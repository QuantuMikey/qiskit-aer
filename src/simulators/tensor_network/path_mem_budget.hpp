// path_mem_budget.hpp — aer-0076: memory-aware sizing of the cotengra
// worker pool.
//
// WHY. The path search runs one loky worker process per parallel slot, and
// every worker holds its own copy of the trial state (hypergraph,
// partitioner working set). Peak host RAM is therefore
//     parent + workers x per_trial(T)
// with T the tensor count. The CPU-only auto-detect (aer-0026) sized the
// pool from cores alone, which is correct only when memory is plentiful:
// battery jobs 21456575/76/77/78/80 (p=6..11 at one GCD, 7 workers) died
// on HOST-RAM OOM at the 64 GB cgroup limit with the GPU untouched. This
// header adds the third signal -- the memory actually granted to the
// process -- and caps the pool so the product fits.
//
// PORTABILITY. Nothing here is LUMI-specific. The budget is discovered in
// order from: cgroup v2 (memory.max), cgroup v1 (memory.limit_in_bytes),
// SLURM's granted-memory environment, and finally /proc/meminfo
// MemAvailable; the MINIMUM of the defined signals binds, because any cap
// that applies is real. Rank sharing on a node is read from
// SLURM_TASKS_PER_NODE ("8" or "8(x2)" forms); outside SLURM it defaults
// to 1 and can be stated explicitly with AER_TN_PATH_LOCAL_RANKS. On
// other HPC systems every path and constant below behaves identically or
// is overridable by environment.
//
// PER-TRIAL MODEL. per_trial(T) = BASE + T x PER_TENSOR, defaults
// BASE = 8500 MB and PER_TENSOR = 192 KiB. The p=5-pass / p=6-OOM pair
// brackets per_trial tightly (both ran 7 workers in the same 64 GB
// share: T=254 fit, T=1482 hit the ceiling), and the defaults are the
// simple line through that bracket, checked against all five measured
// battery points:
//   T=254   (p=5):  8.35 GB/trial -> 7 workers fit 60 GB usable; measured: completed
//   T=1482  (p=6):  8.58 GB/trial -> 6 permitted; measured: 7 hit the ceiling (66.9 GB)
//   T=3012  (p=7):  8.86 GB/trial -> 6 permitted; measured: 7 OOM'd (66.9 GB)
//   T=6078  (p=8):  9.44 GB/trial -> 6 permitted; measured: 7 OOM'd (66.9 GB)
//   T=49068 (p=11): 17.3 GB/trial -> 3 permitted; measured: 7 planned, dead at 37 GB mid-ramp
// The model is deliberately simple and conservative; both constants are
// env-tunable (AER_TN_PATH_TRIAL_BASE_MB, AER_TN_PATH_TRIAL_KB_PER_TENSOR)
// for workloads or systems where the calibration differs. A parent-process
// reserve (default 4 GiB, AER_TN_PATH_PARENT_RESERVE_MB) is subtracted
// from the budget before division.
//
// PRECEDENCE. An explicit AER_TN_PATH_PARALLEL still wins outright (the
// operator said a number; use it). The memory cap applies only to the
// auto-detected path, and the log line states budget, ranks, estimate and
// which signal bound so every job documents its own sizing.
//
// GPU MEMORY is already budgeted elsewhere and needs nothing here: the
// contraction side reads free device memory at set_network ("[AER_TN]
// memory: ... budget" in every log) and slice_target_bytes() clamps
// per-slice peaks to min(target, device budget).

#ifndef _aer_tensor_net_path_mem_budget_hpp_
#define _aer_tensor_net_path_mem_budget_hpp_

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#ifndef _WIN32
#include <sys/resource.h>
#endif

namespace AER {
namespace TensorNetPathMem {

// Read the first whitespace-delimited token of a file into out; false if
// the file is absent or empty. No exceptions, no allocation surprises.
inline bool read_first_token(const char *path, char *out, size_t cap) {
  FILE *f = std::fopen(path, "r");
  if (f == nullptr)
    return false;
  int n = std::fscanf(f, "%63s", out);
  std::fclose(f);
  return n == 1 && cap > 0;
}

// Parse a decimal u64; false on garbage. Values >= 2^60 are treated as
// "unlimited" by callers (cgroup v1 reports a huge sentinel when unset).
inline bool parse_u64(const char *s, uint64_t &val) {
  char *end = nullptr;
  unsigned long long parsed = std::strtoull(s, &end, 10);
  if (end == s)
    return false;
  val = static_cast<uint64_t>(parsed);
  return true;
}

// aer-0077: parse a memory quantity with an optional K/M/G/T suffix
// (SLURM sites differ in whether SLURM_MEM_PER_NODE carries one; a bare
// number is MB per SLURM's documentation and stays MB for the caller).
// Returns bytes; false on garbage.
inline bool parse_mem_mb_or_suffixed(const char *s, uint64_t &bytes) {
  char *end = nullptr;
  unsigned long long v = std::strtoull(s, &end, 10);
  if (end == s || v == 0)
    return false;
  uint64_t mult = 1024ull * 1024ull; // bare number = MB
  switch (*end) {
  case 'K': case 'k': mult = 1024ull; break;
  case 'M': case 'm': mult = 1024ull * 1024ull; break;
  case 'G': case 'g': mult = 1024ull * 1024ull * 1024ull; break;
  case 'T': case 't': mult = 1024ull * 1024ull * 1024ull * 1024ull; break;
  default: break;
  }
  bytes = static_cast<uint64_t>(v) * mult;
  return true;
}

// aer-0077: on cgroup v2 the binding limit usually lives at the
// PROCESS'S OWN cgroup (or an ancestor), not the hierarchy root -- at
// the root memory.max simply reads "max" and a root-only probe walks
// straight past the real limit. Resolve the process's cgroup relative
// path from /proc/self/cgroup ("0::<path>" for v2, "N:memory:<path>"
// for v1's memory controller) and take the MINIMUM limit over the
// cgroup and all its ancestors, since a limit at any level binds.
// Roots and the self file are parameters so the walk is unit-testable
// against fake trees.
inline bool self_cgroup_rel_path(const char *self_file, bool v2, char *out,
                                 size_t cap) {
  FILE *f = std::fopen(self_file, "r");
  if (f == nullptr)
    return false;
  char line[512];
  bool found = false;
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    size_t len = std::strlen(line);
    if (len > 0 && line[len - 1] == '\n')
      line[len - 1] = '\0';
    if (v2) {
      if (std::strncmp(line, "0::", 3) == 0) {
        std::snprintf(out, cap, "%s", line + 3);
        found = true;
        break;
      }
    } else {
      // aer-0078: v1 lines are "N:controller-list:path" with a
      // COMMA-SEPARATED controller list -- "memory,cpuacct",
      // "cpu,memory", or "memory" alone are all real forms, and a
      // substring probe for ":memory:"/",memory:" missed the
      // memory-first list entirely. Split on the two colons and match
      // "memory" as an exact token in the list.
      const char *c1 = std::strchr(line, ':');
      if (c1 == nullptr)
        continue;
      const char *c2p = std::strchr(c1 + 1, ':');
      if (c2p == nullptr)
        continue;
      char ctrls[256];
      size_t clen = static_cast<size_t>(c2p - (c1 + 1));
      if (clen >= sizeof(ctrls))
        continue;
      std::memcpy(ctrls, c1 + 1, clen);
      ctrls[clen] = '\0';
      bool has_memory = false;
      char *save = nullptr;
      for (char *tok = strtok_r(ctrls, ",", &save); tok != nullptr;
           tok = strtok_r(nullptr, ",", &save))
        if (std::strcmp(tok, "memory") == 0) {
          has_memory = true;
          break;
        }
      if (has_memory) {
        std::snprintf(out, cap, "%s", c2p + 1);
        found = true;
        break;
      }
    }
  }
  std::fclose(f);
  return found;
}

// Minimum finite limit over <root><rel>/<file>, walking rel up to "".
// reader is one of the flat-file limit readers above.
template <typename Reader>
inline uint64_t ancestor_min_limit(const char *root, const char *rel,
                                   const char *file, Reader reader) {
  char rel_buf[400];
  std::snprintf(rel_buf, sizeof(rel_buf), "%s", rel);
  uint64_t best = 0;
  for (;;) {
    char path[512];
    std::snprintf(path, sizeof(path), "%s%s/%s", root, rel_buf, file);
    uint64_t v = reader(path);
    if (v > 0 && (best == 0 || v < best))
      best = v;
    char *slash = std::strrchr(rel_buf, '/');
    if (slash == nullptr || slash == rel_buf) {
      if (rel_buf[0] != '\0') {
        rel_buf[0] = '\0';
        continue;
      }
      break;
    }
    *slash = '\0';
  }
  return best;
}

constexpr uint64_t kUnlimited = (1ull << 60);

// cgroup v2 memory.max: decimal bytes or the literal "max".
inline uint64_t cgroup_v2_limit(const char *path) {
  char buf[64];
  uint64_t v = 0;
  if (!read_first_token(path, buf, sizeof(buf)))
    return 0;
  if (std::strcmp(buf, "max") == 0)
    return 0;
  if (!parse_u64(buf, v) || v == 0 || v >= kUnlimited)
    return 0;
  return v;
}

// cgroup v1 memory.limit_in_bytes: decimal bytes; huge sentinel = unset.
inline uint64_t cgroup_v1_limit(const char *path) {
  char buf[64];
  uint64_t v = 0;
  if (!read_first_token(path, buf, sizeof(buf)))
    return 0;
  if (!parse_u64(buf, v) || v == 0 || v >= kUnlimited)
    return 0;
  return v;
}

// SLURM grants: SLURM_MEM_PER_NODE is MB for the node's share (some
// sites carry a K/M/G/T suffix -- both parse); if only SLURM_MEM_PER_CPU
// is set, multiply by the leading integer of SLURM_JOB_CPUS_PER_NODE.
// Zero when neither is present or parseable.
inline uint64_t slurm_granted_bytes() {
  const char *mpn = std::getenv("SLURM_MEM_PER_NODE");
  uint64_t b = 0;
  if (mpn != nullptr && parse_mem_mb_or_suffixed(mpn, b))
    return b;
  const char *mpc = std::getenv("SLURM_MEM_PER_CPU");
  const char *cpn = std::getenv("SLURM_JOB_CPUS_PER_NODE");
  uint64_t per_cpu = 0, cpus = 0;
  if (mpc != nullptr && cpn != nullptr &&
      parse_mem_mb_or_suffixed(mpc, per_cpu) && parse_u64(cpn, cpus) &&
      per_cpu > 0 && cpus > 0)
    return per_cpu * cpus;
  return 0;
}

// /proc/meminfo MemAvailable (kB) — the last-resort signal on systems
// with no cgroup limit and no scheduler environment.
inline uint64_t meminfo_available_bytes(const char *path) {
  FILE *f = std::fopen(path, "r");
  if (f == nullptr)
    return 0;
  char key[64];
  unsigned long long kb = 0;
  uint64_t out = 0;
  while (std::fscanf(f, "%63s %llu kB\n", key, &kb) == 2) {
    if (std::strcmp(key, "MemAvailable:") == 0) {
      out = static_cast<uint64_t>(kb) * 1024ull;
      break;
    }
  }
  std::fclose(f);
  return out;
}

// The node-level budget: minimum of the defined signals, 0 if none.
// aer-0077: cgroup limits are resolved at the PROCESS'S cgroup and its
// ancestors (via self_file), not just the hierarchy root -- the root
// probe alone reads "max" on v2 systems where the job's limit lives at
// the leaf. Roots and self_file are parameters for unit testing; the
// production caller passes the real /sys/fs/cgroup, /sys/fs/cgroup/
// memory and /proc/self/cgroup.
inline uint64_t node_mem_budget_bytes(const char *cg2_root,
                                      const char *cg1_root,
                                      const char *self_file,
                                      const char *meminfo) {
  uint64_t c2 = 0, c1 = 0;
  char rel[400];
  if (self_cgroup_rel_path(self_file, true, rel, sizeof(rel)))
    c2 = ancestor_min_limit(cg2_root, rel, "memory.max",
                            [](const char *p) { return cgroup_v2_limit(p); });
  if (c2 == 0) {
    char root_file[512];
    std::snprintf(root_file, sizeof(root_file), "%s/memory.max", cg2_root);
    c2 = cgroup_v2_limit(root_file);
  }
  if (self_cgroup_rel_path(self_file, false, rel, sizeof(rel)))
    c1 = ancestor_min_limit(cg1_root, rel, "memory.limit_in_bytes",
                            [](const char *p) { return cgroup_v1_limit(p); });
  if (c1 == 0) {
    char root_file[512];
    std::snprintf(root_file, sizeof(root_file), "%s/memory.limit_in_bytes",
                  cg1_root);
    c1 = cgroup_v1_limit(root_file);
  }
  uint64_t sl = slurm_granted_bytes();
  uint64_t best = 0;
  const uint64_t signals[3] = {c2, c1, sl};
  for (uint64_t s : signals)
    if (s > 0 && (best == 0 || s < best))
      best = s;
  if (best == 0)
    best = meminfo_available_bytes(meminfo);
  return best;
}

// Ranks sharing this node's budget. SLURM_TASKS_PER_NODE leads ("8",
// "8(x2)", "6,2" all parse to their LEADING count). On heterogeneous
// layouts the leading count may exceed this node's actual rank count,
// which divides the budget too finely and yields FEWER workers -- the
// safe direction. AER_TN_PATH_LOCAL_RANKS overrides for non-SLURM
// launchers or when precision matters.
inline long local_ranks_sharing_node() {
  const char *ov = std::getenv("AER_TN_PATH_LOCAL_RANKS");
  uint64_t v = 0;
  if (ov != nullptr && parse_u64(ov, v) && v > 0)
    return static_cast<long>(v);
  const char *tpn = std::getenv("SLURM_TASKS_PER_NODE");
  if (tpn != nullptr && parse_u64(tpn, v) && v > 0)
    return static_cast<long>(v);
  return 1;
}

// Env-tunable u64 with default (value read as plain integer in the given
// unit; callers convert).
inline uint64_t env_u64(const char *name, uint64_t dflt) {
  const char *e = std::getenv(name);
  uint64_t v = 0;
  if (e != nullptr && parse_u64(e, v) && v > 0)
    return v;
  return dflt;
}

// per_trial(T) = BASE + T x PER_TENSOR (bytes).
inline uint64_t trial_bytes_estimate(uint64_t num_tensors) {
  uint64_t base = env_u64("AER_TN_PATH_TRIAL_BASE_MB", 8500) * 1024ull * 1024ull;
  uint64_t per_t = env_u64("AER_TN_PATH_TRIAL_KB_PER_TENSOR", 320) * 1024ull;
  return base + num_tensors * per_t;
}

// The memory-permitted worker count for this rank: how many concurrent
// trials fit in this rank's share of the node budget after the parent
// reserve. 0 means "no memory signal; do not cap". Never below 1 when a
// signal exists (serial always remains the floor the caller may choose).
inline long mem_permitted_workers(uint64_t num_tensors, uint64_t node_budget,
                                  long local_ranks) {
  if (node_budget == 0)
    return 0;
  if (local_ranks < 1)
    local_ranks = 1;
  uint64_t per_rank = node_budget / static_cast<uint64_t>(local_ranks);
  uint64_t reserve =
      env_u64("AER_TN_PATH_PARENT_RESERVE_MB", 4096) * 1024ull * 1024ull;
  if (per_rank <= reserve)
    return 1;
  uint64_t usable = per_rank - reserve;
  uint64_t per_trial = trial_bytes_estimate(num_tensors);
  uint64_t n = usable / per_trial;
  if (n < 1)
    return 1;
  if (n > 1024)
    n = 1024;
  return static_cast<long>(n);
}

// aer-0079: kernel-accounted peak-RSS measurement, so the per-trial cost
// can be MEASURED on the live workload instead of modeled. VmHWM is the
// process's peak resident set as tracked by the kernel; writing "5" to
// /proc/self/clear_refs resets the peak counter (no privilege needed --
// verified by direct test: a touched 300 MB allocation measures as a
// 300 MB delta). Both degrade gracefully: 0 / false on systems without
// procfs, and the caller falls back to the model.
// aer-0080: the device memory budget, measured once at set_network
// (hipMemGetInfo minus workspace, MPI-agreed across ranks), published
// here so host-side sizing decisions can be VRAM-derived without any
// new device queries. 0 until set_network has run.
inline uint64_t &device_budget_bytes() {
  static uint64_t b = 0;
  return b;
}

inline uint64_t read_vmhwm_kb() {
  FILE *f = std::fopen("/proc/self/status", "r");
  if (f == nullptr)
    return 0;
  char key[64];
  unsigned long long kb = 0;
  uint64_t out = 0;
  char line[256];
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    if (std::sscanf(line, "%63s %llu", key, &kb) == 2 &&
        std::strcmp(key, "VmHWM:") == 0) {
      out = kb;
      break;
    }
  }
  std::fclose(f);
  return out;
}

inline bool reset_peak_rss() {
  FILE *f = std::fopen("/proc/self/clear_refs", "w");
  if (f == nullptr)
    return false;
  bool ok = std::fputs("5", f) >= 0;
  return (std::fclose(f) == 0) && ok;
}

// aer-0090: a scoped soft cap on the process's data segment, so a search
// trial whose memory demand is unbounded (a degenerate drawn tree's stats
// caches -- 158 GiB on one rank at T=49070, job 21474691) dies as a
// catchable Python MemoryError INSIDE cotengra's per-trial exception
// handler (hyper.py 326-339 catches any Exception, warns, scores the
// trial inf, and moves on) instead of taking the node down. RLIMIT_DATA,
// not RLIMIT_AS: anonymous heap counts under DATA on modern kernels,
// while HIP's large virtual-address reservations do not, so the engine's
// device mappings stay untouched. Soft limit only, never raised above an
// existing tighter limit, restored in the destructor (exception-safe), so
// nothing after the search ever runs capped. Forked search workers
// inherit the soft limit, which is the protective direction. Verified by
// direct test on the exact cotengra classes: under the cap the degenerate
// contract_stats raises MemoryError, the discard frees the memory, and
// the next trial in the same process succeeds.
class ScopedRlimitData {
public:
  explicit ScopedRlimitData(uint64_t cap_bytes) {
#ifndef _WIN32
    if (cap_bytes == 0)
      return;
    if (::getrlimit(RLIMIT_DATA, &saved_) != 0)
      return;
    rlim_t cap = static_cast<rlim_t>(cap_bytes);
    if (saved_.rlim_max != RLIM_INFINITY && cap > saved_.rlim_max)
      cap = saved_.rlim_max;
    if (saved_.rlim_cur != RLIM_INFINITY && saved_.rlim_cur <= cap)
      return; // an existing limit is already at least as tight
    struct rlimit r = saved_;
    r.rlim_cur = cap;
    armed_ = (::setrlimit(RLIMIT_DATA, &r) == 0);
#else
    (void)cap_bytes;
#endif
  }
  ~ScopedRlimitData() {
#ifndef _WIN32
    if (armed_)
      ::setrlimit(RLIMIT_DATA, &saved_);
#endif
  }
  ScopedRlimitData(const ScopedRlimitData &) = delete;
  ScopedRlimitData &operator=(const ScopedRlimitData &) = delete;
  bool armed() const { return armed_; }

private:
#ifndef _WIN32
  struct rlimit saved_ {};
#endif
  bool armed_ = false;
};

} // namespace TensorNetPathMem
} // namespace AER

#endif
