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

// SLURM grants: SLURM_MEM_PER_NODE is MB for the node's share; if only
// SLURM_MEM_PER_CPU is set, multiply by the leading integer of
// SLURM_JOB_CPUS_PER_NODE. Zero when neither is present or parseable.
inline uint64_t slurm_granted_bytes() {
  const char *mpn = std::getenv("SLURM_MEM_PER_NODE");
  uint64_t v = 0;
  if (mpn != nullptr && parse_u64(mpn, v) && v > 0)
    return v * 1024ull * 1024ull;
  const char *mpc = std::getenv("SLURM_MEM_PER_CPU");
  const char *cpn = std::getenv("SLURM_JOB_CPUS_PER_NODE");
  uint64_t per_cpu = 0, cpus = 0;
  if (mpc != nullptr && cpn != nullptr && parse_u64(mpc, per_cpu) &&
      parse_u64(cpn, cpus) && per_cpu > 0 && cpus > 0)
    return per_cpu * cpus * 1024ull * 1024ull;
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
inline uint64_t node_mem_budget_bytes(const char *cg2, const char *cg1,
                                      const char *meminfo) {
  uint64_t best = 0;
  uint64_t c2 = cgroup_v2_limit(cg2);
  uint64_t c1 = cgroup_v1_limit(cg1);
  uint64_t sl = slurm_granted_bytes();
  const uint64_t signals[3] = {c2, c1, sl};
  for (uint64_t s : signals)
    if (s > 0 && (best == 0 || s < best))
      best = s;
  if (best == 0)
    best = meminfo_available_bytes(meminfo);
  return best;
}

// Ranks sharing this node's budget. SLURM_TASKS_PER_NODE leads ("8",
// "8(x2)", "6,2" all start with the local count on homogeneous layouts);
// AER_TN_PATH_LOCAL_RANKS overrides for non-SLURM launchers.
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
  uint64_t per_t = env_u64("AER_TN_PATH_TRIAL_KB_PER_TENSOR", 192) * 1024ull;
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

} // namespace TensorNetPathMem
} // namespace AER

#endif
