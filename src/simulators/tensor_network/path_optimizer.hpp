/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2022.
 * (C) Copyright CSC - IT Center for Science Ltd. 2026.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 */

#ifndef _path_optimizer_hpp_
#define _path_optimizer_hpp_

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include "simulators/tensor_network/path_mem_budget.hpp"

// aer-0026: sched_getaffinity()/CPU_COUNT for the cotengra worker count. Linux
// only, which this file already is (ROCm/HIP, MPICH, LUMI).
#include <sched.h>

#ifdef AER_HIPTENSOR
#include <pybind11/pybind11.h>
// aer-0093: py::exec (the aer-0091 cotengrust block) is declared in
// pybind11/eval.h, not pybind11.h -- build 21476498 failed with "no
// member named 'exec' in namespace 'pybind11'" at exactly that call.
#include <pybind11/eval.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

namespace AER {
namespace TensorNetwork {

static bool path_verbose() {
  static bool checked = false;
  static bool enabled = false;
  if (!checked) {
    const char *val = std::getenv("AER_TN_PATH_VERBOSE");
    enabled = (val != nullptr && std::string(val) == "1");
    checked = true;
  }
  return enabled;
}

// Tiling enable policy. Tiling decomposes an oversized (M>6, N>6, or K>6)
// per-step contraction — and a >12-free-mode output during path search — into a
// grid of m6n6k6 sub-contractions, so a step that hipTensor's kernels cannot
// dispatch produces correct results instead of a hard refusal. Both the planner
// gate (>12-free-mode output, below) and the contractor gate (per-step M/N/K>6,
// in setup_pool_and_cache) read this single policy.
//
// Tristate, default AUTO:
//   AUTO  transparent — an oversized step is detected and tiling is engaged for
//         that run via a one-shot re-plan (see setup_contraction). A circuit
//         that fits without tiling pays nothing: pass 1 runs the exact no-tile
//         planning and, on success, there is no retry and no path change.
//   OFF   force no tiling; the gates refuse oversized steps. The escape hatch
//         for NVIDIA A/B comparison, bug-report reproducers, and
//         AER_TN_DISABLE_RANK_GUARD bisection.
//   ON    tiling always available to the planner and contractor (no re-plan).
//
// Back-compat with the prior boolean knob (no breakage for existing scripts or
// checkpoints): AER_TN_ENABLE_TILING=1 maps to ON, =0 maps to OFF. When
// AER_TN_ENABLE_TILING is unset, AER_TN_TILING is consulted: "on"/"off"/"auto"
// (case-insensitive); unset or unrecognised is AUTO.
enum class TilingMode { Off, On, Auto };

// aer-0030: process-wide latch on the AUTO tiling decision.
//
// WHY. Under AUTO, `engaged` is a LOCAL re-initialised on every
// setup_contraction call, so a workload in which every contraction needs
// tiling rediscovers that fact every single time: pass 1 runs a full cotengra
// search with tiling held back, trips the m6n6k6 gate, throws the plan away,
// and pass 2 searches again with tiling engaged. Job 20539864 (10000-vertex
// random 3-regular MaxCut at depth 3) retried on 48 of its 50 contractions and
// measured 9871 ms of cold path per class -- that is TWO searches of about
// 4935 ms. Roughly 123 s of that 308 s run was spent searching for plans
// discarded by design.
//
// The message the retry already prints claims tiling is enabled "for this
// run". It was not; this makes it true. After the gate has fired once, every
// later setup starts engaged, so a new topology pays one search instead of two
// and a repeat topology skips the doomed pass-1 lookup entirely.
//
// SAFETY. The latch can only ever turn tiling ON, and tiling is the validated
// WS-3 path: an oversized step decomposes into m6n6k6 sub-contractions, and a
// step that does not need it produces no tiles and is untouched. What does
// change is that the planner is handed tiling_available=true from the start,
// so it may return a DIFFERENT contraction order. Any order is exact, so this
// is a performance question and not a correctness one, but it is why the knob
// below exists.
//
// MPI. The latch is per-process, which is safe because the decision is already
// collective: the planner gate throws collectively after an Allreduce, and the
// contractor gate fires on a MINLOC-broadcast plan that is identical on every
// rank. Every rank therefore latches on the same contraction.
//
// AER_TN_TILING_LATCH=0 (or off/false) restores the per-contraction
// rediscovery, which is how to A/B it without a rebuild.
static std::atomic<bool> &tn_tiling_latch_flag() {
  static std::atomic<bool> latched(false);
  return latched;
}

static bool tn_tiling_latch_enabled() {
  static bool checked = false;
  static bool cached = true;
  if (!checked) {
    const char *val = std::getenv("AER_TN_TILING_LATCH");
    if (val != nullptr) {
      std::string s(val);
      for (char &c : s) c = static_cast<char>(std::tolower(c));
      cached = !(s == "0" || s == "off" || s == "false");
    }
    checked = true;
  }
  return cached;
}

static bool tn_tiling_latched() {
  return tn_tiling_latch_enabled() &&
         tn_tiling_latch_flag().load(std::memory_order_relaxed);
}

static void tn_tiling_latch_set() {
  if (tn_tiling_latch_enabled())
    tn_tiling_latch_flag().store(true, std::memory_order_relaxed);
}

static TilingMode tn_tiling_mode() {
  static bool checked = false;
  static TilingMode mode = TilingMode::Auto;
  if (!checked) {
    const char *legacy = std::getenv("AER_TN_ENABLE_TILING");
    if (legacy != nullptr) {
      std::string s(legacy);
      mode = (s == "1") ? TilingMode::On : TilingMode::Off;
    } else {
      const char *val = std::getenv("AER_TN_TILING");
      if (val != nullptr) {
        std::string s(val);
        for (char &c : s) c = static_cast<char>(std::tolower(c));
        if (s == "on")
          mode = TilingMode::On;
        else if (s == "off")
          mode = TilingMode::Off;
        else if (s == "auto")
          mode = TilingMode::Auto;
        else {
          fprintf(stderr,
                  "[AER_TN_PATH] warning: AER_TN_TILING='%s' not recognised, "
                  "using 'auto'. Valid values: auto, on, off.\n",
                  val);
          mode = TilingMode::Auto;
        }
      }
    }
    checked = true;
  }
  return mode;
}

// Typed signal that a gate hit the m6n6k6 envelope while tiling was held back.
// In AUTO, setup_contraction catches THIS TYPE ONLY and re-plans once with
// tiling engaged; OFF lets it convert to a hard refusal at the gate. It is
// never thrown for a real failure (OOM, CK error), so the retry can never mask
// one. Carries the gate that fired and the offending shape for the info line.
class NeedsTilingException : public std::runtime_error {
public:
  enum class Gate { Planner, Contractor };

  NeedsTilingException(Gate gate, const std::string &detail)
      : std::runtime_error(detail), gate_(gate) {}

  Gate gate() const { return gate_; }
  const char *where() const {
    return gate_ == Gate::Planner ? "planner (>12-free-mode output)"
                                  : "contractor (per-step M/N/K>6)";
  }

private:
  Gate gate_;
};

// Read cotengra backend selection from AER_TN_OPTLIB env var.
// Valid values: "optuna" (default), "cmaes", "sbplx", "sses".
// Anything else falls through to "optuna".
static std::string path_optlib() {
  static bool checked = false;
  static std::string cached;
  if (!checked) {
    const char *val = std::getenv("AER_TN_OPTLIB");
    if (val != nullptr) {
      std::string s(val);
      if (s == "optuna" || s == "cmaes" || s == "sbplx" || s == "sses") {
        cached = s;
      } else {
        fprintf(stderr,
                "[AER_TN_PATH] warning: AER_TN_OPTLIB='%s' not recognised, "
                "falling back to 'optuna'. Valid values: optuna, cmaes, "
                "sbplx, sses.\n", val);
        cached = "optuna";
      }
    } else {
      cached = "optuna";
    }
    checked = true;
  }
  return cached;
}

// Per-slice peak-intermediate budget in BYTES, from AER_TN_SLICE_TARGET_BYTES.
// It was written as an OPTIONAL user knob to slice HARDER (smaller per-slice
// memory), sitting under an authoritative clamp -- the CK-safe tiled-operand
// volume enforced by max_tiled_elements(). aer-0013 removed that clamp, WHICH
// MADE THIS ENVELOPE THE ONLY BINDING CONSTRAINT, and the wording here said the
// opposite until aer-0038 corrected it. 2097152 bytes = 131072 elements at 16 bytes per
// logical complex element (double, split-complex: two 8-byte real planes) --
// 2^17, which is roughly 32000x SMALLER than the device-memory term on an
// MI250X GCD (measured: 65137.5 MB free, so about 2^32 elements). The min()
// below therefore always picks this value and the memory budget never wins,
// which is why aer-0013 is named "memory-govern the slicing target" and does
// not. See CAPABILITY_MATRIX_v9 7.8. The budget
// converts to a cotengra `target_size` in ELEMENTS using each optimizer's
// element size, so a single value covers float and double. Lower it to slice
// harder (e.g. to shrink per-slice memory); it can only slice harder, never
// raise the per-slice peak.
// aer-0026: how many worker processes cotengra may use for the path search.
//
// The fork never passed `parallel`, so cotengra applied its own default of
// 'auto'. 'auto' sizes the pool from the machine's CPU count, and inside the
// container that reads the WHOLE NODE: job 20497440 reported 256 visible CPUs
// while the allocation was a handful of cores. Every path search has therefore
// been trying to start ~256 loky workers into a small cgroup, which is the most
// likely cause of the 20-36% worker deaths already recorded -- dead workers
// silently truncate the search, so the plan that comes back is whatever
// survived rather than the best of max_repeats trials.
//
// Return value: -2 leave cotengra's 'auto' (the pre-aer-0026 behaviour),
//               -1 pass False (serial, no pool at all), N>=1 pass N workers.
//
// AER_TN_PATH_PARALLEL: unset = auto-detect (below); "auto" = -2; "0"/"false" =
// -1; a positive integer = that many workers. The knob exists so the three can
// be compared inside one job without another rebuild.
//
// Auto-detection deliberately takes the MINIMUM of two independent signals and
// clamps the result, because neither is guaranteed here and I have not been
// able to verify which one LUMI actually provides through Singularity:
//   * sched_getaffinity() -- the cores this process may run on. Reflects SLURM
//     cpu-binding and the GPU affinity wrapper if either applied.
//   * SLURM_CPUS_PER_TASK -- what was requested. Absent outside SLURM.
// If both fail we fall back to 1 (serial), which is slow but never
// oversubscribes. The cap of 32 stops a whole-node affinity mask from
// reproducing the very problem this fixes.
//
// The chosen value and BOTH inputs are printed once per process, so the first
// job after this lands says outright whether affinity is visible inside the
// container. That question is why this is measured rather than assumed.
static int path_parallel_setting(size_t num_tensors) {
  // aer-0076: the pool is sized from THREE signals, not two -- CPUs
  // available (affinity), CPUs granted (SLURM), and now the host memory
  // actually granted to this process (cgroup/SLURM/meminfo chain in
  // path_mem_budget.hpp). Peak search RSS is parent + workers x
  // per_trial(T), so the memory-permitted count depends on the network
  // size and is computed per search rather than cached; the explicit
  // knob still wins outright. Battery jobs 21456575/76/77/78/80 are the
  // measurements behind this: 7 CPU-sized workers host-OOM'd a 64 GB
  // 1-GCD share at every depth past p=5 while the GPU sat untouched.
  long affinity = -1;
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
    affinity = static_cast<long>(CPU_COUNT(&mask));

  long slurm = -1;
  const char *sc = std::getenv("SLURM_CPUS_PER_TASK");
  if (sc != nullptr) {
    char *end = nullptr;
    long parsed = std::strtol(sc, &end, 10);
    if (end != sc && parsed > 0)
      slurm = parsed;
  }

  uint64_t budget = TensorNetPathMem::node_mem_budget_bytes(
      "/sys/fs/cgroup", "/sys/fs/cgroup/memory", "/proc/self/cgroup",
      "/proc/meminfo");
  long ranks = TensorNetPathMem::local_ranks_sharing_node();
  long mem_workers = TensorNetPathMem::mem_permitted_workers(
      static_cast<uint64_t>(num_tensors), budget, ranks);

  int result = -2;
  const char *v = std::getenv("AER_TN_PATH_PARALLEL");
  std::string source = "auto-detected";
  if (v != nullptr) {
    std::string s(v);
    source = "AER_TN_PATH_PARALLEL";
    if (s == "auto") {
      result = -2;
    } else if (s == "0" || s == "false" || s == "False") {
      result = -1;
    } else {
      char *end = nullptr;
      long parsed = std::strtol(v, &end, 10);
      result = (end != v && parsed > 0) ? static_cast<int>(parsed) : -2;
    }
  } else {
    long n = affinity;
    if (slurm > 0 && (n <= 0 || slurm < n))
      n = slurm;
    if (n <= 0)
      n = 1;
    if (n > 32)
      n = 32;
    if (mem_workers > 0 && mem_workers < n) {
      n = mem_workers;
      source = "memory-capped";
    }
    result = (n <= 1) ? -1 : static_cast<int>(n);
  }

  char desc[64];
  if (result == -2)
    snprintf(desc, sizeof(desc), "cotengra 'auto' (whole-node CPU count)");
  else if (result == -1)
    snprintf(desc, sizeof(desc), "False (serial)");
  else
    snprintf(desc, sizeof(desc), "%d worker(s)", result);
  fprintf(stderr,
          "[AER_TN_PATH] parallel: sched_getaffinity=%ld "
          "SLURM_CPUS_PER_TASK=%ld mem_budget_mb=%llu local_ranks=%ld "
          "trial_est_mb=%llu mem_workers=%ld tensors=%zu -> %s [%s]\n",
          affinity, slurm,
          static_cast<unsigned long long>(budget / (1024ull * 1024ull)),
          ranks,
          static_cast<unsigned long long>(
              TensorNetPathMem::trial_bytes_estimate(
                  static_cast<uint64_t>(num_tensors)) /
              (1024ull * 1024ull)),
          mem_workers, num_tensors, desc, source.c_str());
  return result;
}

static uint64_t slice_target_bytes() {
  // aer-0062: default raised 2 MiB -> 8 GiB. The 2 MiB constant bound at
  // 2^17 elements, ~32,000x below a GCD's memory, so a drop-in
  // memory-bound run over-sliced into the slice-count ceiling and the
  // first user experience was a refusal. 8 GiB is the smallest fence
  // admitting the largest measured-safe executed peak: intermediates land
  // on power-of-two peaks, the budget sweep (job 21310992) bracketed the
  // boundary at 8 GiB peak ok (pool 17.2 GB) / 16 GiB peak OOM, and every
  // fence in [8,16) admits exactly the same 8 GiB maximum peak -- so 8
  // buys the full throughput of any larger sub-16 fence with a full
  // power-of-two of headroom. min(budget, device memory) still applies.
  static bool checked = false;
  static uint64_t cached = 8589934592ULL;
  // aer-0080: when the knob is unset the fence is VRAM-DERIVED --
  // device budget / 8 -- instead of a hardcoded byte count. On this
  // system's MI250X GCDs (65200 MB budget) the rule yields ~8.15 GiB,
  // i.e. the measured-safe 8 GiB constant EMERGES from the rule, while
  // a GPU with more memory scales automatically. The device budget is
  // published by set_network, which runs before any path search (log
  // ordering: "[AER_TN] memory:" precedes "[AER_TN_PATH]"); if it is
  // somehow unavailable the historic 8 GiB constant remains. Explicit
  // AER_TN_SLICE_TARGET_BYTES still wins outright, floor 2 MiB.
  if (!checked) {
    const char *val = std::getenv("AER_TN_SLICE_TARGET_BYTES");
    if (val == nullptr) {
      uint64_t dev = TensorNetPathMem::device_budget_bytes();
      if (dev > 0) {
        uint64_t derived = dev / 8;
        if (derived < 2097152ULL)
          derived = 2097152ULL;
        cached = derived;
      }
    }
    if (val != nullptr) {
      char *end = nullptr;
      unsigned long long parsed = std::strtoull(val, &end, 10);
      if (end != val && parsed > 0) {
        cached = static_cast<uint64_t>(parsed);
      } else {
        fprintf(stderr,
                "[AER_TN_PATH] warning: AER_TN_SLICE_TARGET_BYTES='%s' is "
                "not a positive integer; using default %llu bytes.\n",
                val, (unsigned long long)cached);
      }
    }
    checked = true;
  }
  return cached;
}

// Free-mode envelope of hipTensor's only kernel shape: m6n6k6 admits at most
// 6 M-modes + 6 N-modes per contraction, so one kernel result holds at most
// 2^12 = 4096 elements. This count is one single-kernel result: it grounds the
// planner's oversized-output gate and the FLOOR of the output tiling ceiling
// below (a tiled operand can never be smaller than one kernel result). It is no
// longer any ceiling's DEFAULT: the tiled-operand-volume clamp it once floored
// was max_tiled_elements(), removed by aer-0013 and deleted by aer-0038.
static constexpr size_t kMaxFreeModes = 12; // 6 M-modes + 6 N-modes (m6n6k6)

// aer-0038: max_tiled_elements() and its AER_TN_MAX_TILED_ELEMENTS knob were
// REMOVED here. aer-0013 deleted both call sites and nothing replaced them, so
// the function had been dead code fronting a knob that looked live: a user or a
// sweep setting AER_TN_MAX_TILED_ELEMENTS changed nothing, silently. The clamp
// it used to apply -- 2^16 elements, the m6n6k6 tiled-operand volume -- is
// referred to below as "the removed max_tiled_elements() clamp" where the
// history matters. lumi_tests/run_ck_ceiling_sweep.sh and
// lumi_tests/run_sliced_tiled_probe.sh still set the variable and are annotated
// accordingly. See CAPABILITY_MATRIX_v9 7.5 defect 7.

// Fixed CK-safe descriptor STRIDE ceiling that triggers operand staging in the
// contractor. DECOUPLED from the removed max_tiled_elements() clamp (which clamps the slicing
// target): a tiled sub-block whose max free-mode stride reaches THIS value is
// packed into contiguous scratch instead of handed to hipTensor, which faults
// building a plan on strided views this large. Kept separate because raising the
// per-slice peak target (then AER_TN_MAX_TILED_ELEMENTS, since removed --
// today AER_TN_SLICE_TARGET_BYTES) to reduce slice count must
// NOT also raise the staging trigger -- otherwise the larger descriptors slip
// UNDER the trigger and fault CK (observed: sweep 19756385, every over-ceiling
// step went direct to CK because the trigger tracked the raised ceiling). The
// default is 2^16: bit-exact vs statevector at stride 2^15, CK faults by ~2^18.
// Lowering it (e.g. AER_TN_CK_STRIDE_CEILING=8192 for testing) forces smaller
// descriptors through staging on a circuit that finishes quickly.
static uint64_t ck_stage_stride_ceiling() {
  static bool checked = false;
  static uint64_t cached = 65536;
  if (!checked) {
    const char *val = std::getenv("AER_TN_CK_STRIDE_CEILING");
    if (val != nullptr) {
      char *end = nullptr;
      unsigned long long parsed = std::strtoull(val, &end, 10);
      if (end != val && parsed > 0)
        cached = static_cast<uint64_t>(parsed);
      else
        fprintf(stderr,
                "[AER_TN_PATH] warning: AER_TN_CK_STRIDE_CEILING='%s' is not a "
                "positive integer; using default %llu.\n",
                val, (unsigned long long)cached);
    }
    checked = true;
  }
  return cached;
}

// Hard ceiling, in ELEMENTS, on the (unsliceable) OUTPUT tensor the final
// contraction step may produce. DECOUPLED from the per-slice peak target --
// historically the max_tiled_elements() clamp (AER_TN_MAX_TILED_ELEMENTS),
// removed by aer-0013 and deleted by aer-0038; today AER_TN_SLICE_TARGET_BYTES
// alone -- which governs the per-operand volume of TILED INTERMEDIATE steps and
// trades slice count against per-slice peak. The maximum admissible OUTPUT rank
// is a
// separate concern -- output (open) modes can never be sliced (allow_outer=
// false), so the output tensor's own size is a hard lower bound on the
// per-slice peak, and if the output cannot come from a single m6n6k6 result the
// final step must be TILED and therefore bounded by what tiling+staging can
// emit CK-safely. Coupling the two meant that raising the slicing knob to cut
// slice count on a hard circuit silently raised the accepted output rank (and
// lowering it for stride safety silently forbade legitimate outputs). This
// completes the aer-0011 decouple (which separated the staging STRIDE trigger)
// for the output-ELEMENT concern. Default 2^16 = the same hardware-validated
// CK-safe tiled volume, so behaviour is unchanged until the knob is set. The
// rank-0 amplitude path (get_amplitude, output_elements=1) is trivially under
// this ceiling; the gate exists for save_statevector and reduced-DM outputs.
static uint64_t output_tiling_ceiling() {
  static bool checked = false;
  static uint64_t cached = 65536; // 2^16, CK-safe tiled-output volume
  if (!checked) {
    const char *val = std::getenv("AER_TN_MAX_OUTPUT_ELEMENTS");
    if (val != nullptr) {
      char *end = nullptr;
      unsigned long long parsed = std::strtoull(val, &end, 10);
      if (end != val && parsed > 0) {
        cached = static_cast<uint64_t>(parsed);
        // Floor at one m6n6k6 result: a single untiled kernel already emits
        // 2^kMaxFreeModes elements, so a smaller ceiling would reject outputs
        // that need no tiling at all.
        const uint64_t floor_elems = static_cast<uint64_t>(1) << kMaxFreeModes;
        if (cached < floor_elems) {
          fprintf(stderr,
                  "[AER_TN_PATH] warning: AER_TN_MAX_OUTPUT_ELEMENTS=%llu is "
                  "below one m6n6k6 result (%llu); using %llu.\n",
                  (unsigned long long)cached, (unsigned long long)floor_elems,
                  (unsigned long long)floor_elems);
          cached = floor_elems;
        }
      } else {
        fprintf(stderr,
                "[AER_TN_PATH] warning: AER_TN_MAX_OUTPUT_ELEMENTS='%s' is not "
                "a positive integer; using default %llu.\n",
                val, (unsigned long long)cached);
      }
    }
    checked = true;
  }
  return cached;
}

// Cotengra search objective, from AER_TN_MINIMIZE. "combo" (default) minimizes
// FLOPs+write; it can select plans with very wide single steps (huge
// intermediates) that then require deep m6n6k6 tiling -- the multi-GCD crash
// (job 19229140). "size" minimizes the largest intermediate, favoring
// low-treewidth plans whose per-step shapes stay narrow, so tiling stays
// shallow. Because every ensemble rank then searches for narrow plans, the
// MINLOC(total_flops) pick in MPIParallelPathOptimizer has no wide plan to
// select. "limit", "flops", and "write" are also valid cotengra objective
// names and pass straight through. This is the cause-level lever for the
// deep-tiling problem; default "combo" preserves prior behavior.
static std::string path_minimize() {
  static bool checked = false;
  static std::string cached = "combo";
  if (!checked) {
    const char *val = std::getenv("AER_TN_MINIMIZE");
    if (val != nullptr && *val != '\0') {
      cached = std::string(val);
    }
    checked = true;
  }
  return cached;
}

// Cotengra path-search budget, overridable for tuning and benchmarking
// without recompiling. The HyperOptimizer stops at whichever of these two
// limits it reaches first: max_repeats hyperopt trials, or max_time seconds
// of wall clock. More trials (or more time) tends to find a cheaper
// contraction at the cost of longer planning. Default is 64 trials / 30 s
// (aer-0021): a per-amplitude budget sweep (5 repeats/budget, 25q + 36q,
// natural slicing) showed the historical 128/60 to be over-provisioned --
// path search is 70-90% of wall, and 64/30 finds the same-quality plan (stable
// num_slices across all repeats) in ~1.6x less search time. Going below 64/30
// begins to draw over-sliced plans on a single search (mitigated on multi-GCD
// by the MINLOC best-of-ranks), so 64/30 is the conservative knee. Lower them
// for fast iteration; raise toward 128/60 for a hard high-treewidth network
// where plan quality dominates. Read once and cached, so a process uses one
// consistent budget.
static int path_max_repeats() {
  static bool checked = false;
  static int cached = 64;
  if (!checked) {
    const char *val = std::getenv("AER_TN_PATH_MAX_REPEATS");
    if (val != nullptr) {
      char *end = nullptr;
      long parsed = std::strtol(val, &end, 10);
      if (end != val && parsed > 0) {
        cached = static_cast<int>(parsed);
      } else {
        fprintf(stderr,
                "[AER_TN_PATH] warning: AER_TN_PATH_MAX_REPEATS='%s' is not a "
                "positive integer; using default %d.\n",
                val, cached);
      }
    }
    checked = true;
  }
  return cached;
}

static double path_max_time() {
  static bool checked = false;
  static double cached = 30.0;
  if (!checked) {
    const char *val = std::getenv("AER_TN_PATH_MAX_TIME");
    if (val != nullptr) {
      char *end = nullptr;
      double parsed = std::strtod(val, &end);
      if (end != val && parsed > 0.0) {
        cached = parsed;
      } else {
        fprintf(stderr,
                "[AER_TN_PATH] warning: AER_TN_PATH_MAX_TIME='%s' is not a "
                "positive number; using default %.1f s.\n",
                val, cached);
      }
    }
    checked = true;
  }
  return cached;
}

// AER_TN_FORCE_SLICING=1 forces at least 2 slices even when the natural
// path already fits the per-slice budget. Slicer correctness testing on
// small circuits depends on this: it exercises the full project/execute/
// accumulate machinery on circuits cheap enough to validate element-wise
// against a statevector or NumPy einsum reference.
// aer-0035: floor on the slice count expressed PER RANK.
//
// The slice is the unit MPI divides -- slice_begin_ = myrank_ * num_slices /
// nprocs_ in the contractor -- and nothing anywhere else in the slicing path
// knows the rank count. target_elements is min(device memory, the
// AER_TN_SLICE_TARGET_BYTES envelope) and carries no distribution term, so a
// contraction whose natural peak already fits produces num_slices = 1 no matter
// how many ranks are present. Integer division then gives every rank but one an
// empty range: job 20580891 arm2 shows exactly that, slices=1 with
// slices_local=0 on rank 0 and 1 on rank 1.
//
// That is why every distribution measurement in this project has measured
// nothing. Not because the workloads were small, but because no code path ever
// asks for slices on distribution grounds.
//
// AER_TN_MIN_SLICES_PER_RANK=k asks the planner for at least k * nprocs slices.
// DEFAULT 0 = off, so every validated result stands unchanged and this is an
// A/B with no rebuild. It is off by default because forcing slices is not free:
// slicing replicates work across slices, so on a contraction small enough to
// fit one device the split can cost more than it saves. It exists for the case
// distribution is for -- a contraction too large for one device -- and for
// measuring where the crossover lies.
// aer-0085: FORGE mode. AER_TN_PLAN_FORGE=1 declares this run a plan
// forge: its product is a banked plan for later replay, so the search
// runs SERIALLY (one trial in memory at a time -- the safe shape at
// any network size) and the planner targets a TOTAL slice count
// (AER_TN_PLAN_TARGET_SLICES, default 16) so the captured plan replays
// at every width up to that count through the width-blind
// AER_TN_PLAN_FILE. The total is divided across ranks here in the
// compiled layer, so launchers pass intent, not arithmetic.
static bool tn_plan_forge() {
  const char *v = std::getenv("AER_TN_PLAN_FORGE");
  return v != nullptr && v[0] == '1' && v[1] == '\0';
}

static uint64_t min_slices_per_rank() {
  // aer-0062: default raised 0 -> 1, paired with the slice-target raise
  // above. A big fence lets moderately hard circuits draw 4-16-slice
  // plans; a drop-in MPI user could then land in slices < ranks, the
  // filed idle-rank segfault's home. A floor of 1 guarantees every rank
  // work, is vacuous at one rank, and by construction only ever ADDS
  // slicing (applied after the peak constraint). Explicit
  // AER_TN_MIN_SLICES_PER_RANK=0 restores the old behavior.
  static bool checked = false;
  static uint64_t cached = 1;
  if (!checked) {
    const char *val = std::getenv("AER_TN_MIN_SLICES_PER_RANK");
    if (val != nullptr) {
      char *end = nullptr;
      long long parsed = std::strtoll(val, &end, 10);
      if (end != val && parsed >= 0) {
        cached = static_cast<uint64_t>(parsed);
      } else {
        fprintf(stderr,
                "[AER_TN_PATH] warning: AER_TN_MIN_SLICES_PER_RANK='%s' is not "
                "a non-negative integer; using default %llu.\n",
                val, (unsigned long long)cached);
      }
    }
    checked = true;
  }
  if (tn_plan_forge()) {
    uint64_t target =
        TensorNetPathMem::env_u64("AER_TN_PLAN_TARGET_SLICES", 16);
    int np = 1;
#ifdef AER_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &np);
#endif
    uint64_t per_rank =
        (target + static_cast<uint64_t>(np) - 1) / static_cast<uint64_t>(np);
    if (per_rank > cached)
      return per_rank;
  }
  return cached;
}

static bool force_slicing() {
  static bool checked = false;
  static bool enabled = false;
  if (!checked) {
    const char *val = std::getenv("AER_TN_FORCE_SLICING");
    enabled = (val != nullptr && std::string(val) == "1");
    checked = true;
  }
  return enabled;
}

//=============================================================================
// Data structures
//=============================================================================

struct TensorSpec {
  std::vector<int32_t> modes;
  std::vector<int64_t> extents;

  int64_t num_elements() const {
    int64_t n = 1;
    for (size_t i = 0; i < extents.size(); i++)
      n *= extents[i];
    return n;
  }
};

struct NetworkDescription {
  std::vector<TensorSpec> tensors;
  std::vector<int32_t> output_modes;
  std::vector<int64_t> output_extents;

  std::map<int32_t, int64_t> build_size_dict() const {
    std::map<int32_t, int64_t> sizes;
    for (size_t t = 0; t < tensors.size(); t++) {
      for (size_t i = 0; i < tensors[t].modes.size(); i++) {
        sizes[tensors[t].modes[i]] = tensors[t].extents[i];
      }
    }
    return sizes;
  }
};

struct SliceInfo {
  int32_t mode;
  int64_t extent;
};

struct ContractionStep {
  uint64_t left;
  uint64_t right;
};

struct ContractionPlan {
  std::vector<ContractionStep> steps;
  std::vector<SliceInfo> sliced;
  uint64_t num_slices;
  double total_flops;
  int64_t peak_intermediate_elements;

  std::vector<int64_t> serialize() const {
    std::vector<int64_t> data;
    int64_t num_steps = static_cast<int64_t>(steps.size());
    int64_t num_sliced_modes = static_cast<int64_t>(sliced.size());

    data.push_back(num_steps);
    data.push_back(num_sliced_modes);
    data.push_back(static_cast<int64_t>(num_slices));

    int64_t flops_bits;
    std::memcpy(&flops_bits, &total_flops, sizeof(double));
    data.push_back(flops_bits);
    data.push_back(peak_intermediate_elements);

    for (size_t i = 0; i < steps.size(); i++) {
      data.push_back(static_cast<int64_t>(steps[i].left));
      data.push_back(static_cast<int64_t>(steps[i].right));
    }
    for (size_t i = 0; i < sliced.size(); i++) {
      data.push_back(static_cast<int64_t>(sliced[i].mode));
      data.push_back(sliced[i].extent);
    }
    return data;
  }

  static ContractionPlan deserialize(const std::vector<int64_t> &data) {
    ContractionPlan plan;
    size_t pos = 0;
    int64_t num_steps = data[pos++];
    int64_t num_sliced_modes = data[pos++];
    plan.num_slices = static_cast<uint64_t>(data[pos++]);
    std::memcpy(&plan.total_flops, &data[pos], sizeof(double));
    pos++;
    plan.peak_intermediate_elements = data[pos++];

    plan.steps.resize(num_steps);
    for (int64_t i = 0; i < num_steps; i++) {
      plan.steps[i].left = static_cast<uint64_t>(data[pos++]);
      plan.steps[i].right = static_cast<uint64_t>(data[pos++]);
    }
    plan.sliced.resize(num_sliced_modes);
    for (int64_t i = 0; i < num_sliced_modes; i++) {
      plan.sliced[i].mode = static_cast<int32_t>(data[pos++]);
      plan.sliced[i].extent = data[pos++];
    }
    return plan;
  }
};

//=============================================================================
// aer-0027: cross-instruction contraction-plan cache
//=============================================================================
//
// WHY
// A contractor is created and destroyed per instruction (create_contractor /
// delete contractor throughout tensor_net.hpp), and both gpu_mgr_ and plan_ are
// contractor MEMBERS. So every evaluation re-runs the cotengra search from
// scratch. Measured on grid n=64 p=2 (job 20508223): t_path 3393 ms of 6666 ms
// setup, against t_contract 8.49 ms. The contraction is 0.127% of the cost.
//
// The plan depends only on the network TOPOLOGY: TensorSpec carries {modes,
// extents} and no values, so the identical plan is valid for every choice of
// gate angles. That is what makes this worth caching -- a QAOA angle sweep
// re-contracts one topology thousands of times.
//
// SCOPE: this caches the ContractionPlan only, which is plain integer data
// (steps, sliced, num_slices, total_flops, peak_intermediate_elements) and
// already round-trips through serialize()/deserialize() for the MPI broadcast.
// It holds NO device pointers and allocates NO VRAM. The hipTensor plan cache
// and the memory pool are deliberately NOT persisted here: those own GPU
// allocations whose lifetime is currently bounded by the contractor, and
// changing that is a separate problem with real use-after-free and eviction
// hazards. This half is ~50% of setup at a fraction of the risk.
//
// THE CORRECTNESS TRAP, AND WHY THE KEY IS CANONICAL
// Mode IDs are NOT stable. tensor_net.hpp assigns them from mode_index_, a
// monotonic counter on the TensorNet that advances as gates are applied
// (tensor_net.hpp:534, 840-845, 877-914). Two structurally identical cones
// built at different points in a circuit therefore carry DIFFERENT mode IDs.
// A key built from raw IDs would never hit, and the feature would silently do
// nothing while appearing to work. So modes are relabelled by first appearance
// in tensor order before hashing.
//
// The same trap bites the stored plan. ContractionStep holds {left, right},
// which are tensor POSITIONS and replay safely. But SliceInfo holds a MODE ID.
// Replaying a raw sliced mode into a network whose IDs differ would slice the
// wrong index and silently return a wrong answer. Slices are therefore stored
// as structural coordinates (tensor, position) and re-resolved on every hit --
// precisely what MPIParallelPathOptimizer does after its MPI_Bcast, for
// precisely this reason.
//
// MPI: the cache is consulted ONLY when nprocs_ == 1. create_optimizer()
// returns MPIParallelPathOptimizer under MPI, which runs an allreduce and a
// broadcast; if one rank hit the cache and another missed, the missing rank
// would block in a collective the other never enters. Rather than reason about
// whether canonical keys always agree across ranks, the multi-rank path simply
// does not use the cache. Single-rank is also the intended topology for the
// benchmark driver (SLURM fan-out, not MPI ranks).
//
// Off by default. AER_TN_PLAN_CACHE=1 enables; AER_TN_PLAN_CACHE_MAX caps
// entries (LRU). Host memory only.

static bool tn_plan_cache_enabled() {
  static bool checked = false;
  static bool cached = false;
  if (!checked) {
    // aer-0080: default ON. The in-process plan cache is a pure win
    // validated across every campaign since aer-0027; "0" disables for
    // A/B, matching the POOL_REUSE opt-out convention.
    const char *v = std::getenv("AER_TN_PLAN_CACHE");
    cached = !(v != nullptr && v[0] == '0' && v[1] == '\0');
    checked = true;
  }
  return cached;
}

static size_t tn_plan_cache_max() {
  static bool checked = false;
  static size_t cached = 64;
  if (!checked) {
    const char *v = std::getenv("AER_TN_PLAN_CACHE_MAX");
    if (v != nullptr) {
      char *end = nullptr;
      long long p = std::strtoll(v, &end, 10);
      if (end != v && p > 0)
        cached = static_cast<size_t>(p);
    }
    checked = true;
  }
  return cached;
}

// Mirrors the clamp in CotengPathOptimizer::find_path (target_elements =
// min(budget/element_size, slice_target_bytes()/element_size)). Keying on the
// CLAMPED value matters: the raw budget tracks free VRAM and jitters run to
// run, which would poison the key even though the clamped value is constant.
inline uint64_t plan_key_target_elements(uint64_t memory_budget,
                                         size_t element_size_bytes) {
  const uint64_t es = (element_size_bytes > 0) ? element_size_bytes : 1;
  const uint64_t from_budget = memory_budget / es;
  const uint64_t envelope = std::max<uint64_t>(1, slice_target_bytes() / es);
  return std::min(from_budget, envelope);
}

// Empty return means "do not cache this network" -- used when an output mode
// appears in no tensor and therefore cannot be canonicalised consistently.
// Refusing to key is always safe; a wrong key is not.
inline std::string canonical_network_key(const NetworkDescription &net,
                                         bool engaged,
                                         uint64_t target_elements,
                                         uint64_t seed,
                                         size_t element_size_bytes) {
  std::map<int32_t, int32_t> relabel;
  int32_t next_id = 0;
  std::ostringstream os;
  os << "T" << net.tensors.size();
  for (size_t t = 0; t < net.tensors.size(); t++) {
    const TensorSpec &ts = net.tensors[t];
    if (ts.modes.size() != ts.extents.size())
      return std::string();
    os << "|" << ts.modes.size();
    for (size_t i = 0; i < ts.modes.size(); i++) {
      std::map<int32_t, int32_t>::iterator it = relabel.find(ts.modes[i]);
      if (it == relabel.end())
        it = relabel.insert(std::make_pair(ts.modes[i], next_id++)).first;
      os << "," << it->second << ":" << ts.extents[i];
    }
  }
  // The output spec is part of the key. setup_pool_and_cache() writes the final
  // step's C descriptor in modes_out_ order, so a plan reused against a
  // different output contracts into the wrong shape. This is aer-0024's bug,
  // and across instructions its blast radius is larger.
  if (net.output_modes.size() != net.output_extents.size())
    return std::string();
  os << "|OUT" << net.output_modes.size();
  for (size_t i = 0; i < net.output_modes.size(); i++) {
    std::map<int32_t, int32_t>::const_iterator it =
        relabel.find(net.output_modes[i]);
    if (it == relabel.end())
      return std::string();
    os << "," << it->second << ":" << net.output_extents[i];
  }
  os << "|E" << (engaged ? 1 : 0) << "|TE" << target_elements << "|S" << seed
     << "|ES" << element_size_bytes;
  return os.str();
}

// sliced mode ID -> (tensor, position). Returns false if any sliced mode is not
// found, in which case the plan is simply not cached.
inline bool
sliced_modes_to_coords(const NetworkDescription &net,
                       const ContractionPlan &plan,
                       std::vector<std::pair<int64_t, int64_t>> &out) {
  out.clear();
  for (size_t s = 0; s < plan.sliced.size(); s++) {
    int64_t ft = -1, fp = -1;
    for (size_t t = 0; t < net.tensors.size() && ft < 0; t++) {
      const std::vector<int32_t> &m = net.tensors[t].modes;
      for (size_t p = 0; p < m.size(); p++) {
        if (m[p] == plan.sliced[s].mode) {
          ft = static_cast<int64_t>(t);
          fp = static_cast<int64_t>(p);
          break;
        }
      }
    }
    if (ft < 0)
      return false;
    out.push_back(std::make_pair(ft, fp));
  }
  return true;
}

// (tensor, position) -> this network's mode ID. Returns false rather than
// leaving a stale ID in place, so a caller can fall back to a fresh search.
inline bool
coords_to_sliced_modes(const NetworkDescription &net,
                       const std::vector<std::pair<int64_t, int64_t>> &coords,
                       ContractionPlan &plan) {
  if (coords.size() != plan.sliced.size())
    return false;
  for (size_t s = 0; s < coords.size(); s++) {
    const int64_t t = coords[s].first;
    const int64_t p = coords[s].second;
    if (t < 0 || static_cast<size_t>(t) >= net.tensors.size())
      return false;
    const std::vector<int32_t> &m = net.tensors[t].modes;
    if (p < 0 || static_cast<size_t>(p) >= m.size())
      return false;
    plan.sliced[s].mode = m[p];
  }
  return true;
}

class PlanCache {
public:
  struct Entry {
    ContractionPlan plan;
    std::vector<std::pair<int64_t, int64_t>> sliced_coords;
    uint64_t last_used;
    Entry() : last_used(0) {}
  };

  // Mutex-guarded: Aer may execute circuits on parallel threads, and this is
  // process-global state. The critical section is a map lookup plus a copy of
  // a few hundred integers, against a path search measured in seconds.
  bool get(const std::string &key, const NetworkDescription &net,
           ContractionPlan &out) {
    std::lock_guard<std::mutex> lock(mu_);
    std::unordered_map<std::string, Entry>::iterator it = map_.find(key);
    if (it == map_.end()) {
      misses_++;
      return false;
    }
    ContractionPlan candidate = it->second.plan;
    if (!coords_to_sliced_modes(net, it->second.sliced_coords, candidate)) {
      map_.erase(it);
      misses_++;
      return false;
    }
    it->second.last_used = ++tick_;
    out = candidate;
    hits_++;
    return true;
  }

  void put(const std::string &key, const NetworkDescription &net,
           const ContractionPlan &plan, size_t max_entries) {
    std::vector<std::pair<int64_t, int64_t>> coords;
    if (!sliced_modes_to_coords(net, plan, coords))
      return;
    std::lock_guard<std::mutex> lock(mu_);
    if (map_.size() >= max_entries && map_.find(key) == map_.end())
      evict_lru_locked();
    Entry e;
    e.plan = plan;
    e.sliced_coords = coords;
    e.last_used = ++tick_;
    map_[key] = e;
  }

  void stats(uint64_t &h, uint64_t &m, size_t &n) {
    std::lock_guard<std::mutex> lock(mu_);
    h = hits_;
    m = misses_;
    n = map_.size();
  }

private:
  void evict_lru_locked() {
    std::unordered_map<std::string, Entry>::iterator victim = map_.end();
    uint64_t best = std::numeric_limits<uint64_t>::max();
    for (std::unordered_map<std::string, Entry>::iterator it = map_.begin();
         it != map_.end(); ++it) {
      if (it->second.last_used < best) {
        best = it->second.last_used;
        victim = it;
      }
    }
    if (victim != map_.end())
      map_.erase(victim);
  }

  std::unordered_map<std::string, Entry> map_;
  std::mutex mu_;
  uint64_t tick_ = 0;
  uint64_t hits_ = 0;
  uint64_t misses_ = 0;
};

inline PlanCache &plan_cache_instance() {
  static PlanCache inst;   // C++11 guarantees thread-safe initialisation
  return inst;
}

//=============================================================================
// aer-0049: AER_TN_PLAN_FILE — capture-and-replay of one contraction plan
//=============================================================================
//
// WHY
// The per-rank cotengra search is non-deterministic (CPU probes confirmed no
// HyperOptimizer kwarg, including parallel=False with a fixed seed, makes it
// reproducible), and under MPI the min-FLOP plan is kept — which is the
// fewest-slice / most-tile one, steering plan selection toward the
// AER_TN_MAX_TILES ceiling even at a budget whose runnable window is proven
// (job 20995937: grid 6x5 p=2 clears both ceilings at 1 GiB per slice). Two
// consequences: no two runs share a plan denominator, so no performance
// comparison is valid; and a memory-bound distributed run is not RELIABLE at
// all, because the search will not consistently land a both-ceilings plan.
// Pinning the plan to a file fixes both.
//
// SEMANTICS
//   AER_TN_PLAN_FILE=<path>, unset by default (feature entirely off).
//   File absent  -> CAPTURE: the run searches normally; after the first fully
//                   successful setup (slice ceiling, tile ceiling, prebuild all
//                   passed) rank 0 writes the plan. Written only on success, so
//                   a run that trips a ceiling leaves no file and the next
//                   capture attempt draws a fresh (scattered) search.
//   File present -> REPLAY: the plan is read instead of searched, on every
//                   rank, bypassing both the search and the cross-rank MINLOC
//                   selection. The file is keyed to ONE network topology; a
//                   mismatched network is refused with a warning and searched
//                   fresh (never silently replayed).
//
// FORMAT: plain text over ContractionPlan::serialize()'s int64 vector — the
// MPI-broadcast-proven representation, no new layout. Sliced modes are stored
// as structural (tensor, position) coordinates and re-resolved on load with
// coords_to_sliced_modes(), exactly as PlanCache does, because raw mode IDs
// are not guaranteed stable (see THE CORRECTNESS TRAP above). The key is the
// canonical relabelled topology with target_elements=0 and seed=0: a pinned
// plan is deliberately valid across budget jitter and seed — replaying a plan
// whose peak exceeds the actual budget fails loudly at allocation, which is
// the correct failure.
//
// MPI: rank 0 reads the file and broadcasts the raw bytes; every rank parses
// identical bytes against its own (identical) network, so the verdict is
// deterministic on all ranks and needs no extra agreement collective. The
// load is called only under conditions that are uniform across ranks (env
// value, the collectively-agreed cache verdict, the collectively-set bypass
// flag), so the broadcast cannot desync.

static std::string tn_plan_file() {
  static bool checked = false;
  static std::string cached;
  if (!checked) {
    const char *v = std::getenv("AER_TN_PLAN_FILE");
    if (v != nullptr && v[0] != '\0')
      cached = v;
    checked = true;
  }
  return cached;
}

// aer-0064: AER_TN_PLAN_DIR — the plan LIBRARY. AER_TN_PLAN_FILE stores one
// plan for one topology, which forced launchers evaluating many classes to
// either forgo persistence for all but one class or to shard one class per
// process purely to own a file name. The measured stake (job 21381487,
// rand3 n=10000 p=4): 81 classes, 2377 s of path search against 10 s of
// contraction, and a captured plan replaying in 6 ms where its search took
// 26 s. With a directory, ONE process evaluating all classes captures every
// plan on first contact and replays all of them forever after.
//
// Semantics: each network resolves to <dir>/plan_<fnv1a64(key)>.ctg, where
// key is the same canonical topology key AER_TN_PLAN_FILE uses (aer-0057
// pinning included), so a file is bit-identical to what AER_TN_PLAN_FILE
// would have written for that topology — only the naming is new. Capture,
// replay, the capture-once rule, the AER_TN_PLAN_FILE_MAX_TILES gate, the
// bypass flag and every refusal message behave exactly as for
// AER_TN_PLAN_FILE. Precedence: AER_TN_PLAN_FILE, when set, wins — it is
// the explicit, single-topology instruction and existing scripts must not
// change meaning; the directory is consulted only when the file knob is
// unset. The directory must exist; a missing directory fails the capture
// write with the existing "cannot open" message and costs nothing else.
//
// Concurrency: concurrent processes capturing DIFFERENT topologies write
// different file names and never interact. Concurrent captures of the SAME
// topology are serialized by the per-process temp name (see aer-0064 in
// maybe_write_plan_file); last rename wins with a self-consistent file.
static std::string tn_plan_dir() {
  static bool checked = false;
  static std::string cached;
  if (!checked) {
    const char *v = std::getenv("AER_TN_PLAN_DIR");
    if (v != nullptr && v[0] != '\0') {
      cached = v;
      // normalize: exactly one separator will be appended at use
      while (cached.size() > 1 && cached.back() == '/')
        cached.pop_back();
    }
    checked = true;
  }
  return cached;
}

// FNV-1a 64-bit over the canonical key string. Chosen because it is
// dependency-free, stable across platforms and compilers (the key is ASCII
// text), and collisions across the handful-to-hundreds of topologies a
// campaign holds are vanishingly unlikely; a collision would surface as the
// existing loud "does not replay onto this network" refusal, never as a
// wrong plan, because the full key is verified inside the file on decode.
static std::string tn_plan_key_hash(const std::string &key) {
  uint64_t h = 14695981039346656037ull;
  for (size_t i = 0; i < key.size(); i++) {
    h ^= static_cast<uint64_t>(static_cast<unsigned char>(key[i]));
    h *= 1099511628211ull;
  }
  char buf[17];
  snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
  return std::string(buf);
}

// Resolve the plan path for a network key under the precedence rule above.
// Empty means "no plan persistence configured". Callers must already hold a
// non-empty key (the not-keyable refusal happens before path resolution in
// directory mode, since the path cannot exist without a key).
//
// aer-0068: width tag. A captured plan keeps its slice count, and slicing is
// decided against the capture-time rank count (MIN_SLICES_PER_RANK forces
// >= 1 slice per rank AT PLAN TIME only) -- so a plan library is inherently
// width-locked: replaying an 8-slice plan at 16 ranks lands in the
// slices < ranks idle configuration (defined and loud since aer-0067, but
// half the node does nothing), and replaying it at 1 rank drags the forced
// 8-way recompute overhead forever. Directory entries therefore carry the
// capture width in the name: plan_<hash>.r<N>.ctg for N > 1, untagged for
// single-rank captures. Loads try the width-exact name first and fall back
// to the untagged name, so pre-0068 libraries (all untagged) keep working
// at every width they worked at before; what changes is that a run at a NEW
// width misses, searches AT that width, and captures its own entry instead
// of silently inheriting a mismatched one. AER_TN_PLAN_FILE stays width-
// blind: an explicit file is an explicit instruction.
static std::string tn_plan_path_for_key(const std::string &key,
                                        int width = 0) {
  const std::string file = tn_plan_file();
  if (!file.empty())
    return file;
  const std::string dir = tn_plan_dir();
  if (dir.empty())
    return std::string();
  std::string name = dir + "/plan_" + tn_plan_key_hash(key);
  if (width > 1)
    name += ".r" + std::to_string(width);
  return name + ".ctg";
}

// aer-0051: optional capture-quality gate. The scattered search can land
// anywhere in a config's runnable window (the 6x5 p=2 capture landed 64
// slices / 152,390 tiles where the sizer's sweet-spot row was 2048 / 24,261
// at the same budget), and a tile-heavy plan is a maxGEMM-poor,
// throughput-modest one. When set > 0, a fully-successful setup whose
// prebuild built MORE tiles than this refuses to persist the plan -- the run
// still completes (the plan cleared the hard AER_TN_MAX_TILES ceiling), but
// the file is not written, so re-submitting capture keeps drawing until a
// plan at or under the target lands. 0 (default) = off: any
// both-ceilings-clearing plan is captured, exactly the aer-0049 behaviour.
static uint64_t tn_plan_file_max_tiles() {
  static bool checked = false;
  static uint64_t cached = 0;
  if (!checked) {
    const char *v = std::getenv("AER_TN_PLAN_FILE_MAX_TILES");
    if (v != nullptr) {
      char *end = nullptr;
      long long parsed = std::strtoll(v, &end, 10);
      if (end != v && parsed >= 0) {
        cached = static_cast<uint64_t>(parsed);
      } else {
        fprintf(stderr,
                "[AER_TN_PLAN_FILE] warning: AER_TN_PLAN_FILE_MAX_TILES='%s' "
                "is not a non-negative integer; using default %llu.\n",
                v, (unsigned long long)cached);
      }
    }
    checked = true;
  }
  return cached;
}

// The file's network identity: canonical relabelled topology + output + tiling
// engagement + element size, with target_elements and seed pinned to 0 so the
// key does not move with the jittering VRAM budget or the path seed. Empty
// return means "not keyable" (same rule as the plan cache): refuse to capture
// or replay rather than risk a wrong match.
inline std::string plan_file_network_key(const NetworkDescription &net,
                                         bool engaged,
                                         size_t element_size_bytes) {
  // aer-0057: the tiling-engagement bit is PINNED TRUE in the file key,
  // regardless of the caller's `engaged`. Rationale: every plan file
  // captured before the hipTensor removal was written under the AUTO
  // tiling latch (engaged=true), and the removal makes engagement
  // permanently false at runtime -- without this pin, every banked plan
  // (grid6x5p2.plan, census_amp_5x5p2.plan, p4_maxcone.plan and any user's)
  // would key-mismatch and silently fall back to a fresh search, destroying
  // reproducibility. Tiling no longer affects execution in any way, so the
  // bit carries no information; pinning it preserves byte-compatibility
  // with every existing capture. New captures write the same pinned bit.
  (void)engaged;
  return canonical_network_key(net, /*engaged=*/true, /*target_elements=*/0,
                               /*seed=*/0, element_size_bytes);
}

inline bool plan_file_encode(const NetworkDescription &net,
                             const std::string &key,
                             const ContractionPlan &plan, std::string &out) {
  std::vector<std::pair<int64_t, int64_t>> coords;
  if (!sliced_modes_to_coords(net, plan, coords))
    return false;
  const std::vector<int64_t> data = plan.serialize();
  std::ostringstream os;
  os << "AER_TN_PLAN_FILE v1\n";
  os << "key " << key << "\n";
  os << "coords " << coords.size() << "\n";
  for (size_t i = 0; i < coords.size(); i++)
    os << coords[i].first << " " << coords[i].second << "\n";
  os << "plan " << data.size() << "\n";
  for (size_t i = 0; i < data.size(); i++)
    os << data[i] << "\n";
  os << "end\n";
  out = os.str();
  return true;
}

// Parse + validate + resolve in one step. Never trusts lengths from the file:
// the serialize() layout invariant (5 header values + 2 per step + 2 per
// sliced mode) is checked BEFORE deserialize() indexes the vector, and the
// coords count must match the plan's sliced count. On any failure `why` names
// the reason and the caller falls back to a fresh search.
inline bool plan_file_decode(const std::string &text,
                             const NetworkDescription &net,
                             const std::string &expected_key,
                             ContractionPlan &out, std::string &why) {
  std::istringstream is(text);
  std::string tag, ver, kw, key;
  if (!(is >> tag >> ver) || tag != "AER_TN_PLAN_FILE" || ver != "v1") {
    why = "not an AER_TN_PLAN_FILE v1 file";
    return false;
  }
  if (!(is >> kw >> key) || kw != "key") {
    why = "missing key line";
    return false;
  }
  if (key != expected_key) {
    why = "network key mismatch (different circuit topology, output, tiling "
          "engagement, or element size than the plan was captured for)";
    return false;
  }
  size_t ncoords = 0;
  if (!(is >> kw >> ncoords) || kw != "coords") {
    why = "missing coords section";
    return false;
  }
  std::vector<std::pair<int64_t, int64_t>> coords(ncoords);
  for (size_t i = 0; i < ncoords; i++) {
    if (!(is >> coords[i].first >> coords[i].second)) {
      why = "truncated coords section";
      return false;
    }
  }
  size_t nvals = 0;
  if (!(is >> kw >> nvals) || kw != "plan") {
    why = "missing plan section";
    return false;
  }
  std::vector<int64_t> data(nvals);
  for (size_t i = 0; i < nvals; i++) {
    if (!(is >> data[i])) {
      why = "truncated plan section";
      return false;
    }
  }
  if (!(is >> kw) || kw != "end") {
    why = "missing end marker";
    return false;
  }
  if (data.size() < 5) {
    why = "plan section shorter than the serialize() header";
    return false;
  }
  const int64_t num_steps = data[0];
  const int64_t num_sliced = data[1];
  if (num_steps < 0 || num_sliced < 0 ||
      data.size() != static_cast<size_t>(5 + 2 * num_steps + 2 * num_sliced)) {
    why = "plan section length inconsistent with its step/sliced counts";
    return false;
  }
  if (static_cast<size_t>(num_sliced) != coords.size()) {
    why = "coords count does not match the plan's sliced count";
    return false;
  }
  ContractionPlan plan = ContractionPlan::deserialize(data);
  if (!coords_to_sliced_modes(net, coords, plan)) {
    why = "sliced coordinates do not resolve on this network";
    return false;
  }
  out = plan;
  return true;
}

//=============================================================================
// PathOptimizer interface
//=============================================================================

class PathOptimizer {
public:
  virtual ~PathOptimizer() = default;
  // aer-0065: trial-share hook. (shard, num_shards) divides the repeats
  // budget; the default is the whole budget. A no-op base so optimizers that
  // have no trial budget (or tests) need not care; MPIParallelPathOptimizer
  // is the only caller. Idempotent -- set before every find_path delegation.
  // aer-0066: returns whether the share was ACCEPTED, so the caller's log can
  // tell cooperative sharding (Coteng) from the legacy full-budget ensemble
  // (the Greedy fallback, which has restarts, not a shardable trial budget)
  // instead of announcing a division that is not happening.
  virtual bool set_trial_share(int shard, int num_shards) {
    (void)shard;
    (void)num_shards;
    return false;
  }
  // tiling_available: when false, the planner's >12-free-mode gate throws
  // NeedsTilingException (AUTO) or a hard refusal (OFF, decided by the caller);
  // when true, oversized outputs are admitted because the contractor will tile
  // them. The driver (setup_contraction) sets this per pass so pass 1 and pass 2
  // are explicitly controlled rather than each gate re-reading the environment.
  virtual ContractionPlan find_path(const NetworkDescription &network,
                                    uint64_t memory_limit_bytes,
                                    uint64_t seed,
                                    bool tiling_available) = 0;
};

//=============================================================================
// CotengPathOptimizer — cotengra via pybind11
//
//   We need to hand cotengra an einsum-style description of the network,
//   with a string label for each unique int32_t mode. cotengra/opt_einsum
//   accept any hashable label; the canonical way to scale beyond the
//   52-char a-zA-Z alphabet is cotengra.get_symbol(i), which returns a
//   unique unicode character for any non-negative i — cotengra's own
//   utils.lattice_equation uses this for 100+ index networks. We store
//   labels as std::string (UTF-8) so a unicode char fits unchanged.
//=============================================================================

#ifdef AER_HIPTENSOR

// aer-0099: subtree_reconfigure_ mutates before it recontracts -- cotengra
// removes every branch node of the chosen subtree (core.py:1925) and only
// then rebuilds it via contract_nodes -> find_path (core.py:1931 -> 1377).
// An exception inside that window (job 21479115: OverflowError at
// path_basic.py:232, an int >= 2^1024 cannot convert to float) therefore
// exits with interior nodes deleted and never re-added; every later
// traversal dies on the missing children -- the KeyError: frozenset({...})
// that closed 21479115. "Continuing with the tree as found" was continuing
// with a corrupted tree. This helper makes reconfiguration transactional:
// snapshot via tree.copy() (set_state_from copies the children/info maps
// top-level, core.py:300-347, so removals and insertions on the original
// cannot reach the snapshot), run, and on ANY raise rebind the caller's
// handle to the snapshot. If even the snapshot cannot be taken (e.g. under
// the search RLIMIT), the reconfiguration is SKIPPED outright -- an
// optimisation hint must never risk the tree (the distribution-floor rule).
// Returns true only when the reconfiguration ran and stuck.
static inline bool reconf_tree_transactional(py::object &tree,
                                             const char *site) {
  py::object backup;
  try {
    backup = tree.attr("copy")();
  } catch (py::error_already_set &ce) {
    fprintf(stderr,
            "[AER_TN_PATH] warning: could not snapshot the tree before %s "
            "(%s); reconfiguration skipped, tree unchanged\n",
            site, ce.what());
    ce.discard_as_unraisable("AER_TN_PATH reconf snapshot");
    return false;
  }
  try {
    tree.attr("subtree_reconfigure_")();
    return true;
  } catch (py::error_already_set &re) {
    fprintf(stderr,
            "[AER_TN_PATH] warning: %s raised: %s -- restored the "
            "pre-reconfiguration tree\n",
            site, re.what());
    re.discard_as_unraisable("AER_TN_PATH reconf restore");
    tree = backup;
    return false;
  }
}

class CotengPathOptimizer : public PathOptimizer {
  std::string minimize_;
  int max_repeats_;
  double max_time_;
  std::string preset_;
  size_t element_size_bytes_;
  // aer-0065: this instance's slice of the trial budget (set_trial_share).
  int trial_shard_ = 0;
  int trial_shards_ = 1;
  // aer-0035: distribution floor on the slice count, supplied by the
  // contractor because only it knows nprocs_. 1 means no floor.
  uint64_t min_slices_ = 1;

  // Mode label mapping: int32_t <-> UTF-8 string (built per find_path call).
  // Labels come from cotengra.get_symbol(i): 'a'-'z' for i<26, 'A'-'Z' for
  // 26<=i<52, then unicode chars thereafter. Stored as std::string because
  // the unicode chars are multi-byte in UTF-8.
  std::map<int32_t, std::string> mode_to_label_;
  std::map<std::string, int32_t> label_to_mode_;

  void build_mode_mapping(const NetworkDescription &network) {
    mode_to_label_.clear();
    label_to_mode_.clear();

    py::gil_scoped_acquire gil;
    py::object get_symbol =
        py::module_::import("cotengra").attr("get_symbol");

    int next_idx = 0;
    auto assign = [&](int32_t mode) {
      if (mode_to_label_.find(mode) != mode_to_label_.end()) return;
      std::string label = get_symbol(next_idx++).cast<std::string>();
      mode_to_label_[mode] = label;
      label_to_mode_[label] = mode;
    };

    for (size_t t = 0; t < network.tensors.size(); t++)
      for (size_t m = 0; m < network.tensors[t].modes.size(); m++)
        assign(network.tensors[t].modes[m]);
    for (size_t m = 0; m < network.output_modes.size(); m++)
      assign(network.output_modes[m]);
  }

public:
  CotengPathOptimizer(const std::string &minimize = "combo",
                      int max_repeats = -1, double max_time = -1.0,
                      const std::string &preset = "hyper",
                      size_t element_size_bytes = 16,
                      uint64_t min_slices = 1)
      : minimize_(minimize),
        max_repeats_(max_repeats > 0 ? max_repeats : path_max_repeats()),
        max_time_(max_time > 0.0 ? max_time : path_max_time()),
        preset_(preset), element_size_bytes_(element_size_bytes),
        min_slices_(min_slices > 0 ? min_slices : 1) {}

  // aer-0065: ceil-split so every shard gets >= 1 trial and the shares sum
  // exactly to max_repeats_ (the first budget%shards shards carry one extra).
  bool set_trial_share(int shard, int num_shards) override {
    trial_shard_ = (shard >= 0) ? shard : 0;
    trial_shards_ = (num_shards >= 1) ? num_shards : 1;
    return true;
  }

  int effective_max_repeats() const {
    if (trial_shards_ <= 1)
      return max_repeats_;
    int base = max_repeats_ / trial_shards_;
    int extra = (trial_shard_ < (max_repeats_ % trial_shards_)) ? 1 : 0;
    int n = base + extra;
    return n >= 1 ? n : 1;
  }

  ContractionPlan find_path(const NetworkDescription &network,
                            uint64_t memory_limit_bytes,
                            uint64_t seed,
                            bool tiling_available) override {
    py::gil_scoped_acquire gil;
    auto ctg = py::module_::import("cotengra");

    // Build int → string label mapping
    build_mode_mapping(network);

    // Convert inputs: each tensor's modes → tuple of label strings
    py::list inputs;
    for (size_t i = 0; i < network.tensors.size(); i++) {
      py::list mode_labels;
      for (size_t j = 0; j < network.tensors[i].modes.size(); j++) {
        mode_labels.append(
            py::str(mode_to_label_[network.tensors[i].modes[j]]));
      }
      inputs.append(py::tuple(mode_labels));
    }

    // Convert output modes
    py::list output_labels;
    for (size_t i = 0; i < network.output_modes.size(); i++) {
      output_labels.append(py::str(mode_to_label_[network.output_modes[i]]));
    }
    py::tuple output = py::tuple(output_labels);

    // Convert size_dict: label string keys
    py::dict sizes;
    auto size_dict = network.build_size_dict();
    for (auto it = size_dict.begin(); it != size_dict.end(); ++it) {
      sizes[py::str(mode_to_label_[it->first])] = it->second;
    }

    // Per-slice peak-intermediate budget for cotengra's target_size, in
    // elements. Two ceilings apply: the device-memory budget (we must fit
    // at all) and the m6n6k6 kernel-envelope budget (every per-step
    // descriptor must stay inside hipTensor's only kernel shape). The
    // Two bounds on the per-slice peak intermediate; the smaller wins:
    //   (1) device memory (must fit VRAM at all);
    //   (2) the optional AER_TN_SLICE_TARGET_BYTES envelope -- a user knob to
    //       slice HARDER for less per-slice memory. Its default is slack (2 MB),
    //       so it does not bind unless set.
    //
    // The per-slice peak is NO LONGER clamped to the m6n6k6 tiled-operand
    // volume (the removed max_tiled_elements() clamp). Before operand staging (aer-0010/0011)
    // that clamp was mandatory: it sliced the peak down until every tiled
    // step's strided-view descriptor stayed inside the stride hipTensor/CK
    // could build a plan on. Staging removed that requirement -- a tiled
    // sub-block whose free-mode stride reaches ck_stage_stride_ceiling() is
    // packed into contiguous scratch, contracted, and scattered back, so CK
    // never sees the over-ceiling strided view. Clamping to the removed max_tiled_elements clamp
    // here only forced a needlessly high slice count (the slice-grind): the
    // peak is now governed by memory, and any resulting over-ceiling tiled
    // descriptor is carried by tiling + staging. Under AUTO the first oversized
    // step in prebuild (setup_pool_and_cache) throws NeedsTilingException and
    // the driver re-plans once with tiling engaged; under OFF an oversized step
    // still refuses (unchanged), since a peak above one m6n6k6 kernel cannot be
    // contracted without tiling regardless of this clamp. This is what lets a
    // hard circuit pick a low-slice plan automatically -- drop-in, no flags.
    //
    // The output-feasibility gate below uses output_tiling_ceiling() as the
    // hard cap on the (unsliceable) OUTPUT tensor -- decoupled from the
    // per-slice peak target so tuning slice count does not
    // move the accepted output rank, and vice versa.
    uint64_t target_elements = memory_limit_bytes / element_size_bytes_;
    uint64_t envelope_elements =
        std::max<uint64_t>(1, slice_target_bytes() / element_size_bytes_);
    target_elements = std::min(target_elements, envelope_elements);

    // --- Output feasibility & the m6n6k6 tiling ceiling -------------------
    //
    // Output (open) modes can NEVER be sliced (allow_outer=false; the sliced
    // engine sums slices, which is correct only for summed bonds), so the
    // output tensor's own size is a hard lower bound on the per-slice peak. By
    // output size, three regimes:
    uint64_t output_elements = 1;
    for (auto e : network.output_extents)
      output_elements *= static_cast<uint64_t>(e);

    const uint64_t single_kernel_elements =
        static_cast<uint64_t>(1) << kMaxFreeModes; // 4096: one m6n6k6 result

    if (output_elements > single_kernel_elements) {
      // The output cannot come from a single m6n6k6 result -- the final step
      // must be TILED. Feasible only if the output fits the tiling ceiling,
      // because its open modes cannot be sliced to shrink it.
      if (output_elements > output_tiling_ceiling()) {
        std::ostringstream msg;
        msg << "CotengPathOptimizer: requested output has "
            << network.output_modes.size() << " open modes ("
            << output_elements
            << " elements), exceeding the output tiling ceiling ("
            << output_tiling_ceiling()
            << " elements). Output (open) modes cannot be sliced, so this "
               "output rank exceeds m6n6k6 tiling -- request a lower-rank "
               "output (expectation value, amplitude, or a smaller reduced "
               "density matrix). If larger outputs are known CK-safe on this "
               "hipTensor build, raise AER_TN_MAX_OUTPUT_ELEMENTS after "
               "validation.";
        throw std::runtime_error(msg.str());
      }
      if (!tiling_available) {
        // Tileable, but tiling is not engaged this pass. AUTO: signal the
        // driver to re-plan once with tiling on. OFF: refuse with the reason.
        std::ostringstream msg;
        msg << "CotengPathOptimizer: requested output has "
            << network.output_modes.size() << " open modes, more than the "
            << kMaxFreeModes
            << " a single m6n6k6 step can produce. Enable mode tiling "
               "(AER_TN_TILING=on or auto) to decompose the oversized output "
               "step into m6n6k6 sub-contractions, or request a lower-rank "
               "output.";
        if (tn_tiling_mode() == TilingMode::Auto)
          throw NeedsTilingException(NeedsTilingException::Gate::Planner,
                                     msg.str());
        throw std::runtime_error(msg.str());
      }
      // else: tileable AND tiling engaged -- fall through; the output step is
      // tiled and the clamp below pins the per-slice peak to the (CK-safe)
      // output size.
    }

    // Output-size clamp: slicing can't shrink the output, so the per-slice
    // peak can't drop below it. Raise the target to at least the output size.
    // The checks above guarantee output_elements <= output_tiling_ceiling(), so
    // the target stays inside a CK-safe tiled-output volume.
    if (output_elements > target_elements) {
      if (path_verbose()) {
        fprintf(stderr,
                "[AER_TN_PATH] target_size raised %llu -> %llu elements: "
                "output tensor cannot be sliced (allow_outer=false)\n",
                (unsigned long long)target_elements,
                (unsigned long long)output_elements);
      }
      target_elements = output_elements;
    }

    py::object tree;

    // aer-0090: cap the search's memory. Jobs 21474687-691 (post-0089): at
    // p>=7 essentially every greedy/labels draw on the real network goes
    // degenerate, and the trial then dies at a float conversion (cotengra
    // warns "Trial error: int too large to convert to float" and discards)
    // or, worse, in its stats build -- 158 GiB on one rank at T=49070
    // before any conversion was reached. No model can price a quantity
    // that depends on the drawn tree, so the OS enforces the budget: the
    // process's data segment is soft-capped for the duration of the search
    // (probe and reconf included), turning a monster trial into a
    // MemoryError that cotengra's own per-trial handler absorbs. Default
    // cap: this rank's share of the node budget. AER_TN_PATH_TRIAL_MEM_MB
    // overrides in MB; 0 disables.
    uint64_t search_cap_bytes = 0;
    {
      const char *cap_env = std::getenv("AER_TN_PATH_TRIAL_MEM_MB");
      const bool cap_disabled =
          cap_env != nullptr && cap_env[0] == '0' && cap_env[1] == '\0';
      if (!cap_disabled) {
        uint64_t mb = 0;
        if (cap_env != nullptr && TensorNetPathMem::parse_u64(cap_env, mb) &&
            mb > 0) {
          search_cap_bytes = mb * 1024ull * 1024ull;
        } else if (cap_env == nullptr) {
          const uint64_t budget = TensorNetPathMem::node_mem_budget_bytes(
              "/sys/fs/cgroup", "/sys/fs/cgroup/memory", "/proc/self/cgroup",
              "/proc/meminfo");
          long ranks = TensorNetPathMem::local_ranks_sharing_node();
          if (ranks < 1)
            ranks = 1;
          if (budget > 0)
            search_cap_bytes = budget / static_cast<uint64_t>(ranks);
        }
      }
    }
    auto search_mem_cap =
        std::make_unique<TensorNetPathMem::ScopedRlimitData>(search_cap_bytes);
    if (search_mem_cap->armed())
      fprintf(stderr,
              "[AER_TN_PATH] search memory cap: %llu MB (RLIMIT_DATA, "
              "per-rank share; AER_TN_PATH_TRIAL_MEM_MB overrides, 0 "
              "disables)\n",
              (unsigned long long)(search_cap_bytes >> 20));
    else
      fprintf(stderr,
              "[AER_TN_PATH] search memory cap: off (%s)\n",
              search_cap_bytes == 0
                  ? "disabled or no memory budget signal"
                  : "existing limit already tighter, or setrlimit refused");

    // aer-0091/0096: while the cap is armed, cotengrust must not run. Rust
    // ABORTS the process on allocation failure ("memory allocation of N
    // bytes failed", SIGABRT) instead of raising -- jobs 21476058/21476059
    // died exactly this way. The 0091 shim replaced sys.modules only, and
    // job 21478441's fault-handler backtrace proved that insufficient: at
    // IMPORT TIME cotengra freezes the resolved Rust function into a
    // module-level partial (path_greedy.py:7, ssa_greedy_optimize =
    // functools.partial(get_optimize_greedy(), use_ssa=True)), and
    // trial_greedy calls that captured object forever after -- no module
    // swap can reach a captured function. aer-0097 extends the reach after
    // an audit prompted by the operator: THREE more captures exist as
    // module-level preset INSTANCES constructed at import
    // (presets.greedy_optimize, optimal_optimize, optimal_outer_optimize;
    // each stores the resolved function in _optimize_fn at __init__,
    // path_basic.py 1320/1616), all sentinel-proven to hold Rust under the
    // container's import topology; the 0096 sweep was inert against them
    // (wrong attribute, and no module-level instance carries
    // _optimize_optimal_fn). The sweep below rebinds by attribute and
    // class across presets/path_basic/path_greedy -- validated: 3
    // instances rebound, all resolving pure Python afterwards. Off the
    // forge path today (the fork calls the classes directly, and
    // subtree_reconfigure constructs a fresh OptimalOptimizer per call,
    // core.py 1865, which resolves through the cleared getter), but a
    // loaded footgun for any preset-string use in-process. So the block now
    // does all of it: install a module named cotengrust whose functions
    // ARE cotengra's own Python implementations (find_spec succeeds, the
    // import succeeds), clear the three lru-cached getters so every future
    // resolution lands on the shim, rebind path_greedy's frozen partial
    // directly to the Python implementation, and sweep preset instances
    // for a captured optimal. Validated against a reproduction of the
    // container's import topology: a sentinel "cotengrust" installed
    // FIRST so the import-time capture happens (the sentinel fires,
    // proving the trap), then this exact block, after which the partial,
    // a full hyper greedy search, and every getter resolve pure Python
    // with zero sentinel hits. The dispatch getters are lru_cached, so
    // this is applied once and holds for the process lifetime -- the safe
    // direction. AER_TN_PATH_ALLOW_RUST=1 opts out.
    if (search_mem_cap->armed()) {
      const char *allow_rust = std::getenv("AER_TN_PATH_ALLOW_RUST");
      const bool rust_ok =
          allow_rust != nullptr && allow_rust[0] == '1' && allow_rust[1] == '\0';
      static bool rust_block_done = false;
      if (!rust_ok && !rust_block_done) {
        try {
          py::exec(R"(
import sys, types, importlib.machinery, functools
from cotengra.pathfinders import path_basic as _aer_pb
from cotengra.pathfinders import path_greedy as _aer_pg
from cotengra import presets as _aer_presets
_aer_m = types.ModuleType("cotengrust")
_aer_m.__spec__ = importlib.machinery.ModuleSpec("cotengrust", loader=None)
_aer_m.optimize_greedy = _aer_pb.optimize_greedy
_aer_m.optimize_random_greedy_track_flops = (
    _aer_pb.optimize_random_greedy_track_flops)
_aer_m.optimize_optimal = _aer_pb.optimize_optimal
sys.modules["cotengrust"] = _aer_m
_aer_pb.get_optimize_greedy.cache_clear()
_aer_pb.get_optimize_random_greedy_track_flops.cache_clear()
_aer_pb.get_optimize_optimal.cache_clear()
_aer_pg.ssa_greedy_optimize = functools.partial(
    _aer_pb.optimize_greedy, use_ssa=True)
for _aer_mod in (_aer_presets, _aer_pb, _aer_pg):
    for _aer_obj in list(vars(_aer_mod).values()):
        if hasattr(_aer_obj, "_optimize_optimal_fn"):
            _aer_obj._optimize_optimal_fn = _aer_pb.optimize_optimal
        if hasattr(_aer_obj, "_optimize_fn"):
            if isinstance(_aer_obj, _aer_pb.OptimalOptimizer):
                _aer_obj._optimize_fn = _aer_pb.get_optimize_optimal("auto")
            elif isinstance(_aer_obj, _aer_pb.RandomGreedyOptimizer):
                _aer_obj._optimize_fn = (
                    _aer_pb.get_optimize_random_greedy_track_flops("auto"))
            elif isinstance(_aer_obj, _aer_pb.GreedyOptimizer):
                _aer_obj._optimize_fn = _aer_pb.get_optimize_greedy("auto")
)");
          rust_block_done = true;
          fprintf(stderr,
                  "[AER_TN_PATH] cotengrust disabled for capped searches "
                  "(Rust aborts at the RLIMIT instead of raising); "
                  "AER_TN_PATH_ALLOW_RUST=1 opts out\n");
        } catch (py::error_already_set &rb) {
          fprintf(stderr,
                  "[AER_TN_PATH] warning: could not disable cotengrust: %s "
                  "-- a Rust trial that hits the memory cap will abort the "
                  "process\n",
                  rb.what());
          rb.discard_as_unraisable("AER_TN_PATH cotengrust block");
        }
      } else if (rust_ok && !rust_block_done) {
        fprintf(stderr, "[AER_TN_PATH] cotengrust left enabled "
                        "(AER_TN_PATH_ALLOW_RUST=1); a Rust trial that hits "
                        "the memory cap aborts the process\n");
      }
    }

    if (preset_ == "random-greedy") {
      // RandomGreedyOptimizer takes seed as a direct named kwarg — no routing.
      auto opt = ctg.attr("RandomGreedyOptimizer")(
          py::arg("max_repeats") = effective_max_repeats(),
          py::arg("max_time") = max_time_,
          py::arg("seed") = seed,
          py::arg("progbar") = false);
      tree = opt.attr("search")(inputs, output, sizes);
      // aer-0089: no preset-local slicing. The explicit envelope pass below
      // runs for BOTH presets and is the single place trees are cut to the
      // m6n6k6 budget.
    } else {
      // HyperOptimizer path: backend chosen by AER_TN_OPTLIB env var.
      //
      // *** Seed routing — cotengra 0.7.5 specific ***
      //
      // cotengra 0.7.5's HyperOptimizer.__init__ signature names its **kwargs
      // variable literally `optlib_opts`. Any extra kwargs we pass at the top
      // level land in that dict, which is then unpacked into the backend's
      // init function via:
      //
      //     self._optimizer["init"](self, methods, space, **optlib_opts)
      //
      // For the optuna backend, the init function is:
      //
      //     def optuna_init_optimizers(self, methods, space,
      //                                sampler="TPESampler",
      //                                sampler_opts=None,
      //                                **create_study_opts):
      //         sampler = getattr(optuna.samplers, sampler)(**sampler_opts)
      //         ...
      //
      // So passing sampler_opts={"seed": N} as a top-level kwarg to
      // HyperOptimizer flows into TPESampler(seed=N) — exactly where optuna
      // expects the seed. Anything else would leak into create_study_opts and
      // then to optuna.create_study(), causing TypeError.
      //
      // For the cmaes backend, the 0.7.5 init function is expected to accept
      // seed via its **kwargs which then flows to cmaes.CMA(seed=N). Passing
      // seed= at the top level should work — if it doesn't, we'll iterate.
      //
      // For sbplx and sses we silently omit the seed (they don't support it).
      std::string optlib = path_optlib();

      if (path_verbose()) {
        fprintf(stderr,
                "[AER_TN_PATH] HyperOptimizer: optlib='%s', seed=%lu\n",
                optlib.c_str(), (unsigned long)seed);
      }

      // Build the call args. We ALWAYS pass the standard named params. The
      // seed-routing kwarg is added conditionally per backend using py::dict
      // of extra kwargs merged in via py::kwargs.
      py::dict kwargs;
      // aer-0086: AER_TN_PATH_METHODS restricts cotengra's trial
      // methods (comma-separated, e.g. "greedy" or "greedy,labels").
      // Unset keeps cotengra's default list. The lever exists because
      // trial cost is METHOD-dominated: on the low-treewidth networks
      // of tree-structured circuits, greedy-family trials run in
      // milliseconds at megabytes while a kahypar trial's partitioning
      // arena measures 8-13+ GB at depth -- and on such networks the
      // cheap methods are near-optimal, so a forge restricted to them
      // banks a plan in seconds where the default list needs an hour.
      // The printed plan_flops is the acceptance check either way.
      {
        const char *mth = std::getenv("AER_TN_PATH_METHODS");
        if (mth != nullptr && mth[0] != '\0') {
          py::list mlist;
          std::string cur;
          for (const char *c = mth;; ++c) {
            if (*c == ',' || *c == '\0') {
              if (!cur.empty()) {
                mlist.append(py::str(cur));
                cur.clear();
              }
              if (*c == '\0')
                break;
            } else {
              cur.push_back(*c);
            }
          }
          if (py::len(mlist) > 0)
            kwargs["methods"] = mlist;
        }
      }
      kwargs["minimize"] = path_minimize();
      kwargs["max_repeats"] = effective_max_repeats();
      kwargs["max_time"] = max_time_;
      kwargs["optlib"] = optlib;
      // aer-0089: NO slicing_opts and NO reconf_opts, deliberately. Passing
      // slicing_opts makes cotengra wrap EVERY trial in SlicedTrialFn
      // (hyper.py 545-546), which runs SliceFinder.search(max_repeats=16)
      // on every drawn tree; the finder caches a near-full ContractionCosts
      // copy per accepted index per walk (slicer.py 388, ~110 B x pins
      // each). At depth every draw sits far above the 2^29 target, so every
      // trial paid that cache: 1.27 GB / 100 s at T=1500 on a degenerate
      // draw, and 203 GiB on ONE rank in 3:49 at T=49070 (job 21473960) --
      // the 124-197 GB OOM band of the p>=7 forges, at last named. A
      // reconf_opts of {} additionally wrapped every trial in ReconfTrialFn
      // (hyper.py 553-559, subtree_reconfigure maxiter=500) -- pure time
      // tax. Trials now score RAW trees; the winner is reconfigured once
      // below, and ALL slicing happens in the explicit, cache-free envelope
      // pass after the search. The sizing probe copies these kwargs, so it
      // inherits the same shape.
      kwargs["progbar"] = false;

      // aer-0026: state the worker count instead of letting cotengra's 'auto'
      // read the whole node's CPU count through the container. Omitted entirely
      // when the knob asks for 'auto', so the old behaviour stays reachable for
      // an A/B without a rebuild.
      //
      // This CHANGES PATHS. Worker count decides how many trials finish inside
      // max_time and in what order, so plans found before and after this commit
      // are not comparable and neither are timings taken against them.
      const int par = path_parallel_setting(network.tensors.size());
      if (par == -1)
        kwargs["parallel"] = false;
      else if (par >= 1)
        kwargs["parallel"] = par;

      if (optlib == "optuna") {
        // Flows to optuna_init_optimizers(sampler_opts=...) then TPESampler.
        py::dict sampler_opts;
        sampler_opts["seed"] = seed;
        kwargs["sampler_opts"] = sampler_opts;
      } else if (optlib == "cmaes") {
        // Flows through **optlib_opts into cmaes backend's init kwargs, which
        // forward to cmaes.CMA(seed=N).
        kwargs["seed"] = seed;
      }
      // sbplx, sses: no seed kwarg (backends don't accept it).

      // aer-0079: MEASURED worker sizing. The model in aer-0076 sizes the
      // pool from an estimate that took three battery jobs to calibrate
      // and remains workload-family-specific; the measurement below makes
      // the estimate a fallback and the knobs pure overrides. When the
      // count is auto-decided (no explicit AER_TN_PATH_PARALLEL), a
      // memory budget exists, and more than one worker is on the table:
      // run ONE trial serially in-process through an identical throwaway
      // optimizer, read the kernel's peak-RSS delta around it (VmHWM
      // after clear_refs -- mechanism verified by direct test), price a
      // worker at that delta plus interpreter overhead
      // (AER_TN_PATH_WORKER_OVERHEAD_MB, default 1024), charge the
      // parent's retained trial arena against the budget, and size the
      // pool from what fits -- never above the CPU-derived count. The
      // probe tree is discarded; the main search then runs exactly as
      // before, so plan capture, caching and determinism are untouched.
      // Probe failure of any kind (procfs absent, reset refused, trial
      // raised) falls back to the aer-0076 sizing already in `par`.
      int eff_par = par;
      // aer-0085: a forge runs SERIALLY by definition -- its product is
      // the plan, not the wall, and one trial in memory at a time is
      // the safe shape at any network size.
      if (tn_plan_forge()) {
        eff_par = -1;
        fprintf(stderr, "[AER_TN_PATH] forge: serial search\n");
      }
      // aer-0085: the probe is switchable, bounded, and never silent.
      // AER_TN_PATH_PROBE=0 disables it outright (a flag read: "0"
      // means off, unlike the value knobs where 0 falls to a default).
      // Every non-measured exit below names its branch, because job
      // 21468267 proved a silent fallback is a diagnosis dead-end.
      const char *probe_sw = std::getenv("AER_TN_PATH_PROBE");
      const bool probe_off =
          probe_sw != nullptr && probe_sw[0] == '0' && probe_sw[1] == '\0';
      if (probe_off && !tn_plan_forge())
        fprintf(stderr, "[AER_TN_PATH] probe: disabled by "
                        "AER_TN_PATH_PROBE=0 [skipped]\n");
      if (!tn_plan_forge() && !probe_off &&
          std::getenv("AER_TN_PATH_PARALLEL") == nullptr && par >= 2) {
        uint64_t pb_budget = TensorNetPathMem::node_mem_budget_bytes(
            "/sys/fs/cgroup", "/sys/fs/cgroup/memory", "/proc/self/cgroup",
            "/proc/meminfo");
        long pb_ranks = TensorNetPathMem::local_ranks_sharing_node();
        if (pb_budget == 0) {
          fprintf(stderr, "[AER_TN_PATH] probe: no memory budget signal "
                          "[skipped, model sizing stands]\n");
        } else if (!TensorNetPathMem::reset_peak_rss()) {
          fprintf(stderr, "[AER_TN_PATH] probe: peak-RSS reset unavailable "
                          "[skipped, model sizing stands]\n");
        }
        if (pb_budget > 0 && TensorNetPathMem::reset_peak_rss()) {
          uint64_t rss0 = TensorNetPathMem::read_vmhwm_kb();
          bool probed = false;
          if (rss0 > 0) {
            try {
              // aer-0082: a REAL COPY. py::dict pkw(kwargs) copies the
              // HANDLE -- pybind object copy-construction is reference
              // semantics -- so the probe's overrides would have
              // mutated the main kwargs and every main search would
              // have run max_repeats=1, parallel=False. The Python-
              // level dict.copy() is unambiguous in any binding.
              // aer-0084: the 0082 line above described the right
              // idea with the wrong API. reinterpret_steal exists for
              // RAW PyObject* from the C API; attr("copy")() returns
              // an OWNING py::object temporary, so the steal claimed
              // the reference without incref, the temporary decref'd
              // on destruction, and the fresh copy was freed while
              // pkw still pointed at it -- the first pkw[...] write
              // hit freed memory: the deterministic segfault of jobs
              // 21466846/21466847 directly after the [AER_TN_PATH]
              // parallel line, at both p=1 and p=8. The copy is now a
              // manual shallow item loop: no ownership-transfer API at
              // all, only plain item reads and writes with ordinary
              // refcounting. Shallow is correct and intended: the
              // probe overrides only top-level keys, and sharing
              // sub-objects with the main kwargs mirrors dict.copy().
              py::dict pkw;
              for (auto item : kwargs)
                pkw[item.first] = item.second;
              // aer-0082: several probe trials, not one. Trials sample
              // heterogeneous methods (a cheap greedy draw says nothing
              // about a kahypar draw's 10+ GB), and VmHWM is a
              // high-water mark, so running N serial trials measures
              // the MAX over N method samples at zero extra accounting.
              pkw["max_repeats"] = static_cast<int>(
                  TensorNetPathMem::env_u64("AER_TN_PATH_PROBE_TRIALS", 3));
              pkw["parallel"] = false;
              // aer-0085: the probe carries its OWN time budget instead
              // of inheriting the main search's -- a single serial
              // trial at depth can otherwise run unbounded (the probe
              // cannot preempt a running trial; the bound caps how many
              // start).
              pkw["max_time"] = static_cast<int>(TensorNetPathMem::env_u64(
                  "AER_TN_PATH_PROBE_MAX_TIME", 120));
              auto popt = ctg.attr("HyperOptimizer")(**pkw);
              popt.attr("search")(inputs, output, sizes);
              probed = true;
            } catch (py::error_already_set &pe) {
              fprintf(stderr,
                      "[AER_TN_PATH] probe: trial raised: %s [failed, "
                      "model sizing stands]\n",
                      pe.what());
              pe.discard_as_unraisable("AER_TN_PATH probe trial");
            }
          }
          if (probed) {
            uint64_t rss1 = TensorNetPathMem::read_vmhwm_kb();
            uint64_t trial_b = (rss1 > rss0) ? (rss1 - rss0) * 1024ull : 0;
            if (trial_b == 0)
              fprintf(stderr, "[AER_TN_PATH] probe: zero RSS delta "
                              "[fallback, model sizing stands]\n");
            if (trial_b > 0) {
              uint64_t overhead =
                  TensorNetPathMem::env_u64("AER_TN_PATH_WORKER_OVERHEAD_MB",
                                            1024) *
                  1024ull * 1024ull;
              uint64_t reserve = TensorNetPathMem::env_u64(
                                     "AER_TN_PATH_PARENT_RESERVE_MB", 4096) *
                                 1024ull * 1024ull;
              uint64_t per_rank =
                  pb_budget / static_cast<uint64_t>(pb_ranks > 0 ? pb_ranks
                                                                 : 1);
              uint64_t spent = reserve + trial_b; // parent + retained arena
              uint64_t usable = (per_rank > spent) ? per_rank - spent : 0;
              // aer-0081: the measured trial is ONE SAMPLE of a
              // variable cost -- at T=6078 the p=8 battery cell OOM'd
              // at ~67 GB with 7, 6 AND 5 workers (21456577, 21462112,
              // 21462547), a pattern a fixed per-trial cost cannot
              // produce: later trials (different partitioner draws)
              // peak higher than the first. Price a worker at 3/2 of
              // the measurement (AER_TN_PATH_TRIAL_MARGIN_PCT, default
              // 150) so the pool survives the spread, not just the
              // sample.
              uint64_t margin =
                  TensorNetPathMem::env_u64("AER_TN_PATH_TRIAL_MARGIN_PCT",
                                            150);
              // aer-0082: the model estimate is the FLOOR under the
              // measurement -- if the probe's trials happened to sample
              // only cheap methods, the (deliberately conservative)
              // model keeps the pool from oversizing -- and the
              // variance MARGIN applies to whichever governs, because
              // the max-trial spread the margin covers exists above
              // the floor exactly as above a measurement (a 5-worker
              // pool priced off the unmargined floor at T=6078 is the
              // configuration job 21462547 measured fatal). Costs some
              // parallelism at small networks, where searches are
              // seconds either way; buys a rule that never resizes
              // into a known-fatal pool.
              // aer-0083: num_tensors is the PARAMETER NAME inside
              // path_parallel_setting, not a variable at this site --
              // build 21463409 failed on exactly this identifier. The
              // site's own expression, the same one passed to
              // path_parallel_setting at line ~1668, is used instead.
              uint64_t floor_b = TensorNetPathMem::trial_bytes_estimate(
                  static_cast<uint64_t>(network.tensors.size()));
              uint64_t base_cost = (trial_b > floor_b) ? trial_b : floor_b;
              uint64_t worker_cost = (base_cost * margin) / 100 + overhead;
              long mw = static_cast<long>(usable / worker_cost);
              if (mw < 1)
                mw = 1;
              if (mw < eff_par)
                eff_par = static_cast<int>(mw);
              fprintf(stderr,
                      "[AER_TN_PATH] probe: trial_peak_mb=%llu "
                      "worker_cost_mb=%llu usable_mb=%llu -> %d worker(s) "
                      "[measured]\n",
                      static_cast<unsigned long long>(trial_b /
                                                      (1024ull * 1024ull)),
                      static_cast<unsigned long long>(worker_cost /
                                                      (1024ull * 1024ull)),
                      static_cast<unsigned long long>(usable /
                                                      (1024ull * 1024ull)),
                      eff_par);
            }
          }
        }
      }
      if (eff_par != par) {
        if (eff_par <= 1)
          kwargs["parallel"] = false;
        else
          kwargs["parallel"] = eff_par;
      }

      auto opt = ctg.attr("HyperOptimizer")(**kwargs);
      try {
        tree = opt.attr("search")(inputs, output, sizes);
      } catch (py::error_already_set &e) {
        // cotengra raises KeyError('tree') when every hyperopt trial failed
        // (no contraction tree was ever produced). Surface a named,
        // actionable error instead of the bare KeyError. aer-0090: include
        // the number of trials cotengra actually ran -- jobs 21474687-691
        // logged warning counts far above any 64-repeat budget (15368 per
        // rank at p=8) from a find_path that provably ran once per rank,
        // a multiplier not yet source-located; this counter makes the next
        // occurrence name itself.
        size_t ntrials = 0;
        try {
          ntrials = py::len(opt.attr("scores"));
        } catch (...) {
        }
        std::ostringstream msg;
        msg << "CotengPathOptimizer: cotengra produced no contraction tree ("
            << ntrials
            << " trial(s) ran; every one failed and was discarded -- see the "
               "'Trial error:' warnings above for the per-trial causes). "
               "Underlying error: "
            << e.what();
        throw std::runtime_error(msg.str());
      }
      fprintf(stderr, "[AER_TN_PATH] search done: %zu trial(s)\n",
              (size_t)py::len(opt.attr("scores")));
      // aer-0091: name the winner. Job 21476056 showed 64 clean trials at
      // p=8 whose best still needed 2^70 slices, and nothing in the log
      // said which method won or how bad the best was. One unconditional
      // line makes trial-budget and method-pool tuning evidence-driven.
      try {
        py::dict best = opt.attr("best");
        py::object py_log2 = py::module_::import("math").attr("log2");
        const double lf = py_log2(best["flops"]).cast<double>();
        const double lsz = py_log2(best["size"]).cast<double>();
        std::string mname = "?";
        try {
          mname = py::str(best["params"]["method"]).cast<std::string>();
        } catch (py::error_already_set &me) {
          me.discard_as_unraisable("AER_TN_PATH best method name");
        }
        fprintf(stderr,
                "[AER_TN_PATH] search best: method=%s log2_flops=%.1f "
                "log2_size=%.1f\n",
                mname.c_str(), lf, lsz);
      } catch (py::error_already_set &be) {
        be.discard_as_unraisable("AER_TN_PATH search best");
      }
      // aer-0089: the subtree reconfiguration that ReconfTrialFn previously
      // applied to EVERY trial now runs ONCE, on the winner -- same
      // cotengra defaults (subtree_size=8, maxiter=500), 1/64th the cost
      // under a 64-repeat forge. An optimisation hint must never fail the
      // contraction (the distribution-floor rule), so a raise here warns
      // and continues with the tree as found.
      // aer-0099: transactional; see reconf_tree_transactional. The verbose
      // flops report moved OUT of the protected call and to log2: casting a
      // >= 2^1024-FLOP count to double raises exactly like the
      // reconfiguration itself did on job 21479115, and a report must not be
      // able to roll back a reconfiguration that succeeded.
      if (reconf_tree_transactional(tree, "winner subtree_reconfigure_") &&
          path_verbose()) {
        try {
          py::object py_log2 = py::module_::import("math").attr("log2");
          py::dict st = tree.attr("contract_stats")();
          fprintf(stderr, "[AER_TN_PATH] winner reconf: log2_flops=%.1f\n",
                  py_log2(st["flops"]).cast<double>());
        } catch (py::error_already_set &pe) {
          pe.discard_as_unraisable("AER_TN_PATH winner reconf report");
        }
      }
    }

    // aer-0090: restore the process limit before any post-search work. The
    // winner's slicing, the engine, and HIP must never run capped.
    search_mem_cap.reset();

    // AER_TN_FORCE_SLICING=1: if the natural path needed no slicing, slice
    // anyway (at least 2 slices). This is the validation hook that lets the
    // slicer's projection/accumulation machinery run on small circuits whose
    // results can be checked element-wise against an independent reference.
    if (force_slicing()) {
      py::dict sliced_inds = tree.attr("sliced_inds");
      if (py::len(sliced_inds) == 0) {
        // aer-0089: explicit picker, not tree.slice() -- one code path for
        // every cut this function makes, and no SliceFinder anywhere.
        for (int guard = 0; guard < 8; ++guard) {
          if (tree_num_slices(tree) >= 2)
            break;
          py::object ix = pick_slice_index(tree);
          if (ix.is_none())
            break;
          tree.attr("remove_ind_")(ix);
        }
        if (path_verbose()) {
          fprintf(stderr,
                  "[AER_TN_PATH] AER_TN_FORCE_SLICING=1: sliced an "
                  "already-fitting path over %zu mode(s)\n",
                  (size_t)py::len(tree.attr("sliced_inds")));
        }
      }
    }

    // Enforce the per-slice MEMORY envelope as ACHIEVED, via DYNAMIC
    // SLICING: interleave slicing with subtree reconfiguration. Slicing an
    // index changes which contraction tree is optimal, so a tree cut many
    // modes without re-optimisation can be orders of magnitude more
    // expensive than the same tree re-optimised as it is cut -- the
    // standard technique of the contraction-optimisation literature
    // (arXiv:2002.01935; introduced in arXiv:2005.06787; "interleaving
    // index slicing with local reconfigurations of the contraction order
    // to keep it near-optimal given the already sliced indices",
    // Nat. Comput. Sci. 1, 578 (2021)). Measured on the exact cotengra
    // classes (12x12 grid, mediocre starting tree, target 2^8): slice-only
    // lands at 2^127 total FLOPs; reconfigure-once-then-slice (the
    // aer-0089..0092 pipeline, the shape that produced jobs
    // 21476822/21476823's 2^24-slice explosions) at 2^53 FLOPs / 2^38
    // slices; dynamic slicing at 2^28.5 FLOPs / 8192 slices. Granularity:
    // reconfigure after every AER_TN_SLICE_RECONF_EVERY removals (default
    // 4, the measured-best granularity, matching the literature's
    // slice-factor of ~32 = ~5 binary indices per round; 0 disables the
    // interleave). aer-0094 also REMOVES the slice-count gate from this
    // loop: a fixed ceiling is an arbitrary constant that twice blocked
    // legitimate physics today, and the exact plan cost is now computed
    // and printed at plan time instead -- the wall-derived in-situ gate
    // (measure the first slice, extrapolate, decide) is aer-0095.
    // target_elements is min(device budget, AER_TN_SLICE_TARGET_BYTES) in
    // elements: a memory bound, not a kernel-shape bound. Excluding output
    // modes is allow_outer=false by construction, matching the engine's
    // accumulator and extract_plan's guard.
    {
      py::object py_log2 = py::module_::import("math").attr("log2");
      const double log2_target =
          std::log2(static_cast<double>(target_elements));
      const uint64_t reconf_every =
          TensorNetPathMem::env_u64("AER_TN_SLICE_RECONF_EVERY", 4);
      uint64_t since_reconf = 0;
      bool reconf_pending = false;
      for (int guard = 0; guard < 4096; ++guard) {
        const double cur_log2 =
            py_log2(tree.attr("max_size")()).cast<double>();
        if (cur_log2 <= log2_target + 1e-9) {
          // aer-0097: flush a pending reconfiguration BEFORE declaring the
          // envelope met, then RE-CHECK. Reconfiguration minimizes flops
          // and can trade the peak upward; the 0094 trailing call ran
          // AFTER the last peak check, so a boundary tree could be pushed
          // back over target and thrown as a false "output-bound" hard
          // failure. Flush-and-continue keeps slicing until the peak and
          // the pending flag settle together; the 12x12 regression is
          // identical to the 0094 result (2^28.47 FLOPs, 8192 slices).
          if (reconf_pending) {
            reconf_pending = false;
            since_reconf = 0;
            reconf_tree_transactional(
                tree, "subtree_reconfigure_ during dynamic slicing (flush)");
            continue;
          }
          break;
        }
        py::object ix = pick_slice_index(tree);
        if (ix.is_none())
          break; // no summed bond left to slice: residual peak is output-bound
        tree.attr("remove_ind_")(ix);
        if (reconf_every > 0) {
          reconf_pending = true;
          if (++since_reconf >= reconf_every) {
            since_reconf = 0;
            reconf_pending = false;
            reconf_tree_transactional(
                tree, "subtree_reconfigure_ during dynamic slicing");
          }
        }
      }
      const double final_log2 =
          py_log2(tree.attr("max_size")()).cast<double>();
      if (final_log2 > log2_target + 1e-9) {
        // aer-0098: two truthfully-named exits. With candidates remaining
        // this can only be the 4096-iteration backstop -- reachable in
        // principle on a tree needing thousands of cuts (74k summed
        // indices exist at the deepest campaign networks), and the
        // output-bound wording would then be false. Candidate-free means
        // genuinely output-bound.
        py::object more = pick_slice_index(tree);
        std::ostringstream msg;
        if (!more.is_none()) {
          msg << "CotengPathOptimizer: dynamic slicing stopped at the "
                 "4096-iteration backstop with the peak still 2^"
              << final_log2
              << " elements against the per-slice memory envelope of 2^"
              << log2_target
              << " and sliceable candidates remaining -- the winning tree "
                 "needs thousands of cuts, which no plan survives. This is "
                 "a search-quality failure: raise the trial budget "
                 "(AER_TN_PATH_MAX_REPEATS), raise AER_TN_PATH_MAX_TIME, or "
                 "widen the method pool.";
        } else {
          msg << "CotengPathOptimizer: peak intermediate (2^" << final_log2
              << " elements) exceeds the per-slice memory envelope (2^"
              << log2_target
              << " elements, the smaller of the device budget and "
                 "AER_TN_SLICE_TARGET_BYTES) and cannot be reduced further "
                 "by slicing -- the residual peak is bounded by output "
                 "(open) modes the engine cannot split. Request a "
                 "lower-rank output (expectation value, amplitude, or a "
                 "smaller reduced density matrix), or raise "
                 "AER_TN_SLICE_TARGET_BYTES.";
        }
        throw std::runtime_error(msg.str());
      }
      if (path_verbose()) {
        fprintf(stderr,
                "[AER_TN_PATH] slice envelope enforced: peak %lld <= target "
                "%llu elements (%zu sliced modes, %llu slices)\n",
                (long long)tree.attr("max_size")().cast<int64_t>(),
                (unsigned long long)target_elements,
                (size_t)py::len(tree.attr("sliced_inds")),
                (unsigned long long)tree_num_slices(tree));
      }
    }

    // aer-0035: raise the slice count to the distribution floor, AFTER the
    // envelope block above. Order matters and only in this direction: extra
    // slicing can only lower the peak, so the envelope stays satisfied, whereas
    // doing it first would let the envelope pass re-slice against a target that
    // ignores the rank count.
    //
    // target_slices is the same call form the AER_TN_FORCE_SLICING hook already
    // uses above, so this relies on no cotengra API that is not already
    // exercised. cotengra treats it as a floor and slices to the next reachable
    // power of the sliced bonds' extents, so the result can exceed min_slices_;
    // that is fine, the requirement is only that no rank is left empty.
    //
    // Every rank of the MPI ensemble applies the same floor, so every candidate
    // the MINLOC(FLOPs) pick can choose from already meets it. Without that the
    // pick would actively favour the rank that sliced least, which is the same
    // failure mode the envelope block above documents.
    if (min_slices_ > 1) {
      const size_t before = py::len(tree.attr("sliced_inds"));
      const uint64_t cur_slices = tree_num_slices(tree);
      if (cur_slices < min_slices_) {
        // A failure to REACH the floor must never fail the contraction. The
        // floor is a distribution hint; a plan with fewer slices than ranks is
        // merely under-parallel, and slicing is exact at any count, so whatever
        // state the tree is left in is still a correct plan. Losing a whole
        // job to an optimisation hint is not an acceptable trade.
        bool sliced_ok = true;
        try {
          // aer-0089: explicit picker to the floor. Each removal multiplies
          // the slice count by the removed extent and can only lower the
          // peak, so the envelope above stays satisfied -- the same
          // one-direction argument the original ordering note makes.
          for (int guard = 0; guard < 128; ++guard) {
            if (tree_num_slices(tree) >= min_slices_)
              break;
            py::object ix = pick_slice_index(tree);
            if (ix.is_none())
              break;
            tree.attr("remove_ind_")(ix);
          }
        } catch (py::error_already_set &e) {
          sliced_ok = false;
          fprintf(stderr,
                  "[AER_TN_PATH] warning: the distribution floor of %llu "
                  "slices could not be applied; continuing with the plan as "
                  "found (%llu slices). Underlying error: %s\n",
                  (unsigned long long)min_slices_,
                  (unsigned long long)cur_slices, e.what());
        }
        const uint64_t now = tree_num_slices(tree);
        if (path_verbose() || (sliced_ok && now < min_slices_))
          fprintf(stderr,
                  "[AER_TN_PATH] distribution floor: %llu slices -> %llu "
                  "(target %llu; sliced modes %zu -> %zu)%s\n",
                  (unsigned long long)cur_slices, (unsigned long long)now,
                  (unsigned long long)min_slices_, before,
                  (size_t)py::len(tree.attr("sliced_inds")),
                  (now < min_slices_)
                      ? " -- FLOOR NOT REACHED: too few sliceable summed bonds"
                      : "");
      }
    }

    // aer-0094: the exact plan cost, computed from the sliced tree and
    // printed unconditionally -- multiplicity-inclusive total FLOPs (exact
    // arbitrary-precision integers underneath; reported in log2 so no
    // magnitude can overflow the report), slice count, and steps per
    // slice. This line is the calculated replacement for the retired
    // fixed slice ceiling: the researcher and the aer-0095 in-situ wall
    // gate both read cost, not guess it.
    {
      py::object py_log2 = py::module_::import("math").attr("log2");
      py::dict st = tree.attr("contract_stats")();
      const double lf = py_log2(st["flops"]).cast<double>();
      fprintf(stderr,
              "[AER_TN_PATH] plan cost: total_flops=%.3e (2^%.1f) "
              "slices=%llu steps_per_slice=%lld sliced_modes=%zu\n",
              std::exp2(lf), lf,
              (unsigned long long)tree_num_slices(tree),
              (long long)(tree.attr("N").cast<int64_t>() - 1),
              (size_t)py::len(tree.attr("sliced_inds")));
    }

    return extract_plan(tree, std::set<int32_t>(network.output_modes.begin(),
                                                network.output_modes.end()));
  }

private:
  // aer-0036: the number of slices a tree currently carries, computed the SAME
  // way extract_plan does it -- iterate sliced_inds and multiply each entry's
  // .size. aer-0035 originally read tree.attr("nslices"), an attribute this
  // fork uses nowhere else and which was therefore unverified against cotengra
  // 0.7.5; a wrong name there raises py::error_already_set on every contraction
  // with the floor engaged. sliced_inds and .size are exercised on every
  // contraction this backend has ever run, so this introduces no new API
  // surface at all.
  static uint64_t tree_num_slices(py::object &tree) {
    py::dict sliced_inds = tree.attr("sliced_inds");
    uint64_t n = 1;
    for (auto &item : sliced_inds) {
      const uint64_t e =
          static_cast<uint64_t>(item.second.attr("size").cast<int64_t>());
      if (e == 0)
        continue;
      // aer-0091: saturate instead of wrapping. Job 21476056 (p=8): the
      // envelope cut a degenerate winner over 70 extent-2 modes; 2^70
      // wrapped this product to 0, the distribution floor then saw
      // "0 < 16" and added 128 further modes to a plan already at 2^70
      // slices, and printed FLOOR NOT REACHED from a 2^198-slice plan.
      if (n > (UINT64_MAX / e))
        return UINT64_MAX;
      n *= e;
    }
    return n;
  }

  // aer-0089: the explicit slicing picker. Returns the summed (non-output)
  // index of largest extent on the largest intermediate, or none when the
  // peak is output-bound (nothing sliceable remains on it). The node scan
  // walks traverse() -- the SAME node set max_size() maxes over -- so the
  // peak this reduces is the peak the envelope check reads. remove_ind_
  // strips a removed index from every node's legs, so an already-sliced
  // index can never be returned. Every API touched here (traverse,
  // get_size, get_legs, size_dict, output, remove_ind_) was validated by
  // direct test against cotengra 0.7.5 on the degenerate T=1500 network:
  // same 12 modes as SliceFinder, 0.3 s / 2.3 MB versus 100 s / 1.27 GB.
  // Deterministic and seed-free: argmax in traverse order, ties to the
  // first seen, so identical trees slice identically on every rank.
  static py::object pick_slice_index(py::object &tree) {
    // aer-0090: node sizes are exact Python ints and on degenerate draws
    // dwarf a double (jobs 21474687-691: peaks far beyond 2^1024, where
    // float(size) raises OverflowError -- the 0089 cast<double> here would
    // itself abort). Compare in log2: Python's math.log2 takes
    // arbitrary-precision ints exactly.
    py::object py_log2 = py::module_::import("math").attr("log2");
    py::object best_node = py::none();
    double best_size = -1.0;
    for (auto &item : tree.attr("traverse")()) {
      py::tuple t = item.cast<py::tuple>();
      py::object node = py::reinterpret_borrow<py::object>(t[0]);
      const double s = py_log2(tree.attr("get_size")(node)).cast<double>();
      if (s > best_size) {
        best_size = s;
        best_node = node;
      }
    }
    if (best_node.is_none())
      return py::none();
    std::set<std::string> outset;
    for (auto &o : tree.attr("output"))
      outset.insert(o.cast<std::string>());
    py::dict size_dict = tree.attr("size_dict");
    py::dict legs = tree.attr("get_legs")(best_node);
    py::object best_ix = py::none();
    int64_t best_ext = 1;
    for (auto &kv : legs) {
      if (outset.count(kv.first.cast<std::string>()))
        continue;
      const int64_t d = size_dict[kv.first].cast<int64_t>();
      if (d > best_ext) {
        best_ext = d;
        best_ix = py::reinterpret_borrow<py::object>(kv.first);
      }
    }
    return best_ix;
  }

  ContractionPlan extract_plan(py::object &tree,
                               const std::set<int32_t> &output_modes) {
    ContractionPlan plan;

    py::object ssa_path = tree.attr("get_ssa_path")();
    for (auto &pair : ssa_path) {
      auto t = pair.cast<py::tuple>();
      ContractionStep step;
      step.left = t[0].cast<uint64_t>();
      step.right = t[1].cast<uint64_t>();
      plan.steps.push_back(step);
    }

    // sliced_inds: keys are label strings — map back to int32_t
    py::dict sliced_inds = tree.attr("sliced_inds");
    plan.num_slices = 1;
    for (auto &item : sliced_inds) {
      SliceInfo si;
      std::string mode_str = item.first.cast<std::string>();
      // Primary: reverse-lookup in the label_to_mode_ map we built when
      // sending the network to cotengra.
      auto it = label_to_mode_.find(mode_str);
      if (it != label_to_mode_.end()) {
        si.mode = it->second;
      } else {
        // Fallback: an integer-valued string. Kept for safety even though
        // cotengra preserves label identity in practice.
        try {
          si.mode = static_cast<int32_t>(std::stoi(mode_str));
        } catch (const std::exception &) {
          throw std::runtime_error(
              "CotengPathOptimizer: cannot resolve sliced mode label "
              "returned by cotengra");
        }
      }
      auto slice_info_obj = item.second;
      si.extent = slice_info_obj.attr("size").cast<int64_t>();

      // The execution engine's accumulation rule is a plain element-wise
      // sum, which is only correct for sliced bonds that are summed over
      // (inner indices). A sliced OUTPUT mode would mean each slice owns a
      // disjoint block of the output instead — different semantics the
      // engine does not implement. cotengra doesn't slice outer indices
      // under the options used here; this guard turns a future change of
      // that default into a loud planning error rather than a silent
      // wrong result.
      if (output_modes.count(si.mode)) {
        throw std::runtime_error(
            "CotengPathOptimizer: cotengra sliced an output (outer) mode; "
            "the sliced execution engine only supports slicing summed "
            "bonds. This is a planner configuration bug.");
      }

      plan.sliced.push_back(si);
      plan.num_slices *= static_cast<uint64_t>(si.extent);
    }

    plan.total_flops = tree.attr("total_flops")().cast<double>();

    py::object peak_size = tree.attr("max_size")();
    plan.peak_intermediate_elements = peak_size.cast<int64_t>();

    if (path_verbose()) {
      fprintf(stderr,
              "[AER_TN_PATH] cotengra plan: %zu steps, %zu sliced modes, "
              "%lu slices, %.2e total FLOPs, peak intermediate %ld elements\n",
              plan.steps.size(), plan.sliced.size(),
              (unsigned long)plan.num_slices, plan.total_flops,
              (long)plan.peak_intermediate_elements);
    }

    return plan;
  }
};

#endif // AER_HIPTENSOR

//=============================================================================
// GreedyPathOptimizer — C++ fallback
//=============================================================================

class GreedyPathOptimizer : public PathOptimizer {
  int num_restarts_;

public:
  explicit GreedyPathOptimizer(int num_restarts = 32)
      : num_restarts_(num_restarts) {}

  ContractionPlan find_path(const NetworkDescription &network,
                            uint64_t memory_limit_bytes,
                            uint64_t seed,
                            bool /*tiling_available*/) override {
    auto size_dict = network.build_size_dict();
    ContractionPlan best;
    best.total_flops = std::numeric_limits<double>::max();

    for (int restart = 0; restart < num_restarts_; restart++) {
      ContractionPlan candidate =
          greedy_once(network, size_dict, seed + restart);
      if (candidate.total_flops < best.total_flops)
        best = candidate;
    }

    best.num_slices = 1;
    // Same memory-governed budget as the cotengra path (aer-0013): device
    // memory and the optional AER_TN_SLICE_TARGET_BYTES envelope, in elements
    // (16 bytes per logical complex element). The per-slice peak is NOT clamped
    // to the removed max_tiled_elements() clamp -- an over-ceiling tiled descriptor is carried by
    // tiling + operand staging (aer-0010/0011), and the m6n6k6 envelope is
    // enforced by tiling at execution, not by over-slicing here. Keeping this
    // in sync with the cotengra path avoids a latent slice-grind on the greedy
    // fallback.
    int64_t element_budget =
        static_cast<int64_t>(memory_limit_bytes / 16);
    int64_t envelope_budget =
        std::max<int64_t>(1, static_cast<int64_t>(slice_target_bytes() / 16));
    element_budget = std::min(element_budget, envelope_budget);
    // Output (open) modes must never be sliced -- the sliced engine sums
    // slices, correct only for summed bonds (mirrors the cotengra path's
    // allow_outer=false and extract_plan's guard).
    std::set<int32_t> output_set(network.output_modes.begin(),
                                 network.output_modes.end());
    while (best.peak_intermediate_elements > element_budget &&
           !size_dict.empty()) {
      int32_t best_mode = -1;
      int64_t best_reduction = 0;
      for (auto it = size_dict.begin(); it != size_dict.end(); ++it) {
        if (it->second <= 1)
          continue;
        if (output_set.count(it->first))
          continue;
        int64_t reduction = best.peak_intermediate_elements -
                            best.peak_intermediate_elements / it->second;
        if (reduction > best_reduction) {
          best_reduction = reduction;
          best_mode = it->first;
        }
      }
      if (best_mode < 0)
        break;

      SliceInfo si;
      si.mode = best_mode;
      si.extent = size_dict[best_mode];
      best.sliced.push_back(si);
      best.num_slices *= static_cast<uint64_t>(si.extent);
      best.peak_intermediate_elements /= si.extent;
      size_dict[best_mode] = 1; // sliced: do not pick this mode again
    }

    if (path_verbose()) {
      fprintf(stderr,
              "[AER_TN_PATH] greedy plan (%d restarts): %zu steps, "
              "%zu sliced modes, %lu slices, %.2e total FLOPs\n",
              num_restarts_, best.steps.size(), best.sliced.size(),
              (unsigned long)best.num_slices, best.total_flops);
    }
    return best;
  }

private:
  ContractionPlan greedy_once(const NetworkDescription &network,
                              const std::map<int32_t, int64_t> &size_dict,
                              uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> noise(0.0, 1e-12);

    std::vector<TensorSpec> working;
    working.reserve(network.tensors.size() * 2);
    for (size_t i = 0; i < network.tensors.size(); i++)
      working.push_back(network.tensors[i]);

    std::vector<bool> alive(working.capacity(), false);
    for (size_t i = 0; i < network.tensors.size(); i++)
      alive[i] = true;

    ContractionPlan plan;
    plan.total_flops = 0.0;
    plan.peak_intermediate_elements = 0;

    size_t num_tensors = network.tensors.size();
    for (size_t step = 0; step + 1 < num_tensors; step++) {
      uint64_t best_i = 0, best_j = 0;
      double best_cost = std::numeric_limits<double>::max();

      for (size_t i = 0; i < working.size(); i++) {
        if (!alive[i]) continue;
        for (size_t j = i + 1; j < working.size(); j++) {
          if (!alive[j]) continue;

          bool share = false;
          for (size_t mi = 0; mi < working[i].modes.size() && !share; mi++)
            for (size_t mj = 0; mj < working[j].modes.size() && !share; mj++)
              if (working[i].modes[mi] == working[j].modes[mj])
                share = true;

          double cost = compute_cost(working[i], working[j], size_dict);
          if (!share) cost *= 1e6;
          cost += noise(rng);

          if (cost < best_cost) {
            best_cost = cost;
            best_i = i;
            best_j = j;
          }
        }
      }

      ContractionStep cs;
      cs.left = best_i;
      cs.right = best_j;
      plan.steps.push_back(cs);
      plan.total_flops += best_cost;

      TensorSpec result = contract_spec(working[best_i], working[best_j]);
      plan.peak_intermediate_elements =
          std::max(plan.peak_intermediate_elements, result.num_elements());

      alive[best_i] = false;
      alive[best_j] = false;
      working.push_back(result);
      if (alive.size() < working.size())
        alive.push_back(true);
      else
        alive[working.size() - 1] = true;
    }
    return plan;
  }

  static double compute_cost(const TensorSpec &a, const TensorSpec &b,
                             const std::map<int32_t, int64_t> &) {
    std::map<int32_t, int64_t> all_modes;
    for (size_t i = 0; i < a.modes.size(); i++)
      all_modes[a.modes[i]] = a.extents[i];
    for (size_t i = 0; i < b.modes.size(); i++)
      all_modes[b.modes[i]] = b.extents[i];
    double cost = 1.0;
    for (auto it = all_modes.begin(); it != all_modes.end(); ++it)
      cost *= static_cast<double>(it->second);
    return cost;
  }

  static TensorSpec contract_spec(const TensorSpec &a, const TensorSpec &b) {
    TensorSpec result;
    std::vector<int32_t> shared;
    for (size_t i = 0; i < a.modes.size(); i++)
      for (size_t j = 0; j < b.modes.size(); j++)
        if (a.modes[i] == b.modes[j]) {
          shared.push_back(a.modes[i]);
          break;
        }

    for (size_t i = 0; i < a.modes.size(); i++) {
      bool is_shared = false;
      for (size_t s = 0; s < shared.size(); s++)
        if (a.modes[i] == shared[s]) { is_shared = true; break; }
      if (!is_shared) {
        result.modes.push_back(a.modes[i]);
        result.extents.push_back(a.extents[i]);
      }
    }
    for (size_t i = 0; i < b.modes.size(); i++) {
      bool is_shared = false;
      for (size_t s = 0; s < shared.size(); s++)
        if (b.modes[i] == shared[s]) { is_shared = true; break; }
      if (!is_shared) {
        result.modes.push_back(b.modes[i]);
        result.extents.push_back(b.extents[i]);
      }
    }
    return result;
  }
};

//=============================================================================
// MPIParallelPathOptimizer
//=============================================================================

#ifdef AER_MPI

// aer-0065: MPI-cooperative path search. When ON (default) and nprocs > 1,
// MPIParallelPathOptimizer divides the AER_TN_PATH_MAX_REPEATS trial budget
// across ranks -- rank r runs ~budget/R trials on its own seed stream
// (seed + r, the offset that already exists) and the MINLOC(total_flops)
// collective picks the winner, so the SAME total trial count finishes in
// ~1/R the wall. Measured stake (job 21381487, rand3 n=10000 p=4): 81
// classes x ~15-38 s of search, every rank redundantly searching the full
// budget -- 99.6% of a 40-minute stage, against 10 s of contraction.
//
// SEMANTICS CHANGE, stated plainly: under MPI, AER_TN_PATH_MAX_REPEATS now
// means the TOTAL trial budget across ranks, where the legacy ensemble ran
// budget x R total (full budget per rank, more diversity, no wall saving).
// The legacy behaviour is exactly recovered two ways: AER_TN_PATH_MPI_SHARD=0,
// or shard ON with MAX_REPEATS raised R-fold (identical diversity, same wall
// as one legacy rank). AER_TN_PATH_MAX_TIME stays a PER-RANK cap -- it is a
// ceiling, not a target; a time-bound search that used to hit the cap now
// finishes its smaller shard early, which is the point.
//
// Determinism: the drawn plan is a deterministic function of (seed, R,
// budget) -- reproducible at fixed rank count, DIFFERENT across rank counts
// (as the legacy ensemble's MINLOC winner already was). Cross-width
// reproducibility is plan capture's job (AER_TN_PLAN_FILE / AER_TN_PLAN_DIR),
// unchanged: the captured plan is the MINLOC winner, so first contact pays
// one sharded search and every later run at ANY width replays it.
static int tn_path_mpi_shard() {
  static bool checked = false;
  static int cached = 1;
  if (!checked) {
    const char *v = std::getenv("AER_TN_PATH_MPI_SHARD");
    if (v != nullptr && v[0] != '\0')
      // plain char test, not strcmp: this file never included <cstring>
      // and should not grow the dependency for a one-character compare.
      cached = (v[0] == '0' && v[1] == '\0') ? 0 : 1;
    checked = true;
  }
  return cached;
}

// aer-0066: defined inside the AER_MPI section because it is the file's
// only helper with no non-MPI caller -- at file scope it would be the one
// static function -Wall flags as unused in every non-MPI TU.

class MPIParallelPathOptimizer : public PathOptimizer {
  std::unique_ptr<PathOptimizer> inner_;
  MPI_Comm comm_;

public:
  MPIParallelPathOptimizer(std::unique_ptr<PathOptimizer> inner,
                           MPI_Comm comm = MPI_COMM_WORLD)
      : inner_(std::move(inner)), comm_(comm) {}

  ContractionPlan find_path(const NetworkDescription &network,
                            uint64_t memory_limit_bytes,
                            uint64_t seed,
                            bool tiling_available) override {
    int rank, size;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &size);

    // aer-0065: cooperative search. Divide the trial budget across ranks and
    // let the MINLOC below pick the winner -- same total trials, wall / R.
    // The set is idempotent and explicit in BOTH branches so a process that
    // flips nothing still states its intent every call, and the knob's OFF
    // value exactly restores the legacy full-budget-per-rank ensemble.
    if (tn_path_mpi_shard() && size > 1) {
      const bool accepted = inner_->set_trial_share(rank, size);
      if (path_verbose() && rank == 0) {
        if (accepted)
          fprintf(stderr,
                  "[AER_TN_PATH] MPI cooperative search: %d total trials "
                  "sharded across %d ranks (~%d each), per-rank seeds "
                  "%lu..%lu; AER_TN_PATH_MPI_SHARD=0 restores the legacy "
                  "full-budget ensemble\n",
                  path_max_repeats(), size,
                  (path_max_repeats() + size - 1) / size, (unsigned long)seed,
                  (unsigned long)(seed + size - 1));
        else
          fprintf(stderr,
                  "[AER_TN_PATH] MPI: inner optimizer has no shardable trial "
                  "budget (greedy fallback); legacy full-budget-per-rank "
                  "ensemble\n");
      }
    } else {
      inner_->set_trial_share(0, 1);
    }

    // aer-0100: a rank whose OWN search fails is not a dead rank. After the
    // MINLOC below, every non-winning rank discards its local result and
    // adopts the broadcast winner's plan anyway -- a searchless rank can
    // adopt it exactly the same way. The aer-0037 fail-together contract
    // stays for what it was built for (no rank may unwind past a collective
    // its siblings will enter: every catch below keeps this rank marching
    // through the SAME reductions and broadcasts as everyone else), but the
    // OUTCOME changes: a caught search failure now reports
    // cost = +infinity into the MINLOC instead of aborting all ranks. Jobs
    // 21479115, 21479355 and 21480712 each lost an entire 4- or 8-rank
    // forge to ONE rank whose whole trial shard died at the memory cap
    // while sibling ranks held finished plans (best-of-N searching makes a
    // per-rank wipeout an expected draw, not an anomaly: at the observed
    // ~25% per-rank rate, an 8-rank forge survived with probability
    // 0.75^8 ~= 10%). Only the all-ranks-failed case is fatal now,
    // detected from the reduced MINLOC result -- identical on every rank,
    // BEFORE any broadcast -- so the collectives stay in lockstep and the
    // job dies with the failed-rank count and this rank's own cause.
    // NeedsTilingException keeps its collective semantics unchanged; a
    // searchless rank simply re-plans with everyone else.
    ContractionPlan local;
    int local_status = 0; // 0 proceed, 1 needs tiling
    bool local_ok = true;
    std::string local_msg;
    try {
      local = inner_->find_path(network, memory_limit_bytes, seed + rank,
                                tiling_available);
    } catch (const NeedsTilingException &) {
      // Must stay first: derives from std::runtime_error and is a control
      // signal, not a failure.
      local_status = 1;
    } catch (const std::exception &e) {
      // py::error_already_set derives from std::runtime_error and lands
      // here: cotengra's "produced no contraction tree", a dead loky
      // worker, a std::bad_alloc.
      local_ok = false;
      local_msg = e.what();
    } catch (...) {
      local_ok = false;
      local_msg =
          "[AER_TN] the contraction path search raised an exception of "
          "unknown type on this rank.";
    }
    if (!local_ok) {
      fprintf(stderr,
              "[AER_TN_PATH] rank %d: search produced no plan; this rank "
              "will adopt the winning rank's plan. Cause: %s\n",
              rank, local_msg.c_str());
    }

    // Tiling stays a MAX-collective decision: if ANY rank needs tiling,
    // EVERY rank throws together and the driver's one-shot retry re-plans
    // all ranks -- searchless ranks included -- with tiling engaged.
    int global_status = 0;
    MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX, comm_);
    if (global_status == 1) {
      throw NeedsTilingException(
          NeedsTilingException::Gate::Planner,
          "MPI path search: at least one rank produced an above-envelope "
          "output path; re-planning all ranks with tiling engaged.");
    }

    int local_failed = local_ok ? 0 : 1;
    int failed_ranks = 0;
    MPI_Allreduce(&local_failed, &failed_ranks, 1, MPI_INT, MPI_SUM, comm_);

    struct { double cost; int rank; } local_result, global_result;
    local_result.cost = local_ok
                            ? local.total_flops
                            : std::numeric_limits<double>::infinity();
    local_result.rank = rank;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE_INT,
                  MPI_MINLOC, comm_);

    if (!(global_result.cost < std::numeric_limits<double>::infinity())) {
      // Identical reduced values on every rank -> an identical throw on
      // every rank, still BEFORE any broadcast: the fail-together
      // guarantee holds exactly where it matters.
      std::ostringstream msg;
      msg << "[AER_TN] MPI path search: every rank failed (" << failed_ranks
          << " of " << size
          << " produced no plan); nothing to adopt. This rank's cause: "
          << (local_ok ? std::string("(this rank reported a non-finite plan "
                                     "cost despite succeeding, which should "
                                     "not happen)")
                       : local_msg);
      throw std::runtime_error(msg.str());
    }

    if (path_verbose() && rank == 0) {
      fprintf(stderr,
              "[AER_TN_PATH] MPI: %d ranks, best from rank %d (%.2e FLOPs)"
              "%s\n",
              size, global_result.rank, global_result.cost,
              failed_ranks > 0 ? " -- searchless rank(s) adopt it" : "");
      if (failed_ranks > 0)
        fprintf(stderr,
                "[AER_TN_PATH] MPI: %d of %d rank(s) produced no plan and "
                "adopt the winner's\n",
                failed_ranks, size);
    }

    // The contraction PATH (steps) is positional -- SSA indices into the input
    // tensor list, whose order is identical on every rank -- so it broadcasts
    // verbatim. The SLICED modes are NOT: they are stored as int32 mode IDs,
    // and those IDs are assigned per-process by the network builder, so they
    // differ on every rank (visible as the differing set_output mode values).
    // The winner's int32 sliced IDs name nothing on the other ranks, which
    // would then recognize zero sliced modes and contract UNSLICED -- oversized
    // intermediates that trip the contractor's stride guard (and, before that
    // guard existed, silently miscomputed across ranks). Translate each sliced
    // mode to a rank-PORTABLE structural coordinate -- (tensor index, mode
    // position) of its first occurrence -- on the winner, broadcast that, and
    // resolve it back to each rank's own int32 on receipt. Tensor order and
    // within-tensor mode order are structural (identical across ranks; the IDs
    // differ only by a per-process base), so the coordinate names the same
    // physical bond everywhere. Extents and num_slices are already portable.
    std::vector<int64_t> path_data;
    std::vector<int64_t> sliced_coords; // [tensor, position] per sliced mode
    int path_size = 0;
    int ncoords = 0;
    if (rank == global_result.rank) {
      path_data = local.serialize();
      path_size = static_cast<int>(path_data.size());
      for (const auto &s : local.sliced) {
        int64_t ft = -1, fp = -1;
        for (size_t t = 0; t < network.tensors.size() && ft < 0; t++) {
          const auto &modes = network.tensors[t].modes;
          for (size_t p = 0; p < modes.size(); p++) {
            if (modes[p] == s.mode) {
              ft = static_cast<int64_t>(t);
              fp = static_cast<int64_t>(p);
              break;
            }
          }
        }
        sliced_coords.push_back(ft);
        sliced_coords.push_back(fp);
      }
      ncoords = static_cast<int>(sliced_coords.size());
    }
    MPI_Bcast(&path_size, 1, MPI_INT, global_result.rank, comm_);
    MPI_Bcast(&ncoords, 1, MPI_INT, global_result.rank, comm_);
    path_data.resize(path_size);
    sliced_coords.resize(ncoords);
    MPI_Bcast(path_data.data(), path_size, MPI_INT64_T, global_result.rank,
              comm_);
    MPI_Bcast(sliced_coords.data(), ncoords, MPI_INT64_T, global_result.rank,
              comm_);

    ContractionPlan plan = ContractionPlan::deserialize(path_data);

    // Re-resolve each sliced mode ID into THIS rank's mode namespace via its
    // structural coordinate (extents and num_slices are unchanged).
    for (size_t i = 0; i < plan.sliced.size() && (2 * i + 1) < sliced_coords.size();
         i++) {
      int64_t t = sliced_coords[2 * i];
      int64_t p = sliced_coords[2 * i + 1];
      if (t >= 0 && p >= 0 &&
          static_cast<size_t>(t) < network.tensors.size() &&
          static_cast<size_t>(p) < network.tensors[t].modes.size()) {
        plan.sliced[i].mode = network.tensors[t].modes[p];
      }
    }

    // [MPI DIAG] Every rank dumps each sliced mode's broadcast structural
    // coordinate (t,p), the extent THIS rank finds at that coordinate, the
    // re-resolved local mode ID, and the winner's stored slice extent. The
    // (t,p) pair is broadcast so it is identical everywhere; if the local
    // extent at (t,p) ever differs from the winner's stored extent, the
    // network builder is not structurally identical across ranks and the
    // coordinate names a different physical bond -- i.e. the re-resolution is
    // the bug. Gated so it is silent unless explicitly requested.
    if (getenv("AER_TN_MPI_DIAG") || getenv("AER_TN_MPI_DIAG_FULLSLICE")) {
      for (size_t i = 0;
           i < plan.sliced.size() && (2 * i + 1) < sliced_coords.size(); i++) {
        int64_t t = sliced_coords[2 * i];
        int64_t p = sliced_coords[2 * i + 1];
        int64_t ext = (t >= 0 && p >= 0 &&
                       static_cast<size_t>(t) < network.tensors.size() &&
                       static_cast<size_t>(p) < network.tensors[t].extents.size())
                          ? network.tensors[t].extents[p]
                          : -1;
        fprintf(stderr,
                "[AER_TN_MPIDIAG] rank %d sliced[%zu] coord=(t%ld,p%ld) "
                "local_extent=%ld local_mode=%d winner_extent=%ld\n",
                rank, i, (long)t, (long)p, (long)ext, plan.sliced[i].mode,
                (long)plan.sliced[i].extent);
      }
    }
    return plan;
  }
};

#endif // AER_MPI

} // namespace TensorNetwork
} // namespace AER

#endif // _path_optimizer_hpp_
