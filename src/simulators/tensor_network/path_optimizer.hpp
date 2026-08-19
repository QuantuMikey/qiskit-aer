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

// aer-0026: sched_getaffinity()/CPU_COUNT for the cotengra worker count. Linux
// only, which this file already is (ROCm/HIP, MPICH, LUMI).
#include <sched.h>

#ifdef AER_HIPTENSOR
#include <pybind11/pybind11.h>
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
static int path_parallel_setting() {
  static bool checked = false;
  static int cached = -2;
  if (checked)
    return cached;
  checked = true;

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

  const char *v = std::getenv("AER_TN_PATH_PARALLEL");
  std::string source = "auto-detected";
  if (v != nullptr) {
    std::string s(v);
    source = "AER_TN_PATH_PARALLEL";
    if (s == "auto") {
      cached = -2;
    } else if (s == "0" || s == "false" || s == "False") {
      cached = -1;
    } else {
      char *end = nullptr;
      long parsed = std::strtol(v, &end, 10);
      cached = (end != v && parsed > 0) ? static_cast<int>(parsed) : -2;
    }
  } else {
    long n = affinity;
    if (slurm > 0 && (n <= 0 || slurm < n))
      n = slurm;
    if (n <= 0)
      n = 1;
    if (n > 32)
      n = 32;
    cached = (n <= 1) ? -1 : static_cast<int>(n);
  }

  char desc[64];
  if (cached == -2)
    snprintf(desc, sizeof(desc), "cotengra 'auto' (whole-node CPU count)");
  else if (cached == -1)
    snprintf(desc, sizeof(desc), "False (serial)");
  else
    snprintf(desc, sizeof(desc), "%d worker(s)", cached);
  fprintf(stderr,
          "[AER_TN_PATH] parallel: sched_getaffinity=%ld "
          "SLURM_CPUS_PER_TASK=%ld -> %s [%s]\n",
          affinity, slurm, desc, source.c_str());
  return cached;
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
  if (!checked) {
    const char *val = std::getenv("AER_TN_SLICE_TARGET_BYTES");
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
    const char *v = std::getenv("AER_TN_PLAN_CACHE");
    cached = (v != nullptr && v[0] == '1' && v[1] == '\0');
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

class CotengPathOptimizer : public PathOptimizer {
  std::string minimize_;
  int max_repeats_;
  double max_time_;
  std::string preset_;
  size_t element_size_bytes_;
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

    py::dict slicing_opts;
    slicing_opts["target_size"] = target_elements;
    // Outer (output) modes must never be sliced: the engine's accumulator
    // sums slices, which matches summed-bond slicing only (see extract_plan).
    slicing_opts["allow_outer"] = false;
    py::dict reconf_opts;

    py::object tree;

    if (preset_ == "random-greedy") {
      // RandomGreedyOptimizer takes seed as a direct named kwarg — no routing.
      auto opt = ctg.attr("RandomGreedyOptimizer")(
          py::arg("max_repeats") = max_repeats_,
          py::arg("max_time") = max_time_,
          py::arg("seed") = seed,
          py::arg("progbar") = false);
      tree = opt.attr("search")(inputs, output, sizes);
      // RandomGreedyOptimizer has no slicing_opts; slice the finished tree
      // to the same per-slice budget the HyperOptimizer path uses, so both
      // presets honor the m6n6k6 envelope. No-op when the tree already fits.
      tree.attr("slice")(py::arg("target_size") = target_elements,
                         py::arg("allow_outer") = false,
                         py::arg("seed") = seed,
                         py::arg("inplace") = true);
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
      kwargs["minimize"] = path_minimize();
      kwargs["max_repeats"] = max_repeats_;
      kwargs["max_time"] = max_time_;
      kwargs["optlib"] = optlib;
      kwargs["slicing_opts"] = slicing_opts;
      kwargs["reconf_opts"] = reconf_opts;
      kwargs["progbar"] = false;

      // aer-0026: state the worker count instead of letting cotengra's 'auto'
      // read the whole node's CPU count through the container. Omitted entirely
      // when the knob asks for 'auto', so the old behaviour stays reachable for
      // an A/B without a rebuild.
      //
      // This CHANGES PATHS. Worker count decides how many trials finish inside
      // max_time and in what order, so plans found before and after this commit
      // are not comparable and neither are timings taken against them.
      const int par = path_parallel_setting();
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

      auto opt = ctg.attr("HyperOptimizer")(**kwargs);
      try {
        tree = opt.attr("search")(inputs, output, sizes);
      } catch (py::error_already_set &e) {
        // cotengra raises KeyError('tree') when every hyperopt trial failed
        // (no contraction tree was ever produced). Surface a named,
        // actionable error instead of the bare KeyError.
        std::ostringstream msg;
        msg << "CotengPathOptimizer: cotengra found no feasible contraction "
               "path under the current slicing constraints (target_size="
            << target_elements
            << " elements, allow_outer=false). Underlying error: " << e.what();
        throw std::runtime_error(msg.str());
      }
    }

    // AER_TN_FORCE_SLICING=1: if the natural path needed no slicing, slice
    // anyway (at least 2 slices). This is the validation hook that lets the
    // slicer's projection/accumulation machinery run on small circuits whose
    // results can be checked element-wise against an independent reference.
    if (force_slicing()) {
      py::dict sliced_inds = tree.attr("sliced_inds");
      if (py::len(sliced_inds) == 0) {
        tree.attr("slice")(py::arg("target_slices") = 2,
                           py::arg("allow_outer") = false,
                           py::arg("seed") = seed,
                           py::arg("inplace") = true);
        if (path_verbose()) {
          fprintf(stderr,
                  "[AER_TN_PATH] AER_TN_FORCE_SLICING=1: sliced an "
                  "already-fitting path over %zu mode(s)\n",
                  (size_t)py::len(tree.attr("sliced_inds")));
        }
      }
    }

    // Enforce the m6n6k6 tiling envelope as ACHIEVED, not merely targeted.
    // slicing_opts aim the HyperOptimizer at target_size during search, but the
    // returned tree is not guaranteed to reach it -- the search is heuristic,
    // and a low-FLOPs under-sliced tree can score best; across the MPI ensemble
    // the MINLOC(FLOPs) pick then actively favors the laxest rank (the 8-GCD
    // miscompute). Slice the tree explicitly to target so every plan leaving
    // this function -- and thus every candidate the ensemble can select -- has
    // a peak inside the envelope. allow_outer=false confines this to summed
    // bonds, matching the engine's accumulator and extract_plan's guard.
    {
      for (int guard = 0; guard < 128; ++guard) {
        int64_t cur = tree.attr("max_size")().cast<int64_t>();
        if (static_cast<uint64_t>(cur) <= target_elements)
          break;
        const size_t before = py::len(tree.attr("sliced_inds"));
        tree.attr("slice")(py::arg("target_size") = target_elements,
                           py::arg("allow_outer") = false,
                           py::arg("seed") = seed,
                           py::arg("inplace") = true);
        if (py::len(tree.attr("sliced_inds")) == before)
          break; // no summed bond left to slice: residual peak is output-bound
      }
      const int64_t final_max = tree.attr("max_size")().cast<int64_t>();
      if (static_cast<uint64_t>(final_max) > target_elements) {
        std::ostringstream msg;
        msg << "CotengPathOptimizer: peak intermediate (" << final_max
            << " elements) exceeds the m6n6k6 tiling envelope ("
            << target_elements
            << ") and cannot be reduced further by slicing -- the residual peak "
               "is bounded by output (open) modes the engine cannot split. This "
               "output rank exceeds m6n6k6 tiling; request a lower-rank output "
               "(expectation value, amplitude, or a smaller reduced density "
               "matrix).";
        throw std::runtime_error(msg.str());
      }
      if (path_verbose()) {
        fprintf(stderr,
                "[AER_TN_PATH] tiling envelope enforced: peak %lld <= target "
                "%llu (%zu sliced modes)\n",
                (long long)final_max, (unsigned long long)target_elements,
                (size_t)py::len(tree.attr("sliced_inds")));
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
          tree.attr("slice")(py::arg("target_slices") = min_slices_,
                             py::arg("allow_outer") = false,
                             py::arg("seed") = seed,
                             py::arg("inplace") = true);
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
    for (auto &item : sliced_inds)
      n *= static_cast<uint64_t>(item.second.attr("size").cast<int64_t>());
    return n;
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

    // Per-rank cotengra search uses a rank-dependent seed, so one rank can hit
    // the planner's free-mode gate while others succeed. NeedsTilingException
    // must NOT propagate from a single rank here: the ranks that did not throw
    // would march into the MPI_Allreduce below and desynchronize the collective
    // (deadlock/abort). Instead, catch it locally, turn it into a flag, and make
    // the decision collective: if ANY rank needs tiling, EVERY rank throws
    // together, so the driver's one-shot retry re-plans all ranks with tiling
    // engaged in lockstep. A non-tiling failure (OOM, CK error) is not this type
    // and is left to propagate uniformly.
    ContractionPlan local;
    int local_status = 0; // 0 ok, 1 needs tiling, 2 hard failure
    std::string local_msg;
    try {
      local = inner_->find_path(network, memory_limit_bytes, seed + rank,
                                tiling_available);
    } catch (const NeedsTilingException &) {
      local_status = 1;
    } catch (const std::exception &e) {
      // aer-0037: this catch is new and it closes a real hang. The per-rank
      // seed is seed + rank, so ranks search DIFFERENT trajectories and a
      // failure is genuinely not uniform: cotengra's "no feasible contraction
      // path" error, a py::error_already_set from a dead loky worker, or a
      // std::bad_alloc can strike one rank and not another. Before this, such
      // an exception escaped BEFORE the Allreduce below, so the throwing rank
      // unwound while every sibling blocked in that Allreduce forever.
      // py::error_already_set derives from std::runtime_error, so it is caught
      // here. The NeedsTilingException catch must stay first: it also derives
      // from std::runtime_error and is a control signal, not a failure.
      local_status = 2;
      local_msg = e.what();
    } catch (...) {
      // Nothing of unknown type may escape either: it would unwind this
      // rank alone and leave every sibling in the Allreduce below.
      local_status = 2;
      local_msg =
          "[AER_TN] the contraction path search raised an exception of "
          "unknown type on this rank; caught so the ranks fail together.";
    }

    // MPI_MAX: a hard failure on any rank outranks a tiling retry on another,
    // so a real error is never masked by a re-plan.
    int global_status = 0;
    MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX, comm_);
    if (global_status >= 2) {
      if (local_status >= 2)
        throw std::runtime_error(local_msg);
      throw std::runtime_error(
          "[AER_TN] MPI path search aborted: at least one rank failed during "
          "its contraction-path search, so every rank fails together rather "
          "than blocking at the cross-rank reduction. The failing rank reports "
          "the underlying cause.");
    }
    if (global_status == 1) {
      throw NeedsTilingException(
          NeedsTilingException::Gate::Planner,
          "MPI path search: at least one rank produced an above-envelope "
          "output path; re-planning all ranks with tiling engaged.");
    }

    struct { double cost; int rank; } local_result, global_result;
    local_result.cost = local.total_flops;
    local_result.rank = rank;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE_INT,
                  MPI_MINLOC, comm_);

    if (path_verbose() && rank == 0) {
      fprintf(stderr,
              "[AER_TN_PATH] MPI: %d ranks, best from rank %d (%.2e FLOPs)\n",
              size, global_result.rank, global_result.cost);
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
