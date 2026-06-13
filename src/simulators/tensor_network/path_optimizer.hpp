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
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
// Default derives from the hipTensor m6n6k6 envelope: a kernel-fitting step
// has intermediate rank <= 12, so peak per-slice intermediate is at most
// 2^12 = 4096 elements; one logical complex element occupies 16 bytes
// (double, split-complex: two 8-byte real planes), giving 65536 bytes. The
// budget converts to a cotengra `target_size` in ELEMENTS using each
// optimizer's element size, so a single value covers float and double.
// Lower it to slice harder (e.g. when interior steps still trip the K<=6
// limit); raise it only if a future hipTensor ships higher kernels.
static uint64_t slice_target_bytes() {
  static bool checked = false;
  static uint64_t cached = 65536;
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
// contraction at the cost of longer planning. Defaults match the historical
// hardcoded values (128 trials / 60 s). Lower them for fast iteration; raise
// them for a hard high-treewidth network where plan quality dominates
// execution. Read once and cached, so a process uses one consistent budget.
static int path_max_repeats() {
  static bool checked = false;
  static int cached = 128;
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
  static double cached = 60.0;
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
                      size_t element_size_bytes = 16)
      : minimize_(minimize),
        max_repeats_(max_repeats > 0 ? max_repeats : path_max_repeats()),
        max_time_(max_time > 0.0 ? max_time : path_max_time()),
        preset_(preset), element_size_bytes_(element_size_bytes) {}

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
    // envelope is by far the tighter bound on LUMI (4 Ki elements vs ~7 Gi);
    // taking the min keeps the memory bound authoritative for tiny GPUs and
    // makes the planner's behavior identical on every node class otherwise.
    uint64_t target_elements = memory_limit_bytes / element_size_bytes_;
    uint64_t envelope_elements =
        std::max<uint64_t>(1, slice_target_bytes() / element_size_bytes_);
    target_elements = std::min(target_elements, envelope_elements);

    // --- Output feasibility (OI15 made concrete by the first n=14 GPU run) -
    //
    // 1) Free-mode envelope. The m6n6k6 kernels admit at most 6 M-modes and
    //    6 N-modes per contraction, so no pairwise step can PRODUCE a tensor
    //    with more than 12 free modes. A request whose output carries more
    //    than 12 open modes (e.g. a full statevector save at n > 12) is
    //    infeasible for this backend at any slicing. Refuse up front with
    //    the real reason instead of letting cotengra exhaust dozens of
    //    hyperopt trials ("Ran out of valid indices to slice") and die in a
    //    bare KeyError('tree').
    constexpr size_t kMaxFreeModes = 12; // 6 M-modes + 6 N-modes (m6n6k6)
    if (network.output_modes.size() > kMaxFreeModes && !tiling_available) {
      std::ostringstream msg;
      msg << "CotengPathOptimizer: requested output has "
          << network.output_modes.size()
          << " free modes, but hipTensor's m6n6k6 contraction kernels can "
             "produce at most "
          << kMaxFreeModes
          << " free modes per step. Full-statevector extraction above 12 "
             "qubits is infeasible on this backend; request expectation "
             "values, amplitudes, or other low-rank outputs instead "
             "(slicing is fully effective for those), or enable mode tiling "
             "with AER_TN_TILING=on (or auto) to decompose oversized output "
             "steps into m6n6k6 sub-contractions.";
      if (tn_tiling_mode() == TilingMode::Auto)
        throw NeedsTilingException(NeedsTilingException::Gate::Planner,
                                   msg.str());
      throw std::runtime_error(msg.str());
    }

    // 2) Output-size clamp. Outer (output) modes are never sliced by design
    //    (allow_outer = false everywhere below), so slicing can shrink
    //    intermediates but can NEVER shrink the output tensor itself. Any
    //    target below the output's own element count therefore makes the
    //    slicing problem infeasible by construction, and cotengra correctly
    //    fails every trial. Clamp the target to at least the output size;
    //    intermediates still get pushed toward the envelope wherever the
    //    output permits. (Per-step descriptor feasibility for an
    //    above-envelope output is guaranteed by check (1): <= 12 free modes
    //    of extent 2 is <= 4096 elements <= one m6n6k6 result.)
    uint64_t output_elements = 1;
    for (auto e : network.output_extents)
      output_elements *= static_cast<uint64_t>(e);
    if (output_elements > target_elements) {
      if (path_verbose()) {
        fprintf(stderr,
                "[AER_TN_PATH] target_size raised %llu -> %llu elements: "
                "output tensor cannot be sliced (allow_outer=false) and "
                "bounds the per-slice peak from below\n",
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

    return extract_plan(tree, std::set<int32_t>(network.output_modes.begin(),
                                                network.output_modes.end()));
  }

private:
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
    // Same two-ceiling budget as the cotengra path: device memory and the
    // m6n6k6 envelope, in elements (16 bytes per logical complex element).
    int64_t element_budget =
        static_cast<int64_t>(memory_limit_bytes / 16);
    int64_t envelope_budget =
        std::max<int64_t>(1, static_cast<int64_t>(slice_target_bytes() / 16));
    element_budget = std::min(element_budget, envelope_budget);
    while (best.peak_intermediate_elements > element_budget &&
           !size_dict.empty()) {
      int32_t best_mode = -1;
      int64_t best_reduction = 0;
      for (auto it = size_dict.begin(); it != size_dict.end(); ++it) {
        if (it->second <= 1)
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
    int local_needs_tiling = 0;
    try {
      local = inner_->find_path(network, memory_limit_bytes, seed + rank,
                                tiling_available);
    } catch (const NeedsTilingException &) {
      local_needs_tiling = 1;
    }

    int any_needs_tiling = 0;
    MPI_Allreduce(&local_needs_tiling, &any_needs_tiling, 1, MPI_INT, MPI_MAX,
                  comm_);
    if (any_needs_tiling) {
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

    std::vector<int64_t> path_data;
    int path_size = 0;
    if (rank == global_result.rank) {
      path_data = local.serialize();
      path_size = static_cast<int>(path_data.size());
    }
    MPI_Bcast(&path_size, 1, MPI_INT, global_result.rank, comm_);
    path_data.resize(path_size);
    MPI_Bcast(path_data.data(), path_size, MPI_INT64_T, global_result.rank,
              comm_);
    return ContractionPlan::deserialize(path_data);
  }
};

#endif // AER_MPI

} // namespace TensorNetwork
} // namespace AER

#endif // _path_optimizer_hpp_
