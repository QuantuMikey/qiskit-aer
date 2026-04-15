/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2022.
 * (C) Copyright CSC - IT Center for Science Ltd. 2026.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _path_optimizer_hpp_
#define _path_optimizer_hpp_

/**
 * Path optimization for tensor network contraction.
 *
 * This file contains the contraction path finding subsystem. It has zero
 * GPU dependencies and can be compiled and tested on any machine. The three
 * components are:
 *
 *   CotengPathOptimizer   — calls cotengra's HyperOptimizer via pybind11.
 *                           Same algorithm class as cuTensorNet (graph
 *                           partitioning + subtree reconfiguration + slicing).
 *                           Default for production use at any scale.
 *
 *   GreedyPathOptimizer   — simple C++ greedy. Zero dependencies. Used only
 *                           when cotengra is unavailable (CI, testing).
 *
 *   MPIParallelPathOptimizer — wraps any optimizer. Runs it on every MPI rank
 *                              with a different seed, keeps the best path.
 *                              Core feature for multi-node runs.
 *
 * File layout:
 *   1. Data structures (NetworkDescription, TensorSpec, ContractionPlan)
 *   2. PathOptimizer abstract interface
 *   3. CotengPathOptimizer implementation
 *   4. GreedyPathOptimizer implementation
 *   5. MPIParallelPathOptimizer implementation
 */

#include <algorithm>
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

//=============================================================================
// Diagnostic logging — controlled by environment variables
//=============================================================================

// Check once at first use, cache the result.
// AER_TN_PATH_VERBOSE=1 enables path optimizer diagnostics.
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

//=============================================================================
// 1. Data structures
//=============================================================================

/**
 * Specification of a single tensor in the network.
 *
 * This is the optimizer's view of a tensor — it only needs to know
 * the mode indices and their extents to plan the contraction order.
 * It does not hold tensor data (that lives on the GPU).
 */
struct TensorSpec {
  std::vector<int32_t> modes;   // which edges this tensor connects to
  std::vector<int64_t> extents; // size of each mode (2 for qubits)

  // Number of elements in this tensor (product of extents)
  int64_t num_elements() const {
    int64_t n = 1;
    for (auto e : extents)
      n *= e;
    return n;
  }
};

/**
 * Description of a tensor network to be contracted.
 *
 * This is the input to any PathOptimizer. It describes the topology
 * of the network (which tensors connect to which) without holding
 * any tensor data.
 */
struct NetworkDescription {
  std::vector<TensorSpec> tensors; // all input tensors
  std::vector<int32_t> output_modes;   // modes in the output tensor
  std::vector<int64_t> output_extents; // extents of output modes

  // Build a size dictionary: mode index -> extent.
  // Used by cotengra and by the greedy optimizer.
  std::map<int32_t, int64_t> build_size_dict() const {
    std::map<int32_t, int64_t> sizes;
    for (const auto &t : tensors) {
      for (size_t i = 0; i < t.modes.size(); i++) {
        sizes[t.modes[i]] = t.extents[i];
      }
    }
    return sizes;
  }
};

/**
 * Information about a sliced mode.
 */
struct SliceInfo {
  int32_t mode;   // which mode is sliced
  int64_t extent; // original extent of that mode (number of slice values)
};

/**
 * A single step in the contraction path.
 *
 * Each step contracts two tensors (identified by their current indices
 * in the working list) to produce one intermediate tensor.
 */
struct ContractionStep {
  uint64_t left;  // index of left input tensor
  uint64_t right; // index of right input tensor
};

/**
 * The output of any path optimizer.
 *
 * Contains everything needed to execute the contraction: the ordered
 * steps, the slicing information, and cost estimates. This struct
 * is serializable for MPI broadcast.
 */
struct ContractionPlan {
  // Ordered pairwise contraction steps.
  // Indices refer to positions in a working list that starts as the
  // input tensor list and grows as intermediates are appended.
  // (This is the SSA format that cotengra calls "single static assignment".)
  std::vector<ContractionStep> steps;

  // Modes that were sliced to fit within memory, and their extents.
  std::vector<SliceInfo> sliced;

  // Total number of independent slices (product of sliced extents).
  uint64_t num_slices;

  // Estimated total FLOP count for one full contraction (all slices).
  double total_flops;

  // Estimated peak intermediate tensor size in number of elements.
  int64_t peak_intermediate_elements;

  // ---------------------------------------------------------------
  // Serialization for MPI broadcast.
  //
  // The format is a flat vector of int64_t values with a known layout
  // so any rank can deserialize without struct packing assumptions.
  //
  // Layout:
  //   [0]       = num_steps
  //   [1]       = num_sliced
  //   [2]       = num_slices
  //   [3..4]    = total_flops as two int64 (memcpy of double)
  //   [5]       = peak_intermediate_elements
  //   [6..6+2*num_steps-1] = steps (left, right pairs)
  //   [next..next+2*num_sliced-1] = sliced (mode, extent pairs)
  // ---------------------------------------------------------------

  std::vector<int64_t> serialize() const {
    std::vector<int64_t> data;
    int64_t num_steps = static_cast<int64_t>(steps.size());
    int64_t num_sliced_modes = static_cast<int64_t>(sliced.size());

    data.push_back(num_steps);
    data.push_back(num_sliced_modes);
    data.push_back(static_cast<int64_t>(num_slices));

    // Store double as two int64 via memcpy to avoid type punning UB
    int64_t flops_bits[1];
    std::memcpy(flops_bits, &total_flops, sizeof(double));
    data.push_back(flops_bits[0]);

    data.push_back(peak_intermediate_elements);

    for (const auto &step : steps) {
      data.push_back(static_cast<int64_t>(step.left));
      data.push_back(static_cast<int64_t>(step.right));
    }

    for (const auto &s : sliced) {
      data.push_back(static_cast<int64_t>(s.mode));
      data.push_back(s.extent);
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
// 2. PathOptimizer abstract interface
//=============================================================================

/**
 * Abstract base class for contraction path optimizers.
 *
 * Every optimizer takes a NetworkDescription and a memory limit,
 * and returns a ContractionPlan. The seed parameter enables
 * deterministic randomized search and allows MPI ranks to explore
 * different regions of the search space.
 */
class PathOptimizer {
public:
  virtual ~PathOptimizer() = default;

  virtual ContractionPlan find_path(const NetworkDescription &network,
                                    uint64_t memory_limit_bytes,
                                    uint64_t seed) = 0;
};

//=============================================================================
// 3. CotengPathOptimizer — cotengra HyperOptimizer via pybind11
//=============================================================================

#ifdef AER_HIPTENSOR

/**
 * Calls cotengra's HyperOptimizer with kahypar graph partitioning
 * and subtree reconfiguration. This is the same class of algorithm
 * that NVIDIA's cuTensorNet uses.
 *
 * The Python interpreter is already alive (Aer was called from Python).
 * We acquire the GIL, import cotengra, run the optimizer, extract the
 * results, and release the GIL. cotengrust accelerates the inner
 * greedy subroutines at native Rust speed and releases the GIL
 * during its own computation.
 *
 * All parameters (minimize, max_repeats, max_time, preset) come from
 * the Aer backend configuration. Nothing is hardcoded.
 */
class CotengPathOptimizer : public PathOptimizer {
  std::string minimize_;
  int max_repeats_;
  double max_time_;
  std::string preset_; // "hyper" or "random-greedy"
  size_t element_size_bytes_;

public:
  CotengPathOptimizer(const std::string &minimize = "combo",
                      int max_repeats = 128, double max_time = 60.0,
                      const std::string &preset = "hyper",
                      size_t element_size_bytes = 16)
      : minimize_(minimize), max_repeats_(max_repeats), max_time_(max_time),
        preset_(preset), element_size_bytes_(element_size_bytes) {}

  ContractionPlan find_path(const NetworkDescription &network,
                            uint64_t memory_limit_bytes,
                            uint64_t seed) override {
    // Acquire the GIL — we are calling into Python.
    // The GIL is released automatically when this object destructs.
    py::gil_scoped_acquire gil;

    auto ctg = py::module_::import("cotengra");

    // Convert network to cotengra's input format:
    //   inputs = [tuple of mode indices per tensor]
    //   output = tuple of output mode indices
    //   size_dict = {mode_index: extent}
    py::list inputs;
    for (const auto &tensor : network.tensors) {
      py::tuple modes = py::cast(tensor.modes);
      inputs.append(modes);
    }
    py::tuple output = py::cast(network.output_modes);

    auto size_dict = network.build_size_dict();
    py::dict sizes = py::cast(size_dict);

    // Memory target for cotengra's slicer, in number of elements.
    // cotengra thinks in elements, not bytes. We convert.
    uint64_t target_elements = memory_limit_bytes / element_size_bytes_;

    // Slicing options: cotengra will slice modes until the peak
    // intermediate fits within target_size.
    py::dict slicing_opts;
    slicing_opts["target_size"] = target_elements;

    // Subtree reconfiguration: after finding an initial path via
    // graph partitioning, cotengra optimizes subtrees to further
    // reduce the FLOP count. Empty dict = use cotengra's defaults.
    py::dict reconf_opts;

    py::object tree;

    if (preset_ == "random-greedy") {
      // Fast preset for very large networks (10,000+ tensors).
      // Uses cotengrust at native Rust speed.
      auto opt = ctg.attr("RandomGreedyOptimizer")(
          py::arg("max_repeats") = max_repeats_,
          py::arg("max_time") = max_time_,
          py::arg("seed") = seed,
          py::arg("progbar") = false);
      tree = opt.attr("search")(inputs, output, sizes);
    } else {
      // Default: HyperOptimizer with kahypar graph partitioning
      // and subtree reconfiguration. Same algorithm class as cuTensorNet.
      auto opt = ctg.attr("HyperOptimizer")(
          py::arg("minimize") = minimize_,
          py::arg("max_repeats") = max_repeats_,
          py::arg("max_time") = max_time_,
          py::arg("seed") = seed,
          py::arg("slicing_opts") = slicing_opts,
          py::arg("reconf_opts") = reconf_opts,
          py::arg("progbar") = false);
      tree = opt.attr("search")(inputs, output, sizes);
    }

    // Extract the contraction plan from the ContractionTree
    return extract_plan(tree);
  }

private:
  /**
   * Convert cotengra's ContractionTree to our ContractionPlan.
   */
  ContractionPlan extract_plan(py::object &tree) {
    ContractionPlan plan;

    // Path: list of (int, int) pairs in SSA format
    // cotengra's get_path() returns linear recycled format by default.
    // We use ssa_path for cleaner indexing.
    py::object ssa_path = tree.attr("get_ssa_path")();
    for (auto &pair : ssa_path) {
      auto t = pair.cast<py::tuple>();
      ContractionStep step;
      step.left = t[0].cast<uint64_t>();
      step.right = t[1].cast<uint64_t>();
      plan.steps.push_back(step);
    }

    // Sliced indices: dict of {index_label: SliceInfo}
    py::dict sliced_inds = tree.attr("sliced_inds");
    plan.num_slices = 1;
    for (auto &item : sliced_inds) {
      SliceInfo si;
      si.mode = item.first.cast<int32_t>();
      // cotengra stores SliceInfo objects with a .size attribute
      auto slice_info_obj = item.second;
      si.extent = slice_info_obj.attr("size").cast<int64_t>();
      plan.sliced.push_back(si);
      plan.num_slices *= static_cast<uint64_t>(si.extent);
    }

    // Cost estimates from cotengra
    plan.total_flops = tree.attr("total_flops")().cast<double>();

    // Peak intermediate size in number of elements.
    // cotengra's combo_cost or peak_size method.
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
// 4. GreedyPathOptimizer — C++ fallback, zero dependencies
//=============================================================================

/**
 * Simple greedy contraction path optimizer.
 *
 * At each step, finds the pair of tensors whose contraction has the
 * lowest FLOP cost, contracts them, and repeats. Ties are broken by
 * random perturbation (seeded for reproducibility). Multiple restarts
 * with different seeds explore the space.
 *
 * This is NOT the default optimizer. It exists for two purposes:
 *   1. Unit testing the contraction engine without Python/cotengra.
 *   2. Fallback when cotengra is not installed (CI, bare-metal builds).
 *
 * For production use, CotengPathOptimizer (cotengra HyperOptimizer)
 * produces dramatically better paths.
 */
class GreedyPathOptimizer : public PathOptimizer {
  int num_restarts_;

public:
  explicit GreedyPathOptimizer(int num_restarts = 32)
      : num_restarts_(num_restarts) {}

  ContractionPlan find_path(const NetworkDescription &network,
                            uint64_t memory_limit_bytes,
                            uint64_t seed) override {
    auto size_dict = network.build_size_dict();
    ContractionPlan best;
    best.total_flops = std::numeric_limits<double>::max();

    for (int restart = 0; restart < num_restarts_; restart++) {
      ContractionPlan candidate =
          greedy_once(network, size_dict, seed + restart);
      if (candidate.total_flops < best.total_flops) {
        best = candidate;
      }
    }

    // Apply slicing if peak intermediate exceeds memory limit
    best.num_slices = 1;
    // Greedy slicing: iteratively slice the mode that reduces peak most
    int64_t element_budget =
        static_cast<int64_t>(memory_limit_bytes / 16); // complex<double>
    while (best.peak_intermediate_elements > element_budget &&
           !size_dict.empty()) {
      // Find the mode that, when sliced, reduces peak the most.
      // For qubit circuits (extent 2), any sliced mode halves the peak.
      int32_t best_mode = -1;
      int64_t best_reduction = 0;
      for (auto &[mode, extent] : size_dict) {
        if (extent <= 1)
          continue;
        int64_t reduction = best.peak_intermediate_elements -
                            best.peak_intermediate_elements / extent;
        if (reduction > best_reduction) {
          best_reduction = reduction;
          best_mode = mode;
        }
      }
      if (best_mode < 0)
        break; // nothing left to slice

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
  /**
   * One greedy pass. Contracts the cheapest pair at each step.
   * Ties broken by seeded random perturbation.
   */
  ContractionPlan greedy_once(const NetworkDescription &network,
                              const std::map<int32_t, int64_t> &size_dict,
                              uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> noise(0.0, 1e-12);

    // Working list of tensor specs. Starts as input tensors.
    // New intermediates are appended (SSA format).
    std::vector<TensorSpec> working;
    working.reserve(network.tensors.size() * 2);
    for (const auto &t : network.tensors) {
      working.push_back(t);
    }

    // Track which tensors are still "alive" (not yet consumed)
    std::vector<bool> alive(working.capacity(), false);
    for (size_t i = 0; i < network.tensors.size(); i++)
      alive[i] = true;

    ContractionPlan plan;
    plan.total_flops = 0.0;
    plan.peak_intermediate_elements = 0;

    size_t num_tensors = network.tensors.size();
    // Number of contractions = num_input_tensors - 1
    // (unless there are output modes, in which case it may differ)
    for (size_t step = 0; step + 1 < num_tensors; step++) {
      // Find all pairs of alive tensors that share at least one mode
      uint64_t best_i = 0, best_j = 0;
      double best_cost = std::numeric_limits<double>::max();

      for (size_t i = 0; i < working.size(); i++) {
        if (!alive[i])
          continue;
        for (size_t j = i + 1; j < working.size(); j++) {
          if (!alive[j])
            continue;

          // Check if they share any modes
          bool share = false;
          for (auto m : working[i].modes) {
            for (auto n : working[j].modes) {
              if (m == n) {
                share = true;
                break;
              }
            }
            if (share)
              break;
          }

          // Score: FLOP cost of contracting this pair.
          // Cost = product of all unique extents involved.
          double cost = compute_cost(working[i], working[j], size_dict);

          // If no shared modes, this is an outer product — expensive.
          // Penalize but don't exclude.
          if (!share) {
            cost *= 1e6;
          }

          // Add tiny random noise to break ties deterministically
          cost += noise(rng);

          if (cost < best_cost) {
            best_cost = cost;
            best_i = i;
            best_j = j;
          }
        }
      }

      // Record the step
      ContractionStep cs;
      cs.left = best_i;
      cs.right = best_j;
      plan.steps.push_back(cs);
      plan.total_flops += best_cost;

      // Compute the result tensor spec
      TensorSpec result = contract_spec(working[best_i], working[best_j]);
      plan.peak_intermediate_elements =
          std::max(plan.peak_intermediate_elements, result.num_elements());

      // Mark inputs as consumed, append result
      alive[best_i] = false;
      alive[best_j] = false;
      if (working.size() < working.capacity()) {
        working.push_back(result);
        alive[working.size() - 1] = true;
      } else {
        // Shouldn't happen with reserve, but handle gracefully
        working.push_back(result);
        alive.push_back(true);
      }
    }

    return plan;
  }

  /**
   * FLOP cost of contracting two tensors.
   * Cost = product of all unique mode extents across both tensors.
   */
  static double compute_cost(const TensorSpec &a, const TensorSpec &b,
                             const std::map<int32_t, int64_t> &size_dict) {
    // Collect all unique modes from both tensors
    std::map<int32_t, int64_t> all_modes;
    for (size_t i = 0; i < a.modes.size(); i++)
      all_modes[a.modes[i]] = a.extents[i];
    for (size_t i = 0; i < b.modes.size(); i++)
      all_modes[b.modes[i]] = b.extents[i];

    double cost = 1.0;
    for (auto &[mode, extent] : all_modes)
      cost *= static_cast<double>(extent);
    return cost;
  }

  /**
   * Compute the TensorSpec of the result of contracting a and b.
   * Result modes = union of both modes minus any modes shared between them
   * (shared modes are summed over / contracted).
   */
  static TensorSpec contract_spec(const TensorSpec &a, const TensorSpec &b) {
    TensorSpec result;

    // Find shared modes (the ones being contracted)
    std::vector<int32_t> shared;
    for (auto m : a.modes) {
      for (auto n : b.modes) {
        if (m == n) {
          shared.push_back(m);
          break;
        }
      }
    }

    // Result modes: everything from a and b that isn't shared
    for (size_t i = 0; i < a.modes.size(); i++) {
      bool is_shared = false;
      for (auto s : shared) {
        if (a.modes[i] == s) {
          is_shared = true;
          break;
        }
      }
      if (!is_shared) {
        result.modes.push_back(a.modes[i]);
        result.extents.push_back(a.extents[i]);
      }
    }
    for (size_t i = 0; i < b.modes.size(); i++) {
      bool is_shared = false;
      for (auto s : shared) {
        if (b.modes[i] == s) {
          is_shared = true;
          break;
        }
      }
      if (!is_shared) {
        result.modes.push_back(b.modes[i]);
        result.extents.push_back(b.extents[i]);
      }
    }

    return result;
  }
};

//=============================================================================
// 5. MPIParallelPathOptimizer — core feature for multi-node runs
//=============================================================================

#ifdef AER_MPI

/**
 * Wraps any PathOptimizer and runs it on every MPI rank with a
 * different seed. Uses MPI_MINLOC to find which rank found the
 * best path (lowest total FLOP count), then broadcasts the winner.
 *
 * On a 1024-node run with the default HyperOptimizer (128 repeats
 * per rank), this explores 131,072 independent hyper-optimized
 * contraction paths and keeps the best one — more parallel path
 * exploration than any cuTensorNet deployment.
 *
 * This class is a core architectural feature, not a performance
 * optimization. It is always active when MPI is available.
 */
class MPIParallelPathOptimizer : public PathOptimizer {
  std::unique_ptr<PathOptimizer> inner_;
  MPI_Comm comm_;

public:
  MPIParallelPathOptimizer(std::unique_ptr<PathOptimizer> inner,
                           MPI_Comm comm = MPI_COMM_WORLD)
      : inner_(std::move(inner)), comm_(comm) {}

  ContractionPlan find_path(const NetworkDescription &network,
                            uint64_t memory_limit_bytes,
                            uint64_t seed) override {
    int rank, size;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &size);

    // Every rank runs the optimizer with a different seed.
    // For CotengPathOptimizer, this means each rank samples different
    // hyper-parameters (partition imbalance, greedy temperature, costmod),
    // producing diverse high-quality paths. cotengrust releases the GIL
    // and runs at native Rust speed on each rank's CPU cores.
    ContractionPlan local =
        inner_->find_path(network, memory_limit_bytes, seed + rank);

    // Find which rank got the best path (lowest total FLOP count)
    struct {
      double cost;
      int rank;
    } local_result, global_result;
    local_result.cost = local.total_flops;
    local_result.rank = rank;

    MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE_INT,
                  MPI_MINLOC, comm_);

    if (path_verbose()) {
      if (rank == 0) {
        fprintf(stderr,
                "[AER_TN_PATH] MPI parallel search: %d ranks, "
                "best path from rank %d with %.2e FLOPs "
                "(local rank 0 had %.2e FLOPs)\n",
                size, global_result.rank, global_result.cost,
                local.total_flops);
      }
    }

    // Winner serializes and broadcasts the path.
    // Serialization uses a flat int64 vector with a known layout
    // so there are no struct packing or endianness issues.
    std::vector<int64_t> path_data;
    int path_size = 0;

    if (rank == global_result.rank) {
      path_data = local.serialize();
      path_size = static_cast<int>(path_data.size());
    }

    // Broadcast the size first, then the data
    MPI_Bcast(&path_size, 1, MPI_INT, global_result.rank, comm_);
    path_data.resize(path_size);
    MPI_Bcast(path_data.data(), path_size, MPI_INT64_T, global_result.rank,
              comm_);

    return ContractionPlan::deserialize(path_data);
  }
};

#endif // AER_MPI

//------------------------------------------------------------------------------
} // end namespace TensorNetwork
} // end namespace AER
//------------------------------------------------------------------------------

#endif // _path_optimizer_hpp_
