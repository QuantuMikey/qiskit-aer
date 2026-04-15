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
// Diagnostic logging
//=============================================================================

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
// 2. PathOptimizer abstract interface
//=============================================================================

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

class CotengPathOptimizer : public PathOptimizer {
  std::string minimize_;
  int max_repeats_;
  double max_time_;
  std::string preset_;
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
    py::gil_scoped_acquire gil;

    auto ctg = py::module_::import("cotengra");

    py::list inputs;
    for (size_t i = 0; i < network.tensors.size(); i++) {
      py::tuple modes = py::cast(network.tensors[i].modes);
      inputs.append(modes);
    }
    py::tuple output = py::cast(network.output_modes);

    auto size_dict = network.build_size_dict();
    py::dict sizes = py::cast(size_dict);

    uint64_t target_elements = memory_limit_bytes / element_size_bytes_;

    py::dict slicing_opts;
    slicing_opts["target_size"] = target_elements;

    py::dict reconf_opts;

    py::object tree;

    if (preset_ == "random-greedy") {
      auto opt = ctg.attr("RandomGreedyOptimizer")(
          py::arg("max_repeats") = max_repeats_,
          py::arg("max_time") = max_time_,
          py::arg("seed") = seed,
          py::arg("progbar") = false);
      tree = opt.attr("search")(inputs, output, sizes);
    } else {
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

    return extract_plan(tree);
  }

private:
  ContractionPlan extract_plan(py::object &tree) {
    ContractionPlan plan;

    py::object ssa_path = tree.attr("get_ssa_path")();
    for (auto &pair : ssa_path) {
      auto t = pair.cast<py::tuple>();
      ContractionStep step;
      step.left = t[0].cast<uint64_t>();
      step.right = t[1].cast<uint64_t>();
      plan.steps.push_back(step);
    }

    py::dict sliced_inds = tree.attr("sliced_inds");
    plan.num_slices = 1;
    for (auto &item : sliced_inds) {
      SliceInfo si;
      si.mode = item.first.cast<int32_t>();
      auto slice_info_obj = item.second;
      si.extent = slice_info_obj.attr("size").cast<int64_t>();
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
// 4. GreedyPathOptimizer — C++ fallback, zero dependencies
//=============================================================================

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
    int64_t element_budget =
        static_cast<int64_t>(memory_limit_bytes / 16);
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
    for (size_t i = 0; i < network.tensors.size(); i++) {
      working.push_back(network.tensors[i]);
    }

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
        if (!alive[i])
          continue;
        for (size_t j = i + 1; j < working.size(); j++) {
          if (!alive[j])
            continue;

          bool share = false;
          for (size_t mi = 0; mi < working[i].modes.size(); mi++) {
            for (size_t mj = 0; mj < working[j].modes.size(); mj++) {
              if (working[i].modes[mi] == working[j].modes[mj]) {
                share = true;
                break;
              }
            }
            if (share)
              break;
          }

          double cost = compute_cost(working[i], working[j], size_dict);
          if (!share)
            cost *= 1e6;
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
      if (working.size() < working.capacity()) {
        working.push_back(result);
        alive[working.size() - 1] = true;
      } else {
        working.push_back(result);
        alive.push_back(true);
      }
    }

    return plan;
  }

  static double compute_cost(const TensorSpec &a, const TensorSpec &b,
                             const std::map<int32_t, int64_t> &size_dict) {
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
    for (size_t i = 0; i < a.modes.size(); i++) {
      for (size_t j = 0; j < b.modes.size(); j++) {
        if (a.modes[i] == b.modes[j]) {
          shared.push_back(a.modes[i]);
          break;
        }
      }
    }

    for (size_t i = 0; i < a.modes.size(); i++) {
      bool is_shared = false;
      for (size_t s = 0; s < shared.size(); s++) {
        if (a.modes[i] == shared[s]) {
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
      for (size_t s = 0; s < shared.size(); s++) {
        if (b.modes[i] == shared[s]) {
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
// 5. MPIParallelPathOptimizer
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
                            uint64_t seed) override {
    int rank, size;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &size);

    ContractionPlan local =
        inner_->find_path(network, memory_limit_bytes, seed + rank);

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

//------------------------------------------------------------------------------
} // end namespace TensorNetwork
} // end namespace AER
//------------------------------------------------------------------------------

#endif // _path_optimizer_hpp_
