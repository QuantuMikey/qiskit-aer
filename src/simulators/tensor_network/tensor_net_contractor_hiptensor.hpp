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

#ifndef _tensor_net_contractor_hiptensor_hpp_
#define _tensor_net_contractor_hiptensor_hpp_

#ifdef AER_THRUST_ROCM

/**
 * hipTensor-based tensor network contractor for AMD GPUs.
 *
 * This file composes path_optimizer.hpp (contraction path finding via
 * cotengra) and gpu_resource_manager.hpp (GPU memory and hipTensor
 * management) to implement the TensorNetContractor interface.
 *
 * The contractor's own logic is orchestration — it wires the path
 * optimizer output to the GPU resource manager and executes the
 * contraction plan step by step.
 *
 * File layout:
 *   1. sampling_update_rnd_func (GPU kernel for sampling)
 *   2. TensorNetContractor_HipTensor class declaration
 *   3. Implementation of each interface method
 */

#include <complex>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <vector>

#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include "misc/wrap_thrust.hpp"
#include "simulators/statevector/chunk/thrust_kernels.hpp"

#include "framework/types.hpp"
#include "framework/utils.hpp"

#include "simulators/tensor_network/tensor.hpp"
#include "simulators/tensor_network/tensor_net_contractor.hpp"
#include "simulators/tensor_network/path_optimizer.hpp"
#include "simulators/tensor_network/gpu_resource_manager.hpp"

#ifdef AER_MPI
#include <mpi.h>
#endif

namespace AER {
namespace TensorNetwork {

// Thrust execution policy namespace
namespace thrust_gpu = thrust::hip;

//=============================================================================
// Diagnostic logging
//=============================================================================

static bool tn_verbose() {
  static bool checked = false;
  static bool enabled = false;
  if (!checked) {
    const char *val = std::getenv("AER_TN_VERBOSE");
    enabled = (val != nullptr && std::string(val) == "1");
    checked = true;
  }
  return enabled;
}

//=============================================================================
// 1. Sampling GPU kernel (mirrors cuTensorNet contractor's version)
//=============================================================================

/**
 * GPU functor that updates random numbers by subtracting sampled
 * probabilities. Used in contract_and_sample_measure for iterative
 * measurement sampling.
 */
template <typename data_t> class sampling_update_rnd_func_hip {
  thrust::complex<data_t> *data_;
  uint_t stride_;
  uint_t *index_;
  double *rnds_;

public:
  sampling_update_rnd_func_hip(thrust::complex<data_t> *p, uint_t st,
                               uint_t *idx, double *rnd)
      : data_(p), stride_(st), index_(idx), rnds_(rnd) {}

  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t pos = index_[i];
    if (pos > 0) {
      double t = rnds_[i];
      pos = (pos - 1) * stride_;
      thrust::complex<data_t> d = data_[pos];
      t -= d.real();
      rnds_[i] = t;
    }
  }
};

//=============================================================================
// 2. TensorNetContractor_HipTensor — class declaration
//=============================================================================

template <typename data_t = double>
class TensorNetContractor_HipTensor : public TensorNetContractor<data_t> {
  // --- GPU resources ---
  GPUResourceManager<data_t> gpu_mgr_;
  int num_devices_used_;

  // --- Network description (built from input tensors) ---
  NetworkDescription network_desc_;
  std::vector<std::shared_ptr<Tensor<data_t>>> input_tensors_;
  bool add_sp_tensors_;
  uint_t num_base_tensors_;
  uint_t num_additional_tensors_;

  // Per-tensor device pointers on the primary GPU
  std::vector<void *> tensor_device_ptrs_;

  // --- Output ---
  std::vector<int32_t> modes_out_;
  std::vector<int64_t> extents_out_;
  uint_t out_size_;

  // --- Contraction plan (from path optimizer) ---
  ContractionPlan plan_;
  bool plan_valid_;

  // --- Slice distribution ---
  uint_t slice_begin_;
  uint_t slice_end_;

  // --- MPI ---
  int nprocs_;
  int myrank_;

  // --- Target GPUs ---
  reg_t target_gpus_;

  // --- Previous network topology for path reuse (VQE optimization) ---
  std::vector<std::vector<int32_t>> prev_modes_;
  std::vector<std::vector<int64_t>> prev_extents_;
  bool prev_valid_;

public:
  TensorNetContractor_HipTensor();
  ~TensorNetContractor_HipTensor();

  void set_device(int idev) override {}
  void allocate_additional_tensors(uint_t size) override;

  void set_network(const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors,
                   bool add_sp_tensors = true) override;
  void set_additional_tensors(
      const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors) override;
  void update_additional_tensors(
      const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors) override;
  void set_output(std::vector<int32_t> &modes,
                  std::vector<int64_t> &extents) override;

  void setup_contraction(bool use_autotune = false) override;
  uint_t num_slices(void) override { return plan_.num_slices; }

  void contract(std::vector<std::complex<data_t>> &out) override;
  double contract_and_trace(uint_t num_qubits) override;
  double contract_and_sample_measure(reg_t &samples, std::vector<double> &rnds,
                                     uint_t num_qubits) override;

  void allocate_sampling_buffers(
      uint_t size = AER_TENSOR_NET_MAX_SAMPLING) override;
  void deallocate_sampling_buffers(void) override;

  void set_target_gpus(reg_t &t) override { target_gpus_ = t; }

private:
  // Build a NetworkDescription from the current tensor list
  void build_network_description();

  // Check if the current network topology matches the previous one
  bool topology_matches_previous() const;

  // Cache the current topology for VQE path reuse
  void cache_topology();

  // Execute the contraction plan on the primary GPU for one slice.
  // Writes the result to the primary GPU's output buffer.
  void contract_single_slice(uint_t slice_index);

  // Project input tensor data for a given slice onto GPU.
  // Returns per-tensor device pointers for the projected data.
  void project_slice(uint_t slice_index,
                     std::vector<void *> &projected_ptrs);

  // Accumulate output across GPUs (tree reduction)
  void accumulate_across_gpus();

  // Accumulate output across MPI ranks
  void accumulate_across_mpi();

  // Execute the full contraction (all slices, all GPUs, all MPI ranks)
  void contract_all();

  // Sample measurement outcomes from the output density matrix diagonal
  double sample_measure_on_primary(reg_t &samples, std::vector<double> &rnds,
                                   uint_t num_qubits);

  // Create the path optimizer based on configuration
  std::unique_ptr<PathOptimizer> create_optimizer();
};

//=============================================================================
// 3. Implementation
//=============================================================================

template <typename data_t>
TensorNetContractor_HipTensor<data_t>::TensorNetContractor_HipTensor()
    : num_devices_used_(1), add_sp_tensors_(true), num_base_tensors_(0),
      num_additional_tensors_(0), out_size_(0), plan_valid_(false),
      slice_begin_(0), slice_end_(0), nprocs_(1), myrank_(0),
      prev_valid_(false) {}

template <typename data_t>
TensorNetContractor_HipTensor<data_t>::~TensorNetContractor_HipTensor() {}

// ---- set_network ----

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::set_network(
    const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors,
    bool add_sp_tensors) {
  input_tensors_.clear();
  add_sp_tensors_ = add_sp_tensors;
  num_additional_tensors_ = 0;

  // Store references to input tensors (filtering superop tensors if needed)
  for (const auto &t : tensors) {
    if (add_sp_tensors || !t->sp_tensor()) {
      input_tensors_.push_back(t);
    }
  }
  num_base_tensors_ = input_tensors_.size();

  // Discover GPUs if not already done
  if (gpu_mgr_.num_devices() == 0) {
    std::vector<uint64_t> targets(target_gpus_.begin(), target_gpus_.end());
    gpu_mgr_.discover(targets);
  }

  // Copy tensor data to the primary GPU
  tensor_device_ptrs_ =
      gpu_mgr_.primary().copy_tensor_data(input_tensors_, true);

  // Build the network description for the path optimizer
  build_network_description();

  if (tn_verbose()) {
    fprintf(stderr,
            "[AER_TN] set_network: %zu tensors on %zu GPU(s)\n",
            input_tensors_.size(), gpu_mgr_.num_devices());
  }
}

// ---- allocate_additional_tensors ----

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::allocate_additional_tensors(
    uint_t size) {
  // Pre-allocate space on GPU for measurement tensors that will be
  // added later via set_additional_tensors. The cuTensorNet contractor
  // does this as a separate allocation; we handle it the same way.
  // (The actual allocation happens in set_additional_tensors.)
}

// ---- set_additional_tensors ----

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::set_additional_tensors(
    const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors) {
  // Remove any previously added additional tensors
  if (num_additional_tensors_ > 0) {
    input_tensors_.erase(input_tensors_.end() - num_additional_tensors_,
                         input_tensors_.end());
    tensor_device_ptrs_.erase(
        tensor_device_ptrs_.end() - num_additional_tensors_,
        tensor_device_ptrs_.end());
  }

  // Add the new additional tensors
  num_additional_tensors_ = tensors.size();
  for (const auto &t : tensors) {
    input_tensors_.push_back(t);
  }

  // Copy additional tensor data to GPU and append pointers
  auto additional_ptrs =
      gpu_mgr_.primary().copy_tensor_data(tensors, true);
  tensor_device_ptrs_.insert(tensor_device_ptrs_.end(),
                             additional_ptrs.begin(), additional_ptrs.end());

  // Rebuild network description with additional tensors included
  build_network_description();
}

// ---- update_additional_tensors ----

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::update_additional_tensors(
    const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors) {
  // Update the data for additional tensors without changing the network
  // topology. Used between sampling branches — the measurement projection
  // tensors change values but keep the same modes and extents.
  hipSetDevice(gpu_mgr_.primary().device_id());

  size_t base = num_base_tensors_;
  for (size_t i = 0; i < tensors.size() && (base + i) < tensor_device_ptrs_.size(); i++) {
    check_hip(
        hipMemcpyAsync(
            tensor_device_ptrs_[base + i], tensors[i]->tensor().data(),
            tensors[i]->tensor().size() * sizeof(std::complex<data_t>),
            hipMemcpyHostToDevice, gpu_mgr_.primary().stream()),
        "hipMemcpyAsync(update_additional)", gpu_mgr_.primary().device_id());
  }
  hipStreamSynchronize(gpu_mgr_.primary().stream());
}

// ---- set_output ----

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::set_output(
    std::vector<int32_t> &modes, std::vector<int64_t> &extents) {
  modes_out_ = modes;
  extents_out_ = extents;

  out_size_ = 1;
  for (auto e : extents_out_)
    out_size_ *= e;

  gpu_mgr_.primary().allocate_output(out_size_);
}

// ---- setup_contraction ----

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::setup_contraction(
    bool use_autotune) {
#ifdef AER_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs_);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank_);
#endif

  // Update the network description with output info
  network_desc_.output_modes = modes_out_;
  network_desc_.output_extents = extents_out_;

  // Check for VQE path reuse: if the topology hasn't changed,
  // skip the path optimizer entirely.
  if (prev_valid_ && plan_valid_ && topology_matches_previous()) {
    if (tn_verbose() && myrank_ == 0) {
      fprintf(stderr, "[AER_TN] reusing previous contraction path "
                      "(topology unchanged)\n");
    }
  } else {
    // --- Two-pass memory planning ---
    // Pass 1: query actual free memory after tensor data is on GPU
    gpu_mgr_.primary().refresh_free_memory();
    size_t free_bytes = gpu_mgr_.primary().free_memory();

    // Pass 2: estimate hipTensor workspace for worst-case intermediate
    int64_t max_possible_elements = 1;
    for (auto e : extents_out_)
      max_possible_elements *= e;
    uint64_t workspace_estimate =
        gpu_mgr_.query_workspace_for_size(max_possible_elements);

    // Real budget for cotengra = free memory - workspace
    uint64_t memory_budget = free_bytes;
    if (workspace_estimate < free_bytes)
      memory_budget = free_bytes - workspace_estimate;

    if (tn_verbose() && myrank_ == 0) {
      fprintf(stderr,
              "[AER_TN] memory planning: %.1f MB free, %.1f MB workspace "
              "estimate, %.1f MB budget for path optimizer\n",
              free_bytes / (1024.0 * 1024.0),
              workspace_estimate / (1024.0 * 1024.0),
              memory_budget / (1024.0 * 1024.0));
    }

    // Create and run the path optimizer
    auto optimizer = create_optimizer();
    uint64_t seed = 42; // base seed; MPI wrapper adds rank offset
    plan_ = optimizer->find_path(network_desc_, memory_budget, seed);
    plan_valid_ = true;

    // Corrective second pass: check if actual workspace exceeds estimate
    // (only needed if cotengra returned a plan with a larger intermediate
    // than our worst-case estimate was based on)
    if (plan_.peak_intermediate_elements > max_possible_elements) {
      uint64_t actual_workspace =
          gpu_mgr_.query_workspace_for_size(plan_.peak_intermediate_elements);
      if (plan_.peak_intermediate_elements *
                  (int64_t)(2 * sizeof(data_t)) +
              (int64_t)actual_workspace >
          (int64_t)free_bytes) {
        // Re-plan with corrected budget
        uint64_t corrected = free_bytes - actual_workspace;
        plan_ = optimizer->find_path(network_desc_, corrected, seed);
        if (tn_verbose() && myrank_ == 0) {
          fprintf(stderr,
                  "[AER_TN] corrective re-plan: budget reduced to %.1f MB\n",
                  corrected / (1024.0 * 1024.0));
        }
      }
    }

    cache_topology();
  }

  // Distribute slices across MPI ranks
  slice_begin_ = myrank_ * plan_.num_slices / nprocs_;
  slice_end_ = (myrank_ + 1) * plan_.num_slices / nprocs_;

  // Distribute across GPUs within this rank
  num_devices_used_ = 1;
  if (gpu_mgr_.num_devices() > 1 &&
      (slice_end_ - slice_begin_) > gpu_mgr_.num_devices()) {
    num_devices_used_ = gpu_mgr_.num_devices();
    // Copy tensor data to all GPUs
    for (size_t i = 1; i < gpu_mgr_.num_devices(); i++) {
      gpu_mgr_.device(i).copy_tensor_data_from(gpu_mgr_.primary());
      gpu_mgr_.device(i).allocate_output(out_size_);
    }
  }

  if (tn_verbose() && myrank_ == 0) {
    fprintf(stderr,
            "[AER_TN] setup complete: %zu steps, %lu slices, "
            "%.2e FLOPs, %d GPU(s), %d MPI rank(s)\n",
            plan_.steps.size(), (unsigned long)plan_.num_slices,
            plan_.total_flops, num_devices_used_, nprocs_);
  }
}

// ---- contract ----

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::contract(
    std::vector<std::complex<data_t>> &out) {
  contract_all();
  gpu_mgr_.primary().get_output(out);
}

// ---- contract_and_trace ----

template <typename data_t>
double TensorNetContractor_HipTensor<data_t>::contract_and_trace(
    uint_t num_qubits) {
  contract_all();

  double ret = 0.0;
  for (int idev = 0; idev < num_devices_used_; idev++) {
    ret += gpu_mgr_.device(idev).trace_output(num_qubits);
  }

#ifdef AER_MPI
  if (nprocs_ > 1) {
    double sum = ret;
    MPI_Allreduce(&sum, &ret, 1, MPI_DOUBLE_PRECISION, MPI_SUM,
                  MPI_COMM_WORLD);
  }
#endif

  return ret;
}

// ---- contract_and_sample_measure ----

template <typename data_t>
double TensorNetContractor_HipTensor<data_t>::contract_and_sample_measure(
    reg_t &samples, std::vector<double> &rnds, uint_t num_qubits) {
  contract_all();

  // Accumulate across GPUs onto primary
  accumulate_across_gpus();

#ifdef AER_MPI
  accumulate_across_mpi();
#endif

  // Sample on primary GPU
  return sample_measure_on_primary(samples, rnds, num_qubits);
}

// ---- allocate/deallocate sampling buffers ----

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::allocate_sampling_buffers(
    uint_t size) {
  gpu_mgr_.primary().allocate_sampling_buffers(size);
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::deallocate_sampling_buffers() {
  gpu_mgr_.primary().deallocate_sampling_buffers();
}

//=============================================================================
// Private implementation
//=============================================================================

// ---- Build NetworkDescription from current tensors ----

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::build_network_description() {
  network_desc_.tensors.clear();
  for (const auto &t : input_tensors_) {
    TensorSpec spec;
    spec.modes = t->modes();
    spec.extents.resize(t->modes().size());
    for (size_t i = 0; i < t->modes().size(); i++) {
      spec.extents[i] = t->extents()[i];
    }
    network_desc_.tensors.push_back(spec);
  }
}

// ---- VQE topology comparison ----

template <typename data_t>
bool TensorNetContractor_HipTensor<data_t>::topology_matches_previous() const {
  if (prev_modes_.size() != input_tensors_.size())
    return false;
  for (size_t i = 0; i < input_tensors_.size(); i++) {
    if (input_tensors_[i]->modes() != prev_modes_[i])
      return false;
    auto &ext = input_tensors_[i]->extents();
    if (ext.size() != prev_extents_[i].size())
      return false;
    for (size_t j = 0; j < ext.size(); j++) {
      if (ext[j] != prev_extents_[i][j])
        return false;
    }
  }
  return true;
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::cache_topology() {
  prev_modes_.clear();
  prev_extents_.clear();
  for (const auto &t : input_tensors_) {
    prev_modes_.push_back(t->modes());
    prev_extents_.push_back(t->extents());
  }
  prev_valid_ = true;
}

// ---- Create the path optimizer ----

template <typename data_t>
std::unique_ptr<PathOptimizer>
TensorNetContractor_HipTensor<data_t>::create_optimizer() {
  std::unique_ptr<PathOptimizer> inner;

  // Try cotengra first (production path)
#ifdef AER_HIPTENSOR
  try {
    // Size of one complex element in bytes
    size_t elem_bytes = 2 * sizeof(data_t);
    inner = std::make_unique<CotengPathOptimizer>(
        "combo",  // minimize
        128,      // max_repeats
        60.0,     // max_time
        "hyper",  // preset (HyperOptimizer with kahypar)
        elem_bytes);
  } catch (...) {
    // cotengra import failed — fall back to greedy
    if (tn_verbose()) {
      fprintf(stderr,
              "[AER_TN] cotengra not available, using greedy fallback\n");
    }
    inner = std::make_unique<GreedyPathOptimizer>(32);
  }
#else
  inner = std::make_unique<GreedyPathOptimizer>(32);
#endif

  // Wrap in MPI-parallel optimizer if MPI is available
#ifdef AER_MPI
  if (nprocs_ > 1) {
    inner = std::make_unique<MPIParallelPathOptimizer>(std::move(inner));
  }
#endif

  return inner;
}

// ---- Execute full contraction (all slices, all GPUs) ----

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::contract_all() {
  // For each GPU, process its assigned slices
  for (int idev = 0; idev < num_devices_used_; idev++) {
    uint_t dev_slice_begin =
        slice_begin_ +
        (slice_end_ - slice_begin_) * idev / num_devices_used_;
    uint_t dev_slice_end =
        slice_begin_ +
        (slice_end_ - slice_begin_) * (idev + 1) / num_devices_used_;

    hipSetDevice(gpu_mgr_.device(idev).device_id());

    // Zero the output buffer for accumulation
    if (dev_slice_end - dev_slice_begin > 1 || idev > 0) {
      auto &out_buf = gpu_mgr_.device(idev).output_buffer();
      thrust::fill(thrust_gpu::par.on(gpu_mgr_.device(idev).stream()),
                   out_buf.begin(), out_buf.begin() + out_size_,
                   thrust::complex<data_t>(0.0, 0.0));
    }

    for (uint_t s = dev_slice_begin; s < dev_slice_end; s++) {
      contract_single_slice(s);

      // Accumulate this slice's result into the output buffer.
      // For now, we do a simple element-wise addition.
      // The result of contract_single_slice is in the pool's
      // final intermediate; we need to add it to the output buffer.
      // TODO: This is a placeholder — the full implementation needs
      // the memory pool wired up to track the final intermediate's
      // location and add it to the output buffer.
    }
  }
}

// ---- Contract a single slice ----

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::contract_single_slice(
    uint_t slice_index) {
  // For now, this is the execution skeleton. The full implementation
  // walks the contraction path step by step:
  //
  // 1. If there are sliced modes: project input tensors for this slice
  //    value combination (modify the device pointers to point at
  //    sub-tensors)
  //
  // 2. For each step in plan_.steps:
  //    a. Look up the input tensors (original or intermediate)
  //    b. Look up (or create) the cached hipTensor plan for this
  //       contraction's signature
  //    c. Call gpu_mgr_.primary().execute_contraction() with the
  //       cached plan and pre-computed pointers
  //    d. The result goes to the intermediate's offset in the pool
  //
  // 3. The final step's output is the contraction result for this slice
  //
  // The memory pool, plan cache, and slice projection are all set up
  // during setup_contraction. This function does zero allocation.

  if (plan_.steps.empty())
    return;

  // Walk the contraction path
  // TODO: Full implementation with memory pool integration
  // This requires wiring up:
  // - The pool offsets for each intermediate tensor
  // - The input tensor device pointers (original or projected)
  // - The cached hipTensor plans from the plan cache
  // - The workspace pointer from the pool
  //
  // Each of these pieces exists in gpu_resource_manager.hpp.
  // The wiring connects them in the order specified by plan_.steps.
}

// ---- Slice projection ----

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::project_slice(
    uint_t slice_index, std::vector<void *> &projected_ptrs) {
  // For each sliced mode in plan_.sliced:
  // Compute which value this slice_index assigns to each sliced mode.
  //
  // slice_index is decomposed into per-mode values:
  //   for a network with modes m0 (extent e0) and m1 (extent e1) sliced,
  //   slice_index = v0 * e1 + v1
  //   so v1 = slice_index % e1, v0 = slice_index / e1
  //
  // For each input tensor that contains a sliced mode:
  //   Extract the sub-tensor for the current slice values.
  //   This is a strided copy on the GPU.
  //
  // For input tensors without sliced modes:
  //   Use the original device pointer unchanged.

  projected_ptrs = tensor_device_ptrs_; // start with originals

  // Decompose slice_index into per-mode values
  std::vector<int64_t> slice_values(plan_.sliced.size());
  uint_t remaining = slice_index;
  for (int i = static_cast<int>(plan_.sliced.size()) - 1; i >= 0; i--) {
    slice_values[i] = remaining % plan_.sliced[i].extent;
    remaining /= plan_.sliced[i].extent;
  }

  // TODO: For each input tensor with sliced modes, compute the
  // sub-tensor device pointer based on the slice values and the
  // tensor's memory layout (strides). For base-2 qubits, each
  // sliced mode halves the tensor.
}

// ---- Accumulate across GPUs (tree reduction) ----

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::accumulate_across_gpus() {
  if (num_devices_used_ <= 1)
    return;

  // Tree reduction: pair GPUs and accumulate
  for (int stride = 1; stride < num_devices_used_; stride *= 2) {
    for (int i = 0; i + stride < num_devices_used_; i += stride * 2) {
      int dst_dev = i;
      int src_dev = i + stride;

      hipSetDevice(gpu_mgr_.device(dst_dev).device_id());

      auto &dst_buf = gpu_mgr_.device(dst_dev).output_buffer();
      auto &src_buf = gpu_mgr_.device(src_dev).output_buffer();

      if (gpu_mgr_.device(dst_dev).has_peer_access(
              gpu_mgr_.device(src_dev).device_id())) {
        // Direct P2P accumulation
        if (hipDeviceEnablePeerAccess(
                gpu_mgr_.device(src_dev).device_id(), 0) != hipSuccess)
          hipGetLastError();

        thrust::plus<thrust::complex<data_t>> op;
        thrust::transform(
            thrust_gpu::par.on(gpu_mgr_.device(dst_dev).stream()),
            dst_buf.begin(), dst_buf.begin() + out_size_, src_buf.begin(),
            dst_buf.begin(), op);
      } else {
        // Stage through host
        std::vector<std::complex<data_t>> host_buf(out_size_);
        hipSetDevice(gpu_mgr_.device(src_dev).device_id());
        hipMemcpy(host_buf.data(),
                  thrust::raw_pointer_cast(src_buf.data()),
                  out_size_ * sizeof(std::complex<data_t>),
                  hipMemcpyDeviceToHost);

        hipSetDevice(gpu_mgr_.device(dst_dev).device_id());
        // Copy to a temp device buffer and add
        thrust::device_vector<thrust::complex<data_t>> tmp(out_size_);
        hipMemcpy(thrust::raw_pointer_cast(tmp.data()), host_buf.data(),
                  out_size_ * sizeof(std::complex<data_t>),
                  hipMemcpyHostToDevice);
        thrust::plus<thrust::complex<data_t>> op;
        thrust::transform(
            thrust_gpu::par.on(gpu_mgr_.device(dst_dev).stream()),
            dst_buf.begin(), dst_buf.begin() + out_size_, tmp.begin(),
            dst_buf.begin(), op);
        hipStreamSynchronize(gpu_mgr_.device(dst_dev).stream());
      }
    }
  }
}

// ---- Accumulate across MPI ranks ----

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::accumulate_across_mpi() {
#ifdef AER_MPI
  if (nprocs_ <= 1)
    return;

  std::vector<std::complex<data_t>> out(out_size_);
  gpu_mgr_.primary().get_output(out);

  std::vector<std::complex<data_t>> tmp(out_size_);
  MPI_Datatype mpi_type =
      (sizeof(data_t) == 8) ? MPI_DOUBLE_PRECISION : MPI_FLOAT;
  MPI_Allreduce(out.data(), tmp.data(), out_size_ * 2, mpi_type, MPI_SUM,
                MPI_COMM_WORLD);

  // Copy result back to GPU
  hipSetDevice(gpu_mgr_.primary().device_id());
  hipMemcpyAsync(
      thrust::raw_pointer_cast(gpu_mgr_.primary().output_buffer().data()),
      tmp.data(), out_size_ * sizeof(std::complex<data_t>),
      hipMemcpyHostToDevice, gpu_mgr_.primary().stream());
  hipStreamSynchronize(gpu_mgr_.primary().stream());
#endif
}

// ---- Sample measurement on primary GPU ----

template <typename data_t>
double TensorNetContractor_HipTensor<data_t>::sample_measure_on_primary(
    reg_t &samples, std::vector<double> &rnds, uint_t num_qubits) {
  if (samples.size() < rnds.size())
    samples.resize(rnds.size());

  auto &dev = gpu_mgr_.primary();
  hipSetDevice(dev.device_id());

  uint_t stride = (1ULL << num_qubits) + 1;
  auto *base = (thrust::complex<data_t> *)thrust::raw_pointer_cast(
      dev.output_buffer().data());

  QV::Chunk::strided_range<thrust::complex<data_t> *> iter(
      base, base + out_size_, stride);

  // Inclusive scan to build CDF
  thrust::inclusive_scan(thrust_gpu::par.on(dev.stream()), iter.begin(),
                         iter.end(), iter.begin(),
                         thrust::plus<thrust::complex<data_t>>());

  // Allocate temporary sampling buffers if needed
  thrust::device_vector<double> dev_rnds(rnds.size());
  thrust::device_vector<uint_t> dev_samples(rnds.size());

  // Copy random numbers to device
  hipMemcpyAsync(thrust::raw_pointer_cast(dev_rnds.data()), rnds.data(),
                 rnds.size() * sizeof(double), hipMemcpyHostToDevice,
                 dev.stream());

  // Binary search to find sample positions
  thrust::lower_bound(thrust_gpu::par.on(dev.stream()), iter.begin(),
                      iter.end(), dev_rnds.begin(), dev_rnds.end(),
                      dev_samples.begin(),
                      QV::Chunk::complex_less<data_t>());

  // Update random numbers for hierarchical sampling
  auto ci = thrust::counting_iterator<uint_t>(0);
  thrust::for_each_n(
      thrust_gpu::par.on(dev.stream()), ci, rnds.size(),
      sampling_update_rnd_func_hip<data_t>(
          base, stride,
          (uint_t *)thrust::raw_pointer_cast(dev_samples.data()),
          (double *)thrust::raw_pointer_cast(dev_rnds.data())));

  // Copy results back to host
  hipMemcpyAsync(samples.data(),
                 thrust::raw_pointer_cast(dev_samples.data()),
                 rnds.size() * sizeof(uint_t), hipMemcpyDeviceToHost,
                 dev.stream());
  hipMemcpyAsync(rnds.data(), thrust::raw_pointer_cast(dev_rnds.data()),
                 rnds.size() * sizeof(double), hipMemcpyDeviceToHost,
                 dev.stream());

  hipStreamSynchronize(dev.stream());

  // Return the total trace (last element of the inclusive scan)
  thrust::complex<data_t> trace_val =
      dev.output_buffer()[out_size_ - 1];
  return trace_val.real();
}

//------------------------------------------------------------------------------
} // namespace TensorNetwork
} // end namespace AER
//------------------------------------------------------------------------------

#endif // AER_THRUST_ROCM

#endif // _tensor_net_contractor_hiptensor_hpp_
