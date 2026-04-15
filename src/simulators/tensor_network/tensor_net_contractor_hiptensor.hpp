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

#ifndef _tensor_net_contractor_hiptensor_hpp_
#define _tensor_net_contractor_hiptensor_hpp_

#ifdef AER_THRUST_ROCM

#include <complex>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.hpp>

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

namespace thrust_gpu = thrust::hip;

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
// Helper: compute result modes/extents of a pairwise contraction
//=============================================================================

inline void compute_contraction_result(
    const std::vector<int32_t> &modes_a,
    const std::vector<int64_t> &extents_a,
    const std::vector<int32_t> &modes_b,
    const std::vector<int64_t> &extents_b,
    std::vector<int32_t> &modes_c,
    std::vector<int64_t> &extents_c) {
  modes_c.clear();
  extents_c.clear();

  // Find shared modes (contracted)
  std::set<int32_t> shared;
  for (size_t i = 0; i < modes_a.size(); i++)
    for (size_t j = 0; j < modes_b.size(); j++)
      if (modes_a[i] == modes_b[j])
        shared.insert(modes_a[i]);

  // Result = unshared modes from A then unshared from B
  for (size_t i = 0; i < modes_a.size(); i++) {
    if (shared.find(modes_a[i]) == shared.end()) {
      modes_c.push_back(modes_a[i]);
      extents_c.push_back(extents_a[i]);
    }
  }
  for (size_t i = 0; i < modes_b.size(); i++) {
    if (shared.find(modes_b[i]) == shared.end()) {
      modes_c.push_back(modes_b[i]);
      extents_c.push_back(extents_b[i]);
    }
  }
}

//=============================================================================
// Sampling GPU kernel
//=============================================================================

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
// TensorNetContractor_HipTensor
//=============================================================================

template <typename data_t = double>
class TensorNetContractor_HipTensor : public TensorNetContractor<data_t> {
  GPUResourceManager<data_t> gpu_mgr_;
  int num_devices_used_;

  NetworkDescription network_desc_;
  std::vector<std::shared_ptr<Tensor<data_t>>> input_tensors_;
  bool add_sp_tensors_;
  uint_t num_base_tensors_;
  uint_t num_additional_tensors_;
  std::vector<void *> tensor_device_ptrs_;

  std::vector<int32_t> modes_out_;
  std::vector<int64_t> extents_out_;
  uint_t out_size_;

  ContractionPlan plan_;
  bool plan_valid_;

  uint_t slice_begin_;
  uint_t slice_end_;
  int nprocs_;
  int myrank_;
  reg_t target_gpus_;

  // Sliced tensor descriptions: input specs with sliced modes removed
  std::vector<TensorSpec> sliced_input_specs_;
  // All tensor specs during path walk (inputs + intermediates)
  std::vector<TensorSpec> all_specs_;
  // Sliced mode set for fast lookup
  std::set<int32_t> sliced_mode_set_;
  // Whether pool is set up for current topology
  bool pool_ready_;

  // VQE path reuse
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
  void allocate_sampling_buffers(uint_t size = AER_TENSOR_NET_MAX_SAMPLING) override;
  void deallocate_sampling_buffers(void) override;
  void set_target_gpus(reg_t &t) override { target_gpus_ = t; }

private:
  void build_network_description();
  void build_sliced_specs();
  void setup_pool_and_cache(int device_idx);
  bool topology_matches_previous() const;
  void cache_topology();
  void contract_single_slice(uint_t slice_index, int device_idx);
  void project_slice(uint_t slice_index, std::vector<void *> &projected_ptrs);
  void accumulate_across_gpus();
  void accumulate_across_mpi();
  void contract_all();
  double sample_measure_on_primary(reg_t &samples, std::vector<double> &rnds,
                                   uint_t num_qubits);
  std::unique_ptr<PathOptimizer> create_optimizer();
};

//=============================================================================
// Implementation
//=============================================================================

template <typename data_t>
TensorNetContractor_HipTensor<data_t>::TensorNetContractor_HipTensor()
    : num_devices_used_(1), add_sp_tensors_(true), num_base_tensors_(0),
      num_additional_tensors_(0), out_size_(0), plan_valid_(false),
      slice_begin_(0), slice_end_(0), nprocs_(1), myrank_(0),
      pool_ready_(false), prev_valid_(false) {}

template <typename data_t>
TensorNetContractor_HipTensor<data_t>::~TensorNetContractor_HipTensor() {}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::set_network(
    const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors,
    bool add_sp_tensors) {
  input_tensors_.clear();
  add_sp_tensors_ = add_sp_tensors;
  num_additional_tensors_ = 0;

  for (size_t i = 0; i < tensors.size(); i++)
    if (add_sp_tensors || !tensors[i]->sp_tensor())
      input_tensors_.push_back(tensors[i]);
  num_base_tensors_ = input_tensors_.size();

  if (gpu_mgr_.num_devices() == 0) {
    std::vector<uint64_t> targets(target_gpus_.begin(), target_gpus_.end());
    gpu_mgr_.discover(targets);
  }

  tensor_device_ptrs_ = gpu_mgr_.primary().copy_tensor_data(input_tensors_, true);
  build_network_description();
  pool_ready_ = false;

  if (tn_verbose())
    fprintf(stderr, "[AER_TN] set_network: %zu tensors on %zu GPU(s)\n",
            input_tensors_.size(), gpu_mgr_.num_devices());
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::allocate_additional_tensors(uint_t) {}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::set_additional_tensors(
    const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors) {
  if (num_additional_tensors_ > 0) {
    input_tensors_.erase(input_tensors_.end() - num_additional_tensors_,
                         input_tensors_.end());
    tensor_device_ptrs_.erase(tensor_device_ptrs_.end() - num_additional_tensors_,
                              tensor_device_ptrs_.end());
  }
  num_additional_tensors_ = tensors.size();
  for (size_t i = 0; i < tensors.size(); i++)
    input_tensors_.push_back(tensors[i]);

  auto additional_ptrs = gpu_mgr_.primary().copy_tensor_data(tensors, true);
  tensor_device_ptrs_.insert(tensor_device_ptrs_.end(),
                             additional_ptrs.begin(), additional_ptrs.end());
  build_network_description();
  pool_ready_ = false;
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::update_additional_tensors(
    const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors) {
  hipSetDevice(gpu_mgr_.primary().device_id());
  size_t base = num_base_tensors_;
  for (size_t i = 0; i < tensors.size() && (base + i) < tensor_device_ptrs_.size(); i++) {
    check_hip(hipMemcpyAsync(
        tensor_device_ptrs_[base + i], tensors[i]->tensor().data(),
        tensors[i]->tensor().size() * sizeof(std::complex<data_t>),
        hipMemcpyHostToDevice, gpu_mgr_.primary().stream()),
        "hipMemcpyAsync(update)", gpu_mgr_.primary().device_id());
  }
  hipStreamSynchronize(gpu_mgr_.primary().stream());
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::set_output(
    std::vector<int32_t> &modes, std::vector<int64_t> &extents) {
  modes_out_ = modes;
  extents_out_ = extents;
  out_size_ = 1;
  for (size_t i = 0; i < extents_out_.size(); i++)
    out_size_ *= extents_out_[i];
  gpu_mgr_.primary().allocate_output(out_size_);
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::setup_contraction(bool) {
#ifdef AER_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs_);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank_);
#endif

  network_desc_.output_modes = modes_out_;
  network_desc_.output_extents = extents_out_;

  if (prev_valid_ && plan_valid_ && topology_matches_previous()) {
    if (tn_verbose() && myrank_ == 0)
      fprintf(stderr, "[AER_TN] reusing previous contraction path\n");
  } else {
    gpu_mgr_.primary().refresh_free_memory();
    size_t free_bytes = gpu_mgr_.primary().free_memory();

    int64_t max_possible_elements = 1;
    for (size_t i = 0; i < extents_out_.size(); i++)
      max_possible_elements *= extents_out_[i];
    uint64_t workspace_estimate =
        gpu_mgr_.query_workspace_for_size(max_possible_elements);

    uint64_t memory_budget = free_bytes;
    if (workspace_estimate < free_bytes)
      memory_budget = free_bytes - workspace_estimate;

    if (tn_verbose() && myrank_ == 0)
      fprintf(stderr, "[AER_TN] memory: %.1f MB free, %.1f MB ws, %.1f MB budget\n",
              free_bytes / (1024.0 * 1024.0),
              workspace_estimate / (1024.0 * 1024.0),
              memory_budget / (1024.0 * 1024.0));

    auto optimizer = create_optimizer();
    plan_ = optimizer->find_path(network_desc_, memory_budget, 42);
    plan_valid_ = true;
    pool_ready_ = false;
    cache_topology();
  }

  // Build sliced tensor specs (input specs with sliced modes removed)
  build_sliced_specs();

  // Set up memory pool and plan cache if not already done
  if (!pool_ready_) {
    setup_pool_and_cache(0);
    pool_ready_ = true;
  }

  slice_begin_ = myrank_ * plan_.num_slices / nprocs_;
  slice_end_ = (myrank_ + 1) * plan_.num_slices / nprocs_;

  num_devices_used_ = 1;
  if (gpu_mgr_.num_devices() > 1 &&
      (slice_end_ - slice_begin_) > gpu_mgr_.num_devices()) {
    num_devices_used_ = gpu_mgr_.num_devices();
    for (size_t i = 1; i < gpu_mgr_.num_devices(); i++) {
      gpu_mgr_.device(i).copy_tensor_data_from(gpu_mgr_.primary());
      gpu_mgr_.device(i).allocate_output(out_size_);
      if (!gpu_mgr_.device(i).pool().is_allocated())
        setup_pool_and_cache(i);
    }
  }

  if (tn_verbose() && myrank_ == 0)
    fprintf(stderr, "[AER_TN] setup: %zu steps, %lu slices, %.2e FLOPs, %d GPUs\n",
            plan_.steps.size(), (unsigned long)plan_.num_slices,
            plan_.total_flops, num_devices_used_);
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::build_sliced_specs() {
  sliced_mode_set_.clear();
  for (size_t i = 0; i < plan_.sliced.size(); i++)
    sliced_mode_set_.insert(plan_.sliced[i].mode);

  size_t num_inputs = network_desc_.tensors.size();
  sliced_input_specs_.resize(num_inputs);

  for (size_t t = 0; t < num_inputs; t++) {
    sliced_input_specs_[t].modes.clear();
    sliced_input_specs_[t].extents.clear();
    for (size_t m = 0; m < network_desc_.tensors[t].modes.size(); m++) {
      if (sliced_mode_set_.find(network_desc_.tensors[t].modes[m]) == sliced_mode_set_.end()) {
        sliced_input_specs_[t].modes.push_back(network_desc_.tensors[t].modes[m]);
        sliced_input_specs_[t].extents.push_back(network_desc_.tensors[t].extents[m]);
      }
    }
  }
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::setup_pool_and_cache(int device_idx) {
  GPUDevice<data_t> &dev = gpu_mgr_.device(device_idx);
  hipSetDevice(dev.device_id());

  size_t num_inputs = sliced_input_specs_.size();
  size_t num_steps = plan_.steps.size();
  size_t num_total = num_inputs + num_steps;

  // Walk the path to compute all intermediate shapes
  all_specs_.resize(num_total);
  for (size_t i = 0; i < num_inputs; i++)
    all_specs_[i] = sliced_input_specs_[i];

  for (size_t step = 0; step < num_steps; step++) {
    uint64_t left = plan_.steps[step].left;
    uint64_t right = plan_.steps[step].right;
    size_t result_idx = num_inputs + step;

    compute_contraction_result(
        all_specs_[left].modes, all_specs_[left].extents,
        all_specs_[right].modes, all_specs_[right].extents,
        all_specs_[result_idx].modes, all_specs_[result_idx].extents);
  }

  // Compute tensor lifetimes: last step each tensor is used as input
  std::vector<int> last_used(num_total, -1);
  for (size_t step = 0; step < num_steps; step++) {
    last_used[plan_.steps[step].left] = static_cast<int>(step);
    last_used[plan_.steps[step].right] = static_cast<int>(step);
  }
  // Final result lives until the end
  if (num_steps > 0)
    last_used[num_inputs + num_steps - 1] = static_cast<int>(num_steps - 1);

  // Pre-populate plan cache for all unique contraction signatures
  for (size_t step = 0; step < num_steps; step++) {
    uint64_t left = plan_.steps[step].left;
    uint64_t right = plan_.steps[step].right;
    size_t result_idx = num_inputs + step;

    auto sig = build_signature(
        all_specs_[left].modes, all_specs_[left].extents,
        all_specs_[right].modes, all_specs_[right].extents,
        all_specs_[result_idx].modes, all_specs_[result_idx].extents);

    dev.plan_cache().get_or_create(
        sig,
        all_specs_[left].modes, all_specs_[left].extents,
        all_specs_[right].modes, all_specs_[right].extents,
        all_specs_[result_idx].modes, all_specs_[result_idx].extents);
  }

  // Build memory pool layout for intermediates
  size_t element_bytes = 2 * sizeof(data_t); // sizeof(complex<data_t>)
  std::vector<std::tuple<size_t, int, int, int>> intermediates;

  for (size_t step = 0; step < num_steps; step++) {
    size_t result_idx = num_inputs + step;
    int64_t num_elements = all_specs_[result_idx].num_elements();
    size_t bytes = static_cast<size_t>(num_elements) * element_bytes;
    int birth = static_cast<int>(step);
    int death = last_used[result_idx];
    if (death < 0) death = static_cast<int>(num_steps - 1);

    intermediates.push_back(
        std::make_tuple(bytes, birth, death, static_cast<int>(result_idx)));
  }

  uint64_t max_ws = dev.plan_cache().max_workspace_bytes();

  dev.pool().plan_layout(intermediates, max_ws, static_cast<int>(num_steps));
  dev.pool().allocate(dev.device_id());

  if (tn_verbose())
    fprintf(stderr, "[AER_TN] pool: %zu bytes, %zu plans cached (device %d)\n",
            dev.pool().total_size(), dev.plan_cache().size(), dev.device_id());
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::contract(
    std::vector<std::complex<data_t>> &out) {
  contract_all();
  gpu_mgr_.primary().get_output(out);
}

template <typename data_t>
double TensorNetContractor_HipTensor<data_t>::contract_and_trace(uint_t num_qubits) {
  contract_all();
  double ret = 0.0;
  for (int idev = 0; idev < num_devices_used_; idev++)
    ret += gpu_mgr_.device(idev).trace_output(num_qubits);
#ifdef AER_MPI
  if (nprocs_ > 1) {
    double sum = ret;
    MPI_Allreduce(&sum, &ret, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  return ret;
}

template <typename data_t>
double TensorNetContractor_HipTensor<data_t>::contract_and_sample_measure(
    reg_t &samples, std::vector<double> &rnds, uint_t num_qubits) {
  contract_all();
  accumulate_across_gpus();
#ifdef AER_MPI
  accumulate_across_mpi();
#endif
  return sample_measure_on_primary(samples, rnds, num_qubits);
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::allocate_sampling_buffers(uint_t size) {
  gpu_mgr_.primary().allocate_sampling_buffers(size);
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::deallocate_sampling_buffers() {
  gpu_mgr_.primary().deallocate_sampling_buffers();
}

//=============================================================================
// Private implementation
//=============================================================================

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::build_network_description() {
  network_desc_.tensors.clear();
  for (size_t i = 0; i < input_tensors_.size(); i++) {
    TensorSpec spec;
    spec.modes = input_tensors_[i]->modes();
    spec.extents.resize(input_tensors_[i]->modes().size());
    for (size_t j = 0; j < input_tensors_[i]->modes().size(); j++)
      spec.extents[j] = input_tensors_[i]->extents()[j];
    network_desc_.tensors.push_back(spec);
  }
}

template <typename data_t>
bool TensorNetContractor_HipTensor<data_t>::topology_matches_previous() const {
  if (prev_modes_.size() != input_tensors_.size()) return false;
  for (size_t i = 0; i < input_tensors_.size(); i++) {
    if (input_tensors_[i]->modes() != prev_modes_[i]) return false;
    const std::vector<int64_t> &ext = input_tensors_[i]->extents();
    if (ext.size() != prev_extents_[i].size()) return false;
    for (size_t j = 0; j < ext.size(); j++)
      if (ext[j] != prev_extents_[i][j]) return false;
  }
  return true;
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::cache_topology() {
  prev_modes_.clear();
  prev_extents_.clear();
  for (size_t i = 0; i < input_tensors_.size(); i++) {
    prev_modes_.push_back(input_tensors_[i]->modes());
    prev_extents_.push_back(input_tensors_[i]->extents());
  }
  prev_valid_ = true;
}

template <typename data_t>
std::unique_ptr<PathOptimizer>
TensorNetContractor_HipTensor<data_t>::create_optimizer() {
  std::unique_ptr<PathOptimizer> inner;
#ifdef AER_HIPTENSOR
  try {
    size_t elem_bytes = 2 * sizeof(data_t);
    inner = std::unique_ptr<PathOptimizer>(new CotengPathOptimizer(
        "combo", 128, 60.0, "hyper", elem_bytes));
  } catch (...) {
    if (tn_verbose())
      fprintf(stderr, "[AER_TN] cotengra unavailable, using greedy\n");
    inner = std::unique_ptr<PathOptimizer>(new GreedyPathOptimizer(32));
  }
#else
  inner = std::unique_ptr<PathOptimizer>(new GreedyPathOptimizer(32));
#endif
#ifdef AER_MPI
  if (nprocs_ > 1)
    inner = std::unique_ptr<PathOptimizer>(
        new MPIParallelPathOptimizer(std::move(inner)));
#endif
  return inner;
}

// ---- Core contraction loop ----

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::contract_all() {
  for (int idev = 0; idev < num_devices_used_; idev++) {
    uint_t dev_slice_begin =
        slice_begin_ + (slice_end_ - slice_begin_) * idev / num_devices_used_;
    uint_t dev_slice_end =
        slice_begin_ + (slice_end_ - slice_begin_) * (idev + 1) / num_devices_used_;

    hipSetDevice(gpu_mgr_.device(idev).device_id());

    // Zero output buffer for accumulation
    auto &out_buf = gpu_mgr_.device(idev).output_buffer();
    thrust::fill(thrust_gpu::par.on(gpu_mgr_.device(idev).stream()),
                 out_buf.begin(), out_buf.begin() + out_size_,
                 thrust::complex<data_t>(0.0, 0.0));

    for (uint_t s = dev_slice_begin; s < dev_slice_end; s++) {
      contract_single_slice(s, idev);
    }
  }
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::contract_single_slice(
    uint_t slice_index, int device_idx) {
  if (plan_.steps.empty()) return;

  GPUDevice<data_t> &dev = gpu_mgr_.device(device_idx);
  hipSetDevice(dev.device_id());

  size_t num_inputs = sliced_input_specs_.size();
  size_t num_steps = plan_.steps.size();

  // Get projected pointers for this slice
  std::vector<void *> projected_ptrs;
  project_slice(slice_index, projected_ptrs);

  // Build pointer array: inputs from projected_ptrs, intermediates from pool
  std::vector<void *> all_ptrs(num_inputs + num_steps, nullptr);
  for (size_t i = 0; i < num_inputs; i++)
    all_ptrs[i] = projected_ptrs[i];

  void *workspace = dev.pool().get_workspace_ptr();
  uint64_t ws_size = dev.pool().get_workspace_size();

  // Walk the contraction path
  for (size_t step = 0; step < num_steps; step++) {
    uint64_t left = plan_.steps[step].left;
    uint64_t right = plan_.steps[step].right;
    size_t result_idx = num_inputs + step;

    // Get or retrieve cached plan
    auto sig = build_signature(
        all_specs_[left].modes, all_specs_[left].extents,
        all_specs_[right].modes, all_specs_[right].extents,
        all_specs_[result_idx].modes, all_specs_[result_idx].extents);

    const CachedPlan<data_t> &plan = dev.plan_cache().get_or_create(
        sig,
        all_specs_[left].modes, all_specs_[left].extents,
        all_specs_[right].modes, all_specs_[right].extents,
        all_specs_[result_idx].modes, all_specs_[result_idx].extents);

    // Get result pointer from memory pool
    all_ptrs[result_idx] = dev.pool().get_tensor_ptr(static_cast<int>(result_idx));

    // Execute contraction: D = alpha * A * B + beta * C (beta=0)
    dev.execute_contraction(plan,
        all_ptrs[left], all_ptrs[right], all_ptrs[result_idx],
        workspace, ws_size, false);
  }

  // Accumulate final result into output buffer
  size_t final_idx = num_inputs + num_steps - 1;
  void *final_ptr = all_ptrs[final_idx];
  auto *out_raw = thrust::raw_pointer_cast(dev.output_buffer().data());

  // output_buffer += final_result (buffer was zeroed in contract_all)
  thrust::transform(
      thrust_gpu::par.on(dev.stream()),
      dev.output_buffer().begin(),
      dev.output_buffer().begin() + out_size_,
      static_cast<thrust::complex<data_t> *>(final_ptr),
      dev.output_buffer().begin(),
      thrust::plus<thrust::complex<data_t>>());
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::project_slice(
    uint_t slice_index, std::vector<void *> &projected_ptrs) {
  projected_ptrs = tensor_device_ptrs_;

  if (plan_.sliced.empty()) return;

  // Decompose slice_index into per-mode values
  std::vector<int64_t> slice_values(plan_.sliced.size());
  uint_t remaining = slice_index;
  for (int i = static_cast<int>(plan_.sliced.size()) - 1; i >= 0; i--) {
    slice_values[i] = remaining % plan_.sliced[i].extent;
    remaining /= plan_.sliced[i].extent;
  }

  // Build mode -> slice_value map
  std::map<int32_t, int64_t> slice_map;
  for (size_t i = 0; i < plan_.sliced.size(); i++)
    slice_map[plan_.sliced[i].mode] = slice_values[i];

  size_t element_bytes = 2 * sizeof(data_t);

  // For each input tensor, compute pointer offset from sliced modes
  for (size_t t = 0; t < network_desc_.tensors.size(); t++) {
    const TensorSpec &spec = network_desc_.tensors[t];
    int64_t offset_elements = 0;

    // Compute row-major strides for the ORIGINAL (unsliced) tensor
    std::vector<int64_t> strides(spec.modes.size(), 1);
    for (int i = static_cast<int>(spec.modes.size()) - 2; i >= 0; i--)
      strides[i] = strides[i + 1] * spec.extents[i + 1];

    // Add offset for each sliced mode present in this tensor
    for (size_t m = 0; m < spec.modes.size(); m++) {
      auto it = slice_map.find(spec.modes[m]);
      if (it != slice_map.end())
        offset_elements += it->second * strides[m];
    }

    if (offset_elements > 0)
      projected_ptrs[t] = static_cast<char *>(projected_ptrs[t]) +
                          offset_elements * element_bytes;
  }
}

// ---- Multi-GPU accumulation ----

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::accumulate_across_gpus() {
  if (num_devices_used_ <= 1) return;

  for (int stride = 1; stride < num_devices_used_; stride *= 2) {
    for (int i = 0; i + stride < num_devices_used_; i += stride * 2) {
      int dst_dev = i, src_dev = i + stride;
      hipSetDevice(gpu_mgr_.device(dst_dev).device_id());
      auto &dst_buf = gpu_mgr_.device(dst_dev).output_buffer();
      auto &src_buf = gpu_mgr_.device(src_dev).output_buffer();

      if (gpu_mgr_.device(dst_dev).has_peer_access(
              gpu_mgr_.device(src_dev).device_id())) {
        if (hipDeviceEnablePeerAccess(gpu_mgr_.device(src_dev).device_id(), 0)
            != hipSuccess)
          hipGetLastError();
        thrust::transform(
            thrust_gpu::par.on(gpu_mgr_.device(dst_dev).stream()),
            dst_buf.begin(), dst_buf.begin() + out_size_, src_buf.begin(),
            dst_buf.begin(), thrust::plus<thrust::complex<data_t>>());
      } else {
        std::vector<std::complex<data_t>> host_buf(out_size_);
        hipSetDevice(gpu_mgr_.device(src_dev).device_id());
        hipMemcpy(host_buf.data(), thrust::raw_pointer_cast(src_buf.data()),
                  out_size_ * sizeof(std::complex<data_t>), hipMemcpyDeviceToHost);
        hipSetDevice(gpu_mgr_.device(dst_dev).device_id());
        thrust::device_vector<thrust::complex<data_t>> tmp(out_size_);
        hipMemcpy(thrust::raw_pointer_cast(tmp.data()), host_buf.data(),
                  out_size_ * sizeof(std::complex<data_t>), hipMemcpyHostToDevice);
        thrust::transform(
            thrust_gpu::par.on(gpu_mgr_.device(dst_dev).stream()),
            dst_buf.begin(), dst_buf.begin() + out_size_, tmp.begin(),
            dst_buf.begin(), thrust::plus<thrust::complex<data_t>>());
        hipStreamSynchronize(gpu_mgr_.device(dst_dev).stream());
      }
    }
  }
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::accumulate_across_mpi() {
#ifdef AER_MPI
  if (nprocs_ <= 1) return;
  std::vector<std::complex<data_t>> out(out_size_);
  gpu_mgr_.primary().get_output(out);
  std::vector<std::complex<data_t>> tmp(out_size_);
  MPI_Datatype mpi_type =
      (sizeof(data_t) == 8) ? MPI_DOUBLE_PRECISION : MPI_FLOAT;
  MPI_Allreduce(out.data(), tmp.data(), out_size_ * 2, mpi_type, MPI_SUM,
                MPI_COMM_WORLD);
  hipSetDevice(gpu_mgr_.primary().device_id());
  hipMemcpyAsync(
      thrust::raw_pointer_cast(gpu_mgr_.primary().output_buffer().data()),
      tmp.data(), out_size_ * sizeof(std::complex<data_t>),
      hipMemcpyHostToDevice, gpu_mgr_.primary().stream());
  hipStreamSynchronize(gpu_mgr_.primary().stream());
#endif
}

// ---- Sampling ----

template <typename data_t>
double TensorNetContractor_HipTensor<data_t>::sample_measure_on_primary(
    reg_t &samples, std::vector<double> &rnds, uint_t num_qubits) {
  if (samples.size() < rnds.size()) samples.resize(rnds.size());

  GPUDevice<data_t> &dev = gpu_mgr_.primary();
  hipSetDevice(dev.device_id());

  uint_t stride = (1ULL << num_qubits) + 1;
  thrust::complex<data_t> *base =
      (thrust::complex<data_t> *)thrust::raw_pointer_cast(dev.output_buffer().data());

  QV::Chunk::strided_range<thrust::complex<data_t> *> iter(
      base, base + out_size_, stride);

  thrust::inclusive_scan(thrust_gpu::par.on(dev.stream()), iter.begin(),
                         iter.end(), iter.begin(),
                         thrust::plus<thrust::complex<data_t>>());

  thrust::device_vector<double> dev_rnds(rnds.size());
  thrust::device_vector<uint_t> dev_samples(rnds.size());

  hipMemcpyAsync(thrust::raw_pointer_cast(dev_rnds.data()), rnds.data(),
                 rnds.size() * sizeof(double), hipMemcpyHostToDevice, dev.stream());

  thrust::lower_bound(thrust_gpu::par.on(dev.stream()), iter.begin(),
                      iter.end(), dev_rnds.begin(), dev_rnds.end(),
                      dev_samples.begin(), QV::Chunk::complex_less<data_t>());

  auto ci = thrust::counting_iterator<uint_t>(0);
  thrust::for_each_n(
      thrust_gpu::par.on(dev.stream()), ci, rnds.size(),
      sampling_update_rnd_func_hip<data_t>(
          base, stride,
          (uint_t *)thrust::raw_pointer_cast(dev_samples.data()),
          (double *)thrust::raw_pointer_cast(dev_rnds.data())));

  hipMemcpyAsync(samples.data(), thrust::raw_pointer_cast(dev_samples.data()),
                 rnds.size() * sizeof(uint_t), hipMemcpyDeviceToHost, dev.stream());
  hipMemcpyAsync(rnds.data(), thrust::raw_pointer_cast(dev_rnds.data()),
                 rnds.size() * sizeof(double), hipMemcpyDeviceToHost, dev.stream());

  hipStreamSynchronize(dev.stream());

  thrust::complex<data_t> trace_val = dev.output_buffer()[out_size_ - 1];
  return trace_val.real();
}

} // namespace TensorNetwork
} // namespace AER

#endif // AER_THRUST_ROCM
#endif // _tensor_net_contractor_hiptensor_hpp_
