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

#include <complex>
#include <cstdlib>
#include <memory>
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
// 1. Sampling GPU kernel
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
// 2. TensorNetContractor_HipTensor
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
  void build_network_description();
  bool topology_matches_previous() const;
  void cache_topology();
  void contract_single_slice(uint_t slice_index);
  void project_slice(uint_t slice_index,
                     std::vector<void *> &projected_ptrs);
  void accumulate_across_gpus();
  void accumulate_across_mpi();
  void contract_all();
  double sample_measure_on_primary(reg_t &samples, std::vector<double> &rnds,
                                   uint_t num_qubits);
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

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::set_network(
    const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors,
    bool add_sp_tensors) {
  input_tensors_.clear();
  add_sp_tensors_ = add_sp_tensors;
  num_additional_tensors_ = 0;

  for (size_t i = 0; i < tensors.size(); i++) {
    if (add_sp_tensors || !tensors[i]->sp_tensor()) {
      input_tensors_.push_back(tensors[i]);
    }
  }
  num_base_tensors_ = input_tensors_.size();

  if (gpu_mgr_.num_devices() == 0) {
    std::vector<uint64_t> targets(target_gpus_.begin(), target_gpus_.end());
    gpu_mgr_.discover(targets);
  }

  tensor_device_ptrs_ =
      gpu_mgr_.primary().copy_tensor_data(input_tensors_, true);

  build_network_description();

  if (tn_verbose()) {
    fprintf(stderr,
            "[AER_TN] set_network: %zu tensors on %zu GPU(s)\n",
            input_tensors_.size(), gpu_mgr_.num_devices());
  }
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::allocate_additional_tensors(
    uint_t size) {
  // Pre-allocation handled in set_additional_tensors
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::set_additional_tensors(
    const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors) {
  if (num_additional_tensors_ > 0) {
    input_tensors_.erase(input_tensors_.end() - num_additional_tensors_,
                         input_tensors_.end());
    tensor_device_ptrs_.erase(
        tensor_device_ptrs_.end() - num_additional_tensors_,
        tensor_device_ptrs_.end());
  }

  num_additional_tensors_ = tensors.size();
  for (size_t i = 0; i < tensors.size(); i++) {
    input_tensors_.push_back(tensors[i]);
  }

  auto additional_ptrs =
      gpu_mgr_.primary().copy_tensor_data(tensors, true);
  tensor_device_ptrs_.insert(tensor_device_ptrs_.end(),
                             additional_ptrs.begin(), additional_ptrs.end());

  build_network_description();
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::update_additional_tensors(
    const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors) {
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
void TensorNetContractor_HipTensor<data_t>::setup_contraction(
    bool use_autotune) {
#ifdef AER_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs_);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank_);
#endif

  network_desc_.output_modes = modes_out_;
  network_desc_.output_extents = extents_out_;

  if (prev_valid_ && plan_valid_ && topology_matches_previous()) {
    if (tn_verbose() && myrank_ == 0) {
      fprintf(stderr, "[AER_TN] reusing previous contraction path "
                      "(topology unchanged)\n");
    }
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

    if (tn_verbose() && myrank_ == 0) {
      fprintf(stderr,
              "[AER_TN] memory planning: %.1f MB free, %.1f MB workspace "
              "estimate, %.1f MB budget for path optimizer\n",
              free_bytes / (1024.0 * 1024.0),
              workspace_estimate / (1024.0 * 1024.0),
              memory_budget / (1024.0 * 1024.0));
    }

    auto optimizer = create_optimizer();
    uint64_t seed = 42;
    plan_ = optimizer->find_path(network_desc_, memory_budget, seed);
    plan_valid_ = true;

    cache_topology();
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

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::contract(
    std::vector<std::complex<data_t>> &out) {
  contract_all();
  gpu_mgr_.primary().get_output(out);
}

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

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::build_network_description() {
  network_desc_.tensors.clear();
  for (size_t i = 0; i < input_tensors_.size(); i++) {
    TensorSpec spec;
    spec.modes = input_tensors_[i]->modes();
    spec.extents.resize(input_tensors_[i]->modes().size());
    for (size_t j = 0; j < input_tensors_[i]->modes().size(); j++) {
      spec.extents[j] = input_tensors_[i]->extents()[j];
    }
    network_desc_.tensors.push_back(spec);
  }
}

template <typename data_t>
bool TensorNetContractor_HipTensor<data_t>::topology_matches_previous() const {
  if (prev_modes_.size() != input_tensors_.size())
    return false;
  for (size_t i = 0; i < input_tensors_.size(); i++) {
    if (input_tensors_[i]->modes() != prev_modes_[i])
      return false;
    const std::vector<int64_t> &ext = input_tensors_[i]->extents();
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
    if (tn_verbose()) {
      fprintf(stderr,
              "[AER_TN] cotengra not available, using greedy fallback\n");
    }
    inner = std::unique_ptr<PathOptimizer>(new GreedyPathOptimizer(32));
  }
#else
  inner = std::unique_ptr<PathOptimizer>(new GreedyPathOptimizer(32));
#endif

#ifdef AER_MPI
  if (nprocs_ > 1) {
    inner = std::unique_ptr<PathOptimizer>(
        new MPIParallelPathOptimizer(std::move(inner)));
  }
#endif

  return inner;
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::contract_all() {
  for (int idev = 0; idev < num_devices_used_; idev++) {
    uint_t dev_slice_begin =
        slice_begin_ +
        (slice_end_ - slice_begin_) * idev / num_devices_used_;
    uint_t dev_slice_end =
        slice_begin_ +
        (slice_end_ - slice_begin_) * (idev + 1) / num_devices_used_;

    hipSetDevice(gpu_mgr_.device(idev).device_id());

    if (dev_slice_end - dev_slice_begin > 1 || idev > 0) {
      auto &out_buf = gpu_mgr_.device(idev).output_buffer();
      thrust::fill(thrust_gpu::par.on(gpu_mgr_.device(idev).stream()),
                   out_buf.begin(), out_buf.begin() + out_size_,
                   thrust::complex<data_t>(0.0, 0.0));
    }

    for (uint_t s = dev_slice_begin; s < dev_slice_end; s++) {
      contract_single_slice(s);
      // TODO: accumulate slice result into output buffer
    }
  }
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::contract_single_slice(
    uint_t slice_index) {
  // TODO: walk contraction path step by step using memory pool + plan cache
  if (plan_.steps.empty())
    return;
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::project_slice(
    uint_t slice_index, std::vector<void *> &projected_ptrs) {
  projected_ptrs = tensor_device_ptrs_;

  std::vector<int64_t> slice_values(plan_.sliced.size());
  uint_t remaining = slice_index;
  for (int i = static_cast<int>(plan_.sliced.size()) - 1; i >= 0; i--) {
    slice_values[i] = remaining % plan_.sliced[i].extent;
    remaining /= plan_.sliced[i].extent;
  }

  // TODO: compute sub-tensor device pointers from slice values
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::accumulate_across_gpus() {
  if (num_devices_used_ <= 1)
    return;

  for (int stride = 1; stride < num_devices_used_; stride *= 2) {
    for (int i = 0; i + stride < num_devices_used_; i += stride * 2) {
      int dst_dev = i;
      int src_dev = i + stride;

      hipSetDevice(gpu_mgr_.device(dst_dev).device_id());

      auto &dst_buf = gpu_mgr_.device(dst_dev).output_buffer();
      auto &src_buf = gpu_mgr_.device(src_dev).output_buffer();

      if (gpu_mgr_.device(dst_dev).has_peer_access(
              gpu_mgr_.device(src_dev).device_id())) {
        if (hipDeviceEnablePeerAccess(
                gpu_mgr_.device(src_dev).device_id(), 0) != hipSuccess)
          hipGetLastError();

        thrust::plus<thrust::complex<data_t>> op;
        thrust::transform(
            thrust_gpu::par.on(gpu_mgr_.device(dst_dev).stream()),
            dst_buf.begin(), dst_buf.begin() + out_size_, src_buf.begin(),
            dst_buf.begin(), op);
      } else {
        std::vector<std::complex<data_t>> host_buf(out_size_);
        hipSetDevice(gpu_mgr_.device(src_dev).device_id());
        hipMemcpy(host_buf.data(),
                  thrust::raw_pointer_cast(src_buf.data()),
                  out_size_ * sizeof(std::complex<data_t>),
                  hipMemcpyDeviceToHost);

        hipSetDevice(gpu_mgr_.device(dst_dev).device_id());
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

  hipSetDevice(gpu_mgr_.primary().device_id());
  hipMemcpyAsync(
      thrust::raw_pointer_cast(gpu_mgr_.primary().output_buffer().data()),
      tmp.data(), out_size_ * sizeof(std::complex<data_t>),
      hipMemcpyHostToDevice, gpu_mgr_.primary().stream());
  hipStreamSynchronize(gpu_mgr_.primary().stream());
#endif
}

template <typename data_t>
double TensorNetContractor_HipTensor<data_t>::sample_measure_on_primary(
    reg_t &samples, std::vector<double> &rnds, uint_t num_qubits) {
  if (samples.size() < rnds.size())
    samples.resize(rnds.size());

  GPUDevice<data_t> &dev = gpu_mgr_.primary();
  hipSetDevice(dev.device_id());

  uint_t stride = (1ULL << num_qubits) + 1;
  thrust::complex<data_t> *base =
      (thrust::complex<data_t> *)thrust::raw_pointer_cast(
          dev.output_buffer().data());

  QV::Chunk::strided_range<thrust::complex<data_t> *> iter(
      base, base + out_size_, stride);

  thrust::inclusive_scan(thrust_gpu::par.on(dev.stream()), iter.begin(),
                         iter.end(), iter.begin(),
                         thrust::plus<thrust::complex<data_t>>());

  thrust::device_vector<double> dev_rnds(rnds.size());
  thrust::device_vector<uint_t> dev_samples(rnds.size());

  hipMemcpyAsync(thrust::raw_pointer_cast(dev_rnds.data()), rnds.data(),
                 rnds.size() * sizeof(double), hipMemcpyHostToDevice,
                 dev.stream());

  thrust::lower_bound(thrust_gpu::par.on(dev.stream()), iter.begin(),
                      iter.end(), dev_rnds.begin(), dev_rnds.end(),
                      dev_samples.begin(),
                      QV::Chunk::complex_less<data_t>());

  auto ci = thrust::counting_iterator<uint_t>(0);
  thrust::for_each_n(
      thrust_gpu::par.on(dev.stream()), ci, rnds.size(),
      sampling_update_rnd_func_hip<data_t>(
          base, stride,
          (uint_t *)thrust::raw_pointer_cast(dev_samples.data()),
          (double *)thrust::raw_pointer_cast(dev_rnds.data())));

  hipMemcpyAsync(samples.data(),
                 thrust::raw_pointer_cast(dev_samples.data()),
                 rnds.size() * sizeof(uint_t), hipMemcpyDeviceToHost,
                 dev.stream());
  hipMemcpyAsync(rnds.data(), thrust::raw_pointer_cast(dev_rnds.data()),
                 rnds.size() * sizeof(double), hipMemcpyDeviceToHost,
                 dev.stream());

  hipStreamSynchronize(dev.stream());

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
