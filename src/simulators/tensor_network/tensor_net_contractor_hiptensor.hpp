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

#include <atomic>
#include <climits>
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

static bool tn_debug() {
  static bool checked = false;
  static bool enabled = false;
  if (!checked) {
    const char *val = std::getenv("AER_TN_DEBUG");
    enabled = (val != nullptr && std::string(val) == "1");
    checked = true;
  }
  return enabled;
}

// hipTensor requires every contraction descriptor to have at least one free
// mode on A (the M group), at least one free mode on B (the N group), and at
// least one shared mode (the K group). Pairwise contractions that the
// cotengra-generated path produces for small quantum circuits routinely
// violate this (e.g., matrix-times-vector with no free B modes). When they
// do, hipTensor's plan validation accepts the descriptor, but no registered
// CK kernel matches the shape — hiptensorContraction returns SUCCESS and
// leaves the output buffer untouched. Padding each offending operand with
// a dummy mode of extent 1 makes the descriptor grammar-compliant without
// touching device memory (extent-1 dims are layout-free).
//
// This kill-switch lets you bisect: unset it to reproduce the pre-fix
// behavior. Default is enabled.
static bool tn_mnk_padding_enabled() {
  static bool checked = false;
  static bool enabled = true;
  if (!checked) {
    const char *val = std::getenv("AER_TN_DISABLE_MNK_PADDING");
    enabled = !(val != nullptr && std::string(val) == "1");
    checked = true;
  }
  return enabled;
}

//=============================================================================
// Diagnostic helpers
//=============================================================================

inline std::string modes_to_str(const std::vector<int32_t> &modes) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < modes.size(); i++) {
    if (i > 0) ss << ",";
    ss << modes[i];
  }
  ss << "]";
  return ss.str();
}

inline std::string extents_to_str(const std::vector<int64_t> &extents) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < extents.size(); i++) {
    if (i > 0) ss << ",";
    ss << extents[i];
  }
  ss << "]";
  return ss.str();
}

template <typename data_t>
void dump_device_tensor(const char *label, void *dev_ptr, int64_t num_elements,
                        int max_print = 8) {
  if (!tn_debug()) return;
  std::vector<std::complex<data_t>> host(num_elements);
  hipMemcpy(host.data(), dev_ptr,
            num_elements * sizeof(std::complex<data_t>),
            hipMemcpyDeviceToHost);
  fprintf(stderr, "[AER_TN_DEBUG] %s [%ld elements]:", label, (long)num_elements);
  int n_print = (int)std::min<int64_t>(num_elements, max_print);
  for (int i = 0; i < n_print; i++) {
    fprintf(stderr, " (%.3f,%.3fi)", host[i].real(), host[i].imag());
  }
  if (num_elements > max_print) fprintf(stderr, " ...");
  fprintf(stderr, "\n");
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

  std::set<int32_t> shared;
  for (size_t i = 0; i < modes_a.size(); i++)
    for (size_t j = 0; j < modes_b.size(); j++)
      if (modes_a[i] == modes_b[j])
        shared.insert(modes_a[i]);

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
// Helper: pad a pairwise contraction so hipTensor's M/N/K grammar holds.
//
// hipTensor / Composable Kernel enforce that every contraction has at least
// one free mode on A (M), one free mode on B (N), and one shared mode (K).
// If a natural pairwise contraction violates this (common for rank-1 tensors
// in quantum circuits: statevector legs, small intermediates), we add a
// dummy mode of extent 1 to the appropriate operand plus the output. Extent-
// 1 modes don't change memory layout, so we don't need to touch any device
// data — only the descriptors passed to hiptensorInitTensorDescriptor.
//
// All three tensors' modes/extents are modified in place. If padding was
// applied, the returned pair of ints tells the caller how many dummy modes
// were appended to C so it can strip them back off if the caller tracks
// semantic (unpadded) modes separately. Returned as (num_dummies_on_C).
//
// Dummy modes are drawn from a distinct range (below DUMMY_MODE_BASE) so
// they can never collide with legitimate mode IDs produced by TensorNet's
// mode_index_ counter (which starts at 0 and increments upward; the int32
// "large negative" values we see in logs come from hashing, but they stay
// above INT32_MIN + 2^30). Using INT32_MIN + small-positive-counter for
// dummies keeps them well clear.
//=============================================================================

inline int
pad_contraction_mnk(std::vector<int32_t> &modes_a, std::vector<int64_t> &extents_a,
                    std::vector<int32_t> &modes_b, std::vector<int64_t> &extents_b,
                    std::vector<int32_t> &modes_c, std::vector<int64_t> &extents_c) {
  // Allocator for dummy mode IDs. Process-lifetime counter — collisions across
  // concurrent TensorNetContractor instances are impossible because each plan
  // consumes its own dummies and discards them after hiptensorContraction is
  // called. Value is far from any mode_index_-generated real mode.
  static std::atomic<int32_t> dummy_counter{INT32_MIN + 1};

  std::set<int32_t> sa(modes_a.begin(), modes_a.end());
  std::set<int32_t> sb(modes_b.begin(), modes_b.end());

  bool has_M = false, has_N = false, has_K = false;
  for (int32_t m : modes_a) {
    if (sb.count(m)) has_K = true; else has_M = true;
  }
  for (int32_t m : modes_b) {
    if (!sa.count(m)) { has_N = true; break; }
  }

  int dummies_on_c = 0;
  if (!has_K) {
    // No shared mode: add a shared extent-1 dummy to both A and B.
    // The dummy does NOT go on C (K modes are contracted away).
    int32_t d = dummy_counter.fetch_add(1);
    modes_a.push_back(d); extents_a.push_back(1);
    modes_b.push_back(d); extents_b.push_back(1);
  }
  if (!has_M) {
    // A has no free mode. Add a dummy free mode to A (and to C so it's not
    // contracted). Extent 1 so total element count is unchanged.
    int32_t d = dummy_counter.fetch_add(1);
    modes_a.push_back(d); extents_a.push_back(1);
    modes_c.push_back(d); extents_c.push_back(1);
    dummies_on_c++;
  }
  if (!has_N) {
    int32_t d = dummy_counter.fetch_add(1);
    modes_b.push_back(d); extents_b.push_back(1);
    modes_c.push_back(d); extents_c.push_back(1);
    dummies_on_c++;
  }
  return dummies_on_c;
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

  std::vector<TensorSpec> sliced_input_specs_;
  std::vector<TensorSpec> all_specs_;
  std::set<int32_t> sliced_mode_set_;
  bool pool_ready_;

  // Per-step descriptor shape used when submitting contractions to hipTensor.
  // Each PlanSpec holds the (possibly MNK-padded) modes/extents for A, B, C.
  // Input tensors may appear in multiple steps with different padding, which
  // is why this is per-step, not per-tensor. Rebuilt on every setup_pool_and_cache.
  struct PlanSpec {
    std::vector<int32_t> modes_a, modes_b, modes_c;
    std::vector<int64_t> extents_a, extents_b, extents_c;
  };
  std::vector<PlanSpec> step_plan_specs_;

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

  // Drop tensors whose modes never connect to any other tensor or to the
  // output. Aer's initialize() creates "super qubit" tensors with
  // sp_tensor_=false (set() not set_conj()), so our sp-filter misses them.
  // When add_sp_tensors=false, these remain as orphans with dangling modes.
  // cuTensorNet prunes them internally; we must do it explicitly.
  // Returns vector<bool> mask: keep[i]=true means keep tensor i.
  std::vector<bool> compute_orphan_mask(
      const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors,
      const std::set<int32_t> &output_modes_set) const;
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
std::vector<bool>
TensorNetContractor_HipTensor<data_t>::compute_orphan_mask(
    const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors,
    const std::set<int32_t> &output_modes_set) const {
  // Count mode occurrences across all tensors + output (count output once).
  std::map<int32_t, int> mode_count;
  for (const auto &t : tensors) {
    for (int32_t m : t->modes()) mode_count[m]++;
  }
  for (int32_t m : output_modes_set) mode_count[m]++;

  // A tensor is "connected" if at least one of its modes has count >= 2
  // (i.e., shared with another tensor or with the output). An isolated tensor
  // whose ALL modes have count==1 is an orphan.
  std::vector<bool> keep(tensors.size(), true);
  for (size_t i = 0; i < tensors.size(); i++) {
    bool has_shared_mode = false;
    for (int32_t m : tensors[i]->modes()) {
      if (mode_count[m] >= 2) { has_shared_mode = true; break; }
    }
    if (!has_shared_mode) keep[i] = false;
  }
  return keep;
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::set_network(
    const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors,
    bool add_sp_tensors) {
  input_tensors_.clear();
  add_sp_tensors_ = add_sp_tensors;
  num_additional_tensors_ = 0;

  // Stage 1: Apply Aer's sp_tensor filter
  std::vector<std::shared_ptr<Tensor<data_t>>> sp_filtered;
  sp_filtered.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); i++)
    if (add_sp_tensors || !tensors[i]->sp_tensor())
      sp_filtered.push_back(tensors[i]);

  // Stage 2: Drop orphan tensors.
  // Orphans arise because Aer's initialize() creates "super qubit" tensors
  // with sp_tensor_=false (via set() not set_conj()). When add_sp_tensors=false
  // requested (save_statevector path), the sp filter lets these through, but
  // their modes are disconnected from the rest of the network. cuTensorNet
  // prunes them internally; we must do it explicitly or contracting them as
  // disconnected scalar factors zeroes out the output.
  std::set<int32_t> output_modes_set(modes_out_.begin(), modes_out_.end());
  std::vector<bool> keep = compute_orphan_mask(sp_filtered, output_modes_set);

  size_t dropped = 0;
  for (size_t i = 0; i < sp_filtered.size(); i++) {
    if (keep[i]) {
      input_tensors_.push_back(sp_filtered[i]);
    } else {
      dropped++;
      if (tn_debug()) {
        fprintf(stderr,
                "[AER_TN_DEBUG] dropping orphan tensor (sp=%d) modes=%s\n",
                (int)sp_filtered[i]->sp_tensor(),
                modes_to_str(sp_filtered[i]->modes()).c_str());
        // Dump the orphan's actual values — if these are [1,0] they're benign
        // basis projectors we can safely drop. Anything else means we're
        // dropping meaningful data and this filter is too aggressive.
        const auto &data = sp_filtered[i]->tensor();
        size_t n = data.size();
        size_t n_print = std::min<size_t>(n, 16);
        fprintf(stderr,
                "[AER_TN_DEBUG]   orphan values [%zu total]:", n);
        for (size_t k = 0; k < n_print; k++) {
          fprintf(stderr, " (%.3f,%.3fi)",
                  data[k].real(), data[k].imag());
        }
        if (n > n_print) fprintf(stderr, " ...");
        fprintf(stderr, "\n");
      }
    }
  }
  num_base_tensors_ = input_tensors_.size();

  if (gpu_mgr_.num_devices() == 0) {
    std::vector<uint64_t> targets(target_gpus_.begin(), target_gpus_.end());
    gpu_mgr_.discover(targets);
  }

  tensor_device_ptrs_ = gpu_mgr_.primary().copy_tensor_data(input_tensors_, true);
  build_network_description();
  pool_ready_ = false;

  if (tn_verbose()) {
    fprintf(stderr,
            "[AER_TN] set_network: %zu tensors on %zu GPU(s) "
            "(add_sp=%d, input_count=%zu, sp_filtered=%zu, orphans_dropped=%zu)\n",
            input_tensors_.size(), gpu_mgr_.num_devices(),
            (int)add_sp_tensors, tensors.size(), sp_filtered.size(), dropped);
  }

  if (tn_debug()) {
    fprintf(stderr, "[AER_TN_DEBUG] input tensors (after filter):\n");
    for (size_t i = 0; i < input_tensors_.size(); i++) {
      auto modes = input_tensors_[i]->modes();
      auto extents = input_tensors_[i]->extents();
      fprintf(stderr, "[AER_TN_DEBUG]   T%zu: sp=%d modes=%s extents=%s size=%zu\n",
              i, (int)input_tensors_[i]->sp_tensor(),
              modes_to_str(modes).c_str(),
              extents_to_str(extents).c_str(),
              input_tensors_[i]->tensor().size());
      // Dump HOST-SIDE values (before hipMemcpy). If these match what we later
      // see in "input[i]" from the GPU, upload is fine. If they differ, we
      // have a copy bug. If the host values are already wrong for what the
      // gate should be, the bug is upstream in qiskit-aer's TN construction
      // or our tensor ingestion.
      const auto &data = input_tensors_[i]->tensor();
      size_t n = data.size();
      size_t n_print = std::min<size_t>(n, 32);
      fprintf(stderr, "[AER_TN_DEBUG]     host values:");
      for (size_t k = 0; k < n_print; k++) {
        fprintf(stderr, " (%.3f,%.3fi)", data[k].real(), data[k].imag());
      }
      if (n > n_print) fprintf(stderr, " ...");
      fprintf(stderr, "\n");
    }
  }
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

  if (tn_verbose())
    fprintf(stderr, "[AER_TN] set_additional_tensors: %zu additional (%zu total)\n",
            tensors.size(), input_tensors_.size());
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

  if (tn_verbose())
    fprintf(stderr, "[AER_TN] set_output: modes=%s extents=%s size=%zu\n",
            modes_to_str(modes_out_).c_str(),
            extents_to_str(extents_out_).c_str(),
            (size_t)out_size_);
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

  build_sliced_specs();

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

  all_specs_.resize(num_total);
  for (size_t i = 0; i < num_inputs; i++)
    all_specs_[i] = sliced_input_specs_[i];

  if (tn_debug()) {
    fprintf(stderr, "[AER_TN_DEBUG] setup_pool_and_cache: %zu inputs, %zu steps\n",
            num_inputs, num_steps);
    fprintf(stderr, "[AER_TN_DEBUG] output_modes=%s output_extents=%s\n",
            modes_to_str(modes_out_).c_str(),
            extents_to_str(extents_out_).c_str());
    for (size_t i = 0; i < num_inputs; i++) {
      fprintf(stderr, "[AER_TN_DEBUG]   input[%zu]: modes=%s extents=%s num_el=%ld\n",
              i, modes_to_str(all_specs_[i].modes).c_str(),
              extents_to_str(all_specs_[i].extents).c_str(),
              (long)all_specs_[i].num_elements());
    }
  }

  // Walk path to compute intermediate shapes.
  //
  // For each step we compute the natural (unpadded) output modes/extents.
  // Then, if the step's contraction shape violates hipTensor's M/N/K
  // grammar, we pad the inputs and the output with extent-1 dummy modes.
  // The padded C is stored in all_specs_[result_idx] so downstream steps
  // see the padded shape and stay consistent.
  //
  // We also record the per-step padded A/B/C modes in step_plan_specs_ so
  // contract_single_slice can look up the exact descriptor shape for each
  // hiptensorContraction call. Input tensors may appear in multiple steps
  // with different padding, so we can't pad them in sliced_input_specs_.
  step_plan_specs_.assign(num_steps, PlanSpec{});

  for (size_t step = 0; step < num_steps; step++) {
    uint64_t left = plan_.steps[step].left;
    uint64_t right = plan_.steps[step].right;
    size_t result_idx = num_inputs + step;

    // Natural (unpadded) A and B modes come from all_specs_ (which for inputs
    // reflects sliced_input_specs_, and for earlier intermediates reflects
    // their padded forms — that's fine, the padded form is just the valid
    // shape with extent-1 dummies which don't affect downstream semantics).
    std::vector<int32_t> modes_a = all_specs_[left].modes;
    std::vector<int64_t> extents_a = all_specs_[left].extents;
    std::vector<int32_t> modes_b = all_specs_[right].modes;
    std::vector<int64_t> extents_b = all_specs_[right].extents;

    std::vector<int32_t> modes_c;
    std::vector<int64_t> extents_c;
    compute_contraction_result(modes_a, extents_a, modes_b, extents_b,
                               modes_c, extents_c);

    int dummies_on_c = 0;
    if (tn_mnk_padding_enabled()) {
      dummies_on_c = pad_contraction_mnk(modes_a, extents_a,
                                         modes_b, extents_b,
                                         modes_c, extents_c);
    }

    if (step == num_steps - 1) {
      // Final step: the C we produce must match modes_out_ (plus any
      // padding dummies, which are extent-1 and so don't affect element
      // count or the output buffer's memory layout).
      //
      // Compare sets without the dummy modes — dummies live in the
      // [INT32_MIN + 1, INT32_MIN + N] range.
      std::set<int32_t> natural_set;
      for (int32_t m : modes_c) {
        if (m > INT32_MIN + (1 << 20)) natural_set.insert(m);
      }
      std::set<int32_t> output_set(modes_out_.begin(), modes_out_.end());

      if (natural_set != output_set) {
        if (tn_debug()) {
          fprintf(stderr,
                  "[AER_TN_DEBUG] WARNING: final step modes %s don't match output %s\n",
                  modes_to_str(modes_c).c_str(),
                  modes_to_str(modes_out_).c_str());
        }
        // Fall back to the padded natural modes. Element count still
        // equals num_elements() of modes_out_ because dummies are extent 1.
        all_specs_[result_idx].modes = modes_c;
        all_specs_[result_idx].extents = extents_c;
      } else {
        // Use modes_out_ order for the real modes, keep any dummies
        // appended. This matches the memory layout the caller expects
        // (modes_out_ order) while preserving the padded descriptor.
        std::vector<int32_t> final_modes = modes_out_;
        std::vector<int64_t> final_extents = extents_out_;
        // Append any dummy modes from modes_c that aren't in modes_out_.
        for (size_t i = 0; i < modes_c.size(); i++) {
          if (natural_set.count(modes_c[i]) == 0) {
            final_modes.push_back(modes_c[i]);
            final_extents.push_back(extents_c[i]);
          }
        }
        // Rebuild modes_c/extents_c in the new order so the plan's C
        // descriptor matches the physical layout we want.
        modes_c = final_modes;
        extents_c = final_extents;
        all_specs_[result_idx].modes = final_modes;
        all_specs_[result_idx].extents = final_extents;
      }
    } else {
      all_specs_[result_idx].modes = modes_c;
      all_specs_[result_idx].extents = extents_c;
    }

    // Record the exact per-plan descriptor shape for contract_single_slice.
    step_plan_specs_[step].modes_a = modes_a;
    step_plan_specs_[step].extents_a = extents_a;
    step_plan_specs_[step].modes_b = modes_b;
    step_plan_specs_[step].extents_b = extents_b;
    step_plan_specs_[step].modes_c = modes_c;
    step_plan_specs_[step].extents_c = extents_c;

    if (tn_debug()) {
      fprintf(stderr,
              "[AER_TN_DEBUG]   step %zu: T%ld x T%ld -> T%zu   "
              "modes_l=%s modes_r=%s modes_c=%s num_el=%ld%s%s\n",
              step, (long)left, (long)right, result_idx,
              modes_to_str(modes_a).c_str(),
              modes_to_str(modes_b).c_str(),
              modes_to_str(modes_c).c_str(),
              (long)all_specs_[result_idx].num_elements(),
              (dummies_on_c > 0 || modes_a.size() > all_specs_[left].modes.size() ||
               modes_b.size() > all_specs_[right].modes.size()) ? " [padded]" : "",
              (step == num_steps - 1) ? " (FINAL)" : "");
    }
  }

  std::vector<int> last_used(num_total, -1);
  for (size_t step = 0; step < num_steps; step++) {
    last_used[plan_.steps[step].left] = static_cast<int>(step);
    last_used[plan_.steps[step].right] = static_cast<int>(step);
  }
  if (num_steps > 0)
    last_used[num_inputs + num_steps - 1] = static_cast<int>(num_steps - 1);

  for (size_t step = 0; step < num_steps; step++) {
    uint64_t left = plan_.steps[step].left;
    uint64_t right = plan_.steps[step].right;
    size_t result_idx = num_inputs + step;

    // Use the padded per-step descriptor shape, not the raw all_specs_
    // (which for inputs may differ from what this specific plan needs).
    const PlanSpec &ps = step_plan_specs_[step];

    auto sig = build_signature(
        ps.modes_a, ps.extents_a,
        ps.modes_b, ps.extents_b,
        ps.modes_c, ps.extents_c);

    dev.plan_cache().get_or_create(
        sig,
        ps.modes_a, ps.extents_a,
        ps.modes_b, ps.extents_b,
        ps.modes_c, ps.extents_c);
  }

  size_t element_bytes = 2 * sizeof(data_t);
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

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::contract_all() {
  for (int idev = 0; idev < num_devices_used_; idev++) {
    uint_t dev_slice_begin =
        slice_begin_ + (slice_end_ - slice_begin_) * idev / num_devices_used_;
    uint_t dev_slice_end =
        slice_begin_ + (slice_end_ - slice_begin_) * (idev + 1) / num_devices_used_;

    hipSetDevice(gpu_mgr_.device(idev).device_id());

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

  if (tn_debug())
    fprintf(stderr, "[AER_TN_DEBUG] contract_single_slice(slice=%lu, dev=%d)\n",
            (unsigned long)slice_index, device_idx);

  std::vector<void *> projected_ptrs;
  project_slice(slice_index, projected_ptrs);

  std::vector<void *> all_ptrs(num_inputs + num_steps, nullptr);
  for (size_t i = 0; i < num_inputs; i++)
    all_ptrs[i] = projected_ptrs[i];

  void *workspace = dev.pool().get_workspace_ptr();
  uint64_t ws_size = dev.pool().get_workspace_size();

  if (tn_debug()) {
    for (size_t i = 0; i < num_inputs; i++) {
      char label[64];
      snprintf(label, sizeof(label), "  input[%zu]", i);
      // Print up to 32 elements so 4-leg gates (e.g. CNOT's 16 values) are
      // fully visible — crucial for distinguishing identity from CNOT since
      // they share first 4 values [1, 0, 0, 0].
      dump_device_tensor<data_t>(label, all_ptrs[i],
                                  sliced_input_specs_[i].num_elements(), 32);
    }
  }

  for (size_t step = 0; step < num_steps; step++) {
    uint64_t left = plan_.steps[step].left;
    uint64_t right = plan_.steps[step].right;
    size_t result_idx = num_inputs + step;

    // Use the per-step padded descriptor shape we recorded in
    // setup_pool_and_cache. The plan cache is keyed on these same
    // shapes, so the lookup is O(1) after the first slice.
    const PlanSpec &ps = step_plan_specs_[step];

    auto sig = build_signature(
        ps.modes_a, ps.extents_a,
        ps.modes_b, ps.extents_b,
        ps.modes_c, ps.extents_c);

    const CachedPlan<data_t> &plan = dev.plan_cache().get_or_create(
        sig,
        ps.modes_a, ps.extents_a,
        ps.modes_b, ps.extents_b,
        ps.modes_c, ps.extents_c);

    all_ptrs[result_idx] = dev.pool().get_tensor_ptr(static_cast<int>(result_idx));

    // Defense in depth against two orthogonal failure modes:
    //   1. The pool memory for this intermediate is freshly allocated or
    //      aliased from a prior step, so its contents are undefined.
    //   2. hiptensorContraction's C pointer aliases its D pointer (we pass
    //      ptr_c for both with beta=0). Per the API spec C is only read
    //      when beta!=0, but CK bilinear kernels have historically done an
    //      RMW on the D tile regardless, and under -ffast-math (which the
    //      Aer build uses) reading uninitialized memory can produce NaNs
    //      that propagate via 0*NaN=NaN and get flushed to junk.
    // Zeroing D before the call kills both hazards at microsecond cost.
    {
      size_t c_bytes =
          static_cast<size_t>(all_specs_[result_idx].num_elements()) *
          2 * sizeof(data_t);
      check_hip(hipMemsetAsync(all_ptrs[result_idx], 0, c_bytes, dev.stream()),
                "hipMemsetAsync(C pre-zero)", dev.device_id());
    }

    if (tn_debug()) {
      fprintf(stderr,
              "[AER_TN_DEBUG]   step %zu exec: "
              "A_ptr=%p B_ptr=%p C_ptr=%p ws=%p ws_sz=%lu modes_c=%s\n",
              step, all_ptrs[left], all_ptrs[right], all_ptrs[result_idx],
              workspace, (unsigned long)ws_size,
              modes_to_str(ps.modes_c).c_str());
    }

    dev.execute_contraction(plan,
        all_ptrs[left], all_ptrs[right], all_ptrs[result_idx],
        workspace, ws_size, false);

    if (tn_debug()) {
      hipStreamSynchronize(dev.stream());
      char label[64];
      snprintf(label, sizeof(label), "  after step %zu (T%zu)", step, result_idx);
      dump_device_tensor<data_t>(label, all_ptrs[result_idx],
                                  all_specs_[result_idx].num_elements(), 32);
    }
  }

  size_t final_idx = num_inputs + num_steps - 1;
  void *final_ptr = all_ptrs[final_idx];
  int64_t final_elements = all_specs_[final_idx].num_elements();

  if (tn_debug())
    fprintf(stderr,
            "[AER_TN_DEBUG]   accumulate final_ptr=%p (%ld el) into out_buf (%lu el)\n",
            final_ptr, (long)final_elements, (unsigned long)out_size_);

  if (final_elements != (int64_t)out_size_) {
    fprintf(stderr,
            "[AER_TN] ERROR: final tensor size %ld != out_size_ %lu\n",
            (long)final_elements, (unsigned long)out_size_);
    return;
  }

  thrust::transform(
      thrust_gpu::par.on(dev.stream()),
      dev.output_buffer().begin(),
      dev.output_buffer().begin() + out_size_,
      static_cast<thrust::complex<data_t> *>(final_ptr),
      dev.output_buffer().begin(),
      thrust::plus<thrust::complex<data_t>>());

  if (tn_debug()) {
    hipStreamSynchronize(dev.stream());
    std::vector<std::complex<data_t>> host(out_size_);
    hipMemcpy(host.data(),
              thrust::raw_pointer_cast(dev.output_buffer().data()),
              out_size_ * sizeof(std::complex<data_t>),
              hipMemcpyDeviceToHost);
    fprintf(stderr, "[AER_TN_DEBUG]   output_buffer after accumulate:");
    for (size_t i = 0; i < std::min<size_t>(out_size_, 8); i++)
      fprintf(stderr, " (%.3f,%.3fi)", host[i].real(), host[i].imag());
    fprintf(stderr, "\n");
  }
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::project_slice(
    uint_t slice_index, std::vector<void *> &projected_ptrs) {
  projected_ptrs = tensor_device_ptrs_;

  if (plan_.sliced.empty()) return;

  std::vector<int64_t> slice_values(plan_.sliced.size());
  uint_t remaining = slice_index;
  for (int i = static_cast<int>(plan_.sliced.size()) - 1; i >= 0; i--) {
    slice_values[i] = remaining % plan_.sliced[i].extent;
    remaining /= plan_.sliced[i].extent;
  }

  std::map<int32_t, int64_t> slice_map;
  for (size_t i = 0; i < plan_.sliced.size(); i++)
    slice_map[plan_.sliced[i].mode] = slice_values[i];

  size_t element_bytes = 2 * sizeof(data_t);

  for (size_t t = 0; t < network_desc_.tensors.size(); t++) {
    const TensorSpec &spec = network_desc_.tensors[t];
    int64_t offset_elements = 0;

    std::vector<int64_t> strides(spec.modes.size(), 1);
    for (int i = static_cast<int>(spec.modes.size()) - 2; i >= 0; i--)
      strides[i] = strides[i + 1] * spec.extents[i + 1];

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
