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
#include <unordered_map>
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

// hipTensor 1.5.0 ships kernel instances only at NumDimM=NumDimN=NumDimK=6
// (see contraction_solution_instances.cpp in the upstream library). When a
// per-step contraction descriptor exceeds those bounds — M>6, N>6, or K>6 —
// hipTensor's brute-force dispatcher (HIPTENSOR_ALGO_DEFAULT) silently
// invokes an m6n6k6 kernel against the oversized input. The kernel
// classifies any excess M-axes as K-axes (see contraction_solution_impl.hpp
// where Base::mM and Base::mK are computed from the first MaxNumDimsM=6
// entries vs the rest). The output is silently wrong: extra M-axes are
// summed over instead of preserved as free output dimensions.
//
// This guard refuses to dispatch any step that would hit the m6n6k6
// ceiling. It throws with a clear error message before any kernel runs.
// Without it, users get silent wrong results. Documented as OI11 in the
// design doc.
//
// Kill switch: AER_TN_DISABLE_RANK_GUARD=1 disables the check (for
// bisection or to confirm the silent-wrong-result mode for upstream bug
// reports). Default is enabled — a fast, loud failure beats a slow,
// silent one.
static bool tn_rank_guard_enabled() {
  static bool checked = false;
  static bool enabled = true;
  if (!checked) {
    const char *val = std::getenv("AER_TN_DISABLE_RANK_GUARD");
    enabled = !(val != nullptr && std::string(val) == "1");
    checked = true;
  }
  return enabled;
}

// Cotengra path-search seed, overridable for A/B steering and regression
// probing. NOTE: the loky search pool's trial completion order is
// nondeterministic, so a fixed seed does NOT make a parallel search
// bit-reproducible; this only steers the starting point. The MPI path
// optimizer offsets it per rank (seed + rank) to diversify the ensemble.
// Default 42 preserves the historical hardcoded value.
static uint64_t tn_path_seed() {
  static bool checked = false;
  static uint64_t cached = 42;
  if (!checked) {
    const char *val = std::getenv("AER_TN_PATH_SEED");
    if (val != nullptr) {
      char *end = nullptr;
      long long parsed = std::strtoll(val, &end, 10);
      if (end != val && parsed >= 0) {
        cached = static_cast<uint64_t>(parsed);
      } else {
        fprintf(stderr,
                "[AER_TN] warning: AER_TN_PATH_SEED='%s' is not a non-negative "
                "integer; using default %llu.\n",
                val, (unsigned long long)cached);
      }
    }
    checked = true;
  }
  return cached;
}

// Hard ceiling on a plan's slice count. Suppressing wide steps by tightening
// the per-slice budget trades tiling depth for slice count; past some point
// the contraction is an impractical slice-grind (AER_TN_SLICE_TARGET_BYTES=4096
// produced tens of thousands of slices, job 19231523, which never completed).
// This guard refuses such a plan with a clear error instead of grinding. It
// fires identically on every MPI rank (the MINLOC-broadcast plan is the same
// everywhere), so the failure is collective and clean. Default 8192 is
// generous; recovery is to loosen AER_TN_SLICE_TARGET_BYTES, raise this if the
// cost is acceptable, or use method='statevector'. 0 disables the check.
static uint64_t tn_max_slices() {
  static bool checked = false;
  static uint64_t cached = 8192;
  if (!checked) {
    const char *val = std::getenv("AER_TN_MAX_SLICES");
    if (val != nullptr) {
      char *end = nullptr;
      long long parsed = std::strtoll(val, &end, 10);
      if (end != val && parsed >= 0) {
        cached = static_cast<uint64_t>(parsed);
      } else {
        fprintf(stderr,
                "[AER_TN] warning: AER_TN_MAX_SLICES='%s' is not a non-negative "
                "integer; using default %llu.\n",
                val, (unsigned long long)cached);
      }
    }
    checked = true;
  }
  return cached;
}

// WS-3 mode tiling decomposes an oversized per-step contraction (M>6, N>6, or
// K>6) into a loop of kernel-sized (<=6 each) sub-contractions, so the step
// produces correct results instead of the rank guard throwing. Whether tiling
// is engaged for a given setup_pool_and_cache pass is decided by the driver
// (setup_contraction) and threaded in as the tiling_available parameter, not
// read from the environment at the gate. The enable policy itself lives in
// tn_tiling_mode() (path_optimizer.hpp): OFF refuses, ON always tiles, and AUTO
// (default) lets the driver detect the first oversized step and re-plan once
// with tiling engaged.

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
  // Split-complex tensor slot layout: real plane, then (16B-aligned) imag
  // plane. Copy both planes back to host and assemble complex values.
  std::vector<data_t> host_re(num_elements);
  std::vector<data_t> host_im(num_elements);
  const size_t pb = plane_bytes(num_elements, sizeof(data_t));
  hipMemcpy(host_re.data(), dev_ptr,
            num_elements * sizeof(data_t), hipMemcpyDeviceToHost);
  hipMemcpy(host_im.data(),
            static_cast<char *>(dev_ptr) + pb,
            num_elements * sizeof(data_t), hipMemcpyDeviceToHost);
  fprintf(stderr, "[AER_TN_DEBUG] %s [%ld elements]:", label, (long)num_elements);
  int n_print = (int)std::min<int64_t>(num_elements, max_print);
  for (int i = 0; i < n_print; i++) {
    fprintf(stderr, " (%.3f,%.3fi)",
            static_cast<double>(host_re[i]),
            static_cast<double>(host_im[i]));
  }
  if (num_elements > max_print) fprintf(stderr, " ...");
  fprintf(stderr, "\n");
}

// Plane-pair variant for slice-projected tensors, whose imaginary plane is
// not derivable from the projected element count. Prints leading elements of
// each plane; for strided views these are the parent's leading elements — a
// debug aid for pointer placement, not a strided gather.
template <typename data_t>
void dump_device_planes(const char *label, const data_t *re, const data_t *im,
                        int64_t num_elements, int max_print = 8) {
  if (!tn_debug()) return;
  int n_print = (int)std::min<int64_t>(num_elements, max_print);
  std::vector<data_t> host_re(n_print);
  std::vector<data_t> host_im(n_print);
  hipMemcpy(host_re.data(), re, n_print * sizeof(data_t),
            hipMemcpyDeviceToHost);
  hipMemcpy(host_im.data(), im, n_print * sizeof(data_t),
            hipMemcpyDeviceToHost);
  fprintf(stderr, "[AER_TN_DEBUG] %s [%ld elements]:", label, (long)num_elements);
  for (int i = 0; i < n_print; i++) {
    fprintf(stderr, " (%.3f,%.3fi)",
            static_cast<double>(host_re[i]),
            static_cast<double>(host_im[i]));
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
// Helper: remap mode IDs to a safe range for hipTensor.
//
// hipTensor 1.5.0 on gfx90a silently produces zero output when contraction
// descriptors contain mode IDs in a large-negative range (observed with
// cotengra-hashed IDs around -1.3e9). Small positive IDs always work. This
// helper builds a deterministic remap for the combined A/B/C mode sets so
// hipTensor sees only small positive integers. The mapping is derived from
// the input order and is stable across calls that pass the same mode ID
// sequence, so two identical calls produce identical remapped IDs (and
// therefore identical plan cache signatures).
//
// Does not modify the inputs. Returns fresh remapped arrays in modes_a_out,
// modes_b_out, modes_c_out. Extents are not remapped (they're unaffected).
//
// This is a workaround for a hipTensor/CK bug on AMD gfx90a. cuTensorNet on
// NVIDIA does not exhibit the sensitivity.
//=============================================================================

inline void
remap_modes_to_safe_range(const std::vector<int32_t> &modes_a,
                          const std::vector<int32_t> &modes_b,
                          const std::vector<int32_t> &modes_c,
                          std::vector<int32_t> &modes_a_out,
                          std::vector<int32_t> &modes_b_out,
                          std::vector<int32_t> &modes_c_out) {
  std::unordered_map<int32_t, int32_t> remap;
  auto apply = [&](const std::vector<int32_t> &in, std::vector<int32_t> &out) {
    out.clear();
    out.reserve(in.size());
    for (int32_t m : in) {
      auto it = remap.find(m);
      if (it == remap.end()) {
        int32_t new_id = static_cast<int32_t>(remap.size()) + 1;
        remap.emplace(m, new_id);
        out.push_back(new_id);
      } else {
        out.push_back(it->second);
      }
    }
  };
  apply(modes_a, modes_a_out);
  apply(modes_b, modes_b_out);
  apply(modes_c, modes_c_out);
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

  // Split-complex plane pointers for one tensor: re/im are independent
  // because slice projection advances both planes by the slice's element
  // offset within the FULL parent tensor — after projection the imaginary
  // plane is no longer derivable from the projected element count. This
  // pair is the unit project_slice produces and execute_contraction_planes
  // consumes (design doc §18.4; resolves OI9).
  struct PlanePtrs {
    data_t *re;
    data_t *im;
  };

  // Column-major strides of each input tensor's REMAINING (unsliced) modes
  // within the full parent tensor, aligned index-for-index with
  // sliced_input_specs_[t]. Empty when the tensor carries no sliced mode
  // (packed layout, descriptor strides = nullptr). Built once per plan in
  // build_sliced_specs(); identical for every slice — only pointer offsets
  // differ slice to slice.
  std::vector<std::vector<int64_t>> sliced_input_strides_;

  // Per-step descriptor shape used when submitting contractions to hipTensor.
  // Each PlanSpec holds the (possibly MNK-padded) modes/extents for A, B, C,
  // plus explicit strides for A/B when they are slice-projected views into a
  // larger parent tensor (empty = packed). C is always a packed pool slot.
  // Input tensors may appear in multiple steps with different padding, which
  // is why this is per-step, not per-tensor. Rebuilt on every setup_pool_and_cache.
  struct PlanSpec {
    std::vector<int32_t> modes_a, modes_b, modes_c;
    std::vector<int64_t> extents_a, extents_b, extents_c;
    std::vector<int64_t> strides_a, strides_b;

    // WS-3 mode tiling. Empty `tiles` => in-envelope step, executed exactly as
    // before via the descriptor above (zero behavior change). Non-empty =>
    // this logical step is oversized (M>6, N>6, or K>6) and is executed as a
    // loop of these kernel-sized sub-contractions; each Tile is itself
    // m6n6k6-grammar-compliant. K-block tiles set accumulate=true so their
    // partial products sum into the same C sub-region; M/N-block tiles write
    // disjoint C sub-regions (accumulate=false).
    struct Tile {
      std::vector<int32_t> modes_a, modes_b, modes_c;
      std::vector<int64_t> extents_a, extents_b, extents_c;
      std::vector<int64_t> strides_a, strides_b, strides_c;
      // Element offsets (column-major, mode-0-fastest) into the FULL parent
      // A/B/C tensors for this sub-block's origin. Applied as a base-pointer
      // shift to both split-complex planes, mirroring project_slice().
      int64_t off_a = 0, off_b = 0, off_c = 0;
      bool accumulate = false;   // true for K-block partials (beta=1)
      bool zero_c_first = false; // true on the first writer of each C region
    };
    std::vector<Tile> tiles;
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
  void setup_pool_and_cache(int device_idx, bool tiling_available);
  bool topology_matches_previous() const;
  void cache_topology();
  void contract_single_slice(uint_t slice_index, int device_idx);
  // WS-3: decompose one oversized contraction step (M>6, N>6, or K>6) into a
  // grid of m6n6k6-compliant sub-block Tiles. Returns the tile list; throws if
  // a block still cannot be made grammar-compliant. The caller stores the
  // result in step_plan_specs_[step].tiles.
  std::vector<typename TensorNetContractor_HipTensor<data_t>::PlanSpec::Tile>
  build_tiles_for_step(
      const std::vector<int32_t> &modes_a, const std::vector<int64_t> &extents_a,
      const std::vector<int64_t> &strides_a,
      const std::vector<int32_t> &modes_b, const std::vector<int64_t> &extents_b,
      const std::vector<int64_t> &strides_b,
      const std::vector<int32_t> &modes_c, const std::vector<int64_t> &extents_c);
  void project_slice(uint_t slice_index, int device_idx,
                     std::vector<PlanePtrs> &projected);
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

  // CSC fix (Bell <ZZ> = 0.0 — the REAL root cause):
  // GPUDevice::copy_tensor_data() packs whatever list it is given starting at
  // byte offset 0 of the per-device tensor arena, and only re-allocates when
  // the requested total exceeds the current arena size. Calling it with ONLY
  // the additional (Pauli) tensors therefore wrote them over the FIRST slots
  // of the arena — on top of the network's input tensors. Observed on the
  // n=2 Bell repro: T10/T11 device pointers equaled the T0/T2 slots, the Z
  // matrices' real planes turned T0/T2 into (1,-i) and their imag planes
  // zeroed T1/T3, so every contraction branch through T1/T3 collapsed to 0
  // and the expectation value came out exactly 0.0.
  // Fix: re-upload the FULL input list (network + additional). The additional
  // tensors land in their own slots after the network slots, every entry of
  // tensor_device_ptrs_ stays inside the (possibly re-grown) primary arena —
  // which keeps the per-device slab rebase in the contraction path valid —
  // and update_additional_tensors()' indexing via num_base_tensors_ now
  // points at real, dedicated slots.
  tensor_device_ptrs_ = gpu_mgr_.primary().copy_tensor_data(input_tensors_, true);
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
  hipStream_t stream = gpu_mgr_.primary().stream();
  int dev_id = gpu_mgr_.primary().device_id();
  size_t base = num_base_tensors_;
  // Split-complex H2D: two strided copies (real plane then imag plane).
  // Must match the layout established in GPUDevice::copy_tensor_data.
  for (size_t i = 0; i < tensors.size() && (base + i) < tensor_device_ptrs_.size(); i++) {
    size_t n = tensors[i]->tensor().size();
    size_t pb = plane_bytes(static_cast<int64_t>(n), sizeof(data_t));
    void *dst = tensor_device_ptrs_[base + i];
    const char *src_host =
        reinterpret_cast<const char *>(tensors[i]->tensor().data());
    check_hip(hipMemcpy2DAsync(
        dst, sizeof(data_t),
        src_host, sizeof(std::complex<data_t>),
        sizeof(data_t), n,
        hipMemcpyHostToDevice, stream),
        "hipMemcpy2DAsync(update_re)", dev_id);
    check_hip(hipMemcpy2DAsync(
        static_cast<char *>(dst) + pb, sizeof(data_t),
        src_host + sizeof(data_t), sizeof(std::complex<data_t>),
        sizeof(data_t), n,
        hipMemcpyHostToDevice, stream),
        "hipMemcpy2DAsync(update_im)", dev_id);
  }
  hipStreamSynchronize(stream);
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::set_output(
    std::vector<int32_t> &modes, std::vector<int64_t> &extents) {
  modes_out_ = modes;
  extents_out_ = extents;
  out_size_ = 1;
  for (size_t i = 0; i < extents_out_.size(); i++)
    out_size_ *= extents_out_[i];
  // CSC WS-1: set_output() may now be invoked before set_network() (so the
  // orphan-drop mask sees real output modes). Device discovery normally
  // happens in set_network(); replicate the guard here so primary() is valid
  // and allocate_output() does not dereference an empty devices_ vector when
  // set_output() runs first.
  if (gpu_mgr_.num_devices() == 0) {
    std::vector<uint64_t> targets(target_gpus_.begin(), target_gpus_.end());
    gpu_mgr_.discover(targets);
  }
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

  // Tiling enable policy for THIS setup (AUTO lazy retry).
  //
  //   engaged == false  -> plan and set up as if tiling were unavailable: the
  //                        planner refuses a >12-free-mode output and the
  //                        contractor refuses a per-step M/N/K>6. In AUTO both
  //                        gates throw NeedsTilingException instead of a hard
  //                        error (OFF throws the hard error and never reaches
  //                        the retry).
  //   engaged == true   -> oversized outputs/steps tile instead of refusing.
  //
  // ON starts engaged (always tile, no probe). OFF and AUTO start not-engaged.
  // The retry below runs ONCE: on the first NeedsTilingException in AUTO it
  // re-plans and re-sets-up with engaged = true. A circuit that fits without
  // tiling completes in pass 1 with no path change and no retry; only a circuit
  // that would otherwise have errored pays the extra planning pass.
  const TilingMode tiling_mode = tn_tiling_mode();
  bool engaged = (tiling_mode == TilingMode::On);

  // AUTO short-circuit: if the requested output alone carries more than 12 free
  // modes (full statevector > 12q, large reduced DM), the planner gate is
  // certain to fire in pass 1, so skip the wasted cotengra search and engage
  // tiling directly. An intermediate-only M/N/K>6 step (output <= 12 modes) is
  // not visible here and still needs pass 1 to discover it.
  if (tiling_mode == TilingMode::Auto && modes_out_.size() > 12)
    engaged = true;

  bool retried = false;
  while (true) {
    try {
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
          fprintf(stderr,
                  "[AER_TN] memory: %.1f MB free, %.1f MB ws, %.1f MB budget\n",
                  free_bytes / (1024.0 * 1024.0),
                  workspace_estimate / (1024.0 * 1024.0),
                  memory_budget / (1024.0 * 1024.0));

        auto optimizer = create_optimizer();
        plan_ = optimizer->find_path(network_desc_, memory_budget,
                                     tn_path_seed(), engaged);

        // Slice-count ceiling: refuse an over-sliced plan rather than grind
        // (see tn_max_slices). plan_.num_slices is the MINLOC-broadcast value,
        // identical on every rank, so this throws collectively and cleanly.
        if (tn_max_slices() > 0 && plan_.num_slices > tn_max_slices()) {
          std::stringstream err;
          err << "[AER_TN] plan has " << plan_.num_slices
              << " slices, exceeding the AER_TN_MAX_SLICES ceiling of "
              << tn_max_slices() << ". The per-slice budget is too tight for "
                 "this circuit (a slice-grind). Raise AER_TN_SLICE_TARGET_BYTES, "
                 "raise AER_TN_MAX_SLICES if the cost is acceptable, or use "
                 "method='statevector'.";
          throw std::runtime_error(err.str());
        }

        plan_valid_ = true;
        pool_ready_ = false;
        cache_topology();
      }

      build_sliced_specs();

      if (!pool_ready_) {
        setup_pool_and_cache(0, engaged);
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
            setup_pool_and_cache(i, engaged);
        }
      }

      break;
    } catch (const NeedsTilingException &e) {
      // AUTO only: an oversized output/step was detected with tiling held back.
      // Engage tiling and re-plan/re-setup exactly once. OFF never throws this
      // type (its gates raise std::runtime_error), so the guard is defensive.
      if (tiling_mode != TilingMode::Auto)
        throw;
      if (retried) {
        // Pass 2 already had tiling engaged; tiling always decomposes a step to
        // <= 6, so this should be unreachable. Surface it clearly rather than
        // looping a third time.
        std::stringstream err;
        err << "[AER_TN] tiling was engaged after auto-detection but the "
               "plan still exceeds the m6n6k6 envelope at the "
            << e.where()
            << " gate. This indicates a planning invariant violation, not a "
               "tiling limitation. Underlying detail: "
            << e.what();
        throw std::runtime_error(err.str());
      }
      if (tn_verbose() && myrank_ == 0)
        fprintf(stderr,
                "[AER_TN] oversized step detected at %s; tiling auto-enabled "
                "for this run (re-planning once)\n",
                e.where());
      engaged = true;
      retried = true;
      // Force a fresh path search on pass 2: the held-back plan must not be
      // reused, and pool/topology caches from pass 1 are stale.
      plan_valid_ = false;
      prev_valid_ = false;
      pool_ready_ = false;
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
  sliced_input_strides_.assign(num_inputs, {});

  for (size_t t = 0; t < num_inputs; t++) {
    const TensorSpec &full = network_desc_.tensors[t];
    sliced_input_specs_[t].modes.clear();
    sliced_input_specs_[t].extents.clear();

    bool has_sliced_mode = false;
    int64_t stride = 1; // column-major: mode 0 fastest, parent extents
    std::vector<int64_t> remaining_strides;
    for (size_t m = 0; m < full.modes.size(); m++) {
      if (sliced_mode_set_.find(full.modes[m]) == sliced_mode_set_.end()) {
        sliced_input_specs_[t].modes.push_back(full.modes[m]);
        sliced_input_specs_[t].extents.push_back(full.extents[m]);
        remaining_strides.push_back(stride);
      } else {
        has_sliced_mode = true;
      }
      stride *= full.extents[m];
    }

    // A projected tensor is a strided view inside its parent: the remaining
    // modes keep their PARENT strides (which is what makes projection pure
    // pointer arithmetic). Tensors without sliced modes stay packed, marked
    // by an empty stride vector → descriptor built with nullptr strides.
    if (has_sliced_mode)
      sliced_input_strides_[t] = std::move(remaining_strides);
  }
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::setup_pool_and_cache(int device_idx,
                                                                 bool tiling_available) {
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
  //
  // Dummy stripping: a padding dummy from step N is just a rank-filler for
  // that one contraction's M or N requirement. Once the step completes it
  // carries no information — it's an extent-1 axis with no correspondence
  // in the natural tensor network. But without intervention, dummies ride
  // along as "free" modes through the whole contraction chain, causing
  // the output rank of late steps to balloon. hipTensor 1.5.0 on gfx90a
  // has observed shape-specific correctness bugs (e.g. rank-12 C with 3+
  // consecutive extent-1 modes silently writes only half the output
  // tensor — n=10 QAOA ring triggers this, n=9 doesn't). We eliminate
  // the accumulation by stripping extent-1 modes from A and B's
  // descriptors at the START of each step, letting that step's
  // pad_contraction_mnk re-add only the dummies IT needs. Downstream
  // steps then see the minimum-rank spec for C.
  //
  // Memory layout is unaffected: extent-1 axes contribute nothing to
  // stride or element count, so stripping them is purely a descriptor
  // change. No data movement. Strides (present only for slice-projected
  // input views) drop entries in lockstep so they stay index-aligned
  // with modes/extents.
  auto strip_extent_one = [](std::vector<int32_t> &modes,
                             std::vector<int64_t> &extents,
                             std::vector<int64_t> &strides) {
    std::vector<int32_t> m;
    std::vector<int64_t> e;
    std::vector<int64_t> s;
    m.reserve(modes.size());
    e.reserve(extents.size());
    s.reserve(strides.size());
    const bool has_strides = !strides.empty();
    for (size_t i = 0; i < modes.size(); i++) {
      if (extents[i] > 1) {
        m.push_back(modes[i]);
        e.push_back(extents[i]);
        if (has_strides)
          s.push_back(strides[i]);
      }
    }
    modes = std::move(m);
    extents = std::move(e);
    strides = std::move(s);
  };

  step_plan_specs_.assign(num_steps, PlanSpec{});

  for (size_t step = 0; step < num_steps; step++) {
    uint64_t left = plan_.steps[step].left;
    uint64_t right = plan_.steps[step].right;
    size_t result_idx = num_inputs + step;

    // WS-3: set in the rank guard when this oversized step is to be tiled.
    // The actual build_tiles_for_step runs after the final-step C-reorder
    // below so tiles encode modes_c in its FINAL physical order.
    bool tile_this_step = false;
    int tile_num_m = 0, tile_num_n = 0, tile_num_k = 0;

    // Natural (unpadded) A and B modes come from all_specs_ (which for inputs
    // reflects sliced_input_specs_, and for earlier intermediates reflects
    // their padded forms — that's fine, the padded form is just the valid
    // shape with extent-1 dummies which don't affect downstream semantics).
    std::vector<int32_t> modes_a = all_specs_[left].modes;
    std::vector<int64_t> extents_a = all_specs_[left].extents;
    std::vector<int32_t> modes_b = all_specs_[right].modes;
    std::vector<int64_t> extents_b = all_specs_[right].extents;

    // Slice-projected input views carry their parent strides; intermediates
    // and unsliced inputs are packed (empty → descriptor strides nullptr).
    std::vector<int64_t> strides_a =
        (left < num_inputs) ? sliced_input_strides_[left]
                            : std::vector<int64_t>{};
    std::vector<int64_t> strides_b =
        (right < num_inputs) ? sliced_input_strides_[right]
                             : std::vector<int64_t>{};

    // Strip extent-1 dummies from A and B before this step's planning.
    // Any dummies THIS step genuinely needs will be re-added by
    // pad_contraction_mnk below. Keeping descriptors at their minimum
    // natural rank sidesteps the hipTensor/CK rank-12-plus-consecutive-
    // dummies bug documented above.
    strip_extent_one(modes_a, extents_a, strides_a);
    strip_extent_one(modes_b, extents_b, strides_b);

    std::vector<int32_t> modes_c;
    std::vector<int64_t> extents_c;
    compute_contraction_result(modes_a, extents_a, modes_b, extents_b,
                               modes_c, extents_c);

    int dummies_on_c = 0;
    if (tn_mnk_padding_enabled()) {
      dummies_on_c = pad_contraction_mnk(modes_a, extents_a,
                                         modes_b, extents_b,
                                         modes_c, extents_c);
      // Extend stride lists for appended extent-1 dummies: their stride is
      // never dereferenced (extent 1), so any value works — 1 is the value
      // hipTensor's own packed formula yields for an extent-1 mode at the
      // front, keeping the descriptor maximally unsurprising for CK.
      // Packed tensors (empty stride list) stay packed: hipTensor's nullptr
      // default extends correctly to appended extent-1 dummies.
      if (!strides_a.empty())
        strides_a.resize(modes_a.size(), 1);
      if (!strides_b.empty())
        strides_b.resize(modes_b.size(), 1);
    }

    // OI11 rank guard: refuse to dispatch any step whose padded descriptor
    // exceeds hipTensor 1.5.0's m6n6k6 kernel coverage. Without this guard,
    // contractions where M, N, or K exceed 6 produce silent wrong results
    // because the brute-force dispatcher invokes an m6n6k6 kernel and the
    // kernel sums excess M-axes into the contracted-K dimension. See
    // tn_rank_guard_enabled() above for full rationale and kill switch.
    if (tn_rank_guard_enabled()) {
      // Recompute M/N/K from the (possibly padded) descriptors. M = modes
      // free on A only, N = modes free on B only, K = modes shared.
      std::set<int32_t> sa_post(modes_a.begin(), modes_a.end());
      std::set<int32_t> sb_post(modes_b.begin(), modes_b.end());
      int num_m = 0, num_n = 0, num_k = 0;
      for (int32_t m : modes_a) {
        if (sb_post.count(m)) num_k++; else num_m++;
      }
      for (int32_t m : modes_b) {
        if (!sa_post.count(m)) num_n++;
      }

      if (num_m > 6 || num_n > 6 || num_k > 6) {
        if (tiling_available) {
          // WS-3: decompose this oversized step into a grid of m6n6k6
          // sub-contractions instead of refusing. Works for both unsliced and
          // sliced plans: for a sliced step the input leaves are slice-
          // projected strided views, and build_tiles_for_step derives each
          // block's offsets and free-mode strides from the projected parent
          // strides (passed in strides_a/strides_b). Those strides are slice-
          // independent, so the tiles built once here are reused for every
          // slice; project_slice applies the per-slice base offset, and the
          // tile offset adds on top in the same column-major stride space
          // (slice-offset + tile-offset compose; verified bit-equal to einsum).
          // Slice-sum (across slices, into out_buf) and tile-sum (within a
          // slice: disjoint M/N regions plus K-accumulation into the C slot)
          // are sums at two levels and compose.
          //
          // Defer the actual build_tiles_for_step call until AFTER the
          // final-step C-reorder block below. Tiles encode column-major
          // offsets and strides over modes_c; the final step rewrites modes_c
          // into modes_out_ order, so building here (pre-reorder) would make
          // the final step write C in a different order than downstream steps
          // and output extraction read it. Just flag the step now and record
          // the M/N/K for the verbose line emitted after the build.
          tile_this_step = true;
          tile_num_m = num_m;
          tile_num_n = num_n;
          tile_num_k = num_k;
          // fall through: record the full descriptor as usual below.
        } else {
          std::stringstream err;
          err << "hipTensor m6n6k6 ceiling exceeded at contraction step "
              << step << " of " << num_steps
              << " (T" << left << " x T" << right << " -> T" << result_idx << "): "
              << "M=" << num_m << " N=" << num_n << " K=" << num_k
              << " (hipTensor 1.5.0 ships kernels only at M=N=K=6). "
              << "modes_a=" << modes_to_str(modes_a)
              << " modes_b=" << modes_to_str(modes_b)
              << " modes_c=" << modes_to_str(modes_c) << ". "
              << "This circuit's contraction path requires kernel shapes "
              << "that hipTensor 1.5.0 does not provide. Options: ";
          err << "(1) enable mode tiling with AER_TN_TILING=on (or leave the "
              << "default auto, which engages tiling on demand; decomposes "
              << "oversized steps into m6n6k6 sub-contractions, supports both "
              << "unsliced and sliced plans), ";
          err << "(2) tighten the per-slice budget so the slicer cuts this "
              << "step inside the envelope (lower AER_TN_SLICE_TARGET_BYTES, "
              << "current default 65536), "
              << "(3) use method='statevector' (validated on LUMI to 44 "
              << "qubits at depth 30 on 1024 nodes, CSC April 2025), "
              << "(4) reduce circuit depth or qubit count, "
              << "(5) try a different contraction-path optimizer setting "
              << "via cotengra to see if a lower-rank path exists. "
              << "Set AER_TN_DISABLE_RANK_GUARD=1 to bypass this check "
              << "(produces SILENT WRONG RESULTS — for bug-reporting only).";
          if (tn_tiling_mode() == TilingMode::Auto)
            throw NeedsTilingException(NeedsTilingException::Gate::Contractor,
                                       err.str());
          throw std::runtime_error(err.str());
        }
      }
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

    // WS-3: now that modes_c/extents_c are in their FINAL physical order
    // (after any final-step reorder above), build the sub-block tiles. They
    // encode column-major offsets and strides over this exact C order, which
    // is the order downstream steps and output extraction read C in, so the
    // tiled writes and the reads agree. The recorded full modes_c/extents_c
    // below still describe the untiled output for pool layout; the untiled
    // whole-step plan is never executed for a tiled step.
    if (tile_this_step) {
      step_plan_specs_[step].tiles = build_tiles_for_step(
          modes_a, extents_a, strides_a,
          modes_b, extents_b, strides_b,
          modes_c, extents_c);
      if (tn_verbose())
        fprintf(stderr,
                "[AER_TN] tiling step %zu (M=%d N=%d K=%d) into %zu "
                "m6n6k6 sub-contractions\n",
                step, tile_num_m, tile_num_n, tile_num_k,
                step_plan_specs_[step].tiles.size());

      // Verify the actual tile DESCRIPTOR against hipTensor's strided-view
      // ceiling. The planner bounds intermediate VOLUME (its only lever is
      // slicing summed bonds), which in turn bounds the descriptor's free-mode
      // stride; here we check the EXACT strides the kernel will see -- the true
      // CK constraint, which only the built descriptor exposes (mode order and
      // extent-1 padding can make the volume->stride relation imperfect, and a
      // future high-stride-first reorder would decouple them further). A
      // descriptor whose max free-mode stride reaches the tiling ceiling is
      // what miscomputes and then faults CK plan creation; refuse it here,
      // before any kernel for this step runs, as a clean runtime_error (not a
      // NeedsTilingException) instead of a silent GPU memory fault. This is the
      // contractor-side half of the envelope invariant the planner enforces by
      // slicing: planner volume <= max_tiled_elements => stride < that, so a
      // stride at/above it means the invariant was violated upstream.
      const uint64_t stride_ceiling = max_tiled_elements();
      int64_t max_stride = 0;
      if (!step_plan_specs_[step].tiles.empty()) {
        const auto &t0 = step_plan_specs_[step].tiles.front();
        for (int64_t s : t0.strides_a)
          max_stride = std::max(max_stride, s);
        for (int64_t s : t0.strides_b)
          max_stride = std::max(max_stride, s);
        for (int64_t s : t0.strides_c)
          max_stride = std::max(max_stride, s);
      }
      if (static_cast<uint64_t>(max_stride) >= stride_ceiling) {
        std::stringstream err;
        err << "[AER_TN] tiling step " << step << " of " << num_steps
            << " (M=" << tile_num_m << " N=" << tile_num_n
            << " K=" << tile_num_k << ", " << step_plan_specs_[step].tiles.size()
            << " sub-contractions) produced a tile descriptor with max "
               "free-mode stride "
            << max_stride << ", at/above the m6n6k6 tiling ceiling of "
            << stride_ceiling
            << " elements. hipTensor's m6n6k6 kernels miscompute and then fault "
               "during plan creation on strided views this large. The planner "
               "should have sliced this step inside the envelope; if you raised "
               "AER_TN_SLICE_TARGET_BYTES, lower it so the slicer keeps "
               "per-slice intermediates small, or raise AER_TN_MAX_TILED_ELEMENTS "
               "only after validating that descriptors at this stride build and "
               "compute correctly on your hipTensor build.";
        throw std::runtime_error(err.str());
      }
    }

    // Record the exact per-plan descriptor shape for contract_single_slice.
    step_plan_specs_[step].modes_a = modes_a;
    step_plan_specs_[step].extents_a = extents_a;
    step_plan_specs_[step].modes_b = modes_b;
    step_plan_specs_[step].extents_b = extents_b;
    step_plan_specs_[step].modes_c = modes_c;
    step_plan_specs_[step].extents_c = extents_c;
    step_plan_specs_[step].strides_a = strides_a;
    step_plan_specs_[step].strides_b = strides_b;

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
    // Use the padded per-step descriptor shape, not the raw all_specs_
    // (which for inputs may differ from what this specific plan needs).
    const PlanSpec &ps = step_plan_specs_[step];

    if (tn_verbose())
      fprintf(stderr, "[AER_TN] prebuild step %zu (%s, %zu tiles)\n",
              step, ps.tiles.empty() ? "untiled" : "tiled", ps.tiles.size());

    if (!ps.tiles.empty()) {
      // WS-3 tiled step: pre-build each sub-block plan so its workspace is
      // counted by max_workspace_bytes() below. The untiled whole-step plan is
      // never executed for a tiled step, so we do NOT build it here.
      for (const auto &t : ps.tiles) {
        std::vector<int32_t> ma_safe, mb_safe, mc_safe;
        remap_modes_to_safe_range(t.modes_a, t.modes_b, t.modes_c,
                                  ma_safe, mb_safe, mc_safe);
        auto tsig = build_signature(ma_safe, t.extents_a, mb_safe, t.extents_b,
                                    mc_safe, t.extents_c, t.strides_a, t.strides_b,
                                    t.strides_c);
        dev.plan_cache().get_or_create(
            tsig, ma_safe, t.extents_a, mb_safe, t.extents_b,
            mc_safe, t.extents_c, t.strides_a, t.strides_b, t.strides_c);
      }
      continue;
    }

    // Remap mode IDs to a safe range (see remap_modes_to_safe_range above).
    // Must use identical logic to contract_single_slice so cache
    // signatures match between pre-population here and execution later.
    std::vector<int32_t> modes_a_safe, modes_b_safe, modes_c_safe;
    remap_modes_to_safe_range(ps.modes_a, ps.modes_b, ps.modes_c,
                              modes_a_safe, modes_b_safe, modes_c_safe);

    auto sig = build_signature(
        modes_a_safe, ps.extents_a,
        modes_b_safe, ps.extents_b,
        modes_c_safe, ps.extents_c,
        ps.strides_a, ps.strides_b);

    dev.plan_cache().get_or_create(
        sig,
        modes_a_safe, ps.extents_a,
        modes_b_safe, ps.extents_b,
        modes_c_safe, ps.extents_c,
        ps.strides_a, ps.strides_b);
  }

  // Each intermediate occupies a split-complex tensor slot: two planes
  // (real, imag), each padded to a 16-byte boundary.
  std::vector<std::tuple<size_t, int, int, int>> intermediates;

  for (size_t step = 0; step < num_steps; step++) {
    size_t result_idx = num_inputs + step;
    int64_t num_elements = all_specs_[result_idx].num_elements();
    size_t bytes = tensor_slot_bytes(num_elements, sizeof(data_t));
    int birth = static_cast<int>(step);
    int death = last_used[result_idx];
    if (death < 0) death = static_cast<int>(num_steps - 1);

    intermediates.push_back(
        std::make_tuple(bytes, birth, death, static_cast<int>(result_idx)));
  }

  if (tn_verbose())
    fprintf(stderr, "[AER_TN] prebuild loop done; computing workspace\n");

  uint64_t max_ws = dev.plan_cache().max_workspace_bytes();

  if (tn_verbose())
    fprintf(stderr, "[AER_TN] workspace computed (max_ws=%lu); laying out pool\n",
            (unsigned long)max_ws);

  dev.pool().plan_layout(intermediates, max_ws, static_cast<int>(num_steps));

  if (tn_verbose())
    fprintf(stderr, "[AER_TN] pool layout done; allocating\n");

  dev.pool().allocate(dev.device_id());

  if (tn_verbose())
    fprintf(stderr, "[AER_TN] pool: %zu bytes, %zu plans cached (device %d)\n",
            dev.pool().total_size(), dev.plan_cache().size(), dev.device_id());
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::contract(
    std::vector<std::complex<data_t>> &out) {
  // Each device holds the partial sum of its assigned slice range after
  // contract_all(); each rank holds its slice range after MPI partitioning.
  // Reduce both layers before reading the result off the primary device —
  // skipping either reduction drops every slice not assigned to (rank 0,
  // device 0). Reduction order is fixed by index (P4).
  contract_all();
  accumulate_across_gpus();
#ifdef AER_MPI
  accumulate_across_mpi();
#endif
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
        "combo", -1, -1.0, "hyper", elem_bytes));
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

  // Every operand is a (real plane, imag plane) pointer pair. Inputs come
  // from slice projection (parent slot + slice offset, both planes);
  // intermediates are packed pool slots whose imag plane is at
  // plane_bytes(N) from the slot base.
  std::vector<PlanePtrs> all_planes(num_inputs + num_steps,
                                    PlanePtrs{nullptr, nullptr});
  project_slice(slice_index, device_idx, all_planes);

  void *workspace = dev.pool().get_workspace_ptr();
  uint64_t ws_size = dev.pool().get_workspace_size();

  if (tn_debug()) {
    for (size_t i = 0; i < num_inputs; i++) {
      char label[64];
      snprintf(label, sizeof(label), "  input[%zu]", i);
      // Print up to 32 elements so 4-leg gates (e.g. CNOT's 16 values) are
      // fully visible. For slice-projected strided views this prints the
      // parent's leading elements (debug aid, not a strided gather).
      dump_device_planes<data_t>(label, all_planes[i].re, all_planes[i].im,
                                 sliced_input_specs_[i].num_elements(), 32);
    }
  }

  for (size_t step = 0; step < num_steps; step++) {
    uint64_t left = plan_.steps[step].left;
    uint64_t right = plan_.steps[step].right;
    size_t result_idx = num_inputs + step;

    // Use the per-step padded descriptor shape we recorded in
    // setup_pool_and_cache. The plan cache is keyed on these same
    // shapes (including slice-view strides), so the lookup is O(1) after
    // the first slice — all slices of a step share one plan by
    // construction (§8.3).
    const PlanSpec &ps = step_plan_specs_[step];

    void *c_slot = dev.pool().get_tensor_ptr(static_cast<int>(result_idx));
    const size_t c_plane = plane_bytes(
        all_specs_[result_idx].num_elements(), sizeof(data_t));
    all_planes[result_idx].re = reinterpret_cast<data_t *>(c_slot);
    all_planes[result_idx].im =
        reinterpret_cast<data_t *>(static_cast<char *>(c_slot) + c_plane);

    if (!ps.tiles.empty()) {
      // ---- WS-3 tiled step ----
      // Zero the whole C slot ONCE up front: K-accumulating tiles add into it,
      // and disjoint M/N tiles each overwrite their own region. (Per-tile
      // zeroing would clobber the K running sum — see WS3_tiling_design.md.)
      {
        size_t c_bytes = tensor_slot_bytes(
            all_specs_[result_idx].num_elements(), sizeof(data_t));
        check_hip(hipMemsetAsync(c_slot, 0, c_bytes, dev.stream()),
                  "hipMemsetAsync(tiled C pre-zero)", dev.device_id());
      }
      if (tn_debug())
        fprintf(stderr,
                "[AER_TN_DEBUG]   step %zu TILED into %zu sub-contractions\n",
                step, ps.tiles.size());

      for (size_t ti = 0; ti < ps.tiles.size(); ti++) {
        const auto &t = ps.tiles[ti];

        // Per-block safe-ID remap (identical discipline to the untiled path).
        std::vector<int32_t> ma_safe, mb_safe, mc_safe;
        remap_modes_to_safe_range(t.modes_a, t.modes_b, t.modes_c,
                                  ma_safe, mb_safe, mc_safe);
        auto tsig = build_signature(ma_safe, t.extents_a, mb_safe, t.extents_b,
                                    mc_safe, t.extents_c, t.strides_a, t.strides_b,
                                    t.strides_c);
        const CachedPlan<data_t> &tplan = dev.plan_cache().get_or_create(
            tsig, ma_safe, t.extents_a, mb_safe, t.extents_b,
            mc_safe, t.extents_c, t.strides_a, t.strides_b, t.strides_c);

        // Apply this block's column-major element offsets as base-pointer
        // shifts to BOTH split-complex planes (mirrors project_slice). Use the
        // re/im plane bases already resolved in all_planes[]: for sliced inputs
        // these come from project_slice (parent-tensor plane sizing + slice
        // offset), for intermediates from the packed slot layout below. We must
        // NOT recompute the re->im gap here — project_slice sizes it on the
        // FULL parent, not the sliced element count, so recomputing would be
        // wrong for sliced inputs. Offsetting the already-correct .im base is
        // valid because the tile offset is an element index into the same
        // (parent or packed) column-major layout both planes share.
        data_t *Ar = all_planes[left].re + t.off_a;
        data_t *Ai = all_planes[left].im + t.off_a;
        data_t *Br = all_planes[right].re + t.off_b;
        data_t *Bi = all_planes[right].im + t.off_b;
        data_t *Cr = all_planes[result_idx].re + t.off_c;
        data_t *Ci = all_planes[result_idx].im + t.off_c;

        dev.execute_contraction_planes(tplan, Ar, Ai, Br, Bi, Cr, Ci,
                                       workspace, ws_size, t.accumulate);
      }

      if (tn_debug()) {
        hipStreamSynchronize(dev.stream());
        char label[64];
        snprintf(label, sizeof(label), "  after tiled step %zu (T%zu)", step,
                 result_idx);
        dump_device_planes<data_t>(label, all_planes[result_idx].re,
                                   all_planes[result_idx].im,
                                   all_specs_[result_idx].num_elements(), 32);
      }
      continue; // tiled step done; skip the single-dispatch path below
    }

    // ---- Mode ID remap ----
    // hipTensor / Composable Kernel silently produces zero output when
    // mode IDs are large negative (observed experimentally with
    // cotengra-hashed IDs in the -1.3e9 range). Small positive mode IDs
    // always work. Remap A/B/C mode IDs into {1, 2, 3, ...} per-step so
    // the descriptor hipTensor sees only contains safe values. The
    // semantic mode IDs stay in ps.modes_* for all other bookkeeping
    // (plan cache signature, debug logging, all_specs_).
    //
    // This is a workaround for a bug in hipTensor 1.5.0 on gfx90a.
    // cuTensorNet on NVIDIA does not exhibit this sensitivity. If AMD
    // fixes CK to handle the full int32 range, this remap becomes a
    // no-op and can be removed.
    std::vector<int32_t> modes_a_safe, modes_b_safe, modes_c_safe;
    remap_modes_to_safe_range(ps.modes_a, ps.modes_b, ps.modes_c,
                              modes_a_safe, modes_b_safe, modes_c_safe);

    // Cache signature uses the remapped IDs so identical remapped
    // descriptors hit the cache; strides distinguish projected views from
    // packed tensors of the same shape.
    auto sig = build_signature(
        modes_a_safe, ps.extents_a,
        modes_b_safe, ps.extents_b,
        modes_c_safe, ps.extents_c,
        ps.strides_a, ps.strides_b);

    const CachedPlan<data_t> &plan = dev.plan_cache().get_or_create(
        sig,
        modes_a_safe, ps.extents_a,
        modes_b_safe, ps.extents_b,
        modes_c_safe, ps.extents_c,
        ps.strides_a, ps.strides_b);

    // Defense in depth against two orthogonal failure modes:
    //   1. The pool memory for this intermediate is freshly allocated or
    //      aliased from a prior step, so its contents are undefined.
    //   2. hiptensorContraction's C pointer aliases its D pointer (we pass
    //      ptr_c for both with beta=0). Per the API spec C is only read
    //      when beta!=0, but CK bilinear kernels have historically done an
    //      RMW on the D tile regardless, and under -ffast-math (which the
    //      Aer build uses) reading uninitialized memory can produce NaNs
    //      that propagate via 0*NaN=NaN and get flushed to junk.
    // Zeroing the full slot (both real and imag planes) before the call
    // kills both hazards at microsecond cost.
    {
      size_t c_bytes = tensor_slot_bytes(
          all_specs_[result_idx].num_elements(), sizeof(data_t));
      check_hip(hipMemsetAsync(c_slot, 0, c_bytes, dev.stream()),
                "hipMemsetAsync(C pre-zero)", dev.device_id());
    }

    if (tn_debug()) {
      fprintf(stderr,
              "[AER_TN_DEBUG]   step %zu exec: "
              "A_re=%p B_re=%p C_re=%p ws=%p ws_sz=%lu modes_c=%s "
              "strided=%c%c\n",
              step, (void *)all_planes[left].re, (void *)all_planes[right].re,
              (void *)all_planes[result_idx].re,
              workspace, (unsigned long)ws_size,
              modes_to_str(ps.modes_c).c_str(),
              ps.strides_a.empty() ? '-' : 'A',
              ps.strides_b.empty() ? '-' : 'B');
    }

    // Split-complex decomposition with explicit plane pointers (OI9
    // resolution): each operand's planes were resolved above — by slice
    // projection for inputs and by slot layout for intermediates.
    dev.execute_contraction_planes(plan,
        all_planes[left].re, all_planes[left].im,
        all_planes[right].re, all_planes[right].im,
        all_planes[result_idx].re, all_planes[result_idx].im,
        workspace, ws_size, false);

    if (tn_debug()) {
      hipStreamSynchronize(dev.stream());
      char label[64];
      snprintf(label, sizeof(label), "  after step %zu (T%zu)", step, result_idx);
      dump_device_planes<data_t>(label, all_planes[result_idx].re,
                                 all_planes[result_idx].im,
                                 all_specs_[result_idx].num_elements(), 32);
    }
  }

  size_t final_idx = num_inputs + num_steps - 1;
  int64_t final_elements = all_specs_[final_idx].num_elements();

  if (tn_debug())
    fprintf(stderr,
            "[AER_TN_DEBUG]   accumulate final T%zu (%ld el) into out_buf (%lu el)\n",
            final_idx, (long)final_elements, (unsigned long)out_size_);

  if (final_elements != (int64_t)out_size_) {
    // The final slice tensor must always cover the full unsliced output:
    // cotengra slices only summed (inner) bonds, and the planner rejects
    // plans that slice an output mode. A mismatch is a planner/slicer
    // bookkeeping bug — refuse loudly rather than corrupt the result (P3).
    std::stringstream err;
    err << "TensorNetContractor_HipTensor: final slice tensor has "
        << final_elements << " elements but output expects " << out_size_
        << " (slice " << slice_index << "). A sliced mode leaked into the "
        << "output tensor; this is a slicing bookkeeping bug.";
    throw std::runtime_error(err.str());
  }

  // The final slice tensor is stored split-complex (real plane followed by
  // imag plane). Accumulate it into the interleaved dev_out_ buffer via a
  // small thrust kernel that reads both planes and forms complex values
  // on the fly. Slices on one device run sequentially in increasing index
  // order, so the per-device accumulation order is fixed (P4).
  dev.accumulate_planar_to_output(all_planes[final_idx].re, out_size_);

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

// WS-3 mode tiling: decompose one oversized contraction (M>6, N>6, or K>6)
// into a grid of m6n6k6-compliant sub-contractions. Verified bit-equal to
// einsum (M-tiling exact; K-tiling to fp roundoff) before implementation.
//
// Model (mirrors project_slice's column-major offset arithmetic): for each
// axis that exceeds 6 modes, take the EXCESS modes as "tile modes" and iterate
// every mixed-radix combination of their values. For a fixed combination:
//   * M tile-modes select a disjoint sub-region of C and the matching rows of
//     A  -> the block writes its own C region (accumulate=false).
//   * N tile-modes  -> disjoint C region and matching rows of B
//     (accumulate=false).
//   * K tile-modes select matching slabs of A AND B and the partial product
//     must be SUMMED into the same C region -> accumulate=true.
// Each block's remaining (free) modes number <=6 on every axis, so it is a
// legal m6n6k6 call. Offsets are element offsets in the FULL parent tensor's
// column-major (mode-0-fastest) layout; contract_single_slice applies them as
// base-pointer shifts to both split-complex planes.
template <typename data_t>
std::vector<typename TensorNetContractor_HipTensor<data_t>::PlanSpec::Tile>
TensorNetContractor_HipTensor<data_t>::build_tiles_for_step(
    const std::vector<int32_t> &modes_a, const std::vector<int64_t> &extents_a,
    const std::vector<int64_t> &strides_a,
    const std::vector<int32_t> &modes_b, const std::vector<int64_t> &extents_b,
    const std::vector<int64_t> &strides_b,
    const std::vector<int32_t> &modes_c, const std::vector<int64_t> &extents_c) {
  using Tile = typename PlanSpec::Tile;

  // strides_a/strides_b are the parent input strides: non-empty (and
  // slice-independent) for a slice-projected leaf input, empty for a packed
  // input (intermediate, or unsliced leaf). They feed sa/sb below so a block is
  // a correct strided view into either a projected or a packed parent. This is
  // what lets tiling compose with slicing.

  std::set<int32_t> set_a(modes_a.begin(), modes_a.end());
  std::set<int32_t> set_b(modes_b.begin(), modes_b.end());
  std::set<int32_t> set_c(modes_c.begin(), modes_c.end());

  // Classify modes. M = in A & C not B; N = in B & C not A; K = in A & B not C.
  std::vector<int32_t> M, N, K;
  for (int32_t m : modes_a) {
    if (set_b.count(m)) { if (!set_c.count(m)) K.push_back(m); }
    else if (set_c.count(m)) M.push_back(m);
  }
  for (int32_t m : modes_b) {
    if (!set_a.count(m) && set_c.count(m)) N.push_back(m);
  }

  // Column-major element stride of each mode within its parent tensor (mode 0
  // fastest), so an offset for fixing mode == value is value * stride.
  auto stride_map = [](const std::vector<int32_t> &modes,
                       const std::vector<int64_t> &extents) {
    std::map<int32_t, int64_t> s;
    int64_t st = 1;
    for (size_t i = 0; i < modes.size(); i++) { s[modes[i]] = st; st *= extents[i]; }
    return s;
  };
  // sa/sb are the parent column-major element strides used both for a block's
  // base-pointer offsets and for its surviving free-mode descriptor strides.
  // When an input is a SLICE-PROJECTED view, its parent strides arrive in
  // strides_a/strides_b (non-empty) and are slice-INDEPENDENT — project_slice
  // fixes only the per-slice base offset, never the strides — so the tiles
  // built here from them are a correct strided view into the projected parent
  // for every slice. When the input is packed (an intermediate, or an unsliced
  // leaf) the stride list is empty and we derive packed column-major strides
  // from extents, exactly as the unsliced increment did. C is always a packed
  // intermediate slot (never projected), so sc stays packed.
  auto strides_to_map = [](const std::vector<int32_t> &modes,
                           const std::vector<int64_t> &strides) {
    std::map<int32_t, int64_t> s;
    for (size_t i = 0; i < modes.size() && i < strides.size(); i++)
      s[modes[i]] = strides[i];
    return s;
  };
  std::map<int32_t, int64_t> sa = strides_a.empty()
      ? stride_map(modes_a, extents_a) : strides_to_map(modes_a, strides_a);
  std::map<int32_t, int64_t> sb = strides_b.empty()
      ? stride_map(modes_b, extents_b) : strides_to_map(modes_b, strides_b);
  std::map<int32_t, int64_t> sc = stride_map(modes_c, extents_c);
  std::map<int32_t, int64_t> ext;
  for (size_t i = 0; i < modes_a.size(); i++) ext[modes_a[i]] = extents_a[i];
  for (size_t i = 0; i < modes_b.size(); i++) ext[modes_b[i]] = extents_b[i];
  for (size_t i = 0; i < modes_c.size(); i++) ext[modes_c[i]] = extents_c[i];

  // Tile modes = the excess beyond 6 on each oversized axis. Free modes stay.
  // Only EXTENT>1 modes are eligible to be tiled: extent-1 dummies (added by
  // pad_contraction_mnk upstream) carry no data, so keeping them free and
  // tiling only real modes guarantees each block's free axis holds <=6 real
  // modes irrespective of where padding placed the dummies.
  auto excess = [&ext](const std::vector<int32_t> &axis) {
    std::vector<int32_t> real;
    for (int32_t m : axis) if (ext[m] > 1) real.push_back(m);
    return (real.size() > 6)
               ? std::vector<int32_t>(real.begin() + 6, real.end())
               : std::vector<int32_t>{};
  };
  std::vector<int32_t> tM = excess(M), tN = excess(N), tK = excess(K);
  std::set<int32_t> tile_set;
  for (int32_t m : tM) tile_set.insert(m);
  for (int32_t m : tN) tile_set.insert(m);
  for (int32_t m : tK) tile_set.insert(m);

  // Ordered list of all tile modes, with which operands each one offsets and
  // whether it is a (summed) K mode. Mixed-radix iteration order is fixed and
  // deterministic (P4).
  struct TileMode { int32_t mode; bool in_a, in_b, in_c; bool is_k; int64_t extent; };
  std::vector<TileMode> tile_modes;
  for (int32_t m : tM) tile_modes.push_back({m, true, false, true, false, ext[m]});
  for (int32_t m : tN) tile_modes.push_back({m, false, true, true, false, ext[m]});
  for (int32_t m : tK) tile_modes.push_back({m, true, true, false, true, ext[m]});

  bool any_k_tiled = !tK.empty();

  // The per-block (free) descriptor: drop tile modes from A/B/C mode lists.
  // The surviving free modes keep their PARENT column-major strides (sa/sb/sc
  // computed above), NOT re-packed strides. This is what makes a block a
  // correct strided VIEW into its parent: when a non-trailing axis is tiled
  // away, the kept axes are no longer contiguous in the parent buffer, so a
  // packed descriptor would mis-address every element past the gap (the bug
  // this fixes — packed-contiguous writes were only correct for N-only and
  // K-only tiling). sa/sb are the parent's true strides whether the parent is
  // packed (unsliced/intermediate) or slice-projected; sc is always packed (C
  // is an intermediate slot). Identical for every block; offsets differ.
  auto drop_tiles = [&tile_set](const std::vector<int32_t> &modes,
                                const std::vector<int64_t> &extents,
                                const std::map<int32_t, int64_t> &pstride,
                                std::vector<int32_t> &om,
                                std::vector<int64_t> &oe,
                                std::vector<int64_t> &os) {
    for (size_t i = 0; i < modes.size(); i++) {
      if (tile_set.count(modes[i])) continue;
      om.push_back(modes[i]);
      oe.push_back(extents[i]);
      os.push_back(pstride.at(modes[i]));
    }
  };
  std::vector<int32_t> bma, bmb, bmc;
  std::vector<int64_t> bea, beb, bec, bsa, bsb, bsc;
  drop_tiles(modes_a, extents_a, sa, bma, bea, bsa);
  drop_tiles(modes_b, extents_b, sb, bmb, beb, bsb);
  drop_tiles(modes_c, extents_c, sc, bmc, bec, bsc);

  // Total block count.
  uint64_t nblocks = 1;
  for (const auto &tm : tile_modes) nblocks *= static_cast<uint64_t>(tm.extent);

  std::vector<Tile> tiles;
  tiles.reserve(nblocks);

  for (uint64_t idx = 0; idx < nblocks; idx++) {
    // Decompose idx into per-tile-mode values, first tile mode fastest (fixed,
    // deterministic order).
    uint64_t rem = idx;
    int64_t off_a = 0, off_b = 0, off_c = 0;
    for (const auto &tm : tile_modes) {
      int64_t v = static_cast<int64_t>(rem % static_cast<uint64_t>(tm.extent));
      rem /= static_cast<uint64_t>(tm.extent);
      if (tm.in_a) off_a += v * sa[tm.mode];
      if (tm.in_b) off_b += v * sb[tm.mode];
      if (tm.in_c) off_c += v * sc[tm.mode];
    }

    Tile t;
    t.modes_a = bma; t.extents_a = bea; t.strides_a = bsa;
    t.modes_b = bmb; t.extents_b = beb; t.strides_b = bsb;
    t.modes_c = bmc; t.extents_c = bec; t.strides_c = bsc;
    t.off_a = off_a; t.off_b = off_b; t.off_c = off_c;

    // Pad each block to satisfy the m6n6k6 grammar exactly as a normal step
    // does (extent-1 dummies). Done per block so each descriptor is itself
    // grammar-compliant (R1: never hand a block a known-bad shape).
    if (tn_mnk_padding_enabled()) {
      pad_contraction_mnk(t.modes_a, t.extents_a, t.modes_b, t.extents_b,
                          t.modes_c, t.extents_c);
      // Dummies are appended at the end of each mode list and are extent-1, so
      // their stride is never dereferenced; pad all three stride lists to the
      // new rank with value 1 (hipTensor's packed default for an extent-1
      // axis). All three are now non-empty (real strided views), so unlike the
      // old packed path these resizes always run.
      t.strides_a.resize(t.modes_a.size(), 1);
      t.strides_b.resize(t.modes_b.size(), 1);
      t.strides_c.resize(t.modes_c.size(), 1);
    }

    // K-tiled blocks accumulate into the shared C region. When K is tiled,
    // multiple blocks target the SAME off_c, so only the first writer zeroes
    // it; the rest add (beta=1). When K is not tiled, every block has a unique
    // off_c (disjoint M/N region) and writes fresh.
    if (any_k_tiled) {
      // off_c repeats every (#K-combinations) blocks. With tile_modes ordered
      // M,N,K and idx mixed-radix first-fastest, the K values are the highest-
      // order digits, so blocks sharing an off_c are NOT contiguous. Instead
      // detect the first writer by whether ALL K tile-mode values are zero.
      uint64_t r = idx;
      bool first_k = true;
      for (const auto &tm : tile_modes) {
        int64_t v = static_cast<int64_t>(r % static_cast<uint64_t>(tm.extent));
        r /= static_cast<uint64_t>(tm.extent);
        if (tm.is_k && v != 0) { first_k = false; break; }
      }
      t.accumulate = !first_k;       // beta=1 for all but the first K writer
      t.zero_c_first = first_k;      // first K writer zeroes its C region
    } else {
      t.accumulate = false;
      t.zero_c_first = true;         // every block writes a fresh disjoint region
    }

    tiles.push_back(std::move(t));
  }

  return tiles;
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::project_slice(
    uint_t slice_index, int device_idx, std::vector<PlanePtrs> &projected) {
  // Project every input tensor onto one slice (fixed value of every sliced
  // mode) for execution on device_idx. Pure pointer arithmetic — no copies,
  // no data movement (design doc §18.3): for each tensor carrying a sliced
  // mode, both split-complex planes advance by the slice's element offset
  // within the FULL parent tensor. The element offset uses COLUMN-major
  // strides (mode 0 fastest), matching hipTensor's packed-stride default for
  // the unsliced descriptors. The remaining (unsliced) modes are presented to
  // hipTensor as a strided view via PlanSpec::strides_a/b, so the data is
  // read in place inside the parent layout.
  //
  // Plane bookkeeping (resolves OI9): the imaginary plane is always
  // plane_bytes(N_full) from the slot base — N_full, not the sliced count —
  // which is why the projection produces explicit re/im pairs instead of a
  // single slot pointer.
  const size_t num_inputs = network_desc_.tensors.size();
  // Fill the input prefix; intermediates (slots after num_inputs) belong to
  // the caller. Grow if needed, never shrink.
  if (projected.size() < num_inputs)
    projected.resize(num_inputs, PlanePtrs{nullptr, nullptr});

  // Inputs are replicated per device as a slab copy: rebase each primary
  // pointer onto this device's slab before applying slice offsets.
  GPUDevice<data_t> &dev = gpu_mgr_.device(device_idx);
  const char *primary_base =
      static_cast<const char *>(gpu_mgr_.primary().tensor_data_ptr());
  char *device_base = static_cast<char *>(dev.tensor_data_ptr());

  // slice_index → per-mode values, mixed radix, last sliced mode fastest.
  // The decomposition order is fixed by plan_.sliced and identical on every
  // rank and device, preserving deterministic accumulation (P4).
  std::vector<int64_t> slice_values(plan_.sliced.size());
  uint_t remaining = slice_index;
  for (int i = static_cast<int>(plan_.sliced.size()) - 1; i >= 0; i--) {
    slice_values[i] = remaining % plan_.sliced[i].extent;
    remaining /= plan_.sliced[i].extent;
  }

  std::map<int32_t, int64_t> slice_map;
  for (size_t i = 0; i < plan_.sliced.size(); i++)
    slice_map[plan_.sliced[i].mode] = slice_values[i];

  for (size_t t = 0; t < num_inputs; t++) {
    const TensorSpec &full = network_desc_.tensors[t];

    char *slot = device_base +
                 (static_cast<const char *>(tensor_device_ptrs_[t]) -
                  primary_base);

    int64_t offset_elements = 0;
    if (!plan_.sliced.empty()) {
      int64_t stride = 1; // column-major: mode 0 fastest
      for (size_t m = 0; m < full.modes.size(); m++) {
        auto it = slice_map.find(full.modes[m]);
        if (it != slice_map.end())
          offset_elements += it->second * stride;
        stride *= full.extents[m];
      }
    }

    const size_t plane = plane_bytes(full.num_elements(), sizeof(data_t));
    const size_t shift = static_cast<size_t>(offset_elements) * sizeof(data_t);
    projected[t].re = reinterpret_cast<data_t *>(slot + shift);
    projected[t].im = reinterpret_cast<data_t *>(slot + plane + shift);
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

  std::vector<std::complex<data_t>> acc(out_size_);
  gpu_mgr_.primary().get_output(acc);

  // Deterministic binary-tree reduction by rank index (P4): at stride s,
  // rank r receives from rank r+s and adds in fixed element order. The
  // pairing depends only on rank indices and rank count, never on timing,
  // so the floating-point sum is bit-identical run to run. (MPI_Allreduce
  // makes no such ordering guarantee.) Slice→rank assignment is contiguous
  // by index, so the combined order across all slices is the global index
  // order regardless of which ranks executed which slices.
  std::vector<std::complex<data_t>> incoming(out_size_);
  MPI_Datatype mpi_type = (sizeof(data_t) == 8) ? MPI_DOUBLE : MPI_FLOAT;
  for (int stride = 1; stride < nprocs_; stride *= 2) {
    if (myrank_ % (2 * stride) == 0) {
      int src = myrank_ + stride;
      if (src < nprocs_) {
        MPI_Recv(incoming.data(), static_cast<int>(out_size_) * 2, mpi_type,
                 src, stride, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (size_t i = 0; i < out_size_; i++)
          acc[i] += incoming[i];
      }
    } else if (myrank_ % (2 * stride) == stride) {
      MPI_Send(acc.data(), static_cast<int>(out_size_) * 2, mpi_type,
               myrank_ - stride, stride, MPI_COMM_WORLD);
      break; // this rank's partial has been handed off
    }
  }

  // Aer expects the full result on every rank.
  MPI_Bcast(acc.data(), static_cast<int>(out_size_) * 2, mpi_type, 0,
            MPI_COMM_WORLD);

  hipSetDevice(gpu_mgr_.primary().device_id());
  hipMemcpyAsync(
      thrust::raw_pointer_cast(gpu_mgr_.primary().output_buffer().data()),
      acc.data(), out_size_ * sizeof(std::complex<data_t>),
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
