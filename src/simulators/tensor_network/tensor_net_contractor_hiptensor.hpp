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

// aer-0059: since aer-0057 removed the hipTensor dispatch, hipBLAS is not an
// accelerator for this backend -- it IS the backend. A configure that finds
// hipTensor but not hipBLAS would previously fall back to
// hiptensorContraction; now it would compile cleanly and then refuse every
// contraction step at runtime. Make the misconfiguration fail at build time,
// where it is one CMake line to fix, not in a batch job.
#if defined(AER_HIPTENSOR) && !defined(AER_HIPBLAS)
#error "qiskit-aer tensor-network (ROCm): AER_HIPBLAS is required since the hipTensor dispatch was removed (aer-0057). hipBLAS was not found or not enabled -- fix the hipBLAS discovery in CMake; there is no fallback engine."
#endif

#ifdef AER_THRUST_ROCM

#include <atomic>
#include <chrono>
#include <algorithm>
#include <climits>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>

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

// aer-0024: keep the prebuilt plan cache and memory pool across a
// topology-matching set_network() / set_additional_tensors(). Default ON.
// Set AER_TN_POOL_REUSE=0 to restore the pre-aer-0024 behaviour, in which
// every such call invalidated pool_ready_ and re-ran the whole prebuild pass.
// The knob exists so both arms can be measured in ONE job: prebuild count,
// pool allocations and tiled-plan-creation exposure are all directly
// comparable without a rebuild between arms.
static bool tn_pool_reuse() {
  static bool checked = false;
  static bool enabled = true;
  if (!checked) {
    const char *val = std::getenv("AER_TN_POOL_REUSE");
    enabled = !(val != nullptr && std::string(val) == "0");
    checked = true;
  }
  return enabled;
}

// Opt-in wall-clock profiling. Inert unless AER_TN_PROFILE=1: when off, no
// timers are read and no GPU stream syncs are added, so the default execution
// path is byte-for-byte unchanged in both result and performance. When on, each
// reduction entrypoint prints a one-line per-rank phase breakdown to stderr.
static bool tn_profile() {
  static bool checked = false;
  static bool enabled = false;
  if (!checked) {
    const char *val = std::getenv("AER_TN_PROFILE");
    enabled = (val != nullptr && std::string(val) == "1");
    checked = true;
  }
  return enabled;
}

// Milliseconds elapsed since a captured steady_clock time point.
static inline double
tn_ms_since(const std::chrono::steady_clock::time_point &t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
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
// aer-0040: is this step already a matrix product?
//
// A packed tensor's element index is sum_i idx_i * prod_{j<i} ext_j -- stride 1
// on modes[0], multiplying forward (stride_map below). So a packed tensor whose
// declared order is [free modes][contracted modes] IS a column-major matrix of
// (prod free) rows by (prod contracted) columns with leading dimension equal to
// the row count, and one whose order is [contracted][free] is that matrix
// transposed. hipblas's HIPBLAS_OP_N / HIPBLAS_OP_T take either. Any other
// order interleaves free and contracted axes and would need a physical permute,
// which this route does not do: it declines the step instead.
//
// run_split() returns how many maximal runs of like-classified modes the order
// contains, and the operand is admissible exactly when that is at most two.
// It also reports which class comes first, which is what selects the transpose.
//
// Extent-1 modes are ignored. pad_contraction_mnk appends them only when a
// category is empty, they contribute a factor of one to every product, and
// counting them would reject an otherwise perfectly contiguous operand.
inline bool gemm_operand_layout(const std::vector<int32_t> &modes,
                                const std::vector<int64_t> &extents,
                                const std::set<int32_t> &contracted,
                                bool &transposed, int64_t &free_extent,
                                int64_t &contracted_extent) {
  free_extent = 1;
  contracted_extent = 1;
  int runs = 0;
  int prev = -1;
  bool first_is_contracted = false;
  for (size_t i = 0; i < modes.size(); i++) {
    if (extents[i] <= 1)
      continue;
    const int cur = contracted.count(modes[i]) ? 1 : 0;
    if (cur)
      contracted_extent *= extents[i];
    else
      free_extent *= extents[i];
    if (cur != prev) {
      if (runs == 0)
        first_is_contracted = (cur == 1);
      runs++;
      prev = cur;
    }
  }
  transposed = first_is_contracted;
  return runs <= 2;
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

// --- WS-3 operand staging: strided <-> packed device copies -----------------
// When a tiled sub-block's descriptor stride for an operand reaches the CK-safe
// ceiling (max_tiled_elements), hipTensor faults while BUILDING the plan on that
// strided view (observed as a GPU memory access fault during prebuild; sweep
// 19751464). Staging routes around it: pack the strided operand into a packed
// scratch buffer, contract on the packed buffer (small strides CK can build),
// and for the output scatter the packed result back into the strided parent.
// This is value-identical to the direct strided path -- only the memory layout
// differs -- proved bit-exact on CPU in prove_operand_staging.py.
static constexpr int kStageMaxRank = 32;

template <typename data_t> struct strided_pack_func_hip {
  const data_t *src; // parent plane, already base-shifted to the tile origin
  data_t *dst;       // packed scratch plane (contiguous, column-major)
  int rank;
  int64_t ext[kStageMaxRank]; // free-mode extents (packed/descriptor order)
  int64_t str[kStageMaxRank]; // parent strides of those free modes
  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t rem = i;
    int64_t off = 0;
    for (int k = 0; k < rank; ++k) {
      uint_t e = static_cast<uint_t>(ext[k]);
      off += static_cast<int64_t>(rem % e) * str[k];
      rem /= e;
    }
    dst[i] = src[off];
  }
};

template <typename data_t> struct strided_scatter_func_hip {
  const data_t *src; // packed scratch plane
  data_t *dst;       // parent plane, already base-shifted to the tile origin
  bool accumulate;   // add into the parent (K-partials) vs overwrite
  int rank;
  int64_t ext[kStageMaxRank];
  int64_t str[kStageMaxRank];
  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t rem = i;
    int64_t off = 0;
    for (int k = 0; k < rank; ++k) {
      uint_t e = static_cast<uint_t>(ext[k]);
      off += static_cast<int64_t>(rem % e) * str[k];
      rem /= e;
    }
    if (accumulate)
      dst[off] += src[i];
    else
      dst[off] = src[i];
  }
};

// Number of elements in a tile sub-block (product of its free-mode extents).
static uint64_t stage_block_elems(const std::vector<int64_t> &ext) {
  uint64_t n = 1;
  for (int64_t e : ext)
    n *= (e > 0 ? static_cast<uint64_t>(e) : 1);
  return n;
}

// --- aer-0055: interleaved-complex staging for the native-ZGEMM sub-path ---
// The gather below is stage_pack's index math with an interleaved destination:
// element i of the (possibly permuted) source order lands at dst[2i] (real)
// and dst[2i+1] (imag), which is exactly hipblasDoubleComplex/hipblasComplex
// memory layout. One pass fuses the permute and the interleave.
template <typename data_t> struct interleave_pack_func_hip {
  const data_t *src_re;
  const data_t *src_im;
  data_t *dst; // interleaved complex scratch
  int rank;
  int64_t ext[kStageMaxRank];
  int64_t str[kStageMaxRank];
  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t rem = i;
    int64_t off = 0;
    for (int k = 0; k < rank; ++k) {
      uint_t e = static_cast<uint_t>(ext[k]);
      off += static_cast<int64_t>(rem % e) * str[k];
      rem /= e;
    }
    dst[2 * i] = src_re[off];
    dst[2 * i + 1] = src_im[off];
  }
};

// The scatter twin: interleaved source element i goes to the strided offset
// in each destination plane. With ext={n}, str={1} it is a plain
// deinterleave; with the corder scat vectors it fuses the deinterleave and
// the declared-order scatter into one pass.
template <typename data_t> struct deinterleave_scatter_func_hip {
  const data_t *src; // interleaved complex scratch
  data_t *dst_re;
  data_t *dst_im;
  int rank;
  int64_t ext[kStageMaxRank];
  int64_t str[kStageMaxRank];
  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t rem = i;
    int64_t off = 0;
    for (int k = 0; k < rank; ++k) {
      uint_t e = static_cast<uint_t>(ext[k]);
      off += static_cast<int64_t>(rem % e) * str[k];
      rem /= e;
    }
    dst_re[off] = src[2 * i];
    dst_im[off] = src[2 * i + 1];
  }
};

template <typename data_t>
static void interleave_pack(hipStream_t stream, const data_t *src_re,
                            const data_t *src_im, data_t *dst,
                            const std::vector<int64_t> &extents,
                            const std::vector<int64_t> &strides) {
  const uint64_t n = stage_block_elems(extents);
  interleave_pack_func_hip<data_t> f;
  const int rank = static_cast<int>(extents.size());
  f.rank = rank < kStageMaxRank ? rank : kStageMaxRank;
  for (int k = 0; k < f.rank; ++k) {
    f.ext[k] = extents[k];
    f.str[k] = strides[k];
  }
  f.src_re = src_re;
  f.src_im = src_im;
  f.dst = dst;
  auto ci = thrust::counting_iterator<uint_t>(0);
  thrust::for_each_n(thrust_gpu::par.on(stream), ci, n, f);
}

template <typename data_t>
static void deinterleave_scatter(hipStream_t stream, const data_t *src,
                                 data_t *dst_re, data_t *dst_im,
                                 const std::vector<int64_t> &extents,
                                 const std::vector<int64_t> &strides) {
  const uint64_t n = stage_block_elems(extents);
  deinterleave_scatter_func_hip<data_t> f;
  const int rank = static_cast<int>(extents.size());
  f.rank = rank < kStageMaxRank ? rank : kStageMaxRank;
  for (int k = 0; k < f.rank; ++k) {
    f.ext[k] = extents[k];
    f.str[k] = strides[k];
  }
  f.src = src;
  f.dst_re = dst_re;
  f.dst_im = dst_im;
  auto ci = thrust::counting_iterator<uint_t>(0);
  thrust::for_each_n(thrust_gpu::par.on(stream), ci, n, f);
}

// Pack a strided parent operand (both split-complex planes) into packed
// scratch, on the given stream. src planes are already base-shifted to the tile
// origin; parent_strides index the parent's packed layout.
template <typename data_t>
static void stage_pack(hipStream_t stream, const data_t *src_re,
                       const data_t *src_im, data_t *dst_re, data_t *dst_im,
                       const std::vector<int64_t> &extents,
                       const std::vector<int64_t> &parent_strides) {
  const uint64_t n = stage_block_elems(extents);
  strided_pack_func_hip<data_t> f;
  const int rank = static_cast<int>(extents.size());
  f.rank = rank < kStageMaxRank ? rank : kStageMaxRank;
  for (int k = 0; k < f.rank; ++k) {
    f.ext[k] = extents[k];
    f.str[k] = parent_strides[k];
  }
  auto ci = thrust::counting_iterator<uint_t>(0);
  f.src = src_re;
  f.dst = dst_re;
  thrust::for_each_n(thrust_gpu::par.on(stream), ci, n, f);
  f.src = src_im;
  f.dst = dst_im;
  thrust::for_each_n(thrust_gpu::par.on(stream), ci, n, f);
}

// Scatter a packed scratch result (both planes) back into the strided parent.
template <typename data_t>
static void stage_scatter(hipStream_t stream, const data_t *src_re,
                          const data_t *src_im, data_t *dst_re, data_t *dst_im,
                          const std::vector<int64_t> &extents,
                          const std::vector<int64_t> &parent_strides,
                          bool accumulate) {
  const uint64_t n = stage_block_elems(extents);
  strided_scatter_func_hip<data_t> f;
  f.accumulate = accumulate;
  const int rank = static_cast<int>(extents.size());
  f.rank = rank < kStageMaxRank ? rank : kStageMaxRank;
  for (int k = 0; k < f.rank; ++k) {
    f.ext[k] = extents[k];
    f.str[k] = parent_strides[k];
  }
  auto ci = thrust::counting_iterator<uint_t>(0);
  f.src = src_re;
  f.dst = dst_re;
  thrust::for_each_n(thrust_gpu::par.on(stream), ci, n, f);
  f.src = src_im;
  f.dst = dst_im;
  thrust::for_each_n(thrust_gpu::par.on(stream), ci, n, f);
}

//=============================================================================
// TensorNetContractor_HipTensor
//=============================================================================

template <typename data_t = double>
class TensorNetContractor_HipTensor : public TensorNetContractor<data_t> {
  GPUResourceManager<data_t> gpu_mgr_;
  int num_devices_used_;

  // Wall-clock phase accumulators (milliseconds) for the current contraction,
  // populated only when tn_profile() is on. setup/path are filled by
  // setup_contraction(); contract/reduce_* by the reduction entrypoints; the
  // per-rank breakdown is emitted by prof_report(). Zero effect when profiling
  // is off (guarded at every site).
  double prof_setup_ms_ = 0.0;
  double prof_path_ms_ = 0.0;
  double prof_contract_ms_ = 0.0;
  double prof_reduce_gpu_ms_ = 0.0;
  double prof_reduce_mpi_ms_ = 0.0;

  // aer-0047: the (t_setup_ms - t_path_ms) remainder, itemised. It is 41-57
  // percent of runtime at p=2 and had no counter of any kind. All three are
  // charged INSIDE the same setup bracket as prof_setup_ms_ (setup_contraction
  // opens it at prof_setup_t0 and closes it at the += below), so
  //   prof_setup_ms_ - prof_path_ms_ - prof_specs_ms_ - prof_prebuild_ms_
  //     - prof_pool_ms_
  // is the residual (windows 1/2/3a, the per-device copy loop). Every one of
  // these phases is host-side and synchronous -- hipTensor plan get_or_create
  // builds descriptors and queries workspace with no kernel launch, and
  // pool().allocate() is a blocking hipMalloc -- so they are timed with plain
  // steady_clock and NOT prof_sync_devices(), exactly as prof_setup_ms_ is.
  // Adding a sync here would charge unrelated in-flight work to setup.
  double prof_specs_ms_ = 0.0;      // build_sliced_specs()
  double prof_prebuild_ms_ = 0.0;   // per-step descriptor walk + plan get_or_create
  double prof_pool_ms_ = 0.0;       // pool plan_layout + hipMalloc

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
  // aer-0040: how the GEMM route classified every step of the last plan. Reset
  // in setup_pool_and_cache, printed by prof_report and by the verbose setup
  // line, and the reason no separate job is needed to count how often a step's
  // operands are already matrix-shaped.
  uint64_t gemm_steps_routed_ = 0;
  uint64_t gemm_steps_declined_tiled_ = 0;
  uint64_t gemm_steps_declined_strided_ = 0;
  uint64_t gemm_steps_declined_layout_ = 0;
  // aer-0044: declined because C's declared mode order is not
  // [M modes in A's order][N modes in B's order]. Only the final step can trip
  // this, since it is the only one that rewrites modes_c into modes_out_ order.
  // Replaces aer-0040's blanket exclusion of the final step.
  uint64_t gemm_steps_declined_corder_ = 0;
  // aer-0045: declined because the contracted modes appear in a different order
  // on A than on B, which a single GEMM k index cannot express and a transpose
  // cannot repair.
  uint64_t gemm_steps_declined_korder_ = 0;
  // aer-0052: layout/korder declines rescued by the permute layer. Counted
  // separately from gemm_steps_routed_ because a permuted dispatch pays an
  // extra pack per operand per slice and the two populations must be
  // distinguishable in any timing read.
  uint64_t gemm_steps_permuted_ = 0;
  // aer-0046: plans NOT built in the setup prebuild because the route will take
  // the step. This is the count that should track the setup saving.
  uint64_t gemm_plans_skipped_ = 0;
  // aer-0042: per-phase attribution for the GEMM route. Without it the contract
  // phase lumps routed and declined steps together and a timing comparison
  // cannot say which produced the difference.
  //
  // WHAT prof_gemm_ms_ MEASURES, PRECISELY: HOST-SIDE ENQUEUE COST, not
  // execution time. There is no synchronise at the call site. t_contract_ms is
  // NOT accumulated the same way -- prof_sync_devices() runs a
  // hipStreamSynchronize on every device stream at the end of the phase, so
  // t_contract_ms includes kernel EXECUTION while t_gemm_ms does not. The two
  // are therefore not the same kind of quantity, and t_gemm_ms / t_contract_ms
  // is NOT the fraction of contraction time the route owns.
  //
  // Enqueue cost is nonetheless the quantity this pivot is about: job 20644277
  // measured hipTensor at 77.3 us per call against hipblasZgemm's ~7.4 us
  // dispatch floor, and dispatch is what dominates at the ~10^3 FLOP steps this
  // backend produces. A synchronise here would make t_gemm_ms an execution
  // time, but it would also serialise the routed path against a hipTensor path
  // that is not serialised, and the comparison would be worthless.
  //
  // The measurement that includes execution is t_contract_ms with the route off
  // against t_contract_ms with the route on, same workload, same plan. That is
  // the number to read; t_gemm_ms says how much of the difference is dispatch.
  //
  // TWO THINGS THESE COUNTS ARE NOT.
  //  (1) They are DISPATCHES, not steps. contract_single_slice runs once per
  //      slice, so a plan with S slices reports steps x slices_local, which is
  //      the same "call" unit the 77.3 us per-call regression uses. The census
  //      line [AER_TN_GEMM] is the per-STEP count; these are not equal and are
  //      not meant to be.
  //  (2) Under AER_TN_GEMM_VERIFY the hiptensorContraction run that produces
  //      the reference is NOT counted in ht_calls and its time is NOT in
  //      t_gemm_ms, though it IS in t_contract_ms. So with verification on,
  //      gemm_calls + ht_calls under-counts the dispatches actually issued and
  //      t_gemm_ms understates the fraction of t_contract_ms the route owns.
  //      Do not read a timing arm that has verification enabled.
  double prof_gemm_ms_ = 0.0;
  uint64_t prof_gemm_calls_ = 0;
  uint64_t prof_ht_calls_ = 0;

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

    // aer-0040: the GEMM route. gemm_ok is set only when this step is a matrix
    // product already -- see gemm_eligible() for the three conditions and why
    // each is necessary. When it is false the step runs the unchanged
    // hiptensorContraction path, so an ineligible step costs nothing.
    bool gemm_ok = false;
    bool gemm_a_trans = false, gemm_b_trans = false;
    int64_t gemm_m = 0, gemm_n = 0, gemm_k = 0;

    // aer-0052: the permute layer. When gemm_perm_a/b is set, the operand is
    // packed through stage_pack into GEMM scratch under the permuted mode
    // order [free modes, ORIGINAL relative order][contracted modes, canonical
    // order := A's contracted-mode appearance order] (extent-1 modes at the
    // tail; they move no data). perm_ext_X is the destination-order extents
    // and perm_str_X the SOURCE column-major strides reordered by the same
    // permutation -- exactly the (extents, parent_strides) pair stage_pack
    // consumes. Preserving the free modes' relative order is the load-bearing
    // correctness decision: C is written [M in A's original free order][N in
    // B's original free order], which is byte-identical to the slot
    // compute_contraction_result declares, so the aer-0044/0045 wrong-slot
    // class is closed by construction. Mapping audited standalone BEFORE this
    // code was written (audit_aer0052_permute_design.py: 3,288 exhaustive +
    // 111,941 randomised cases, zero mismatches).
    bool gemm_perm_a = false, gemm_perm_b = false;
    std::vector<int64_t> perm_ext_a, perm_str_a, perm_ext_b, perm_str_b;
    // aer-0054: corder handling. When gemm_scatter_c is set the GEMM writes
    // its natural [M in A's free order][N in B's free order] result into
    // scratch, and stage_scatter places it into the DECLARED C slot:
    // scat_ext is the natural-order extent list (extent>1 free modes) and
    // scat_str the declared-C column-major strides reordered to that natural
    // order. Audited in audit_aer0054_design.py CLAIM C, including every
    // declared order of a 3-mode output exhaustively.
    bool gemm_scatter_c = false;
    std::vector<int64_t> scat_ext, scat_str;

    // WS-3 mode tiling. Empty `tiles` => in-envelope step, executed exactly as
    // before via the descriptor above (zero behavior change). Non-empty =>
    // this logical step is oversized (M>6, N>6, or K>6) and is executed as a
    // loop of these kernel-sized sub-contractions; each Tile is itself
    // m6n6k6-grammar-compliant. K-block tiles set accumulate=true so their
    // partial products sum into the same C sub-region; M/N-block tiles write
    // disjoint C sub-regions (accumulate=false).
    // aer-0060: the Tile machinery is deleted; every step runs whole on the
    // GEMM route.
  };
  std::vector<PlanSpec> step_plan_specs_;

  // WS-3 operand-staging scratch. Sized in the prebuild pass to the largest
  // staged sub-block; 6 planes per device (re+im for A, B, C) laid out
  // contiguously. Zero-cost when no step needs staging (stays unallocated).
  uint64_t stage_scratch_elems_ = 0;
  // aer-0033: sub-blocks materialised so far in the CURRENT
  // setup_pool_and_cache pass. Reset at the top of that function, because
  // each pass rebuilds step_plan_specs_ from scratch (including the extra
  // per-device passes), so the count is per pass and not cumulative.
  uint64_t tiles_built_ = 0;
  std::unordered_map<int, thrust::device_vector<data_t>> stage_scratch_;

  // aer-0052: GEMM permute scratch. Sized in the prebuild pass to the largest
  // permuted operand; 4 planes per device (re+im for A and B; C never
  // permutes -- the free-order-preserving target writes C straight into its
  // declared slot). Zero-cost when no step permutes (stays unallocated).
  uint64_t gemm_perm_scratch_elems_ = 0;
  std::unordered_map<int, thrust::device_vector<data_t>> gemm_perm_scratch_;

  std::vector<std::vector<int32_t>> prev_modes_;
  std::vector<std::vector<int64_t>> prev_extents_;
  // aer-0024: the OUTPUT spec the cached plan and the cached step_plan_specs_
  // were built for. setup_pool_and_cache() reorders the final step's C
  // descriptor into modes_out_ order (see the `step == num_steps - 1` branch),
  // so the prebuilt specs are only valid for the output they were built with.
  // Before aer-0024 that was masked: every set_additional_tensors() cleared
  // pool_ready_ and the specs were rebuilt from the current modes_out_. Now
  // that they persist, the reuse predicate must cover the output too, or a
  // caller that varied modes_out_ while holding the inputs fixed would contract
  // with a stale C mode order and return a silently wrong answer.
  std::vector<int32_t> prev_modes_out_;
  std::vector<int64_t> prev_extents_out_;
  bool prev_valid_;
  // Tiling-engaged state the cached plan was built with. A reused plan must be
  // set up with the same engaged value; otherwise setup_pool_and_cache re-trips
  // the m6n6k6 gate (throwing NeedsTilingException) on a plan that needed
  // tiling, forcing a wasteful re-plan and defeating the cache.
  bool prev_engaged_;

  // aer-0049: when true, AER_TN_PLAN_FILE is neither read nor written by this
  // contractor. Set through set_plan_file_bypass() by the aer-0025 terminal
  // re-plan so the last-ditch recovery searches fresh instead of replaying
  // the pinned plan. The expval guard's retry decision is collective (the
  // guarded value is post-Allreduce), so under MPI this flag is set on every
  // rank together and the load/skip decision stays uniform.
  bool plan_file_bypass_;

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
  void set_plan_file_bypass(bool bypass) override { plan_file_bypass_ = bypass; }

private:
  void build_network_description();
  void build_sliced_specs();
  void setup_pool_and_cache(int device_idx, bool tiling_available);
  bool topology_matches_previous() const;
  void cache_topology();
  void contract_single_slice(uint_t slice_index, int device_idx);
  void project_slice(uint_t slice_index, int device_idx,
                     std::vector<PlanePtrs> &projected);
  void accumulate_across_gpus();
  void accumulate_across_mpi();
#ifdef AER_MPI
  // Collective fail-together for the reduction-bearing contraction
  // entrypoints. A per-rank throw during local contraction (a HIP/CK fault or
  // a slicing-bookkeeping guard on THIS rank's slice range) must not leave the
  // other ranks blocked at the cross-rank reduction below. Each caller runs its
  // local work in a try/catch and reports a 0/1 flag here; Allreduce(MAX)
  // agrees a global verdict, and if any rank failed, every rank throws together
  // BEFORE the reduction collective is entered. Mirrors the planner's
  // MPIParallelPathOptimizer collective-agreement pattern. No-op single-rank.
  void agree_or_fail_together(int local_failed, const std::string &local_msg);
  bool agree_plan_cache_hit(bool local_hit);
#endif
  // aer-0037. Declared OUTSIDE the AER_MPI guard above, matching its definition:
  // it is written to work single-rank as well, where it degenerates to
  // rethrow-locally, so setup_contraction keeps one code path instead of two.
  // Both call sites are likewise unguarded. If this were inside the guard, a
  // build without AER_MPI would fail to compile -- the definition would name a
  // non-member and the calls an undeclared function.
  int agree_setup_status(int local_status, const std::string &local_msg);
  // aer-0049. Both declared outside any AER_MPI guard: single-rank they
  // degenerate to a plain local file read/write, and try_load_plan_file's
  // broadcast is itself guarded internally. try_load_plan_file must only be
  // called under rank-uniform conditions (it runs an MPI_Bcast when
  // nprocs_ > 1); maybe_write_plan_file is rank-0-only and collective-free.
  bool try_load_plan_file(bool engaged);
  void maybe_write_plan_file(bool engaged);
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

  // Profiling helpers (tn_profile()). prof_sync_devices() blocks until every
  // used device stream is idle so async kernel time is attributed to the phase
  // it belongs to rather than leaking into the next wall-clock measurement.
  // prof_report() prints a machine-parseable per-rank breakdown to stderr.
  void prof_sync_devices();
  void prof_report(const char *entrypoint);
};

//=============================================================================
// Implementation
//=============================================================================

template <typename data_t>
TensorNetContractor_HipTensor<data_t>::TensorNetContractor_HipTensor()
    : num_devices_used_(1), add_sp_tensors_(true), num_base_tensors_(0),
      num_additional_tensors_(0), out_size_(0), plan_valid_(false),
      slice_begin_(0), slice_end_(0), nprocs_(1), myrank_(0),
      pool_ready_(false), prev_valid_(false), prev_engaged_(false),
      plan_file_bypass_(false) {}

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
  // aer-0024: do NOT invalidate the pool here. The pool holds only
  // plan-derived intermediates and the shape-keyed hipTensor plan cache;
  // neither depends on tensor VALUES, and input data is addressed through
  // tensor_device_ptrs_, which was just refreshed above and is re-read on
  // every contraction. A topology or output-spec change is caught by
  // topology_matches_previous() in setup_contraction(), whose cache-miss
  // branch clears pool_ready_ itself. Keeping it set turns get_amplitudes()'
  // N-amplitude loop from N full prebuild passes into 1.
  if (!tn_pool_reuse())
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
  // aer-0024: same reasoning as set_network(). The Pauli leaves of a
  // save_expval batch differ only in their VALUES; their modes and extents are
  // identical across I/X/Y/Z, so the plan, the step specs and the pool stay
  // valid. This is the site that made aer-0022's reuse partial: the path
  // search was skipped but the ~101-step prebuild still ran once per term,
  // which is what multiplied tiled-plan-creation fault exposure by M.
  if (!tn_pool_reuse())
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
void TensorNetContractor_HipTensor<data_t>::setup_contraction(
    bool use_autotune) {
#ifdef AER_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs_);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank_);
#endif

  // aer-0037 WINDOW 0: everything before the retry loop. The network_desc_
  // assignments allocate, and none of it was reachable by any agreement -- a
  // throw here left the siblings blocked at window 1's collective below.
  //
  // The two network_desc_ assignments moved ABOVE the profiling reset so they
  // sit inside this guard. They are two vector copies of a handful of ints; the
  // sub-microsecond they no longer contribute to t_setup_ms is far below the
  // measurement's resolution.
  {
    int w0_status = 0;
    std::string w0_msg;
    try {
      // aer-0060: the hipTensor algorithm selector is deleted with its
      // engine; the autotune flag has nothing to latch. Accepted and ignored
      // so the shared driver interface is unchanged.
      (void)use_autotune;
      network_desc_.output_modes = modes_out_;
      network_desc_.output_extents = extents_out_;
    } catch (const std::exception &e) {
      w0_status = 2;
      w0_msg = e.what();
    } catch (...) {
      w0_status = 2;
      w0_msg =
          "[AER_TN] setup raised an exception of unknown type; caught so the "
          "ranks fail together rather than one rank unwinding alone.";
    }
    // ONE agreement for the whole pre-loop region rather than two. Both halves
    // sit above the profiling reset, so neither this collective nor the work it
    // covers is charged to t_setup_ms -- the reason the trim was placed there in
    // the first place. Merging matters because this region is on the WARM path
    // too: a reused topology skips windows 1 and 2 entirely, so its whole MPI
    // cost is this agreement plus the end-of-pass one, two collectives against a
    // 5.25 ms warm setup. Single-rank pays nothing at all: agree_setup_status
    // makes no MPI call when nprocs_ <= 1.
    (void)agree_setup_status(w0_status, w0_msg);
  }

  // setup_contraction is the first call of the setup -> contract sequence, so
  // it owns resetting the profiling accumulators for this contraction. The
  // per-rank report is emitted later, by the reduction entrypoint.
  std::chrono::steady_clock::time_point prof_setup_t0;
  if (tn_profile()) {
    prof_setup_ms_ = 0.0;
    prof_path_ms_ = 0.0;
    prof_contract_ms_ = 0.0;
    prof_gemm_ms_ = 0.0;
    prof_gemm_calls_ = 0;
    prof_ht_calls_ = 0;
    prof_reduce_gpu_ms_ = 0.0;
    prof_reduce_mpi_ms_ = 0.0;
    prof_specs_ms_ = 0.0;
    prof_prebuild_ms_ = 0.0;
    prof_pool_ms_ = 0.0;
    prof_setup_t0 = std::chrono::steady_clock::now();
  }

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

  // aer-0030: if the m6n6k6 gate has already fired anywhere in this process,
  // do not pay a search to rediscover it. An intermediate-only oversized step
  // is invisible until a plan exists, so the FIRST contraction still pays the
  // discovery; every later one starts where that one ended. Set in the catch
  // below; see tn_tiling_latch_set() for why this is MPI-safe.
  if (tiling_mode == TilingMode::Auto && tn_tiling_latched())
    engaged = true;

  bool retried = false;
  while (true) {
    try {
      // aer-0037: one status for every throw site AFTER the last collective of
      // this pass. 0 = ok, 1 = this rank needs tiling, 2 = hard failure. The
      // agreement at the end of the pass takes MPI_MAX, so a hard failure on
      // any rank outranks a tiling retry on another -- masking a real failure
      // behind a retry would be worse than either.
      int setup_status = 0;
      std::string setup_msg;
      std::exception_ptr setup_tiling;
      if (prev_valid_ && plan_valid_ && topology_matches_previous()) {
        // Restore the tiling state this plan was planned/set-up with. engaged is
        // re-initialized to false each setup (AUTO + scalar output), so without
        // this a tiled plan would re-trip the m6n6k6 gate below and force a
        // needless re-plan -- the reuse must skip the search AND the re-setup.
        engaged = prev_engaged_;
        if (tn_verbose() && myrank_ == 0)
          fprintf(stderr, "[AER_TN] reusing previous contraction path\n");
      } else {
        // aer-0037 WINDOW 1. Everything from here to the budget Allreduce is
        // local and hardware-dependent: refresh_free_memory() throws through
        // check_hip(hipMemGetInfo), which depends on THIS rank's device, so
        // it is not uniform across ranks. A throw here escapes
        // setup_contraction -- there is no try/catch anywhere in
        // tensor_net.hpp or tensor_net_state.hpp, and agree_or_fail_together
        // wraps only contract_all/accumulate_across_gpus inside the contract
        // entrypoints -- while every sibling walks into the budget Allreduce
        // below and blocks forever. Catch, agree, and fail together instead.
        size_t free_bytes = 0;
        uint64_t workspace_estimate = 0;
        {
          int w1_status = 0;
          std::string w1_msg;
          try {
            gpu_mgr_.primary().refresh_free_memory();
            free_bytes = gpu_mgr_.primary().free_memory();

          } catch (const std::exception &e) {
            w1_status = 2;
            w1_msg = e.what();
          } catch (...) {
            w1_status = 2;
            w1_msg =
                "[AER_TN] setup raised an exception of unknown type; caught so the "
                "ranks fail together rather than one rank unwinding alone.";
          }
          (void)agree_setup_status(w1_status, w1_msg);
        }

        uint64_t memory_budget = free_bytes;
        if (workspace_estimate < free_bytes)
          memory_budget = free_bytes - workspace_estimate;

#ifdef AER_MPI
        // aer-0032: agree the budget across ranks BEFORE anything consumes it.
        // It feeds two quantities that must be identical on every rank -- the
        // planner's per-slice target_size and the plan-cache key computed by
        // plan_key_target_elements() -- and free VRAM is a per-rank number that
        // jitters independently. Job 20516615 printed five distinct budgets
        // (65133.0, 65133.5, 65137.5, 65244.5 and 65276.0 MB) across the
        // contractions of a single job on a single device.
        //
        // MPI_MIN is the conservative direction: the agreed budget fits every
        // rank by construction. This sits in the same branch as find_path's own
        // allreduce and broadcast, so it introduces no divergence risk that is
        // not already present -- whether a rank takes this branch depends only
        // on the driver's call sequence, which is identical on every rank.
        if (nprocs_ > 1) {
          // MPI_UINT64_T rather than MPI_UNSIGNED_LONG_LONG: memory_budget is
          // uint64_t, so this is type-exact and needs no casts, and the C99
          // fixed-width family is already proven present in this build --
          // MPIParallelPathOptimizer broadcasts with MPI_INT64_T. The fork has
          // never used MPI_UNSIGNED_LONG_LONG.
          uint64_t local_budget = memory_budget;
          uint64_t agreed_budget = 0;
          MPI_Allreduce(&local_budget, &agreed_budget, 1, MPI_UINT64_T, MPI_MIN,
                        MPI_COMM_WORLD);
          memory_budget = agreed_budget;
        }
#endif

        if (tn_verbose() && myrank_ == 0)
          fprintf(stderr,
                  "[AER_TN] memory: %.1f MB free, %.1f MB ws, %.1f MB budget\n",
                  free_bytes / (1024.0 * 1024.0),
                  workspace_estimate / (1024.0 * 1024.0),
                  memory_budget / (1024.0 * 1024.0));

        // aer-0027: consult the cross-instruction plan cache before searching.
        //
        // aer-0032 removed the nprocs_ == 1 guard. The hazard that guard
        // avoided is real: under MPI create_optimizer() returns
        // MPIParallelPathOptimizer, whose find_path runs an allreduce and a
        // broadcast, so a rank that hit the cache while a sibling missed would
        // skip collectives the sibling enters and the job would hang. The answer
        // is to make the VERDICT collective rather than to switch the feature
        // off -- see agree_plan_cache_hit(). Leaving it off costs a distributed
        // run a full cotengra path search on every contraction, which is exactly
        // the cost the two caches exist to remove (4595.1 ms warm setup with
        // them off against 5.8 ms with them on -- job 20516615, arm4 vs arm5).
        //
        // put() below needs no equivalent treatment: under MPI plan_ after
        // find_path is the MINLOC-broadcast plan, so every rank stores the same
        // plan. A rank whose key happens to differ simply misses next time and
        // drags the communicator into a search, which is safe.
        //
        // elem_bytes mirrors create_optimizer() exactly (2 * sizeof(data_t)),
        // not sizeof(std::complex<data_t>), so the key cannot drift from the
        // value the optimizer actually clamps against.
        // aer-0037 WINDOW 2. The optimizer is constructed HERE rather than
        // lazily at the search below, so that a failure to construct it is
        // caught inside this guarded region and agreed by the cache verdict's
        // collective. Constructed lazily it sat between two collectives with no
        // agreement point, and a throw there hung every sibling inside
        // find_path's allreduce. Hoisting is free: CotengPathOptimizer's
        // constructor is pure member initialisation -- the cotengra import
        // lives in build_mode_mapping(), which find_path calls -- so building
        // it on a cache hit costs two small heap objects and no Python at all.
        const size_t plan_elem_bytes = 2 * sizeof(data_t);
        std::string plan_key;
        bool plan_from_cache = false;
        std::unique_ptr<PathOptimizer> optimizer;
        int w2_status = 0;
        std::string w2_msg;
        try {
          optimizer = create_optimizer();
        } catch (const std::exception &e) {
          w2_status = 2;
          w2_msg = e.what();
        } catch (...) {
          w2_status = 2;
          w2_msg =
              "[AER_TN] setup raised an exception of unknown type; caught so the "
              "ranks fail together rather than one rank unwinding alone.";
        }
        if (tn_plan_cache_enabled() && w2_status == 0) {
          plan_key = canonical_network_key(
              network_desc_, engaged,
              plan_key_target_elements(memory_budget, plan_elem_bytes),
              tn_path_seed(), plan_elem_bytes);
          if (!plan_key.empty())
            plan_from_cache =
                plan_cache_instance().get(plan_key, network_desc_, plan_);
        }
        // Agreed AFTER the cache block, not inside it: a rank whose optimizer
        // failed to construct must still reach this collective, or the ranks
        // that succeeded would block in it alone. A failed rank reports a miss,
        // which is the safe direction -- everyone searches -- and the hard
        // failure is then agreed immediately below.
#ifdef AER_MPI
        if (nprocs_ > 1)
          plan_from_cache =
              agree_plan_cache_hit(w2_status == 0 && plan_from_cache);
#endif
        (void)agree_setup_status(w2_status, w2_msg);

        // aer-0049: replay a pinned plan from AER_TN_PLAN_FILE, if configured
        // and present, INSTEAD of searching. Placed after the two agreement
        // collectives above so every condition it is gated on is uniform
        // across ranks: plan_from_cache is the collectively-agreed verdict,
        // the env value is identical on every rank, and plan_file_bypass_ is
        // set collectively (see the member). find_path is then skipped on
        // every rank together, so MPIParallelPathOptimizer's collectives are
        // skipped in lockstep. A replayed plan still faces the slice ceiling
        // below and the tile ceiling in the prebuild, so a stale file that
        // violates a ceiling refuses loudly rather than grinding.
        bool plan_from_file = false;
        if (!plan_from_cache)
          plan_from_file = try_load_plan_file(engaged);

        std::chrono::steady_clock::time_point prof_path_t0;
        if (tn_profile())
          prof_path_t0 = std::chrono::steady_clock::now();
        if (!plan_from_cache && !plan_from_file) {
          plan_ = optimizer->find_path(network_desc_, memory_budget,
                                       tn_path_seed(), engaged);
        }
        // The put also covers the file-replayed plan: later setups in this
        // process then hit the cache with the identical pinned plan, keeping
        // every contraction of the run on one denominator.
        if (!plan_from_cache && !plan_key.empty())
          plan_cache_instance().put(plan_key, network_desc_, plan_,
                                    tn_plan_cache_max());
        // += so an AUTO re-plan (pass 2) adds to pass 1's search cost rather
        // than hiding it. A cache hit contributes ~0 here, which is the whole
        // point: t_path_ms collapsing is how a hit is visible in the profile.
        if (tn_profile())
          prof_path_ms_ += tn_ms_since(prof_path_t0);

        // Printed per setup so a run shows the cache working live rather than
        // only in aggregate at the end.
        if (tn_plan_cache_enabled() && tn_verbose() && myrank_ == 0) {
          uint64_t ch = 0, cm = 0;
          size_t cn = 0;
          plan_cache_instance().stats(ch, cm, cn);
          fprintf(stderr,
                  "[AER_TN_PLAN_CACHE] %s hits=%llu misses=%llu entries=%zu%s\n",
                  plan_from_cache ? "HIT " : "miss",
                  (unsigned long long)ch, (unsigned long long)cm, cn,
                  plan_key.empty() ? " (network not keyable; not cached)" : "");
        }

        // aer-0037 WINDOW 3a: the tail of this branch runs AFTER the last
        // collective of the pass, so a throw here leaves the siblings blocked
        // at the end-of-pass agreement below rather than at a collective. It is
        // caught into setup_status for the same reason.
        try {
          // Slice-count ceiling: refuse an over-sliced plan rather than grind
          // (see tn_max_slices). plan_.num_slices is the MINLOC-broadcast
          // value, identical on every rank, so every rank raises this together
          // and each reports its own message.
          if (tn_max_slices() > 0 && plan_.num_slices > tn_max_slices()) {
            std::stringstream err;
            err << "[AER_TN] plan has " << plan_.num_slices
                << " slices, exceeding the AER_TN_MAX_SLICES ceiling of "
                << tn_max_slices() << ". The per-slice budget is too tight for "
                   "this circuit (a slice-grind). Raise "
                   "AER_TN_SLICE_TARGET_BYTES, raise AER_TN_MAX_SLICES if the "
                   "cost is acceptable, lower AER_TN_MIN_SLICES_PER_RANK if "
                   "the distribution floor put it here (aer-0035 asks for at "
                   "least that many slices PER RANK, so the floor scales with "
                   "the rank count), or use method='statevector'.";
            throw std::runtime_error(err.str());
          }

          plan_valid_ = true;
          pool_ready_ = false;
          cache_topology();
          prev_engaged_ = engaged;
        } catch (const NeedsTilingException &) {
          // Nothing in 3a raises this today -- the slice ceiling is a
          // runtime_error and cache_topology only copies vectors. It is handled
          // anyway so 3a and 3b behave identically: without it, a tiling gate
          // added here later would be silently demoted to a hard failure by the
          // std::exception catch below, because NeedsTilingException derives from
          // std::runtime_error.
          setup_status = 1;
          setup_tiling = std::current_exception();
        } catch (const std::exception &e) {
          setup_status = 2;
          setup_msg = e.what();
        } catch (...) {
          setup_status = 2;
          setup_msg =
              "[AER_TN] setup raised an exception of unknown type; caught so the "
              "ranks fail together rather than one rank unwinding alone.";
        }
      }

      // aer-0037 WINDOW 3b: build_sliced_specs, the pool prebuild and the
      // per-device setup. This is where hipMalloc for the pool, hipTensor plan
      // creation and aer-0033's tile-list ceiling live -- every one of them
      // resource-dependent and therefore NOT uniform across ranks. Skipped when
      // 3a already failed, so the first cause is the one reported.
      //
      // NeedsTilingException is separated out and NOT treated as a failure: it
      // is the AUTO retry signal, and folding it into the hard-failure path
      // would disable mode tiling entirely. It is agreed on its own code so the
      // ranks retry in lockstep, exactly as MPIParallelPathOptimizer does for
      // the planner gate.
      if (setup_status == 0) {
        try {
          {
            // aer-0047: descriptor/spec phase.
            std::chrono::steady_clock::time_point prof_specs_t0;
            if (tn_profile())
              prof_specs_t0 = std::chrono::steady_clock::now();
            build_sliced_specs();
            if (tn_profile())
              prof_specs_ms_ += tn_ms_since(prof_specs_t0);
          }

          if (!pool_ready_) {
            setup_pool_and_cache(0, engaged);
            pool_ready_ = true;
          } else if (tn_verbose() && myrank_ == 0) {
            // aer-0024: the counterpart of "reusing previous contraction path".
            // Its ABSENCE on a term that printed the path-reuse line is exactly
            // the partial-reuse symptom aer-0024 removes, so the diagnostic
            // greps for this line rather than inferring reuse from a missing
            // prebuild block.
            fprintf(stderr,
                    "[AER_TN] reusing prebuilt plan cache and memory pool\n");
          }

          slice_begin_ = myrank_ * plan_.num_slices / nprocs_;
          slice_end_ = (myrank_ + 1) * plan_.num_slices / nprocs_;

      // [MPI DIAG] AER_TN_MPI_DIAG_FULLSLICE forces every rank to contract the
      // whole slice range; contract() then skips the cross-rank reduction, so
      // each rank independently reproduces the single-GCD result. A rank whose
      // full-slice result diverges from the reference has a wrong per-rank plan;
      // if every rank matches the reference yet the normal distributed sum is
      // wrong, the bug is in the partition or the reduction. The dump (also
      // available without the override via AER_TN_MPI_DIAG) prints from every
      // rank so num_slices and the partition bounds are directly comparable.
          const bool mpi_diag_fullslice =
              getenv("AER_TN_MPI_DIAG_FULLSLICE") != nullptr;
          if (mpi_diag_fullslice) {
            slice_begin_ = 0;
            slice_end_ = plan_.num_slices;
          }
          if (mpi_diag_fullslice || getenv("AER_TN_MPI_DIAG"))
            fprintf(stderr,
                    "[AER_TN_MPIDIAG] rank %d/%d num_slices=%lu "
                    "sliced_modes=%zu begin=%lu end=%lu fullslice=%d\n",
                    myrank_, nprocs_, (unsigned long)plan_.num_slices,
                    plan_.sliced.size(), (unsigned long)slice_begin_,
                    (unsigned long)slice_end_, mpi_diag_fullslice ? 1 : 0);

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
        } catch (const NeedsTilingException &) {
          setup_status = 1;
          setup_tiling = std::current_exception();
        } catch (const std::exception &e) {
          setup_status = 2;
          setup_msg = e.what();
        } catch (...) {
          setup_status = 2;
          setup_msg =
              "[AER_TN] setup raised an exception of unknown type; caught so the "
              "ranks fail together rather than one rank unwinding alone.";
        }
      }

      // aer-0037: the single agreement point for this pass. Throws on every
      // rank if any rank failed hard; returns 1 if any rank needs tiling and
      // none failed. Reached by every rank, because every throw site after the
      // last collective has been caught above.
      if (agree_setup_status(setup_status, setup_msg) == 1) {
        // Re-raise the ORIGINAL exception on the rank that produced it, so its
        // gate and shape reach the retry's info line unchanged; ranks that did
        // not throw raise an equivalent one so the retry is collective. Mirrors
        // MPIParallelPathOptimizer's handling of the planner gate.
        if (setup_tiling)
          std::rethrow_exception(setup_tiling);
        throw NeedsTilingException(
            NeedsTilingException::Gate::Contractor,
            "a sibling MPI rank hit the m6n6k6 envelope during prebuild; "
            "re-planning all ranks with tiling engaged in lockstep.");
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
                "%s (re-planning once)\n",
                e.where(),
                tn_tiling_latch_enabled() ? "for the rest of this process"
                                          : "for this contraction only");
      engaged = true;
      retried = true;
      // aer-0030: record it process-wide so the next topology does not repeat
      // the discovery. This is the whole point: the discovery costs a full
      // cotengra search whose plan is then thrown away.
      tn_tiling_latch_set();
      // Force a fresh path search on pass 2: the held-back plan must not be
      // reused, and pool/topology caches from pass 1 are stale.
      plan_valid_ = false;
      prev_valid_ = false;
      pool_ready_ = false;
    }
  }

  // aer-0049: capture point. Reached only after the whole setup succeeded —
  // path search, slice ceiling (3a), specs, prebuild, pool allocation — so
  // a written plan is by
  // construction one that clears BOTH ceilings and sets up on this device.
  // A run that trips a ceiling throws before reaching here and leaves no
  // file, so re-running capture retries with a fresh (scattered) search
  // until an in-window plan lands. Rank-0-only and collective-free.
  maybe_write_plan_file(engaged);

  if (tn_verbose() && myrank_ == 0)
    fprintf(stderr, "[AER_TN] setup: %zu steps, %lu slices, %.2e FLOPs, %d GPUs\n",
            plan_.steps.size(), (unsigned long)plan_.num_slices,
            plan_.total_flops, num_devices_used_);

  if (tn_profile())
    prof_setup_ms_ += tn_ms_since(prof_setup_t0);
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
  // aer-0033: this pass rebuilds every step's tile list, so the materialised
  // sub-block count starts from zero here.
  tiles_built_ = 0;
  gemm_steps_routed_ = 0;
  gemm_steps_declined_tiled_ = 0;
  gemm_steps_declined_strided_ = 0;
  gemm_steps_declined_layout_ = 0;
  gemm_steps_declined_corder_ = 0;
  gemm_steps_declined_korder_ = 0;
  gemm_steps_permuted_ = 0;
  gemm_plans_skipped_ = 0;
  gemm_perm_scratch_elems_ = 0;
  GPUDevice<data_t> &dev = gpu_mgr_.device(device_idx);
  hipSetDevice(dev.device_id());

  // aer-0047: everything from here to plan_layout is the plan-prebuild phase
  // (the per-step descriptor walk and the hipTensor plan get_or_create loop,
  // the 31.6 ms-per-plan term aer-0046 skips for routed steps); plan_layout and
  // pool().allocate() are the pool phase. Both host-side/synchronous; += so a
  // multi-device setup and an AUTO re-plan accumulate rather than overwrite.
  std::chrono::steady_clock::time_point prof_prebuild_t0, prof_pool_t0;
  if (tn_profile())
    prof_prebuild_t0 = std::chrono::steady_clock::now();

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

    // aer-0057: the m6n6k6 envelope guard and the tiling decision are gone
    // with the engine that needed them. A GEMM has no shape envelope; the
    // permute layer generalises every step (aer-0052/0054), and the only
    // per-step shape limits are the ones the route enforces loudly itself:
    // int32 GEMM dimensions (hipblas_narrow) and kStageMaxRank modes per
    // operand (a named refusal in the classifier below). aer-0060 executed
    // the textual deletion of the tile construction and Tile machinery.

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
    step_plan_specs_[step].strides_a = strides_a;
    step_plan_specs_[step].strides_b = strides_b;

    // aer-0040: decide the GEMM route for this step, here, where the finished
    // descriptors already exist. Three conditions, each traced to the reason it
    // is necessary:
    //
    //  (1) NOT TILED. A tiled step's sub-blocks are strided VIEWS into their
    //      parents with their own offsets and per-block strides. Tiling exists
    //      only to fit hipTensor's m6n6k6 envelope, which a GEMM does not have,
    //      so an eligible step is by construction one that never needed it.
    //  (2) BOTH OPERANDS PACKED. project_slice hands a sliced input the PARENT's
    //      strides so projection stays pointer arithmetic (the OI9 note), and a
    //      view with arbitrary parent strides is not expressible as one lda.
    //      strides_a / strides_b are empty exactly when the operand is packed --
    //      build_sliced_specs leaves them empty for tensors carrying no sliced
    //      mode, and intermediates are always packed pool slots.
    //  (3) FREE AND CONTRACTED MODES EACH CONTIGUOUS, per gemm_operand_layout.
    //
    // C needs no test: compute_contraction_result builds it as
    // [A-only modes][B-only modes], which is [M][N], and it is always a packed
    // pool slot. On the final step modes_c is re-ordered into modes_out_ order,
    // which can break that, so the final step is excluded as well.
    step_plan_specs_[step].gemm_ok = false;
#ifdef AER_HIPBLAS
    {
      // aer-0054: with the permute layer on, tiled and strided steps proceed
      // into the analysis and are rescued there -- the permute pack reads a
      // strided view by reordering its PARENT strides (one gather, no
      // composition step; CLAIM S), and a tiled-class step routes as a WHOLE
      // step because tiling exists only for hipTensor's m6n6k6 envelope,
      // which a GEMM does not have (CLAIM T). EXCEPTION: under
      // AER_TN_GEMM_VERIFY a tiled step stays declined, because its
      // per-element reference would need the untiled whole-step hipTensor
      // plan, which violates the envelope -- exactly what tiling exists to
      // avoid. Tiled-class correctness is instead certified end-to-end by
      // the pinned route-on/route-off A/B, whose amplitudes are bit-level
      // comparable on a replayed plan.
      {
        const std::set<int32_t> sa(modes_a.begin(), modes_a.end());
        const std::set<int32_t> sb(modes_b.begin(), modes_b.end());
        std::set<int32_t> shared_modes;
        for (int32_t m : modes_a)
          if (sb.count(m))
            shared_modes.insert(m);

        // aer-0044: C must be laid out [M modes in A's order][N modes in B's
        // order], because that is what the GEMM writes with ldc = M.
        // compute_contraction_result produces exactly that, so on an ordinary
        // step this check cannot fail -- there it makes a guarantee explicit
        // rather than assumed. The FINAL step is what can break it: it rewrites
        // modes_c into modes_out_ order plus appended dummies. aer-0040
        // excluded the final step wholesale for that reason, which cost 12.7
        // percent of all steps across jobs 20657149 and 20657243. Checking the
        // actual post-rewrite modes_c instead recovers the ones still in the
        // right order -- and for a SCALAR output that is all of them, because
        // TensorNet::get_amplitudes calls set_output with EMPTY mode vectors,
        // so the final C carries no real modes and there is nothing to order.
        // On the amplitude path that recovers the final full contraction, which
        // is the largest step in the plan.
        //
        // Extent-1 modes are ignored on both sides: pad_contraction_mnk appends
        // them, they contribute a factor of one to every product and index zero
        // to every offset, and counting them would reject a correctly ordered C.
        std::vector<int32_t> want_c, have_c;
        for (size_t i = 0; i < modes_a.size(); i++)
          if (!sb.count(modes_a[i]) && extents_a[i] > 1)
            want_c.push_back(modes_a[i]);
        for (size_t i = 0; i < modes_b.size(); i++)
          if (!sa.count(modes_b[i]) && extents_b[i] > 1)
            want_c.push_back(modes_b[i]);
        for (size_t i = 0; i < modes_c.size(); i++)
          if (extents_c[i] > 1)
            have_c.push_back(modes_c[i]);
        const bool c_ok = (want_c == have_c);

        // aer-0045: the contracted modes must appear in the SAME ORDER on A and
        // on B. A GEMM has one k index; it walks A's K block and B's K block
        // together, so the mode each position of k stands for has to match.
        // aer-0040 never checked this. gemm_operand_layout inspects one operand
        // at a time -- it counts runs and multiplies extents -- so A = [m,k1,k2]
        // against B = [k2,k1,n] passes every guard: two contiguous runs each,
        // K extents agree at 4, and C is [m][n] so aer-0044 accepts it too. The
        // GEMM then pairs A's (k1,k2) with B's (k2,k1) and returns a plausible,
        // wrong number with no error anywhere. Reproduced on paper against a
        // reference contraction: all four elements of C wrong.
        //
        // This is a genuine decline, not something a transpose can repair.
        // HIPBLAS_OP_T swaps an operand's two BLOCKS; it cannot reorder modes
        // WITHIN a block. Only a physical permute could, and this route does
        // not permute.
        std::vector<int32_t> ka_seq, kb_seq;
        for (size_t i = 0; i < modes_a.size(); i++)
          if (sb.count(modes_a[i]) && extents_a[i] > 1)
            ka_seq.push_back(modes_a[i]);
        for (size_t i = 0; i < modes_b.size(); i++)
          if (sa.count(modes_b[i]) && extents_b[i] > 1)
            kb_seq.push_back(modes_b[i]);
        const bool k_ok = (ka_seq == kb_seq);

        bool at = false, bt = false;
        int64_t m_ext = 1, ka_ext = 1, n_ext = 1, kb_ext = 1;
        const bool a_ok = gemm_operand_layout(modes_a, extents_a, shared_modes,
                                              at, m_ext, ka_ext);
        const bool b_ok = gemm_operand_layout(modes_b, extents_b, shared_modes,
                                              bt, n_ext, kb_ext);
        // K seen from A and from B must agree, and M*N must be exactly the C
        // slot. The second is not redundant: compute_contraction_result drops
        // EVERY shared mode from C, while a batch-mode layout keeps one
        // present in A, B and C. Those two disagree for a batch mode, and if one ever appeared this route would write an M*N block
        // into a slot of a different size. Refusing on the invariant is the
        // difference between a declined step and silent corruption.
        const int64_t c_elems_expected =
            all_specs_[result_idx].num_elements();
        // The classic (no-copy) accept now states what the early declines
        // used to imply: packed operands, untiled. A tiled or strided step
        // reaching this point can only be rescued below.
        if (strides_a.empty() && strides_b.empty() &&
            a_ok && b_ok && c_ok && k_ok && ka_ext == kb_ext &&
            m_ext * n_ext == c_elems_expected) {
          step_plan_specs_[step].gemm_ok = true;
          // A is (M x K) when its order is [free][contracted] -- OP_N, lda = M
          // -- and (K x M) when it is [contracted][free], which is the
          // transpose of what is wanted -- OP_T, lda = K. gemm_operand_layout
          // reports which class comes first, so at is already the flag.
          step_plan_specs_[step].gemm_a_trans = at;
          // B is wanted as (K x N). Stored [contracted][free] it already is
          // that -- OP_N -- and stored [free][contracted] it is (N x K) and
          // must be transposed. So the flag is the NEGATION of B's report.
          step_plan_specs_[step].gemm_b_trans = !bt;
          step_plan_specs_[step].gemm_m = m_ext;
          step_plan_specs_[step].gemm_n = n_ext;
          step_plan_specs_[step].gemm_k = ka_ext;
          // aer-0055: a classic (no-copy) accept still needs interleaved
          // scratch under the complex sub-path, so its operand and result
          // sizes join the scratch maximum. Without this, a plan with no
          // rescued step would size the scratch at zero.
          if (tn_gemm_complex()) {
            gemm_perm_scratch_elems_ = std::max<uint64_t>(
                gemm_perm_scratch_elems_,
                stage_block_elems(extents_a));
            gemm_perm_scratch_elems_ = std::max<uint64_t>(
                gemm_perm_scratch_elems_,
                stage_block_elems(extents_b));
            gemm_perm_scratch_elems_ = std::max<uint64_t>(
                gemm_perm_scratch_elems_,
                static_cast<uint64_t>(c_elems_expected));
          }
          gemm_steps_routed_++;
        } else if (ka_ext == kb_ext &&
                   m_ext * n_ext == c_elems_expected &&
                   modes_a.size() <= static_cast<size_t>(kStageMaxRank) &&
                   modes_b.size() <= static_cast<size_t>(kStageMaxRank) &&
                   modes_c.size() <= static_cast<size_t>(kStageMaxRank)) {
          // aer-0052: rescue a layout or korder decline by physically packing
          // both operands into scratch under the audited target orders
          //   A' = [A free modes, original relative order][K, canonical]
          //   B' = [K, canonical][B free modes, original relative order]
          // canonical := A's contracted-mode appearance order (extent-1 modes
          // at each tail; a mode's extent is a network-global fact, so an
          // extent-1 contracted mode is extent-1 on BOTH operands and drops
          // from K identically on each side). After the pack A' is (M x K)
          // and B' is (K x N), both OP_N, and C is written [M in A's original
          // free order][N in B's original free order] -- byte-identical to
          // the slot compute_contraction_result declares, because the free
          // relative order was never touched. Mapping audited standalone in
          // audit_aer0052_permute_design.py (115,229 cases, 0 mismatches)
          // BEFORE this code existed, per the aer-0045 method. c_ok, the K
          // product agreement and the M*N == C-slot invariant are still
          // REQUIRED: the permute repairs order, never membership.
          //
          // Both operands are routed through the permute even if one is
          // already contiguous -- the audit covers exactly that uniform
          // shape, and one code path is one thing to trust.
          // aer-0054 extends the rescue to STRIDED sources (perm_str is the
          // view's parent strides reordered -- project_slice's base shift is
          // already on the plane pointer, so the pack composes for free),
          // TILED-class steps (routed whole; tiles cleared below), and
          // CORDER (C scattered into the declared order; scatter-eligible
          // means the declared extent>1 modes are the same SET as the
          // natural ones -- the permute repairs order, never membership).
          bool scatter_needed = !c_ok;
          bool membership_ok = true;
          if (scatter_needed) {
            std::vector<int32_t> wsort = want_c, hsort = have_c;
            std::sort(wsort.begin(), wsort.end());
            std::sort(hsort.begin(), hsort.end());
            membership_ok = (wsort == hsort);
          }
          std::vector<size_t> permA, permB;
          std::vector<int32_t> kA;
          for (size_t i = 0; i < modes_a.size(); i++)
            if (!sb.count(modes_a[i]) && extents_a[i] > 1)
              permA.push_back(i);
          for (size_t i = 0; i < modes_a.size(); i++)
            if (sb.count(modes_a[i]) && extents_a[i] > 1) {
              permA.push_back(i);
              kA.push_back(modes_a[i]);
            }
          for (size_t i = 0; i < modes_a.size(); i++)
            if (extents_a[i] <= 1)
              permA.push_back(i);
          std::map<int32_t, size_t> pos_b;
          for (size_t i = 0; i < modes_b.size(); i++)
            pos_b[modes_b[i]] = i;
          bool perm_ok = true;
          for (size_t j = 0; j < kA.size() && perm_ok; j++) {
            std::map<int32_t, size_t>::iterator it = pos_b.find(kA[j]);
            if (it == pos_b.end() || extents_b[it->second] <= 1)
              perm_ok = false;
            else
              permB.push_back(it->second);
          }
          if (perm_ok && membership_ok) {
            for (size_t i = 0; i < modes_b.size(); i++)
              if (!sa.count(modes_b[i]) && extents_b[i] > 1)
                permB.push_back(i);
            for (size_t i = 0; i < modes_b.size(); i++)
              if (extents_b[i] <= 1)
                permB.push_back(i);
            PlanSpec &psr = step_plan_specs_[step];
            auto fill_perm = [](const std::vector<int64_t> &ext,
                                const std::vector<size_t> &perm,
                                std::vector<int64_t> &pe,
                                std::vector<int64_t> &pstr) {
              std::vector<int64_t> src(ext.size());
              int64_t acc = 1;
              for (size_t i = 0; i < ext.size(); i++) {
                src[i] = acc;
                acc *= (ext[i] > 0 ? ext[i] : 1);
              }
              pe.clear();
              pstr.clear();
              for (size_t j = 0; j < perm.size(); j++) {
                pe.push_back(ext[perm[j]]);
                pstr.push_back(src[perm[j]]);
              }
            };
            fill_perm(extents_a, permA, psr.perm_ext_a, psr.perm_str_a);
            fill_perm(extents_b, permB, psr.perm_ext_b, psr.perm_str_b);
            // Strided source: replace the packed column-major strides the
            // lambda derived with the view's PARENT strides, reordered by
            // the same permutation (CLAIM S). Extents are unchanged.
            if (!strides_a.empty())
              for (size_t j = 0; j < permA.size(); j++)
                psr.perm_str_a[j] = strides_a[permA[j]];
            if (!strides_b.empty())
              for (size_t j = 0; j < permB.size(); j++)
                psr.perm_str_b[j] = strides_b[permB[j]];
            // Corder: the GEMM's natural C is [want_c order]; scatter it
            // into the declared slot with declared column-major strides
            // reordered to natural order (CLAIM C).
            psr.gemm_scatter_c = scatter_needed;
            if (scatter_needed) {
              std::vector<int64_t> cstr(extents_c.size());
              int64_t cacc = 1;
              for (size_t i = 0; i < extents_c.size(); i++) {
                cstr[i] = cacc;
                cacc *= (extents_c[i] > 0 ? extents_c[i] : 1);
              }
              std::map<int32_t, size_t> pos_c;
              for (size_t i = 0; i < modes_c.size(); i++)
                if (extents_c[i] > 1)
                  pos_c[modes_c[i]] = i;
              psr.scat_ext.clear();
              psr.scat_str.clear();
              const std::map<int32_t, size_t> *pa2 = nullptr;
              (void)pa2;
              for (size_t j = 0; j < want_c.size(); j++) {
                const size_t ci = pos_c[want_c[j]];
                psr.scat_ext.push_back(extents_c[ci]);
                psr.scat_str.push_back(cstr[ci]);
              }
            }
            // Tiled-class: the whole-step route replaces the sub-blocks.
            // Clearing them sends prebuild and execution down the untiled
            // path and releases their host memory. (Their construction cost
            // was already paid this pass; hoisting the classifier above tile
            // materialisation is deliberately left to the removal era, when
            // the tiling machinery is deleted outright.)
            psr.gemm_ok = true;
            psr.gemm_perm_a = true;
            psr.gemm_perm_b = true;
            // After the pack A' is [free][contracted] and B' is
            // [contracted][free]: OP_N on both, by construction.
            psr.gemm_a_trans = false;
            psr.gemm_b_trans = false;
            psr.gemm_m = m_ext;
            psr.gemm_n = n_ext;
            psr.gemm_k = ka_ext;
            gemm_perm_scratch_elems_ = std::max<uint64_t>(
                gemm_perm_scratch_elems_,
                stage_block_elems(psr.perm_ext_a));
            gemm_perm_scratch_elems_ = std::max<uint64_t>(
                gemm_perm_scratch_elems_,
                stage_block_elems(psr.perm_ext_b));
            if (scatter_needed || tn_gemm_complex())
              gemm_perm_scratch_elems_ = std::max<uint64_t>(
                  gemm_perm_scratch_elems_,
                  static_cast<uint64_t>(c_elems_expected));
            gemm_steps_permuted_++;
          } else {
            // aer-0057: no fallback engine exists, so an unroutable step is
            // a loud, named refusal at setup -- never a silent wrong answer,
            // never a quiet decline. perm_ok false means a contracted mode
            // of A is missing or extent-1-mismatched on B, and
            // membership_ok false means the declared C modes are not the
            // set the contraction produces -- both indicate a malformed
            // network upstream, not a layout problem.
            std::stringstream err;
            err << "[AER_TN] step " << step
                << " cannot be routed to the GEMM path (and the hipTensor "
                   "fallback was removed in aer-0057): "
                << (!perm_ok ? "contracted-mode set mismatch between operands"
                             : "declared C modes are not the contraction's "
                               "output set")
                << ". modes_a=" << modes_to_str(modes_a)
                << " modes_b=" << modes_to_str(modes_b)
                << " modes_c=" << modes_to_str(modes_c) << ".";
            throw std::runtime_error(err.str());
          }
        } else {
          // Rescue ineligible: name the reason. ka!=kb is unreachable for a
          // well-formed network (one size_dict per network -- a shared mode
          // has one extent), so it is reported as the internal error it is;
          // the rank cap is the one honest capacity limit and says so.
          std::stringstream err;
          err << "[AER_TN] step " << step
              << " cannot be routed to the GEMM path (and the hipTensor "
                 "fallback was removed in aer-0057): ";
          if (ka_ext != kb_ext)
            err << "internal error: contracted-extent product differs "
                   "between operands (" << ka_ext << " vs " << kb_ext
                << ") -- a shared mode has one network-global extent, so "
                   "this indicates descriptor corruption";
          else if (m_ext * n_ext != c_elems_expected)
            err << "internal error: M*N (" << m_ext * n_ext
                << ") does not equal the C slot's element count ("
                << c_elems_expected << ")";
          else
            err << "an operand or the output exceeds kStageMaxRank = "
                << kStageMaxRank
                << " modes; raise the constant and rebuild, or lower the "
                   "per-slice budget so the slicer cuts this step smaller";
          err << ". modes_a=" << modes_to_str(modes_a)
              << " modes_b=" << modes_to_str(modes_b)
              << " modes_c=" << modes_to_str(modes_c) << ".";
          throw std::runtime_error(err.str());
        }
      }
    }
#endif

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
      fprintf(stderr, "[AER_TN] prebuild step %zu\n", step);

    // aer-0046: a step the GEMM route will take never calls
    // hiptensorContraction, so building its plan HERE is the waste aer-0043
    // aimed at and missed. aer-0043 guarded the call in contract_single_slice,
    // where the cache is already warm and get_or_create is a lookup. THIS loop
    // is where the construction happens, and ps.gemm_ok is already decided by
    // the time it runs, so the same condition can be applied where it pays.
    //
    // WHAT IT IS WORTH, from job 20659936 at the shipped path budget. Route
    // off, 16 qubits: t_setup_ms 9404.310, t_path_ms 6171.214, so 3233.096 ms
    // is setup that is neither path search nor contraction, over 96 steps
    // (census, job 20662067) = 33.7 ms per step. 25 qubits: 16133.620 minus
    // 10872.088 = 5261.532 ms over 150 steps = 35.1 ms per step. A flat
    // per-step cost on two different circuits, which is the shape a per-step
    // get_or_create has, and it is 34 and 32 percent of wall. Contraction over
    // the same two runs is 9.671 ms and 193.300 ms -- 0.10 and 1.18 percent.
    // At the measured 69.7 percent routed this skip removes roughly 2.2 s of
    // 9.4 s at 16 qubits.
    //
    // aer-0057: every step routes; no hipTensor plan is ever built. A step
    // that could not route threw a named refusal in the classifier above,
    // so reaching here with gemm_ok unset is impossible by construction.
    gemm_plans_skipped_++;
    continue;

  }


  // aer-0040: the routing census. Printed whenever the route is enabled, not
  // behind AER_TN_VERBOSE, because a run whose steps were all declined would
  // otherwise look identical to one that routed everything.
#ifdef AER_HIPBLAS
  {
    fprintf(stderr,
            "[AER_TN_GEMM] steps=%zu routed=%llu permuted=%llu declined: "
            "tiled=%llu "
            "strided=%llu layout=%llu corder=%llu korder=%llu sum=%llu "
            "plans_skipped=%llu\n",
            num_steps, (unsigned long long)gemm_steps_routed_,
            (unsigned long long)gemm_steps_permuted_,
            (unsigned long long)gemm_steps_declined_tiled_,
            (unsigned long long)gemm_steps_declined_strided_,
            (unsigned long long)gemm_steps_declined_layout_,
            (unsigned long long)gemm_steps_declined_corder_,
            (unsigned long long)gemm_steps_declined_korder_,
            (unsigned long long)(gemm_steps_routed_ +
                                 gemm_steps_permuted_ +
                                 gemm_steps_declined_tiled_ +
                                 gemm_steps_declined_strided_ +
                                 gemm_steps_declined_layout_ +
                                 gemm_steps_declined_corder_ +
                                 gemm_steps_declined_korder_),
            (unsigned long long)gemm_plans_skipped_);
    if (tn_gemm_complex())
      fprintf(stderr, "[AER_TN_GEMM] complex sub-path active: one native "
                      "complex GEMM per dispatch (interleaved scratch)\n");
  }
#endif

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
    fprintf(stderr, "[AER_TN] prebuild loop done; laying out pool\n");

  if (tn_profile()) {
    prof_prebuild_ms_ += tn_ms_since(prof_prebuild_t0);
    prof_pool_t0 = std::chrono::steady_clock::now();
  }
  dev.pool().plan_layout(intermediates, static_cast<int>(num_steps));

  if (tn_verbose())
    fprintf(stderr, "[AER_TN] pool layout done; allocating\n");

  dev.pool().allocate(dev.device_id());
  if (tn_profile())
    prof_pool_ms_ += tn_ms_since(prof_pool_t0);

  if (tn_verbose())
    fprintf(stderr, "[AER_TN] pool: %zu bytes (device %d)\n",
            dev.pool().total_size(), dev.device_id());
}

#ifdef AER_MPI
template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::agree_or_fail_together(
    int local_failed, const std::string &local_msg) {
  // Single rank: no sibling to hang, so just re-raise the local failure.
  if (nprocs_ <= 1) {
    if (local_failed)
      throw std::runtime_error(local_msg);
    return;
  }

  // Every rank contributes its 0/1 verdict; MPI_MAX makes the outcome 1 iff
  // ANY rank failed. This Allreduce is the ONLY collective every rank is
  // guaranteed to reach after local contraction (the failing rank caught its
  // throw and continued to here), so it cannot itself desync.
  int any_failed = 0;
  MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  if (!any_failed)
    return;

  // The rank that actually failed re-raises its own message so the true cause
  // (HIP/CK fault, OOM, or a slicing-bookkeeping guard) reaches its RESULT
  // MESSAGE -- not its stderr, which stays empty; the text travels through
  // circuit_executor.hpp:726. The ranks that succeeded fail together with a
  // pointer to it, rather than blocking forever at the reduction below waiting
  // on a rank that has already unwound (observed job 19214242).
  if (local_failed)
    throw std::runtime_error(local_msg);
  throw std::runtime_error(
      "[AER_TN] contraction aborted: a sibling MPI rank threw during its "
      "assigned slice range, so this rank fails together rather than blocking "
      "at the cross-rank reduction. The failing rank reports the underlying "
      "cause through the same result message channel as this one -- it does "
      "NOT print to stderr; the text reaches result.message via "
      "circuit_executor.hpp:726.");
}

// aer-0032: turn a per-rank plan-cache verdict into a collective one, so the
// cross-instruction cache can be USED under MPI instead of being switched off.
//
// STAGE 1 -- all or nothing. MPI_MIN over a 0/1 flag is 1 only if EVERY rank
// hit. A rank that missed, or whose canonical key differed, or whose entry
// failed sliced-mode re-resolution (PlanCache::get erases the entry and reports
// a miss in that case) drags the whole communicator onto the search path. So
// correctness does NOT depend on canonical keys agreeing across ranks:
// disagreement costs a path search, never a hang and never a divergent plan.
//
// STAGE 2 -- the replayed plans must also BE the same plan. Stage 1 only proves
// each rank found something under its own key. Compare a fingerprint of what
// each rank actually replayed and discard on any difference. Under MPI a stored
// plan came from find_path's MINLOC broadcast and is identical everywhere, so
// this is expected to pass; it is here because a divergent plan is a silently
// wrong answer rather than a crash, and "expected" is not a guarantee.
//
// Two collectives rather than one, because a miss leaves plan_ untouched
// (PlanCache::get returns before assigning to out), so the fingerprint is
// meaningless until stage 1 has established that every rank hit.
template <typename data_t>
bool TensorNetContractor_HipTensor<data_t>::agree_plan_cache_hit(
    bool local_hit) {
  int hit = local_hit ? 1 : 0;
  int all_hit = 0;
  MPI_Allreduce(&hit, &all_hit, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!all_hit)
    return false;

  int64_t fp[4];
  fp[0] = static_cast<int64_t>(plan_.steps.size());
  fp[1] = static_cast<int64_t>(plan_.num_slices);
  fp[2] = static_cast<int64_t>(plan_.sliced.size());
  // Bit pattern, not value: MIN and MAX of the reinterpreted integers agree if
  // and only if every rank's double is bit-identical, which is the question
  // being asked. The ordering semantics of the reinterpretation do not matter.
  double flops = plan_.total_flops;
  std::memcpy(&fp[3], &flops, sizeof(double));

  int64_t lo[4], hi[4];
  MPI_Allreduce(fp, lo, 4, MPI_INT64_T, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(fp, hi, 4, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
  for (int i = 0; i < 4; i++) {
    if (lo[i] != hi[i]) {
      if (tn_verbose() && myrank_ == 0)
        fprintf(stderr,
                "[AER_TN_PLAN_CACHE] cross-rank plan fingerprint mismatch at "
                "field %d; discarding the replayed plan and searching "
                "collectively instead\n",
                i);
      return false;
    }
  }
  return true;
}
#endif

// aer-0049: replay leg. Reads AER_TN_PLAN_FILE and, if it decodes and keys
// onto the current network, installs it as plan_. Returns false (fresh search)
// on any of: knob unset, bypass set, file absent, wrong key, malformed file,
// unresolvable sliced coordinates. Every refusal except "knob unset / file
// absent" prints its reason unconditionally — replaying a plan is a
// deliberate, high-stakes act and a silent fall-through to a scattered search
// would make the pinned run's numbers wrong without a trace.
//
// MPI: rank 0 reads the bytes and broadcasts them; every rank then parses the
// SAME bytes against its own network (identical on every rank by
// construction), so the verdict is deterministic and uniform with no further
// agreement collective. Callers must invoke this only under rank-uniform
// conditions — see the call site in setup_contraction.
template <typename data_t>
bool TensorNetContractor_HipTensor<data_t>::try_load_plan_file(bool engaged) {
  const std::string path = tn_plan_file();
  if (path.empty() || plan_file_bypass_)
    return false;

  std::string text;
  bool have_text = false;
#ifdef AER_MPI
  if (nprocs_ > 1) {
    int64_t len = -1;
    if (myrank_ == 0) {
      std::ifstream f(path.c_str());
      if (f) {
        std::ostringstream ss;
        ss << f.rdbuf();
        text = ss.str();
        len = static_cast<int64_t>(text.size());
      }
    }
    MPI_Bcast(&len, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    if (len < 0)
      return false;
    text.resize(static_cast<size_t>(len));
    if (len > 0)
      MPI_Bcast(&text[0], static_cast<int>(len), MPI_CHAR, 0, MPI_COMM_WORLD);
    have_text = true;
  }
#endif
  if (!have_text) {
    std::ifstream f(path.c_str());
    if (!f)
      return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    text = ss.str();
  }

  const size_t elem_bytes = 2 * sizeof(data_t);
  const std::string key =
      plan_file_network_key(network_desc_, engaged, elem_bytes);
  if (key.empty()) {
    if (myrank_ == 0)
      fprintf(stderr,
              "[AER_TN_PLAN_FILE] this network is not keyable; ignoring %s and "
              "searching fresh\n",
              path.c_str());
    return false;
  }
  std::string why;
  ContractionPlan replayed;
  if (!plan_file_decode(text, network_desc_, key, replayed, why)) {
    if (myrank_ == 0)
      fprintf(stderr,
              "[AER_TN_PLAN_FILE] %s does not replay onto this network: %s; "
              "searching fresh instead\n",
              path.c_str(), why.c_str());
    return false;
  }
  plan_ = replayed;
  if (myrank_ == 0)
    fprintf(stderr,
            "[AER_TN_PLAN_FILE] replayed plan from %s (steps=%zu slices=%llu "
            "flops=%.3e)\n",
            path.c_str(), plan_.steps.size(),
            (unsigned long long)plan_.num_slices, plan_.total_flops);
  return true;
}

// aer-0049: capture leg. Rank-0-only, collective-free, capture-once (an
// existing file is never overwritten — delete the file to re-capture). Called
// exclusively from the end of a fully successful setup_contraction, so the
// written plan has already cleared the slice ceiling, the tile ceiling and
// the prebuild on this device. Written via a temp file + rename so a job
// killed mid-write cannot leave a truncated file that a later replay would
// half-parse (rename within one directory is atomic on Lustre).
template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::maybe_write_plan_file(
    bool engaged) {
  const std::string path = tn_plan_file();
  if (path.empty() || plan_file_bypass_ || myrank_ != 0)
    return;
  {
    std::ifstream probe(path.c_str());
    if (probe)
      return;
  }
  // aer-0051: capture-quality gate. Checked against tiles_built_, the same
  // counter the (now-deleted) hard tile ceiling read, so the two verdicts
  // cannot drift. Refusing to WRITE (not to run) makes re-submission a
  // retry-until-quality loop with no wasted result: this run's answer stands.
  if (tn_plan_file_max_tiles() > 0 && tiles_built_ > tn_plan_file_max_tiles()) {
    fprintf(stderr,
            "[AER_TN_PLAN_FILE] plan prebuilt %llu tiles, above the "
            "AER_TN_PLAN_FILE_MAX_TILES capture target of %llu; plan NOT "
            "captured to %s (the run itself completed -- re-submit capture to "
            "draw a fresh search, or raise the target)\n",
            (unsigned long long)tiles_built_,
            (unsigned long long)tn_plan_file_max_tiles(), path.c_str());
    return;
  }
  const size_t elem_bytes = 2 * sizeof(data_t);
  const std::string key =
      plan_file_network_key(network_desc_, engaged, elem_bytes);
  if (key.empty()) {
    fprintf(stderr,
            "[AER_TN_PLAN_FILE] this network is not keyable; plan not "
            "captured to %s\n",
            path.c_str());
    return;
  }
  std::string text;
  if (!plan_file_encode(network_desc_, key, plan_, text)) {
    fprintf(stderr,
            "[AER_TN_PLAN_FILE] a sliced mode of this plan is not present in "
            "the network (cannot store structural coordinates); plan not "
            "captured to %s\n",
            path.c_str());
    return;
  }
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp.c_str(), std::ios::trunc);
    if (!f) {
      fprintf(stderr,
              "[AER_TN_PLAN_FILE] cannot open %s for writing (missing "
              "directory?); plan not captured\n",
              tmp.c_str());
      return;
    }
    f << text;
    f.flush();
    if (!f) {
      fprintf(stderr,
              "[AER_TN_PLAN_FILE] write to %s failed; plan not captured\n",
              tmp.c_str());
      f.close();
      std::remove(tmp.c_str());
      return;
    }
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    fprintf(stderr,
            "[AER_TN_PLAN_FILE] rename %s -> %s failed; plan not captured\n",
            tmp.c_str(), path.c_str());
    std::remove(tmp.c_str());
    return;
  }
  fprintf(stderr,
          "[AER_TN_PLAN_FILE] captured plan to %s (steps=%zu slices=%llu "
          "tiles_prebuilt=%llu flops=%.3e)\n",
          path.c_str(), plan_.steps.size(),
          (unsigned long long)plan_.num_slices,
          (unsigned long long)tiles_built_, plan_.total_flops);
}

// aer-0037: agree the outcome of a setup phase across ranks.
//
//   0 = this rank is fine
//   1 = this rank needs tiling (the AUTO retry signal)
//   2 = this rank failed hard
//
// MPI_MAX, so a hard failure anywhere outranks a tiling retry anywhere. The
// other order would leave the run re-planning a network that cannot be set up,
// which is worse than either outcome alone. Returns the agreed code, 0 or 1; a
// 2 never returns, because every rank throws.
//
// WHY THIS EXISTS. setup_contraction is not covered by agree_or_fail_together,
// which wraps only contract_all/accumulate_across_gpus inside the three contract
// entrypoints, and there is no try/catch anywhere in tensor_net.hpp or
// tensor_net_state.hpp -- all nine setup_contraction call sites are unguarded.
// So a device-dependent throw during setup (hipMemGetInfo, the workspace query,
// the pool hipMalloc, hipTensor plan creation, aer-0033's tile-list ceiling)
// unwound one rank while its siblings blocked forever in the next collective.
// The symptom was a hang with no error anywhere, which is the worst way for a
// distributed run to fail.
//
// Deliberately defined OUTSIDE #ifdef AER_MPI. With one rank it degenerates to
// "rethrow locally", which is the correct single-rank behaviour, and keeps one
// code path rather than two.
//
// The sibling message says what it can honestly say: agree_or_fail_together's
// comment claims the failing rank's cause reaches its stderr, and it does not --
// nothing prints, the text reaches result.message via circuit_executor.hpp:726.
// This message does not repeat that claim.
template <typename data_t>
int TensorNetContractor_HipTensor<data_t>::agree_setup_status(
    int local_status, const std::string &local_msg) {
  int global_status = local_status;
#ifdef AER_MPI
  if (nprocs_ > 1)
    MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
#endif
  if (global_status < 2)
    return global_status;
  if (local_status >= 2)
    throw std::runtime_error(local_msg);
  throw std::runtime_error(
      "[AER_TN] contraction setup aborted: a sibling MPI rank threw while "
      "preparing this contraction -- device memory query, contraction path "
      "search, memory pool allocation, plan prebuild or the tile-list ceiling "
      "-- so this rank fails together rather than blocking at the next "
      "collective. The failing rank reports the underlying cause through the "
      "same result message channel as this one.");
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::prof_sync_devices() {
  for (int idev = 0; idev < num_devices_used_; idev++) {
    hipSetDevice(gpu_mgr_.device(idev).device_id());
    hipStreamSynchronize(gpu_mgr_.device(idev).stream());
  }
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::prof_report(
    const char *entrypoint) {
  const uint_t slices_local = slice_end_ - slice_begin_;
  // This rank contracts slices_local of num_slices; attribute that fraction of
  // the plan's total FLOPs to it for an honest per-rank throughput. Guard the
  // divide: a rank-0 scalar output can plan to num_slices==0/1.
  const double local_flops =
      (plan_.num_slices > 0)
          ? plan_.total_flops * (double)slices_local / (double)plan_.num_slices
          : plan_.total_flops;
  const double contract_s = prof_contract_ms_ / 1000.0;
  const double gflops =
      (contract_s > 0.0) ? (local_flops / contract_s / 1.0e9) : 0.0;
  const double t_total = prof_setup_ms_ + prof_contract_ms_ +
                         prof_reduce_gpu_ms_ + prof_reduce_mpi_ms_;
  fprintf(stderr,
          "[AER_TN_PROFILE] entry=%s rank=%d nprocs=%d ndev=%d slices=%lu "
          "slices_local=%lu plan_flops=%.3e local_flops=%.3e "
          "t_setup_ms=%.3f t_path_ms=%.3f t_contract_ms=%.3f "
          "t_reduce_gpu_ms=%.3f t_reduce_mpi_ms=%.3f t_total_ms=%.3f "
          "gflops_local=%.3f out_size=%lu "
          "gemm_route=%d t_gemm_ms=%.3f gemm_calls=%lu ht_calls=%lu "
          "t_specs_ms=%.3f t_prebuild_ms=%.3f t_pool_ms=%.3f\n",
          entrypoint, myrank_, nprocs_, num_devices_used_,
          (unsigned long)plan_.num_slices, (unsigned long)slices_local,
          plan_.total_flops, local_flops, prof_setup_ms_, prof_path_ms_,
          prof_contract_ms_, prof_reduce_gpu_ms_, prof_reduce_mpi_ms_, t_total,
          gflops, (unsigned long)out_size_,
#ifdef AER_HIPBLAS
          1,
#else
          0,
#endif
          prof_gemm_ms_, (unsigned long)prof_gemm_calls_,
          (unsigned long)prof_ht_calls_,
          prof_specs_ms_, prof_prebuild_ms_, prof_pool_ms_);
}

template <typename data_t>
void TensorNetContractor_HipTensor<data_t>::contract(
    std::vector<std::complex<data_t>> &out) {
  // Each device holds the partial sum of its assigned slice range after
  // contract_all(); each rank holds its slice range after MPI partitioning.
  // Reduce both layers before reading the result off the primary device —
  // skipping either reduction drops every slice not assigned to (rank 0,
  // device 0). Reduction order is fixed by index (P4).
#ifdef AER_MPI
  // Run the per-rank local contraction under a fail-together guard: a HIP/CK
  // fault or slicing-bookkeeping throw on THIS rank's slice range is caught,
  // agreed collectively, and turned into a job-wide abort BEFORE
  // accumulate_across_mpi() — so no surviving rank blocks at the reduction.
  int local_failed = 0;
  std::string local_msg;
  try {
    std::chrono::steady_clock::time_point pt;
    if (tn_profile()) pt = std::chrono::steady_clock::now();
    contract_all();
    if (tn_profile()) { prof_sync_devices(); prof_contract_ms_ += tn_ms_since(pt); }
    if (tn_profile()) pt = std::chrono::steady_clock::now();
    accumulate_across_gpus();
    if (tn_profile()) { prof_sync_devices(); prof_reduce_gpu_ms_ += tn_ms_since(pt); }
  } catch (const std::exception &e) {
    local_failed = 1;
    local_msg = e.what();
  }
  agree_or_fail_together(local_failed, local_msg);

  if (getenv("AER_TN_MPI_DIAG_FULLSLICE")) {
    // Every rank already contracted the full slice range; skip the reduction
    // so each rank's buffer holds its OWN independent full result. Print it for
    // cross-rank comparison against the single-GCD reference, then leave rank
    // 0's (un-reduced) full result in place for get_output below.
    std::vector<std::complex<data_t>> r;
    gpu_mgr_.primary().get_output(r);
    fprintf(stderr,
            "[AER_TN_MPIDIAG] rank %d full-slice out_size=%zu result[0]=(%.15e,%.15e)\n",
            myrank_, out_size_, r.empty() ? 0.0 : (double)r[0].real(),
            r.empty() ? 0.0 : (double)r[0].imag());
  } else {
    std::chrono::steady_clock::time_point pt;
    if (tn_profile()) pt = std::chrono::steady_clock::now();
    accumulate_across_mpi();
    if (tn_profile()) prof_reduce_mpi_ms_ += tn_ms_since(pt);
  }
#else
  {
    std::chrono::steady_clock::time_point pt;
    if (tn_profile()) pt = std::chrono::steady_clock::now();
    contract_all();
    if (tn_profile()) { prof_sync_devices(); prof_contract_ms_ += tn_ms_since(pt); }
    if (tn_profile()) pt = std::chrono::steady_clock::now();
    accumulate_across_gpus();
    if (tn_profile()) { prof_sync_devices(); prof_reduce_gpu_ms_ += tn_ms_since(pt); }
  }
#endif
  gpu_mgr_.primary().get_output(out);
  if (tn_profile()) prof_report("contract");
}

template <typename data_t>
double TensorNetContractor_HipTensor<data_t>::contract_and_trace(uint_t num_qubits) {
#ifdef AER_MPI
  // Fail-together guard (see agree_or_fail_together): a per-rank throw in
  // contract_all() must abort the job collectively rather than leaving siblings
  // blocked at the MPI_Allreduce(SUM) below.
  double ret = 0.0;
  int local_failed = 0;
  std::string local_msg;
  try {
    std::chrono::steady_clock::time_point pt;
    if (tn_profile()) pt = std::chrono::steady_clock::now();
    contract_all();
    if (tn_profile()) { prof_sync_devices(); prof_contract_ms_ += tn_ms_since(pt); }
    if (tn_profile()) pt = std::chrono::steady_clock::now();
    for (int idev = 0; idev < num_devices_used_; idev++)
      ret += gpu_mgr_.device(idev).trace_output(num_qubits);
    if (tn_profile()) prof_reduce_gpu_ms_ += tn_ms_since(pt);
  } catch (const std::exception &e) {
    local_failed = 1;
    local_msg = e.what();
  }
  agree_or_fail_together(local_failed, local_msg);
  if (nprocs_ > 1) {
    std::chrono::steady_clock::time_point pt;
    if (tn_profile()) pt = std::chrono::steady_clock::now();
    double sum = ret;
    MPI_Allreduce(&sum, &ret, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD);
    if (tn_profile()) prof_reduce_mpi_ms_ += tn_ms_since(pt);
  }
  if (tn_profile()) prof_report("contract_and_trace");
  return ret;
#else
  std::chrono::steady_clock::time_point pt;
  if (tn_profile()) pt = std::chrono::steady_clock::now();
  contract_all();
  if (tn_profile()) { prof_sync_devices(); prof_contract_ms_ += tn_ms_since(pt); }
  double ret = 0.0;
  if (tn_profile()) pt = std::chrono::steady_clock::now();
  for (int idev = 0; idev < num_devices_used_; idev++)
    ret += gpu_mgr_.device(idev).trace_output(num_qubits);
  if (tn_profile()) { prof_reduce_gpu_ms_ += tn_ms_since(pt); prof_report("contract_and_trace"); }
  return ret;
#endif
}

template <typename data_t>
double TensorNetContractor_HipTensor<data_t>::contract_and_sample_measure(
    reg_t &samples, std::vector<double> &rnds, uint_t num_qubits) {
#ifdef AER_MPI
  // Fail-together guard (see agree_or_fail_together): a per-rank throw in the
  // local contraction must abort the job collectively rather than leaving
  // siblings blocked at accumulate_across_mpi() below.
  int local_failed = 0;
  std::string local_msg;
  try {
    std::chrono::steady_clock::time_point pt;
    if (tn_profile()) pt = std::chrono::steady_clock::now();
    contract_all();
    if (tn_profile()) { prof_sync_devices(); prof_contract_ms_ += tn_ms_since(pt); }
    if (tn_profile()) pt = std::chrono::steady_clock::now();
    accumulate_across_gpus();
    if (tn_profile()) { prof_sync_devices(); prof_reduce_gpu_ms_ += tn_ms_since(pt); }
  } catch (const std::exception &e) {
    local_failed = 1;
    local_msg = e.what();
  }
  agree_or_fail_together(local_failed, local_msg);
  {
    std::chrono::steady_clock::time_point pt;
    if (tn_profile()) pt = std::chrono::steady_clock::now();
    accumulate_across_mpi();
    if (tn_profile()) prof_reduce_mpi_ms_ += tn_ms_since(pt);
  }
#else
  {
    std::chrono::steady_clock::time_point pt;
    if (tn_profile()) pt = std::chrono::steady_clock::now();
    contract_all();
    if (tn_profile()) { prof_sync_devices(); prof_contract_ms_ += tn_ms_since(pt); }
    if (tn_profile()) pt = std::chrono::steady_clock::now();
    accumulate_across_gpus();
    if (tn_profile()) { prof_sync_devices(); prof_reduce_gpu_ms_ += tn_ms_since(pt); }
  }
#endif
  if (tn_profile()) prof_report("contract_and_sample_measure");
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
  // aer-0024: the output spec is part of the topology. The cached plan was
  // searched against network_desc_.output_modes, and the cached step specs
  // carry a final-step C descriptor written in modes_out_ order. Reusing
  // either against a different output is wrong, and now that pool_ready_
  // survives set_network()/set_additional_tensors() it would be silently
  // wrong rather than merely wasteful. Both batch callers hold the output
  // fixed across the batch, so this costs nothing on the paths that matter.
  if (modes_out_ != prev_modes_out_) return false;
  if (extents_out_ != prev_extents_out_) return false;
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
  // aer-0024: record the output this plan and these step specs were built for.
  // Called from the cache-miss branch of setup_contraction(), after
  // network_desc_.output_modes/_extents have been assigned from modes_out_.
  prev_modes_out_ = modes_out_;
  prev_extents_out_ = extents_out_;
  prev_valid_ = true;
}

template <typename data_t>
std::unique_ptr<PathOptimizer>
TensorNetContractor_HipTensor<data_t>::create_optimizer() {
  std::unique_ptr<PathOptimizer> inner;
#ifdef AER_HIPTENSOR
  try {
    size_t elem_bytes = 2 * sizeof(data_t);
    // aer-0035: the planner cannot see the rank count, so hand it the
    // distribution floor here -- this is the only place that knows both.
    // AER_TN_MIN_SLICES_PER_RANK defaults to 0, which leaves min_slices at 1
    // and the planner's behaviour exactly as before.
    uint64_t min_slices = 1;
    if (nprocs_ > 1 && min_slices_per_rank() > 0)
      min_slices = min_slices_per_rank() * static_cast<uint64_t>(nprocs_);
    inner = std::unique_ptr<PathOptimizer>(new CotengPathOptimizer(
        "combo", -1, -1.0, "hyper", elem_bytes, min_slices));
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

  // Fault-injection hook (inert unless set; never a drop-in default, like
  // AER_TN_FORCE_SLICING). AER_TN_FAULT_INJECT_RANK selects the rank that
  // throws mid-contraction; AER_TN_FAULT_INJECT_SLICE optionally pins the
  // slice index (default: the rank's first slice). This deterministically
  // reproduces the per-rank throw that the fail-together guard must convert
  // into a fast collective abort instead of a reduction hang (job 19214242).
  if (const char *fr = getenv("AER_TN_FAULT_INJECT_RANK")) {
    const char *fs = getenv("AER_TN_FAULT_INJECT_SLICE");
    uint_t inj_slice = fs ? (uint_t)strtoull(fs, nullptr, 10) : slice_begin_;
    if (atoi(fr) == myrank_ && slice_index == inj_slice) {
      std::stringstream err;
      err << "[AER_TN_FAULT_INJECT] rank " << myrank_ << " forced throw at slice "
          << slice_index << " (validation hook; unset AER_TN_FAULT_INJECT_RANK "
          << "to disable).";
      throw std::runtime_error(err.str());
    }
  }

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

    // aer-0058: restore the routing flag that the aer-0057 excision of the
    // tiled branch and plan-lookup block took with it (build 21195415:
    // "use of undeclared identifier 'will_route'"). With the classifier
    // throwing a named refusal for any unroutable step, gemm_ok is true for
    // every step that reaches execution -- this flag now only feeds the
    // internal-consistency guard below.
    bool will_route = false;
#ifdef AER_HIPBLAS
    will_route = ps.gemm_ok;
#endif

    {
      size_t c_bytes = tensor_slot_bytes(
          all_specs_[result_idx].num_elements(), sizeof(data_t));
      check_hip(hipMemsetAsync(c_slot, 0, c_bytes, dev.stream()),
                "hipMemsetAsync(C pre-zero)", dev.device_id());
    }

    if (tn_debug()) {
      fprintf(stderr,
              "[AER_TN_DEBUG]   step %zu exec: "
              "A_re=%p B_re=%p C_re=%p modes_c=%s "
              "strided=%c%c\n",
              step, (void *)all_planes[left].re, (void *)all_planes[right].re,
              (void *)all_planes[result_idx].re,
              modes_to_str(ps.modes_c).c_str(),
              ps.strides_a.empty() ? '-' : 'A',
              ps.strides_b.empty() ? '-' : 'B');
    }

    // Split-complex decomposition with explicit plane pointers (OI9
    // resolution): each operand's planes were resolved above — by slice
    // projection for inputs and by slot layout for intermediates.
    //
    // aer-0040: when this step was found to be a matrix product already and
    // AER_TN_GEMM=1, the same four alphas and betas go to hipblas*gemm instead.
    // C was zeroed just above, so accumulate=false matches the hipTensor call
    // it replaces exactly.
    bool routed = false;
#ifdef AER_HIPBLAS
    if (will_route) {
      routed = true;
      std::chrono::steady_clock::time_point prof_g0;
      if (tn_profile())
        prof_g0 = std::chrono::steady_clock::now();
      // aer-0052: a permuted step packs its operands into scratch first (the
      // same stage_pack the WS-3 tile staging uses -- the permutation IS a
      // strided view: destination-order extents with the source's column-major
      // strides reordered by the permutation). C never permutes; the target
      // orders write it straight into its declared slot. Reference planes for
      // the verify arm above are untouched: the pack READS the originals.
      // Non-const: execute_contraction_gemm's signature takes data_t*
      // (hipblas's own pointer types are non-const), and the scratch planes
      // are writable by definition. The pack call below still cannot mutate
      // its source -- stage_pack takes const src pointers.
      data_t *g_a_re = all_planes[left].re;
      data_t *g_a_im = all_planes[left].im;
      data_t *g_b_re = all_planes[right].re;
      data_t *g_b_im = all_planes[right].im;
      // aer-0054: the scratch is 6 planes -- A re/im, B re/im, and C re/im
      // for corder steps, whose GEMM result lands in scratch and is then
      // scattered into the declared slot.
      data_t *g_c_re = all_planes[result_idx].re;
      data_t *g_c_im = all_planes[result_idx].im;
      if (ps.gemm_perm_a || ps.gemm_perm_b || ps.gemm_scatter_c) {
        thrust::device_vector<data_t> &pbuf =
            gemm_perm_scratch_[dev.device_id()];
        const uint64_t pe = gemm_perm_scratch_elems_;
        if (pbuf.size() < 6 * pe)
          pbuf.resize(6 * pe);
        data_t *pbase = thrust::raw_pointer_cast(pbuf.data());
        if (ps.gemm_perm_a) {
          stage_pack<data_t>(dev.stream(), g_a_re, g_a_im, pbase + 0 * pe,
                             pbase + 1 * pe, ps.perm_ext_a, ps.perm_str_a);
          g_a_re = pbase + 0 * pe;
          g_a_im = pbase + 1 * pe;
        }
        if (ps.gemm_perm_b) {
          stage_pack<data_t>(dev.stream(), g_b_re, g_b_im, pbase + 2 * pe,
                             pbase + 3 * pe, ps.perm_ext_b, ps.perm_str_b);
          g_b_re = pbase + 2 * pe;
          g_b_im = pbase + 3 * pe;
        }
        if (ps.gemm_scatter_c) {
          g_c_re = pbase + 4 * pe;
          g_c_im = pbase + 5 * pe;
        }
      }
      if (tn_gemm_complex()) {
        // aer-0055: one native complex GEMM on interleaved scratch. The pack
        // reads the SAME (extents, strides) the real path would -- permuted
        // vectors when the step was rescued, identity packed order when it
        // was a classic accept -- and emits the interleaved layout in the
        // same pass, so the permute costs nothing extra here. Regions of the
        // 6-plane scratch: A at 0 (2*pe data_t = pe complex), B at 2*pe,
        // C at 4*pe. The deinterleave-scatter places C into the declared
        // slot: with the corder scat vectors when set, or as a plain linear
        // deinterleave (ext={c_elems}, str={1}) otherwise -- one pass either
        // way. Verify compares the declared slot afterwards, unchanged.
        thrust::device_vector<data_t> &zbuf =
            gemm_perm_scratch_[dev.device_id()];
        const uint64_t pe = gemm_perm_scratch_elems_;
        if (zbuf.size() < 6 * pe)
          zbuf.resize(6 * pe);
        data_t *zbase = thrust::raw_pointer_cast(zbuf.data());
        std::vector<int64_t> id_ext, id_str;
        auto pick = [&](const std::vector<int64_t> &pext,
                        const std::vector<int64_t> &pstr,
                        const std::vector<int64_t> &full_ext)
            -> std::pair<const std::vector<int64_t> *,
                         const std::vector<int64_t> *> {
          if (!pext.empty())
            return {&pext, &pstr};
          id_ext = full_ext;
          id_str.assign(full_ext.size(), 0);
          int64_t acc = 1;
          for (size_t q = 0; q < full_ext.size(); q++) {
            id_str[q] = acc;
            acc *= (full_ext[q] > 0 ? full_ext[q] : 1);
          }
          return {&id_ext, &id_str};
        };
        {
          auto pr = pick(ps.perm_ext_a, ps.perm_str_a, ps.extents_a);
          interleave_pack<data_t>(dev.stream(), all_planes[left].re,
                                  all_planes[left].im, zbase + 0 * pe,
                                  *pr.first, *pr.second);
        }
        {
          auto pr = pick(ps.perm_ext_b, ps.perm_str_b, ps.extents_b);
          interleave_pack<data_t>(dev.stream(), all_planes[right].re,
                                  all_planes[right].im, zbase + 2 * pe,
                                  *pr.first, *pr.second);
        }
        dev.execute_contraction_gemm_complex(
            ps.gemm_m, ps.gemm_n, ps.gemm_k, ps.gemm_a_trans,
            ps.gemm_b_trans, step, zbase + 0 * pe, zbase + 2 * pe,
            zbase + 4 * pe, false);
        const int64_t c_nat = ps.gemm_m * ps.gemm_n;
        if (ps.gemm_scatter_c) {
          deinterleave_scatter<data_t>(dev.stream(), zbase + 4 * pe,
                                       all_planes[result_idx].re,
                                       all_planes[result_idx].im,
                                       ps.scat_ext, ps.scat_str);
        } else {
          const std::vector<int64_t> lin_ext(1, c_nat);
          const std::vector<int64_t> lin_str(1, 1);
          deinterleave_scatter<data_t>(dev.stream(), zbase + 4 * pe,
                                       all_planes[result_idx].re,
                                       all_planes[result_idx].im,
                                       lin_ext, lin_str);
        }
      } else {
      dev.execute_contraction_gemm(
          ps.gemm_m, ps.gemm_n, ps.gemm_k, ps.gemm_a_trans, ps.gemm_b_trans,
          step,
          g_a_re, g_a_im,
          g_b_re, g_b_im,
          g_c_re, g_c_im, false);
      }
      // Corder: place the natural-order result into the declared slot. The
      // slot was pre-zeroed above; extent-1 declared modes contribute index
      // zero, and the M*N == C-slot invariant makes the scatter a bijection
      // onto the extent>1 lattice, so overwrite (accumulate=false) fills the
      // slot exactly. Runs BEFORE the verify read-back, which compares the
      // declared slot -- so corder steps verify like any other.
      if (ps.gemm_scatter_c && !tn_gemm_complex()) {
        stage_scatter<data_t>(dev.stream(), g_c_re, g_c_im,
                              all_planes[result_idx].re,
                              all_planes[result_idx].im,
                              ps.scat_ext, ps.scat_str, false);
      }
      if (tn_profile()) {
        // Host-side enqueue cost. Deliberately NOT synchronised: see the note
        // on prof_gemm_ms_ at its declaration for why this is not the same
        // quantity as t_contract_ms and must not be read as a fraction of it.
        prof_gemm_ms_ += tn_ms_since(prof_g0);
        prof_gemm_calls_++;
      }

    }
#endif
    if (!routed) {
      // aer-0057: there is no fallback engine. Every step is classified for
      // the GEMM route at setup, and a step that cannot route throws a named
      // refusal THERE (with its modes and the reason); reaching this point
      // unrouted is therefore an internal inconsistency, not a user error.
      throw std::runtime_error(
          "[AER_TN] internal error: step reached execution without a GEMM "
          "routing (the hipTensor dispatch was removed in aer-0057; "
          "classification must refuse loudly at setup instead)");
    }

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
