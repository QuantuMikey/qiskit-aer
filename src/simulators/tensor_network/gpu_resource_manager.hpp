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

#ifndef _gpu_resource_manager_hpp_
#define _gpu_resource_manager_hpp_

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>
#ifdef AER_HIPBLAS
#include <hipblas/hipblas.h>
#endif


#include "misc/wrap_thrust.hpp"
#include "simulators/statevector/chunk/thrust_kernels.hpp"

#include "simulators/tensor_network/path_optimizer.hpp"

namespace thrust_gpu = thrust::hip;

namespace AER {
namespace TensorNetwork {

//=============================================================================
// Alignment
//=============================================================================
// Alignment we promise hipTensor for every tensor pointer we hand it (A, B, C,
// and D). Must be a value that is actually satisfied by every pointer we pass —
// both input tensor slabs and intermediate-pool offsets.
//
// hiptensorGetAlignmentRequirement, when queried on a real pool pointer for
// HIP_C_64F, returns 16 (one complex-double element width). That is the true
// hardware requirement for Composable Kernel's bilinear kernels on gfx90a.
//
// Prior versions of this file used 256, which was wrong: the pool's
// find_offset() only rounds up to this value on overlap bumps, not on every
// allocation, so closely-packed intermediates end up at offsets like +0x80 or
// +0xc0 within a 256-byte-aligned base. Telling hipTensor align=256 while
// passing a 128-byte-aligned pointer caused the kernel to compute wrong
// byte offsets for downstream reads and silently produced zero output
// beyond the first contraction (manifested as the Gate 1.5 Bell-state
// failure: step 0 correct, step 1+ all zero).
//
// 16 matches both the library's actual requirement and what the pool
// naturally produces when sizing allocations in element-granular units.
// If a future hardware target raises this, also update the find_offset()
// rounding in the memory pool to match.
static constexpr uint32_t TENSOR_POINTER_ALIGN = 16;

//=============================================================================
// Split-complex tensor layout
//=============================================================================
// hipTensor 1.5.0 on gfx90a has broken complex contractions: kernels either
// silently no-op or run a real kernel that leaves the imaginary plane
// uninitialized. Real contractions work correctly. To support complex
// semantics we store each tensor's real and imaginary values in separate
// contiguous planes and issue four real contractions per complex
// contraction:
//
//   Dr = Ar*Br - Ai*Bi
//   Di = Ar*Bi + Ai*Br
//
// Each tensor occupies 2 * plane_bytes(N) bytes in the pool. The real plane
// starts at offset 0; the imaginary plane at offset plane_bytes(N). Planes
// are padded up to a 16-byte boundary so the imag plane is properly aligned
// for hipTensor regardless of N's parity.
inline size_t plane_bytes(int64_t num_elements, size_t element_size) {
  size_t raw = static_cast<size_t>(num_elements) * element_size;
  return (raw + 15u) & ~size_t{15u};
}

// Total bytes a single split-complex tensor occupies in the pool.
inline size_t tensor_slot_bytes(int64_t num_elements, size_t element_size) {
  return 2 * plane_bytes(num_elements, element_size);
}

//=============================================================================
// Diagnostic logging
//=============================================================================

static bool gpu_verbose() {
  static bool checked = false;
  static bool enabled = false;
  if (!checked) {
    const char *val = std::getenv("AER_TN_GPU_VERBOSE");
    enabled = (val != nullptr && std::string(val) == "1");
    checked = true;
  }
  return enabled;
}

static bool memory_verbose() {
  static bool checked = false;
  static bool enabled = false;
  if (!checked) {
    const char *val = std::getenv("AER_TN_MEMORY_VERBOSE");
    enabled = (val != nullptr && std::string(val) == "1");
    checked = true;
  }
  return enabled;
}

//=============================================================================
// Error handling
//=============================================================================

inline void check_hip(hipError_t err, const char *func, int device_id = -1) {
  if (err != hipSuccess) {
    std::stringstream ss;
    ss << "HIP error in " << func << ": " << hipGetErrorString(err);
    if (device_id >= 0) ss << " (device " << device_id << ")";
    throw std::runtime_error(ss.str());
  }
}

//=============================================================================
// No-throw GPU capability probe (Piece 0 — CPU-node operability)
//=============================================================================
// On a CPU-only node hipGetDeviceCount returns hipErrorNoDevice. That is a
// normal condition, not an error: callers use this probe to choose CPU vs
// GPU execution deliberately instead of discovering GPU absence via a crash.
// Never throws; never aborts; clears the sticky HIP error so later HIP calls
// are unaffected.
inline int tn_gpu_device_count() noexcept {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess) {
    (void)hipGetLastError();
    return 0;
  }
  return count;
}

inline bool tn_gpu_available() noexcept { return tn_gpu_device_count() > 0; }

//=============================================================================
// ContractionSignature
//=============================================================================

struct ContractionSignature {
  std::vector<int32_t> data;
  bool operator==(const ContractionSignature &other) const {
    return data == other.data;
  }
};

struct ContractionSignatureHash {
  size_t operator()(const ContractionSignature &sig) const {
    size_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < sig.data.size(); i++) {
      hash ^= static_cast<size_t>(sig.data[i]);
      hash *= 1099511628211ULL;
    }
    return hash;
  }
};

inline ContractionSignature
build_signature(const std::vector<int32_t> &modes_a,
                const std::vector<int64_t> &extents_a,
                const std::vector<int32_t> &modes_b,
                const std::vector<int64_t> &extents_b,
                const std::vector<int32_t> &modes_c,
                const std::vector<int64_t> &extents_c,
                const std::vector<int64_t> &strides_a = {},
                const std::vector<int64_t> &strides_b = {},
                const std::vector<int64_t> &strides_c = {}) {
  ContractionSignature sig;
  sig.data.push_back(static_cast<int32_t>(modes_a.size()));
  for (size_t i = 0; i < modes_a.size(); i++) sig.data.push_back(modes_a[i]);
  for (size_t i = 0; i < extents_a.size(); i++) sig.data.push_back(static_cast<int32_t>(extents_a[i]));
  sig.data.push_back(static_cast<int32_t>(modes_b.size()));
  for (size_t i = 0; i < modes_b.size(); i++) sig.data.push_back(modes_b[i]);
  for (size_t i = 0; i < extents_b.size(); i++) sig.data.push_back(static_cast<int32_t>(extents_b[i]));
  sig.data.push_back(static_cast<int32_t>(modes_c.size()));
  for (size_t i = 0; i < modes_c.size(); i++) sig.data.push_back(modes_c[i]);
  for (size_t i = 0; i < extents_c.size(); i++) sig.data.push_back(static_cast<int32_t>(extents_c[i]));
  // Strides distinguish a strided sliced-input view from a packed tensor of
  // identical shape. Empty means packed (descriptor built with nullptr
  // strides — hipTensor's column-major default); a sentinel separates the
  // empty case from an explicit stride list so the two can never alias.
  sig.data.push_back(static_cast<int32_t>(strides_a.size()));
  for (size_t i = 0; i < strides_a.size(); i++) sig.data.push_back(static_cast<int32_t>(strides_a[i]));
  sig.data.push_back(static_cast<int32_t>(strides_b.size()));
  for (size_t i = 0; i < strides_b.size(); i++) sig.data.push_back(static_cast<int32_t>(strides_b[i]));
  // C strides: empty for packed pool-slot C (the common case), non-empty for
  // WS-3 tiled strided-C blocks. Same sentinel discipline as A/B so a packed
  // C can never collide with a strided C of identical mode/extent shape.
  sig.data.push_back(static_cast<int32_t>(strides_c.size()));
  for (size_t i = 0; i < strides_c.size(); i++) sig.data.push_back(static_cast<int32_t>(strides_c[i]));
  return sig;
}

// aer-0052: AER_TN_GEMM_PERMUTE=1 rescues steps the GEMM route declines for
// `layout` (free and contracted modes interleaved) or `korder` (contracted
// modes ordered differently on A and B) by physically packing each operand
// into scratch under a GEMM-compatible mode order before the dispatch. The
// pinned census (job 21011485) counted layout=100 + korder=15 of 150 steps --
// the route's largest untapped population -- but the permute costs one extra
// read+write per operand PER SLICE, which on GEMM-poor steps can exceed the
// GEMM's gain. Default OFF; gate any default flip on the 8/16-GCD scaling
// runs against a pinned plan, never on correctness alone (the correctness is
// closed by construction and by the standalone audit -- see PlanSpec).
// aer-0055: AER_TN_GEMM_COMPLEX=1 executes each routed or rescued step as ONE
// native complex GEMM (hipblasZgemm / hipblasCgemm) on interleaved scratch
// instead of the four real GEMMs of the split-complex decomposition. The pack
// that already reads the operand (permuted or identity order) emits the
// interleaved layout directly, so no storage migration happens -- planes in,
// planes out. Complex correctness of hipblasZgemm on gfx90a was established
// by job 20644277 (50/50 cases, worst difference exactly zero), and the
// interleave -> one-GEMM -> deinterleave equivalence to both the four-GEMM
// decomposition and a mode-label complex reference is CLAIM Z of
// audit_aer0054_design.py (110,392 cases, zero mismatches). Default OFF; the
// A/B against the four-GEMM path gates any default flip.
static bool tn_gemm_complex() {
  static bool checked = false;
  static bool cached = false;
  if (!checked) {
    const char *v = std::getenv("AER_TN_GEMM_COMPLEX");
    cached = (v != nullptr && v[0] == '1' && v[1] == '\0');
    checked = true;
  }
  return cached;
}

// aer-0048: AER_TN_GEMM_ATOMICS=1 restores rocBLAS's DEFAULT atomic reductions
// on the GEMM handle. Default OFF keeps HIPBLAS_ATOMICS_NOT_ALLOWED, which is
// the aer-0041 setting that the deterministic-reduction capability and the
// AER_TN_GEMM_VERIFY exact-equality gate both depend on. This knob exists only
// to ATTRIBUTE the in-situ dispatch gap (job 20666446: ~19 ms per routed call
// against the probe's 25.692 us for four gemms, and the probe ran with atomics
// ALLOWED because it called hipblasCreate and nothing else). Flip it on a
// timing arm; never on a correctness or verify arm, where a non-deterministic
// gemm would break the exact comparison silently on the routed steps only.
static bool tn_gemm_atomics() {
  static bool checked = false;
  static bool cached = false;
  if (!checked) {
    const char *v = std::getenv("AER_TN_GEMM_ATOMICS");
    cached = (v != nullptr && v[0] == '1' && v[1] == '\0');
    checked = true;
  }
  return cached;
}

// aer-0048: AER_TN_GEMM_WARMUP=1 issues one throwaway four-plane dispatch right
// after the handle is created, so rocBLAS/Tensile library initialisation and
// the first Tensile solution lookup are paid at setup rather than on the first
// timed routed call. The other named candidate for the dispatch gap (per-shape
// Tensile selection): if a single warm-up shape closes most of the 175x, the
// cost was library-level lazy init; if it does not, it is per-shape and a
// single warm-up cannot fix it.
//
// aer-0050: default flipped to ON. The pinned 4-arm census (job 21011485, all
// arms replaying the identical 150-step/8-slice plan) attributed the gap:
// warm-up alone collapsed t_gemm_ms 471.6 -> 53.0 ms (t_contract 653.3 ->
// 231.6 ms, 2.82x), and atomics added NOTHING once warm (arm AW 238.4 /
// 53.6 ms, within noise of arm W) -- so the warm-up belongs on the route
// unconditionally and HIPBLAS_ATOMICS_NOT_ALLOWED (bitwise determinism) is
// kept for free. The residual ~414 us/call against the isolated 7.4 us floor
// is per-shape Tensile selection, which one warm-up cannot fix (measured, not
// assumed). AER_TN_GEMM_WARMUP=0 restores the aer-0048 lazy behaviour for
// A/B; any other value (including 1) leaves it on.
static bool tn_gemm_warmup() {
  static bool checked = false;
  static bool cached = true;
  if (!checked) {
    const char *v = std::getenv("AER_TN_GEMM_WARMUP");
    if (v != nullptr && v[0] == '0' && v[1] == '\0')
      cached = false;
    checked = true;
  }
  return cached;
}

//=============================================================================
// MemoryPool
//=============================================================================

struct PoolAllocation {
  size_t offset;
  size_t size;
  int birth_step;
  int death_step;
  int tensor_index;
};

class MemoryPool {
  void *pool_ptr_;
  size_t pool_size_;
  size_t capacity_;    // aer-0070: arena high-water (bytes actually malloc'd)
  uint64_t epoch_;     // aer-0070: base-pointer generation
  int device_id_;
  bool allocated_;
  std::vector<PoolAllocation> allocations_;

public:
  MemoryPool() : pool_ptr_(nullptr), pool_size_(0), capacity_(0), epoch_(0),
                 device_id_(0), allocated_(false) {}
  ~MemoryPool() { release(); }
  MemoryPool(const MemoryPool &) = delete;
  MemoryPool &operator=(const MemoryPool &) = delete;
  MemoryPool(MemoryPool &&other) noexcept
      : pool_ptr_(other.pool_ptr_), pool_size_(other.pool_size_),
        capacity_(other.capacity_), epoch_(other.epoch_),
        device_id_(other.device_id_), allocated_(other.allocated_),
        allocations_(std::move(other.allocations_)) {
    other.pool_ptr_ = nullptr;
    other.allocated_ = false;
    other.capacity_ = 0;
  }

  void plan_layout(
      const std::vector<std::tuple<size_t, int, int, int>> &intermediates,
      int num_steps) {
    allocations_.clear();

    std::vector<size_t> order(intermediates.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) {
                return std::get<0>(intermediates[a]) >
                       std::get<0>(intermediates[b]);
              });

    for (size_t oi = 0; oi < order.size(); oi++) {
      size_t idx = order[oi];
      size_t sz = std::get<0>(intermediates[idx]);
      int birth = std::get<1>(intermediates[idx]);
      int death = std::get<2>(intermediates[idx]);
      int tensor_idx = std::get<3>(intermediates[idx]);
      if (sz == 0) continue;

      size_t offset = find_offset(sz, birth, death);
      PoolAllocation alloc;
      alloc.offset = offset; alloc.size = sz;
      alloc.birth_step = birth; alloc.death_step = death;
      alloc.tensor_index = tensor_idx;
      allocations_.push_back(alloc);
    }

    pool_size_ = 0;
    for (size_t i = 0; i < allocations_.size(); i++)
      pool_size_ = std::max(pool_size_, allocations_[i].offset + allocations_[i].size);
    pool_size_ = ((pool_size_ + TENSOR_POINTER_ALIGN - 1) / TENSOR_POINTER_ALIGN) * TENSOR_POINTER_ALIGN;

    if (memory_verbose()) {
      fprintf(stderr, "[AER_TN_MEMORY] pool: %zu allocs, %zu bytes (%.2f MB)\n",
              allocations_.size(), pool_size_, pool_size_ / (1024.0 * 1024.0));
    }
  }

  // aer-0070: grow-only arena, mirroring copy_tensor_data's existing slab
  // pattern. If the new layout fits the current capacity, the base pointer
  // is INTENTIONALLY kept -- get_tensor_ptr() offsets are recomputed per
  // plan_layout(), so a returning topology sees identical slot addresses,
  // which is what lets aer-0069's captured graphs survive a class switch.
  // A larger layout frees and re-mallocs, bumping the epoch so every baked
  // pointer is declared dead. This also fixes a leak: the previous code
  // hipMalloc'd a fresh pool per topology cache-miss without freeing the
  // old one. AER_TN_POOL_ARENA=0 restores alloc-per-topology (leak still
  // fixed): the epoch then moves on every switch, which reproduces the
  // pre-0070 graph lifetime for A/B purposes.
  static bool arena_enabled() {
    static bool checked = false;
    static bool cached = true;
    if (!checked) {
      const char *v = std::getenv("AER_TN_POOL_ARENA");
      cached = !(v != nullptr && v[0] == '0');
      checked = true;
    }
    return cached;
  }

  void allocate(int device_id) {
    device_id_ = device_id;
    if (pool_size_ == 0) return;
    if (allocated_ && pool_ptr_ != nullptr && arena_enabled() &&
        pool_size_ <= capacity_)
      return;
    hipSetDevice(device_id_);
    if (allocated_ && pool_ptr_ != nullptr) {
      hipFree(pool_ptr_);
      pool_ptr_ = nullptr;
    }
    check_hip(hipMalloc(&pool_ptr_, pool_size_), "hipMalloc(pool)", device_id_);
    capacity_ = pool_size_;
    epoch_++;
    allocated_ = true;
  }

  // aer-0070: consumers holding device addresses derived from this pool
  // (captured graphs) compare this before trusting them.
  uint64_t epoch() const { return epoch_; }

  void *get_tensor_ptr(int tensor_index) const {
    for (size_t i = 0; i < allocations_.size(); i++)
      if (allocations_[i].tensor_index == tensor_index)
        return static_cast<char *>(pool_ptr_) + allocations_[i].offset;
    std::stringstream ss;
    ss << "MemoryPool: tensor index " << tensor_index << " not found";
    throw std::runtime_error(ss.str());
  }

  size_t total_size() const { return pool_size_; }
  bool is_allocated() const { return allocated_; }

  void release() {
    if (allocated_ && pool_ptr_) {
      hipSetDevice(device_id_);
      hipFree(pool_ptr_);
      pool_ptr_ = nullptr;
      capacity_ = 0;
      epoch_++;
      allocated_ = false;
    }
    allocations_.clear();
    pool_size_ = 0;
  }

private:
  size_t find_offset(size_t size, int birth, int death) {
    size_t offset = 0;
    bool placed = false;
    while (!placed) {
      placed = true;
      for (size_t i = 0; i < allocations_.size(); i++) {
        bool lt_overlap = (birth <= allocations_[i].death_step &&
                          death >= allocations_[i].birth_step);
        if (!lt_overlap) continue;
        bool sp_overlap = (offset < allocations_[i].offset + allocations_[i].size &&
                          offset + size > allocations_[i].offset);
        if (sp_overlap) {
          offset = allocations_[i].offset + allocations_[i].size;
          offset = ((offset + TENSOR_POINTER_ALIGN - 1) / TENSOR_POINTER_ALIGN) * TENSOR_POINTER_ALIGN;
          placed = false;
          break;
        }
      }
    }
    return offset;
  }
};

//=============================================================================
// AccumulatePlanarFunctor
//=============================================================================
// Adds a split-complex tensor into an interleaved thrust::complex output
// buffer. Used by GPUDevice::accumulate_planar_to_output. We use an
// explicit functor rather than a device lambda to match the Aer build's
// existing pattern and avoid extended-lambda compile flag dependencies.
template <typename data_t> struct AccumulatePlanarFunctor {
  thrust::complex<data_t> *out;
  const data_t *re;
  const data_t *im;
  AccumulatePlanarFunctor(thrust::complex<data_t> *o, const data_t *r,
                          const data_t *i)
      : out(o), re(r), im(i) {}
  __host__ __device__ void operator()(const size_t &i) const {
    out[i] += thrust::complex<data_t>(re[i], im[i]);
  }
};

//=============================================================================
// GPUDevice
//=============================================================================

#ifdef AER_HIPBLAS
//=============================================================================
// aer-0040: the GEMM contraction route.
//
// A contraction whose A and B are PACKED column-major and whose free and
// contracted modes each occupy one contiguous run of the declared mode order
// IS a matrix product already -- no permute, no copy, only a reinterpretation
// of the same bytes. stride_map (tensor_net_contractor_hiptensor.hpp:3020)
// gives stride 1 to modes[0] and multiplies forward, so a packed tensor whose
// order is [free...][contracted...] is an (F x C) column-major matrix with
// leading dimension F, and one whose order is [contracted...][free...] is its
// transpose. HIPBLAS_OP_N / HIPBLAS_OP_T cover both, which is why no permute
// layer is needed for the steps this route accepts.
//
// FOUR REAL GEMMs, NOT ONE COMPLEX GEMM. This backend stores complex values as
// two separate real planes (historically for hipTensor's broken complex path;
// kept on the route's own merits), and
// execute_contraction_gemm below takes six plane pointers. hipblasZgemm
// requires interleaved complex, which this layout is not, and interleaving it
// would mean changing the pool, the upload, project_slice's plane arithmetic
// and accumulate_planar_to_output -- the machinery the OI9 comment identifies
// as what makes slicing and split-complex compose. Job 20644277 measured both
// forms on one GCD at 64^3: four hipblasDgemm at 25.692 us against the
// memset-plus-four-hiptensorContraction path at 117.007 us, so the split form
// alone is 4.55x with no layout change at all. hipblasZgemm's 6.923 us is a
// separate decision about storage layout and is deliberately not taken here.
//
// The alphas and betas are the same four this backend already uses:
//   Cr  = Ar*Br      beta = accumulate ? 1 : 0
//   Cr -= Ai*Bi      beta = 1
//   Ci  = Ar*Bi      beta = accumulate ? 1 : 0
//   Ci += Ai*Br      beta = 1
// so accumulation semantics are unchanged. beta=1 is already how per-tile
// K-accumulation works today; this route moves nothing.
//=============================================================================

inline void check_hipblas(hipblasStatus_t st, const char *what, int device) {
  if (st != HIPBLAS_STATUS_SUCCESS) {
    std::stringstream err;
    err << "[AER_TN] " << what << " failed on device " << device
        << " with hipblasStatus_t " << static_cast<int>(st) << ".";
    throw std::runtime_error(err.str());
  }
}

// hipblas*gemm takes int for m/n/k/lda/ldb/ldc while this backend carries
// int64_t extents throughout. The narrowing is explicit and loud.
inline int hipblas_narrow(int64_t v, const char *what, size_t step) {
  if (v <= 0 || v > static_cast<int64_t>(2147483647)) {
    std::stringstream err;
    err << "[AER_TN] GEMM route: " << what << " = " << v << " at step " << step
        << " does not fit the int the hipblas*gemm interface takes. There "
           "is no fallback engine (aer-0057); tighten the per-slice budget "
           "(SS_SLICE_BYTES / AER_TN_SLICE_TARGET_BYTES) so the slicer cuts "
           "this step smaller, or reduce circuit width/depth.";
    throw std::runtime_error(err.str());
  }
  return static_cast<int>(v);
}

template <typename data_t> struct HipblasGemm;

template <> struct HipblasGemm<double> {
  static hipblasStatus_t call(hipblasHandle_t h, hipblasOperation_t ta,
                              hipblasOperation_t tb, int m, int n, int k,
                              const double *alpha, const double *A, int lda,
                              const double *B, int ldb, const double *beta,
                              double *C, int ldc) {
    return hipblasDgemm(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C,
                        ldc);
  }
};

template <> struct HipblasGemm<float> {
  static hipblasStatus_t call(hipblasHandle_t h, hipblasOperation_t ta,
                              hipblasOperation_t tb, int m, int n, int k,
                              const float *alpha, const float *A, int lda,
                              const float *B, int ldb, const float *beta,
                              float *C, int ldc) {
    return hipblasSgemm(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C,
                        ldc);
  }
};

// aer-0055: the native-complex dispatch. The pointers are INTERLEAVED
// complex buffers (re, im, re, im, ...), which is exactly hipblas's
// hipblasDoubleComplex / hipblasComplex memory layout, so the casts below
// are layout-preserving reinterpretations, not conversions.
template <typename data_t> struct HipblasGemmC;

template <> struct HipblasGemmC<double> {
  typedef hipblasDoubleComplex cplx;
  static hipblasStatus_t call(hipblasHandle_t h, hipblasOperation_t ta,
                              hipblasOperation_t tb, int m, int n, int k,
                              const cplx *alpha, const cplx *A, int lda,
                              const cplx *B, int ldb, const cplx *beta,
                              cplx *C, int ldc) {
    return hipblasZgemm(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C,
                        ldc);
  }
};

template <> struct HipblasGemmC<float> {
  typedef hipblasComplex cplx;
  static hipblasStatus_t call(hipblasHandle_t h, hipblasOperation_t ta,
                              hipblasOperation_t tb, int m, int n, int k,
                              const cplx *alpha, const cplx *A, int lda,
                              const cplx *B, int ldb, const cplx *beta,
                              cplx *C, int ldc) {
    return hipblasCgemm(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C,
                        ldc);
  }
};
#endif // AER_HIPBLAS

template <typename data_t> class GPUDevice {
  int device_id_;
  std::string architecture_;
  size_t total_memory_;
  size_t free_memory_;
  hipStream_t stream_;
  bool handle_valid_;
#ifdef AER_HIPBLAS
  // aer-0040: the device's GEMM handle, destroyed in release() alongside
  // the stream. Never shared across contractors: a hipBLAS handle owns no
  // plans, so there is nothing to keep alive.
  hipblasHandle_t bl_handle_;
  bool bl_handle_valid_;
#endif

  // aer-0028: when shared-plan-cache mode is on this points at the process-wide
  // slot for this device and plan_cache_ above goes unused. The pool, the
  // arena, the stream and the thrust buffers are deliberately NOT shared --
  // they own device memory whose lifetime is already correct per contractor,
  // and only the handle and the plans need to outlive one.
  MemoryPool pool_;
  std::vector<bool> peer_access_;
  void *tensor_data_ptr_;
  size_t tensor_data_size_;
  uint64_t tensor_data_epoch_ = 0; // aer-0070: slab base generation

  thrust::device_vector<thrust::complex<data_t>> dev_out_;
  thrust::device_vector<double> sampling_rnds_;
  thrust::device_vector<uint64_t> sampling_out_;

public:
  GPUDevice()
      : device_id_(-1), total_memory_(0), free_memory_(0), stream_(nullptr),
        handle_valid_(false),
#ifdef AER_HIPBLAS
        bl_handle_(nullptr), bl_handle_valid_(false),
#endif
        tensor_data_ptr_(nullptr), tensor_data_size_(0) {}
  ~GPUDevice() { release(); }
  GPUDevice(const GPUDevice &) = delete;
  GPUDevice &operator=(const GPUDevice &) = delete;

  void init(int device_id, int total_device_count) {
    device_id_ = device_id;
    hipSetDevice(device_id_);

    hipDeviceProp_t props;
    check_hip(hipGetDeviceProperties(&props, device_id_),
              "hipGetDeviceProperties", device_id_);
    architecture_ = props.gcnArchName;
    total_memory_ = props.totalGlobalMem;

    size_t total;
    check_hip(hipMemGetInfo(&free_memory_, &total), "hipMemGetInfo", device_id_);

    check_hip(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
              "hipStreamCreateWithFlags", device_id_);

#ifdef AER_HIPBLAS
    {
      check_hipblas(hipblasCreate(&bl_handle_), "hipblasCreate", device_id_);
      bl_handle_valid_ = true;
      // Same stream as every device call on this device, so the GEMM route
      // keeps the existing ordering and the existing one-sync-per-phase
      // profiling model stays valid.
      check_hipblas(hipblasSetStream(bl_handle_, stream_), "hipblasSetStream",
                    device_id_);
      // aer-0041: ATOMICS OFF. The hipBLAS API reference states that some
      // functions use atomic operations for performance, that this "may cause
      // functions to not give bit-wise reproducible results", and that the
      // rocBLAS backend ALLOWS atomics BY DEFAULT. Two things in this project
      // depend on that not happening:
      //
      //   1. "Deterministic reduction at a fixed rank count" is a documented
      //      user-visible capability of this fork -- repeated runs at the same
      //      rank count give bit-identical answers, which matters because an
      //      optimiser differentiating an expectation value follows a
      //      different trajectory if the last bits move. A non-deterministic
      //      gemm would break that silently, on the routed steps only.
      //   2. AER_TN_GEMM_VERIFY compares the routed result against the
      //      hiptensorContraction result with EXACT equality, deliberately:
      //      a tolerance would hide a transposed operand whose error happens
      //      to be small. That gate is only sound if the gemm is reproducible.
      //
      // The cost is whatever atomics were buying on these shapes, which is
      // unmeasured. Determinism is a shipped capability and speed is the thing
      // being evaluated, so the capability wins by default. If atomics turn
      // out to be worth a lot, that is a separate, measured decision with its
      // own knob -- not a silent default.
      // aer-0048: default NOT_ALLOWED, unchanged; AER_TN_GEMM_ATOMICS=1 flips it
      // to ALLOWED for the dispatch-gap attribution arm only.
      check_hipblas(
          hipblasSetAtomicsMode(bl_handle_,
                                tn_gemm_atomics() ? HIPBLAS_ATOMICS_ALLOWED
                                                  : HIPBLAS_ATOMICS_NOT_ALLOWED),
          "hipblasSetAtomicsMode", device_id_);
      // aer-0048: pay rocBLAS/Tensile lazy init here, off the timed path.
      if (tn_gemm_warmup())
        gemm_warmup();
    }
#endif

    peer_access_.resize(total_device_count, false);
    for (int other = 0; other < total_device_count; other++) {
      if (other == device_id_) continue;
      int can_access = 0;
      hipDeviceCanAccessPeer(&can_access, device_id_, other);
      peer_access_[other] = (can_access != 0);
    }

    if (gpu_verbose()) {
      fprintf(stderr, "[AER_TN_GPU] device %d: %s, %.1f GB total, %.1f GB free\n",
              device_id_, architecture_.c_str(),
              total_memory_ / (1024.0 * 1024.0 * 1024.0),
              free_memory_ / (1024.0 * 1024.0 * 1024.0));
    }
  }

  std::vector<void *> copy_tensor_data(
      const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors,
      bool add_sp_tensors) {
    hipSetDevice(device_id_);

    // Split-complex tensor layout. For each tensor we reserve
    // 2 * plane_bytes(N, sizeof(data_t)) bytes — enough for a real plane
    // and an imag plane, with the imag plane 16-byte-aligned relative to
    // the tensor slot start. Total device bytes is the sum of slot sizes.
    size_t total_bytes = 0;
    for (size_t i = 0; i < tensors.size(); i++)
      if (add_sp_tensors || !tensors[i]->sp_tensor())
        total_bytes += tensor_slot_bytes(
            static_cast<int64_t>(tensors[i]->tensor().size()), sizeof(data_t));

    if (tensor_data_size_ < total_bytes) {
      if (tensor_data_ptr_) hipFree(tensor_data_ptr_);
      check_hip(hipMalloc(&tensor_data_ptr_, total_bytes),
                "hipMalloc(tensor_data)", device_id_);
      tensor_data_size_ = total_bytes;
      // aer-0070: slab base moved; input pointers baked into captured
      // graphs are dead. Same-or-smaller layouts keep the base, and the
      // per-tensor offsets are a deterministic function of the tensor
      // sequence -- a returning topology's H2D refresh lands in the SAME
      // slots, which is why captured graphs stay valid across angle
      // changes (rebinding refreshes DATA, not pointers).
      tensor_data_epoch_++;
    }

    std::vector<void *> ptrs;
    size_t byte_offset = 0;
    for (size_t i = 0; i < tensors.size(); i++) {
      if (add_sp_tensors || !tensors[i]->sp_tensor()) {
        size_t n = tensors[i]->tensor().size();
        size_t pb = plane_bytes(static_cast<int64_t>(n), sizeof(data_t));
        void *dst = static_cast<char *>(tensor_data_ptr_) + byte_offset;
        ptrs.push_back(dst);

        // Host tensor data is interleaved std::complex<data_t>. Split into
        // real and imag planes via two strided H2D copies. hipMemcpy2DAsync
        // semantics: (dst, dpitch, src, spitch, width, height, kind).
        // width = sizeof(data_t) (one scalar per row), height = n.
        const char *src_host =
            reinterpret_cast<const char *>(tensors[i]->tensor().data());

        // Real plane: dst at offset 0, src starts at byte 0, src stride
        // sizeof(std::complex<data_t>) skips the imag word each row.
        check_hip(hipMemcpy2DAsync(
            dst, sizeof(data_t),
            src_host, sizeof(std::complex<data_t>),
            sizeof(data_t), n,
            hipMemcpyHostToDevice, stream_),
            "hipMemcpy2DAsync(tensor_data_re)", device_id_);

        // Imag plane: dst at offset plane_bytes, src starts at the imag
        // half of the first complex word (sizeof(data_t) bytes in), stride
        // sizeof(std::complex<data_t>).
        check_hip(hipMemcpy2DAsync(
            static_cast<char *>(dst) + pb, sizeof(data_t),
            src_host + sizeof(data_t), sizeof(std::complex<data_t>),
            sizeof(data_t), n,
            hipMemcpyHostToDevice, stream_),
            "hipMemcpy2DAsync(tensor_data_im)", device_id_);

        byte_offset += tensor_slot_bytes(
            static_cast<int64_t>(n), sizeof(data_t));
      }
    }
    hipStreamSynchronize(stream_);
    size_t total;
    check_hip(hipMemGetInfo(&free_memory_, &total), "hipMemGetInfo", device_id_);
    return ptrs;
  }

  uint64_t tensor_data_epoch() const { return tensor_data_epoch_; }

  void copy_tensor_data_from(const GPUDevice<data_t> &src) {
    hipSetDevice(device_id_);
    size_t bytes = src.tensor_data_size_;
    if (tensor_data_size_ < bytes) {
      if (tensor_data_ptr_) hipFree(tensor_data_ptr_);
      check_hip(hipMalloc(&tensor_data_ptr_, bytes),
                "hipMalloc(tensor_data_peer)", device_id_);
      tensor_data_size_ = bytes;
    }
    if (peer_access_[src.device_id_]) {
      if (hipDeviceEnablePeerAccess(src.device_id_, 0) != hipSuccess)
        hipGetLastError();
      check_hip(hipMemcpyPeerAsync(tensor_data_ptr_, device_id_,
                    src.tensor_data_ptr_, src.device_id_, bytes, stream_),
          "hipMemcpyPeerAsync", device_id_);
    } else {
      std::vector<char> host_buf(bytes);
      hipSetDevice(src.device_id_);
      check_hip(hipMemcpy(host_buf.data(), src.tensor_data_ptr_, bytes,
                    hipMemcpyDeviceToHost), "hipMemcpy(D2H)", src.device_id_);
      hipSetDevice(device_id_);
      check_hip(hipMemcpyAsync(tensor_data_ptr_, host_buf.data(), bytes,
                    hipMemcpyHostToDevice, stream_), "hipMemcpy(H2D)", device_id_);
    }
    hipStreamSynchronize(stream_);
  }

  void allocate_output(size_t num_elements) {
    hipSetDevice(device_id_);
    dev_out_.resize(num_elements);
  }

  void allocate_sampling_buffers(size_t num_samples) {
    hipSetDevice(device_id_);
    sampling_rnds_.resize(num_samples);
    sampling_out_.resize(num_samples);
  }

  void deallocate_sampling_buffers() {
    hipSetDevice(device_id_);
    sampling_rnds_.clear(); sampling_rnds_.shrink_to_fit();
    sampling_out_.clear(); sampling_out_.shrink_to_fit();
  }

#ifdef AER_HIPBLAS
  // aer-0048: one throwaway routed-shape dispatch to pay rocBLAS/Tensile lazy
  // initialisation at handle-creation time, off the timed path. Self-contained:
  // allocates its own small scratch, runs the SAME four-gemm split-complex form
  // execute_contraction_gemm uses, synchronises so the init actually completes,
  // and frees. Shape 8x8x8 is inside the m6n6k6 routed envelope and large
  // enough to select a non-trivial Tensile solution. Errors are swallowed: a
  // warm-up that fails must not take down a run whose real dispatches would
  // have succeeded. Only reached from init() when
  // tn_gemm_warmup(); default off leaves the handle exactly as before.
  void gemm_warmup() {
    hipSetDevice(device_id_);
    const int wm = 8, wn = 8, wk = 8;
    const size_t na = (size_t)wm * wk, nb = (size_t)wk * wn, nc = (size_t)wm * wn;
    void *Ar = nullptr, *Ai = nullptr, *Br = nullptr, *Bi = nullptr,
         *Cr = nullptr, *Ci = nullptr;
    bool ok = (hipMalloc(&Ar, na * sizeof(data_t)) == hipSuccess) &&
              (hipMalloc(&Ai, na * sizeof(data_t)) == hipSuccess) &&
              (hipMalloc(&Br, nb * sizeof(data_t)) == hipSuccess) &&
              (hipMalloc(&Bi, nb * sizeof(data_t)) == hipSuccess) &&
              (hipMalloc(&Cr, nc * sizeof(data_t)) == hipSuccess) &&
              (hipMalloc(&Ci, nc * sizeof(data_t)) == hipSuccess);
    if (ok) {
      hipMemsetAsync(Ar, 0, na * sizeof(data_t), stream_);
      hipMemsetAsync(Ai, 0, na * sizeof(data_t), stream_);
      hipMemsetAsync(Br, 0, nb * sizeof(data_t), stream_);
      hipMemsetAsync(Bi, 0, nb * sizeof(data_t), stream_);
      const data_t pos_one = static_cast<data_t>(1.0);
      const data_t neg_one = static_cast<data_t>(-1.0);
      const data_t zero = static_cast<data_t>(0.0);
      data_t *ar = static_cast<data_t *>(Ar), *ai = static_cast<data_t *>(Ai);
      data_t *br = static_cast<data_t *>(Br), *bi = static_cast<data_t *>(Bi);
      data_t *cr = static_cast<data_t *>(Cr), *ci = static_cast<data_t *>(Ci);
      HipblasGemm<data_t>::call(bl_handle_, HIPBLAS_OP_N, HIPBLAS_OP_N, wm, wn,
                                wk, &pos_one, ar, wm, br, wk, &zero, cr, wm);
      HipblasGemm<data_t>::call(bl_handle_, HIPBLAS_OP_N, HIPBLAS_OP_N, wm, wn,
                                wk, &neg_one, ai, wm, bi, wk, &pos_one, cr, wm);
      HipblasGemm<data_t>::call(bl_handle_, HIPBLAS_OP_N, HIPBLAS_OP_N, wm, wn,
                                wk, &pos_one, ar, wm, bi, wk, &zero, ci, wm);
      HipblasGemm<data_t>::call(bl_handle_, HIPBLAS_OP_N, HIPBLAS_OP_N, wm, wn,
                                wk, &pos_one, ai, wm, br, wk, &pos_one, ci, wm);
      // aer-0055: the complex GEMM has its own Tensile lazy-init; warm it too
      // when the complex sub-path is enabled. 4x4x4 complex needs 16 complex
      // = 32 data_t per operand, within the 64 data_t each buffer holds.
      if (tn_gemm_complex()) {
        typedef typename HipblasGemmC<data_t>::cplx cplx;
        const cplx cone{pos_one, zero};
        const cplx czero{zero, zero};
        HipblasGemmC<data_t>::call(bl_handle_, HIPBLAS_OP_N, HIPBLAS_OP_N, 4,
                                   4, 4, &cone,
                                   reinterpret_cast<const cplx *>(ar), 4,
                                   reinterpret_cast<const cplx *>(br), 4,
                                   &czero, reinterpret_cast<cplx *>(cr), 4);
      }
      hipStreamSynchronize(stream_);
    }
    hipFree(Ar); hipFree(Ai); hipFree(Br); hipFree(Bi);
    hipFree(Cr); hipFree(Ci);
  }

  // aer-0040: the contraction executor. Six plane pointers; four
  // hipblas*gemm calls per step (or one native Z/Cgemm on the aer-0055
  // sub-path). No workspace: GEMM needs none.
  //
  // a_trans / b_trans say which of the two admissible packed layouts each
  // operand is in, decided in setup_pool_and_cache and carried on PlanSpec:
  //   a_trans == false: A is [M-modes][K-modes], an (M x K) matrix, lda = M
  //   a_trans == true : A is [K-modes][M-modes], so A^T is wanted, lda = K
  //   b_trans == false: B is [K-modes][N-modes], a (K x N) matrix, ldb = K
  //   b_trans == true : B is [N-modes][K-modes], so B^T is wanted, ldb = N
  // C is always a packed pool slot in [M-modes][N-modes] order, which is what
  // compute_contraction_result produces, so ldc = M with no transpose.
  void execute_contraction_gemm(int64_t M64, int64_t N64, int64_t K64,
                                bool a_trans, bool b_trans, size_t step,
                                data_t *Ar, data_t *Ai,
                                data_t *Br, data_t *Bi,
                                data_t *Cr, data_t *Ci,
                                bool accumulate) {
    hipSetDevice(device_id_);
    const int m = hipblas_narrow(M64, "M", step);
    const int n = hipblas_narrow(N64, "N", step);
    const int k = hipblas_narrow(K64, "K", step);
    const int lda = a_trans ? k : m;
    const int ldb = b_trans ? n : k;
    const int ldc = m;
    const hipblasOperation_t ta = a_trans ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    const hipblasOperation_t tb = b_trans ? HIPBLAS_OP_T : HIPBLAS_OP_N;

    const data_t pos_one = static_cast<data_t>(1.0);
    const data_t neg_one = static_cast<data_t>(-1.0);
    const data_t zero = static_cast<data_t>(0.0);
    const data_t initial_beta = accumulate ? pos_one : zero;

    check_hipblas(HipblasGemm<data_t>::call(bl_handle_, ta, tb, m, n, k,
                                            &pos_one, Ar, lda, Br, ldb,
                                            &initial_beta, Cr, ldc),
                  "hipblas gemm(Cr=Ar*Br)", device_id_);
    check_hipblas(HipblasGemm<data_t>::call(bl_handle_, ta, tb, m, n, k,
                                            &neg_one, Ai, lda, Bi, ldb,
                                            &pos_one, Cr, ldc),
                  "hipblas gemm(Cr-=Ai*Bi)", device_id_);
    check_hipblas(HipblasGemm<data_t>::call(bl_handle_, ta, tb, m, n, k,
                                            &pos_one, Ar, lda, Bi, ldb,
                                            &initial_beta, Ci, ldc),
                  "hipblas gemm(Ci=Ar*Bi)", device_id_);
    check_hipblas(HipblasGemm<data_t>::call(bl_handle_, ta, tb, m, n, k,
                                            &pos_one, Ai, lda, Br, ldb,
                                            &pos_one, Ci, ldc),
                  "hipblas gemm(Ci+=Ai*Br)", device_id_);
  }

  // aer-0055: one native complex GEMM on interleaved buffers, replacing the
  // four real dispatches above. Aiv/Biv/Civ point at interleaved complex
  // scratch (see the 6-plane GEMM scratch: A at 0, B at 2*pe, C at 4*pe,
  // each region 2*pe data_t = pe complex elements).
  void execute_contraction_gemm_complex(int64_t M64, int64_t N64, int64_t K64,
                                        bool a_trans, bool b_trans,
                                        size_t step,
                                        const data_t *Aiv, const data_t *Biv,
                                        data_t *Civ, bool accumulate) {
    hipSetDevice(device_id_);
    const int m = hipblas_narrow(M64, "M", step);
    const int n = hipblas_narrow(N64, "N", step);
    const int k = hipblas_narrow(K64, "K", step);
    const int lda = a_trans ? k : m;
    const int ldb = b_trans ? n : k;
    const int ldc = m;
    const hipblasOperation_t ta = a_trans ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    const hipblasOperation_t tb = b_trans ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    typedef typename HipblasGemmC<data_t>::cplx cplx;
    const cplx one{static_cast<data_t>(1.0), static_cast<data_t>(0.0)};
    const cplx beta{accumulate ? static_cast<data_t>(1.0)
                               : static_cast<data_t>(0.0),
                    static_cast<data_t>(0.0)};
    check_hipblas(HipblasGemmC<data_t>::call(
                      bl_handle_, ta, tb, m, n, k, &one,
                      reinterpret_cast<const cplx *>(Aiv), lda,
                      reinterpret_cast<const cplx *>(Biv), ldb, &beta,
                      reinterpret_cast<cplx *>(Civ), ldc),
                  "hipblas gemm complex(C=A*B)", device_id_);
  }
#endif // AER_HIPBLAS

  // Accumulate a split-complex tensor (stored with real plane followed by
  // imag plane, each of num_elements data_t values) into the interleaved
  // thrust complex output buffer dev_out_. Used at the end of each slice
  // contraction to add the final slice's contribution to the running
  // total.
  void accumulate_planar_to_output(void *planar_tensor_ptr,
                                   size_t num_elements) {
    hipSetDevice(device_id_);
    const size_t pb = plane_bytes(
        static_cast<int64_t>(num_elements), sizeof(data_t));
    const data_t *re = reinterpret_cast<const data_t *>(planar_tensor_ptr);
    const data_t *im = reinterpret_cast<const data_t *>(
        reinterpret_cast<const char *>(planar_tensor_ptr) + pb);

    thrust::complex<data_t> *out =
        thrust::raw_pointer_cast(dev_out_.data());

    AccumulatePlanarFunctor<data_t> fn(out, re, im);
    thrust::for_each_n(
        thrust_gpu::par.on(stream_),
        thrust::counting_iterator<size_t>(0),
        num_elements,
        fn);
  }

  void get_output(std::vector<std::complex<data_t>> &out) {
    hipSetDevice(device_id_);
    size_t n = dev_out_.size();
    if (out.size() < n) out.resize(n);
    check_hip(hipMemcpyAsync(out.data(),
                  thrust::raw_pointer_cast(dev_out_.data()),
                  n * sizeof(std::complex<data_t>),
                  hipMemcpyDeviceToHost, stream_),
        "hipMemcpyAsync(output D2H)", device_id_);
    hipStreamSynchronize(stream_);
  }

  double trace_output(uint64_t num_qubits) {
    hipSetDevice(device_id_);
    uint64_t stride = (1ULL << num_qubits) + 1;
    auto *base = (thrust::complex<data_t> *)thrust::raw_pointer_cast(dev_out_.data());
    QV::Chunk::strided_range<thrust::complex<data_t> *> iter(
        base, base + dev_out_.size(), stride);
    thrust::complex<data_t> ret = thrust::reduce(
        thrust_gpu::par.on(stream_), iter.begin(), iter.end());
    return ret.real();
  }

  int device_id() const { return device_id_; }
  const std::string &architecture() const { return architecture_; }
  size_t free_memory() const { return free_memory_; }
  size_t total_memory() const { return total_memory_; }
  bool has_peer_access(int other_device) const { return peer_access_[other_device]; }
  hipStream_t stream() const { return stream_; }

  // aer-0072: replace the device stream after a failed graph capture. A
  // capture aborted mid-flight (job 21451954: an instance-local scratch
  // resize threw std::bad_alloc under capture) can leave the stream in a
  // permanently failed state -- and since aer-0071 the stream is
  // process-persistent, so the poison outlived the contractor and killed
  // every later async op with "previous error during capture". Recreating
  // the stream is the only reliable reset; callers hold no cached
  // references to the raw handle (stream() is re-read at every use).
  void recreate_stream() {
    hipSetDevice(device_id_);
    if (stream_) {
      hipStreamDestroy(stream_);
      stream_ = nullptr;
    }
    check_hip(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
              "hipStreamCreateWithFlags(recreate)", device_id_);
  }
  MemoryPool &pool() { return pool_; }
  thrust::device_vector<thrust::complex<data_t>> &output_buffer() { return dev_out_; }
  void *tensor_data_ptr() const { return tensor_data_ptr_; }
  size_t tensor_data_size() const { return tensor_data_size_; }

  void release() {
    if (device_id_ < 0) return;
    hipSetDevice(device_id_);
    pool_.release();
    if (tensor_data_ptr_) { hipFree(tensor_data_ptr_); tensor_data_ptr_ = nullptr; }
    tensor_data_size_ = 0;
    dev_out_.clear(); dev_out_.shrink_to_fit();
    deallocate_sampling_buffers();
#ifdef AER_HIPBLAS
    if (bl_handle_valid_) { hipblasDestroy(bl_handle_); bl_handle_ = nullptr; bl_handle_valid_ = false; }
#endif
    if (stream_) { hipStreamDestroy(stream_); stream_ = nullptr; }
  }

  void refresh_free_memory() {
    hipSetDevice(device_id_);
    size_t total;
    check_hip(hipMemGetInfo(&free_memory_, &total), "hipMemGetInfo", device_id_);
  }
};

//=============================================================================
// GPUResourceManager
//=============================================================================

template <typename data_t> class GPUResourceManager {
  std::vector<std::unique_ptr<GPUDevice<data_t>>> devices_;
  std::vector<int> device_ids_;

public:
  GPUResourceManager() = default;
  ~GPUResourceManager() = default;
  GPUResourceManager(const GPUResourceManager &) = delete;
  GPUResourceManager &operator=(const GPUResourceManager &) = delete;

  void discover(const std::vector<uint64_t> &target_gpus = {},
                size_t min_memory = 256 * 1024 * 1024) {
    // Piece 0 (CPU-node operability): use the no-throw probe rather than
    // check_hip so that "no GPU on this node" surfaces as a single clean,
    // catchable exception naming the cause — not as a HIP error string and
    // never as a process abort. GPU absence is a normal condition on LUMI
    // `standard` nodes; only an explicit request for the GPU-only contractor
    // makes it an error, which is exactly the path that reaches discover().
    int device_count = tn_gpu_device_count();
    if (device_count == 0)
      throw std::runtime_error(
          "tensor_network method requires a ROCm-capable GPU; none detected "
          "on this node — for circuits of this size consider a CPU-supported "
          "method (e.g. method='statevector' with device='CPU').");

    std::vector<int> candidates;
    if (!target_gpus.empty()) {
      for (size_t i = 0; i < target_gpus.size(); i++)
        candidates.push_back(static_cast<int>(target_gpus[i]));
    } else {
      for (int i = 0; i < device_count; i++)
        candidates.push_back(i);
    }

    for (size_t ci = 0; ci < candidates.size(); ci++) {
      int dev_id = candidates[ci];
      if (dev_id >= device_count) continue;
      auto device = std::unique_ptr<GPUDevice<data_t>>(new GPUDevice<data_t>());
      try {
        device->init(dev_id, device_count);
      } catch (const std::runtime_error &e) {
        if (gpu_verbose())
          fprintf(stderr, "[AER_TN_GPU] skipping device %d: %s\n", dev_id, e.what());
        continue;
      }
      if (device->free_memory() < min_memory) continue;
      device_ids_.push_back(dev_id);
      devices_.push_back(std::move(device));
    }

    if (devices_.empty())
      throw std::runtime_error("No usable GPUs found.");
    if (gpu_verbose())
      fprintf(stderr, "[AER_TN_GPU] discovered %zu usable GPU(s)\n", devices_.size());
  }

  size_t min_free_memory() const {
    size_t min_mem = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < devices_.size(); i++)
      min_mem = std::min(min_mem, devices_[i]->free_memory());
    return min_mem;
  }

  size_t num_devices() const { return devices_.size(); }
  GPUDevice<data_t> &device(size_t i) { return *devices_[i]; }
  const GPUDevice<data_t> &device(size_t i) const { return *devices_[i]; }
  GPUDevice<data_t> &primary() { return *devices_[0]; }
  const std::vector<int> &device_ids() const { return device_ids_; }
};

// aer-0071: process-lifetime resource manager, selected by AER_TN_GPU_PERSIST.
// The contractor is constructed and DELETED per expval call
// (tensor_net.hpp: create_contractor .. delete contractor), so every
// instance-owned pool, slab and captured graph dies between contractions --
// which is why aer-0069/0070 measured graph_captures=0: no instance ever
// reached a second visit. This singleton is the plan_cache_instance()
// precedent applied to GPU state: devices are discovered once, the tensor
// slab and the aer-0070 pool arena persist, and their epochs finally govern
// something longer-lived than one contraction. One instance per data_t.
// CONCURRENCY CONTRACT: persistent mode assumes ONE active contractor per
// process at a time (true for the tensor_network expval/statevector flows;
// the OMP-parallel executor paths must leave the knob off, which is why the
// default is off).
template <typename data_t>
GPUResourceManager<data_t> &gpu_manager_singleton() {
  static GPUResourceManager<data_t> mgr;
  return mgr;
}

} // namespace TensorNetwork
} // namespace AER

#endif // _gpu_resource_manager_hpp_
