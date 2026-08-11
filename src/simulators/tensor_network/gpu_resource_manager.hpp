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
#include <hiptensor/hiptensor.hpp>
#ifdef AER_HIPBLAS
#include <hipblas/hipblas.h>
#endif


// aer-0031: expose hipTensor's algorithm selector.
//
// WHY THIS IS A PARITY ITEM. tensor_net_state.hpp:434 reads
// config.use_cuTensorNet_autotuning into a user-facing flag, tensor_net.hpp
// passes it to setup_contraction() from nine call sites, and the cuTensorNet
// contractor honours it by calling cutensornetContractionAutotune
// (tensor_net_contractor_cuTensorNet.hpp:600). The hipTensor contractor takes
// the same flag and discards it -- the parameter at
// tensor_net_contractor_hiptensor.hpp:1093 is unnamed -- and hardcodes
// HIPTENSOR_ALGO_DEFAULT at both plan-creation sites. A user who sets that
// option gets autotuning on NVIDIA and silence on AMD.
//
// It was not vacuous. Job 20563826 read the shipped headers at
// /opt/rocm-6.4.2/include/hiptensor/ and found THREE selectors:
//   HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_ALGO_DEFAULT_PATIENT,
//   HIPTENSOR_ALGO_ACTOR_CRITIC
// Two of them have never been used by this backend.
//
// WHY THIS MAY MATTER MORE THAN TUNING. OI11 is stated specifically against
// "HIPTENSOR_ALGO_DEFAULT's brute-force dispatcher", which invokes an m6n6k6
// kernel against an oversized descriptor and sums over the excess M-axes with
// no error returned. Whether DEFAULT_PATIENT or ACTOR_CRITIC validate the
// descriptor instead is unmeasured. If either does, that is a correctness
// result, not a performance one, and it is reachable now without a rebuild.
//
// LATCHED PROCESS-WIDE, DELIBERATELY. The shared hipTensor plan cache is keyed
// by tensor shape and not by algorithm, so a selector that could change during
// a run would let a plan built under one be replayed under another. Latching on
// first read makes that impossible without touching the cache key. The
// consequence, stated rather than hidden: if the workspace probe below runs
// before any setup_contraction(), the latch takes DEFAULT and a later
// use_autotune request is ignored. That is the safe direction -- it can only
// leave current behaviour in place.
//
// An unrecognised value throws instead of falling back, because a silent
// fallback is how a measurement of "patient" ends up measuring "default".
static hiptensorAlgo_t tn_hiptensor_algo(bool autotune_requested = false) {
  static bool checked = false;
  static hiptensorAlgo_t cached = HIPTENSOR_ALGO_DEFAULT;
  if (!checked) {
    const char *val = std::getenv("AER_TN_HIPTENSOR_ALGO");
    if (val != nullptr) {
      std::string want(val);
      for (char &c : want)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      if (want == "default") {
        cached = HIPTENSOR_ALGO_DEFAULT;
      } else if (want == "patient" || want == "default_patient") {
        cached = HIPTENSOR_ALGO_DEFAULT_PATIENT;
      } else if (want == "actor_critic" || want == "actor-critic") {
        cached = HIPTENSOR_ALGO_ACTOR_CRITIC;
      } else {
        std::stringstream err;
        err << "[AER_TN] AER_TN_HIPTENSOR_ALGO='" << val
            << "' is not recognised. Use default, patient, or actor_critic.";
        throw std::runtime_error(err.str());
      }
    } else if (autotune_requested) {
      cached = HIPTENSOR_ALGO_DEFAULT_PATIENT;
    }
    checked = true;
  }
  return cached;
}

static const char *tn_hiptensor_algo_name() {
  switch (tn_hiptensor_algo()) {
  case HIPTENSOR_ALGO_DEFAULT_PATIENT:
    return "DEFAULT_PATIENT";
  case HIPTENSOR_ALGO_ACTOR_CRITIC:
    return "ACTOR_CRITIC";
  default:
    return "DEFAULT";
  }
}
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

inline void check_hiptensor(hiptensorStatus_t err, const char *func,
                            int device_id = -1) {
  if (err != HIPTENSOR_STATUS_SUCCESS) {
    std::stringstream ss;
    ss << "hipTensor error in " << func << ": " << hiptensorGetErrorString(err);
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
// ContractionSignature and HipTensorPlanCache
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

// aer-0040: AER_TN_GEMM=1 routes eligible contraction steps through
// hipblas*gemm instead of hiptensorContraction. Default OFF, so an image built
// with -DAER_HIPBLAS behaves byte-identically to one without it until the knob
// is set. Read here rather than in the contractor because GPUDevice::init()
// needs it to decide whether to create the hipBLAS handle at all.
static bool tn_gemm_route() {
  static bool checked = false;
  static bool cached = false;
  if (!checked) {
    const char *v = std::getenv("AER_TN_GEMM");
    cached = (v != nullptr && v[0] == '1' && v[1] == '\0');
    checked = true;
  }
  return cached;
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
static bool tn_gemm_permute() {
  static bool checked = false;
  static bool cached = false;
  if (!checked) {
    const char *v = std::getenv("AER_TN_GEMM_PERMUTE");
    cached = (v != nullptr && v[0] == '1' && v[1] == '\0');
    checked = true;
  }
  return cached;
}

// aer-0040: AER_TN_GEMM_VERIFY=1 runs BOTH paths on every routed step and
// compares the results bit for bit, reporting the first disagreement with the
// step index and shape. Expensive by construction -- it doubles the work and
// adds a device-to-host copy per step -- and off by default. It exists because
// the failure mode a GEMM route introduces is a wrong lda or a wrong transpose
// flag, which is exactly as silent as the hipTensor defect the rank guard was
// written for, and no guard catches it.
static bool tn_gemm_verify() {
  static bool checked = false;
  static bool cached = false;
  if (!checked) {
    const char *v = std::getenv("AER_TN_GEMM_VERIFY");
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

template <typename data_t> struct CachedPlan {
  hiptensorContractionDescriptor_t desc;
  hiptensorContractionFind_t find;
  hiptensorContractionPlan_t plan;
  uint64_t workspace_bytes;
  hiptensorTensorDescriptor_t desc_a;
  hiptensorTensorDescriptor_t desc_b;
  hiptensorTensorDescriptor_t desc_c;

  // Persistent copies of the mode and extent arrays we hand to hipTensor.
  // hiptensorInitContractionDescriptor and hiptensorInitTensorDescriptor
  // retain the caller-provided pointers (modes_a.data(), extents_a.data(),
  // etc.) rather than copying. If we pass pointers to caller-local vectors
  // those become dangling the moment the caller's stack frame unwinds, and
  // any later use of this plan reads freed memory - observed as silent
  // zero-output. Owning the arrays here keeps them alive for the plan's
  // lifetime.
  std::vector<int32_t> modes_a_storage;
  std::vector<int32_t> modes_b_storage;
  std::vector<int32_t> modes_c_storage;
  std::vector<int64_t> extents_a_storage;
  std::vector<int64_t> extents_b_storage;
  std::vector<int64_t> extents_c_storage;
  // Explicit strides for sliced-input views (empty = packed, descriptor
  // built with nullptr strides). Same lifetime rule as the mode/extent
  // arrays: hipTensor retains the pointer, so the plan must own the data.
  std::vector<int64_t> strides_a_storage;
  std::vector<int64_t> strides_b_storage;
  // Explicit C strides. Packed pool-slot C steps leave this empty (nullptr
  // descriptor, as before). WS-3 tiled sub-blocks set it: a block is a
  // strided VIEW into the packed parent C, so when a non-trailing C axis is
  // tiled away the surviving axes are no longer contiguous and C must carry
  // explicit strides — same mechanism as the A/B sliced-view strides above.
  std::vector<int64_t> strides_c_storage;

  // aer-0028: LRU stamp. A plain integer, so adding it cannot move the storage
  // vectors above -- hipTensor retains pointers INTO those, and relocating them
  // is the documented silent-zero-output failure.
  uint64_t last_used = 0;
};

template <typename data_t> class HipTensorPlanCache {
  std::unordered_map<ContractionSignature, CachedPlan<data_t>,
                     ContractionSignatureHash> cache_;
  // Storage slot used only when AER_TN_DISABLE_PLAN_CACHE=1. Initialized
  // fresh on every get_or_create call so the returned reference is valid
  // until the next call. Not thread-safe, but neither is the cache itself.
  CachedPlan<data_t> overwritten_slot_;
  hiptensorHandle_t *handle_;
  int device_id_;
  // aer-0028: guards cache_ so a process-wide instance can be reached from
  // concurrent contractors. Uncontended and negligible for the per-contractor
  // instance. NOT recursive: get_or_create must not call trim.
  // mutable: max_workspace_bytes() and size() are const readers that iterate
  // cache_, and under a process-wide instance a concurrent insert from another
  // contractor makes that iteration a data race. aer-0028 introduced the
  // sharing, so aer-0028 owns guarding them.
  mutable std::mutex mu_;
  uint64_t tick_ = 0;
  uint64_t hits_ = 0;
  uint64_t misses_ = 0;

public:
  HipTensorPlanCache() : handle_(nullptr), device_id_(0) {}

  void init(hiptensorHandle_t *handle, int device_id) {
    handle_ = handle;
    device_id_ = device_id;
  }

  const CachedPlan<data_t> &
  get_or_create(const ContractionSignature &sig,
                const std::vector<int32_t> &modes_a,
                const std::vector<int64_t> &extents_a,
                const std::vector<int32_t> &modes_b,
                const std::vector<int64_t> &extents_b,
                const std::vector<int32_t> &modes_c,
                const std::vector<int64_t> &extents_c,
                const std::vector<int64_t> &strides_a = {},
                const std::vector<int64_t> &strides_b = {},
                const std::vector<int64_t> &strides_c = {}) {
    // Diagnostic kill switch: forces a fresh plan build on every call via
    // an always-rebuilt slot outside the cache map. Useful when suspecting
    // cache-relocation hazards; AER_TN_DISABLE_PLAN_CACHE=1 activates it.
    const char *disable_cache_env = std::getenv("AER_TN_DISABLE_PLAN_CACHE");
    bool cache_disabled =
        (disable_cache_env != nullptr && std::string(disable_cache_env) == "1");

    // aer-0028: the lock covers only the map operations. The returned
    // reference is used by the caller outside it, which is sound because
    // nothing erases from cache_ except trim(), and trim() is called only from
    // setup_contraction() before any reference exists. Insert may rehash;
    // unordered_map preserves references to elements across rehash.
    std::lock_guard<std::mutex> cache_lock(mu_);
    if (!cache_disabled) {
      auto it = cache_.find(sig);
      if (it != cache_.end()) {
        it->second.last_used = ++tick_;
        hits_++;
        return it->second;
      }
      misses_++;
    }

    if (gpu_verbose()) {
      fprintf(stderr,
              "[AER_TN_GPU] plan %s — A(%zu) x B(%zu) -> C(%zu) (dev %d)\n",
              cache_disabled ? "rebuild (cache disabled)" : "cache miss",
              modes_a.size(), modes_b.size(), modes_c.size(), device_id_);
    }

    // CRITICAL LIFETIME RULE:
    // hiptensorInitContractionDescriptor may store &desc_a / &desc_b / &desc_c
    // inside the returned desc, and hiptensorInitContractionPlan may likewise
    // store pointers into the descriptor members. If we initialize these
    // objects on the stack and then copy them into the cache map, the map's
    // copy ends up with internal pointers to the discarded stack frame —
    // manifesting as silent zero-output on subsequent contractions because the
    // plan internally dereferences freed / overwritten memory.
    //
    // Fix: insert an empty CachedPlan into the map FIRST, take a reference
    // to the stored object, and initialize descriptors and plan directly on
    // that stored object. All hipTensor-internal pointers then refer to the
    // map entry's permanent heap address, which remains valid for the
    // lifetime of the cache.
    //
    // For the diagnostic cache-disabled path we still need a stable storage
    // slot. Use a persistent member (overwritten_slot_) so the returned
    // reference is valid until the next call to get_or_create.
    CachedPlan<data_t> *slot = nullptr;
    if (cache_disabled) {
      slot = &overwritten_slot_;
    } else {
      auto result = cache_.emplace(sig, CachedPlan<data_t>{});
      slot = &result.first->second;
    }
    CachedPlan<data_t> &cp = *slot;

    // Copy the mode and extent arrays into persistent storage owned by the
    // CachedPlan. hipTensor retains the pointers we pass to
    // hiptensorInitTensorDescriptor and hiptensorInitContractionDescriptor
    // rather than deep-copying; if we hand it pointers into caller-local
    // vectors, those pointers dangle the moment the caller returns. Copying
    // into cp's own storage and handing hipTensor cp's pointers keeps them
    // valid for the plan's full lifetime.
    cp.modes_a_storage = modes_a;
    cp.modes_b_storage = modes_b;
    cp.modes_c_storage = modes_c;
    cp.extents_a_storage = extents_a;
    cp.extents_b_storage = extents_b;
    cp.extents_c_storage = extents_c;
    cp.strides_a_storage = strides_a;
    cp.strides_b_storage = strides_b;
    cp.strides_c_storage = strides_c;

    hipSetDevice(device_id_);

    // Split-complex contractions: descriptors see REAL tensors. hipTensor
    // 1.5.0's complex path is broken on gfx90a; we decompose each complex
    // contraction into four real contractions in execute_contraction. One
    // plan (one shape, one compute type) serves all four calls.
    hipDataType hip_dtype;
    hiptensorComputeType_t compute_type;
    if (sizeof(data_t) == 8) {
      hip_dtype = HIP_R_64F;
      compute_type = HIPTENSOR_COMPUTE_64F;
    } else {
      hip_dtype = HIP_R_32F;
      compute_type = HIPTENSOR_COMPUTE_32F;
    }

    uint32_t align = TENSOR_POINTER_ALIGN;
    // A and B may be strided views into a larger parent tensor (slice
    // projection keeps device data in place and presents the unsliced
    // remainder through parent strides). nullptr = hipTensor's packed
    // column-major default, identical to an explicit packed stride list.
    // C is packed for ordinary pool-slot steps (strides_c empty → nullptr),
    // but a WS-3 tiled sub-block is a strided VIEW into the packed parent C
    // and supplies explicit C strides (see strides_c_storage).
    const int64_t *sa = cp.strides_a_storage.empty()
                            ? nullptr : cp.strides_a_storage.data();
    const int64_t *sb = cp.strides_b_storage.empty()
                            ? nullptr : cp.strides_b_storage.data();
    const int64_t *sc = cp.strides_c_storage.empty()
                            ? nullptr : cp.strides_c_storage.data();
    check_hiptensor(hiptensorInitTensorDescriptor(
        handle_, &cp.desc_a, static_cast<uint32_t>(cp.modes_a_storage.size()),
        cp.extents_a_storage.data(), sa, hip_dtype, HIPTENSOR_OP_IDENTITY),
        "hiptensorInitTensorDescriptor(A)", device_id_);

    check_hiptensor(hiptensorInitTensorDescriptor(
        handle_, &cp.desc_b, static_cast<uint32_t>(cp.modes_b_storage.size()),
        cp.extents_b_storage.data(), sb, hip_dtype, HIPTENSOR_OP_IDENTITY),
        "hiptensorInitTensorDescriptor(B)", device_id_);

    check_hiptensor(hiptensorInitTensorDescriptor(
        handle_, &cp.desc_c, static_cast<uint32_t>(cp.modes_c_storage.size()),
        cp.extents_c_storage.data(), sc, hip_dtype, HIPTENSOR_OP_IDENTITY),
        "hiptensorInitTensorDescriptor(C)", device_id_);

    check_hiptensor(hiptensorInitContractionDescriptor(
        handle_, &cp.desc,
        &cp.desc_a, cp.modes_a_storage.data(), align,
        &cp.desc_b, cp.modes_b_storage.data(), align,
        &cp.desc_c, cp.modes_c_storage.data(), align,
        &cp.desc_c, cp.modes_c_storage.data(), align,
        compute_type),
        "hiptensorInitContractionDescriptor", device_id_);

    check_hiptensor(hiptensorInitContractionFind(
        handle_, &cp.find, tn_hiptensor_algo()),
        "hiptensorInitContractionFind", device_id_);

    cp.workspace_bytes = 0;
    check_hiptensor(hiptensorContractionGetWorkspaceSize(
        handle_, &cp.desc, &cp.find,
        HIPTENSOR_WORKSPACE_RECOMMENDED, &cp.workspace_bytes),
        "hiptensorContractionGetWorkspaceSize", device_id_);

    check_hiptensor(hiptensorInitContractionPlan(
        handle_, &cp.plan, &cp.desc, &cp.find, cp.workspace_bytes),
        "hiptensorInitContractionPlan", device_id_);

    // cp is already stored in its final location (either the cache map
    // entry or overwritten_slot_), so hipTensor's internal pointers into
    // &cp.desc_a / &cp.desc_b / &cp.desc_c / &cp.desc / &cp.find are
    // stable for the lifetime of that storage. No further copy needed.
    return cp;
  }

  // WARNING: this is the max over EVERY plan in the cache. With the aer-0028
  // process-wide cache that spans every circuit ever run on this device, not
  // the circuit being set up, so sizing a pool from it over-reserves by
  // whatever the largest plan ever seen needed. setup_pool_and_cache()
  // therefore accumulates its own per-circuit maximum from the prebuild loop
  // and does not call this. Kept for diagnostics only.
  uint64_t max_workspace_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t max_ws = 0;
    for (auto it = cache_.begin(); it != cache_.end(); ++it)
      max_ws = std::max(max_ws, it->second.workspace_bytes);
    return max_ws;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return cache_.size();
  }
  void clear() {
    std::lock_guard<std::mutex> lock(mu_);
    cache_.clear();
  }

  // aer-0028: evict down to max_entries, least-recently-used first.
  //
  // MUST NOT be called while any CachedPlan reference is live: hipTensor holds
  // pointers into the entry's own storage, so erasing one under a live
  // reference is a use-after-free that shows up as silent zero-output rather
  // than a fault. The only call site is the top of setup_contraction(), before
  // the prebuild loop and long before contract_single_slice takes a reference.
  void trim(size_t max_entries) {
    std::lock_guard<std::mutex> lock(mu_);
    while (cache_.size() > max_entries) {
      auto victim = cache_.end();
      uint64_t best = UINT64_MAX;
      for (auto it = cache_.begin(); it != cache_.end(); ++it) {
        if (it->second.last_used < best) {
          best = it->second.last_used;
          victim = it;
        }
      }
      if (victim == cache_.end())
        break;
      cache_.erase(victim);
    }
  }

  void stats(uint64_t &h, uint64_t &m, size_t &n) {
    std::lock_guard<std::mutex> lock(mu_);
    h = hits_;
    m = misses_;
    n = cache_.size();
  }
};

//=============================================================================
// aer-0028: process-wide hipTensor plan cache
//=============================================================================
//
// WHY
// GPUResourceManager is a MEMBER of the contractor, and a contractor is built
// and destroyed per instruction (create_contractor / delete contractor
// throughout tensor_net.hpp). So GPUDevice::init() runs hiptensorCreate and
// starts an EMPTY plan cache every evaluation, and ~101 hipTensor plans are
// rebuilt from nothing each time.
//
// aer-0027 removed the cotengra search (t_path_ms=0 on a repeat). Job 20512566
// measured what is left: with t_path already zero, t_setup is still 3431 ms
// (grid n=16) and 2603 ms (rand3 n=64) against a GPU contraction of 7.8 ms and
// 5.7 ms. The contraction is 0.22% of its own setup, and essentially all the
// remainder is this cache being thrown away and rebuilt. Target: ~450x.
//
// WHY ONLY THE HANDLE AND THE PLAN CACHE MOVE
// CachedPlan owns hipTensor descriptors, the plan object, and HOST-side mode /
// extent / stride vectors. It owns no device memory and calls no hipMalloc.
// Every device pointer, the workspace and the stream are passed as arguments to
// hiptensorContraction at call time (see contract_split_complex). So the pool,
// the tensor arena, the stream and the thrust buffers stay per-contractor,
// where their lifetime is already correct, and only {handle, plan cache} are
// shared. The handle must move WITH the plans because the descriptors are
// initialised against it; a plan built on one handle and executed on another is
// undefined.
//
// THE LIFETIME TRAP
// hiptensorInitContractionDescriptor and hiptensorInitContractionPlan retain
// pointers INTO the CachedPlan's own storage (documented at the top of
// get_or_create). Destroying or relocating an entry while a reference to it is
// live is a use-after-free whose symptom is silent zero-output, not a crash.
// contract_single_slice holds `const CachedPlan &` across a per-tile loop, so:
//   * rehash on insert is SAFE -- unordered_map preserves references to
//     elements across rehash, invalidating only iterators;
//   * erase is NOT safe mid-contraction.
// Eviction is therefore never performed inside get_or_create. trim() exists as
// a separate call, made only from setup_contraction() before any reference has
// been taken.
//
// THREAD SAFETY
// A process-wide cache can be reached from concurrent contractors. get_or_create
// and trim take a mutex; the returned reference is used outside it, which is
// sound only because of the no-erase rule above. Plan CREATION on the shared
// handle is serialised by the same mutex.
//
// MPI: setup_pool_and_cache contains no collectives, so this is rank-local and
// needs none of aer-0027's nprocs_ == 1 restriction. Each rank is its own
// process and gets its own registry.
//
// Off by default. AER_TN_SHARED_PLAN_CACHE=1 enables;
// AER_TN_SHARED_PLAN_CACHE_MAX caps entries (default 4096, host memory only,
// roughly a kilobyte or two each).

static bool tn_shared_plan_cache() {
  static bool checked = false;
  static bool cached = false;
  if (!checked) {
    const char *v = std::getenv("AER_TN_SHARED_PLAN_CACHE");
    bool want = (v != nullptr && v[0] == '1' && v[1] == '\0');

    // AER_TN_DISABLE_PLAN_CACHE wins. Its path returns a reference to
    // overwritten_slot_, a single MEMBER of the cache that every call
    // rewrites; the comment on it says the reference is valid only "until the
    // next call to get_or_create". That is tolerable for a per-contractor
    // cache, where the sole caller consumes the reference before calling
    // again. It is NOT tolerable for a process-wide one, where a second
    // contractor's call can rewrite the slot while the first still holds the
    // reference -- and the reference is used outside the mutex, so no lock
    // makes it safe. The two knobs are mutually exclusive rather than silently
    // corrupting, and the diagnostic kill switch takes precedence because its
    // whole purpose is to rule out cache-relocation hazards.
    const char *d = std::getenv("AER_TN_DISABLE_PLAN_CACHE");
    if (want && d != nullptr && std::string(d) == "1") {
      fprintf(stderr,
              "[AER_TN] AER_TN_DISABLE_PLAN_CACHE=1 overrides "
              "AER_TN_SHARED_PLAN_CACHE=1: the disable path hands back a "
              "reference to one overwritten slot, which cannot be shared "
              "between contractors. Plan-cache sharing is OFF for this run.\n");
      want = false;
    }
    cached = want;
    checked = true;
  }
  return cached;
}

static size_t tn_shared_plan_cache_max() {
  static bool checked = false;
  static size_t cached = 4096;
  if (!checked) {
    const char *v = std::getenv("AER_TN_SHARED_PLAN_CACHE_MAX");
    if (v != nullptr) {
      char *end = nullptr;
      long long p = std::strtoll(v, &end, 10);
      if (end != v && p > 0)
        cached = static_cast<size_t>(p);
    }
    checked = true;
  }
  return cached;
}

// One {handle, cache} per device id, created on first use and DELIBERATELY
// never destroyed. Tearing a hiptensorHandle_t down during static destruction
// would order it against the HIP runtime's own teardown, and any plan still
// referencing it would be used-after-free; leaking one handle per device for
// the life of the process is the safer trade.
template <typename data_t> class SharedPlanRegistry {
public:
  struct Slot {
    hiptensorHandle_t *handle;
    HipTensorPlanCache<data_t> cache;
    Slot() : handle(nullptr) {}
  };

  static Slot *acquire(int device_id) {
    static std::mutex reg_mu;
    static std::map<int, Slot *> slots;
    std::lock_guard<std::mutex> lock(reg_mu);
    typename std::map<int, Slot *>::iterator it = slots.find(device_id);
    if (it != slots.end())
      return it->second;
    Slot *s = new Slot();          // never deleted, by design (see above)
    check_hiptensor(hiptensorCreate(&s->handle), "hiptensorCreate(shared)",
                    device_id);
    s->cache.init(s->handle, device_id);
    slots[device_id] = s;
    return s;
  }
};

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
  int device_id_;
  bool allocated_;
  std::vector<PoolAllocation> allocations_;

public:
  MemoryPool() : pool_ptr_(nullptr), pool_size_(0), device_id_(0),
                 allocated_(false) {}
  ~MemoryPool() { release(); }
  MemoryPool(const MemoryPool &) = delete;
  MemoryPool &operator=(const MemoryPool &) = delete;
  MemoryPool(MemoryPool &&other) noexcept
      : pool_ptr_(other.pool_ptr_), pool_size_(other.pool_size_),
        device_id_(other.device_id_), allocated_(other.allocated_),
        allocations_(std::move(other.allocations_)) {
    other.pool_ptr_ = nullptr;
    other.allocated_ = false;
  }

  void plan_layout(
      const std::vector<std::tuple<size_t, int, int, int>> &intermediates,
      size_t workspace_bytes, int num_steps) {
    allocations_.clear();

    if (workspace_bytes > 0) {
      PoolAllocation ws;
      ws.offset = 0; ws.size = workspace_bytes;
      ws.birth_step = 0; ws.death_step = num_steps - 1;
      ws.tensor_index = -1;
      allocations_.push_back(ws);
    }

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

  void allocate(int device_id) {
    device_id_ = device_id;
    if (pool_size_ == 0) return;
    hipSetDevice(device_id_);
    check_hip(hipMalloc(&pool_ptr_, pool_size_), "hipMalloc(pool)", device_id_);
    allocated_ = true;
  }

  void *get_tensor_ptr(int tensor_index) const {
    for (size_t i = 0; i < allocations_.size(); i++)
      if (allocations_[i].tensor_index == tensor_index)
        return static_cast<char *>(pool_ptr_) + allocations_[i].offset;
    std::stringstream ss;
    ss << "MemoryPool: tensor index " << tensor_index << " not found";
    throw std::runtime_error(ss.str());
  }

  void *get_workspace_ptr() const {
    for (size_t i = 0; i < allocations_.size(); i++)
      if (allocations_[i].tensor_index == -1)
        return static_cast<char *>(pool_ptr_) + allocations_[i].offset;
    return nullptr;
  }

  size_t get_workspace_size() const {
    for (size_t i = 0; i < allocations_.size(); i++)
      if (allocations_[i].tensor_index == -1) return allocations_[i].size;
    return 0;
  }

  size_t total_size() const { return pool_size_; }
  bool is_allocated() const { return allocated_; }

  void release() {
    if (allocated_ && pool_ptr_) {
      hipSetDevice(device_id_);
      hipFree(pool_ptr_);
      pool_ptr_ = nullptr;
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
// two separate real planes (see the OI7 note on hipTensor's complex path), and
// execute_contraction_planes below takes six plane pointers. hipblasZgemm
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
        << " does not fit the int the hipblas*gemm interface takes. Unset "
           "AER_TN_GEMM to run this circuit on the hiptensorContraction path.";
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
#endif // AER_HIPBLAS

template <typename data_t> class GPUDevice {
  int device_id_;
  std::string architecture_;
  size_t total_memory_;
  size_t free_memory_;
  hipStream_t stream_;
  hiptensorHandle_t *ht_handle_;
  bool handle_valid_;
#ifdef AER_HIPBLAS
  // aer-0040: created only when the GEMM route is enabled, destroyed in
  // release() alongside the stream. Never shared through SharedPlanRegistry:
  // that slot exists to keep hipTensor PLANS alive across contractors, and a
  // hipBLAS handle owns no plans.
  hipblasHandle_t bl_handle_;
  bool bl_handle_valid_;
#endif

  HipTensorPlanCache<data_t> plan_cache_;
  // aer-0028: when shared-plan-cache mode is on this points at the process-wide
  // slot for this device and plan_cache_ above goes unused. The pool, the
  // arena, the stream and the thrust buffers are deliberately NOT shared --
  // they own device memory whose lifetime is already correct per contractor,
  // and only the handle and the plans need to outlive one.
  typename SharedPlanRegistry<data_t>::Slot *shared_slot_ = nullptr;
  MemoryPool pool_;
  std::vector<bool> peer_access_;
  void *tensor_data_ptr_;
  size_t tensor_data_size_;

  thrust::device_vector<thrust::complex<data_t>> dev_out_;
  thrust::device_vector<double> sampling_rnds_;
  thrust::device_vector<uint64_t> sampling_out_;

public:
  GPUDevice()
      : device_id_(-1), total_memory_(0), free_memory_(0), stream_(nullptr),
        ht_handle_(nullptr), handle_valid_(false),
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

    if (tn_shared_plan_cache()) {
      // Reuse the process-wide handle. handle_valid_ stays FALSE so release()
      // does not hiptensorDestroy it out from under plans that outlive this
      // contractor -- that would be a use-after-free on the next instruction.
      shared_slot_ = SharedPlanRegistry<data_t>::acquire(device_id_);
      ht_handle_ = shared_slot_->handle;
      handle_valid_ = false;
    } else {
      check_hiptensor(hiptensorCreate(&ht_handle_), "hiptensorCreate",
                      device_id_);
      handle_valid_ = true;
      plan_cache_.init(ht_handle_, device_id_);
    }

#ifdef AER_HIPBLAS
    if (tn_gemm_route()) {
      check_hipblas(hipblasCreate(&bl_handle_), "hipblasCreate", device_id_);
      bl_handle_valid_ = true;
      // Same stream as every hipTensor call on this device, so the GEMM route
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

  // Execute pairwise contraction. accumulate=true uses beta=1 (D += A*B).
  //
  // Split-complex decomposition: A, B, C are stored with real and imaginary
  // planes in separate contiguous regions of the pool slot. Given pointers
  // ptr_{a,b,c} to slot starts and the element counts, we derive plane
  // pointers and issue four real contractions (one hipTensor plan, four
  // different pointer combinations and alpha/beta values):
  //
  //   Cr = Ar*Br - Ai*Bi
  //   Ci = Ar*Bi + Ai*Br
  //
  // When accumulate=true the initial beta is 1 instead of 0 so we add into
  // the pre-existing Cr / Ci values. The two subsequent calls always use
  // beta=1 since they accumulate onto the partial result from the first
  // call of each pair.
  void execute_contraction(const CachedPlan<data_t> &plan,
                           void *ptr_a, size_t num_elements_a,
                           void *ptr_b, size_t num_elements_b,
                           void *ptr_c, size_t num_elements_c,
                           void *workspace, uint64_t workspace_size,
                           bool accumulate = false) {
    // Packed-slot convenience form: derive each tensor's imaginary plane
    // from its slot layout ([real plane][imag plane], imag at
    // plane_bytes(N_full)). Valid only for tensors that occupy a whole
    // slot — pool intermediates and unprojected inputs. Slice-projected
    // inputs must use the explicit-plane overload below, because their
    // imag-plane offset depends on the FULL parent tensor size, not the
    // projected element count (resolves OI9).
    const size_t plane_a = plane_bytes(
        static_cast<int64_t>(num_elements_a), sizeof(data_t));
    const size_t plane_b = plane_bytes(
        static_cast<int64_t>(num_elements_b), sizeof(data_t));
    const size_t plane_c = plane_bytes(
        static_cast<int64_t>(num_elements_c), sizeof(data_t));

    execute_contraction_planes(
        plan,
        reinterpret_cast<data_t *>(ptr_a),
        reinterpret_cast<data_t *>(reinterpret_cast<char *>(ptr_a) + plane_a),
        reinterpret_cast<data_t *>(ptr_b),
        reinterpret_cast<data_t *>(reinterpret_cast<char *>(ptr_b) + plane_b),
        reinterpret_cast<data_t *>(ptr_c),
        reinterpret_cast<data_t *>(reinterpret_cast<char *>(ptr_c) + plane_c),
        workspace, workspace_size, accumulate);
  }

  // Explicit-plane form. The caller provides the real and imaginary plane
  // pointers for every operand. Slice projection advances both planes by
  // the same element offset within the parent tensor's slot, so a sliced
  // sub-tensor's planes are no longer related by plane_bytes(N_sub) — this
  // overload is what makes slicing and split-complex compose.
  void execute_contraction_planes(const CachedPlan<data_t> &plan,
                                  data_t *Ar, data_t *Ai,
                                  data_t *Br, data_t *Bi,
                                  data_t *Cr, data_t *Ci,
                                  void *workspace, uint64_t workspace_size,
                                  bool accumulate = false) {
    hipSetDevice(device_id_);
    const data_t pos_one = static_cast<data_t>(1.0);
    const data_t neg_one = static_cast<data_t>(-1.0);
    const data_t zero    = static_cast<data_t>(0.0);
    const data_t initial_beta = accumulate ? pos_one : zero;

    // Cr = Ar * Br  (beta = initial)
    check_hiptensor(
        hiptensorContraction(ht_handle_, &plan.plan,
                             &pos_one, Ar, Br,
                             &initial_beta, Cr, Cr,
                             workspace, workspace_size, stream_),
        "hiptensorContraction(Cr=Ar*Br)", device_id_);
    // Cr -= Ai * Bi  (beta = 1, alpha = -1)
    check_hiptensor(
        hiptensorContraction(ht_handle_, &plan.plan,
                             &neg_one, Ai, Bi,
                             &pos_one, Cr, Cr,
                             workspace, workspace_size, stream_),
        "hiptensorContraction(Cr-=Ai*Bi)", device_id_);
    // Ci = Ar * Bi  (beta = initial)
    check_hiptensor(
        hiptensorContraction(ht_handle_, &plan.plan,
                             &pos_one, Ar, Bi,
                             &initial_beta, Ci, Ci,
                             workspace, workspace_size, stream_),
        "hiptensorContraction(Ci=Ar*Bi)", device_id_);
    // Ci += Ai * Br  (beta = 1, alpha = 1)
    check_hiptensor(
        hiptensorContraction(ht_handle_, &plan.plan,
                             &pos_one, Ai, Br,
                             &pos_one, Ci, Ci,
                             workspace, workspace_size, stream_),
        "hiptensorContraction(Ci+=Ai*Br)", device_id_);
  }

#ifdef AER_HIPBLAS
  // aer-0048: one throwaway routed-shape dispatch to pay rocBLAS/Tensile lazy
  // initialisation at handle-creation time, off the timed path. Self-contained:
  // allocates its own small scratch, runs the SAME four-gemm split-complex form
  // execute_contraction_gemm uses, synchronises so the init actually completes,
  // and frees. Shape 8x8x8 is inside the m6n6k6 routed envelope and large
  // enough to select a non-trivial Tensile solution. Errors are swallowed: a
  // warm-up that fails must not take down a run whose real dispatches would
  // have succeeded. Only reached from init() when tn_gemm_route() &&
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
      hipStreamSynchronize(stream_);
    }
    hipFree(Ar); hipFree(Ai); hipFree(Br); hipFree(Bi);
    hipFree(Cr); hipFree(Ci);
  }

  // aer-0040: the routed form of execute_contraction_planes. Same six plane
  // pointers, same alphas, same betas, same stream; four hipblas*gemm calls
  // instead of four hiptensorContraction calls. No workspace: GEMM needs none.
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
  hiptensorHandle_t *handle() { return ht_handle_; }
  HipTensorPlanCache<data_t> &plan_cache() {
    return shared_slot_ ? shared_slot_->cache : plan_cache_;
  }
  bool plan_cache_is_shared() const { return shared_slot_ != nullptr; }
  MemoryPool &pool() { return pool_; }
  thrust::device_vector<thrust::complex<data_t>> &output_buffer() { return dev_out_; }
  void *tensor_data_ptr() const { return tensor_data_ptr_; }
  size_t tensor_data_size() const { return tensor_data_size_; }

  void release() {
    if (device_id_ < 0) return;
    hipSetDevice(device_id_);
    pool_.release();
    // aer-0028: never clear the shared cache here -- surviving this
    // contractor is the entire point, and the plans are still valid because
    // the shared handle is never destroyed. The private cache is cleared as
    // before so non-shared mode is byte-for-byte unchanged.
    if (!shared_slot_)
      plan_cache_.clear();
    if (tensor_data_ptr_) { hipFree(tensor_data_ptr_); tensor_data_ptr_ = nullptr; }
    tensor_data_size_ = 0;
    dev_out_.clear(); dev_out_.shrink_to_fit();
    deallocate_sampling_buffers();
    if (handle_valid_) { hiptensorDestroy(ht_handle_); ht_handle_ = nullptr; handle_valid_ = false; }
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

  uint64_t query_workspace_for_size(int64_t max_intermediate_elements) {
    if (devices_.empty()) return 0;
    GPUDevice<data_t> &primary = *devices_[0];
    hipSetDevice(primary.device_id());

    int64_t n = static_cast<int64_t>(
        std::sqrt(static_cast<double>(max_intermediate_elements)));
    if (n < 2) n = 2;
    std::vector<int64_t> extents = {n, n};
    std::vector<int32_t> modes_a = {0, 1};
    std::vector<int32_t> modes_b = {1, 2};
    std::vector<int32_t> modes_c = {0, 2};

    hipDataType hip_dtype = (sizeof(data_t) == 8) ? HIP_R_64F : HIP_R_32F;
    hiptensorComputeType_t compute_type =
        (sizeof(data_t) == 8) ? HIPTENSOR_COMPUTE_64F : HIPTENSOR_COMPUTE_32F;

    hiptensorTensorDescriptor_t da, db, dc;
    hiptensorContractionDescriptor_t desc;
    hiptensorContractionFind_t find;
    uint64_t workspace = 0;
    uint32_t align = TENSOR_POINTER_ALIGN;

    auto status = hiptensorInitTensorDescriptor(
        primary.handle(), &da, 2, extents.data(), nullptr,
        hip_dtype, HIPTENSOR_OP_IDENTITY);
    if (status != HIPTENSOR_STATUS_SUCCESS) {
      fprintf(stderr,
              "[AER_TN_GPU] WARNING: workspace probe could not init tensor "
              "descriptor (hipTensor status %d); falling back to 64 MB budget\n",
              (int)status);
      return 64 * 1024 * 1024;
    }

    hiptensorInitTensorDescriptor(primary.handle(), &db, 2, extents.data(),
        nullptr, hip_dtype, HIPTENSOR_OP_IDENTITY);
    hiptensorInitTensorDescriptor(primary.handle(), &dc, 2, extents.data(),
        nullptr, hip_dtype, HIPTENSOR_OP_IDENTITY);

    status = hiptensorInitContractionDescriptor(
        primary.handle(), &desc,
        &da, modes_a.data(), align, &db, modes_b.data(), align,
        &dc, modes_c.data(), align, &dc, modes_c.data(), align,
        compute_type);
    if (status != HIPTENSOR_STATUS_SUCCESS) {
      fprintf(stderr,
              "[AER_TN_GPU] WARNING: workspace probe could not init contraction "
              "descriptor (hipTensor status %d); falling back to 64 MB budget\n",
              (int)status);
      return 64 * 1024 * 1024;
    }

    hiptensorInitContractionFind(primary.handle(), &find, tn_hiptensor_algo());
    hiptensorContractionGetWorkspaceSize(
        primary.handle(), &desc, &find,
        HIPTENSOR_WORKSPACE_RECOMMENDED, &workspace);
    return workspace;
  }

  size_t num_devices() const { return devices_.size(); }
  GPUDevice<data_t> &device(size_t i) { return *devices_[i]; }
  const GPUDevice<data_t> &device(size_t i) const { return *devices_[i]; }
  GPUDevice<data_t> &primary() { return *devices_[0]; }
  const std::vector<int> &device_ids() const { return device_ids_; }
};

} // namespace TensorNetwork
} // namespace AER

#endif // _gpu_resource_manager_hpp_
