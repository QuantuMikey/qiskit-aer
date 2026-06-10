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
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.hpp>

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
                const std::vector<int64_t> &strides_b = {}) {
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
  return sig;
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
                const std::vector<int64_t> &strides_b = {}) {
    // Diagnostic kill switch: forces a fresh plan build on every call via
    // an always-rebuilt slot outside the cache map. Useful when suspecting
    // cache-relocation hazards; AER_TN_DISABLE_PLAN_CACHE=1 activates it.
    const char *disable_cache_env = std::getenv("AER_TN_DISABLE_PLAN_CACHE");
    bool cache_disabled =
        (disable_cache_env != nullptr && std::string(disable_cache_env) == "1");

    if (!cache_disabled) {
      auto it = cache_.find(sig);
      if (it != cache_.end())
        return it->second;
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
    // C is always a packed pool slot, so it never carries strides.
    const int64_t *sa = cp.strides_a_storage.empty()
                            ? nullptr : cp.strides_a_storage.data();
    const int64_t *sb = cp.strides_b_storage.empty()
                            ? nullptr : cp.strides_b_storage.data();
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
        cp.extents_c_storage.data(), nullptr, hip_dtype, HIPTENSOR_OP_IDENTITY),
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
        handle_, &cp.find, HIPTENSOR_ALGO_DEFAULT),
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

  uint64_t max_workspace_bytes() const {
    uint64_t max_ws = 0;
    for (auto it = cache_.begin(); it != cache_.end(); ++it)
      max_ws = std::max(max_ws, it->second.workspace_bytes);
    return max_ws;
  }

  size_t size() const { return cache_.size(); }
  void clear() { cache_.clear(); }
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

template <typename data_t> class GPUDevice {
  int device_id_;
  std::string architecture_;
  size_t total_memory_;
  size_t free_memory_;
  hipStream_t stream_;
  hiptensorHandle_t *ht_handle_;
  bool handle_valid_;

  HipTensorPlanCache<data_t> plan_cache_;
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

    check_hiptensor(hiptensorCreate(&ht_handle_), "hiptensorCreate", device_id_);
    handle_valid_ = true;
    plan_cache_.init(ht_handle_, device_id_);

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
  HipTensorPlanCache<data_t> &plan_cache() { return plan_cache_; }
  MemoryPool &pool() { return pool_; }
  thrust::device_vector<thrust::complex<data_t>> &output_buffer() { return dev_out_; }
  void *tensor_data_ptr() const { return tensor_data_ptr_; }
  size_t tensor_data_size() const { return tensor_data_size_; }

  void release() {
    if (device_id_ < 0) return;
    hipSetDevice(device_id_);
    pool_.release();
    plan_cache_.clear();
    if (tensor_data_ptr_) { hipFree(tensor_data_ptr_); tensor_data_ptr_ = nullptr; }
    tensor_data_size_ = 0;
    dev_out_.clear(); dev_out_.shrink_to_fit();
    deallocate_sampling_buffers();
    if (handle_valid_) { hiptensorDestroy(ht_handle_); ht_handle_ = nullptr; handle_valid_ = false; }
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

    hiptensorInitContractionFind(primary.handle(), &find, HIPTENSOR_ALGO_DEFAULT);
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
