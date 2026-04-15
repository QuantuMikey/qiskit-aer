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

#ifndef _gpu_resource_manager_hpp_
#define _gpu_resource_manager_hpp_

/**
 * GPU resource management for hipTensor-based tensor network contraction.
 *
 * This file manages all GPU-side resources: device discovery, memory
 * allocation, hipTensor handles, contraction plan caching, and memory
 * pools. It has zero cotengra dependencies — it can be tested with
 * simple manual contractions without any path optimization.
 *
 * File layout:
 *   1. Diagnostic logging
 *   2. Error handling helpers
 *   3. ContractionSignature and HipTensorPlanCache
 *   4. MemoryPool with lifetime-based offset assignment
 *   5. GPUDevice — per-GPU state and operations
 *   6. GPUResourceManager — multi-GPU orchestration
 */

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
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include "misc/wrap_thrust.hpp"
#include "simulators/statevector/chunk/thrust_kernels.hpp"

#include "simulators/tensor_network/path_optimizer.hpp"

// Thrust execution policy namespace: thrust::hip on ROCm
namespace thrust_gpu = thrust::hip;

namespace AER {
namespace TensorNetwork {

//=============================================================================
// 1. Diagnostic logging
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
// 2. Error handling helpers
//=============================================================================

/**
 * Check a HIP API return code. On failure, throw an exception with
 * the function name, error string, and device ID.
 */
inline void check_hip(hipError_t err, const char *func, int device_id = -1) {
  if (err != hipSuccess) {
    std::stringstream ss;
    ss << "HIP error in " << func << ": " << hipGetErrorString(err);
    if (device_id >= 0)
      ss << " (device " << device_id << ")";
    throw std::runtime_error(ss.str());
  }
}

/**
 * Check a hipTensor API return code. On failure, throw an exception
 * with the function name and error code.
 */
inline void check_hiptensor(hiptensorStatus_t err, const char *func,
                            int device_id = -1) {
  if (err != HIPTENSOR_STATUS_SUCCESS) {
    std::stringstream ss;
    ss << "hipTensor error in " << func << ": error code " << (int)err;
    if (device_id >= 0)
      ss << " (device " << device_id << ")";
    throw std::runtime_error(ss.str());
  }
}

//=============================================================================
// 3. ContractionSignature and HipTensorPlanCache
//=============================================================================

/**
 * Uniquely identifies a pairwise tensor contraction by shape.
 *
 * Two contractions with the same signature use the same hipTensor
 * kernels and workspace, so the plan can be reused. In quantum
 * circuits, most contractions have identical signatures because
 * gates produce tensors of the same rank and extent.
 *
 * The signature is a flat vector of int32 values:
 *   [num_modes_A, modes_A..., extents_A...,
 *    num_modes_B, modes_B..., extents_B...,
 *    num_modes_C, modes_C..., extents_C...]
 *
 * This is used as a hash key for the plan cache.
 */
struct ContractionSignature {
  std::vector<int32_t> data;

  bool operator==(const ContractionSignature &other) const {
    return data == other.data;
  }
};

// Hash function for ContractionSignature
struct ContractionSignatureHash {
  size_t operator()(const ContractionSignature &sig) const {
    // FNV-1a hash
    size_t hash = 14695981039346656037ULL;
    for (auto v : sig.data) {
      hash ^= static_cast<size_t>(v);
      hash *= 1099511628211ULL;
    }
    return hash;
  }
};

/**
 * Build a contraction signature from the mode/extent lists of
 * two input tensors and one output tensor.
 */
inline ContractionSignature
build_signature(const std::vector<int32_t> &modes_a,
                const std::vector<int64_t> &extents_a,
                const std::vector<int32_t> &modes_b,
                const std::vector<int64_t> &extents_b,
                const std::vector<int32_t> &modes_c,
                const std::vector<int64_t> &extents_c) {
  ContractionSignature sig;
  sig.data.push_back(static_cast<int32_t>(modes_a.size()));
  for (auto m : modes_a)
    sig.data.push_back(m);
  for (auto e : extents_a)
    sig.data.push_back(static_cast<int32_t>(e));

  sig.data.push_back(static_cast<int32_t>(modes_b.size()));
  for (auto m : modes_b)
    sig.data.push_back(m);
  for (auto e : extents_b)
    sig.data.push_back(static_cast<int32_t>(e));

  sig.data.push_back(static_cast<int32_t>(modes_c.size()));
  for (auto m : modes_c)
    sig.data.push_back(m);
  for (auto e : extents_c)
    sig.data.push_back(static_cast<int32_t>(e));

  return sig;
}

/**
 * A cached hipTensor contraction plan.
 *
 * Contains everything needed to execute a pairwise contraction
 * of a given shape: the plan itself, the workspace size it needs,
 * and all the descriptor/find objects that hipTensor requires.
 *
 * These are created during setup_contraction (planning phase) and
 * reused during contract (execution phase). No hipTensor plan
 * creation happens during the hot loop.
 */
template <typename data_t> struct CachedPlan {
  hiptensorContractionDescriptor_t desc;
  hiptensorContractionFind_t find;
  hiptensorContractionPlan_t plan;
  uint64_t workspace_bytes;

  // Tensor descriptors (needed by hipTensor API)
  hiptensorTensorDescriptor_t desc_a;
  hiptensorTensorDescriptor_t desc_b;
  hiptensorTensorDescriptor_t desc_c;
};

/**
 * Cache of hipTensor contraction plans, keyed by shape signature.
 *
 * Populated during setup_contraction by walking the contraction path
 * and creating plans for each unique signature. During contract,
 * every pairwise contraction is a cache hit.
 */
template <typename data_t> class HipTensorPlanCache {
  std::unordered_map<ContractionSignature, CachedPlan<data_t>,
                     ContractionSignatureHash>
      cache_;
  hiptensorHandle_t *handle_; // owned by GPUDevice, not by us
  int device_id_;

public:
  HipTensorPlanCache() : handle_(nullptr), device_id_(0) {}

  void init(hiptensorHandle_t *handle, int device_id) {
    handle_ = handle;
    device_id_ = device_id;
  }

  /**
   * Get or create a plan for the given contraction signature.
   *
   * On first call for a signature: creates hipTensor descriptors,
   * queries workspace size, creates the plan. Subsequent calls
   * return the cached plan immediately.
   *
   * Returns the maximum workspace size across all cached plans,
   * which is needed for memory pool sizing.
   */
  const CachedPlan<data_t> &
  get_or_create(const ContractionSignature &sig,
                const std::vector<int32_t> &modes_a,
                const std::vector<int64_t> &extents_a,
                const std::vector<int32_t> &modes_b,
                const std::vector<int64_t> &extents_b,
                const std::vector<int32_t> &modes_c,
                const std::vector<int64_t> &extents_c) {
    auto it = cache_.find(sig);
    if (it != cache_.end()) {
      if (gpu_verbose()) {
        fprintf(stderr, "[AER_TN_GPU] plan cache hit (device %d)\n",
                device_id_);
      }
      return it->second;
    }

    if (gpu_verbose()) {
      fprintf(stderr,
              "[AER_TN_GPU] plan cache miss — creating plan for "
              "A(%zu modes) x B(%zu modes) -> C(%zu modes) (device %d)\n",
              modes_a.size(), modes_b.size(), modes_c.size(), device_id_);
    }

    CachedPlan<data_t> cp;
    hipSetDevice(device_id_);

    // Determine data type and compute type from template parameter
    hipDataType hip_dtype;
    hiptensorComputeType_t compute_type;
    if (sizeof(data_t) == 8) {
      hip_dtype = HIP_C_64F;
      compute_type = HIPTENSOR_COMPUTE_64F;
    } else {
      hip_dtype = HIP_C_32F;
      compute_type = HIPTENSOR_COMPUTE_32F;
    }

    // Create tensor descriptors
    uint32_t align = 256; // alignment in bytes
    check_hiptensor(
        hiptensorInitTensorDescriptor(
            handle_, &cp.desc_a, modes_a.size(),
            extents_a.data(), nullptr /* strides = row-major */,
            hip_dtype, HIPTENSOR_OP_IDENTITY),
        "hiptensorInitTensorDescriptor(A)", device_id_);

    check_hiptensor(
        hiptensorInitTensorDescriptor(
            handle_, &cp.desc_b, modes_b.size(),
            extents_b.data(), nullptr,
            hip_dtype, HIPTENSOR_OP_IDENTITY),
        "hiptensorInitTensorDescriptor(B)", device_id_);

    check_hiptensor(
        hiptensorInitTensorDescriptor(
            handle_, &cp.desc_c, modes_c.size(),
            extents_c.data(), nullptr,
            hip_dtype, HIPTENSOR_OP_IDENTITY),
        "hiptensorInitTensorDescriptor(C)", device_id_);

    // Create contraction descriptor
    check_hiptensor(
        hiptensorInitContractionDescriptor(
            handle_, &cp.desc,
            &cp.desc_a, modes_a.data(), align,
            &cp.desc_b, modes_b.data(), align,
            &cp.desc_c, modes_c.data(), align,
            &cp.desc_c, modes_c.data(), align,
            compute_type),
        "hiptensorInitContractionDescriptor", device_id_);

    // Find algorithm
    check_hiptensor(
        hiptensorInitContractionFind(handle_, &cp.find,
                                     HIPTENSOR_ALGO_DEFAULT),
        "hiptensorInitContractionFind", device_id_);

    // Query workspace size
    cp.workspace_bytes = 0;
    check_hiptensor(
        hiptensorContractionGetWorkspaceSize(
            handle_, &cp.desc, &cp.find,
            HIPTENSOR_WORKSPACE_RECOMMENDED, &cp.workspace_bytes),
        "hiptensorContractionGetWorkspaceSize", device_id_);

    // Create the plan
    check_hiptensor(
        hiptensorInitContractionPlan(handle_, &cp.plan, &cp.desc,
                                     &cp.find, cp.workspace_bytes),
        "hiptensorInitContractionPlan", device_id_);

    auto result = cache_.emplace(sig, cp);
    return result.first->second;
  }

  /**
   * Maximum workspace size across all cached plans.
   * Used to size the workspace region in the memory pool.
   */
  uint64_t max_workspace_bytes() const {
    uint64_t max_ws = 0;
    for (const auto &[sig, plan] : cache_)
      max_ws = std::max(max_ws, plan.workspace_bytes);
    return max_ws;
  }

  size_t size() const { return cache_.size(); }

  void clear() { cache_.clear(); }
};

//=============================================================================
// 4. MemoryPool with lifetime-based offset assignment
//=============================================================================

/**
 * An allocation within the memory pool.
 *
 * Each allocation has a byte offset into the pool, a size, and a
 * lifetime (the range of contraction steps during which this memory
 * is in use). Two allocations may share the same memory region only
 * if their lifetimes do not overlap.
 */
struct PoolAllocation {
  size_t offset;    // byte offset into the pool
  size_t size;      // size in bytes
  int birth_step;   // step that creates this intermediate
  int death_step;   // last step that reads this intermediate
  int tensor_index; // which intermediate tensor this is for
};

/**
 * Pre-allocated GPU memory pool with lifetime-based offset assignment.
 *
 * The pool is sized and laid out during setup_contraction (planning).
 * During contract (execution), every intermediate tensor is accessed
 * via its pre-computed offset — zero runtime allocation.
 *
 * The layout algorithm:
 *   1. For each intermediate tensor in the contraction path, compute
 *      its size, birth step (when created), and death step (when last
 *      consumed).
 *   2. Sort intermediates by size (largest first).
 *   3. For each intermediate, find the lowest offset where it fits
 *      without overlapping any other allocation that is alive at the
 *      same time.
 *   4. The pool size is max(offset + size) across all allocations.
 *   5. hipTensor workspace is included as a special allocation that
 *      is alive during every step.
 *
 * This is essentially register allocation for GPU memory.
 */
class MemoryPool {
  void *pool_ptr_;       // the single hipMalloc'd region
  size_t pool_size_;     // total size in bytes
  int device_id_;
  bool allocated_;

  // Layout computed during plan_layout()
  std::vector<PoolAllocation> allocations_;

public:
  MemoryPool() : pool_ptr_(nullptr), pool_size_(0), device_id_(0),
                 allocated_(false) {}

  ~MemoryPool() { release(); }

  // Non-copyable, movable
  MemoryPool(const MemoryPool &) = delete;
  MemoryPool &operator=(const MemoryPool &) = delete;
  MemoryPool(MemoryPool &&other) noexcept
      : pool_ptr_(other.pool_ptr_), pool_size_(other.pool_size_),
        device_id_(other.device_id_), allocated_(other.allocated_),
        allocations_(std::move(other.allocations_)) {
    other.pool_ptr_ = nullptr;
    other.allocated_ = false;
  }

  /**
   * Plan the memory layout for a set of intermediate tensors.
   *
   * Each intermediate is described by its size, birth step, and death step.
   * The workspace is a special region alive during all steps.
   *
   * Call this during setup_contraction. After this returns, call
   * allocate() to actually hipMalloc the pool.
   *
   * @param intermediates  List of (size_bytes, birth_step, death_step, index)
   * @param workspace_bytes  hipTensor workspace size (alive during all steps)
   * @param num_steps  Total number of contraction steps
   */
  void plan_layout(
      const std::vector<std::tuple<size_t, int, int, int>> &intermediates,
      size_t workspace_bytes, int num_steps) {
    allocations_.clear();

    // Start with workspace — alive during every step, offset 0
    if (workspace_bytes > 0) {
      PoolAllocation ws;
      ws.offset = 0;
      ws.size = workspace_bytes;
      ws.birth_step = 0;
      ws.death_step = num_steps - 1;
      ws.tensor_index = -1; // sentinel: this is workspace, not a tensor
      allocations_.push_back(ws);
    }

    // Build list of intermediates sorted by size (largest first).
    // Largest-first packing minimizes fragmentation.
    std::vector<size_t> order(intermediates.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) {
                return std::get<0>(intermediates[a]) >
                       std::get<0>(intermediates[b]);
              });

    for (size_t idx : order) {
      auto [size, birth, death, tensor_idx] = intermediates[idx];
      if (size == 0)
        continue;

      // Find the lowest offset where this allocation fits without
      // overlapping any existing allocation with an overlapping lifetime.
      size_t offset = find_offset(size, birth, death);

      PoolAllocation alloc;
      alloc.offset = offset;
      alloc.size = size;
      alloc.birth_step = birth;
      alloc.death_step = death;
      alloc.tensor_index = tensor_idx;
      allocations_.push_back(alloc);
    }

    // Pool size = max(offset + size) across all allocations
    pool_size_ = 0;
    for (const auto &a : allocations_)
      pool_size_ = std::max(pool_size_, a.offset + a.size);

    // Align pool size to 256 bytes
    pool_size_ = ((pool_size_ + 255) / 256) * 256;

    if (memory_verbose()) {
      fprintf(stderr,
              "[AER_TN_MEMORY] pool layout: %zu allocations, "
              "pool size %zu bytes (%.2f MB), workspace %zu bytes\n",
              allocations_.size(), pool_size_,
              pool_size_ / (1024.0 * 1024.0), workspace_bytes);
      for (const auto &a : allocations_) {
        fprintf(stderr,
                "[AER_TN_MEMORY]   tensor %d: offset %zu, size %zu, "
                "alive steps [%d, %d]\n",
                a.tensor_index, a.offset, a.size, a.birth_step,
                a.death_step);
      }
    }
  }

  /**
   * Allocate the pool on the GPU. Call after plan_layout().
   */
  void allocate(int device_id) {
    device_id_ = device_id;
    if (pool_size_ == 0)
      return;

    hipSetDevice(device_id_);
    check_hip(hipMalloc(&pool_ptr_, pool_size_), "hipMalloc(pool)",
              device_id_);
    allocated_ = true;

    if (gpu_verbose()) {
      fprintf(stderr,
              "[AER_TN_GPU] allocated memory pool: %zu bytes (%.2f MB) "
              "on device %d\n",
              pool_size_, pool_size_ / (1024.0 * 1024.0), device_id_);
    }
  }

  /**
   * Get a pointer to the memory region for a given tensor index.
   * This is a constant-time lookup — no allocation happens.
   */
  void *get_tensor_ptr(int tensor_index) const {
    for (const auto &a : allocations_) {
      if (a.tensor_index == tensor_index) {
        return static_cast<char *>(pool_ptr_) + a.offset;
      }
    }
    std::stringstream ss;
    ss << "MemoryPool::get_tensor_ptr: tensor index " << tensor_index
       << " not found in pool layout";
    throw std::runtime_error(ss.str());
  }

  /**
   * Get a pointer to the workspace region.
   * Workspace has tensor_index = -1.
   */
  void *get_workspace_ptr() const {
    for (const auto &a : allocations_) {
      if (a.tensor_index == -1)
        return static_cast<char *>(pool_ptr_) + a.offset;
    }
    return nullptr; // no workspace needed
  }

  /**
   * Get the size of the workspace region.
   */
  size_t get_workspace_size() const {
    for (const auto &a : allocations_) {
      if (a.tensor_index == -1)
        return a.size;
    }
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
  /**
   * Find the lowest offset where an allocation of the given size
   * fits without overlapping any existing allocation whose lifetime
   * overlaps [birth, death].
   *
   * Simple first-fit algorithm: try offset 0, check for conflicts,
   * if conflict found jump past the conflicting allocation, repeat.
   */
  size_t find_offset(size_t size, int birth, int death) {
    size_t offset = 0;
    bool placed = false;

    while (!placed) {
      placed = true;
      for (const auto &existing : allocations_) {
        // Check lifetime overlap
        bool lifetime_overlap =
            (birth <= existing.death_step && death >= existing.birth_step);
        if (!lifetime_overlap)
          continue;

        // Check spatial overlap at this offset
        bool spatial_overlap =
            (offset < existing.offset + existing.size &&
             offset + size > existing.offset);
        if (spatial_overlap) {
          // Jump past this allocation and try again
          offset = existing.offset + existing.size;
          // Align to 256 bytes
          offset = ((offset + 255) / 256) * 256;
          placed = false;
          break;
        }
      }
    }

    return offset;
  }
};

//=============================================================================
// 5. GPUDevice — per-GPU state and operations
//=============================================================================

/**
 * Represents a single GPU (GCD on MI250X) and its resources.
 *
 * Created by GPUResourceManager during discovery. Holds the HIP
 * stream, hipTensor handle, plan cache, and memory pool for this GPU.
 */
template <typename data_t> class GPUDevice {
  int device_id_;
  std::string architecture_;
  size_t total_memory_;
  size_t free_memory_;
  hipStream_t stream_;
  hiptensorHandle_t ht_handle_;
  bool handle_valid_;

  HipTensorPlanCache<data_t> plan_cache_;
  MemoryPool pool_;

  // Peer access capability with other devices
  std::vector<bool> peer_access_;

  // Input tensor data stored on this device
  void *tensor_data_ptr_;
  size_t tensor_data_size_;

  // Output buffer
  thrust::device_vector<thrust::complex<data_t>> dev_out_;

  // Sampling buffers
  thrust::device_vector<double> sampling_rnds_;
  thrust::device_vector<uint64_t> sampling_out_;

public:
  GPUDevice()
      : device_id_(-1), total_memory_(0), free_memory_(0), stream_(nullptr),
        handle_valid_(false), tensor_data_ptr_(nullptr), tensor_data_size_(0) {}

  ~GPUDevice() { release(); }

  // Non-copyable
  GPUDevice(const GPUDevice &) = delete;
  GPUDevice &operator=(const GPUDevice &) = delete;

  /**
   * Initialize this device. Queries properties, creates stream and
   * hipTensor handle. Called once during GPUResourceManager::discover().
   */
  void init(int device_id, int total_device_count) {
    device_id_ = device_id;
    hipSetDevice(device_id_);

    // Query architecture
    hipDeviceProp_t props;
    check_hip(hipGetDeviceProperties(&props, device_id_),
              "hipGetDeviceProperties", device_id_);
    architecture_ = props.gcnArchName;
    total_memory_ = props.totalGlobalMem;

    // Query current free memory
    size_t total;
    check_hip(hipMemGetInfo(&free_memory_, &total), "hipMemGetInfo",
              device_id_);

    // Create stream
    check_hip(
        hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
        "hipStreamCreateWithFlags", device_id_);

    // Create hipTensor handle
    check_hiptensor(hiptensorCreate(&ht_handle_), "hiptensorCreate",
                    device_id_);
    handle_valid_ = true;

    // Initialize plan cache with our handle
    plan_cache_.init(&ht_handle_, device_id_);

    // Query peer access to all other devices
    peer_access_.resize(total_device_count, false);
    for (int other = 0; other < total_device_count; other++) {
      if (other == device_id_)
        continue;
      int can_access = 0;
      hipDeviceCanAccessPeer(&can_access, device_id_, other);
      peer_access_[other] = (can_access != 0);
    }

    if (gpu_verbose()) {
      fprintf(stderr,
              "[AER_TN_GPU] device %d: %s, %.1f GB total, %.1f GB free\n",
              device_id_, architecture_.c_str(),
              total_memory_ / (1024.0 * 1024.0 * 1024.0),
              free_memory_ / (1024.0 * 1024.0 * 1024.0));
    }
  }

  /**
   * Copy input tensor data from host to this GPU.
   * Call after init(), before setup_contraction.
   *
   * The data is stored as a contiguous array of all tensor elements
   * concatenated. The per-tensor offsets (pointers into this array)
   * are returned so the contraction engine knows where each tensor
   * starts.
   *
   * @param tensors  The input tensors (host-side data)
   * @param add_sp_tensors  Whether to include superop tensors
   * @return Per-tensor device pointers
   */
  std::vector<void *> copy_tensor_data(
      const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors,
      bool add_sp_tensors) {
    hipSetDevice(device_id_);

    // Compute total size
    size_t total_elements = 0;
    for (const auto &t : tensors) {
      if (add_sp_tensors || !t->sp_tensor())
        total_elements += t->tensor().size();
    }

    // Allocate
    size_t total_bytes = total_elements * sizeof(std::complex<data_t>);
    if (tensor_data_size_ < total_bytes) {
      if (tensor_data_ptr_)
        hipFree(tensor_data_ptr_);
      check_hip(hipMalloc(&tensor_data_ptr_, total_bytes),
                "hipMalloc(tensor_data)", device_id_);
      tensor_data_size_ = total_bytes;
    }

    // Copy each tensor's data contiguously
    std::vector<void *> ptrs;
    size_t offset = 0;
    for (const auto &t : tensors) {
      if (add_sp_tensors || !t->sp_tensor()) {
        void *dst = static_cast<char *>(tensor_data_ptr_) +
                    offset * sizeof(std::complex<data_t>);
        ptrs.push_back(dst);
        check_hip(
            hipMemcpyAsync(dst, t->tensor().data(),
                           t->tensor().size() * sizeof(std::complex<data_t>),
                           hipMemcpyHostToDevice, stream_),
            "hipMemcpyAsync(tensor_data)", device_id_);
        offset += t->tensor().size();
      }
    }

    hipStreamSynchronize(stream_);

    // Update free memory after tensor copy
    size_t total;
    check_hip(hipMemGetInfo(&free_memory_, &total), "hipMemGetInfo",
              device_id_);

    if (gpu_verbose()) {
      fprintf(stderr,
              "[AER_TN_GPU] copied %zu tensors (%zu elements, %.2f MB) "
              "to device %d, %.1f GB free\n",
              ptrs.size(), total_elements,
              total_bytes / (1024.0 * 1024.0), device_id_,
              free_memory_ / (1024.0 * 1024.0 * 1024.0));
    }

    return ptrs;
  }

  /**
   * Copy tensor data from another GPU to this one.
   * Uses peer-to-peer if available, otherwise stages through host.
   */
  void copy_tensor_data_from(const GPUDevice<data_t> &src) {
    hipSetDevice(device_id_);

    size_t bytes = src.tensor_data_size_;
    if (tensor_data_size_ < bytes) {
      if (tensor_data_ptr_)
        hipFree(tensor_data_ptr_);
      check_hip(hipMalloc(&tensor_data_ptr_, bytes),
                "hipMalloc(tensor_data_peer)", device_id_);
      tensor_data_size_ = bytes;
    }

    if (peer_access_[src.device_id_]) {
      // Direct GPU-to-GPU copy
      if (hipDeviceEnablePeerAccess(src.device_id_, 0) != hipSuccess)
        hipGetLastError(); // ignore "already enabled" error
      check_hip(
          hipMemcpyPeerAsync(tensor_data_ptr_, device_id_,
                             src.tensor_data_ptr_, src.device_id_, bytes,
                             stream_),
          "hipMemcpyPeerAsync", device_id_);
    } else {
      // Stage through host
      std::vector<char> host_buf(bytes);
      hipSetDevice(src.device_id_);
      check_hip(
          hipMemcpy(host_buf.data(), src.tensor_data_ptr_, bytes,
                    hipMemcpyDeviceToHost),
          "hipMemcpy(D2H staging)", src.device_id_);
      hipSetDevice(device_id_);
      check_hip(
          hipMemcpyAsync(tensor_data_ptr_, host_buf.data(), bytes,
                         hipMemcpyHostToDevice, stream_),
          "hipMemcpyAsync(H2D staging)", device_id_);
    }

    hipStreamSynchronize(stream_);
  }

  /**
   * Allocate the output buffer for contraction results.
   */
  void allocate_output(size_t num_elements) {
    hipSetDevice(device_id_);
    dev_out_.resize(num_elements);
  }

  /**
   * Allocate sampling buffers (random numbers and output indices).
   */
  void allocate_sampling_buffers(size_t num_samples) {
    hipSetDevice(device_id_);
    sampling_rnds_.resize(num_samples);
    sampling_out_.resize(num_samples);
  }

  void deallocate_sampling_buffers() {
    hipSetDevice(device_id_);
    sampling_rnds_.clear();
    sampling_rnds_.shrink_to_fit();
    sampling_out_.clear();
    sampling_out_.shrink_to_fit();
  }

  /**
   * Execute a single pairwise contraction using a cached plan.
   * All pointers are pre-computed offsets — no allocation or lookup.
   */
  void execute_contraction(const CachedPlan<data_t> &plan, void *ptr_a,
                           void *ptr_b, void *ptr_c, void *workspace,
                           uint64_t workspace_size) {
    hipSetDevice(device_id_);
    std::complex<data_t> alpha(1.0, 0.0);
    std::complex<data_t> beta(0.0, 0.0);

    check_hiptensor(
        hiptensorContraction(&ht_handle_, &plan.plan, &alpha, ptr_a, ptr_b,
                             &beta, ptr_c, ptr_c, workspace, workspace_size,
                             stream_),
        "hiptensorContraction", device_id_);
  }

  /**
   * Copy contraction output from GPU to host.
   */
  void get_output(std::vector<std::complex<data_t>> &out) {
    hipSetDevice(device_id_);
    size_t n = dev_out_.size();
    if (out.size() < n)
      out.resize(n);
    check_hip(
        hipMemcpyAsync(out.data(),
                       thrust::raw_pointer_cast(dev_out_.data()),
                       n * sizeof(std::complex<data_t>),
                       hipMemcpyDeviceToHost, stream_),
        "hipMemcpyAsync(output D2H)", device_id_);
    hipStreamSynchronize(stream_);
  }

  /**
   * Compute the trace of the output tensor.
   * Sum of diagonal elements: output[i * stride + i] for all i.
   */
  double trace_output(uint64_t num_qubits) {
    hipSetDevice(device_id_);
    uint64_t stride = (1ULL << num_qubits) + 1;
    auto *base =
        (thrust::complex<data_t> *)thrust::raw_pointer_cast(dev_out_.data());
    QV::Chunk::strided_range<thrust::complex<data_t> *> iter(
        base, base + dev_out_.size(), stride);

    thrust::complex<data_t> ret = thrust::reduce(
        thrust_gpu::par.on(stream_), iter.begin(), iter.end());
    return ret.real();
  }

  // --- Accessors ---
  int device_id() const { return device_id_; }
  const std::string &architecture() const { return architecture_; }
  size_t free_memory() const { return free_memory_; }
  size_t total_memory() const { return total_memory_; }
  bool has_peer_access(int other_device) const {
    return peer_access_[other_device];
  }
  hipStream_t stream() const { return stream_; }
  hiptensorHandle_t *handle() { return &ht_handle_; }
  HipTensorPlanCache<data_t> &plan_cache() { return plan_cache_; }
  MemoryPool &pool() { return pool_; }

  thrust::device_vector<thrust::complex<data_t>> &output_buffer() {
    return dev_out_;
  }

  void *tensor_data_ptr() const { return tensor_data_ptr_; }
  size_t tensor_data_size() const { return tensor_data_size_; }

  void release() {
    if (device_id_ < 0)
      return;
    hipSetDevice(device_id_);

    pool_.release();
    plan_cache_.clear();

    if (tensor_data_ptr_) {
      hipFree(tensor_data_ptr_);
      tensor_data_ptr_ = nullptr;
    }
    tensor_data_size_ = 0;

    dev_out_.clear();
    dev_out_.shrink_to_fit();

    deallocate_sampling_buffers();

    if (handle_valid_) {
      hiptensorDestroy(&ht_handle_);
      handle_valid_ = false;
    }
    if (stream_) {
      hipStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  /**
   * Refresh the free memory reading.
   * Call after any allocation to get an accurate budget.
   */
  void refresh_free_memory() {
    hipSetDevice(device_id_);
    size_t total;
    check_hip(hipMemGetInfo(&free_memory_, &total), "hipMemGetInfo",
              device_id_);
  }
};

//=============================================================================
// 6. GPUResourceManager — multi-GPU orchestration
//=============================================================================

/**
 * Discovers available GPUs, manages per-GPU resources, and provides
 * the interface used by the contractor to set up and execute contractions.
 *
 * Created once per TensorNetContractor_HipTensor instance. Discovers
 * GPUs at construction time. All GPU-specific state is accessed through
 * this manager.
 */
template <typename data_t> class GPUResourceManager {
  std::vector<std::unique_ptr<GPUDevice<data_t>>> devices_;
  std::vector<int> device_ids_; // IDs of usable devices

public:
  GPUResourceManager() = default;
  ~GPUResourceManager() = default;

  // Non-copyable
  GPUResourceManager(const GPUResourceManager &) = delete;
  GPUResourceManager &operator=(const GPUResourceManager &) = delete;

  /**
   * Discover available GPUs. Only GPUs with sufficient free memory
   * are included. Called once at contractor construction.
   *
   * @param target_gpus  If non-empty, only consider these device IDs.
   *                     If empty, discover all available devices.
   * @param min_memory   Minimum free memory in bytes to consider a GPU usable.
   */
  void discover(const std::vector<uint64_t> &target_gpus = {},
                size_t min_memory = 256 * 1024 * 1024) {
    int device_count = 0;
    check_hip(hipGetDeviceCount(&device_count), "hipGetDeviceCount");

    if (device_count == 0) {
      throw std::runtime_error(
          "No HIP-capable GPUs found. tensor_network method requires "
          "at least one AMD GPU with hipTensor support.");
    }

    // Determine which devices to probe
    std::vector<int> candidates;
    if (!target_gpus.empty()) {
      for (auto id : target_gpus)
        candidates.push_back(static_cast<int>(id));
    } else {
      for (int i = 0; i < device_count; i++)
        candidates.push_back(i);
    }

    // Initialize each candidate device and check usability
    for (int dev_id : candidates) {
      if (dev_id >= device_count)
        continue;

      auto device = std::make_unique<GPUDevice<data_t>>();
      try {
        device->init(dev_id, device_count);
      } catch (const std::runtime_error &e) {
        if (gpu_verbose()) {
          fprintf(stderr,
                  "[AER_TN_GPU] skipping device %d: %s\n", dev_id, e.what());
        }
        continue;
      }

      if (device->free_memory() < min_memory) {
        if (gpu_verbose()) {
          fprintf(stderr,
                  "[AER_TN_GPU] skipping device %d: only %.1f MB free "
                  "(need %.1f MB)\n",
                  dev_id, device->free_memory() / (1024.0 * 1024.0),
                  min_memory / (1024.0 * 1024.0));
        }
        continue;
      }

      device_ids_.push_back(dev_id);
      devices_.push_back(std::move(device));
    }

    if (devices_.empty()) {
      throw std::runtime_error(
          "No usable GPUs found. All devices have insufficient free memory "
          "or failed initialization.");
    }

    if (gpu_verbose()) {
      fprintf(stderr, "[AER_TN_GPU] discovered %zu usable GPU(s)\n",
              devices_.size());
    }
  }

  /**
   * Query the minimum free memory across all devices.
   * Used as the memory limit for the path optimizer.
   * Called after tensor data has been copied to the primary device.
   */
  size_t min_free_memory() const {
    size_t min_mem = std::numeric_limits<size_t>::max();
    for (const auto &dev : devices_) {
      min_mem = std::min(min_mem, dev->free_memory());
    }
    return min_mem;
  }

  /**
   * Query hipTensor workspace for the worst-case intermediate tensor
   * on the primary device. Used in the two-pass memory planning
   * to determine the real budget for the path optimizer.
   *
   * @param max_intermediate_elements  Size of the largest plausible
   *                                   intermediate tensor in elements.
   * @return Workspace size in bytes.
   */
  uint64_t query_workspace_for_size(int64_t max_intermediate_elements) {
    if (devices_.empty())
      return 0;

    auto &primary = devices_[0];
    hipSetDevice(primary->device_id());

    // Build a dummy contraction descriptor for the worst-case shape.
    // We use a simple square matrix contraction as the workspace estimate
    // because hipTensor's workspace depends mainly on the output size.
    int64_t n = static_cast<int64_t>(
        std::sqrt(static_cast<double>(max_intermediate_elements)));
    if (n < 2) n = 2;

    std::vector<int64_t> extents = {n, n};
    std::vector<int32_t> modes_a = {0, 1};
    std::vector<int32_t> modes_b = {1, 2};
    std::vector<int32_t> modes_c = {0, 2};

    hipDataType hip_dtype = (sizeof(data_t) == 8) ? HIP_C_64F : HIP_C_32F;
    hiptensorComputeType_t compute_type =
        (sizeof(data_t) == 8) ? HIPTENSOR_COMPUTE_64F : HIPTENSOR_COMPUTE_32F;

    hiptensorTensorDescriptor_t da, db, dc;
    hiptensorContractionDescriptor_t desc;
    hiptensorContractionFind_t find;
    uint64_t workspace = 0;
    uint32_t align = 256;

    // These calls may fail for unusual shapes — in that case,
    // return a conservative estimate.
    auto status = hiptensorInitTensorDescriptor(
        primary->handle(), &da, 2, extents.data(), nullptr,
        hip_dtype, HIPTENSOR_OP_IDENTITY);
    if (status != HIPTENSOR_STATUS_SUCCESS)
      return 64 * 1024 * 1024; // 64 MB conservative fallback

    hiptensorInitTensorDescriptor(
        primary->handle(), &db, 2, extents.data(), nullptr,
        hip_dtype, HIPTENSOR_OP_IDENTITY);
    hiptensorInitTensorDescriptor(
        primary->handle(), &dc, 2, extents.data(), nullptr,
        hip_dtype, HIPTENSOR_OP_IDENTITY);

    status = hiptensorInitContractionDescriptor(
        primary->handle(), &desc,
        &da, modes_a.data(), align,
        &db, modes_b.data(), align,
        &dc, modes_c.data(), align,
        &dc, modes_c.data(), align,
        compute_type);
    if (status != HIPTENSOR_STATUS_SUCCESS)
      return 64 * 1024 * 1024;

    hiptensorInitContractionFind(primary->handle(), &find,
                                 HIPTENSOR_ALGO_DEFAULT);
    hiptensorContractionGetWorkspaceSize(
        primary->handle(), &desc, &find,
        HIPTENSOR_WORKSPACE_RECOMMENDED, &workspace);

    if (memory_verbose()) {
      fprintf(stderr,
              "[AER_TN_MEMORY] workspace estimate for %ld-element "
              "intermediate: %lu bytes (%.2f MB)\n",
              (long)max_intermediate_elements, (unsigned long)workspace,
              workspace / (1024.0 * 1024.0));
    }

    return workspace;
  }

  // --- Accessors ---
  size_t num_devices() const { return devices_.size(); }
  GPUDevice<data_t> &device(size_t i) { return *devices_[i]; }
  const GPUDevice<data_t> &device(size_t i) const { return *devices_[i]; }
  GPUDevice<data_t> &primary() { return *devices_[0]; }
  const std::vector<int> &device_ids() const { return device_ids_; }
};

//------------------------------------------------------------------------------
} // end namespace TensorNetwork
} // end namespace AER
//------------------------------------------------------------------------------

#endif // _gpu_resource_manager_hpp_
