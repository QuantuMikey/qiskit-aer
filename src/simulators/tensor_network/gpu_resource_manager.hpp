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

inline void check_hip(hipError_t err, const char *func, int device_id = -1) {
  if (err != hipSuccess) {
    std::stringstream ss;
    ss << "HIP error in " << func << ": " << hipGetErrorString(err);
    if (device_id >= 0)
      ss << " (device " << device_id << ")";
    throw std::runtime_error(ss.str());
  }
}

inline void check_hiptensor(hiptensorStatus_t err, const char *func,
                            int device_id = -1) {
  if (err != HIPTENSOR_STATUS_SUCCESS) {
    std::stringstream ss;
    ss << "hipTensor error in " << func << ": " << hiptensorGetErrorString(err);
    if (device_id >= 0)
      ss << " (device " << device_id << ")";
    throw std::runtime_error(ss.str());
  }
}

//=============================================================================
// 3. ContractionSignature and HipTensorPlanCache
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
                const std::vector<int64_t> &extents_c) {
  ContractionSignature sig;
  sig.data.push_back(static_cast<int32_t>(modes_a.size()));
  for (size_t i = 0; i < modes_a.size(); i++)
    sig.data.push_back(modes_a[i]);
  for (size_t i = 0; i < extents_a.size(); i++)
    sig.data.push_back(static_cast<int32_t>(extents_a[i]));

  sig.data.push_back(static_cast<int32_t>(modes_b.size()));
  for (size_t i = 0; i < modes_b.size(); i++)
    sig.data.push_back(modes_b[i]);
  for (size_t i = 0; i < extents_b.size(); i++)
    sig.data.push_back(static_cast<int32_t>(extents_b[i]));

  sig.data.push_back(static_cast<int32_t>(modes_c.size()));
  for (size_t i = 0; i < modes_c.size(); i++)
    sig.data.push_back(modes_c[i]);
  for (size_t i = 0; i < extents_c.size(); i++)
    sig.data.push_back(static_cast<int32_t>(extents_c[i]));

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
};

template <typename data_t> class HipTensorPlanCache {
  std::unordered_map<ContractionSignature, CachedPlan<data_t>,
                     ContractionSignatureHash>
      cache_;
  hiptensorHandle_t *handle_; // opaque pointer, owned by GPUDevice
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
                const std::vector<int64_t> &extents_c) {
    auto it = cache_.find(sig);
    if (it != cache_.end()) {
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

    // Complex data types for hipTensor
    hipDataType hip_dtype;
    hiptensorComputeType_t compute_type;
    if (sizeof(data_t) == 8) {
      hip_dtype = HIP_C_64F;
      compute_type = HIPTENSOR_COMPUTE_C64F;
    } else {
      hip_dtype = HIP_C_32F;
      compute_type = HIPTENSOR_COMPUTE_C32F;
    }

    uint32_t align = 256;
    check_hiptensor(
        hiptensorInitTensorDescriptor(
            handle_, &cp.desc_a,
            static_cast<uint32_t>(modes_a.size()),
            extents_a.data(), nullptr,
            hip_dtype, HIPTENSOR_OP_IDENTITY),
        "hiptensorInitTensorDescriptor(A)", device_id_);

    check_hiptensor(
        hiptensorInitTensorDescriptor(
            handle_, &cp.desc_b,
            static_cast<uint32_t>(modes_b.size()),
            extents_b.data(), nullptr,
            hip_dtype, HIPTENSOR_OP_IDENTITY),
        "hiptensorInitTensorDescriptor(B)", device_id_);

    check_hiptensor(
        hiptensorInitTensorDescriptor(
            handle_, &cp.desc_c,
            static_cast<uint32_t>(modes_c.size()),
            extents_c.data(), nullptr,
            hip_dtype, HIPTENSOR_OP_IDENTITY),
        "hiptensorInitTensorDescriptor(C)", device_id_);

    check_hiptensor(
        hiptensorInitContractionDescriptor(
            handle_, &cp.desc,
            &cp.desc_a, modes_a.data(), align,
            &cp.desc_b, modes_b.data(), align,
            &cp.desc_c, modes_c.data(), align,
            &cp.desc_c, modes_c.data(), align,
            compute_type),
        "hiptensorInitContractionDescriptor", device_id_);

    check_hiptensor(
        hiptensorInitContractionFind(handle_, &cp.find,
                                     HIPTENSOR_ALGO_DEFAULT),
        "hiptensorInitContractionFind", device_id_);

    cp.workspace_bytes = 0;
    check_hiptensor(
        hiptensorContractionGetWorkspaceSize(
            handle_, &cp.desc, &cp.find,
            HIPTENSOR_WORKSPACE_RECOMMENDED, &cp.workspace_bytes),
        "hiptensorContractionGetWorkspaceSize", device_id_);

    check_hiptensor(
        hiptensorInitContractionPlan(handle_, &cp.plan, &cp.desc,
                                     &cp.find, cp.workspace_bytes),
        "hiptensorInitContractionPlan", device_id_);

    auto result = cache_.emplace(sig, cp);
    return result.first->second;
  }

  uint64_t max_workspace_bytes() const {
    uint64_t max_ws = 0;
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
      max_ws = std::max(max_ws, it->second.workspace_bytes);
    }
    return max_ws;
  }

  size_t size() const { return cache_.size(); }
  void clear() { cache_.clear(); }
};

//=============================================================================
// 4. MemoryPool with lifetime-based offset assignment
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
      ws.offset = 0;
      ws.size = workspace_bytes;
      ws.birth_step = 0;
      ws.death_step = num_steps - 1;
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
      if (sz == 0)
        continue;

      size_t offset = find_offset(sz, birth, death);

      PoolAllocation alloc;
      alloc.offset = offset;
      alloc.size = sz;
      alloc.birth_step = birth;
      alloc.death_step = death;
      alloc.tensor_index = tensor_idx;
      allocations_.push_back(alloc);
    }

    pool_size_ = 0;
    for (size_t i = 0; i < allocations_.size(); i++)
      pool_size_ = std::max(pool_size_, allocations_[i].offset + allocations_[i].size);

    pool_size_ = ((pool_size_ + 255) / 256) * 256;

    if (memory_verbose()) {
      fprintf(stderr,
              "[AER_TN_MEMORY] pool layout: %zu allocations, "
              "pool size %zu bytes (%.2f MB), workspace %zu bytes\n",
              allocations_.size(), pool_size_,
              pool_size_ / (1024.0 * 1024.0), workspace_bytes);
    }
  }

  void allocate(int device_id) {
    device_id_ = device_id;
    if (pool_size_ == 0)
      return;

    hipSetDevice(device_id_);
    check_hip(hipMalloc(&pool_ptr_, pool_size_), "hipMalloc(pool)",
              device_id_);
    allocated_ = true;
  }

  void *get_tensor_ptr(int tensor_index) const {
    for (size_t i = 0; i < allocations_.size(); i++) {
      if (allocations_[i].tensor_index == tensor_index) {
        return static_cast<char *>(pool_ptr_) + allocations_[i].offset;
      }
    }
    std::stringstream ss;
    ss << "MemoryPool::get_tensor_ptr: tensor index " << tensor_index
       << " not found in pool layout";
    throw std::runtime_error(ss.str());
  }

  void *get_workspace_ptr() const {
    for (size_t i = 0; i < allocations_.size(); i++) {
      if (allocations_[i].tensor_index == -1)
        return static_cast<char *>(pool_ptr_) + allocations_[i].offset;
    }
    return nullptr;
  }

  size_t get_workspace_size() const {
    for (size_t i = 0; i < allocations_.size(); i++) {
      if (allocations_[i].tensor_index == -1)
        return allocations_[i].size;
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
  size_t find_offset(size_t size, int birth, int death) {
    size_t offset = 0;
    bool placed = false;

    while (!placed) {
      placed = true;
      for (size_t i = 0; i < allocations_.size(); i++) {
        bool lifetime_overlap =
            (birth <= allocations_[i].death_step &&
             death >= allocations_[i].birth_step);
        if (!lifetime_overlap)
          continue;

        bool spatial_overlap =
            (offset < allocations_[i].offset + allocations_[i].size &&
             offset + size > allocations_[i].offset);
        if (spatial_overlap) {
          offset = allocations_[i].offset + allocations_[i].size;
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

template <typename data_t> class GPUDevice {
  int device_id_;
  std::string architecture_;
  size_t total_memory_;
  size_t free_memory_;
  hipStream_t stream_;
  hiptensorHandle_t *ht_handle_; // opaque pointer from hiptensorCreate
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
    check_hip(hipMemGetInfo(&free_memory_, &total), "hipMemGetInfo",
              device_id_);

    check_hip(
        hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
        "hipStreamCreateWithFlags", device_id_);

    // hiptensorCreate takes hiptensorHandle_t** and allocates the handle
    check_hiptensor(hiptensorCreate(&ht_handle_), "hiptensorCreate",
                    device_id_);
    handle_valid_ = true;

    plan_cache_.init(ht_handle_, device_id_);

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

  std::vector<void *> copy_tensor_data(
      const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors,
      bool add_sp_tensors) {
    hipSetDevice(device_id_);

    size_t total_elements = 0;
    for (size_t i = 0; i < tensors.size(); i++) {
      if (add_sp_tensors || !tensors[i]->sp_tensor())
        total_elements += tensors[i]->tensor().size();
    }

    size_t total_bytes = total_elements * sizeof(std::complex<data_t>);
    if (tensor_data_size_ < total_bytes) {
      if (tensor_data_ptr_)
        hipFree(tensor_data_ptr_);
      check_hip(hipMalloc(&tensor_data_ptr_, total_bytes),
                "hipMalloc(tensor_data)", device_id_);
      tensor_data_size_ = total_bytes;
    }

    std::vector<void *> ptrs;
    size_t offset = 0;
    for (size_t i = 0; i < tensors.size(); i++) {
      if (add_sp_tensors || !tensors[i]->sp_tensor()) {
        void *dst = static_cast<char *>(tensor_data_ptr_) +
                    offset * sizeof(std::complex<data_t>);
        ptrs.push_back(dst);
        check_hip(
            hipMemcpyAsync(dst, tensors[i]->tensor().data(),
                           tensors[i]->tensor().size() * sizeof(std::complex<data_t>),
                           hipMemcpyHostToDevice, stream_),
            "hipMemcpyAsync(tensor_data)", device_id_);
        offset += tensors[i]->tensor().size();
      }
    }

    hipStreamSynchronize(stream_);

    size_t total;
    check_hip(hipMemGetInfo(&free_memory_, &total), "hipMemGetInfo",
              device_id_);

    return ptrs;
  }

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
      if (hipDeviceEnablePeerAccess(src.device_id_, 0) != hipSuccess)
        hipGetLastError();
      check_hip(
          hipMemcpyPeerAsync(tensor_data_ptr_, device_id_,
                             src.tensor_data_ptr_, src.device_id_, bytes,
                             stream_),
          "hipMemcpyPeerAsync", device_id_);
    } else {
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
    sampling_rnds_.clear();
    sampling_rnds_.shrink_to_fit();
    sampling_out_.clear();
    sampling_out_.shrink_to_fit();
  }

  void execute_contraction(const CachedPlan<data_t> &plan, void *ptr_a,
                           void *ptr_b, void *ptr_c, void *workspace,
                           uint64_t workspace_size) {
    hipSetDevice(device_id_);
    std::complex<data_t> alpha(1.0, 0.0);
    std::complex<data_t> beta(0.0, 0.0);

    // ht_handle_ is already hiptensorHandle_t* — pass directly
    check_hiptensor(
        hiptensorContraction(ht_handle_, &plan.plan, &alpha, ptr_a, ptr_b,
                             &beta, ptr_c, ptr_c, workspace, workspace_size,
                             stream_),
        "hiptensorContraction", device_id_);
  }

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

  // Returns the opaque handle pointer for hipTensor API calls
  hiptensorHandle_t *handle() { return ht_handle_; }

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
      // hiptensorDestroy takes hiptensorHandle_t* directly
      hiptensorDestroy(ht_handle_);
      ht_handle_ = nullptr;
      handle_valid_ = false;
    }
    if (stream_) {
      hipStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

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
    int device_count = 0;
    check_hip(hipGetDeviceCount(&device_count), "hipGetDeviceCount");

    if (device_count == 0) {
      throw std::runtime_error(
          "No HIP-capable GPUs found. tensor_network method requires "
          "at least one AMD GPU with hipTensor support.");
    }

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
      if (dev_id >= device_count)
        continue;

      auto device = std::unique_ptr<GPUDevice<data_t>>(new GPUDevice<data_t>());
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

  size_t min_free_memory() const {
    size_t min_mem = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < devices_.size(); i++) {
      min_mem = std::min(min_mem, devices_[i]->free_memory());
    }
    return min_mem;
  }

  uint64_t query_workspace_for_size(int64_t max_intermediate_elements) {
    if (devices_.empty())
      return 0;

    GPUDevice<data_t> &primary = *devices_[0];
    hipSetDevice(primary.device_id());

    int64_t n = static_cast<int64_t>(
        std::sqrt(static_cast<double>(max_intermediate_elements)));
    if (n < 2) n = 2;

    std::vector<int64_t> extents = {n, n};
    std::vector<int32_t> modes_a = {0, 1};
    std::vector<int32_t> modes_b = {1, 2};
    std::vector<int32_t> modes_c = {0, 2};

    hipDataType hip_dtype = (sizeof(data_t) == 8) ? HIP_C_64F : HIP_C_32F;
    hiptensorComputeType_t compute_type =
        (sizeof(data_t) == 8) ? HIPTENSOR_COMPUTE_C64F : HIPTENSOR_COMPUTE_C32F;

    hiptensorTensorDescriptor_t da, db, dc;
    hiptensorContractionDescriptor_t desc;
    hiptensorContractionFind_t find;
    uint64_t workspace = 0;
    uint32_t align = 256;

    auto status = hiptensorInitTensorDescriptor(
        primary.handle(), &da, 2, extents.data(), nullptr,
        hip_dtype, HIPTENSOR_OP_IDENTITY);
    if (status != HIPTENSOR_STATUS_SUCCESS)
      return 64 * 1024 * 1024; // 64 MB conservative fallback

    hiptensorInitTensorDescriptor(
        primary.handle(), &db, 2, extents.data(), nullptr,
        hip_dtype, HIPTENSOR_OP_IDENTITY);
    hiptensorInitTensorDescriptor(
        primary.handle(), &dc, 2, extents.data(), nullptr,
        hip_dtype, HIPTENSOR_OP_IDENTITY);

    status = hiptensorInitContractionDescriptor(
        primary.handle(), &desc,
        &da, modes_a.data(), align,
        &db, modes_b.data(), align,
        &dc, modes_c.data(), align,
        &dc, modes_c.data(), align,
        compute_type);
    if (status != HIPTENSOR_STATUS_SUCCESS)
      return 64 * 1024 * 1024;

    hiptensorInitContractionFind(primary.handle(), &find,
                                 HIPTENSOR_ALGO_DEFAULT);
    hiptensorContractionGetWorkspaceSize(
        primary.handle(), &desc, &find,
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
