#pragma once
/**
 * @file raii.hpp
 * @brief C++ RAII wrappers around cuEST C opaque handles.
 *
 * Every cuEST C handle type gets a template wrapper that:
 *   - Destroys the handle in the destructor
 *   - Is move-only (no copies)
 *   - Provides implicit conversion to the raw handle for API calls
 *   - Carries a cuestType_t tag for typed cuestQuery helpers
 */

#include <cuda_runtime.h>
#include <cuest.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cuest {

// ---------------------------------------------------------------------------
// Status checking
// ---------------------------------------------------------------------------
inline void check(cuestStatus_t status, const char* expr,
                  const char* file, int line) {
  if (status != CUEST_STATUS_SUCCESS) {
    std::string msg = std::string("cuEST error: ") + expr + " at " + file +
                      ":" + std::to_string(line) + " code=" +
                      std::to_string(static_cast<int>(status));
    throw std::runtime_error(msg);
  }
}
#define CUEST_CHECK(call) cuest::check((call), #call, __FILE__, __LINE__)

inline void check_cuda(cudaError_t err, const char* expr,
                       const char* file, int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + expr + " at " +
                             file + ":" + std::to_string(line) + " — " +
                             cudaGetErrorString(err));
  }
}
#define CUDA_CHECK(call) cuest::check_cuda((call), #call, __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// Forward declarations of parameter wrappers
// ---------------------------------------------------------------------------
template <cuestParametersType_t Type>
class Parameters;

// ---------------------------------------------------------------------------
// Generic RAII handle wrapper
// ---------------------------------------------------------------------------
// Type is the cuestType_t tag for cuestQuery (e.g. CUEST_AOBASIS).
template <typename HandleT, auto DestroyFn, cuestType_t Type>
class Handle {
 public:
  static constexpr cuestType_t object_type = Type;

  Handle() : h_(nullptr) {}
  explicit Handle(HandleT h) : h_(h) {}
  ~Handle() { reset(); }

  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;

  Handle(Handle&& other) noexcept : h_(other.h_) { other.h_ = nullptr; }
  Handle& operator=(Handle&& other) noexcept {
    if (this != &other) {
      reset();
      h_ = other.h_;
      other.h_ = nullptr;
    }
    return *this;
  }

  [[nodiscard]] HandleT get() const { return h_; }
  operator HandleT() const { return h_; }
  [[nodiscard]] bool valid() const { return h_ != nullptr; }

  HandleT* ptr() { return &h_; }

  void reset() noexcept {
    if (h_) {
      cuestStatus_t st = DestroyFn(h_);
      if (st != CUEST_STATUS_SUCCESS) {
        std::cerr << "Warning: cuEST destroy failed with status " << st << "\n";
      }
      h_ = nullptr;
    }
  }

  HandleT release() noexcept {
    HandleT tmp = h_;
    h_ = nullptr;
    return tmp;
  }

  /// Typed cuestQuery using this handle's object type tag.
  template <typename T, typename AttrT>
  [[nodiscard]] T query(cuestHandle_t ctx, AttrT attr) const {
    T val{};
    CUEST_CHECK(cuestQuery(ctx, Type, h_, attr, &val, sizeof(val)));
    return val;
  }

 private:
  HandleT h_;
};

// ---------------------------------------------------------------------------
// Concrete handle types
// ---------------------------------------------------------------------------
using CuESTHandle = Handle<cuestHandle_t, cuestDestroy, CUEST_HANDLE>;
using AOShellHandle = Handle<cuestAOShell_t, cuestAOShellDestroy, CUEST_AOSHELL>;
using AOBasisHandle =
    Handle<cuestAOBasis_t, cuestAOBasisDestroy, CUEST_AOBASIS>;
using AOPairListHandle =
    Handle<cuestAOPairList_t, cuestAOPairListDestroy, CUEST_AOPAIRLIST>;
using ECPShellHandle =
    Handle<cuestECPShell_t, cuestECPShellDestroy, CUEST_ECPSHELL>;
using ECPAtomHandle =
    Handle<cuestECPAtom_t, cuestECPAtomDestroy, CUEST_ECPATOM>;
using OEIntPlanHandle =
    Handle<cuestOEIntPlan_t, cuestOEIntPlanDestroy, CUEST_OEINTPLAN>;
using ECPIntPlanHandle =
    Handle<cuestECPIntPlan_t, cuestECPIntPlanDestroy, CUEST_ECPINTPLAN>;
using DFIntPlanHandle =
    Handle<cuestDFIntPlan_t, cuestDFIntPlanDestroy, CUEST_DFINTPLAN>;
using AtomGridHandle =
    Handle<cuestAtomGrid_t, cuestAtomGridDestroy, CUEST_ATOMGRID>;
using MolecularGridHandle =
    Handle<cuestMolecularGrid_t, cuestMolecularGridDestroy,
           CUEST_MOLECULARGRID>;
using XCIntPlanHandle =
    Handle<cuestXCIntPlan_t, cuestXCIntPlanDestroy, CUEST_XCINTPLAN>;

// ---------------------------------------------------------------------------
// Device memory RAII wrapper (typed GPU buffer; size is in bytes)
// ---------------------------------------------------------------------------
template <typename T>
class DeviceArray {
 public:
  DeviceArray() = default;
  explicit DeviceArray(size_t bytes) { alloc(bytes); }

  ~DeviceArray() {
    if (ptr_) cudaFree(ptr_);
  }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  DeviceArray(DeviceArray&& other) noexcept
      : ptr_(other.ptr_), capacity_(other.capacity_) {
    other.ptr_ = nullptr;
    other.capacity_ = 0;
  }
  DeviceArray& operator=(DeviceArray&& other) noexcept {
    if (this != &other) {
      if (ptr_) cudaFree(ptr_);
      ptr_ = other.ptr_;
      capacity_ = other.capacity_;
      other.ptr_ = nullptr;
      other.capacity_ = 0;
    }
    return *this;
  }

  /// Free and allocate exactly `bytes` (0 frees).
  void alloc(size_t bytes) {
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    capacity_ = 0;
    if (bytes == 0) return;
    T* fresh = nullptr;
    if (cudaMalloc(&fresh, bytes) != cudaSuccess) {
      throw std::runtime_error("Failed to allocate GPU memory (" +
                               std::to_string(bytes) + " bytes)");
    }
    ptr_ = fresh;
    capacity_ = bytes;
  }

  /// Grow-only: no-op if capacity already covers `bytes`.
  void ensure(size_t bytes) {
    if (bytes <= capacity_) return;
    alloc(bytes);
  }

  [[nodiscard]] size_t capacity() const { return capacity_; }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  operator T*() { return ptr_; }
  operator const T*() const { return ptr_; }
  bool valid() const { return ptr_ != nullptr; }

  std::vector<T> copy_to_host(size_t count) const {
    std::vector<T> host(count);
    if (cudaMemcpy(host.data(), ptr_, count * sizeof(T),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("Device to host copy failed");
    }
    return host;
  }

  void copy_from_host(const std::vector<T>& host) {
    copy_from_host(host.data(), host.size());
  }

  void copy_from_host(const T* host, size_t count) {
    if (cudaMemcpy(ptr_, host, count * sizeof(T),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("Host to device copy failed");
    }
  }

 private:
  T* ptr_{nullptr};
  size_t capacity_{0};
};

// ---------------------------------------------------------------------------
// Workspace RAII wrapper (cuEST scratch: host vector + device bytes)
// ---------------------------------------------------------------------------
class Workspace {
 public:
  Workspace() = default;

  explicit Workspace(const cuestWorkspaceDescriptor_t& desc) { ensure(desc); }

  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  Workspace(Workspace&& other) noexcept
      : ws_(other.ws_),
        host_buf_(std::move(other.host_buf_)),
        dev_buf_(std::move(other.dev_buf_)) {
    other.ws_ = {};
  }

  Workspace& operator=(Workspace&& other) noexcept {
    if (this != &other) {
      ws_ = other.ws_;
      host_buf_ = std::move(other.host_buf_);
      dev_buf_ = std::move(other.dev_buf_);
      other.ws_ = {};
    }
    return *this;
  }

  /// Grow host/device scratch to cover `desc`; reuse capacity across calls.
  void ensure(const cuestWorkspaceDescriptor_t& desc) {
    if (desc.hostBufferSizeInBytes > host_buf_.size())
      host_buf_.resize(desc.hostBufferSizeInBytes);
    if (desc.deviceBufferSizeInBytes > 0)
      dev_buf_.ensure(desc.deviceBufferSizeInBytes);

    ws_.hostBuffer = host_buf_.empty()
                         ? 0
                         : reinterpret_cast<uintptr_t>(host_buf_.data());
    // Report available capacity so cuEST can reuse a larger pooled buffer.
    ws_.hostBufferSizeInBytes = host_buf_.size();
    ws_.deviceBuffer = reinterpret_cast<uintptr_t>(dev_buf_.get());
    ws_.deviceBufferSizeInBytes = dev_buf_.capacity();
  }

  cuestWorkspace_t* ptr() { return &ws_; }
  cuestWorkspace_t& get() { return ws_; }

 private:
  cuestWorkspace_t ws_{};
  std::vector<uint8_t> host_buf_;
  DeviceArray<uint8_t> dev_buf_;
};

// ---------------------------------------------------------------------------
// Parameter RAII wrapper (create/destroy is type-specific)
// Each parameter type is typedef'd to void*, so we store void*.
// ---------------------------------------------------------------------------
template <cuestParametersType_t Type>
class Parameters {
 public:
  Parameters() : params_(nullptr) {
    CUEST_CHECK(cuestParametersCreate(Type, &params_));
  }
  ~Parameters() {
    if (params_)
      cuestParametersDestroy(Type, params_);
  }

  Parameters(const Parameters&) = delete;
  Parameters& operator=(const Parameters&) = delete;
  Parameters(Parameters&& other) noexcept : params_(other.params_) {
    other.params_ = nullptr;
  }
  Parameters& operator=(Parameters&& other) noexcept {
    if (this != &other) {
      if (params_) cuestParametersDestroy(Type, params_);
      params_ = other.params_;
      other.params_ = nullptr;
    }
    return *this;
  }

  void* get() const { return params_; }
  operator void*() const { return params_; }

 private:
  void* params_{nullptr};
};

// Type aliases for all parameter types
using HandleParams = Parameters<CUEST_HANDLE_PARAMETERS>;
using AOShellParams = Parameters<CUEST_AOSHELL_PARAMETERS>;
using AOBasisParams = Parameters<CUEST_AOBASIS_PARAMETERS>;
using AOPairListParams = Parameters<CUEST_AOPAIRLIST_PARAMETERS>;
using ECPShellParams = Parameters<CUEST_ECPSHELL_PARAMETERS>;
using ECPAtomParams = Parameters<CUEST_ECPATOM_PARAMETERS>;
using OEIntPlanParams = Parameters<CUEST_OEINTPLAN_PARAMETERS>;
using ECPIntPlanParams = Parameters<CUEST_ECPINTPLAN_PARAMETERS>;
using DFIntPlanParams = Parameters<CUEST_DFINTPLAN_PARAMETERS>;
using AtomGridParams = Parameters<CUEST_ATOMGRID_PARAMETERS>;
using MolecularGridParams = Parameters<CUEST_MOLECULARGRID_PARAMETERS>;
using XCIntPlanParams = Parameters<CUEST_XCINTPLAN_PARAMETERS>;
using OverlapComputeParams = Parameters<CUEST_OVERLAPCOMPUTE_PARAMETERS>;
using KineticComputeParams = Parameters<CUEST_KINETICCOMPUTE_PARAMETERS>;
using PotentialComputeParams = Parameters<CUEST_POTENTIALCOMPUTE_PARAMETERS>;
using DFCoulombComputeParams =
    Parameters<CUEST_DFCOULOMBCOMPUTE_PARAMETERS>;
using DFSymmetricExchangeComputeParams =
    Parameters<CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS>;
using ECPComputeParams = Parameters<CUEST_ECPCOMPUTE_PARAMETERS>;
using XCPotentialRKSComputeParams =
    Parameters<CUEST_XCPOTENTIALRKSCOMPUTE_PARAMETERS>;
using XCPotentialUKSComputeParams =
    Parameters<CUEST_XCPOTENTIALUKSCOMPUTE_PARAMETERS>;

}  // namespace cuest
