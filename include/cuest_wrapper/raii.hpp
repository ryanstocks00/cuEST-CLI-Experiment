#pragma once
/**
 * @file raii.hpp
 * @brief C++ RAII wrappers around cuEST C opaque handles.
 *
 * Every cuEST C handle type gets a template wrapper that:
 *   - Destroys the handle in the destructor
 *   - Is move-only (no copies)
 *   - Provides implicit conversion to the raw handle for API calls
 *   - Has factory static methods for standard create/destroy lifecycle
 */

#include <cuest.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

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

// ---------------------------------------------------------------------------
// Forward declarations of parameter wrappers
// ---------------------------------------------------------------------------
template <cuestParametersType_t Type>
class Parameters;

// ---------------------------------------------------------------------------
// Generic RAII handle wrapper
// ---------------------------------------------------------------------------
template <typename HandleT, auto DestroyFn, auto... QueryFns>
class Handle {
 public:
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

  void reset() {
    if (h_) {
      cuestStatus_t st = DestroyFn(h_);
      if (st != CUEST_STATUS_SUCCESS) {
        std::cerr << "Warning: cuEST destroy failed with status " << st << "\n";
      }
      h_ = nullptr;
    }
  }

  HandleT release() {
    HandleT tmp = h_;
    h_ = nullptr;
    return tmp;
  }

 private:
  HandleT h_;
};

// ---------------------------------------------------------------------------
// Concrete handle types
// ---------------------------------------------------------------------------

// cuEST global handle
struct CuEST {
  using HandleT = cuestHandle_t;
  static cuestStatus_t destroy(cuestHandle_t h) { return cuestDestroy(h); }
};
using CuESTHandle = Handle<cuestHandle_t, cuestDestroy, CUEST_HANDLE>;

// AO Shell
struct AOShellTraits {
  using HandleT = cuestAOShell_t;
  static cuestStatus_t destroy(cuestAOShell_t h) { return cuestAOShellDestroy(h); }
};
using AOShellHandle = Handle<cuestAOShell_t, cuestAOShellDestroy, CUEST_AOSHELL>;

// AO Basis
using AOBasisHandle =
    Handle<cuestAOBasis_t, cuestAOBasisDestroy, CUEST_AOBASIS>;

// AO PairList
using AOPairListHandle =
    Handle<cuestAOPairList_t, cuestAOPairListDestroy, CUEST_AOPAIRLIST>;

// ECP Shell
using ECPShellHandle =
    Handle<cuestECPShell_t, cuestECPShellDestroy, CUEST_ECPSHELL>;

// ECP Atom
using ECPAtomHandle =
    Handle<cuestECPAtom_t, cuestECPAtomDestroy, CUEST_ECPATOM>;

// One-electron integral plan
using OEIntPlanHandle =
    Handle<cuestOEIntPlan_t, cuestOEIntPlanDestroy, CUEST_OEINTPLAN>;

// ECP integral plan
using ECPIntPlanHandle =
    Handle<cuestECPIntPlan_t, cuestECPIntPlanDestroy, CUEST_ECPINTPLAN>;

// DF integral plan
using DFIntPlanHandle =
    Handle<cuestDFIntPlan_t, cuestDFIntPlanDestroy, CUEST_DFINTPLAN>;

// Atom grid
using AtomGridHandle =
    Handle<cuestAtomGrid_t, cuestAtomGridDestroy, CUEST_ATOMGRID>;

// Molecular grid
using MolecularGridHandle =
    Handle<cuestMolecularGrid_t, cuestMolecularGridDestroy,
           CUEST_MOLECULARGRID>;

// XC integral plan
using XCIntPlanHandle =
    Handle<cuestXCIntPlan_t, cuestXCIntPlanDestroy, CUEST_XCINTPLAN>;

// ---------------------------------------------------------------------------
// Workspace RAII wrapper
// ---------------------------------------------------------------------------
class Workspace {
 public:
  Workspace() = default;

  explicit Workspace(const cuestWorkspaceDescriptor_t& desc) {
    if (desc.hostBufferSizeInBytes) {
      host_buf_ = std::unique_ptr<uint8_t[]>{
          new uint8_t[desc.hostBufferSizeInBytes]};
    }
    if (desc.deviceBufferSizeInBytes) {
      uint8_t* dev = nullptr;
      if (cudaMalloc(&dev, desc.deviceBufferSizeInBytes) != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device workspace");
      }
      dev_buf_ = dev;
    }
    ws_.hostBuffer = reinterpret_cast<uintptr_t>(host_buf_.get());
    ws_.hostBufferSizeInBytes = desc.hostBufferSizeInBytes;
    ws_.deviceBuffer = reinterpret_cast<uintptr_t>(dev_buf_);
    ws_.deviceBufferSizeInBytes = desc.deviceBufferSizeInBytes;
  }

  ~Workspace() {
    if (dev_buf_) cudaFree(dev_buf_);
  }

  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  Workspace(Workspace&& other) noexcept
      : ws_(other.ws_),
        host_buf_(std::move(other.host_buf_)),
        dev_buf_(other.dev_buf_) {
    other.ws_ = {};
    other.dev_buf_ = nullptr;
  }

  Workspace& operator=(Workspace&& other) noexcept {
    if (this != &other) {
      if (dev_buf_) cudaFree(dev_buf_);
      ws_ = other.ws_;
      host_buf_ = std::move(other.host_buf_);
      dev_buf_ = other.dev_buf_;
      other.ws_ = {};
      other.dev_buf_ = nullptr;
    }
    return *this;
  }

  cuestWorkspace_t* ptr() { return &ws_; }
  cuestWorkspace_t& get() { return ws_; }

 private:
  cuestWorkspace_t ws_{};
  std::unique_ptr<uint8_t[]> host_buf_{nullptr};
  uint8_t* dev_buf_{nullptr};
};

// ---------------------------------------------------------------------------
// Device memory RAII wrapper
// ---------------------------------------------------------------------------
class DeviceArray {
 public:
  DeviceArray() = default;
  explicit DeviceArray(size_t bytes) { alloc(bytes); }

  ~DeviceArray() {
    if (ptr_) cudaFree(ptr_);
  }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  DeviceArray(DeviceArray&& other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }
  DeviceArray& operator=(DeviceArray&& other) noexcept {
    if (this != &other) {
      if (ptr_) cudaFree(ptr_);
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  void alloc(size_t bytes) {
    if (ptr_) cudaFree(ptr_);
    if (cudaMalloc(&ptr_, bytes) != cudaSuccess) {
      throw std::runtime_error("Failed to allocate GPU memory (" +
                               std::to_string(bytes) + " bytes)");
    }
  }

  double* get() { return ptr_; }
  const double* get() const { return ptr_; }
  operator double*() { return ptr_; }
  operator const double*() const { return ptr_; }
  bool valid() const { return ptr_ != nullptr; }

  std::vector<double> copy_to_host(size_t count) const {
    std::vector<double> host(count);
    if (cudaMemcpy(host.data(), ptr_, count * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("Device to host copy failed");
    }
    return host;
  }

  void copy_from_host(const std::vector<double>& host) {
    copy_from_host(host.data(), host.size());
  }

  void copy_from_host(const double* host, size_t count) {
    if (cudaMemcpy(ptr_, host, count * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("Host to device copy failed");
    }
  }

 private:
  double* ptr_{nullptr};
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
