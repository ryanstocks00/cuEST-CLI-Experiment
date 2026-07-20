#pragma once
/**
 * @file context.hpp
 * @brief cuEST context (global handle) management.
 */

#include <cuest.h>

#include <cstdint>

#include "raii.hpp"

namespace cuest {

class CuESTContext {
 public:
  CuESTContext() {
    cuestHandleParameters_t hp = nullptr;
    CUEST_CHECK(cuestParametersCreate(CUEST_HANDLE_PARAMETERS, &hp));
    cuestStatus_t st = cuestCreate(hp, &handle_);
    cuestParametersDestroy(CUEST_HANDLE_PARAMETERS, hp);
    CUEST_CHECK(st);  // check after destroying hp to avoid leak
  }

  ~CuESTContext() {
    if (handle_) cuestDestroy(handle_);
  }

  CuESTContext(const CuESTContext&) = delete;
  CuESTContext& operator=(const CuESTContext&) = delete;
  CuESTContext(CuESTContext&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }
  CuESTContext& operator=(CuESTContext&& other) noexcept {
    if (this != &other) {
      if (handle_) cuestDestroy(handle_);
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  cuestHandle_t get() const { return handle_; }
  operator cuestHandle_t() const { return handle_; }

  // Query helpers
  template <typename T, typename AttrT>
  T query(cuestType_t type, void* obj, AttrT attr) const {
    T val{};
    CUEST_CHECK(cuestQuery(handle_, type, obj, attr, &val, sizeof(val)));
    return val;
  }

  // Query AO basis num_ao
  uint64_t query_nao(cuestAOBasis_t basis) const {
    uint64_t nao = 0;
    CUEST_CHECK(cuestQuery(handle_, CUEST_AOBASIS, basis, CUEST_AOBASIS_NUM_AO,
                            &nao, sizeof(nao)));
    return nao;
  }

 private:
  cuestHandle_t handle_{nullptr};
};

}  // namespace cuest
