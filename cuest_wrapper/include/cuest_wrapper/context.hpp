#pragma once
/**
 * @file context.hpp
 * @brief cuEST context (global handle) management.
 */

#include <cuest.h>

#include <cstdint>
#include <string>

#include "nvtx.hpp"
#include "raii.hpp"

namespace cuest {

/// Handle-level cuEST tuning knobs (CUEST_HANDLE_PARAMETERS_*). Defaults here
/// match cuEST's own defaults, so a default-constructed config reproduces the
/// previous unconditional-defaults behavior exactly.
struct CuESTContextConfig {
  uint64_t max_gauss_hermite{20};
  uint64_t max_l_solid_harmonic{10};
  uint64_t max_rys{0};  // 0 = largest available
  std::string jit_cache_dir;  // "" = derived default (~/.cuest_cache/...)
  int32_t jit_compile_threads{16};
};

class CuESTContext {
 public:
  CuESTContext() : CuESTContext(CuESTContextConfig{}) {}

  explicit CuESTContext(const CuESTContextConfig& cfg) {
    cuestHandleParameters_t hp = nullptr;
    CUEST_CHECK(cuestParametersCreate(CUEST_HANDLE_PARAMETERS, &hp));
    cuestParametersConfigure(CUEST_HANDLE_PARAMETERS, hp,
        CUEST_HANDLE_PARAMETERS_MAX_GAUSS_HERMITE,
        &cfg.max_gauss_hermite, sizeof(cfg.max_gauss_hermite));
    cuestParametersConfigure(CUEST_HANDLE_PARAMETERS, hp,
        CUEST_HANDLE_PARAMETERS_MAX_L_SOLID_HARMONIC,
        &cfg.max_l_solid_harmonic, sizeof(cfg.max_l_solid_harmonic));
    cuestParametersConfigure(CUEST_HANDLE_PARAMETERS, hp,
        CUEST_HANDLE_PARAMETERS_MAX_RYS,
        &cfg.max_rys, sizeof(cfg.max_rys));
    // The attribute value being configured is itself a const char*, so
    // cuestParametersConfigure needs the address of that pointer variable
    // (not the pointer's value) — passing the string pointer directly here
    // previously caused cuEST to read garbage as a pointer and segfault.
    const char* jit_cache_dir = cfg.jit_cache_dir.c_str();
    cuestParametersConfigure(CUEST_HANDLE_PARAMETERS, hp,
        CUEST_HANDLE_PARAMETERS_JIT_CACHE_DIR,
        &jit_cache_dir, sizeof(jit_cache_dir));
    cuestParametersConfigure(CUEST_HANDLE_PARAMETERS, hp,
        CUEST_HANDLE_PARAMETERS_JIT_COMPILE_THREADS,
        &cfg.jit_compile_threads, sizeof(cfg.jit_compile_threads));

    cuestStatus_t st;
    {
      NvtxRange range("cuestCreate");
      st = cuestCreate(hp, &handle_);
    }
    cuestParametersDestroy(CUEST_HANDLE_PARAMETERS, hp);
    CUEST_CHECK(st);  // check after destroying hp to avoid leak
  }

  ~CuESTContext() {
    if (handle_) cuestDestroy(handle_);
  }

  CuESTContext(const CuESTContext&) = delete;
  CuESTContext& operator=(const CuESTContext&) = delete;
  CuESTContext(CuESTContext&& other) noexcept
      : handle_(other.handle_), scratch_(std::move(other.scratch_)) {
    other.handle_ = nullptr;
  }
  CuESTContext& operator=(CuESTContext&& other) noexcept {
    if (this != &other) {
      if (handle_) cuestDestroy(handle_);
      handle_ = other.handle_;
      scratch_ = std::move(other.scratch_);
      other.handle_ = nullptr;
    }
    return *this;
  }

  cuestHandle_t get() const { return handle_; }
  operator cuestHandle_t() const { return handle_; }

  // Query helpers — prefer Handle::query when you hold an RAII handle.
  template <typename T, typename AttrT>
  T query(cuestType_t type, void* obj, AttrT attr) const {
    T val{};
    CUEST_CHECK(cuestQuery(handle_, type, obj, attr, &val, sizeof(val)));
    return val;
  }

  template <typename T, typename HandleT, auto DestroyFn, cuestType_t Type,
            typename AttrT>
  T query(const Handle<HandleT, DestroyFn, Type>& obj, AttrT attr) const {
    return obj.template query<T>(handle_, attr);
  }

  uint64_t query_nao(const AOBasisHandle& basis) const {
    return basis.query<uint64_t>(handle_, CUEST_AOBASIS_NUM_AO);
  }

  uint64_t query_nao(cuestAOBasis_t basis) const {
    uint64_t nao = 0;
    CUEST_CHECK(cuestQuery(handle_, CUEST_AOBASIS, basis, CUEST_AOBASIS_NUM_AO,
                            &nao, sizeof(nao)));
    return nao;
  }

  /// Grow-only scratch shared by all integral / gradient computes.
  /// Peak device usage is max(query), not sum — callers must not overlap.
  Workspace& scratch() { return scratch_; }

 private:
  cuestHandle_t handle_{nullptr};
  Workspace scratch_;
};

}  // namespace cuest
