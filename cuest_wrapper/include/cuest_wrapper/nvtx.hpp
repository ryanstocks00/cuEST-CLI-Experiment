#pragma once
/**
 * @file nvtx.hpp
 * @brief RAII NVTX range push/pop for Nsight Systems / nvprof timelines.
 *
 * Usage:
 *   {
 *     NvtxRange range("cuestAOBasisCreate");
 *     CUEST_CHECK(cuestAOBasisCreate(...));
 *   }
 *   // or: CUEST_NVTX("name", cuestFoo(...));
 */

#include <nvtx3/nvToolsExt.h>

#include "raii.hpp"

namespace cuest {

/// Push an NVTX range on construction; pop on destruction (including exceptions).
class NvtxRange {
 public:
  explicit NvtxRange(const char* name) noexcept {
    nvtxRangePushA(name);
  }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
  ~NvtxRange() { nvtxRangePop(); }
};

}  // namespace cuest

/// Run a cuEST call inside an NVTX range, then check status.
#define CUEST_NVTX(name, call)                                                 \
  do {                                                                         \
    ::cuest::NvtxRange _cuest_nvtx_range(name);                                \
    ::cuest::check((call), #call, __FILE__, __LINE__);                         \
  } while (0)
