#pragma once
/**
 * @file functionals.hpp
 * @brief Canonical string ↔ XCBuilder::Functional registry (CLI boundary only).
 */
#include <cstring>
#include <optional>
#include <string_view>

#include "cuest_wrapper/integrals.hpp"

namespace cuest {

struct FunctionalInfo {
  const char* name;
  XCBuilder::Functional id;
};

/// Single source of truth for CLI names. Order matches XCBuilder::Functional.
inline constexpr FunctionalInfo kFunctionals[] = {
    {"HF", XCBuilder::XC_HF},
    {"PBE", XCBuilder::XC_PBE},
    {"B3LYP", XCBuilder::XC_B3LYP},
    {"B3LYP5", XCBuilder::XC_B3LYP5},
    {"PBE0", XCBuilder::XC_PBE0},
    {"CAM-B3LYP", XCBuilder::XC_CAM_B3LYP},
    {"WB97X-V", XCBuilder::XC_WB97X_V},
    {"WB97M-V", XCBuilder::XC_WB97M_V},
    {"HSE06", XCBuilder::XC_HSE06},
    {"M06", XCBuilder::XC_M06},
    {"M06-2X", XCBuilder::XC_M062X},
    {"LC-WPBE", XCBuilder::XC_LC_WPBE},
    {"LC-WPBEH", XCBuilder::XC_LC_WPBEH},
    {"WB97X", XCBuilder::XC_WB97X},
};

[[nodiscard]] inline std::optional<XCBuilder::Functional> functional_from_name(
    std::string_view name) {
  for (const auto& fi : kFunctionals) {
    if (name == fi.name) return fi.id;
  }
  return std::nullopt;
}

[[nodiscard]] inline const char* functional_to_name(XCBuilder::Functional id) {
  for (const auto& fi : kFunctionals) {
    if (fi.id == id) return fi.name;
  }
  return "PBE";
}

}  // namespace cuest
