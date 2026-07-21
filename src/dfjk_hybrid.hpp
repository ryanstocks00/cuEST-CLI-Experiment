#pragma once
/**
 * @file dfjk_hybrid.hpp
 * @brief HF-exchange fractions to bake into a DF-JK plan for a functional.
 *
 * Shared by main.cpp (molecular DF-JK) and sac_guess.cpp (atomic reference
 * DF-JK) so the two never drift apart on which fractions a hybrid uses.
 */
#include "cuest_wrapper/integrals.hpp"

namespace cuest {

/// Mirrors cuEST's example matrix for the range-separated hybrids.
inline void hybrid_dfjk_fractions(XCBuilder::Functional functional, XCBuilder& xc,
                                  double& ex_frac, double& lrc_frac,
                                  double& lrc_omega) {
  ex_frac = 0.0;
  lrc_frac = 0.0;
  lrc_omega = 0.0;

  if (xc.is_hf()) {
    ex_frac = 1.0;
    return;
  }
  if (!xc.is_hybrid()) return;

  if (!xc.is_lrc()) {
    // Global hybrids (B3LYP, PBE0, ...): EXCHANGE_FRACTION = HF scale.
    ex_frac = xc.exchange_scale();
    return;
  }

  // Range-separated: plan holds both full-range and LRC fractions.
  switch (functional) {
    case XCBuilder::XC_CAM_B3LYP:
      ex_frac = 0.19; lrc_frac = 0.46; lrc_omega = 0.33; break;
    case XCBuilder::XC_WB97X:
    case XCBuilder::XC_WB97X_V:
      ex_frac = 0.157706; lrc_frac = 0.842294; lrc_omega = 0.3; break;
    case XCBuilder::XC_WB97M_V:
      ex_frac = 0.15; lrc_frac = 0.85; lrc_omega = 0.3; break;
    case XCBuilder::XC_LC_WPBE:
    case XCBuilder::XC_LC_WPBEH:
      ex_frac = 0.0; lrc_frac = 1.0; lrc_omega = 0.4; break;
    case XCBuilder::XC_HSE06:
      ex_frac = 0.25; lrc_frac = -0.25; lrc_omega = 0.11; break;
    default:
      // Fallback: use XC query scale as full-range HF only.
      ex_frac = xc.exchange_scale();
      break;
  }
}

}  // namespace cuest
