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

/// Queries cuEST's own XC plan for the HF-exchange fractions to bake into
/// the DF-JK plan, rather than hardcoding literature values: cuEST already
/// exposes CUEST_XCINTPLAN_EXCHANGE_SCALE / _LRC_EXCHANGE_SCALE / _LRC_OMEGA
/// as queryable attributes tied to the functional chosen at XC-plan creation
/// (the same query-don't-hardcode approach used for VV10's constants). This
/// previously hardcoded a literature table that silently mismapped
/// XC_WB97X_V onto bare WB97X's (different) range-separation parameters.
inline void hybrid_dfjk_fractions(XCBuilder::Functional /*functional*/, XCBuilder& xc,
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

  ex_frac = xc.exchange_scale();
  if (xc.is_lrc()) {
    lrc_frac = xc.lrc_exchange_scale();
    lrc_omega = xc.lrc_omega();
  }
}

}  // namespace cuest
