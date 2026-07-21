#pragma once
/**
 * @file constants.hpp
 * @brief Physical constants used across the cuEST wrapper.
 *
 * Single source of truth — change here and everything updates.
 * CODATA 2018 values unless noted otherwise.
 */

namespace cuest {
namespace constants {

// ---------------------------------------------------------------------------
// Length
// ---------------------------------------------------------------------------

/// Angstrom per bohr (Bohr radius in angstrom) — matches PySCF for exact comparison
constexpr double angstrom_per_bohr = 0.52917721092;

/// Bohr per angstrom
constexpr double bohr_per_angstrom = 1.0 / angstrom_per_bohr;  // ≈ 1.8897259887

// ---------------------------------------------------------------------------
// Energy
// ---------------------------------------------------------------------------

/// eV per Hartree (CODATA 2018)
constexpr double ev_per_hartree = 27.211386245988;

/// Hartree per eV
constexpr double hartree_per_ev = 1.0 / ev_per_hartree;

}  // namespace constants
}  // namespace cuest
