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

/// Bohr radius in angstrom: a0 = 1 bohr ≈ 0.529177210903 Å (CODATA 2018)
constexpr double bohr_per_angstrom = 0.529177210903;

/// Angstrom per bohr
constexpr double angstrom_per_bohr = 1.0 / bohr_per_angstrom;  // ≈ 1.8897259886

// ---------------------------------------------------------------------------
// Energy
// ---------------------------------------------------------------------------

/// Hartree to eV conversion (CODATA 2018)
constexpr double hartree_per_ev = 27.211386245988;

/// eV to Hartree
constexpr double ev_per_hartree = 1.0 / hartree_per_ev;

}  // namespace constants
}  // namespace cuest
