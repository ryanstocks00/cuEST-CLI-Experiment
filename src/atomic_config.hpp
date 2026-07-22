#pragma once
/**
 * @file atomic_config.hpp
 * @brief Reference atomic configurations and the fractional occupations they
 *        imply for a spherically symmetric free atom.
 *
 * Once the atomic Fock has been projected onto the SO(3) commutant (see
 * atomic_symmetry.hpp) its eigenvalues come out in exactly (2l+1)-fold
 * degenerate manifolds, to machine precision rather than to a tolerance. That
 * has two consequences this file relies on:
 *
 *   1. A manifold's *size* identifies its angular momentum, l = (size-1)/2,
 *      with no need for a Cartesian→spherical transform or any AO-index
 *      bookkeeping. This works identically for pure and Cartesian bases; a
 *      Cartesian d shell's l=0 contaminant simply shows up as another s
 *      manifold, which is exactly what it physically is.
 *
 *   2. Occupying whole manifolds with equal fractional occupancy is
 *      well-defined: the individual eigenvectors within a degenerate manifold
 *      are arbitrary, but any equal-weight combination of the whole manifold
 *      gives the same density. That is what makes the spherically symmetric
 *      constrained minimum representable at all.
 */
#include <cstdint>
#include <vector>

namespace cuest {

#include "atomic_config_table.inc"

/// One group of exactly-degenerate MOs from a symmetrized atomic Fock.
struct OrbitalManifold {
  int lo{0};       ///< first MO index
  int size{0};     ///< number of MOs (== 2l+1)
  int l{0};        ///< angular momentum, (size-1)/2
  double energy{0.0};
  double occ{0.0};  ///< electrons per orbital, in [0, 2]
};

/// Core shells per l channel replaced by an ECP with `nelec_core` electrons,
/// mirroring PySCF's gto.ecp.core_configuration. Returns {n_s, n_p, n_d, n_f}
/// counts of *shells* (not electrons). Returns false if `nelec_core` is not a
/// recognised core size.
[[nodiscard]] bool ecp_core_shells(int nelec_core, int out[4]);

/// Occupied manifolds, with fractional occupancies, for element `Z` with
/// `nelec_core` electrons replaced by an ECP (0 if none).
///
/// Walks `mo_energies` upward from the bottom, forming one exactly-degenerate
/// group at a time and handing each to its l channel, and stops as soon as the
/// reference configuration is satisfied. Stopping early matters: the high
/// virtual orbitals of a large basis can be accidentally near-degenerate, and
/// grouping them would risk merging two manifolds for no benefit — nothing
/// above the last occupied shell affects the guess.
///
/// Within each l channel the manifolds fill in energy order: whole shells take
/// occ = 2, and the remaining electrons spread evenly over the next manifold,
/// giving it a fractional occupancy. This is PySCF AtomSphAverageRHF.get_occ,
/// and it is what makes carbon's 2p come out as 2/3 of an electron in each of
/// px, py, pz instead of an arbitrary choice of which two real p orbitals to
/// fill.
///
/// `tol` can be tight because the input comes from a projected (exactly
/// spherical) Fock: this is grouping numerically-identical eigenvalues, not
/// guessing at near-degeneracy.
///
/// Throws if a manifold that the configuration needs has even size — a
/// spherically symmetric Fock only produces (2l+1)-fold degeneracies, so that
/// means the Fock was never projected (or two l accidentally coincided), and
/// silently mis-assigning l would be worse than failing.
[[nodiscard]] std::vector<OrbitalManifold> occupied_manifolds(
    const std::vector<double>& mo_energies, int Z, int nelec_core,
    double tol = 1e-8);

}  // namespace cuest
