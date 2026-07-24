#pragma once
/**
 * @file sad_guess.hpp
 * @brief Superposition-of-Atomic-Densities (SAD) initial guess.
 *
 * For each distinct element we solve an isolated free atom under an explicit
 * spherical-symmetry constraint, then superpose the resulting atomic densities
 * at each atom's AO offset. The converged atomic reference is cached to disk so
 * repeated runs (every finite-difference displacement in grad_numerical.hpp,
 * say) don't re-solve the same atom.
 *
 * ## Imposing the constraint
 *
 * The atomic Fock is projected onto the SO(3) commutant every iteration (see
 * atomic_symmetry.hpp). That is the direct statement of "minimise the energy
 * over spherically symmetric densities": by Schur's lemma the commutant *is*
 * the l-block structure, so projecting is exactly the constraint, and it needs
 * only the rotation action on AOs rather than any explicit l-decomposition.
 *
 * Doing it that way is what lets Cartesian and pure bases share one code path.
 * A Cartesian shell of angular momentum L spans L ⊕ L-2 ⊕ ... and so is not a
 * single irrep, which is why the usual "reshape the l-block and average over
 * m" trick (PySCF's AtomSphAverageRHF.eig) cannot be applied to it at all.
 * PySCF sidesteps this by solving the atom in a spherical basis and mapping up
 * with cart2sph, which leaves the Cartesian contaminant functions with exactly
 * zero density. Here they are simply part of the variational space, so a
 * Cartesian guess is slightly *better* than PySCF's — and correspondingly not
 * bit-comparable to it. The spherical path does reproduce PySCF's SAD.
 *
 * ## Fractional occupations
 *
 * After projection the eigenvalues are exactly (2l+1)-fold degenerate, so a
 * manifold's size identifies its l and whole manifolds can be occupied at equal
 * fractional occupancy (see atomic_config.hpp). Carbon's 2p comes out as 2/3 of
 * an electron in each of px, py, pz rather than an arbitrary choice of which
 * two real p orbitals to fill.
 *
 * The reference is stored as *weighted* orbitals rather than as a density
 * matrix: each column is pre-scaled by sqrt(its occupation), so that
 * D_alpha = C_w C_w^T reproduces the fractionally-occupied density exactly
 * while the columns stay directly usable as Cocc for a density-fitted exchange
 * build. That matters because cuEST's DF-K API takes occupied MO coefficients,
 * not an arbitrary density matrix — the sqrt-weighting is what lets a genuine
 * SAD density feed a real Hcore+J-K(+Vxc) Fock on the very first iteration.
 */
#include <cstdint>
#include <string>
#include <vector>

#include "cuest_wrapper/integrals.hpp"

namespace cuest {

class CuESTContext;

/// Everything needed to reproduce a per-atom reference SCF.
struct SADGuessConfig {
  std::string basis_path;
  std::string aux_basis_path;
  bool is_pure{true};
  bool use_jit{true};

  /// Solve the atoms with the parent calculation's functional instead of HF.
  ///
  /// Default is HF, matching PySCF: its SAD uses spherically averaged atomic
  /// *HF* regardless of the molecular functional. HF atoms need no XC grid and
  /// are functional-independent, so one cached atom serves every functional —
  /// and it is what makes the guess comparable to PySCF's. Set this to run the
  /// atoms with `functional`/grid below instead, which is more self-consistent
  /// with what the guess seeds but costs a cache entry per functional.
  bool use_parent_functional{false};
  XCBuilder::Functional functional{XCBuilder::XC_PBE};
  int radial_pts{75};
  int angular_pts{302};

  // DF fitting knobs — kept in sync with the molecular DFJKBuilder's so the
  // atomic reference is solved with the same DF numerics as the calculation it
  // seeds. They deliberately do *not* enter the disk-cache key: they change the
  // atom by far less than the guess needs to be accurate to, and keying on them
  // would fragment the cache across otherwise-identical runs.
  double fitting_cutoff{1.0e-12};
  bool fitting_relative_conditioning{true};
  cuestDFIntPlanParametersFittingAlgorithm_t fitting_algorithm{
      CUEST_DFINTPLAN_PARAMETERS_FITTING_ALGORITHM_QR};
};

/// Converged spherically symmetric free-atom reference, in that element's own
/// AO basis — the same shell set/order the element contributes inside any
/// molecule built from the same basis file.
///
/// The atom is spin-restricted (its constrained ground state is spin-averaged),
/// so there is a single set of columns rather than separate alpha/beta ones.
/// `Cocc` is nao x n_cols column-major with every column scaled by sqrt of its
/// occupancy, normalised so that Cocc @ Cocc^T is the *alpha* density
/// (i.e. half the atom's total density), matching the D_alpha convention used
/// throughout SCFSolver.
struct AtomicGuessOrbitals {
  std::vector<double> Cocc;
  uint64_t n_cols{0};
  uint64_t nao{0};
  double energy{0.0};   ///< converged atomic energy (diagnostics only)
  int nelec{0};         ///< electrons represented (Z minus ECP core)
};

/// Compute (or load from an on-disk cache) the converged atomic reference for
/// element Z. `ctx` is the caller's cuEST context, reused for the atomic
/// sub-calculation (cheap and safe: growable scratch, no shared mutable state
/// across sequential calls).
[[nodiscard]] AtomicGuessOrbitals get_atomic_guess_orbitals(
    CuESTContext& ctx, int Z, const SADGuessConfig& cfg);

}  // namespace cuest
