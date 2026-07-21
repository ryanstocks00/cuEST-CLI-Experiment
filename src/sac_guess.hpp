#pragma once
/**
 * @file sac_guess.hpp
 * @brief Superposition-of-Atomic-Coefficients (SAC) initial guess.
 *
 * For each distinct element in the molecule we run an isolated-atom UKS SCF
 * (starting from a zero-density / core-Hamiltonian guess, so there is no
 * recursion into this same guess) in the *same* basis, auxiliary basis,
 * functional, and grid as the parent calculation, and cache the converged
 * atomic reference to disk so repeated runs (e.g. every finite-difference
 * displacement in grad_numerical.hpp) don't re-solve the same atom.
 *
 * Each atomic reference is captured as *weighted occupied orbitals*, not a
 * density matrix: every column is pre-scaled by sqrt(its occupation weight),
 * so D = C_w C_w^T reproduces the exact spherically-averaged density, but
 * the columns themselves remain usable directly as Cocc for a real
 * Coulomb+exchange Fock build (J and K are both linear in the occupation-
 * weighted density, i.e. quadratic in these scaled columns — scaling a
 * column by sqrt(w) scales its contribution to J/K by exactly w).
 *
 * The molecular guess is the placement of these weighted columns at each
 * atom's real AO offset (BasisBuilder::ao_offsets), rescaled to the target
 * electron count. Because the result is real orbitals (not just a density),
 * it needs no special-cased "guess Fock": it's fed straight into the same
 * build_fock_rks()/build_fock_uks() every later iteration uses, so the very
 * first SCF iteration already gets a genuine Hcore+J-K(+Vxc) Fock instead of
 * an exchange-blind Hcore+J one.
 */
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "cuest_wrapper/integrals.hpp"

namespace cuest {

class CuESTContext;

/// Everything needed to reproduce a per-atom reference SCF identically to
/// the parent molecular calculation (same basis, functional, grid, kernels)
/// so the guess is consistent with what it's seeding.
struct SACGuessConfig {
  std::string basis_path;
  std::string aux_basis_path;
  XCBuilder::Functional functional{XCBuilder::XC_PBE};
  int radial_pts{75};
  int angular_pts{302};
  bool is_pure{true};
  bool use_jit{true};
};

/// Converged neutral (or ECP-valence-neutral) free-atom reference, in that
/// element's own AO basis — the same shell set/order the element
/// contributes inside any molecule built from the same basis file.
/// Cocc_alpha/Cocc_beta are nao x n_cols_alpha/n_cols_beta (column-major),
/// with every column already scaled by sqrt(its occupation weight): the
/// alpha/beta density is exactly Cocc @ Cocc^T, and the columns are also
/// directly usable as Cocc for a density-fitted exchange (K) build.
struct AtomicGuessOrbitals {
  std::vector<double> Cocc_alpha, Cocc_beta;
  uint64_t n_cols_alpha{0}, n_cols_beta{0};
  uint64_t nao{0};
};

/// Ground-state neutral-atom multiplicity for `nelec` electrons, from
/// generic Aufbau (Madelung order) filling + Hund's rule on the last
/// partially-filled subshell. This is exact for main-group elements; a
/// handful of transition metals/lanthanides have anomalous experimental
/// ground states (Cr, Cu, Nb, ...) that this simple rule doesn't reproduce.
/// That's fine here — this only seeds an SCF guess, not the final answer.
[[nodiscard]] int aufbau_hund_multiplicity(int nelec);

/// Degenerate MO group [lo, hi) straddling the HOMO/LUMO boundary at `nocc`,
/// found by expanding outward from that boundary while consecutive orbital
/// energies differ by less than `tol`. An isolated atom's true partially
/// filled shell is exactly this group (its Fock operator is rotationally
/// symmetric, so degenerate orbitals really are numerically degenerate);
/// a fully filled or fully empty frontier collapses to a 1-wide group.
[[nodiscard]] std::pair<int, int> degenerate_frontier_group(
    const std::vector<double>& mo_energies, int nocc, double tol = 1e-5);

/// The occupied-orbital columns [0, hi) of `C`, with every column in
/// [lo, hi) scaled by sqrt((nocc-lo)/(hi-lo)) instead of the hard integer
/// cutoff at `nocc` implied by columns [0, nocc) alone — the standard
/// "spherically averaged" atomic occupation expressed as weighted orbitals
/// rather than a density matrix, so a partially filled p/d/f shell doesn't
/// bias the result toward whichever specific real orbital (px vs py vs pz,
/// say) an ordinary Aufbau SCF happens to fill, while the returned columns
/// remain directly usable as Cocc for a J/K build (D = C_w C_w^T reproduces
/// the exact fractionally-averaged density; see AtomicGuessOrbitals).
/// Basis-representation agnostic: operates on the Fock eigenbasis, not on
/// AO shell indices, so it works the same whether the AO basis is spherical
/// or Cartesian. [lo, hi) should come from degenerate_frontier_group()
/// applied ONCE to an exactly-symmetric eigenspectrum (e.g. from a bare
/// Hcore diagonalization) and then held fixed for the rest of the SCF —
/// re-deriving it fresh from each iteration's (possibly already slightly
/// split) eigenvalues is unstable, since any accidental hard cutoff feeds
/// an anisotropic density back into J/K/Vxc, which then splits the
/// degeneracy for real.
[[nodiscard]] std::vector<double> weighted_occupied_coefficients(
    const std::vector<double>& C, int lo, int hi, uint64_t nocc, uint64_t nao);

/// D = C_w C_w^T for C_w = weighted_occupied_coefficients(C, lo, hi, nocc, nao).
[[nodiscard]] std::vector<double> spherically_averaged_density(
    const std::vector<double>& C, int lo, int hi, uint64_t nao, uint64_t nocc);

/// Compute (or load from an on-disk cache) the converged atomic reference
/// orbitals for element Z using the given basis/functional/grid. `ctx` is
/// the caller's cuEST context, reused for the atomic sub-calculation
/// (cheap and safe: growable scratch, no shared mutable state across
/// sequential calls).
[[nodiscard]] AtomicGuessOrbitals get_atomic_guess_orbitals(
    CuESTContext& ctx, int Z, const SACGuessConfig& cfg);

}  // namespace cuest
