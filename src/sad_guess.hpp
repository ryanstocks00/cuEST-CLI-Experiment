#pragma once
/**
 * @file sad_guess.hpp
 * @brief Superposition-of-Atomic-Densities initial guess.
 *
 * For each distinct element in the molecule we run an isolated-atom UKS SCF
 * (starting from a zero-density / core-Hamiltonian guess, so there is no
 * recursion into this same guess) in the *same* basis, auxiliary basis,
 * functional, and grid as the parent calculation, and cache the converged
 * atomic density to disk so repeated runs (e.g. every finite-difference
 * displacement in grad_numerical.hpp) don't re-solve the same atom.
 *
 * The molecular guess density is then the block-diagonal placement of these
 * atomic densities at each atom's real AO offset (BasisBuilder::ao_offsets),
 * rescaled to the target electron count and run through the ordinary
 * Fock build (Hcore + J/K/Vxc of the guess density) rather than being added
 * to Hcore directly.
 */
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cuest {

class CuESTContext;

/// Everything needed to reproduce a per-atom reference SCF identically to
/// the parent molecular calculation (same basis, functional, grid, kernels)
/// so the guess is consistent with what it's seeding.
struct SADGuessConfig {
  std::string basis_path;
  std::string aux_basis_path;
  int functional_id{0};
  int radial_pts{75};
  int angular_pts{302};
  bool is_pure{true};
  bool use_jit{true};
};

/// Converged neutral (or ECP-valence-neutral) free-atom alpha/beta densities
/// in that element's own AO basis — the same shell set/order the element
/// contributes inside any molecule built from the same basis file.
struct AtomicGuessDensity {
  std::vector<double> D_alpha, D_beta;
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

/// Density built from MOs `C` with every orbital in [lo, hi) given equal
/// fractional occupation (nocc - lo)/(hi - lo) instead of a hard integer
/// cutoff at `nocc` — the standard "spherically averaged" atomic occupation,
/// so a partially filled p/d/f shell doesn't bias the density toward
/// whichever specific real orbital (px vs py vs pz, say) an ordinary Aufbau
/// SCF happens to fill. Basis-representation agnostic: operates on the Fock
/// eigenbasis, not on AO shell indices, so it works the same whether the AO
/// basis is spherical or Cartesian. [lo, hi) should come from
/// degenerate_frontier_group() applied ONCE to an exactly-symmetric
/// eigenspectrum (e.g. from a bare Hcore diagonalization) and then held
/// fixed for the rest of the SCF — re-deriving it fresh from each
/// iteration's (possibly already slightly split) eigenvalues is unstable,
/// since any accidental hard cutoff feeds an anisotropic density back into
/// J/K/Vxc, which then splits the degeneracy for real.
[[nodiscard]] std::vector<double> spherically_averaged_density(
    const std::vector<double>& C, int lo, int hi, uint64_t nao, uint64_t nocc);

/// Compute (or load from an on-disk cache) the converged atomic reference
/// density for element Z using the given basis/functional/grid. `ctx` is
/// the caller's cuEST context, reused for the atomic sub-calculation
/// (cheap and safe: growable scratch, no shared mutable state across
/// sequential calls).
[[nodiscard]] AtomicGuessDensity get_atomic_guess_density(
    CuESTContext& ctx, int Z, const SADGuessConfig& cfg);

}  // namespace cuest
