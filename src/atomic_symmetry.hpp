#pragma once
/**
 * @file atomic_symmetry.hpp
 * @brief SO(3) commutant projector for free-atom matrices.
 *
 * A free atom's exact Fock operator is a scalar under rotation, so by
 * Wigner-Eckart it commutes with the AO rotation representation: block
 * diagonal in angular momentum l, proportional to the identity in m, and with
 * no coupling between different l (not even between different l of the same
 * parity — <s|F|d> vanishes). The matrices with that property form the
 * *commutant* of the rotation representation, and "the density is spherically
 * symmetric" is exactly the statement that D lies in it (Schur's lemma).
 *
 * The point of this file is that we impose that constraint by *projection*
 * rather than by changing basis into the l-decomposition:
 *
 *     P(X) = ∫_SO(3) dR  U(R) X U(R)^T
 *
 * which needs only the rotation *action* U(R) on the AOs, never the
 * decomposition itself. That is what makes it representation-agnostic: no
 * Cartesian→spherical transform, no hardcoded solid-harmonic tables, and no
 * need to align the m-ordering of different shells sharing an l. A Cartesian
 * shell of angular momentum L spans L ⊕ L-2 ⊕ ... and so is *not* a single
 * irrep, but it needs no special handling here.
 *
 * Projecting the Fock matrix each SCF iteration makes the 2l+1 degeneracies
 * exact to machine precision instead of something a tolerance has to detect,
 * which in turn lets the occupied manifolds be identified by their size.
 *
 * Two implementations, one interface:
 *   - Pure (spherical) shells are already single irreps, so the projection
 *     collapses to averaging the m-diagonal of each equal-l block and zeroing
 *     everything else. That needs no rotation matrices and is exactly what
 *     PySCF's AtomSphAverageRHF.eig does, so it reproduces PySCF's SAD.
 *   - Cartesian shells use the integral above, evaluated by an Euler-angle
 *     quadrature that is exact for the polynomial degree involved.
 *
 * Cost stays negligible because P acts independently on each shell-pair block
 * and depends only on (l_A, l_B): the Cartesian block projectors are built
 * once and thereafter applied as small fixed matrices.
 */
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace cuest {

/// Number of AOs in a shell of angular momentum `l`.
[[nodiscard]] constexpr int shell_size(int l, bool is_pure) {
  return is_pure ? (2 * l + 1) : ((l + 1) * (l + 2) / 2);
}

/// Shell layout of one free atom's AO basis, in AO order.
struct AtomShells {
  std::vector<int> l;       ///< angular momentum of each shell
  std::vector<int> offset;  ///< AO offset of each shell, size nshell+1
  bool is_pure{true};
  int nao{0};

  [[nodiscard]] int nshell() const { return static_cast<int>(l.size()); }
  [[nodiscard]] int size(int s) const { return offset[s + 1] - offset[s]; }
  [[nodiscard]] int max_l() const;
};

/// Build an AtomShells from per-shell angular momenta.
[[nodiscard]] AtomShells make_atom_shells(const std::vector<int>& shell_l,
                                          bool is_pure);

// ---------------------------------------------------------------------------
// Cartesian rotation representation
// ---------------------------------------------------------------------------

/// Cartesian monomial exponents (a, b, c) with a+b+c == l, in cuEST/libcint
/// shell order: `a` descending, then `b` descending within each `a`.
/// For l=2 this is xx, xy, xz, yy, yz, zz.
[[nodiscard]] std::vector<std::array<int, 3>> cartesian_monomials(int l);

/// Rotation representation U(R) for one *normalized* Cartesian shell of
/// angular momentum `l`, row-major (n x n) with n = shell_size(l, false).
///
/// Defined by the active rotation of the basis function, χ'(r) = χ(R^-1 r),
/// re-expanded on the same shell: the l-th symmetric power of R, obtained by
/// multinomial expansion of (R^T r)_x^a (R^T r)_y^b (R^T r)_z^c.
///
/// No per-component rescaling appears because cuEST (like libcint) normalizes
/// a Cartesian shell with a *single* radial constant, chosen for the (l,0,0)
/// component — so <d_xx|d_xx> = 3 <d_xy|d_xy> rather than both being 1, and
/// the normalized AOs differ from the raw monomials by one shell-wide factor
/// that cancels out of the similarity transform. verify_rotation_rep() checks
/// that against a real cuEST overlap matrix.
///
/// `R` is row-major 3x3.
[[nodiscard]] std::vector<double> cartesian_shell_rotation(int l, const double* R);

/// Uniformly-distributed random rotation matrix (row-major 3x3) from `seed`.
[[nodiscard]] std::vector<double> random_rotation(uint64_t seed);

/// Self-test: max |U(R)^T S U(R) - S| over several random rotations, on a real
/// single-atom overlap matrix.
///
/// S is a genuinely spherically symmetric operator, so this one condition
/// validates the AO ordering *and* the per-component normalization *and* the
/// rotation code at once — a wrong monomial order or an unexpected
/// normalization convention fails loudly rather than silently producing a
/// subtly wrong guess. Cartesian only; pure shells never build U(R).
[[nodiscard]] double verify_rotation_rep(const std::vector<double>& S,
                                         const AtomShells& shells);

// ---------------------------------------------------------------------------
// The projector
// ---------------------------------------------------------------------------

/// Which transformation law a matrix obeys under rotation.
///
/// This distinction is *not* cosmetic in a Cartesian basis. The AOs within a
/// Cartesian shell are not orthonormal (<d_xx|d_yy> != 0), so U(R) is not an
/// orthogonal matrix and the two laws give genuinely different projectors:
///
///   Operator (covariant, X_uv = <chi_u|O|chi_v>):  invariant <=> U^T X U = X
///   Density  (contravariant, rho = sum D_uv chi_u chi_v): invariant <=> U D U^T = D
///
/// For *pure* shells U is orthogonal and the two coincide, which is why PySCF
/// needs only one routine. Use Operator for Fock/Hcore/overlap and Density for
/// density matrices; mixing them up silently destroys the symmetry you were
/// trying to impose.
enum class RotationLaw { Operator, Density };

/// Projector onto the SO(3) commutant for one atom's AO basis.
///
/// Construction is the expensive part (Cartesian only, and only once per
/// distinct (l_A, l_B, law) triple); apply() is cheap enough to call every SCF
/// iteration.
class SphericalProjector {
 public:
  explicit SphericalProjector(AtomShells shells);

  /// X <- P(X). `X` is nao x nao row-major and symmetric.
  void apply(std::vector<double>& X, RotationLaw law) const;

  /// max |P(P(X)) - P(X)| for a deterministic pseudo-random symmetric X.
  /// A projector must be idempotent, so this catches an under-resolved
  /// quadrature; it is independent of verify_rotation_rep(), which instead
  /// catches a wrong U(R).
  [[nodiscard]] double check_idempotent(RotationLaw law) const;

  [[nodiscard]] const AtomShells& shells() const { return shells_; }

 private:
  /// Dense (n_A*n_B) x (n_A*n_B) block projector for a Cartesian l-pair,
  /// acting on vec(X_AB) in row-major order.
  [[nodiscard]] const std::vector<double>& block_projector(int la, int lb,
                                                           RotationLaw law) const;

  AtomShells shells_;
  mutable std::map<std::tuple<int, int, int>, std::vector<double>> cart_blocks_;
};

}  // namespace cuest
