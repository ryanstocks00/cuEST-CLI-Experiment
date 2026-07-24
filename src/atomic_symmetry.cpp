/**
 * @file atomic_symmetry.cpp — SO(3) commutant projector (see header).
 */
#include "atomic_symmetry.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace cuest {

int AtomShells::max_l() const {
  return l.empty() ? 0 : *std::max_element(l.begin(), l.end());
}

AtomShells make_atom_shells(const std::vector<int>& shell_l, bool is_pure) {
  AtomShells s;
  s.l = shell_l;
  s.is_pure = is_pure;
  s.offset.assign(shell_l.size() + 1, 0);
  for (size_t i = 0; i < shell_l.size(); i++)
    s.offset[i + 1] = s.offset[i] + shell_size(shell_l[i], is_pure);
  s.nao = s.offset.back();
  return s;
}

// ---------------------------------------------------------------------------
// Cartesian rotation representation
// ---------------------------------------------------------------------------

std::vector<std::array<int, 3>> cartesian_monomials(int l) {
  std::vector<std::array<int, 3>> out;
  out.reserve(static_cast<size_t>(shell_size(l, false)));
  for (int a = l; a >= 0; a--)
    for (int b = l - a; b >= 0; b--) out.push_back({a, b, l - a - b});
  return out;
}

namespace {

double factorial(int n) {
  double f = 1.0;
  for (int i = 2; i <= n; i++) f *= i;
  return f;
}

/// Index of monomial (a,b,c) within its shell, matching cartesian_monomials().
/// With `a` descending then `b` descending, the monomials with first exponent
/// > a occupy sum_{a'>a} (l-a'+1) slots, and within a fixed `a` the offset is
/// (l-a) - b.
int monomial_index(int l, int a, int b) {
  int idx = 0;
  for (int ap = l; ap > a; ap--) idx += l - ap + 1;
  return idx + (l - a - b);
}

}  // namespace

std::vector<double> cartesian_shell_rotation(int l, const double* R) {
  const int n = shell_size(l, false);
  std::vector<double> U(static_cast<size_t>(n) * n, 0.0);
  const auto mons = cartesian_monomials(l);

  // chi'_alpha(r) = chi_alpha(R^-1 r), and (R^T r)_i = sum_j R_ji r_j.
  // Expanding u_x^a u_y^b u_z^c by multinomial gives the column `alpha` of U.
  for (int col = 0; col < n; col++) {
    const int a = mons[col][0], b = mons[col][1], c = mons[col][2];

    // Multinomial over the three factors independently.
    for (int p1 = 0; p1 <= a; p1++)
      for (int p2 = 0; p2 <= a - p1; p2++) {
        const int p3 = a - p1 - p2;
        const double cp = factorial(a) / (factorial(p1) * factorial(p2) * factorial(p3)) *
                          std::pow(R[0 * 3 + 0], p1) * std::pow(R[1 * 3 + 0], p2) *
                          std::pow(R[2 * 3 + 0], p3);
        if (cp == 0.0) continue;

        for (int q1 = 0; q1 <= b; q1++)
          for (int q2 = 0; q2 <= b - q1; q2++) {
            const int q3 = b - q1 - q2;
            const double cq = factorial(b) / (factorial(q1) * factorial(q2) * factorial(q3)) *
                              std::pow(R[0 * 3 + 1], q1) * std::pow(R[1 * 3 + 1], q2) *
                              std::pow(R[2 * 3 + 1], q3);
            if (cq == 0.0) continue;

            for (int s1 = 0; s1 <= c; s1++)
              for (int s2 = 0; s2 <= c - s1; s2++) {
                const int s3 = c - s1 - s2;
                const double cs = factorial(c) /
                                  (factorial(s1) * factorial(s2) * factorial(s3)) *
                                  std::pow(R[0 * 3 + 2], s1) * std::pow(R[1 * 3 + 2], s2) *
                                  std::pow(R[2 * 3 + 2], s3);
                if (cs == 0.0) continue;

                const int ea = p1 + q1 + s1;
                const int eb = p2 + q2 + s2;
                const int row = monomial_index(l, ea, eb);
                U[static_cast<size_t>(row) * n + col] += cp * cq * cs;
              }
          }
      }
  }
  return U;
}

namespace {

/// Deterministic splitmix64 -> uniform double in [0, 1).
double rand01(uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z = z ^ (z >> 31);
  return static_cast<double>(z >> 11) / 9007199254740992.0;
}

/// Row-major 3x3 rotation from ZYZ Euler angles.
std::vector<double> euler_rotation(double alpha, double beta, double gamma) {
  const double ca = std::cos(alpha), sa = std::sin(alpha);
  const double cb = std::cos(beta), sb = std::sin(beta);
  const double cg = std::cos(gamma), sg = std::sin(gamma);
  // Rz(alpha) Ry(beta) Rz(gamma)
  return {ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb,
          sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb,
          -sb * cg,               sb * sg,                 cb};
}

}  // namespace

std::vector<double> random_rotation(uint64_t seed) {
  uint64_t s = seed * 0x2545F4914F6CDD1DULL + 1;
  const double alpha = 2.0 * M_PI * rand01(s);
  const double beta = std::acos(2.0 * rand01(s) - 1.0);
  const double gamma = 2.0 * M_PI * rand01(s);
  return euler_rotation(alpha, beta, gamma);
}

double verify_rotation_rep(const std::vector<double>& S, const AtomShells& shells) {
  if (shells.is_pure) return 0.0;  // pure never builds U(R)
  const int nao = shells.nao;
  if (static_cast<int>(S.size()) != nao * nao)
    throw std::runtime_error("verify_rotation_rep: overlap size mismatch");

  double worst = 0.0;
  for (uint64_t trial = 0; trial < 4; trial++) {
    const auto R = random_rotation(trial + 1);

    // Block-diagonal U over shells; build S' = U^T S U blockwise.
    std::vector<std::vector<double>> U(shells.nshell());
    for (int s = 0; s < shells.nshell(); s++)
      U[s] = cartesian_shell_rotation(shells.l[s], R.data());

    for (int sa = 0; sa < shells.nshell(); sa++) {
      const int na = shells.size(sa), oa = shells.offset[sa];
      for (int sb = 0; sb < shells.nshell(); sb++) {
        const int nb = shells.size(sb), ob = shells.offset[sb];
        // (U_a^T S_ab U_b)_{ij} = sum_{p,q} U_a[p][i] S[oa+p][ob+q] U_b[q][j]
        for (int i = 0; i < na; i++)
          for (int j = 0; j < nb; j++) {
            double acc = 0.0;
            for (int p = 0; p < na; p++) {
              const double uap = U[sa][static_cast<size_t>(p) * na + i];
              if (uap == 0.0) continue;
              for (int q = 0; q < nb; q++)
                acc += uap * S[static_cast<size_t>(oa + p) * nao + (ob + q)] *
                       U[sb][static_cast<size_t>(q) * nb + j];
            }
            worst = std::max(worst,
                             std::abs(acc - S[static_cast<size_t>(oa + i) * nao + (ob + j)]));
          }
      }
    }
  }
  return worst;
}

// ---------------------------------------------------------------------------
// Projector
// ---------------------------------------------------------------------------

SphericalProjector::SphericalProjector(AtomShells shells)
    : shells_(std::move(shells)) {}

namespace {

/// Gauss-Legendre nodes/weights on [-1, 1] via Newton iteration on P_n.
void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
  x.assign(n, 0.0);
  w.assign(n, 0.0);
  for (int i = 0; i < n; i++) {
    double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));
    double pp = 0.0;
    for (int it = 0; it < 100; it++) {
      double p0 = 1.0, p1 = 0.0;
      for (int j = 0; j < n; j++) {
        const double p2 = p1;
        p1 = p0;
        p0 = ((2.0 * j + 1.0) * z * p1 - j * p2) / (j + 1);
      }
      pp = n * (z * p0 - p1) / (z * z - 1.0);
      const double dz = -p0 / pp;
      z += dz;
      if (std::abs(dz) < 1e-15) break;
    }
    x[i] = z;
    w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
  }
}

}  // namespace

const std::vector<double>& SphericalProjector::block_projector(int la, int lb,
                                                               RotationLaw law) const {
  const auto key = std::make_tuple(la, lb, static_cast<int>(law));
  auto it = cart_blocks_.find(key);
  if (it != cart_blocks_.end()) return it->second;

  const int na = shell_size(la, false), nb = shell_size(lb, false);
  const size_t dim = static_cast<size_t>(na) * nb;

  // U_A(R) entries are degree-la polynomials in R, so U_A ⊗ U_B carries Wigner
  // D of order up to L = la+lb. Uniform quadrature in the two azimuthal Euler
  // angles is exact with 2L+1 points; Gauss-Legendre in cos(beta) with L+1
  // points covers the polar dependence. check_idempotent() verifies that the
  // result really is a projector, which is what would catch under-resolution.
  const int L = la + lb;
  const int n_az = 2 * L + 1;
  const int n_beta = L + 1;

  // The ZYZ parametrisation R = Rz(alpha) Ry(beta) Rz(gamma) factorises the
  // integral. U is a group homomorphism and (A1 A2) ⊗ (B1 B2) = (A1⊗B1)(A2⊗B2),
  // so the integrand is a product of three factors each depending on one angle:
  //
  //     P = [∫dalpha Kz] · [∫dbeta Ky] · [∫dgamma Kz]
  //
  // Building the three factors separately and multiplying costs two dense
  // matmuls instead of n_az^2 * n_beta rank-1 accumulations — for a QZ-basis
  // transition metal that is the difference between ~30 s and well under a
  // second.
  auto kron_at = [&](const std::vector<double>& R, double weight,
                     std::vector<double>& acc) {
    const auto Ua = cartesian_shell_rotation(la, R.data());
    const auto Ub = (la == lb) ? Ua : cartesian_shell_rotation(lb, R.data());

    // Density:  (U X U^T)_ij = sum_pq Ua[i][p] X_pq Ub[j][q]
    // Operator: (U^T X U)_ij = sum_pq Ua[p][i] X_pq Ub[q][j]
    // i.e. the operator law is the same contraction with both U transposed.
    // See RotationLaw.
    const bool op = (law == RotationLaw::Operator);
    for (int i = 0; i < na; i++)
      for (int j = 0; j < nb; j++) {
        const size_t row = static_cast<size_t>(i) * nb + j;
        for (int p = 0; p < na; p++) {
          const double uap = op ? Ua[static_cast<size_t>(p) * na + i]
                                : Ua[static_cast<size_t>(i) * na + p];
          if (uap == 0.0) continue;
          const double wu = weight * uap;
          for (int q = 0; q < nb; q++)
            acc[row * dim + static_cast<size_t>(p) * nb + q] +=
                wu * (op ? Ub[static_cast<size_t>(q) * nb + j]
                         : Ub[static_cast<size_t>(j) * nb + q]);
        }
      }
  };

  std::vector<double> Kz(dim * dim, 0.0);
  for (int i = 0; i < n_az; i++)
    kron_at(euler_rotation(2.0 * M_PI * i / n_az, 0.0, 0.0),
            1.0 / static_cast<double>(n_az), Kz);

  std::vector<double> tb, wb;
  gauss_legendre(n_beta, tb, wb);
  std::vector<double> Ky(dim * dim, 0.0);
  for (int i = 0; i < n_beta; i++)
    kron_at(euler_rotation(0.0, std::acos(tb[i]), 0.0), wb[i] * 0.5, Ky);

  auto matmul = [dim](const std::vector<double>& A, const std::vector<double>& B) {
    std::vector<double> C(dim * dim, 0.0);
    for (size_t i = 0; i < dim; i++)
      for (size_t k = 0; k < dim; k++) {
        const double a = A[i * dim + k];
        if (a == 0.0) continue;
        for (size_t j = 0; j < dim; j++) C[i * dim + j] += a * B[k * dim + j];
      }
    return C;
  };

  std::vector<double> P = matmul(matmul(Kz, Ky), Kz);
  return cart_blocks_.emplace(key, std::move(P)).first->second;
}

void SphericalProjector::apply(std::vector<double>& X, RotationLaw law) const {
  const int nao = shells_.nao;
  if (static_cast<int>(X.size()) != nao * nao)
    throw std::runtime_error("SphericalProjector::apply: size mismatch");

  std::vector<double> out(X.size(), 0.0);

  for (int sa = 0; sa < shells_.nshell(); sa++) {
    const int la = shells_.l[sa], na = shells_.size(sa), oa = shells_.offset[sa];
    for (int sb = 0; sb < shells_.nshell(); sb++) {
      const int lb = shells_.l[sb], nb = shells_.size(sb), ob = shells_.offset[sb];

      if (shells_.is_pure) {
        // A pure shell is a single irrep and its U(R) is orthogonal, so both
        // rotation laws collapse to the same thing: keep only equal-l blocks
        // and, within one, only the m-diagonal average. This is PySCF
        // AtomSphAverageRHF.eig's 'piqi->pq' / degen.
        if (la != lb) continue;  // leaves the block zero
        double avg = 0.0;
        for (int m = 0; m < na; m++)
          avg += X[static_cast<size_t>(oa + m) * nao + (ob + m)];
        avg /= static_cast<double>(na);
        for (int m = 0; m < na; m++)
          out[static_cast<size_t>(oa + m) * nao + (ob + m)] = avg;
      } else {
        const auto& P = block_projector(la, lb, law);
        const size_t dim = static_cast<size_t>(na) * nb;
        for (int i = 0; i < na; i++)
          for (int j = 0; j < nb; j++) {
            const size_t row = static_cast<size_t>(i) * nb + j;
            double acc = 0.0;
            for (int p = 0; p < na; p++)
              for (int q = 0; q < nb; q++)
                acc += P[row * dim + static_cast<size_t>(p) * nb + q] *
                       X[static_cast<size_t>(oa + p) * nao + (ob + q)];
            out[static_cast<size_t>(oa + i) * nao + (ob + j)] = acc;
          }
      }
    }
  }

  X.swap(out);
}

double SphericalProjector::check_idempotent(RotationLaw law) const {
  const int nao = shells_.nao;
  std::vector<double> X(static_cast<size_t>(nao) * nao);
  uint64_t s = 12345;
  for (int i = 0; i < nao; i++)
    for (int j = 0; j <= i; j++) {
      const double v = 2.0 * rand01(s) - 1.0;
      X[static_cast<size_t>(i) * nao + j] = v;
      X[static_cast<size_t>(j) * nao + i] = v;
    }

  apply(X, law);
  std::vector<double> Y = X;
  apply(Y, law);

  double worst = 0.0;
  for (size_t i = 0; i < X.size(); i++)
    worst = std::max(worst, std::abs(Y[i] - X[i]));
  return worst;
}

}  // namespace cuest
