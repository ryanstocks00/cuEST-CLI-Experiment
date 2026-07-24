/**
 * @file test_atomic_symmetry.cpp
 * @brief Self-contained checks for the rotation representation and the SO(3)
 *        commutant projector. No CUDA, no cuEST, no GPU — runs in well under a
 *        second, so it is the cheap first line of defence on the trickiest
 *        maths in the SAD guess.
 *
 * Covers what can be checked without touching cuEST: that the Cartesian
 * monomial ordering is the assumed one, that U is a genuine representation
 * (U(I) = I, U(R1 R2) = U(R1) U(R2)), that it preserves the *analytic* monomial
 * Gram matrix, and that both projectors are idempotent.
 *
 * The one thing it cannot check is whether cuEST actually lays its AOs out the
 * way all of this assumes — that needs a real overlap matrix, and lives in
 * tools/probe_symmetry.cpp.
 */
#include "atomic_symmetry.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace cuest;

// Raw Cartesian-monomial shell overlap, up to a shell-wide constant:
//   <m_a|m_b> ∝ prod_i df(a_i + b_i),  df(n) = (n-1)!! for even n, 0 for odd.
// With cuEST/libcint's single-constant-per-shell normalization this is the
// actual overlap block up to that constant, which cancels in U^T G U = G.
static double df(int n) {
  if (n % 2 != 0) return 0.0;
  double r = 1.0;
  for (int k = n - 1; k > 0; k -= 2) r *= k;
  return r;
}

static std::vector<double> shell_gram(int l) {
  const auto m = cartesian_monomials(l);
  const int n = (int)m.size();
  std::vector<double> G((size_t)n * n, 0.0);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      G[(size_t)i * n + j] = df(m[i][0] + m[j][0]) * df(m[i][1] + m[j][1]) *
                             df(m[i][2] + m[j][2]);
  return G;
}

static std::vector<double> matmul(const std::vector<double>& A,
                                  const std::vector<double>& B, int n) {
  std::vector<double> C((size_t)n * n, 0.0);
  for (int i = 0; i < n; i++)
    for (int k = 0; k < n; k++) {
      double a = A[(size_t)i * n + k];
      if (a == 0.0) continue;
      for (int j = 0; j < n; j++) C[(size_t)i * n + j] += a * B[(size_t)k * n + j];
    }
  return C;
}

int main() {
  int failures = 0;

  // 1. Ordering sanity for l=2: xx, xy, xz, yy, yz, zz.
  {
    const auto m = cartesian_monomials(2);
    const int want[6][3] = {{2,0,0},{1,1,0},{1,0,1},{0,2,0},{0,1,1},{0,0,2}};
    for (int i = 0; i < 6; i++)
      if (m[i][0] != want[i][0] || m[i][1] != want[i][1] || m[i][2] != want[i][2]) {
        printf("FAIL ordering l=2 idx %d: got (%d,%d,%d)\n", i, m[i][0], m[i][1], m[i][2]);
        failures++;
      }
    printf("ordering l=2: ok\n");
  }

  // 2. U(identity) == I.
  for (int l = 0; l <= 5; l++) {
    const double I3[9] = {1,0,0, 0,1,0, 0,0,1};
    const auto U = cartesian_shell_rotation(l, I3);
    const int n = shell_size(l, false);
    double worst = 0.0;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        worst = std::max(worst, std::abs(U[(size_t)i*n+j] - (i == j ? 1.0 : 0.0)));
    printf("U(I)=I  l=%d: %.3e %s\n", l, worst, worst < 1e-12 ? "ok" : "FAIL");
    if (worst >= 1e-12) failures++;
  }

  // 3. Representation property: U(R1 R2) == U(R1) U(R2).
  for (int l = 0; l <= 5; l++) {
    const auto R1 = random_rotation(11);
    const auto R2 = random_rotation(22);
    std::vector<double> R12(9, 0.0);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++) R12[i*3+j] += R1[i*3+k] * R2[k*3+j];

    const int n = shell_size(l, false);
    const auto U12 = cartesian_shell_rotation(l, R12.data());
    const auto U1 = cartesian_shell_rotation(l, R1.data());
    const auto U2 = cartesian_shell_rotation(l, R2.data());
    const auto prod = matmul(U1, U2, n);
    double worst = 0.0;
    for (size_t i = 0; i < U12.size(); i++)
      worst = std::max(worst, std::abs(U12[i] - prod[i]));
    printf("homomorphism l=%d: %.3e %s\n", l, worst, worst < 1e-10 ? "ok" : "FAIL");
    if (worst >= 1e-10) failures++;
  }

  // 4. U^T G U == G against the analytic monomial Gram matrix.
  for (int l = 0; l <= 5; l++) {
    const int n = shell_size(l, false);
    const auto G = shell_gram(l);
    double worst = 0.0;
    double scale = 0.0;
    for (double v : G) scale = std::max(scale, std::abs(v));
    for (uint64_t t = 1; t <= 3; t++) {
      const auto R = random_rotation(t);
      const auto U = cartesian_shell_rotation(l, R.data());
      // U^T G U
      std::vector<double> Ut((size_t)n*n);
      for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) Ut[(size_t)i*n+j] = U[(size_t)j*n+i];
      const auto tmp = matmul(Ut, G, n);
      const auto res = matmul(tmp, U, n);
      for (size_t i = 0; i < G.size(); i++)
        worst = std::max(worst, std::abs(res[i] - G[i]));
    }
    printf("U^T G U = G  l=%d: %.3e (scale %.1f) %s\n", l, worst, scale,
           worst < 1e-9 * scale ? "ok" : "FAIL");
    if (worst >= 1e-9 * scale) failures++;
  }

  // 5. Projector idempotency, Cartesian and pure.
  for (int maxl = 1; maxl <= 4; maxl++) {
    std::vector<int> shell_l;
    for (int l = 0; l <= maxl; l++) { shell_l.push_back(l); shell_l.push_back(l); }
    for (bool pure : {true, false}) {
      SphericalProjector proj(make_atom_shells(shell_l, pure));
      const double d = std::max(proj.check_idempotent(RotationLaw::Operator), proj.check_idempotent(RotationLaw::Density));
      printf("idempotent maxl=%d %s: %.3e %s\n", maxl, pure ? "pure" : "cart", d,
             d < 1e-10 ? "ok" : "FAIL");
      if (d >= 1e-10) failures++;
    }
  }

  printf("\n%s (%d failures)\n", failures ? "FAILURES" : "ALL PASS", failures);
  return failures ? 1 : 0;
}
