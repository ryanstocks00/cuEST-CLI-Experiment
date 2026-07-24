/**
 * @file probe_symmetry.cpp
 * @brief Validate the Cartesian rotation representation against a real cuEST
 *        overlap matrix.
 *
 * atomic_symmetry.cpp assumes cuEST lays out Cartesian AOs in libcint order
 * (monomial exponent `a` descending, then `b`) and normalizes a shell with a
 * single shell-wide constant. Both assumptions are checked here by the one
 * condition that catches either being wrong: the overlap of a free atom is a
 * spherically symmetric operator, so U(R)^T S U(R) == S must hold exactly.
 *
 * Usage: probe_symmetry <basis.json> <Z> [--cartesian|--spherical]
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "atomic_symmetry.hpp"
#include "cuest_wrapper/basis.hpp"
#include "cuest_wrapper/context.hpp"
#include "cuest_wrapper/integrals.hpp"
#include "cuest_wrapper/molecule.hpp"
#include "cuest_wrapper/raii.hpp"

using namespace cuest;

/// Gauss-Jordan inverse of a row-major n x n matrix (probe-only; n is small).
static std::vector<double> invert(const std::vector<double>& A_in, int n) {
  std::vector<double> A = A_in;
  std::vector<double> I(static_cast<size_t>(n) * n, 0.0);
  for (int i = 0; i < n; i++) I[static_cast<size_t>(i) * n + i] = 1.0;

  for (int c = 0; c < n; c++) {
    int piv = c;
    for (int r = c + 1; r < n; r++)
      if (std::abs(A[static_cast<size_t>(r) * n + c]) >
          std::abs(A[static_cast<size_t>(piv) * n + c]))
        piv = r;
    for (int j = 0; j < n; j++) {
      std::swap(A[static_cast<size_t>(c) * n + j], A[static_cast<size_t>(piv) * n + j]);
      std::swap(I[static_cast<size_t>(c) * n + j], I[static_cast<size_t>(piv) * n + j]);
    }
    const double d = A[static_cast<size_t>(c) * n + c];
    for (int j = 0; j < n; j++) {
      A[static_cast<size_t>(c) * n + j] /= d;
      I[static_cast<size_t>(c) * n + j] /= d;
    }
    for (int r = 0; r < n; r++) {
      if (r == c) continue;
      const double f = A[static_cast<size_t>(r) * n + c];
      if (f == 0.0) continue;
      for (int j = 0; j < n; j++) {
        A[static_cast<size_t>(r) * n + j] -= f * A[static_cast<size_t>(c) * n + j];
        I[static_cast<size_t>(r) * n + j] -= f * I[static_cast<size_t>(c) * n + j];
      }
    }
  }
  return I;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <basis.json> <Z> [--cartesian|--spherical]\n", argv[0]);
    return 2;
  }
  const std::string basis_path = argv[1];
  const int Z = std::atoi(argv[2]);
  bool is_pure = false;
  std::string dump_overlap;
  for (int i = 3; i < argc; i++) {
    if (std::string(argv[i]) == "--spherical") is_pure = true;
    if (std::string(argv[i]) == "--cartesian") is_pure = false;
    if (std::string(argv[i]) == "--dump-overlap" && i + 1 < argc)
      dump_overlap = argv[++i];
  }

  CuESTContext ctx;
  Molecule mol;
  mol.add_atom_bohr(Z, 0.0, 0.0, 0.0);

  BasisBuilder basis(ctx, mol, is_pure ? 1 : 0);
  basis.build_from_json(basis_path);
  const int nao = static_cast<int>(basis.nao());

  DeviceArray<double> d_S;
  d_S.alloc(static_cast<size_t>(nao) * nao * sizeof(double));
  {
    const OwnedAOPairList& pl = basis.pair_list();
    OneElectronIntegrals oe(ctx, basis.basis(), pl.get(), /*use_jit=*/true);
    oe.compute_overlap(d_S);
  }
  std::vector<double> S(static_cast<size_t>(nao) * nao);
  CUDA_CHECK(cudaMemcpy(S.data(), d_S.get(), S.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));

  const auto shells = make_atom_shells(basis.shell_angular_momenta(), is_pure);
  std::printf("Z=%d  basis=%s  %s  nao=%d  nshell=%d  max_l=%d\n", Z,
              basis_path.c_str(), is_pure ? "spherical" : "cartesian", nao,
              shells.nshell(), shells.max_l());
  if (shells.nao != nao) {
    std::printf("FAIL: shell layout nao=%d != basis nao=%d\n", shells.nao, nao);
    return 1;
  }

  // Report the diagonal of the first d shell (if any): with a single shell-wide
  // normalization constant we expect <d_xx|d_xx> = 3 <d_xy|d_xy>, which is what
  // PySCF/libcint produce and what the rotation code assumes.
  if (!is_pure) {
    for (int s = 0; s < shells.nshell(); s++) {
      if (shells.l[s] != 2) continue;
      const int o = shells.offset[s];
      const double xx = S[static_cast<size_t>(o) * nao + o];
      const double xy = S[static_cast<size_t>(o + 1) * nao + (o + 1)];
      std::printf("  first d shell: S_xx=%.10f  S_xy=%.10f  ratio=%.10f\n", xx, xy,
                  xx / xy);
      break;
    }
  }

  if (!dump_overlap.empty()) {
    std::ofstream out(dump_overlap);
    out.precision(17);
    out << nao << "\n";
    for (int i = 0; i < nao; i++) {
      for (int j = 0; j < nao; j++) out << S[static_cast<size_t>(i) * nao + j] << " ";
      out << "\n";
    }
    std::printf("  wrote overlap to %s\n", dump_overlap.c_str());
  }

  const double dev = verify_rotation_rep(S, shells);
  std::printf("  max |U^T S U - S| = %.3e  %s\n", dev, dev < 1e-9 ? "PASS" : "FAIL");

  SphericalProjector proj(shells);
  bool ok = dev < 1e-9;

  for (auto law : {RotationLaw::Operator, RotationLaw::Density}) {
    const char* name = (law == RotationLaw::Operator) ? "operator" : "density ";
    const double idem = proj.check_idempotent(law);
    std::printf("  %s  max |P(P(X)) - P(X)| = %.3e  %s\n", name, idem,
                idem < 1e-9 ? "PASS" : "FAIL");
    ok = ok && idem < 1e-9;

    // A genuinely spherical matrix of the matching variance must survive
    // untouched: S itself is covariant (an operator matrix), while S^-1 is
    // contravariant and so plays the role of an invariant density.
    const std::vector<double> ref =
        (law == RotationLaw::Operator) ? S : invert(S, nao);
    std::vector<double> P = ref;
    proj.apply(P, law);
    double sdev = 0.0, scale = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
      sdev = std::max(sdev, std::abs(P[i] - ref[i]));
      scale = std::max(scale, std::abs(ref[i]));
    }
    const bool pass = sdev < 1e-8 * std::max(scale, 1.0);
    std::printf("  %s  max |P(X) - X| = %.3e (scale %.3g)  %s\n", name, sdev, scale,
                pass ? "PASS" : "FAIL");
    ok = ok && pass;
  }

  return ok ? 0 : 1;
}
