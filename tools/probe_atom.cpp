/**
 * @file probe_atom.cpp
 * @brief Run the spherically symmetric free-atom reference for one element and
 *        report its energy and density, for comparison against PySCF's
 *        AtomSphAverageRHF.
 *
 * Usage: probe_atom <basis.json> <aux.json> <Z> [--cartesian|--spherical]
 *                   [--dump-density <file>]
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "cuest_wrapper/basis.hpp"
#include "cuest_wrapper/context.hpp"
#include "cuest_wrapper/molecule.hpp"
#include "sad_guess.hpp"

using namespace cuest;

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <basis.json> <aux.json> <Z> "
                 "[--cartesian|--spherical] [--dump-density <file>]\n",
                 argv[0]);
    return 2;
  }

  SADGuessConfig cfg;
  cfg.basis_path = argv[1];
  cfg.aux_basis_path = argv[2];
  const int Z = std::atoi(argv[3]);
  cfg.is_pure = true;
  std::string dump;
  for (int i = 4; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--spherical") cfg.is_pure = true;
    else if (a == "--cartesian") cfg.is_pure = false;
    else if (a == "--dump-density" && i + 1 < argc) dump = argv[++i];
  }

  CuESTContext ctx;
  const AtomicGuessOrbitals atom = get_atomic_guess_orbitals(ctx, Z, cfg);

  std::printf("Z=%d  %s  nao=%llu  ncols=%llu  nelec=%d\n", Z,
              cfg.is_pure ? "spherical" : "cartesian",
              static_cast<unsigned long long>(atom.nao),
              static_cast<unsigned long long>(atom.n_cols), atom.nelec);
  std::printf("ATOM_ENERGY %.12f\n", atom.energy);

  // Emit the shell angular momenta in our AO order. PySCF re-sorts an atom's
  // shells by l when building a molecule, whereas BasisBuilder keeps them in
  // basis-file order — which differ for any basis whose JSON is not already
  // l-sorted (Pople SP shells, notably). Same span and same energy, permuted
  // AOs; the validator uses this to undo the permutation before comparing
  // density matrices element by element.
  {
    Molecule m;
    m.add_atom_bohr(Z, 0.0, 0.0, 0.0);
    BasisBuilder b(ctx, m, cfg.is_pure ? 1 : 0);
    b.build_from_json(cfg.basis_path);
    std::printf("SHELL_L");
    for (int l : b.shell_angular_momenta()) std::printf(" %d", l);
    std::printf("\n");
  }

  // D_total = 2 * Cocc Cocc^T (Cocc is weighted to give the alpha density).
  const int n = static_cast<int>(atom.nao);
  std::vector<double> D(static_cast<size_t>(n) * n, 0.0);
  for (uint64_t c = 0; c < atom.n_cols; c++)
    for (int i = 0; i < n; i++) {
      const double v = atom.Cocc[c * n + i];
      if (v == 0.0) continue;
      for (int j = 0; j < n; j++)
        D[static_cast<size_t>(i) * n + j] += 2.0 * v * atom.Cocc[c * n + j];
    }

  if (!dump.empty()) {
    std::ofstream out(dump);
    out.precision(17);
    out << n << "\n";
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) out << D[static_cast<size_t>(i) * n + j] << " ";
      out << "\n";
    }
    std::printf("wrote density to %s\n", dump.c_str());
  }
  return 0;
}
