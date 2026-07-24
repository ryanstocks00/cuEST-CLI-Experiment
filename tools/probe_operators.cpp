/**
 * @file probe_operators.cpp
 * @brief Evaluate the exchange (K) and XC operators on a *supplied* density,
 *        so they can be compared against another code term by term.
 *
 * Comparing converged SCF energies cannot localise a discrepancy: if the
 * operators differ at all, the converged densities differ too, and every
 * energy term then differs for two reasons at once. This tool removes that
 * confound by taking the occupied orbitals as input, so both codes evaluate
 * their operators on bit-identical data and any difference is the operator.
 *
 * Prints, for the given functional:
 *   EX_FRAC / LRC_FRAC / LRC_OMEGA  the range-separation parameters used
 *   E_K    = -Tr[D_a K(D_a)]        exact-exchange energy (fractions baked in)
 *   E_XC                            semilocal XC energy
 * and optionally dumps the K and Vxc matrices.
 *
 * Usage: probe_operators <xyz> <basis.json> <aux.json> <functional> <cocc.txt>
 *                        [--cartesian|--spherical] [--radial N] [--angular N]
 *                        [--dump-k FILE] [--dump-vxc FILE]
 *
 * cocc.txt: first line "nao ncols", then nao*ncols values, column-major
 * (column k is occupied orbital k), such that D_alpha = Cocc Cocc^T.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "cuest_wrapper/basis.hpp"
#include "cuest_wrapper/context.hpp"
#include "cuest_wrapper/grid.hpp"
#include "cuest_wrapper/integrals.hpp"
#include "cuest_wrapper/molecule.hpp"
#include "cuest_wrapper/raii.hpp"
#include "dfjk_hybrid.hpp"
#include "functionals.hpp"
#include "io/parsers.hpp"

using namespace cuest;

static void dump_matrix(const std::string& path, const std::vector<double>& M, int n) {
  std::ofstream out(path);
  out.precision(17);
  out << n << "\n";
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) out << M[static_cast<size_t>(i) * n + j] << " ";
    out << "\n";
  }
}

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr,
                 "usage: %s <xyz> <basis.json> <aux.json> <functional> <cocc.txt> "
                 "[--cartesian|--spherical] [--radial N] [--angular N] "
                 "[--dump-k FILE] [--dump-vxc FILE]\n", argv[0]);
    return 2;
  }
  const std::string xyz_path = argv[1], basis_path = argv[2];
  const std::string aux_path = argv[3], func_name = argv[4], cocc_path = argv[5];
  int is_pure = 1, radial = 200, angular = 770;
  std::string dump_k, dump_vxc, dump_ovlp;
  bool split_rsh = false;
  for (int i = 6; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--cartesian") is_pure = 0;
    else if (a == "--spherical") is_pure = 1;
    else if (a == "--radial" && i + 1 < argc) radial = std::atoi(argv[++i]);
    else if (a == "--angular" && i + 1 < argc) angular = std::atoi(argv[++i]);
    else if (a == "--dump-k" && i + 1 < argc) dump_k = argv[++i];
    else if (a == "--dump-vxc" && i + 1 < argc) dump_vxc = argv[++i];
    else if (a == "--dump-overlap" && i + 1 < argc) dump_ovlp = argv[++i];
    else if (a == "--split-rsh") split_rsh = true;
  }

  auto functional = functional_from_name(func_name);
  if (!functional) { std::fprintf(stderr, "unknown functional %s\n", func_name.c_str()); return 2; }

  CuESTContext ctx;
  auto xyz_data = parse_xyz(xyz_path);
  Molecule mol;
  for (size_t i = 0; i < xyz_data.n_atoms; i++)
    mol.add_atom_bohr(xyz_data.symbols[i], xyz_data.xyz[3 * i],
                      xyz_data.xyz[3 * i + 1], xyz_data.xyz[3 * i + 2]);

  BasisBuilder basis(ctx, mol, is_pure);
  basis.build_from_json(basis_path);
  const int nao = static_cast<int>(basis.nao());

  AuxBasis aux(ctx, mol, 1);
  aux.build_from_json(aux_path);

  // --- read the supplied occupied orbitals ---
  std::ifstream in(cocc_path);
  if (!in) { std::fprintf(stderr, "cannot read %s\n", cocc_path.c_str()); return 2; }
  int n_in = 0, ncols = 0;
  in >> n_in >> ncols;
  if (n_in != nao) {
    std::fprintf(stderr, "cocc nao=%d but basis nao=%d\n", n_in, nao);
    return 2;
  }
  std::vector<double> Cocc(static_cast<size_t>(nao) * ncols);
  for (auto& v : Cocc) in >> v;

  GridBuilder gb(ctx, mol, radial, angular);
  auto grid = gb.build();
  XCBuilder xc(ctx, basis.basis(), grid, *functional);

  double ex_frac = 0, lrc_frac = 0, lrc_omega = 0;
  hybrid_dfjk_fractions(*functional, xc, ex_frac, lrc_frac, lrc_omega);
  std::printf("EX_FRAC %.12f\nLRC_FRAC %.12f\nLRC_OMEGA %.12f\n",
              ex_frac, lrc_frac, lrc_omega);

  auto xyz_h = mol.xyz_host();
  // Default: one DF plan carrying both fractions, as the SCF uses it.
  // --split-rsh instead builds two plans -- full-Coulomb and attenuated -- and
  // combines their K matrices here, to test whether cuEST's combined plan fits
  // the attenuated kernel with its own metric or reuses the Coulomb one.
  DFJKBuilder dfjk(ctx, basis.basis(), aux.basis(), xyz_h.data(), mol.natom(),
                   split_rsh ? 1.0 : ex_frac, split_rsh ? 0.0 : lrc_frac,
                   split_rsh ? 0.0 : lrc_omega, /*use_jit=*/true);

  const size_t n2 = static_cast<size_t>(nao) * nao;
  DeviceArray<double> d_Cocc, d_K, d_Vxc;
  d_Cocc.alloc(Cocc.size() * sizeof(double));
  d_K.alloc(n2 * sizeof(double));
  d_Vxc.alloc(n2 * sizeof(double));
  CUDA_CHECK(cudaMemcpy(d_Cocc, Cocc.data(), Cocc.size() * sizeof(double),
                        cudaMemcpyHostToDevice));

  // D_alpha = Cocc Cocc^T, on the host (this tool is a diagnostic, not a kernel).
  std::vector<double> D(n2, 0.0);
  for (int c = 0; c < ncols; c++)
    for (int i = 0; i < nao; i++) {
      const double v = Cocc[static_cast<size_t>(c) * nao + i];
      if (v == 0.0) continue;
      for (int j = 0; j < nao; j++)
        D[static_cast<size_t>(i) * nao + j] += v * Cocc[static_cast<size_t>(c) * nao + j];
    }

  // Sanity: Tr[D S] must equal the number of alpha electrons. If it does not,
  // the supplied orbitals are not in this code's AO ordering and every
  // comparison below is meaningless.
  {
    DeviceArray<double> d_S;
    d_S.alloc(n2 * sizeof(double));
    const OwnedAOPairList& pl = basis.pair_list();
    OneElectronIntegrals oe(ctx, basis.basis(), pl.get(), true);
    oe.compute_overlap(d_S);
    std::vector<double> S(n2);
    CUDA_CHECK(cudaMemcpy(S.data(), d_S.get(), n2 * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double tr = 0.0;
    for (size_t i = 0; i < n2; i++) tr += D[i] * S[i];
    std::printf("TR_DS %.12f\n", tr);
    if (!dump_ovlp.empty()) dump_matrix(dump_ovlp, S, nao);
  }

  std::vector<double> K(n2, 0.0), Vxc(n2, 0.0);
  double e_k = 0.0, e_xc = 0.0;

  if (ex_frac != 0.0 || lrc_frac != 0.0) {
    dfjk.compute_K(static_cast<uint64_t>(ncols), d_Cocc, d_K);
    CUDA_CHECK(cudaMemcpy(K.data(), d_K.get(), n2 * sizeof(double),
                          cudaMemcpyDeviceToHost));
    if (split_rsh) {
      for (auto& v : K) v *= ex_frac;   // this plan returned the bare K_full
      if (lrc_frac != 0.0) {
        DFJKBuilder dfjk_lr(ctx, basis.basis(), aux.basis(), xyz_h.data(),
                            mol.natom(), 0.0, 1.0, lrc_omega, /*use_jit=*/true);
        DeviceArray<double> d_Klr;
        d_Klr.alloc(n2 * sizeof(double));
        dfjk_lr.compute_K(static_cast<uint64_t>(ncols), d_Cocc, d_Klr);
        std::vector<double> Klr(n2);
        CUDA_CHECK(cudaMemcpy(Klr.data(), d_Klr.get(), n2 * sizeof(double),
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n2; i++) K[i] += lrc_frac * Klr[i];
      }
    }
    for (size_t i = 0; i < n2; i++) e_k -= D[i] * K[i];  // -Tr[D K], both symmetric
  }
  if (!xc.is_hf()) {
    xc.compute_vxc_rks(static_cast<uint64_t>(ncols), d_Cocc, &e_xc, d_Vxc);
    CUDA_CHECK(cudaMemcpy(Vxc.data(), d_Vxc.get(), n2 * sizeof(double),
                          cudaMemcpyDeviceToHost));
  }

  std::printf("E_K %.14f\nE_XC %.14f\n", e_k, e_xc);
  if (!dump_k.empty()) dump_matrix(dump_k, K, nao);
  if (!dump_vxc.empty()) dump_matrix(dump_vxc, Vxc, nao);
  return 0;
}
