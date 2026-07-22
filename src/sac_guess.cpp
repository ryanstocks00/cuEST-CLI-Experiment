/**
 * @file sac_guess.cpp — isolated-atom UKS reference orbitals, disk-cached.
 */
#include "sac_guess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <utility>

#include "cuest_wrapper/basis.hpp"
#include "cuest_wrapper/context.hpp"
#include "cuest_wrapper/grid.hpp"
#include "cuest_wrapper/integrals.hpp"
#include "cuest_wrapper/molecule.hpp"
#include "dfjk_hybrid.hpp"
#include "scf.hpp"

namespace cuest {

int aufbau_hund_multiplicity(int nelec) {
  if (nelec <= 0) return 1;

  // Madelung filling order (1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p 7s
  // 5f 6d 7p): only angular momentum l matters for degeneracy/spin here.
  static const int order_l[] = {0, 0, 1, 0, 1, 0, 2, 1, 0, 2,
                                1, 0, 3, 2, 1, 0, 3, 2, 1};

  int remaining = nelec;
  for (int l : order_l) {
    const int g = 2 * l + 1;    // orbital degeneracy of this subshell
    const int cap = 2 * g;      // max electrons (Pauli)
    if (remaining <= cap) {
      // Last (possibly partially filled) subshell: Hund's rule fills each
      // of the g degenerate orbitals singly before any pairing.
      const int n = remaining;
      const int unpaired = (n <= g) ? n : (2 * g - n);
      return unpaired + 1;  // multiplicity = 2S+1, S = unpaired/2
    }
    remaining -= cap;
  }
  return 1;  // beyond Z=118; shouldn't happen in practice
}

std::pair<int, int> degenerate_frontier_group(const std::vector<double>& mo_energies,
                                              int nocc, double tol) {
  const int N = static_cast<int>(mo_energies.size());
  if (nocc <= 0 || nocc >= N) return {nocc, nocc};
  int lo = nocc - 1, hi = nocc;
  bool changed = true;
  while (changed) {
    changed = false;
    if (lo > 0 && std::abs(mo_energies[lo] - mo_energies[lo - 1]) < tol) {
      lo--;
      changed = true;
    }
    if (hi < N && std::abs(mo_energies[hi] - mo_energies[hi - 1]) < tol) {
      hi++;
      changed = true;
    }
  }
  return {lo, hi};
}

std::vector<double> weighted_occupied_coefficients(const std::vector<double>& C,
                                                   int lo, int hi,
                                                   uint64_t nocc, uint64_t nao) {
  const int N = static_cast<int>(nao);
  std::vector<double> Cw(static_cast<size_t>(N) * std::max(hi, 0), 0.0);
  if (hi <= 0) return Cw;

  const int noccI = static_cast<int>(nocc);
  const double w_partial =
      (hi > lo) ? static_cast<double>(noccI - lo) / static_cast<double>(hi - lo) : 0.0;
  const double s_partial = std::sqrt(std::max(w_partial, 0.0));

  for (int k = 0; k < hi; k++) {
    const double s = (k < lo) ? 1.0 : s_partial;
    for (int i = 0; i < N; i++)
      Cw[static_cast<size_t>(i) + static_cast<size_t>(k) * N] = s * C[static_cast<size_t>(i) + static_cast<size_t>(k) * N];
  }
  return Cw;
}

std::vector<double> spherically_averaged_density(const std::vector<double>& C,
                                                  int lo, int hi,
                                                  uint64_t nao, uint64_t nocc) {
  const int N = static_cast<int>(nao);
  std::vector<double> D(static_cast<size_t>(N) * N, 0.0);
  if (nocc == 0) return D;

  const auto Cw = weighted_occupied_coefficients(C, lo, hi, nocc, nao);
  for (int k = 0; k < hi; k++)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        D[static_cast<size_t>(i) * N + j] +=
            Cw[static_cast<size_t>(i) + static_cast<size_t>(k) * N] *
            Cw[static_cast<size_t>(j) + static_cast<size_t>(k) * N];
  return D;
}

namespace {

// ---------------------------------------------------------------------------
// Disk cache: key = element + basis/aux-basis file names + guess parameters.
// Keyed on filename, not file content: basis set files are static, versioned
// data (a change means a new file/name in practice, not an in-place edit),
// so this avoids reading and hashing the whole file just to build a cache
// key. If you ever DO hand-edit a basis JSON in place, clear the cache dir.
// ---------------------------------------------------------------------------
std::string cache_dir() {
  if (const char* env = std::getenv("CUEST_SAC_CACHE_DIR"); env && *env)
    return env;
  if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg)
    return std::string(xdg) + "/cuest_dft/sac_guess";
  const char* home = std::getenv("HOME");
  return std::string(home ? home : ".") + "/.cache/cuest_dft/sac_guess";
}

std::string cache_key(int Z, const SACGuessConfig& cfg) {
  namespace fs = std::filesystem;
  std::ostringstream ss;
  ss << "Z" << Z
     << "_" << fs::path(cfg.basis_path).stem().string()
     << "_" << fs::path(cfg.aux_basis_path).stem().string()
     << "_f" << static_cast<int>(cfg.functional)
     << "_r" << cfg.radial_pts << "a" << cfg.angular_pts
     << "_p" << (cfg.is_pure ? 1 : 0)
     << "_j" << (cfg.use_jit ? 1 : 0);
  return ss.str();
}

constexpr uint32_t kCacheMagic = 0x53414331u;  // "SAC1"

bool try_load_cache(const std::string& path, int Z, AtomicGuessOrbitals& out) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return false;
  const std::streamsize total = in.tellg();
  in.seekg(0);

  uint32_t magic = 0;
  int32_t zz = 0;
  uint64_t nao = 0, n_cols_a = 0, n_cols_b = 0;
  const std::streamsize header = static_cast<std::streamsize>(
      sizeof(magic) + sizeof(zz) + sizeof(nao) + sizeof(n_cols_a) + sizeof(n_cols_b));
  if (total < header) return false;

  in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  in.read(reinterpret_cast<char*>(&zz), sizeof(zz));
  in.read(reinterpret_cast<char*>(&nao), sizeof(nao));
  in.read(reinterpret_cast<char*>(&n_cols_a), sizeof(n_cols_a));
  in.read(reinterpret_cast<char*>(&n_cols_b), sizeof(n_cols_b));
  if (!in || magic != kCacheMagic || zz != Z || nao == 0) return false;

  const std::streamsize expected = header + static_cast<std::streamsize>(
      (nao * n_cols_a + nao * n_cols_b) * sizeof(double));
  if (total != expected) return false;

  out.nao = nao;
  out.n_cols_alpha = n_cols_a;
  out.n_cols_beta = n_cols_b;
  out.Cocc_alpha.resize(nao * n_cols_a);
  out.Cocc_beta.resize(nao * n_cols_b);
  in.read(reinterpret_cast<char*>(out.Cocc_alpha.data()),
          static_cast<std::streamsize>(sizeof(double) * out.Cocc_alpha.size()));
  in.read(reinterpret_cast<char*>(out.Cocc_beta.data()),
          static_cast<std::streamsize>(sizeof(double) * out.Cocc_beta.size()));
  return in.good() || in.eof();
}

void write_cache(const std::string& path, int Z, const AtomicGuessOrbitals& d) {
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::create_directories(fs::path(path).parent_path(), ec);

  const std::string tmp_path =
      path + ".tmp" + std::to_string(static_cast<long>(getpid()));
  {
    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    if (!out) return;  // best-effort cache; missing write access isn't fatal
    const uint32_t magic = kCacheMagic;
    const int32_t zz = Z;
    const uint64_t nao = d.nao, n_cols_a = d.n_cols_alpha, n_cols_b = d.n_cols_beta;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&zz), sizeof(zz));
    out.write(reinterpret_cast<const char*>(&nao), sizeof(nao));
    out.write(reinterpret_cast<const char*>(&n_cols_a), sizeof(n_cols_a));
    out.write(reinterpret_cast<const char*>(&n_cols_b), sizeof(n_cols_b));
    out.write(reinterpret_cast<const char*>(d.Cocc_alpha.data()),
              static_cast<std::streamsize>(sizeof(double) * d.Cocc_alpha.size()));
    out.write(reinterpret_cast<const char*>(d.Cocc_beta.data()),
              static_cast<std::streamsize>(sizeof(double) * d.Cocc_beta.size()));
    if (!out) {
      std::error_code rm_ec;
      fs::remove(tmp_path, rm_ec);
      return;
    }
  }
  fs::rename(tmp_path, path, ec);
  if (ec) {
    std::error_code rm_ec;
    fs::remove(tmp_path, rm_ec);
  }
}

// ---------------------------------------------------------------------------
// Isolated-atom UKS reference SCF.
// ---------------------------------------------------------------------------
AtomicGuessOrbitals compute_atomic_orbitals(CuESTContext& ctx, int Z,
                                            const SACGuessConfig& cfg,
                                            bool& converged) {
  converged = false;
  Molecule mol;
  mol.add_atom_bohr(Z, 0.0, 0.0, 0.0);
  mol.set_charge(0);

  BasisBuilder atom_basis(ctx, mol, cfg.is_pure ? 1 : 0);
  atom_basis.build_from_json(cfg.basis_path);
  const uint64_t nao = atom_basis.nao();

  AtomicGuessOrbitals result;
  result.nao = nao;

  ECPBuilder ecp_builder(ctx, mol);
  ecp_builder.build_from_json(cfg.basis_path);
  const int n_ecp = static_cast<int>(ecp_builder.total_ecp_electrons());

  const int nelec = mol.nelec(n_ecp);
  if (nelec <= 0) {
    converged = true;  // fully ECP-replaced core; nothing to solve
    return result;
  }

  const int mult = aufbau_hund_multiplicity(nelec);
  mol.set_multiplicity(mult);

  AuxBasis aux_basis(ctx, mol, /*is_pure=*/1);
  aux_basis.build_from_json(cfg.aux_basis_path);

  std::unique_ptr<ECPIntegrals> ecp_int;
  if (ecp_builder.has_ecp()) {
    auto xyz_h = mol.xyz_host();
    ecp_int = std::make_unique<ECPIntegrals>(
        ctx, atom_basis.basis(), xyz_h.data(), ecp_builder.num_active_atoms(),
        ecp_builder.ecp_indices().data(), ecp_builder.ecp_atoms().data());
  }

  GridBuilder grid_builder(ctx, mol, cfg.radial_pts, cfg.angular_pts);
  auto mol_grid = grid_builder.build();

  XCBuilder xc(ctx, atom_basis.basis(), mol_grid, cfg.functional);

  double ex_frac = 0.0, lrc_frac = 0.0, lrc_omega = 0.0;
  hybrid_dfjk_fractions(cfg.functional, xc, ex_frac, lrc_frac, lrc_omega);

  auto xyz = mol.xyz_host();
  DFJKBuilder dfjk(ctx, atom_basis.basis(), aux_basis.basis(), xyz.data(),
                   mol.natom(), ex_frac, lrc_frac, lrc_omega, cfg.use_jit,
                   cfg.fitting_cutoff, cfg.fitting_relative_conditioning,
                   cfg.fitting_algorithm);

  SCFParams sp;
  sp.verbose = false;
  sp.print_mos = false;
  sp.print_level = 0;
  sp.force_hcore_guess = true;  // zero-density guess: no SAC recursion
  sp.suppress_output = true;    // don't interleave an ENERGY_COMPONENTS block
  sp.use_jit = cfg.use_jit;
  // Open-shell atoms starting from a raw Hcore guess (no SAC to bootstrap
  // from — this SCF *is* the SAC reference) oscillate more than molecules
  // do; damping + more iterations makes convergence reliable without
  // touching the main molecular SCF's defaults.
  sp.damping = 0.3;
  sp.max_iter = 500;
  // Spherically average the occupation of the HOMO/LUMO-straddling
  // degenerate shell at every iteration (not just at the end): only filling
  // one arbitrary real orbital (e.g. px, py but never pz) out of a
  // partially-filled shell feeds an anisotropic density into J/K/Vxc, which
  // then self-consistently splits the degeneracy — so symmetrizing after
  // convergence would be too late.
  sp.spherical_average_occupation = true;

  SCFSolver atom_scf(ctx, atom_basis, dfjk, &xc, &ecp_builder, ecp_int.get(),
                     mol, sp);
  atom_scf.run();
  converged = atom_scf.converged();
  if (!converged) {
    std::cerr << "Warning: atomic SAC reference for Z=" << Z
              << " did not converge; using best-effort orbitals for the guess.\n";
  }

  // spherical_average_occupation above means the frontier group locked in
  // during convergence is exactly the atom's true degenerate shell (its own
  // Fock is exactly rotationally symmetric throughout) — reuse it directly
  // rather than re-deriving from the final eigenvalues.
  const auto [lo_a, hi_a] = atom_scf.frontier_group_alpha();
  const auto [lo_b, hi_b] = atom_scf.frontier_group_beta();
  result.n_cols_alpha = static_cast<uint64_t>(std::max(hi_a, 0));
  result.n_cols_beta = static_cast<uint64_t>(std::max(hi_b, 0));
  result.Cocc_alpha = weighted_occupied_coefficients(
      atom_scf.mo_coefficients_host(), lo_a, hi_a, atom_scf.nocc_alpha(), nao);
  result.Cocc_beta = weighted_occupied_coefficients(
      atom_scf.mo_coefficients_beta_host(), lo_b, hi_b, atom_scf.nocc_beta(), nao);
  return result;
}

}  // namespace

AtomicGuessOrbitals get_atomic_guess_orbitals(CuESTContext& ctx, int Z,
                                              const SACGuessConfig& cfg) {
  const std::string path = cache_dir() + "/" + cache_key(Z, cfg) + ".sac";

  AtomicGuessOrbitals result;
  if (try_load_cache(path, Z, result)) return result;

  bool converged = false;
  result = compute_atomic_orbitals(ctx, Z, cfg, converged);
  if (converged) write_cache(path, Z, result);
  return result;
}

}  // namespace cuest
