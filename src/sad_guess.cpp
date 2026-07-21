/**
 * @file sad_guess.cpp — isolated-atom UKS reference densities, disk-cached.
 */
#include "sad_guess.hpp"

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

std::vector<double> spherically_averaged_density(const std::vector<double>& C,
                                                  int lo, int hi,
                                                  uint64_t nao, uint64_t nocc) {
  const int N = static_cast<int>(nao);
  std::vector<double> D(static_cast<size_t>(N) * N, 0.0);
  if (nocc == 0) return D;

  const int noccI = static_cast<int>(nocc);
  const double w_partial =
      (hi > lo) ? static_cast<double>(noccI - lo) / static_cast<double>(hi - lo) : 0.0;

  for (int k = 0; k < hi; k++) {
    const double w = (k < lo) ? 1.0 : w_partial;
    if (w == 0.0) continue;
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        D[static_cast<size_t>(i) * N + j] += w * C[i + k * N] * C[j + k * N];
  }
  return D;
}

namespace {

// ---------------------------------------------------------------------------
// Disk cache: key = element + content hash of basis/aux + guess parameters.
// ---------------------------------------------------------------------------
uint64_t fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ULL) {
  const auto* p = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < n; i++) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

uint64_t hash_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("SAD guess: cannot open '" + path + "'");
  std::ostringstream ss;
  ss << in.rdbuf();
  const std::string contents = ss.str();
  return fnv1a(contents.data(), contents.size());
}

std::string cache_dir() {
  if (const char* env = std::getenv("CUEST_SAD_CACHE_DIR"); env && *env)
    return env;
  if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg)
    return std::string(xdg) + "/cuest_dft/sad_guess";
  const char* home = std::getenv("HOME");
  return std::string(home ? home : ".") + "/.cache/cuest_dft/sad_guess";
}

std::string cache_key(int Z, const SADGuessConfig& cfg, uint64_t basis_hash,
                       uint64_t aux_hash) {
  std::ostringstream ss;
  ss << "Z" << Z << "_f" << cfg.functional_id << "_r" << cfg.radial_pts << "a"
     << cfg.angular_pts << "_p" << (cfg.is_pure ? 1 : 0) << "_j"
     << (cfg.use_jit ? 1 : 0) << "_b" << std::hex << basis_hash << "_x"
     << std::hex << aux_hash;
  return ss.str();
}

constexpr uint32_t kCacheMagic = 0x53414432u;  // "SAD2"

bool try_load_cache(const std::string& path, int Z, AtomicGuessDensity& out) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return false;
  const std::streamsize total = in.tellg();
  in.seekg(0);

  uint32_t magic = 0;
  int32_t zz = 0;
  uint64_t nao = 0;
  const std::streamsize header =
      static_cast<std::streamsize>(sizeof(magic) + sizeof(zz) + sizeof(nao));
  if (total < header) return false;

  in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  in.read(reinterpret_cast<char*>(&zz), sizeof(zz));
  in.read(reinterpret_cast<char*>(&nao), sizeof(nao));
  if (!in || magic != kCacheMagic || zz != Z || nao == 0) return false;

  const std::streamsize expected =
      header + 2 * static_cast<std::streamsize>(nao * nao * sizeof(double));
  if (total != expected) return false;

  out.nao = nao;
  out.D_alpha.resize(nao * nao);
  out.D_beta.resize(nao * nao);
  in.read(reinterpret_cast<char*>(out.D_alpha.data()),
          static_cast<std::streamsize>(sizeof(double) * nao * nao));
  in.read(reinterpret_cast<char*>(out.D_beta.data()),
          static_cast<std::streamsize>(sizeof(double) * nao * nao));
  return in.good() || in.eof();
}

void write_cache(const std::string& path, int Z, const AtomicGuessDensity& d) {
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
    const uint64_t nao = d.nao;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&zz), sizeof(zz));
    out.write(reinterpret_cast<const char*>(&nao), sizeof(nao));
    out.write(reinterpret_cast<const char*>(d.D_alpha.data()),
              static_cast<std::streamsize>(sizeof(double) * d.D_alpha.size()));
    out.write(reinterpret_cast<const char*>(d.D_beta.data()),
              static_cast<std::streamsize>(sizeof(double) * d.D_beta.size()));
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
AtomicGuessDensity compute_atomic_density(CuESTContext& ctx, int Z,
                                          const SADGuessConfig& cfg) {
  Molecule mol;
  mol.add_atom_bohr(Z, 0.0, 0.0, 0.0);
  mol.set_charge(0);

  BasisBuilder atom_basis(ctx, mol, cfg.is_pure ? 1 : 0);
  atom_basis.build_from_json(cfg.basis_path);
  const uint64_t nao = atom_basis.nao();

  AtomicGuessDensity result;
  result.nao = nao;
  result.D_alpha.assign(nao * nao, 0.0);
  result.D_beta.assign(nao * nao, 0.0);

  ECPBuilder ecp_builder(ctx, mol);
  ecp_builder.build_from_json(cfg.basis_path);
  const int n_ecp = static_cast<int>(ecp_builder.total_ecp_electrons());

  const int nelec = mol.nelec(n_ecp);
  if (nelec <= 0) return result;  // fully ECP-replaced core; no electrons to place

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

  XCBuilder xc(ctx, atom_basis.basis(), mol_grid, cfg.functional_id);

  double ex_frac = 0.0, lrc_frac = 0.0, lrc_omega = 0.0;
  hybrid_dfjk_fractions(cfg.functional_id, xc, ex_frac, lrc_frac, lrc_omega);

  auto xyz = mol.xyz_host();
  DFJKBuilder dfjk(ctx, atom_basis.basis(), aux_basis.basis(), xyz.data(),
                   mol.natom(), ex_frac, lrc_frac, lrc_omega, cfg.use_jit);

  SCFParams sp;
  sp.verbose = false;
  sp.print_mos = false;
  sp.print_level = 0;
  sp.force_hcore_guess = true;  // zero-density guess: no SAD recursion
  sp.suppress_output = true;    // don't interleave an ENERGY_COMPONENTS block
  sp.use_jit = cfg.use_jit;
  // Open-shell atoms starting from a raw Hcore guess (no SAD to bootstrap
  // from — this SCF *is* the SAD reference) oscillate more than molecules
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
  if (!atom_scf.converged()) {
    std::cerr << "Warning: atomic SAD reference for Z=" << Z
              << " did not converge; using best-effort density for the guess.\n";
  }

  // spherical_average_occupation above means d_D_a_/d_D_b_ are already the
  // spherically-averaged density at self-consistency — no post-hoc fixup.
  result.D_alpha = atom_scf.density_alpha_host();
  result.D_beta = atom_scf.density_beta_host();
  return result;
}

}  // namespace

AtomicGuessDensity get_atomic_guess_density(CuESTContext& ctx, int Z,
                                            const SADGuessConfig& cfg) {
  const uint64_t basis_hash = hash_file(cfg.basis_path);
  const uint64_t aux_hash = hash_file(cfg.aux_basis_path);
  const std::string path =
      cache_dir() + "/" + cache_key(Z, cfg, basis_hash, aux_hash) + ".sad";

  AtomicGuessDensity result;
  if (try_load_cache(path, Z, result)) return result;

  result = compute_atomic_density(ctx, Z, cfg);
  write_cache(path, Z, result);
  return result;
}

}  // namespace cuest
