/**
 * @file sad_guess.cpp — spherically symmetric free-atom reference, disk-cached.
 */
#include "sad_guess.hpp"

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

#include "atomic_config.hpp"
#include "atomic_symmetry.hpp"
#include "cuest_wrapper/basis.hpp"
#include "cuest_wrapper/context.hpp"
#include "cuest_wrapper/grid.hpp"
#include "cuest_wrapper/integrals.hpp"
#include "cuest_wrapper/molecule.hpp"
#include "cuest_wrapper/raii.hpp"
#include "dfjk_hybrid.hpp"

namespace cuest {
namespace {

// ---------------------------------------------------------------------------
// Small dense host linear algebra. The atom has at most a few hundred AOs, so
// these run in milliseconds and keep the atomic solver independent of the
// cuBLAS/cuSOLVER handles SCFSolver owns.
//
// Convention: symmetric matrices (S, F, D, Hcore) are row-major, which for a
// symmetric matrix is the same bytes as column-major and so interchanges freely
// with cuEST. Coefficient matrices (X, C) are column-major — column k is
// orbital k — matching what cuEST's DF-K expects for Cocc.
// ---------------------------------------------------------------------------

/// Cyclic Jacobi eigensolver for a symmetric row-major matrix.
///
/// Chosen over a QR-based solver because it is unconditionally accurate for
/// clustered and exactly-degenerate spectra, which is precisely what a
/// projected atomic Fock produces — the (2l+1)-fold degeneracies have to come
/// out numerically identical for manifold detection to work.
///
/// `evals` ascending; `evecs` column-major with column k the k-th eigenvector.
void jacobi_symmetric(std::vector<double> A, int n, std::vector<double>& evals,
                      std::vector<double>& evecs) {
  evecs.assign(static_cast<size_t>(n) * n, 0.0);
  for (int i = 0; i < n; i++) evecs[static_cast<size_t>(i) * n + i] = 1.0;

  auto at = [&](int i, int j) -> double& { return A[static_cast<size_t>(i) * n + j]; };
  // evecs column-major: element (row, col)
  auto vt = [&](int r, int c) -> double& { return evecs[static_cast<size_t>(c) * n + r]; };

  for (int sweep = 0; sweep < 100; sweep++) {
    double off = 0.0;
    for (int i = 0; i < n; i++)
      for (int j = i + 1; j < n; j++) off += at(i, j) * at(i, j);
    if (std::sqrt(2.0 * off) < 1e-14) break;

    for (int p = 0; p < n - 1; p++)
      for (int q = p + 1; q < n; q++) {
        const double apq = at(p, q);
        if (std::abs(apq) < 1e-300) continue;
        const double theta = (at(q, q) - at(p, p)) / (2.0 * apq);
        const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                         (std::abs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0);
        const double s = t * c;

        for (int k = 0; k < n; k++) {
          const double akp = at(k, p), akq = at(k, q);
          at(k, p) = c * akp - s * akq;
          at(k, q) = s * akp + c * akq;
        }
        for (int k = 0; k < n; k++) {
          const double apk = at(p, k), aqk = at(q, k);
          at(p, k) = c * apk - s * aqk;
          at(q, k) = s * apk + c * aqk;
        }
        for (int k = 0; k < n; k++) {
          const double vkp = vt(k, p), vkq = vt(k, q);
          vt(k, p) = c * vkp - s * vkq;
          vt(k, q) = s * vkp + c * vkq;
        }
      }
  }

  std::vector<int> order(n);
  for (int i = 0; i < n; i++) order[i] = i;
  std::vector<double> diag(n);
  for (int i = 0; i < n; i++) diag[i] = at(i, i);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return diag[a] < diag[b]; });

  std::vector<double> sorted_vecs(static_cast<size_t>(n) * n);
  evals.resize(n);
  for (int k = 0; k < n; k++) {
    evals[k] = diag[order[k]];
    for (int r = 0; r < n; r++)
      sorted_vecs[static_cast<size_t>(k) * n + r] =
          evecs[static_cast<size_t>(order[k]) * n + r];
  }
  evecs.swap(sorted_vecs);
}

/// Canonical orthogonalizer X (nao x nmo, column-major) with X^T S X = I,
/// dropping eigenvalues of S below `thresh` to handle near-linear dependence.
std::vector<double> canonical_orthogonalizer(const std::vector<double>& S, int nao,
                                             int& nmo, double thresh = 1e-7) {
  std::vector<double> s, U;
  jacobi_symmetric(S, nao, s, U);

  nmo = 0;
  for (int i = 0; i < nao; i++)
    if (s[i] > thresh) nmo++;

  std::vector<double> X(static_cast<size_t>(nao) * nmo, 0.0);
  int col = 0;
  for (int i = 0; i < nao; i++) {
    if (s[i] <= thresh) continue;
    const double inv = 1.0 / std::sqrt(s[i]);
    for (int r = 0; r < nao; r++)
      X[static_cast<size_t>(col) * nao + r] = U[static_cast<size_t>(i) * nao + r] * inv;
    col++;
  }
  return X;
}

/// C = A^T B for column-major A (n x p) and row-major symmetric B (n x n),
/// returning row-major (p x n). Helper for the X^T F X congruence below.
std::vector<double> congruence(const std::vector<double>& X, int nao, int nmo,
                               const std::vector<double>& F) {
  // T = F X  (nao x nmo, column-major)
  std::vector<double> T(static_cast<size_t>(nao) * nmo, 0.0);
  for (int c = 0; c < nmo; c++)
    for (int i = 0; i < nao; i++) {
      double acc = 0.0;
      for (int k = 0; k < nao; k++)
        acc += F[static_cast<size_t>(i) * nao + k] * X[static_cast<size_t>(c) * nao + k];
      T[static_cast<size_t>(c) * nao + i] = acc;
    }
  // Ft = X^T T  (nmo x nmo, row-major; symmetric)
  std::vector<double> Ft(static_cast<size_t>(nmo) * nmo, 0.0);
  for (int a = 0; a < nmo; a++)
    for (int b = 0; b < nmo; b++) {
      double acc = 0.0;
      for (int k = 0; k < nao; k++)
        acc += X[static_cast<size_t>(a) * nao + k] * T[static_cast<size_t>(b) * nao + k];
      Ft[static_cast<size_t>(a) * nmo + b] = acc;
    }
  return Ft;
}

double frobenius_dot(const std::vector<double>& A, const std::vector<double>& B) {
  double acc = 0.0;
  for (size_t i = 0; i < A.size(); i++) acc += A[i] * B[i];
  return acc;
}

/// Pulay DIIS on the Fock matrix in the orthonormal basis, where the error is
/// simply the commutator [F, D] (no overlap factors needed).
class HostDIIS {
 public:
  explicit HostDIIS(int max_space) : max_space_(max_space) {}

  void push(std::vector<double> F, std::vector<double> err) {
    F_.push_back(std::move(F));
    E_.push_back(std::move(err));
    if (static_cast<int>(F_.size()) > max_space_) {
      F_.erase(F_.begin());
      E_.erase(E_.begin());
    }
  }

  /// Replace `F` by the DIIS extrapolant. No-op with fewer than two vectors.
  void extrapolate(std::vector<double>& F) const {
    const int m = static_cast<int>(F_.size());
    if (m < 2) return;

    // Solve the Pulay system [B 1; 1 0][c; -lambda] = [0; 1].
    const int dim = m + 1;
    std::vector<double> B(static_cast<size_t>(dim) * dim, 0.0);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < m; j++)
        B[static_cast<size_t>(i) * dim + j] = frobenius_dot(E_[i], E_[j]);
    for (int i = 0; i < m; i++) {
      B[static_cast<size_t>(i) * dim + m] = -1.0;
      B[static_cast<size_t>(m) * dim + i] = -1.0;
    }
    std::vector<double> rhs(dim, 0.0);
    rhs[m] = -1.0;

    // Gaussian elimination with partial pivoting; bail out to plain damping if
    // the history has gone linearly dependent.
    for (int c = 0; c < dim; c++) {
      int piv = c;
      for (int r = c + 1; r < dim; r++)
        if (std::abs(B[static_cast<size_t>(r) * dim + c]) >
            std::abs(B[static_cast<size_t>(piv) * dim + c]))
          piv = r;
      if (std::abs(B[static_cast<size_t>(piv) * dim + c]) < 1e-14) return;
      if (piv != c) {
        for (int j = 0; j < dim; j++)
          std::swap(B[static_cast<size_t>(c) * dim + j],
                    B[static_cast<size_t>(piv) * dim + j]);
        std::swap(rhs[c], rhs[piv]);
      }
      for (int r = c + 1; r < dim; r++) {
        const double f = B[static_cast<size_t>(r) * dim + c] /
                         B[static_cast<size_t>(c) * dim + c];
        if (f == 0.0) continue;
        for (int j = c; j < dim; j++)
          B[static_cast<size_t>(r) * dim + j] -= f * B[static_cast<size_t>(c) * dim + j];
        rhs[r] -= f * rhs[c];
      }
    }
    std::vector<double> x(dim, 0.0);
    for (int r = dim - 1; r >= 0; r--) {
      double acc = rhs[r];
      for (int j = r + 1; j < dim; j++) acc -= B[static_cast<size_t>(r) * dim + j] * x[j];
      x[r] = acc / B[static_cast<size_t>(r) * dim + r];
    }

    std::fill(F.begin(), F.end(), 0.0);
    for (int i = 0; i < m; i++)
      for (size_t k = 0; k < F.size(); k++) F[k] += x[i] * F_[i][k];
  }

 private:
  int max_space_;
  std::vector<std::vector<double>> F_, E_;
};

// ---------------------------------------------------------------------------
// Disk cache: key = element + basis/aux-basis file names + guess parameters.
// Keyed on filename, not file content: basis set files are static, versioned
// data (a change means a new file/name in practice, not an in-place edit), so
// this avoids reading and hashing the whole file just to build a cache key. If
// you ever DO hand-edit a basis JSON in place, clear the cache dir.
// ---------------------------------------------------------------------------
std::string cache_dir() {
  if (const char* env = std::getenv("CUEST_SAD_CACHE_DIR"); env && *env) return env;
  if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg)
    return std::string(xdg) + "/cuest_dft/sad_guess";
  const char* home = std::getenv("HOME");
  return std::string(home ? home : ".") + "/.cache/cuest_dft/sad_guess";
}

std::string cache_key(int Z, const SADGuessConfig& cfg) {
  namespace fs = std::filesystem;
  std::ostringstream ss;
  ss << "Z" << Z << "_" << fs::path(cfg.basis_path).stem().string() << "_"
     << fs::path(cfg.aux_basis_path).stem().string() << "_p"
     << (cfg.is_pure ? 1 : 0) << "_j" << (cfg.use_jit ? 1 : 0);
  // Functional and grid only enter the key when the atoms actually use them.
  // The HF default is functional-independent, so one cached atom serves every
  // functional the molecule might be run with.
  if (cfg.use_parent_functional)
    ss << "_f" << static_cast<int>(cfg.functional) << "_r" << cfg.radial_pts << "a"
       << cfg.angular_pts;
  else
    ss << "_hf";
  return ss.str();
}

constexpr uint32_t kCacheMagic = 0x53414432u;  // "SAD2"

bool try_load_cache(const std::string& path, int Z, AtomicGuessOrbitals& out) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return false;
  const std::streamsize total = in.tellg();
  in.seekg(0);

  uint32_t magic = 0;
  int32_t zz = 0, nelec = 0;
  uint64_t nao = 0, n_cols = 0;
  double energy = 0.0;
  const std::streamsize header = static_cast<std::streamsize>(
      sizeof(magic) + sizeof(zz) + sizeof(nelec) + sizeof(nao) + sizeof(n_cols) +
      sizeof(energy));
  if (total < header) return false;

  in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  in.read(reinterpret_cast<char*>(&zz), sizeof(zz));
  in.read(reinterpret_cast<char*>(&nelec), sizeof(nelec));
  in.read(reinterpret_cast<char*>(&nao), sizeof(nao));
  in.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));
  in.read(reinterpret_cast<char*>(&energy), sizeof(energy));
  if (!in || magic != kCacheMagic || zz != Z) return false;

  const std::streamsize expected =
      header + static_cast<std::streamsize>(nao * n_cols * sizeof(double));
  if (total != expected) return false;

  out.nao = nao;
  out.n_cols = n_cols;
  out.energy = energy;
  out.nelec = nelec;
  out.Cocc.resize(nao * n_cols);
  in.read(reinterpret_cast<char*>(out.Cocc.data()),
          static_cast<std::streamsize>(sizeof(double) * out.Cocc.size()));
  return in.good() || in.eof();
}

void write_cache(const std::string& path, int Z, const AtomicGuessOrbitals& d) {
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::create_directories(fs::path(path).parent_path(), ec);

  const std::string tmp_path = path + ".tmp" + std::to_string(static_cast<long>(getpid()));
  {
    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    if (!out) return;  // best-effort cache; missing write access isn't fatal
    const uint32_t magic = kCacheMagic;
    const int32_t zz = Z, nelec = d.nelec;
    const uint64_t nao = d.nao, n_cols = d.n_cols;
    const double energy = d.energy;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&zz), sizeof(zz));
    out.write(reinterpret_cast<const char*>(&nelec), sizeof(nelec));
    out.write(reinterpret_cast<const char*>(&nao), sizeof(nao));
    out.write(reinterpret_cast<const char*>(&n_cols), sizeof(n_cols));
    out.write(reinterpret_cast<const char*>(&energy), sizeof(energy));
    out.write(reinterpret_cast<const char*>(d.Cocc.data()),
              static_cast<std::streamsize>(sizeof(double) * d.Cocc.size()));
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
// The constrained atomic SCF.
// ---------------------------------------------------------------------------
AtomicGuessOrbitals compute_atomic_orbitals(CuESTContext& ctx, int Z,
                                            const SADGuessConfig& cfg,
                                            bool& converged) {
  converged = false;
  Molecule mol;
  mol.add_atom_bohr(Z, 0.0, 0.0, 0.0);
  mol.set_charge(0);

  BasisBuilder atom_basis(ctx, mol, cfg.is_pure ? 1 : 0);
  atom_basis.build_from_json(cfg.basis_path);
  const int nao = static_cast<int>(atom_basis.nao());

  AtomicGuessOrbitals result;
  result.nao = static_cast<uint64_t>(nao);

  ECPBuilder ecp_builder(ctx, mol);
  ecp_builder.build_from_json(cfg.basis_path);
  const int n_ecp = static_cast<int>(ecp_builder.total_ecp_electrons());

  const int nelec = mol.nelec(n_ecp);
  result.nelec = nelec;
  if (nelec <= 0) {
    converged = true;  // fully ECP-replaced core, or a ghost centre
    return result;
  }

  AuxBasis aux_basis(ctx, mol, /*is_pure=*/1);
  aux_basis.build_from_json(cfg.aux_basis_path);

  std::unique_ptr<ECPIntegrals> ecp_int;
  auto xyz_h = mol.xyz_host();
  if (ecp_builder.has_ecp())
    ecp_int = std::make_unique<ECPIntegrals>(
        ctx, atom_basis.basis(), xyz_h.data(), ecp_builder.num_active_atoms(),
        ecp_builder.ecp_indices().data(), ecp_builder.ecp_atoms().data());

  // XC only in the opt-in functional-consistent mode; the HF default needs no
  // grid at all, which is most of why it is the default.
  std::unique_ptr<GridBuilder> grid_builder;
  std::unique_ptr<XCBuilder> xc;
  MolecularGridHandle mol_grid;
  double ex_frac = 1.0, lrc_frac = 0.0, lrc_omega = 0.0;
  if (cfg.use_parent_functional) {
    grid_builder =
        std::make_unique<GridBuilder>(ctx, mol, cfg.radial_pts, cfg.angular_pts);
    mol_grid = grid_builder->build();
    xc = std::make_unique<XCBuilder>(ctx, atom_basis.basis(), mol_grid, cfg.functional);
    hybrid_dfjk_fractions(cfg.functional, *xc, ex_frac, lrc_frac, lrc_omega);
  }

  DFJKBuilder dfjk(ctx, atom_basis.basis(), aux_basis.basis(), xyz_h.data(),
                   mol.natom(), ex_frac, lrc_frac, lrc_omega, cfg.use_jit,
                   cfg.fitting_cutoff, cfg.fitting_relative_conditioning,
                   cfg.fitting_algorithm);

  // --- one-electron integrals ---
  const size_t n2 = static_cast<size_t>(nao) * nao;
  DeviceArray<double> d_S, d_T, d_V, d_ECP, d_D, d_J, d_K, d_Cocc, d_Vxc;
  d_S.alloc(n2 * sizeof(double));
  d_T.alloc(n2 * sizeof(double));
  d_V.alloc(n2 * sizeof(double));
  d_D.alloc(n2 * sizeof(double));
  d_J.alloc(n2 * sizeof(double));
  d_K.alloc(n2 * sizeof(double));

  DeviceArray<double> d_xyz, d_charges;
  auto chg_h = mol.charges_host(ecp_builder.ecp_electrons_per_atom());
  d_xyz.alloc(xyz_h.size() * sizeof(double));
  d_charges.alloc(chg_h.size() * sizeof(double));
  CUDA_CHECK(cudaMemcpy(d_xyz, xyz_h.data(), xyz_h.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_charges, chg_h.data(), chg_h.size() * sizeof(double),
                        cudaMemcpyHostToDevice));

  {
    const OwnedAOPairList& pl = atom_basis.pair_list();
    OneElectronIntegrals oe(ctx, atom_basis.basis(), pl.get(), cfg.use_jit);
    oe.compute_kinetic(d_T);
    oe.compute_potential(d_V, mol.natom(), d_xyz, d_charges);
    oe.compute_overlap(d_S);
  }

  std::vector<double> S(n2), Hcore(n2), Tmat(n2);
  CUDA_CHECK(cudaMemcpy(S.data(), d_S.get(), n2 * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(Tmat.data(), d_T.get(), n2 * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(Hcore.data(), d_V.get(), n2 * sizeof(double), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n2; i++) Hcore[i] += Tmat[i];
  if (ecp_int) {
    d_ECP.alloc(n2 * sizeof(double));
    ecp_int->compute(d_ECP);
    std::vector<double> E(n2);
    CUDA_CHECK(cudaMemcpy(E.data(), d_ECP.get(), n2 * sizeof(double),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n2; i++) Hcore[i] += E[i];
  }

  // --- symmetry projector, with the checks that catch a convention mismatch ---
  const auto shells =
      make_atom_shells(atom_basis.shell_angular_momenta(), cfg.is_pure);
  if (shells.nao != nao)
    throw std::runtime_error("SAD guess: shell layout disagrees with basis nao");
  SphericalProjector proj(shells);

  if (!cfg.is_pure) {
    const double rot_dev = verify_rotation_rep(S, shells);
    if (rot_dev > 1e-8)
      throw std::runtime_error(
          "SAD guess: Cartesian rotation representation does not preserve the "
          "overlap (max deviation " + std::to_string(rot_dev) +
          ") — cuEST's Cartesian AO ordering or normalization is not what "
          "atomic_symmetry.cpp assumes");
  }
  {
    // The overlap of a free atom is spherically symmetric by construction, so
    // the projector must leave it exactly alone. This validates the projector
    // against this specific basis, not just in the abstract.
    std::vector<double> PS = S;
    proj.apply(PS, RotationLaw::Operator);
    double dev = 0.0;
    for (size_t i = 0; i < n2; i++) dev = std::max(dev, std::abs(PS[i] - S[i]));
    if (dev > 1e-8)
      throw std::runtime_error(
          "SAD guess: commutant projector alters the overlap matrix (max "
          "deviation " + std::to_string(dev) + ") — projector is not correct "
          "for this basis");
  }

  int nmo = 0;
  const std::vector<double> X = canonical_orthogonalizer(S, nao, nmo);

  // --- SCF ---
  std::vector<double> D(n2, 0.0);      // alpha density
  std::vector<double> Dt;              // alpha density in the orthonormal basis
  std::vector<double> F = Hcore;
  proj.apply(F, RotationLaw::Operator);

  std::vector<double> Cocc_w;
  uint64_t ncols = 0;
  double energy = 0.0, e_prev = 0.0;
  HostDIIS diis(8);
  const bool one_electron = (nelec == 1);

  const int kMaxIter = 200;
  for (int iter = 1; iter <= kMaxIter; iter++) {
    std::vector<double> Ft = congruence(X, nao, nmo, F);

    // Commutator error in the orthonormal basis: no overlap factors needed.
    if (!Dt.empty()) {
      std::vector<double> err(static_cast<size_t>(nmo) * nmo, 0.0);
      for (int i = 0; i < nmo; i++)
        for (int j = 0; j < nmo; j++) {
          double acc = 0.0;
          for (int k = 0; k < nmo; k++)
            acc += Ft[static_cast<size_t>(i) * nmo + k] * Dt[static_cast<size_t>(k) * nmo + j] -
                   Dt[static_cast<size_t>(i) * nmo + k] * Ft[static_cast<size_t>(k) * nmo + j];
          err[static_cast<size_t>(i) * nmo + j] = acc;
        }
      const double err_norm =
          std::sqrt(frobenius_dot(err, err) / static_cast<double>(nmo * nmo));

      if (iter > 1 && err_norm < 1e-9 && std::abs(energy - e_prev) < 1e-11) {
        converged = true;
        break;
      }
      diis.push(Ft, std::move(err));
      diis.extrapolate(Ft);
    }

    std::vector<double> e, Cp;
    jacobi_symmetric(Ft, nmo, e, Cp);

    // Exactly (2l+1)-fold degenerate by construction, so grouping is reading
    // off machine-precision-identical eigenvalues rather than guessing at
    // near-degeneracy. Scale the tolerance with the orbital energy so deep
    // cores (~-1000 Ha for heavy elements) group as reliably as valence ones.
    const double tol = 1e-8 * std::max(1.0, std::abs(e.front()));
    const auto manifolds = occupied_manifolds(e, Z, n_ecp, tol);

    // C = X Cp (nao x nmo, column-major)
    std::vector<double> C(static_cast<size_t>(nao) * nmo, 0.0);
    for (int c = 0; c < nmo; c++)
      for (int r = 0; r < nao; r++) {
        double acc = 0.0;
        for (int k = 0; k < nmo; k++)
          acc += X[static_cast<size_t>(k) * nao + r] * Cp[static_cast<size_t>(c) * nmo + k];
        C[static_cast<size_t>(c) * nao + r] = acc;
      }

    // Weighted occupied columns: scale by sqrt(occ/2) so that Cocc Cocc^T is
    // the *alpha* density, matching SCFSolver's D_alpha convention, and so the
    // columns remain directly usable as Cocc for a DF exchange build.
    Cocc_w.clear();
    std::vector<double> Cocc_p;  // same, in the orthonormal basis, for Dt
    ncols = 0;
    for (const auto& m : manifolds) {
      if (m.occ <= 0.0) continue;
      const double w = std::sqrt(m.occ * 0.5);
      for (int k = 0; k < m.size; k++) {
        const int col = m.lo + k;
        for (int r = 0; r < nao; r++)
          Cocc_w.push_back(w * C[static_cast<size_t>(col) * nao + r]);
        for (int r = 0; r < nmo; r++)
          Cocc_p.push_back(w * Cp[static_cast<size_t>(col) * nmo + r]);
        ncols++;
      }
    }

    auto gram = [](const std::vector<double>& M, int n, uint64_t cols) {
      std::vector<double> G(static_cast<size_t>(n) * n, 0.0);
      for (uint64_t c = 0; c < cols; c++)
        for (int i = 0; i < n; i++) {
          const double mi = M[c * n + i];
          if (mi == 0.0) continue;
          for (int j = 0; j < n; j++)
            G[static_cast<size_t>(i) * n + j] += mi * M[c * n + j];
        }
      return G;
    };
    D = gram(Cocc_w, nao, ncols);
    Dt = gram(Cocc_p, nmo, ncols);
    // Equal fractional occupancy of whole manifolds already gives a spherical
    // density; this removes only the accumulated round-off.
    proj.apply(D, RotationLaw::Density);

    // --- Fock build: F = Hcore + 2 J[D_alpha] - K[D_alpha] (+ Vxc) ---
    //
    // A single electron is a special case: the restricted fractional-occupancy
    // Fock does not cancel its self-interaction (2J - K leaves 0.5 J[CC^T] for
    // one half-filled orbital), so the two-electron terms have to be dropped
    // outright. PySCF does the same via its separate AtomHF1e path. This
    // matters for more than hydrogen — an ECP that leaves one valence electron
    // (Na with a 10-electron core, say) lands here too.
    if (one_electron) {
      F = Hcore;
      proj.apply(F, RotationLaw::Operator);
      e_prev = energy;
      energy = 2.0 * frobenius_dot(D, Hcore);
      if (iter > 1) converged = true;
      if (converged) break;
      continue;
    }

    CUDA_CHECK(cudaMemcpy(d_D, D.data(), n2 * sizeof(double), cudaMemcpyHostToDevice));
    dfjk.compute_J(d_D, d_J);

    const size_t cocc_bytes = static_cast<size_t>(nao) * std::max<uint64_t>(ncols, 1) *
                              sizeof(double);
    d_Cocc.ensure(cocc_bytes);
    CUDA_CHECK(cudaMemset(d_Cocc, 0, cocc_bytes));
    if (ncols > 0)
      CUDA_CHECK(cudaMemcpy(d_Cocc, Cocc_w.data(), Cocc_w.size() * sizeof(double),
                            cudaMemcpyHostToDevice));

    const bool want_k = !xc || xc->is_hybrid() || xc->is_hf();
    std::vector<double> Jm(n2), Km(n2, 0.0), Vxc;
    CUDA_CHECK(cudaMemcpy(Jm.data(), d_J.get(), n2 * sizeof(double), cudaMemcpyDeviceToHost));
    if (want_k && ncols > 0) {
      dfjk.compute_K(ncols, d_Cocc, d_K);
      CUDA_CHECK(cudaMemcpy(Km.data(), d_K.get(), n2 * sizeof(double),
                            cudaMemcpyDeviceToHost));
    }

    double e_xc = 0.0;
    if (xc && !xc->is_hf() && ncols > 0) {
      d_Vxc.ensure(n2 * sizeof(double));
      xc->compute_vxc_rks(ncols, d_Cocc, &e_xc, d_Vxc);
      Vxc.resize(n2);
      CUDA_CHECK(cudaMemcpy(Vxc.data(), d_Vxc.get(), n2 * sizeof(double),
                            cudaMemcpyDeviceToHost));
    }

    F = Hcore;
    for (size_t i = 0; i < n2; i++) F[i] += 2.0 * Jm[i] - Km[i];
    if (!Vxc.empty())
      for (size_t i = 0; i < n2; i++) F[i] += Vxc[i];
    proj.apply(F, RotationLaw::Operator);

    // Matches SCFSolver's RKS energy convention (see build_fock_rks): the
    // exchange fraction is already baked into the DF plan, so K needs no
    // further scaling here.
    e_prev = energy;
    energy = 2.0 * frobenius_dot(D, Hcore) + 2.0 * frobenius_dot(D, Jm) -
             frobenius_dot(D, Km) + e_xc;
  }

  if (!converged)
    std::cerr << "Warning: atomic SAD reference for Z=" << Z
              << " did not converge; using best-effort orbitals for the guess.\n";

  result.Cocc = std::move(Cocc_w);
  result.n_cols = ncols;
  result.energy = energy;
  return result;
}

}  // namespace

AtomicGuessOrbitals get_atomic_guess_orbitals(CuESTContext& ctx, int Z,
                                             const SADGuessConfig& cfg) {
  const std::string path = cache_dir() + "/" + cache_key(Z, cfg) + ".sad";

  AtomicGuessOrbitals result;
  if (try_load_cache(path, Z, result)) return result;

  bool converged = false;
  result = compute_atomic_orbitals(ctx, Z, cfg, converged);
  if (converged) write_cache(path, Z, result);
  return result;
}

}  // namespace cuest
