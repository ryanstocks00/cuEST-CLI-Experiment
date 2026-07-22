/**
 * @file scf.cpp — RKS / UKS SCF with device DIIS (cuSOLVER dsygvd).
 */
#include "scf.hpp"
#include "cuest_wrapper/nvtx.hpp"
#include "sac_guess.hpp"
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace cuest {

SCFSolver::SCFSolver(CuESTContext& ctx, BasisBuilder& basis,
                     DFJKBuilder& dfjk, XCBuilder* xc,
                     ECPBuilder* ecp_builder, ECPIntegrals* ecp_int,
                     const Molecule& mol, SCFParams params,
                     const SACGuessConfig* sac_config)
    : ctx_(ctx), basis_(basis), dfjk_(dfjk), xc_(xc),
      ecp_builder_(ecp_builder), ecp_int_(ecp_int),
      mol_(mol), params_(params), sac_config_(sac_config) {
  (void)ecp_builder_;
  if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cublasCreate failed");
  if (cusolverDnCreate(&cusolver_) != CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error("cusolverDnCreate failed");

  nao_ = ctx_.query_nao(basis_.basis_handle());
  const int n_ecp = ecp_builder_
                        ? static_cast<int>(ecp_builder_->total_ecp_electrons())
                        : 0;
  const std::vector<int>* ecp_cores =
      ecp_builder_ ? &ecp_builder_->ecp_electrons_per_atom() : nullptr;
  nelec_ = mol_.nelec(n_ecp);
  if (nelec_ < 0)
    throw std::runtime_error("Negative electron count after charge/ECP");

  nocc_a_ = static_cast<uint64_t>(mol_.nalpha(n_ecp));
  nocc_b_ = static_cast<uint64_t>(mol_.nbeta(n_ecp));
  if (static_cast<int>(nocc_a_ + nocc_b_) != nelec_)
    throw std::runtime_error("nalpha+nbeta != nelec — check multiplicity/charge");
  if (nocc_a_ < nocc_b_)
    throw std::runtime_error("nalpha < nbeta — multiplicity inconsistent with nelec");

  uks_ = (mol_.multiplicity() != 1) || (nocc_a_ != nocc_b_);
  nocc_ = uks_ ? nocc_a_ : static_cast<uint64_t>(nelec_ / 2);
  if (!uks_ && (nelec_ % 2 != 0))
    throw std::runtime_error("Odd electron count requires multiplicity > 1 (UKS)");

  e_nuc_ = ecp_cores ? mol_.nuclear_repulsion(*ecp_cores)
                     : mol_.nuclear_repulsion();

  size_t N2 = nao_ * nao_;
  size_t n2_bytes = N2 * sizeof(double);
  d_Hcore_.alloc(n2_bytes);
  d_S_.alloc(n2_bytes);
  d_T_.alloc(n2_bytes);
  d_V_.alloc(n2_bytes);
  d_ECP_.alloc(n2_bytes);
  d_Fock_a_.alloc(n2_bytes);
  d_Fock_b_.alloc(n2_bytes);
  d_D_a_.alloc(n2_bytes);
  d_D_b_.alloc(n2_bytes);
  d_D_tot_.alloc(n2_bytes);
  d_D_old_a_.alloc(n2_bytes);
  d_D_old_b_.alloc(n2_bytes);
  d_C_a_.alloc(n2_bytes);
  d_C_b_.alloc(n2_bytes);
  d_J_.alloc(n2_bytes);
  d_K_a_.alloc(n2_bytes);
  d_K_b_.alloc(n2_bytes);
  d_Vxc_a_.alloc(n2_bytes);
  d_Vxc_b_.alloc(n2_bytes);
  d_Vnlc_a_.alloc(n2_bytes);
  d_eigvals_.alloc(nao_ * sizeof(double));
  d_Fwork_.alloc(n2_bytes);
  d_Swork_.alloc(n2_bytes);
  d_rms_scratch_.alloc(n2_bytes);
  d_info_.alloc(sizeof(int));

  // Occupied MO packs (pad empty UKS channel to 1 column of zeros).
  const uint64_t nocc_a_pad = std::max<uint64_t>(uks_ ? nocc_a_ : nocc_, 1);
  const uint64_t nocc_b_pad = std::max<uint64_t>(nocc_b_, 1);
  d_Cocc_a_.alloc(nao_ * nocc_a_pad * sizeof(double));
  d_Cocc_b_.alloc(nao_ * nocc_b_pad * sizeof(double));

  diis_a_.init(params_.diis_max_space, static_cast<int>(nao_));
  diis_b_.init(params_.diis_max_space, static_cast<int>(nao_));

  auto xyz_h = mol_.xyz_host();
  auto chg_h = ecp_cores ? mol_.charges_host(*ecp_cores) : mol_.charges_host();
  d_xyz_.alloc(xyz_h.size() * sizeof(double));
  d_charges_.alloc(chg_h.size() * sizeof(double));
  CUDA_CHECK(cudaMemcpy(d_xyz_, xyz_h.data(), xyz_h.size()*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_charges_, chg_h.data(), chg_h.size()*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D_a_, 0, n2_bytes));
  CUDA_CHECK(cudaMemset(d_D_b_, 0, n2_bytes));
  CUDA_CHECK(cudaMemset(d_D_tot_, 0, n2_bytes));
  CUDA_CHECK(cudaMemset(d_ECP_, 0, n2_bytes));
}

SCFSolver::~SCFSolver() {
  if (cublas_) cublasDestroy(cublas_);
  if (cusolver_) cusolverDnDestroy(cusolver_);
}

double SCFSolver::frobenius_dot(const double* d_A, const double* d_B, int n2) {
  double result = 0.0;
  CUBLAS_CHECK(cublasDdot(cublas_, n2, d_A, 1, d_B, 1, &result));
  return result;
}

double SCFSolver::compute_rms_delta_device(const double* d_D,
                                            const double* d_D_old, int n2) {
  // scratch = D - D_old
  CUBLAS_CHECK(cublasDcopy(cublas_, n2, d_D, 1, d_rms_scratch_, 1));
  double minus_one = -1.0;
  CUBLAS_CHECK(cublasDaxpy(cublas_, n2, &minus_one, d_D_old, 1, d_rms_scratch_, 1));
  double ss = 0.0;
  CUBLAS_CHECK(cublasDdot(cublas_, n2, d_rms_scratch_, 1, d_rms_scratch_, 1, &ss));
  return std::sqrt(ss / static_cast<double>(n2));
}

void SCFSolver::build_core_hamiltonian() {
  NvtxRange range("build_core_hamiltonian");
  const OwnedAOPairList& pair_list = basis_.pair_list();
  OneElectronIntegrals oe(ctx_, basis_.basis(), pair_list.get(), params_.use_jit);
  oe.compute_kinetic(d_T_);
  oe.compute_potential(d_V_, mol_.natom(), d_xyz_, d_charges_);
  oe.compute_overlap(d_S_);

  double one = 1.0;
  cublasDcopy(cublas_, nao_*nao_, d_T_, 1, d_Hcore_, 1);
  cublasDaxpy(cublas_, nao_*nao_, &one, d_V_, 1, d_Hcore_, 1);

  if (ecp_int_) {
    ecp_int_->compute(d_ECP_);
    cublasDaxpy(cublas_, nao_*nao_, &one, d_ECP_, 1, d_Hcore_, 1);
  }
  if (params_.verbose)
    std::cout << "Computing one-electron integrals...\n";
}

void SCFSolver::build_density(DeviceArray<double>& d_C, uint64_t nocc,
                              DeviceArray<double>& d_D) {
  NvtxRange range("build_density");
  int N = static_cast<int>(nao_);
  int Nocc = static_cast<int>(nocc);
  double one = 1.0, zero = 0.0;
  if (Nocc <= 0) {
    CUDA_CHECK(cudaMemset(d_D, 0, nao_ * nao_ * sizeof(double)));
    return;
  }
  cublasDgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
              N, N, Nocc, &one,
              d_C, N, d_C, N,
              &zero, d_D, N);
}

void SCFSolver::build_density_auto(DeviceArray<double>& d_C, uint64_t nocc,
                                   const std::vector<double>& mo_energies,
                                   DeviceArray<double>& d_D, FrontierGroup& group) {
  if (!params_.spherical_average_occupation) {
    build_density(d_C, nocc, d_D);
    return;
  }
  NvtxRange range("build_density_spherical");
  const int N = static_cast<int>(nao_);
  if (nocc == 0) {
    CUDA_CHECK(cudaMemset(d_D, 0, static_cast<size_t>(N) * N * sizeof(double)));
    return;
  }
  if (!group.ready) {
    // Only trustworthy on the first call, from a bare-Hcore diagonalization
    // (exactly rotationally symmetric for an isolated atom) — locking it in
    // here avoids re-deriving it from a potentially already-split (and
    // therefore self-reinforcing) eigenspectrum on later iterations.
    auto [lo, hi] = degenerate_frontier_group(mo_energies, static_cast<int>(nocc));
    group.lo = lo;
    group.hi = hi;
    group.ready = true;
  }
  std::vector<double> C_h(static_cast<size_t>(N) * N);
  CUDA_CHECK(cudaMemcpy(C_h.data(), d_C, C_h.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
  std::vector<double> D_h =
      spherically_averaged_density(C_h, group.lo, group.hi, nao_, nocc);
  CUDA_CHECK(cudaMemcpy(d_D, D_h.data(), D_h.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
}

void SCFSolver::form_total_density() {
  double one = 1.0;
  cublasDcopy(cublas_, nao_*nao_, d_D_a_, 1, d_D_tot_, 1);
  if (uks_)
    cublasDaxpy(cublas_, nao_*nao_, &one, d_D_b_, 1, d_D_tot_, 1);
  else {
    double two = 2.0;
    cublasDscal(cublas_, nao_*nao_, &two, d_D_tot_, 1);
  }
}

void SCFSolver::build_fock_rks() {
  NvtxRange range("build_fock_rks");
  dfjk_.compute_J(d_D_a_, d_J_);

  double two = 2.0;
  cublasDcopy(cublas_, nao_*nao_, d_Hcore_, 1, d_Fock_a_, 1);
  cublasDaxpy(cublas_, nao_*nao_, &two, d_J_, 1, d_Fock_a_, 1);

  const uint64_t ncols = use_sac_cocc_ ? sac_ncols_a_ : nocc_;
  if (!use_sac_cocc_) {
    CUDA_CHECK(cudaMemcpy(d_Cocc_a_, d_C_a_, nao_ * nocc_ * sizeof(double),
                          cudaMemcpyDeviceToDevice));
  }

  if (xc_ && (xc_->is_hybrid() || xc_->is_hf()) && ncols > 0) {
    const double k_factor = -1.0;
    dfjk_.compute_K(ncols, d_Cocc_a_, d_K_a_);
    cublasDaxpy(cublas_, nao_*nao_, &k_factor, d_K_a_, 1, d_Fock_a_, 1);
  }

  if (xc_ && !xc_->is_hf() && ncols > 0) {
    double exc_val = 0.0;
    xc_->compute_vxc_rks(ncols, d_Cocc_a_, &exc_val, d_Vxc_a_);
    e_xc_ = exc_val;
    double one = 1.0;
    cublasDaxpy(cublas_, nao_*nao_, &one, d_Vxc_a_, 1, d_Fock_a_, 1);

    if (xc_->is_vv10()) {
      double enlc_val = 0.0;
      xc_->compute_vv10_rks(ncols, d_Cocc_a_, &enlc_val, d_Vnlc_a_);
      e_xc_ += enlc_val;
      cublasDaxpy(cublas_, nao_*nao_, &one, d_Vnlc_a_, 1, d_Fock_a_, 1);
    }
  } else {
    e_xc_ = 0.0;
  }
}

void SCFSolver::build_fock_uks() {
  NvtxRange range("build_fock_uks");
  {
    double one = 1.0;
    cublasDcopy(cublas_, nao_*nao_, d_D_a_, 1, d_D_tot_, 1);
    cublasDaxpy(cublas_, nao_*nao_, &one, d_D_b_, 1, d_D_tot_, 1);
  }
  dfjk_.compute_J(d_D_tot_, d_J_);

  cublasDcopy(cublas_, nao_*nao_, d_Hcore_, 1, d_Fock_a_, 1);
  cublasDcopy(cublas_, nao_*nao_, d_Hcore_, 1, d_Fock_b_, 1);
  double one = 1.0;
  cublasDaxpy(cublas_, nao_*nao_, &one, d_J_, 1, d_Fock_a_, 1);
  cublasDaxpy(cublas_, nao_*nao_, &one, d_J_, 1, d_Fock_b_, 1);

  const uint64_t ncols_a = use_sac_cocc_ ? sac_ncols_a_ : nocc_a_;
  const uint64_t ncols_b = use_sac_cocc_ ? sac_ncols_b_ : nocc_b_;
  const uint64_t nocc_a_pad = std::max<uint64_t>(ncols_a, 1);
  const uint64_t nocc_b_pad = std::max<uint64_t>(ncols_b, 1);

  if (!use_sac_cocc_) {
    CUDA_CHECK(cudaMemset(d_Cocc_a_, 0, nao_ * nocc_a_pad * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_Cocc_b_, 0, nao_ * nocc_b_pad * sizeof(double)));
    if (nocc_a_ > 0)
      CUDA_CHECK(cudaMemcpy(d_Cocc_a_, d_C_a_, nao_ * nocc_a_ * sizeof(double),
                            cudaMemcpyDeviceToDevice));
    if (nocc_b_ > 0)
      CUDA_CHECK(cudaMemcpy(d_Cocc_b_, d_C_b_, nao_ * nocc_b_ * sizeof(double),
                            cudaMemcpyDeviceToDevice));
  }

  if (xc_ && (xc_->is_hybrid() || xc_->is_hf())) {
    const double k_factor = -1.0;
    if (ncols_a > 0) {
      dfjk_.compute_K(ncols_a, d_Cocc_a_, d_K_a_);
      cublasDaxpy(cublas_, nao_*nao_, &k_factor, d_K_a_, 1, d_Fock_a_, 1);
    } else {
      CUDA_CHECK(cudaMemset(d_K_a_, 0, nao_ * nao_ * sizeof(double)));
    }
    if (ncols_b > 0) {
      dfjk_.compute_K(ncols_b, d_Cocc_b_, d_K_b_);
      cublasDaxpy(cublas_, nao_*nao_, &k_factor, d_K_b_, 1, d_Fock_b_, 1);
    } else {
      CUDA_CHECK(cudaMemset(d_K_b_, 0, nao_ * nao_ * sizeof(double)));
    }
  }

  if (xc_ && !xc_->is_hf()) {
    double exc_val = 0.0;
    xc_->compute_vxc_uks(nocc_a_pad, nocc_b_pad, d_Cocc_a_, d_Cocc_b_,
                         &exc_val, d_Vxc_a_, d_Vxc_b_);
    e_xc_ = exc_val;
    cublasDaxpy(cublas_, nao_*nao_, &one, d_Vxc_a_, 1, d_Fock_a_, 1);
    cublasDaxpy(cublas_, nao_*nao_, &one, d_Vxc_b_, 1, d_Fock_b_, 1);

    if (xc_->is_vv10()) {
      double enlc_val = 0.0;
      xc_->compute_vv10_uks(nocc_a_pad, nocc_b_pad, d_Cocc_a_, d_Cocc_b_,
                            &enlc_val, d_Vnlc_a_);
      e_xc_ += enlc_val;
      // VV10 potential is shared by both spin channels (functional of total density only).
      cublasDaxpy(cublas_, nao_*nao_, &one, d_Vnlc_a_, 1, d_Fock_a_, 1);
      cublasDaxpy(cublas_, nao_*nao_, &one, d_Vnlc_a_, 1, d_Fock_b_, 1);
    }
  } else if (xc_ && xc_->is_hf()) {
    e_xc_ = 0.0;
  }
}

void SCFSolver::diagonalize_fock(DeviceArray<double>& d_Fock,
                                 DeviceArray<double>& d_C,
                                 std::vector<double>& mo_energies) {
  NvtxRange range("diagonalize_fock");
  int N = static_cast<int>(nao_);
  int lda = N;

  cublasDcopy(cublas_, N*N, d_Fock, 1, d_Fwork_, 1);
  cublasDcopy(cublas_, N*N, d_S_, 1, d_Swork_, 1);

  int lwork = 0;
  cusolverDnDsygvd_bufferSize(cusolver_, CUSOLVER_EIG_TYPE_1,
      CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      N, d_Fwork_, lda, d_Swork_, lda,
      d_eigvals_, &lwork);

  if (lwork > syev_lwork_) {
    d_syev_work_.alloc(static_cast<size_t>(lwork) * sizeof(double));
    syev_lwork_ = lwork;
  }

  cusolverStatus_t st = cusolverDnDsygvd(cusolver_, CUSOLVER_EIG_TYPE_1,
      CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      N, d_Fwork_, lda, d_Swork_, lda,
      d_eigvals_, d_syev_work_, syev_lwork_, d_info_);

  int info_h = 0;
  CUDA_CHECK(cudaMemcpy(&info_h, d_info_, sizeof(int), cudaMemcpyDeviceToHost));

  if (st != CUSOLVER_STATUS_SUCCESS || info_h != 0) {
    throw std::runtime_error(
        "cuSOLVER dsygvd failed (status=" + std::to_string(static_cast<int>(st)) +
        ", info=" + std::to_string(info_h) + ")");
  }

  cublasDcopy(cublas_, N*N, d_Fwork_, 1, d_C, 1);
  mo_energies.resize(static_cast<size_t>(N));
  CUDA_CHECK(cudaMemcpy(mo_energies.data(), d_eigvals_,
             static_cast<size_t>(N) * sizeof(double), cudaMemcpyDeviceToHost));
}

void SCFSolver::break_beta_symmetry() {
  // Originally gated to nocc_a_==nocc_b_ (the "force an open-shell solution
  // out of an artificially closed-shell guess" trick). But a *naturally*
  // open-shell radical can have this exact same problem at its β HOMO/LUMO
  // boundary: e.g. OH's unpaired electron sits in one of two orbitals that
  // are genuinely degenerate by symmetry (a non-bonding π-like pair
  // perpendicular to the bond axis) — which one is arbitrary. A spherically
  // -averaged atomic guess (see sac_guess.hpp) gives that pair *exactly*
  // equal weight, so the molecular guess starts precisely on the knife-edge
  // between the two equivalent solutions, and the SCF has to let floating-
  // point noise slowly (and irreproducibly) nudge it off that ridge —
  // dozens to hundreds of extra iterations, with wildly run-to-run-variable
  // counts. Applying the same fixed-angle β HOMO/LUMO mix here regardless of
  // nocc_a_ vs nocc_b_ breaks that tie deterministically up front instead.
  if (nocc_b_ == 0 || nocc_b_ >= nao_) return;
  double theta = params_.break_symmetry;
  if (std::abs(theta) < 1e-12) return;

  std::vector<double> C(nao_ * nao_);
  CUDA_CHECK(cudaMemcpy(C.data(), d_C_b_, nao_*nao_*sizeof(double),
                        cudaMemcpyDeviceToHost));

  const size_t homo = nocc_b_ - 1;
  const size_t lumo = nocc_b_;
  const double c = std::cos(theta), s = std::sin(theta);
  for (size_t i = 0; i < nao_; i++) {
    double cho = C[i + homo * nao_];
    double clu = C[i + lumo * nao_];
    C[i + homo * nao_] = c * cho + s * clu;
    C[i + lumo * nao_] = -s * cho + c * clu;
  }
  CUDA_CHECK(cudaMemcpy(d_C_b_, C.data(), nao_*nao_*sizeof(double),
                        cudaMemcpyHostToDevice));
  build_density(d_C_b_, nocc_b_, d_D_b_);
  if (params_.print_level >= 2)
    std::cout << "  Symmetry breaking: β HOMO/LUMO mix θ=" << theta << " rad\n";
}

namespace {
// D = C C^T for an N x ncols column-major host matrix (D is N x N).
std::vector<double> gram_from_columns(const std::vector<double>& C, int N, uint64_t ncols) {
  std::vector<double> D(static_cast<size_t>(N) * N, 0.0);
  for (uint64_t k = 0; k < ncols; k++)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        D[static_cast<size_t>(i) * N + j] +=
            C[static_cast<size_t>(i) + k * N] * C[static_cast<size_t>(j) + k * N];
  return D;
}
}  // namespace

void SCFSolver::assemble_sac_guess() {
  const auto& offsets = basis_.ao_offsets();
  const int N = static_cast<int>(nao_);
  if (offsets.size() != mol_.natom() + 1)
    throw std::runtime_error("SAC guess: BasisBuilder AO offsets unavailable");

  // Fetch (or compute+cache) each distinct element's converged atomic
  // reference once, as weighted occupied orbitals (not a density — see
  // sac_guess.hpp), then embed those columns at that atom's real AO offset.
  std::unordered_map<int, AtomicGuessOrbitals> per_element;
  auto get_atom = [&](int Z) -> const AtomicGuessOrbitals& {
    auto it = per_element.find(Z);
    if (it == per_element.end())
      it = per_element.emplace(Z, get_atomic_guess_orbitals(ctx_, Z, *sac_config_)).first;
    return it->second;
  };

  uint64_t total_cols_a = 0, total_cols_b = 0;
  for (size_t a = 0; a < mol_.natom(); a++) {
    const AtomicGuessOrbitals& ad = get_atom(mol_.atom(a).atomic_number);
    total_cols_a += ad.n_cols_alpha;
    total_cols_b += ad.n_cols_beta;
  }

  std::vector<double> Ca(static_cast<size_t>(N) * std::max<uint64_t>(total_cols_a, 1), 0.0);
  std::vector<double> Cb(static_cast<size_t>(N) * std::max<uint64_t>(total_cols_b, 1), 0.0);
  uint64_t col_a = 0, col_b = 0;
  for (size_t a = 0; a < mol_.natom(); a++) {
    const int Z = mol_.atom(a).atomic_number;
    const AtomicGuessOrbitals& ad = get_atom(Z);
    const uint64_t start = offsets[a];
    const uint64_t count = offsets[a + 1] - offsets[a];
    if (ad.nao != count)
      throw std::runtime_error(
          "SAC guess: atomic AO count (" + std::to_string(ad.nao) +
          ") != molecular block size (" + std::to_string(count) +
          ") for atom " + std::to_string(a) + " (Z=" + std::to_string(Z) + ")");

    for (uint64_t k = 0; k < ad.n_cols_alpha; k++)
      for (uint64_t i = 0; i < count; i++)
        Ca[(start + i) + (col_a + k) * static_cast<uint64_t>(N)] =
            ad.Cocc_alpha[i + k * count];
    col_a += ad.n_cols_alpha;

    for (uint64_t k = 0; k < ad.n_cols_beta; k++)
      for (uint64_t i = 0; i < count; i++)
        Cb[(start + i) + (col_b + k) * static_cast<uint64_t>(N)] =
            ad.Cocc_beta[i + k * count];
    col_b += ad.n_cols_beta;
  }

  std::vector<double> S_h(static_cast<size_t>(N) * N);
  CUDA_CHECK(cudaMemcpy(S_h.data(), d_S_, S_h.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
  auto trace_ds = [&](const std::vector<double>& D) {
    double tr = 0.0;
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        tr += D[static_cast<size_t>(i) * N + j] * S_h[static_cast<size_t>(j) * N + i];
    return tr;
  };

  // Rescale the superposed orbitals to the molecule's actual electron count.
  // Scaling every column by sqrt(scale) scales D = C C^T by exactly `scale`.
  std::vector<double> Cocc_a, Cocc_b;
  uint64_t ncols_a = 0, ncols_b = 0;
  if (uks_) {
    const double scale_a =
        static_cast<double>(nocc_a_) / std::max(trace_ds(gram_from_columns(Ca, N, total_cols_a)), 1e-10);
    const double target_b = static_cast<double>(nocc_b_);
    const double scale_b = target_b > 0.0
        ? target_b / std::max(trace_ds(gram_from_columns(Cb, N, total_cols_b)), 1e-10)
        : 0.0;
    const double s_a = std::sqrt(scale_a), s_b = std::sqrt(scale_b);
    Cocc_a = Ca; for (auto& c : Cocc_a) c *= s_a;
    Cocc_b = Cb; for (auto& c : Cocc_b) c *= s_b;
    ncols_a = total_cols_a;
    ncols_b = target_b > 0.0 ? total_cols_b : 0;
  } else {
    // Restricted: fold both spin channels' atomic columns into one set
    // (concatenating both and halving reproduces 0.5*(D_a+D_b)), then rescale.
    // Size Cr from Ca/Cb's actual (already padded-to-≥1-column) buffer
    // sizes, not total_cols_a/total_cols_b directly: an element like H whose
    // isolated atomic reference has zero beta-channel columns pads Cb to one
    // all-zero column, and sizing Cr from the unpadded total undercounts by
    // N, overflowing Cr's buffer when Cb is copied in. The extra all-zero
    // padding column contributes nothing to the Gram matrix, so including it
    // in ncols_r below is harmless.
    const uint64_t cols_a_padded = Ca.size() / static_cast<size_t>(N);
    const uint64_t cols_b_padded = Cb.size() / static_cast<size_t>(N);
    std::vector<double> Cr(static_cast<size_t>(N) * (cols_a_padded + cols_b_padded));
    std::copy(Ca.begin(), Ca.end(), Cr.begin());
    std::copy(Cb.begin(), Cb.end(), Cr.begin() + static_cast<long>(Ca.size()));
    const double half = std::sqrt(0.5);
    for (auto& c : Cr) c *= half;
    const uint64_t ncols_r = cols_a_padded + cols_b_padded;
    const double scale =
        static_cast<double>(nocc_) / std::max(trace_ds(gram_from_columns(Cr, N, ncols_r)), 1e-10);
    const double s = std::sqrt(scale);
    for (auto& c : Cr) c *= s;
    Cocc_a = Cr;
    Cocc_b = Cr;
    ncols_a = ncols_b = ncols_r;
  }

  const std::vector<double> D_a_guess = gram_from_columns(Cocc_a, N, ncols_a);
  const std::vector<double> D_b_guess = gram_from_columns(Cocc_b, N, ncols_b);
  CUDA_CHECK(cudaMemcpy(d_D_a_, D_a_guess.data(),
                        D_a_guess.size() * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_D_b_, D_b_guess.data(),
                        D_b_guess.size() * sizeof(double), cudaMemcpyHostToDevice));

  // Grow Cocc buffers if SAC column count exceeds the ctor pad (nocc).
  const size_t need_a = static_cast<size_t>(N) * std::max<uint64_t>(ncols_a, 1) * sizeof(double);
  const size_t need_b = static_cast<size_t>(N) * std::max<uint64_t>(ncols_b, 1) * sizeof(double);
  d_Cocc_a_.ensure(need_a);
  d_Cocc_b_.ensure(need_b);
  CUDA_CHECK(cudaMemset(d_Cocc_a_, 0, need_a));
  CUDA_CHECK(cudaMemset(d_Cocc_b_, 0, need_b));
  if (ncols_a > 0)
    CUDA_CHECK(cudaMemcpy(d_Cocc_a_, Cocc_a.data(), Cocc_a.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
  if (ncols_b > 0)
    CUDA_CHECK(cudaMemcpy(d_Cocc_b_, Cocc_b.data(), Cocc_b.size() * sizeof(double),
                          cudaMemcpyHostToDevice));

  sac_ncols_a_ = ncols_a;
  sac_ncols_b_ = ncols_b;
  use_sac_cocc_ = true;
}

void SCFSolver::initial_guess() {
  NvtxRange range("initial_guess");
  int N = static_cast<int>(nao_);
  const bool use_sac = (sac_config_ != nullptr) && !params_.force_hcore_guess;

  if (use_sac) {
    assemble_sac_guess();
    // First SCF iteration builds a normal Fock from SAC D/Cocc (incl. Vxc).
    // β HOMO/LUMO mix needs a full MO space — deferred until after first diag.
    return;
  }

  cublasDcopy(cublas_, N*N, d_Hcore_, 1, d_Fock_a_, 1);
  if (uks_) cublasDcopy(cublas_, N*N, d_Hcore_, 1, d_Fock_b_, 1);

  diagonalize_fock(d_Fock_a_, d_C_a_, mo_energies_a_);
  build_density_auto(d_C_a_, uks_ ? nocc_a_ : nocc_, mo_energies_a_, d_D_a_, sph_group_a_);

  if (uks_) {
    diagonalize_fock(d_Fock_b_, d_C_b_, mo_energies_b_);
    build_density_auto(d_C_b_, nocc_b_, mo_energies_b_, d_D_b_, sph_group_b_);
    break_beta_symmetry();
    symmetry_broken_ = true;
  } else {
    cublasDcopy(cublas_, N*N, d_C_a_, 1, d_C_b_, 1);
    cublasDcopy(cublas_, N*N, d_D_a_, 1, d_D_b_, 1);
    mo_energies_b_ = mo_energies_a_;
  }
}

void SCFSolver::run() {
  NvtxRange scf_range("SCFSolver::run");
  if (params_.verbose) {
    std::cout << "\n=== Starting SCF Calculation ("
              << (uks_ ? "UKS" : "RKS") << ") ===\n"
              << "  Basis functions: " << nao_ << "\n"
              << "  Occupied: " << (uks_
                  ? ("α=" + std::to_string(nocc_a_) + " β=" + std::to_string(nocc_b_))
                  : std::to_string(nocc_)) << "\n"
              << "  Electrons: " << nelec_ << "\n"
              << "  Multiplicity: " << mol_.multiplicity() << "\n"
              << "  Nuclear repulsion: " << std::setprecision(12) << e_nuc_
              << " Ha\n";
    if (xc_) std::cout << "  XC: enabled"
                       << (xc_->is_hybrid() ? " (hybrid)" : "") << "\n";
  }

  build_core_hamiltonian();
  initial_guess();

  int N = static_cast<int>(nao_);
  int n2 = N * N;
  double e_prev = 0.0;
  diis_a_.clear();
  diis_b_.clear();

  for (iter_ = 1; iter_ <= params_.max_iter; iter_++) {
    NvtxRange iter_range("SCF iteration");

    if (uks_) build_fock_uks();
    else      build_fock_rks();

    // Energy on device (symmetric matrices → Frobenius = Tr(AB))
    {
      NvtxRange energy_range("SCF energy");
      if (uks_) {
      e_hcore_ = frobenius_dot(d_D_a_, d_Hcore_, n2) +
                 frobenius_dot(d_D_b_, d_Hcore_, n2);
      double one = 1.0;
      cublasDcopy(cublas_, n2, d_D_a_, 1, d_D_tot_, 1);
      cublasDaxpy(cublas_, n2, &one, d_D_b_, 1, d_D_tot_, 1);
      e_j_ = 0.5 * frobenius_dot(d_D_tot_, d_J_, n2);
      e_k_ = 0.0;
      if (xc_ && (xc_->is_hybrid() || xc_->is_hf())) {
        e_k_ = -0.5 * (frobenius_dot(d_D_a_, d_K_a_, n2) +
                       frobenius_dot(d_D_b_, d_K_b_, n2));
      }
      e_kin_ = frobenius_dot(d_D_a_, d_T_, n2) + frobenius_dot(d_D_b_, d_T_, n2);
      e_ne_  = frobenius_dot(d_D_a_, d_V_, n2) + frobenius_dot(d_D_b_, d_V_, n2);
      tr_ds_ = frobenius_dot(d_D_a_, d_S_, n2) + frobenius_dot(d_D_b_, d_S_, n2);
    } else {
      double tr_dh = frobenius_dot(d_D_a_, d_Hcore_, n2);
      double tr_dj = frobenius_dot(d_D_a_, d_J_, n2);
      e_hcore_ = 2.0 * tr_dh;
      e_j_ = 2.0 * tr_dj;
      e_k_ = 0.0;
      if (xc_ && (xc_->is_hybrid() || xc_->is_hf()))
        e_k_ = -frobenius_dot(d_D_a_, d_K_a_, n2);
      e_kin_ = 2.0 * frobenius_dot(d_D_a_, d_T_, n2);
      e_ne_  = 2.0 * frobenius_dot(d_D_a_, d_V_, n2);
      tr_ds_ = frobenius_dot(d_D_a_, d_S_, n2);
    }
    e_elec_ = e_hcore_ + e_j_ + e_k_ + e_xc_;
    e_total_ = e_elec_ + e_nuc_;
    }

    double dE = (iter_ > 1) ? (e_total_ - e_prev) : 0.0;

    double rms_a = 1.0, rms_b = 0.0;
    {
      NvtxRange rms_range("SCF RMS");
      rms_a = (iter_ > 1) ? compute_rms_delta_device(d_D_a_, d_D_old_a_, n2) : 1.0;
      rms_b = (uks_ && iter_ > 1)
                         ? compute_rms_delta_device(d_D_b_, d_D_old_b_, n2)
                         : 0.0;
    }
    double rms_d = uks_ ? std::sqrt(0.5 * (rms_a * rms_a + rms_b * rms_b)) : rms_a;

    // DIIS commutator RMS ||FDS-SDF|| (raw Fock/D, before any extrapolation):
    // how far D is from being a stationary eigenspace of F. Unlike rmsD,
    // this stays flat under a rotation within an exactly-degenerate frontier
    // subspace (e.g. an open-shell radical's degenerate SOMO), so it doesn't
    // stall on that oscillation the way rmsD can.
    double err_a = diis_a_.compute_residual(cublas_, d_S_, d_Fock_a_, d_D_a_);
    double err_b = uks_ ? diis_b_.compute_residual(cublas_, d_S_, d_Fock_b_, d_D_b_) : 0.0;
    double diis_err = uks_ ? std::sqrt(0.5 * (err_a * err_a + err_b * err_b)) : err_a;

    if (params_.print_level >= 1) {
      std::cout << "  Iter " << std::setw(3) << iter_
                << "  Etot = " << std::setw(16) << std::setprecision(10) << std::fixed << e_total_
                << "  dE = " << std::setw(12) << std::scientific << dE
                << "  rmsD = " << std::setw(10) << rms_d
                << "  |FDS-SDF| = " << std::setw(10) << diis_err;
      if (xc_) std::cout << "  Exc = " << std::setw(14) << std::fixed << std::setprecision(8) << e_xc_;
      std::cout << "\n" << std::flush;
    }

    if (iter_ > 1 && diis_err < params_.conv_thresh &&
        std::abs(dE) < params_.energy_conv_thresh) {
      converged_ = true;
      if (params_.verbose)
        std::cout << "\n  SCF converged in " << iter_ << " iterations\n";
      break;
    }

    if (iter_ >= params_.diis_start) {
      diis_a_.extrapolate(cublas_, d_Fock_a_);
      if (uks_)
        diis_b_.extrapolate(cublas_, d_Fock_b_);
    }

    cublasDcopy(cublas_, n2, d_D_a_, 1, d_D_old_a_, 1);
    if (uks_)
      cublasDcopy(cublas_, n2, d_D_b_, 1, d_D_old_b_, 1);

    diagonalize_fock(d_Fock_a_, d_C_a_, mo_energies_a_);
    build_density_auto(d_C_a_, uks_ ? nocc_a_ : nocc_, mo_energies_a_, d_D_a_, sph_group_a_);
    if (uks_) {
      diagonalize_fock(d_Fock_b_, d_C_b_, mo_energies_b_);
      build_density_auto(d_C_b_, nocc_b_, mo_energies_b_, d_D_b_, sph_group_b_);
      if (!symmetry_broken_) {
        break_beta_symmetry();
        symmetry_broken_ = true;
      }
    } else {
      cublasDcopy(cublas_, n2, d_C_a_, 1, d_C_b_, 1);
      cublasDcopy(cublas_, n2, d_D_a_, 1, d_D_b_, 1);
      mo_energies_b_ = mo_energies_a_;
    }
    // After the first diagonalization, drop SAC Cocc and use standard nocc.
    use_sac_cocc_ = false;

    if (params_.damping > 0.0) {
      double damp = params_.damping;
      double one_minus = 1.0 - damp;
      cublasDscal(cublas_, n2, &one_minus, d_D_a_, 1);
      cublasDaxpy(cublas_, n2, &damp, d_D_old_a_, 1, d_D_a_, 1);
      if (uks_) {
        cublasDscal(cublas_, n2, &one_minus, d_D_b_, 1);
        cublasDaxpy(cublas_, n2, &damp, d_D_old_b_, 1, d_D_b_, 1);
      }
    }

    e_prev = e_total_;
  }

  form_total_density();
  if (uks_) {
    double one = 1.0;
    cublasDcopy(cublas_, n2, d_D_a_, 1, d_D_tot_, 1);
    cublasDaxpy(cublas_, n2, &one, d_D_b_, 1, d_D_tot_, 1);
  }

  if (params_.verbose) {
    std::cout << "\n=== SCF Results ===\n"
              << "Mode: " << (uks_ ? "UKS" : "RKS") << "\n"
              << "Converged: " << (converged_ ? "Yes" : "No") << "\n"
              << "Iterations: " << iter_ << "\n"
              << std::setprecision(12) << std::fixed
              << "Nuclear repulsion: " << e_nuc_ << " Ha\n"
              << "E_Hcore:           " << e_hcore_ << " Ha\n"
              << "E_J:               " << e_j_ << " Ha\n";
    if (xc_ && (xc_->is_hybrid() || xc_->is_hf()))
      std::cout << "E_K (exchange):    " << e_k_ << " Ha\n";
    if (xc_ && !xc_->is_hf())
      std::cout << "XC energy:         " << e_xc_ << " Ha\n";
    else if (xc_ && xc_->is_hf())
      std::cout << "XC energy:         0 (HF)\n";
    std::cout << "Electronic energy: " << e_elec_ << " Ha\n"
              << "Total energy:      " << e_total_ << " Ha\n"
              << "Tr[D*S]:           " << tr_ds_ << "\n";

    if (params_.print_mos) {
      auto print_mos = [&](const char* label, const std::vector<double>& eps, uint64_t nocc) {
        std::cout << "\n" << label << " orbital energies (Ha):\n";
        for (size_t i = 0; i < eps.size(); i++) {
          std::cout << "  MO " << std::setw(3) << i << ": "
                    << std::setw(14) << eps[i];
          if (i + 1 == nocc) std::cout << "  (HOMO)";
          else if (i == nocc) std::cout << "  (LUMO)";
          std::cout << "\n";
        }
      };
      if (uks_) {
        print_mos("Alpha", mo_energies_a_, nocc_a_);
        print_mos("Beta", mo_energies_b_, nocc_b_);
      } else {
        print_mos("Restricted", mo_energies_a_, nocc_);
      }
    }
  }

  if (params_.suppress_output) return;

  std::cout << std::setprecision(16) << std::fixed
            << "=== ENERGY_COMPONENTS ===\n"
            << "E_NUC " << e_nuc_ << "\n"
            << "E_KIN " << e_kin_ << "\n"
            << "E_NE " << e_ne_ << "\n"
            << "E_HCORE " << e_hcore_ << "\n"
            << "E_J " << e_j_ << "\n"
            << "E_K " << e_k_ << "\n"
            << "E_XC " << e_xc_ << "\n"
            << "E_ELEC " << e_elec_ << "\n"
            << "E_TOT " << e_total_ << "\n"
            << "TR_DS " << tr_ds_ << "\n"
            << "NAO " << nao_ << "\n"
            << "NOCC " << nocc() << "\n"
            << "NOCC_A " << nocc_a_ << "\n"
            << "NOCC_B " << nocc_b_ << "\n"
            << "NELEC " << nelec_ << "\n"
            << "UKS " << (uks_ ? 1 : 0) << "\n"
            << "N_SCF " << iter_ << "\n"
            << "CONVERGED " << (converged_ ? 1 : 0) << "\n"
            << "=== END_ENERGY_COMPONENTS ===\n";
}

}  // namespace cuest
