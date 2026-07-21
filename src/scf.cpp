/**
 * @file scf.cpp — RKS / UKS SCF with device DIIS (cuSOLVER dsygvd).
 */
#include "scf.hpp"
#include "cuest_wrapper/nvtx.hpp"
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace cuest {

SCFSolver::SCFSolver(CuESTContext& ctx, BasisBuilder& basis,
                     DFJKBuilder& dfjk, XCBuilder* xc,
                     ECPBuilder* ecp_builder, ECPIntegrals* ecp_int,
                     const Molecule& mol, SCFParams params)
    : ctx_(ctx), basis_(basis), dfjk_(dfjk), xc_(xc),
      ecp_builder_(ecp_builder), ecp_int_(ecp_int),
      mol_(mol), params_(params) {
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
  d_Fock_save_a_.alloc(n2_bytes);
  d_Fock_save_b_.alloc(n2_bytes);
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
  cublasDdot(cublas_, n2, d_A, 1, d_B, 1, &result);
  return result;
}

double SCFSolver::compute_rms_delta_device(const double* d_D,
                                            const double* d_D_old, int n2) {
  // scratch = D - D_old
  cublasDcopy(cublas_, n2, d_D, 1, d_rms_scratch_, 1);
  double minus_one = -1.0;
  cublasDaxpy(cublas_, n2, &minus_one, d_D_old, 1, d_rms_scratch_, 1);
  double ss = 0.0;
  cublasDdot(cublas_, n2, d_rms_scratch_, 1, d_rms_scratch_, 1, &ss);
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

  CUDA_CHECK(cudaMemcpy(d_Cocc_a_, d_C_a_, nao_ * nocc_ * sizeof(double),
                        cudaMemcpyDeviceToDevice));

  if (xc_ && (xc_->is_hybrid() || xc_->is_hf())) {
    const double k_factor = -1.0;
    dfjk_.compute_K(nocc_, d_Cocc_a_, d_K_a_);
    cublasDaxpy(cublas_, nao_*nao_, &k_factor, d_K_a_, 1, d_Fock_a_, 1);
  }

  if (xc_ && !xc_->is_hf()) {
    double exc_val = 0.0;
    xc_->compute_vxc_rks(nocc_, d_Cocc_a_, &exc_val, d_Vxc_a_);
    e_xc_ = exc_val;
    double one = 1.0;
    cublasDaxpy(cublas_, nao_*nao_, &one, d_Vxc_a_, 1, d_Fock_a_, 1);
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

  uint64_t nocc_a_pad = std::max<uint64_t>(nocc_a_, 1);
  uint64_t nocc_b_pad = std::max<uint64_t>(nocc_b_, 1);
  CUDA_CHECK(cudaMemset(d_Cocc_a_, 0, nao_ * nocc_a_pad * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_Cocc_b_, 0, nao_ * nocc_b_pad * sizeof(double)));
  if (nocc_a_ > 0)
    CUDA_CHECK(cudaMemcpy(d_Cocc_a_, d_C_a_, nao_ * nocc_a_ * sizeof(double),
                          cudaMemcpyDeviceToDevice));
  if (nocc_b_ > 0)
    CUDA_CHECK(cudaMemcpy(d_Cocc_b_, d_C_b_, nao_ * nocc_b_ * sizeof(double),
                          cudaMemcpyDeviceToDevice));

  if (xc_ && (xc_->is_hybrid() || xc_->is_hf())) {
    const double k_factor = -1.0;
    if (nocc_a_ > 0) {
      dfjk_.compute_K(nocc_a_, d_Cocc_a_, d_K_a_);
      cublasDaxpy(cublas_, nao_*nao_, &k_factor, d_K_a_, 1, d_Fock_a_, 1);
    } else {
      CUDA_CHECK(cudaMemset(d_K_a_, 0, nao_ * nao_ * sizeof(double)));
    }
    if (nocc_b_ > 0) {
      dfjk_.compute_K(nocc_b_, d_Cocc_b_, d_K_b_);
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
  cudaMemcpy(mo_energies.data(), d_eigvals_,
             static_cast<size_t>(N) * sizeof(double), cudaMemcpyDeviceToHost);
}

void SCFSolver::break_beta_symmetry() {
  if (nocc_a_ != nocc_b_) return;
  if (nocc_b_ == 0 || nocc_b_ >= nao_) return;
  double theta = params_.break_symmetry;
  if (std::abs(theta) < 1e-12) return;

  std::vector<double> C(nao_ * nao_);
  cudaMemcpy(C.data(), d_C_b_, nao_*nao_*sizeof(double), cudaMemcpyDeviceToHost);

  const size_t homo = nocc_b_ - 1;
  const size_t lumo = nocc_b_;
  const double c = std::cos(theta), s = std::sin(theta);
  for (size_t i = 0; i < nao_; i++) {
    double cho = C[i + homo * nao_];
    double clu = C[i + lumo * nao_];
    C[i + homo * nao_] = c * cho + s * clu;
    C[i + lumo * nao_] = -s * cho + c * clu;
  }
  cudaMemcpy(d_C_b_, C.data(), nao_*nao_*sizeof(double), cudaMemcpyHostToDevice);
  build_density(d_C_b_, nocc_b_, d_D_b_);
  if (params_.print_level >= 2)
    std::cout << "  Symmetry breaking: β HOMO/LUMO mix θ=" << theta << " rad\n";
}

void SCFSolver::initial_guess() {
  NvtxRange range("initial_guess");
  bool use_hcore = (ecp_builder_ && ecp_builder_->has_ecp());
  int N = static_cast<int>(nao_);

  if (use_hcore) {
    cublasDcopy(cublas_, N*N, d_Hcore_, 1, d_Fock_a_, 1);
  } else {
    std::vector<double> S_h(static_cast<size_t>(N*N));
    cudaMemcpy(S_h.data(), d_S_, static_cast<size_t>(N*N)*sizeof(double),
               cudaMemcpyDeviceToHost);
    std::vector<double> D_sad(static_cast<size_t>(N*N), 0.0);
    std::vector<int> ao_per_atom(mol_.natom(), 0), ao_start(mol_.natom(), 0);
    int ao_fill = 0;
    for (size_t a = 0; a < mol_.natom(); a++) {
      ao_start[a] = ao_fill;
      int Z = mol_.atom(a).atomic_number;
      int nao = (Z <= 2) ? 5 : 14;
      if (Z > 10) nao = (Z <= 18) ? 18 : 27;
      if (Z > 36) nao = 32;
      ao_per_atom[a] = std::min(nao, N - ao_fill);
      ao_fill += ao_per_atom[a];
    }
    auto nval = [](int Z) {
      if (Z <= 2) return Z;
      if (Z <= 10) return Z - 2;
      if (Z <= 18) return Z - 10;
      if (Z <= 36) return Z - 18;
      return Z - 36;
    };
    for (size_t a = 0; a < mol_.natom(); a++) {
      int nv = nval(mol_.atom(a).atomic_number);
      int start = ao_start[a], count = ao_per_atom[a];
      if (count <= 0) continue;
      double occ = static_cast<double>(nv) / count * 0.5;
      for (int i = start; i < start + count && i < N; i++)
        D_sad[static_cast<size_t>(i*N + i)] = occ;
    }
    double trDS = 0.0;
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        trDS += D_sad[static_cast<size_t>(i*N+j)] * S_h[static_cast<size_t>(j*N+i)];
    double target = uks_ ? static_cast<double>(nocc_a_) : static_cast<double>(nocc_);
    double scale = target / std::max(trDS, 1e-10);
    for (auto& d : D_sad) d *= scale;

    std::vector<double> F_init(static_cast<size_t>(N*N));
    cudaMemcpy(F_init.data(), d_Hcore_, static_cast<size_t>(N*N)*sizeof(double),
               cudaMemcpyDeviceToHost);
    const double mix = 0.5;
    for (int i = 0; i < N*N; i++) F_init[static_cast<size_t>(i)] += mix * D_sad[static_cast<size_t>(i)];
    cudaMemcpy(d_Fock_a_, F_init.data(), static_cast<size_t>(N*N)*sizeof(double),
               cudaMemcpyHostToDevice);
  }

  diagonalize_fock(d_Fock_a_, d_C_a_, mo_energies_a_);
  build_density(d_C_a_, uks_ ? nocc_a_ : nocc_, d_D_a_);

  if (uks_) {
    cublasDcopy(cublas_, N*N, d_C_a_, 1, d_C_b_, 1);
    mo_energies_b_ = mo_energies_a_;
    build_density(d_C_b_, nocc_b_, d_D_b_);
    break_beta_symmetry();
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

    // Snapshot raw Fock before DIIS (for trust-region reject).
    cublasDcopy(cublas_, n2, d_Fock_a_, 1, d_Fock_save_a_, 1);
    if (uks_)
      cublasDcopy(cublas_, n2, d_Fock_b_, 1, d_Fock_save_b_, 1);

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

    if (params_.print_level >= 1) {
      std::cout << "  Iter " << std::setw(3) << iter_
                << "  Etot = " << std::setw(16) << std::setprecision(10) << std::fixed << e_total_
                << "  dE = " << std::setw(12) << std::scientific << dE
                << "  rmsD = " << std::setw(10) << rms_d;
      if (xc_) std::cout << "  Exc = " << std::setw(14) << std::fixed << std::setprecision(8) << e_xc_;
      std::cout << "\n" << std::flush;
    }

    if (iter_ > 1 && rms_d < params_.conv_thresh &&
        std::abs(dE) < params_.energy_conv_thresh) {
      converged_ = true;
      if (params_.verbose)
        std::cout << "\n  SCF converged in " << iter_ << " iterations\n";
      break;
    }

    if (iter_ >= params_.diis_start) {
      diis_a_.extrapolate(cublas_, d_S_, d_Fock_a_, d_D_a_);
      if (uks_)
        diis_b_.extrapolate(cublas_, d_S_, d_Fock_b_, d_D_b_);
    }

    cublasDcopy(cublas_, n2, d_D_a_, 1, d_D_old_a_, 1);
    if (uks_)
      cublasDcopy(cublas_, n2, d_D_b_, 1, d_D_old_b_, 1);

    diagonalize_fock(d_Fock_a_, d_C_a_, mo_energies_a_);
    build_density(d_C_a_, uks_ ? nocc_a_ : nocc_, d_D_a_);
    if (uks_) {
      diagonalize_fock(d_Fock_b_, d_C_b_, mo_energies_b_);
      build_density(d_C_b_, nocc_b_, d_D_b_);
    } else {
      cublasDcopy(cublas_, n2, d_C_a_, 1, d_C_b_, 1);
      cublasDcopy(cublas_, n2, d_D_a_, 1, d_D_b_, 1);
      mo_energies_b_ = mo_energies_a_;
    }

    // DIIS trust-region: reject unphysical densities and reset the subspace.
    if (iter_ >= params_.diis_start + 2) {
      double trA = frobenius_dot(d_D_a_, d_S_, n2);
      double expect = uks_ ? static_cast<double>(nocc_a_) : static_cast<double>(nocc_);
      if (std::abs(trA - expect) > 0.5 * std::max(expect, 1.0)) {
        if (params_.print_level >= 2)
          std::cout << "  DIIS rejected (Tr[Da*S]=" << trA << "), using raw Fock\n";
        cublasDcopy(cublas_, n2, d_Fock_save_a_, 1, d_Fock_a_, 1);
        diagonalize_fock(d_Fock_a_, d_C_a_, mo_energies_a_);
        build_density(d_C_a_, uks_ ? nocc_a_ : nocc_, d_D_a_);
        diis_a_.clear();
        if (uks_) {
          cublasDcopy(cublas_, n2, d_Fock_save_b_, 1, d_Fock_b_, 1);
          diagonalize_fock(d_Fock_b_, d_C_b_, mo_energies_b_);
          build_density(d_C_b_, nocc_b_, d_D_b_);
          diis_b_.clear();
        }
      }
    }

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

    if (iter_ >= 25 && (iter_ % 15) == 0 && !converged_) {
      diis_a_.clear();
      diis_b_.clear();
      if (params_.print_level >= 2)
        std::cout << "  DIIS subspace flushed at iter " << iter_ << "\n";
    }
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
