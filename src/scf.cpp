/**
 * @file scf.cpp — RKS / UKS SCF with DIIS (cuSOLVER dsygvd). Application layer.
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
  d_Hcore_.alloc(N2 * sizeof(double));
  d_S_.alloc(N2 * sizeof(double));
  d_T_.alloc(N2 * sizeof(double));
  d_V_.alloc(N2 * sizeof(double));
  d_ECP_.alloc(N2 * sizeof(double));
  d_Fock_a_.alloc(N2 * sizeof(double));
  d_Fock_b_.alloc(N2 * sizeof(double));
  d_D_a_.alloc(N2 * sizeof(double));
  d_D_b_.alloc(N2 * sizeof(double));
  d_D_tot_.alloc(N2 * sizeof(double));
  d_D_old_a_.alloc(N2 * sizeof(double));
  d_D_old_b_.alloc(N2 * sizeof(double));
  d_C_a_.alloc(N2 * sizeof(double));
  d_C_b_.alloc(N2 * sizeof(double));
  d_J_.alloc(N2 * sizeof(double));
  d_K_a_.alloc(N2 * sizeof(double));
  d_K_b_.alloc(N2 * sizeof(double));
  d_Vxc_a_.alloc(N2 * sizeof(double));
  d_Vxc_b_.alloc(N2 * sizeof(double));
  d_eigvals_.alloc(nao_ * sizeof(double));
  d_Fwork_.alloc(N2 * sizeof(double));
  d_Swork_.alloc(N2 * sizeof(double));

  auto xyz_h = mol_.xyz_host();
  auto chg_h = ecp_cores ? mol_.charges_host(*ecp_cores) : mol_.charges_host();
  d_xyz_.alloc(xyz_h.size() * sizeof(double));
  d_charges_.alloc(chg_h.size() * sizeof(double));
  CUDA_CHECK(cudaMemcpy(d_xyz_, xyz_h.data(), xyz_h.size()*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_charges_, chg_h.data(), chg_h.size()*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D_a_, 0, N2 * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_D_b_, 0, N2 * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_D_tot_, 0, N2 * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_ECP_, 0, N2 * sizeof(double)));
}

SCFSolver::~SCFSolver() {
  if (cublas_) cublasDestroy(cublas_);
  if (cusolver_) cusolverDnDestroy(cusolver_);
}

double SCFSolver::trace_dot(const double* d_A, const double* d_B, int N) {
  std::vector<double> A(N*N), B(N*N);
  cudaMemcpy(A.data(), d_A, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(B.data(), d_B, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  double tr = 0.0;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      tr += A[i*N+j] * B[j*N+i];
  return tr;
}

void SCFSolver::build_core_hamiltonian() {
  const OwnedAOPairList& pair_list = basis_.pair_list();
  OneElectronIntegrals oe(ctx_, basis_.basis(), pair_list.get());
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

void SCFSolver::build_density(DeviceArray<double>& d_C, uint64_t nocc, DeviceArray<double>& d_D) {
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
    // RKS: D_tot = 2 * D_alpha for reporting; internal D_a is D_alpha
    double two = 2.0;
    cublasDscal(cublas_, nao_*nao_, &two, d_D_tot_, 1);
  }
}

void SCFSolver::build_fock_rks() {
  // D_a holds D_alpha; J from D_alpha with F = H + 2J - K + Vxc
  dfjk_.compute_J(d_D_a_, d_J_);

  double two = 2.0;
  cublasDcopy(cublas_, nao_*nao_, d_Hcore_, 1, d_Fock_a_, 1);
  cublasDaxpy(cublas_, nao_*nao_, &two, d_J_, 1, d_Fock_a_, 1);

  DeviceArray<double> d_Cocc;
  d_Cocc.alloc(nao_ * nocc_ * sizeof(double));
  CUDA_CHECK(cudaMemcpy(d_Cocc, d_C_a_, nao_ * nocc_ * sizeof(double),
                        cudaMemcpyDeviceToDevice));

  if (xc_ && (xc_->is_hybrid() || xc_->is_hf())) {
    const double k_factor = -1.0;
    dfjk_.compute_K(nocc_, d_Cocc, d_K_a_);
    cublasDaxpy(cublas_, nao_*nao_, &k_factor, d_K_a_, 1, d_Fock_a_, 1);
  }

  // Pure HF: cuEST XC potential API rejects FUNCTIONAL_HF — Exc = 0.
  if (xc_ && !xc_->is_hf()) {
    double exc_val = 0.0;
    xc_->compute_vxc_rks(nocc_, d_Cocc, &exc_val, d_Vxc_a_);
    e_xc_ = exc_val;
    double one = 1.0;
    cublasDaxpy(cublas_, nao_*nao_, &one, d_Vxc_a_, 1, d_Fock_a_, 1);
  } else {
    e_xc_ = 0.0;
  }
}

void SCFSolver::build_fock_uks() {
  // J from total density D_a + D_b
  form_total_density();
  // form_total_density scales RKS; for UKS d_D_tot_ = D_a+D_b already via copy+axpy
  // Recompute without the RKS ×2 path:
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

  DeviceArray<double> d_Cocc_a, d_Cocc_b;
  // cuEST UKS requires nocc > 0 for both channels; pad empty spin with zeros.
  uint64_t nocc_a_pad = std::max<uint64_t>(nocc_a_, 1);
  uint64_t nocc_b_pad = std::max<uint64_t>(nocc_b_, 1);
  d_Cocc_a.alloc(nao_ * nocc_a_pad * sizeof(double));
  d_Cocc_b.alloc(nao_ * nocc_b_pad * sizeof(double));
  CUDA_CHECK(cudaMemset(d_Cocc_a, 0, nao_ * nocc_a_pad * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_Cocc_b, 0, nao_ * nocc_b_pad * sizeof(double)));
  if (nocc_a_ > 0)
    CUDA_CHECK(cudaMemcpy(d_Cocc_a, d_C_a_, nao_ * nocc_a_ * sizeof(double),
                          cudaMemcpyDeviceToDevice));
  if (nocc_b_ > 0)
    CUDA_CHECK(cudaMemcpy(d_Cocc_b, d_C_b_, nao_ * nocc_b_ * sizeof(double),
                          cudaMemcpyDeviceToDevice));

  if (xc_ && (xc_->is_hybrid() || xc_->is_hf())) {
    const double k_factor = -1.0;
    if (nocc_a_ > 0) {
      dfjk_.compute_K(nocc_a_, d_Cocc_a, d_K_a_);
      cublasDaxpy(cublas_, nao_*nao_, &k_factor, d_K_a_, 1, d_Fock_a_, 1);
    } else {
      CUDA_CHECK(cudaMemset(d_K_a_, 0, nao_ * nao_ * sizeof(double)));
    }
    if (nocc_b_ > 0) {
      dfjk_.compute_K(nocc_b_, d_Cocc_b, d_K_b_);
      cublasDaxpy(cublas_, nao_*nao_, &k_factor, d_K_b_, 1, d_Fock_b_, 1);
    } else {
      CUDA_CHECK(cudaMemset(d_K_b_, 0, nao_ * nao_ * sizeof(double)));
    }
  }

  if (xc_ && !xc_->is_hf()) {
    double exc_val = 0.0;
    xc_->compute_vxc_uks(nocc_a_pad, nocc_b_pad, d_Cocc_a, d_Cocc_b,
                         &exc_val, d_Vxc_a_, d_Vxc_b_);
    e_xc_ = exc_val;
    cublasDaxpy(cublas_, nao_*nao_, &one, d_Vxc_a_, 1, d_Fock_a_, 1);
    cublasDaxpy(cublas_, nao_*nao_, &one, d_Vxc_b_, 1, d_Fock_b_, 1);
  } else if (xc_ && xc_->is_hf()) {
    e_xc_ = 0.0;
  }
}

void SCFSolver::diagonalize_fock(DeviceArray<double>& d_Fock, DeviceArray<double>& d_C,
                                 std::vector<double>& mo_energies) {
  int N = static_cast<int>(nao_);
  int lda = N;

  cublasDcopy(cublas_, N*N, d_Fock, 1, d_Fwork_, 1);
  cublasDcopy(cublas_, N*N, d_S_, 1, d_Swork_, 1);

  int lwork = 0;
  cusolverDnDsygvd_bufferSize(cusolver_, CUSOLVER_EIG_TYPE_1,
      CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      N, d_Fwork_, lda, d_Swork_, lda,
      d_eigvals_, &lwork);

  double* d_work = nullptr;
  CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));
  int* d_info = nullptr;
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

  cusolverStatus_t st = cusolverDnDsygvd(cusolver_, CUSOLVER_EIG_TYPE_1,
      CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      N, d_Fwork_, lda, d_Swork_, lda,
      d_eigvals_, d_work, lwork, d_info);

  int info_h = 0;
  CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_work));
  CUDA_CHECK(cudaFree(d_info));

  if (st != CUSOLVER_STATUS_SUCCESS || info_h != 0) {
    throw std::runtime_error(
        "cuSOLVER dsygvd failed (status=" + std::to_string(static_cast<int>(st)) +
        ", info=" + std::to_string(info_h) + ")");
  }

  cublasDcopy(cublas_, N*N, d_Fwork_, 1, d_C, 1);
  mo_energies.resize(N);
  cudaMemcpy(mo_energies.data(), d_eigvals_, N*sizeof(double), cudaMemcpyDeviceToHost);
}

void SCFSolver::break_beta_symmetry() {
  // Mix β HOMO with LUMO to break α/β equivalence on the initial guess.
  // Only useful when nα == nβ (broken-symmetry singlets). For ordinary
  // open-shell (nα ≠ nβ) the guess is already spin-polarized; mixing here
  // can destabilize DIIS on small bases.
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
  bool use_hcore = (ecp_builder_ && ecp_builder_->has_ecp());
  int N = static_cast<int>(nao_);

  if (use_hcore) {
    cublasDcopy(cublas_, N*N, d_Hcore_, 1, d_Fock_a_, 1);
  } else {
    // SAD-like diagonal guess mixed into Hcore
    std::vector<double> S_h(N*N);
    cudaMemcpy(S_h.data(), d_S_, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    std::vector<double> D_sad(N*N, 0.0);
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
        D_sad[i*N + i] = occ;
    }
    double trDS = 0.0;
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        trDS += D_sad[i*N+j] * S_h[j*N+i];
    double target = uks_ ? static_cast<double>(nocc_a_) : static_cast<double>(nocc_);
    double scale = target / std::max(trDS, 1e-10);
    for (auto& d : D_sad) d *= scale;

    std::vector<double> F_init(N*N);
    cudaMemcpy(F_init.data(), d_Hcore_, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    const double mix = 0.5;
    for (int i = 0; i < N*N; i++) F_init[i] += mix * D_sad[i];
    cudaMemcpy(d_Fock_a_, F_init.data(), N*N*sizeof(double), cudaMemcpyHostToDevice);
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

double SCFSolver::compute_rms_delta(const std::vector<double>& D_new,
                                     const std::vector<double>& D_old) {
  double sum = 0.0;
  for (size_t i = 0; i < D_new.size(); i++) {
    double diff = D_new[i] - D_old[i];
    sum += diff * diff;
  }
  return std::sqrt(sum / D_new.size());
}

void SCFSolver::diis_extrapolate(std::vector<std::vector<double>>& errs,
                                 std::vector<std::vector<double>>& focks,
                                 const std::vector<double>& F_host,
                                 const std::vector<double>& D_host,
                                 DeviceArray<double>& d_Fock) {
  int N = static_cast<int>(nao_);
  std::vector<double> S_host(N*N);
  cudaMemcpy(S_host.data(), d_S_, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  std::vector<double> FD(N*N, 0.0), FDS(N*N, 0.0), SD(N*N, 0.0), SDF(N*N, 0.0);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double fd = 0.0, sd = 0.0;
      for (int k = 0; k < N; k++) {
        fd += F_host[i*N+k] * D_host[k*N+j];
        sd += S_host[i*N+k] * D_host[k*N+j];
      }
      FD[i*N+j] = fd; SD[i*N+j] = sd;
    }
  }
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      double fds = 0.0, sdf = 0.0;
      for (int k = 0; k < N; k++) {
        fds += FD[i*N+k] * S_host[k*N+j];
        sdf += SD[i*N+k] * F_host[k*N+j];
      }
      FDS[i*N+j] = fds; SDF[i*N+j] = sdf;
    }

  std::vector<double> err(N*N);
  for (int i = 0; i < N*N; i++) err[i] = FDS[i] - SDF[i];

  if (static_cast<int>(errs.size()) >= params_.diis_max_space) {
    errs.erase(errs.begin());
    focks.erase(focks.begin());
  }
  errs.push_back(err);
  focks.push_back(F_host);

  int nvec = static_cast<int>(errs.size());
  if (nvec < 2) return;

  int m = nvec + 1;
  std::vector<double> B(m*m, 0.0);
  for (int i = 0; i < nvec; i++)
    for (int j = 0; j <= i; j++) {
      double dot = 0.0;
      for (int k = 0; k < N*N; k++) dot += errs[i][k] * errs[j][k];
      B[i*m+j] = B[j*m+i] = dot;
    }
  for (int i = 0; i < nvec; i++) B[i*m+nvec] = B[nvec*m+i] = -1.0;
  B[nvec*m+nvec] = 0.0;
  std::vector<double> rhs(m, 0.0); rhs[nvec] = -1.0;

  auto Bc = B; auto rhsc = rhs;
  bool ok = true;
  for (int col = 0; col < m; col++) {
    int pivot = col;
    for (int r = col+1; r < m; r++)
      if (std::fabs(Bc[r*m+col]) > std::fabs(Bc[pivot*m+col])) pivot = r;
    if (std::fabs(Bc[pivot*m+col]) < 1e-20) { ok = false; break; }
    if (pivot != col) {
      for (int j = 0; j < m; j++) std::swap(Bc[col*m+j], Bc[pivot*m+j]);
      std::swap(rhsc[col], rhsc[pivot]);
    }
    for (int r = col+1; r < m; r++) {
      double f = Bc[r*m+col] / Bc[col*m+col];
      for (int j = col; j < m; j++) Bc[r*m+j] -= f * Bc[col*m+j];
      rhsc[r] -= f * rhsc[col];
    }
  }
  if (!ok) return;

  std::vector<double> coeffs(m, 0.0);
  for (int i = m-1; i >= 0; i--) {
    double s = rhsc[i];
    for (int j = i+1; j < m; j++) s -= Bc[i*m+j] * coeffs[j];
    coeffs[i] = s / Bc[i*m+i];
  }

  std::vector<double> F_diis(N*N, 0.0);
  for (int i = 0; i < nvec; i++)
    for (int j = 0; j < N*N; j++)
      F_diis[j] += coeffs[i] * focks[i][j];
  cudaMemcpy(d_Fock, F_diis.data(), N*N*sizeof(double), cudaMemcpyHostToDevice);
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

  std::vector<std::vector<double>> diis_errs_a, diis_focks_a;
  std::vector<std::vector<double>> diis_errs_b, diis_focks_b;
  int N = static_cast<int>(nao_);
  double e_prev = 0.0;

  for (iter_ = 1; iter_ <= params_.max_iter; iter_++) {
    NvtxRange iter_range("SCF iteration");
    std::vector<double> Da(N*N), Db(N*N), Da_old(N*N), Db_old(N*N);
    cudaMemcpy(Da.data(), d_D_a_, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Db.data(), d_D_b_, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Da_old.data(), d_D_old_a_, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Db_old.data(), d_D_old_b_, N*N*sizeof(double), cudaMemcpyDeviceToHost);

    if (uks_) build_fock_uks();
    else      build_fock_rks();

    std::vector<double> Fa(N*N), Fb(N*N);
    cudaMemcpy(Fa.data(), d_Fock_a_, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    if (uks_)
      cudaMemcpy(Fb.data(), d_Fock_b_, N*N*sizeof(double), cudaMemcpyDeviceToHost);

    // Energy
    if (uks_) {
      e_hcore_ = trace_dot(d_D_a_, d_Hcore_, N) + trace_dot(d_D_b_, d_Hcore_, N);
      // Rebuild D_tot = Da+Db for J energy
      double one = 1.0;
      cublasDcopy(cublas_, N*N, d_D_a_, 1, d_D_tot_, 1);
      cublasDaxpy(cublas_, N*N, &one, d_D_b_, 1, d_D_tot_, 1);
      e_j_ = 0.5 * trace_dot(d_D_tot_, d_J_, N);
      e_k_ = 0.0;
      if (xc_ && (xc_->is_hybrid() || xc_->is_hf())) {
        e_k_ = -0.5 * (trace_dot(d_D_a_, d_K_a_, N) + trace_dot(d_D_b_, d_K_b_, N));
      }
      e_kin_ = trace_dot(d_D_a_, d_T_, N) + trace_dot(d_D_b_, d_T_, N);
      e_ne_  = trace_dot(d_D_a_, d_V_, N) + trace_dot(d_D_b_, d_V_, N);
      tr_ds_ = trace_dot(d_D_a_, d_S_, N) + trace_dot(d_D_b_, d_S_, N);
    } else {
      double tr_dh = trace_dot(d_D_a_, d_Hcore_, N);
      double tr_dj = trace_dot(d_D_a_, d_J_, N);
      e_hcore_ = 2.0 * tr_dh;
      e_j_ = 2.0 * tr_dj;
      e_k_ = 0.0;
      if (xc_ && (xc_->is_hybrid() || xc_->is_hf()))
        e_k_ = -trace_dot(d_D_a_, d_K_a_, N);
      e_kin_ = 2.0 * trace_dot(d_D_a_, d_T_, N);
      e_ne_  = 2.0 * trace_dot(d_D_a_, d_V_, N);
      tr_ds_ = trace_dot(d_D_a_, d_S_, N);
    }
    e_elec_ = e_hcore_ + e_j_ + e_k_ + e_xc_;
    e_total_ = e_elec_ + e_nuc_;
    double dE = (iter_ > 1) ? (e_total_ - e_prev) : 0.0;

    double rms_a = (iter_ > 1) ? compute_rms_delta(Da, Da_old) : 1.0;
    double rms_b = (uks_ && iter_ > 1) ? compute_rms_delta(Db, Db_old) : 0.0;
    double rms_d = uks_ ? std::sqrt(0.5*(rms_a*rms_a + rms_b*rms_b)) : rms_a;

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
      diis_extrapolate(diis_errs_a, diis_focks_a, Fa, Da, d_Fock_a_);
      if (uks_)
        diis_extrapolate(diis_errs_b, diis_focks_b, Fb, Db, d_Fock_b_);
    }

    cudaMemcpy(d_D_old_a_, Da.data(), N*N*sizeof(double), cudaMemcpyHostToDevice);
    if (uks_)
      cudaMemcpy(d_D_old_b_, Db.data(), N*N*sizeof(double), cudaMemcpyHostToDevice);

    diagonalize_fock(d_Fock_a_, d_C_a_, mo_energies_a_);
    build_density(d_C_a_, uks_ ? nocc_a_ : nocc_, d_D_a_);
    if (uks_) {
      diagonalize_fock(d_Fock_b_, d_C_b_, mo_energies_b_);
      build_density(d_C_b_, nocc_b_, d_D_b_);
    } else {
      cublasDcopy(cublas_, N*N, d_C_a_, 1, d_C_b_, 1);
      cublasDcopy(cublas_, N*N, d_D_a_, 1, d_D_b_, 1);
      mo_energies_b_ = mo_energies_a_;
    }

    // DIIS trust-region: reject unphysical densities and reset the DIIS
    // subspace so poisoned error vectors cannot keep the SCF oscillating.
    if (iter_ >= params_.diis_start + 2) {
      double trA = trace_dot(d_D_a_, d_S_, N);
      double expect = uks_ ? static_cast<double>(nocc_a_) : static_cast<double>(nocc_);
      if (std::abs(trA - expect) > 0.5 * std::max(expect, 1.0)) {
        if (params_.print_level >= 2)
          std::cout << "  DIIS rejected (Tr[Da*S]=" << trA << "), using raw Fock\n";
        cudaMemcpy(d_Fock_a_, Fa.data(), N*N*sizeof(double), cudaMemcpyHostToDevice);
        diagonalize_fock(d_Fock_a_, d_C_a_, mo_energies_a_);
        build_density(d_C_a_, uks_ ? nocc_a_ : nocc_, d_D_a_);
        diis_errs_a.clear();
        diis_focks_a.clear();
        if (uks_) {
          cudaMemcpy(d_Fock_b_, Fb.data(), N*N*sizeof(double), cudaMemcpyHostToDevice);
          diagonalize_fock(d_Fock_b_, d_C_b_, mo_energies_b_);
          build_density(d_C_b_, nocc_b_, d_D_b_);
          diis_errs_b.clear();
          diis_focks_b.clear();
        }
      }
    }

    if (params_.damping > 0.0) {
      double damp = params_.damping;
      double one_minus = 1.0 - damp;
      cublasDscal(cublas_, N*N, &one_minus, d_D_a_, 1);
      cublasDaxpy(cublas_, N*N, &damp, d_D_old_a_, 1, d_D_a_, 1);
      if (uks_) {
        cublasDscal(cublas_, N*N, &one_minus, d_D_b_, 1);
        cublasDaxpy(cublas_, N*N, &damp, d_D_old_b_, 1, d_D_b_, 1);
      }
    }

    e_prev = e_total_;

    // Periodic DIIS flush: CH4/STO-3G and similar cases can oscillate near
    // convergence when a poisoned DIIS subspace never triggers the Tr[DS]
    // trust check. Flushing regularly recovers without global damping.
    if (iter_ >= 25 && (iter_ % 15) == 0 && !converged_) {
      diis_errs_a.clear();
      diis_focks_a.clear();
      diis_errs_b.clear();
      diis_focks_b.clear();
      if (params_.print_level >= 2)
        std::cout << "  DIIS subspace flushed at iter " << iter_ << "\n";
    }
  }

  form_total_density();
  if (!uks_) {
    // RKS reporting: D_tot = 2*D_alpha already done in form_total_density
  } else {
    double one = 1.0;
    cublasDcopy(cublas_, N*N, d_D_a_, 1, d_D_tot_, 1);
    cublasDaxpy(cublas_, N*N, &one, d_D_b_, 1, d_D_tot_, 1);
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
