/**
 * @file scf.cpp — SCF solver using cuSOLVER dsygvd + correct DFT energy.
 */
#include "cuest_wrapper/scf.hpp"
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
namespace cuest {

// ---------------------------------------------------------------------------
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
  nao_ = ctx_.query_nao(basis_.basis());
  // Molecule::nelec() already accounts for ECP (Z_eff) and molecular charge
  nelec_ = mol_.nelec();
  if (nelec_ < 0)
    throw std::runtime_error("Negative electron count after charge/ECP");
  if (nelec_ % 2 != 0)
    throw std::runtime_error(
        "Odd electron count — only closed-shell RKS is supported");
  nocc_ = static_cast<uint64_t>(mol_.nocc());
  e_nuc_ = mol_.nuclear_repulsion();

  size_t N2 = nao_ * nao_;
  d_Hcore_.alloc(N2 * sizeof(double));
  d_S_.alloc(N2 * sizeof(double));
  d_T_.alloc(N2 * sizeof(double));
  d_V_.alloc(N2 * sizeof(double));
  d_ECP_.alloc(N2 * sizeof(double));
  d_Fock_.alloc(N2 * sizeof(double));
  d_D_.alloc(N2 * sizeof(double));
  d_D_old_.alloc(N2 * sizeof(double));
  d_C_.alloc(N2 * sizeof(double));
  d_J_.alloc(N2 * sizeof(double));
  d_K_.alloc(N2 * sizeof(double));
  d_Vxc_.alloc(N2 * sizeof(double));
  d_eigvals_.alloc(nao_ * sizeof(double));
  d_Fwork_.alloc(N2 * sizeof(double));
  d_Swork_.alloc(N2 * sizeof(double));

  auto xyz_h = mol_.xyz_host();
  auto chg_h = mol_.charges_host();
  d_xyz_.alloc(xyz_h.size() * sizeof(double));
  d_charges_.alloc(chg_h.size() * sizeof(double));
  CUDA_CHECK(cudaMemcpy(d_xyz_, xyz_h.data(), xyz_h.size()*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_charges_, chg_h.data(), chg_h.size()*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D_, 0, N2 * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_ECP_, 0, N2 * sizeof(double)));
}

SCFSolver::~SCFSolver() {
  if (cublas_) cublasDestroy(cublas_);
  if (cusolver_) cusolverDnDestroy(cusolver_);
}

// ---------------------------------------------------------------------------
// Trace of product: Tr[A^T * B] = sum_ij A_ji * B_ji
// For symmetric A: Tr[A * B] = sum_ij A_ij * B_ji
// ---------------------------------------------------------------------------
double SCFSolver::trace_dot(const double* d_A, const double* d_B, int N) {
  // Use cuBLAS dot: Tr[A * B] = sum(d_A * d_B^T) if B is row-major
  // d_A and d_B are column-major (Fortran convention from cuSOLVER)
  // Tr[A*B] = sum_i sum_j A_ij * B_ji
  // We can compute: cublasDdot(N*N, d_A, 1, d_B_transposed, 1) where
  // d_B_transposed(i,j) = d_B(j,i)
  // Or just copy to host for small matrices:
  std::vector<double> A(N*N), B(N*N);
  cudaMemcpy(A.data(), d_A, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(B.data(), d_B, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  double tr = 0.0;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      tr += A[i*N+j] * B[j*N+i];
  return tr;
}

// ---------------------------------------------------------------------------
// Core Hamiltonian: Hcore = T + V (+ V_ECP if present)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Fock matrix: F = Hcore + 2*J - a*K + Vxc  (supports hybrids)
// ---------------------------------------------------------------------------
void SCFSolver::build_fock_matrix(const std::vector<double>& D_host) {
  cudaMemcpy(d_D_, D_host.data(), nao_*nao_*sizeof(double), cudaMemcpyHostToDevice);

  // J matrix from density
  dfjk_.compute_J(d_D_, d_J_);

  // F = Hcore + 2*J
  double two = 2.0;
  cublasDcopy(cublas_, nao_*nao_, d_Hcore_, 1, d_Fock_, 1);
  cublasDaxpy(cublas_, nao_*nao_, &two, d_J_, 1, d_Fock_, 1);

  // Read Cocc from device ONCE, reuse for both K and Vxc
  DeviceArray d_Cocc_dev;  // may remain empty if not needed
  bool cocc_cached = false;

  // Hybrid: subtract exact exchange
  // For standard hybrids (B3LYP, PBE0): DF plan has EXCHANGE_FRACTION=1.0,
  //   K = full HF exchange, F -= exchange_scale * K
  // For LRC (CAM-B3LYP, wB97X): DF plan has LRC params set,
  //   K already range-separated, F -= 1.0 * K
  if (xc_ && xc_->is_hybrid()) {
    bool is_lrc = xc_->is_lrc();
    double k_factor = is_lrc ? -1.0 : -xc_->exchange_scale();

    // Cache Cocc from device (reused below for Vxc)
    std::vector<double> Cocc(nao_ * nocc_);
    cudaMemcpy(Cocc.data(), d_C_, nao_*nocc_*sizeof(double), cudaMemcpyDeviceToHost);
    d_Cocc_dev.alloc(nao_ * nocc_ * sizeof(double));
    cudaMemcpy(d_Cocc_dev, Cocc.data(), nao_*nocc_*sizeof(double), cudaMemcpyHostToDevice);
    cocc_cached = true;

    dfjk_.compute_K(nocc_, d_Cocc_dev, d_K_);
    cublasDaxpy(cublas_, nao_*nao_, &k_factor, d_K_, 1, d_Fock_, 1);
  }

  // Add Vxc
  if (xc_) {
    double exc_val = 0.0;
    if (!cocc_cached) {
      std::vector<double> Cocc(nao_ * nocc_);
      cudaMemcpy(Cocc.data(), d_C_, nao_*nocc_*sizeof(double), cudaMemcpyDeviceToHost);
      d_Cocc_dev.alloc(nao_ * nocc_ * sizeof(double));
      cudaMemcpy(d_Cocc_dev, Cocc.data(), nao_*nocc_*sizeof(double), cudaMemcpyHostToDevice);
    }
    xc_->compute_vxc_rks(nocc_, d_Cocc_dev, &exc_val, d_Vxc_);
    e_xc_ = exc_val;

    double one = 1.0;
    cublasDaxpy(cublas_, nao_*nao_, &one, d_Vxc_, 1, d_Fock_, 1);
  }
}

// ---------------------------------------------------------------------------
// Diagonalize: FC = SCE using cuSOLVER dsygvd (GPU, generalized symmetric)
// ---------------------------------------------------------------------------
void SCFSolver::diagonalize_fock() {
  int N = static_cast<int>(nao_);
  int lda = N;

  // Copy Fock and Overlap to work arrays (dsygvd destroys inputs)
  cublasDcopy(cublas_, N*N, d_Fock_, 1, d_Fwork_, 1);
  cublasDcopy(cublas_, N*N, d_S_, 1, d_Swork_, 1);

  // Query workspace size
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

  // d_Fwork_ now contains the eigenvectors C (column-major)
  // d_eigvals_ contains eigenvalues
  // Copy eigenvectors to d_C_
  cublasDcopy(cublas_, N*N, d_Fwork_, 1, d_C_, 1);

  // Get eigenvalues
  mo_energies_.resize(N);
  cudaMemcpy(mo_energies_.data(), d_eigvals_, N*sizeof(double), cudaMemcpyDeviceToHost);
}

// ---------------------------------------------------------------------------
// Density matrix: D = C_occ * C_occ^T  (for RKS)
// ---------------------------------------------------------------------------
void SCFSolver::build_density_matrix() {
  int N = static_cast<int>(nao_);
  int Nocc = static_cast<int>(nocc_);
  double one = 1.0, zero = 0.0;

  // D = C_occ * C_occ^T (D_alpha, Tr[D*S] = nocc = nelec/2)
  // cuEST Fock matrix uses D_alpha convention: F = Hcore + 2*J + Vxc
  cublasDgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
              N, N, Nocc, &one,
              d_C_, N, d_C_, N,
              &zero, d_D_, N);

  double trDS = trace_dot(d_D_, d_S_, N);
  if (params_.print_level >= 2) {
    std::cout << "  Tr[D*S] = " << std::fixed << std::setprecision(8)
              << trDS << " (expected " << nocc_ << ")\n";
  }
}

// ---------------------------------------------------------------------------
double SCFSolver::compute_rms_delta(const std::vector<double>& D_new,
                                     const std::vector<double>& D_old) {
  double sum = 0.0;
  for (size_t i = 0; i < D_new.size(); i++) {
    double diff = D_new[i] - D_old[i];
    sum += diff * diff;
  }
  return std::sqrt(sum / D_new.size());
}

// ---------------------------------------------------------------------------
// Main SCF loop
// ---------------------------------------------------------------------------
void SCFSolver::run() {
  if (params_.verbose) {
    std::cout << "\n=== Starting SCF Calculation ===\n"
              << "  Basis functions: " << nao_ << "\n"
              << "  Occupied orbitals: " << nocc_ << "\n"
              << "  Electrons: " << nelec_ << "\n"
              << "  Nuclear repulsion: " << std::setprecision(12) << e_nuc_
              << " Ha\n";
    if (xc_) std::cout << "  XC: enabled"
                       << (xc_->is_hybrid() ? " (hybrid)" : "") << "\n";
    if (params_.damping > 0.0)
      std::cout << "  Damping: " << params_.damping << "\n";
    if (mol_.charge() != 0)
      std::cout << "  Charge: " << mol_.charge() << "\n";
    if (mol_.total_ecp_electrons() > 0)
      std::cout << "  ECP electrons removed: " << mol_.total_ecp_electrons() << "\n";
  }

  build_core_hamiltonian();

  // Initial guess.
  // Hcore guess for ECP systems (SAD unreliable with valence-only basis).
  // SAD guess for all-electron (much better for heavy atoms like Br).
  bool use_hcore_guess = (mol_.total_ecp_electrons() > 0);
  int N_guess = static_cast<int>(nao_);

  if (use_hcore_guess) {
    cublasDcopy(cublas_, N_guess*N_guess, d_Hcore_, 1, d_Fock_, 1);
    diagonalize_fock();
    build_density_matrix();
    if (params_.print_level >= 2)
      std::cout << "  Hcore initial guess: Tr[D*S] = " << std::fixed
                << std::setprecision(6) << trace_dot(d_D_, d_S_, N_guess) << "\n";
  } else {
    // SAD: superposition of atomic densities
    std::vector<double> S_h(N_guess*N_guess);
    cudaMemcpy(S_h.data(), d_S_, N_guess*N_guess*sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<double> D_sad(N_guess*N_guess, 0.0);
    std::vector<int> ao_per_atom(mol_.natom(), 0);
    std::vector<int> ao_start(mol_.natom(), 0);

    int ao_fill = 0;
    for (size_t a = 0; a < mol_.natom(); a++) {
      ao_start[a] = ao_fill;
      int Z = mol_.atom(a).atomic_number;
      int nao = (Z <= 2) ? 5 : 14;
      if (Z > 10) nao = (Z <= 18) ? 18 : 27;
      if (Z > 36) nao = 32;
      ao_per_atom[a] = std::min(nao, N_guess - ao_fill);
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
      int start = ao_start[a];
      int count = ao_per_atom[a];
      if (count <= 0) continue;
      double occ_per_ao = static_cast<double>(nv) / count * 0.5;
      for (int i = start; i < start + count && i < N_guess; i++)
        D_sad[i*N_guess + i] = occ_per_ao;
    }

    double trDS = 0.0;
    for (int i = 0; i < N_guess; i++)
      for (int j = 0; j < N_guess; j++)
        trDS += D_sad[i*N_guess + j] * S_h[j*N_guess + i];
    double scale = nocc_ / std::max(trDS, 1e-10);
    for (auto& d : D_sad) d *= scale;

    std::vector<double> F_init(N_guess*N_guess);
    cudaMemcpy(F_init.data(), d_Hcore_, N_guess*N_guess*sizeof(double), cudaMemcpyDeviceToHost);
    const double mix = 0.5;
    for (int i = 0; i < N_guess*N_guess; i++) F_init[i] += mix * D_sad[i];
    cudaMemcpy(d_Fock_, F_init.data(), N_guess*N_guess*sizeof(double), cudaMemcpyHostToDevice);

    diagonalize_fock();
    build_density_matrix();
    if (params_.print_level >= 2)
      std::cout << "  SAD initial guess: Tr[D*S] = " << std::fixed
                << std::setprecision(6) << trace_dot(d_D_, d_S_, N_guess) << "\n";
  }

  // DIIS arrays
  std::vector<std::vector<double>> diis_errs, diis_focks;
  int N = static_cast<int>(nao_);
  double e_prev = 0.0;

  for (iter_ = 1; iter_ <= params_.max_iter; iter_++) {
    // Get current density
    std::vector<double> D_curr(N*N), D_old(N*N);
    cudaMemcpy(D_curr.data(), d_D_, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(D_old.data(), d_D_old_, N*N*sizeof(double), cudaMemcpyDeviceToHost);

    // Build Fock from density
    build_fock_matrix(D_curr);

    // Extract Fock and Hcore for energy
    std::vector<double> F_host(N*N), Hcore_host(N*N);
    cudaMemcpy(F_host.data(), d_Fock_, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Hcore_host.data(), d_Hcore_, N*N*sizeof(double), cudaMemcpyDeviceToHost);

    // === Energy for RKS: D = D_alpha, need factor of 2 for total ===
    // E_elec = 2*Tr[D*Hcore] + 2*Tr[D*J] + Exc
    // (since F = Hcore + 2*J + Vxc and eigenvalues match, D = D_alpha)
    double tr_dh = trace_dot(d_D_, d_Hcore_, N);
    double tr_dj = trace_dot(d_D_, d_J_, N);
    e_elec_ = 2.0 * (tr_dh + tr_dj) + e_xc_;
    bool hybrid = (xc_ && xc_->is_hybrid());
    if (hybrid) {
      double k_energy_factor = (xc_ && xc_->is_lrc()) ? 1.0 : xc_->exchange_scale();
      double tr_dk = trace_dot(d_D_, d_K_, N);
      e_elec_ -= k_energy_factor * tr_dk;
    }
    e_total_ = e_elec_ + e_nuc_;
    double dE = (iter_ > 1) ? (e_total_ - e_prev) : 0.0;

    // RMS density change
    double rms_d = (iter_ > 1) ? compute_rms_delta(D_curr, D_old) : 1.0;

    if (params_.print_level >= 1) {
      std::cout << "  Iter " << std::setw(3) << iter_
                << "  Etot = " << std::setw(16) << std::setprecision(10) << std::fixed << e_total_
                << "  dE = " << std::setw(12) << std::scientific << dE
                << "  rmsD = " << std::setw(10) << rms_d;
      if (xc_) std::cout << "  Exc = " << std::setw(14) << std::fixed << std::setprecision(8) << e_xc_;
      std::cout << "\n" << std::flush;
    }

    // Check convergence
    if (iter_ > 1 && rms_d < params_.conv_thresh &&
        std::abs(dE) < params_.energy_conv_thresh) {
      converged_ = true;
      if (params_.verbose)
        std::cout << "\n  SCF converged in " << iter_ << " iterations\n";
      break;
    }

    // DIIS
    if (iter_ >= params_.diis_start) {
      // Error = FDS - SDF
      std::vector<double> S_host(N*N);
      cudaMemcpy(S_host.data(), d_S_, N*N*sizeof(double), cudaMemcpyDeviceToHost);
      std::vector<double> FD(N*N, 0.0), FDS(N*N, 0.0), SD(N*N, 0.0), SDF(N*N, 0.0);
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          double fd = 0.0, sd = 0.0;
          for (int k = 0; k < N; k++) {
            fd += F_host[i*N+k] * D_curr[k*N+j];
            sd += S_host[i*N+k] * D_curr[k*N+j];
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

      if (static_cast<int>(diis_errs.size()) >= params_.diis_max_space) {
        diis_errs.erase(diis_errs.begin());
        diis_focks.erase(diis_focks.begin());
      }
      diis_errs.push_back(err);
      diis_focks.push_back(F_host);

      // Solve DIIS (small linear system)
      int nvec = diis_errs.size();
      if (nvec >= 2) {
        int m = nvec + 1;
        std::vector<double> B(m*m, 0.0);
        for (int i = 0; i < nvec; i++)
          for (int j = 0; j <= i; j++) {
            double dot = 0.0;
            for (int k = 0; k < N*N; k++) dot += diis_errs[i][k] * diis_errs[j][k];
            B[i*m+j] = B[j*m+i] = dot;
          }
        for (int i = 0; i < nvec; i++) B[i*m+nvec] = B[nvec*m+i] = -1.0;
        B[nvec*m+nvec] = 0.0;
        std::vector<double> rhs(m, 0.0); rhs[nvec] = -1.0;

        // Solve via Gaussian elimination
        auto Bc = B; auto rhsc = rhs;
        for (int col = 0; col < m; col++) {
          int pivot = col;
          for (int r = col+1; r < m; r++)
            if (fabs(Bc[r*m+col]) > fabs(Bc[pivot*m+col])) pivot = r;
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
        std::vector<double> coeffs(m, 0.0);
        for (int i = m-1; i >= 0; i--) {
          double s = rhsc[i];
          for (int j = i+1; j < m; j++) s -= Bc[i*m+j] * coeffs[j];
          coeffs[i] = s / Bc[i*m+i];
        }

        // Use raw Pulay coefficients; trust-region below rejects bad DIIS.
        std::vector<double> F_diis(N*N, 0.0);
        for (int i = 0; i < nvec; i++)
          for (int j = 0; j < N*N; j++)
            F_diis[j] += coeffs[i] * diis_focks[i][j];

        cudaMemcpy(d_Fock_, F_diis.data(), N*N*sizeof(double), cudaMemcpyHostToDevice);
      }
    }

    // Save old density
    cudaMemcpy(d_D_old_, D_curr.data(), N*N*sizeof(double), cudaMemcpyHostToDevice);

    // Diagonalize Fock to get new MOs
    diagonalize_fock();

    // Build new density
    build_density_matrix();

    // DIIS trust-region: if DIIS overshoots (Tr[D*S] far from nocc),
    // revert to the raw Fock matrix for this iteration.
    if (iter_ >= params_.diis_start + 2) {
      double trDS = trace_dot(d_D_, d_S_, N);
      if (std::abs(trDS - nocc_) > 0.5 * nocc_) {
        if (params_.print_level >= 2)
          std::cout << "  DIIS rejected (Tr[D*S]=" << trDS
                    << "), using raw Fock\n";
        cudaMemcpy(d_Fock_, F_host.data(), N*N*sizeof(double), cudaMemcpyHostToDevice);
        diagonalize_fock();
        build_density_matrix();
      }
    }

    // Density damping: D = (1-damping)*D_new + damping*D_old
    if (params_.damping > 0.0) {
      double damp = params_.damping;
      double one_minus = 1.0 - damp;
      // d_D_ currently holds D_new; d_D_old_ holds D_curr (previous iteration)
      // D_damped = one_minus * D_new + damp * D_old
      cublasDscal(cublas_, N*N, &one_minus, d_D_, 1);
      cublasDaxpy(cublas_, N*N, &damp, d_D_old_, 1, d_D_, 1);
      if (params_.print_level >= 2) {
        std::cout << "  Damping: " << std::fixed << std::setprecision(3)
                  << damp << "\n";
      }
    }

    e_prev = e_total_;
  }

  // Final output
  if (params_.verbose) {
    std::cout << "\n=== SCF Results ===\n"
              << "Converged: " << (converged_ ? "Yes" : "No") << "\n"
              << "Iterations: " << iter_ << "\n"
              << std::setprecision(12) << std::fixed
              << "Nuclear repulsion: " << e_nuc_ << " Ha\n"
              << "Electronic energy: " << e_elec_ << " Ha\n";
    if (xc_) std::cout << "XC energy: " << e_xc_ << " Ha\n";
    std::cout << "Total energy:      " << e_total_ << " Ha\n";

    if (params_.print_mos && !mo_energies_.empty()) {
      std::cout << "\nOrbital energies (Ha):\n";
      for (size_t i = 0; i < mo_energies_.size(); i++) {
        std::cout << "  MO " << std::setw(3) << i << ": "
                  << std::setw(14) << mo_energies_[i];
        if (static_cast<int>(i) == static_cast<int>(nocc_) - 1)
          std::cout << "  (HOMO)";
        else if (static_cast<int>(i) == static_cast<int>(nocc_))
          std::cout << "  (LUMO)";
        std::cout << "\n";
      }
    }
  }
}

}  // namespace cuest
