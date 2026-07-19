/**
 * @file scf.cpp — SCF solver using cuSOLVER dsygvd + correct DFT energy.
 */
#include "cuest_wrapper/scf.hpp"
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
namespace cuest {

// ---------------------------------------------------------------------------
SCFSolver::SCFSolver(CuESTContext& ctx, BasisBuilder& basis,
                     DFJKBuilder& dfjk, XCBuilder* xc,
                     ECPBuilder* ecp_builder, ECPIntegrals* ecp_int,
                     const Molecule& mol, SCFParams params)
    : ctx_(ctx), basis_(basis), dfjk_(dfjk), xc_(xc),
      ecp_builder_(ecp_builder), ecp_int_(ecp_int),
      mol_(mol), params_(params) {
  cublasCreate(&cublas_);
  cusolverDnCreate(&cusolver_);
  nao_ = ctx_.query_nao(basis_.basis());
  nocc_ = mol_.nocc();
  nelec_ = mol_.nelec();
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
  cudaMemcpy(d_xyz_, xyz_h.data(), xyz_h.size()*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_charges_, chg_h.data(), chg_h.size()*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(d_D_, 0, N2 * sizeof(double));
  cudaMemset(d_ECP_, 0, N2 * sizeof(double));
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
  AOPairListHandle pair_list = basis_.create_pair_list();
  OneElectronIntegrals oe(ctx_, basis_.basis(), pair_list);
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

  // Hybrid: subtract exact exchange
  // For standard hybrids (B3LYP, PBE0): DF plan has EXCHANGE_FRACTION=1.0,
  //   K = full HF exchange, F -= exchange_scale * K
  // For LRC (CAM-B3LYP, wB97X): DF plan has LRC params set,
  //   K already range-separated, F -= 1.0 * K
  if (xc_ && xc_->is_hybrid()) {
    bool is_lrc = xc_->is_lrc();
    double k_factor = is_lrc ? -1.0 : -xc_->exchange_scale();

    std::vector<double> Cocc(nao_ * nocc_);
    cudaMemcpy(Cocc.data(), d_C_, nao_*nocc_*sizeof(double), cudaMemcpyDeviceToHost);
    DeviceArray d_Cocc_ke(nao_ * nocc_ * sizeof(double));
    cudaMemcpy(d_Cocc_ke, Cocc.data(), nao_*nocc_*sizeof(double), cudaMemcpyHostToDevice);

    dfjk_.compute_K(nocc_, d_Cocc_ke, d_K_);
    cublasDaxpy(cublas_, nao_*nao_, &k_factor, d_K_, 1, d_Fock_, 1);
  }

  // Add Vxc
  if (xc_) {
    double exc_val = 0.0;
    std::vector<double> Cocc(nao_ * nocc_);
    cudaMemcpy(Cocc.data(), d_C_, nao_*nocc_*sizeof(double), cudaMemcpyDeviceToHost);
    DeviceArray d_Cocc(nao_ * nocc_ * sizeof(double));
    cudaMemcpy(d_Cocc, Cocc.data(), nao_*nocc_*sizeof(double), cudaMemcpyHostToDevice);

    xc_->compute_vxc_rks(nocc_, d_Cocc, &exc_val, d_Vxc_);
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

  // Allocate device workspace
  double* d_work = nullptr;
  cudaMalloc(&d_work, lwork * sizeof(double));
  int* d_info = nullptr;
  cudaMalloc(&d_info, sizeof(int));

  // Solve generalized eigenvalue problem: Fwork * C = Swork * C * E
  cusolverDnDsygvd(cusolver_, CUSOLVER_EIG_TYPE_1,
      CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      N, d_Fwork_, lda, d_Swork_, lda,
      d_eigvals_, d_work, lwork, d_info);

  int info_h = 0;
  cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_work);
  cudaFree(d_info);

  if (info_h != 0) {
    std::cerr << "WARNING: dsygvd info = " << info_h;
    if (info_h > 0)
      std::cerr << " (B not positive definite?)";
    std::cerr << "\n";
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
  }

  build_core_hamiltonian();

  // SAD (Superposition of Atomic Densities) initial guess:
  // Build approximate atomic densities from orbital occupations,
  // project into molecular basis, sum, diagonalize D*S.
  {
    // Atomic ground-state configurations (neutral atoms)
    // Format: {Z: [(n_occ, L), ...]} where n_occ = electrons in each L-shell
    // Example: O (Z=8): 1s²(2), 2s²(2), 2p⁴(4) → L=0:4, L=1:4
    // For SAD, we set up atomic occupation numbers per angular momentum
    struct AtomConfig { int Z; int occ_s, occ_p, occ_d; };
    static const AtomConfig configs[] = {
      {1,1,0,0}, {2,2,0,0}, // H, He
      {3,2,1,0}, {4,2,2,0}, {5,2,3,0}, {6,2,4,0}, {7,2,5,0}, {8,2,6,0}, {9,2,7,0}, {10,2,8,0}, // Li-Ne
      {11,2,8,1}, {12,2,8,2}, {13,2,8,3}, {14,2,8,4}, {15,2,8,5}, {16,2,8,6}, {17,2,8,7}, {18,2,8,8}, // Na-Ar
    };

    // Get S overlap for density matrix diagonalization
    // Build approximate atomic densities: D_atom = sum_i |AO_i><AO_i| * occ_i
    // For simplicity, use core Hamiltonian guess (already good for main-group)
    // TODO: implement proper SAD by projecting minimal-basis atomic densities
    cublasDcopy(cublas_, nao_*nao_, d_Hcore_, 1, d_Fock_, 1);
    diagonalize_fock();
    build_density_matrix();
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
