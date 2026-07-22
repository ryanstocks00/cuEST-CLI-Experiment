#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>
#include "cuest_wrapper/basis.hpp"
#include "cuest_wrapper/context.hpp"
#include "cuest_wrapper/integrals.hpp"
#include "cuest_wrapper/raii.hpp"
#include "diis.hpp"

namespace cuest {

struct SADGuessConfig;

struct SCFParams {
  int max_iter{250};
  /// Threshold on the orbital-gradient 2-norm ‖g‖ = ‖2 C_vir^T F C_occ‖₂ — the
  /// same quantity PySCF converges on (its `conv_tol_grad`), and the derivative
  /// of the energy with respect to occupied→virtual rotations.
  ///
  /// This replaced a threshold on the RMS of the DIIS commutator FDS-SDF. Both
  /// vanish at the solution and both are insensitive to harmless rotation
  /// within a degenerate occupied subspace (unlike a raw density RMS), but the
  /// commutator RMS divides by nao² while the signal lives in the nocc·nvir
  /// occupied-virtual block. Enlarging the basis at fixed molecule therefore
  /// shrinks it like sqrt(nocc/nao), so a fixed threshold silently gets
  /// *looser* for better bases — backwards. The gradient norm has no such
  /// arbitrary denominator and relates directly to the energy error, which
  /// goes as ‖g‖²/gap because the energy is stationary.
  ///
  /// FDS-SDF is still what DIIS extrapolates on; only the convergence test
  /// changed. See SCFSolver::orbital_gradient_norm.
  ///
  /// The default follows PySCF's rule of thumb, grad tolerance =
  /// sqrt(energy tolerance): the energy is stationary, so its error goes as
  /// ‖g‖². With energy_conv_thresh = 1e-8 that gives 1e-4, which measured
  /// across H2O/C2H4/SO2/I2 and open-shell OH lands every energy within 5e-8 Ha
  /// — two orders inside a 1e-6 target.
  ///
  /// A *force* error is only first order in ‖g‖, so a tighter threshold might
  /// be expected to matter for the analytic gradients (validated at 5e-4). It
  /// does not: measured on the worst gradient cases, tightening to 1e-5 moves
  /// the gradient RMS by under 3% (9.0e-5 → 8.8e-5). At this level the error
  /// against PySCF is dominated by XC-grid and density-fitting differences, not
  /// by SCF convergence, so the extra iterations buy nothing.
  double conv_thresh{1e-4};
  double energy_conv_thresh{1e-8};
  int diis_start{1};
  /// Large history windows make the Pulay DIIS solve numerically fragile
  /// for near-degenerate open-shell systems (nearly-linearly-dependent
  /// history vectors amplify floating-point noise into macroscopically
  /// different iteration counts); 6 is fast and reproducible across the
  /// validated systems, including hard radical cases, without regressing
  /// well-behaved closed-shell molecules.
  int diis_max_space{6};
  double damping{0.0};
  bool verbose{true};
  bool print_mos{false};
  int print_level{2};
  /// Mix angle (radians) for HOMO–LUMO symmetry breaking on the β guess.
  /// Applied only when the α and β occupations are equal (an artificially
  /// closed-shell UKS guess) — a genuinely spin-polarised system has no
  /// symmetry to break. See SCFSolver::break_beta_symmetry.
  ///
  /// It cannot simply be disabled: at 0 a stretched H2 under `unrestricted`
  /// stays on the restricted solution (-0.9372 Ha) instead of finding the
  /// broken-symmetry one (-0.9979 Ha). But 0.3 over-perturbs systems that have
  /// no broken-symmetry solution and must then relax back out of it — 0.1
  /// finds the same solutions with roughly a third of the iterations
  /// (H2O 38 -> 15, stretched H2 190 -> 47).
  double break_symmetry{0.1};
  /// Solve unrestricted (UKS) even at multiplicity 1. Multiplicity > 1 is
  /// always unrestricted; this only adds the case where the two are separable —
  /// a broken-symmetry singlet, homolytic dissociation, antiferromagnetic
  /// coupling. Combined with break_symmetry it is what lets a spin-symmetric
  /// guess relax into a broken-symmetry solution.
  bool unrestricted{false};
  /// Use cuEST JIT kernels (DF-J, nuclear potential, …). Off ⇒ AOT + fp64.
  bool use_jit{true};
  /// Force the zero-density (core Hamiltonian) initial guess, skipping SAD.
  /// Exposed on the CLI as --hcore-guess, mainly so the SAD guess can be A/B'd
  /// against the guess it replaced.
  bool force_hcore_guess{false};
};

class SCFSolver {
 public:
  /// `sad_config`, if non-null, enables the superposition-of-atomic-densities
  /// initial guess (see sad_guess.hpp); must outlive this SCFSolver. Pass
  /// nullptr (or set params.force_hcore_guess) to fall back to the
  /// core-Hamiltonian guess.
  SCFSolver(CuESTContext& ctx, BasisBuilder& basis,
            DFJKBuilder& dfjk, XCBuilder* xc,
            ECPBuilder* ecp_builder, ECPIntegrals* ecp_int,
            const Molecule& mol, SCFParams params,
            const SADGuessConfig* sad_config = nullptr);
  ~SCFSolver();

  void run();
  double total_energy() const { return e_total_; }
  double nuclear_repulsion() const { return e_nuc_; }
  double electronic_energy() const { return e_elec_; }
  double xc_energy() const { return e_xc_; }
  double hcore_energy() const { return e_hcore_; }
  double kinetic_energy() const { return e_kin_; }
  double nuclear_attraction_energy() const { return e_ne_; }
  double coulomb_energy() const { return e_j_; }
  double exchange_energy() const { return e_k_; }
  double tr_ds() const { return tr_ds_; }
  bool is_uks() const { return uks_; }
  uint64_t nocc_alpha() const { return nocc_a_; }
  uint64_t nocc_beta() const { return nocc_b_; }
  /// RKS: nocc. UKS: nocc_alpha (for gradient APIs that expect one channel).
  uint64_t nocc() const { return uks_ ? nocc_a_ : nocc_; }
  const std::vector<double>& orbital_energies() const { return mo_energies_a_; }
  const std::vector<double>& orbital_energies_beta() const { return mo_energies_b_; }
  std::vector<double> mo_coefficients_host() {
    std::vector<double> C(nao_ * nao_);
    CUDA_CHECK(cudaMemcpy(C.data(), d_C_a_.get(), nao_*nao_*sizeof(double),
                          cudaMemcpyDeviceToHost));
    return C;
  }
  std::vector<double> mo_coefficients_beta_host() {
    std::vector<double> C(nao_ * nao_);
    CUDA_CHECK(cudaMemcpy(C.data(), d_C_b_.get(), nao_*nao_*sizeof(double),
                          cudaMemcpyDeviceToHost));
    return C;
  }
  /// Spin-α density (RKS: D_alpha; UKS: D_alpha). Gradients use D_alpha convention.
  std::vector<double> density_host() {
    std::vector<double> D(nao_ * nao_);
    CUDA_CHECK(cudaMemcpy(D.data(), d_D_a_.get(), nao_*nao_*sizeof(double),
                          cudaMemcpyDeviceToHost));
    return D;
  }
  std::vector<double> density_alpha_host() { return density_host(); }
  std::vector<double> density_beta_host() {
    std::vector<double> D(nao_ * nao_);
    CUDA_CHECK(cudaMemcpy(D.data(), d_D_b_.get(), nao_*nao_*sizeof(double),
                          cudaMemcpyDeviceToHost));
    return D;
  }
  std::vector<double> density_total_host() {
    std::vector<double> D(nao_ * nao_);
    CUDA_CHECK(cudaMemcpy(D.data(), d_D_tot_.get(), nao_*nao_*sizeof(double),
                          cudaMemcpyDeviceToHost));
    return D;
  }
  uint64_t nao() const { return nao_; }
  int n_iter() const { return iter_; }
  bool converged() const { return converged_; }

 private:
  void build_core_hamiltonian();
  void build_fock_rks();
  void build_fock_uks();
  void diagonalize_fock(DeviceArray<double>& d_Fock, DeviceArray<double>& d_C,
                        std::vector<double>& mo_energies);
  void build_density(DeviceArray<double>& d_C, uint64_t nocc, DeviceArray<double>& d_D);
  /// 2-norm of the occupied→virtual orbital-gradient block for one spin
  /// channel: ‖2 · C_vir^T F C_occ‖₂. This is literally dE/d(orbital rotation),
  /// so it is exactly zero at the SCF solution.
  double orbital_gradient_norm(const DeviceArray<double>& d_Fock,
                               const DeviceArray<double>& d_C, uint64_t nocc);
  /// Total orbital-gradient norm over both spin channels (RKS: the single
  /// channel, scaled so a closed-shell system matches the UKS value for the
  /// same physical state). This is the convergence metric — see SCFParams.
  double orbital_gradient_norm_total();
  void form_total_density();
  void initial_guess();
  /// Place cached atomic weighted Cocc at AO offsets, rescale to the target
  /// electron count, and upload D + Cocc for the first SCF Fock build.
  void assemble_sad_guess();
  void break_beta_symmetry();
  /// Frobenius Tr(A^T B) ≡ ∑ A_ij B_ij (valid for symmetric energy matrices).
  double frobenius_dot(const double* d_A, const double* d_B, int n2);
  double compute_rms_delta_device(const double* d_D, const double* d_D_old, int n2);

  CuESTContext& ctx_;
  BasisBuilder& basis_;
  DFJKBuilder& dfjk_;
  XCBuilder* xc_;
  ECPBuilder* ecp_builder_;
  ECPIntegrals* ecp_int_;
  const Molecule& mol_;
  SCFParams params_;
  const SADGuessConfig* sad_config_;

  cublasHandle_t cublas_{};
  cusolverDnHandle_t cusolver_{};

  bool uks_{false};
  uint64_t nao_{0}, nocc_{0}, nocc_a_{0}, nocc_b_{0};
  int nelec_{0};
  double e_nuc_{0.0}, e_elec_{0.0}, e_xc_{0.0}, e_total_{0.0};
  double e_hcore_{0.0}, e_kin_{0.0}, e_ne_{0.0}, e_j_{0.0}, e_k_{0.0};
  double tr_ds_{0.0};
  std::vector<double> mo_energies_a_, mo_energies_b_;
  int iter_{0};
  bool converged_{false};
  /// First Fock uses the SAD weighted Cocc (ncols may differ from nocc_*).
  bool use_sad_cocc_{false};
  uint64_t sad_ncols_a_{0}, sad_ncols_b_{0};
  /// UKS: apply β HOMO/LUMO mix once after the first post-guess diagonalization.
  bool symmetry_broken_{false};

  DeviceArray<double> d_Hcore_, d_S_, d_T_, d_V_, d_ECP_;
  DeviceArray<double> d_Fock_a_, d_Fock_b_;
  DeviceArray<double> d_D_a_, d_D_b_, d_D_tot_, d_D_old_a_, d_D_old_b_;
  DeviceArray<double> d_C_a_, d_C_b_;
  DeviceArray<double> d_Cocc_a_, d_Cocc_b_;
  DeviceArray<double> d_J_, d_K_a_, d_K_b_, d_Vxc_a_, d_Vxc_b_;
  DeviceArray<double> d_Vnlc_a_;  // VV10 nonlocal XC contribution (shared across spin channels)
  DeviceArray<double> d_eigvals_, d_xyz_, d_charges_;
  DeviceArray<double> d_Fwork_, d_Swork_;
  DeviceArray<double> d_syev_work_;
  DeviceArray<int> d_info_;
  int syev_lwork_{0};

  DeviceArray<double> d_rms_scratch_;
  /// Scratch for orbital_gradient_norm(): F·C_occ and then C_vir^T(F C_occ).
  DeviceArray<double> d_grad_t_, d_grad_g_;
  /// MO coefficients are only meaningful after the first diagonalization; the
  /// SAD guess path reaches iteration 1 without having diagonalized anything.
  bool mos_valid_{false};
  DIIS diis_a_, diis_b_;
};

}  // namespace cuest
