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
#include <vector>
#include "cuest_wrapper/basis.hpp"
#include "cuest_wrapper/context.hpp"
#include "cuest_wrapper/integrals.hpp"
#include "cuest_wrapper/raii.hpp"
#include "diis.hpp"

namespace cuest {

struct SADGuessConfig;

/// Cached [lo, hi) degenerate-frontier orbital range for one spin channel
/// (see sad_guess.hpp) — detected once from the first, exactly-symmetric
/// Hcore diagonalization, then held fixed for the rest of the SCF.
struct FrontierGroup {
  int lo{-1};
  int hi{-1};
  bool ready{false};
};

struct SCFParams {
  int max_iter{250};
  /// Threshold on the DIIS commutator RMS ||FDS-SDF|| (see DIIS::error_rms),
  /// not density RMS — insensitive to harmless rotation within an exactly-
  /// degenerate occupied subspace, unlike a raw density-RMS criterion.
  double conv_thresh{1e-6};
  double energy_conv_thresh{1e-8};
  int diis_start{1};
  int diis_max_space{15};
  double damping{0.0};
  bool verbose{true};
  bool print_mos{false};
  int print_level{2};
  /// Mix angle (radians) for HOMO–LUMO symmetry breaking on the β guess.
  double break_symmetry{0.3};
  /// Use cuEST JIT kernels (DF-J, nuclear potential, …). Off ⇒ AOT + fp64.
  bool use_jit{true};
  /// Force the zero-density (core Hamiltonian) initial guess, skipping SAD.
  /// Used for the isolated-atom reference SCF that SAD itself calls (see
  /// sad_guess.cpp) so it doesn't recurse into building its own SAD guess.
  bool force_hcore_guess{false};
  /// Suppress the machine-readable ENERGY_COMPONENTS block (printed
  /// unconditionally otherwise, even under --quiet). Used for the isolated
  /// atomic reference SCFs so they don't interleave a second energy block
  /// ahead of the real molecular result on stdout.
  bool suppress_output{false};
  /// Give every orbital in the HOMO/LUMO-straddling degenerate group equal
  /// fractional occupation (see sad_guess.hpp) instead of a hard integer
  /// cutoff, at *every* iteration. Needed for the isolated-atom SAD
  /// reference: occupying one arbitrary real orbital out of a partially
  /// filled degenerate shell (e.g. only px, py of a p2 configuration) feeds
  /// an anisotropic density into J/K/Vxc, which then splits the degeneracy
  /// self-consistently — symmetrizing only the final density isn't enough.
  bool spherical_average_occupation{false};
};

class SCFSolver {
 public:
  /// `sad_config`, if non-null, enables the atomic-superposition initial
  /// guess (see sad_guess.hpp); must outlive this SCFSolver. Pass nullptr
  /// (or set params.force_hcore_guess) to fall back to the core-Hamiltonian
  /// guess.
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
    cudaMemcpy(C.data(), d_C_a_.get(), nao_*nao_*sizeof(double), cudaMemcpyDeviceToHost);
    return C;
  }
  std::vector<double> mo_coefficients_beta_host() {
    std::vector<double> C(nao_ * nao_);
    cudaMemcpy(C.data(), d_C_b_.get(), nao_*nao_*sizeof(double), cudaMemcpyDeviceToHost);
    return C;
  }
  /// Spin-α density (RKS: D_alpha; UKS: D_alpha). Gradients use D_alpha convention.
  std::vector<double> density_host() {
    std::vector<double> D(nao_ * nao_);
    cudaMemcpy(D.data(), d_D_a_.get(), nao_*nao_*sizeof(double), cudaMemcpyDeviceToHost);
    return D;
  }
  std::vector<double> density_alpha_host() { return density_host(); }
  std::vector<double> density_beta_host() {
    std::vector<double> D(nao_ * nao_);
    cudaMemcpy(D.data(), d_D_b_.get(), nao_*nao_*sizeof(double), cudaMemcpyDeviceToHost);
    return D;
  }
  std::vector<double> density_total_host() {
    std::vector<double> D(nao_ * nao_);
    cudaMemcpy(D.data(), d_D_tot_.get(), nao_*nao_*sizeof(double), cudaMemcpyDeviceToHost);
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
  /// Dispatches to build_density(), or to the spherically-averaged density
  /// (sad_guess.hpp) when params_.spherical_average_occupation is set —
  /// in which case `group` is detected once (from the first call) and then
  /// held fixed; see FrontierGroup.
  void build_density_auto(DeviceArray<double>& d_C, uint64_t nocc,
                          const std::vector<double>& mo_energies,
                          DeviceArray<double>& d_D, FrontierGroup& group);
  void form_total_density();
  void initial_guess();
  /// Assemble the block-diagonal SAD guess density (from cached atomic
  /// references, placed at basis_.ao_offsets()) and build d_Fock_a_/d_Fock_b_
  /// as Hcore + J[D_guess]. K/Vxc are deliberately omitted here: they need
  /// occupied MO coefficients, which don't exist until this Fock is
  /// diagonalized for the first time.
  void build_sad_guess_fock();
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

  DeviceArray<double> d_Hcore_, d_S_, d_T_, d_V_, d_ECP_;
  DeviceArray<double> d_Fock_a_, d_Fock_b_;
  DeviceArray<double> d_Fock_save_a_, d_Fock_save_b_;
  DeviceArray<double> d_D_a_, d_D_b_, d_D_tot_, d_D_old_a_, d_D_old_b_;
  DeviceArray<double> d_C_a_, d_C_b_;
  DeviceArray<double> d_Cocc_a_, d_Cocc_b_;
  DeviceArray<double> d_J_, d_K_a_, d_K_b_, d_Vxc_a_, d_Vxc_b_;
  DeviceArray<double> d_eigvals_, d_xyz_, d_charges_;
  DeviceArray<double> d_Fwork_, d_Swork_;
  DeviceArray<double> d_syev_work_;
  DeviceArray<int> d_info_;
  int syev_lwork_{0};

  DeviceArray<double> d_rms_scratch_;
  DIIS diis_a_, diis_b_;

  FrontierGroup sph_group_a_, sph_group_b_;
};

}  // namespace cuest
