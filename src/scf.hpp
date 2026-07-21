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

namespace cuest {

struct SCFParams {
  int max_iter{250};
  double conv_thresh{1e-8};
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
};

class SCFSolver {
 public:
  SCFSolver(CuESTContext& ctx, BasisBuilder& basis,
            DFJKBuilder& dfjk, XCBuilder* xc,
            ECPBuilder* ecp_builder, ECPIntegrals* ecp_int,
            const Molecule& mol, SCFParams params);
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
  void form_total_density();
  void initial_guess();
  void break_beta_symmetry();
  double compute_rms_delta(const std::vector<double>& D_new,
                           const std::vector<double>& D_old);
  double trace_dot(const double* d_A, const double* d_B, int N);
  void diis_extrapolate(std::vector<std::vector<double>>& errs,
                        std::vector<std::vector<double>>& focks,
                        const std::vector<double>& F_host,
                        const std::vector<double>& D_host,
                        DeviceArray<double>& d_Fock);

  CuESTContext& ctx_;
  BasisBuilder& basis_;
  DFJKBuilder& dfjk_;
  XCBuilder* xc_;
  ECPBuilder* ecp_builder_;
  ECPIntegrals* ecp_int_;
  const Molecule& mol_;
  SCFParams params_;

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
  DeviceArray<double> d_D_a_, d_D_b_, d_D_tot_, d_D_old_a_, d_D_old_b_;
  DeviceArray<double> d_C_a_, d_C_b_;
  DeviceArray<double> d_J_, d_K_a_, d_K_b_, d_Vxc_a_, d_Vxc_b_;
  DeviceArray<double> d_eigvals_, d_xyz_, d_charges_;
  DeviceArray<double> d_Fwork_, d_Swork_;
};

}  // namespace cuest
