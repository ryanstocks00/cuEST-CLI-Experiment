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
#include "basis.hpp"
#include "context.hpp"
#include "integrals.hpp"
#include "raii.hpp"

namespace cuest {

struct SCFParams {
  int max_iter{150};
  double conv_thresh{1e-8};
  double energy_conv_thresh{1e-8};
  int diis_start{1};
  int diis_max_space{10};
  double damping{0.0};
  double level_shift{0.0};
  bool verbose{true};
  bool print_mos{false};
  int print_level{2};
  uint64_t ecp_electrons{0};     // core electrons removed by ECP
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
  const std::vector<double>& orbital_energies() const { return mo_energies_; }
  std::vector<double> mo_coefficients_host() {
    std::vector<double> C(nao_ * nao_);
    cudaMemcpy(C.data(), d_C_.get(), nao_*nao_*sizeof(double), cudaMemcpyDeviceToHost);
    return C;
  }
  std::vector<double> density_host() {
    std::vector<double> D(nao_ * nao_);
    cudaMemcpy(D.data(), d_D_.get(), nao_*nao_*sizeof(double), cudaMemcpyDeviceToHost);
    return D;
  }
  uint64_t nao() const { return nao_; }
  uint64_t nocc() const { return nocc_; }
  int n_iter() const { return iter_; }
  bool converged() const { return converged_; }

 private:
  void build_core_hamiltonian();
  void build_fock_matrix(const std::vector<double>& D_host);
  void diagonalize_fock();  // GPU-based: FC=SCE via cuSOLVER dsygvd
  void build_density_matrix();
  double compute_rms_delta(const std::vector<double>& D_new,
                           const std::vector<double>& D_old);
  double trace_dot(const double* d_A, const double* d_B, int N); // Tr[A*B] on host

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

  uint64_t nao_, nocc_;
  int nelec_;
  double e_nuc_{0.0}, e_elec_{0.0}, e_xc_{0.0}, e_total_{0.0};
  std::vector<double> mo_energies_;
  int iter_{0};
  bool converged_{false};

  // GPU matrices
  DeviceArray d_Hcore_, d_S_, d_T_, d_V_, d_ECP_;
  DeviceArray d_Fock_, d_D_, d_D_old_, d_C_;
  DeviceArray d_J_, d_K_, d_Vxc_;
  DeviceArray d_eigvals_, d_xyz_, d_charges_;
  DeviceArray d_Fwork_, d_Swork_;  // workspaces
};

}  // namespace cuest
