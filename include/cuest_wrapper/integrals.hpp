#pragma once
/**
 * @file integrals.hpp
 * @brief Integral computation wrappers for one-electron, DF-J/K, XC, and ECP
 *        integrals using cuEST.
 */

#include <cuda_runtime.h>
#include <cuest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "raii.hpp"

namespace cuest {

class CuESTContext;

// ---------------------------------------------------------------------------
// One-electron integrals: overlap (S), kinetic (T), potential (V)
// ---------------------------------------------------------------------------
class OneElectronIntegrals {
 public:
  OneElectronIntegrals(CuESTContext& ctx, cuestAOBasis_t basis,
                        cuestAOPairList_t pair_list);

  void compute_overlap(double* d_S);  // d_S[nao*nao] device memory
  void compute_kinetic(double* d_T);
  void compute_potential(double* d_V, uint64_t natom,
                          const double* d_xyz, const double* d_charges);
  cuestOEIntPlan_t plan() const { return plan_.get(); }

 private:
  CuESTContext& ctx_;
  OEIntPlanHandle plan_;
  cuestWorkspaceDescriptor_t persistent_desc_{};
  Workspace persistent_ws_;
};

// ---------------------------------------------------------------------------
// Density-fitted J and K matrix builder
// ---------------------------------------------------------------------------
class DFJKBuilder {
 public:
  DFJKBuilder(CuESTContext& ctx, cuestAOBasis_t primary_basis,
               cuestAOBasis_t aux_basis,
               const double* xyz_host, uint64_t natom);

  // Compute Coulomb matrix J from density matrix D
  void compute_J(const double* d_D, double* d_J);

  // Compute exchange matrix K from occupied orbitals Cocc
  void compute_K(uint64_t nocc, const double* d_Cocc, double* d_K,
                 size_t variable_buf_bytes = 2000000000ULL);

  ~DFJKBuilder();
  cuestDFIntPlan_t plan() const { return plan_.get(); }

 private:
  CuESTContext& ctx_;
  DFIntPlanHandle plan_;
  void* persist_plan_dev_{nullptr};
};

// ---------------------------------------------------------------------------
// XC potential builder (local + nonlocal)
// ---------------------------------------------------------------------------
class XCBuilder {
 public:
  // Supported functionals -- mirrored from cuEST parameter types
  enum Functional {
    XC_PBE = 0,             // CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE
    XC_B3LYP,               // CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B3LYP
    XC_B3LYP5,              // etc.
    XC_PBE0,
    XC_CAM_B3LYP,
    XC_WB97X_V,
    XC_WB97M_V,
    XC_HSE06,
    XC_M06,
    XC_M062X,
    XC_LC_WPBE,
    XC_LC_WPBEH,
    XC_WB97X,
    NUM_FUNCTIONALS
  };

  XCBuilder(CuESTContext& ctx, cuestAOBasis_t basis,
             cuestMolecularGrid_t mol_grid,
             int functional_id,
             int radial_pts = 75, int angular_pts = 302);

  // RKS: restricted Kohn-Sham Vxc
  void compute_vxc_rks(uint64_t nocc, const double* d_Cocc,
                        double* exc, double* d_Vxc,
                        size_t variable_buf_bytes = 2000000000ULL);

  // UKS: unrestricted Kohn-Sham Vxc
  void compute_vxc_uks(uint64_t nocc_a, uint64_t nocc_b,
                        const double* d_Cocc_a, const double* d_Cocc_b,
                        double* exc,
                        double* d_Vxc_a, double* d_Vxc_b,
                        size_t variable_buf_bytes = 2000000000ULL);

  ~XCBuilder();
  cuestXCIntPlan_t plan() const { return plan_.get(); }
  // Query if functional is hybrid (has exact exchange)
  bool is_hybrid();
  double exchange_scale();  // fraction of HF exchange for hybrids

 private:
  static cuestXCIntPlanParametersFunctional_t to_cuest_functional(int id);

  CuESTContext& ctx_;
  cuestMolecularGrid_t mol_grid_;  // raw handle, NOT owned (owned by caller)
  XCIntPlanHandle plan_;
  void* xc_persist_dev_{nullptr};  // KEPT ALIVE for compute
};

// ---------------------------------------------------------------------------
// ECP integral builder
// ---------------------------------------------------------------------------
class ECPIntegrals {
 public:
  ECPIntegrals(CuESTContext& ctx, cuestAOBasis_t basis,
                const double* xyz_host, uint64_t num_ecp_atoms,
                const uint64_t* ecp_indices, const cuestECPAtom_t* ecp_atoms);

  void compute(double* d_ECP, size_t variable_buf_bytes = 2000000000ULL);
  cuestECPIntPlan_t plan() const { return plan_.get(); }

 private:
  CuESTContext& ctx_;
  ECPIntPlanHandle plan_;
  cuestWorkspaceDescriptor_t persistent_desc_{};
  Workspace persistent_ws_;
};

}  // namespace cuest
