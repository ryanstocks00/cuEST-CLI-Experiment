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
  cuestWorkspaceDescriptor_t persistent_desc_{};
  Workspace persistent_ws_;  // outlives plan_
  OEIntPlanHandle plan_;
};

// Named constant for max GPU variable buffer (2 GB)
constexpr size_t kDefaultVariableBufBytes = 2000000000ULL;

// ---------------------------------------------------------------------------
// Density-fitted J and K matrix builder
// ---------------------------------------------------------------------------
class DFJKBuilder {
 public:
  DFJKBuilder(CuESTContext& ctx, cuestAOBasis_t primary_basis,
               cuestAOBasis_t aux_basis,
               const double* xyz_host, uint64_t natom,
               double exchange_frac=0.0, double lrc_frac=0.0, double lrc_omega=0.0);

  // Compute Coulomb matrix J from density matrix D
  void compute_J(const double* d_D, double* d_J);

  // Compute exchange matrix K from occupied orbitals Cocc
  void compute_K(uint64_t nocc, const double* d_Cocc, double* d_K,
                 size_t variable_buf_bytes = kDefaultVariableBufBytes);

  ~DFJKBuilder();
  [[nodiscard]] cuestDFIntPlan_t plan() const { return plan_.get(); }

 private:
  CuESTContext& ctx_;
  // Persist buffers outlive the handles that use them (destroyed last).
  Workspace pair_list_persist_;
  Workspace plan_persist_;
  AOPairListHandle pair_list_;
  DFIntPlanHandle plan_;
};

// ---------------------------------------------------------------------------
// XC potential builder (local + nonlocal)
// ---------------------------------------------------------------------------
class XCBuilder {
 public:
  // Supported functionals -- mirrored from cuEST parameter types
  enum Functional {
    XC_HF = 0,              // CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_HF (no grid XC)
    XC_PBE,                 // CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE
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
             int functional_id);

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

  cuestXCIntPlan_t plan() const { return plan_.get(); }
  // Query if functional is hybrid (has exact exchange)
  bool is_hybrid();
  bool is_lrc();            // has long-range correction
  bool is_hf() const { return functional_id_ == XC_HF; }
  double exchange_scale();  // fraction of HF exchange for hybrids

 private:
  static cuestXCIntPlanParametersFunctional_t to_cuest_functional(int id);

  CuESTContext& ctx_;
  cuestMolecularGrid_t mol_grid_;  // raw handle, NOT owned (owned by caller)
  // persist outlives plan (destroyed last)
  Workspace xc_persist_ws_;
  XCIntPlanHandle plan_;
  int functional_id_{-1};
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
  cuestWorkspaceDescriptor_t persistent_desc_{};
  Workspace persistent_ws_;  // outlives plan_
  ECPIntPlanHandle plan_;
};

}  // namespace cuest
