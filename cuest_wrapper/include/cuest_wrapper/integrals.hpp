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
                        cuestAOPairList_t pair_list, bool use_jit = true);

  void compute_overlap(double* d_S);  // d_S[nao*nao] device memory
  void compute_kinetic(double* d_T);
  void compute_potential(double* d_V, uint64_t natom,
                          const double* d_xyz, const double* d_charges);
  cuestOEIntPlan_t plan() const { return plan_.get(); }

 private:
  CuESTContext& ctx_;
  bool use_jit_{true};
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
               double exchange_frac=0.0, double lrc_frac=0.0, double lrc_omega=0.0,
               bool use_jit=true,
               double fitting_cutoff=1.0e-12,
               bool fitting_relative_conditioning=true,
               cuestDFIntPlanParametersFittingAlgorithm_t fitting_algorithm=
                   CUEST_DFINTPLAN_PARAMETERS_FITTING_ALGORITHM_QR);

  // Compute Coulomb matrix J from density matrix D
  void compute_J(const double* d_D, double* d_J);

  // Compute exchange matrix K from occupied orbitals Cocc
  void compute_K(uint64_t nocc, const double* d_Cocc, double* d_K,
                 size_t variable_buf_bytes = kDefaultVariableBufBytes);

  void set_use_jit(bool use_jit) { use_jit_ = use_jit; }
  [[nodiscard]] bool use_jit() const { return use_jit_; }

  ~DFJKBuilder();
  [[nodiscard]] cuestDFIntPlan_t plan() const { return plan_.get(); }

 private:
  CuESTContext& ctx_;
  bool use_jit_{true};
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
             Functional functional);

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

  // RKS: nonlocal (VV10) XC contribution. Reuses the same plan_ as
  // compute_vxc_rks (cuEST has no separate "nonlocal plan" type); the VV10
  // scale/C/b constants are queried from the plan itself (tied to the
  // functional chosen at construction), not hardcoded here. Only meaningful
  // when is_vv10() is true.
  void compute_vv10_rks(uint64_t nocc, const double* d_Cocc,
                        double* enlc, double* d_Vnlc,
                        size_t variable_buf_bytes = 2000000000ULL);

  // UKS: nonlocal (VV10) XC contribution. VV10 is a functional of the total
  // density only, so cuEST returns a single potential matrix shared by both
  // spin channels (unlike compute_vxc_uks's separate alpha/beta outputs).
  void compute_vv10_uks(uint64_t nocc_a, uint64_t nocc_b,
                        const double* d_Cocc_a, const double* d_Cocc_b,
                        double* enlc, double* d_Vnlc,
                        size_t variable_buf_bytes = 2000000000ULL);

  cuestXCIntPlan_t plan() const { return plan_.get(); }
  // Query if functional is hybrid (has exact exchange)
  bool is_hybrid();
  bool is_lrc();            // has long-range correction
  bool is_hf() const { return functional_ == XC_HF; }
  // Does this functional include VV10 nonlocal correlation (e.g. WB97X-V, WB97M-V)?
  bool is_vv10();
  // VV10 constants tied to the functional (only meaningful when is_vv10());
  // queried from cuEST's plan, not hardcoded — used for both the nonlocal
  // potential (compute_vv10_rks/uks) and the nonlocal gradient (GradientComputer).
  double vv10_scale();
  double vv10_c();
  double vv10_b();
  Functional functional() const { return functional_; }
  double exchange_scale();      // fraction of (short-range, for LRC) HF exchange for hybrids
  double lrc_exchange_scale();  // additional long-range HF exchange fraction (LRC hybrids only)
  double lrc_omega();           // range-separation omega (LRC hybrids only)

 private:
  static cuestXCIntPlanParametersFunctional_t to_cuest_functional(Functional id);

  CuESTContext& ctx_;
  cuestMolecularGrid_t mol_grid_;  // raw handle, NOT owned (owned by caller)
  // persist outlives plan (destroyed last)
  Workspace xc_persist_ws_;
  XCIntPlanHandle plan_;
  Functional functional_{XC_PBE};
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
