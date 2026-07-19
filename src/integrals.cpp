/**
 * @file integrals.cpp
 * @brief Implementations of cuEST integral computation wrappers.
 */

#include "cuest_wrapper/integrals.hpp"
#include "cuest_wrapper/context.hpp"
#include "cuest_wrapper/raii.hpp"

extern "C" {
cuestStatus_t nvidia_create_df_plan(
    cuestHandle_t handle, cuestAOBasis_t primary, cuestAOBasis_t aux,
    const double* xyz, uint64_t natom, cuestDFIntPlan_t* outPlan,
    double exchange_frac, double lrc_frac, double lrc_omega);
}

namespace cuest {

// ---------------------------------------------------------------------------
// OneElectronIntegrals
// ---------------------------------------------------------------------------
OneElectronIntegrals::OneElectronIntegrals(CuESTContext& ctx,
                                             cuestAOBasis_t basis,
                                             cuestAOPairList_t pair_list)
    : ctx_(ctx) {
  cuestWorkspaceDescriptor_t temp_desc{};
  OEIntPlanParams oe_params;

  CUEST_CHECK(cuestOEIntPlanCreateWorkspaceQuery(
      ctx_, basis, pair_list,
      oe_params,
      &persistent_desc_, &temp_desc, plan_.ptr()));

  void* oe_tdev = nullptr;
  if (temp_desc.deviceBufferSizeInBytes) cudaMalloc(&oe_tdev, temp_desc.deviceBufferSizeInBytes);
  cuestWorkspace_t oe_tws = {0, temp_desc.hostBufferSizeInBytes, (uintptr_t)oe_tdev, temp_desc.deviceBufferSizeInBytes};

  persistent_ws_ = Workspace(persistent_desc_);

  CUEST_CHECK(cuestOEIntPlanCreate(
      ctx_, basis, pair_list,
      oe_params,
      persistent_ws_.ptr(), &oe_tws, plan_.ptr()));

  if (oe_tdev) cudaFree(oe_tdev);
}

void OneElectronIntegrals::compute_overlap(double* d_S) {
  cuestWorkspaceDescriptor_t temp_desc{};
  CUEST_CHECK(cuestOverlapComputeWorkspaceQuery(
      ctx_, plan_, cuest::OverlapComputeParams{},
      &temp_desc, d_S));
  Workspace temp_ws(temp_desc);

  CUEST_CHECK(cuestOverlapCompute(
      ctx_, plan_, cuest::OverlapComputeParams{},
      temp_ws.ptr(), d_S));
}

void OneElectronIntegrals::compute_kinetic(double* d_T) {
  cuestWorkspaceDescriptor_t temp_desc{};
  CUEST_CHECK(cuestKineticComputeWorkspaceQuery(
      ctx_, plan_, cuest::KineticComputeParams{},
      &temp_desc, d_T));
  Workspace temp_ws(temp_desc);

  CUEST_CHECK(cuestKineticCompute(
      ctx_, plan_, cuest::KineticComputeParams{},
      temp_ws.ptr(), d_T));
}

void OneElectronIntegrals::compute_potential(double* d_V, uint64_t natom,
                                              const double* d_xyz,
                                              const double* d_charges) {
  cuestWorkspaceDescriptor_t temp_desc{};
  CUEST_CHECK(cuestPotentialComputeWorkspaceQuery(
      ctx_, plan_, cuest::PotentialComputeParams{},
      &temp_desc, natom, d_xyz, d_charges, d_V));
  Workspace temp_ws(temp_desc);

  CUEST_CHECK(cuestPotentialCompute(
      ctx_, plan_, cuest::PotentialComputeParams{},
      temp_ws.ptr(), natom, d_xyz, d_charges, d_V));
}

// ---------------------------------------------------------------------------
// DFJKBuilder
// ---------------------------------------------------------------------------
DFJKBuilder::DFJKBuilder(CuESTContext& ctx, cuestAOBasis_t primary_basis,
                           cuestAOBasis_t aux_basis,
                           const double* xyz_host, uint64_t natom,
                           double exchange_frac, double lrc_frac, double lrc_omega)
    : ctx_(ctx) {
  // Use pure-C wrapper with exchange parameters for hybrid/LRC functionals
  cuestDFIntPlan_t raw_plan = nullptr;
  CUEST_CHECK(nvidia_create_df_plan(
      static_cast<cuestHandle_t>(ctx_), primary_basis, aux_basis,
      xyz_host, natom, &raw_plan,
      exchange_frac, lrc_frac, lrc_omega));
  *plan_.ptr() = raw_plan;
}

DFJKBuilder::~DFJKBuilder() {
  if (persist_plan_dev_) cudaFree(persist_plan_dev_);
}

void DFJKBuilder::compute_J(const double* d_D, double* d_J) {
  cuestWorkspaceDescriptor_t temp_desc{};
  CUEST_CHECK(cuestDFCoulombComputeWorkspaceQuery(
      ctx_, plan_, cuest::DFCoulombComputeParams{},
      &temp_desc, d_D, d_J));
  Workspace temp_ws(temp_desc);
  CUEST_CHECK(cuestDFCoulombCompute(
      ctx_, plan_, cuest::DFCoulombComputeParams{},
      temp_ws.ptr(), d_D, d_J));
}

void DFJKBuilder::compute_K(uint64_t nocc, const double* d_Cocc, double* d_K,
                              size_t variable_buf_bytes) {
  cuestWorkspaceDescriptor_t temp_desc{};
  cuestWorkspaceDescriptor_t var_buf{};
  var_buf.hostBufferSizeInBytes = 0;
  var_buf.deviceBufferSizeInBytes = variable_buf_bytes;

  CUEST_CHECK(cuestDFSymmetricExchangeComputeWorkspaceQuery(
      ctx_, plan_, cuest::DFSymmetricExchangeComputeParams{},
      &var_buf, &temp_desc, nocc, d_Cocc, d_K));
  Workspace temp_ws(temp_desc);
  CUEST_CHECK(cuestDFSymmetricExchangeCompute(
      ctx_, plan_, cuest::DFSymmetricExchangeComputeParams{},
      &var_buf, temp_ws.ptr(), nocc, d_Cocc, d_K));
}

// ---------------------------------------------------------------------------
// XCBuilder
// ---------------------------------------------------------------------------
cuestXCIntPlanParametersFunctional_t XCBuilder::to_cuest_functional(int id) {
  // Map our internal IDs to cuEST cuestXCIntPlanParametersFunctional_t
  // Values from cuest_parameter_types.h
  switch (id) {
    case XC_PBE:      return CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE;
    case XC_B3LYP:    return CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B3LYP1;
    case XC_B3LYP5:   return CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B3LYP5;
    case XC_PBE0:     return CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE0;
    case XC_CAM_B3LYP:return CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_CAMB3LYP;
    case XC_WB97X_V:  return CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_WB97XV;
    case XC_WB97M_V:  return CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_WB97MV;
    case XC_HSE06:    return CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_HSE06;
    case XC_M06:      return CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_M06;
    case XC_M062X:    return CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_M062X;
    case XC_LC_WPBE:  return CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_LCWPBE;
    case XC_LC_WPBEH: return CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_LCWPBEH;
    case XC_WB97X:    return CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_WB97X;
    default:          return CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE;
  }
}

XCBuilder::XCBuilder(CuESTContext& ctx, cuestAOBasis_t basis,
                       cuestMolecularGrid_t mol_grid,
                       int functional_id,
                       int /*radial_pts*/, int /*angular_pts*/)
    : ctx_(ctx), mol_grid_(mol_grid) {
  cuestWorkspaceDescriptor_t pers_desc{}, temp_desc{};
  XCIntPlanParams xc_params;

  CUEST_CHECK(cuestXCIntPlanCreateWorkspaceQuery(
      ctx_, basis, mol_grid_,
      to_cuest_functional(functional_id),
      xc_params,
      &pers_desc, &temp_desc, plan_.ptr()));

  void* xc_pp_dev = nullptr, *xc_tp_dev = nullptr;
  if (pers_desc.deviceBufferSizeInBytes) cudaMalloc(&xc_pp_dev, pers_desc.deviceBufferSizeInBytes);
  if (temp_desc.deviceBufferSizeInBytes) cudaMalloc(&xc_tp_dev, temp_desc.deviceBufferSizeInBytes);
  cuestWorkspace_t xc_pp_ws = {0, pers_desc.hostBufferSizeInBytes,
                                (uintptr_t)xc_pp_dev, pers_desc.deviceBufferSizeInBytes};
  cuestWorkspace_t xc_tp_ws = {0, temp_desc.hostBufferSizeInBytes,
                                (uintptr_t)xc_tp_dev, temp_desc.deviceBufferSizeInBytes};

  CUEST_CHECK(cuestXCIntPlanCreate(
      ctx_, basis, mol_grid_,
      to_cuest_functional(functional_id),
      xc_params,
      &xc_pp_ws, &xc_tp_ws, plan_.ptr()));

  // KEEP persist alive for compute!
  xc_persist_dev_ = xc_pp_dev; xc_pp_dev = nullptr;
  if (xc_tp_dev) cudaFree(xc_tp_dev);  // temp can be freed
}

XCBuilder::~XCBuilder() {
  if (xc_persist_dev_) cudaFree(xc_persist_dev_);
}

void XCBuilder::compute_vxc_rks(uint64_t nocc, const double* d_Cocc,
                                 double* exc, double* d_Vxc,
                                 size_t variable_buf_bytes) {
  cuestWorkspaceDescriptor_t temp_desc{};
  cuestWorkspaceDescriptor_t var_buf{};
  var_buf.hostBufferSizeInBytes = 0;
  var_buf.deviceBufferSizeInBytes = variable_buf_bytes;

  XCPotentialRKSComputeParams xc_comp_params;
  CUEST_CHECK(cuestXCPotentialRKSComputeWorkspaceQuery(
      ctx_, plan_,
      xc_comp_params,
      &var_buf, &temp_desc,
      nocc, d_Cocc, exc, d_Vxc));

  void* tdev=nullptr;
  if(temp_desc.deviceBufferSizeInBytes)cudaMalloc(&tdev,temp_desc.deviceBufferSizeInBytes);
  cuestWorkspace_t tws={0,temp_desc.hostBufferSizeInBytes,(uintptr_t)tdev,temp_desc.deviceBufferSizeInBytes};
  CUEST_CHECK(cuestXCPotentialRKSCompute(
      ctx_, plan_,
      xc_comp_params,
      &var_buf, &tws,
      nocc, d_Cocc, exc, d_Vxc));
  if(tdev)cudaFree(tdev);
}

void XCBuilder::compute_vxc_uks(uint64_t nocc_a, uint64_t nocc_b,
                                 const double* d_Cocc_a,
                                 const double* d_Cocc_b,
                                 double* exc,
                                 double* d_Vxc_a, double* d_Vxc_b,
                                 size_t variable_buf_bytes) {
  cuestWorkspaceDescriptor_t temp_desc{};
  cuestWorkspaceDescriptor_t var_buf{};
  var_buf.hostBufferSizeInBytes = 0;
  var_buf.deviceBufferSizeInBytes = variable_buf_bytes;

  CUEST_CHECK(cuestXCPotentialUKSComputeWorkspaceQuery(
      ctx_, plan_,
      cuest::XCPotentialUKSComputeParams{},
      &var_buf, &temp_desc,
      nocc_a, nocc_b,
      d_Cocc_a, d_Cocc_b,
      exc, d_Vxc_a, d_Vxc_b));

  Workspace temp_ws(temp_desc);
  CUEST_CHECK(cuestXCPotentialUKSCompute(
      ctx_, plan_,
      cuest::XCPotentialUKSComputeParams{},
      &var_buf, temp_ws.ptr(),
      nocc_a, nocc_b,
      d_Cocc_a, d_Cocc_b,
      exc, d_Vxc_a, d_Vxc_b));
}

bool XCBuilder::is_hybrid() {
  int32_t hyb = 0;
  CUEST_CHECK(cuestQuery(ctx_, CUEST_XCINTPLAN, plan_.get(),
                          CUEST_XCINTPLAN_IS_HYBRID,
                          &hyb, sizeof(hyb)));
  return hyb != 0;
}

bool XCBuilder::is_lrc() {
  int32_t lrc = 0;
  CUEST_CHECK(cuestQuery(ctx_, CUEST_XCINTPLAN, plan_.get(),
                          CUEST_XCINTPLAN_IS_LRC_HYBRID,
                          &lrc, sizeof(lrc)));
  return lrc != 0;
}

double XCBuilder::exchange_scale() {
  double scale = 0.0;
  CUEST_CHECK(cuestQuery(ctx_, CUEST_XCINTPLAN, plan_.get(),
                          CUEST_XCINTPLAN_EXCHANGE_SCALE,
                          &scale, sizeof(scale)));
  return scale;
}

// ---------------------------------------------------------------------------
// ECPIntegrals
// ---------------------------------------------------------------------------
ECPIntegrals::ECPIntegrals(CuESTContext& ctx, cuestAOBasis_t basis,
                             const double* xyz_host,
                             uint64_t num_ecp_atoms,
                             const uint64_t* ecp_indices,
                             const cuestECPAtom_t* ecp_atoms)
    : ctx_(ctx) {
  cuestWorkspaceDescriptor_t temp_desc{};

  CUEST_CHECK(cuestECPIntPlanCreateWorkspaceQuery(
      ctx_, basis, xyz_host,
      num_ecp_atoms, ecp_indices, ecp_atoms,
      cuest::ECPIntPlanParams{},
      &persistent_desc_, &temp_desc, plan_.ptr()));

  persistent_ws_ = Workspace(persistent_desc_);
  Workspace temp_ws(temp_desc);

  CUEST_CHECK(cuestECPIntPlanCreate(
      ctx_, basis, xyz_host,
      num_ecp_atoms, ecp_indices, ecp_atoms,
      cuest::ECPIntPlanParams{},
      persistent_ws_.ptr(), temp_ws.ptr(), plan_.ptr()));
}

void ECPIntegrals::compute(double* d_ECP, size_t variable_buf_bytes) {
  cuestWorkspaceDescriptor_t temp_desc{};
  cuestWorkspaceDescriptor_t var_buf{};
  var_buf.hostBufferSizeInBytes = 0;
  var_buf.deviceBufferSizeInBytes = variable_buf_bytes;

  CUEST_CHECK(cuestECPComputeWorkspaceQuery(
      ctx_, plan_,
      cuest::ECPComputeParams{},
      &var_buf, &temp_desc, d_ECP));

  Workspace temp_ws(temp_desc);
  CUEST_CHECK(cuestECPCompute(
      ctx_, plan_,
      cuest::ECPComputeParams{},
      &var_buf, temp_ws.ptr(), d_ECP));
}

}  // namespace cuest
