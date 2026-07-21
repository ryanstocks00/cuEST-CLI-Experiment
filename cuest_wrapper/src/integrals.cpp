/**
 * @file integrals.cpp
 * @brief Implementations of cuEST integral computation wrappers.
 *
 * Compute scratch is shared via CuESTContext::scratch() so peak temp memory
 * is max(query) across OE / DF / XC / ECP, not the sum.
 */

#include "cuest_wrapper/integrals.hpp"
#include "cuest_wrapper/context.hpp"
#include "cuest_wrapper/nvtx.hpp"
#include "cuest_wrapper/raii.hpp"

namespace cuest {
namespace {

/// When JIT is off, ffloat must also be off (cuEST requirement).
template <typename ParamsT, typename JitAttr, typename FfloatAttr>
void apply_jit_mode(ParamsT& params, cuestParametersType_t ptype,
                    JitAttr jit_attr, FfloatAttr ffloat_attr, bool use_jit) {
  if (use_jit) return;
  cuestJITUsageMode_t jit = CUEST_JIT_USAGE_MODE_OFF;
  cuestFfloatUsageMode_t ffloat = CUEST_FFLOAT_USAGE_MODE_OFF;
  cuestParametersConfigure(ptype, params, jit_attr, &jit, sizeof(jit));
  cuestParametersConfigure(ptype, params, ffloat_attr, &ffloat, sizeof(ffloat));
}

}  // namespace

// ---------------------------------------------------------------------------
// OneElectronIntegrals
// ---------------------------------------------------------------------------
OneElectronIntegrals::OneElectronIntegrals(CuESTContext& ctx,
                                             cuestAOBasis_t basis,
                                             cuestAOPairList_t pair_list,
                                             bool use_jit)
    : ctx_(ctx), use_jit_(use_jit) {
  cuestWorkspaceDescriptor_t temp_desc{};
  OEIntPlanParams oe_params;

  CUEST_CHECK(cuestOEIntPlanCreateWorkspaceQuery(
      ctx_, basis, pair_list,
      oe_params,
      &persistent_desc_, &temp_desc, plan_.ptr()));

  persistent_ws_ = Workspace(persistent_desc_);
  ctx_.scratch().ensure(temp_desc);

  CUEST_NVTX("cuestOEIntPlanCreate",
             cuestOEIntPlanCreate(
                 ctx_, basis, pair_list,
                 oe_params,
                 persistent_ws_.ptr(), ctx_.scratch().ptr(), plan_.ptr()));
}

void OneElectronIntegrals::compute_overlap(double* d_S) {
  cuestWorkspaceDescriptor_t temp_desc{};
  CUEST_CHECK(cuestOverlapComputeWorkspaceQuery(
      ctx_, plan_, cuest::OverlapComputeParams{},
      &temp_desc, d_S));
  ctx_.scratch().ensure(temp_desc);

  CUEST_NVTX("cuestOverlapCompute",
             cuestOverlapCompute(
                 ctx_, plan_, cuest::OverlapComputeParams{},
                 ctx_.scratch().ptr(), d_S));
}

void OneElectronIntegrals::compute_kinetic(double* d_T) {
  cuestWorkspaceDescriptor_t temp_desc{};
  CUEST_CHECK(cuestKineticComputeWorkspaceQuery(
      ctx_, plan_, cuest::KineticComputeParams{},
      &temp_desc, d_T));
  ctx_.scratch().ensure(temp_desc);

  CUEST_NVTX("cuestKineticCompute",
             cuestKineticCompute(
                 ctx_, plan_, cuest::KineticComputeParams{},
                 ctx_.scratch().ptr(), d_T));
}

void OneElectronIntegrals::compute_potential(double* d_V, uint64_t natom,
                                              const double* d_xyz,
                                              const double* d_charges) {
  PotentialComputeParams params;
  apply_jit_mode(params, CUEST_POTENTIALCOMPUTE_PARAMETERS,
                 CUEST_POTENTIALCOMPUTE_PARAMETERS_JIT_USAGE_MODE,
                 CUEST_POTENTIALCOMPUTE_PARAMETERS_FFLOAT_USAGE_MODE,
                 use_jit_);

  cuestWorkspaceDescriptor_t temp_desc{};
  CUEST_CHECK(cuestPotentialComputeWorkspaceQuery(
      ctx_, plan_, params,
      &temp_desc, natom, d_xyz, d_charges, d_V));
  ctx_.scratch().ensure(temp_desc);

  CUEST_NVTX("cuestPotentialCompute",
             cuestPotentialCompute(
                 ctx_, plan_, params,
                 ctx_.scratch().ptr(), natom, d_xyz, d_charges, d_V));
}

// ---------------------------------------------------------------------------
// DFJKBuilder
// ---------------------------------------------------------------------------
DFJKBuilder::DFJKBuilder(CuESTContext& ctx, cuestAOBasis_t primary_basis,
                           cuestAOBasis_t aux_basis,
                           const double* xyz_host, uint64_t natom,
                           double exchange_frac, double lrc_frac, double lrc_omega,
                           bool use_jit)
    : ctx_(ctx), use_jit_(use_jit) {
  {
    cuestWorkspaceDescriptor_t pers_desc{}, temp_desc{};
    AOPairListParams pl_params;
    CUEST_CHECK(cuestAOPairListCreateWorkspaceQuery(
        static_cast<cuestHandle_t>(ctx_), primary_basis, natom, xyz_host,
        1e-14, pl_params, &pers_desc, &temp_desc, pair_list_.ptr()));
    pair_list_persist_ = Workspace(pers_desc);
    ctx_.scratch().ensure(temp_desc);
    CUEST_NVTX("cuestAOPairListCreate",
               cuestAOPairListCreate(
                   static_cast<cuestHandle_t>(ctx_), primary_basis, natom, xyz_host,
                   1e-14, pl_params, pair_list_persist_.ptr(), ctx_.scratch().ptr(),
                   pair_list_.ptr()));
  }

  {
    DFIntPlanParams df_params;
    cuestParametersConfigure(CUEST_DFINTPLAN_PARAMETERS, df_params,
        CUEST_DFINTPLAN_PARAMETERS_EXCHANGE_FRACTION, &exchange_frac, sizeof(double));
    cuestParametersConfigure(CUEST_DFINTPLAN_PARAMETERS, df_params,
        CUEST_DFINTPLAN_PARAMETERS_LRC_EXCHANGE_FRACTION, &lrc_frac, sizeof(double));
    cuestParametersConfigure(CUEST_DFINTPLAN_PARAMETERS, df_params,
        CUEST_DFINTPLAN_PARAMETERS_LRC_EXCHANGE_OMEGA, &lrc_omega, sizeof(double));

    cuestWorkspaceDescriptor_t pers_desc{}, temp_desc{};
    CUEST_CHECK(cuestDFIntPlanCreateWorkspaceQuery(
        static_cast<cuestHandle_t>(ctx_), primary_basis, aux_basis,
        pair_list_.get(), df_params, &pers_desc, &temp_desc, plan_.ptr()));
    plan_persist_ = Workspace(pers_desc);
    ctx_.scratch().ensure(temp_desc);
    CUEST_NVTX("cuestDFIntPlanCreate",
               cuestDFIntPlanCreate(
                   static_cast<cuestHandle_t>(ctx_), primary_basis, aux_basis,
                   pair_list_.get(), df_params, plan_persist_.ptr(),
                   ctx_.scratch().ptr(), plan_.ptr()));
  }
}

DFJKBuilder::~DFJKBuilder() {
  // Destroy handles before persist workspaces (members would also do this
  // if declaration order is correct; be explicit for clarity).
  plan_.reset();
  pair_list_.reset();
}

void DFJKBuilder::compute_J(const double* d_D, double* d_J) {
  DFCoulombComputeParams params;
  apply_jit_mode(params, CUEST_DFCOULOMBCOMPUTE_PARAMETERS,
                 CUEST_DFCOULOMBCOMPUTE_PARAMETERS_JIT_USAGE_MODE,
                 CUEST_DFCOULOMBCOMPUTE_PARAMETERS_FFLOAT_USAGE_MODE,
                 use_jit_);

  cuestWorkspaceDescriptor_t temp_desc{};
  CUEST_CHECK(cuestDFCoulombComputeWorkspaceQuery(
      ctx_, plan_, params,
      &temp_desc, d_D, d_J));
  ctx_.scratch().ensure(temp_desc);
  CUEST_NVTX("cuestDFCoulombCompute",
             cuestDFCoulombCompute(
                 ctx_, plan_, params,
                 ctx_.scratch().ptr(), d_D, d_J));
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
  ctx_.scratch().ensure(temp_desc);
  CUEST_NVTX("cuestDFSymmetricExchangeCompute",
             cuestDFSymmetricExchangeCompute(
                 ctx_, plan_, cuest::DFSymmetricExchangeComputeParams{},
                 &var_buf, ctx_.scratch().ptr(), nocc, d_Cocc, d_K));
}

// ---------------------------------------------------------------------------
// XCBuilder
// ---------------------------------------------------------------------------
cuestXCIntPlanParametersFunctional_t XCBuilder::to_cuest_functional(Functional id) {
  switch (id) {
    case XC_HF:       return CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_HF;
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
                       Functional functional)
    : ctx_(ctx), mol_grid_(mol_grid), functional_(functional) {
  cuestWorkspaceDescriptor_t pers_desc{}, temp_desc{};
  XCIntPlanParams xc_params;

  CUEST_CHECK(cuestXCIntPlanCreateWorkspaceQuery(
      ctx_, basis, mol_grid_,
      to_cuest_functional(functional),
      xc_params,
      &pers_desc, &temp_desc, plan_.ptr()));

  xc_persist_ws_ = Workspace(pers_desc);
  ctx_.scratch().ensure(temp_desc);

  CUEST_NVTX("cuestXCIntPlanCreate",
             cuestXCIntPlanCreate(
                 ctx_, basis, mol_grid_,
                 to_cuest_functional(functional),
                 xc_params,
                 xc_persist_ws_.ptr(), ctx_.scratch().ptr(), plan_.ptr()));
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

  ctx_.scratch().ensure(temp_desc);
  CUEST_NVTX("cuestXCPotentialRKSCompute",
             cuestXCPotentialRKSCompute(
                 ctx_, plan_,
                 xc_comp_params,
                 &var_buf, ctx_.scratch().ptr(),
                 nocc, d_Cocc, exc, d_Vxc));
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

  ctx_.scratch().ensure(temp_desc);
  CUEST_NVTX("cuestXCPotentialUKSCompute",
             cuestXCPotentialUKSCompute(
                 ctx_, plan_,
                 cuest::XCPotentialUKSComputeParams{},
                 &var_buf, ctx_.scratch().ptr(),
                 nocc_a, nocc_b,
                 d_Cocc_a, d_Cocc_b,
                 exc, d_Vxc_a, d_Vxc_b));
}

bool XCBuilder::is_hybrid() {
  return plan_.query<int32_t>(ctx_, CUEST_XCINTPLAN_IS_HYBRID) != 0;
}

bool XCBuilder::is_lrc() {
  return plan_.query<int32_t>(ctx_, CUEST_XCINTPLAN_IS_LRC_HYBRID) != 0;
}

double XCBuilder::exchange_scale() {
  return plan_.query<double>(ctx_, CUEST_XCINTPLAN_EXCHANGE_SCALE);
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
  ctx_.scratch().ensure(temp_desc);

  CUEST_NVTX("cuestECPIntPlanCreate",
             cuestECPIntPlanCreate(
                 ctx_, basis, xyz_host,
                 num_ecp_atoms, ecp_indices, ecp_atoms,
                 cuest::ECPIntPlanParams{},
                 persistent_ws_.ptr(), ctx_.scratch().ptr(), plan_.ptr()));
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

  ctx_.scratch().ensure(temp_desc);
  CUEST_NVTX("cuestECPCompute",
             cuestECPCompute(
                 ctx_, plan_,
                 cuest::ECPComputeParams{},
                 &var_buf, ctx_.scratch().ptr(), d_ECP));
}

}  // namespace cuest
