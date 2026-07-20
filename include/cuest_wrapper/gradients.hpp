#pragma once
#include <cuda_runtime.h>
#include <cuest.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "basis.hpp"
#include "context.hpp"
#include "integrals.hpp"
#include "molecule.hpp"
#include "raii.hpp"

namespace cuest {

// Nuclear repulsion gradient: dEnuc/dR (uses Z_eff for ECP atoms)
[[nodiscard]] inline std::vector<double> nuc_grad(const Molecule& mol) {
  size_t n = mol.natom();
  std::vector<double> g(3 * n, 0.0);
  for (size_t i = 0; i < n; i++) {
    double zi = static_cast<double>(mol.zeff(i));
    for (size_t j = i + 1; j < n; j++) {
      double dx = mol.atom(i).x - mol.atom(j).x;
      double dy = mol.atom(i).y - mol.atom(j).y;
      double dz = mol.atom(i).z - mol.atom(j).z;
      double r2 = dx*dx + dy*dy + dz*dz;
      if (r2 < 1e-20) continue;
      double r = std::sqrt(r2);
      double f = zi * static_cast<double>(mol.zeff(j)) / (r * r2);
      g[3*i+0] += f * dx;  g[3*i+1] += f * dy;  g[3*i+2] += f * dz;
      g[3*j+0] -= f * dx;  g[3*j+1] -= f * dy;  g[3*j+2] -= f * dz;
    }
  }
  return g;
}

// Energy-weighted density matrix for RKS: W = sum_k^occ eps_k * C_k * C_k^T
// NO factor of 2 — the occupation factor is applied when combining gradient terms.
[[nodiscard]] inline std::vector<double> ewd(
    const std::vector<double>& eps, const std::vector<double>& C,
    uint64_t nao, uint64_t nocc) {
  std::vector<double> W(nao * nao, 0.0);
  for (size_t i = 0; i < nao; i++)
    for (size_t j = 0; j < nao; j++)
      for (size_t k = 0; k < nocc; k++)
        W[i + j*nao] += eps[k] * C[i + k*nao] * C[j + k*nao];
  return W;
}

class GradientComputer {
 public:
  GradientComputer(CuESTContext& ctx, BasisBuilder& basis, DFJKBuilder& dfjk,
                   XCBuilder* xc_builder, ECPIntegrals* ecp_int, const Molecule& mol,
                   uint64_t nocc,
                   const std::vector<double>& eps, const std::vector<double>& C,
                   const std::vector<double>& D_alpha);
  std::vector<double> compute();
  const std::vector<double>& ov() const { return ov_; }
  const std::vector<double>& ke() const { return ke_; }
  const std::vector<double>& po() const { return po_; }
  const std::vector<double>& df() const { return df_; }
  const std::vector<double>& xc() const { return xc_; }
  const std::vector<double>& nu() const { return nu_; }
  const std::vector<double>& ecpg() const { return ecp_; }
  const std::vector<double>& pc() const { return pc_; }
 private:
  CuESTContext& ctx_; BasisBuilder& basis_; DFJKBuilder& dfjk_;
  XCBuilder* xc_builder_; ECPIntegrals* ecp_int_; const Molecule& mol_;
  uint64_t nao_, nocc_, natom_;
  DeviceArray d_Da_, d_Wa_, d_Co_;
  std::vector<double> ov_, ke_, po_, pc_, df_, nu_, xc_, ecp_;
};

inline GradientComputer::GradientComputer(
    CuESTContext& ctx, BasisBuilder& basis, DFJKBuilder& dfjk,
    XCBuilder* xc_builder, ECPIntegrals* ecp_int, const Molecule& mol,
    uint64_t nocc,
    const std::vector<double>& eps, const std::vector<double>& C,
    const std::vector<double>& D_alpha)
  : ctx_(ctx), basis_(basis), dfjk_(dfjk), xc_builder_(xc_builder), ecp_int_(ecp_int), mol_(mol){
  nao_=basis.nao(); nocc_=nocc; natom_=mol.natom();
  d_Da_.alloc(nao_*nao_*sizeof(double));
  CUDA_CHECK(cudaMemcpy(d_Da_,D_alpha.data(),nao_*nao_*sizeof(double),cudaMemcpyHostToDevice));
  auto W=ewd(eps,C,nao_,nocc_);
  d_Wa_.alloc(nao_*nao_*sizeof(double));
  CUDA_CHECK(cudaMemcpy(d_Wa_,W.data(),nao_*nao_*sizeof(double),cudaMemcpyHostToDevice));
  // C_occ: SAME layout as SCF — flat copy of first nocc columns
  std::vector<double> Co(nao_*nocc_);
  for(size_t k=0;k<nocc_;k++)for(size_t i=0;i<nao_;i++) Co[i+k*nao_]=C[i+k*nao_];
  d_Co_.alloc(nao_*nocc_*sizeof(double));
  CUDA_CHECK(cudaMemcpy(d_Co_,Co.data(),nao_*nocc_*sizeof(double),cudaMemcpyHostToDevice));
}

// Helper: allocate temp workspace, run derivative compute, free temp
namespace {
template <typename ParamsT, typename QueryFn, typename ComputeFn>
void run_derivative(cuestWorkspaceDescriptor_t& var_buf,
                    ParamsT& params, cuestParametersType_t ptype,
                    QueryFn query, ComputeFn compute) {
  cuestWorkspaceDescriptor_t td{};
  query(&var_buf, &td);
  Workspace temp_ws(td);
  CUDA_CHECK(cudaDeviceSynchronize());
  cuestStatus_t st = compute(&var_buf, temp_ws.ptr());
  CUDA_CHECK(cudaDeviceSynchronize());
  cuestParametersDestroy(ptype, params);
  if (st != CUEST_STATUS_SUCCESS)
    throw std::runtime_error("Gradient derivative compute failed with code " +
                             std::to_string(static_cast<int>(st)));
}
}  // namespace

inline std::vector<double> GradientComputer::compute() {
  uint64_t n3 = 3 * natom_;
  nu_ = nuc_grad(mol_);

  const OwnedAOPairList& pl = basis_.pair_list();
  OneElectronIntegrals oe(ctx_, basis_.basis(), pl.get());
  auto xh = mol_.xyz_host(), ch = mol_.charges_host();
  DeviceArray dx(3 * natom_ * sizeof(double)), dq(natom_ * sizeof(double));
  CUDA_CHECK(cudaMemcpy(dx, xh.data(), 3 * natom_ * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dq, ch.data(), natom_ * sizeof(double), cudaMemcpyHostToDevice));

  DeviceArray d_ov(n3 * sizeof(double)), d_ke(n3 * sizeof(double));
  DeviceArray d_po(n3 * sizeof(double)), d_p2(n3 * sizeof(double));
  DeviceArray d_df(n3 * sizeof(double)), d_xc(n3 * sizeof(double));

  cuestWorkspaceDescriptor_t var_buf{};
  var_buf.deviceBufferSizeInBytes = kDefaultVariableBufBytes;

  // Overlap: Tr[W * dS/dR]
  {
    cuestOverlapDerivativeComputeParameters_t p;
    CUEST_CHECK(cuestParametersCreate(CUEST_OVERLAPDERIVATIVECOMPUTE_PARAMETERS, &p));
    run_derivative(var_buf, p, CUEST_OVERLAPDERIVATIVECOMPUTE_PARAMETERS,
      [&](auto* /*vb*/, auto* td) {
        cuestOverlapDerivativeComputeWorkspaceQuery(ctx_, oe.plan(), p, td, d_Wa_, nullptr);
      },
      [&](auto* /*vb*/, auto* tw) {
        return cuestOverlapDerivativeCompute(ctx_, oe.plan(), p, tw, d_Wa_, d_ov);
      });
  }

  // Kinetic: Tr[D * dT/dR]
  {
    cuestKineticDerivativeComputeParameters_t p;
    CUEST_CHECK(cuestParametersCreate(CUEST_KINETICDERIVATIVECOMPUTE_PARAMETERS, &p));
    run_derivative(var_buf, p, CUEST_KINETICDERIVATIVECOMPUTE_PARAMETERS,
      [&](auto* /*vb*/, auto* td) {
        cuestKineticDerivativeComputeWorkspaceQuery(ctx_, oe.plan(), p, td, d_Da_, nullptr);
      },
      [&](auto* /*vb*/, auto* tw) {
        return cuestKineticDerivativeCompute(ctx_, oe.plan(), p, tw, d_Da_, d_ke);
      });
  }

  // Potential: Tr[D * dV/dR]
  {
    cuestPotentialDerivativeComputeParameters_t p;
    CUEST_CHECK(cuestParametersCreate(CUEST_POTENTIALDERIVATIVECOMPUTE_PARAMETERS, &p));
    run_derivative(var_buf, p, CUEST_POTENTIALDERIVATIVECOMPUTE_PARAMETERS,
      [&](auto* /*vb*/, auto* td) {
        cuestPotentialDerivativeComputeWorkspaceQuery(ctx_, oe.plan(), p, td, natom_, dx, dq, d_Da_, nullptr, nullptr);
      },
      [&](auto* /*vb*/, auto* tw) {
        return cuestPotentialDerivativeCompute(ctx_, oe.plan(), p, tw, natom_, dx, dq, d_Da_, d_po, d_p2);
      });
  }

  // ECP derivative
  if (ecp_int_) {
    DeviceArray d_ecp(n3 * sizeof(double));
    cuestECPDerivativeComputeParameters_t p;
    CUEST_CHECK(cuestParametersCreate(CUEST_ECPDERIVATIVECOMPUTE_PARAMETERS, &p));
    run_derivative(var_buf, p, CUEST_ECPDERIVATIVECOMPUTE_PARAMETERS,
      [&](auto* vb, auto* td) {
        cuestECPDerivativeComputeWorkspaceQuery(ctx_, ecp_int_->plan(), p, vb, td, d_Da_, nullptr);
      },
      [&](auto* vb, auto* tw) {
        return cuestECPDerivativeCompute(ctx_, ecp_int_->plan(), p, vb, tw, d_Da_, d_ecp);
      });
    ecp_.resize(n3);
    cudaMemcpy(ecp_.data(), d_ecp, n3 * sizeof(double), cudaMemcpyDeviceToHost);
  } else {
    ecp_.assign(n3, 0.0);
  }

  // DF J/K derivative
  {
    const double ds = 2.0;
    bool is_hybrid = (xc_builder_ && xc_builder_->is_hybrid());
    bool is_lrc    = (xc_builder_ && xc_builder_->is_lrc());
    double cs = is_hybrid ? (is_lrc ? -1.0 : -xc_builder_->exchange_scale()) : 0.0;
    uint64_t ncm = is_hybrid ? 1ULL : 0ULL;
    const double* cmat = is_hybrid ? d_Co_.get() : nullptr;

    cuestDFSymmetricDerivativeComputeParameters_t p;
    CUEST_CHECK(cuestParametersCreate(CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS, &p));
    run_derivative(var_buf, p, CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS,
      [&](auto* vb, auto* td) {
        cuestDFSymmetricDerivativeComputeWorkspaceQuery(ctx_, dfjk_.plan(), p, vb, td, ds, d_Da_, cs, ncm, &nocc_, cmat, nullptr);
      },
      [&](auto* vb, auto* tw) {
        return cuestDFSymmetricDerivativeCompute(ctx_, dfjk_.plan(), p, vb, tw, ds, d_Da_, cs, ncm, &nocc_, cmat, d_df);
      });
  }

  // XC derivative
  if (xc_builder_) {
    cuestXCDerivativeRKSComputeParameters_t p;
    CUEST_CHECK(cuestParametersCreate(CUEST_XCDERIVATIVERKSCOMPUTE_PARAMETERS, &p));
    run_derivative(var_buf, p, CUEST_XCDERIVATIVERKSCOMPUTE_PARAMETERS,
      [&](auto* vb, auto* td) {
        cuestXCDerivativeRKSComputeWorkspaceQuery(ctx_, xc_builder_->plan(), p, vb, td, nocc_, d_Co_, nullptr);
      },
      [&](auto* vb, auto* tw) {
        return cuestXCDerivativeRKSCompute(ctx_, xc_builder_->plan(), p, vb, tw, nocc_, d_Co_, d_xc);
      });
  } else {
    xc_.assign(n3, 0.0);
  }

  // Fetch all results from device
  cudaDeviceSynchronize();
  ov_.resize(n3); cudaMemcpy(ov_.data(), d_ov, n3 * sizeof(double), cudaMemcpyDeviceToHost);
  ke_.resize(n3); cudaMemcpy(ke_.data(), d_ke, n3 * sizeof(double), cudaMemcpyDeviceToHost);
  po_.resize(n3); cudaMemcpy(po_.data(), d_po, n3 * sizeof(double), cudaMemcpyDeviceToHost);
  pc_.resize(n3); cudaMemcpy(pc_.data(), d_p2, n3 * sizeof(double), cudaMemcpyDeviceToHost);
  df_.resize(n3); cudaMemcpy(df_.data(), d_df, n3 * sizeof(double), cudaMemcpyDeviceToHost);
  if (xc_builder_) {
    xc_.resize(n3);
    cudaMemcpy(xc_.data(), d_xc, n3 * sizeof(double), cudaMemcpyDeviceToHost);
  }

  // Analytical total gradient for RKS with density fitting.
  // Verified against PySCF numerical gradients:
  //   dE/dR = -dEnuc/dR + 2*(ke+po+pc+ecp) - 2*ov + df + xc
  std::vector<double> total(n3, 0.0);
  for (size_t i = 0; i < n3; i++)
    total[i] = -nu_[i] + 2.0 * (ke_[i] + po_[i] + pc_[i] + ecp_[i])
               - 2.0 * ov_[i] + df_[i] + xc_[i];
  return total;
}

}  // namespace cuest
