#pragma once
#include <cuda_runtime.h>
#include <cuest.h>
#include <algorithm>
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

// Energy-weighted density: W = sum_k^occ eps_k * C_k * C_k^T
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

[[nodiscard]] inline std::vector<double> pack_cocc(
    const std::vector<double>& C, uint64_t nao, uint64_t nocc) {
  std::vector<double> Co(nao * nocc, 0.0);
  for (size_t k = 0; k < nocc; k++)
    for (size_t i = 0; i < nao; i++)
      Co[i + k * nao] = C[i + k * nao];
  return Co;
}

class GradientComputer {
 public:
  /// RKS: D_alpha, C_alpha, eps_alpha; occupation factor applied in assemble().
  GradientComputer(CuESTContext& ctx, BasisBuilder& basis, DFJKBuilder& dfjk,
                   XCBuilder* xc_builder, ECPIntegrals* ecp_int, const Molecule& mol,
                   uint64_t nocc,
                   const std::vector<double>& eps, const std::vector<double>& C,
                   const std::vector<double>& D_alpha);

  /// UKS: α/β channels. One-electron terms use D_tot / W_tot (no ×2).
  GradientComputer(CuESTContext& ctx, BasisBuilder& basis, DFJKBuilder& dfjk,
                   XCBuilder* xc_builder, ECPIntegrals* ecp_int, const Molecule& mol,
                   uint64_t nocc_a, uint64_t nocc_b,
                   const std::vector<double>& eps_a, const std::vector<double>& eps_b,
                   const std::vector<double>& C_a, const std::vector<double>& C_b,
                   const std::vector<double>& D_a, const std::vector<double>& D_b);

  std::vector<double> compute();
  bool is_uks() const { return uks_; }
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
  bool uks_{false};
  uint64_t nao_, nocc_, nocc_a_, nocc_b_, natom_;
  // RKS: d_Da_ = D_alpha, d_Wa_ = W_alpha, d_Co_ = Cocc_alpha
  // UKS: d_Da_ = D_tot, d_Wa_ = W_tot; d_Co_/d_Cob_ = padded Cocc α/β;
  //      d_Cconcat_ = [Cocc_a | Cocc_b] for DF hybrid exchange.
  DeviceArray d_Da_, d_Wa_, d_Co_, d_Cob_, d_Cconcat_;
  std::vector<double> ov_, ke_, po_, pc_, df_, nu_, xc_, ecp_;
};

inline GradientComputer::GradientComputer(
    CuESTContext& ctx, BasisBuilder& basis, DFJKBuilder& dfjk,
    XCBuilder* xc_builder, ECPIntegrals* ecp_int, const Molecule& mol,
    uint64_t nocc,
    const std::vector<double>& eps, const std::vector<double>& C,
    const std::vector<double>& D_alpha)
  : ctx_(ctx), basis_(basis), dfjk_(dfjk), xc_builder_(xc_builder), ecp_int_(ecp_int), mol_(mol),
    uks_(false), nao_(basis.nao()), nocc_(nocc), nocc_a_(nocc), nocc_b_(nocc), natom_(mol.natom()) {
  d_Da_.alloc(nao_ * nao_ * sizeof(double));
  CUDA_CHECK(cudaMemcpy(d_Da_, D_alpha.data(), nao_ * nao_ * sizeof(double), cudaMemcpyHostToDevice));
  auto W = ewd(eps, C, nao_, nocc_);
  d_Wa_.alloc(nao_ * nao_ * sizeof(double));
  CUDA_CHECK(cudaMemcpy(d_Wa_, W.data(), nao_ * nao_ * sizeof(double), cudaMemcpyHostToDevice));
  auto Co = pack_cocc(C, nao_, nocc_);
  d_Co_.alloc(nao_ * nocc_ * sizeof(double));
  CUDA_CHECK(cudaMemcpy(d_Co_, Co.data(), nao_ * nocc_ * sizeof(double), cudaMemcpyHostToDevice));
}

inline GradientComputer::GradientComputer(
    CuESTContext& ctx, BasisBuilder& basis, DFJKBuilder& dfjk,
    XCBuilder* xc_builder, ECPIntegrals* ecp_int, const Molecule& mol,
    uint64_t nocc_a, uint64_t nocc_b,
    const std::vector<double>& eps_a, const std::vector<double>& eps_b,
    const std::vector<double>& C_a, const std::vector<double>& C_b,
    const std::vector<double>& D_a, const std::vector<double>& D_b)
  : ctx_(ctx), basis_(basis), dfjk_(dfjk), xc_builder_(xc_builder), ecp_int_(ecp_int), mol_(mol),
    uks_(true), nao_(basis.nao()), nocc_(nocc_a), nocc_a_(nocc_a), nocc_b_(nocc_b),
    natom_(mol.natom()) {
  // One-electron / overlap: total density and total energy-weighted density.
  std::vector<double> Dtot(nao_ * nao_), Wtot(nao_ * nao_);
  auto Wa = ewd(eps_a, C_a, nao_, nocc_a_);
  auto Wb = ewd(eps_b, C_b, nao_, nocc_b_);
  for (size_t i = 0; i < nao_ * nao_; i++) {
    Dtot[i] = D_a[i] + D_b[i];
    Wtot[i] = Wa[i] + Wb[i];
  }
  d_Da_.alloc(nao_ * nao_ * sizeof(double));
  d_Wa_.alloc(nao_ * nao_ * sizeof(double));
  CUDA_CHECK(cudaMemcpy(d_Da_, Dtot.data(), nao_ * nao_ * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wa_, Wtot.data(), nao_ * nao_ * sizeof(double), cudaMemcpyHostToDevice));

  // cuEST UKS APIs require nocc > 0 per channel — pad empty spin with zeros.
  const uint64_t nocc_a_pad = std::max<uint64_t>(nocc_a_, 1);
  const uint64_t nocc_b_pad = std::max<uint64_t>(nocc_b_, 1);
  auto Coa = pack_cocc(C_a, nao_, nocc_a_pad);
  auto Cob = pack_cocc(C_b, nao_, nocc_b_pad);
  if (nocc_a_ == 0) std::fill(Coa.begin(), Coa.end(), 0.0);
  if (nocc_b_ == 0) std::fill(Cob.begin(), Cob.end(), 0.0);

  d_Co_.alloc(nao_ * nocc_a_pad * sizeof(double));
  d_Cob_.alloc(nao_ * nocc_b_pad * sizeof(double));
  CUDA_CHECK(cudaMemcpy(d_Co_, Coa.data(), nao_ * nocc_a_pad * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Cob_, Cob.data(), nao_ * nocc_b_pad * sizeof(double), cudaMemcpyHostToDevice));

  // Concatenated [Cα|Cβ] for DF hybrid exchange (cuEST UKS example).
  std::vector<double> Cconcat;
  Cconcat.reserve(Coa.size() + Cob.size());
  Cconcat.insert(Cconcat.end(), Coa.begin(), Coa.end());
  Cconcat.insert(Cconcat.end(), Cob.begin(), Cob.end());
  d_Cconcat_.alloc(Cconcat.size() * sizeof(double));
  CUDA_CHECK(cudaMemcpy(d_Cconcat_, Cconcat.data(),
                        Cconcat.size() * sizeof(double), cudaMemcpyHostToDevice));
}

// Helper: allocate temp workspace, run derivative compute, free temp
namespace {
template <typename ParamsT, typename QueryFn, typename ComputeFn>
void run_derivative(cuestWorkspaceDescriptor_t& var_buf,
                    ParamsT& params, cuestParametersType_t ptype,
                    QueryFn query, ComputeFn compute,
                    const char* label = "derivative") {
  cuestWorkspaceDescriptor_t td{};
  cuestStatus_t qst = query(&var_buf, &td);
  if (qst != CUEST_STATUS_SUCCESS) {
    cuestParametersDestroy(ptype, params);
    throw std::runtime_error(std::string("Gradient ") + label +
                             " workspace query failed with code " +
                             std::to_string(static_cast<int>(qst)));
  }
  Workspace temp_ws(td);
  CUDA_CHECK(cudaDeviceSynchronize());
  cuestStatus_t st = compute(&var_buf, temp_ws.ptr());
  CUDA_CHECK(cudaDeviceSynchronize());
  cuestParametersDestroy(ptype, params);
  if (st != CUEST_STATUS_SUCCESS)
    throw std::runtime_error(std::string("Gradient ") + label +
                             " compute failed with code " +
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

  // Overlap: Tr[W * dS/dR]  (RKS: W_α; UKS: W_α+W_β)
  {
    cuestOverlapDerivativeComputeParameters_t p;
    CUEST_CHECK(cuestParametersCreate(CUEST_OVERLAPDERIVATIVECOMPUTE_PARAMETERS, &p));
    run_derivative(var_buf, p, CUEST_OVERLAPDERIVATIVECOMPUTE_PARAMETERS,
      [&](auto* /*vb*/, auto* td) {
        return cuestOverlapDerivativeComputeWorkspaceQuery(ctx_, oe.plan(), p, td, d_Wa_, nullptr);
      },
      [&](auto* /*vb*/, auto* tw) {
        return cuestOverlapDerivativeCompute(ctx_, oe.plan(), p, tw, d_Wa_, d_ov);
      }, "overlap");
  }

  // Kinetic / potential / ECP: Tr[D * dH/dR]  (RKS: D_α; UKS: D_tot)
  {
    cuestKineticDerivativeComputeParameters_t p;
    CUEST_CHECK(cuestParametersCreate(CUEST_KINETICDERIVATIVECOMPUTE_PARAMETERS, &p));
    run_derivative(var_buf, p, CUEST_KINETICDERIVATIVECOMPUTE_PARAMETERS,
      [&](auto* /*vb*/, auto* td) {
        return cuestKineticDerivativeComputeWorkspaceQuery(ctx_, oe.plan(), p, td, d_Da_, nullptr);
      },
      [&](auto* /*vb*/, auto* tw) {
        return cuestKineticDerivativeCompute(ctx_, oe.plan(), p, tw, d_Da_, d_ke);
      }, "kinetic");
  }

  {
    cuestPotentialDerivativeComputeParameters_t p;
    CUEST_CHECK(cuestParametersCreate(CUEST_POTENTIALDERIVATIVECOMPUTE_PARAMETERS, &p));
    run_derivative(var_buf, p, CUEST_POTENTIALDERIVATIVECOMPUTE_PARAMETERS,
      [&](auto* /*vb*/, auto* td) {
        return cuestPotentialDerivativeComputeWorkspaceQuery(ctx_, oe.plan(), p, td, natom_, dx, dq, d_Da_, nullptr, nullptr);
      },
      [&](auto* /*vb*/, auto* tw) {
        return cuestPotentialDerivativeCompute(ctx_, oe.plan(), p, tw, natom_, dx, dq, d_Da_, d_po, d_p2);
      }, "potential");
  }

  if (ecp_int_) {
    DeviceArray d_ecp(n3 * sizeof(double));
    cuestECPDerivativeComputeParameters_t p;
    CUEST_CHECK(cuestParametersCreate(CUEST_ECPDERIVATIVECOMPUTE_PARAMETERS, &p));
    run_derivative(var_buf, p, CUEST_ECPDERIVATIVECOMPUTE_PARAMETERS,
      [&](auto* vb, auto* td) {
        return cuestECPDerivativeComputeWorkspaceQuery(ctx_, ecp_int_->plan(), p, vb, td, d_Da_, nullptr);
      },
      [&](auto* vb, auto* tw) {
        return cuestECPDerivativeCompute(ctx_, ecp_int_->plan(), p, vb, tw, d_Da_, d_ecp);
      }, "ecp");
    ecp_.resize(n3);
    cudaMemcpy(ecp_.data(), d_ecp, n3 * sizeof(double), cudaMemcpyDeviceToHost);
  } else {
    ecp_.assign(n3, 0.0);
  }

  // DF J/K derivative
  // RKS: ds=2, cs=-1, ncm=1  (cuEST B3LYP RKS example)
  // UKS: ds=0.5, cs=-0.5, ncm=2 with D=Da+Db (cuEST PBE0 UKS example)
  {
    const bool is_hybrid = (xc_builder_ &&
                            (xc_builder_->is_hybrid() || xc_builder_->is_hf()));
    const double ds = uks_ ? 0.5 : 2.0;
    const double cs = is_hybrid ? (uks_ ? -0.5 : -1.0) : 0.0;
    const uint64_t ncm = is_hybrid ? (uks_ ? 2ULL : 1ULL) : 0ULL;

    uint64_t nocc_arr[2] = {0, 0};
    const uint64_t* nocc_ptr = nullptr;
    const double* cmat = nullptr;
    if (is_hybrid) {
      if (uks_) {
        nocc_arr[0] = std::max<uint64_t>(nocc_a_, 1);
        nocc_arr[1] = std::max<uint64_t>(nocc_b_, 1);
        nocc_ptr = nocc_arr;
        cmat = d_Cconcat_.get();
      } else {
        nocc_arr[0] = nocc_;
        nocc_ptr = nocc_arr;
        cmat = d_Co_.get();
      }
    }

    cuestDFSymmetricDerivativeComputeParameters_t p;
    CUEST_CHECK(cuestParametersCreate(CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS, &p));
    {
      auto pol = CUEST_DFSYMMETRICDERIVATIVECOMPUTE_MEMORY_POLICY_OVERWRITE;
      CUEST_CHECK(cuestParametersConfigure(
          CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS, p,
          CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS_MEMORY_POLICY,
          &pol, sizeof(pol)));
    }
    run_derivative(var_buf, p, CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS,
      [&](auto* vb, auto* td) {
        return cuestDFSymmetricDerivativeComputeWorkspaceQuery(
            ctx_, dfjk_.plan(), p, vb, td, ds, d_Da_, cs, ncm, nocc_ptr, cmat, nullptr);
      },
      [&](auto* vb, auto* tw) {
        return cuestDFSymmetricDerivativeCompute(
            ctx_, dfjk_.plan(), p, vb, tw, ds, d_Da_, cs, ncm, nocc_ptr, cmat, d_df);
      }, "df-jk");
  }

  // XC derivative (skipped for pure HF — cuEST XC grad rejects FUNCTIONAL_HF)
  if (xc_builder_ && !xc_builder_->is_hf()) {
    if (uks_) {
      const uint64_t nocc_a_pad = std::max<uint64_t>(nocc_a_, 1);
      const uint64_t nocc_b_pad = std::max<uint64_t>(nocc_b_, 1);
      cuestXCDerivativeUKSComputeParameters_t p;
      CUEST_CHECK(cuestParametersCreate(CUEST_XCDERIVATIVEUKSCOMPUTE_PARAMETERS, &p));
      run_derivative(var_buf, p, CUEST_XCDERIVATIVEUKSCOMPUTE_PARAMETERS,
        [&](auto* vb, auto* td) {
          return cuestXCDerivativeUKSComputeWorkspaceQuery(
              ctx_, xc_builder_->plan(), p, vb, td,
              nocc_a_pad, nocc_b_pad, d_Co_, d_Cob_, nullptr);
        },
        [&](auto* vb, auto* tw) {
          return cuestXCDerivativeUKSCompute(
              ctx_, xc_builder_->plan(), p, vb, tw,
              nocc_a_pad, nocc_b_pad, d_Co_, d_Cob_, d_xc);
        }, "xc-uks");
    } else {
      cuestXCDerivativeRKSComputeParameters_t p;
      CUEST_CHECK(cuestParametersCreate(CUEST_XCDERIVATIVERKSCOMPUTE_PARAMETERS, &p));
      run_derivative(var_buf, p, CUEST_XCDERIVATIVERKSCOMPUTE_PARAMETERS,
        [&](auto* vb, auto* td) {
          return cuestXCDerivativeRKSComputeWorkspaceQuery(
              ctx_, xc_builder_->plan(), p, vb, td, nocc_, d_Co_, nullptr);
        },
        [&](auto* vb, auto* tw) {
          return cuestXCDerivativeRKSCompute(
              ctx_, xc_builder_->plan(), p, vb, tw, nocc_, d_Co_, d_xc);
        }, "xc");
    }
  } else {
    xc_.assign(n3, 0.0);
  }

  cudaDeviceSynchronize();
  ov_.resize(n3); cudaMemcpy(ov_.data(), d_ov, n3 * sizeof(double), cudaMemcpyDeviceToHost);
  ke_.resize(n3); cudaMemcpy(ke_.data(), d_ke, n3 * sizeof(double), cudaMemcpyDeviceToHost);
  po_.resize(n3); cudaMemcpy(po_.data(), d_po, n3 * sizeof(double), cudaMemcpyDeviceToHost);
  pc_.resize(n3); cudaMemcpy(pc_.data(), d_p2, n3 * sizeof(double), cudaMemcpyDeviceToHost);
  df_.resize(n3); cudaMemcpy(df_.data(), d_df, n3 * sizeof(double), cudaMemcpyDeviceToHost);
  if (xc_builder_ && !xc_builder_->is_hf()) {
    xc_.resize(n3);
    cudaMemcpy(xc_.data(), d_xc, n3 * sizeof(double), cudaMemcpyDeviceToHost);
  } else if (xc_.size() != n3) {
    xc_.assign(n3, 0.0);
  }

  // Assemble total gradient.
  // RKS (D_α convention):  -dEnuc + 2*(ke+po+pc+ecp) - 2*ov + df + xc
  // UKS (D_tot convention): -dEnuc + 1*(ke+po+pc+ecp) - 1*ov + df + xc
  const double oe_fac = uks_ ? 1.0 : 2.0;
  std::vector<double> total(n3, 0.0);
  for (size_t i = 0; i < n3; i++)
    total[i] = -nu_[i] + oe_fac * (ke_[i] + po_[i] + pc_[i] + ecp_[i])
               - oe_fac * ov_[i] + df_[i] + xc_[i];
  return total;
}

}  // namespace cuest
