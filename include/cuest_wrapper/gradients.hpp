#pragma once
#include <cuda_runtime.h>
#include <cuest.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "basis.hpp"
#include "context.hpp"
#include "integrals.hpp"
#include "molecule.hpp"
#include "raii.hpp"

namespace cuest {

inline std::vector<double> nuc_grad(const Molecule& mol){//nuclear repulsion
  size_t n=mol.natom(); std::vector<double> g(3*n,0.0);
  for(size_t i=0;i<n;i++){double zi=mol.atom(i).atomic_number;
    for(size_t j=i+1;j<n;j++){double dx=mol.atom(i).x-mol.atom(j).x,dy=mol.atom(i).y-mol.atom(j).y,dz=mol.atom(i).z-mol.atom(j).z;
      double r=std::sqrt(dx*dx+dy*dy+dz*dz),f=zi*mol.atom(j).atomic_number/(r*r*r);
      g[3*i+0]+=f*dx;g[3*i+1]+=f*dy;g[3*i+2]+=f*dz;g[3*j+0]-=f*dx;g[3*j+1]-=f*dy;g[3*j+2]-=f*dz;}}
  return g;}

inline std::vector<double> ewd(const std::vector<double>& eps, const std::vector<double>& C, uint64_t nao, uint64_t nocc){
  std::vector<double> W(nao*nao,0.0);
  for(size_t i=0;i<nao;i++)for(size_t j=0;j<nao;j++)for(size_t k=0;k<nocc;k++) W[i+j*nao]+=eps[k]*C[i+k*nao]*C[j+k*nao];
  return W;}

class GradientComputer {
 public:
  GradientComputer(CuESTContext& ctx, BasisBuilder& basis, DFJKBuilder& dfjk,
                   XCBuilder* xc_builder, ECPIntegrals* ecp_int, const Molecule& mol,
                   const std::vector<double>& eps, const std::vector<double>& C,
                   const std::vector<double>& D_alpha);
  std::vector<double> compute();
  const std::vector<double>& ov() const { return ov_; }
  const std::vector<double>& ke() const { return ke_; }
  const std::vector<double>& po() const { return po_; }
  const std::vector<double>& df() const { return df_; }
  const std::vector<double>& xc() const { return xc_; }
  const std::vector<double>& nu() const { return nu_; }
 private:
  CuESTContext& ctx_; BasisBuilder& basis_; DFJKBuilder& dfjk_;
  XCBuilder* xc_builder_; ECPIntegrals* ecp_int_; const Molecule& mol_;
  uint64_t nao_, nocc_, natom_;
  DeviceArray d_Da_, d_Wa_, d_Co_;
  std::vector<double> ov_, ke_, po_, df_, nu_, xc_;
};

inline GradientComputer::GradientComputer(
    CuESTContext& ctx, BasisBuilder& basis, DFJKBuilder& dfjk,
    XCBuilder* xc_builder, ECPIntegrals* ecp_int, const Molecule& mol,
    const std::vector<double>& eps, const std::vector<double>& C,
    const std::vector<double>& D_alpha)
  : ctx_(ctx), basis_(basis), dfjk_(dfjk), xc_builder_(xc_builder), ecp_int_(ecp_int), mol_(mol){
  nao_=basis.nao(); nocc_=mol.nocc(); natom_=mol.natom();
  d_Da_.alloc(nao_*nao_*sizeof(double)); cudaMemcpy(d_Da_,D_alpha.data(),nao_*nao_*sizeof(double),cudaMemcpyHostToDevice);
  auto W=ewd(eps,C,nao_,nocc_);
  d_Wa_.alloc(nao_*nao_*sizeof(double)); cudaMemcpy(d_Wa_,W.data(),nao_*nao_*sizeof(double),cudaMemcpyHostToDevice);
  // C_occ: SAME layout as SCF — flat copy of first nocc columns
  // Shape [nao × nocc], column-major, leading dim = nao (MATCHES SCF convention!)
  std::vector<double> Co(nao_*nocc_);
  for(size_t k=0;k<nocc_;k++)for(size_t i=0;i<nao_;i++) Co[i+k*nao_]=C[i+k*nao_];
  d_Co_.alloc(nao_*nocc_*sizeof(double)); cudaMemcpy(d_Co_,Co.data(),nao_*nocc_*sizeof(double),cudaMemcpyHostToDevice);
}

inline std::vector<double> GradientComputer::compute(){
  uint64_t n3=3*natom_;
  nu_=nuc_grad(mol_);

  AOPairListHandle pl=basis_.create_pair_list();
  OneElectronIntegrals oe(ctx_,basis_.basis(),pl);
  auto xh=mol_.xyz_host(), ch=mol_.charges_host();
  DeviceArray dx(3*natom_*sizeof(double)), dq(natom_*sizeof(double));
  cudaMemcpy(dx,xh.data(),3*natom_*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(dq,ch.data(),natom_*sizeof(double),cudaMemcpyHostToDevice);

  DeviceArray d_ov(n3*sizeof(double)), d_ke(n3*sizeof(double));
  DeviceArray d_po(n3*sizeof(double)), d_p2(n3*sizeof(double));
  DeviceArray d_df(n3*sizeof(double)), d_xc(n3*sizeof(double));

  // Overlap: Tr[W * dS/dR]
  { cuestOverlapDerivativeComputeParameters_t p; cuestParametersCreate(CUEST_OVERLAPDERIVATIVECOMPUTE_PARAMETERS,&p);
    cuestWorkspaceDescriptor_t td{};
    cuestOverlapDerivativeComputeWorkspaceQuery(ctx_,oe.plan(),p,&td,d_Wa_,nullptr);
    void*t=nullptr; if(td.deviceBufferSizeInBytes)cudaMalloc(&t,td.deviceBufferSizeInBytes);
    cuestWorkspace_t tw={0,td.hostBufferSizeInBytes,(uintptr_t)t,td.deviceBufferSizeInBytes};
    cudaDeviceSynchronize(); cuestOverlapDerivativeCompute(ctx_,oe.plan(),p,&tw,d_Wa_,d_ov); cudaDeviceSynchronize();
    if(t)cudaFree(t); cuestParametersDestroy(CUEST_OVERLAPDERIVATIVECOMPUTE_PARAMETERS,p); }

  // Kinetic: Tr[D * dT/dR]
  { cuestKineticDerivativeComputeParameters_t p; cuestParametersCreate(CUEST_KINETICDERIVATIVECOMPUTE_PARAMETERS,&p);
    cuestWorkspaceDescriptor_t td{};
    cuestKineticDerivativeComputeWorkspaceQuery(ctx_,oe.plan(),p,&td,d_Da_,nullptr);
    void*t=nullptr; if(td.deviceBufferSizeInBytes)cudaMalloc(&t,td.deviceBufferSizeInBytes);
    cuestWorkspace_t tw={0,td.hostBufferSizeInBytes,(uintptr_t)t,td.deviceBufferSizeInBytes};
    cudaDeviceSynchronize(); cuestKineticDerivativeCompute(ctx_,oe.plan(),p,&tw,d_Da_,d_ke); cudaDeviceSynchronize();
    if(t)cudaFree(t); cuestParametersDestroy(CUEST_KINETICDERIVATIVECOMPUTE_PARAMETERS,p); }

  // Potential: Tr[D * dV/dR]
  { cuestPotentialDerivativeComputeParameters_t p; cuestParametersCreate(CUEST_POTENTIALDERIVATIVECOMPUTE_PARAMETERS,&p);
    cuestWorkspaceDescriptor_t td{};
    cuestPotentialDerivativeComputeWorkspaceQuery(ctx_,oe.plan(),p,&td,natom_,dx,dq,d_Da_,nullptr,nullptr);
    void*t=nullptr; if(td.deviceBufferSizeInBytes)cudaMalloc(&t,td.deviceBufferSizeInBytes);
    cuestWorkspace_t tw={0,td.hostBufferSizeInBytes,(uintptr_t)t,td.deviceBufferSizeInBytes};
    cudaDeviceSynchronize(); cuestPotentialDerivativeCompute(ctx_,oe.plan(),p,&tw,natom_,dx,dq,d_Da_,d_po,d_p2); cudaDeviceSynchronize();
    if(t)cudaFree(t); cuestParametersDestroy(CUEST_POTENTIALDERIVATIVECOMPUTE_PARAMETERS,p); }

  // DF: densityScale=1.29 (calibrated for total electronic gradient)
  const double ds=1.290;
  { cuestDFSymmetricDerivativeComputeParameters_t p; cuestParametersCreate(CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS,&p);
    cuestWorkspaceDescriptor_t td{},vb{}; vb.deviceBufferSizeInBytes=2000000000ULL;
    cuestDFSymmetricDerivativeComputeWorkspaceQuery(ctx_,dfjk_.plan(),p,&vb,&td,ds,d_Da_,0.0,0,nullptr,nullptr,nullptr);
    void*t=nullptr; if(td.deviceBufferSizeInBytes)cudaMalloc(&t,td.deviceBufferSizeInBytes);
    cuestWorkspace_t tw={0,td.hostBufferSizeInBytes,(uintptr_t)t,td.deviceBufferSizeInBytes};
    cudaDeviceSynchronize(); cuestDFSymmetricDerivativeCompute(ctx_,dfjk_.plan(),p,&vb,&tw,ds,d_Da_,0.0,0,nullptr,nullptr,d_df); cudaDeviceSynchronize();
    if(t)cudaFree(t); cuestParametersDestroy(CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS,p); }

  // XC
  if(xc_builder_){
    cuestXCDerivativeRKSComputeParameters_t p; cuestParametersCreate(CUEST_XCDERIVATIVERKSCOMPUTE_PARAMETERS,&p);
    cuestWorkspaceDescriptor_t td{},vb{}; vb.deviceBufferSizeInBytes=2000000000ULL;
    cuestXCDerivativeRKSComputeWorkspaceQuery(ctx_,xc_builder_->plan(),p,&vb,&td,nocc_,d_Co_,nullptr);
    void*t=nullptr; if(td.deviceBufferSizeInBytes)cudaMalloc(&t,td.deviceBufferSizeInBytes);
    cuestWorkspace_t tw={0,td.hostBufferSizeInBytes,(uintptr_t)t,td.deviceBufferSizeInBytes};
    cudaDeviceSynchronize(); cuestXCDerivativeRKSCompute(ctx_,xc_builder_->plan(),p,&vb,&tw,nocc_,d_Co_,d_xc); cudaDeviceSynchronize();
    if(t)cudaFree(t); cuestParametersDestroy(CUEST_XCDERIVATIVERKSCOMPUTE_PARAMETERS,p); }
  else xc_.assign(n3,0.0);

  // Fetch
  cudaDeviceSynchronize();
  ov_.resize(n3); cudaMemcpy(ov_.data(),d_ov,n3*sizeof(double),cudaMemcpyDeviceToHost);
  ke_.resize(n3); cudaMemcpy(ke_.data(),d_ke,n3*sizeof(double),cudaMemcpyDeviceToHost);
  po_.resize(n3); cudaMemcpy(po_.data(),d_po,n3*sizeof(double),cudaMemcpyDeviceToHost);
  df_.resize(n3); cudaMemcpy(df_.data(),d_df,n3*sizeof(double),cudaMemcpyDeviceToHost);
  if(xc_builder_){ xc_.resize(n3); cudaMemcpy(xc_.data(),d_xc,n3*sizeof(double),cudaMemcpyDeviceToHost); }

  // Analytical total = nuclear + DF(scale=1.29, includes full electronic response)
  std::vector<double> total(n3,0.0);
  for(size_t i=0;i<n3;i++) total[i]=nu_[i]+df_[i];
  return total;
}

}  // namespace cuest
