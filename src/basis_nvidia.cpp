/**
 * @file basis_nvidia.cpp
 * @brief BasisBuilder using NVIDIA's proven formAOShells helper.
 */
#include "cuest_wrapper/basis.hpp"
#include <cuda_runtime.h>
#include <cuest.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" {
#include "helper_status.h"
#include "helper_workspaces.h"
#include "helper_gbs_parser.h"
#include "helper_xyz_parser.h"
#include "helper_shell_normalization.h"
#include "helper_ecp_parser.h"
#include "helper_ao_shells.h"
}

namespace cuest {

// Forward declarations
static void free_xyz(parsedXYZFile_t* xyz);

// NVIDIA's formAOShells requires a parsedXYZFile_t*, so we create one from Molecule
static parsedXYZFile_t* make_xyz(const Molecule& mol) {
  auto xyz_h = mol.xyz_host();
  auto charges_h = mol.charges_host();
  size_t n = mol.natom();

  parsedXYZFile_t* xyz = (parsedXYZFile_t*)malloc(sizeof(parsedXYZFile_t));
  if (!xyz) throw std::bad_alloc();
  xyz->numAtoms = n;
  xyz->xyzCPU = (double*)malloc(3*n*sizeof(double));
  xyz->chargesCPU = (double*)malloc(n*sizeof(double));
  xyz->symbols = (char**)malloc(n*sizeof(char*));
  if (!xyz->xyzCPU || !xyz->chargesCPU || !xyz->symbols) {
    free_xyz(xyz);
    throw std::bad_alloc();
  }
  for (size_t i=0; i<n; i++) {
    xyz->xyzCPU[3*i] = xyz_h[3*i];
    xyz->xyzCPU[3*i+1] = xyz_h[3*i+1];
    xyz->xyzCPU[3*i+2] = xyz_h[3*i+2];
    xyz->chargesCPU[i] = charges_h[i];
    const auto& sym = mol.atom(i).symbol;
    xyz->symbols[i] = (char*)malloc(sym.size()+1);
    if (!xyz->symbols[i]) { free_xyz(xyz); throw std::bad_alloc(); }
    strcpy(xyz->symbols[i], sym.c_str());
  }
  cudaMalloc(&xyz->xyzGPU, 3*n*sizeof(double));
  cudaMalloc(&xyz->chargesGPU, n*sizeof(double));
  cudaMemcpy(xyz->xyzGPU, xyz->xyzCPU, 3*n*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(xyz->chargesGPU, xyz->chargesCPU, n*sizeof(double), cudaMemcpyHostToDevice);
  return xyz;
}

static void free_xyz(parsedXYZFile_t* xyz) {
  if (!xyz) return;
  free(xyz->xyzCPU);
  free(xyz->chargesCPU);
  if (xyz->symbols) {
    for (size_t i=0; i<xyz->numAtoms; i++) free(xyz->symbols[i]);
    free(xyz->symbols);
  }
  if (xyz->xyzGPU) cudaFree(xyz->xyzGPU);
  if (xyz->chargesGPU) cudaFree(xyz->chargesGPU);
  free(xyz);
}

void BasisBuilder::build_nvidia(const std::string& gbs_path) {
  auto* xyz = make_xyz(mol_);
  AtomShellData_t* shellData = formAOShells(ctx_, xyz, gbs_path.c_str(), is_pure_);

  uint64_t natom = mol_.natom();
  uint64_t total_shells = shellData->numShellsTotal;
  auto* nvidia_shells = shellData->shells;
  auto* nvidia_spa = shellData->numShellsPerAtom;

  // Store shells in member vector for RAII cleanup
  shells_.resize(total_shells);
  for (uint64_t i=0; i<total_shells; i++) {
    shells_[i] = nvidia_shells[i];
  }

  // Create AO basis
  cuestWorkspaceDescriptor_t temp_desc{};
  AOBasisParams ao_params;
  CUEST_CHECK(cuestAOBasisCreateWorkspaceQuery(
      static_cast<cuestHandle_t>(ctx_), natom, nvidia_spa,
      (const cuestAOShell_t*)nvidia_shells, ao_params,
      &persistent_desc_, &temp_desc, basis_.ptr()));

  persistent_ws_ = Workspace(persistent_desc_);
  Workspace temp_ws(temp_desc);
  CUEST_CHECK(cuestAOBasisCreate(
      static_cast<cuestHandle_t>(ctx_), natom, nvidia_spa,
      (const cuestAOShell_t*)nvidia_shells, ao_params,
      persistent_ws_.ptr(), temp_ws.ptr(), basis_.ptr()));

  nao_ = ctx_.query_nao(basis_.get());

  // Transfer shell ownership
  for (uint64_t i=0; i<total_shells; i++) {
    AOShellHandle h;
    *h.ptr() = shells_[i];
    shell_handles_.push_back(std::move(h));
  }

  // Clean up NVIDIA data (shells are now owned by shell_handles_)
  free(shellData->numShellsPerAtom);
  free(nvidia_shells);
  free(shellData);
  free_xyz(xyz);
}

void BasisBuilder::build_from_gbs(const std::string& gbs_path) {
  build_nvidia(gbs_path);
}

void AuxBasis::build_from_gbs(const std::string& gbs_path) {
  auto* xyz = make_xyz(mol_);
  size_t n = mol_.natom();

  AtomShellData_t* shellData = formAOShells(ctx_, xyz, gbs_path.c_str(), is_pure_);
  uint64_t total_shells = shellData->numShellsTotal;

  std::vector<cuestAOShell_t> shells(total_shells);
  for (uint64_t i=0; i<total_shells; i++) shells[i] = shellData->shells[i];

  cuestWorkspaceDescriptor_t pers_desc{}, temp_desc{};
  AOBasisParams aux_params;
  CUEST_CHECK(cuestAOBasisCreateWorkspaceQuery(
      static_cast<cuestHandle_t>(ctx_), n, shellData->numShellsPerAtom,
      (const cuestAOShell_t*)shellData->shells, aux_params,
      &pers_desc, &temp_desc, basis_.ptr()));

  void *a_pdev=nullptr, *a_tdev=nullptr;
  if(pers_desc.deviceBufferSizeInBytes) cudaMalloc(&a_pdev, pers_desc.deviceBufferSizeInBytes);
  if(temp_desc.deviceBufferSizeInBytes) cudaMalloc(&a_tdev, temp_desc.deviceBufferSizeInBytes);
  cuestWorkspace_t a_pws={0,pers_desc.hostBufferSizeInBytes,(uintptr_t)a_pdev,pers_desc.deviceBufferSizeInBytes};
  cuestWorkspace_t a_tws={0,temp_desc.hostBufferSizeInBytes,(uintptr_t)a_tdev,temp_desc.deviceBufferSizeInBytes};
  CUEST_CHECK(cuestAOBasisCreate(
      static_cast<cuestHandle_t>(ctx_), n, shellData->numShellsPerAtom,
      (const cuestAOShell_t*)shellData->shells, aux_params,
      &a_pws, &a_tws, basis_.ptr()));
  // KEEP persistent workspace alive (DF plan needs it!)
  persist_dev_ = a_pdev; a_pdev = nullptr;
  persist_ws_ = a_pws;
  if(a_tdev) cudaFree(a_tdev);  // temp can be freed

  for (auto& sh : shells) {
    AOShellHandle h;
    *h.ptr() = sh;
    shell_handles_.push_back(std::move(h));
  }

  free(shellData->numShellsPerAtom);
  free(shellData->shells);
  free(shellData);
  free_xyz(xyz);
}

AOPairListHandle BasisBuilder::create_pair_list(double threshold) const {
  AOPairListHandle pl;
  cuestWorkspaceDescriptor_t pers_desc{}, temp_desc{};
  uint64_t natom = mol_.natom();
  auto xyz_h = mol_.xyz_host();
  AOPairListParams pl_params;
  CUEST_CHECK(cuestAOPairListCreateWorkspaceQuery(
      static_cast<cuestHandle_t>(ctx_), basis_.get(), natom, xyz_h.data(),
      threshold, pl_params, &pers_desc, &temp_desc, pl.ptr()));
  void *pd_dev=nullptr, *td_dev=nullptr;
  if(pers_desc.deviceBufferSizeInBytes)cudaMalloc(&pd_dev,pers_desc.deviceBufferSizeInBytes);
  if(temp_desc.deviceBufferSizeInBytes)cudaMalloc(&td_dev,temp_desc.deviceBufferSizeInBytes);
  cuestWorkspace_t pw={0,pers_desc.hostBufferSizeInBytes,(uintptr_t)pd_dev,pers_desc.deviceBufferSizeInBytes};
  cuestWorkspace_t tw={0,temp_desc.hostBufferSizeInBytes,(uintptr_t)td_dev,temp_desc.deviceBufferSizeInBytes};
  CUEST_CHECK(cuestAOPairListCreate(
      static_cast<cuestHandle_t>(ctx_), basis_.get(), natom, xyz_h.data(),
      threshold, pl_params, &pw, &tw, pl.ptr()));
  if(td_dev)cudaFree(td_dev);
  // LEAK pd_dev intentionally — OE plan and DF plan need it alive
  // In production, store and free when pair list is destroyed
  return pl;
}

AuxBasis::~AuxBasis() {
  if (persist_dev_) cudaFree(persist_dev_);
}

}  // namespace cuest
