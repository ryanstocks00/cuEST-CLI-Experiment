/**
 * @file basis_ecp.cpp
 * @brief ECP builder implementation (uses cuEST C API directly).
 */
#include "cuest_wrapper/basis.hpp"
#include <cuda_runtime.h>
#include <cuest.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cuest {

void ECPBuilder::build_from_file(const std::string& ecp_path) {
  auto unique = mol_.unique_symbols();
  std::vector<ECPShellSet> element_ecps;
  for (const auto& sym : unique) {
    element_ecps.push_back(parse_ecp_for_element(ecp_path, sym));
  }

  for (size_t i = 0; i < mol_.natom(); i++) {
    int bidx = -1;
    for (size_t j = 0; j < unique.size(); j++)
      if (mol_.atom(i).symbol == unique[j]) { bidx = static_cast<int>(j); break; }
    if (element_ecps[bidx].n_shells > 0)
      ecp_indices_.push_back(i);
  }

  num_active_ecp_ = ecp_indices_.size();
  has_ecp_ = num_active_ecp_ > 0;
  if (!has_ecp_) return;

  // Build ECP shells and atoms using raw C API for reliability
  for (size_t u = 0; u < unique.size(); u++) {
    const auto& ecp = element_ecps[u];
    if (ecp.n_shells == 0) continue;
    size_t num_shells = ecp.n_shells - 1;
    ECPShellHandle top;
    CUEST_CHECK(cuestECPShellCreate(
        static_cast<cuestHandle_t>(ctx_),
        ecp.shell_types[0], ecp.num_primitives[0],
        ecp.Ns.data(), ecp.coefficients.data(), ecp.exponents.data(),
        ECPShellParams{}, top.ptr()));
    ecp_top_shell_handles_.push_back(std::move(top));
    for (size_t k = 0; k < num_shells; k++) {
      uint64_t offset = ecp.primitive_offsets[k + 1];
      ECPShellHandle sh;
      CUEST_CHECK(cuestECPShellCreate(
          static_cast<cuestHandle_t>(ctx_),
          ecp.shell_types[k + 1], ecp.num_primitives[k + 1],
          &ecp.Ns[offset], &ecp.coefficients[offset], &ecp.exponents[offset],
          ECPShellParams{}, sh.ptr()));
      ecp_shell_handles_.push_back(std::move(sh));
    }
  }

  // Create ECP atoms: one per active atom center (not per unique element!)
  uint64_t shell_head = 0, top_head = 0;
  for (size_t a = 0; a < mol_.natom(); a++) {
    // Find which unique element this atom is
    int u = -1;
    for (size_t j = 0; j < unique.size(); j++)
      if (mol_.atom(a).symbol == unique[j]) { u = (int)j; break; }
    if (u < 0) continue;

    const auto& ecp = element_ecps[u];
    if (ecp.n_shells == 0) continue;

    size_t num_shells = ecp.n_shells - 1;
    // Find the shell handles for this unique element
    uint64_t u_shell_head = 0, u_top_head = 0;
    for (int j = 0; j < u; j++) {
      if (element_ecps[j].n_shells > 0) {
        u_shell_head += element_ecps[j].n_shells - 1;
        u_top_head += 1;
      }
    }

    std::vector<cuestECPShell_t> sh_raw(num_shells);
    for (size_t k = 0; k < num_shells; k++)
      sh_raw[k] = ecp_shell_handles_[u_shell_head + k].get();

    ECPAtomHandle atom;
    CUEST_CHECK(cuestECPAtomCreate(
        static_cast<cuestHandle_t>(ctx_),
        ecp.n_elec, num_shells, sh_raw.data(),
        ecp_top_shell_handles_[u_top_head].get(),
        ECPAtomParams{}, atom.ptr()));
    ecp_atom_handles_.push_back(std::move(atom));
  }

  for (auto& a : ecp_atom_handles_)
    ecp_atoms_raw_.push_back(a.get());
}

ECPIntPlanHandle ECPBuilder::create_ecp_int_plan(cuestAOBasis_t basis,
                                                   const double* xyz_host) const {
  ECPIntPlanHandle plan;
  cuestWorkspaceDescriptor_t pers_desc{}, temp_desc{};
  ECPIntPlanParams ecp_params;
  CUEST_CHECK(cuestECPIntPlanCreateWorkspaceQuery(
      static_cast<cuestHandle_t>(ctx_), basis, xyz_host,
      num_active_ecp_, ecp_indices_.data(), ecp_atoms_raw_.data(),
      ecp_params, &pers_desc, &temp_desc, plan.ptr()));
  void *pd_dev=nullptr, *td_dev=nullptr;
  if(pers_desc.deviceBufferSizeInBytes)cudaMalloc(&pd_dev,pers_desc.deviceBufferSizeInBytes);
  if(temp_desc.deviceBufferSizeInBytes)cudaMalloc(&td_dev,temp_desc.deviceBufferSizeInBytes);
  cuestWorkspace_t pw={0,pers_desc.hostBufferSizeInBytes,(uintptr_t)pd_dev,pers_desc.deviceBufferSizeInBytes};
  cuestWorkspace_t tw={0,temp_desc.hostBufferSizeInBytes,(uintptr_t)td_dev,temp_desc.deviceBufferSizeInBytes};
  CUEST_CHECK(cuestECPIntPlanCreate(
      static_cast<cuestHandle_t>(ctx_), basis, xyz_host,
      num_active_ecp_, ecp_indices_.data(), ecp_atoms_raw_.data(),
      ecp_params, &pw, &tw, plan.ptr()));
  if(pd_dev)cudaFree(pd_dev); if(td_dev)cudaFree(td_dev);
  return plan;
}

}  // namespace cuest
