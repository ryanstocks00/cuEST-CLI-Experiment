/**
 * @file basis_ecp.cpp
 * @brief ECP builder — extracts ECP data from BSE JSON.
 */
#include "cuest_wrapper/basis.hpp"
#include "cuest_wrapper/basis_json.hpp"
#include <cuda_runtime.h>
#include <cuest.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cuest {

void ECPBuilder::build_from_json(const std::string& json_path) {
  BSEJsonReader reader(json_path);

  std::vector<int> atom_to_ecp_idx(mol_.natom(), -1);
  std::vector<JsonECP> ecp_data;

  for (size_t a = 0; a < mol_.natom(); a++) {
    int Z = mol_.atom(a).atomic_number;
    if (reader.has_ecp(Z)) {
      ecp_indices_.push_back(a);
      atom_to_ecp_idx[a] = static_cast<int>(ecp_data.size());
      ecp_data.push_back(reader.get_ecp(Z));
    }
  }

  num_active_ecp_ = ecp_indices_.size();
  total_ecp_electrons_ = 0;
  for (const auto& e : ecp_data) total_ecp_electrons_ += e.n_elec;
  has_ecp_ = num_active_ecp_ > 0;
  if (!has_ecp_) return;

  // Build ECP shells
  for (size_t u = 0; u < ecp_data.size(); u++) {
    const auto& ecp = ecp_data[u];
    if (ecp.shell_types.empty()) continue;

    ECPShellHandle top;
    CUEST_CHECK(cuestECPShellCreate(
        static_cast<cuestHandle_t>(ctx_),
        ecp.shell_types[0], ecp.num_primitives[0],
        ecp.Ns.data(), ecp.coefficients.data(), ecp.exponents.data(),
        ECPShellParams{}, top.ptr()));
    ecp_top_shell_handles_.push_back(std::move(top));

    size_t offset = ecp.num_primitives[0];
    for (size_t k = 1; k < ecp.shell_types.size(); k++) {
      ECPShellHandle sh;
      CUEST_CHECK(cuestECPShellCreate(
          static_cast<cuestHandle_t>(ctx_),
          ecp.shell_types[k], ecp.num_primitives[k],
          &ecp.Ns[offset], &ecp.coefficients[offset], &ecp.exponents[offset],
          ECPShellParams{}, sh.ptr()));
      ecp_shell_handles_.push_back(std::move(sh));
      offset += ecp.num_primitives[k];
    }
  }

  // Create ECP atoms — one per active atom center
  for (size_t a = 0; a < mol_.natom(); a++) {
    int u = atom_to_ecp_idx[a];
    if (u < 0) continue;

    const auto& ecp = ecp_data[u];
    size_t num_shells = ecp.shell_types.size() - 1;

    uint64_t shell_head = 0, top_head = 0;
    for (int j = 0; j < u; j++) {
      if (!ecp_data[j].shell_types.empty()) {
        shell_head += ecp_data[j].shell_types.size() - 1;
        top_head += 1;
      }
    }

    std::vector<cuestECPShell_t> sh_raw(num_shells);
    for (size_t k = 0; k < num_shells; k++)
      sh_raw[k] = ecp_shell_handles_[shell_head + k].get();

    ECPAtomHandle atom;
    CUEST_CHECK(cuestECPAtomCreate(
        static_cast<cuestHandle_t>(ctx_),
        ecp.n_elec, num_shells, sh_raw.data(),
        ecp_top_shell_handles_[top_head].get(),
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
  void* pd_dev = nullptr;
  void* td_dev = nullptr;
  if (pers_desc.deviceBufferSizeInBytes)
    cudaMalloc(&pd_dev, pers_desc.deviceBufferSizeInBytes);
  if (temp_desc.deviceBufferSizeInBytes)
    cudaMalloc(&td_dev, temp_desc.deviceBufferSizeInBytes);
  cuestWorkspace_t pw = {0, pers_desc.hostBufferSizeInBytes,
                         (uintptr_t)pd_dev, pers_desc.deviceBufferSizeInBytes};
  cuestWorkspace_t tw = {0, temp_desc.hostBufferSizeInBytes,
                         (uintptr_t)td_dev, temp_desc.deviceBufferSizeInBytes};
  CUEST_CHECK(cuestECPIntPlanCreate(
      static_cast<cuestHandle_t>(ctx_), basis, xyz_host,
      num_active_ecp_, ecp_indices_.data(), ecp_atoms_raw_.data(),
      ecp_params, &pw, &tw, plan.ptr()));
  if (td_dev) cudaFree(td_dev);
  return plan;
}

}  // namespace cuest
