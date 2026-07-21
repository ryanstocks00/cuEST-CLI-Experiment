/**
 * @file basis_ecp_from_json.cpp
 * @brief ECP builder — extracts ECP data from BSE JSON (app layer).
 */
#include "cuest_wrapper/basis.hpp"
#include "cuest_wrapper/nvtx.hpp"
#include "io/basis_json.hpp"
#include <cuda_runtime.h>
#include <cuest.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cuest {

void ECPBuilder::build_from_json(const std::string& json_path) {
  BSEJsonReader reader(json_path);

  ecp_electrons_per_atom_.assign(mol_.natom(), 0);
  std::vector<int> atom_to_ecp_idx(mol_.natom(), -1);
  std::vector<JsonECP> ecp_data;

  for (size_t a = 0; a < mol_.natom(); a++) {
    int Z = mol_.atom(a).atomic_number;
    if (reader.has_ecp(Z)) {
      ecp_indices_.push_back(a);
      atom_to_ecp_idx[a] = static_cast<int>(ecp_data.size());
      JsonECP ecp = reader.get_ecp(Z);
      ecp_electrons_per_atom_[a] = static_cast<int>(ecp.n_elec);
      ecp_data.push_back(std::move(ecp));
    }
  }

  num_active_ecp_ = ecp_indices_.size();
  total_ecp_electrons_ = 0;
  for (const auto& e : ecp_data) total_ecp_electrons_ += e.n_elec;
  has_ecp_ = num_active_ecp_ > 0;
  if (!has_ecp_) return;

  for (size_t u = 0; u < ecp_data.size(); u++) {
    const auto& ecp = ecp_data[u];
    if (ecp.shell_types.empty()) continue;

    // Validate / sort so shells[i] has angular momentum L == i (cuEST contract).
    // BSE typically provides ul (max L) first, then L=0..Lmax-1.
    ECPShellHandle top;
    CUEST_NVTX("cuestECPShellCreate",
               cuestECPShellCreate(
                   static_cast<cuestHandle_t>(ctx_),
                   ecp.shell_types[0], ecp.num_primitives[0],
                   ecp.Ns.data(), ecp.coefficients.data(), ecp.exponents.data(),
                   ECPShellParams{}, top.ptr()));
    ecp_top_shell_handles_.push_back(std::move(top));

    size_t offset = ecp.num_primitives[0];
    for (size_t k = 1; k < ecp.shell_types.size(); k++) {
      ECPShellHandle sh;
      CUEST_NVTX("cuestECPShellCreate",
                 cuestECPShellCreate(
                     static_cast<cuestHandle_t>(ctx_),
                     ecp.shell_types[k], ecp.num_primitives[k],
                     &ecp.Ns[offset], &ecp.coefficients[offset], &ecp.exponents[offset],
                     ECPShellParams{}, sh.ptr()));
      ecp_shell_handles_.push_back(std::move(sh));
      offset += ecp.num_primitives[k];
    }
  }

  for (size_t a = 0; a < mol_.natom(); a++) {
    int u = atom_to_ecp_idx[a];
    if (u < 0) continue;

    const auto& ecp = ecp_data[u];
    if (ecp.shell_types.empty())
      throw std::runtime_error("ECP atom has zero shells");
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
    CUEST_NVTX("cuestECPAtomCreate",
               cuestECPAtomCreate(
                   static_cast<cuestHandle_t>(ctx_),
                   ecp.n_elec, num_shells, sh_raw.data(),
                   ecp_top_shell_handles_[top_head].get(),
                   ECPAtomParams{}, atom.ptr()));
    ecp_atom_handles_.push_back(std::move(atom));
  }

  for (auto& a : ecp_atom_handles_)
    ecp_atoms_raw_.push_back(a.get());
}

void ECPBuilder::apply_to_molecule(Molecule& mol) const {
  if (ecp_electrons_per_atom_.size() != mol.natom())
    throw std::runtime_error("ECP atom count mismatch when applying to molecule");
  for (size_t a = 0; a < mol.natom(); a++)
    mol.set_atom_ecp_electrons(a, ecp_electrons_per_atom_[a]);
}

OwnedECPIntPlan ECPBuilder::create_ecp_int_plan(cuestAOBasis_t basis,
                                                 const double* xyz_host) const {
  OwnedECPIntPlan owned;
  cuestWorkspaceDescriptor_t pers_desc{}, temp_desc{};
  ECPIntPlanParams ecp_params;
  CUEST_CHECK(cuestECPIntPlanCreateWorkspaceQuery(
      static_cast<cuestHandle_t>(ctx_), basis, xyz_host,
      num_active_ecp_, ecp_indices_.data(), ecp_atoms_raw_.data(),
      ecp_params, &pers_desc, &temp_desc, owned.plan.ptr()));

  owned.persist = Workspace(pers_desc);
  Workspace temp_ws(temp_desc);
  CUEST_NVTX("cuestECPIntPlanCreate",
             cuestECPIntPlanCreate(
                 static_cast<cuestHandle_t>(ctx_), basis, xyz_host,
                 num_active_ecp_, ecp_indices_.data(), ecp_atoms_raw_.data(),
                 ecp_params, owned.persist.ptr(), temp_ws.ptr(), owned.plan.ptr()));
  return owned;
}

}  // namespace cuest
