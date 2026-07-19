/**
 * @file basis.cpp
 * @brief Basis set builder implementations.
 */

#include "cuest_wrapper/basis.hpp"

#include <cuda_runtime.h>
#include <cuest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cuest {

// ---------------------------------------------------------------------------
// BasisBuilder
// ---------------------------------------------------------------------------
void BasisBuilder::build_from_gbs(const std::string& gbs_path) {
  // Get unique elements and parse basis for each
  auto unique = mol_.unique_symbols();
  std::vector<AtomBasisSet> element_bases;
  for (const auto& sym : unique) {
    element_bases.push_back(parse_gbs_for_element(gbs_path, sym));
  }

  // Count total shells
  uint64_t total_shells = 0;
  std::vector<uint64_t> shells_per_atom(mol_.natom());
  for (size_t i = 0; i < mol_.natom(); i++) {
    int idx = -1;
    for (size_t j = 0; j < unique.size(); j++) {
      if (mol_.atom(i).symbol == unique[j]) { idx = static_cast<int>(j); break; }
    }
    uint64_t ns = element_bases[idx].n_shells;
    shells_per_atom[i] = ns;
    total_shells += ns;
  }

  // Create AO shells
  shells_.resize(total_shells);

  uint64_t counter = 0;
  for (size_t a = 0; a < mol_.natom(); a++) {
    int bidx = -1;
    for (size_t j = 0; j < unique.size(); j++) {
      if (mol_.atom(a).symbol == unique[j]) { bidx = static_cast<int>(j); break; }
    }
    const auto& basis = element_bases[bidx];

    for (size_t s = 0; s < basis.n_shells; s++, counter++) {
      uint64_t L = basis.shell_types[s];
      uint64_t nprim = basis.num_primitives[s];

      // Get exponents and normalize coefficients
      const double* exp_ptr = &basis.exponents[basis.primitive_offsets[s]];
      const double* coeff_ptr = &basis.coefficients[basis.primitive_offsets[s]];

      auto norm_coeffs = compute_normalized_coefficients(L, nprim, exp_ptr, coeff_ptr);

      CUEST_CHECK(cuestAOShellCreate(
          static_cast<cuestHandle_t>(ctx_), is_pure_, L, nprim,
          exp_ptr, norm_coeffs.data(),
          cuest::AOShellParams{}, &shells_[counter]));
    }
  }

  // Create AO basis
  cuestWorkspaceDescriptor_t temp_desc{};
  {
    AOBasisParams params;
    CUEST_CHECK(cuestAOBasisCreateWorkspaceQuery(
        static_cast<cuestHandle_t>(ctx_), mol_.natom(), shells_per_atom.data(),
        shells_.data(), params,
        &persistent_desc_, &temp_desc, basis_.ptr()));
  }

  persistent_ws_ = Workspace(persistent_desc_);
  Workspace temp_ws(temp_desc);
  {
    AOBasisParams params;
    CUEST_CHECK(cuestAOBasisCreate(
        static_cast<cuestHandle_t>(ctx_), mol_.natom(), shells_per_atom.data(),
        shells_.data(), params,
        persistent_ws_.ptr(), temp_ws.ptr(), basis_.ptr()));
  }

  // Query number of AOs
  nao_ = ctx_.query_nao(basis_.get());

  // Store shells as owned handles for RAII cleanup
  for (auto& sh : shells_) {
    AOShellHandle h;
    *h.ptr() = sh;
    shell_handles_.push_back(std::move(h));
  }
}

AOPairListHandle BasisBuilder::create_pair_list(double threshold) const {
  AOPairListHandle pl;
  cuestWorkspaceDescriptor_t pers_desc{}, temp_desc{};

  uint64_t natom = mol_.natom();
  auto xyz = mol_.xyz_host();

  CUEST_CHECK(cuestAOPairListCreateWorkspaceQuery(
      ctx_, basis_.get(), natom, xyz.data(),
      threshold, AOPairListParams{}, &pers_desc, &temp_desc, pl.ptr()));

  Workspace pers_ws(pers_desc);
  Workspace temp_ws(temp_desc);

  CUEST_CHECK(cuestAOPairListCreate(
      ctx_, basis_.get(), natom, xyz.data(),
      threshold, AOPairListParams{},
      pers_ws.ptr(), temp_ws.ptr(), pl.ptr()));

  return pl;
}

// ---------------------------------------------------------------------------
// AuxBasis
// ---------------------------------------------------------------------------
void AuxBasis::build_from_gbs(const std::string& gbs_path) {
  auto unique = mol_.unique_symbols();
  std::vector<AtomBasisSet> element_bases;
  for (const auto& sym : unique) {
    element_bases.push_back(parse_gbs_for_element(gbs_path, sym));
  }

  uint64_t total_shells = 0;
  std::vector<uint64_t> shells_per_atom(mol_.natom());
  for (size_t i = 0; i < mol_.natom(); i++) {
    int idx = -1;
    for (size_t j = 0; j < unique.size(); j++) {
      if (mol_.atom(i).symbol == unique[j]) { idx = static_cast<int>(j); break; }
    }
    uint64_t ns = element_bases[idx].n_shells;
    shells_per_atom[i] = ns;
    total_shells += ns;
  }

  std::vector<cuestAOShell_t> shells(total_shells);

  uint64_t counter = 0;
  for (size_t a = 0; a < mol_.natom(); a++) {
    int bidx = -1;
    for (size_t j = 0; j < unique.size(); j++) {
      if (mol_.atom(a).symbol == unique[j]) { bidx = static_cast<int>(j); break; }
    }
    const auto& basis = element_bases[bidx];

    for (size_t s = 0; s < basis.n_shells; s++, counter++) {
      uint64_t L = basis.shell_types[s];
      uint64_t nprim = basis.num_primitives[s];
      const double* exp_ptr = &basis.exponents[basis.primitive_offsets[s]];
      const double* coeff_ptr = &basis.coefficients[basis.primitive_offsets[s]];
      auto norm = compute_normalized_coefficients(L, nprim, exp_ptr, coeff_ptr);

      CUEST_CHECK(cuestAOShellCreate(
          static_cast<cuestHandle_t>(ctx_), is_pure_, L, nprim, exp_ptr, norm.data(),
          AOShellParams{}, &shells[counter]));
    }
  }

  cuestWorkspaceDescriptor_t pers_desc{}, temp_desc{};

  CUEST_CHECK(cuestAOBasisCreateWorkspaceQuery(
      ctx_, mol_.natom(), shells_per_atom.data(),
      shells.data(), AOBasisParams{},
      &pers_desc, &temp_desc, basis_.ptr()));

  // Manual workspace for aux basis
  void* a_pdev = nullptr, *a_tdev = nullptr;
  if (pers_desc.deviceBufferSizeInBytes) cudaMalloc(&a_pdev, pers_desc.deviceBufferSizeInBytes);
  if (temp_desc.deviceBufferSizeInBytes) cudaMalloc(&a_tdev, temp_desc.deviceBufferSizeInBytes);
  cuestWorkspace_t a_pws = {0, pers_desc.hostBufferSizeInBytes, (uintptr_t)a_pdev, pers_desc.deviceBufferSizeInBytes};
  cuestWorkspace_t a_tws = {0, temp_desc.hostBufferSizeInBytes, (uintptr_t)a_tdev, temp_desc.deviceBufferSizeInBytes};

  CUEST_CHECK(cuestAOBasisCreate(
      ctx_, mol_.natom(), shells_per_atom.data(),
      shells.data(), AOBasisParams{},
      &a_pws, &a_tws, basis_.ptr()));

  if (a_pdev) cudaFree(a_pdev);
  if (a_tdev) cudaFree(a_tdev);

  // Transfer shell ownership to member variable (outlives basis)
  for (auto& sh : shells) {
    AOShellHandle h;
    *h.ptr() = sh;
    shell_handles_.push_back(std::move(h));
  }
}

// ---------------------------------------------------------------------------
// ECPBuilder
// ---------------------------------------------------------------------------
void ECPBuilder::build_from_file(const std::string& ecp_path) {
  auto unique = mol_.unique_symbols();
  std::vector<ECPShellSet> element_ecps;

  for (const auto& sym : unique) {
    element_ecps.push_back(parse_ecp_for_element(ecp_path, sym));
  }

  // Count active ECP atoms
  for (size_t i = 0; i < mol_.natom(); i++) {
    int bidx = -1;
    for (size_t j = 0; j < unique.size(); j++) {
      if (mol_.atom(i).symbol == unique[j]) { bidx = static_cast<int>(j); break; }
    }
    if (element_ecps[bidx].n_shells > 0) {
      ecp_indices_.push_back(i);
    }
  }

  num_active_ecp_ = ecp_indices_.size();
  has_ecp_ = num_active_ecp_ > 0;
  if (!has_ecp_) return;

  // Build ECP shells and atoms
  for (size_t u = 0; u < unique.size(); u++) {
    const auto& ecp = element_ecps[u];
    if (ecp.n_shells == 0) continue;

    size_t num_shells = ecp.n_shells - 1;  // excluding top shell

    // Create top shell (first shell)
    ECPShellHandle top;
    CUEST_CHECK(cuestECPShellCreate(
        ctx_,
        ecp.shell_types[0],
        ecp.num_primitives[0],
        ecp.Ns.data(),
        ecp.coefficients.data(),
        ecp.exponents.data(),
        ECPShellParams{},
        top.ptr()));
    ecp_top_shell_handles_.push_back(std::move(top));

    // Create remaining shells
    for (size_t k = 0; k < num_shells; k++) {
      uint64_t offset = ecp.primitive_offsets[k + 1];
      ECPShellHandle sh;
      CUEST_CHECK(cuestECPShellCreate(
          ctx_,
          ecp.shell_types[k + 1],
          ecp.num_primitives[k + 1],
          &ecp.Ns[offset],
          &ecp.coefficients[offset],
          &ecp.exponents[offset],
          ECPShellParams{},
          sh.ptr()));
      ecp_shell_handles_.push_back(std::move(sh));
    }
  }

  // Create ECP atoms
  // Need to index into the stored shells by unique element
  uint64_t shell_head = 0, top_head = 0;
  for (size_t u = 0; u < unique.size(); u++) {
    const auto& ecp = element_ecps[u];
    if (ecp.n_shells == 0) continue;

    size_t num_shells = ecp.n_shells - 1;

    // Get pointers to the raw shell handles for this element
    std::vector<cuestECPShell_t> sh_raw(num_shells);
    for (size_t k = 0; k < num_shells; k++) {
      sh_raw[k] = ecp_shell_handles_[shell_head + k].get();
    }

    ECPAtomHandle atom;
    CUEST_CHECK(cuestECPAtomCreate(
        ctx_,
        ecp.n_elec,
        num_shells,
        sh_raw.data(),
        ecp_top_shell_handles_[top_head].get(),
        ECPAtomParams{},
        atom.ptr()));
    ecp_atom_handles_.push_back(std::move(atom));

    shell_head += num_shells;
    top_head += 1;
  }

  // Collect raw atom pointers
  for (auto& a : ecp_atom_handles_) {
    ecp_atoms_raw_.push_back(a.get());
  }
}

ECPIntPlanHandle ECPBuilder::create_ecp_int_plan(cuestAOBasis_t basis,
                                                   const double* xyz_host) const {
  ECPIntPlanHandle plan;
  cuestWorkspaceDescriptor_t pers_desc{}, temp_desc{};

  CUEST_CHECK(cuestECPIntPlanCreateWorkspaceQuery(
      ctx_, basis, xyz_host,
      num_active_ecp_, ecp_indices_.data(), ecp_atoms_raw_.data(),
      ECPIntPlanParams{}, &pers_desc, &temp_desc, plan.ptr()));

  Workspace pers_ws(pers_desc);
  Workspace temp_ws(temp_desc);

  CUEST_CHECK(cuestECPIntPlanCreate(
      ctx_, basis, xyz_host,
      num_active_ecp_, ecp_indices_.data(), ecp_atoms_raw_.data(),
      ECPIntPlanParams{}, pers_ws.ptr(), temp_ws.ptr(), plan.ptr()));

  return plan;
}

}  // namespace cuest
