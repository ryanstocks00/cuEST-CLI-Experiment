/**
 * @file basis_nvidia.cpp
 * @brief BasisBuilder and AuxBasis — build cuEST AO shells from BSE JSON.
 */
#include "cuest_wrapper/basis.hpp"
#include "cuest_wrapper/basis_json.hpp"
#include <cuda_runtime.h>
#include <cuest.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cuest {

void BasisBuilder::build_from_json(const std::string& json_path) {
  BSEJsonReader reader(json_path);

  uint64_t natom = mol_.natom();
  std::vector<uint64_t> shells_per_atom(natom);
  std::vector<cuestAOShell_t> all_shells;

  for (size_t a = 0; a < natom; a++) {
    int Z = mol_.atom(a).atomic_number;
    auto json_shells = reader.get_shells(Z);
    shells_per_atom[a] = 0;

    for (auto& js : json_shells) {
      for (auto& coeffs : js.all_coefficients) {
        auto norm = compute_normalized_coefficients(js.L, js.nprim,
                                                     js.exponents.data(),
                                                     coeffs.data());
        cuestAOShell_t sh;
        CUEST_CHECK(cuestAOShellCreate(
            static_cast<cuestHandle_t>(ctx_), is_pure_, js.L, js.nprim,
            js.exponents.data(), norm.data(),
            AOShellParams{}, &sh));
        all_shells.push_back(sh);
        shells_per_atom[a]++;
        AOShellHandle h;
        *h.ptr() = sh;
        shell_handles_.push_back(std::move(h));
      }
    }
  }

  cuestWorkspaceDescriptor_t pers_desc{}, temp_desc{};
  AOBasisParams ao_params;
  CUEST_CHECK(cuestAOBasisCreateWorkspaceQuery(
      static_cast<cuestHandle_t>(ctx_), natom, shells_per_atom.data(),
      all_shells.data(), ao_params,
      &pers_desc, &temp_desc, basis_.ptr()));

  persistent_ws_ = Workspace(pers_desc);
  Workspace temp_ws(temp_desc);
  CUEST_CHECK(cuestAOBasisCreate(
      static_cast<cuestHandle_t>(ctx_), natom, shells_per_atom.data(),
      all_shells.data(), ao_params,
      persistent_ws_.ptr(), temp_ws.ptr(), basis_.ptr()));

  nao_ = ctx_.query_nao(basis_.get());
  pair_list_ready_ = false;
}

void AuxBasis::build_from_json(const std::string& json_path) {
  BSEJsonReader reader(json_path);

  uint64_t natom = mol_.natom();
  std::vector<uint64_t> shells_per_atom(natom);
  std::vector<cuestAOShell_t> all_shells;

  for (size_t a = 0; a < natom; a++) {
    int Z = mol_.atom(a).atomic_number;
    auto json_shells = reader.get_shells(Z);
    shells_per_atom[a] = 0;

    for (auto& js : json_shells) {
      for (auto& coeffs : js.all_coefficients) {
        auto norm = compute_normalized_coefficients(js.L, js.nprim,
                                                     js.exponents.data(),
                                                     coeffs.data());
        cuestAOShell_t sh;
        CUEST_CHECK(cuestAOShellCreate(
            static_cast<cuestHandle_t>(ctx_), is_pure_, js.L, js.nprim,
            js.exponents.data(), norm.data(),
            AOShellParams{}, &sh));
        all_shells.push_back(sh);
        shells_per_atom[a]++;
        AOShellHandle h;
        *h.ptr() = sh;
        shell_handles_.push_back(std::move(h));
      }
    }
  }

  cuestWorkspaceDescriptor_t pers_desc{}, temp_desc{};
  AOBasisParams aux_params;
  CUEST_CHECK(cuestAOBasisCreateWorkspaceQuery(
      static_cast<cuestHandle_t>(ctx_), natom, shells_per_atom.data(),
      all_shells.data(), aux_params,
      &pers_desc, &temp_desc, basis_.ptr()));

  persist_ws_ = Workspace(pers_desc);
  Workspace temp_ws(temp_desc);
  CUEST_CHECK(cuestAOBasisCreate(
      static_cast<cuestHandle_t>(ctx_), natom, shells_per_atom.data(),
      all_shells.data(), aux_params,
      persist_ws_.ptr(), temp_ws.ptr(), basis_.ptr()));
}

const OwnedAOPairList& BasisBuilder::pair_list(double threshold) const {
  if (pair_list_ready_ && threshold == pair_list_threshold_)
    return pair_list_;

  OwnedAOPairList pl;
  cuestWorkspaceDescriptor_t pers_desc{}, temp_desc{};
  uint64_t natom = mol_.natom();
  auto xyz_h = mol_.xyz_host();
  AOPairListParams pl_params;
  CUEST_CHECK(cuestAOPairListCreateWorkspaceQuery(
      static_cast<cuestHandle_t>(ctx_), basis_.get(), natom, xyz_h.data(),
      threshold, pl_params, &pers_desc, &temp_desc, pl.handle.ptr()));

  pl.persist = Workspace(pers_desc);
  Workspace temp_ws(temp_desc);
  CUEST_CHECK(cuestAOPairListCreate(
      static_cast<cuestHandle_t>(ctx_), basis_.get(), natom, xyz_h.data(),
      threshold, pl_params, pl.persist.ptr(), temp_ws.ptr(), pl.handle.ptr()));

  pair_list_ = std::move(pl);
  pair_list_ready_ = true;
  pair_list_threshold_ = threshold;
  return pair_list_;
}

}  // namespace cuest
