#pragma once
/**
 * @file basis.hpp
 * @brief Basis set builder - creates cuEST AO shells, AO basis, and pair lists
 *        from BSE JSON data.
 */

#include <cuda_runtime.h>
#include <cuest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "context.hpp"
#include "molecule.hpp"
#include "parsers.hpp"
#include "raii.hpp"
#include "shell_norm.hpp"

namespace cuest {

// Pair list + persistent workspace that outlives the handle.
// Member order: persist destroyed last, handle destroyed first.
struct OwnedAOPairList {
  Workspace persist;
  AOPairListHandle handle;

  OwnedAOPairList() = default;
  OwnedAOPairList(OwnedAOPairList&& o) noexcept
      : persist(std::move(o.persist)), handle(std::move(o.handle)) {}
  OwnedAOPairList& operator=(OwnedAOPairList&& o) noexcept {
    if (this != &o) {
      handle.reset();  // destroy handle before freeing persist
      persist = std::move(o.persist);
      handle = std::move(o.handle);
    }
    return *this;
  }
  OwnedAOPairList(const OwnedAOPairList&) = delete;
  OwnedAOPairList& operator=(const OwnedAOPairList&) = delete;

  [[nodiscard]] cuestAOPairList_t get() const { return handle.get(); }
  operator cuestAOPairList_t() const { return handle.get(); }
  [[nodiscard]] bool valid() const { return handle.valid(); }
};

// ECP plan + persistent workspace.
struct OwnedECPIntPlan {
  Workspace persist;
  ECPIntPlanHandle plan;

  OwnedECPIntPlan() = default;
  OwnedECPIntPlan(OwnedECPIntPlan&& o) noexcept
      : persist(std::move(o.persist)), plan(std::move(o.plan)) {}
  OwnedECPIntPlan& operator=(OwnedECPIntPlan&& o) noexcept {
    if (this != &o) {
      plan.reset();
      persist = std::move(o.persist);
      plan = std::move(o.plan);
    }
    return *this;
  }
  OwnedECPIntPlan(const OwnedECPIntPlan&) = delete;
  OwnedECPIntPlan& operator=(const OwnedECPIntPlan&) = delete;

  [[nodiscard]] cuestECPIntPlan_t get() const { return plan.get(); }
  operator cuestECPIntPlan_t() const { return plan.get(); }
  [[nodiscard]] bool valid() const { return plan.valid(); }
};

// ---------------------------------------------------------------------------
// Basis set builder
// ---------------------------------------------------------------------------
class BasisBuilder {
 public:
  BasisBuilder(CuESTContext& ctx, const Molecule& mol, int is_pure = 1)
      : ctx_(ctx), mol_(mol), is_pure_(is_pure) {}

  void build_from_json(const std::string& json_path);

  [[nodiscard]] uint64_t nao() const { return nao_; }
  [[nodiscard]] uint64_t nshells() const { return shell_handles_.size(); }

  [[nodiscard]] cuestAOBasis_t basis() const { return basis_.get(); }
  [[nodiscard]] const AOBasisHandle& basis_handle() const { return basis_; }

  // Cached pair list (persist owned for lifetime of BasisBuilder).
  [[nodiscard]] const OwnedAOPairList& pair_list(double threshold = 1e-14) const;

 private:
  CuESTContext& ctx_;
  const Molecule& mol_;
  int is_pure_{1};

  // Declaration order: persist outlives basis; shells outlive basis destroy.
  Workspace persistent_ws_;
  std::vector<AOShellHandle> shell_handles_;
  AOBasisHandle basis_;
  uint64_t nao_{0};

  mutable OwnedAOPairList pair_list_;
  mutable bool pair_list_ready_{false};
  mutable double pair_list_threshold_{1e-14};
};

// ---------------------------------------------------------------------------
// Auxiliary (density fitting) basis
// ---------------------------------------------------------------------------
class AuxBasis {
 public:
  AuxBasis(CuESTContext& ctx, const Molecule& mol, int is_pure = 1)
      : ctx_(ctx), mol_(mol), is_pure_(is_pure) {}

  void build_from_json(const std::string& json_path);
  [[nodiscard]] cuestAOBasis_t basis() const { return basis_.get(); }
  [[nodiscard]] const AOBasisHandle& handle() const { return basis_; }

 private:
  CuESTContext& ctx_;
  const Molecule& mol_;
  int is_pure_{1};
  // persist outlives basis (destroyed last)
  Workspace persist_ws_;
  std::vector<AOShellHandle> shell_handles_;
  AOBasisHandle basis_;
};

// ---------------------------------------------------------------------------
// ECP builder
// ---------------------------------------------------------------------------
class ECPBuilder {
 public:
  ECPBuilder(CuESTContext& ctx, const Molecule& mol)
      : ctx_(ctx), mol_(mol) {}

  void build_from_json(const std::string& json_path);

  // Apply per-atom ECP core counts onto a Molecule (sets Z_eff).
  void apply_to_molecule(Molecule& mol) const;

  [[nodiscard]] bool has_ecp() const { return has_ecp_; }
  [[nodiscard]] uint64_t num_active_atoms() const { return num_active_ecp_; }
  [[nodiscard]] uint64_t total_ecp_electrons() const { return total_ecp_electrons_; }
  [[nodiscard]] const std::vector<uint64_t>& ecp_indices() const { return ecp_indices_; }
  [[nodiscard]] const std::vector<cuestECPAtom_t>& ecp_atoms() const { return ecp_atoms_raw_; }
  [[nodiscard]] const std::vector<int>& ecp_electrons_per_atom() const {
    return ecp_electrons_per_atom_;
  }

  [[nodiscard]] OwnedECPIntPlan create_ecp_int_plan(cuestAOBasis_t basis,
                                                     const double* xyz_host) const;

 private:
  CuESTContext& ctx_;
  const Molecule& mol_;
  bool has_ecp_{false};
  uint64_t num_active_ecp_{0};
  uint64_t total_ecp_electrons_{0};

  std::vector<uint64_t> ecp_indices_;
  std::vector<int> ecp_electrons_per_atom_;
  std::vector<cuestECPAtom_t> ecp_atoms_raw_;
  std::vector<ECPAtomHandle> ecp_atom_handles_;
  std::vector<ECPShellHandle> ecp_shell_handles_;
  std::vector<ECPShellHandle> ecp_top_shell_handles_;
};

}  // namespace cuest
