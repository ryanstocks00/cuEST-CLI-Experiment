#pragma once
/**
 * @file basis.hpp
 * @brief Basis set builder - creates cuEST AO shells, AO basis, and pair lists
 *        from parsed GBS data.
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

// ---------------------------------------------------------------------------
// Basis set builder
// ---------------------------------------------------------------------------
class BasisBuilder {
 public:
  BasisBuilder(CuESTContext& ctx, const Molecule& mol, int is_pure = 1)
      : ctx_(ctx), mol_(mol), is_pure_(is_pure) {}

  // Parse a GBS file and build the AO basis
  void build_from_gbs(const std::string& gbs_path);
  void build_nvidia(const std::string& gbs_path);

  // Accessors
  uint64_t nao() const { return nao_; }
  uint64_t nshells() const { return shells_.size(); }

  cuestAOBasis_t basis() const { return basis_.get(); }
  const AOBasisHandle& basis_handle() const { return basis_; }

  // Create and return an AO pair list
  AOPairListHandle create_pair_list(double threshold = 1e-14) const;

 private:
  CuESTContext& ctx_;
  const Molecule& mol_;
  int is_pure_{1};

  cuestWorkspaceDescriptor_t persistent_desc_{};
  Workspace persistent_ws_;
  AOBasisHandle basis_;          // destroyed BEFORE shells (must outlive shells)
  uint64_t nao_{0};
  std::vector<AOShellHandle> shell_handles_;  // shells destroyed after basis
  std::vector<cuestAOShell_t> shells_;        // raw handles (same as shell_handles_)
};

// ---------------------------------------------------------------------------
// Auxiliary (density fitting) basis
// ---------------------------------------------------------------------------
class AuxBasis {
 public:
  AuxBasis(CuESTContext& ctx, const Molecule& mol, int is_pure = 1)
      : ctx_(ctx), mol_(mol), is_pure_(is_pure) {}

  void build_from_gbs(const std::string& gbs_path);
  ~AuxBasis();
  cuestAOBasis_t basis() const { return basis_.get(); }
  const AOBasisHandle& handle() const { return basis_; }

 private:
  CuESTContext& ctx_;
  const Molecule& mol_;
  int is_pure_{1};
  AOBasisHandle basis_;
  std::vector<AOShellHandle> shell_handles_;  // shells outlive basis
  void* persist_dev_{nullptr};  // KEPT ALIVE for DF plan
  cuestWorkspace_t persist_ws_{};  // stored for DF plan dependency
};

// ---------------------------------------------------------------------------
// ECP builder
// ---------------------------------------------------------------------------
class ECPBuilder {
 public:
  ECPBuilder(CuESTContext& ctx, const Molecule& mol)
      : ctx_(ctx), mol_(mol) {}

  void build_from_file(const std::string& ecp_path);

  bool has_ecp() const { return has_ecp_; }
  uint64_t num_active_atoms() const { return num_active_ecp_; }
  const std::vector<uint64_t>& ecp_indices() const { return ecp_indices_; }
  const std::vector<cuestECPAtom_t>& ecp_atoms() const { return ecp_atoms_raw_; }

  ECPIntPlanHandle create_ecp_int_plan(cuestAOBasis_t basis,
                                        const double* xyz_host) const;

 private:
  CuESTContext& ctx_;
  const Molecule& mol_;
  bool has_ecp_{false};
  uint64_t num_active_ecp_{0};

  std::vector<uint64_t> ecp_indices_;
  std::vector<cuestECPAtom_t> ecp_atoms_raw_;
  std::vector<ECPAtomHandle> ecp_atom_handles_;
  std::vector<ECPShellHandle> ecp_shell_handles_;
  std::vector<ECPShellHandle> ecp_top_shell_handles_;
};

}  // namespace cuest
