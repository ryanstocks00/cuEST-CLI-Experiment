#pragma once
/**
 * @file grid.hpp
 * @brief DFT integration grid builder (Becke grid + Lebedev angular quadrature).
 */

#include <cuest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "context.hpp"
#include "molecule.hpp"
#include "raii.hpp"

namespace cuest {

// ---------------------------------------------------------------------------
// Ahlrichs radial quadrature builder
// ---------------------------------------------------------------------------
inline void build_ahlrichs_radial_quadrature(size_t n_points, double radius,
                                              std::vector<double>& nodes,
                                              std::vector<double>& weights) {
  nodes.resize(n_points);
  weights.resize(n_points);
  const double alpha = 0.6;

  for (size_t i = 1; i <= n_points; i++) {
    double z = i * M_PI / (n_points + 1.0);
    double x = std::cos(z);
    double y = std::sin(z);
    double u = std::log((1.0 - x) / 2.0);
    double v = std::pow(1.0 + x, alpha) / std::log(2.0);
    double r = -radius * v * u;
    double w = M_PI / (n_points + 1.0) * y * radius * v *
               (-alpha * u / (1.0 + x) + 1.0 / (1.0 - x)) * r * r;
    nodes[n_points - i] = r;
    weights[n_points - i] = w;
  }
}

// Ahlrichs radii for elements up to Kr (Z=36)
inline double ahlrichs_radius(int atomic_number) {
  static const double radii[] = {
      1.00,  // X
      0.80,  // H
      0.90,  // He
      1.80, 1.40, 1.30, 1.10, 0.90, 0.90, 0.90, 0.90,  // Li-Ne
      1.40, 1.30, 1.30, 1.20, 1.10, 1.00, 1.00, 1.00,  // Na-Ar
      1.50, 1.40, 1.30, 1.20, 1.20, 1.20, 1.20, 1.20, 1.20, 1.10, 1.10,
      1.10, 1.10, 1.00, 0.90, 0.90, 0.90, 0.90  // K-Kr
  };
  if (atomic_number >= 0 && atomic_number <= 36) return radii[atomic_number];
  return 1.0;  // default for heavy atoms
}

// ---------------------------------------------------------------------------
// Grid builder
// ---------------------------------------------------------------------------
class GridBuilder {
 public:
  GridBuilder(CuESTContext& ctx, const Molecule& mol,
              int radial_points = 75, int angular_points = 302)
      : ctx_(ctx), mol_(mol),
        radial_pts_(radial_points), angular_pts_(angular_points) {}
  ~GridBuilder() { if (grid_persist_dev_) cudaFree(grid_persist_dev_); }

  MolecularGridHandle build();
  void* persist_dev() const { return grid_persist_dev_; }

 private:
  CuESTContext& ctx_;
  const Molecule& mol_;
  int radial_pts_;
  int angular_pts_;
  void* grid_persist_dev_{nullptr};  // kept alive for XC plan
  cuestWorkspace_t grid_persist_ws_{};
};

inline MolecularGridHandle GridBuilder::build() {
  // Create per-atom grids
  uint64_t natom = mol_.natom();
  std::vector<cuestAtomGrid_t> atom_grids(natom);
  std::vector<AtomGridHandle> atom_grid_handles;

  std::vector<double> radial_nodes(radial_pts_);
  std::vector<double> radial_weights(radial_pts_);
  std::vector<uint64_t> angular_pts_per_radial(radial_pts_, angular_pts_);

  for (uint64_t i = 0; i < natom; i++) {
    double radius = ahlrichs_radius(mol_.atom(i).atomic_number);
    build_ahlrichs_radial_quadrature(radial_pts_, radius,
                                      radial_nodes, radial_weights);

    CUEST_CHECK(cuestAtomGridCreate(
        ctx_, radial_pts_,
        radial_nodes.data(), radial_weights.data(),
        angular_pts_per_radial.data(),
        AtomGridParams{}, &atom_grids[i]));
  }

  // Create molecular grid
  MolecularGridHandle mol_grid;
  cuestWorkspaceDescriptor_t pers_desc{}, temp_desc{};

  auto xyz_h = mol_.xyz_host();

  CUEST_CHECK(cuestMolecularGridCreateWorkspaceQuery(
      ctx_, natom, atom_grids.data(), xyz_h.data(),
      MolecularGridParams{}, &pers_desc, &temp_desc, mol_grid.ptr()));

  // KEEP persist workspace alive — XC plan depends on it!
  void *pdev=nullptr, *tdev=nullptr;
  if (pers_desc.deviceBufferSizeInBytes) cudaMalloc(&pdev, pers_desc.deviceBufferSizeInBytes);
  if (temp_desc.deviceBufferSizeInBytes) cudaMalloc(&tdev, temp_desc.deviceBufferSizeInBytes);
  cuestWorkspace_t pws = {0,pers_desc.hostBufferSizeInBytes,(uintptr_t)pdev,pers_desc.deviceBufferSizeInBytes};
  cuestWorkspace_t tws = {0,temp_desc.hostBufferSizeInBytes,(uintptr_t)tdev,temp_desc.deviceBufferSizeInBytes};

  CUEST_CHECK(cuestMolecularGridCreate(
      ctx_, natom, atom_grids.data(), xyz_h.data(),
      MolecularGridParams{}, &pws, &tws, mol_grid.ptr()));

  // Free temp, KEEP persist
  if (tdev) cudaFree(tdev);
  grid_persist_dev_ = pdev;
  grid_persist_ws_ = pws;

  // Destroy atom grids (molecular grid copies what it needs)
  for (uint64_t i = 0; i < natom; i++)
    cuestAtomGridDestroy(atom_grids[i]);

  return mol_grid;
}

}  // namespace cuest
