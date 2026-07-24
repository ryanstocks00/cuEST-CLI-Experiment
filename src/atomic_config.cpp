/**
 * @file atomic_config.cpp — reference configurations and fractional occupations.
 */
#include "atomic_config.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace cuest {

bool ecp_core_shells(int nelec_core, int out[4]) {
  // Mirrors PySCF gto.ecp.core_configuration. Entries are shell counts per
  // {s, p, d, f}. The f-in-core lanthanide/actinide special cases PySCF adds
  // for specific elements are deliberately not replicated: they only affect
  // f-in-core ECPs, which this code does not currently support, and silently
  // guessing at them would be worse than failing.
  struct Entry {
    int nelec;
    int shells[4];
  };
  static constexpr Entry kTable[] = {
      {0, {0, 0, 0, 0}},   {2, {1, 0, 0, 0}},   {10, {2, 1, 0, 0}},
      {18, {3, 2, 0, 0}},  {28, {3, 2, 1, 0}},  {36, {4, 3, 1, 0}},
      {46, {4, 3, 2, 0}},  {54, {5, 4, 2, 0}},  {60, {4, 3, 2, 1}},
      {68, {5, 4, 2, 1}},  {78, {5, 4, 3, 1}},  {92, {5, 4, 3, 2}},
  };
  for (const auto& e : kTable) {
    if (e.nelec != nelec_core) continue;
    for (int i = 0; i < 4; i++) out[i] = e.shells[i];
    return true;
  }
  return false;
}

std::vector<OrbitalManifold> occupied_manifolds(const std::vector<double>& mo_energies,
                                                int Z, int nelec_core, double tol) {
  if (Z < 0 || Z > kMaxConfigurationZ)
    throw std::runtime_error("occupied_manifolds: no reference configuration for Z=" +
                             std::to_string(Z));

  int config[4] = {kNRSRHFConfiguration[Z][0], kNRSRHFConfiguration[Z][1],
                   kNRSRHFConfiguration[Z][2], kNRSRHFConfiguration[Z][3]};

  if (nelec_core > 0) {
    int core[4] = {0, 0, 0, 0};
    if (!ecp_core_shells(nelec_core, core))
      throw std::runtime_error("occupied_manifolds: unrecognised ECP core size " +
                               std::to_string(nelec_core) +
                               " for Z=" + std::to_string(Z));
    // Remove the electrons living in the ECP-replaced core shells: a full
    // shell of angular momentum l holds 2*(2l+1) electrons.
    for (int l = 0; l < 4; l++) {
      config[l] -= core[l] * 2 * (2 * l + 1);
      if (config[l] < 0)
        throw std::runtime_error(
            "occupied_manifolds: ECP core for Z=" + std::to_string(Z) +
            " removes more l=" + std::to_string(l) +
            " electrons than the reference configuration has");
    }
  }

  auto satisfied = [&] {
    return config[0] == 0 && config[1] == 0 && config[2] == 0 && config[3] == 0;
  };

  std::vector<OrbitalManifold> out;
  const int n = static_cast<int>(mo_energies.size());
  int i = 0;
  while (i < n && !satisfied()) {
    int j = i + 1;
    while (j < n && std::abs(mo_energies[j] - mo_energies[i]) < tol) j++;

    OrbitalManifold m;
    m.lo = i;
    m.size = j - i;
    m.energy = mo_energies[i];
    if (m.size % 2 == 0)
      throw std::runtime_error(
          "occupied_manifolds: manifold of even size " + std::to_string(m.size) +
          " at MO " + std::to_string(i) +
          " — a spherically symmetric Fock only produces (2l+1)-fold "
          "degeneracies, so either the Fock was not projected onto the SO(3) "
          "commutant or two different l accidentally coincided");
    m.l = (m.size - 1) / 2;

    // Hand the manifold to its l channel. Manifolds arrive in energy order, so
    // each channel fills lowest-first without any extra sorting.
    if (m.l < 4 && config[m.l] > 0) {
      const int per_shell = 2 * (2 * m.l + 1);  // electrons in a full shell
      const int take = std::min(config[m.l], per_shell);
      // Electrons per orbital: 2 for a full shell, a fraction otherwise. Every
      // orbital in the manifold gets the *same* occupancy, which is what keeps
      // the resulting density spherically symmetric.
      m.occ = static_cast<double>(take) / static_cast<double>(2 * m.l + 1);
      config[m.l] -= take;
    }

    out.push_back(m);
    i = j;
  }

  if (!satisfied()) {
    for (int l = 0; l < 4; l++) {
      if (config[l] == 0) continue;
      throw std::runtime_error(
          "occupied_manifolds: Z=" + std::to_string(Z) + " needs " +
          std::to_string(config[l]) + " more l=" + std::to_string(l) +
          " electrons than the basis provides shells for — basis too small for "
          "this element's reference configuration");
    }
  }
  return out;
}

}  // namespace cuest
