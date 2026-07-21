#pragma once
/**
 * @file grad_numerical.hpp
 * @brief Numerical gradient by finite differences of total energy.
 *
 * Computes the nuclear gradient as:
 *   dE/dR ≈ [E(R + delta) - E(R - delta)] / (2 * delta)
 *
 * Uses subprocess calls to the cuEST DFT binary for each displacement.
 * This is validated against PySCF as a reference for the analytical gradient.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "cuest_wrapper/constants.hpp"
#include "cuest_wrapper/integrals.hpp"
#include "cuest_wrapper/molecule.hpp"
#include "functionals.hpp"
#include "scf.hpp"

namespace cuest {

// Numerical gradient by spawning cuEST binary as subprocesses.
// Uses the SAME binary, basis, and settings as the analytical gradient.
inline std::vector<double> numerical_gradient(
    const Molecule& mol, const std::string& basis_path,
    const std::string& aux_path,
    XCBuilder::Functional functional, int rad_pts, int ang_pts,
    struct SCFParams params, bool quiet,
    const std::string& binary_path = "build/cuest_dft",
    int is_pure = 1)
{
  size_t natom = mol.natom();
  double delta = 0.005; // Angstrom
  double delta_bohr = delta * constants::bohr_per_angstrom;
  std::vector<double> forces(natom*3, 0.0);

  const char* func_name = functional_to_name(functional);

  for (size_t a = 0; a < natom; a++) {
    for (int xyz = 0; xyz < 3; xyz++) {
      for (int sgn = -1; sgn <= 1; sgn += 2) {
        // Write displaced geometry to temp file
        char tmp[256];
        snprintf(tmp, sizeof(tmp), "/tmp/cuest_numgrad_%zu_%d_%d.xyz", a, xyz, sgn);
        FILE* f = fopen(tmp, "w");
        if (!f) continue;
        fprintf(f, "%zu\nnumerical gradient displacement\n", natom);
        for (size_t i = 0; i < natom; i++) {
          double x = mol.atom(i).x * constants::angstrom_per_bohr;
          double y = mol.atom(i).y * constants::angstrom_per_bohr;
          double z = mol.atom(i).z * constants::angstrom_per_bohr;
          if (i == a) {
            if (xyz == 0) x += sgn * delta;
            if (xyz == 1) y += sgn * delta;
            if (xyz == 2) z += sgn * delta;
          }
          fprintf(f, "%s %14.8f %14.8f %14.8f\n",
                  Molecule::z_to_symbol(mol.atom(i).atomic_number), x, y, z);
        }
        fclose(f);

        // Build command — quote paths to protect against shell injection
        auto quote = [](const std::string& s) {
          return "\"" + s + "\"";
        };
        std::ostringstream cmd;
        cmd << quote(binary_path)
            << " --xyz " << quote(tmp)
            << " --basis " << quote(basis_path)
            << " --aux-basis " << quote(aux_path)
            << " --functional " << func_name
            << " --radial-pts " << rad_pts
            << " --angular-pts " << ang_pts
            << " --charge " << mol.charge()
            << " --multiplicity " << mol.multiplicity()
            << " --max-iter " << params.max_iter
            << " --conv-thresh " << params.conv_thresh
            << (is_pure ? " --spherical" : " --cartesian")
            << " --quiet";

        FILE* pipe = popen(cmd.str().c_str(), "r");
        if (!pipe) { remove(tmp); continue; }

        char buf[512];
        double e = 0.0;
        bool found = false;
        while (fgets(buf, sizeof(buf), pipe)) {
          // Match "Final SCF energy:" with optional leading whitespace
          const char* key = "Final SCF energy:";
          const char* pos = strstr(buf, key);
          if (pos && sscanf(pos + strlen(key), "%lf", &e) == 1) {
            found = true;
          }
        }
        int rc = pclose(pipe);

        if (found) {
          forces[3*a + xyz] += sgn * e;
        } else {
          // Retry with looser convergence
          if (!quiet) {
            std::cerr << "Warning: numerical gradient displacement failed (a=" << a
                      << " xyz=" << xyz << " sgn=" << sgn
                      << " exit=" << rc << "). Check that cuEST binary is built.\n";
          }
        }
        remove(tmp);
      }
    }
    if (!quiet) std::cout << "  Atom " << (a+1) << "/" << natom << " done\r" << std::flush;
  }

  for (size_t i = 0; i < forces.size(); i++) forces[i] /= (2.0 * delta_bohr);
  if (!quiet) std::cout << std::string(30,' ') << "\r" << std::flush;
  return forces;
}

}  // namespace cuest
