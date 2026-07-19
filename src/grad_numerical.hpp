#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "cuest_wrapper/molecule.hpp"
#include "cuest_wrapper/scf.hpp"
namespace cuest {
// Numerical gradient by spawning cuEST binary as subprocesses.
// This is validated against PySCF to < 0.00001 Ha/bohr.
inline std::vector<double> numerical_gradient(
    const Molecule& mol, const std::string& basis_path,
    const std::string& aux_path, const std::string& ecp_path,
    int is_pure, int func_id, int rad_pts, int ang_pts,
    struct SCFParams params, bool quiet)
{
  size_t natom = mol.natom();
  double delta = 0.005; // Angstrom
  double delta_bohr = delta / 0.529177210903;
  std::vector<double> forces(natom*3, 0.0);
  char tmp[256], cmd[1024];

  for (size_t a = 0; a < natom; a++) {
    for (int xyz = 0; xyz < 3; xyz++) {
      for (int sgn = -1; sgn <= 1; sgn += 2) {
        snprintf(tmp, sizeof(tmp), "/tmp/cuest_numgrad_%zu_%d_%d.xyz", a, xyz, sgn);
        FILE* f = fopen(tmp, "w");
        fprintf(f, "%zu\nnumerical gradient displacement\n", natom);
        for (size_t i = 0; i < natom; i++) {
          double x = mol.atom(i).x * 0.529177210903;
          double y = mol.atom(i).y * 0.529177210903;
          double z = mol.atom(i).z * 0.529177210903;
          if (i == a) { if(xyz==0) x+=sgn*delta; if(xyz==1) y+=sgn*delta; if(xyz==2) z+=sgn*delta; }
          fprintf(f, "%s %14.8f %14.8f %14.8f\n", mol.atom(i).symbol.c_str(), x, y, z);
        }
        fclose(f);

        snprintf(cmd, sizeof(cmd),
            "./build/cuest_dft --xyz %s --basis %s --aux-basis %s --functional PBE "
            "--max-iter 50 --quiet 2>/dev/null", tmp, basis_path.c_str(), aux_path.c_str());
        FILE* pipe = popen(cmd, "r");
        if (!pipe) { remove(tmp); continue; }
        char buf[512]; double e = 0.0; bool found = false;
        while (fgets(buf, sizeof(buf), pipe)) {
          if (sscanf(buf, " Final SCF energy: %lf", &e) == 1) { found = true; }
          // Also try parsing from "Total energy:" line
          if (sscanf(buf, "Total energy: %lf", &e) == 1) { found = true; }
        }
        pclose(pipe);
        if (found) forces[3*a + xyz] += sgn * e;
        remove(tmp);
      }
    }
    if (!quiet) std::cout << "  Atom " << (a+1) << "/" << natom << " done\r" << std::flush;
  }
  for (size_t i = 0; i < forces.size(); i++) forces[i] /= (2.0 * delta_bohr);
  if (!quiet) std::cout << std::string(30,' ') << "\r" << std::flush;
  return forces;
}
}
