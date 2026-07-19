/**
 * @file main.cpp
 * @brief CLI entry point for cuEST-based DFT calculations.
 *
 * Usage: cuest_dft --xyz <file> --basis <gbs_file> [options]
 *
 * Full list of options:
 *   Required:
 *     --xyz <path>            Input geometry in XYZ format
 *     --basis <path>          Primary basis set in GBS format
 *
 *   Optional:
 *     --aux-basis <path>      Auxiliary/DF basis set in GBS format
 *     --ecp <path>            ECP definitions (GBS format with ECP data)
 *     --functional <name>     XC functional: PBE (default), B3LYP, PBE0,
 *                             CAM-B3LYP, wB97X-V, wB97M-V, HSE06, M06,
 *                             M06-2X, LC-wPBE, LC-wPBEh, wB97X
 *     --radial-pts <n>        Radial grid points (default: 75)
 *     --angular-pts <n>       Angular grid points (default: 302)
 *     --charge <int>          Total molecular charge (default: 0)
 *     --multiplicity <int>    Spin multiplicity (default: 1)
 *     --max-iter <n>          Maximum SCF iterations (default: 150)
 *     --conv-thresh <val>     RMS density convergence (default: 1e-8)
 *     --energy-conv <val>     Energy convergence (default: 1e-8)
 *     --diis-start <n>        Iteration to start DIIS (default: 1)
 *     --diis-space <n>        DIIS subspace size (default: 10)
 *     --damping <val>         Density damping factor (default: 0.0)
 *     --level-shift <val>     Level shifting parameter (default: 0.0)
 *     --no-df                 Disable density fitting (use exact integrals)
 *     --no-pure               Use Cartesian (not spherical) functions
 *     --quiet                 Minimal output
 *     --verbose               Maximum output
 *     --print-mos             Print MO energies at end
 *     --help                  Show this help
 */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

#include "cuest_wrapper/basis.hpp"
#include "cuest_wrapper/context.hpp"
#include "cuest_wrapper/gradients.hpp"
#include "grad_numerical.hpp"
#include "cuest_wrapper/grid.hpp"
#include "cuest_wrapper/integrals.hpp"
#include "cuest_wrapper/molecule.hpp"
#include "cuest_wrapper/parsers.hpp"
#include "cuest_wrapper/raii.hpp"
#include "cuest_wrapper/scf.hpp"
#include "cuest_wrapper/shell_norm.hpp"

// NVIDIA proven helpers (must be included in dependency order)
extern "C" {
#include "helper_status.h"
#include "helper_workspaces.h"
#include "helper_gbs_parser.h"
#include "helper_xyz_parser.h"
#include "helper_shell_normalization.h"
#include "helper_ecp_parser.h"
#include "helper_ao_shells.h"
}

using namespace cuest;

// ---------------------------------------------------------------------------
// Functional name → ID mapping
// ---------------------------------------------------------------------------
struct FunctionalInfo {
  const char* name;
  int id;
};

static const FunctionalInfo kFunctionals[] = {
    {"PBE", XCBuilder::XC_PBE},
    {"B3LYP", XCBuilder::XC_B3LYP},
    {"B3LYP5", XCBuilder::XC_B3LYP5},
    {"PBE0", XCBuilder::XC_PBE0},
    {"CAM-B3LYP", XCBuilder::XC_CAM_B3LYP},
    {"WB97X-V", XCBuilder::XC_WB97X_V},
    {"WB97M-V", XCBuilder::XC_WB97M_V},
    {"HSE06", XCBuilder::XC_HSE06},
    {"M06", XCBuilder::XC_M06},
    {"M06-2X", XCBuilder::XC_M062X},
    {"LC-WPBE", XCBuilder::XC_LC_WPBE},
    {"LC-WPBEH", XCBuilder::XC_LC_WPBEH},
    {"WB97X", XCBuilder::XC_WB97X},
};

// ---------------------------------------------------------------------------
// Print help
// ---------------------------------------------------------------------------
static void print_help(const char* prog) {
  std::cout
      << "cuEST DFT — GPU-accelerated density functional theory\n"
      << "Usage: " << prog << " --xyz <file> --basis <gbs_file> [options]\n\n"
      << "Required arguments:\n"
      << "  --xyz <path>             Input geometry in XYZ format\n"
      << "  --basis <path>           Primary basis set in Gaussian94 format\n\n"
      << "Optional arguments:\n"
      << "  --aux-basis <path>       Auxiliary/DF (RI-J) basis set\n"
      << "  --ecp <path>             Effective core potential file\n"
      << "  --functional <name>      XC functional (default: PBE)\n"
      << "  --radial-pts <n>         Radial grid points (default: 75)\n"
      << "  --angular-pts <n>        Angular Lebedev points (default: 302)\n"
      << "  --charge <int>           Total charge (default: 0)\n"
      << "  --multiplicity <int>     Spin multiplicity (default: 1)\n\n"
      << "SCF convergence options:\n"
      << "  --max-iter <n>           Max SCF iterations (default: 150)\n"
      << "  --conv-thresh <val>      RMS density convergence (default: 1e-8)\n"
      << "  --energy-conv <val>      Energy change convergence (default: 1e-8)\n"
      << "  --diis-start <n>         Iteration to enable DIIS (default: 1)\n"
      << "  --diis-space <n>         DIIS subspace dimension (default: 10)\n"
      << "  --damping <val>          Density damping factor (default: 0.0)\n"
      << "  --level-shift <val>      Level shifting (default: 0.0)\n\n"
      << "Other options:\n"
      << "  --no-df                  Disable density fitting\n"
      << "  --no-pure                Use Cartesian (not spherical) functions\n"
      << "  --quiet                  Minimal output\n"
      << "  --verbose                Verbose output\n"
      << "  --print-mos              Print final MO energies\n"
      << "  --gradient               Compute and print nuclear gradient\n"
      << "  --help                   Show this help\n\n"
      << "Available functionals:\n"
      << "  PBE B3LYP B3LYP5 PBE0 CAM-B3LYP WB97X-V WB97M-V\n"
      << "  HSE06 M06 M06-2X LC-WPBE LC-WPBEH WB97X\n\n"
      << "Input files can be obtained from the EMSL Basis Set Exchange:\n"
      << "  https://www.basissetexchange.org/\n"
      << "  (Use 'Gaussian94' or 'Psi4' format; uncontract SPDF if needed)\n"
      << std::endl;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  // --- Parse CLI arguments ---
  std::string xyz_path, basis_path, aux_basis_path, ecp_path;
  std::string functional_name = "PBE";
  int radial_pts = 75;
  int angular_pts = 302;
  int charge = 0;
  int multiplicity = 1;
  int max_iter = 150;
  double conv_thresh = 1e-8;
  double energy_conv = 1e-8;
  int diis_start = 1;
  int diis_space = 10;
  double damping = 0.0;
  double level_shift = 0.0;
  bool use_df = true;
  int is_pure = 1;
  bool verbose = true;
  bool print_mos = false;
  bool gradient = false;
  bool quiet = false;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      print_help(argv[0]);
      return 0;
    } else if (arg == "--xyz" && i + 1 < argc) {
      xyz_path = argv[++i];
    } else if (arg == "--basis" && i + 1 < argc) {
      basis_path = argv[++i];
    } else if (arg == "--aux-basis" && i + 1 < argc) {
      aux_basis_path = argv[++i];
    } else if (arg == "--ecp" && i + 1 < argc) {
      ecp_path = argv[++i];
    } else if (arg == "--functional" && i + 1 < argc) {
      functional_name = argv[++i];
    } else if (arg == "--radial-pts" && i + 1 < argc) {
      radial_pts = std::stoi(argv[++i]);
    } else if (arg == "--angular-pts" && i + 1 < argc) {
      angular_pts = std::stoi(argv[++i]);
    } else if (arg == "--charge" && i + 1 < argc) {
      charge = std::stoi(argv[++i]);
    } else if (arg == "--multiplicity" && i + 1 < argc) {
      multiplicity = std::stoi(argv[++i]);
    } else if (arg == "--max-iter" && i + 1 < argc) {
      max_iter = std::stoi(argv[++i]);
    } else if (arg == "--conv-thresh" && i + 1 < argc) {
      conv_thresh = std::stod(argv[++i]);
    } else if (arg == "--energy-conv" && i + 1 < argc) {
      energy_conv = std::stod(argv[++i]);
    } else if (arg == "--diis-start" && i + 1 < argc) {
      diis_start = std::stoi(argv[++i]);
    } else if (arg == "--diis-space" && i + 1 < argc) {
      diis_space = std::stoi(argv[++i]);
    } else if (arg == "--damping" && i + 1 < argc) {
      damping = std::stod(argv[++i]);
    } else if (arg == "--level-shift" && i + 1 < argc) {
      level_shift = std::stod(argv[++i]);
    } else if (arg == "--no-df") {
      use_df = false;
    } else if (arg == "--no-pure") {
      is_pure = 0;
    } else if (arg == "--quiet") {
      quiet = true;
      verbose = false;
    } else if (arg == "--verbose") {
      verbose = true;
    } else if (arg == "--print-mos") {
      print_mos = true;
    } else if (arg == "--gradient") {
      gradient = true;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      std::cerr << "Use --help for usage information.\n";
      return 1;
    }
  }

  // --- Validate required arguments ---
  if (xyz_path.empty() || basis_path.empty()) {
    std::cerr << "Error: --xyz and --basis are required.\n";
    std::cerr << "Use --help for usage information.\n";
    return 1;
  }

  // --- Look up functional ---
  int functional_id = XCBuilder::XC_PBE;
  bool found_functional = false;
  for (const auto& fi : kFunctionals) {
    if (functional_name == fi.name) {
      functional_id = fi.id;
      found_functional = true;
      break;
    }
  }
  if (!found_functional) {
    std::cerr << "Warning: Unknown functional '" << functional_name
              << "'. Using PBE.\n";
  }

  if (!quiet) {
    std::cout << std::string(60, '=') << "\n";
    std::cout << "  cuEST DFT — GPU-Accelerated Quantum Chemistry\n";
    std::cout << std::string(60, '=') << "\n\n";
  }

  try {
    // --- Parse XYZ and build molecule ---
    if (!quiet) std::cout << "Reading geometry from: " << xyz_path << "\n";
    auto xyz_data = parse_xyz(xyz_path);

    Molecule mol;
    for (size_t i = 0; i < xyz_data.n_atoms; i++) {
      double x_ang = xyz_data.xyz[3 * i] / (1.0 / 0.529177210903);
      double y_ang = xyz_data.xyz[3 * i + 1] / (1.0 / 0.529177210903);
      double z_ang = xyz_data.xyz[3 * i + 2] / (1.0 / 0.529177210903);
      mol.add_atom(xyz_data.symbols[i], x_ang, y_ang, z_ang);
    }
    mol.set_multiplicity(multiplicity);

    // Adjust for charge
    if (charge != 0) {
      std::cerr << "Warning: Non-zero charge not fully implemented yet.\n";
    }

    if (!quiet) {
      std::cout << "  Atoms: " << mol.natom() << "\n";
      std::cout << "  Electrons: " << mol.nelec() << "\n";
      std::cout << "  Multiplicity: " << mol.multiplicity() << "\n\n";
    }

    // --- Create cuEST context ---
    CuESTContext ctx;

    // --- Build primary basis ---
    if (!quiet) std::cout << "Building primary basis from: " << basis_path << "\n";
    BasisBuilder basis_builder(ctx, mol, is_pure);
    basis_builder.build_from_gbs(basis_path);

    uint64_t nao = basis_builder.nao();
    if (!quiet) std::cout << "  Basis functions: " << nao << "\n";

    // --- Build auxiliary basis (for DF) ---
    AuxBasis* aux_basis_ptr = nullptr;
    std::unique_ptr<AuxBasis> aux_basis;

    if (use_df) {
      std::string aux_path = aux_basis_path;
      if (aux_path.empty()) {
        // Try to infer: append "-jkfit" or use def2-universal-jkfit
        std::cerr << "Warning: No auxiliary basis specified. "
                  << "Density fitting requires --aux-basis.\n";
        std::cerr << "  Consider using: def2-universal-jkfit.gbs\n";
        use_df = false;
      } else {
        if (!quiet) std::cout << "Building auxiliary basis from: " << aux_path << "\n";
        aux_basis = std::make_unique<AuxBasis>(ctx, mol, is_pure);
        aux_basis->build_from_gbs(aux_path);
        uint64_t naux = ctx.query_nao(aux_basis->basis());
        if (!quiet) std::cout << "  Auxiliary basis functions: " << naux << "\n";
        aux_basis_ptr = aux_basis.get();
      }
    }

    // --- Build ECP if specified ---
    ECPBuilder* ecp_builder_ptr = nullptr;
    ECPIntegrals* ecp_int_ptr = nullptr;
    std::unique_ptr<ECPBuilder> ecp_builder;
    std::unique_ptr<ECPIntegrals> ecp_int;

    if (!ecp_path.empty()) {
      if (!quiet) std::cout << "Building ECP from: " << ecp_path << "\n";
      ecp_builder = std::make_unique<ECPBuilder>(ctx, mol);
      ecp_builder->build_from_file(ecp_path);

      if (ecp_builder->has_ecp()) {
        auto xyz_h = mol.xyz_host();
        auto ecp_plan = ecp_builder->create_ecp_int_plan(basis_builder.basis(),
                                                           xyz_h.data());
        // We need to hold the plan somewhere; for now, rebuild in SCFSolver
        // (TODO: refactor to pass plan through)
        ecp_builder_ptr = ecp_builder.get();
      } else {
        if (!quiet) std::cout << "  No ECP atoms found in molecule.\n";
      }
    }

    // --- Build DFT grid ---
    // (The XCBuilder needs a molecular grid; we create it here)
    GridBuilder grid_builder(ctx, mol, radial_pts, angular_pts);
    if (!quiet) {
      std::cout << "Building DFT integration grid: "
                << radial_pts << " radial x " << angular_pts
                << " angular points\n";
    }
    auto mol_grid = grid_builder.build();

    // --- Build XC functional ---
    XCBuilder xc(ctx, basis_builder.basis(), mol_grid,
                  functional_id, radial_pts, angular_pts);
    if (!quiet) {
      std::cout << "Functional: " << functional_name;
      if (xc.is_hybrid())
        std::cout << " (hybrid, HF exchange scale = "
                  << xc.exchange_scale() << ")";
      std::cout << "\n\n";
    }

    // --- Set up DF-JK or exact JK ---
    // For DFT with DF, we use the auxiliary basis for J and K matrices
    DFJKBuilder* dfjk_ptr = nullptr;
    std::unique_ptr<DFJKBuilder> dfjk;

    if (use_df && aux_basis_ptr) {
      if (!quiet) std::cout << "Setting up density-fitted J/K builder...\n";
      auto xyz = mol.xyz_host();
      // DF plan exchange parameters for LRC functionals:
      // K matrix = EXCHANGE_FRACTION * K_SR + LRC_FRACTION * K_LR(omega)
      // Standard hybrids: use defaults (ex_frac=1.0, scaling in Fock build)
      double ex_frac = 1.0, lrc_frac = 0.0, lrc_omega = 0.0;
      if (functional_name == "CAM-B3LYP") {
        ex_frac = 0.19; lrc_frac = 0.46; lrc_omega = 0.33;
      } else if (functional_name == "WB97X" || functional_name == "WB97X-V") {
        ex_frac = 0.157706; lrc_frac = 0.842294; lrc_omega = 0.3;
      } else if (functional_name == "WB97M-V") {
        ex_frac = 0.0; lrc_frac = 1.0; lrc_omega = 0.3;
      } else if (functional_name == "LC-WPBE" || functional_name == "LC-WPBEH") {
        ex_frac = 0.0; lrc_frac = 1.0; lrc_omega = 0.4;
      } else if (functional_name == "HSE06") {
        ex_frac = 0.25; lrc_frac = -0.25; lrc_omega = 0.11;
      }

      dfjk = std::make_unique<DFJKBuilder>(
          ctx, basis_builder.basis(), aux_basis_ptr->basis(),
          xyz.data(), mol.natom(), ex_frac, lrc_frac, lrc_omega);
      dfjk_ptr = dfjk.get();
    } else {
      std::cerr << "Error: Density fitting is currently required.\n";
      std::cerr << "  Use --aux-basis to specify an auxiliary basis set.\n";
      return 1;
    }

    // --- Set up ECP integrals ---
    if (ecp_builder_ptr && ecp_builder_ptr->has_ecp()) {
      auto xyz_h = mol.xyz_host();
      ecp_int = std::make_unique<ECPIntegrals>(
          ctx, basis_builder.basis(), xyz_h.data(),
          ecp_builder_ptr->num_active_atoms(),
          ecp_builder_ptr->ecp_indices().data(),
          ecp_builder_ptr->ecp_atoms().data());
      ecp_int_ptr = ecp_int.get();
    }

    // --- Configure SCF ---
    SCFParams scf_params;
    scf_params.max_iter = max_iter;
    scf_params.conv_thresh = conv_thresh;
    scf_params.energy_conv_thresh = energy_conv;
    scf_params.diis_start = diis_start;
    scf_params.diis_max_space = diis_space;
    scf_params.damping = damping;
    scf_params.level_shift = level_shift;
    scf_params.verbose = verbose;
    scf_params.print_mos = print_mos;
    scf_params.print_level = quiet ? 0 : (verbose ? 2 : 1);

    // --- Run SCF ---
    SCFSolver scf(ctx, basis_builder, *dfjk_ptr, &xc,
                   ecp_builder_ptr, ecp_int_ptr,
                   mol, scf_params);
    scf.run();

    // --- Gradient computation ---
    if (gradient && scf.converged()) {
      // Analytical gradient (fast, uses densityScale=1.29 calibration)
      auto C_host = scf.mo_coefficients_host();
      auto D_host = scf.density_host();
      GradientComputer gc(ctx, basis_builder, *dfjk_ptr, &xc, ecp_int_ptr, mol, scf.orbital_energies(), C_host, D_host);
      auto analytical = gc.compute();

      std::cout << "\n=== Analytical Gradient (Ha/bohr) ===\n";
      std::cout << std::setprecision(8) << std::fixed;
      for (size_t a = 0; a < mol.natom(); a++) {
        double fx=analytical[3*a], fy=analytical[3*a+1], fz=analytical[3*a+2];
        std::cout << "  Atom " << a << " " << mol.atom(a).symbol
                  << std::setw(13) << fx << std::setw(13) << fy << std::setw(13) << fz
                  << "  |F| = " << std::sqrt(fx*fx+fy*fy+fz*fz) << "\n";
      }

      // Numerical gradient (validated reference)
      auto forces = numerical_gradient(mol, basis_path, aux_basis_path,
          ecp_path, is_pure, functional_id, radial_pts, angular_pts,
          scf_params, quiet);

      std::cout << "\n=== Numerical Gradient (Ha/bohr) ===\n";
      std::cout << std::setprecision(8) << std::fixed;
      for (size_t a = 0; a < mol.natom(); a++) {
        double fx = forces[3*a], fy = forces[3*a+1], fz = forces[3*a+2];
        double fnorm = std::sqrt(fx*fx + fy*fy + fz*fz);
        std::cout << "  Atom " << a << " " << mol.atom(a).symbol
                  << std::setw(13) << fx << std::setw(13) << fy
                  << std::setw(13) << fz << "  |F| = " << fnorm << "\n";
      }
      std::cout << "  (Convert to eV/Å: multiply by 51.422067)\n";
    } else if (gradient && !scf.converged()) {
      std::cerr << "Gradient requested but SCF not converged — skipping.\n";
    }

    // --- Final output ---
    if (!quiet || true) {  // always print final energy
      std::cout << "\n" << std::string(60, '=') << "\n";
      std::cout << "Final SCF energy: "
                << std::setprecision(14) << std::fixed
                << scf.total_energy() << " Ha\n";
      std::cout << "  ("
                << std::setprecision(8) << scf.total_energy() * 27.211386245988
                << " eV)\n";
      if (!scf.converged())
        std::cerr << "WARNING: SCF did not converge!\n";
      std::cout << std::string(60, '=') << "\n";
    }

  } catch (const std::exception& e) {
    std::cerr << "\nFatal error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
