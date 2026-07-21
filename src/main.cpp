/**
 * @file main.cpp
 * @brief CLI entry point for cuEST-based DFT calculations.
 *
 * Usage: cuest_dft --xyz <file> --basis <json_file> [options]
 *
 * Basis sets must be in BSE (Basis Set Exchange) JSON format.
 * ECP data is auto-detected from the JSON basis file for heavy elements.
 */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include "cuest_wrapper/basis.hpp"
#include "cuest_wrapper/context.hpp"
#include "cuest_wrapper/gradients.hpp"
#include "grad_numerical.hpp"
#include "cuest_wrapper/grid.hpp"
#include "cuest_wrapper/integrals.hpp"
#include "cuest_wrapper/molecule.hpp"
#include "io/parsers.hpp"
#include "cuest_wrapper/raii.hpp"
#include "scf.hpp"

using namespace cuest;

// ---------------------------------------------------------------------------
// Functional name -> ID mapping
// ---------------------------------------------------------------------------
struct FunctionalInfo {
  const char* name;
  int id;
};

static const FunctionalInfo kFunctionals[] = {
    {"HF",        XCBuilder::XC_HF},
    {"PBE",       XCBuilder::XC_PBE},
    {"B3LYP",     XCBuilder::XC_B3LYP},
    {"B3LYP5",    XCBuilder::XC_B3LYP5},
    {"PBE0",      XCBuilder::XC_PBE0},
    {"CAM-B3LYP", XCBuilder::XC_CAM_B3LYP},
    {"WB97X-V",   XCBuilder::XC_WB97X_V},
    {"WB97M-V",   XCBuilder::XC_WB97M_V},
    {"HSE06",     XCBuilder::XC_HSE06},
    {"M06",       XCBuilder::XC_M06},
    {"M06-2X",    XCBuilder::XC_M062X},
    {"LC-WPBE",   XCBuilder::XC_LC_WPBE},
    {"LC-WPBEH",  XCBuilder::XC_LC_WPBEH},
    {"WB97X",     XCBuilder::XC_WB97X},
};

// ---------------------------------------------------------------------------
// Print help
// ---------------------------------------------------------------------------
static void print_help(const char* prog) {
  std::cout
      << "cuEST DFT — GPU-accelerated density functional theory\n"
      << "Usage: " << prog << " --xyz <file> --basis <json_file> [options]\n\n"
      << "Required arguments:\n"
      << "  --xyz <path>             Input geometry in XYZ format\n"
      << "  --basis <path>           Basis set in BSE JSON format\n\n"
      << "Optional arguments:\n"
      << "  --aux-basis <path>       Auxiliary/DF (RI-J) basis set (BSE JSON)\n"
      << "  --functional <name>      XC functional (default: PBE)\n"
      << "  --radial-pts <n>         Radial grid points (default: 75)\n"
      << "  --angular-pts <n>        Angular Lebedev points (default: 302)\n"
      << "  --charge <int>           Total charge (default: 0)\n"
      << "  --multiplicity <int>     Spin multiplicity (default: 1; UKS if ≠ 1)\n"
      << "  --break-symmetry <rad>   UKS β HOMO/LUMO mix angle (default: 0.3)\n\n"
      << "SCF convergence options:\n"
      << "  --max-iter <n>           Max SCF iterations (default: 250)\n"
      << "  --conv-thresh <val>      RMS density convergence (default: 1e-8)\n"
      << "  --energy-conv <val>      Energy change convergence (default: 1e-8)\n"
      << "  --diis-start <n>         Iteration to enable DIIS (default: 1)\n"
      << "  --diis-space <n>         DIIS subspace dimension (default: 15)\n"
      << "  --damping <val>          Density damping factor (default: 0.0)\n\n"
      << "Other options:\n"
      << "  --quiet                  Minimal output\n"
      << "  --verbose                Verbose output\n"
      << "  --print-mos              Print final MO energies\n"
      << "  --spherical              Spherical (pure) orbital Gaussians (default)\n"
      << "  --cartesian              Cartesian orbital Gaussians\n"
      << "  --gradient               Nuclear gradient (analytic + numerical)\n"
      << "  --analytic-gradient      Analytic nuclear gradient only\n"
      << "  --help                   Show this help\n\n"
      << "Notes:\n"
      << "  Density fitting is required (--aux-basis). Closed-shell RKS\n"
      << "  (multiplicity 1) or unrestricted UKS (multiplicity > 1) are supported.\n"
      << "  UKS: --break-symmetry mixes β HOMO/LUMO when nα=nβ (BS-UKS).\n"
      << "  Analytic gradients (RKS and UKS) via --gradient / --analytic-gradient.\n"
      << "  The DF auxiliary basis is always spherical; --cartesian applies\n"
      << "  to the orbital basis only.\n\n"
      << "Available functionals:\n"
      << "  HF PBE B3LYP B3LYP5 PBE0 CAM-B3LYP WB97X-V WB97M-V\n"
      << "  HSE06 M06 M06-2X LC-WPBE LC-WPBEH WB97X\n\n"
      << "Basis sets in BSE JSON format can be downloaded from:\n"
      << "  https://www.basissetexchange.org/api/basis/<name>/format/json/\n"
      << "ECP data for heavy elements is auto-detected from the JSON file.\n"
      << std::endl;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  // --- Parse CLI arguments ---
  std::string xyz_path, basis_path, aux_basis_path;
  std::string functional_name = "PBE";
  int radial_pts = 75;
  int angular_pts = 302;
  int charge = 0;
  int multiplicity = 1;
  double break_symmetry = 0.3;
  int max_iter = 250;
  double conv_thresh = 1e-8;
  double energy_conv = 1e-8;
  int diis_start = 1;
  int diis_space = 15;
  double damping = 0.0;
  bool verbose = true;
  bool print_mos = false;
  bool gradient = false;
  bool numerical_gradient_flag = true;
  bool quiet = false;
  int is_pure = 1;  // orbital basis: 1 = spherical, 0 = Cartesian

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
    } else if (arg == "--break-symmetry" && i + 1 < argc) {
      break_symmetry = std::stod(argv[++i]);
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
    } else if (arg == "--no-df") {
      std::cerr << "Error: --no-df is not supported; density fitting is required.\n"
                << "  Provide --aux-basis with a BSE JSON auxiliary basis.\n";
      return 1;
    } else if (arg == "--level-shift") {
      std::cerr << "Error: --level-shift is not implemented.\n";
      return 1;
    } else if (arg == "--quiet") {
      quiet = true;
      verbose = false;
    } else if (arg == "--verbose") {
      verbose = true;
    } else if (arg == "--print-mos") {
      print_mos = true;
    } else if (arg == "--spherical") {
      is_pure = 1;
    } else if (arg == "--cartesian") {
      is_pure = 0;
    } else if (arg == "--gradient") {
      gradient = true;
      numerical_gradient_flag = true;
    } else if (arg == "--analytic-gradient") {
      gradient = true;
      numerical_gradient_flag = false;
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
      mol.add_atom_bohr(xyz_data.symbols[i],
                        xyz_data.xyz[3 * i],
                        xyz_data.xyz[3 * i + 1],
                        xyz_data.xyz[3 * i + 2]);
    }
    mol.set_charge(charge);
    mol.set_multiplicity(multiplicity);

    if (multiplicity < 1) {
      std::cerr << "Error: multiplicity must be >= 1.\n";
      return 1;
    }

    if (!quiet) {
      std::cout << "  Atoms: " << mol.natom() << "\n";
      std::cout << "  Charge: " << mol.charge() << "\n";
      std::cout << "  Multiplicity: " << mol.multiplicity()
                << (multiplicity == 1 ? " (RKS)" : " (UKS)") << "\n";
    }

    // --- Create cuEST context ---
    CuESTContext ctx;

    // --- Build primary basis from JSON ---
    if (!quiet) {
      std::cout << "Building primary basis from: " << basis_path << "\n";
      std::cout << "  Orbital shells: "
                << (is_pure ? "spherical (pure)" : "Cartesian") << "\n";
    }
    BasisBuilder basis_builder(ctx, mol, is_pure);
    basis_builder.build_from_json(basis_path);

    uint64_t nao = basis_builder.nao();
    if (!quiet) std::cout << "  Basis functions: " << nao << "\n";

    // --- Build auxiliary basis (required for DF) ---
    // cuEST DF currently requires a spherical auxiliary basis.
    if (aux_basis_path.empty()) {
      std::cerr << "Error: Density fitting is required.\n"
                << "  Use --aux-basis with a BSE JSON auxiliary basis.\n";
      return 1;
    }
    if (!quiet) std::cout << "Building auxiliary basis from: " << aux_basis_path
                          << " (spherical)\n";
    auto aux_basis = std::make_unique<AuxBasis>(ctx, mol, /*is_pure=*/1);
    aux_basis->build_from_json(aux_basis_path);
    uint64_t naux = ctx.query_nao(aux_basis->handle());
    if (!quiet) std::cout << "  Auxiliary basis functions: " << naux << "\n";

    // --- Build ECP from JSON (auto-detected) and apply Z_eff ---
    ECPBuilder* ecp_builder_ptr = nullptr;
    ECPIntegrals* ecp_int_ptr = nullptr;
    std::unique_ptr<ECPBuilder> ecp_builder;
    std::unique_ptr<ECPIntegrals> ecp_int;

    ecp_builder = std::make_unique<ECPBuilder>(ctx, mol);
    ecp_builder->build_from_json(basis_path);
    ecp_builder_ptr = ecp_builder.get();
    if (ecp_builder->has_ecp()) {
      ecp_builder->apply_to_molecule(mol);
      if (!quiet) {
        std::cout << "  ECP active: " << ecp_builder->total_ecp_electrons()
                  << " core electrons replaced (using Z_eff)\n";
      }
    } else if (!quiet) {
      std::cout << "  No ECP atoms found in molecule.\n";
    }

    if (mol.nelec() < 0) {
      std::cerr << "Error: negative electron count after charge/ECP (got "
                << mol.nelec() << ").\n";
      return 1;
    }
    if (mol.nalpha() < mol.nbeta() || mol.nalpha() + mol.nbeta() != mol.nelec()) {
      std::cerr << "Error: multiplicity " << multiplicity
                << " inconsistent with electron count " << mol.nelec() << ".\n";
      return 1;
    }
    if (!quiet) {
      std::cout << "  Electrons (valence): " << mol.nelec() << "\n";
      if (multiplicity == 1)
        std::cout << "  Occupied orbitals: " << mol.nocc() << "\n\n";
      else
        std::cout << "  Occupied orbitals: α=" << mol.nalpha()
                  << " β=" << mol.nbeta() << "\n\n";
    }

    // --- Build DFT grid ---
    GridBuilder grid_builder(ctx, mol, radial_pts, angular_pts);
    if (!quiet) {
      std::cout << "Building DFT integration grid: "
                << radial_pts << " radial x " << angular_pts
                << " angular points\n";
    }
    auto mol_grid = grid_builder.build();

    // --- Build XC functional ---
    XCBuilder xc(ctx, basis_builder.basis(), mol_grid, functional_id);
    if (!quiet) {
      std::cout << "Functional: " << functional_name;
      if (xc.is_hybrid())
        std::cout << " (hybrid, HF exchange scale = "
                  << xc.exchange_scale() << ")";
      std::cout << "\n\n";
    }

    // --- Set up DF-JK ---
    // Bake HF / LRC exchange fractions into the DF plan (NVIDIA recommended).
    // SCF and gradients then use coefficientScale = -1 for all hybrids.
    if (!quiet) std::cout << "Setting up density-fitted J/K builder...\n";
    auto xyz = mol.xyz_host();

    double ex_frac = 0.0, lrc_frac = 0.0, lrc_omega = 0.0;
    if (xc.is_hf()) {
      ex_frac = 1.0;
    } else if (xc.is_hybrid()) {
      if (xc.is_lrc()) {
        // Range-separated: plan holds both full-range and LRC fractions.
        if (functional_name == "CAM-B3LYP") {
          ex_frac = 0.19; lrc_frac = 0.46; lrc_omega = 0.33;
        } else if (functional_name == "WB97X" || functional_name == "WB97X-V") {
          ex_frac = 0.157706; lrc_frac = 0.842294; lrc_omega = 0.3;
        } else if (functional_name == "WB97M-V") {
          ex_frac = 0.15; lrc_frac = 0.85; lrc_omega = 0.3;
        } else if (functional_name == "LC-WPBE" || functional_name == "LC-WPBEH") {
          ex_frac = 0.0; lrc_frac = 1.0; lrc_omega = 0.4;
        } else if (functional_name == "HSE06") {
          ex_frac = 0.25; lrc_frac = -0.25; lrc_omega = 0.11;
        } else {
          // Fallback: use XC query scale as full-range HF only
          ex_frac = xc.exchange_scale();
        }
      } else {
        // Global hybrids (B3LYP, PBE0, ...): EXCHANGE_FRACTION = HF scale
        ex_frac = xc.exchange_scale();
      }
    }

    auto dfjk = std::make_unique<DFJKBuilder>(
        ctx, basis_builder.basis(), aux_basis->basis(),
        xyz.data(), mol.natom(), ex_frac, lrc_frac, lrc_omega);
    DFJKBuilder* dfjk_ptr = dfjk.get();

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
    scf_params.verbose = verbose;
    scf_params.print_mos = print_mos;
    scf_params.print_level = quiet ? 0 : (verbose ? 2 : 1);
    scf_params.break_symmetry = break_symmetry;

    // --- Run SCF ---
    SCFSolver scf(ctx, basis_builder, *dfjk_ptr, &xc,
                   ecp_builder_ptr, ecp_int_ptr,
                   mol, scf_params);
    scf.run();

    // --- Gradient computation ---
    if (gradient && scf.converged()) {
      try {
        std::unique_ptr<GradientComputer> gc_ptr;
        if (scf.is_uks()) {
          gc_ptr = std::make_unique<GradientComputer>(
              ctx, basis_builder, *dfjk_ptr, &xc, ecp_int_ptr, mol,
              scf.nocc_alpha(), scf.nocc_beta(),
              scf.orbital_energies(), scf.orbital_energies_beta(),
              scf.mo_coefficients_host(), scf.mo_coefficients_beta_host(),
              scf.density_alpha_host(), scf.density_beta_host());
        } else {
          gc_ptr = std::make_unique<GradientComputer>(
              ctx, basis_builder, *dfjk_ptr, &xc, ecp_int_ptr, mol,
              scf.nocc(), scf.orbital_energies(),
              scf.mo_coefficients_host(), scf.density_host());
        }
        GradientComputer& gc = *gc_ptr;
        auto analytical = gc.compute();

        if (verbose) {
          auto print_comp = [&](const char* label, const std::vector<double>& g) {
            std::cout << "  " << label;
            for (size_t a = 0; a < mol.natom(); a++)
              std::cout << " [" << std::setw(9) << g[3*a] << " " << std::setw(9)
                        << g[3*a+1] << " " << std::setw(9) << g[3*a+2] << "]";
            std::cout << "\n";
          };
          std::cout << "\n=== Gradient Components (Ha/bohr) ===\n";
          std::cout << std::setprecision(6) << std::fixed;
          print_comp("NUC    ", gc.nu());
          print_comp("OVERLAP", gc.ov());
          print_comp("KINETIC", gc.ke());
          print_comp("POT_B  ", gc.po());
          print_comp("POT_C  ", gc.pc());
          if (ecp_int_ptr) print_comp("ECP    ", gc.ecpg());
          print_comp("DF-J   ", gc.df());
          print_comp("XC     ", gc.xc());
          double fx=0,fy=0,fz=0;
          for(size_t i=0;i<mol.natom()*3;i+=3){fx+=analytical[i];fy+=analytical[i+1];fz+=analytical[i+2];}
          std::cout << "  Sum of forces: [" << fx << " " << fy << " " << fz << "] (should be ~0)\n\n";
        }

        std::cout << "=== Analytical Gradient (Ha/bohr) ===\n";
        std::cout << std::setprecision(8) << std::fixed;
        for (size_t a = 0; a < mol.natom(); a++) {
          double fx=analytical[3*a], fy=analytical[3*a+1], fz=analytical[3*a+2];
          std::cout << "  Atom " << a << " " << mol.atom(a).symbol
                    << std::setw(13) << fx << std::setw(13) << fy << std::setw(13) << fz
                    << "  |F| = " << std::sqrt(fx*fx+fy*fy+fz*fz) << "\n";
        }

        if (numerical_gradient_flag) {
          const char* exe_path = argv[0];
          auto forces = numerical_gradient(mol, basis_path, aux_basis_path,
              functional_id, radial_pts, angular_pts,
              scf_params, quiet, exe_path, is_pure);

          std::cout << "\n=== Numerical Gradient (Ha/bohr) ===\n";
          for (size_t a = 0; a < mol.natom(); a++) {
            double fx = forces[3*a], fy = forces[3*a+1], fz = forces[3*a+2];
            double fnorm = std::sqrt(fx*fx + fy*fy + fz*fz);
            std::cout << "  Atom " << a << " " << mol.atom(a).symbol
                      << std::setw(13) << fx << std::setw(13) << fy
                      << std::setw(13) << fz << "  |F| = " << fnorm << "\n";
          }
        }
      } catch (const std::exception& e) {
        // Keep SCF energy usable even if a cuEST derivative kernel fails
        // (known issue: hybrid DF grads on some SP-only bases).
        std::cerr << "WARNING: Analytic gradient failed: " << e.what() << "\n";
      }
    } else if (gradient && !scf.converged()) {
      std::cerr << "Gradient requested but SCF not converged — skipping.\n";
    }

    // --- Final output ---
    {
      std::cout << "\n" << std::string(60, '=') << "\n";
      std::cout << "Final SCF energy: "
                << std::setprecision(14) << std::fixed
                << scf.total_energy() << " Ha\n";
      std::cout << "  ("
                << std::setprecision(8) << scf.total_energy() * constants::ev_per_hartree
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
