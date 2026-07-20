#!/usr/bin/env python3
"""
Test cuEST DFT against PySCF for H2O with PBE/def2-SVP.

Usage:
    python3 test/test_water.py
    python3 test/test_water.py --functional PBE0
    python3 test/test_water.py --basis def2-svp --aux-basis def2-universal-jkfit

Requirements:
    pyscf, numpy
"""

import subprocess
import sys
import os
import re
import argparse
import numpy as np

ANG_TO_BOHR = 1.0 / 0.529177210903

# Water geometry (Angstrom)
H2O_XYZ = """3
Water molecule
O   0.000000   0.000000   0.117790
H   0.756950   0.000000  -0.471160
H  -0.756950   0.000000  -0.471160"""


def parse_xyz(xyz_string):
    """Parse XYZ string into atom list."""
    atoms = []
    lines = xyz_string.strip().split("\n")
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            sym = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append((sym, (x, y, z)))
    return atoms


def run_pyscf(atoms, basis="def2svp", aux_basis="def2-universal-jkfit",
              functional="PBE", charge=0, spin=0):
    """Run DFT calculation with PySCF and return total energy."""
    from pyscf import gto, dft, df
    from pyscf.dft import gen_grid

    mol = gto.Mole()
    mol.atom = [(sym, coord) for sym, coord in atoms]
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 4
    mol.build()

    # Use density fitting
    mf = dft.RKS(mol).density_fit()
    mf.with_df.auxbasis = aux_basis
    mf.xc = functional
    # Match grid: (75, 302) is the default
    mf.grids = gen_grid.Grids(mol)
    mf.grids.level = 3  # (75, 302) grid

    e_dft = mf.kernel()
    mo_energies = mf.mo_energy
    return e_dft, mo_energies, mol.nao_nr()


def run_cuest(xyz_path, basis_path, aux_basis_path, functional="PBE",
              binary="build/cuest_dft"):
    """Run cuEST DFT binary and parse output."""
    cmd = [
        binary,
        "--xyz", xyz_path,
        "--basis", basis_path,
        "--aux-basis", aux_basis_path,
        "--functional", functional,
        "--print-mos",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        print("cuEST binary failed:")
        print(stderr)
        print(stdout)
        return None

    # Parse total energy
    energy_match = re.search(r"Final SCF energy:\s+([-\d.]+)\s+Ha", stdout)
    if energy_match:
        e_total = float(energy_match.group(1))
    else:
        print("Could not parse total energy from cuEST output.")
        print(stdout)
        return None

    # Parse MO energies if available
    mo_lines = []
    in_mo_section = False
    for line in stdout.split("\n"):
        if "Orbital energies" in line:
            in_mo_section = True
            continue
        if in_mo_section and "MO" in line:
            parts = line.split()
            if len(parts) >= 3:
                mo_lines.append(float(parts[2]))

    return e_total, mo_lines


def main():
    parser = argparse.ArgumentParser(
        description="Compare cuEST DFT with PySCF reference")
    parser.add_argument("--functional", default="PBE",
                        help="XC functional (default: PBE)")
    parser.add_argument("--basis", default="def2-svp",
                        help="Basis set name for PySCF")
    parser.add_argument("--aux-basis", default="def2-universal-jkfit",
                        help="Auxiliary basis for density fitting")
    parser.add_argument("--cuest-binary", default="build/cuest_dft",
                        help="Path to cuEST DFT binary")
    parser.add_argument("--xyz", default=None,
                        help="XYZ file path (or use built-in H2O)")
    parser.add_argument("--basis-file", default="data/def2-svp.gbs",
                        help="GBS basis file for cuEST")
    parser.add_argument("--aux-basis-file",
                        default="data/def2-universal-jkfit.gbs",
                        help="GBS aux basis file for cuEST")
    args = parser.parse_args()

    print("=" * 60)
    print("  cuEST DFT vs PySCF Validation")
    print("=" * 60)
    print(f"  Functional: {args.functional}")
    print(f"  Basis: {args.basis}")
    print(f"  Aux basis: {args.aux_basis}")
    print()

    # Setup geometry
    if args.xyz:
        with open(args.xyz) as f:
            xyz_data = f.read()
    else:
        xyz_data = H2O_XYZ

    atoms = parse_xyz(xyz_data)
    print(f"  Molecule: {len(atoms)} atoms")
    for sym, (x, y, z) in atoms:
        print(f"    {sym:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")

    # Run PySCF
    print(f"\n--- PySCF ({args.functional}/{args.basis}) ---")
    try:
        e_pyscf, mo_pyscf, nao = run_pyscf(
            atoms, basis=args.basis, aux_basis=args.aux_basis,
            functional=args.functional)
        print(f"  PySCF total energy: {e_pyscf:.12f} Ha")
        print(f"  Number of AOs: {nao}")
        print(f"  HOMO energy: {mo_pyscf[nao//2-1]:.8f} Ha")
        print(f"  LUMO energy: {mo_pyscf[nao//2]:.8f} Ha")
    except ImportError:
        print("  PySCF not available - skipping reference calculation")
        e_pyscf = None
    except Exception as e:
        print(f"  PySCF error: {e}")
        e_pyscf = None

    # Run cuEST
    print(f"\n--- cuEST ({args.functional}) ---")

    # Write XYZ file for cuEST
    xyz_path = "/tmp/test_h2o.xyz"
    with open(xyz_path, "w") as f:
        f.write(xyz_data)

    try:
        result = run_cuest(
            xyz_path,
            args.basis_file,
            args.aux_basis_file,
            functional=args.functional,
            binary=args.cuest_binary)
    except FileNotFoundError:
        print(f"  cuEST binary not found: {args.cuest_binary}")
        print("  Build with: cmake --build build")
        result = None
    except Exception as e:
        print(f"  cuEST error: {e}")
        result = None

    if result:
        e_cuest, mo_cuest = result
        print(f"  cuEST total energy: {e_cuest:.12f} Ha")
        if mo_cuest:
            nocc = len(atoms) * 5 // 2  # rough for water
            if len(mo_cuest) > nocc:
                print(f"  HOMO energy: {mo_cuest[min(nocc-1, len(mo_cuest)-1)]:.8f} Ha")
                print(f"  LUMO energy: {mo_cuest[min(nocc, len(mo_cuest)-1)]:.8f} Ha")
    else:
        e_cuest = None

    # Compare
    if e_pyscf is not None and e_cuest is not None:
        diff = abs(e_pyscf - e_cuest)
        print(f"\n{'='*60}")
        print(f"  Energy difference: {diff:.10f} Ha ({diff*27.2114:.6f} eV)")
        if diff < 1e-4:
            print("  PASS: Energies agree within 0.1 mHa")
        elif diff < 1e-2:
            print("  WARNING: Energy difference > 0.1 mHa")
        else:
            print("  FAIL: Energy difference > 10 mHa")
        print(f"{'='*60}")
        return 0 if diff < 1e-2 else 1
    else:
        print("\n  No comparison possible (missing reference or cuEST result)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
