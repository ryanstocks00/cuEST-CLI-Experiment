#!/usr/bin/env python3
"""
Test harness: compare cuEST DFT results against PySCF.

Usage:
    python3 test/run_pyscf_test.py

The test computes DFT energies for H2O with PBE/def2-SVP and
compares against PySCF reference values.
"""

import subprocess
import sys
import os
from pathlib import Path

PROJ_DIR = Path(__file__).parent.parent
DATA_DIR = PROJ_DIR / "data"
BUILD_DIR = PROJ_DIR / "build"
EXE = BUILD_DIR / "cuest_dft"


def run_cuest_dft(xyz_path, basis_path, aux_basis_path,
                  functional="PBE",
                  radial_pts=75, angular_pts=302,
                  max_iter=150):
    """Run the cuEST DFT executable and parse its output."""
    cmd = [
        str(EXE),
        "--xyz", str(xyz_path),
        "--basis", str(basis_path),
        "--aux-basis", str(aux_basis_path),
        "--functional", functional,
        "--radial-pts", str(radial_pts),
        "--angular-pts", str(angular_pts),
        "--max-iter", str(max_iter),
        "--conv-thresh", "1e-8",
        "--print-mos",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        print(f"C++ program failed with exit code {result.returncode}")
        return None

    # Parse energy from output
    energy = None
    for line in result.stdout.splitlines():
        if "Final SCF energy:" in line:
            parts = line.split()
            energy = float(parts[-2])  # "energy Ha" -> energy is second to last
            print(f"\nParsed cuEST total energy: {energy:.10f} Ha")
            break
    return energy


def run_pyscf_reference(xyz_path, functional="PBE", basis="def2SVP"):
    """Compute reference DFT energy using PySCF."""
    import numpy as np
    from pyscf import gto, dft

    # Read XYZ coordinates
    with open(xyz_path) as f:
        lines = f.readlines()
    natom = int(lines[0].strip())
    atoms = []
    for line in lines[2:2 + natom]:
        parts = line.split()
        if len(parts) >= 4:
            atoms.append((parts[0], (float(parts[1]), float(parts[2]), float(parts[3]))))

    # Build PySCF molecule
    mol = gto.M(
        atom=atoms,
        basis=basis,
        verbose=0,
    )

    # Map functional names to PySCF
    xc_map = {
        "PBE": "pbe,pbe",
        "B3LYP": "b3lyp",
        "PBE0": "pbe0",
        "CAM-B3LYP": "cam-b3lyp",
        "HSE06": "hse06",
        "M06": "m06",
        "M06-2X": "m062x",
        "WB97X-V": "wb97x_v",
        "WB97M-V": "wb97m_v",
        "LC-WPBE": "lc-wpbe",
        "WB97X": "wb97x",
    }
    xc = xc_map.get(functional, "pbe,pbe")

    # Run DFT
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.max_cycle = 150
    mf.conv_tol = 1e-8
    energy = mf.kernel()

    print(f"\nPySCF reference ({functional}/{basis}):")
    print(f"  Total energy: {energy:.10f} Ha")
    print(f"  HOMO: {mf.mo_energy[mol.nelec//2 - 1]:.6f} Ha")
    print(f"  LUMO: {mf.mo_energy[mol.nelec//2]:.6f} Ha")

    return energy


def main():
    print("=" * 60)
    print("  cuEST DFT vs PySCF Validation Test")
    print("=" * 60)

    xyz_path = DATA_DIR / "h2o.xyz"
    basis_path = DATA_DIR / "def2-svp.gbs"
    aux_path = DATA_DIR / "def2-universal-jkfit.gbs"

    if not EXE.exists():
        print(f"ERROR: cuEST executable not found at {EXE}")
        print("Build first: cmake --build build -j$(nproc)")
        sys.exit(1)

    functional = "PBE"
    print(f"\nTest: H2O, {functional}/def2-SVP (DF)")
    print("-" * 40)

    # Run cuEST DFT
    cuest_energy = run_cuest_dft(
        xyz_path, basis_path, aux_path,
        functional=functional,
    )

    if cuest_energy is None:
        print("cuEST run failed! Check errors above.")
        sys.exit(1)

    # Run PySCF reference
    try:
        pyscf_energy = run_pyscf_reference(xyz_path, functional=functional)
    except Exception as e:
        print(f"PySCF run failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Compare
    diff = abs(cuest_energy - pyscf_energy)
    print(f"\n{'=' * 60}")
    print(f"Results comparison:")
    print(f"  cuEST:  {cuest_energy:.10f} Ha")
    print(f"  PySCF:  {pyscf_energy:.10f} Ha")
    print(f"  Diff:   {diff:.10f} Ha ({diff * 27.211386:.6f} eV)")
    print(f"{'=' * 60}")

    # Tolerance check (1e-4 Ha = ~2.7 meV is reasonable for DFT grid differences)
    if diff < 1e-4:
        print("PASS: Energies agree within 1e-4 Ha")
        return 0
    elif diff < 1e-3:
        print("WARN: Energies differ by more than 1e-4 but less than 1e-3 Ha")
        print("      This may be due to grid or basis set differences.")
        return 0
    else:
        print("FAIL: Energy difference exceeds 1e-3 Ha")
        print("      Check basis set parser and integral accuracy.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
