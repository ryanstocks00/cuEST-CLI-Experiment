#!/usr/bin/env python3
"""
Comprehensive validation: cuEST DFT vs PySCF.
Tests multiple molecules, functionals, and basis sets.
Downloads basis sets from Basis Set Exchange in Gaussian94 format.
"""
import subprocess, numpy as np, json, sys, os, tempfile, shutil, urllib.request
from pathlib import Path
from datetime import datetime

PROJ = Path(__file__).parent.parent
EXE = PROJ / "build" / "cuest_dft"
DATA = PROJ / "data"
BSE_API = "https://www.basissetexchange.org/api/basis"

# PySCF Bohr radius (standardize on this)
BOHR = 0.52917721092

# ======== CONFIGURATION ========
MOLECULES = {
    "H2O": """
O  0.000000  0.000000  0.000000
H  0.756950  0.585882  0.000000
H -0.756950  0.585882  0.000000
""",
    "NH3": """
N  0.000000  0.000000  0.116300
H  0.000000  0.938800 -0.271300
H  0.812900 -0.469400 -0.271300
H -0.812900 -0.469400 -0.271300
""",
    "CH4": """
C  0.000000  0.000000  0.000000
H  0.629118  0.629118  0.629118
H -0.629118 -0.629118  0.629118
H  0.629118 -0.629118 -0.629118
H -0.629118  0.629118 -0.629118
""",
    "CO2": """
C  0.000000  0.000000  0.000000
O  1.160000  0.000000  0.000000
O -1.160000  0.000000  0.000000
""",
    "N2": """
N  0.000000  0.000000  0.550000
N  0.000000  0.000000 -0.550000
""",
    "HF": """
F  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.917000
""",
    "H2": """
H  0.000000  0.000000  0.370424
H  0.000000  0.000000 -0.370424
""",
    "C2H4": """
C  0.000000  0.000000  0.669500
C  0.000000  0.000000 -0.669500
H  0.000000  0.928900  1.232100
H  0.000000 -0.928900  1.232100
H  0.000000  0.928900 -1.232100
H  0.000000 -0.928900 -1.232100
""",
    "BH3": """
B  0.000000  0.000000  0.000000
H  0.000000  1.190000  0.000000
H  1.030700 -0.595000  0.000000
H -1.030700 -0.595000  0.000000
""",
    "SO2": """
S  0.000000  0.000000  0.369700
O  1.240600  0.000000 -0.184800
O -1.240600  0.000000 -0.184800
""",
}

BASIS_SETS = {
    "def2-SVP":    "def2-svp",
    "def2-TZVP":   "def2-tzvp",
    "def2-QZVP":   "def2-qzvp",
    "cc-pVDZ":     "cc-pvdz",
    "cc-pVTZ":     "cc-pvtz",
    "6-31G*":      "6-31g-d",
    "6-311G**":    "6-311g-2d1f",
}

FUNCTIONALS = ["PBE", "B3LYP", "PBE0"]

# ======== DOWNLOAD BASIS SETS ========
def download_basis(name, fmt="gaussian94"):
    """Download basis set from BSE and save as Gaussian94 .gbs file."""
    out_path = DATA / f"{name}.gbs"
    if out_path.exists():
        return out_path

    url = f"{BSE_API}/{name}/format/{fmt}/?version=1&optimize_general=true"
    print(f"  Downloading {name}...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cuEST-dft/1.0"})
        data = urllib.request.urlopen(req, timeout=30).read()
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"OK ({len(data)} bytes)")
        return out_path
    except Exception as e:
        print(f"FAILED: {e}")
        return None

def download_jkfit(name):
    """Download matching JKfit auxiliary basis."""
    jkfit_name = name.replace("def2-", "def2-universal-jkfit")
    # Map: cc-pV*Z -> cc-pV*Z-JKFIT, 6-31G* -> def2-universal-jkfit
    if "cc-p" in jkfit_name:
        jkfit_name = jkfit_name.replace("universal-", "") + "-jkfit"
    if "6-31" in jkfit_name or "6-311" in jkfit_name:
        jkfit_name = "def2-universal-jkfit"

    out_path = DATA / f"{jkfit_name}.gbs"
    if out_path.exists():
        return out_path

    url = f"{BSE_API}/{jkfit_name}/format/gaussian94/?version=1&optimize_general=true"
    print(f"  Downloading JKfit {jkfit_name}...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cuEST-dft/1.0"})
        data = urllib.request.urlopen(req, timeout=30).read()
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"OK ({len(data)} bytes)")
        return out_path
    except Exception as e:
        print(f"FAILED: {e} — using def2-universal-jkfit as fallback")
        return DATA / "def2-universal-jkfit.gbs"

# ======== RUN CUEST ========
def run_cuest(xyz_path, basis_path, aux_path, functional="PBE", max_iter=100):
    cmd = [
        str(EXE), "--xyz", str(xyz_path), "--basis", str(basis_path),
        "--aux-basis", str(aux_path), "--functional", functional,
        "--max-iter", str(max_iter), "--conv-thresh", "1e-8",
        "--quiet"
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        # Parse energy
        for line in r.stdout.splitlines():
            if "Final SCF energy:" in line:
                return float(line.split()[-2]), True
        # Fallback: parse last Etot
        for line in reversed(r.stdout.splitlines()):
            if "Etot =" in line:
                return float(line.split("Etot =")[1].split()[0]), True
        return None, False
    except Exception as e:
        return str(e), False

# ======== RUN PySCF ========
def run_pyscf(atoms_str, basis_name, functional="PBE"):
    from pyscf import gto, dft

    atoms = []
    for line in atoms_str.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 4:
            atoms.append((parts[0], (float(parts[1]), float(parts[2]), float(parts[3]))))

    xc_map = {"PBE": "pbe,pbe", "B3LYP": "b3lyp", "PBE0": "pbe0"}
    xc = xc_map.get(functional, "pbe,pbe")

    mol = gto.M(atom=atoms, basis=basis_name, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = xc
    mf = mf.density_fit()
    mf.max_cycle = 200
    mf.conv_tol = 1e-8

    try:
        e = mf.kernel()
        if not mf.converged:
            return None, False
        return e, True
    except Exception as ex:
        return str(ex), False

# ======== MAIN ========
def main():
    print("=" * 70)
    print(f"  cuEST DFT — Comprehensive Validation")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if not EXE.exists():
        print(f"\nERROR: {EXE} not found. Build first: cmake --build build")
        sys.exit(1)

    # Download basis sets
    print("\n--- Downloading Basis Sets ---")
    basis_files = {}
    jkfit_files = {}
    for name, bse_id in BASIS_SETS.items():
        path = download_basis(bse_id)
        if path:
            basis_files[name] = path
            if name.startswith("def2-"):
                jkfit = download_jkfit(bse_id)
                if jkfit:
                    jkfit_files[name] = jkfit

    # Also ensure def2-universal-jkfit exists as default
    default_jk = download_basis("def2-universal-jkfit")
    if default_jk:
        jkfit_files["default"] = default_jk

    # Create molecule XYZ files
    print("\n--- Creating Molecule XYZ files ---")
    xyz_files = {}
    for name, atoms_str in MOLECULES.items():
        xyz_path = DATA / f"test_{name}.xyz"
        # Convert Angstrom to same Bohr convention as PySCF
        lines = atoms_str.strip().split("\n")
        natom = len(lines)
        # Convert coordinates to Angstrom (they're already in Angstrom in MOLECULES dict)
        with open(xyz_path, "w") as f:
            f.write(f"{natom}\n{name} test molecule\n")
            for line in lines:
                parts = line.split()
                f.write(f"{parts[0]}  {float(parts[1]):10.6f}  {float(parts[2]):10.6f}  {float(parts[3]):10.6f}\n")
        xyz_files[name] = xyz_path

    # Run validation
    print("\n" + "=" * 70)
    print("  Validation Results")
    print("=" * 70)

    results = []

    for mol_name in sorted(MOLECULES.keys()):
        for basis_name in sorted(basis_files.keys()):
            # Skip large basis sets for large molecules to save time
            if mol_name in ("C2H4", "SO2") and "QZ" in basis_name:
                continue
            if mol_name in ("C2H4", "SO2") and "cc-pVTZ" in basis_name:
                continue

            for func in FUNCTIONALS:
                xyz_path = xyz_files[mol_name]
                basis_path = basis_files[basis_name]
                aux_path = jkfit_files.get(basis_name, jkfit_files.get("default"))

                if aux_path is None:
                    continue

                # Run cuEST
                cu_e, cu_ok = run_cuest(xyz_path, basis_path, aux_path, func)

                # Run PySCF
                py_e, py_ok = run_pyscf(MOLECULES[mol_name],
                                         BASIS_SETS[basis_name], func)

                status = "???"
                diff = None
                if cu_ok and py_ok and isinstance(cu_e, float) and isinstance(py_e, float):
                    diff = abs(cu_e - py_e)
                    if diff < 1e-4:
                        status = "✅ PASS"
                    elif diff < 1e-3:
                        status = "⚠️  WARN"
                    else:
                        status = "❌ FAIL"
                elif not cu_ok:
                    status = f"❌ cuEST failed: {cu_e}"
                elif not py_ok:
                    status = f"❌ PySCF failed: {py_e}"

                cu_str = f"{cu_e:18.10f}" if isinstance(cu_e, float) else f"{str(cu_e):>18s}"
                diff_str = f"{diff:10.6f}" if diff else f"{'N/A':>10s}"
                line = f"  {mol_name:5s} {basis_name:12s} {func:6s}  cuEST={cu_str}  diff={diff_str}  {status}"
                print(line)
                results.append({
                    "molecule": mol_name, "basis": basis_name, "functional": func,
                    "cuest_energy": cu_e, "pyscf_energy": py_e, "diff": diff,
                    "status": status
                })

    # Summary
    n_pass = sum(1 for r in results if "PASS" in r["status"])
    n_warn = sum(1 for r in results if "WARN" in r["status"])
    n_fail = sum(1 for r in results if "FAIL" in r["status"])
    n_total = len(results)

    print(f"\n{'=' * 70}")
    print(f"  Summary: {n_pass}/{n_total} PASS, {n_warn} WARN, {n_fail} FAIL")
    print(f"{'=' * 70}")

    # Save results
    with open(PROJ / "test" / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return 0 if n_fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
