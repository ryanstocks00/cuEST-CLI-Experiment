#!/usr/bin/env python3
"""
Comprehensive energy validation suite: cuEST DFT vs PySCF reference.

Loops over a matrix of molecules × functionals × basis sets × options,
runs both cuEST and PySCF, and writes a structured JSON results file.

Usage:
    python3 test/validate_energy.py                    # all tests
    python3 test/validate_energy.py --quick            # subset of quick tests
    python3 test/validate_energy.py --functional PBE   # single functional
    python3 test/validate_energy.py --molecule H2O     # single molecule
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJ_DIR = Path(__file__).resolve().parent.parent
MOLECULES_DIR = PROJ_DIR / "data" / "molecules"
BASIS_DIR = PROJ_DIR / "data" / "basis_sets"
BUILD_DIR = PROJ_DIR / "build"
EXE = BUILD_DIR / "cuest_dft"
RESULTS_FILE = PROJ_DIR / "test" / "validation_results.json"

# ---------------------------------------------------------------------------
# Test matrix definition
# ---------------------------------------------------------------------------

# Each entry: (molecule_xyz, basis, aux_basis, ecp_file or None, label)
BASIS_SETS = {
    "def2SVP": {
        "basis": "def2-svp.json",
        "aux": "def2-universal-jkfit.json",
    },
    "def2TZVP": {
        "basis": "def2-tzvp.json",
        "aux": "def2-universal-jkfit.json",
    },
    "def2SVPD": {
        "basis": "def2-svpd.json",
        "aux": "def2-universal-jkfit.json",
    },
    "cc-pVDZ": {
        "basis": "cc-pvdz.json",
        "aux": "def2-universal-jkfit.json",
    },
    "cc-pVTZ": {
        "basis": "cc-pvtz.json",
        "aux": "def2-universal-jkfit.json",
    },
}

FUNCTIONALS = [
    "PBE",
    "B3LYP",
    "PBE0",
    "CAM-B3LYP",
    "WB97X",
    "HSE06",
    "M06",
    "M06-2X",
]

# PySCF functional name mapping
PYSCF_XC_MAP = {
    "PBE": "pbe,pbe",
    "B3LYP": "b3lyp",
    "B3LYP5": "b3lyp5",
    "PBE0": "pbe0",
    "CAM-B3LYP": "cam-b3lyp",
    "WB97X-V": "wb97x_v",
    "WB97M-V": "wb97m_v",
    "HSE06": "hse06",
    "M06": "m06",
    "M06-2X": "m062x",
    "LC-WPBE": "lc-wpbe",
    "LC-WPBEH": "lc-wpbeh",
    "WB97X": "wb97x",
}

def build_test_matrix(quick=False):
    """Return list of test configurations as dicts."""
    molecules = [
        ("h2o.xyz", "H2O"),
        ("nh3.xyz", "NH3"),
        ("h2.xyz", "H2"),
        ("hf.xyz", "HF"),
        ("n2.xyz", "N2"),
        ("co2.xyz", "CO2"),
        ("ch4.xyz", "CH4"),
        ("c2h4.xyz", "C2H4"),
        ("bh3.xyz", "BH3"),
        ("so2.xyz", "SO2"),
    ]

    matrix = []

    for xyz_file, label in molecules:
        xyz_path = MOLECULES_DIR / xyz_file
        if not xyz_path.exists():
            continue

        for bs_key, bs in BASIS_SETS.items():
            basis_file = BASIS_DIR / bs["basis"]
            aux_file = BASIS_DIR / bs["aux"]
            if not basis_file.exists() or not aux_file.exists():
                continue

            # Pick functionals: all for common molecules, PBE-only for exotic
            if quick:
                funcs = ["PBE"]
            elif label in ("H2O", "NH3"):
                funcs = FUNCTIONALS
            else:
                funcs = ["PBE"]

            for func in funcs:
                # Skip exotic combos in quick mode
                if quick and bs_key not in ("def2SVP",):
                    continue

                matrix.append({
                    "molecule": label,
                    "xyz": str(xyz_path),
                    "basis": str(basis_file),
                    "aux_basis": str(aux_file),
                    "ecp": None,  # ECP auto-detected from JSON
                    "functional": func,
                    "basis_label": bs_key,
                    "has_ecp": False,
                })

    # Add heavy-element tests (ECP auto-detected from JSON)
    ecp_tests = [
        ("br2.xyz", "Br2"),
        ("i2.xyz", "I2"),
        ("ch2i2.xyz", "CH2I2"),
    ]
    for xyz_file, label in ecp_tests:
        xyz_path = MOLECULES_DIR / xyz_file
        basis_path = BASIS_DIR / "def2-svp.json"
        aux_path = BASIS_DIR / "def2-universal-jkfit.json"
        if not all(p.exists() for p in [xyz_path, basis_path, aux_path]):
            continue
        funcs = ["PBE"] if quick else ["PBE", "B3LYP", "PBE0"]
        for func in funcs:
            matrix.append({
                "molecule": label,
                "xyz": str(xyz_path),
                "basis": str(basis_path),
                "aux_basis": str(aux_path),
                "ecp": None,  # auto-detected from JSON
                "functional": func,
                "basis_label": "def2SVP",
                "has_ecp": True,
            })

    return matrix


# ---------------------------------------------------------------------------
# cuEST DFT runner
# ---------------------------------------------------------------------------

def run_cuest(config, timeout=300):
    """Run cuEST DFT and return parsed energy dict."""
    cmd = [
        str(EXE),
        "--xyz", config["xyz"],
        "--basis", config["basis"],
        "--aux-basis", config["aux_basis"],
        "--functional", config["functional"],
        "--max-iter", "150",
        "--conv-thresh", "1e-8",
        "--energy-conv", "1e-8",
        "--quiet",
    ]
    if config.get("ecp"):  # explicit ECP file override (legacy)
        cmd.extend(["--ecp", config["ecp"]])

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "wall_seconds": timeout}
    except Exception as e:
        return {"ok": False, "error": str(e), "wall_seconds": time.time() - start}

    elapsed = time.time() - start

    if result.returncode != 0:
        return {
            "ok": False,
            "error": f"exit code {result.returncode}",
            "stderr_tail": "\n".join(result.stderr.splitlines()[-10:]),
            "wall_seconds": elapsed,
        }

    # Parse energy and convergence
    energy = None
    converged = False
    for line in result.stdout.splitlines():
        if "Final SCF energy:" in line:
            key = "Final SCF energy:"
            pos = line.find(key)
            try:
                energy = float(line[pos + len(key):].strip().split()[0])
            except (ValueError, IndexError):
                pass
        # Check both stdout and stderr for convergence indicators
        if "SCF converged" in line.lower():
            converged = True
        if "Converged:" in line and "Yes" in line:
            converged = True
    # If no explicit convergence message but energy was printed without a
    # "did not converge" warning, assume converged (quiet mode behavior)
    if not converged:
        converged = ("WARNING: SCF did not converge" not in result.stdout
                     and "did not converge" not in result.stderr)

    if energy is None:
        return {
            "ok": False,
            "error": "could not parse energy",
            "stdout_tail": "\n".join(result.stdout.splitlines()[-20:]),
            "wall_seconds": elapsed,
        }

    return {
        "ok": True,
        "energy": energy,
        "converged": converged,
        "wall_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# BSE JSON to PySCF basis converter
# ---------------------------------------------------------------------------

def bse_json_to_pyscf(basis_path):
    """Parse a BSE JSON basis file into PySCF format using gto.basis.parse."""
    import json
    from pyscf import gto

    with open(basis_path) as f:
        data = json.load(f)

    _ELEMENTS = [
        "X",  "H",  "HE", "LI", "BE", "B",  "C",  "N",  "O",  "F",  "NE",
        "NA", "MG", "AL", "SI", "P",  "S",  "CL", "AR", "K",  "CA",
        "SC", "TI", "V",  "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
        "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y",  "ZR",
        "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN",
        "SB", "TE", "I",  "XE", "CS", "BA", "LA", "CE", "PR", "ND",
        "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB",
        "LU", "HF", "TA", "W",  "RE", "OS", "IR", "PT", "AU", "HG",
        "TL", "PB", "BI", "PO", "AT", "RN", "FR", "RA", "AC", "TH",
        "PA", "U",  "NP", "PU", "AM", "CM", "BK", "CF", "ES", "FM",
        "MD", "NO", "LR", "RF", "DB", "SG", "BH", "HS", "MT", "DS",
        "RG", "CN", "NH", "FL", "MC", "LV", "TS", "OG",
    ]
    am_chars = "SPDFGHIKLMNOQR"

    basis_dict = {}
    ecp_dict = {}

    for z_str, edata in data.get("elements", {}).items():
        z = int(z_str)
        if z <= 0 or z >= len(_ELEMENTS):
            continue
        symbol = _ELEMENTS[z]

        # --- Electron shells in NWChem format ---
        shells = edata.get("electron_shells", [])
        if shells:
            lines = []
            for sh in shells:
                for L in sh["angular_momentum"]:
                    am = am_chars[L] if L < len(am_chars) else "S"
                    lines.append(f"{symbol}    {am}")
                    for p in range(len(sh["exponents"])):
                        exp = float(sh["exponents"][p])
                        coeff_strs = [f"{float(ca[p]):.12f}" for ca in sh["coefficients"]]
                        lines.append(f"    {exp:.12f}  " + "  ".join(coeff_strs))
            basis_dict[symbol] = gto.basis.parse("\n".join(lines))

        # --- ECP in NWChem format ---
        ecp_pots = edata.get("ecp_potentials", [])
        if ecp_pots:
            nelec = edata.get("ecp_electrons", 0)
            lines = [f"{symbol} nelec {nelec}"]
            for pot in ecp_pots:
                L = pot["angular_momentum"][0]
                am = am_chars[L].lower() if L < len(am_chars) else "s"
                lines.append(f"{symbol} {am}-ul")
                nprim = len(pot["r_exponents"])
                lines.append(f"  {nprim}")
                for i in range(nprim):
                    r = pot["r_exponents"][i]
                    g = float(pot["gaussian_exponents"][i])
                    c = float(pot["coefficients"][0][i])
                    lines.append(f"  {r}  {g:.12f}  {c:.12f}")
            ecp_dict[symbol] = (nelec, "\n".join(lines))

    return basis_dict, (ecp_dict if ecp_dict else None)

# ---------------------------------------------------------------------------
# PySCF reference runner
# ---------------------------------------------------------------------------

def run_pyscf(config):
    """Compute PySCF reference energy using BSE JSON basis directly."""
    try:
        from pyscf import gto, dft
    except ImportError:
        return {"ok": False, "error": "PySCF not installed"}

    # Read XYZ
    try:
        with open(config["xyz"]) as f:
            lines = f.readlines()
        natom = int(lines[0].strip())
        atoms = []
        for line in lines[2:2 + natom]:
            parts = line.split()
            if len(parts) >= 4:
                atoms.append((parts[0], (float(parts[1]), float(parts[2]), float(parts[3]))))
    except Exception as e:
        return {"ok": False, "error": f"XYZ parse: {e}"}

    # Parse BSE JSON basis → PySCF format
    try:
        basis_dict, ecp_data = bse_json_to_pyscf(config["basis"])
    except Exception as e:
        return {"ok": False, "error": f"Basis parse: {e}"}

    # Build molecule with custom basis.
    try:
        # Parse ECP strings via PySCF's native parser
        mol_ecp = None
        if ecp_data:
            symbols_in_mol = {a[0].upper() for a in atoms}
            ecp_parsed = {}
            for s in symbols_in_mol:
                if s in ecp_data:
                    nelec, ecp_str = ecp_data[s]
                    ecp_parsed[s] = (nelec, gto.basis.parse_ecp(ecp_str))
            if ecp_parsed:
                mol_ecp = ecp_parsed
        if mol_ecp:
            mol = gto.M(atom=atoms, basis=basis_dict, ecp=mol_ecp, verbose=0)
        else:
            mol = gto.M(atom=atoms, basis=basis_dict, verbose=0)
    except Exception as e:
        return {"ok": False, "error": f"PySCF mol build: {e}"}

    xc = PYSCF_XC_MAP.get(config["functional"], "pbe,pbe")

    start = time.time()
    try:
        mf = dft.RKS(mol)
        mf.xc = xc
        mf.max_cycle = 200
        mf.conv_tol = 1e-10
        mf.grids.level = 5  # fine grid for accuracy
        mf.init_guess = '1e'  # core Hamiltonian guess (MINAO fails with custom basis)
        energy = mf.kernel()
        converged = mf.converged
        nelec = mol.nelec
        if isinstance(nelec, (tuple, list)):
            nelec = nelec[0] + nelec[1]
        nocc = nelec // 2
        homo = float(mf.mo_energy[nocc - 1])
        lumo = float(mf.mo_energy[nocc])
    except Exception as e:
        return {"ok": False, "error": f"PySCF kernel: {e}"}

    elapsed = time.time() - start

    if not converged:
        return {"ok": False, "error": "PySCF did not converge", "wall_seconds": elapsed}

    return {
        "ok": True,
        "energy": float(energy),
        "homo": homo,
        "lumo": lumo,
        "homo_lumo_gap": lumo - homo,
        "nelec": nelec,
        "wall_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------

def run_validation(matrix, results_file, skip_existing=False):
    """Run all tests and write results incrementally."""
    # Load existing results if any
    existing = {}
    if skip_existing and results_file.exists():
        with open(results_file) as f:
            for r in json.load(f):
                key = (r["molecule"], r["functional"], r["basis_label"], r.get("has_ecp", False))
                existing[key] = r

    results = list(existing.values())
    seen = set(existing.keys())

    n_total = len(matrix)
    n_pass = sum(1 for r in existing.values() if r.get("energy_ok") is True)
    n_fail = sum(1 for r in existing.values() if r.get("energy_ok") is False)
    n_skip = len(existing)

    for i, config in enumerate(matrix):
        key = (config["molecule"], config["functional"], config["basis_label"], config.get("has_ecp", False))
        if key in seen:
            continue

        label = f"{config['molecule']}/{config['functional']}/{config['basis_label']}"
        if config.get("has_ecp"):
            label += "+ECP"

        print(f"\n[{i+1}/{n_total}] {label}")
        print("-" * 60)

        # Run cuEST
        print("  cuEST...", end=" ", flush=True)
        cuest_r = run_cuest(config)
        if cuest_r["ok"]:
            print(f"{cuest_r['energy']:.10f} Ha ({cuest_r['wall_seconds']:.1f}s)")
        else:
            print(f"FAILED: {cuest_r.get('error', 'unknown')}")

        # Run PySCF
        print("  PySCF...", end=" ", flush=True)
        pyscf_r = run_pyscf(config)
        if pyscf_r["ok"]:
            print(f"{pyscf_r['energy']:.10f} Ha ({pyscf_r['wall_seconds']:.1f}s)")
        else:
            print(f"FAILED: {pyscf_r.get('error', 'unknown')}")

        # Compare
        diff_ha = None
        diff_ev = None
        ok = False
        if cuest_r["ok"] and pyscf_r["ok"]:
            diff_ha = cuest_r["energy"] - pyscf_r["energy"]
            from constants import HARTREE_PER_EV
            diff_ev = diff_ha * HARTREE_PER_EV
            ok = abs(diff_ha) < 1e-4  # 0.1 mHa tolerance
            status = "PASS" if ok else "FAIL"
            print(f"  Diff: {diff_ha:+.6e} Ha ({diff_ev:+.6f} eV)  [{status}]")
        else:
            print(f"  Diff: N/A (one or both failed)")

        # Collect system info on first run
        system_info = {}
        if i == 0 and cuest_r["ok"]:
            system_info["cuest_version"] = "0.1.0"

        result = {
            "molecule": config["molecule"],
            "functional": config["functional"],
            "basis_label": config["basis_label"],
            "has_ecp": config.get("has_ecp", False),
            "xyz_file": Path(config["xyz"]).name,
            "cuest_ok": cuest_r["ok"],
            "cuest_energy_ha": cuest_r.get("energy"),
            "cuest_converged": cuest_r.get("converged"),
            "cuest_wall_s": cuest_r.get("wall_seconds"),
            "cuest_error": cuest_r.get("error"),
            "pyscf_ok": pyscf_r["ok"],
            "pyscf_energy_ha": pyscf_r.get("energy"),
            "pyscf_homo_ha": pyscf_r.get("homo"),
            "pyscf_lumo_ha": pyscf_r.get("lumo"),
            "pyscf_wall_s": pyscf_r.get("wall_seconds"),
            "pyscf_error": pyscf_r.get("error"),
            "energy_diff_ha": diff_ha,
            "energy_diff_ev": diff_ev,
            "energy_ok": ok if (cuest_r["ok"] and pyscf_r["ok"]) else None,
        }
        results.append(result)

        if ok:
            n_pass += 1
        elif cuest_r["ok"] and pyscf_r["ok"]:
            n_fail += 1

        # Write incrementally (so partial results survive crashes)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        sys.stdout.flush()

    return results, n_pass, n_fail, n_skip


def print_summary(results, n_pass, n_fail, n_skip):
    """Print a formatted summary table."""
    n_total = len(results)
    n_tested = n_pass + n_fail
    n_error = n_total - n_tested - n_skip

    print("\n" + "=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Total configurations: {n_total}")
    print(f"  Previously cached:   {n_skip}")
    print(f"  Tested this run:     {n_tested}")
    print(f"  PASS (|diff| < 0.1 mHa): {n_pass}")
    print(f"  FAIL:                {n_fail}")
    if n_error > 0:
        print(f"  ERROR (run failed):  {n_error}")
    print("-" * 70)

    # Group by molecule
    by_mol = {}
    for r in results:
        mol = r["molecule"]
        if mol not in by_mol:
            by_mol[mol] = []
        by_mol[mol].append(r)

    for mol, entries in sorted(by_mol.items()):
        passes = sum(1 for e in entries if e.get("energy_ok") is True)
        fails = sum(1 for e in entries if e.get("energy_ok") is False)
        errors = sum(1 for e in entries if e.get("energy_ok") is None)
        print(f"\n  {mol}:")
        for e in entries:
            if e.get("energy_ok") is True:
                icon = "✓"
            elif e.get("energy_ok") is False:
                icon = "✗"
            else:
                icon = "?"
            diff_str = ""
            if e.get("energy_diff_ha") is not None:
                diff_str = f"  Δ={e['energy_diff_ha']:+.3e} Ha"
            ecp_flag = " +ECP" if e.get("has_ecp") else ""
            print(f"    {icon} {e['functional']:12s} / {e['basis_label']:12s}{ecp_flag}{diff_str}")

    print("\n" + "=" * 70)
    if n_fail == 0 and n_error == 0:
        print("  ALL TESTS PASSED")
    else:
        if n_fail > 0:
            print(f"  {n_fail} FAILURE(S) — check validation_results.json")
        if n_error > 0:
            print(f"  {n_error} ERROR(S) — cuEST or PySCF run failed")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="cuEST DFT energy validation suite")
    parser.add_argument("--quick", action="store_true",
                        help="Run a quick subset of tests")
    parser.add_argument("--functional", type=str,
                        help="Filter to a single functional")
    parser.add_argument("--molecule", type=str,
                        help="Filter to a single molecule")
    parser.add_argument("--output", type=str, default=str(RESULTS_FILE),
                        help="Output JSON file path")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tests already in the output file")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing results file")
    args = parser.parse_args()

    # Check binary
    if not EXE.exists():
        print(f"ERROR: cuEST binary not found at {EXE}")
        print("Build first: cmake --build build -j$(nproc)")
        sys.exit(1)

    # Check PySCF
    try:
        import pyscf  # noqa: F401
    except ImportError:
        print("ERROR: PySCF not installed. Install with: pip install pyscf")
        sys.exit(1)

    # Build matrix
    matrix = build_test_matrix(quick=args.quick)

    # Filter
    if args.functional:
        matrix = [c for c in matrix if c["functional"] == args.functional]
    if args.molecule:
        matrix = [c for c in matrix if c["molecule"] == args.molecule]

    if not matrix:
        print("No tests match the given filters.")
        sys.exit(1)

    print("=" * 70)
    print("  cuEST DFT — Energy Validation Suite")
    print("=" * 70)
    print(f"  Configurations to test: {len(matrix)}")
    print(f"  Binary: {EXE}")
    print(f"  Results: {args.output}")
    print(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # Handle existing results
    if args.force and Path(args.output).exists():
        Path(args.output).unlink()
        print("  (removed existing results file)")

    results, n_pass, n_fail, n_skip = run_validation(
        matrix, Path(args.output),
        skip_existing=args.skip_existing and not args.force,
    )

    print_summary(results, n_pass, n_fail, n_skip)

    # Exit code
    if n_fail > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
