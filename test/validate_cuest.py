#!/usr/bin/env python3
"""
Validate cuEST DFT energies against a PySCF reference file.

Usage:
    # First, generate the reference:
    python3 test/generate_reference.py

    # Then validate cuEST against it:
    python3 test/validate_cuest.py                    # all configs
    python3 test/validate_cuest.py --molecule H2O     # single molecule
    python3 test/validate_cuest.py --functional PBE   # single functional
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJ_DIR = Path(__file__).resolve().parent.parent
EXE = PROJ_DIR / "build" / "cuest_dft"
REFERENCE_FILE = PROJ_DIR / "test" / "reference.json"
RESULTS_FILE = PROJ_DIR / "test" / "cuest_results.json"


# ---------------------------------------------------------------------------
# cuEST runner
# ---------------------------------------------------------------------------

def run_cuest(config, aux_path, timeout=300):
    """Run cuEST DFT and return parsed energy dict."""
    cmd = [
        str(EXE),
        "--xyz", config["xyz"],
        "--basis", config["basis"],
        "--aux-basis", aux_path,
        "--functional", config["functional"],
        "--max-iter", "150",
        "--conv-thresh", "1e-8",
        "--energy-conv", "1e-8",
        "--quiet",
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "wall_s": timeout}
    except Exception as e:
        return {"ok": False, "error": str(e), "wall_s": time.time() - start}

    elapsed = time.time() - start

    if result.returncode != 0:
        return {"ok": False, "error": f"exit {result.returncode}",
                "stderr": "\n".join(result.stderr.splitlines()[-5:]),
                "wall_s": elapsed}

    energy = None
    converged = False
    for line in result.stdout.splitlines():
        if "Final SCF energy:" in line:
            pos = line.find("Final SCF energy:")
            try:
                energy = float(line[pos + len("Final SCF energy:"):].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if "SCF converged" in line.lower():
            converged = True
        if "Converged:" in line and "Yes" in line:
            converged = True
    if not converged:
        converged = ("did not converge" not in result.stdout
                     and "did not converge" not in result.stderr)

    if energy is None:
        return {"ok": False, "error": "could not parse energy", "wall_s": elapsed}

    return {"ok": True, "energy_ha": energy, "converged": converged, "wall_s": elapsed}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Validate cuEST against PySCF reference")
    p.add_argument("--reference", type=str, default=str(REFERENCE_FILE),
                   help="Path to reference JSON")
    p.add_argument("--molecule", type=str)
    p.add_argument("--functional", type=str)
    p.add_argument("--tolerance-ha", type=float, default=1e-4,
                   help="Max energy difference in Ha (default: 1e-4 = 0.1 mHa)")
    args = p.parse_args()

    if not EXE.exists():
        print(f"ERROR: cuEST binary not found at {EXE}")
        print("Build first: cmake --build build -j$(nproc)")
        sys.exit(1)

    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"ERROR: Reference file not found: {ref_path}")
        print("Generate it first: python3 test/generate_reference.py")
        sys.exit(1)

    with open(ref_path) as f:
        ref_data = json.load(f)

    references = ref_data["results"]
    print(f"Loaded {len(references)} references from {ref_path}")
    print(f"Generated: {ref_data.get('generated', 'unknown')}")

    # Build aux path lookup
    aux_path = str(PROJ_DIR / "data" / "basis_sets" / "def2-universal-jkfit.json")

    # Filter references by molecule/functional
    refs = references
    if args.molecule:
        refs = [r for r in refs if r["molecule"] == args.molecule]
    if args.functional:
        refs = [r for r in refs if r["functional"] == args.functional]

    # Build molecule -> xyz path lookup
    mol_xyz = {}
    for f in (PROJ_DIR / "data" / "molecules").glob("*.xyz"):
        # map molecule names to xyz files
        pass  # we look up by molecule label below

    # Map molecule label -> basis path (from generate_reference.py config)
    def get_basis_path(ref):
        """Infer basis path from reference entry."""
        bs_map = {
            "def2SVP": "def2-svp.json", "def2TZVP": "def2-tzvp.json",
            "def2SVPD": "def2-svpd.json", "cc-pVDZ": "cc-pvdz.json",
            "cc-pVTZ": "cc-pvtz.json",
        }
        bs_file = bs_map.get(ref["basis_label"], "def2-svp.json")
        return str(PROJ_DIR / "data" / "basis_sets" / bs_file)

    def get_xyz_path(mol_label):
        """Map molecule label to xyz file."""
        name_map = {
            "H2O": "h2o.xyz", "NH3": "nh3.xyz", "H2": "h2.xyz",
            "HF": "hf.xyz", "N2": "n2.xyz", "CO2": "co2.xyz",
            "CH4": "ch4.xyz", "C2H4": "c2h4.xyz", "BH3": "bh3.xyz",
            "SO2": "so2.xyz", "Br2": "br2.xyz", "I2": "i2.xyz",
            "CH2I2": "ch2i2.xyz",
        }
        fname = name_map.get(mol_label, f"{mol_label.lower()}.xyz")
        return str(PROJ_DIR / "data" / "molecules" / fname)

    # Build test list from references directly
    testable = []
    for ref in refs:
        xyz = get_xyz_path(ref["molecule"])
        basis = get_basis_path(ref)
        if not Path(xyz).exists() or not Path(basis).exists():
            continue
        testable.append({
            "molecule": ref["molecule"],
            "functional": ref["functional"],
            "basis_label": ref["basis_label"],
            "xyz": xyz,
            "basis": basis,
        })

    if not testable:
        print("No configs with references match filters.")
        sys.exit(1)

    print(f"\nTesting {len(testable)} configurations...\n")

    results = []
    n_pass, n_fail, n_error = 0, 0, 0

    for i, config in enumerate(testable):
        label = f"{config['molecule']}/{config['functional']}/{config['basis_label']}"
        # Find matching reference
        ref = next((r for r in refs if r["molecule"] == config["molecule"]
                    and r["functional"] == config["functional"]
                    and r["basis_label"] == config["basis_label"]), None)

        if not ref or not ref.get("ok"):
            print(f"[{i+1}/{len(testable)}] {label} — SKIP (ref failed: {ref.get('error','?') if ref else 'missing'})")
            results.append({**config, "ref_ok": False, "ref_error": ref.get("error") if ref else "missing"})
            n_error += 1
            continue

        print(f"[{i+1}/{len(testable)}] {label} ...", end=" ", flush=True)
        cuest_r = run_cuest(config, aux_path)

        if not cuest_r["ok"]:
            print(f"FAILED: {cuest_r['error']}")
            results.append({**config, "cuest_ok": False, "cuest_error": cuest_r["error"],
                           "ref_energy_ha": ref["energy_ha"]})
            n_error += 1
            continue

        diff = cuest_r["energy_ha"] - ref["energy_ha"]
        passed = abs(diff) < args.tolerance_ha
        status = "PASS" if passed else "FAIL"
        print(f"{cuest_r['energy_ha']:.10f} Ha  Δ={diff:+.3e}  [{status}]")

        if passed:
            n_pass += 1
        else:
            n_fail += 1

        results.append({
            "molecule": config["molecule"],
            "functional": config["functional"],
            "basis_label": config["basis_label"],
            "ref_energy_ha": ref["energy_ha"],
            "ref_homo_ha": ref.get("homo_ha"),
            "ref_lumo_ha": ref.get("lumo_ha"),
            "cuest_energy_ha": cuest_r["energy_ha"],
            "cuest_converged": cuest_r["converged"],
            "cuest_wall_s": cuest_r["wall_s"],
            "energy_diff_ha": diff,
            "energy_ok": passed,
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS: {n_pass} PASS, {n_fail} FAIL, {n_error} ERROR  (of {len(testable)})")
    print(f"  Tolerance: {args.tolerance_ha} Ha")
    print(f"{'='*60}")

    if n_fail > 0:
        print("\nFailures:")
        for r in results:
            if r.get("energy_ok") is False:
                print(f"  {r['molecule']}/{r['functional']}/{r['basis_label']}: "
                      f"Δ={r['energy_diff_ha']:+.3e} Ha")
    if n_error > 0:
        print("\nErrors:")
        for r in results:
            if r.get("cuest_ok") is False:
                print(f"  {r['molecule']}/{r['functional']}/{r['basis_label']}: {r.get('cuest_error','?')}")
            elif r.get("ref_ok") is False:
                print(f"  {r['molecule']}/{r['functional']}/{r['basis_label']}: ref: {r.get('ref_error','?')}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULTS_FILE}")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
