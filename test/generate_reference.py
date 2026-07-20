#!/usr/bin/env python3
"""
Generate reference energies for all validation configurations using PySCF.

Reads the same BSE JSON basis files as cuEST, computes reference DFT
energies, and writes them to a reference JSON file.

Usage:
    python3 test/generate_reference.py                    # all configs
    python3 test/generate_reference.py --quick            # PBE-only subset
    python3 test/generate_reference.py --molecule H2O     # single molecule
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    AUX_JSON, BASIS_DIR, MOLECULES_DIR, PROJ_DIR, PYSCF_XC_MAP,
    load_xyz, run_pyscf_df,
)

REFERENCE_FILE = PROJ_DIR / "test" / "reference.json"

BASIS_SETS = {
    "def2SVP":  {"basis": "def2-svp.json", "aux": AUX_JSON},
    "def2TZVP": {"basis": "def2-tzvp.json", "aux": AUX_JSON},
    "def2SVPD": {"basis": "def2-svpd.json", "aux": AUX_JSON},
    "cc-pVDZ":  {"basis": "cc-pvdz.json", "aux": AUX_JSON},
    "cc-pVTZ":  {"basis": "cc-pvtz.json", "aux": AUX_JSON},
}

FUNCTIONALS = ["PBE", "B3LYP", "PBE0", "CAM-B3LYP", "WB97X", "HSE06", "M06", "M06-2X"]


# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

def build_test_matrix(quick=False):
    molecules = [
        ("h2o.xyz", "H2O"), ("nh3.xyz", "NH3"), ("h2.xyz", "H2"),
        ("hf.xyz", "HF"), ("n2.xyz", "N2"), ("co2.xyz", "CO2"),
        ("ch4.xyz", "CH4"), ("c2h4.xyz", "C2H4"), ("bh3.xyz", "BH3"),
        ("so2.xyz", "SO2"),
    ]
    matrix = []
    for xyz_file, label in molecules:
        xyz_path = MOLECULES_DIR / xyz_file
        if not xyz_path.exists():
            continue
        for bs_key, bs in BASIS_SETS.items():
            basis_file = BASIS_DIR / bs["basis"]
            if not basis_file.exists():
                continue
            funcs = ["PBE"] if quick else (FUNCTIONALS if label in ("H2O", "NH3") else ["PBE"])
            for func in funcs:
                if quick and bs_key not in ("def2SVP",):
                    continue
                matrix.append({
                    "molecule": label, "xyz": str(xyz_path),
                    "basis": str(basis_file),
                    "aux_basis": str(BASIS_DIR / bs["aux"]),
                    "functional": func,
                    "basis_label": bs_key,
                })

    # Heavy-element tests (ECP auto-detected)
    for xyz_file, label in [("br2.xyz", "Br2"), ("i2.xyz", "I2"), ("ch2i2.xyz", "CH2I2")]:
        xyz_path = MOLECULES_DIR / xyz_file
        basis_path = BASIS_DIR / "def2-svp.json"
        aux_path = BASIS_DIR / AUX_JSON
        if not xyz_path.exists() or not basis_path.exists():
            continue
        funcs = ["PBE"] if quick else ["PBE", "B3LYP", "PBE0"]
        for func in funcs:
            matrix.append({
                "molecule": label, "xyz": str(xyz_path),
                "basis": str(basis_path),
                "aux_basis": str(aux_path),
                "functional": func,
                "basis_label": "def2SVP", "has_ecp": True,
            })
    return matrix


# ---------------------------------------------------------------------------
# PySCF-DF runner (aligned with cuEST)
# ---------------------------------------------------------------------------

def run_pyscf(config):
    """Compute density-fitted reference energy with PySCF."""
    atoms = load_xyz(config["xyz"])
    start = time.time()
    r = run_pyscf_df(
        atoms, config["basis"], config["aux_basis"], config["functional"],
        grid_level=3,
    )
    elapsed = time.time() - start
    if not r["ok"]:
        return {"ok": False, "error": r.get("error", "PySCF failed"), "wall_s": elapsed}
    return {
        "ok": True,
        "energy_ha": r["energy"],
        "homo_ha": r["homo"],
        "lumo_ha": r["lumo"],
        "wall_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate PySCF reference energies")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--molecule", type=str)
    p.add_argument("--functional", type=str)
    p.add_argument("--output", type=str, default=str(REFERENCE_FILE))
    args = p.parse_args()

    matrix = build_test_matrix(quick=args.quick)
    if args.molecule:
        matrix = [c for c in matrix if c["molecule"] == args.molecule]
    if args.functional:
        matrix = [c for c in matrix if c["functional"] == args.functional]
    if not matrix:
        print("No configurations match filters."); sys.exit(1)

    print(f"Generating {len(matrix)} PySCF references...")
    print(f"Output: {args.output}")
    results = []
    n_ok, n_fail = 0, 0

    for i, config in enumerate(matrix):
        label = f"{config['molecule']}/{config['functional']}/{config['basis_label']}"
        print(f"[{i+1}/{len(matrix)}] {label} ...", end=" ", flush=True)

        try:
            r = run_pyscf(config)
        except Exception as e:
            r = {"ok": False, "error": str(e)}

        if r["ok"]:
            print(f"{r['energy_ha']:.10f} Ha ({r['wall_s']:.1f}s)")
            n_ok += 1
        else:
            print(f"FAILED: {r['error']}")
            n_fail += 1

        results.append({
            "molecule": config["molecule"],
            "functional": config["functional"],
            "basis_label": config["basis_label"],
            **r,
        })

        with open(args.output, "w") as f:
            json.dump({
                "generated": datetime.now(timezone.utc).isoformat(),
                "pyscf_basis_source": "bse_json_density_fit",
                "results": results,
            }, f, indent=2)

    print(f"\nDone: {n_ok} OK, {n_fail} failed  -> {args.output}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
