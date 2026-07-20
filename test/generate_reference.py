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
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJ_DIR = Path(__file__).resolve().parent.parent
MOLECULES_DIR = PROJ_DIR / "data" / "molecules"
BASIS_DIR = PROJ_DIR / "data" / "basis_sets"
REFERENCE_FILE = PROJ_DIR / "test" / "reference.json"

# ---------------------------------------------------------------------------
# Configuration (same as validate_energy.py)
# ---------------------------------------------------------------------------

BASIS_SETS = {
    "def2SVP":  {"basis": "def2-svp.json"},
    "def2TZVP": {"basis": "def2-tzvp.json"},
    "def2SVPD": {"basis": "def2-svpd.json"},
    "cc-pVDZ":  {"basis": "cc-pvdz.json"},
    "cc-pVTZ":  {"basis": "cc-pvtz.json"},
}

FUNCTIONALS = ["PBE", "B3LYP", "PBE0", "CAM-B3LYP", "WB97X", "HSE06", "M06", "M06-2X"]

PYSCF_XC_MAP = {
    "PBE": "pbe,pbe", "B3LYP": "b3lyp", "PBE0": "pbe0",
    "CAM-B3LYP": "cam-b3lyp", "WB97X": "wb97x", "HSE06": "hse06",
    "M06": "m06", "M06-2X": "m062x",
}

_ELEMENTS = [
    "X","H","HE","LI","BE","B","C","N","O","F","NE","NA","MG","AL","SI","P","S","CL","AR",
    "K","CA","SC","TI","V","CR","MN","FE","CO","NI","CU","ZN","GA","GE","AS","SE","BR","KR",
    "RB","SR","Y","ZR","NB","MO","TC","RU","RH","PD","AG","CD","IN","SN","SB","TE","I","XE",
    "CS","BA","LA","CE","PR","ND","PM","SM","EU","GD","TB","DY","HO","ER","TM","YB",
    "LU","HF","TA","W","RE","OS","IR","PT","AU","HG","TL","PB","BI","PO","AT","RN",
    "FR","RA","AC","TH","PA","U","NP","PU","AM","CM","BK","CF","ES","FM",
    "MD","NO","LR","RF","DB","SG","BH","HS","MT","DS","RG","CN","NH","FL","MC","LV","TS","OG",
]
AM_CHARS = "SPDFGHIKLMNOQR"


# ---------------------------------------------------------------------------
# BSE JSON -> PySCF basis converter
# ---------------------------------------------------------------------------

def bse_json_to_pyscf(basis_path):
    """Parse BSE JSON into PySCF basis + ECP dicts."""
    from pyscf import gto

    with open(basis_path) as f:
        data = json.load(f)

    basis_dict = {}
    ecp_dict = {}

    for z_str, edata in data.get("elements", {}).items():
        z = int(z_str)
        if z <= 0 or z >= len(_ELEMENTS):
            continue
        symbol = _ELEMENTS[z]

        # Electron shells
        shells = edata.get("electron_shells", [])
        if shells:
            lines = []
            for sh in shells:
                for L in sh["angular_momentum"]:
                    am = AM_CHARS[L] if L < len(AM_CHARS) else "S"
                    lines.append(f"{symbol}    {am}")
                    for p in range(len(sh["exponents"])):
                        exp = float(sh["exponents"][p])
                        coeff_strs = [f"{float(ca[p]):.12f}" for ca in sh["coefficients"]]
                        lines.append(f"    {exp:.12f}  " + "  ".join(coeff_strs))
            basis_dict[symbol] = gto.basis.parse("\n".join(lines))

        # ECP (NWChem format: highest-L is local "ul", rest are projectors)
        ecp_pots = edata.get("ecp_potentials", [])
        if ecp_pots:
            nelec = edata.get("ecp_electrons", 0)
            max_L = max(pot["angular_momentum"][0] for pot in ecp_pots)
            lines = [f"{symbol} nelec {nelec}"]
            for pot in ecp_pots:
                L = pot["angular_momentum"][0]
                nprim = len(pot["r_exponents"])
                if L == max_L:
                    label = f"{symbol} ul"
                else:
                    am = AM_CHARS[L] if L < len(AM_CHARS) else str(L)
                    label = f"{symbol} {am}"
                lines.append(label)
                lines.append(f"  {nprim}")
                for i in range(nprim):
                    r = pot["r_exponents"][i]
                    g = float(pot["gaussian_exponents"][i])
                    c = float(pot["coefficients"][0][i])
                    lines.append(f"  {r}  {g:.12f}  {c:.12f}")
            ecp_dict[symbol] = (nelec, "\n".join(lines))

    return basis_dict, (ecp_dict if ecp_dict else None)


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
                    "basis": str(basis_file), "functional": func,
                    "basis_label": bs_key,
                })

    # Heavy-element tests (ECP auto-detected)
    for xyz_file, label in [("br2.xyz", "Br2"), ("i2.xyz", "I2"), ("ch2i2.xyz", "CH2I2")]:
        xyz_path = MOLECULES_DIR / xyz_file
        basis_path = BASIS_DIR / "def2-svp.json"
        if not xyz_path.exists() or not basis_path.exists():
            continue
        funcs = ["PBE"] if quick else ["PBE", "B3LYP", "PBE0"]
        for func in funcs:
            matrix.append({
                "molecule": label, "xyz": str(xyz_path),
                "basis": str(basis_path), "functional": func,
                "basis_label": "def2SVP", "has_ecp": True,
            })
    return matrix


# ---------------------------------------------------------------------------
# PySCF runner
# ---------------------------------------------------------------------------

def run_pyscf(config):
    """Compute reference energy with PySCF using BSE JSON basis."""
    from pyscf import gto, dft

    with open(config["xyz"]) as f:
        lines = f.readlines()
    natom = int(lines[0].strip())
    atoms = [(p[0], (float(p[1]), float(p[2]), float(p[3])))
             for line in lines[2:2+natom] if len(p := line.split()) >= 4]

    basis_dict, ecp_data = bse_json_to_pyscf(config["basis"])

    mol_ecp = None
    if ecp_data:
        symbols = {a[0].upper() for a in atoms}
        ecp_parsed = {}
        for s in symbols:
            if s in ecp_data:
                _, ecp_str = ecp_data[s]  # nelec is embedded in parse_ecp result
                ecp_parsed[s] = gto.basis.parse_ecp(ecp_str)
        if ecp_parsed:
            mol_ecp = ecp_parsed

    mol = gto.M(atom=atoms, basis=basis_dict, ecp=mol_ecp, verbose=0) if mol_ecp \
          else gto.M(atom=atoms, basis=basis_dict, verbose=0)

    xc = PYSCF_XC_MAP.get(config["functional"], "pbe,pbe")
    mf = dft.RKS(mol)
    mf.xc = xc; mf.max_cycle = 200; mf.conv_tol = 1e-10
    mf.grids.level = 5; mf.init_guess = '1e'

    start = time.time()
    energy = mf.kernel()
    elapsed = time.time() - start

    if not mf.converged:
        return {"ok": False, "error": "PySCF did not converge", "wall_s": elapsed}

    nelec = mol.nelec
    if isinstance(nelec, (tuple, list)):
        nelec = nelec[0] + nelec[1]
    nocc = nelec // 2

    return {
        "ok": True,
        "energy_ha": float(energy),
        "homo_ha": float(mf.mo_energy[nocc - 1]),
        "lumo_ha": float(mf.mo_energy[nocc]),
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
                "pyscf_basis_source": "bse_json_direct",
                "results": results,
            }, f, indent=2)

    print(f"\nDone: {n_ok} OK, {n_fail} failed  -> {args.output}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
