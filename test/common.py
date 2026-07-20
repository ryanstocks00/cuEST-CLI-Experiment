#!/usr/bin/env python3
"""Shared helpers for cuEST validation scripts."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

PROJ_DIR = Path(__file__).resolve().parent.parent
MOLECULES_DIR = PROJ_DIR / "data" / "molecules"
BASIS_DIR = PROJ_DIR / "data" / "basis_sets"
BUILD_DIR = PROJ_DIR / "build"
EXE = BUILD_DIR / "cuest_dft"

ELEMENTS = [
    "X", "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE",
    "NA", "MG", "AL", "SI", "P", "S", "CL", "AR", "K", "CA",
    "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
    "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y", "ZR",
    "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN",
    "SB", "TE", "I", "XE", "CS", "BA", "LA", "CE", "PR", "ND",
    "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB",
    "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG",
    "TL", "PB", "BI", "PO", "AT", "RN", "FR", "RA", "AC", "TH",
    "PA", "U", "NP", "PU", "AM", "CM", "BK", "CF", "ES", "FM",
    "MD", "NO", "LR", "RF", "DB", "SG", "BH", "HS", "MT", "DS",
    "RG", "CN", "NH", "FL", "MC", "LV", "TS", "OG",
]
AM_CHARS = "SPDFGHIKLMNOQR"

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

BASIS_JSON = {
    "def2SVP": "def2-svp.json",
    "def2TZVP": "def2-tzvp.json",
    "def2SVPD": "def2-svpd.json",
    "cc-pVDZ": "cc-pvdz.json",
    "cc-pVTZ": "cc-pvtz.json",
}
AUX_JSON = "def2-universal-jkfit.json"


def is_finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def load_xyz(path) -> list:
    with open(path) as f:
        lines = f.readlines()
    natom = int(lines[0].strip())
    atoms = []
    for line in lines[2:2 + natom]:
        parts = line.split()
        if len(parts) >= 4:
            atoms.append((parts[0], (float(parts[1]), float(parts[2]), float(parts[3]))))
    return atoms


def bse_json_to_pyscf(basis_path):
    """Parse BSE JSON into PySCF basis + ECP dicts (correct NWChem ECP labels)."""
    from pyscf import gto

    with open(basis_path) as f:
        data = json.load(f)

    basis_dict = {}
    ecp_dict = {}

    for z_str, edata in data.get("elements", {}).items():
        z = int(z_str)
        if z <= 0 or z >= len(ELEMENTS):
            continue
        symbol = ELEMENTS[z]

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


def parse_cuest_energy(stdout: str):
    """Return (energy, converged) or (None, False). Rejects non-finite values."""
    energy = None
    for line in stdout.splitlines():
        if "Final SCF energy:" in line:
            key = "Final SCF energy:"
            pos = line.find(key)
            try:
                energy = float(line[pos + len(key):].strip().split()[0])
            except (ValueError, IndexError):
                pass
    if energy is not None and not is_finite(energy):
        return None, False
    converged = (
        "WARNING: SCF did not converge" not in stdout
        and "did not converge" not in stdout.lower()
        and energy is not None
    )
    if "Converged: No" in stdout:
        converged = False
    if "Converged: Yes" in stdout or "SCF converged" in stdout.lower():
        converged = True
    return energy, converged


def parse_gradient_block(stdout: str, header: str):
    """Parse 'Atom N SYM fx fy fz |F| = ...' lines after a header."""
    lines = stdout.splitlines()
    start = None
    for i, line in enumerate(lines):
        if header in line:
            start = i + 1
            break
    if start is None:
        return None
    grad = []
    for line in lines[start:]:
        if line.strip().startswith("===") or not line.strip():
            if grad:
                break
            continue
        m = re.match(
            r"\s*Atom\s+(\d+)\s+\S+\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)",
            line,
        )
        if m:
            grad.extend([float(m.group(2)), float(m.group(3)), float(m.group(4))])
        elif grad:
            break
    return grad if grad else None


def run_cuest_cmd(cmd, timeout=300):
    env = dict(os.environ)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "returncode": -1, "stdout": "", "stderr": ""}
    except Exception as e:
        return {"ok": False, "error": str(e), "returncode": -1, "stdout": "", "stderr": ""}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": None if result.returncode == 0 else f"exit code {result.returncode}",
    }


def run_pyscf_df(atoms, basis_path, aux_path, functional, charge=0, spin=0,
                 grid_level=3):
    """PySCF RKS with density fitting using the same BSE JSON bases as cuEST."""
    from pyscf import gto, dft

    basis_dict, ecp_data = bse_json_to_pyscf(basis_path)
    aux_dict, _ = bse_json_to_pyscf(aux_path)

    mol_ecp = None
    if ecp_data:
        symbols = {a[0].upper() for a in atoms}
        ecp_parsed = {}
        for s in symbols:
            if s in ecp_data:
                _, ecp_str = ecp_data[s]
                ecp_parsed[s] = gto.basis.parse_ecp(ecp_str)
        if ecp_parsed:
            mol_ecp = ecp_parsed

    kwargs = dict(atom=atoms, basis=basis_dict, charge=charge, spin=spin, verbose=0)
    if mol_ecp:
        kwargs["ecp"] = mol_ecp
    mol = gto.M(**kwargs)

    xc = PYSCF_XC_MAP.get(functional, functional.lower())
    mf = dft.RKS(mol).density_fit()
    mf.with_df.auxbasis = aux_dict
    mf.xc = xc
    mf.max_cycle = 200
    mf.conv_tol = 1e-10
    mf.grids.level = grid_level
    mf.init_guess = "1e"
    energy = mf.kernel()
    if not mf.converged or not is_finite(energy):
        return {"ok": False, "error": "PySCF did not converge or non-finite energy",
                "mf": mf, "mol": mol}
    nelec = mol.nelec
    if isinstance(nelec, (tuple, list)):
        nelec = nelec[0] + nelec[1]
    nocc = nelec // 2
    return {
        "ok": True,
        "energy": float(energy),
        "homo": float(mf.mo_energy[nocc - 1]),
        "lumo": float(mf.mo_energy[nocc]),
        "nelec": nelec,
        "nocc": nocc,
        "mf": mf,
        "mol": mol,
    }
