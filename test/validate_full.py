#!/usr/bin/env python3
"""
Full validation: energies + analytical/numerical gradients vs PySCF-DF.
Uses BSE JSON bases under data/basis_sets/ and molecules under data/molecules/.
"""
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    AUX_JSON, BASIS_DIR, BASIS_JSON, EXE, MOLECULES_DIR, PROJ_DIR,
    is_finite, parse_cuest_energy, parse_gradient_block, run_cuest_cmd,
    run_pyscf_df,
)

TARGETS = [
    # (molecule_xyz, label, functional, basis_key, check_gradient)
    ("h2o.xyz", "H2O", "PBE", "def2SVP", True),
    ("h2o.xyz", "H2O", "B3LYP", "def2SVP", True),
    ("h2o.xyz", "H2O", "PBE0", "def2SVP", True),
    ("h2o.xyz", "H2O", "CAM-B3LYP", "def2SVP", True),
    ("h2o.xyz", "H2O", "WB97X", "def2SVP", False),
    ("nh3.xyz", "NH3", "PBE", "def2TZVP", False),
    ("nh3.xyz", "NH3", "PBE", "def2SVP", True),
    ("i2.xyz", "I2", "PBE", "def2SVP", False),
    ("br2.xyz", "Br2", "PBE", "def2SVP", False),
]


def run_cuest_energy(xyz, basis, aux, functional, max_iter=100):
    cmd = [str(EXE), "--xyz", str(xyz), "--basis", str(basis),
           "--aux-basis", str(aux), "--functional", functional,
           "--max-iter", str(max_iter), "--conv-thresh", "1e-8", "--quiet"]
    r = run_cuest_cmd(cmd, timeout=300)
    energy, converged = parse_cuest_energy(r["stdout"] + "\n" + r["stderr"])
    if not r["ok"] or energy is None or not converged:
        return None, False
    return energy, True


def run_cuest_gradient(xyz, basis, aux, functional, max_iter=100):
    cmd = [str(EXE), "--xyz", str(xyz), "--basis", str(basis),
           "--aux-basis", str(aux), "--functional", functional,
           "--max-iter", str(max_iter), "--gradient", "--quiet"]
    r = run_cuest_cmd(cmd, timeout=900)
    text = r["stdout"] + "\n" + r["stderr"]
    energy, converged = parse_cuest_energy(text)
    if not r["ok"] or energy is None or not converged:
        return None, None, None
    ana = parse_gradient_block(text, "Analytical Gradient")
    num = parse_gradient_block(text, "Numerical Gradient")
    return energy, ana, num


def main():
    print("=" * 70)
    print("  cuEST DFT — FULL Validation Suite")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if not EXE.exists():
        print(f"ERROR: {EXE} not found. Build first.")
        return 1

    aux_path = BASIS_DIR / AUX_JSON
    if not aux_path.exists():
        print(f"ERROR: missing aux basis {aux_path}")
        return 1

    results = []
    n_pass = n_fail = 0

    for xyz_name, mol_name, func, basis_key, check_grad in TARGETS:
        xyz_path = MOLECULES_DIR / xyz_name
        basis_path = BASIS_DIR / BASIS_JSON[basis_key]
        if not xyz_path.exists() or not basis_path.exists():
            print(f"  SKIP {mol_name}/{func}/{basis_key} (missing files)")
            n_fail += 1
            continue

        if check_grad:
            cu_e, cu_ana, cu_num = run_cuest_gradient(
                xyz_path, basis_path, aux_path, func)
        else:
            cu_e, ok = run_cuest_energy(xyz_path, basis_path, aux_path, func)
            if not ok:
                cu_e = None
            cu_ana = cu_num = None

        from common import load_xyz
        atoms = load_xyz(xyz_path)
        try:
            py = run_pyscf_df(atoms, str(basis_path), str(aux_path), func)
        except Exception as ex:
            py = {"ok": False, "error": str(ex)}

        py_e = py.get("energy") if py.get("ok") else None
        py_grad = None
        if check_grad and py.get("ok"):
            try:
                from pyscf import grad
                g = grad.RKS(py["mf"]).kernel()
                py_grad = g.flatten().tolist()
            except Exception:
                py_grad = None

        e_diff = None
        e_ok = False
        if is_finite(cu_e) and is_finite(py_e):
            e_diff = abs(cu_e - py_e)
            e_ok = e_diff < 1e-4

        g_ana_ok = g_num_ok = False
        g_ana_rms = g_num_rms = g_self_rms = None
        if check_grad and cu_ana is not None and py_grad is not None:
            ca = np.array(cu_ana)
            pg = np.array(py_grad)
            if len(ca) == len(pg):
                g_ana_rms = float(np.sqrt(np.mean((ca - pg) ** 2)))
                g_ana_ok = g_ana_rms < 1e-4
        if check_grad and cu_num is not None and py_grad is not None:
            cn = np.array(cu_num)
            pg = np.array(py_grad)
            if len(cn) == len(pg):
                g_num_rms = float(np.sqrt(np.mean((cn - pg) ** 2)))
                g_num_ok = g_num_rms < 1e-3
        if check_grad and cu_ana is not None and cu_num is not None:
            ca = np.array(cu_ana)
            cn = np.array(cu_num)
            if len(ca) == len(cn):
                g_self_rms = float(np.sqrt(np.mean((ca - cn) ** 2)))

        passed = e_ok
        if check_grad:
            # Gradient must be present and within tolerance when requested
            passed = e_ok and g_ana_ok and (g_num_ok or cu_num is None)

        parts = [
            "PASS" if passed else "FAIL",
            f"E={cu_e:.8f}" if is_finite(cu_e) else "E=ERR",
            f"dE={e_diff * 1000:.3f}mHa" if e_diff is not None else "dE=N/A",
        ]
        if check_grad:
            parts.append(f"ana={g_ana_rms:.2e}" if g_ana_rms is not None else "ana=N/A")
            parts.append(f"num={g_num_rms:.2e}" if g_num_rms is not None else "num=N/A")
            parts.append(f"a-n={g_self_rms:.2e}" if g_self_rms is not None else "a-n=N/A")

        label = f"{mol_name:5s}/{func:10s}/{basis_key:8s}"
        print(f"  {label:40s} {' '.join(parts)}")

        results.append({
            "molecule": mol_name, "functional": func, "basis": basis_key,
            "energy_ok": e_ok,
            "energy_diff_mHa": (e_diff * 1000) if e_diff is not None else None,
            "grad_ana_rms": g_ana_rms, "grad_num_rms": g_num_rms,
            "grad_self_rms": g_self_rms,
            "passed": passed,
        })
        if passed:
            n_pass += 1
        else:
            n_fail += 1

    print(f"\n{'=' * 70}")
    print(f"  Summary: {n_pass} PASS, {n_fail} FAIL")
    print(f"{'=' * 70}")

    with open(PROJ_DIR / "test" / "results_full.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
