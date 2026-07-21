#!/usr/bin/env python3
"""Validate density-fitted Hartree–Fock (cuEST vs PySCF).

DF-HF has no XC grid, so residuals isolate DF J/K (and basis/aux matching).
Optionally compare the same geometries under PBE / WB97X to show how much
extra error comes from the XC quadrature.

Usage:
    python3 test/validate_hf.py
    python3 test/validate_hf.py --quick
    python3 test/validate_hf.py --compare-dft   # also run PBE/WB97X on same cells
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BASIS_DIR,
    BASIS_SETS,
    EXE,
    MOLECULES,
    MOLECULES_DIR,
    as_grad_nx3,
    flatten_grad,
    load_xyz,
    parse_cuest_energy,
    parse_gradient_block,
    run_pyscf_df,
)

# HF uses exact exchange → same known-bad DF JK derivative cases as hybrids.
_SP_ONLY = {"STO-3G", "6-31G"}
_HF_GRAD_SKIP = {("H2", "6-31G*"), ("H2O", "6-31G*"), ("HF", "6-31G*")}


def want_hf_gradient(molecule: str, basis_label: str, shell: str) -> bool:
    if shell != "spherical":
        return False
    if basis_label in _SP_ONLY:
        return False
    if (molecule, basis_label) in _HF_GRAD_SKIP:
        return False
    return True


def grad_rms(a, b):
    fa, fb = flatten_grad(a), flatten_grad(b)
    if fa is None or fb is None or len(fa) != len(fb) or not fa:
        return None
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(fa, fb)) / len(fa))


def run_one(xyz, label, bl, func, shell, tol_e, tol_g, check_grad):
    bs = BASIS_SETS[bl]
    basis = str(BASIS_DIR / bs["basis"])
    aux = str(BASIS_DIR / bs["aux"])
    atoms = load_xyz(str(MOLECULES_DIR / xyz))
    tag = f"{label}/{func}/{bl}/{shell}"

    py = run_pyscf_df(
        atoms, basis, aux, func, shell=shell, compute_gradient=check_grad,
    )
    if not py.get("ok"):
        return tag, False, None, None, f"PySCF {py.get('error')}"

    cmd = [
        str(EXE), "--xyz", str(MOLECULES_DIR / xyz),
        "--basis", basis, "--aux-basis", aux,
        "--functional", func, f"--{shell}", "--quiet",
        "--max-iter", "250",
    ]
    if check_grad:
        cmd.append("--analytic-gradient")
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + "\n" + r.stderr
    cu, conv = parse_cuest_energy(out)
    if cu is None or r.returncode != 0 or not conv:
        return tag, False, None, None, f"cuEST rc={r.returncode} conv={conv}"

    dE = cu - py["energy"]
    energy_ok = abs(dE) <= tol_e
    g_rms = None
    grad_ok = True
    note = ""
    if check_grad:
        cg = parse_gradient_block(out, "Analytical Gradient")
        if cg is not None:
            cg = as_grad_nx3(cg)
        g_rms = grad_rms(cg, py.get("gradient_ha_bohr"))
        if g_rms is None:
            if "Analytic gradient failed" in out:
                note = " gSKIP"
            else:
                grad_ok = False
                note = " gMISSING"
        else:
            grad_ok = g_rms < tol_g
            note = f" gRMS={g_rms:.2e}"

    ok = energy_ok and grad_ok
    return tag, ok, dE, g_rms, note


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tolerance-ha", type=float, default=1e-3)
    ap.add_argument("--grad-tol", type=float, default=5e-4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--compare-dft", action="store_true",
                    help="Also run PBE and WB97X on the same cells")
    ap.add_argument("--molecule", type=str)
    ap.add_argument("--basis", action="append", dest="bases")
    args = ap.parse_args()

    if not EXE.is_file():
        print(f"Missing binary: {EXE}", file=sys.stderr)
        return 1

    if args.quick:
        mols = [("h2o.xyz", "H2O"), ("n2.xyz", "N2"), ("ch4.xyz", "CH4")]
        bases = ["def2SVP", "6-31G*"]
    else:
        mols = list(MOLECULES)
        bases = args.bases or [
            "STO-3G", "6-31G", "6-31G*", "def2SVP", "def2TZVP", "cc-pVDZ",
        ]
    if args.molecule:
        mols = [(x, l) for x, l in mols if l == args.molecule]

    funcs = ["HF"]
    if args.compare_dft:
        funcs += ["PBE", "WB97X"]

    # Collect |dE| by functional for the summary
    by_func = {f: [] for f in funcs}
    n_pass = n_fail = 0

    for xyz, label in mols:
        for bl in bases:
            if bl not in BASIS_SETS:
                continue
            for shell in ("spherical", "cartesian"):
                for func in funcs:
                    check_grad = (
                        func == "HF"
                        and want_hf_gradient(label, bl, shell)
                    ) or (
                        func != "HF"
                        and shell == "spherical"
                        and not (func != "PBE" and bl in _SP_ONLY)
                        and (label, bl) not in _HF_GRAD_SKIP
                    )
                    # For DFT compare, skip grads (energy isolation is the point)
                    if args.compare_dft and func != "HF":
                        check_grad = False

                    tag, ok, dE, g_rms, note = run_one(
                        xyz, label, bl, func, shell,
                        args.tolerance_ha, args.grad_tol, check_grad,
                    )
                    if dE is None and not ok:
                        print(f"FAIL  {tag}: {note}")
                        n_fail += 1
                        continue
                    status = "PASS" if ok else "FAIL"
                    if ok:
                        n_pass += 1
                        by_func[func].append(abs(dE) * 1000.0)
                    else:
                        n_fail += 1
                    gnote = note if note.startswith(" ") or note.startswith("g") else (
                        f" {note}" if note else ""
                    )
                    print(
                        f"{status}  {tag}: dE={dE*1000:+.4f} mHa{gnote}"
                    )

    print(f"\n{n_pass} PASS / {n_fail} FAIL")
    print("\n|ΔE| summary (mHa) — DF residuals (HF has no XC grid):")
    for f in funcs:
        vals = by_func[f]
        if not vals:
            continue
        vals = sorted(vals)
        mean = sum(vals) / len(vals)
        med = vals[len(vals) // 2]
        print(
            f"  {f:8s}  n={len(vals):3d}  mean={mean:.4f}  "
            f"median={med:.4f}  max={vals[-1]:.4f}"
        )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
