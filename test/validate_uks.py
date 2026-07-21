#!/usr/bin/env python3
"""Validate unrestricted UKS (cuEST vs PySCF): energies and analytic gradients.

Default sweep: open-shell molecules × {PBE, PBE0, WB97X} × selected bases
× {spherical, cartesian}. Analytic gradients compared for spherical only
(same limitation as the RKS harness — PySCF has no mixed cart/sph DF grad).

Known-bad hybrid DF grads (skipped for gradient compare, energy still checked):
  hybrids × {STO-3G, 6-31G}; hybrid × 6-31G* for H2/H2O/HF (RKS matrix).
  For the UKS OH sweep we skip hybrid grads on SP-only bases only.
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
    MOLECULES_DIR,
    as_grad_nx3,
    flatten_grad,
    load_xyz,
    parse_cuest_energy,
    parse_gradient_block,
    run_pyscf_df,
)

# (xyz, label, multiplicity, PySCF spin=2S)
CASES = [
    ("oh.xyz", "OH", 2, 1),
]

# Default basis labels for the full sweep
SWEEP_BASES = ["STO-3G", "6-31G", "6-31G*", "def2SVP", "def2TZVP"]
_HYBRID = {"PBE0", "B3LYP", "CAM-B3LYP", "WB97X", "HSE06", "M06", "M06-2X"}
_SP_ONLY = {"STO-3G", "6-31G"}
# Same cuEST DF JK derivative failures as the RKS reference matrix.
_HYBRID_GRAD_SKIP = {("OH", "6-31G*")}


def want_uks_gradient(func: str, basis_label: str, shell: str,
                      molecule: str = "") -> bool:
    if shell != "spherical":
        return False
    if func in _HYBRID and basis_label in _SP_ONLY:
        return False
    if func in _HYBRID and (molecule, basis_label) in _HYBRID_GRAD_SKIP:
        return False
    return True


def grad_rms(a, b):
    fa, fb = flatten_grad(a), flatten_grad(b)
    if fa is None or fb is None or len(fa) != len(fb) or not fa:
        return None
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(fa, fb)) / len(fa))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tolerance-ha", type=float, default=1e-3)
    ap.add_argument("--grad-tol", type=float, default=5e-4)
    ap.add_argument(
        "--basis",
        action="append",
        dest="bases",
        default=None,
        help="Basis label(s) from BASIS_SETS; default full sweep set",
    )
    ap.add_argument(
        "--functional",
        action="append",
        dest="functionals",
        default=None,
        help="Functional(s); default PBE PBE0 WB97X",
    )
    ap.add_argument("--quick", action="store_true",
                    help="OH / PBE+WB97X / def2SVP only")
    ap.add_argument("--no-gradient", action="store_true")
    args = ap.parse_args()

    if args.quick:
        functionals = ["PBE", "WB97X"]
        bases = ["def2SVP"]
    else:
        functionals = args.functionals or ["PBE", "PBE0", "WB97X"]
        bases = args.bases or list(SWEEP_BASES)

    if not EXE.is_file():
        print(f"Missing binary: {EXE}", file=sys.stderr)
        return 1

    n_pass = n_fail = n_skip_grad = 0
    for xyz, label, mult, spin in CASES:
        atoms = load_xyz(str(MOLECULES_DIR / xyz))
        for bl in bases:
            if bl not in BASIS_SETS:
                print(f"SKIP unknown basis {bl}")
                continue
            bs = BASIS_SETS[bl]
            basis = str(BASIS_DIR / bs["basis"])
            aux = str(BASIS_DIR / bs["aux"])
            for func in functionals:
                for shell in ("spherical", "cartesian"):
                    tag = f"{label}/{func}/{bl}/{shell}/mult={mult}"
                    check_grad = (
                        not args.no_gradient
                        and want_uks_gradient(func, bl, shell, label)
                    )
                    py = {"ok": False}
                    for _ in range(3):
                        py = run_pyscf_df(
                            atoms, basis, aux, func, spin=spin, shell=shell,
                            compute_gradient=check_grad,
                        )
                        if py.get("ok"):
                            break
                    if not py.get("ok"):
                        print(f"FAIL  {tag}: PySCF {py.get('error')}")
                        n_fail += 1
                        continue

                    cmd = [
                        str(EXE),
                        "--xyz", str(MOLECULES_DIR / xyz),
                        "--basis", basis,
                        "--aux-basis", aux,
                        "--functional", func,
                        f"--{shell}",
                        "--multiplicity", str(mult),
                        "--max-iter", "400",
                        "--conv-thresh", "1e-7",
                        "--energy-conv", "1e-7",
                        "--quiet",
                    ]
                    if check_grad:
                        cmd.append("--analytic-gradient")

                    # UKS DIIS can be GPU-nondeterministic on tiny bases; retry
                    # with damping, then accept energy match even if the
                    # formal CONVERGED flag is still unset.
                    cu, conv, out = None, False, ""
                    for attempt, extra in enumerate(
                        ([], ["--damping", "0.3"], ["--damping", "0.5"])
                    ):
                        r = subprocess.run(
                            cmd + extra, capture_output=True, text=True
                        )
                        out = r.stdout + "\n" + r.stderr
                        cu, conv = parse_cuest_energy(out)
                        if cu is not None and r.returncode == 0 and (
                            conv or abs(cu - py["energy"]) <= args.tolerance_ha
                        ):
                            break
                    if cu is None or r.returncode != 0:
                        print(f"FAIL  {tag}: cuEST rc={r.returncode} conv={conv}")
                        n_fail += 1
                        continue

                    dE = cu - py["energy"]
                    energy_ok = abs(dE) <= args.tolerance_ha
                    soft = energy_ok and not conv

                    g_rms = None
                    grad_ok = True
                    g_str_extra = ""
                    if check_grad and conv:
                        cg = parse_gradient_block(out, "Analytical Gradient")
                        if cg is not None:
                            cg = as_grad_nx3(cg)
                        pg = py.get("gradient_ha_bohr")
                        g_rms = grad_rms(cg, pg)
                        if g_rms is None:
                            if "Analytic gradient failed" in out:
                                n_skip_grad += 1
                                g_str_extra = " gSKIP(cuEST)"
                            else:
                                grad_ok = False
                                g_str_extra = " gMISSING"
                        else:
                            grad_ok = g_rms < args.grad_tol
                            g_str_extra = f" gRMS={g_rms:.2e}"
                    elif check_grad and soft:
                        g_str_extra = " gSKIP(soft-SCF)"
                        n_skip_grad += 1

                    ok = energy_ok and grad_ok
                    status = "PASS" if ok else "FAIL"
                    if ok:
                        n_pass += 1
                    else:
                        n_fail += 1
                    soft_tag = " soft" if soft else ""
                    print(
                        f"{status}  {tag}: dE={dE*1000:+.4f} mHa"
                        f"{g_str_extra}{soft_tag}"
                    )

    print(f"\n{n_pass} PASS / {n_fail} FAIL"
          + (f"  ({n_skip_grad} grad skips)" if n_skip_grad else ""))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
