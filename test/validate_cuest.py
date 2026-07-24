#!/usr/bin/env python3
"""
Validate cuEST DFT against a PySCF reference file.

Compares energy and SCF iteration count. Analytic gradients are compared when
the reference provides them (spherical-orbital refs). Cartesian refs are
energy-only.

Usage:
    python3 test/generate_reference.py
    python3 test/validate_cuest.py
    python3 test/validate_cuest.py --molecule H2O --shell cartesian
"""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BASIS_DIR, BASIS_SETS, EXE, LRC_FUNCTIONALS, MOLECULES_DIR, PROJ_DIR,
    as_grad_nx3, aux_json_for, aux_json_from_label, aux_label_for, flatten_grad,
    is_finite, parse_cuest_energy, parse_cuest_scf_iterations, parse_gradient_block,
)

REFERENCE_FILE = PROJ_DIR / "test" / "reference.json"
RESULTS_FILE = PROJ_DIR / "test" / "cuest_results.json"


def run_cuest(config, timeout=600, radial_pts=150, angular_pts=770):
    """Run cuEST DFT and return parsed results.

    Requests an analytic gradient only when the config asks for it (spherical
    refs). Cartesian validation is energy-only.
    """
    aux_path = config.get("aux_basis") or str(BASIS_DIR / aux_json_for(config["basis_label"]))
    shell = config.get("shell", "spherical")
    want_grad = config.get("check_gradient", shell == "spherical")
    multiplicity = int(config.get("multiplicity", 1))
    # Tight and uniform regardless of multiplicity: the old multiplicity != 1
    # branch loosened both thresholds by 10x to work around independent-channel
    # DIIS oscillating on open-shell systems (OH needed up to 251 iterations at
    # 1e-8). Joint DIIS (one Pulay coefficient set over both spin channels,
    # matching PySCF) fixed that at the root — OH now converges in single
    # digits to low tens of iterations at these same tight thresholds — so the
    # workaround no longer earns its keep and only hid real convergence issues.
    cmd = [
        str(EXE),
        "--xyz", config["xyz"],
        "--basis", config["basis"],
        "--aux-basis", aux_path,
        "--functional", config["functional"],
        "--multiplicity", str(multiplicity),
        "--max-iter", "400",
        "--conv-thresh", "1e-7",
        "--energy-conv", "1e-10",
        # cuEST's own default grid (75 radial / 302 angular) is NOT converged
        # to the tolerances used below: on H2O/PBE/def2SVP it alone accounts
        # for essentially the entire ~1.5e-6 Ha gap against a grid-7 PySCF
        # reference, which vanishes (down to ~3e-8) once the grid is
        # tightened. The references are grid-converged (see
        # generate_reference.py); this side has to match or every non-LRC
        # comparison is dominated by our own grid truncation, not physics.
        "--radial-pts", str(radial_pts),
        "--angular-pts", str(angular_pts),
        "--quiet",
        "--cartesian" if shell == "cartesian" else "--spherical",
    ]
    if want_grad:
        cmd.append("--analytic-gradient")

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env=dict(os.environ),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "wall_s": timeout}
    except Exception as e:
        return {"ok": False, "error": str(e), "wall_s": time.time() - start}

    elapsed = time.time() - start
    text = result.stdout + "\n" + result.stderr

    if result.returncode != 0:
        return {"ok": False, "error": f"exit {result.returncode}",
                "stderr": "\n".join(result.stderr.splitlines()[-5:]),
                "wall_s": elapsed}

    energy, converged = parse_cuest_energy(text)
    if energy is None:
        return {"ok": False, "error": "could not parse energy or non-finite", "wall_s": elapsed}
    # Require real convergence for both RKS and UKS. The UKS soft-accept this
    # used to have (finite energy but no convergence flag) was working around
    # independent-channel DIIS oscillating near the minimum on open-shell
    # systems; joint DIIS fixed that, so a non-converged UKS run now signals a
    # real problem rather than an expected rough edge.
    if not converged:
        return {"ok": False, "error": "SCF did not converge", "energy_ha": energy,
                "converged": False, "wall_s": elapsed}

    scf_iterations = parse_cuest_scf_iterations(text)
    grad = parse_gradient_block(text, "Analytical Gradient")
    if grad is not None:
        grad = as_grad_nx3(grad)
        if not all(is_finite(x) for row in grad for x in row):
            grad = None

    return {
        "ok": True,
        "energy_ha": energy,
        "converged": bool(converged),
        "scf_iterations": scf_iterations,
        "gradient_ha_bohr": grad,
        "wall_s": elapsed,
    }


def grad_rms(a, b):
    fa, fb = flatten_grad(a), flatten_grad(b)
    if fa is None or fb is None or len(fa) != len(fb) or not fa:
        return None
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(fa, fb)) / len(fa))


def main():
    import argparse
    p = argparse.ArgumentParser(description="Validate cuEST against PySCF reference")
    p.add_argument("--reference", type=str, default=str(REFERENCE_FILE))
    p.add_argument("--molecule", type=str)
    p.add_argument("--functional", type=str)
    p.add_argument("--basis", type=str)
    p.add_argument("--shell", type=str, choices=["spherical", "cartesian"])
    p.set_defaults(density_fitting=None)
    p.add_argument("--df", dest="density_fitting", action="store_const", const=True,
                   help="Only density-fitted references")
    p.add_argument("--exact", dest="density_fitting", action="store_const", const=False,
                   help="Only non-DF (exact integral) references")
    p.add_argument("--tolerance-ha", type=float, default=1e-7,
                   help="Max |ΔE| in Ha for non-LRC functionals (default: 1e-7)")
    p.add_argument("--grad-tol", type=float, default=1e-5,
                   help="Max RMS |Δg| in Ha/bohr for non-LRC functionals "
                        "(default: 1e-5)")
    p.add_argument("--lrc-tolerance-ha", type=float, default=3e-4,
                   help="Max |ΔE| in Ha for LRC_FUNCTIONALS (default: 3e-4; "
                        "see common.LRC_FUNCTIONALS for why these need a floor "
                        "-- cc-pvtz-jkfit is a further, larger outlier within "
                        "this group at ~1.7e-4)")
    p.add_argument("--lrc-grad-tol", type=float, default=1e-4,
                   help="Max RMS |Δg| in Ha/bohr for LRC_FUNCTIONALS (default: 1e-4)")
    p.add_argument("--radial-pts", type=int, default=150,
                   help="cuEST radial grid points (default: 150; must be "
                        "converged to match the grid-7 PySCF references. "
                        "The 99/590 'standard' combo fails (H2O/def2QZVPP is "
                        "off by ~1.2e-7); the missing ingredient is angular, "
                        "not radial -- 200/590 still leaves ~1.1e-7 on CH4, "
                        "so 590 is simply insufficient regardless of radial "
                        "points. Angular=770 fixes it at any radial>=150 "
                        "(worst residual ~2.4e-8 across 13 molecule/basis "
                        "combos tested); 200/770 gives no further improvement "
                        "over 150/770, so 150 is used to save cost)")
    p.add_argument("--angular-pts", type=int, default=770,
                   help="cuEST angular (Lebedev) grid points (default: 770)")
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

    refs = references
    if args.molecule:
        refs = [r for r in refs if r["molecule"] == args.molecule]
    if args.functional:
        refs = [r for r in refs if r["functional"] == args.functional]
    if args.basis:
        refs = [r for r in refs if r["basis_label"] == args.basis]
    if args.shell:
        refs = [r for r in refs if r.get("shell", "spherical") == args.shell]
    if args.density_fitting is True:
        refs = [r for r in refs if r.get("density_fitting", True)]
    elif args.density_fitting is False:
        refs = [r for r in refs if not r.get("density_fitting", True)]

    def get_xyz_path(mol_label):
        name_map = {
            "H2O": "h2o.xyz", "NH3": "nh3.xyz", "H2": "h2.xyz",
            "HF": "hf.xyz", "N2": "n2.xyz", "CO2": "co2.xyz",
            "CH4": "ch4.xyz", "C2H4": "c2h4.xyz", "BH3": "bh3.xyz",
            "SO2": "so2.xyz", "Br2": "br2.xyz", "I2": "i2.xyz",
            "CH2I2": "ch2i2.xyz", "OH": "oh.xyz",
        }
        fname = name_map.get(mol_label, f"{mol_label.lower()}.xyz")
        return str(MOLECULES_DIR / fname)

    testable = []
    for ref in refs:
        if "energy_ha" not in ref:
            continue
        xyz = get_xyz_path(ref["molecule"])
        bs = BASIS_SETS.get(ref["basis_label"])
        if not bs:
            continue
        basis = str(BASIS_DIR / bs["basis"])
        df = bool(ref.get("density_fitting", True))
        if not df:
            continue
        aux_label = ref.get("aux_basis_label") or aux_label_for(ref["basis_label"])
        aux = str(BASIS_DIR / aux_json_from_label(aux_label))
        if not Path(xyz).exists() or not Path(basis).exists() or not Path(aux).exists():
            continue
        testable.append({
            "molecule": ref["molecule"],
            "functional": ref["functional"],
            "basis_label": ref["basis_label"],
            "shell": ref.get("shell", "spherical"),
            "multiplicity": int(ref.get("multiplicity", 1)),
            "density_fitting": df,
            "aux_basis_label": aux_label,
            "xyz": xyz,
            "basis": basis,
            "aux_basis": aux,
            "check_gradient": ref.get("gradient_ha_bohr") is not None,
            "ref": ref,
        })

    if not testable:
        print("No configs with references match filters.")
        sys.exit(1)

    print(f"\nTesting {len(testable)} configurations...\n")

    results = []
    n_pass = n_fail = n_error = 0

    for i, config in enumerate(testable):
        ref = config["ref"]
        label = (f"{config['molecule']}/{config['functional']}/"
                 f"{config['basis_label']}/{config['shell']}"
                 f"/df:{config['aux_basis_label']}")
        print(f"[{i+1}/{len(testable)}] {label} ...", end=" ", flush=True)
        cuest_r = run_cuest(config, radial_pts=args.radial_pts,
                           angular_pts=args.angular_pts)

        if not cuest_r["ok"]:
            print(f"FAILED: {cuest_r['error']}")
            results.append({**{k: config[k] for k in
                               ("molecule", "functional", "basis_label", "shell")},
                            "cuest_ok": False, "cuest_error": cuest_r["error"],
                            "ref_energy_ha": ref["energy_ha"]})
            n_error += 1
            continue

        is_lrc = config["functional"] in LRC_FUNCTIONALS
        tol_e = args.lrc_tolerance_ha if is_lrc else args.tolerance_ha
        tol_g = args.lrc_grad_tol if is_lrc else args.grad_tol

        diff = cuest_r["energy_ha"] - ref["energy_ha"]
        energy_ok = abs(diff) < tol_e

        g_rms = grad_rms(cuest_r.get("gradient_ha_bohr"),
                         ref.get("gradient_ha_bohr"))
        grad_ok = g_rms is not None and g_rms < tol_g
        if ref.get("gradient_ha_bohr") is None:
            grad_ok = True
            g_rms = None

        ref_iters = ref.get("scf_iterations")
        cu_iters = cuest_r.get("scf_iterations")
        iter_delta = (None if ref_iters is None or cu_iters is None
                      else cu_iters - ref_iters)

        passed = energy_ok and grad_ok
        status = "PASS" if passed else "FAIL"
        g_str = f" gRMS={g_rms:.2e}" if g_rms is not None else ""
        i_str = (f" iters={cu_iters}"
                 + (f"(Δ{iter_delta:+d})" if iter_delta is not None else ""))
        print(f"{cuest_r['energy_ha']:.10f} Ha  ΔE={diff:+.3e}{g_str}  "
              f"{i_str}  [{status}]")

        if passed:
            n_pass += 1
        else:
            n_fail += 1

        results.append({
            "molecule": config["molecule"],
            "functional": config["functional"],
            "basis_label": config["basis_label"],
            "shell": config["shell"],
            "density_fitting": config.get("density_fitting", True),
            "aux_basis_label": config.get("aux_basis_label"),
            "ref_energy_ha": ref["energy_ha"],
            "ref_scf_iterations": ref_iters,
            "ref_gradient_ha_bohr": ref.get("gradient_ha_bohr"),
            "cuest_energy_ha": cuest_r["energy_ha"],
            "cuest_scf_iterations": cu_iters,
            "cuest_gradient_ha_bohr": cuest_r.get("gradient_ha_bohr"),
            "cuest_converged": cuest_r["converged"],
            "cuest_wall_s": cuest_r["wall_s"],
            "energy_diff_ha": diff,
            "grad_rms_ha_bohr": g_rms,
            "lrc_tolerance_applied": is_lrc,
            "scf_iter_delta": iter_delta,
            "energy_ok": energy_ok,
            "grad_ok": grad_ok,
            "ok": passed,
        })

    print(f"\n{'='*60}")
    print(f"  RESULTS: {n_pass} PASS, {n_fail} FAIL, {n_error} ERROR  "
          f"(of {len(testable)})")
    print(f"  Energy tol: {args.tolerance_ha} Ha | Grad RMS tol: {args.grad_tol}")
    print(f"  LRC functionals ({', '.join(sorted(LRC_FUNCTIONALS))}):")
    print(f"    Energy tol: {args.lrc_tolerance_ha} Ha | Grad RMS tol: {args.lrc_grad_tol}")
    print(f"{'='*60}")

    if n_fail > 0:
        print("\nFailures:")
        for r in results:
            if r.get("ok") is False:
                print(f"  {r['molecule']}/{r['functional']}/{r['basis_label']}/"
                      f"{r['shell']}: ΔE={r.get('energy_diff_ha'):+.3e} "
                      f"gRMS={r.get('grad_rms_ha_bohr')}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULTS_FILE}")

    sys.exit(0 if (n_fail == 0 and n_error == 0) else 1)


if __name__ == "__main__":
    main()
