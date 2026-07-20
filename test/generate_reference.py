#!/usr/bin/env python3
"""
Generate PySCF-DF reference energies, SCF iteration counts, and gradients.

Default matrix: all molecules × {PBE, WB97X} × all bases × {spherical, cartesian}.
DF auxiliary bases are always spherical (matching cuEST).

Each reference entry records density_fitting=true and aux_basis_label so
exact-integral (non-DF) refs can be stored alongside later.

Gradients: analytic for spherical orbitals only. Cartesian-orbital refs are
energy (+ SCF iters) only — PySCF has no mixed cart/sph DF analytic gradient,
and finite-difference grads are not accurate enough for validation.

Usage:
    python3 test/generate_reference.py [--quick] [--merge] [--molecule]
        [--functional] [--basis] [--shell] [--output]
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    PROJ_DIR, build_energy_matrix, load_xyz, run_pyscf_df,
)

REFERENCE_FILE = PROJ_DIR / "test" / "reference.json"


def ref_key(r):
    """Identity for a reference entry.

    Includes DF flag + aux label so density-fitted and exact-integral refs
    can coexist for the same molecule/functional/orbital basis/shell.
    """
    df = bool(r.get("density_fitting", True))
    aux = r.get("aux_basis_label") if df else None
    return (
        r["molecule"],
        r["functional"],
        r["basis_label"],
        r.get("shell", "spherical"),
        df,
        aux,
    )


def run_pyscf(config):
    atoms = load_xyz(config["xyz"])
    start = time.time()
    shell = config.get("shell", "spherical")
    r = run_pyscf_df(
        atoms, config["basis"], config["aux_basis"], config["functional"],
        grid_level=3,
        shell=shell,
        compute_gradient=(shell == "spherical"),
    )
    elapsed = time.time() - start
    if not r["ok"]:
        return {"error": r.get("error", "PySCF failed"), "wall_s": elapsed}
    return {
        "energy_ha": r["energy"],
        "homo_ha": r["homo"],
        "lumo_ha": r["lumo"],
        "scf_iterations": r["scf_iterations"],
        "gradient_ha_bohr": r["gradient_ha_bohr"],
        "nao": r.get("nao"),
        "wall_s": elapsed,
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate PySCF reference data")
    p.add_argument("--quick", action="store_true",
                   help="H2O/PBE/def2SVP/spherical only")
    p.add_argument("--merge", action="store_true",
                   help="Merge into existing reference.json (replace matching keys)")
    p.add_argument("--molecule", type=str)
    p.add_argument("--functional", type=str)
    p.add_argument("--basis", type=str)
    p.add_argument("--shell", type=str, choices=["spherical", "cartesian"])
    p.add_argument("--output", type=str, default=str(REFERENCE_FILE))
    args = p.parse_args()

    matrix = build_energy_matrix(quick=args.quick)
    if args.molecule:
        matrix = [c for c in matrix if c["molecule"] == args.molecule]
    if args.functional:
        matrix = [c for c in matrix if c["functional"] == args.functional]
    if args.basis:
        matrix = [c for c in matrix if c["basis_label"] == args.basis]
    if args.shell:
        matrix = [c for c in matrix if c["shell"] == args.shell]
    if not matrix:
        print("No configurations match filters.")
        sys.exit(1)

    out_path = Path(args.output)
    by_key = {}
    if args.merge and out_path.exists():
        prev = json.loads(out_path.read_text())
        for r in prev.get("results", []):
            by_key[ref_key(r)] = r
        print(f"Merging into {len(by_key)} existing references")

    shells = sorted({c["shell"] for c in matrix})
    grad_note = ("energy only" if shells == ["cartesian"]
                 else "analytic grads for spherical; energy only for cartesian")
    print(f"Generating {len(matrix)} PySCF-DF references ({grad_note})...")
    print(f"  Bases:       {sorted({c['basis_label'] for c in matrix})}")
    print(f"  Functionals: {sorted({c['functional'] for c in matrix})}")
    print(f"  Shells:      {shells}")
    print(f"  Molecules:   {sorted({c['molecule'] for c in matrix})}")
    print(f"Output: {out_path}")
    n_ok, n_fail = 0, 0

    for i, config in enumerate(matrix):
        df = bool(config.get("density_fitting", True))
        aux_label = (config.get("aux_basis_label")
                     or Path(Path(config["aux_basis"]).name).stem)
        if not df:
            aux_label = None
        label = (f"{config['molecule']}/{config['functional']}/"
                 f"{config['basis_label']}/{config['shell']}")
        if df:
            label += f"/df:{aux_label}"
        else:
            label += "/exact"
        print(f"[{i+1}/{len(matrix)}] {label} ...", end=" ", flush=True)

        try:
            r = run_pyscf(config)
        except Exception as e:
            r = {"error": str(e)}

        key = ref_key({
            "molecule": config["molecule"],
            "functional": config["functional"],
            "basis_label": config["basis_label"],
            "shell": config["shell"],
            "density_fitting": df,
            "aux_basis_label": aux_label,
        })
        if "energy_ha" in r:
            print(f"{r['energy_ha']:.10f} Ha  iters={r['scf_iterations']} "
                  f"({r.get('wall_s', 0):.1f}s)")
            n_ok += 1
            by_key[key] = {
                "molecule": config["molecule"],
                "functional": config["functional"],
                "basis_label": config["basis_label"],
                "shell": config["shell"],
                "density_fitting": df,
                "aux_basis_label": aux_label,
                "energy_ha": r["energy_ha"],
                "homo_ha": r["homo_ha"],
                "lumo_ha": r["lumo_ha"],
                "scf_iterations": r["scf_iterations"],
                "gradient_ha_bohr": r["gradient_ha_bohr"],
                "nao": r.get("nao"),
            }
        else:
            print(f"FAILED: {r.get('error', 'unknown')}")
            n_fail += 1
            by_key.pop(key, None)

        results = sorted(
            by_key.values(),
            key=lambda r: (
                r["molecule"], r["functional"], r["basis_label"],
                r.get("shell", "spherical"),
                not bool(r.get("density_fitting", True)),
                r.get("aux_basis_label") or "",
            ),
        )
        with open(out_path, "w") as f:
            json.dump({
                "generated": datetime.now(timezone.utc).isoformat(),
                "results": results,
            }, f, indent=2)
            f.write("\n")

    print(f"\nDone: {n_ok} OK, {n_fail} failed  -> {out_path} "
          f"({len(by_key)} total refs)")
    sys.exit(0 if n_ok > 0 else 1)


if __name__ == "__main__":
    main()
