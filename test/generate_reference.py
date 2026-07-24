#!/usr/bin/env python3
"""
Generate PySCF-DF reference energies, SCF iteration counts, and gradients.

Default matrix: closed-shell × {HF, PBE, WB97X} × all bases × {spherical, cartesian},
plus UKS OH × {HF, PBE, PBE0, WB97X} × UKS bases.
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

# XC grid level for the PySCF reference. Level 3 (PySCF's default) is NOT
# converged: it sits ~1e-5 Ha from the grid limit for a range-separated
# functional and ~1e-6 for a GGA, which put a floor under how tight the
# validation tolerance could be.
#
# Level 5 is already converged to ~2e-9 against level 6 for LIGHT elements,
# which is why 7 was chosen originally for headroom. That check was never
# redone for heavy elements, though, and it turns out level 7 is NOT converged
# there: for Br2/PBE the level-7 energy is 1.1e-7 Ha off cuEST's own
# independently grid-converged answer, while level 9 lands within 1.2e-8 (a
# ~10x improvement); I2/PBE shows the same pattern (4.5e-7 -> 1.7e-7). Rather
# than special-case heavy elements, 9 is used for the whole reference set.
DEFAULT_GRID_LEVEL = 9

# cuEST cuestDFSymmetricDerivativeCompute throws on hybrids for SP-only
# orbital bases (STO-3G, 6-31G). Skip storing grads for those refs.
# Also skip hybrid + 6-31G* for H2 (no D on H ⇒ SP-equivalent, wrong grads)
# and H2O/HF (library throws CUEST_STATUS_EXCEPTION on DF JK derivative).
_HYBRID_FUNCS = {
    "HF",  # 100% exact exchange — same SP-only DF JK derivative failures
    "B3LYP", "B3LYP5", "PBE0", "CAM-B3LYP", "WB97X", "WB97X-V", "WB97M-V",
    "HSE06", "M06", "M06-2X", "LC-WPBE", "LC-WPBEH",
}
_SP_ONLY_BASES = {"STO-3G", "6-31G"}
_HYBRID_GRAD_SKIP = {
    ("H2", "6-31G*"),   # no D on H ⇒ SP-equivalent; wrong grads
    ("H2O", "6-31G*"),  # DF JK derivative throws
    ("HF", "6-31G*"),   # DF JK derivative throws
    ("OH", "6-31G*"),   # DF JK derivative throws (UKS)
}


def want_gradient(config) -> bool:
    if config.get("shell", "spherical") != "spherical":
        return False
    func = config["functional"]
    basis = config["basis_label"]
    if func in _HYBRID_FUNCS and basis in _SP_ONLY_BASES:
        return False
    if func in _HYBRID_FUNCS and (config["molecule"], basis) in _HYBRID_GRAD_SKIP:
        return False
    return True


def ref_key(r):
    """Identity for a reference entry.

    Includes DF flag + aux label so density-fitted and exact-integral refs
    can coexist for the same molecule/functional/orbital basis/shell.
    Multiplicity distinguishes RKS vs UKS entries.
    """
    df = bool(r.get("density_fitting", True))
    aux = r.get("aux_basis_label") if df else None
    return (
        r["molecule"],
        r["functional"],
        r["basis_label"],
        r.get("shell", "spherical"),
        int(r.get("multiplicity", 1)),
        df,
        aux,
    )


def run_pyscf(config, grid_level=DEFAULT_GRID_LEVEL):
    atoms = load_xyz(config["xyz"])
    start = time.time()
    shell = config.get("shell", "spherical")
    spin = int(config.get("spin", 0))
    r = run_pyscf_df(
        atoms, config["basis"], config["aux_basis"], config["functional"],
        charge=int(config.get("charge", 0)),
        spin=spin,
        grid_level=grid_level,
        shell=shell,
        compute_gradient=want_gradient(config),
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
    p.add_argument("--grid-level", type=int, default=DEFAULT_GRID_LEVEL,
                   help=f"PySCF XC grid level (default {DEFAULT_GRID_LEVEL}; "
                        "3 is PySCF's default but is NOT grid-converged)")
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
            r = run_pyscf(config, grid_level=args.grid_level)
        except Exception as e:
            r = {"error": str(e)}

        key = ref_key({
            "molecule": config["molecule"],
            "functional": config["functional"],
            "basis_label": config["basis_label"],
            "shell": config["shell"],
            "multiplicity": config.get("multiplicity", 1),
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
                "multiplicity": int(config.get("multiplicity", 1)),
                "uks": bool(config.get("uks", False)),
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
                int(r.get("multiplicity", 1)),
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
