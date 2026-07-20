#!/usr/bin/env python3
"""
Smoke test: cuEST DFT vs PySCF-DF for H2O (PBE/def2-SVP).

Usage:
    python3 test/test_water.py
    python3 test/test_water.py --functional PBE0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BASIS_DIR, EXE, MOLECULES_DIR, parse_cuest_energy, run_cuest_cmd, run_pyscf_df,
)


def main():
    parser = argparse.ArgumentParser(description="Compare cuEST DFT with PySCF-DF")
    parser.add_argument("--functional", default="PBE")
    parser.add_argument("--xyz", default=str(MOLECULES_DIR / "h2o.xyz"))
    parser.add_argument("--basis-file", default=str(BASIS_DIR / "def2-svp.json"))
    parser.add_argument("--aux-basis-file",
                        default=str(BASIS_DIR / "def2-universal-jkfit.json"))
    parser.add_argument("--cuest-binary", default=str(EXE))
    args = parser.parse_args()

    print("=" * 60)
    print("  cuEST DFT vs PySCF-DF Validation")
    print("=" * 60)
    print(f"  Functional: {args.functional}")
    print(f"  XYZ: {args.xyz}")
    print(f"  Basis: {args.basis_file}")
    print(f"  Aux:   {args.aux_basis_file}")
    print()

    if not Path(args.cuest_binary).exists():
        print(f"ERROR: cuEST binary not found: {args.cuest_binary}")
        return 1
    for p in (args.xyz, args.basis_file, args.aux_basis_file):
        if not Path(p).exists():
            print(f"ERROR: missing file: {p}")
            return 1

    from common import load_xyz
    atoms = load_xyz(args.xyz)
    print(f"  Molecule: {len(atoms)} atoms")
    for sym, (x, y, z) in atoms:
        print(f"    {sym:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")

    print(f"\n--- PySCF-DF ({args.functional}) ---")
    try:
        pyscf_r = run_pyscf_df(
            atoms, args.basis_file, args.aux_basis_file, args.functional
        )
    except ImportError:
        print("  PySCF not available")
        return 1
    except Exception as e:
        print(f"  PySCF error: {e}")
        return 1

    if not pyscf_r["ok"]:
        print(f"  PySCF failed: {pyscf_r.get('error')}")
        return 1

    e_pyscf = pyscf_r["energy"]
    print(f"  PySCF total energy: {e_pyscf:.12f} Ha")
    print(f"  HOMO: {pyscf_r['homo']:.8f} Ha  LUMO: {pyscf_r['lumo']:.8f} Ha")

    print(f"\n--- cuEST ({args.functional}) ---")
    cmd = [
        args.cuest_binary,
        "--xyz", args.xyz,
        "--basis", args.basis_file,
        "--aux-basis", args.aux_basis_file,
        "--functional", args.functional,
        "--print-mos",
    ]
    r = run_cuest_cmd(cmd, timeout=300)
    if not r["ok"]:
        print("cuEST binary failed:")
        print(r.get("stderr", ""))
        print(r.get("stdout", "")[-2000:])
        return 1

    e_cuest, converged = parse_cuest_energy(r["stdout"])
    if e_cuest is None or not converged:
        print("Could not parse finite converged energy from cuEST output.")
        print(r["stdout"][-2000:])
        return 1

    print(f"  cuEST total energy: {e_cuest:.12f} Ha")

    diff = abs(e_pyscf - e_cuest)
    print(f"\n{'=' * 60}")
    print(f"  Energy difference: {diff:.10f} Ha ({diff * 27.2114:.6f} eV)")
    if diff < 1e-4:
        print("  PASS: Energies agree within 0.1 mHa")
        rc = 0
    else:
        print("  FAIL: Energy difference >= 0.1 mHa")
        rc = 1
    print(f"{'=' * 60}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
