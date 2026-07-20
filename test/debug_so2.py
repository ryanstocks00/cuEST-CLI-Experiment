#!/usr/bin/env python3
"""
Piece-by-piece SO2 energy comparison: cuEST vs PySCF-DF.

Compares geometry, Enuc, electron/basis counts, and energy components
(E_HCORE, E_J, E_XC, E_TOT) under matched DF / grid settings.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BASIS_DIR, EXE, MOLECULES_DIR, load_xyz, run_pyscf_df, bse_json_to_pyscf,
)

BOHR = 0.52917721092  # PySCF / cuEST shared Bohr radius in Angstrom


def parse_components(text: str) -> dict:
    comps = {}
    in_block = False
    for line in text.splitlines():
        if "=== ENERGY_COMPONENTS ===" in line:
            in_block = True
            continue
        if "=== END_ENERGY_COMPONENTS ===" in line:
            break
        if not in_block:
            continue
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0]
            try:
                comps[key] = float(parts[1]) if "." in parts[1] or "e" in parts[1].lower() \
                    else int(parts[1])
            except ValueError:
                comps[key] = parts[1]
    return comps


def run_cuest(xyz, basis, aux, functional="PBE", radial=75, angular=302):
    cmd = [
        str(EXE), "--xyz", str(xyz), "--basis", str(basis),
        "--aux-basis", str(aux), "--functional", functional,
        "--radial-pts", str(radial), "--angular-pts", str(angular),
        "--max-iter", "150", "--conv-thresh", "1e-8", "--energy-conv", "1e-8",
        "--quiet",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    text = r.stdout + "\n" + r.stderr
    comps = parse_components(text)
    comps["_returncode"] = r.returncode
    comps["_stdout_tail"] = "\n".join(text.splitlines()[-40:])
    return comps


def enuc_from_xyz(atoms):
    """Nuclear repulsion from XYZ (Angstrom) using Z_full (no ECP)."""
    from pyscf.data import elements
    zs = []
    coords = []
    for sym, xyz in atoms:
        z = elements.charge(sym)
        zs.append(z)
        coords.append(np.array(xyz) / BOHR)  # -> bohr
    enuc = 0.0
    for i in range(len(zs)):
        for j in range(i + 1, len(zs)):
            r = np.linalg.norm(coords[i] - coords[j])
            enuc += zs[i] * zs[j] / r
    return enuc, zs, np.array(coords)


def pyscf_decompose(atoms, basis_path, aux_path, functional="PBE",
                    grid_level=3, rad=None, ang=None):
    from pyscf import gto, dft, df
    from pyscf.dft import gen_grid

    basis_dict, ecp_data = bse_json_to_pyscf(basis_path)
    aux_dict, _ = bse_json_to_pyscf(aux_path)

    mol = gto.M(atom=atoms, basis=basis_dict, unit="Angstrom", verbose=0)
    mol.build()

    mf = dft.RKS(mol).density_fit()
    mf.with_df.auxbasis = aux_dict
    mf.xc = "pbe,pbe" if functional == "PBE" else functional.lower()
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    mf.init_guess = "1e"

    if rad is not None and ang is not None:
        # Match cuEST-style (radial, angular) Lebedev grid as closely as possible
        mf.grids = gen_grid.Grids(mol)
        mf.grids.atom_grid = (rad, ang)
        mf.grids.prune = None
    else:
        mf.grids.level = grid_level

    e_tot = mf.kernel()
    assert mf.converged

    dm = mf.make_rdm1()          # total density (closed shell)
    h1e = mf.get_hcore()
    s = mol.intor("int1e_ovlp")
    t = mol.intor("int1e_kin")
    v = mol.intor("int1e_nuc")

    # Coulomb from DF
    vj = mf.get_j(mol, dm)
    # XC
    ni = mf._numint
    nelec, exc, vxc = ni.nr_rks(mol, mf.grids, mf.xc, dm)
    # Energy pieces matching cuEST: D_alpha = dm/2
    # cuEST: E_HCORE = 2*Tr[Da*H] = Tr[Dtot*H]
    # cuEST: E_J     = 2*Tr[Da*J(Da)].  With J built from Da, J(Da)=0.5*J(Dtot),
    #         so 2*Tr[Da*J(Da)] = 2*Tr[(D/2)*(J/2)] = 0.5*Tr[D*J] = E_J_pyscf
    e_hcore = float(np.einsum("ij,ji", h1e, dm))
    e_j = 0.5 * float(np.einsum("ij,ji", vj, dm))
    e_xc = float(exc)
    e_nuc = float(mol.energy_nuc())
    e_kin = float(np.einsum("ij,ji", t, dm))
    e_ne = float(np.einsum("ij,ji", v, dm))
    tr_ds = 0.5 * float(np.einsum("ij,ji", s, dm))  # Tr[Da*S] = nocc

    # Orbital energies
    nocc = mol.nelectron // 2
    mo = mf.mo_energy

    return {
        "E_TOT": float(e_tot),
        "E_NUC": e_nuc,
        "E_HCORE": e_hcore,
        "E_KIN": e_kin,
        "E_NE": e_ne,
        "E_J": e_j,
        "E_XC": e_xc,
        "E_ELEC": e_hcore + e_j + e_xc,
        "TR_DS": tr_ds,
        "NAO": mol.nao_nr(),
        "NOCC": nocc,
        "NELEC": mol.nelectron,
        "NAUX": mf.with_df.get_naoaux() if hasattr(mf.with_df, "get_naoaux") else None,
        "HOMO": float(mo[nocc - 1]),
        "LUMO": float(mo[nocc]),
        "S_eigs": np.linalg.eigvalsh(s),
        "grids_level": getattr(mf.grids, "level", None),
        "atom_grid": getattr(mf.grids, "atom_grid", None),
        "mf": mf,
        "mol": mol,
        "dm": dm,
    }


def fmt(v, w=16):
    if v is None:
        return f"{'N/A':>{w}}"
    if isinstance(v, (int, np.integer)):
        return f"{v:>{w}d}"
    return f"{v:>{w}.10f}"


def row(label, a, b, scale=1.0):
    if a is None or b is None:
        print(f"  {label:18s}  {fmt(a)}  {fmt(b)}  {'N/A':>14}")
        return
    da = float(a) - float(b)
    print(f"  {label:18s}  {fmt(a)}  {fmt(b)}  {da*scale:>+14.6e}")


def main():
    xyz = MOLECULES_DIR / "so2.xyz"
    basis = BASIS_DIR / "def2-svp.json"
    aux = BASIS_DIR / "def2-universal-jkfit.json"
    atoms = load_xyz(xyz)

    print("=" * 72)
    print("  SO2 detailed comparison: cuEST vs PySCF-DF")
    print("=" * 72)

    # --- Geometry / Enuc ---
    enuc_ref, zs, coords_bohr = enuc_from_xyz(atoms)
    print("\n[1] Geometry (Angstrom → Bohr with BOHR=0.52917721092)")
    for (sym, xyz_a), z, c in zip(atoms, zs, coords_bohr):
        print(f"  {sym:2s}  Z={z:2d}  Ang={xyz_a}  Bohr={tuple(c)}")
    print(f"  Reference Enuc from XYZ: {enuc_ref:.12f} Ha")

    # --- PySCF with several grid settings ---
    print("\n[2] PySCF-DF energy components")
    grids_to_try = [
        ("level=3 (default validate)", dict(grid_level=3)),
        ("level=5 (finer)", dict(grid_level=5)),
        ("(75,302) unpruned", dict(rad=75, ang=302)),
        ("(99,590) unpruned", dict(rad=99, ang=590)),
    ]
    pyscf_runs = {}
    for label, kw in grids_to_try:
        print(f"\n  --- PySCF {label} ---")
        r = pyscf_decompose(atoms, str(basis), str(aux), **kw)
        pyscf_runs[label] = r
        print(f"  NAO={r['NAO']} NELEC={r['NELEC']} NOCC={r['NOCC']} NAUX={r['NAUX']}")
        print(f"  E_NUC   {r['E_NUC']:.12f}")
        print(f"  E_KIN   {r['E_KIN']:.12f}")
        print(f"  E_NE    {r['E_NE']:.12f}")
        print(f"  E_HCORE {r['E_HCORE']:.12f}  (= E_KIN + E_NE)")
        print(f"  E_J     {r['E_J']:.12f}")
        print(f"  E_XC    {r['E_XC']:.12f}")
        print(f"  E_ELEC  {r['E_ELEC']:.12f}")
        print(f"  E_TOT   {r['E_TOT']:.12f}")
        print(f"  Tr[Da*S]{r['TR_DS']:.12f}")
        print(f"  HOMO/LUMO {r['HOMO']:.8f} / {r['LUMO']:.8f}")
        print(f"  S eig range [{r['S_eigs'][0]:.6e}, {r['S_eigs'][-1]:.6e}]")

    # --- cuEST ---
    print("\n[3] cuEST (75,302) energy components")
    if not EXE.exists():
        print(f"ERROR: missing binary {EXE}")
        return 1
    cu = run_cuest(xyz, basis, aux, radial=75, angular=302)
    if "E_TOT" not in cu:
        print("Failed to parse cuEST components:")
        print(cu.get("_stdout_tail", ""))
        return 1
    for k in ["NAO", "NELEC", "NOCC", "E_NUC", "E_HCORE", "E_J", "E_XC",
              "E_ELEC", "E_TOT", "TR_DS"]:
        print(f"  {k:8s} {cu[k]}")

    # --- Side-by-side vs best-matched PySCF ---
    print("\n[4] Side-by-side vs PySCF (75,302) unpruned")
    py = pyscf_runs["(75,302) unpruned"]
    print(f"  {'component':18s}  {'cuEST':>16}  {'PySCF':>16}  {'cuEST-PySCF':>14}")
    row("E_NUC", cu["E_NUC"], py["E_NUC"])
    row("E_HCORE", cu["E_HCORE"], py["E_HCORE"])
    row("E_J", cu["E_J"], py["E_J"])
    row("E_XC", cu["E_XC"], py["E_XC"])
    row("E_ELEC", cu["E_ELEC"], py["E_ELEC"])
    row("E_TOT", cu["E_TOT"], py["E_TOT"])
    row("Tr[Da*S]", cu["TR_DS"], py["TR_DS"])
    row("NAO", cu["NAO"], py["NAO"])
    row("NELEC", cu["NELEC"], py["NELEC"])

    print("\n[5] E_TOT vs all PySCF grids (mHa)")
    for label, r in pyscf_runs.items():
        d = (cu["E_TOT"] - r["E_TOT"]) * 1000
        print(f"  vs {label:28s}: Δ = {d:+.6f} mHa   PySCF={r['E_TOT']:.12f}")

    # --- Attribute the total error ---
    print("\n[6] Error budget (cuEST − PySCF@75,302) in mHa")
    for key in ["E_NUC", "E_HCORE", "E_J", "E_XC", "E_TOT"]:
        d = (cu[key] - py[key]) * 1000
        print(f"  Δ{key:8s} = {d:+10.6f} mHa")

    # --- Conventional (no DF) PySCF for reference ---
    print("\n[7] PySCF conventional (no DF) vs DF — isolates fitting error")
    from pyscf import gto, dft
    basis_dict, _ = bse_json_to_pyscf(str(basis))
    mol = gto.M(atom=atoms, basis=basis_dict, unit="Angstrom", verbose=0)
    mf4 = dft.RKS(mol)
    mf4.xc = "pbe,pbe"
    mf4.grids.atom_grid = (75, 302)
    mf4.grids.prune = None
    mf4.conv_tol = 1e-10
    mf4.init_guess = "1e"
    e4 = mf4.kernel()
    print(f"  PySCF no-DF (75,302): {e4:.12f} Ha")
    print(f"  PySCF DF    (75,302): {py['E_TOT']:.12f} Ha")
    print(f"  DF error (DF−4c):     {(py['E_TOT']-e4)*1000:+.6f} mHa")
    print(f"  cuEST − no-DF:        {(cu['E_TOT']-e4)*1000:+.6f} mHa")
    print(f"  cuEST − DF:           {(cu['E_TOT']-py['E_TOT'])*1000:+.6f} mHa")

    # --- Check S basis / shell counts ---
    print("\n[8] Basis composition")
    mol = py["mol"]
    print(f"  PySCF AO labels (first 20): {mol.ao_labels()[:20]}")
    print(f"  PySCF nbas={mol.nbas} nao={mol.nao}")

    print("\n" + "=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
