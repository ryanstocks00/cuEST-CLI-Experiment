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
MOLECULES_ROOT = PROJ_DIR / "data" / "molecules"
MOLECULES_DIR = MOLECULES_ROOT / "small"  # single-molecule smoke/validation set
BASIS_DIR = PROJ_DIR / "data" / "basis_sets"

# VV10 nonlocal-correlation grid level (see run_pyscf_df).
NLC_GRID_LEVEL = 5

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
    "HF": "hf",  # used only for name checks; HF uses scf.RHF/UHF below
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

# Full functional catalog (closed-shell RKS + DF)
FUNCTIONALS = [
    "HF",
    "PBE",
    "B3LYP",
    "PBE0",
    "CAM-B3LYP",
    "WB97X",
    "HSE06",
    "M06",
    "M06-2X",
]

# Default reference / energy-matrix functionals (closed-shell).
# HF isolates DF J/K (no XC grid); PBE/WB97X exercise grid + hybrids.
# WB97X-V/WB97M-V additionally exercise the VV10 nonlocal-correlation path.
REF_FUNCTIONALS = ["HF", "PBE", "WB97X", "WB97X-V", "WB97M-V"]
SHELL_TYPES = ("spherical", "cartesian")

# Open-shell UKS reference matrix (multiplicity > 1)
UKS_MOLECULES = [
    ("oh.xyz", "OH", 2, 1),  # xyz, label, multiplicity, PySCF spin=2S
]
UKS_REF_FUNCTIONALS = ["HF", "PBE", "PBE0", "WB97X", "WB97X-V", "WB97M-V"]
# Keep UKS bases moderate (OH is light; full Ahlrichs/Dunning matrix is enough)
UKS_BASIS_LABELS = [
    "STO-3G", "6-31G", "6-31G*", "def2SVP", "def2TZVP",
    "def2SVPD", "cc-pVDZ", "cc-pVTZ",
]

# Orbital basis → matching DF auxiliary (BSE JSON filenames under data/basis_sets/).
# Prefer JK-fit for hybrids (exact exchange). Dunning: cc-pVTZ/VQZ-jkfit on BSE;
# no cc-pvdz-jkfit → def2-universal-jkfit. Ahlrichs/Pople/STO → def2-universal-jkfit.
BASIS_SETS = {
    "STO-3G": {
        "basis": "sto-3g.json",
        "aux": "def2-universal-jkfit.json",
    },
    "6-31G": {
        "basis": "6-31g.json",
        "aux": "def2-universal-jkfit.json",
    },
    "6-31G*": {
        "basis": "6-31gs.json",
        "aux": "def2-universal-jkfit.json",
    },
    "def2SVP": {
        "basis": "def2-svp.json",
        "aux": "def2-universal-jkfit.json",
    },
    "def2TZVP": {
        "basis": "def2-tzvp.json",
        "aux": "def2-universal-jkfit.json",
    },
    "def2SVPD": {
        "basis": "def2-svpd.json",
        "aux": "def2-universal-jkfit.json",
    },
    "def2QZVPP": {
        "basis": "def2-qzvpp.json",
        "aux": "def2-universal-jkfit.json",
    },
    # Dunning: prefer JK-fit aux for hybrids (RIFIT is Coulomb-only).
    # BSE has no cc-pvdz-jkfit → use def2-universal-jkfit.
    "cc-pVDZ": {
        "basis": "cc-pvdz.json",
        "aux": "def2-universal-jkfit.json",
    },
    "cc-pVTZ": {
        "basis": "cc-pvtz.json",
        "aux": "cc-pvtz-jkfit.json",
    },
    "cc-pVQZ": {
        "basis": "cc-pvqz.json",
        "aux": "cc-pvqz-jkfit.json",
    },
}
BASIS_JSON = {k: v["basis"] for k, v in BASIS_SETS.items()}
AUX_JSON = "def2-universal-jkfit.json"  # default / Ahlrichs-family aux

# Validation molecules (closed-shell singlets)
MOLECULES = [
    ("h2o.xyz", "H2O"),
    ("nh3.xyz", "NH3"),
    ("h2.xyz", "H2"),
    ("hf.xyz", "HF"),
    ("n2.xyz", "N2"),
    ("co2.xyz", "CO2"),
    ("ch4.xyz", "CH4"),
    ("c2h4.xyz", "C2H4"),
    ("bh3.xyz", "BH3"),
    ("so2.xyz", "SO2"),
    ("br2.xyz", "Br2"),
    ("i2.xyz", "I2"),
    ("ch2i2.xyz", "CH2I2"),
]


def aux_json_for(basis_label: str) -> str:
    """Return auxiliary JSON filename for a basis label."""
    return BASIS_SETS.get(basis_label, {}).get("aux", AUX_JSON)


def aux_label_for(basis_label: str) -> str:
    """Stable aux-basis id (filename stem), e.g. 'def2-universal-jkfit'."""
    return Path(aux_json_for(basis_label)).stem


def aux_json_from_label(aux_basis_label: str) -> str:
    """Map aux label / filename to a JSON filename under data/basis_sets/."""
    if not aux_basis_label:
        return AUX_JSON
    name = aux_basis_label
    if not name.endswith(".json"):
        name = f"{name}.json"
    return name


def _basis_has_elements(basis_path: Path, symbols: set[str]) -> bool:
    """True if BSE JSON contains electron_shells for every element symbol."""
    try:
        with open(basis_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    elements = data.get("elements", {})
    for sym in symbols:
        z = ELEMENTS.index(sym.upper()) if sym.upper() in ELEMENTS else -1
        if z < 0 or str(z) not in elements:
            return False
        if not elements[str(z)].get("electron_shells"):
            return False
    return True


def build_energy_matrix(quick=False):
    """Energy / gradient validation matrix.

    Default: closed-shell molecules × REF_FUNCTIONALS (HF, PBE, WB97X)
    × all BASIS_SETS × spherical/cartesian, plus UKS (OH) × UKS_REF_FUNCTIONALS
    × UKS_BASIS_LABELS.
    --quick: H2O × PBE × def2-SVP × spherical only.

    Skips molecule/basis pairs where the orbital or aux JSON lacks required
    elements. Also skips all-electron STO-3G for I-containing molecules
    (pathological heavy-atom / minimal-basis combination).
    """
    matrix = []
    if quick:
        mols = [("h2o.xyz", "H2O")]
        funcs = ["PBE"]
        bases = {"def2SVP": BASIS_SETS["def2SVP"]}
        shells = ("spherical",)
        uks_mols = []
        uks_funcs = []
        uks_bases = {}
    else:
        mols = MOLECULES
        funcs = list(REF_FUNCTIONALS)
        bases = BASIS_SETS
        shells = SHELL_TYPES
        uks_mols = list(UKS_MOLECULES)
        uks_funcs = list(UKS_REF_FUNCTIONALS)
        uks_bases = {k: BASIS_SETS[k] for k in UKS_BASIS_LABELS if k in BASIS_SETS}

    def _append_configs(xyz_file, label, funcs_list, bases_dict, shells_list,
                        multiplicity=1, spin=0):
        xyz_path = MOLECULES_DIR / xyz_file
        if not xyz_path.exists():
            return
        symbols = {a[0].upper() for a in load_xyz(xyz_path)}
        for bs_key, bs in bases_dict.items():
            if bs_key == "STO-3G" and symbols & {"I"}:
                continue
            basis_file = BASIS_DIR / bs["basis"]
            aux_file = BASIS_DIR / bs["aux"]
            if not basis_file.exists() or not aux_file.exists():
                continue
            if not _basis_has_elements(basis_file, symbols):
                continue
            if not _basis_has_elements(aux_file, symbols):
                continue
            for func in funcs_list:
                for shell in shells_list:
                    matrix.append({
                        "molecule": label,
                        "xyz": str(xyz_path),
                        "basis": str(basis_file),
                        "aux_basis": str(aux_file),
                        "ecp": None,
                        "functional": func,
                        "basis_label": bs_key,
                        "shell": shell,
                        "density_fitting": True,
                        "aux_basis_label": Path(bs["aux"]).stem,
                        "has_ecp": label in ("Br2", "I2", "CH2I2"),
                        "multiplicity": multiplicity,
                        "spin": spin,
                        "uks": multiplicity != 1 or spin != 0,
                    })

    for xyz_file, label in mols:
        _append_configs(xyz_file, label, funcs, bases, shells)

    for xyz_file, label, mult, spin in uks_mols:
        _append_configs(xyz_file, label, uks_funcs, uks_bases, shells,
                        multiplicity=mult, spin=spin)

    return matrix


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
                ams = sh["angular_momentum"]
                exps = sh["exponents"]
                coeff_arrays = sh["coefficients"]
                if len(ams) == 1:
                    # Single-L (possibly generally contracted): all coeff columns
                    L = ams[0]
                    am = AM_CHARS[L] if L < len(AM_CHARS) else "S"
                    lines.append(f"{symbol}    {am}")
                    for p in range(len(exps)):
                        exp = float(exps[p])
                        coeff_strs = [f"{float(ca[p]):.12f}" for ca in coeff_arrays]
                        lines.append(f"    {exp:.12f}  " + "  ".join(coeff_strs))
                else:
                    # Multi-L SP/SPD: coefficients[i] belongs to angular_momentum[i]
                    if len(ams) != len(coeff_arrays):
                        raise ValueError(
                            f"BSE multi-L shell size mismatch for {symbol}"
                        )
                    for L, ca in zip(ams, coeff_arrays):
                        am = AM_CHARS[L] if L < len(AM_CHARS) else "S"
                        lines.append(f"{symbol}    {am}")
                        for p in range(len(exps)):
                            exp = float(exps[p])
                            lines.append(f"    {exp:.12f}  {float(ca[p]):.12f}")
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
    # Prefer machine-readable CONVERGED flag when present
    for line in stdout.splitlines():
        if line.startswith("CONVERGED "):
            try:
                converged = int(line.split()[1]) == 1
            except (ValueError, IndexError):
                pass
    return energy, converged


def parse_cuest_scf_iterations(stdout: str):
    """Return SCF iteration count from ENERGY_COMPONENTS or verbose summary."""
    for line in stdout.splitlines():
        if line.startswith("N_SCF "):
            try:
                return int(line.split()[1])
            except (ValueError, IndexError):
                pass
    for line in stdout.splitlines():
        if line.startswith("Iterations:"):
            try:
                return int(line.split()[1])
            except (ValueError, IndexError):
                pass
        m = re.search(r"SCF converged in\s+(\d+)\s+iterations", line, re.I)
        if m:
            return int(m.group(1))
    return None


def as_grad_nx3(grad):
    """Normalize a gradient to an N×3 nested list (or None)."""
    if grad is None:
        return None
    if not grad:
        return []
    # Already N×3
    if isinstance(grad[0], (list, tuple)):
        out = [[float(x), float(y), float(z)] for x, y, z in grad]
        return out
    # Flat length-3N
    if len(grad) % 3 != 0:
        raise ValueError(f"flat gradient length {len(grad)} not divisible by 3")
    return [
        [float(grad[i]), float(grad[i + 1]), float(grad[i + 2])]
        for i in range(0, len(grad), 3)
    ]


def flatten_grad(grad):
    """Flatten an N×3 (or flat) gradient to a 1-D list."""
    g = as_grad_nx3(grad)
    if g is None:
        return None
    return [c for row in g for c in row]


def parse_gradient_block(stdout: str, header: str):
    """Parse gradient after header; return N×3 list or None."""
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
            grad.append([float(m.group(2)), float(m.group(3)), float(m.group(4))])
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



def _make_spherical_auxmol(mol, auxbasis):
    """Independent spherical (pure) DF auxiliary Mole — does not touch mol."""
    from pyscf.df.addons import make_auxmol

    mol_sph = mol.copy(deep=True)
    if mol_sph.cart:
        mol_sph.cart = False
        mol_sph.build(dump_input=False, parse_arg=False)
    auxmol = make_auxmol(mol_sph, auxbasis)
    if auxmol.cart:
        raise RuntimeError("failed to build spherical DF auxiliary Mole")
    return auxmol


def _build_cderi_cart_orb_sph_aux(mol, auxbasis):
    """(ij|P) DF tensor for Cartesian orbitals + spherical aux.

    PySCF has no mixed cart/sph 3c2e intor, so evaluate all-Cartesian 3c
    integrals and project the aux index with cart2sph. Respects mol.omega
    for range-separated hybrids.
    """
    import numpy
    import scipy.linalg
    from pyscf.df import incore
    from pyscf.df.addons import make_auxmol

    aux_sph = _make_spherical_auxmol(mol, auxbasis)
    aux_cart = make_auxmol(mol, auxbasis)  # inherits mol.cart=True
    omega = getattr(mol, "omega", None) or 0.0
    if omega:
        aux_sph.omega = omega
        aux_cart.omega = omega
    c2s = aux_cart.cart2sph_coeff()

    j3c_cart = incore.aux_e2(mol, aux_cart, intor="int3c2e", aosym="s2ij")
    j3c = numpy.asarray(j3c_cart @ c2s, order="C")
    j2c = aux_sph.intor("int2c2e", hermi=1)
    try:
        low = scipy.linalg.cholesky(j2c, lower=True)
        cderi = scipy.linalg.solve_triangular(
            low, j3c.T, lower=True, overwrite_b=False
        ).copy()
    except scipy.linalg.LinAlgError:
        # Range-separated (P|Q)_ω is often indefinite / rank-deficient.
        w, v = scipy.linalg.eigh(j2c)
        mask = w > 1e-12
        cderi = (v[:, mask] * (1.0 / numpy.sqrt(w[mask]))).T @ j3c.T
        cderi = numpy.asarray(cderi, order="C").copy()
    return aux_sph, numpy.asarray(cderi, order="C")


def _build_spherical_df(mf, auxbasis):
    """Attach spherical DF aux (+ _cderi), matching cuEST DF requirements."""
    from pyscf import __config__
    from pyscf.df import df_jk

    mol = mf.mol
    mf.with_df.auxbasis = auxbasis

    if not mol.cart:
        mf.with_df.build()
        if mf.with_df.auxmol is None or mf.with_df.auxmol.cart:
            raise RuntimeError("expected spherical DF auxmol for spherical orbitals")
        return mf

    # Cartesian orbitals: pin spherical-aux CDERI. Also override get_jk so
    # range-separated hybrids rebuild LR/SR CDERI the same way (PySCF's
    # range_coulomb() copy().reset() would otherwise leave _cderi=None).
    df = mf.with_df
    aux_sph, cderi = _build_cderi_cart_orb_sph_aux(mol, auxbasis)
    df.auxmol = aux_sph
    df._cderi = cderi
    df.build = lambda *a, **k: df

    _tol = getattr(__config__, "scf_hf_SCF_direct_scf_tol", 1e-13)

    def _get_jk(dm, hermi=1, with_j=True, with_k=True,
                direct_scf_tol=_tol, omega=None, _df=df):
        if omega is None:
            return df_jk.get_jk(_df, dm, hermi, with_j, with_k, direct_scf_tol)
        key = "%.6f" % omega
        rsh_df = _df._rsh_df.get(key)
        if rsh_df is None:
            old = _df.mol.omega
            _df.mol.omega = omega
            try:
                aux, cderi_w = _build_cderi_cart_orb_sph_aux(_df.mol, _df.auxbasis)
            finally:
                _df.mol.omega = old
            rsh_df = _df.copy().reset()
            rsh_df.auxbasis = _df.auxbasis
            rsh_df.auxmol = aux
            rsh_df._cderi = cderi_w
            rsh_df.build = lambda *a, _r=rsh_df, **k: _r
            _df._rsh_df[key] = rsh_df
        return df_jk.get_jk(rsh_df, dm, hermi, with_j, with_k, direct_scf_tol)

    df.get_jk = _get_jk
    return mf


def run_pyscf_df(atoms, basis_path, aux_path, functional, charge=0, spin=0,
                 grid_level=3, shell="spherical", compute_gradient=False):
    """PySCF RKS with density fitting using the same BSE JSON bases as cuEST.

    shell: 'spherical' (pure) or 'cartesian' for the *orbital* basis.
    The DF auxiliary basis is always spherical, matching cuEST.

    Gradients: analytic DF for spherical orbitals only. Cartesian is energy-only
    (PySCF has no mixed cart/sph DF analytic gradient).
    """
    from pyscf import gto, dft

    cart = shell == "cartesian"
    basis_dict, ecp_data = bse_json_to_pyscf(basis_path)
    aux_dict, _ = bse_json_to_pyscf(aux_path)

    symbols = sorted({a[0].upper() for a in atoms})
    missing = [s for s in symbols if s not in basis_dict]
    if missing:
        return {
            "ok": False,
            "error": f"basis missing elements: {', '.join(missing)}",
        }
    missing_aux = [s for s in symbols if s not in aux_dict]
    if missing_aux:
        return {
            "ok": False,
            "error": f"aux basis missing elements: {', '.join(missing_aux)}",
        }

    mol_ecp = None
    if ecp_data:
        ecp_parsed = {}
        for s in symbols:
            if s in ecp_data:
                _, ecp_str = ecp_data[s]
                ecp_parsed[s] = gto.basis.parse_ecp(ecp_str)
        if ecp_parsed:
            mol_ecp = ecp_parsed

    def _energy_for_atoms(atom_list):
        from pyscf import scf as pyscf_scf

        kwargs = dict(
            atom=atom_list, basis=basis_dict, charge=charge, spin=spin,
            verbose=0, cart=cart,
        )
        if mol_ecp:
            kwargs["ecp"] = mol_ecp
        mol = gto.M(**kwargs)
        # spin = 2S = multiplicity - 1. Nonzero → unrestricted.
        # Pure HF: no XC grid — isolates DF J/K vs cuEST.
        if functional.upper() == "HF":
            mf = (pyscf_scf.RHF(mol) if spin == 0 else pyscf_scf.UHF(mol)).density_fit()
        elif spin == 0:
            mf = dft.RKS(mol).density_fit()
            mf.xc = PYSCF_XC_MAP.get(functional, functional.lower())
            mf.grids.level = grid_level
            if functional.upper() in ("WB97X-V", "WB97M-V"):
                # PySCF's NLC grid is a separate object whose level does not
                # follow mf.grids.level, so set it explicitly. It does not need
                # to match: VV10 is a smooth long-range functional, already
                # converged to ~2e-12 by level 3, so level 5 is headroom. What
                # it must not be is PySCF's default of 1, which sits ~1.6e-7
                # off and would put a floor under the validation tolerance.
                mf.nlcgrids.level = min(grid_level, NLC_GRID_LEVEL)
        else:
            mf = dft.UKS(mol).density_fit()
            mf.xc = PYSCF_XC_MAP.get(functional, functional.lower())
            mf.grids.level = grid_level
            if functional.upper() in ("WB97X-V", "WB97M-V"):
                mf.nlcgrids.level = min(grid_level, NLC_GRID_LEVEL)
        mf.max_cycle = 200
        mf.conv_tol = 1e-10
        # minao/sad: '1e' (Hcore) can converge to false minima for some
        # heavy-atom + LRC cases (e.g. Br2/WB97X/cartesian).
        mf.init_guess = "minao"
        _build_spherical_df(mf, aux_dict)
        energy = mf.kernel()
        if not mf.converged or not is_finite(energy):
            return None, None
        if mf.with_df.auxmol is None or mf.with_df.auxmol.cart:
            return None, None
        return float(energy), mf

    energy, mf = _energy_for_atoms(atoms)
    if energy is None or mf is None:
        return {"ok": False, "error": "PySCF did not converge or non-finite energy / bad aux"}

    mol = mf.mol
    nelec_ab = mol.nelec
    if isinstance(nelec_ab, (tuple, list)):
        nalpha, nbeta = int(nelec_ab[0]), int(nelec_ab[1])
        nelec = nalpha + nbeta
    else:
        nelec = int(nelec_ab)
        nalpha = nbeta = nelec // 2
    nocc = nelec // 2
    scf_iterations = int(getattr(mf, "cycles", 0) or 0)

    gradient = None
    # Analytic DF grads: spherical orbitals only (PySCF has no mixed cart/sph DF grad).
    # Supported for both RKS (spin=0) and UKS (spin>0).
    if compute_gradient and not cart:
        try:
            g = mf.nuc_grad_method().kernel()
            gradient = [[float(x) for x in row] for row in g.reshape(-1, 3)]
            if not all(is_finite(x) for row in gradient for x in row):
                return {"ok": False, "error": "non-finite gradient", "mf": mf, "mol": mol}
        except Exception as e:
            return {"ok": False, "error": f"gradient failed: {e}", "mf": mf, "mol": mol}

    # HOMO/LUMO: RKS uses mo_energy; UKS uses max occupied / min virtual over spins.
    if spin == 0:
        homo = float(mf.mo_energy[nocc - 1])
        lumo = float(mf.mo_energy[nocc])
    else:
        ea, eb = mf.mo_energy
        occ_e = list(ea[:nalpha]) + list(eb[:nbeta])
        vir_e = list(ea[nalpha:]) + list(eb[nbeta:])
        homo = float(max(occ_e)) if occ_e else float("nan")
        lumo = float(min(vir_e)) if vir_e else float("nan")

    return {
        "ok": True,
        "energy": float(energy),
        "homo": homo,
        "lumo": lumo,
        "nelec": nelec,
        "nalpha": nalpha,
        "nbeta": nbeta,
        "nocc": nocc,
        "scf_iterations": scf_iterations,
        "gradient_ha_bohr": gradient,
        "shell": shell,
        "nao": int(mol.nao),
        "naux": int(mf.with_df.auxmol.nao),
        "uks": spin != 0,
        "mf": mf,
        "mol": mol,
    }
