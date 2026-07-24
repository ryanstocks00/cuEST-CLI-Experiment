#!/usr/bin/env python3
"""
Validate the SAD atomic reference against PySCF.

The spherical path is expected to reproduce PySCF's SAD essentially exactly:
both solve the same spherically averaged atomic RHF, so with the *same* density
fitting the only difference should be arithmetic ordering. Energies and the full
atomic density matrix are both compared.

The Cartesian path is deliberately NOT expected to match. PySCF builds its atom
in a spherical basis and maps it up with cart2sph, which leaves the Cartesian
contaminant functions (the s in a 6d shell, the p in a 10f shell) with exactly
zero density. Here the atom is solved directly in the Cartesian space, so those
functions are part of the variational space. The check there is therefore the
variational one: our Cartesian energy must be <= the spherical energy, and close
to it.

Usage:
    python3 test/validate_sad.py                # default element/basis sweep
    python3 test/validate_sad.py --elements 6,8 --basis cc-pVDZ
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy

PROJ_DIR = Path(__file__).resolve().parent.parent
BASIS_DIR = PROJ_DIR / "data" / "basis_sets"
PROBE = PROJ_DIR / "build" / "probe_atom"

# Orbital basis label -> (our JSON, our aux JSON). PySCF is fed the *same* JSON
# via common.bse_json_to_pyscf rather than its own internal basis of the same
# name: BSE's contraction of e.g. cc-pVDZ is an equivalent but different set of
# individual basis functions, which leaves the SCF energy identical while making
# an AO-by-AO density comparison meaningless.
BASIS_SETS = {
    "STO-3G":   ("sto-3g.json",    "def2-universal-jkfit.json"),
    "6-31G":    ("6-31g.json",     "def2-universal-jkfit.json"),
    "6-31G*":   ("6-31gs.json",    "def2-universal-jkfit.json"),
    "cc-pVDZ":  ("cc-pvdz.json",   "def2-universal-jkfit.json"),
    "cc-pVTZ":  ("cc-pvtz.json",   "cc-pvtz-jkfit.json"),
    "def2SVP":  ("def2-svp.json",  "def2-universal-jkfit.json"),
    "def2TZVP": ("def2-tzvp.json", "def2-universal-jkfit.json"),
}

# Light main group, a partly-filled 3d transition metal, a filled-d case, and
# several ECP-bearing heavy elements (Br/Ag/I/Xe/Au) so the ECP core-subtraction
# path in atomic_config.cpp is actually exercised. Elements a given basis JSON
# lacks are skipped automatically.
DEFAULT_ELEMENTS = [1, 3, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18, 20, 26, 29, 30,
                    35, 36, 47, 53, 54, 79]

SKIP = "skip"  # sentinel: element absent from the basis JSON

ENERGY_TOL = 1e-8   # Ha; same method + same DF => should agree to ~1e-10
DENSITY_TOL = 1e-6  # max abs element of the AO density matrix
DF_FLOOR_CAP = 1e-3 # ceiling on the DF-noise-floor escape (see pyscf_df_error)


def run_probe(basis_json, aux_json, Z, cartesian, dump_path, cache_dir):
    cmd = [str(PROBE), str(BASIS_DIR / basis_json), str(BASIS_DIR / aux_json), str(Z),
           "--cartesian" if cartesian else "--spherical",
           "--dump-density", str(dump_path)]
    env = dict(os.environ, CUEST_SAD_CACHE_DIR=str(cache_dir))
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    if res.returncode != 0:
        # A basis JSON that simply lacks this element is a data gap, not a
        # failure of the guess — skip it the same way a missing PySCF basis is.
        if "not found in basis set JSON" in res.stderr:
            return None, SKIP
        return None, f"probe_atom exit {res.returncode}: {res.stderr.strip()[-300:]}"
    m = re.search(r"ATOM_ENERGY\s+(-?\d+\.\d+)", res.stdout)
    if not m:
        return None, "no ATOM_ENERGY in probe output"
    sm = re.search(r"SHELL_L([\d ]*)", res.stdout)
    shell_l = [int(x) for x in sm.group(1).split()] if sm else []
    return (float(m.group(1)), shell_l), None


def ao_permutation(shell_l, is_pure):
    """Map our AO order onto PySCF's.

    PySCF sorts an atom's shells by angular momentum when building the molecule;
    BasisBuilder keeps basis-file order. For a basis whose JSON is already
    l-sorted (Dunning, def2) this is the identity, but Pople SP shells come out
    interleaved (s, s, p, s, p) against PySCF's (s, s, s, p, p). Returns `perm`
    with perm[i] = our AO index for PySCF AO i.
    """
    sizes = [(2 * l + 1) if is_pure else ((l + 1) * (l + 2) // 2) for l in shell_l]
    offsets, acc = [], 0
    for s in sizes:
        offsets.append(acc)
        acc += s
    order = sorted(range(len(shell_l)), key=lambda i: shell_l[i])  # stable
    perm = []
    for s in order:
        perm.extend(range(offsets[s], offsets[s] + sizes[s]))
    return perm


def read_density(path):
    with open(path) as fh:
        n = int(fh.readline())
        rows = [[float(x) for x in fh.readline().split()] for _ in range(n)]
    return numpy.array(rows)


def pyscf_atom(Z, basis_json, aux_json, cart):
    from pyscf import gto
    from pyscf.scf import atom_hf
    from pyscf.data import elements

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import bse_json_to_pyscf

    sym = elements.ELEMENTS[Z]
    basis_dict, ecp_data = bse_json_to_pyscf(BASIS_DIR / basis_json)
    aux_dict, _ = bse_json_to_pyscf(BASIS_DIR / aux_json)

    # common.py keys these dicts with its own uppercase symbols ('RB'), while
    # PySCF's element table is mixed case ('Rb'). Match case-insensitively —
    # comparing directly silently "passes" only the single-letter elements and
    # reports every two-letter one as a missing basis.
    def lookup(d, what):
        if d:
            for k in d:
                if k.upper() == sym.upper():
                    return k
        raise KeyError(f"{sym} not in {what}")

    bkey = lookup(basis_dict, basis_json)
    lookup(aux_dict, aux_json)

    kwargs = dict(atom=f"{sym} 0 0 0", basis={sym: basis_dict[bkey]}, spin=Z % 2,
                  cart=cart, verbose=0)
    # ECP entries are (nelec, NWChem-format string); parse as common.py does.
    if ecp_data:
        for k in ecp_data:
            if k.upper() == sym.upper():
                kwargs["ecp"] = {sym: gto.basis.parse_ecp(ecp_data[k][1])}
                break
    mol = gto.M(**kwargs)
    mol._sad_aux = aux_dict  # stashed for pyscf_df_error()

    if mol.nelectron == 1:
        # PySCF routes one-electron atoms around the SCF entirely: a lone
        # electron has no self-interaction for 2J-K to cancel.
        mf = atom_hf.AtomHF1e(mol)
    else:
        mf = atom_hf.AtomSphAverageRHF(mol).density_fit(auxbasis=aux_dict)
        # PySCF's default conv_tol leaves the density itself uncertain at the
        # 1e-6 level even when the energy looks converged — a fractionally
        # occupied shell keeps moving after the energy has stopped changing.
        # Tighten it so the comparison measures our error, not PySCF's.
        mf.conv_tol = 1e-13
    mf.verbose = 0
    mf.run()
    dm = numpy.asarray(mf.make_rdm1())
    if dm.ndim == 3:
        dm = dm[0] + dm[1]  # AtomHF1e is ROHF-based and returns (alpha, beta)
    return mf.e_tot, dm, mol


def pyscf_df_error(mol):
    """|E_DF - E_exact| for PySCF's own atomic SCF — the density-fitting noise
    floor for this element and basis.

    For heavy ECP elements the def2-universal-jkfit fit is ill-conditioned
    enough that PySCF's own DF error reaches ~5e-5 Ha (Xe, I), which is larger
    than any plausible disagreement between two correct implementations. Below
    that floor, "cuEST vs PySCF-DF" is comparing two arbitrary resolutions of
    the same ill-conditioned fit, not measuring correctness — so the check falls
    back to this when the tight comparison fails. Only computed on failure:
    exact ERIs for a heavy atom are not cheap.
    """
    from pyscf.scf import atom_hf
    if mol.nelectron == 1:
        return 0.0
    ex = atom_hf.AtomSphAverageRHF(mol)
    ex.conv_tol = 1e-13
    ex.verbose = 0
    ex.run()
    df = atom_hf.AtomSphAverageRHF(mol).density_fit(auxbasis=mol._sad_aux)
    df.conv_tol = 1e-13
    df.verbose = 0
    df.run()
    return abs(df.e_tot - ex.e_tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", default=None, help="comma-separated Z values")
    ap.add_argument("--basis", default=None, help="comma-separated basis labels")
    ap.add_argument("--skip-cartesian", action="store_true")
    args = ap.parse_args()

    if not PROBE.exists():
        print(f"error: {PROBE} not built (cmake --build build --target probe_atom)")
        return 2

    elements_list = ([int(x) for x in args.elements.split(",")]
                     if args.elements else DEFAULT_ELEMENTS)
    basis_list = (args.basis.split(",") if args.basis else list(BASIS_SETS))

    failures, checked = [], 0
    tmp = Path(tempfile.mkdtemp(prefix="sad_validate_"))
    cache = tmp / "cache"

    for label in basis_list:
        if label not in BASIS_SETS:
            print(f"skip unknown basis {label}")
            continue
        bjson, auxjson = BASIS_SETS[label]

        for Z in elements_list:
            # --- spherical: expected to reproduce PySCF ---
            dump = tmp / f"d_{label}_{Z}_sph.txt"
            got, err = run_probe(bjson, auxjson, Z, False, dump, cache)
            if err is SKIP:
                print(f"  {label:9s} Z={Z:3d} spherical: element not in basis JSON")
                continue
            if err:
                failures.append(f"{label} Z={Z} spherical: {err}")
                continue
            ours, shell_l = got
            try:
                ref, ref_dm, ref_mol = pyscf_atom(Z, bjson, auxjson, cart=False)
            except Exception as e:  # basis may not cover this element
                print(f"  {label:9s} Z={Z:3d} spherical: pyscf unavailable ({e})")
                continue

            de = abs(ours - ref)
            our_dm = read_density(dump)
            if our_dm.shape != ref_dm.shape:
                failures.append(f"{label} Z={Z} spherical: density shape "
                                f"{our_dm.shape} != pyscf {ref_dm.shape}")
                continue
            perm = ao_permutation(shell_l, is_pure=True)
            our_dm = our_dm[numpy.ix_(perm, perm)]
            dd = numpy.max(numpy.abs(our_dm - ref_dm))
            checked += 1
            ok = de < ENERGY_TOL and dd < DENSITY_TOL
            note = ""
            if not ok:
                # Fall back to the DF noise floor: below PySCF's own DF error
                # there is nothing meaningful left to compare. Capped, because
                # the floor is not always small — pairing STO-3G with
                # def2-universal-jkfit gives a fit so poor for heavy elements
                # that PySCF's own DF error reaches tens of Ha, and an uncapped
                # floor would rubber-stamp anything.
                floor = min(pyscf_df_error(ref_mol), DF_FLOOR_CAP)
                if de <= max(floor, ENERGY_TOL):
                    ok = True
                    note = f"  [within PySCF's own DF error {floor:.1e}]"
            status = "ok" if ok else "FAIL"
            print(f"  {label:9s} Z={Z:3d} spherical: dE={de:.2e} dD={dd:.2e}  "
                  f"{status}{note}")
            if not ok:
                failures.append(f"{label} Z={Z} spherical: dE={de:.3e} dD={dd:.3e} "
                                f"(ours {ours:.10f} vs pyscf {ref:.10f})")

            if args.skip_cartesian:
                continue

            # --- cartesian: variational check, not a PySCF comparison ---
            dumpc = tmp / f"d_{label}_{Z}_cart.txt"
            got_c, err = run_probe(bjson, auxjson, Z, True, dumpc, cache)
            if err is SKIP:
                continue
            if err:
                failures.append(f"{label} Z={Z} cartesian: {err}")
                continue
            ours_c, _ = got_c
            # Solving in the larger Cartesian space (the l-2, l-4, ... shell
            # contaminants are real extra variational freedom) can only lower
            # the constrained minimum relative to the spherical one.
            gain = ours - ours_c  # >= 0 expected
            variational_ok = gain >= -1e-9
            # Sanity bound only, to catch a wildly wrong answer rather than a
            # legitimately large gain. It has to be *relative*: in a flexible
            # basis the contaminants buy ~1e-4 Ha, but in a minimal basis they
            # are genuinely new functions and buy of order 1 Ha per contaminant
            # on a heavy atom (STO-3G Xe gains 2 AOs and ~3 Ha).
            sane_ok = gain < 0.02 * abs(ours)
            ok_c = variational_ok and sane_ok
            status = "ok" if ok_c else "FAIL"
            print(f"  {label:9s} Z={Z:3d} cartesian: E={ours_c:.10f} "
                  f"(sph {ours:.10f}, dE={ours_c - ours:+.2e})  {status}")
            if not variational_ok:
                failures.append(f"{label} Z={Z} cartesian: E={ours_c:.10f} is ABOVE "
                                f"spherical {ours:.10f} — the larger Cartesian "
                                f"space cannot raise the constrained minimum")
            elif not sane_ok:
                failures.append(f"{label} Z={Z} cartesian: gain {gain:.3f} Ha over "
                                f"spherical {ours:.10f} exceeds {0.02*abs(ours):.3f} Ha "
                                f"({2}% of |E|) — implausibly large, check the guess")

    print(f"\n{checked} spherical comparisons against PySCF")
    if checked == 0:
        print("error: nothing was actually compared")
        return 2
    if failures:
        print(f"{len(failures)} FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
