#!/usr/bin/env python3
"""
FULL validation: gradients, ECPs, range-separated hybrids, functionals.
Compares cuEST analytical+numerical gradients against PySCF references.
"""
import subprocess, numpy as np, json, sys, os, tempfile, shutil, urllib.request
from pathlib import Path
from datetime import datetime

PROJ = Path(__file__).parent.parent
EXE = PROJ / "build" / "cuest_dft"
DATA = PROJ / "data"
BSE = "https://www.basissetexchange.org/api/basis"

BOHR = 0.52917721092
ANG2BOHR = 1.0 / BOHR

# ======== DOWNLOAD HELPERS ========
def download_basis(name, fmt="gaussian94"):
    out = DATA / f"{name}.gbs"
    if out.exists(): return out
    url = f"{BSE}/{name}/format/{fmt}/?version=1&optimize_general=true"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cuEST/1.0"})
        data = urllib.request.urlopen(req, timeout=30).read()
        with open(out, "wb") as f: f.write(data)
        return out
    except: return None

# Ensure we have ECP basis
for b in ["def2-svp-ecp", "def2-universal-jkfit", "def2-svp", "def2-tzvp"]:
    download_basis(b)

# ======== TEST CONFIGURATIONS ========
# (molecule_name, xyz_string, functional, basis, use_ecp, check_gradient)
TARGETS = [
    # --- Gradient tests: H2O with multiple functionals ---
    ("H2O", """O 0 0 0; H 0.75695 0.585882 0; H -0.75695 0.585882 0""",
     "PBE", "def2SVP", False, True),
    ("H2O", """O 0 0 0; H 0.75695 0.585882 0; H -0.75695 0.585882 0""",
     "B3LYP", "def2SVP", False, True),
    ("H2O", """O 0 0 0; H 0.75695 0.585882 0; H -0.75695 0.585882 0""",
     "PBE0", "def2SVP", False, True),
    ("H2O", """O 0 0 0; H 0.75695 0.585882 0; H -0.75695 0.585882 0""",
     "CAM-B3LYP", "def2SVP", False, True),

    # --- Range-separated hybrids ---
    ("H2O", """O 0 0 0; H 0.75695 0.585882 0; H -0.75695 0.585882 0""",
     "WB97X", "def2SVP", False, False),

    # --- Larger basis ---
    ("NH3", """N 0 0 0.1163; H 0 0.9388 -0.2713; H 0.8129 -0.4694 -0.2713; H -0.8129 -0.4694 -0.2713""",
     "PBE", "def2TZVP", False, False),

    # --- Gradient test: NH3 ---
    ("NH3", """N 0 0 0.1163; H 0 0.9388 -0.2713; H 0.8129 -0.4694 -0.2713; H -0.8129 -0.4694 -0.2713""",
     "PBE", "def2SVP", False, True),

    # --- ECP tests ---
    ("I2", """I 0 0 1.335; I 0 0 -1.335""",
     "PBE", "def2SVP", True, False),
    ("Br2", """Br 0 0 1.15; Br 0 0 -1.15""",
     "PBE", "def2SVP", True, False),
]

def make_xyz(atoms_str, path):
    """Convert 'C 0 0 0; H 1 0 0' format to XYZ file."""
    atoms = [a.split() for a in atoms_str.split(";")]
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\nvalidation geometry\n")
        for a in atoms:
            f.write(f"{a[0]}  {float(a[1]):10.6f}  {float(a[2]):10.6f}  {float(a[3]):10.6f}\n")

def run_cuest(xyz_path, basis_path, aux_path, functional, use_ecp=False, ecp_path=None, max_iter=100):
    cmd = [str(EXE), "--xyz", str(xyz_path), "--basis", str(basis_path),
           "--aux-basis", str(aux_path), "--functional", functional,
           "--max-iter", str(max_iter), "--conv-thresh", "1e-8", "--quiet"]
    if use_ecp and ecp_path:
        cmd += ["--ecp", str(ecp_path)]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        for line in r.stdout.splitlines():
            if "Final SCF energy:" in line:
                return float(line.split()[-2]), True
        for line in reversed(r.stdout.splitlines()):
            if "Etot =" in line:
                return float(line.split("Etot =")[1].split()[0]), True
        return None, False
    except Exception as e:
        return str(e), False

def run_cuest_gradient(xyz_path, basis_path, aux_path, functional,
                        use_ecp=False, ecp_path=None, max_iter=100):
    """Run cuEST with --gradient, parse both analytical and numerical outputs."""
    cmd = [str(EXE), "--xyz", str(xyz_path), "--basis", str(basis_path),
           "--aux-basis", str(aux_path), "--functional", functional,
           "--max-iter", str(max_iter), "--gradient", "--quiet"]
    if use_ecp and ecp_path:
        cmd += ["--ecp", str(ecp_path)]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        # Parse analytical gradient from RAW_DF + RAW_NU on stderr
        # Parse numerical gradient from stdout
        energy = None
        num_grad = None
        ana_grad_nu = None
        ana_grad_df = None

        for line in r.stdout.splitlines():
            if "Final SCF energy:" in line:
                energy = float(line.split()[-2])
            if "Numerical Gradient" in line:
                num_grad = []
            if num_grad is not None and "Atom" in line and "|F|" in line:
                parts = line.split()
                try:
                    idx = parts.index("Atom") + 1
                    fx, fy, fz = float(parts[idx+1]), float(parts[idx+2]), float(parts[idx+3])
                    num_grad.extend([fx, fy, fz])
                except: pass

        for line in r.stderr.splitlines():
            if "RAW_NU" in line:
                ana_grad_nu = [float(x) for x in line.split()[1:]]
            if "RAW_DF" in line:
                ana_grad_df = [float(x) for x in line.split()[1:]]

        # Analytical total = NU + DF (our formula)
        ana_grad = None
        if ana_grad_nu and ana_grad_df and len(ana_grad_nu) == len(ana_grad_df):
            ana_grad = [ana_grad_nu[i] + ana_grad_df[i] for i in range(len(ana_grad_nu))]

        return energy, ana_grad, num_grad
    except Exception as e:
        return str(e), None, None

def run_pyscf(atoms_str, basis_name, functional="PBE", use_ecp=False):
    from pyscf import gto, dft, grad

    atoms = []
    for a in atoms_str.split(";"):
        parts = a.split()
        atoms.append((parts[0], (float(parts[1]), float(parts[2]), float(parts[3]))))

    xc_map = {
        "PBE": "pbe,pbe", "B3LYP": "b3lyp", "PBE0": "pbe0",
        "CAM-B3LYP": "cam-b3lyp", "WB97X": "wb97x"
    }

    mol = gto.M(atom=atoms, basis=basis_name, verbose=0)
    if use_ecp:
        mol.ecp = basis_name  # use ECP from basis set

    mf = dft.RKS(mol)
    mf.xc = xc_map.get(functional, "pbe,pbe")
    mf = mf.density_fit()
    mf.max_cycle = 200
    mf.conv_tol = 1e-8

    try:
        e = mf.kernel()
        if not mf.converged:
            return None, None, False

        # Gradient
        g = grad.RKS(mf).kernel()
        grad_flat = g.flatten().tolist()

        return e, grad_flat, True
    except Exception as ex:
        return str(ex), None, False

def main():
    print("=" * 70)
    print("  cuEST DFT — FULL Validation Suite")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if not EXE.exists():
        print(f"ERROR: {EXE} not found. Build first.")
        sys.exit(1)

    # Paths
    aux_path = DATA / "def2-universal-jkfit.gbs"
    svp_path = DATA / "def2-svp.gbs"
    tzvp_path = DATA / "def2-tzvp.gbs"
    ecp_path = DATA / "def2-svp-ecp.gbs"

    results = []
    n_pass = n_fail = n_skip = 0

    for mol_name, atoms_str, func, basis, use_ecp, check_grad in TARGETS:
        # Create XYZ
        xyz_path = DATA / f"v_{mol_name}_{func}_{basis}.xyz"
        make_xyz(atoms_str, xyz_path)

        # Select basis file
        bf = svp_path if "SVP" in basis else tzvp_path

        # Run cuEST
        if check_grad:
            cu_e, cu_ana, cu_num = run_cuest_gradient(
                xyz_path, bf, aux_path, func, use_ecp, ecp_path if use_ecp else None)
        else:
            cu_e, _ = run_cuest(xyz_path, bf, aux_path, func, use_ecp, ecp_path if use_ecp else None)
            cu_ana = cu_num = None

        # Run PySCF
        py_e, py_grad, py_ok = run_pyscf(atoms_str, basis, func, use_ecp)

        # Energy check
        if isinstance(cu_e, float) and py_ok:
            e_diff = abs(cu_e - py_e)
            e_ok = e_diff < 1e-4
        else:
            e_diff = None
            e_ok = False

        # Gradient check
        g_ana_ok = g_num_ok = False
        g_ana_rms = g_num_rms = None

        if check_grad and cu_ana is not None and py_grad is not None:
            ca = np.array(cu_ana); pg = np.array(py_grad)
            if len(ca) == len(pg):
                g_ana_rms = np.sqrt(np.mean((ca - pg)**2))
                g_ana_ok = g_ana_rms < 0.01

        if check_grad and cu_num is not None and py_grad is not None:
            cn = np.array(cu_num); pg = np.array(py_grad)
            if len(cn) == len(pg):
                g_num_rms = np.sqrt(np.mean((cn - pg)**2))
                g_num_ok = g_num_rms < 0.001

        # Status
        parts = []
        parts.append("✅" if e_ok else "❌")
        parts.append(f"E={cu_e:.8f}" if isinstance(cu_e,float) else f"E=ERR")
        parts.append(f"dE={e_diff*1000:.3f}mHa" if e_diff else "dE=N/A")
        if check_grad:
            parts.append(f"ana={g_ana_rms:.4f}" if g_ana_rms else "ana=N/A")
            parts.append(f"num={g_num_rms:.4f}" if g_num_rms else "num=N/A")

        label = f"{mol_name:5s}/{func:10s}/{basis:8s}"
        if use_ecp: label += "(ECP)"
        print(f"  {label:40s} {' '.join(parts)}")

        results.append({
            "molecule": mol_name, "functional": func, "basis": basis,
            "ecp": use_ecp,
            "energy_ok": e_ok, "energy_diff_mHa": e_diff*1000 if e_diff else None,
            "grad_ana_rms": g_ana_rms, "grad_num_rms": g_num_rms,
        })

        if e_ok: n_pass += 1
        else: n_fail += 1

    print(f"\n{'=' * 70}")
    print(f"  Summary: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP")
    print(f"{'=' * 70}")

    with open(PROJ / "test" / "results_full.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return 0 if n_fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
