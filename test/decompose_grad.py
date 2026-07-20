#!/usr/bin/env python3
"""Compare cuEST and PySCF gradient components term-by-term."""
import numpy as np
from pyscf import gto, dft, grad
from pyscf.dft import gen_grid

def print_comp(label, g, natm):
    parts = []
    for a in range(natm):
        parts.append(f"[{g[3*a]:10.6f} {g[3*a+1]:10.6f} {g[3*a+2]:10.6f}]")
    print(f"  {label:10s} {' '.join(parts)}")

def main():
    atoms = [
        ('O', (0.000000, 0.000000, 0.117790)),
        ('H', (0.756950, 0.000000, -0.471160)),
        ('H', (-0.756950, 0.000000, -0.471160)),
    ]
    a2b = 1.0 / 0.529177210903
    atoms_b = [(s, (x*a2b, y*a2b, z*a2b)) for s,(x,y,z) in atoms]

    mol = gto.M(atom=atoms, basis='def2SVP', verbose=0)
    mol.build()
    natm = mol.natm
    nao = mol.nao_nr()
    nocc = mol.nelec[0]

    mf = dft.RKS(mol).density_fit()
    mf.with_df.auxbasis = 'def2-universal-JKFIT'
    mf.xc = 'PBE'
    mf.grids = gen_grid.Grids(mol)
    mf.grids.level = 3
    mf.max_cycle = 150
    mf.conv_tol = 1e-8
    e_tot = mf.kernel()
    print(f"PySCF energy: {e_tot:.12f}")

    # Density matrices
    S = mol.intor('int1e_ovlp')
    D_total = mf.make_rdm1()
    D_alpha = D_total / 2.0
    print(f"Tr[D_total*S] = {np.trace(D_total @ S):.6f}")
    print(f"Tr[D_alpha*S] = {np.trace(D_alpha @ S):.6f}")

    C = mf.mo_coeff
    eps = mf.mo_energy
    C_occ = C[:, :nocc]
    eps_occ = eps[:nocc]
    W_alpha = C_occ @ np.diag(eps_occ) @ C_occ.T

    # Nuclear repulsion
    nuc_grad = np.zeros((natm, 3))
    for i in range(natm):
        Zi = mol.atom_charge(i)
        ri = np.array(atoms_b[i][1])
        for j in range(i+1, natm):
            Zj = mol.atom_charge(j)
            rj = np.array(atoms_b[j][1])
            d = ri - rj
            r = np.linalg.norm(d)
            f = Zi * Zj / r**3 * d
            nuc_grad[i] += f
            nuc_grad[j] -= f

    # Total gradient
    total_grad = mf.nuc_grad_method().kernel()
    de_elec = total_grad - nuc_grad

    # Compute one-electron gradient manually using PySCF integral derivatives
    # int1e_ipkin, int1e_ipnuc, int1e_ipovlp give (3, nao, nao) shell-wise derivatives
    # We need to contract with D and map to atoms
    kin_deriv = mol.intor('int1e_ipkin', comp=3)
    nuc_deriv = mol.intor('int1e_ipnuc', comp=3)
    ovlp_deriv = mol.intor('int1e_ipovlp', comp=3)

    # Map shells to atoms
    shell_atom = [mol.bas_atom(ib) for ib in range(mol.nbas)]

    # For each atom and direction, contract D with the derivative integrals
    grad_kin = np.zeros((natm, 3))
    grad_nuc = np.zeros((natm, 3))
    grad_ov = np.zeros((natm, 3))

    # PySCF integral derivatives are in the format where derivative is with
    # respect to the basis function CENTER (shell center = nucleus position).
    # The (comp, nao, nao) array contains dI_μν/dR_A where A is the atom
    # that basis function μ or ν is centered on.
    #
    # We can't easily map this to per-atom without PySCF's internal
    # contraction utilities. Let's use a simpler approach.

    # APPROACH: Use grad.rhf.get_hcore for one-electron part
    from pyscf.grad.rhf import get_hcore as rhf_get_hcore
    from pyscf.grad import rhf as rhf_grad

    # get_hcore(mol) returns (3, nao, nao) — shell-wise derivatives
    # These are dH_μν/dR_shell for each direction
    hcore_deriv = rhf_get_hcore(mol)  # (3, nao, nao)

    # The actual per-atom Hcore gradient for RHF is:
    # E1 = sum_A 2*Tr[D_hf * dHcore/dR_A]
    # where D_hf = make_rdm1 of the HF object (TOTAL density)
    # This is done by contracting hcore_deriv with D

    # For RKS with alpha-spin density D_alpha:
    # E1_alpha = 2 * Tr[D_alpha * (dT/dR + dV/dR)] - 2 * Tr[W_alpha * dS/dR]
    # Where dT/dR, dV/dR, dS/dR are the INTEGRAL derivatives

    # The per-atom contraction requires summing over shell contributions
    # Let's compute this manually

    # For derivative w.r.t. atom A, sum contributions from shells on atom A
    # Each integral derivative (comp, nao, nao) gives dI/dR for each basis fn center

    # Actually, the int1e_ip* integrals give dI_μν/dR for EACH shell
    # The mapping is: dI_μν/dR_C where C is the center of basis fn μ or ν
    # This is a 4-index quantity and the per-atom gradient is the sum

    # PySCF provides a helper for the one-electron contraction:
    # grad.rhf._make_h1 is the inner function but it uses the mol + D/W
    # Let's try another approach: use the full RHF gradient without DF

    # Compute RHF (no DF) gradient as reference for 1e contribution
    mf_nodf = dft.RKS(mol)
    mf_nodf.xc = 'PBE'
    mf_nodf.grids = gen_grid.Grids(mol)
    mf_nodf.grids.level = 3
    mf_nodf.max_cycle = 150
    mf_nodf.conv_tol = 1e-8
    mf_nodf.kernel()
    # This gives gradient with exact 4-center integrals (no DF)
    total_nodf = mf_nodf.nuc_grad_method().kernel()

    # The difference DF vs no-DF is the DF fitting error
    # For gradient decomposition, let's compute the 1e part
    # using the rhf gradient infrastructure

    # The cleanest way: compute the PySCF grad components by
    # running RHF grad (no XC, no DF) and extracting pieces

    # Actually, let me just read PySCF's grad/rhf.py source to understand
    # the internal structure, then extract what I need

    # SIMPLEST VIABLE APPROACH:
    # 1. PySCF total grad = nuc + electronic (we have both)
    # 2. Electronic = one_electron + two_electron + XC
    # 3. Use PySCF grad.get_hcore to get the one-e contribution

    # get_hcore returns the per-atom one-electron gradient
    from pyscf.grad.rhf import Gradients as GRHF
    gr = GRHF(mf)

    # get_hcore expects (mol) and returns per-atom h1 gradient
    # But the 2nd arg might be the density matrix
    h1_grad_atom = gr.get_hcore(mol)
    print(f"h1_grad_atom shape: {h1_grad_atom.shape}")

    # If it returns (3, nao, nao), we need to contract with D
    if len(h1_grad_atom.shape) == 3:
        # Shell-wise derivatives: contract with density
        # For RHF convention: D_total
        h1_per_atom = np.zeros((natm, 3))
        S_inv = np.linalg.inv(S)  # need this for mapping?

        # This is getting complex. Let me try yet another approach.
        pass

    # NEW APPROACH: just compare total PySCF gradient with cuEST total
    # If cuEST formula is correct, they should match

    print(f"\n{'='*60}")
    print(f"  PySCF TOTAL gradient (truth):")
    print_comp("TOT", total_grad.flatten(), natm)
    print(f"  PySCF NUCLEAR:")
    print_comp("NUC", nuc_grad.flatten(), natm)
    print(f"  PySCF ELECTRONIC (total - nuc):")
    print_comp("ELEC", de_elec.flatten(), natm)

    # cuEST values from our run (hardcoded)
    ce_nu = np.array([0.000,-2.993,0.000, 2.056,1.497,0.000,-2.056,1.497,0.000])
    ce_ke = np.array([0.000,0.464,0.000,-0.310,-0.232,0.000,0.310,-0.232,0.000])
    ce_po = np.array([0.000,-2.991,0.000, 1.816,1.356,0.000,-1.816,1.356,0.000])
    ce_ov = np.array([0.000,-0.137,0.000, 0.081,0.069,0.000,-0.081,0.069,0.000])
    ce_df = np.array([0.000,4.676,0.000,-3.144,-2.338,0.000,3.144,-2.338,0.000])
    ce_xc = np.array([0.000,-0.459,0.000,0.305,0.229,0.000,-0.305,0.229,0.000])

    ce_1e = 2*ce_ke + 2*ce_po - 2*ce_ov  # cuEST one-electron
    ce_2e_xc = ce_df + ce_xc               # cuEST 2e+XC
    ce_total = ce_nu + ce_1e + ce_2e_xc

    print(f"\n  cuEST one-electron (2*ke+2*po-2*ov):")
    print_comp("1E_CU", ce_1e, natm)
    print(f"  cuEST 2e+XC (df+xc):")
    print_comp("2E_CU", ce_2e_xc, natm)
    print(f"  cuEST TOTAL:")
    print_comp("TOT_CU", ce_total, natm)

    # COMPARISON
    py_nuc = nuc_grad.flatten()
    py_total = total_grad.flatten()

    print(f"\n{'='*60}")
    print(f"  COMPARISON: cuEST vs PySCF")
    print(f"{'='*60}")

    # Nuclear repulsion comparison
    nu_diff = ce_nu - py_nuc
    nu_rms = np.sqrt(np.mean(nu_diff**2))
    print(f"  NUC repulsion RMS: {nu_rms:.6f}  (should be ~0)")

    # Total comparison
    total_diff = ce_total - py_total
    total_rms = np.sqrt(np.mean(total_diff**2))
    print(f"  TOTAL gradient RMS: {total_rms:.6f}")

    # The electronic part comparison
    py_elec = de_elec.flatten()
    ce_elec = ce_1e + ce_2e_xc
    elec_diff = ce_elec - py_elec
    elec_rms = np.sqrt(np.mean(elec_diff**2))
    print(f"  ELECTRONIC RMS: {elec_rms:.6f}")

    # Individual component comparison
    print(f"\n  Per-component values:")
    for label, ce, py in [('nu', ce_nu, py_nuc), ('total', ce_total, py_total)]:
        for a, sym in enumerate(['O ', 'H1', 'H2']):
            i = 3*a
            d = ce[i:i+3] - py[i:i+3]
            print(f"    {label} {sym}: cuEST=[{ce[i]:7.4f} {ce[i+1]:7.4f} {ce[i+2]:7.4f}]  PySCF=[{py[i]:7.4f} {py[i+1]:7.4f} {py[i+2]:7.4f}]  diff=[{d[0]:7.4f} {d[1]:7.4f} {d[2]:7.4f}]")

if __name__ == '__main__':
    main()
