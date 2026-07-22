# cuEST DFT — GPU-Accelerated Quantum Chemistry

A C++ DFT package built on [NVIDIA cuEST](https://developer.nvidia.com/cuda/cuda-x-libraries/cuest) (v0.2.0) with RAII wrappers, analytic gradients, and PySCF-validation.

> **Note:** The initial guess is a Superposition of Atomic *Coefficients* (SAC), not the
> more standard Superposition of Atomic *Densities* (SAD). cuEST's density-fitted
> exchange (K) API only accepts occupied MO coefficients, not an arbitrary density
> matrix, so a density guess would require a custom fock build or no exchange on the first
> iteration. A density guess would be preferable to allow spherical averaging.

## Features

- **Full SCF**: RKS (Restricted Kohn-Sham) with density fitting (RI-JK)
- **DIIS**: Direct inversion in iterative subspace, configurable space and start
- **SAD initial guess**: Superposition of atomic densities from spherically
  symmetric free-atom HF, with fractionally occupied MOs
- **Gradients**: Analytic or numerical finite differences
- **Functionals**: PBE, B3LYP, B3LYP5, PBE0, CAM-B3LYP, ωB97X, ωB97X-V, ωB97M-V, HSE06, M06, M06-2X, LC-ωPBE, LC-ωPBEh
  (ωB97X-V/ωB97M-V include the VV10 nonlocal-correlation term via cuEST's separate nonlocal XC potential API)
- **ECP support**: Effective core potentials
- **C++ RAII**: Memory-safe wrappers around the cuEST C API
- **Validated**: SCF energies match PySCF-DF to ~0.01 mHa (light systems)

## Requirements

- **CUDA Toolkit 13.x** (13.0.2+ for JIT)
- **NVIDIA GPU** (Compute Capability 8.0+)
- **cuEST 0.2.0 C SDK** (headers + shared library)
- **CMake 3.20+**, **GCC 13+**
- **PySCF** (for validation tests)

## Quick Start

### 1. Get cuEST SDK

Download from [NVIDIA cuEST Downloads](https://developer.nvidia.com/cuest-downloads):
```bash
# Extract and place in external/
mkdir -p external/cuest_include external/lib
# Copy cuest.h + headers → external/cuest_include/
# Copy libcuest.so.0.2.0 → external/ (symlink as libcuest.so)
ln -sf libcuest.so.0.2.0 external/libcuest.so
```

### 2. Build

```bash
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES="80;89;90"
cmake --build build -j$(nproc)
```

### 3. Basis sets

BSE JSON bases live under `data/basis_sets/`. Refresh / fetch missing files with:

```bash
python3 test/download_bases.py
```

Orbital ↔ auxiliary pairings used by the validators (`test/common.py`):

| Orbital | Auxiliary |
|---------|-----------|
| STO-3G, 6-31G, 6-31G* | def2-universal-jkfit |
| def2-SVP / TZVP / SVPD / QZVPP | def2-universal-jkfit |
| cc-pVDZ | def2-universal-jkfit (no cc-pvdz-jkfit on BSE) |
| cc-pVTZ / VQZ | matching `cc-pV*Z-jkfit` |

Default `test/reference.json` matrix: closed-shell molecules × {HF, PBE, WB97X,
WB97X-V, WB97M-V} × all bases × {spherical, Cartesian}, plus UKS OH (mult=2)
× {HF, PBE, PBE0, WB97X, WB97X-V, WB97M-V} × selected bases. SCF iteration
counts included. Gradients are stored for
spherical-orbital refs only (analytic DF; hybrids skip known-bad SP-only cases).
The DF auxiliary basis is always spherical (cuEST requirement; `--cartesian`
applies to the primary basis only). Regenerate with:

```bash
python3 test/generate_reference.py
python3 test/generate_reference.py --shell cartesian --merge
python3 test/validate_cuest.py
```

### 4. Run

```bash
./build/cuest_dft \
  --xyz data/molecules/small/h2o.xyz \
  --basis data/basis_sets/def2-svp.json \
  --aux-basis data/basis_sets/def2-universal-jkfit.json \
  --functional PBE \
  --gradient
```

## Usage

```
cuEST DFT — GPU-accelerated density functional theory
Usage: ./build/cuest_dft --xyz <file> --basis <json_file> [options]

Required arguments:
  --xyz <path>             Input geometry in XYZ format
  --basis <path>           Basis set in BSE JSON format

Optional arguments:
  --aux-basis <path>       Auxiliary/DF (RI-J) basis set (BSE JSON)
  --functional <name>      XC functional (default: PBE)
  --radial-pts <n>         Radial grid points (default: 75)
  --angular-pts <n>        Angular Lebedev points (default: 302)
  --charge <int>           Total charge (default: 0)
  --multiplicity <int>     Spin multiplicity (default: 1)
  --unrestricted, --uks    Solve unrestricted even at multiplicity 1
  --break-symmetry <rad>   UKS β HOMO/LUMO mix angle (default: 0.1);
                           only applied when nalpha == nbeta

Initial guess options:
  --hcore-guess            Use the bare core-Hamiltonian guess instead of SAD
  --sad-functional-atoms   Solve the SAD reference atoms with the molecular
                           functional instead of HF (default HF matches PySCF
                           and needs no XC grid)

SCF convergence options:
  --max-iter <n>           Max SCF iterations (default: 250)
  --conv-thresh <val>      Orbital-gradient norm |g| convergence (default: 1e-4)
  --energy-conv <val>      Energy change convergence (default: 1e-8)
  --diis-start <n>         Iteration to enable DIIS (default: 1)
  --diis-space <n>         DIIS subspace dimension (default: 6)
  --damping <val>          Density damping factor (default: 0.0)

Other options:
  --quiet                  Minimal output
  --verbose                Verbose output
  --print-mos              Print final MO energies
  --spherical              Spherical (pure) orbital Gaussians (default)
  --cartesian              Cartesian orbital Gaussians
  --gradient               Nuclear gradient (analytic + numerical)
  --analytic-gradient      Analytic nuclear gradient only
  --jit                    Enable cuEST JIT kernels (default)
  --no-jit                 Disable JIT (AOT kernels, fp64)
  --help                   Show this help

Advanced/tuning options (cuEST engine parameters; defaults match cuEST's own):
  --max-gauss-hermite <n>       Max Gauss-Hermite quadrature points (default: 20)
  --max-l-solid-harmonic <n>    Max angular momentum for solid-harmonic
                                transforms (default: 10)
  --max-rys <n>                 Max Rys quadrature points; 0 = largest
                                available (default: 0)
  --jit-cache-dir <path>        JIT kernel cache directory; must be a
                                trusted, non-world-writable path (default:
                                derived ~/.cuest_cache/...)
  --jit-compile-threads <n>     Parallel JIT precompile worker threads,
                                clamped to 256 (default: 16)
  --df-fitting-cutoff <val>     Eigenvalue threshold below which DF metric
                                eigenvalues are discarded (default: 1e-12)
  --df-fitting-absolute         Treat --df-fitting-cutoff as absolute, not
                                relative to the largest eigenvalue (default:
                                relative)
  --df-fitting-algorithm <alg>  DF metric inversion: qr (default, most
                                robust) or matrixpower
  Note: --rys-scheme is intentionally not exposed — cuEST currently
  defines only one scheme, so the flag would have no effect.

Notes:
  Density fitting is required. Closed-shell RKS (multiplicity 1) and
  unrestricted UKS (multiplicity > 1) are supported. For UKS,
  `--break-symmetry` mixes the β HOMO/LUMO after the first diagonalization,
  but only when nalpha == nbeta (an artificially closed-shell guess); a
  genuinely spin-polarised system has no symmetry to break. Analytic
  gradients are available for both RKS and UKS (spherical orbitals).
  ECP data is auto-detected from the JSON basis.

Known-bad hybrid DF analytic gradients (cuEST library limitation — energy
still computed; refs skip storing grads for these):
  - All hybrids × {STO-3G, 6-31G} (SP-only orbital bases): DF JK derivative
    throws `CUEST_STATUS_EXCEPTION`.
  - Hybrid × 6-31G* for H2 (no D on H ⇒ SP-equivalent; wrong grads).
  - Hybrid × 6-31G* for H2O, HF, OH: DF JK derivative throws.

Available functionals:
  HF PBE B3LYP B3LYP5 PBE0 CAM-B3LYP WB97X-V WB97M-V
  HSE06 M06 M06-2X LC-WPBE LC-WPBEH WB97X
```

## Initial guess (SAD)

The SCF starts from a superposition of atomic densities. Each distinct element
is solved once as an isolated free atom under an explicit spherical-symmetry
constraint, cached to disk (`$CUEST_SAD_CACHE_DIR`, else
`~/.cache/cuest_dft/sad_guess`), and superposed at each atom's AO offset.

**Imposing spherical symmetry.** The constraint is applied by *projecting* the
atomic Fock onto the commutant of the AO rotation representation every
iteration:

```
P(X) = ∫dR U(R) X U(R)ᵀ
```

By Schur's lemma that commutant is exactly the l-block structure, so this is
precisely "minimise the energy over spherically symmetric densities" — but it
needs only the rotation *action* on the AOs, never an explicit l-decomposition.
That is what lets pure and Cartesian bases share one code path. A Cartesian
shell of angular momentum L spans L ⊕ L−2 ⊕ … and is therefore not a single
irrep, so the usual "reshape the l-block and average over m" trick (PySCF's
`AtomSphAverageRHF.eig`) cannot be applied to it at all.

Two details that are easy to get wrong:

- Cartesian AOs within a shell are **not orthonormal**, so `U(R)` is not
  orthogonal and operator matrices (`F`, `S`) and density matrices transform
  differently — `UᵀXU = X` versus `U D Uᵀ = D`. Both projectors exist; using one
  where the other belongs silently destroys the symmetry.
- The Euler-angle integral factorises (`U` is a homomorphism and
  `(A₁A₂)⊗(B₁B₂) = (A₁⊗B₁)(A₂⊗B₂)`), which turns the quadrature into two dense
  matmuls — ~15× faster for a QZ-basis transition metal.

**Fractional occupations.** After projection the eigenvalues are exactly
(2l+1)-fold degenerate, so a manifold's size identifies its l and whole
manifolds are occupied at equal fractional occupancy. Carbon's 2p comes out as
2/3 of an electron in each of px, py, pz rather than an arbitrary choice of
which two real p orbitals to fill. Occupations come from the reference
configurations in `atomic_config_table.inc` (PySCF's `NRSRHF_CONFIGURATION`,
from Phys. Rev. A **101**, 012516), which beat naive Aufbau for transition
metals.

The reference is stored as columns pre-scaled by √occupancy, so
`D = C diag(occ) Cᵀ = (C√occ)(C√occ)ᵀ` feeds cuEST's DF-K directly — that API
takes occupied MO coefficients, not an arbitrary density matrix.

**Atoms are HF by default**, matching PySCF (whose SAD uses atomic HF whatever
the molecular functional). That needs no XC grid and is functional-independent,
so one cached atom serves every functional. `--sad-functional-atoms` solves them
with the molecular functional instead; `--hcore-guess` disables SAD entirely.

## Validation

### SAD atomic reference vs PySCF

```bash
python3 test/validate_sad.py
```

Spherical-basis atoms reproduce PySCF's `AtomSphAverageRHF` to ~1e-12 Ha and
~1e-8 in every density-matrix element (same method, same DF auxiliary).

Two lower-level checks back that up:

```bash
./build/test_atomic_symmetry                     # GPU-free: rotation algebra
./build/probe_symmetry <basis.json> <Z> --cartesian   # vs a real cuEST overlap
```

`probe_symmetry` is the one that matters for conventions: a free atom's overlap
is spherically symmetric, so `U(R)ᵀ S U(R) = S` must hold exactly. That single
condition validates cuEST's Cartesian AO ordering *and* its per-shell
normalization *and* the rotation code at once, so a convention change fails
loudly instead of quietly degrading the guess.

Cartesian atoms deliberately do **not** match: PySCF forms its atom in a
spherical basis and maps it up with `cart2sph`, leaving the Cartesian
contaminant functions at exactly zero density, whereas here they are part of
the variational space. The result is a slightly *better* guess (energy lower by
~1e-4 Ha), so it is checked variationally — Cartesian energy ≤ spherical — not
against PySCF.

### Energy matrix vs PySCF-DF

```bash
python3 test/validate_cuest.py --functional PBE
```

### Density-fitted Hartree–Fock (no XC grid)

DF-HF isolates J/K / aux-basis error from XC quadrature. Use `--compare-dft`
to print |ΔE| summaries for HF vs PBE vs WB97X on the same cells:

```bash
python3 test/validate_hf.py --quick --compare-dft
python3 test/validate_hf.py
```

### Unrestricted UKS (open-shell)

```bash
python3 test/validate_uks.py            # full OH energy+grad sweep
python3 test/validate_uks.py --quick    # def2SVP smoke
# OH radical example:
./build/cuest_dft --xyz data/molecules/small/oh.xyz \
  --basis data/basis_sets/def2-svp.json \
  --aux-basis data/basis_sets/def2-universal-jkfit.json \
  --functional PBE --multiplicity 2 --analytic-gradient
```

### Gradients

```bash
python3 test/validate_full.py
```

### Energy (H2O equilibrium geometry, DF)

| Functional | cuEST (Ha) | PySCF-DF (Ha) | Diff (mHa) |
|---|---|---|---|
| PBE | -76.2721181 | -76.2721198 | 0.0017 |
| PBE0 | -76.2763001 | -76.2763014 | 0.0013 |
| B3LYP | -76.3581997 | -76.3582013 | 0.0016 |

All functionals agree within 0.0017 mHa (0.05 meV)

### Gradient

Analytical gradients use all 6 cuEST derivative APIs:
```
dE/dR = -nu + 2*(ke + po + pc) - 2*ov + df + xc
```
where `nu`=nuclear, `ke`=kinetic, `po`/`pc`=potential (basis/charge centers),
`ov`=overlap, `df`=DF Coulomb+exchange, `xc`=exchange-correlation.
Agreement with numerical finite-difference and PySCF reference gradients is < 1e-5 Ha/bohr for PBE, B3LYP, PBE0, and CAM-B3LYP.

## Architecture

```
cuest_wrapper/                 # Minimal C++ RAII over cuEST C API (STATIC)
├── include/cuest_wrapper/
│   ├── raii.hpp               # Handle / Workspace / Parameters RAII
│   ├── nvtx.hpp               # NvtxRange + CUEST_NVTX for Nsight timelines
│   ├── context.hpp            # cuEST context
│   ├── molecule.hpp           # Molecular geometry + Z_eff / charge
│   ├── constants.hpp          # Physical constants / unit helpers
│   ├── shell_norm.hpp         # Gaussian shell normalization
│   ├── basis.hpp              # AO / aux / ECP ownership (build_from_json in app)
│   ├── integrals.hpp          # 1e, DF-J/K, XC, ECP wrappers
│   ├── grid.hpp               # DFT integration grid (Becke+Ahlrichs)
│   └── gradients.hpp          # Analytical gradient APIs
└── src/
    └── integrals.cpp

src/                           # CLI / application layer
├── main.cpp                   # cuest_dft entry point
├── functionals.hpp            # CLI string ↔ XCBuilder::Functional registry
├── scf.hpp / scf.cpp          # SCF solver with DIIS + cuSOLVER
├── diis.hpp / diis.cpp        # Device Pulay DIIS
├── sad_guess.hpp / sad_guess.cpp  # SAD initial guess (cached atomic HF)
├── atomic_symmetry.hpp / .cpp # SO(3) commutant projector for free atoms
├── atomic_config.hpp / .cpp   # Reference configs + fractional occupations
├── dfjk_hybrid.hpp            # Hybrid/LRC DF-JK exchange fractions (queried from cuEST's XC plan)
├── basis_from_json.cpp        # BasisBuilder / AuxBasis from BSE JSON
├── basis_ecp_from_json.cpp    # ECPBuilder from BSE JSON
├── grad_numerical.hpp         # FD gradients via subprocess
└── io/
    ├── parsers.hpp            # XYZ file parsers
    └── basis_json.hpp         # BSE JSON reader

test/
├── common.py                  # Shared BSE→PySCF, parsers, runners
├── download_bases.py          # Fetch orbital/aux bases from BSE
├── generate_reference.py      # Build test/reference.json (PySCF-DF)
├── validate_cuest.py          # Compare cuEST vs reference.json
├── validate_sad.py            # Compare the SAD atomic reference vs PySCF
├── validate_energy.py         # Live side-by-side energy matrix
├── validate_full.py           # Gradient spot checks
└── test_water.py              # H2O smoke test
```
