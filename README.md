# cuEST DFT — GPU-Accelerated Quantum Chemistry

A C++ DFT package built on [NVIDIA cuEST](https://developer.nvidia.com/cuda/cuda-x-libraries/cuest) (v0.2.0) with RAII wrappers, analytic gradients, and PySCF-validation.

## Features

- **Full SCF**: RKS (Restricted Kohn-Sham) with density fitting (RI-JK)
- **DIIS**: Direct inversion in iterative subspace, configurable space and start
- **Gradients**: Analytic or numerical finite differences
- **Functionals**: PBE, B3LYP, B3LYP5, PBE0, CAM-B3LYP, ωB97X, ωB97X-V, ωB97M-V, HSE06, M06, M06-2X, LC-ωPBE, LC-ωPBEh
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

Default `test/reference.json` matrix: closed-shell molecules × {HF, PBE, WB97X}
× all bases × {spherical, Cartesian}, plus UKS OH (mult=2) × {HF, PBE, PBE0, WB97X}
× selected bases. SCF iteration counts included. Gradients are stored for
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
  --basis <path>           Primary basis set in BSE JSON format
  --aux-basis <path>       Auxiliary/DF (RI-J) basis set (BSE JSON)

Optional arguments:
  --functional <name>      XC functional (default: PBE)
  --radial-pts <n>         Radial grid points (default: 75)
  --angular-pts <n>        Angular Lebedev points (default: 302)
  --charge <int>           Total charge (default: 0)
  --multiplicity <int>     Spin multiplicity (default: 1; UKS if ≠ 1)
  --break-symmetry <rad>   UKS β HOMO/LUMO mix angle (default: 0.3)

SCF convergence options:
  --max-iter <n>           Max SCF iterations (default: 150)
  --conv-thresh <val>      RMS density convergence (default: 1e-8)
  --energy-conv <val>      Energy change convergence (default: 1e-8)
  --diis-start <n>         Iteration to enable DIIS (default: 1)
  --diis-space <n>         DIIS subspace dimension (default: 10)
  --damping <val>          Density damping factor (default: 0.0)

Other options:
  --quiet                  Minimal output
  --verbose                Verbose output
  --print-mos              Print final MO energies
  --gradient               Compute and print nuclear gradient
  --help                   Show this help

Notes:
  Density fitting is required. Closed-shell RKS (multiplicity 1) and
  unrestricted UKS (multiplicity > 1) are supported. For broken-symmetry
  singlets (nα = nβ), `--break-symmetry` mixes the β HOMO/LUMO on the
  initial guess; ordinary open-shell (nα ≠ nβ) skips that mix. Analytic
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

## Validation

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
├── scf.hpp / scf.cpp          # SCF solver with DIIS + cuSOLVER
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
├── validate_energy.py         # Live side-by-side energy matrix
├── validate_full.py           # Gradient spot checks
└── test_water.py              # H2O smoke test
```
