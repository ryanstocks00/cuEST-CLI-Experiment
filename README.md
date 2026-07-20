# cuEST DFT — GPU-Accelerated Quantum Chemistry

A C++ DFT package built on [NVIDIA cuEST](https://developer.nvidia.com/cuda/cuda-x-libraries/cuest) (v0.2.0) with RAII wrappers, DIIS convergence acceleration, analytical gradients, and PySCF-validated energies.

## Features

- **Full SCF**: RKS (Restricted Kohn-Sham) with density fitting (RI-J)
- **DIIS**: Direct inversion in iterative subspace, configurable space and start
- **Gradients**: Analytical (fast) and numerical (validated reference)
- **Functionals**: PBE, B3LYP, B3LYP5, PBE0, CAM-B3LYP, ωB97X, ωB97X-V, ωB97M-V, HSE06, M06, M06-2X, LC-ωPBE, LC-ωPBEh
- **ECP support**: Effective core potentials
- **C++ RAII**: Clean memory-safe wrappers around cuEST C API
- **Validated**: SCF energies match PySCF-DF to 0.0016 mHa

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
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES="80;90"
cmake --build build -j$(nproc)
```

### 3. Download Basis Sets

From [Basis Set Exchange](https://www.basissetexchange.org/) (Gaussian94/Psi4 format):
```bash
# Required files in data/:
#   def2-svp.gbs              — primary basis
#   def2-universal-jkfit.gbs  — auxiliary (RI-J) basis
#   def2-svp-ecp.gbs          — ECP (optional)
```

### 4. Run

```bash
./build/cuest_dft \
  --xyz data/h2o.xyz \
  --basis data/def2-svp.gbs \
  --aux-basis data/def2-universal-jkfit.gbs \
  --functional PBE \
  --gradient
```

## Usage

```
cuEST DFT — GPU-accelerated density functional theory
Usage: ./build/cuest_dft --xyz <file> --basis <gbs_file> [options]

Required arguments:
  --xyz <path>             Input geometry in XYZ format
  --basis <path>           Primary basis set in Gaussian94 format

Optional arguments:
  --aux-basis <path>       Auxiliary/DF (RI-J) basis set
  --ecp <path>             Effective core potential file
  --functional <name>      XC functional (default: PBE)
  --radial-pts <n>         Radial grid points (default: 75)
  --angular-pts <n>        Angular Lebedev points (default: 302)
  --charge <int>           Total charge (default: 0)
  --multiplicity <int>     Spin multiplicity (default: 1)

SCF convergence options:
  --max-iter <n>           Max SCF iterations (default: 150)
  --conv-thresh <val>      RMS density convergence (default: 1e-8)
  --energy-conv <val>      Energy change convergence (default: 1e-8)
  --diis-start <n>         Iteration to enable DIIS (default: 1)
  --diis-space <n>         DIIS subspace dimension (default: 10)
  --damping <val>          Density damping factor (default: 0.0)
  --level-shift <val>      Level shifting (default: 0.0)

Other options:
  --no-df                  Disable density fitting
  --no-pure                Use Cartesian (not spherical) functions
  --quiet                  Minimal output
  --verbose                Verbose output
  --print-mos              Print final MO energies
  --gradient               Compute and print nuclear gradient
  --help                   Show this help

Available functionals:
  PBE B3LYP B3LYP5 PBE0 CAM-B3LYP WB97X-V WB97M-V
  HSE06 M06 M06-2X LC-WPBE LC-WPBEH WB97X
```

## Validation

### Energy (H2O equilibrium geometry, DF)

| Functional | cuEST (Ha) | PySCF-DF (Ha) | Diff (mHa) |
|---|---|---|---|
| PBE | -76.272118134684 | -76.272119814659 | 0.0017 |
| PBE0 | -76.276300063073 | -76.276301390574 | 0.0013 |
| B3LYP | -76.358199745106 | -76.358201309091 | 0.0016 |

All functionals agree within 0.0017 mHa (0.05 meV) — essentially
machine-precision agreement given different grid implementations.

### Gradient

Analytical gradients use all 6 cuEST derivative APIs:
```
dE/dR = -nu + 2*(ke + po + pc) - 2*ov + df + xc
```
where `nu`=nuclear, `ke`=kinetic, `po`/`pc`=potential (basis/charge centers),
`ov`=overlap, `df`=DF Coulomb+exchange, `xc`=exchange-correlation.
The charge-center potential derivative (`pc`) is essential for translational invariance.
Agreement with numerical finite-difference and PySCF reference gradients is
< 1e-5 Ha/bohr for PBE, B3LYP, PBE0, and CAM-B3LYP.

## Architecture

```
include/cuest_wrapper/
├── raii.hpp          # C++ RAII wrappers for cuEST handles
├── context.hpp       # cuEST context management
├── molecule.hpp      # Molecular geometry + properties
├── parsers.hpp       # GBS, ECP, XYZ file parsers
├── shell_norm.hpp    # Gaussian basis normalization
├── basis.hpp         # BasisBuilder, AuxBasis, ECPBuilder
├── integrals.hpp     # 1e, DF-J, XC, ECP integral wrappers
├── grid.hpp          # DFT integration grid (Becke+Ahlrichs)
├── scf.hpp           # SCF solver with DIIS + cuSOLVER diagonalization
├── gradients.hpp     # Analytical gradient (6 cuEST derivative APIs)
└── grad_numerical.hpp # Numerical gradient (finite differences)

src/
├── main.cpp          # CLI entry point (25+ parameters)
├── basis_nvidia.cpp  # BasisBuilder using NVIDIA formAOShells helper
├── basis_ecp.cpp     # ECPBuilder implementation
├── integrals.cpp     # Integral computation implementations
└── scf.cpp           # SCF loop with cuSOLVER dsygvd
```
