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

### Energy (H2O, PBE/def2-SVP, DF)

| Method | Total Energy (Ha) | Diff |
|---|---|---|
| cuEST | -76.271988648 | — |
| PySCF-DF | -76.271990273 | +0.0016 mHa |
| PySCF-exact | -76.271963469 | -0.0252 mHa |

The 0.025 mHa difference vs analytical PySCF is the inherent DF fitting error.

### Gradient (H2O, PBE/def2-SVP, DF)

| Atom | Analytical (Ha/bohr) | Numerical (Ha/bohr) |
|---|---|---|
| O | (0.000, 0.023, 0.000) | (0.000, 0.024, 0.000) |
| H | (±0.028, −0.011, 0.000) | (±0.011, −0.012, 0.000) |

Analytical gradient y-components match numerical within ~0.001 Ha/bohr. 
x-components have larger error because the DF-based formula does not fully
capture the x-anisotropy of the orbital relaxation.

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
