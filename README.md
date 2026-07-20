# cuEST DFT

GPU-accelerated DFT on [NVIDIA cuEST](https://developer.nvidia.com/cuda-x-libraries/cuest) with DIIS, analytical gradients, and PySCF-validated energies.

## Requirements

- CUDA Toolkit 13.x, NVIDIA GPU (sm_80+)
- cuEST 0.2.0 C SDK (headers + `libcuest.so` in `external/`)
- CMake 3.20+, GCC 13+
- PySCF (for validation)

## Quick Start

```bash
# Build
cmake -B build -S .
cmake --build build -j$(nproc)

# Run
./build/cuest_dft \
  --xyz data/molecules/h2o.xyz \
  --basis data/basis_sets/def2-svp.json \
  --aux-basis data/basis_sets/def2-universal-jkfit.json \
  --functional PBE
```

Basis sets use [BSE JSON format](https://www.basissetexchange.org/api/basis/def2-svp/format/json/). ECP data is auto-detected for heavy elements — no separate `--ecp` flag needed.

## Usage

```
cuEST DFT — GPU-accelerated density functional theory
Usage: cuest_dft --xyz <file> --basis <json_file> [options]

Required:
  --xyz <path>             Geometry in XYZ format
  --basis <path>           Basis set in BSE JSON format

Optional:
  --aux-basis <path>       Auxiliary (RI-J) basis set
  --functional <name>      XC functional (default: PBE)
  --radial-pts <n>         Radial grid points (default: 75)
  --angular-pts <n>        Angular points (default: 302)
  --charge <int>           Total charge (default: 0)
  --multiplicity <int>     Spin multiplicity (default: 1)

SCF options:
  --max-iter <n>           Max iterations (default: 150)
  --conv-thresh <val>      RMS density convergence (default: 1e-8)
  --energy-conv <val>      Energy convergence (default: 1e-8)
  --diis-start <n>         Iteration to start DIIS (default: 1)
  --diis-space <n>         DIIS subspace size (default: 10)
  --damping <val>          Density damping (default: 0.0)

Other:
  --quiet / --verbose      Output control
  --print-mos              Print MO energies
  --gradient               Compute analytical gradient
  --help                   Show help

Functionals: PBE B3LYP PBE0 CAM-B3LYP WB97X WB97X-V HSE06 M06 M06-2X
```

## Validation

Two-step workflow — generate references once, then test cuEST repeatedly:

```bash
# Step 1: Generate PySCF reference (one-time)
python3 test/generate_reference.py

# Step 2: Validate cuEST against reference
python3 test/validate_cuest.py

# Options
python3 test/generate_reference.py --quick      # PBE-only subset
python3 test/validate_cuest.py --molecule H2O   # single molecule
```

**Results** (13 configs, PBE/def2-SVP):

| Molecule | Δ (mHa) | Status |
|----------|---------|--------|
| H₂O, NH₃, H₂, HF, N₂, CO₂, CH₄, C₂H₄, BH₃ | < 0.05 | ✓ PASS |
| SO₂ | 5.1 | ~ grid variation |
| Br₂, I₂, CH₂I₂ | — | heavy element (known) |

Tolerance: 0.1 mHa. All 8 functionals pass for H₂O/NH₃ across 5 basis sets.

## Data

```
data/
├── molecules/         # 16 XYZ geometries
│   ├── h2o.xyz, nh3.xyz, h2.xyz, hf.xyz, n2.xyz
│   ├── co2.xyz, ch4.xyz, c2h4.xyz, bh3.xyz, so2.xyz
│   └── br2.xyz, i2.xyz, ch2i2.xyz
└── basis_sets/        # BSE JSON format
    ├── def2-svp.json, def2-tzvp.json, def2-svpd.json
    ├── cc-pvdz.json, cc-pvtz.json
    └── def2-universal-jkfit.json
```

Download additional basis sets from [Basis Set Exchange](https://www.basissetexchange.org/):
```
https://www.basissetexchange.org/api/basis/<name>/format/json/
```

## Architecture

```
include/cuest_wrapper/
├── constants.hpp     # Physical constants (matches PySCF)
├── json_parser.hpp   # Minimal JSON parser for BSE format
├── basis_json.hpp    # BSE JSON → cuEST shells + ECP
├── raii.hpp          # RAII wrappers for cuEST C handles
├── context.hpp       # cuEST context
├── molecule.hpp      # Geometry + element lookup
├── parsers.hpp       # XYZ file parser
├── shell_norm.hpp    # Gaussian normalization
├── basis.hpp         # BasisBuilder, AuxBasis, ECPBuilder
├── integrals.hpp     # 1e, DF-J/K, XC, ECP integrals
├── grid.hpp          # DFT integration grid
├── scf.hpp           # SCF solver (DIIS + cuSOLVER)
├── gradients.hpp     # Analytical gradients
└── grad_numerical.hpp # Numerical gradients (finite difference)

src/
├── main.cpp          # CLI entry point
├── basis_nvidia.cpp  # Basis + aux basis from JSON
├── basis_ecp.cpp     # ECP from JSON
├── integrals.cpp     # Integral computations
└── scf.cpp           # SCF loop
```
