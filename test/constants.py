"""
Physical constants matching include/cuest_wrapper/constants.hpp.
CODATA 2018 values. Single source of truth for Python scripts.
"""

# Bohr radius in angstrom: 1 bohr = 0.529177210903 Å
BOHR_PER_ANGSTROM = 0.529177210903

# Angstrom per bohr
ANGSTROM_PER_BOHR = 1.0 / BOHR_PER_ANGSTROM  # ≈ 1.8897259886

# Hartree to eV
HARTREE_PER_EV = 27.211386245988

# eV to Hartree
EV_PER_HARTREE = 1.0 / HARTREE_PER_EV
