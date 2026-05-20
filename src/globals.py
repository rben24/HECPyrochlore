"""
Globals will be declared here

"""
import numpy as np

# ── element sets ────────────────────────────────────────────────────────────

# Rare-earth / Y cations that occupy the 8-coordinated A-site
# pulled from pymatgen using is_metal() and common_oxidation_state of 3
KNOWN_A: frozenset[str] = frozenset({
    'Ac', 'Al', 'Am', 'Au', 'Bi', 'Bk', 'Ce', 'Cf', 'Cm', 'Co',
    'Cr', 'Dy', 'Er', 'Es', 'Eu', 'Fe', 'Fm', 'Ga', 'Gd', 'Ho',
    'In', 'Ir', 'La', 'Lr', 'Lu', 'Md', 'Nd', 'No', 'Pm', 'Pr',
    'Rh', 'Ru', 'Sc', 'Sm', 'Tb', 'Tl', 'Tm', 'Y', 'Yb'
})

# Transition-metal cations that occupy the 6-coordinated B-site
# pulled from pymatgen using is_metal() and common_oxidation_state of 4
KNOWN_B: frozenset[str] = frozenset({
    'Ce', 'Hf', 'Ir', 'Mn', 'Mo', 'Os', 'Pb', 'Pd', 'Pt', 'Pu',
    'Re', 'Ru', 'Sn', 'Tc', 'Th', 'Ti', 'W', 'Zr'
})

# # ── element sets (shared with load_icsd.py) ───────────────────────────────────
# KNOWN_A: frozenset = frozenset({
#     'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
#     'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Y',
# })
# KNOWN_B: frozenset = frozenset({
#     'Ti', 'Zr', 'Hf', 'Sn', 'Ir', 'Nb',
# })

# Ce can sit on either site depending on oxidation state; handled separately
CE_AMBIGUOUS = 'Ce'

# Pyrochlore structure-type identifiers used in the ICSD file
PYROCHLORE_STRUCTURE_TYPES: frozenset[str] = frozenset({
    'Ca2Nb2O7',   # standard Fd-3m pyrochlore prototype
    'Eu2Zr2O7',   # alternate ICSD label for the same structure
    'Bi2Ti2O7',
})

# Physical bounds
LATTICE_MIN = 9.5    # Å
LATTICE_MAX = 11.5   # Å
TEMP_MIN    = 285.0  # K
TEMP_MAX    = 305.0  # K
A_STOICH_RANGE = (1.5, 2.5)
B_STOICH_RANGE = (1.5, 2.5)

# Pyrochlore stability window for r_A/r_B (Shannon ionic radii)
# Outside this range → likely defect-fluorite or other polymorph
RA_RB_MIN = 1.40
RA_RB_MAX = 1.90

# ── Compound-type enum strings ───────────────────────────────────────────────

PRISTINE      = 'pristine'
HIGH_ENTROPY  = 'high_entropy'
NON_PYROCHLORE = 'non_pyrochlore'

# ── Constants ────────────────────────────────────────────────────────────────

K_B = 1.380649 * np.float_power(10, -23) # Boltzmann constant J/K
R_GAS = 8.314  # J/(mol·K)