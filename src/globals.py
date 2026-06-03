"""
Globals will be declared here

"""
import numpy as np

# ── element sets ────────────────────────────────────────────────────────────

# Rare-earth / Y cations that occupy the 8-coordinated A-site
# pulled from pymatgen using is_metal() and common_oxidation_state of 3
# **Also, there can possibly be metals with oxidation state 2+ in A site
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

# Some elements can sit on either site depending on oxidation state; handled separately
KNOWN_AMBIGUOUS = [
    'Ce', 'Ir', 'Sn', 'Ru',
]
CE_AMBIGUOUS = 'Ce'

# Pyrochlore structure-type identifiers used in the ICSD file
PYROCHLORE_STRUCTURE_TYPES: frozenset[str] = frozenset({
    'Ca2Nb2O7',   # standard Fd-3m pyrochlore prototype
    'Eu2Zr2O7',   # alternate ICSD label for the same structure
    'Bi2Ti2O7',
})

# Physical bounds
LATTICE_MIN = 8.0    # Å
LATTICE_MAX = 13.5   # Å
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
ROOM_TEMP = 300 # K (technically 298.15K by IUPAC)

# ── Column Names ─────────────────────────────────────────────────────────────

# Single Phase columns
PRISTINE_COLS = [
    'Composition',
    'Sample A',
    'Sample B',
    'Thermal Conductivity (W/m/K)',
    'Lattice Parameter (Angstrom)',
    'Density',
    'Energy per Atom',
    'Formation Energy per Atom',
    'Enthalpy',
    'Magnetic Moment',
    'Band Gap',
    'Band Gap Type',
    'Valence',
    'Bulk Modulus (VRH)',
    'Shear Modulus (VRH)',
    'Youngs Modulus (VRH)',
    'Poisson Ratio',
    'AEL Debye Temperature',
    'Temperature',
    'Thermal Expansion',
    'Energy Above Hull',
    'Synthesis Method',
    'compound_type',
    'data_source',
]

# High Entropy Columns
HEC_COLS = [
    'Composition',
    'Sample A',
    'Sample B',
    'a_stoich_json',
    'b_stoich_json',
    'Lattice Parameter (Angstrom)',
    'Thermal Conductivity (W/m/K)',
    'compound_type',
    'Temperature',
    'data_source',
]

# Canonical Columns
CANONICAL_COLS = [
    'Composition',
    'Sample A',
    'Sample B',
    'Thermal Conductivity (W/m/K)',
    'Lattice Parameter (Angstrom)',
    'Relative Density %',
    'Is Single Phase',
    'Synthesis Method',
    'data_source',
    'b_o_distance',
    'b_o_b_angle',
    'oxygen_param_x',
    'compound_type',
    'a_stoich_json',
    'b_stoich_json',
    'Temperature',
    'Thermal Expansion',
    'density_calc',
]