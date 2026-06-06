"""
Globals will be declared here

"""
import numpy as np
from typing import List, Tuple

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
    'Ionic Radius A (Angstrom)',
    'Ionic Radius B (Angstrom)',
    'Electronegativity A',
    'Electronegativity B',
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

ROM_COLS = [
    "ROM_Lattice_Parameter",
    "ROM_Lattice_Distortion",
    "ROM_Ionic_Radius_A",
    "ROM_Ionic_Radius_B",
    "ROM_Radius_Ratio_rA_rB",
    "ROM_Electronegativity_A",
    "ROM_Electronegativity_B",
    "ROM_Electronegativity_Diff",
    "ROM_Lattice_Distortion_A",
    "ROM_Lattice_Distortion_B",
    "ROM_Bulk_Modulus_GPa",
    "ROM_Shear_Modulus_GPa",
    "ROM_Youngs_Modulus_GPa",
    "ROM_Poisson_Ratio",
    "ROM_Thermal_Conductivity_W_mK",
    "ROM_Thermal_Expansion",
]

ROM_LATT_FEAT_COLS = [
    "ROM_Lattice_Parameter",
    "ROM_Lattice_Distortion",
    "ROM_Ionic_Radius_A",
    "ROM_Ionic_Radius_B",
    # "ROM_Radius_Ratio_rA_rB",
    # "ROM_Electronegativity_A",
    "ROM_Electronegativity_B",
    "ROM_Electronegativity_Diff",
    # "ROM_Lattice_Distortion_A",
    # "ROM_Lattice_Distortion_B",
    "ROM_Bulk_Modulus_GPa",
    "ROM_Shear_Modulus_GPa",
    "ROM_Youngs_Modulus_GPa",
    "ROM_Poisson_Ratio",
    # "ROM_Thermal_Conductivity_W_mK",
    # "Temperature",
]
# ROM_LATT_FEAT_COLS = ROM_COLS + ['Temperature']

# ── Outlier Compositions ──────────────────────────────────────────────────
OUTLIER_COMPS = [
    'Y2ZrSnO7',
    'Eu0.02Y1.98Zr1Sn1O7',
    'La1.65Y0.35Ti2O7',
    'La1.7Y0.3Ti2O7',
    'La1.75Y0.25Ti2O7',
    'Gd2Zr0.3Ti1.7O7',
    'Y2Ti0.5Sn1.5O7',
]

# ── Physical property tables ──────────────────────────────────────────────────

# Shannon ionic radii (Å) — 8-coord for A-site, 6-coord for B-site
# taken from Shannon, R. D. “Revised Effective Ionic Radii and Systematic Studies of Interatomic Distances in Halides
# and Chalcogenides.” Acta Crystallographica Section A, vol. 32, no. 5, 1 Sept. 1976, pp. 751–767,
# https://doi.org/10.1107/s0567739476001551.
IONIC_RADII_8 = {
    'La': 1.160, 'Ce': 1.143, 'Pr': 1.126, 'Nd': 1.109, 'Sm': 1.079,
    'Eu': 1.066, 'Gd': 1.053, 'Tb': 1.040, 'Dy': 1.027, 'Ho': 1.015,
    'Er': 1.004, 'Tm': 0.994, 'Yb': 0.985, 'Lu': 0.977, 'Y':  1.019,
    # Additional A-site cations
    'Ac': 1.120,
    'Al': 0.675,
    'Am': 1.090,
    'Au': 1.020,
    'Bi': 1.170,
    'Bk': 1.010,
    'Cf': 1.010,
    'Cm': 1.025,
    'Co': 0.900,
    'Cr': 0.840,
    'Es': 1.000,
    'Fe': 0.920,
    'Fm': 0.990,
    'Ga': 0.762,
    'In': 1.100,
    'Ir': 1.000,
    'Lr': 0.970,
    'Md': 0.980,
    'No': 0.975,
    'Pm': 1.093,
    'Rh': 0.880,
    'Ru': 0.900,
    'Sc': 0.885,
    'Tl': 1.155,
}

IONIC_RADII_6 = {
    'Ti': 0.605, 'Zr': 0.720, 'Hf': 0.710, 'Sn': 0.690, 'Ir': 0.625,
    'Ce': 0.870, 'Nb': 0.640,
    # Additional B-site cations
    'Mn': 0.670,
    'Mo': 0.790,
    'Os': 0.630,
    'Pb': 0.785,
    'Pd': 0.755,
    'Pt': 0.740,
    'Pu': 0.900,
    'Re': 0.720,
    'Ru': 0.760,
    'Tc': 0.740,
    'Th': 1.050,
    'W': 0.740,
}

# Molar masses (g/mol)
# from https://iupac.qmul.ac.uk/AtWt/
MOLAR_MASSES = {
    'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24, 'Pm': 145.00,
    'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93, 'Dy': 162.50,
    'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.04, 'Lu': 174.97,
    'Y':   88.91, 'Ti':  47.87, 'Zr':  91.22, 'Hf': 178.49, 'Sn': 118.71,
    'Ir': 192.22, 'Nb':  92.91, 'O':   16.00,
    # Additional elements
    'Ac': 227.03,
    'Al': 26.98,
    'Am': 243.00,
    'Au': 196.97,
    'Bi': 208.98,
    'Bk': 247.00,
    'Cf': 251.00,
    'Cm': 247.00,
    'Co': 58.93,
    'Cr': 51.996,
    'Es': 252.00,
    'Fe': 55.845,
    'Fm': 257.00,
    'Ga': 69.723,
    'In': 114.818,
    'Lr': 262.00,
    'Md': 258.00,
    'Mn': 54.938,
    'Mo': 95.95,
    'No': 259.00,
    'Os': 190.23,
    'Pb': 207.2,
    'Pd': 106.42,
    'Pt': 195.085,
    'Pu': 244.00,
    'Re': 186.207,
    'Rh': 102.91,
    'Ru': 101.07,
    'Sc': 44.956,
    'Tc': 98.00,
    'Th': 232.04,
    'Tl': 204.38,
    'W': 183.84,
}

# Pauling electronegativity
# from Wikipedia
ELECTRONEGATIVITY = {
    'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14, 'Sm': 1.17,
    'Eu': 1.20, 'Gd': 1.20, 'Tb': 1.22, 'Dy': 1.23, 'Ho': 1.24,
    'Er': 1.24, 'Tm': 1.25, 'Yb': 1.10, 'Lu': 1.27, 'Y':  1.22,
    'Ti': 1.54, 'Zr': 1.33, 'Hf': 1.30, 'Sn': 1.96, 'Ir': 2.20,
    'Nb': 1.60,
    # Additional elements
    'Ac': 1.10,
    'Al': 1.61,
    'Am': 1.30,
    'Au': 2.54,
    'Bi': 2.02,
    'Bk': 1.30,
    'Cf': 1.30,
    'Cm': 1.30,
    'Co': 1.88,
    'Cr': 1.66,
    'Es': 1.30,
    'Fe': 1.83,
    'Fm': 1.30,
    'Ga': 1.81,
    'In': 1.78,
    'Lr': 1.30,
    'Md': 1.30,
    'Mn': 1.55,
    'Mo': 2.16,
    'No': 1.30,
    'Os': 2.20,
    'Pb': 2.33,
    'Pd': 2.20,
    'Pt': 2.28,
    'Pu': 1.28,
    'Re': 1.90,
    'Rh': 2.28,
    'Ru': 2.20,
    'Sc': 1.36,
    'Tc': 1.90,
    'Th': 1.30,
    'Tl': 1.62,
    'W': 2.36,
}

# Atomic numbers
ATOMIC_NUMBER = {
    'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Sm': 62, 'Eu': 63,
    'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69,
    'Yb': 70, 'Lu': 71, 'Y': 39, 'Ti': 22, 'Zr': 40, 'Hf': 72,
    'Sn': 50, 'Ir': 77, 'Nb': 41,
    # Additional elements
    'Ac': 89, 'Al': 13, 'Am': 95, 'Au': 79, 'Bi': 83, 'Bk': 97,
    'Cf': 98, 'Cm': 96, 'Co': 27, 'Cr': 24, 'Es': 99, 'Fe': 26,
    'Fm': 100, 'Ga': 31, 'In': 49, 'Lr': 103, 'Md': 101, 'Mn': 25,
    'Mo': 42, 'No': 102, 'Os': 76, 'Pb': 82, 'Pd': 46, 'Pt': 78,
    'Pu': 94, 'Re': 75, 'Rh': 45, 'Ru': 44, 'Sc': 21, 'Tc': 43,
    'Th': 90, 'Tl': 81, 'W': 74,
}