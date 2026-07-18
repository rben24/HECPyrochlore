"""
Globals are declared here
Includes elements for pyrochlore sites, elementary constants
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from pymatgen.core import Element, Composition
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT = _HERE.parent

# ── element sets ────────────────────────────────────────────────────────────

# Rare-earth / Y cations that occupy the 8-coordinated A-site
# pulled from pymatgen using is_metal() and common_oxidation_state of 3 or 2
# **Also, there can possibly be metals with oxidation state 2+ in A site
# KNOWN_A_3 = [ #: frozenset[str] = frozenset({
#     'Ac', 'Al', 'Am', 'Au', 'Bi', 'Bk', 'Ce', 'Cf', 'Cm', 'Co',
#     'Cr', 'Dy', 'Er', 'Es', 'Eu', 'Fe', 'Fm', 'Ga', 'Gd', 'Ho',
#     'In', 'Ir', 'La', 'Lr', 'Lu', 'Md', 'Nd', 'No', 'Pm', 'Pr',
#     'Sc', 'Sm', 'Tb', 'Tl', 'Tm', 'Y', 'Yb'
# ] # })
#
# KNOWN_A_2 = [
#     'Ba', 'Be', 'Ca', 'Cd', 'Co', 'Cu', 'Eu', 'Fe', 'Hg', 'Mg',
#     'Mn', 'Ni', 'Pb', 'Pd', 'Pt', 'Ra', 'Sn', 'Sr', 'Zn',
# ]
#
# # Transition-metal cations that occupy the 6-coordinated B-site
# # pulled from pymatgen using is_metal() and common_oxidation_state of 4 or 5
# KNOWN_B_4 = [ #: frozenset[str] = frozenset({
#     'Ce', 'Hf', 'Ir', 'Mn', 'Mo', 'Os', 'Pb', 'Pd', 'Pt', 'Pu',
#     'Re', 'Rh', 'Ru', 'Sn', 'Ta', 'Tc', 'Th', 'Ti', 'W', 'Zr'
# ] #})
#
# KNOWN_B_5 = [
#     'Nb', 'Np', 'Pa', 'Ta', 'V',
# ]

# Some elements can sit on either site depending on oxidation state; handled separately
# KNOWN_AMBIGUOUS = [
#     'Ce', 'Ir', 'Sn', 'Ru', 'Rh', 'Pb', 'Mn', #'Ta'
# ]

# Pure A-site only (do NOT appear in any B list)
KNOWN_A_3_ONLY = [
    'Ac', 'Al', 'Am', 'Au', 'Bi', 'Bk', 'Cf', 'Cm',
    'Cr', 'Dy', 'Er', 'Es', 'Fm', 'Ga', 'Gd', 'Ho',
    'In', 'La', 'Lr', 'Lu', 'Md', 'Nd', 'No', 'Pm', 'Pr',
    'Sc', 'Sm', 'Tb', 'Tl', 'Tm', 'Y', 'Yb',
]

KNOWN_A_2_ONLY = [
    'Ba', 'Be', 'Ca', 'Cd', 'Cu', 'Hg', 'Mg',
    'Ni', 'Ra', 'Sr', 'Zn',
]

# Pure B-site only (do NOT appear in any A list)
KNOWN_B_4_ONLY = [
    'Hf', 'Mo', 'Re', 'Ta', 'Tc', 'Th', 'Ti', 'W', 'Zr',
    'Rh', 'Ru', 'Os', 'Te',
]

KNOWN_B_5_ONLY = [
    'Nb', 'Np', 'Pa', 'Ta', 'V',
]

# Structure: element → list of (oxidation_state, site_type) pairs
# site_type: 'A2' = A-site +2, 'A3' = A-site +3, 'B4' = B-site +4, 'B5' = B-site +5
KNOWN_AMBIGUOUS_SMALL = {
    'Ce': [3, 4],      # ±3 or ±4
    'Co': [3, 2],
    'Eu': [3, 2],
    'Fe': [3, 2],
    'Ge': [2, 4],
    'Ir': [3, 4],
    'Mn': [4, 2],
    'Pb': [4, 2],
    'Pd': [4, 2],
    'Pt': [4, 2],
    # 'Rh': [3, 4],
    # 'Ru': [3, 4],
    'Sn': [4, 2],
    'Ta': [5, 4],  # Prefers +5 as B-site
    'V':  [2, 3, 4, 5],  # Ultra-ambiguous
    # 'Mo': [('B', 4), ('B', 5), ('A', 3)],  # Mostly B-site
    # 'W':  [('B', 4), ('B', 5), ('A', 3)],  # Mostly B-site
}

KNOWN_AMBIGUOUS = {
    'Ce': [('A', 3), ('B', 4)],      # ±3 or ±4
    'Co': [('A', 3), ('A', 2)],
    'Eu': [('A', 3), ('A', 2)],
    'Fe': [('A', 3), ('A', 2)],
    'Ge': [('A', 2), ('B', 4)],
    'Ir': [('A', 3), ('B', 4)],
    'Mn': [('B', 4), ('A', 2)],
    'Pb': [('B', 4), ('A', 2)],
    'Pd': [('B', 4), ('A', 2)],
    'Pt': [('B', 4), ('A', 2)],
    # 'Rh': [('A', 3), ('B', 4)],
    # 'Ru': [('A', 3), ('B', 4)],
    'Sn': [('B', 4), ('A', 2)],
    'Ta': [('B', 5), ('B', 4)],  # Prefers +5 as B-site
    'V':  [('A', 2), ('A', 3), ('B', 4), ('B', 5)],  # Ultra-ambiguous
    # 'Mo': [('B', 4), ('B', 5), ('A', 3)],  # Mostly B-site
    # 'W':  [('B', 4), ('B', 5), ('A', 3)],  # Mostly B-site
}

# Fallback hardcoded sets (for elements not matching standard oxidation patterns)
KNOWN_A = set(KNOWN_A_3_ONLY) | set(KNOWN_A_2_ONLY) | set(KNOWN_AMBIGUOUS_SMALL.keys())
KNOWN_B = set(KNOWN_B_4_ONLY) | set(KNOWN_B_5_ONLY) | set(KNOWN_AMBIGUOUS_SMALL.keys())

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
NA = 6.02214076e23 # Avogadro's number


# ── Column Names ─────────────────────────────────────────────────────────────

# Single Phase columns
PRISTINE_COLS = [
    'Composition',
    'Sample A',
    'Sample B',
    'Oxidation State A',
    'Oxidation State B',
    'Thermal Conductivity (W/m/K)',
    'Lattice Parameter (Å)',
    'Vickers Hardness (GPa)',
    'CTE (K^-1)',
    'Relative Density %',
    'Density Measured',
    'Density Calculated',
    'Energy per Atom',
    'Formation Energy per Atom',
    'Enthalpy',
    'Activation Energy (eV)',
    'Magnetic Moment',
    'Band Gap',
    'Band Gap Type',
    'Valence',
    'Bulk Modulus (GPa)',
    'Shear Modulus (GPa)',
    'Youngs Modulus (GPa)',
    'Poisson Ratio',
    'Fracture Toughness (Mpa*m^.5)',
    'Specific Heat (Jg^−1K^−1)',
    'Thermal Diffusivity  (mm² s⁻¹)',
    'High Temp CTE',
    'AEL Debye Temperature',
    'Temperature (K)',
    'Energy Above Hull',
    'Ionic Radius A (Å)',
    'Ionic Radius B (Å)',
    'rA/rB (Å)',
    'Size disorder (δ %)',
    'Porosity (%)',
    'Grain Size (μm)',
    'Electronegativity A',
    'Electronegativity B',
    'oxygen_param_x',
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
    'Lattice Parameter (Å)',
    'Thermal Conductivity (W/m/K)',
    'Vickers Hardness (GPa)',
    'CTE (K^-1)', # °K⁻¹
    'Relative Density %',
    'Density Measured',
    'Density Calculated',
    'Bulk Modulus (GPa)',
    'Shear Modulus (GPa)',
    'Youngs Modulus (GPa)',
    'Poisson Ratio',
    'Fracture Toughness (Mpa*m^.5)',
    'Specific Heat (Jg^−1K^−1)', # Jg⁻¹K⁻¹
    'Thermal Diffusivity  (mm² s⁻¹)',
    'High Temp CTE',
    'Temperature (K)',
    'Ionic Radius A (Å)',
    'Ionic Radius B (Å)',
    'oxygen_param_x',
    'rA/rB (Å)',
    'Size disorder (δ %)',
    'Porosity (%)',
    'Grain Size (μm)',
    'compound_type',
    'data_source',
]

# Canonical Columns
CANONICAL_COLS = [
    'Composition',
    'Sample A',
    'Sample B',
    'Thermal Conductivity (W/m/K)',
    'Lattice Parameter (Å)',
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
    "ROM_Vickers_Hardness",
    "ROM_Fracture_Toughness",
    "ROM_Specific_Heat",
    "ROM_Thermal_Diffusivity",
]

ROM_LATT_FEAT_COLS = [
    "Thermal Conductivity (W/m/K)"
    'Vickers Hardness (GPa)',
    'CTE (K^-1)',  # °K⁻¹
    'Relative Density %',
    'Density Measured',
    'Density Calculated',
    'Bulk Modulus (GPa)',
    'Shear Modulus (GPa)',
    'Youngs Modulus (GPa)',
    'Poisson Ratio',
    'Fracture Toughness (Mpa*m^.5)',
    'Specific Heat (Jg^−1K^−1)',  # Jg⁻¹K⁻¹
    'Thermal Diffusivity  (mm² s⁻¹)',
    'High Temp CTE',
    'Temperature (K)',
    'Ionic Radius A (Å)',
    'Ionic Radius B (Å)',
    'oxygen_param_x',
    'rA/rB (Å)',
    'Size disorder (δ %)',
    'Porosity (%)',
    'Grain Size (μm)',
    "ROM_Lattice_Parameter",
    "ROM_Lattice_Distortion",
    "ROM_Ionic_Radius_A",
    "ROM_Ionic_Radius_B",
    # "ROM_Radius_Ratio_rA_rB",
    "ROM_Electronegativity_A",
    "ROM_Electronegativity_B",
    # "ROM_Electronegativity_Diff",
    "ROM_Lattice_Distortion_A",
    "ROM_Lattice_Distortion_B",
    "ROM_Bulk_Modulus_GPa",
    "ROM_Shear_Modulus_GPa",
    "ROM_Youngs_Modulus_GPa",
    "ROM_Poisson_Ratio",
    # "ROM_Thermal_Conductivity_W_mK",
    'ROM_Vickers_Hardness',
    'ROM_Fracture_Toughness',
    'ROM_Specific_Heat',
    'ROM_Thermal_Diffusivity',
]
# ROM_LATT_FEAT_COLS = ROM_COLS + HEC_COLS

ROM_THERM_COND_FEAT_COLS = [
    'Lattice Parameter (Å)',
    'Vickers Hardness (GPa)',
    'CTE (K^-1)',  # °K⁻¹
    'Relative Density %',
    'Density Measured',
    'Density Calculated',
    'Bulk Modulus (GPa)',
    'Shear Modulus (GPa)',
    'Youngs Modulus (GPa)',
    'Poisson Ratio',
    'Fracture Toughness (Mpa*m^.5)',
    'Specific Heat (Jg^−1K^−1)',  # Jg⁻¹K⁻¹
    'Thermal Diffusivity  (mm² s⁻¹)',
    'High Temp CTE',
    'Temperature (K)',
    'Ionic Radius A (Å)',
    'Ionic Radius B (Å)',
    'oxygen_param_x',
    'rA/rB (Å)',
    'Size disorder (δ %)',
    'Porosity (%)',
    'Grain Size (μm)',
    "ROM_Lattice_Parameter",
    "ROM_Lattice_Distortion",
    # "ROM_Ionic_Radius_A",
    # "ROM_Ionic_Radius_B",
    # "ROM_Radius_Ratio_rA_rB",
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
    'ROM_Vickers_Hardness',
    'ROM_Fracture_Toughness',
    'ROM_Specific_Heat',
    'ROM_Thermal_Diffusivity',
]
# ROM_THERM_COND_FEAT_COLS = ROM_COLS + HEC_COLS

ROM_HARDNESS_FEAT_COLS = [
    'Lattice Parameter (Å)',
    'Thermal Conductivity (W/m/K)',
    'CTE (K^-1)',  # °K⁻¹
    'Relative Density %',
    'Density Measured',
    'Density Calculated',
    'Bulk Modulus (GPa)',
    'Shear Modulus (GPa)',
    'Youngs Modulus (GPa)',
    'Poisson Ratio',
    'Fracture Toughness (Mpa*m^.5)',
    'Specific Heat (Jg^−1K^−1)',  # Jg⁻¹K⁻¹
    'Thermal Diffusivity  (mm² s⁻¹)',
    'High Temp CTE',
    'Temperature (K)',
    'Ionic Radius A (Å)',
    'Ionic Radius B (Å)',
    'oxygen_param_x',
    'rA/rB (Å)',
    'Size disorder (δ %)',
    'Porosity (%)',
    'Grain Size (μm)',
    "ROM_Lattice_Parameter",
    "ROM_Lattice_Distortion",
    # "ROM_Ionic_Radius_A",
    # "ROM_Ionic_Radius_B",
    # "ROM_Radius_Ratio_rA_rB",
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
    'ROM_Vickers_Hardness',
    'ROM_Fracture_Toughness',
    'ROM_Specific_Heat',
    'ROM_Thermal_Diffusivity',
]
# ROM_HARDNESS_FEAT_COLS = ROM_COLS + HEC_COLS

ROM_CTE_FEAT_COLS = [
    'Lattice Parameter (Å)',
    'Thermal Conductivity (W/m/K)'
    'Vickers Hardness (GPa)',
    'CTE (K^-1)',  # °K⁻¹
    'Relative Density %',
    'Density Measured',
    'Density Calculated',
    'Bulk Modulus (GPa)',
    'Shear Modulus (GPa)',
    'Youngs Modulus (GPa)',
    'Poisson Ratio',
    'Fracture Toughness (Mpa*m^.5)',
    'Specific Heat (Jg^−1K^−1)',  # Jg⁻¹K⁻¹
    'Thermal Diffusivity  (mm² s⁻¹)',
    'High Temp CTE',
    'Temperature (K)',
    'Ionic Radius A (Å)',
    'Ionic Radius B (Å)',
    'oxygen_param_x',
    'rA/rB (Å)',
    'Size disorder (δ %)',
    'Porosity (%)',
    'Grain Size (μm)',
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
    'ROM_Vickers_Hardness',
    'ROM_Fracture_Toughness',
    'ROM_Specific_Heat',
    'ROM_Thermal_Diffusivity',
]
# ROM_CTE_FEAT_COLS = ROM_COLS + HEC_COLS

ELEMENT_COLS = [
    'Element',
    'Atomic Mass',
    'Electronegativity',
    'Atomic Radius',
    'Metallic Radius',
    'Melting Point',
    'Thermal Expansion Coeff',
    'Thermal Conductivity',
    'Vickers Hardness',
    'Bulk Modulus',
    'Youngs Modulus',
    'Shear Modulus',
    'Poissons Ratio',
]

# ── Outlier Compositions ──────────────────────────────────────────────────
OUTLIER_COMPS = [
    'Y2ZrSnO7',                     # over 2 * stddev
    'Eu0.02Y1.98Zr1Sn1O7',
    'La1.65Y0.35Ti2O7',             # over 3 * stddev
    'La1.7Y0.3Ti2O7',               # over 3 * stddev
    'La1.75Y0.25Ti2O7',             # over 3 * stddev
    'Gd2Zr0.3Ti1.7O7',              # over 3 * stddev
    'Y2Ti0.5Sn1.5O7',               # over 2 * stddev
    'LaYTiZr TF',
    'La0.35Y1.65Ti2O7',
    'La0.3Y1.7Ti2O7',
    'La0.25Y1.75Ti2O7',
    'La0.2Y1.8Ti2O7',
    'Yb0.1Y1.9Sn2O7',
]

# ── Physical property tables ──────────────────────────────────────────────────

# Shannon ionic radii (Å) — 8-coord for A-site, 6-coord for B-site
# taken from Shannon, R. D. “Revised Effective Ionic Radii and Systematic Studies of Interatomic Distances in Halides
# and Chalcogenides.” Acta Crystallographica Section A, vol. 32, no. 5, 1 Sept. 1976, pp. 751–767,
# https://doi.org/10.1107/s0567739476001551.
IONIC_RADII_8_3 = {
    'La': 1.160, 'Ce': 1.143, 'Pr': 1.126, 'Nd': 1.109, 'Sm': 1.079,
    'Eu': 1.066, 'Gd': 1.053, 'Tb': 1.040, 'Dy': 1.027, 'Ho': 1.015,
    'Er': 1.004, 'Tm': 0.994, 'Yb': 0.985, 'Lu': 0.977, 'Y':  1.019,
    # Additional A-site cations
    'Am': 1.090,
    'Bi': 1.170,
    'Fe': 0.780,
    'In': 0.920,
    'Pm': 1.093,
    'Sc': 0.870,
    'Tl': 0.980,
}

IONIC_RADII_8_2 = {
    'Ba': 1.420, 'Ca': 1.120, 'Cd': 1.100, 'Co': 0.900, 'Eu': 1.250,
    'Fe': 0.920, 'Hg': 1.140, 'Mg': 0.890, 'Mn': 0.960, 'Pb': 1.290,
    'Ra': 1.480, 'Sr': 1.260, 'Zn': 0.900,
}

IONIC_RADII_6_4 = {
    'Ti': 0.605, 'Zr': 0.720, 'Hf': 0.710, 'Sn': 0.690, 'Ir': 0.625,
    'Ce': 0.870, 'Nb': 0.680,
    # Additional B-site cations
    'Ge': 0.530,
    'Mn': 0.530,
    'Mo': 0.650,
    'Os': 0.630,
    'Pb': 0.775,
    'Pd': 0.615,
    'Pt': 0.625,
    'Pu': 0.860,
    'Re': 0.630,
    'Rh': 0.600,
    'Ru': 0.620,
    'Si': 0.400,
    'Ta': 0.680,
    'Tc': 0.645,
    'Te': 0.970,
    'Th': 0.940,
    'W': 0.660,
}

IONIC_RADII_6_5 = {
    'Ir': 0.570, 'Nb': 0.640, 'Np': 0.750, 'Os': 0.575, 'Pa': 0.780, 'Re': 0.580, 'Ru': 0.565,
    'Ta': 0.640, 'Tc': 0.600,
    'V': 0.540,
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
    'Ge': 72.630,
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
    'Ta': 180.947,
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
    'Ca': 1.00,
    'Cd': 1.69,
    'Cf': 1.30,
    'Cm': 1.30,
    'Co': 1.88,
    'Cr': 1.66,
    'Es': 1.30,
    'Fe': 1.83,
    'Fm': 1.30,
    'Ga': 1.81,
    'Ge': 2.01,
    'Hg': 2.00,
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
    'Ta': 1.50,
    'Tc': 1.90,
    'Te': 2.10,
    'Th': 1.30,
    'Tl': 1.62,
    'W': 2.36,
}

# Atomic numbers
ATOMIC_NUMBER = {
    'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Sm': 62, 'Eu': 63,
    'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69,
    'Yb': 70, 'Lu': 71, 'Y': 39, 'Ti': 22, 'Zr': 40, 'Hf': 72, 'Ta': 73,
    'Sn': 50, 'Ir': 77, 'Nb': 41,
    # Additional elements
    'Ac': 89, 'Al': 13, 'Am': 95, 'Au': 79, 'Bi': 83, 'Bk': 97,
    'Cf': 98, 'Cm': 96, 'Co': 27, 'Cr': 24, 'Es': 99, 'Fe': 26,
    'Fm': 100, 'Ga': 31, 'Ge': 32, 'In': 49, 'Lr': 103, 'Md': 101, 'Mn': 25,
    'Mo': 42, 'No': 102, 'Os': 76, 'Pb': 82, 'Pd': 46, 'Pt': 78,
    'Pu': 94, 'Re': 75, 'Rh': 45, 'Ru': 44, 'Sc': 21, 'Tc': 43,
    'Th': 90, 'Tl': 81, 'W': 74,
}

# =============================================================================
# ELEMENTAL MECHANICAL & THERMAL PROPERTIES
# Sources: user-provided CSV (images) + standard reference (Haynes, CRC 2023)
# K  = bulk modulus  (GPa)
# E  = Young's modulus (GPa)
# nu = Poisson ratio (dimensionless)
# G  = shear modulus (GPa); computed as E/(2*(1+nu)) if absent
# k  = thermal conductivity (W/m·K)
# cte= thermal expansion coefficient (K⁻¹)
# =============================================================================
ELEM_PROPS: Dict[str, Dict] = {
    # ── A-site lanthanides + Y ────────────────────────────────────────────────
    'La': {'K':  27.9, 'E':  36.6, 'nu': 0.280, 'k': 13.0, 'cte': 1.21e-5},
    'Ce': {'K':  21.5, 'E':  33.6, 'nu': 0.248, 'k': 11.0, 'cte': 6.30e-6},
    'Pr': {'K':  28.8, 'E':  37.3, 'nu': 0.280, 'k': 13.0, 'cte': 6.70e-6},
    'Nd': {'K':  32.0, 'E':  41.0, 'nu': 0.280, 'k': 17.0, 'cte': 9.60e-6},
    'Pm': {'K':  33.0, 'E':  46.0, 'nu': 0.280, 'k': 15.0, 'cte': 1.10e-5},
    'Sm': {'K':  38.0, 'E':  50.0, 'nu': 0.270, 'k': 13.0, 'cte': 1.27e-5},
    'Eu': {'K':   8.3, 'E':  18.2, 'nu': 0.152, 'k': 14.0, 'cte': 3.50e-5},
    'Gd': {'K':  37.9, 'E':  54.8, 'nu': 0.259, 'k': 11.0, 'cte': 9.40e-6},
    'Tb': {'K':  38.7, 'E':  55.7, 'nu': 0.261, 'k': 11.0, 'cte': 1.03e-5},
    'Dy': {'K':  40.5, 'E':  61.4, 'nu': 0.247, 'k': 11.0, 'cte': 9.90e-6},
    'Ho': {'K':  40.2, 'E':  64.8, 'nu': 0.231, 'k': 16.0, 'cte': 1.12e-5},
    'Er': {'K':  44.4, 'E':  69.9, 'nu': 0.237, 'k': 15.0, 'cte': 1.22e-5},
    'Tm': {'K':  45.0, 'E':  74.0, 'nu': 0.213, 'k': 17.0, 'cte': 1.33e-5},
    'Yb': {'K':  30.5, 'E':  23.9, 'nu': 0.207, 'k': 39.0, 'cte': 2.63e-5},
    'Lu': {'K':  47.6, 'E':  68.6, 'nu': 0.261, 'k': 16.0, 'cte': 9.90e-6},
    'Y':  {'K':  41.2, 'E':  63.5, 'nu': 0.243, 'k': 17.0, 'cte': 1.06e-5},
    'Sc': {'K':  56.6, 'E':  74.4, 'nu': 0.279, 'k': 16.0, 'cte': 1.02e-5},
    'Bi': {'K':  31.0, 'E':  32.0, 'nu': 0.330, 'k':  8.0, 'cte': 1.34e-5},
    'Tl': {'K':  43.0, 'E':   8.0, 'nu': 0.450, 'k': 46.0, 'cte': 2.99e-5},
    'In': {'K':  39.0, 'E':  11.0, 'nu': 0.450, 'k': 82.0, 'cte': 3.21e-5},
    'Am': {'K':  None, 'E':  None, 'nu':  None, 'k': 10.0, 'cte':  None},
    'Pu': {'K':  None, 'E':  96.0, 'nu': 0.210, 'k':  6.0, 'cte':  None},
    'Th': {'K':  54.0, 'E':  79.0, 'nu': 0.270, 'k': 54.0, 'cte': 1.10e-5},
    'Cm': {'K':  None, 'E':  None, 'nu':  None, 'k':  8.8, 'cte':  None},
    # ── B-site transition metals ───────────────────────────────────────────────
    'Ti': {'K': 110.0, 'E': 116.0, 'nu': 0.320, 'k': 22.0, 'cte': 8.60e-6},
    'Zr': {'K':  94.0, 'E':  68.0, 'nu': 0.340, 'k': 23.0, 'cte': 5.70e-6},
    'Hf': {'K': 110.0, 'E':  78.0, 'nu': 0.370, 'k': 23.0, 'cte': 5.90e-6},
    'Sn': {'K':  58.0, 'E':  50.0, 'nu': 0.360, 'k': 67.0, 'cte': 2.20e-5},
    'Ir': {'K': 320.0, 'E': 528.0, 'nu': 0.260, 'k':150.0, 'cte': 6.40e-6},
    'Nb': {'K': 170.0, 'E': 105.0, 'nu': 0.400, 'k': 54.0, 'cte': 7.30e-6},
    'Ta': {'K': 200.0, 'E': 186.0, 'nu': 0.340, 'k': 57.0, 'cte': 6.30e-6},
    'Mo': {'K': 230.0, 'E': 329.0, 'nu': 0.310, 'k':139.0, 'cte': 4.80e-6},
    'W':  {'K': 310.0, 'E': 411.0, 'nu': 0.280, 'k':170.0, 'cte': 4.50e-6},
    'Re': {'K': 370.0, 'E': 463.0, 'nu': 0.300, 'k': 48.0, 'cte': 6.20e-6},
    'Ru': {'K': 220.0, 'E': 447.0, 'nu': 0.300, 'k':120.0, 'cte': 6.40e-6},
    'Rh': {'K': 380.0, 'E': 275.0, 'nu': 0.260, 'k':150.0, 'cte': 8.20e-6},
    'Os': {'K': 462.0, 'E': 586.0, 'nu': 0.250, 'k': 88.0, 'cte': 5.10e-6},
    'Pd': {'K': 180.0, 'E': 121.0, 'nu': 0.390, 'k': 72.0, 'cte': 1.18e-5},
    'Pt': {'K': 230.0, 'E': 168.0, 'nu': 0.380, 'k': 72.0, 'cte': 8.80e-6},
    'Mn': {'K': 120.0, 'E': 198.0, 'nu': 0.240, 'k':  7.8, 'cte': 2.17e-5},
    'Pb': {'K':  46.0, 'E':  16.0, 'nu': 0.440, 'k': 35.0, 'cte': 2.89e-5},
    'Tc': {'K':  None, 'E':  None, 'nu':  None, 'k': 51.0, 'cte':  None},
}

# ── Helper functions ────────────────────────────────────────────────────────────

def get_pymatgen_oxi_preference(element_str: str) -> List[int]:
    """Return pymatgen's common oxidation states in order of prevalence."""
    try:
        return list(Element(element_str).common_oxidation_states)
    except Exception:
        return []


def _infer_b_oxi_from_b_comp(b_comp: Dict[str, float]) -> Optional[int]:
    """
    Infer the B-site oxidation state from already-assigned B-site elements.

    Rules
    -----
    - If all present elements are in KNOWN_B_4_ONLY → +4
    - If all present elements are in KNOWN_B_5_ONLY → +5
    - If mixed, return None (caller must handle)
    """
    b4 = sum(1 for e in b_comp if e in KNOWN_B_4_ONLY)
    b5 = sum(1 for e in b_comp if e in KNOWN_B_5_ONLY)
    if b4 > 0 and b5 == 0:
        return 4
    if b5 > 0 and b4 == 0:
        return 5
    return None  # mixed or unknown


def resolve_ambiguous_element(
        elem: str,
        a_oxidation_state: Optional[int],
        required_b_oxi: Optional[int],
        a_comp: Dict[str, float],
        b_comp: Dict[str, float],
) -> Tuple[str, Optional[int]]:
    """
    Resolve which site an ambiguous element belongs to.

    Returns
    -------
    site      : 'A' or 'B'
    oxi_state : oxidation state for that site
    """
    possible_sites = KNOWN_AMBIGUOUS[elem]

    # ── PRIORITY 1: Match already-known A or B oxidation state ──────────────
    if a_oxidation_state is not None:
        matching_a = [oxi for site, oxi in possible_sites
                      if site == 'A' and oxi == a_oxidation_state]
        if matching_a:
            return 'A', a_oxidation_state

        if required_b_oxi is not None:
            matching_b = [oxi for site, oxi in possible_sites
                          if site == 'B' and oxi == required_b_oxi]
            if matching_b:
                return 'B', required_b_oxi

    if required_b_oxi is not None:
        matching_b = [oxi for site, oxi in possible_sites
                      if site == 'B' and oxi == required_b_oxi]
        if matching_b:
            return 'B', required_b_oxi

    # ── PRIORITY 2: Placement guided by which sites are already occupied ─────
    if a_comp:
        # A-site already has members → prefer B-site to avoid double-counting
        b_options = [oxi for site, oxi in possible_sites if site == 'B']
        if b_options:
            return 'B', b_options[0]
        a_options = [oxi for site, oxi in possible_sites if site == 'A']
        if a_options:
            return 'A', a_options[0]

    if b_comp:
        # B-site already has members
        b_options = [oxi for site, oxi in possible_sites if site == 'B']
        if b_options:
            return 'B', b_options[0]
        # FIX: element cannot go to B-site; fall back to A-site rather than
        #      letting pymatgen prevalence pick a conflicting oxidation state.
        a_options = [oxi for site, oxi in possible_sites if site == 'A']
        if a_options:
            # Honor charge balance if possible
            if a_oxidation_state is not None and a_oxidation_state in a_options:
                return 'A', a_oxidation_state
            return 'A', a_options[0]

    # ── PRIORITY 3: Use pymatgen's oxidation-state prevalence ────────────────
    for oxi in get_pymatgen_oxi_preference(elem):
        for site, s_oxi in possible_sites:
            if s_oxi == oxi:
                return site, oxi

    # ── PRIORITY 4: First option in list ─────────────────────────────────────
    if possible_sites:
        site, oxi = possible_sites[0]
        return site, oxi

    return 'unknown', None


def resolve_all_flexible_a(
        flexible_a_elements: Dict[str, float],
) -> Tuple[int, int]:
    """
    Resolve A-site oxidation when every cation is a flexible A-site element.

    Strategy
    --------
    1. If any element also supports B-site +4 → assume A(+3) / B(+4)
    2. If any element also supports B-site +5 → assume A(+2) / B(+5)
    3. Default to A(+3) / B(+4)
    """
    can_be_b4 = []
    can_be_b5 = []

    for elem in flexible_a_elements:
        try:
            common = set(Element(elem).common_oxidation_states)
        except Exception:
            common = set()
        if 4 in common:
            can_be_b4.append(elem)
        if 5 in common:
            can_be_b5.append(elem)

    if can_be_b4:
        return 3, 4
    if can_be_b5:
        return 2, 5

    print(f"WARNING: All flexible elements {list(flexible_a_elements)} "
          f"lack B-site capability. Defaulting to A(+3), B(+4).")
    return 3, 4


# ── Main assignment function ──────────────────────────────────────────────────

def assign_sites(
        comp: Composition,
) -> Tuple[Dict[str, float], Dict[str, float],
           Dict[str, float], Optional[Tuple[float, float]]]:
    """
    Split cation elements into A-site (3+/2+), B-site (4+/5+), and unknown dicts.
    Stoichiometries are mole fractions (sum to 1 per site).

    Pyrochlore charge balance constraint
    -------------------------------------
    A₂B₂O₇  →  2·A_oxi + 2·B_oxi = 14
        A(+3) / B(+4)   or   A(+2) / B(+5)

    Assignment priority (per element)
    -----------------------------------
    1. Oxygen is always skipped.
    2. Ambiguous elements are held aside and resolved last.
    3. For every other cation the KNOWN_* look-up tables are consulted.
    4. Anything not in a known list goes to *unknown*.
    5. After the first pass, ``required_b_oxi`` is inferred from already
       assigned B-site elements (fixing the Eu₂Ru₂O₇-class of errors), then
       ``a_oxidation_state`` is back-derived if still unknown.
    6. Ambiguous elements are resolved with full A/B context available.

    Parameters
    ----------
    comp : pymatgen Composition (need not be reduced; reduced internally)

    Returns
    -------
    a_comp     : {element: mole_fraction}  A-site (sums to 1)
    b_comp     : {element: mole_fraction}  B-site (sums to 1)
    unknown    : {element: mole_fraction}  unassigned cations
    oxi_states : (a_site_oxi_state, b_site_oxi_state) or (None, None)
    """
    reduced = Composition(comp).reduced_composition
    raw: Dict[str, float] = {
        str(el): amt
        for el, amt in reduced.items()
        if str(el) != 'O'
    }
    print(raw)
    exit(0)

    a_comp:        Dict[str, float] = {}
    b_comp:        Dict[str, float] = {}
    unknown:       Dict[str, float] = {}
    ambig_elements: Dict[str, float] = {}
    a_oxidation_state: Optional[int] = None

    # ── FIRST PASS: assign non-ambiguous elements ────────────────────────────
    for elem, amt in raw.items():
        if elem in KNOWN_AMBIGUOUS:
            ambig_elements[elem] = amt
        elif elem in KNOWN_A_3_ONLY:
            a_comp[elem] = amt
            if a_oxidation_state is None:
                a_oxidation_state = 3
        elif elem in KNOWN_A_2_ONLY:
            a_comp[elem] = amt
            if a_oxidation_state is None:
                a_oxidation_state = 2
        elif elem in KNOWN_B_4_ONLY:
            b_comp[elem] = amt
        elif elem in KNOWN_B_5_ONLY:
            b_comp[elem] = amt
        else:
            unknown[elem] = amt

    # ── INFER OXIDATION STATES from both A and B evidence ───────────────────
    # FIX: derive required_b_oxi from B-site elements *before* resolving
    #      ambiguous elements — this is what caused Eu₂Ru₂O₇ → Eu(+2)/Ru(NaN).
    required_b_oxi: Optional[int] = None

    if a_oxidation_state == 3:
        required_b_oxi = 4
    elif a_oxidation_state == 2:
        required_b_oxi = 5

    if required_b_oxi is None and b_comp:
        required_b_oxi = _infer_b_oxi_from_b_comp(b_comp)

    # Back-derive a_oxidation_state from B-site if still unknown
    if a_oxidation_state is None:
        if required_b_oxi == 4:
            a_oxidation_state = 3
        elif required_b_oxi == 5:
            a_oxidation_state = 2

    # Forward-derive required_b_oxi from A-site if still unknown
    if required_b_oxi is None:
        if a_oxidation_state == 3:
            required_b_oxi = 4
        elif a_oxidation_state == 2:
            required_b_oxi = 5

    # ── RESOLVE AMBIGUOUS ELEMENTS (full context now available) ──────────────
    for elem, amt in ambig_elements.items():
        site, oxi = resolve_ambiguous_element(
            elem,
            a_oxidation_state,
            required_b_oxi,
            a_comp,
            b_comp,
        )

        if site == 'A':
            a_comp[elem] = amt
            if a_oxidation_state is None:
                a_oxidation_state = oxi
                required_b_oxi = 4 if oxi == 3 else (5 if oxi == 2 else required_b_oxi)
        elif site == 'B':
            b_comp[elem] = amt
            if required_b_oxi is None:
                required_b_oxi = oxi
                a_oxidation_state = 3 if oxi == 4 else (2 if oxi == 5 else a_oxidation_state)
        else:
            unknown[elem] = amt

    oxi_states: Optional[Tuple[float, float]] = (a_oxidation_state, required_b_oxi)

    def _to_fracs(d: Dict[str, float]) -> Dict[str, float]:
        total = sum(d.values())
        return {k: v / total for k, v in d.items()} if total else {}

    return _to_fracs(a_comp), _to_fracs(b_comp), unknown, oxi_states



def reconcile_ambig_to_sites(
    a_comp: Dict[str, float],
    b_comp: Dict[str, float],
    ambig_elements: Dict[str, float],  # elem -> amt
    a_oxidation_state: Optional[int],  # may be None if not fixed by A_3_ONLY/A_2_ONLY
    b_oxidation_state: Optional[int],  # may be None if not fixed by B_4_ONLY/B_5_ONLY
) -> Tuple[Dict[str, float], Dict[str, float], Optional[int], Optional[int], Dict[str, Tuple[str, int]]]:
    target_A: float = 2.0
    target_B: float = 2.0

    # Preference: prefer A=3 and B=4 (over A=2 and B=5)
    def option_priority(site: str, ox: int, curr_a_ox: Optional[int], curr_b_ox: Optional[int], elem: str) -> int:
        score = 0

        # Strong preference for 3/4
        if site == "A" and ox == 3:
            score += 50
        if site == "B" and ox == 4:
            score += 50

        # Discourage 2/5
        if site == "A" and ox == 2:
            score -= 20
        if site == "B" and ox == 5:
            score -= 20

        # If already fixed oxidation on that site, reward exact match (but mismatches should be disallowed anyway)
        if site == "A" and curr_a_ox is not None:
            score += 30 if ox == curr_a_ox else -100
        if site == "B" and curr_b_ox is not None:
            score += 30 if ox == curr_b_ox else -100

        # # Optional: mild boost for Ta's comment ("prefers +5 as B-site")
        # if elem == "Ta" and site == "B" and ox == 5:
        #     score += 10

        return score

    ambig_items = list(ambig_elements.items())

    initial_a_total = sum(a_comp.values())
    initial_b_total = sum(b_comp.values())

    best = None
    # best = (err, -score, assignments_dict, final_a_ox, final_b_ox)

    def backtrack(
        i: int,
        curr_a: Dict[str, float],
        curr_b: Dict[str, float],
        curr_a_total: float,
        curr_b_total: float,
        curr_a_ox: Optional[int],
        curr_b_ox: Optional[int],
        curr_score: int,
        assignments: Dict[str, Tuple[str, int]],
    ):
        nonlocal best

        if i == len(ambig_items):
            err = abs(curr_a_total - target_A) + abs(curr_b_total - target_B)
            cand = (err, -curr_score, dict(assignments), curr_a_ox, curr_b_ox)
            if best is None or cand < best:
                best = cand
            return

        elem, amt = ambig_items[i]
        options = KNOWN_AMBIGUOUS[elem]  # list of (site, ox)

        for site, ox in options:
            # Enforce single oxidation per site:
            if site == "A":
                if curr_a_ox is not None and ox != curr_a_ox:
                    continue  # disallow inconsistent oxidation on A-site
                next_a_ox = curr_a_ox if curr_a_ox is not None else ox
                next_a_total = curr_a_total + amt
                next_a = dict(curr_a)
                next_a[elem] = next_a.get(elem, 0.0) + amt

                pr = option_priority(site, ox, curr_a_ox, curr_b_ox, elem)
                assignments[elem] = (site, ox)
                backtrack(
                    i + 1,
                    next_a, curr_b,
                    next_a_total, curr_b_total,
                    next_a_ox, curr_b_ox,
                    curr_score + pr,
                    assignments
                )
                del assignments[elem]

            else:  # site == "B"
                if curr_b_ox is not None and ox != curr_b_ox:
                    continue  # disallow inconsistent oxidation on B-site
                next_b_ox = curr_b_ox if curr_b_ox is not None else ox
                next_b_total = curr_b_total + amt
                next_b = dict(curr_b)
                next_b[elem] = next_b.get(elem, 0.0) + amt

                pr = option_priority(site, ox, curr_a_ox, curr_b_ox, elem)
                assignments[elem] = (site, ox)
                backtrack(
                    i + 1,
                    curr_a, next_b,
                    curr_a_total, next_b_total,
                    curr_a_ox, next_b_ox,
                    curr_score + pr,
                    assignments
                )
                del assignments[elem]

    backtrack(
        0,
        dict(a_comp), dict(b_comp),
        initial_a_total, initial_b_total,
        a_oxidation_state, b_oxidation_state,
        curr_score=0,
        assignments={}
    )

    if best is None:
        return a_comp, b_comp, a_oxidation_state, b_oxidation_state, {}

    _, _, assignments, final_a_ox, final_b_ox = best

    # Build final compositions by applying assignments to the provided base comps
    final_a = dict(a_comp)
    final_b = dict(b_comp)
    for elem, amt in ambig_elements.items():
        site, _ox = assignments[elem]
        if site == "A":
            final_a[elem] = final_a.get(elem, 0.0) + amt
        else:
            final_b[elem] = final_b.get(elem, 0.0) + amt

    return final_a, final_b, final_a_ox, final_b_ox, assignments


def assign_sites_pristine(comp: Composition) -> Tuple[Dict[str, float], Dict[str, float],
           Dict[str, float], Optional[Tuple[float, float]]]:
    """
            Split cation elements into A-site (3+/2+), B-site (4+/5+), and unknown dicts.
            Stoichiometries are mole fractions (sum to 1 per site).

            Pyrochlore charge balance constraint
            -------------------------------------
            A₂B₂O₇  →  2·A_oxi + 2·B_oxi = 14
                A(+3) / B(+4)   or   A(+2) / B(+5)

            Assignment priority (per element)
            -----------------------------------
            1. Oxygen is always skipped.
            2. Ambiguous elements are held aside and resolved last.
            3. For every other cation the KNOWN_* look-up tables are consulted.
            4. Anything not in a known list goes to *unknown*.
            5. After the first pass, ``required_b_oxi`` is inferred from already
               assigned B-site elements (fixing the Eu₂Ru₂O₇-class of errors), then
               ``a_oxidation_state`` is back-derived if still unknown.
            6. Ambiguous elements are resolved with full A/B context available.

            Parameters
            ----------
            comp : pymatgen Composition (need not be reduced; reduced internally)

            Returns
            -------
            a_comp     : {element: mole_fraction}  A-site (sums to 1)
            b_comp     : {element: mole_fraction}  B-site (sums to 1)
            unknown    : {element: mole_fraction}  unassigned cations
            oxi_states : (a_site_oxi_state, b_site_oxi_state) or (None, None)
            """

    reduced = Composition(comp).reduced_composition
    raw: Dict[str, float] = {
        str(el): amt
        for el, amt in reduced.items()
        if str(el) != 'O'
    }


    a_comp: Dict[str, float] = {}
    b_comp: Dict[str, float] = {}
    unknown: Dict[str, float] = {}
    ambig_elements: Dict[str, float] = {}
    a_oxidation_state: Optional[int] = None
    b_oxidation_state: Optional[int] = None

    for elem, amt in raw.items():
        if elem in KNOWN_AMBIGUOUS:
            # options = KNOWN_AMBIGUOUS[elem]
            # sites = {site for site, ox in options}
            # if sites == {'A', 'B'}:
            #     ambig_elements[elem] = amt
            # elif sites == {'A'}:
            #     a_comp[elem] = amt
            # elif sites == {'B'}:
            #     b_comp[elem] = amt
            # else:
            ambig_elements[elem] = amt
        elif elem in KNOWN_A_3_ONLY:
            a_comp[elem] = amt
            if a_oxidation_state is None:
                a_oxidation_state = 3
        elif elem in KNOWN_A_2_ONLY:
            a_comp[elem] = amt
            if a_oxidation_state is None:
                a_oxidation_state = 2
        elif elem in KNOWN_B_4_ONLY:
            b_comp[elem] = amt
            if b_oxidation_state is None:
                b_oxidation_state = 4
        elif elem in KNOWN_B_5_ONLY:
            b_comp[elem] = amt
            if b_oxidation_state is None:
                b_oxidation_state = 5
        else:
            unknown[elem] = amt

    if ambig_elements:
        a_comp, b_comp, a_oxidation_state, b_oxidation_state, _assignments = reconcile_ambig_to_sites(
            a_comp=a_comp,
            b_comp=b_comp,
            ambig_elements=ambig_elements,
            a_oxidation_state=a_oxidation_state,
            b_oxidation_state=b_oxidation_state,
        )

    return a_comp, b_comp, unknown, (a_oxidation_state, b_oxidation_state)