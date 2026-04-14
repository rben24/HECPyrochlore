"""
Feature Engineering for Pyrochlore Oxide Dataset
Calculates compositional, thermodynamic, and structural features
from A-site and B-site cation compositions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# ── Physical property tables ─────────────────────────────────────────────────

# Shannon ionic radii (Angstroms, 8-coord for A-site, 6-coord for B-site)
IONIC_RADII_8 = {  # A-site: 8-coordination
    'La': 1.160, 'Ce': 1.143, 'Pr': 1.126, 'Nd': 1.109, 'Sm': 1.079,
    'Eu': 1.066, 'Gd': 1.053, 'Tb': 1.040, 'Dy': 1.027, 'Ho': 1.015,
    'Er': 1.004, 'Tm': 0.994, 'Yb': 0.985, 'Lu': 0.977, 'Y':  1.019,
}

IONIC_RADII_6 = {  # B-site: 6-coordination
    'Ti': 0.605, 'Zr': 0.720, 'Hf': 0.710, 'Sn': 0.690, 'Ir': 0.625,
    'Ce': 0.870, 'Nb': 0.640,
}

# Molar masses (g/mol)
MOLAR_MASSES = {
    'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24, 'Pm': 145.00,
    'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93, 'Dy': 162.50,
    'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.04, 'Lu': 174.97,
    'Y':   88.91, 'Ti':  47.87, 'Zr':  91.22, 'Hf': 178.49, 'Sn': 118.71,
    'Ir': 192.22, 'O':   16.00,
}

# Pauling electronegativity
ELECTRONEGATIVITY = {
    'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14, 'Sm': 1.17,
    'Eu': 1.20, 'Gd': 1.20, 'Tb': 1.22, 'Dy': 1.23, 'Ho': 1.24,
    'Er': 1.24, 'Tm': 1.25, 'Yb': 1.10, 'Lu': 1.27, 'Y':  1.22,
    'Ti': 1.54, 'Zr': 1.33, 'Hf': 1.30, 'Sn': 1.96, 'Ir': 2.20,
}

# Atomic number (proxy for electron configuration effects)
ATOMIC_NUMBER = {
    'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Sm': 62, 'Eu': 63,
    'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69,
    'Yb': 70, 'Lu': 71, 'Y': 39, 'Ti': 22, 'Zr': 40, 'Hf': 72,
    'Sn': 50, 'Ir': 77,
}

R_GAS = 8.314  # J/(mol·K)


# ── Parser ───────────────────────────────────────────────────────────────────

def parse_composition(comp_str: str) -> Dict[str, float]:
    """
    Parse a comma-separated element string into equiatomic fractions.
    E.g. "La,Gd,Lu" -> {'La': 0.333, 'Gd': 0.333, 'Lu': 0.333}
    """
    if pd.isna(comp_str) or str(comp_str).strip() == '':
        return {}
    elements = [e.strip() for e in str(comp_str).split(',') if e.strip()]
    n = len(elements)
    return {elem: 1.0 / n for elem in elements}


# ── Per-site feature calculators ─────────────────────────────────────────────

def configurational_entropy(comp: Dict[str, float]) -> float:
    """S_config = -R * Σ x_i ln(x_i)  [J/(mol·K)]"""
    if not comp:
        return np.nan
    fracs = [f for f in comp.values() if f > 0]
    return -R_GAS * sum(f * np.log(f) for f in fracs)


def mean_radius(comp: Dict[str, float], radii_table: Dict[str, float]) -> float:
    vals = [radii_table.get(e, np.nan) * f for e, f in comp.items()]
    if any(np.isnan(v) for v in vals):
        return np.nan
    return sum(vals)


def radius_variance(comp: Dict[str, float], radii_table: Dict[str, float]) -> float:
    r_mean = mean_radius(comp, radii_table)
    if np.isnan(r_mean):
        return np.nan
    return sum(f * (radii_table.get(e, np.nan) - r_mean) ** 2
               for e, f in comp.items())


def radius_std(comp: Dict[str, float], radii_table: Dict[str, float]) -> float:
    var = radius_variance(comp, radii_table)
    return np.nan if np.isnan(var) else np.sqrt(var)


def delta_parameter(comp: Dict[str, float], radii_table: Dict[str, float]) -> float:
    """δ = sqrt(Σ x_i*(1 - r_i/r_mean)^2) — lattice distortion index"""
    r_mean = mean_radius(comp, radii_table)
    if np.isnan(r_mean) or r_mean == 0:
        return np.nan
    return np.sqrt(sum(f * (1 - radii_table.get(e, np.nan) / r_mean) ** 2
                       for e, f in comp.items()))


def en_mean(comp: Dict[str, float]) -> float:
    vals = [ELECTRONEGATIVITY.get(e, np.nan) * f for e, f in comp.items()]
    if any(np.isnan(v) for v in vals):
        return np.nan
    return sum(vals)


def en_variance(comp: Dict[str, float]) -> float:
    mu = en_mean(comp)
    if np.isnan(mu):
        return np.nan
    return sum(f * (ELECTRONEGATIVITY.get(e, np.nan) - mu) ** 2
               for e, f in comp.items())


def mean_atomic_number(comp: Dict[str, float]) -> float:
    vals = [ATOMIC_NUMBER.get(e, np.nan) * f for e, f in comp.items()]
    if any(np.isnan(v) for v in vals):
        return np.nan
    return sum(vals)


def mean_molar_mass(comp: Dict[str, float]) -> float:
    vals = [MOLAR_MASSES.get(e, np.nan) * f for e, f in comp.items()]
    if any(np.isnan(v) for v in vals):
        return np.nan
    return sum(vals)


# ── Full-composition features ────────────────────────────────────────────────

def a_site_b_site_radius_ratio(r_a: float, r_b: float) -> float:
    """r_A / r_B — governs pyrochlore stability field"""
    if np.isnan(r_a) or np.isnan(r_b) or r_b == 0:
        return np.nan
    return r_a / r_b


def phonon_scattering_factor(s_config: float, delta: float) -> float:
    """Higher → more phonon scattering → lower thermal conductivity"""
    if np.isnan(s_config) or np.isnan(delta):
        return np.nan
    return s_config * delta


def theoretical_density(a_comp: Dict, b_comp: Dict, lattice_a: float) -> float:
    """ρ_th = Z·M / (V·N_A),  Z=8 formula units, V in cm³"""
    if np.isnan(lattice_a) or lattice_a <= 0:
        return np.nan
    M_a = 2 * sum(MOLAR_MASSES.get(e, np.nan) * f for e, f in a_comp.items())
    M_b = 2 * sum(MOLAR_MASSES.get(e, np.nan) * f for e, f in b_comp.items())
    M_o = 7 * MOLAR_MASSES['O']
    M = M_a + M_b + M_o
    V = (lattice_a * 1e-8) ** 3  # cm³
    return (8 * M) / (V * 6.022e23)


# ── Row-level feature builder ─────────────────────────────────────────────────

def build_features_for_row(row: pd.Series) -> Dict[str, float]:
    """Compute all engineered features for one data row."""
    a_comp = parse_composition(row.get('Sample A', ''))
    b_comp = parse_composition(row.get('Sample B', ''))

    lattice_a = row.get('Lattice Parameter (Angstrom)', np.nan)
    if pd.isna(lattice_a):
        lattice_a = row.get('Lattice Parameter a (A)', np.nan)
    try:
        lattice_a = float(lattice_a)
    except (TypeError, ValueError):
        lattice_a = np.nan

    # ── A-site
    a_S   = configurational_entropy(a_comp)
    a_r   = mean_radius(a_comp, IONIC_RADII_8)
    a_var = radius_variance(a_comp, IONIC_RADII_8)
    a_std = radius_std(a_comp, IONIC_RADII_8)
    a_del = delta_parameter(a_comp, IONIC_RADII_8)
    a_en  = en_mean(a_comp)
    a_env = en_variance(a_comp)
    a_Z   = mean_atomic_number(a_comp)
    a_M   = mean_molar_mass(a_comp)
    a_n   = len(a_comp)

    # ── B-site
    b_S   = configurational_entropy(b_comp)
    b_r   = mean_radius(b_comp, IONIC_RADII_6)
    b_var = radius_variance(b_comp, IONIC_RADII_6)
    b_std = radius_std(b_comp, IONIC_RADII_6)
    b_del = delta_parameter(b_comp, IONIC_RADII_6)
    b_en  = en_mean(b_comp)
    b_env = en_variance(b_comp)
    b_Z   = mean_atomic_number(b_comp)
    b_M   = mean_molar_mass(b_comp)
    b_n   = len(b_comp)

    # ── Cross-site
    r_ratio = a_site_b_site_radius_ratio(a_r, b_r)
    total_S = (a_S if not np.isnan(a_S) else 0) + (b_S if not np.isnan(b_S) else 0)
    total_delta = (a_del if not np.isnan(a_del) else 0) + (b_del if not np.isnan(b_del) else 0)
    phonon_factor = phonon_scattering_factor(total_S, total_delta)

    # ── Lattice-derived
    lattice_vol = lattice_a ** 3 if not np.isnan(lattice_a) else np.nan
    rho_th = theoretical_density(a_comp, b_comp, lattice_a)

    feat = {
        # A-site
        'a_site_n_elements':       float(a_n),
        'a_site_entropy':          a_S,
        'a_site_mean_radius':      a_r,
        'a_site_radius_variance':  a_var,
        'a_site_radius_std':       a_std,
        'a_site_delta':            a_del,
        'a_site_en_mean':          a_en,
        'a_site_en_variance':      a_env,
        'a_site_mean_atomic_num':  a_Z,
        'a_site_mean_molar_mass':  a_M,
        # B-site
        'b_site_n_elements':       float(b_n),
        'b_site_entropy':          b_S,
        'b_site_mean_radius':      b_r,
        'b_site_radius_variance':  b_var,
        'b_site_radius_std':       b_std,
        'b_site_delta':            b_del,
        'b_site_en_mean':          b_en,
        'b_site_en_variance':      b_env,
        'b_site_mean_atomic_num':  b_Z,
        'b_site_mean_molar_mass':  b_M,
        # Cross-site
        'a_b_radius_ratio':        r_ratio,
        'total_entropy':           total_S,
        'total_delta':             total_delta,
        'phonon_scattering_factor': phonon_factor,
        # Lattice
        'lattice_parameter':       lattice_a,
        'lattice_volume':          lattice_vol,
        'density_theoretical':     rho_th,
    }
    return feat


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features to a DataFrame."""
    records = []
    for _, row in df.iterrows():
        records.append(build_features_for_row(row))
    feat_df = pd.DataFrame(records, index=df.index)
    return pd.concat([df, feat_df], axis=1)


# ── Feature name helpers ──────────────────────────────────────────────────────

FEATURE_COLS = [
    'a_site_n_elements', 'a_site_entropy', 'a_site_mean_radius',
    'a_site_radius_variance', 'a_site_radius_std', 'a_site_delta',
    'a_site_en_mean', 'a_site_en_variance', 'a_site_mean_atomic_num',
    'a_site_mean_molar_mass',
    'b_site_n_elements', 'b_site_entropy', 'b_site_mean_radius',
    'b_site_radius_variance', 'b_site_radius_std', 'b_site_delta',
    'b_site_en_mean', 'b_site_en_variance', 'b_site_mean_atomic_num',
    'b_site_mean_molar_mass',
    'a_b_radius_ratio', 'total_entropy', 'total_delta',
    'phonon_scattering_factor',
]

LATTICE_EXTRA_FEATURES = FEATURE_COLS  # no lattice_parameter itself
THERMAL_EXTRA_FEATURES = FEATURE_COLS + ['lattice_parameter']


if __name__ == '__main__':
    # Quick sanity check
    test = pd.Series({
        'Sample A': 'Pr,Sm,Gd,Ho,Lu',
        'Sample B': 'Ti',
        'Lattice Parameter (Angstrom)': 10.178,
        'TPS Cond W/m/K': 1.566,
    })
    feats = build_features_for_row(test)
    print("Feature engineering smoke test — PrSmGdHoLu / Ti:")
    for k, v in feats.items():
        print(f"  {k:35s}: {v:.6g}" if not np.isnan(v) else f"  {k:35s}: NaN")
