"""
Feature Engineering for Pyrochlore Oxide Dataset
=================================================
Calculates compositional, thermodynamic, and structural features
from A-site and B-site cation compositions.

Key change vs. original
-----------------------
``parse_composition`` now supports **non-equiatomic** compositions by reading
pre-computed mole-fraction dicts from the ``a_stoich_json`` / ``b_stoich_json``
columns that are written by the ICSD loader.  If those columns are absent (or
NaN), it falls back to the original equiatomic assumption so backward
compatibility with the Safin / NLM / parent-component sources is preserved.

New features added
------------------
  n_total_elements   : total distinct cation count across both sites
  site_asymmetry     : |n_A - n_B|  (degree of site-compositional imbalance)
  en_site_contrast   : |ēn_A - ēn_B|  (electronegativity contrast across sites)
  mass_site_contrast : |M̄_A - M̄_B|  (molar-mass contrast across sites)
"""

from __future__ import annotations

import json
import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# ── Physical property tables ──────────────────────────────────────────────────

# Shannon ionic radii (Å) — 8-coord for A-site, 6-coord for B-site
# taken from Shannon, R. D. “Revised Effective Ionic Radii and Systematic Studies of Interatomic Distances in Halides
# and Chalcogenides.” Acta Crystallographica Section A, vol. 32, no. 5, 1 Sept. 1976, pp. 751–767,
# https://doi.org/10.1107/s0567739476001551.
IONIC_RADII_8: Dict[str, float] = {
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

IONIC_RADII_6: Dict[str, float] = {
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
MOLAR_MASSES: Dict[str, float] = {
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
ELECTRONEGATIVITY: Dict[str, float] = {
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
ATOMIC_NUMBER: Dict[str, int] = {
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

R_GAS = 8.314  # J/(mol·K)


# ── Composition parser ────────────────────────────────────────────────────────

def parse_composition(
    comp_str: str,
    stoich_json: Optional[str] = None,
) -> Dict[str, float]:
    """
    Convert a composition description to a mole-fraction dict.

    Priority order
    --------------
    1. If ``stoich_json`` is a non-empty JSON string (written by the ICSD
       loader), parse it directly — this preserves the true non-equiatomic
       fractions (e.g. Gd₁.₉Ce₀.₁Ti₂O₇).
    2. Otherwise, fall back to splitting ``comp_str`` on commas and assuming
       equiatomic fractions — identical to the original behaviour used for
       the Safin / parent-component / NLM datasets.

    Parameters
    ----------
    comp_str    : comma-separated element symbols, e.g. "La,Gd,Lu"
    stoich_json : optional JSON string from a_stoich_json / b_stoich_json,
                  e.g. '{"Gd": 0.95, "Ce": 0.05}'

    Returns
    -------
    Dict mapping element symbol → mole fraction (values sum to 1)
    """
    # --- try JSON stoichiometry first ---
    if stoich_json and not pd.isna(stoich_json):
        try:
            raw = json.loads(str(stoich_json))
            total = sum(raw.values())
            if total > 0:
                return {e: v / total for e, v in raw.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # --- equiatomic fallback ---
    if pd.isna(comp_str) or str(comp_str).strip() == '':
        return {}
    elements = [e.strip() for e in str(comp_str).split(',') if e.strip()]
    n = len(elements)
    if n == 0:
        return {}
    return {elem: 1.0 / n for elem in elements}


# ── Per-site feature calculators ─────────────────────────────────────────────

def configurational_entropy(comp: Dict[str, float]) -> float:
    """S_config = -R × Σ xᵢ ln(xᵢ)   [J/(mol·K)]"""
    if not comp:
        return np.nan
    fracs = [f for f in comp.values() if f > 0]
    return -R_GAS * sum(f * np.log(f) for f in fracs)


def mean_radius(comp: Dict[str, float], radii_table: Dict[str, float]) -> float:
    """Composition-weighted mean ionic radius."""
    vals = [radii_table.get(e, np.nan) * f for e, f in comp.items()]
    if any(np.isnan(v) for v in vals) or not vals:
        return np.nan
    return float(sum(vals))


def radius_variance(comp: Dict[str, float], radii_table: Dict[str, float]) -> float:
    """Composition-weighted variance of ionic radii."""
    r_mean = mean_radius(comp, radii_table)
    if np.isnan(r_mean):
        return np.nan
    return float(sum(
        f * (radii_table.get(e, np.nan) - r_mean) ** 2
        for e, f in comp.items()
    ))


def radius_std(comp: Dict[str, float], radii_table: Dict[str, float]) -> float:
    var = radius_variance(comp, radii_table)
    return np.nan if np.isnan(var) else float(np.sqrt(var))


def delta_parameter(comp: Dict[str, float], radii_table: Dict[str, float]) -> float:
    """δ = √(Σ xᵢ(1 − rᵢ/r̄)²)  — lattice distortion index."""
    """https://www.tandfonline.com/doi/full/10.1080/21663831.2024.2326014#d1e452"""
    r_mean = mean_radius(comp, radii_table)
    if np.isnan(r_mean) or r_mean == 0:
        return np.nan
    return float(np.sqrt(sum(
        f * (1 - radii_table.get(e, np.nan) / r_mean) ** 2
        for e, f in comp.items()
    )))


def en_mean(comp: Dict[str, float]) -> float:
    vals = [ELECTRONEGATIVITY.get(e, np.nan) * f for e, f in comp.items()]
    if any(np.isnan(v) for v in vals) or not vals:
        return np.nan
    return float(sum(vals))


def en_variance(comp: Dict[str, float]) -> float:
    mu = en_mean(comp)
    if np.isnan(mu):
        return np.nan
    return float(sum(
        f * (ELECTRONEGATIVITY.get(e, np.nan) - mu) ** 2
        for e, f in comp.items()
    ))


def mean_atomic_number(comp: Dict[str, float]) -> float:
    vals = [ATOMIC_NUMBER.get(e, np.nan) * f for e, f in comp.items()]
    if any(np.isnan(v) for v in vals) or not vals:
        return np.nan
    return float(sum(vals))


def mean_molar_mass(comp: Dict[str, float]) -> float:
    vals = [MOLAR_MASSES.get(e, np.nan) * f for e, f in comp.items()]
    if any(np.isnan(v) for v in vals) or not vals:
        return np.nan
    return float(sum(vals))


# ── Cross-site feature calculators ───────────────────────────────────────────

def a_site_b_site_radius_ratio(r_a: float, r_b: float) -> float:
    """r_A / r_B — governs pyrochlore stability field (ideal: 1.46–1.78)."""
    if np.isnan(r_a) or np.isnan(r_b) or r_b == 0:
        return np.nan
    return r_a / r_b


def phonon_scattering_factor(s_config: float, delta: float) -> float:
    """S_config × δ_total — composite proxy for phonon scattering strength."""
    if np.isnan(s_config) or np.isnan(delta):
        return np.nan
    return s_config * delta


def theoretical_density(
    a_comp: Dict[str, float],
    b_comp: Dict[str, float],
    lattice_a: float,
) -> float:
    """ρ_th = Z·M / (V·N_A),  Z = 8 formula units per unit cell."""
    if np.isnan(lattice_a) or lattice_a <= 0:
        return np.nan
    M_a = 2 * sum(MOLAR_MASSES.get(e, np.nan) * f for e, f in a_comp.items())
    M_b = 2 * sum(MOLAR_MASSES.get(e, np.nan) * f for e, f in b_comp.items())
    M_o = 7 * MOLAR_MASSES['O']
    M   = M_a + M_b + M_o
    V   = (lattice_a * 1e-8) ** 3   # cm³
    return (8 * M) / (V * 6.022e23)


# ── Row-level feature builder ─────────────────────────────────────────────────

def build_features_for_row(row: pd.Series) -> Dict[str, float]:
    """
    Compute all engineered features for one data row.

    Reads ``a_stoich_json`` / ``b_stoich_json`` when present so that
    non-equiatomic ICSD entries are handled correctly.
    """
    a_comp = parse_composition(
        row.get('Sample A', ''),
        stoich_json=row.get('a_stoich_json', None),
    )
    b_comp = parse_composition(
        row.get('Sample B', ''),
        stoich_json=row.get('b_stoich_json', None),
    )

    # Lattice parameter (may come from different column names)
    lattice_a = row.get('Lattice Parameter (Angstrom)', np.nan)
    if pd.isna(lattice_a):
        lattice_a = row.get('Lattice Parameter a (A)', np.nan)
    try:
        lattice_a = float(lattice_a)
    except (TypeError, ValueError):
        lattice_a = np.nan

    # ── A-site features ───────────────────────────────────────────────────────
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

    # ── B-site features ───────────────────────────────────────────────────────
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

    # ── Cross-site features ───────────────────────────────────────────────────
    r_ratio     = a_site_b_site_radius_ratio(a_r, b_r)
    total_S     = (a_S if not np.isnan(a_S) else 0.0) + \
                  (b_S if not np.isnan(b_S) else 0.0)
    total_delta = (a_del if not np.isnan(a_del) else 0.0) + \
                  (b_del if not np.isnan(b_del) else 0.0)
    phonon_fac  = phonon_scattering_factor(total_S, total_delta)

    # New cross-site contrast features
    n_total        = float(a_n + b_n)
    site_asymmetry = float(abs(a_n - b_n))
    en_contrast    = abs(a_en - b_en) if not (np.isnan(a_en) or np.isnan(b_en)) \
                     else np.nan
    mass_contrast  = abs(a_M - b_M) if not (np.isnan(a_M) or np.isnan(b_M)) \
                     else np.nan

    # ── Lattice-derived features ──────────────────────────────────────────────
    lattice_vol = lattice_a ** 3 if not np.isnan(lattice_a) else np.nan
    rho_th = (theoretical_density(a_comp, b_comp, lattice_a) + float(row.get('density_calc'))) / 2 \
        if pd.notna(row.get('density_calc')) \
        else theoretical_density(a_comp, b_comp, lattice_a)

    return {
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
        'a_b_radius_ratio':         r_ratio,
        'total_entropy':            total_S,
        'total_delta':              total_delta,
        'phonon_scattering_factor': phonon_fac,
        'n_total_elements':         n_total,
        'site_asymmetry':           site_asymmetry,
        'en_site_contrast':         en_contrast,
        'mass_site_contrast':       mass_contrast,
        # Lattice-derived (used in thermal model only)
        'lattice_parameter':        lattice_a,
        'lattice_volume':           lattice_vol,
        'density_theoretical':      rho_th,
    }


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features as new columns to a DataFrame."""
    records = [build_features_for_row(row) for _, row in df.iterrows()]
    feat_df = pd.DataFrame(records, index=df.index)
    return pd.concat([df, feat_df], axis=1)


# ── Feature-name lists ────────────────────────────────────────────────────────

# Core compositional features (no lattice parameter)
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
    # new cross-site contrast features
    'n_total_elements', 'site_asymmetry', 'en_site_contrast', 'mass_site_contrast',
]

LATTICE_EXTRA_FEATURES = FEATURE_COLS   # lattice_param itself is the target
THERMAL_EXTRA_FEATURES = FEATURE_COLS + ['lattice_parameter']


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Test 1: equiatomic (Safin-style)
    test_eq = pd.Series({
        'Sample A': 'Pr,Sm,Gd,Ho,Lu',
        'Sample B': 'Ti',
        'Lattice Parameter (Angstrom)': 10.178,
        'TPS Cond W/m/K': 1.566,
    })
    feats_eq = build_features_for_row(test_eq)
    print("Equiatomic PrSmGdHoLu / Ti:")
    for k, v in feats_eq.items():
        print(f"  {k:35s}: {v:.6g}" if not np.isnan(v) else f"  {k:35s}: NaN")

    # Test 2: non-equiatomic (ICSD-style)
    import json as _json
    test_ne = pd.Series({
        'Sample A': 'Ce,Gd',
        'Sample B': 'Ce,Ti',
        'a_stoich_json': _json.dumps({'Ce': 0.05, 'Gd': 0.95}),
        'b_stoich_json': _json.dumps({'Ce': 0.05, 'Ti': 0.95}),
        'Lattice Parameter (Angstrom)': 10.171,
    })
    feats_ne = build_features_for_row(test_ne)
    print("\nNon-equiatomic Gd₁.₉Ce₀.₁Ti₂O₇:")
    for k, v in feats_ne.items():
        print(f"  {k:35s}: {v:.6g}" if not np.isnan(v) else f"  {k:35s}: NaN")

    print(f"\nTotal feature count: {len(FEATURE_COLS)} (compositional) "
          f"+ 1 lattice = {len(THERMAL_EXTRA_FEATURES)} (thermal)")