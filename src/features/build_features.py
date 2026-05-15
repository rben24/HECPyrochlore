"""
build_features.py
=================
Feature engineering for the HEC Pyrochlore project.

Formula input formats accepted
-------------------------------
All the following are handled through a unified parser:

  1. Pyrochlore formula  : "(La0.2Yb0.8)2(Zr0.7Ce0.3)2O7"
     Also plain formulas : "La2Ti2O7", "La2(Ti0.5Zr0.5)2O7"
  2. Comma-separated     : "La,Gd,Lu" + "Ti,Zr"   (equiatomic assumption)
  3. Stoich JSON         : '{"La":0.2,"Yb":0.8}'   (from ICSD loader)

The parser always returns {element: mole_fraction} dicts that sum to 1.0
for each site independently.

Feature design
--------------
Redundant features removed vs. previous version:
  • a_site_radius_std  — algebraic duplicate of sqrt(a_site_radius_variance)
  • b_site_radius_std  — same
  • n_total_elements   — exact sum of a_site_n_elements + b_site_n_elements

New features added:
  latt_param_calc       Vegard-law predicted lattice parameter (Å)
                          a = 1.9395·r_A + 3.2702·r_B + 6.1433
                          fitted on 169 data points, R² = 0.948
  a/b_site_VEC          composition-weighted valence electron count
  a/b_site_mixing_enthalpy  Miedema ΔH_mix = Σ 4Ω_ij xi xj  (kJ/mol)
  a/b_site_omega        HEA stability Ω = T_m S_config / |H_mix|
  radius_gamma          r_min / r_max across all cations on both sites
  en_site_contrast      |ēn_A − ēn_B|
  mass_site_contrast    |M̄_A − M̄_B|
  site_asymmetry        |n_A − n_B|
"""

from __future__ import annotations

import json
import re
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .. import globals

warnings.filterwarnings('ignore')

# # ── Element sets ──────────────────────────────────────────────────────────────
#
# KNOWN_A: frozenset = frozenset({
#     'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
#     'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Y',
# })
# KNOWN_B: frozenset = frozenset({
#     'Ti', 'Zr', 'Hf', 'Sn', 'Ir', 'Nb',
# })

REF_DENSITY = 1 # g/cm³ of water
# ── Physical property tables ──────────────────────────────────────────────────

IONIC_RADII_8: Dict[str, float] = {
    'La': 1.160, 'Ce': 1.143, 'Pr': 1.126, 'Nd': 1.109, 'Sm': 1.079,
    'Eu': 1.066, 'Gd': 1.053, 'Tb': 1.040, 'Dy': 1.027, 'Ho': 1.015,
    'Er': 1.004, 'Tm': 0.994, 'Yb': 0.985, 'Lu': 0.977, 'Y':  1.019,
}
IONIC_RADII_6: Dict[str, float] = {
    'Ti': 0.605, 'Zr': 0.720, 'Hf': 0.710, 'Sn': 0.690,
    'Ir': 0.625, 'Ce': 0.870, 'Nb': 0.640,
}
MOLAR_MASSES: Dict[str, float] = {
    'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24, 'Pm': 145.00,
    'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93, 'Dy': 162.50,
    'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.04, 'Lu': 174.97,
    'Y':   88.91, 'Ti':  47.87, 'Zr':  91.22, 'Hf': 178.49, 'Sn': 118.71,
    'Ir': 192.22, 'Nb':  92.91, 'O':   16.00,
}
ELECTRONEGATIVITY: Dict[str, float] = {
    'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14, 'Sm': 1.17,
    'Eu': 1.20, 'Gd': 1.20, 'Tb': 1.22, 'Dy': 1.23, 'Ho': 1.24,
    'Er': 1.24, 'Tm': 1.25, 'Yb': 1.10, 'Lu': 1.27, 'Y':  1.22,
    'Ti': 1.54, 'Zr': 1.33, 'Hf': 1.30, 'Sn': 1.96, 'Ir': 2.20,
    'Nb': 1.60,
}
ATOMIC_NUMBER: Dict[str, int] = {
    'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Sm': 62, 'Eu': 63,
    'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69,
    'Yb': 70, 'Lu': 71, 'Y': 39, 'Ti': 22, 'Zr': 40, 'Hf': 72,
    'Sn': 50, 'Ir': 77, 'Nb': 41,
}
VALENCE_ELECTRONS: Dict[str, float] = {
    'La': 3, 'Ce': 3, 'Pr': 3, 'Nd': 3, 'Sm': 3, 'Eu': 3,
    'Gd': 3, 'Tb': 3, 'Dy': 3, 'Ho': 3, 'Er': 3, 'Tm': 3,
    'Yb': 3, 'Lu': 3, 'Y': 3,
    'Ti': 4, 'Zr': 4, 'Hf': 4, 'Sn': 4, 'Ir': 4, 'Nb': 5,
}
MELTING_POINT: Dict[str, float] = {
    'La': 1193, 'Ce': 1068, 'Pr': 1208, 'Nd': 1297, 'Sm': 1345,
    'Eu': 1095, 'Gd': 1585, 'Tb': 1629, 'Dy': 1680, 'Ho': 1734,
    'Er': 1802, 'Tm': 1818, 'Yb': 1097, 'Lu': 1925, 'Y': 1799,
    'Ti': 1941, 'Zr': 2128, 'Hf': 2506, 'Sn': 505, 'Ir': 2719,
    'Nb': 2750,
}
# Miedema-style binary interaction parameters (kJ/mol)
_OMEGA_PAIRS: Dict[frozenset, float] = {
    frozenset({'La', 'Ce'}):  -2.0, frozenset({'La', 'Pr'}):  -2.5,
    frozenset({'La', 'Nd'}):  -3.0, frozenset({'La', 'Sm'}):  -4.0,
    frozenset({'La', 'Eu'}):  -4.0, frozenset({'La', 'Gd'}):  -4.5,
    frozenset({'La', 'Tb'}):  -5.0, frozenset({'La', 'Dy'}):  -5.5,
    frozenset({'La', 'Ho'}):  -5.5, frozenset({'La', 'Er'}):  -6.0,
    frozenset({'La', 'Tm'}):  -6.0, frozenset({'La', 'Yb'}):  -6.0,
    frozenset({'La', 'Lu'}):  -6.5, frozenset({'La', 'Y'}):   -3.5,
    frozenset({'Gd', 'Lu'}):  -2.5,
    frozenset({'Ti', 'Zr'}):  -9.0, frozenset({'Ti', 'Hf'}):  -8.5,
    frozenset({'Zr', 'Hf'}):  -3.0, frozenset({'Ti', 'Sn'}):  -6.0,
    frozenset({'Zr', 'Sn'}):  -5.0,
}
_DEFAULT_OMEGA = -3.0

# Vegard-law coefficients fitted on 169 pyrochlore data points, R² = 0.948
_VEGARD_K_A = 1.9395
_VEGARD_K_B = 3.2702
_VEGARD_K_0 = 6.1433

R_GAS = 8.314   # J/(mol·K)
K_B = 1.380694 * np.float_power(10,-23) # J/K


# ── Formula parser ─────────────────────────────────────────────────────────────

def _looks_like_formula(s: str) -> bool:
    """True when s contains digits or parentheses — i.e. a chemical formula."""
    return bool(re.search(r'[\d(]', s))


def _parse_pymatgen_formula(
    formula: str,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Parse a full pyrochlore formula with pymatgen and return per-site
    mole-fraction dicts.

    Ce ambiguity is resolved by stoichiometry: Ce is assigned to the site
    whose total would thereby become closest to 2.0.
    """
    try:
        from pymatgen.core import Composition as PmgComp
    except ImportError:
        raise ImportError(
            "pymatgen is required for formula-string parsing.\n"
            "Install with:  pip install pymatgen"
        )
    pmg = PmgComp(formula)
    raw = {str(el): float(amt) for el, amt in pmg.items() if str(el) != 'O'}

    a_raw = {e: v for e, v in raw.items() if e in globals.KNOWN_A and e != 'Ce'}
    b_raw = {e: v for e, v in raw.items() if e in globals.KNOWN_B and e != 'Ce'}
    unkn  = {e: v for e, v in raw.items() if e not in globals.KNOWN_A and e not in globals.KNOWN_B}
    ce_amt = raw.get('Ce', 0.0)

    if ce_amt > 0.0:
        a_gap = 2.0 - sum(a_raw.values())
        b_gap = 2.0 - sum(b_raw.values())
        if a_gap >= ce_amt and a_raw:
            a_raw['Ce'] = ce_amt
        elif b_gap >= ce_amt:
            b_raw['Ce'] = ce_amt
        elif a_raw:
            a_raw['Ce'] = ce_amt
        else:
            b_raw['Ce'] = ce_amt

    def _norm(d):
        t = sum(d.values())
        return {e: v / t for e, v in d.items()} if t > 0 else {}

    return _norm(a_raw), _norm(b_raw), _norm(unkn)


def parse_composition(
    comp_str:    Optional[str] = None,
    stoich_json: Optional[str] = None,
    formula:     Optional[str] = None,
    site:        Optional[str] = None,
) -> Dict[str, float]:
    """
    Unified composition parser returning {element: mole_fraction} with Σ = 1.

    Priority: full formula > stoich JSON > comma-separated.
    """
    if formula and not pd.isna(formula):
        a_f, b_f, _ = _parse_pymatgen_formula(str(formula))
        return a_f if site == 'A' else b_f if site == 'B' else (a_f or b_f)

    if stoich_json and not pd.isna(stoich_json):
        try:
            raw = json.loads(str(stoich_json))
            total = sum(raw.values())
            if total > 0:
                return {e: v / total for e, v in raw.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    if comp_str is not None and not pd.isna(comp_str):
        s = str(comp_str).strip()
        if _looks_like_formula(s):
            return parse_composition(formula=s, site=site)
        elems = [e.strip() for e in s.split(',') if e.strip()]
        if elems:
            return {e: 1.0 / len(elems) for e in elems}

    return {}


# ── Per-site calculators ───────────────────────────────────────────────────────

def _w(comp, table):
    """Weighted sum helper; returns nan if any element is missing."""
    vals = [table.get(e, np.nan) * f for e, f in comp.items()]
    return float(sum(vals)) if vals and not any(np.isnan(v) for v in vals) else np.nan


def configurational_entropy(comp: Dict[str, float]) -> float:
    """S_config = −K_boltman Σ xᵢ ln xᵢ  [J/(mol·K)]"""
    if not comp:
        return np.nan
    return float(-K_B * sum(f * np.log(f) for f in comp.values() if f > 0))


def mean_radius(comp: Dict[str, float], radii: Dict[str, float]) -> float:
    return _w(comp, radii)


def radius_variance(comp: Dict[str, float], radii: Dict[str, float]) -> float:
    """σ²_r = Σ xᵢ (rᵢ − r̄)²"""
    r_bar = mean_radius(comp, radii)
    if np.isnan(r_bar):
        return np.nan
    return float(sum(f * (radii.get(e, np.nan) - r_bar) ** 2 for e, f in comp.items()))


def delta_parameter(comp: Dict[str, float], radii: Dict[str, float]) -> float:
    """δ = √[Σ xᵢ (1 − rᵢ/r̄)²]  — normalised lattice distortion index"""
    r_bar = mean_radius(comp, radii)
    if np.isnan(r_bar) or r_bar == 0:
        return np.nan
    return float(np.sqrt(sum(
        f * (1 - radii.get(e, np.nan) / r_bar) ** 2 for e, f in comp.items()
    )))


def en_mean(comp):     return _w(comp, ELECTRONEGATIVITY)
def en_variance(comp):
    mu = en_mean(comp)
    if np.isnan(mu):
        return np.nan
    return float(sum(f * (ELECTRONEGATIVITY.get(e, np.nan) - mu)**2 for e, f in comp.items()))
def mean_atomic_number(comp): return _w(comp, ATOMIC_NUMBER)
def mean_molar_mass(comp):    return _w(comp, MOLAR_MASSES)
def mean_VEC(comp):           return _w(comp, VALENCE_ELECTRONS)
def mean_melting_point(comp): return _w(comp, MELTING_POINT)


def mixing_enthalpy(comp: Dict[str, float]) -> float:
    """ΔH_mix = Σ_{i>j} 4Ω_ij xᵢ xⱼ  [kJ/mol]"""
    if len(comp) < 2:
        return 0.0
    elems = list(comp.keys())
    h = 0.0
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            omega = _OMEGA_PAIRS.get(frozenset({elems[i], elems[j]}), _DEFAULT_OMEGA)
            h += 4.0 * omega * comp[elems[i]] * comp[elems[j]]
    return float(h)


def omega_parameter(comp: Dict[str, float]) -> float:
    """Ω = T̄_m · S_config / |ΔH_mix| — HEA solid-solution stability."""
    s = configurational_entropy(comp)
    h = mixing_enthalpy(comp)
    tm = mean_melting_point(comp)
    if np.isnan(s) or np.isnan(tm) or abs(h) < 1e-9:
        return np.nan
    return float(tm * s / (abs(h) * 1000))


def radius_gamma(comp_a: Dict[str, float], comp_b: Dict[str, float]) -> float:
    """γ = r_min / r_max across all cations on both sites combined."""
    all_r = [IONIC_RADII_8.get(e, np.nan) for e in comp_a] + \
            [IONIC_RADII_6.get(e, np.nan) for e in comp_b]
    all_r = [r for r in all_r if not np.isnan(r)]
    if len(all_r) < 2:
        return np.nan
    return float(min(all_r) / max(all_r))


def a_site_b_site_radius_ratio(r_a: float, r_b: float) -> float:
    """r̄_A / r̄_B — pyrochlore stability indicator (ideal: 1.46–1.78)"""
    if np.isnan(r_a) or np.isnan(r_b) or r_b == 0:
        return np.nan
    return float(r_a / r_b)


def phonon_scattering_factor(s: float, delta: float) -> float:
    return float(s * delta) if not (np.isnan(s) or np.isnan(delta)) else np.nan


def latt_param_calc(r_a: float, r_b: float) -> float:
    """
    Vegard-law estimate of lattice parameter for A₂B₂O₇ pyrochlore (Å).
      a_calc = 1.9395·r̄_A + 3.2702·r̄_B + 6.1433
    R² = 0.948, RMSE = 0.045 Å on 169 training points.
    """
    if np.isnan(r_a) or np.isnan(r_b):
        return np.nan
    return float(_VEGARD_K_A * r_a + _VEGARD_K_B * r_b + _VEGARD_K_0)


def theoretical_density(a_comp, b_comp, lattice_a: float) -> float:
    """ρ_th = 8M / (V_cell · N_A)  for A₂B₂O₇  [g/cm³]"""
    if np.isnan(lattice_a) or lattice_a <= 0:
        return np.nan
    M = (2 * sum(MOLAR_MASSES.get(e, np.nan) * f for e, f in a_comp.items())
       + 2 * sum(MOLAR_MASSES.get(e, np.nan) * f for e, f in b_comp.items())
       + 7 * MOLAR_MASSES['O'])
    V = (lattice_a * 1e-8) ** 3
    return float(8 * M / (V * 6.022e23))

def rel_to_calc_density(rel_density: float) -> float:
    """act_dens = rel_density * ref_density  [g/cm³]"""
    return rel_density * REF_DENSITY


def _safe_add(a: float, b: float) -> float:
    if np.isnan(a) and np.isnan(b):
        return np.nan
    return float((0.0 if np.isnan(a) else a) + (0.0 if np.isnan(b) else b))


# ── Row-level feature builder ──────────────────────────────────────────────────

def build_features_for_row(row: pd.Series) -> Dict[str, float]:
    """
    Compute all engineered features for one dataset row.

    Input resolution per site:
      1. 'Composition' column contains a full pyrochlore formula
      2. 'a_stoich_json' / 'b_stoich_json' columns (ICSD non-equiatomic)
      3. 'Sample A' / 'Sample B' comma-separated element lists
    """
    composition = row.get('Composition', None)

    if composition and not pd.isna(composition) and _looks_like_formula(str(composition)):
        try:
            a_comp, b_comp, _ = _parse_pymatgen_formula(str(composition))
        except Exception:
            a_comp, b_comp = {}, {}
    else:
        a_comp, b_comp = {}, {}

    if not a_comp:
        a_comp = parse_composition(
            comp_str=row.get('Sample A', None),
            stoich_json=row.get('a_stoich_json', None),
        )
    if not b_comp:
        b_comp = parse_composition(
            comp_str=row.get('Sample B', None),
            stoich_json=row.get('b_stoich_json', None),
        )

    lattice_a = row.get('Lattice Parameter (Angstrom)', np.nan)
    if pd.isna(lattice_a):
        lattice_a = row.get('Lattice Parameter a (A)', np.nan)
    try:
        lattice_a = float(lattice_a)
    except (TypeError, ValueError):
        lattice_a = np.nan

    # A-site
    a_n   = len(a_comp)
    a_S   = configurational_entropy(a_comp)
    a_r   = mean_radius(a_comp, IONIC_RADII_8)
    a_var = radius_variance(a_comp, IONIC_RADII_8)
    a_del = delta_parameter(a_comp, IONIC_RADII_8)
    a_en  = en_mean(a_comp)
    a_env = en_variance(a_comp)
    a_Z   = mean_atomic_number(a_comp)
    a_M   = mean_molar_mass(a_comp)
    a_VEC = mean_VEC(a_comp)
    a_Hm  = mixing_enthalpy(a_comp)
    a_Om  = omega_parameter(a_comp)

    # B-site
    b_n   = len(b_comp)
    b_S   = configurational_entropy(b_comp)
    b_r   = mean_radius(b_comp, IONIC_RADII_6)
    b_var = radius_variance(b_comp, IONIC_RADII_6)
    b_del = delta_parameter(b_comp, IONIC_RADII_6)
    b_en  = en_mean(b_comp)
    b_env = en_variance(b_comp)
    b_Z   = mean_atomic_number(b_comp)
    b_M   = mean_molar_mass(b_comp)
    b_VEC = mean_VEC(b_comp)
    b_Hm  = mixing_enthalpy(b_comp)
    b_Om  = omega_parameter(b_comp)

    # Cross-site
    total_S   = _safe_add(a_S, b_S)
    total_del = _safe_add(a_del, b_del)

    return {
        'a_site_n_elements':        float(a_n),
        'a_site_entropy':           a_S,
        'a_site_mean_radius':       a_r,
        'a_site_radius_variance':   a_var,
        'a_site_delta':             a_del,
        'a_site_en_mean':           a_en,
        'a_site_en_variance':       a_env,
        'a_site_mean_atomic_num':   a_Z,
        'a_site_mean_molar_mass':   a_M,
        'a_site_VEC':               a_VEC,
        'a_site_mixing_enthalpy':   a_Hm,
        'a_site_omega':             a_Om,
        'b_site_n_elements':        float(b_n),
        'b_site_entropy':           b_S,
        'b_site_mean_radius':       b_r,
        'b_site_radius_variance':   b_var,
        'b_site_delta':             b_del,
        'b_site_en_mean':           b_en,
        'b_site_en_variance':       b_env,
        'b_site_mean_atomic_num':   b_Z,
        'b_site_mean_molar_mass':   b_M,
        'b_site_VEC':               b_VEC,
        'b_site_mixing_enthalpy':   b_Hm,
        'b_site_omega':             b_Om,
        'a_b_radius_ratio':         a_site_b_site_radius_ratio(a_r, b_r),
        'total_entropy':            total_S,
        'total_delta':              total_del,
        'phonon_scattering_factor': phonon_scattering_factor(total_S, total_del),
        'en_site_contrast':         abs(a_en - b_en) if not (np.isnan(a_en) or np.isnan(b_en)) else np.nan,
        'mass_site_contrast':       abs(a_M  - b_M)  if not (np.isnan(a_M)  or np.isnan(b_M))  else np.nan,
        'site_asymmetry':           float(abs(a_n - b_n)),
        'radius_gamma':             radius_gamma(a_comp, b_comp),
        'latt_param_calc':          latt_param_calc(a_r, b_r),
        # Lattice-derived (target for lattice model; input feature for thermal)
        'lattice_parameter':        lattice_a,
        'lattice_volume':           lattice_a ** 3 if not np.isnan(lattice_a) else np.nan,
        'density_theoretical':      theoretical_density(a_comp, b_comp, lattice_a),
    }


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append all engineered features as new columns to a DataFrame."""
    records = [build_features_for_row(row) for _, row in df.iterrows()]
    feat_df = pd.DataFrame(records, index=df.index)
    existing = [c for c in feat_df.columns if c in df.columns]
    feat_df = feat_df.drop(columns=existing)
    return pd.concat([df, feat_df], axis=1)


# ── Feature-name lists ─────────────────────────────────────────────────────────

FEATURE_COLS: List[str] = [
    'a_site_n_elements',    'a_site_entropy',        'a_site_mean_radius',
    'a_site_radius_variance','a_site_delta',
    'a_site_en_mean',       'a_site_en_variance',
    'a_site_mean_atomic_num','a_site_mean_molar_mass',
    'a_site_VEC',           'a_site_mixing_enthalpy', 'a_site_omega',
    'b_site_n_elements',    'b_site_entropy',         'b_site_mean_radius',
    'b_site_radius_variance','b_site_delta',
    'b_site_en_mean',       'b_site_en_variance',
    'b_site_mean_atomic_num','b_site_mean_molar_mass',
    'b_site_VEC',           'b_site_mixing_enthalpy', 'b_site_omega',
    'a_b_radius_ratio',     'total_entropy',          'total_delta',
    'phonon_scattering_factor',
    'en_site_contrast',     'mass_site_contrast',
    'site_asymmetry',       'radius_gamma',           'latt_param_calc',
]

LATTICE_EXTRA_FEATURES = FEATURE_COLS
THERMAL_EXTRA_FEATURES = FEATURE_COLS + ['lattice_parameter']


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Test 1: comma-separated equiatomic ===")
    r1 = pd.Series({'Sample A': 'Pr,Sm,Gd,Ho,Lu', 'Sample B': 'Ti',
                    'Lattice Parameter (Angstrom)': 10.178})
    f1 = build_features_for_row(r1)
    for k, v in f1.items():
        tag = f"{v:.5g}" if not (isinstance(v, float) and np.isnan(v)) else "NaN"
        print(f"  {k:<32}: {tag}")

    print("\n=== Test 2: full pyrochlore formula ===")
    r2 = pd.Series({'Composition': '(La0.2Yb0.8)2(Zr0.7Ce0.3)2O7',
                    'Lattice Parameter (Angstrom)': 10.36})
    f2 = build_features_for_row(r2)
    print(f"  a_site_mean_radius  : {f2['a_site_mean_radius']:.4f}")
    print(f"  b_site_mean_radius  : {f2['b_site_mean_radius']:.4f}")
    print(f"  latt_param_calc     : {f2['latt_param_calc']:.4f}")
    print(f"  radius_gamma        : {f2['radius_gamma']:.4f}")

    print("\n=== Test 3: stoich JSON ===")
    r3 = pd.Series({'Sample A': 'Ce,Gd',
                    'a_stoich_json': '{"Ce":0.05,"Gd":0.95}',
                    'Sample B': 'Ti', 'b_stoich_json': '{"Ti":1.0}'})
    f3 = build_features_for_row(r3)
    exp = 0.95 * 1.053 + 0.05 * 1.143
    print(f"  a_site_mean_radius  : {f3['a_site_mean_radius']:.4f}  (expected {exp:.4f})")
    print(f"  latt_param_calc     : {f3['latt_param_calc']:.4f}")

    print(f"\nFeature count: {len(FEATURE_COLS)} compositional + 1 lattice = "
          f"{len(THERMAL_EXTRA_FEATURES)} (thermal)")