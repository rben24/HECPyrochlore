"""
Stage 1: Rule-of-Mixtures Gap Filler for Pristine A2B2O7 Pyrochlore Data
=========================================================================
Fills missing values in the pristine single-phase dataset using deterministic
lookups and elemental Rule-of-Mixtures estimates.

Fill priority / confidence:
  ★★★★  Exact lookup    → Ionic Radius A/B, Electronegativity A/B
  ★★★☆  Empirical fit   → Lattice Parameter  (±0.05 Å, <0.5% for Ln pyrochlores)
  ★★★☆  Crystal formula → Density            (exact given lattice parameter)
  ★★☆☆  Elemental RoM   → Bulk / Shear / Young's Modulus, Poisson Ratio,
                           Thermal Conductivity, Thermal Expansion

Columns NOT touched (require DFT / experiment):
  Formation Energy, Energy per Atom, Band Gap, Band Gap Type,
  Magnetic Moment, Enthalpy, Energy Above Hull, AEL Debye Temperature,
  Temperature, Valence, Synthesis Method, compound_type, data_source

Usage
-----
    from fill_pristine_gaps import fill_pristine_gaps
    df_filled, report = fill_pristine_gaps(df_pristine)
    print(report)
    df_filled.to_csv("pristine_filled.csv", index=False)
"""

import math
import re
import warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from src.globals import (
        IONIC_RADII_8  as _RADII_8,
        IONIC_RADII_6  as _RADII_6,
        ELECTRONEGATIVITY as _EN,
        MOLAR_MASSES   as _MM,
        NA,
)

_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent
DATA  = _PROJECT / 'data'
INPUT_FILE = DATA / 'raw' / 'element_database.csv'
OUTPUT_FILE = DATA / 'processed' / 'pristine_pyrochlore.csv'

# Avogadro's number
_NA = NA

# ── Empirical lattice-parameter formula ────────────────────────────────────────
# Derived from a linear fit on ~30 experimentally confirmed pyrochlores:
#   a (Å) = α·r_A(VIII) + β·r_B(VI) + γ
# Coefficients validated against La2Zr2O7 (10.802 Å), Gd2Zr2O7 (10.520 Å),
# Y2Ti2O7 (10.090 Å), La2Hf2O7 (10.773 Å), La2Sn2O7 (10.713 Å), etc.
_LP_ALPHA = 2.636   # r_A coefficient
_LP_BETA  = 2.957   # r_B coefficient
_LP_GAMMA = 5.615   # intercept (Å)


_ELEM_PROPS = pd.DataFrame(pd.read_csv(INPUT_FILE))#.set_index('Element').to_dict('index')


# =============================================================================
# FORMULA PARSER
# =============================================================================

_FORMULA_RE = re.compile(r"^([A-Z][a-z]?)([0-9.]+)?([A-Z][a-z]?)([0-9.]+)?O([0-9.]+)$")


def _parse_ab(formula: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract (A_symbol, B_symbol) from formulas like 'A2B2O7', 'Bi2Ru2O6.952', etc.
    Normalizes to A2B2O7 stoichiometry.
    Returns (None, None) on failure.
    """
    if not isinstance(formula, str):
        return None, None

    formula = formula.strip()
    m = _FORMULA_RE.match(formula)

    if not m:
        return None, None

    a_symbol = m.group(1)
    a_coeff = float(m.group(2)) if m.group(2) else 1.0
    b_symbol = m.group(3)
    b_coeff = float(m.group(4)) if m.group(4) else 1.0
    o_coeff = float(m.group(5))

    # Check if stoichiometry is approximately A2B2O7 (with tolerance for rounding)
    # We do this by checking if the ratios match
    tolerance = 0.1
    if (abs(a_coeff - 2.0) < tolerance and
            abs(b_coeff - 2.0) < tolerance and
            abs(o_coeff - 7.0) < tolerance):
        return a_symbol, b_symbol

    return None, None


# =============================================================================
# INDIVIDUAL PROPERTY ESTIMATORS
# =============================================================================

def _ionic_radius_A(a_elem: str) -> Optional[float]:
    return _RADII_8.get(a_elem)


def _ionic_radius_B(b_elem: str) -> Optional[float]:
    return _RADII_6.get(b_elem)


def _en_A(a_elem: str) -> Optional[float]:
    return _EN.get(a_elem)


def _en_B(b_elem: str) -> Optional[float]:
    return _EN.get(b_elem)


def _lattice_parameter(a_elem: str, b_elem: str) -> Optional[float]:
    """
    Empirical ionic-radii linear model for pyrochlore lattice parameter.
    Valid for lanthanide A-site + {Ti, Zr, Hf, Sn, Nb, Ir, ...} B-site.
    Estimated accuracy: ±0.05 Å (<0.5 % for Ln pyrochlores).
    """
    r_A = _RADII_8.get(a_elem)
    r_B = _RADII_6.get(b_elem)
    if r_A is None or r_B is None:
        return None
    return _LP_ALPHA * r_A + _LP_BETA * r_B + _LP_GAMMA


def _density(a_elem: str, b_elem: str, a_ang: float) -> Optional[float]:
    """
    Crystallographic density (g/cm³) for A2B2O7 pyrochlore.
    Z = 8 formula units per conventional cubic cell.

    ρ = 8 · M_formula / (N_A · a³)
    """
    m_A = _MM.get(a_elem)
    m_B = _MM.get(b_elem)
    m_O = _MM.get('O', 16.00)
    if m_A is None or m_B is None or a_ang is None or math.isnan(a_ang):
        return None
    m_formula = 2 * m_A + 2 * m_B + 7 * m_O       # g/mol per formula unit
    a_cm = a_ang * 1e-8                              # Å → cm
    vol_cm3 = a_cm ** 3
    rho = (8 * m_formula) / (_NA * vol_cm3)          # g/cm³
    return rho


def _elemental_cation_rom(
    a_elem: str,
    b_elem: str,
    prop_key: str,
    # el_df: pd.DataFrame,
) -> Optional[float]:
    """
    Equal-weight (50/50) Rule of Mixtures over the two cation species only.
    Oxygen is excluded because elemental O properties are not meaningful
    for an oxide compound.

    Returns None if either cation's property is unavailable.
    """
    val_A = _ELEM_PROPS.loc[_ELEM_PROPS['Element'] == a_elem, prop_key].values[0]
    val_B = _ELEM_PROPS.loc[_ELEM_PROPS['Element'] == b_elem, prop_key].values[0]
    if val_A is None or val_B is None:
        return None
    return 0.5 * val_A + 0.5 * val_B

def _elemental_rom(
    a_elem: str,
    b_elem: str,
    prop_key: str,
) -> Optional[float]:
    """
    Weight (2/2/7)/11 Rule of Mixtures over the two cation species and O.

    Returns None if either cation's property is unavailable.
    """
    val_A = _ELEM_PROPS.loc[_ELEM_PROPS['Element'] == a_elem, prop_key].values[0]
    val_B = _ELEM_PROPS.loc[_ELEM_PROPS['Element'] == b_elem, prop_key].values[0]
    val_C = _ELEM_PROPS.loc[_ELEM_PROPS['Element'] == 'O', prop_key].values[0]
    if val_A is None or val_B is None or val_C is None:
        return None
    return 2/11 * val_A + 2/11 * val_B + 7/11 * val_C


def _shear_from_E_nu(E: float, nu: float) -> Optional[float]:
    """G = E / (2·(1+ν)) — exact for isotropic materials."""
    if E is None or nu is None or math.isnan(E) or math.isnan(nu):
        return None
    return E / (2.0 * (1.0 + nu))


# =============================================================================
# FILL DISPATCH TABLE
# =============================================================================
# Each entry: (csv_column, filler_function_or_method, confidence_tag)
# The filler receives (a_elem, b_elem, current_row_Series) and returns a value.

def _make_fillers():
    """Return ordered list of (col_name, filler_fn, confidence) tuples."""

    def fill_ir_A(a, b, row):
        return _ionic_radius_A(a)

    def fill_ir_B(a, b, row):
        return _ionic_radius_B(b)

    def fill_en_A(a, b, row):
        return _en_A(a)

    def fill_en_B(a, b, row):
        return _en_B(b)

    def fill_lattice(a, b, row):
        return _lattice_parameter(a, b)

    def fill_density(a, b, row):
        # Prefer existing lattice parameter; fall back to estimated one
        a_val = _coerce(row.get('Lattice Parameter (Angstrom)'))
        if a_val is None:
            a_val = _lattice_parameter(a, b)
        return _density(a, b, a_val)

    def fill_bulk(a, b, row):
        return _elemental_cation_rom(a, b, 'Bulk Modulus')

    def fill_shear(a, b, row):
        # Try direct elemental RoM first, then derive from E + nu
        g = _elemental_cation_rom(a, b, 'Shear Modulus')    # use bulk as proxy if needed
        E  = _elemental_cation_rom(a, b, 'Youngs Modulus')
        nu = _elemental_cation_rom(a, b, 'Poissons Ratio')
        if g:
            return g
        return _shear_from_E_nu(E, nu)

    def fill_youngs(a, b, row):
        return _elemental_cation_rom(a, b, 'Youngs Modulus')

    def fill_poisson(a, b, row):
        return _elemental_cation_rom(a, b, 'Poissons Ratio')

    def fill_k_therm(a, b, row):
        return _elemental_rom(a, b, 'Thermal Conductivity')

    def fill_cte(a, b, row):
        return _elemental_cation_rom(a, b, 'Thermal Expansion')

    def fill_vickers(a, b, row):
        return _elemental_cation_rom(a, b, 'Vickers Hardness')

    return [
        # (column_name,                     filler_fn,    confidence_stars)
        ('Ionic Radius A (Å)',        fill_ir_A,    '★★★★'),
        ('Ionic Radius B (Å)',        fill_ir_B,    '★★★★'),
        ('Electronegativity A',              fill_en_A,    '★★★★'),
        ('Electronegativity B',              fill_en_B,    '★★★★'),
        ('Lattice Parameter (Å)',     fill_lattice, '★★★☆'),
        ('Density',                          fill_density, '★★★☆'),
        ('Bulk Modulus (VRH)',               fill_bulk,    '★★☆☆'),
        ('Shear Modulus (VRH)',              fill_shear,   '★★☆☆'),
        ('Youngs Modulus (VRH)',             fill_youngs,  '★★☆☆'),
        ('Poisson Ratio',                    fill_poisson, '★★☆☆'),
        ('Vickers Hardness',                 fill_vickers, '★★☆☆'),
        ('Thermal Conductivity (W/m/K)',     fill_k_therm, '★★☆☆'),
        ('Thermal Expansion',                fill_cte,     '★★☆☆'),
    ]


# =============================================================================
# HELPERS
# =============================================================================

def _coerce(val) -> Optional[float]:
    """Convert a DataFrame cell to float; return None for NaN / non-numeric."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _is_missing(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    try:
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return False   # non-numeric string → treat as present


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def fill_pristine_gaps(
    df: pd.DataFrame,
    composition_col: str = 'Composition',
    verbose: bool = True,
    tag_filled: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    Fill missing values in the pristine pyrochlore DataFrame using
    Rule-of-Mixtures / lookup-table estimates.

    Parameters
    ----------
    df               : Input DataFrame (pristine_pyrochlore.csv).
    composition_col  : Column containing 'A2B2O7' formula strings.
    verbose          : Print per-column fill summary.
    tag_filled       : If True, add a 'RoM_fill_flags' column recording
                       which properties were estimated (comma-separated).

    Returns
    -------
    df_out : pd.DataFrame  — copy of df with gaps filled where possible.
    report : str           — human-readable fill summary.
    """
    df_out = df.copy()
    fillers = _make_fillers()

    # Ensure all fillable columns exist (add as NaN if absent)
    for col, _, _ in fillers:
        if col not in df_out.columns:
            df_out[col] = np.nan

    # Per-column counters
    fill_counts: Dict[str, int] = {col: 0 for col, _, _ in fillers}
    skip_no_formula: int = 0
    skip_no_form = []
    skip_no_radii: int = 0

    # Column for fill provenance tagging
    fill_flags: List[str] = [''] * len(df_out)

    for idx in df_out.index:
        row = df_out.loc[idx]
        formula = row.get(composition_col, '')
        a_elem, b_elem = _parse_ab(str(formula))

        if a_elem is None or b_elem is None:
            skip_no_formula += 1
            skip_no_form.append(formula)
            continue

        # Check whether ionic radii are available (needed for lattice/density)
        if _RADII_8.get(a_elem) is None or _RADII_6.get(b_elem) is None:
            skip_no_radii += 1

        flags_this_row: List[str] = []

        for col, filler_fn, confidence in fillers:
            # Only fill if the cell is genuinely missing
            if not _is_missing(row[col]):
                continue

            estimated = filler_fn(a_elem, b_elem, row)
            if estimated is None:
                continue

            df_out.at[idx, col] = estimated
            fill_counts[col] += 1
            flags_this_row.append(f"{col}[{confidence}]")

            # After filling lattice parameter, immediately make it available
            # for the density filler in the same row
            if col == 'Lattice Parameter (Å)':
                row = df_out.loc[idx]   # refresh row view

        if tag_filled:
            fill_flags[idx if isinstance(idx, int) else list(df_out.index).index(idx)] = (
                '; '.join(flags_this_row)
            )

    if tag_filled:
        df_out['RoM_fill_flags'] = fill_flags

    # ── Build report ──────────────────────────────────────────────────────────
    n_rows = len(df_out)
    report_lines = [
        "=" * 68,
        "  Stage 1: Rule-of-Mixtures Gap Fill — Summary",
        f"  Rows processed : {n_rows}",
        f"  Unparseable formulae : {skip_no_formula} {skip_no_form}",
        f"  Rows w/ missing ionic radii (limited coverage) : {skip_no_radii}",
        "=" * 68,
        f"  {'Column':<38} {'Filled':>6}  {'Confidence':>12}",
        "  " + "─" * 60,
    ]
    for col, _, conf in fillers:
        n_filled = fill_counts[col]
        if n_filled > 0:
            report_lines.append(f"  {col:<38} {n_filled:>6}  {conf:>12}")

    total_filled = sum(fill_counts.values())
    report_lines += [
        "  " + "─" * 60,
        f"  Total cells filled : {total_filled}",
        "",
        "  Confidence legend:",
        "    ★★★★  Exact lookup (Shannon radii / Pauling EN)",
        "    ★★★☆  Empirical fit  (ionic-radii → lattice param; crystal formula → density)",
        "    ★★☆☆  Elemental cation RoM  (50/50 A-site : B-site, no O contribution)",
        "           — use these values as features, not ground truth",
        "=" * 68,
    ]
    report = "\n".join(report_lines)
    if verbose:
        print(report)
    return df_out, report


# =============================================================================
# QUICK DIAGNOSTICS
# =============================================================================

def coverage_report(
    df: pd.DataFrame,
    composition_col: str = 'Composition',
) -> pd.DataFrame:
    """
    Show how many values are missing per fillable column, before and after
    a dry-run of fill_pristine_gaps().

    Useful for deciding whether Stage 2 ML is needed.
    """
    fillers = _make_fillers()
    cols = [col for col, _, _ in fillers]

    before_missing = {
        col: int(df[col].isna().sum()) if col in df.columns else len(df)
        for col in cols
    }

    df_filled, _ = fill_pristine_gaps(df.copy(), composition_col=composition_col,
                                       verbose=False, tag_filled=False)

    after_missing = {
        col: int(df_filled[col].isna().sum())
        for col in cols
    }

    conf_map = {col: conf for col, _, conf in fillers}
    records = []
    for col in cols:
        bm = before_missing[col]
        am = after_missing[col]
        records.append({
            'Column':            col,
            'Missing (before)':  bm,
            'Filled by Stage 1': bm - am,
            'Still missing':     am,
            'Confidence':        conf_map[col],
            'Needs ML (Stage 2)': am > 0,
        })

    return pd.DataFrame(records)


# =============================================================================
# CLI / SMOKE TEST
# =============================================================================

if __name__ == '__main__':
    csv_path = OUTPUT_FILE

    if csv_path is None:
        print("pristine_pyrochlore.csv not found — running on synthetic sample.")
        df_in = pd.DataFrame({
            'Composition':                ['Gd2Zr2O7', 'Ho2Ti2O7', 'La2Hf2O7',
                                           'Nd2Zr2O7',  'Yb2Ti2O7', 'Dy2Sn2O7'],
            'Lattice Parameter (Å)': [10.520,    None,       10.770,
                                              None,      None,       None],
            'Density':                      [None]*6,
            'Ionic Radius A (Å)':    [None]*6,
            'Ionic Radius B (Å)':    [None]*6,
            'Electronegativity A':          [None]*6,
            'Electronegativity B':          [None]*6,
            'Bulk Modulus (GPa)':           [None]*6,
            'Shear Modulus (GPa)':          [None]*6,
            'Youngs Modulus (GPa)':         [None]*6,
            'Poisson Ratio':                [None]*6,
            'Thermal Conductivity (W/m/K)': [None]*6,
            'Thermal Expansion':            [None]*6,
        })
    else:
        df_in = pd.read_csv(csv_path)
        print(f"Loaded {len(df_in)} rows from {csv_path.name}")

    # ── coverage before / after ───────────────────────────────────────────────
    print("\n── Coverage report ──────────────────────────────────────────────")
    cov = coverage_report(df_in)
    print(cov.to_string(index=False))

    # ── fill and save ─────────────────────────────────────────────────────────
    df_out, report = fill_pristine_gaps(df_in, verbose=True)

    out_path = (csv_path.parent / 'pristine_stage1_filled.csv'
                if csv_path else Path('pristine_stage1_filled.csv'))
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved filled dataset → {out_path}")
