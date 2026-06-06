"""
Rule of Mixtures Calculator for High Entropy Pyrochlores
=========================================================
Data source  : pristine_pyrochlore.csv  (loaded at import time via load_pristine_db())
Site fractions: a_stoich_json / b_stoich_json columns in the input DataFrame

Primary entry points
--------------------
  rom_from_dataframe(df)          – batch ROM over a DataFrame
  full_report(a_site, b_site)     – detailed single-composition report
  reload_db(csv_path)             – reload PYROCHLORE_DB at runtime
"""

import json
import math
import warnings
from itertools import product
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent
_PROJECT      = _HERE.parent.parent
DATA          = _PROJECT / 'data' / 'processed'
PRISTINE_DATA = DATA / 'pristine_pyrochlore.csv'
HEC_DATA      = DATA / 'hec_pyrochlore.csv'


# =============================================================================
# PROPERTIES DATACLASS
# =============================================================================

@dataclass
class PyrochloreProperties:
    """Properties for a single-phase A2B2O7 pyrochlore."""
    lattice_parameter:    float = None   # Å
    ionic_radius_A:       float = None   # Å  (8-coord)
    ionic_radius_B:       float = None   # Å  (6-coord)
    electronegativity_A:  float = None   # Pauling
    electronegativity_B:  float = None   # Pauling
    charge_A:             int   = 3      # formal oxidation state
    charge_B:             int   = 4
    formation_enthalpy:   float = None   # kJ/mol
    bulk_modulus:         float = None   # GPa
    shear_modulus:        float = None   # GPa
    youngs_modulus:       float = None   # GPa
    poisson_ratio:        float = None
    thermal_conductivity: float = None   # W/m·K
    thermal_expansion:    float = None   # °C⁻¹


# =============================================================================
# CSV → PYROCHLORE_DB LOADER
# =============================================================================

# Maps PyrochloreProperties fields → candidate CSV column names (case-insensitive).
_COL_MAP: Dict[str, str] = {
    "lattice_parameter":    "lattice parameter (angstrom)",
    "ionic_radius_A":       "ionic radius a (angstrom)",
    "ionic_radius_B":       "ionic radius b (angstrom)",
    "electronegativity_A":  "electronegativity a",
    "electronegativity_B":  "electronegativity b",
    # "charge_A":             ["charge_a", "valence_a", "oxidation_state_a"],
    # "charge_B":             ["charge_b", "valence_b", "oxidation_state_b"],
    "formation_enthalpy":   "formation energy per atom",
    "bulk_modulus":         "bulk modulus (vrh)",
    "shear_modulus":        "shear modulus (vrh)",
    "youngs_modulus":       "youngs modulus (vrh)",
    "poisson_ratio":        "poisson ratio",
    "thermal_conductivity": "thermal conductivity (w/m/k)",
    "thermal_expansion":    "thermal_expansion",
}

# Candidate column names for the A2B2O7 compound key (checked in order)
_KEY_COL = "composition"

def load_pristine_db(
    csv_path: Union[str, Path] = PRISTINE_DATA,
) -> Dict[str, PyrochloreProperties]:
    """
    Build PYROCHLORE_DB from a CSV of single-phase pyrochlore data.

    CSV requirements
    ----------------
    • A column named exactly (case-insensitive) ``Composition`` whose
      values are formatted as "A2B2O7" (e.g. "Gd2Zr2O7").
    • Any subset of columns listed in _COL_MAP. Unrecognised columns
      are silently ignored.

    Returns
    -------
    dict  {formula_str: PyrochloreProperties}
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        warnings.warn(
            f"[load_pristine_db] CSV not found at:\n  {csv_path}\n"
            "  PYROCHLORE_DB will be empty — all ROM results will be NaN.",
            stacklevel=2,
        )
        return {}

    df = pd.read_csv(csv_path)

    # Build a lowercase → original-case column map for case-insensitive lookup
    col_map: Dict[str, str] = {c.strip().lower(): c for c in df.columns}

    # Locate the compound-key column
    key_col_lower = _KEY_COL.lower()
    if key_col_lower not in col_map:
        raise ValueError(
            f"[load_pristine_db] Key column '{_KEY_COL}' not found in {csv_path.name}.\n"
            f"  Available columns: {list(df.columns)}"
        )
    key_col = col_map[key_col_lower]

    # Resolve _COL_MAP entries against actual CSV columns (case-insensitive)
    resolved: Dict[str, str] = {}          # field_name → actual CSV column name
    for field_name, csv_col in _COL_MAP.items():
        actual = col_map.get(csv_col.strip().lower())
        if actual is not None:
            resolved[field_name] = actual
        else:
            warnings.warn(
                f"[load_pristine_db] Column '{csv_col}' not found in CSV "
                f"(field: {field_name}) — will be None for all entries.",
                stacklevel=2,
            )

    db: Dict[str, PyrochloreProperties] = {}

    for _, row in df.iterrows():
        key = str(row[key_col]).strip()
        if not key or key.lower() == "nan":
            continue

        kwargs: Dict[str, Optional[float]] = {}
        for field_name, actual_col in resolved.items():
            raw = row[actual_col]
            try:
                val = float(raw)
                kwargs[field_name] = None if math.isnan(val) else val
            except (ValueError, TypeError):
                kwargs[field_name] = None

        db[key] = PyrochloreProperties(**kwargs)

    return db


# Load at import time.  Call reload_db() to refresh without restarting.
PYROCHLORE_DB: Dict[str, PyrochloreProperties] = load_pristine_db()


def reload_db(csv_path: Union[str, Path] = PRISTINE_DATA) -> None:
    """Reload PYROCHLORE_DB from *csv_path* in place (no restart needed)."""
    global PYROCHLORE_DB
    PYROCHLORE_DB = load_pristine_db(csv_path)
    print(f"[reload_db] Loaded {len(PYROCHLORE_DB)} entries from {Path(csv_path).name}")

# =============================================================================
# HELPERS
# =============================================================================

def composition_label(a_site: Dict[str, float], b_site: Dict[str, float]) -> str:
    a_str = "".join(f"{e}{f:.2g}" for e, f in a_site.items())
    b_str = "".join(f"{e}{f:.2g}" for e, f in b_site.items())
    return f"({a_str})2({b_str})2O7"


def get_missing_endpoints(
    a_site: Dict[str, float],
    b_site: Dict[str, float],
) -> List[str]:
    """
    Return a list of all A_i2B_j2O7 endpoint keys that are absent from
    PYROCHLORE_DB.  An empty list means full coverage.
    """
    return [
        f"{a_elem}2{b_elem}2O7"
        for a_elem, b_elem in product(a_site.keys(), b_site.keys())
        if f"{a_elem}2{b_elem}2O7" not in PYROCHLORE_DB
    ]


def all_endpoints_available(
    a_site: Dict[str, float],
    b_site: Dict[str, float],
) -> bool:
    """
    Return True only when every A_i2B_j2O7 endpoint required by the
    composition is present in PYROCHLORE_DB.
    """
    return len(get_missing_endpoints(a_site, b_site)) == 0



# =============================================================================
# CORE RULE OF MIXTURES ENGINE
# =============================================================================

def rule_of_mixtures(
    a_site: Dict[str, float],
    b_site: Dict[str, float],
    prop_getter,
    prop_name: str = "property",
    verbose: bool = False,
) -> Optional[float]:
    """
    Weighted-average rule of mixtures over all (A_i, B_j) endpoint pairs.

        ROM = Σ_{i,j} (x_i · p_j · val_ij) / Σ_{i,j} (x_i · p_j)

    Endpoints with missing DB entries or None property values are skipped
    and their weight is excluded from the denominator.

    Note: callers in rom_from_dataframe are only reached after
    all_endpoints_available() returns True, so missing-key skips here
    serve only as a safety net for direct / full_report usage.
    """
    numerator   = 0.0
    denominator = 0.0
    missing: List[str] = []

    if verbose:
        print(f"\n  Rule of Mixtures — {prop_name}")
        print(f"  {'Phase':<22} {'Weight':>8}  {'Value':>10}  {'Contrib':>12}")
        print("  " + "─" * 58)

    for (a_elem, a_frac), (b_elem, b_frac) in product(
        a_site.items(), b_site.items()
    ):
        key    = f"{a_elem}2{b_elem}2O7"
        weight = a_frac * b_frac

        if key not in PYROCHLORE_DB:
            missing.append(key)
            if verbose:
                print(f"  {key:<22} {weight:>8.4f}  {'MISSING':>10}")
            continue

        val = prop_getter(PYROCHLORE_DB[key])
        if val is None:
            missing.append(f"{key}[{prop_name}=None]")
            if verbose:
                print(f"  {key:<22} {weight:>8.4f}  {'N/A':>10}")
            continue

        contribution = weight * val
        numerator   += contribution
        denominator += weight

        if verbose:
            print(f"  {key:<22} {weight:>8.4f}  {val:>10.4f}  {contribution:>12.6f}")

    if missing and verbose:
        print(f"  ⚠  Skipped: {missing}")

    if denominator == 0.0:
        return None

    result = numerator / denominator
    if verbose:
        print(f"  {'ROM =':>44}  {result:>12.6f}")

    return result


# =============================================================================
# PROPERTY CALCULATORS
# =============================================================================

def calc_lattice_parameter(a_site, b_site, verbose=False):
    """ROM lattice parameter (Å)."""
    return rule_of_mixtures(a_site, b_site,
        lambda p: p.lattice_parameter, "Lattice Parameter (Å)", verbose)

def calculate_lattice_distortion(
        a_site: Dict[str, float],
        b_site: Dict[str, float],
        distortion_metric: str = "weighted_average",
        verbose: bool = False,
) -> Optional[float]:
    """
    Calculate lattice distortion as deviation from Vegard's law prediction.

    The distortion quantifies how much the actual (or average) lattice
    parameters deviate from what a linear mixing model would predict.

    Parameters
    ----------
    a_site : dict
        A-site cation fractions.
    b_site : dict
        B-site cation fractions.
    distortion_metric : {'quadratic_elongation', 'max_deviation', 'mean_absolute_deviation'}
        Type of distortion metric:
        • 'quadratic_elongation': sqrt(mean((a_i / a_pred)^2)) - 1
          (analog of octahedral distortion in pyrochlores)
        • 'max_deviation': max(|a_i - a_pred| / a_pred)
          (largest single deviation)
        • 'mean_absolute_deviation': mean(|a_i - a_pred|) / a_pred
          (average normalized deviation)
    verbose : bool
        If True, print detailed breakdown.

    Returns
    -------
    float or None
        Distortion metric (typically 0–0.1 for pyrochlores), or None
        if insufficient data.
    """
    # Get ROM-predicted lattice parameter
    a_pred = calc_lattice_parameter(a_site, b_site, verbose=verbose)
    if a_pred is None:
        return None

    # Collect all single-phase lattice parameters
    single_phase_params = []
    for a_elem, b_elem in product(a_site.keys(), b_site.keys()):
        key = f"{a_elem}2{b_elem}2O7"
        if key in PYROCHLORE_DB:
            val = PYROCHLORE_DB[key].lattice_parameter
            if val is not None:
                single_phase_params.append(val)

    if not single_phase_params:
        return None

    if verbose:
        print(f"\n  Lattice Distortion — {distortion_metric}")
        print(f"  ROM prediction: {a_pred:.6f} Å")
        print(f"  Endpoints: {single_phase_params}")

    if distortion_metric == "quadratic_elongation":
        # Similar to octahedral distortion: sqrt(mean((a_i / a_avg)^2)) - 1
        mean_sq = sum((a / a_pred) ** 2 for a in single_phase_params) / len(single_phase_params)
        distortion = (mean_sq ** 0.5) - 1.0

    elif distortion_metric == "max_deviation":
        # Maximum relative deviation
        distortion = max(abs(a - a_pred) / a_pred for a in single_phase_params)

    elif distortion_metric == "mean_absolute_deviation":
        # Mean absolute relative deviation
        distortion = sum(abs(a - a_pred) for a in single_phase_params) / len(single_phase_params)
        distortion = distortion / a_pred

    elif distortion_metric == "weighted_average":
        distortion = sum(abs(a-a_pred) for a in single_phase_params) / a_pred

    else:
        raise ValueError(f"Unknown distortion_metric: {distortion_metric}")

    if verbose:
        print(f"  Distortion ({distortion_metric}): {distortion:.8f}")

    return distortion


def calc_ionic_radius_A(a_site, b_site, verbose=False):
    """ROM mean A-site ionic radius (Å, 8-coord)."""
    return rule_of_mixtures(a_site, b_site,
        lambda p: p.ionic_radius_A, "Ionic Radius A (Å)", verbose)

def calc_ionic_radius_B(a_site, b_site, verbose=False):
    """ROM mean B-site ionic radius (Å, 6-coord)."""
    return rule_of_mixtures(a_site, b_site,
        lambda p: p.ionic_radius_B, "Ionic Radius B (Å)", verbose)

def calc_electronegativity_A(a_site, b_site, verbose=False):
    """ROM mean A-site Pauling electronegativity."""
    return rule_of_mixtures(a_site, b_site,
        lambda p: p.electronegativity_A, "Electronegativity A", verbose)

def calc_electronegativity_B(a_site, b_site, verbose=False):
    """ROM mean B-site Pauling electronegativity."""
    return rule_of_mixtures(a_site, b_site,
        lambda p: p.electronegativity_B, "Electronegativity B", verbose)

def calc_electronegativity_difference(a_site, b_site, verbose=False):
    """ROM mean |χ_A − χ_B|."""
    return rule_of_mixtures(a_site, b_site,
        lambda p: abs(p.electronegativity_A - p.electronegativity_B)
                  if p.electronegativity_A is not None and p.electronegativity_B is not None
                  else None,
        "|Δχ| A−B", verbose)

def calc_bulk_modulus(a_site, b_site, verbose=False):
    """ROM bulk modulus (GPa)."""
    return rule_of_mixtures(a_site, b_site,
        lambda p: p.bulk_modulus, "Bulk Modulus (GPa)", verbose)

def calc_shear_modulus(a_site, b_site, verbose=False):
    """ROM shear modulus (GPa)."""
    return rule_of_mixtures(a_site, b_site,
        lambda p: p.shear_modulus, "Shear Modulus (GPa)", verbose)

def calc_youngs_modulus(a_site, b_site, verbose=False):
    """ROM youngs modulus (GPa)."""
    return rule_of_mixtures(a_site, b_site,
        lambda p: p.youngs_modulus, "Youngs Modulus (GPa)", verbose)

def calc_poisson_ratio(a_site, b_site, verbose=False):
    """ROM Poisson Ratio."""
    return rule_of_mixtures(a_site, b_site,
        lambda p: p.poisson_ratio, "Poisson Ratio", verbose)

def calc_thermal_conductivity(a_site, b_site, verbose=False):
    """ROM thermal conductivity (W/m·K)."""
    return rule_of_mixtures(a_site, b_site,
        lambda p: p.thermal_conductivity, "Thermal Conductivity (W/m·K)", verbose)

def calc_thermal_expansion(a_site, b_site, verbose=False):
    """ROM thermal expansion coefficient (°C⁻¹)."""
    return rule_of_mixtures(a_site, b_site,
        lambda p: p.thermal_expansion, "Thermal Expansion (°C⁻¹)", verbose)

def calc_radius_ratio(a_site, b_site, verbose=False):
    """r_A / r_B stability criterion. Pyrochlore stable: 1.46–1.78."""
    r_A = calc_ionic_radius_A(a_site, b_site, verbose)
    r_B = calc_ionic_radius_B(a_site, b_site, verbose)
    return (r_A / r_B) if (r_A and r_B) else None

def calc_lattice_distortion_A(a_site, b_site, verbose=False, t_ideal=1.0):
    """
    A-site RMS lattice distortion:
        δ_A = sqrt(Σ x_i (r_i − <r_A>)²) / <r_A>
    r_i is averaged over available B-site partners for each A element.

    Parameters:
        t_ideal: ideal tolerance for undistorted pyrochlore
    """
    # r_A_mean = calc_ionic_radius_A(a_site, b_site)
    # if r_A_mean is None:
    #     return None
    # variance = 0.0
    # for a_elem, a_frac in a_site.items():
    #     r_vals = [
    #         PYROCHLORE_DB[f"{a_elem}2{b_elem}2O7"].ionic_radius_A
    #         for b_elem in b_site
    #         if f"{a_elem}2{b_elem}2O7" in PYROCHLORE_DB
    #         and PYROCHLORE_DB[f"{a_elem}2{b_elem}2O7"].ionic_radius_A is not None
    #     ]
    #     if r_vals:
    #         r_i = sum(r_vals) / len(r_vals)
    #         variance += a_frac * (r_i - r_A_mean) ** 2
    # delta = math.sqrt(variance) / r_A_mean if r_A_mean else None
    # if verbose and delta is not None:
    #     print(f"\n  A-site δ_A = {delta:.6f}")
    # return delta
    r_A_mean = calc_ionic_radius_A(a_site, b_site)
    r_B_mean = calc_ionic_radius_B(a_site, b_site)
    if r_A_mean is None or r_B_mean is None or r_B_mean == 0:
        return None

    t = r_A_mean / (math.sqrt(2) * r_B_mean)
    distortion = abs(t - t_ideal)

    if verbose:
        print(f"\n  r_A_mean = {r_A_mean:.6f} Å")
        print(f"  r_B_mean = {r_B_mean:.6f} Å")
        print(f"  t = {t:.6f} (using denominator sqrt(2)*r_B_mean)")
        print(f"  Distortion (|t - {t_ideal}|) = {distortion:.6f}")

    return distortion

def calc_lattice_distortion_B(a_site, b_site, verbose=False, t_ideal=1.0):
    """
    B-site RMS lattice distortion:
        δ_B = sqrt(Σ p_j (r_j − <r_B>)²) / <r_B>
    r_j is averaged over available A-site partners for each B element.
    """
    # r_B_mean = calc_ionic_radius_B(a_site, b_site)
    # if r_B_mean is None:
    #     return None
    # variance = 0.0
    # for b_elem, b_frac in b_site.items():
    #     r_vals = [
    #         PYROCHLORE_DB[f"{a_elem}2{b_elem}2O7"].ionic_radius_B
    #         for a_elem in a_site
    #         if f"{a_elem}2{b_elem}2O7" in PYROCHLORE_DB
    #         and PYROCHLORE_DB[f"{a_elem}2{b_elem}2O7"].ionic_radius_B is not None
    #     ]
    #     if r_vals:
    #         r_j = sum(r_vals) / len(r_vals)
    #         variance += b_frac * (r_j - r_B_mean) ** 2
    # delta = math.sqrt(variance) / r_B_mean if r_B_mean else None
    # if verbose and delta is not None:
    #     print(f"\n  B-site δ_B = {delta:.6f}")
    # return delta
    r_A_mean = calc_ionic_radius_A(a_site, b_site)
    r_B_mean = calc_ionic_radius_B(a_site, b_site)
    if r_A_mean is None or r_B_mean is None or r_B_mean == 0:
        return None

    t = r_A_mean / (math.sqrt(2) * r_B_mean)
    distortion_B = abs(t - t_ideal)

    if verbose:
        print(f"\n  r_A_mean = {r_A_mean:.6f} Å")
        print(f"  r_B_mean = {r_B_mean:.6f} Å")
        print(f"  t = {t:.6f} (using denominator sqrt(2)*r_B_mean)")
        print(f"  B-site Distortion (|t - {t_ideal}|) = {distortion_B:.6f}")

    return distortion_B

# =============================================================================
# ORDERED CALCULATION REGISTRY
# (col_name, calculator) — drives both full_report and rom_from_dataframe
# =============================================================================

_CALCS: List[Tuple[str, callable]] = [
    ("ROM_Lattice_Parameter",           calc_lattice_parameter),
    ("ROM_Lattice_Distortion",          calculate_lattice_distortion),
    ("ROM_Ionic_Radius_A",              calc_ionic_radius_A),
    ("ROM_Ionic_Radius_B",              calc_ionic_radius_B),
    ("ROM_Radius_Ratio_rA_rB",          calc_radius_ratio),
    ("ROM_Electronegativity_A",         calc_electronegativity_A),
    ("ROM_Electronegativity_B",         calc_electronegativity_B),
    ("ROM_Electronegativity_Diff",      calc_electronegativity_difference),
    ("ROM_Lattice_Distortion_A",        calc_lattice_distortion_A),
    ("ROM_Lattice_Distortion_B",        calc_lattice_distortion_B),
    ("ROM_Bulk_Modulus_GPa",            calc_bulk_modulus),
    ("ROM_Shear_Modulus_GPa",           calc_shear_modulus),
    ("ROM_Youngs_Modulus_GPa",          calc_youngs_modulus),
    ("ROM_Poisson_Ratio",               calc_poisson_ratio),
    ("ROM_Thermal_Conductivity_W_mK",   calc_thermal_conductivity),
    ("ROM_Thermal_Expansion",           calc_thermal_expansion),
]


# =============================================================================
# FULL REPORT  (single composition)
# =============================================================================


def full_report(
    a_site: Dict[str, float],
    b_site: Dict[str, float],
    verbose: bool = True,
) -> Dict[str, Optional[float]]:
    """
    Compute all ROM properties for a single composition and print a summary.

    Parameters
    ----------
    a_site  : {element: fraction}  e.g. {"Ho": 0.5, "Gd": 0.5}
    b_site  : {element: fraction}  e.g. {"Zr": 0.5, "Hf": 0.5}
    verbose : print per-endpoint breakdown for every property

    Returns
    -------
    dict {col_name: value}  — same keys as the ROM columns added by rom_from_dataframe.
    All values are None when any endpoint is missing from PYROCHLORE_DB.
    """
    label = composition_label(a_site, b_site)
    print("=" * 64)
    print("  High-Entropy Pyrochlore — Rule of Mixtures Report")
    print(f"  Composition : {label}")
    print(f"  A-site      : { {k: round(v, 4) for k, v in a_site.items()} }")
    print(f"  B-site      : { {k: round(v, 4) for k, v in b_site.items()} }")
    print("=" * 64)

    # ── Completeness guard ────────────────────────────────────────────────────
    missing = get_missing_endpoints(a_site, b_site)
    if missing:
        print(f"\n  ✗ Incomplete DB coverage — {len(missing)} endpoint(s) missing:")
        for key in missing:
            print(f"      • {key}")
        print("\n  ROM properties cannot be calculated. Returning all None.\n")
        print("=" * 64 + "\n")
        results = {col_name: None for col_name, _ in _CALCS}
        results["ROM_Pyrochlore_Stable"] = None
        return results

    results: Dict[str, Optional[float]] = {}
    for col_name, func in _CALCS:
        results[col_name] = func(a_site, b_site, verbose=verbose)

    ratio = results.get("ROM_Radius_Ratio_rA_rB")
    results["ROM_Pyrochlore_Stable"] = (
        (1.46 <= ratio <= 1.78) if ratio is not None else None
    )

    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    for name, val in results.items():
        if isinstance(val, bool) or val is None:
            display = f"{'Yes' if val else 'No':>10}" if isinstance(val, bool) else f"{'N/A':>10}"
        else:
            display = f"{val:>10.4f}"
        print(f"  {name:<42} {display}")

    if ratio is not None:
        status = "✓ Pyrochlore-stable" if 1.46 <= ratio <= 1.78 else "✗ Outside stability window"
        print(f"\n  Stability (1.46 ≤ r_A/r_B ≤ 1.78): {status}")
    print("=" * 64 + "\n")

    return results


# =============================================================================
# DATAFRAME BATCH CALCULATOR
# =============================================================================

def _parse_stoich_json(raw) -> Optional[Dict[str, float]]:
    """
    Parse a site-fraction dict from a DataFrame cell.

    Accepts
    -------
    • dict already                 {"Ho": 0.5, "Gd": 0.5}
    • standard JSON string         '{"Ho": 0.5, "Gd": 0.5}'
    • Python-repr string           "{'Ho': 0.5, 'Gd': 0.5}"

    Returns None on parse failure.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:                                    # handle single-quote repr
                return json.loads(raw.replace("'", '"'))
            except json.JSONDecodeError:
                return None
    return None


def rom_from_dataframe(
    df: pd.DataFrame,
    a_col: str = "a_stoich_json",
    b_col: str = "b_stoich_json",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute all ROM properties for every row in *df* and return an enriched copy.

    Site fractions are read directly from *a_col* and *b_col* — no composition
    string parsing is performed.  Values are looked up in PYROCHLORE_DB which
    is loaded from pristine_pyrochlore.csv at import time (call reload_db() to
    refresh without restarting the kernel).

    Completeness requirement
    ------------------------
    Every A_i2B_j2O7 endpoint implied by the A-site × B-site Cartesian product
    must be present in PYROCHLORE_DB.  If **any** endpoint is missing, all ROM
    columns for that row are set to NaN and a warning is issued listing the
    absent structures.  Partial interpolation is intentionally not performed so
    that results are never silently biased by an incomplete endpoint set.

    Parameters
    ----------
    df      : DataFrame containing at least *a_col* and *b_col*.
    a_col   : Column with A-site fractions — dict or JSON/Python-repr string.
              e.g.  '{"Ho": 0.5, "Gd": 0.5}'
    b_col   : Column with B-site fractions — same format.
              e.g.  '{"Zr": 0.5, "Hf": 0.5}'
    verbose : Forward to each ROM calculator (very chatty on large DataFrames;
              useful for debugging individual rows).

    Returns
    -------
    DataFrame — original columns + ROM columns below.
    Rows whose fractions cannot be parsed, or whose endpoint set is not fully
    covered by PYROCHLORE_DB, receive NaN in all ROM columns.

    ROM columns appended
    --------------------
    ROM_Lattice_Parameter_A         Å
    ROM_Ionic_Radius_A              Å  (8-coord)
    ROM_Ionic_Radius_B              Å  (6-coord)
    ROM_Radius_Ratio_rA_rB          dimensionless  (stable: 1.46–1.78)
    ROM_Electronegativity_A         Pauling
    ROM_Electronegativity_B         Pauling
    ROM_Electronegativity_Diff      |χ_A − χ_B|
    ROM_Lattice_Distortion_A        δ_A  (dimensionless)
    ROM_Lattice_Distortion_B        δ_B  (dimensionless)
    ROM_Bulk_Modulus_GPa            GPa
    ROM_Shear_Modulus_GPa           GPa
    ROM_Youngs_Modulus_GPa          GPa
    ROM_Poisson_Ratio               float
    ROM_Thermal_Conductivity_W_mK   W/m·K
    ROM_Thermal_Expansion           °C⁻¹
    ROM_Pyrochlore_Stable           bool  (True if 1.46 ≤ r_A/r_B ≤ 1.78)
    """
    if a_col not in df.columns:
        raise ValueError(f"A-site column '{a_col}' not found in DataFrame. "
                         f"Available columns: {list(df.columns)}")
    if b_col not in df.columns:
        raise ValueError(f"B-site column '{b_col}' not found in DataFrame. "
                         f"Available columns: {list(df.columns)}")

    rom_col_names = [name for name, _ in _CALCS] + ["ROM_Pyrochlore_Stable"]
    _nan_row = {c: float("nan") for c in rom_col_names}
    records: List[Dict] = []

    for idx, row in df.iterrows():
        a_site = _parse_stoich_json(row[a_col])
        b_site = _parse_stoich_json(row[b_col])

        # ── Parse failure ──────────────────────────────────────────────────
        if a_site is None or b_site is None:
            warnings.warn(
                f"Row {idx}: could not parse site fractions "
                f"(a={row[a_col]!r}, b={row[b_col]!r}). "
                "ROM columns set to NaN.",
                stacklevel=2,
            )
            records.append(_nan_row.copy())
            continue

        # ── Completeness guard ─────────────────────────────────────────────
        missing = get_missing_endpoints(a_site, b_site)
        if missing:
            warnings.warn(
                f"Row {idx}: {len(missing)} endpoint(s) missing from "
                f"PYROCHLORE_DB — ROM columns set to NaN.\n"
                f"  Missing: {missing}",
                stacklevel=2,
            )
            records.append(_nan_row.copy())
            continue

        # ── All endpoints present — compute ROM ────────────────────────────
        row_results: Dict = {}
        for col_name, func in _CALCS:
            row_results[col_name] = func(a_site, b_site, verbose=verbose)

        ratio = row_results.get("ROM_Radius_Ratio_rA_rB")
        row_results["ROM_Pyrochlore_Stable"] = (
            (1.46 <= ratio <= 1.78) if ratio is not None else None
        )
        records.append(row_results)

    rom_df = pd.DataFrame(records, index=df.index)
    return pd.concat([df, rom_df], axis=1)



# =============================================================================
# EXAMPLE / SMOKE TEST
# =============================================================================

if __name__ == "__main__":

    print(f"PYROCHLORE_DB: {len(PYROCHLORE_DB)} entries loaded from pristine CSV.\n")
    if PYROCHLORE_DB:
        sample_key = next(iter(PYROCHLORE_DB))
        print(f"  Sample entry → {sample_key}: {PYROCHLORE_DB[sample_key]}\n")

    # # ── Single composition ─────────────────────────────────────────────────────
    # full_report(
    #     a_site={"Ho": 0.25, "Gd": 0.25, "Nd": 0.25, "Sm": 0.25},
    #     b_site={"Zr": 0.5,  "Hf": 0.5},
    # )

    # ── DataFrame batch ────────────────────────────────────────────────────────
    sample_data = {
        "Composition": [
            "(Ho0.5Gd0.5)2(Zr1.0)2O7",
            "(Gd0.5Nd0.5)2(Zr0.5Hf0.5)2O7",
        ],
        "Sample A": ["HG-Zr", "GN-ZH"],
        "Sample B": ["batch-1", "batch-2"],
        "a_stoich_json": [
            '{"Ho": 0.5, "Gd": 0.5}',
            '{"Gd": 0.5, "Nd": 0.5}',
        ],
        "b_stoich_json": [
            '{"Zr": 1.0}',
            '{"Zr": 0.5, "Hf": 0.5}',
        ],
        "Lattice Parameter (Angstrom)": [10.41, 10.51],
        "Thermal Conductivity (W/m/K)": [1.8,    1.6],
    }
    data = pd.read_csv(HEC_DATA)
    df_in  = pd.DataFrame(data)
    df_out = rom_from_dataframe(df_in, verbose=True)

    rom_cols = [c for c in df_out.columns if c.startswith("ROM_")]
    print("ROM Results:")
    print(df_out[["Composition"] + rom_cols].to_string(index=False))
