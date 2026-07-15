"""
Rule of Mixtures Calculator for High Entropy Pyrochlores
=========================================================
Data source  : pristine_pyrochlore.csv  (loaded at import time via load_pristine_db())
Site fractions: a_stoich_json / b_stoich_json columns in the input DataFrame

Primary entry points
--------------------
  rom_from_dataframe(df)                 – batch ROM over a DataFrame
  full_report(a_site, b_site)            – detailed single-composition report
  reload_db(csv_path)                    – reload PYROCHLORE_DB at runtime

DB management
-------------
  add_db_entry(formula, properties)      – add/update one entry in memory
                                           (auto-enriches ionic radii & EN from
                                            src.data.build_pristine lookups)
  upsert_db_entries(entries_dict)        – batch add/update entries in memory
  save_db(csv_path, backup=True)         – persist PYROCHLORE_DB → CSV
  scan_missing_endpoints(df)             – report which pristine endpoints are
                                           absent and how many rows need them
"""

import json
import math
import re
import shutil
import warnings
from itertools import product
from dataclasses import dataclass, fields as dc_fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE           = Path(__file__).resolve().parent
_PROJECT        = _HERE.parent.parent
DATA            = _PROJECT / 'data' / 'processed'
PRISTINE_DATA   = DATA / 'pristine_pyrochlore.csv'
PRISTINE_EXTRA  = DATA / 'pristine_pyrochlore_extra.csv'
HEC_DATA        = DATA / 'hec_pyrochlore.csv'
ELE_DATA        = DATA / 'element_database.csv'

# ── Element property lookups (src.data.build_pristine) ────────────────────────
try:
    from src.data.build_pristine import (
        get_ionic_radius_A,
        get_ionic_radius_B,
        get_electronegativity,
    )
    _ELEMENT_PROPS = True
except ImportError:
    _ELEMENT_PROPS = False
    warnings.warn(
        "[rom_calculator] Could not import element property functions from "
        "src.data.build_pristine.  New DB entries will NOT be auto-enriched "
        "with ionic radii / electronegativity.",
        stacklevel=2,
    )


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
    vickers_hardness:     float = None   # GPa
    fracture_toughness:   float = None   # Mpa*m^.5
    specific_heat:        float = None   # Jg^−1K^−1
    thermal_diffusivity:  float = None   # mm² s⁻¹


# =============================================================================
# CSV → PYROCHLORE_DB LOADER
# =============================================================================

# Maps PyrochloreProperties fields → canonical CSV column names (case-insensitive).
_COL_MAP: Dict[str, str] = {
    "lattice_parameter":    "lattice parameter (Å)",
    "ionic_radius_A":       "ionic radius a (Å)",
    "ionic_radius_B":       "ionic radius b (Å)",
    "electronegativity_A":  "electronegativity a",
    "electronegativity_B":  "electronegativity b",
    "formation_enthalpy":   "formation energy per atom",
    "bulk_modulus":         "bulk modulus (gpa)",
    "shear_modulus":        "shear modulus (gpa)",
    "youngs_modulus":       "youngs modulus (gpa)",
    "poisson_ratio":        "poisson ratio",
    "thermal_conductivity": "thermal conductivity (w/m/k)",
    "thermal_expansion":    "cte (k^-1)",
    "vickers_hardness":     "vickers hardness (gpa)",
    "fracture_toughness":   "fracture toughness (mpa*m^.5)",
    "specific_heat":        "specific heat (jg^−1k^−1)",
    "thermal_diffusivity":  "thermal diffusivity  (mm² s⁻¹)",
}

# Candidate column names for the A2B2O7 compound key (checked in order)
_KEY_COL = "composition"

# Reverse map: field_name → canonical CSV header (preserving original casing)
_FIELD_TO_CSV_COL: Dict[str, str] = {
    "lattice_parameter":    "Lattice Parameter (Å)",
    "ionic_radius_A":       "Ionic Radius A (Å)",
    "ionic_radius_B":       "Ionic Radius B (Å)",
    "electronegativity_A":  "Electronegativity A",
    "electronegativity_B":  "Electronegativity B",
    "formation_enthalpy":   "Formation Energy per Atom",
    "bulk_modulus":         "Bulk Modulus (GPa)",
    "shear_modulus":        "Shear Modulus (GPa)",
    "youngs_modulus":       "Youngs Modulus (GPa)",
    "poisson_ratio":        "Poisson Ratio",
    "thermal_conductivity": "Thermal Conductivity (W/m/K)",
    "thermal_expansion":    "CTE (K^-1)",
    "vickers_hardness":     "Vickers Hardness (GPa)",
    "fracture_toughness":   "Fracture Toughness (Mpa*m^.5)",
    "specific_heat":        "Specific Heat (Jg^−1K^−1)",
    "thermal_diffusivity":  "Thermal Diffusivity  (mm² s⁻¹)",
}


def load_pristine_db(
    csv_path: Union[str, Path] = PRISTINE_DATA,
) -> Dict[str, "PyrochloreProperties"]:
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
    col_map: Dict[str, str] = {c.strip().lower(): c for c in df.columns}

    key_col_lower = _KEY_COL.lower()
    if key_col_lower not in col_map:
        raise ValueError(
            f"[load_pristine_db] Key column '{_KEY_COL}' not found in {csv_path.name}.\n"
            f"  Available columns: {list(df.columns)}"
        )
    key_col = col_map[key_col_lower]

    resolved: Dict[str, str] = {}
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
# FORMULA PARSING  &  AUTO-ENRICHMENT
# =============================================================================

# Matches standard "A2B2O7" pyrochlore formula, e.g. "Gd2Zr2O7", "La2Hf2O7"
_FORMULA_RE = re.compile(r"^([A-Z][a-z]*)2([A-Z][a-z]*)2O7$")


def parse_ab_from_formula(formula: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract the A and B element symbols from an "A2B2O7" formula string.

    Returns
    -------
    (a_symbol, b_symbol) — both str if matched, both None if pattern fails.

    Examples
    --------
    >>> parse_ab_from_formula("Gd2Zr2O7")
    ('Gd', 'Zr')
    >>> parse_ab_from_formula("bad_input")
    (None, None)
    """
    m = _FORMULA_RE.match(formula.strip())
    if m:
        return m.group(1), m.group(2)
    return None, None



def enrich_properties(
    formula: str,
    props: PyrochloreProperties,
) -> PyrochloreProperties:
    """
    Fill any ``None`` ionic-radius / electronegativity fields in *props*
    using element-level lookups from ``src.data.build_pristine``.

    Only ``None`` fields are touched — any value you already set is preserved.
    If the element-property module is unavailable, *props* is returned unchanged.

    Parameters
    ----------
    formula : str
        Standard "A2B2O7" formula, e.g. ``"Gd2Zr2O7"``.
    props   : PyrochloreProperties
        Possibly partial properties object (may have ``None`` fields).

    Returns
    -------
    PyrochloreProperties
        The same object, mutated in place, with enriched fields.
    """
    if not _ELEMENT_PROPS:
        return props

    a_sym, b_sym = parse_ab_from_formula(formula)
    if a_sym is None:
        warnings.warn(
            f"[enrich_properties] Cannot parse A/B symbols from '{formula}'; "
            "skipping element-level enrichment.",
            stacklevel=2,
        )
        return props

    if props.ionic_radius_A is None:
        props.ionic_radius_A = get_ionic_radius_A(a_sym)

    if props.ionic_radius_B is None:
        props.ionic_radius_B = get_ionic_radius_B(b_sym)

    if props.electronegativity_A is None:
        props.electronegativity_A = get_electronegativity(a_sym)

    if props.electronegativity_B is None:
        props.electronegativity_B = get_electronegativity(b_sym)

    return props


# =============================================================================
# DB MUTATION  (in-memory)
# =============================================================================

def add_db_entry(
    formula: str,
    properties: Optional[PyrochloreProperties] = None,
    overwrite: bool = False,
    auto_enrich: bool = True,
) -> None:
    """
    Add or update a single entry in the in-memory ``PYROCHLORE_DB``.

    Parameters
    ----------
    formula    : "A2B2O7" string, e.g. ``"La2Hf2O7"``.
    properties : ``PyrochloreProperties`` instance (or ``None`` to create a
                 blank partial entry enriched from element lookups only).
    overwrite  : If ``False`` (default), raise a warning rather than
                 silently replacing an existing entry.
    auto_enrich: If ``True`` (default), call :func:`enrich_properties` to
                 fill any ``None`` ionic-radius / EN fields from element-level
                 lookups before storing.
    """
    formula = formula.strip()

    if formula in PYROCHLORE_DB and not overwrite:
        warnings.warn(
            f"[add_db_entry] '{formula}' already exists in PYROCHLORE_DB. "
            "Pass overwrite=True to replace it.",
            stacklevel=2,
        )
        return

    if properties is None:
        properties = PyrochloreProperties()

    if auto_enrich:
        properties = enrich_properties(formula, properties)

    PYROCHLORE_DB[formula] = properties


def upsert_db_entries(
    entries: Dict[str, PyrochloreProperties],
    overwrite: bool = True,
    auto_enrich: bool = True,
) -> None:
    """
    Batch add/update entries in ``PYROCHLORE_DB``.

    Parameters
    ----------
    entries     : ``{formula: PyrochloreProperties}`` mapping.
    overwrite   : Passed through to :func:`add_db_entry` (default ``True``
                  for batch upserts).
    auto_enrich : Enrich each entry before storing (default ``True``).
    """
    for formula, props in entries.items():
        add_db_entry(formula, props, overwrite=overwrite, auto_enrich=auto_enrich)
    print(f"[upsert_db_entries] Upserted {len(entries)} entries into PYROCHLORE_DB.")


# =============================================================================
# SAVE DB  (memory → CSV)
# =============================================================================

def save_db(
    csv_path: Union[str, Path] = PRISTINE_DATA,
    backup: bool = True,
) -> None:
    """
    Persist the in-memory ``PYROCHLORE_DB`` to *csv_path*.

    Merge strategy
    --------------
    • Existing rows whose formula is **not** in ``PYROCHLORE_DB`` are preserved
      verbatim (non-destructive merge).
    • Rows present in both the file and ``PYROCHLORE_DB`` are updated with the
      in-memory values.
    • Rows present only in ``PYROCHLORE_DB`` are appended.

    Parameters
    ----------
    csv_path : Destination CSV path (defaults to ``PRISTINE_DATA``).
    backup   : If ``True`` (default), write a ``.bak.csv`` copy of the
               existing file before overwriting.
    """
    csv_path = Path(csv_path)

    # ── Column name for the composition key (Title-cased for output) ──────────
    key_col = "Composition"

    # ── Build a DataFrame from PYROCHLORE_DB ──────────────────────────────────
    records = []
    for formula, props in PYROCHLORE_DB.items():
        row: Dict = {key_col: formula}
        for field_obj in dc_fields(props):
            csv_col = _FIELD_TO_CSV_COL.get(field_obj.name)
            if csv_col:
                row[csv_col] = getattr(props, field_obj.name)
        records.append(row)
    db_df = pd.DataFrame(records)

    # ── Merge with any existing CSV ───────────────────────────────────────────
    if csv_path.exists():
        if backup:
            bak_path = csv_path.with_suffix(".bak.csv")
            shutil.copy2(csv_path, bak_path)
            print(f"[save_db] Backup written → {bak_path.name}")

        existing = pd.read_csv(csv_path)

        # Normalise the key column name for merging
        existing_key_col = next(
            (c for c in existing.columns if c.strip().lower() == "composition"),
            None,
        )
        if existing_key_col and existing_key_col != key_col:
            existing = existing.rename(columns={existing_key_col: key_col})

        # Drop rows that are being replaced by PYROCHLORE_DB entries
        if existing_key_col:
            existing = existing[~existing[key_col].isin(db_df[key_col])]

        merged = pd.concat([existing, db_df], ignore_index=True, sort=False)
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        merged = db_df

    merged.to_csv(csv_path, index=False)
    print(f"[save_db] {len(PYROCHLORE_DB)} entries saved → {csv_path}")


# =============================================================================
# DIAGNOSTIC  —  scan for missing endpoints in a DataFrame
# =============================================================================

def scan_missing_endpoints(
    df: pd.DataFrame,
    a_col: str = "a_stoich_json",
    b_col: str = "b_stoich_json",
) -> Dict[str, int]:
    """
    Scan *df* and report which pristine endpoints are absent from
    ``PYROCHLORE_DB`` and how many rows are affected by each.

    Returns
    -------
    dict  {formula: affected_row_count}  sorted by count descending.
    An empty dict means full DB coverage.
    """
    tally: Dict[str, int] = {}

    for _, row in df.iterrows():
        a_site = _parse_stoich_json(row.get(a_col))
        b_site = _parse_stoich_json(row.get(b_col))
        if a_site is None or b_site is None:
            continue
        for missing in get_missing_endpoints(a_site, b_site):
            tally[missing] = tally.get(missing, 0) + 1

    return dict(sorted(tally.items(), key=lambda kv: kv[1], reverse=True))


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
    """True only when every A_i2B_j2O7 endpoint is present in PYROCHLORE_DB."""
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
    Lattice distortion as deviation from Vegard's law prediction.

    Metrics: 'quadratic_elongation', 'max_deviation',
             'mean_absolute_deviation', 'weighted_average'
    """
    a_pred = calc_lattice_parameter(a_site, b_site, verbose=verbose)
    if a_pred is None:
        return None

    single_phase_params = []
    for a_elem, b_elem in product(a_site.keys(), b_site.keys()):
        key = f"{a_elem}2{b_elem}2O7"
        if key in PYROCHLORE_DB:
            val = PYROCHLORE_DB[key].lattice_parameter
            if val is not None:
                single_phase_params.append(val)

    if not single_phase_params:
        return None

    if distortion_metric == "quadratic_elongation":
        mean_sq = sum((a / a_pred) ** 2 for a in single_phase_params) / len(single_phase_params)
        return (mean_sq ** 0.5) - 1.0
    elif distortion_metric == "max_deviation":
        return max(abs(a - a_pred) / a_pred for a in single_phase_params)
    elif distortion_metric == "mean_absolute_deviation":
        return (sum(abs(a - a_pred) for a in single_phase_params)
                / len(single_phase_params) / a_pred)
    elif distortion_metric == "weighted_average":
        return sum(abs(a - a_pred) for a in single_phase_params) / a_pred
    else:
        raise ValueError(f"Unknown distortion_metric: {distortion_metric}")


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
    """ROM thermal expansion coefficient (°K⁻¹)."""
    return rule_of_mixtures(a_site, b_site,
        lambda p: p.thermal_expansion, "Thermal Expansion (°K⁻¹)", verbose)

def calc_vickers_hardness(a_site, b_site, verbose=False):
    """ROM Vickers Hardness (GPa)."""
    return rule_of_mixtures(a_site, b_site,
        lambda p: p.vickers_hardness, "Vickers Hardness (GPa)", verbose)

def calc_fracture_toughness(a_site, b_site, verbose=False):
    """ROM Fracture Toughness (Mpa*m^.5)."""
    return rule_of_mixtures(a_site, b_site,
        lambda p: p.fracture_toughness, "Fracture Toughness (Mpa*m^.5)", verbose)

def calc_specific_heat(a_site, b_site, verbose=False):
    """ROM Specific Heat (Jg^−1K^−1)."""
    return rule_of_mixtures(a_site, b_site,
        lambda p: p.specific_heat, "Specific Heat (Jg^−1K^−1)", verbose)

def calc_thermal_diffusivity(a_site, b_site, verbose=False):
    """ROM Thermal Diffusivity (mm² s⁻¹)."""
    return rule_of_mixtures(a_site, b_site,
        lambda p: p.thermal_diffusivity, "Thermal Diffusivity  (mm² s⁻¹)", verbose)

def calc_radius_ratio(a_site, b_site, verbose=False):
    """r_A / r_B stability criterion. Pyrochlore stable: 1.46–1.78."""
    r_A = calc_ionic_radius_A(a_site, b_site, verbose)
    r_B = calc_ionic_radius_B(a_site, b_site, verbose)
    return (r_A / r_B) if (r_A and r_B) else None

def calc_lattice_distortion_A(a_site, b_site, verbose=False, t_ideal=1.0):
    """Tolerance-factor distortion |t − t_ideal| using mean A/B radii."""
    r_A_mean = calc_ionic_radius_A(a_site, b_site)
    r_B_mean = calc_ionic_radius_B(a_site, b_site)
    if r_A_mean is None or r_B_mean is None or r_B_mean == 0:
        return None
    t = r_A_mean / (math.sqrt(2) * r_B_mean)
    distortion = abs(t - t_ideal)
    if verbose:
        print(f"\n  r_A_mean={r_A_mean:.6f} Å  r_B_mean={r_B_mean:.6f} Å  "
              f"t={t:.6f}  |t-{t_ideal}|={distortion:.6f}")
    return distortion

def calc_lattice_distortion_B(a_site, b_site, verbose=False, t_ideal=1.0):
    """Tolerance-factor distortion |t − t_ideal| using mean A/B radii (B-site view)."""
    return calc_lattice_distortion_A(a_site, b_site, verbose=verbose, t_ideal=t_ideal)


# =============================================================================
# ORDERED CALCULATION REGISTRY
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
    ("ROM_Vickers_Hardness",            calc_vickers_hardness),
    ("ROM_Fracture_Toughness",          calc_fracture_toughness),
    ("ROM_Specific_Heat",               calc_specific_heat),
    ("ROM_Thermal_Diffusivity",         calc_thermal_diffusivity),
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
    dict {col_name: value}
    """
    label = composition_label(a_site, b_site)
    print("=" * 64)
    print("  High-Entropy Pyrochlore — Rule of Mixtures Report")
    print(f"  Composition : {label}")
    print(f"  A-site      : { {k: round(v, 4) for k, v in a_site.items()} }")
    print(f"  B-site      : { {k: round(v, 4) for k, v in b_site.items()} }")
    print("=" * 64)

    missing = get_missing_endpoints(a_site, b_site)
    if missing:
        print(f"\n  ✗ Incomplete DB coverage — {len(missing)} endpoint(s) missing:")
        for key in missing:
            print(f"      • {key}")
        print(
            "\n  Tip: use add_db_entry() / upsert_db_entries() to register missing\n"
            "  endpoints, then call save_db() to persist them to the CSV.\n"
        )
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
        if isinstance(val, bool):
            display = f"{'Yes' if val else 'No':>10}"
        elif val is None:
            display = f"{'N/A':>10}"
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
    Accepts dict, JSON string, or Python-repr string.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                return json.loads(raw.replace("'", '"'))
            except json.JSONDecodeError:
                return None
    return None


def rom_from_dataframe(
    df: pd.DataFrame,
    a_col: str = "a_stoich_json",
    b_col: str = "b_stoich_json",
    verbose: bool = False,
    auto_add_missing: bool = True,
    auto_save: bool = True,
    csv_path: Union[str, Path] = PRISTINE_DATA,
    supplement: Optional[Dict[str, "PyrochloreProperties"]] = None,
    persist_supplement: bool = False,
) -> pd.DataFrame | None:
    """
    Compute all ROM properties for every row in *df* and return an enriched copy.

    Missing endpoint handling
    -------------------------
    When ``auto_add_missing=True`` (default), any ``A_i2B_j2O7`` formula that
    is absent from ``PYROCHLORE_DB`` is automatically added as a *partial* row
    populated with only the element-level properties derivable from the formula
    (ionic radii and Pauling EN via ``src.data.build_pristine`` lookups).

    The partial entry is stored in ``PYROCHLORE_DB`` immediately so that:

    • ROM columns that depend only on ionic radii / EN (radius ratio, EN
      difference, lattice distortions) can still be computed for that row.
    • ROM columns whose underlying property is genuinely unknown (lattice
      parameter, moduli, thermal properties) return NaN — they are not
      invented or extrapolated.

    When ``auto_save=True`` (default), after all rows have been processed the
    updated DB (including all newly created partial entries) is flushed to
    *csv_path*.  Partial rows can be filled in later by calling
    ``add_db_entry()`` / ``upsert_db_entries()`` and ``save_db()``.

    When ``auto_add_missing=False`` the old behaviour is restored: any row
    whose endpoint set is not fully covered receives NaN in all ROM columns and
    a warning is emitted.

    Parameters
    ----------
    df               : Input DataFrame.
    a_col            : Column with A-site fractions (dict or JSON string).
    b_col            : Column with B-site fractions (dict or JSON string).
    verbose          : Forward verbose flag to each ROM calculator.
    auto_add_missing : Auto-create partial DB entries for missing endpoints.
    auto_save        : Flush DB to CSV after processing (only when new entries
                       were added).
    csv_path         : CSV path used when auto_save=True.
    supplement       : Transient ``{formula: PyrochloreProperties}`` entries
                       injected only for this call.
    persist_supplement: Keep supplemental entries in PYROCHLORE_DB after the
                       call (default False — they are removed on exit).

    Returns
    -------
    DataFrame — original columns + ROM columns appended.
    """
    if a_col not in df.columns:
        raise ValueError(f"A-site column '{a_col}' not found. "
                         f"Available: {list(df.columns)}")
    if b_col not in df.columns:
        raise ValueError(f"B-site column '{b_col}' not found. "
                         f"Available: {list(df.columns)}")

    # ── Inject transient supplement entries ───────────────────────────────────
    _supplement_keys: List[str] = []
    if supplement:
        for formula, props in supplement.items():
            enriched = enrich_properties(formula, props)
            PYROCHLORE_DB[formula] = enriched
            _supplement_keys.append(formula)

    rom_col_names = [name for name, _ in _CALCS] + ["ROM_Pyrochlore_Stable"]
    _nan_row = {c: float("nan") for c in rom_col_names}

    records: List[Dict] = []
    _newly_added: List[str] = []   # track formulae auto-added this call

    try:
        for idx, row in df.iterrows():
            a_site = _parse_stoich_json(row[a_col])
            b_site = _parse_stoich_json(row[b_col])

            # ── Parse failure ──────────────────────────────────────────────
            if a_site is None or b_site is None:
                warnings.warn(
                    f"Row {idx}: could not parse site fractions "
                    f"(a={row[a_col]!r}, b={row[b_col]!r}). "
                    "ROM columns set to NaN.",
                    stacklevel=2,
                )
                records.append(_nan_row.copy())
                continue

            # ── Auto-add missing endpoints as partial entries ───────────────
            missing = get_missing_endpoints(a_site, b_site)
            if missing:
                if auto_add_missing:
                    for formula in missing:
                        # add_db_entry with auto_enrich fills ionic radii + EN
                        add_db_entry(
                            formula,
                            properties=None,   # blank → enriched from element lookups
                            overwrite=False,   # never clobber an existing entry
                            auto_enrich=True,
                        )
                        _newly_added.append(formula)
                        if verbose:
                            props = PYROCHLORE_DB.get(formula)
                            print(
                                f"[rom_from_dataframe] Row {idx}: auto-added "
                                f"partial entry '{formula}' "
                                f"(r_A={props.ionic_radius_A}, "
                                f"r_B={props.ionic_radius_B}, "
                                f"χ_A={props.electronegativity_A}, "
                                f"χ_B={props.electronegativity_B})"
                            )
                else:
                    # Legacy behaviour: skip and warn
                    warnings.warn(
                        f"Row {idx}: {len(missing)} endpoint(s) missing from "
                        f"PYROCHLORE_DB — ROM columns set to NaN.\n"
                        f"  Missing: {missing}\n"
                        f"  Tip: pass auto_add_missing=True to create partial "
                        "entries automatically.",
                        stacklevel=2,
                    )
                    records.append(_nan_row.copy())
                    continue

            # ── Compute ROM (partial entries produce NaN for unknown props) ─
            row_results: Dict = {}
            for col_name, func in _CALCS:
                row_results[col_name] = func(a_site, b_site, verbose=verbose)

            ratio = row_results.get("ROM_Radius_Ratio_rA_rB")
            row_results["ROM_Pyrochlore_Stable"] = (
                (1.46 <= ratio <= 1.78) if ratio is not None else None
            )
            records.append(row_results)

    finally:
        # ── Remove transient supplement entries (unless persisted) ────────
        if not persist_supplement:
            for key in _supplement_keys:
                PYROCHLORE_DB.pop(key, None)

    # ── Flush newly created partial entries to CSV ────────────────────────────
    unique_new = list(dict.fromkeys(_newly_added))   # preserve order, deduplicate
    if unique_new and auto_save:
        print(
            f"\n[rom_from_dataframe] Auto-added {len(unique_new)} new partial "
            f"endpoint(s) to PYROCHLORE_DB:\n"
            + "\n".join(f"    • {f}" for f in unique_new)
        )
        save_db(csv_path=PRISTINE_EXTRA, backup=False)
    elif unique_new:
        warnings.warn(
            f"[rom_from_dataframe] {len(unique_new)} new endpoint(s) were added "
            "to PYROCHLORE_DB in memory but NOT saved (auto_save=False).\n"
            "  Call save_db() to persist them.",
            stacklevel=2,
        )

    rom_df = pd.DataFrame(records, index=df.index)
    return pd.concat([df, rom_df], axis=1)


# =============================================================================
# EXAMPLE / SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    # ── Add a missing endpoint with partial data; let auto-enrich fill the rest ** ONLY EXAMPLE
    # add_db_entry(
    #     "La2Hf2O7",
    #     PyrochloreProperties(lattice_parameter=10.77, bulk_modulus=180.0),
    #     auto_enrich=True,   # fills ionic_radius_A/B, electronegativity_A/B
    # )
    # save_db()   # persist to CSV with backup

    # ── Scan a DataFrame for missing endpoints before running batch ROM ───────
    # data = pd.read_csv(HEC_DATA)
    # missing = scan_missing_endpoints(data)
    #
    # ── Batch ROM (pass supplement= for one-off additions without touching CSV)
    # df_out = rom_from_dataframe(
    #     data,
    #     supplement={
    #         "La2Hf2O7": PyrochloreProperties(lattice_parameter=10.77),
    #     },
    #     persist_supplement=False,
    # )
    # rom_cols = [c for c in df_out.columns if c.startswith("ROM_")]
    # print(df_out[["Composition"] + rom_cols].to_string(index=False))

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
    df_out = rom_from_dataframe(df_in, verbose=False)

    rom_cols = [c for c in df_out.columns if c.startswith("ROM_")]
    print("ROM Results:")
    print(df_out[["Composition"] + rom_cols].to_string(index=False))
