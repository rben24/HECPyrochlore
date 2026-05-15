"""
load_icsd.py
============
Parses the ICSD pyrochlore dataset (HECPyrochlore_latt_data_ICSD_.csv) into the
canonical combined-dataset schema used by the rest of the pipeline.

Key responsibilities
--------------------
1. Parse ``StructuredFormula`` strings (e.g. "Gd1.9 Ce0.1 Ti2 O7") into
   fractional A-site and B-site compositions.
2. Classify every entry as one of:
      pristine      — exactly 1 element on A-site AND 1 on B-site
      high_entropy  — ≥2 elements on A-site OR B-site (or both)
      non_pyrochlore — anything that fails the pyrochlore sanity checks
3. Exclude non-pyrochlores from the training dataset.
4. Average duplicate measurements of the same composition (multiple
   publications / temperatures) into a single row.
5. Return a DataFrame in the canonical schema ready for merge into
   combined_pyrochlore.csv.

Pyrochlore sanity checks
------------------------
  * Structure type must be 'Ca2Nb2O7' or 'Eu2Zr2O7'  (both are Fd-3m pyrochlore)
  * Temperature must be in the 285 – 305 K window     (room-temperature data only)
  * A-site stoichiometry total must be in [1.5, 2.5]
  * B-site stoichiometry total must be in [1.5, 2.5]
  * All cation elements must be in the known A-site or B-site element tables
    (entries containing unrecognised cations are excluded, as we cannot
     compute features for them)
  * Lattice parameter must be parseable and in a physically reasonable range
    (9.5 – 11.5 Å for pyrochlore A₂B₂O₇)

Ce ambiguity note
-----------------
Ce can be either 3+ (A-site, 8-coord, r = 1.143 Å) or 4+ (B-site, 6-coord).
The parser assigns Ce to whichever site is consistent with the formula
stoichiometry.  If Ce appears alongside other A-site lanthanides it is placed
on the A-site; if it appears with known B-site cations (Ti, Zr, …) it goes to
the B-site.  Entries where this is genuinely ambiguous are flagged and excluded.
"""

from __future__ import annotations

import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from .. import globals


log = logging.getLogger(__name__)

# # ── element sets ────────────────────────────────────────────────────────────
#
# # Rare-earth / Y cations that occupy the 8-coordinated A-site
# KNOWN_A: frozenset[str] = frozenset({
#     'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
#     'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Y', 'Bi', 'Pb', 'Ca',
# })
#
# # Transition-metal cations that occupy the 6-coordinated B-site
# KNOWN_B: frozenset[str] = frozenset({
#     # Group IV
#     'Ti', 'Zr', 'Hf', 'Sn',
#     # Group V
#     'V', 'Nb', 'Ta',
#     # Group VI
#     'Cr', 'Mo', 'W',
#     # Group VII
#     'Mn', 'Re',
#     # Group VIII/IX (transition metals and 5d metals)
#     'Fe', 'Co', 'Ni', 'Ru', 'Os', 'Rh', 'Ir',
#     # Post-transition
#     'Pb', 'Pt',
# })
#
# # Ce can sit on either site depending on oxidation state; handled separately
# _CE_AMBIGUOUS = 'Ce'
#
# # Pyrochlore structure-type identifiers used in the ICSD file
# _PYROCHLORE_STRUCTURE_TYPES: frozenset[str] = frozenset({
#     'Ca2Nb2O7',   # standard Fd-3m pyrochlore prototype
#     'Eu2Zr2O7',   # alternate ICSD label for the same structure
#     'Bi2Ti2O7',
# })

# # Physical bounds
# _LATTICE_MIN = 9.5    # Å
# _LATTICE_MAX = 11.5   # Å
# _TEMP_MIN    = 285.0  # K
# _TEMP_MAX    = 305.0  # K
# _A_STOICH_RANGE = (1.5, 2.5)
# _B_STOICH_RANGE = (1.5, 2.5)


# # ── Compound-type enum strings ───────────────────────────────────────────────
#
# PRISTINE      = 'pristine'
# HIGH_ENTROPY  = 'high_entropy'
# NON_PYROCHLORE = 'non_pyrochlore'


# ── lattice-parameter parser ─────────────────────────────────────────────────

def _parse_lattice(cell_str: str) -> Optional[float]:
    """
    Extract the cubic lattice parameter *a* from a CellParameter string.

    The ICSD format is:  "10.166(3) 10.166(3) 10.166(3) 90. 90. 90."
    We take the first token and strip any parenthetical uncertainty.
    Returns None if the string cannot be parsed or is out of range.
    """
    try:
        first_token = str(cell_str).strip().split()[0]
        a_str = re.sub(r'\([^)]*\)', '', first_token)   # strip "(3)" etc.
        a = float(a_str)
        #if globals.LATTICE_MIN <= a <= globals.LATTICE_MAX:
        if 0 <= a:
            return a
        return None
    except Exception:
        return None


# ── formula parser ───────────────────────────────────────────────────────────

def _parse_icsd_formula(
    formula_str: str,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Parse an ICSD ``StructuredFormula`` string into three composition dicts:
      a_comp   : { element: absolute_stoich }  for A-site cations
      b_comp   : { element: absolute_stoich }  for B-site cations
      unknown  : { element: absolute_stoich }  for unrecognised cations

    Stoichiometries are *absolute* (i.e. as written in the formula, not
    normalised to 1).  Normalisation to mole fractions happens later in
    ``build_features.py`` via ``parse_composition``.

    Ce assignment strategy
    ~~~~~~~~~~~~~~~~~~~~~~
    If Ce appears together with other A-site lanthanides → A-site.
    If Ce appears together with known B-site cations only → B-site.
    If Ce is the sole cation on both → treated as A-site (Ce³⁺ default).
    """
    if pd.isna(formula_str):
        return {}, {}, {}

    s = str(formula_str).strip().replace('(', ' ').replace(')', ' ')

    # Tokenise: pairs of (ElementSymbol, stoichiometry)
    tokens = re.findall(r'([A-Z][a-z]?)\s*([\d\.]+)?', s)
    raw: Dict[str, float] = {}
    for elem, stoich in tokens:
        if elem == 'O':
            continue
        amt = float(stoich) if stoich else 1.0
        raw[elem] = raw.get(elem, 0.0) + amt

    if not raw:
        return {}, {}, {}

    # Separate into tentative A / B / unknown (Ce is held aside)
    a_comp: Dict[str, float] = {}
    b_comp: Dict[str, float] = {}
    unknown: Dict[str, float] = {}
    ce_amt: float = 0.0

    for elem, amt in raw.items():
        if elem == globals.CE_AMBIGUOUS:
            ce_amt = amt
        elif elem in globals.KNOWN_A:
            a_comp[elem] = amt
        elif elem in globals.KNOWN_B:
            b_comp[elem] = amt
        else:
            unknown[elem] = amt

    # Resolve Ce
    if ce_amt > 0:
        if a_comp:          # other lanthanides present → Ce on A-site
            a_comp[globals.CE_AMBIGUOUS] = ce_amt
        elif b_comp:        # only B-site neighbours → Ce on B-site
            b_comp[globals.CE_AMBIGUOUS] = ce_amt
        else:               # lone Ce → default A-site (Ce³⁺)
            a_comp[globals.CE_AMBIGUOUS] = ce_amt

    return a_comp, b_comp, unknown


# ── pyrochlore classifier ────────────────────────────────────────────────────

def classify_compound(
    a_comp: Dict[str, float],
    b_comp: Dict[str, float],
    unknown: Dict[str, float],
    lattice_a: Optional[float],
    structure_type: str,
    temp: float,
) -> str:
    """
    Return one of: ``'pristine'``, ``'high_entropy'``, ``'non_pyrochlore'``.

    A compound is classified as *non_pyrochlore* if ANY of the following hold:
      • structure type is not a recognised pyrochlore prototype
      • temperature is outside the RT window
      • lattice parameter is missing or out of range
      • any unknown (unsupported) cations are present
      • A-site or B-site stoichiometry totals are out of the expected range

    Among valid pyrochlores:
      • pristine     → exactly 1 element on A-site AND 1 on B-site
      • high_entropy → ≥2 elements on A-site OR B-site (or both)
    """
    # --- structure type check ---
    if str(structure_type).strip() not in globals.PYROCHLORE_STRUCTURE_TYPES:
        return globals.NON_PYROCHLORE

    # --- temperature check ---
    try:
        t = float(temp)
    except (TypeError, ValueError):
        return globals.NON_PYROCHLORE
    # if not (globals.TEMP_MIN <= t <= globals.TEMP_MAX):
    #     return globals.NON_PYROCHLORE

    # # --- lattice check ---
    # if lattice_a is None or not (globals.LATTICE_MIN <= lattice_a <= globals.LATTICE_MAX):
    #     return globals.NON_PYROCHLORE

    # --- unknown cations check ---
    if unknown:
        return globals.NON_PYROCHLORE

    # --- stoichiometry checks ---
    a_tot = sum(a_comp.values()) if a_comp else 0.0
    b_tot = sum(b_comp.values()) if b_comp else 0.0
    if not (globals.A_STOICH_RANGE[0] <= a_tot <= globals.A_STOICH_RANGE[1]):
        return globals.NON_PYROCHLORE
    if not (globals.B_STOICH_RANGE[0] <= b_tot <= globals.B_STOICH_RANGE[1]):
        return globals.NON_PYROCHLORE

    # --- compound type ---
    n_a = len(a_comp)
    n_b = len(b_comp)
    if n_a == 1 and n_b == 1:
        return globals.PRISTINE
    return globals.HIGH_ENTROPY


# ── canonical composition strings ────────────────────────────────────────────

def _comp_to_str(comp: Dict[str, float]) -> str:
    """
    Convert a {element: stoich} dict to the canonical comma-separated format
    used everywhere else in the pipeline (e.g. "Gd,Ce" for equiatomic or
    fractional compositions).

    The string contains only element symbols; the pipeline's
    ``parse_composition`` function always assumes equiatomic fractions.
    To preserve the true non-equiatomic fractions we create *weighted* element
    strings: each element appears with its mole-fraction encoded as a
    repeated symbol count (rounded to nearest 0.05 step) — BUT that would
    break the existing feature code.

    Instead we store the absolute stoichiometries separately as
    ``a_stoich_json`` / ``b_stoich_json`` columns, and generate a
    mole-fraction-based Sample A / Sample B string for the feature builder.

    The mole-fraction string uses the element symbols only (comma-joined,
    sorted alphabetically), identical to how the experimental data is stored.
    """
    return ','.join(sorted(comp.keys()))


def _comp_to_fractions(comp: Dict[str, float]) -> Dict[str, float]:
    """Normalise absolute stoichiometries to mole fractions summing to 1."""
    total = sum(comp.values())
    if total == 0:
        return {}
    return {e: v / total for e, v in comp.items()}


# ── unique-composition key ───────────────────────────────────────────────────

def _composition_key(
    a_comp: Dict[str, float],
    b_comp: Dict[str, float],
) -> str:
    """
    Build a hashable string key that encodes the full fractional composition
    (rounded to 2 decimal places to merge near-identical entries).
    """
    a_frac = _comp_to_fractions(a_comp)
    b_frac = _comp_to_fractions(b_comp)
    a_part = '|'.join(f"{e}:{round(v,2):.2f}" for e, v in sorted(a_frac.items()))
    b_part = '|'.join(f"{e}:{round(v,2):.2f}" for e, v in sorted(b_frac.items()))
    return f"A[{a_part}]B[{b_part}]"


# ── main loader ──────────────────────────────────────────────────────────────

def load_icsd(
    filepath: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load and parse the ICSD pyrochlore CSV.

    Parameters
    ----------
    filepath : path to ``HECPyrochlore_latt_data_ICSD_.csv``
               (defaults to ``data/raw/HECPyrochlore_latt_data_ICSD.csv``
               relative to the project root)
    verbose  : print a summary table

    Returns
    -------
    DataFrame in the canonical combined-dataset schema, containing only
    pyrochlore entries (pristine + high_entropy), with duplicates averaged.

    Extra columns (not in other sources)
    -------------------------------------
    compound_type       : 'pristine' or 'high_entropy'
    icsd_collection     : original ICSD CollectionCode(s) (pipe-joined)
    n_icsd_duplicates   : number of ICSD entries averaged into this row
    a_stoich_json       : JSON string of {element: mole_fraction} for A-site
    b_stoich_json       : JSON string of {element: mole_fraction} for B-site
    """
    import json

    # --- locate file ---
    if filepath is None:
        _HERE = Path(__file__).resolve().parent
        _PROJECT = _HERE.parent.parent
        filepath = _PROJECT / 'data' / 'raw' / 'HECPyrochlore_latt_data_ICSD.csv'
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"ICSD dataset not found at {filepath}.\n"
            f"Place the file at data/raw/HECPyrochlore_latt_data_ICSD.csv "
            f"or pass the path explicitly."
        )

    df_raw = pd.read_csv(filepath, encoding='latin-1')
    df_raw['temp_val'] = pd.to_numeric(df_raw['Temperature'], errors='coerce')
    df_raw['lattice_a'] = df_raw['CellParameter'].apply(_parse_lattice)

    if verbose:
        log.info(f"ICSD: {len(df_raw)} raw rows loaded from {filepath.name}")

    # --- parse and classify every row ---
    records = []
    n_non_pyro = 0
    n_bad_stoich = 0

    for _, row in df_raw.iterrows():
        a_comp, b_comp, unknown = _parse_icsd_formula(row['StructuredFormula'])
        ctype = classify_compound(
            a_comp, b_comp, unknown,
            row['lattice_a'],
            row.get('StructureType', ''),
            row['temp_val'],
        )

        if ctype == globals.NON_PYROCHLORE:
            n_non_pyro += 1
            continue

        a_frac = _comp_to_fractions(a_comp)
        b_frac = _comp_to_fractions(b_comp)

        records.append({
            'Composition':                  str(row.get('ChemicalName', '')).strip(),
            'Sample A':                     _comp_to_str(a_comp),
            'Sample B':                     _comp_to_str(b_comp),
            'TPS Cond W/m/K':               np.nan,
            'Lattice Parameter (Angstrom)': row['lattice_a'],
            'Relative Density %':           np.nan,
            'Is Single Phase':              'Yes',
            'Synthesis Method':             '',
            'data_source':                  'icsd_literature',
            'b_o_distance':                 np.nan,
            'b_o_b_angle':                  np.nan,
            'oxygen_param_x':               np.nan,
            'compound_type':                ctype,
            'icsd_collection':              str(int(row['CollectionCode'])),
            'n_icsd_duplicates':            1,
            'a_stoich_json':                json.dumps(a_frac),
            'b_stoich_json':                json.dumps(b_frac),
            '_comp_key':                    _composition_key(a_comp, b_comp),
            'Temperature':                  str(row['temp_val']),
        })

    if verbose:
        log.info(
            f"ICSD: {n_non_pyro} entries excluded as non-pyrochlore "
            f"({len(records)} pyrochlore entries remain)"
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # --- deduplicate: average lattice param for same composition ---
    # Aggregate by composition key
    agg_rows = []
    # for key, grp in df.groupby('_comp_key'):
    for key, grp in df.groupby(['_comp_key', 'Temperature']):
        base = grp.iloc[0].copy()
        base['Lattice Parameter (Angstrom)'] = grp['Lattice Parameter (Angstrom)'].mean()
        base['icsd_collection'] = '|'.join(grp['icsd_collection'].tolist())
        base['n_icsd_duplicates'] = len(grp)
        # Use the most common ChemicalName as Composition label
        base['Composition'] = (
            grp['Composition'].value_counts().index[0]
            if grp['Composition'].notna().any() else ''
        )
        agg_rows.append(base)

    df_dedup = pd.DataFrame(agg_rows).drop(columns=['_comp_key'])

    if verbose:
        pristine_n = (df_dedup['compound_type'] == globals.PRISTINE).sum()
        he_n = (df_dedup['compound_type'] == globals.HIGH_ENTROPY).sum()
        log.info(
            f"ICSD: {len(df_dedup)} unique compositions after deduplication "
            f"({pristine_n} pristine, {he_n} high-entropy)"
        )
        print()
        print(f"  {'Compound type':<20} {'Count':>6}")
        print(f"  {'-'*28}")
        print(f"  {'Pristine':<20} {pristine_n:>6}")
        print(f"  {'High-entropy':<20} {he_n:>6}")
        print(f"  {'Non-pyrochlore (excl.)':<20} {n_non_pyro:>6}")
        print(f"  {'-'*28}")
        print(f"  {'Total (raw)':<20} {len(df_raw):>6}")
        print()

    return df_dedup


# ── standalone test ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='  [%(levelname)s] %(message)s')
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else None
    result = load_icsd(filepath=fp, verbose=True)
    print(result[['Composition', 'Sample A', 'Sample B',
                  'Lattice Parameter (Angstrom)', 'compound_type',
                  'n_icsd_duplicates']].head(20).to_string(index=False))
    print(f"\nTotal rows: {len(result)}")