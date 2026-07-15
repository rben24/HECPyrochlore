"""
load_lit_extract.py
===================
Parses the literature-extraction pyrochlore dataset
(Pyrochlore_literature_extraction.csv) into the canonical combined-dataset
schema used by the rest of the pipeline.

Key responsibilities
--------------------
1. Expand compound-mixture formulas BEFORE any other processing:
     (Ca2Nb2O7)0.05(Gd2Zr2O7)0.95  →  (Ca0.05Gd0.95)2(Nb0.05Zr0.95)2O7
   Pure end-members expressed as mixtures (fraction = 0 or 1) are handled
   correctly; zero-contribution elements are dropped automatically.

2. Clean ``Composition`` strings:
     - Strip invisible Unicode artefacts (zero-width spaces, BOM, etc.)
     - Remove trailing whitespace and non-breaking spaces.
     - Deduplicate entries whose formula text was pasted twice.
     - Strip space-separated annotation tags (e.g. "(G5)").

3. Parse cleaned formulas — including parenthetical fraction notation such
   as ``(Ho1/2Y1/2)2Ti2O7`` — into A-site / B-site composition dicts.

4. Standardize every valid formula to the canonical (A1x1A2x2...)2(B1y1B2y2...)2O7
   form where fractions inside parentheses are mole fractions (sum = 1) and
   the outer subscript is always 2.  Single-element sites are written without
   parentheses: La2Ti2O7.

5. Classify every entry as one of:
       pristine       — exactly 1 element on A-site AND 1 on B-site
       high_entropy   — ≥2 elements on A-site OR B-site (or both)
       non_pyrochlore — anything that fails the pyrochlore sanity checks

6. Exclude non-pyrochlores from the training dataset.

7. Optionally average duplicate measurements of the same composition into a
   single row.

8. Return a DataFrame in the canonical schema ready for merge into
   combined_pyrochlore.csv.

Pyrochlore filter
-----------------
  * ``Phase`` column must be ``'P'`` (pyrochlore) or ``'SP'`` (single-phase).
    All other phases (F, DF, F+P, Mix, M, …) are excluded.
  * All cation elements must be in the known A-site or B-site element tables.
    Entries with unrecognised cations are excluded.

Thermal Conductivity values
---------------------------
  Values marked with ``*`` were estimated by AI from a figure; `` ` `` marks
  indicate the value is unreliable.  Both markers are stripped and the
  numeric value is retained with a separate flag column
  ``tc_ai_estimated`` (True / False).

Ce ambiguity note
-----------------
  Ce can be 3+ (A-site) or 4+ (B-site).  Resolution follows a
  stoichiometry-based tie-breaker: whichever site has the larger remaining
  deficit from 2.0 receives Ce.  If there is a tie, Ce goes to A-site (3+
  default).
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.io.abinit.abiobjects import lattice_from_abivars
from sympy.tensor.array.expressions import convert_array_to_indexed

from src import globals

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Phase values treated as pyrochlore.
PYROCHLORE_PHASES: frozenset[str] = frozenset({'P', 'SP', 'F+P', 'P+M'})

# ---------------------------------------------------------------------------
# Diagnostic accumulators (inspectable after a run)
# ---------------------------------------------------------------------------
latt_err:    list = []
struct_err:  list = []
unknown_err: list = []
stoic_err:   list = []

# ---------------------------------------------------------------------------
# Utility parsers
# ---------------------------------------------------------------------------

def _clean_and_ai(val) -> Tuple[Optional[float], bool]:
    """
    Parse val for ``*`` (AI-estimated from figure), ``± value``,
    and/or `` ` `` (unreliable). Clean and strip the value.

    :param val: float

    :return:
    (numeric_value_or_None, ai_estimated_flag)
    """
    if pd.isna(val):
        return None, False
    s = str(val)
    ai_flag = '*' in s
    s = s.split(' ', 1)[0]
    s = s.replace('*', '').replace('`', '').strip()

    try:
        return float(s), ai_flag
    except ValueError:
        return None, False

def _parse_lattice(val) -> Tuple[Optional[float], bool]:
    """
    Parse the ``Lattice constant (Å)`` column.

    Strips ``*`` (AI-estimated from figure)
    Stored as a plain float in this CSV; parenthetical uncertainties are
    stripped defensively.

    Returns
    -------
    (numeric_value_or_None, ai_estimated_flag)
    None if outside the physically reasonable
    range defined in globals.
    """
    try:
        a, ai_flag = _clean_and_ai(val)
        if globals.LATTICE_MIN <= a <= globals.LATTICE_MAX:
            return a, ai_flag
        latt_err.append(val)
        return None, False
    except Exception:
        return None, False


# def _parse_tc(val) -> Tuple[Optional[float], bool]:
#     """
#     Parse a Thermal Conductivity value.
#
#     Strips ``*`` (AI-estimated from figure) and `` ` `` (unreliable) markers.
#     Scientific notation (e.g. ``9.47E-01``) is handled natively by ``float()``.
#
#     Returns
#     -------
#     (numeric_value_or_None, ai_estimated_flag)
#     """
#     s, ai_flag = _clean_and_ai(val)
#     try:
#         return s, ai_flag
#     except ValueError:
#         return None, False


def _parse_relative_density(val) -> Tuple[Optional[float], bool]:
    """
    Parse the ``Relative density`` column.

    Values are stored inconsistently:
      - As a fraction (0.934 → 93.4 %) for most rows
      - As a percentage (96.2 → 96.2 %) for some rows

    A value > 1.5 is assumed to already be in percent.
    """
    if pd.isna(val):
        return None, False
    try:
        s, ai_flag = _clean_and_ai(val)
        v = float(str(val).replace('%', '').strip())
        return (v if v > 1.5 else v * 100.0), ai_flag
    except ValueError:
        return None, False


# ---------------------------------------------------------------------------
# Formula cleaning
# ---------------------------------------------------------------------------

def _clean_formula(s) -> Optional[str]:
    """
    Clean a raw ``Composition`` string from the CSV.

    Steps
    -----
    1. Remove all Unicode control/format characters (zero-width spaces,
       zero-width non-joiners, BOM remnants, etc.).
    2. Strip non-breaking spaces (``\\xa0``) and surrounding whitespace.
    3. Remove everything after the first space — annotation tags such as
       ``(G5)`` or ``(S1)`` are always space-separated from the formula.
    4. Detect and collapse duplicated formula text:
       e.g. ``"(La1/3Nd1/3Gd1/3)2Zr2O7(La1/3Nd1/3Gd1/3)2Zr2O7"``
       → ``"(La1/3Nd1/3Gd1/3)2Zr2O7"``.

    Returns None for missing or empty strings.
    """
    if pd.isna(s):
        return None

    # 1 & 2: strip invisible unicode and non-breaking spaces
    s = ''.join(c for c in str(s) if not unicodedata.category(c).startswith('C'))
    s = s.replace('\xa0', '').strip()

    # 3: strip space-separated annotation tags
    s = s.split()[0] if s else s

    # 4: detect exact duplication (search half-lengths from 10 upwards)
    n = len(s)
    for half in range(10, n // 2 + 1):
        if s[:half] == s[half: half * 2]:
            s = s[:half]
            break

    return s or None


def _is_compound_mixture(formula: str) -> bool:
    """
    Return True for compound-mixture formulas such as
    ``(Ca2Nb2O7)0.05(Gd2Zr2O7)0.95``.

    Identified by an oxygen atom appearing *inside* a parenthetical group.
    """
    return bool(re.search(r'\([^)]*O[^)]*\)', formula))


# ---------------------------------------------------------------------------
# Formula parser helpers
# ---------------------------------------------------------------------------

def _parse_amount(s: str) -> float:
    """Convert ``'1/3'``, ``'0.25'``, ``'2'``, ``''`` → float."""
    s = s.strip()
    if not s:
        return 1.0
    if '/' in s:
        num, den = s.split('/', 1)
        return float(num) / float(den)
    return float(s)


def _parse_group(content: str, multiplier: float) -> Dict[str, float]:
    """
    Parse the interior of a parenthetical group and apply the outer multiplier.
    Oxygen atoms inside are ignored.

    Example
    -------
    ``_parse_group('Ho1/2Y1/2', 2.0)`` → ``{'Ho': 1.0, 'Y': 1.0}``
    """
    result: Dict[str, float] = {}
    for m in re.finditer(r'([A-Z][a-z]?)([\d/]+(?:\.\d+)?)?', content):
        elem = m.group(1)
        if elem == 'O':
            continue
        amt = _parse_amount(m.group(2) or '1')
        result[elem] = result.get(elem, 0.0) + amt * multiplier
    return result


def _parse_lit_formula(
    formula_str: str,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Parse a literature-extraction formula into (a_comp, b_comp, unknown).

    Supported notations
    -------------------
    * Simple:            ``Eu2Ti2O7``
    * A-site grouped:    ``(Ho1/2Y1/2)2Ti2O7``
    * B-site grouped:    ``La2(Ce0.2Zr0.8)2O7``
    * Both grouped:      ``(La1/3Nd1/3Gd1/3)2(Zr3/4Ce1/4)2O7``
    * Ungrouped mixed:   ``La1.9Ce0.1Ti2O7``

    Stoichiometries are *absolute* (as written in the formula).
    Normalisation to mole fractions is done by ``_comp_to_fractions()``.

    Ambiguous element assignment (e.g. Ce)
    ---------------------------------------
    Whichever site has the larger deficit from 2.0 receives the ambiguous
    element.  On a tie, A-site is preferred (3+ default).
    """
    if not formula_str:
        return {}, {}, {}

    raw_elems: Dict[str, float] = {}
    consumed: set = set()

    # --- parenthetical groups with outer multiplier ---
    for m in re.finditer(r'\(([^)]+)\)([\d/]+(?:\.\d+)?)?', formula_str):
        content    = m.group(1)
        multiplier = _parse_amount(m.group(2) or '1')
        for elem, amt in _parse_group(content, multiplier).items():
            raw_elems[elem] = raw_elems.get(elem, 0.0) + amt
        consumed.update(range(m.start(), m.end()))

    # --- remaining elements outside parentheses ---
    for m in re.finditer(r'([A-Z][a-z]?)([\d/]+(?:\.\d+)?)?', formula_str):
        if any(p in consumed for p in range(m.start(), m.end())):
            continue
        elem = m.group(1)
        if elem == 'O':
            continue
        amt = _parse_amount(m.group(2) or '1')
        raw_elems[elem] = raw_elems.get(elem, 0.0) + amt

    # --- assign to sites ---
    a_comp:    Dict[str, float] = {}
    b_comp:    Dict[str, float] = {}
    unknown:   Dict[str, float] = {}
    ambiguous: Dict[str, float] = {}

    for elem, amt in raw_elems.items():
        if elem in globals.KNOWN_AMBIGUOUS:
            ambiguous[elem] = amt
        elif elem in globals.KNOWN_A and elem not in globals.KNOWN_B:
            a_comp[elem] = amt
        elif elem in globals.KNOWN_B and elem not in globals.KNOWN_A:
            b_comp[elem] = amt
        else:
            unknown[elem] = amt

    # --- resolve ambiguous by stoichiometry deficit ---
    for elem, amt in ambiguous.items():
        a_deficit = 2.0 - sum(a_comp.values())
        b_deficit = 2.0 - sum(b_comp.values())
        if a_deficit >= b_deficit:
            a_comp[elem] = amt   # tie-breaker: A-site (3+ default)
        else:
            b_comp[elem] = amt

    return a_comp, b_comp, unknown


# ---------------------------------------------------------------------------
# Composition utilities
# ---------------------------------------------------------------------------

def _comp_to_str(comp: Dict[str, float]) -> str:
    """Comma-joined sorted element symbols for a site (e.g. ``'Gd,La'``)."""
    return ','.join(sorted(comp.keys()))


def _comp_to_fractions(comp: Dict[str, float]) -> Dict[str, float]:
    """Normalise absolute stoichiometries to mole fractions summing to 1."""
    total = sum(comp.values())
    if total == 0:
        return {}
    return {e: v / total for e, v in comp.items()}


def _to_standard_formula(
    a_comp: Dict[str, float],
    b_comp: Dict[str, float],
) -> str:
    """
    Convert A-site and B-site composition dicts to the canonical
    ``(A1x1A2x2...)2(B1y1B2y2...)2O7`` string.

    Rules
    -----
    * Fractions *inside* parentheses are mole fractions (sum to 1).
    * The outer subscript is always ``2`` (one formula unit = A₂B₂O₇).
    * Single-element sites are written without parentheses: ``La2Ti2O7``.
    * Elements within each site are sorted alphabetically.
    * Fractions are formatted with up to 4 significant figures (``:.4g``),
      which cleanly represents 1/2 → 0.5, 1/3 → 0.3333, 3/4 → 0.75, etc.

    Examples
    --------
    ``{'La': 2}``          , ``{'Ti': 2}``           → ``La2Ti2O7``
    ``{'Ca': 0.05, 'Gd': 0.95}``, ``{'Nb': 0.05, 'Zr': 0.95}``
                                                       → ``(Ca0.05Gd0.95)2(Nb0.05Zr0.95)2O7``
    ``{'La':1/3,'Nd':1/3,'Gd':1/3}``, ``{'Zr':1}``  → ``(Gd0.3333La0.3333Nd0.3333)2Zr2O7``
    """
    a_frac = _comp_to_fractions(a_comp)
    b_frac = _comp_to_fractions(b_comp)

    def _fmt_group(frac_dict: Dict[str, float]) -> str:
        parts = []
        for elem in sorted(frac_dict):
            frac = frac_dict[elem]
            # If fraction is effectively 1 (single-element site after
            # filtering), omit the subscript entirely
            parts.append(elem if frac > 0.9995 else f"{elem}{frac:.4g}")
        return ''.join(parts)

    a_str  = _fmt_group(a_frac)
    b_str  = _fmt_group(b_frac)
    a_part = f"({a_str})2" if len(a_frac) > 1 else f"{a_str}2"
    b_part = f"({b_str})2" if len(b_frac) > 1 else f"{b_str}2"
    return f"{a_part}{b_part}O7"


def _expand_compound_mixture(
    formula: str,
) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
    """
    Expand a compound-mixture formula into ``(a_comp, b_comp)`` dicts whose
    values are **mole fractions** (each site sums to 1.0).

    Algorithm
    ---------
    For each ``(EndMember)fraction`` pair:
      1. Parse ``EndMember`` with ``_parse_lit_formula()`` to get its A/B site
         compositions.
      2. Normalise each site to mole fractions (handles non-stoichiometric
         end-members gracefully).
      3. Weight by ``fraction`` and accumulate into the combined dicts.
      4. Drop elements whose combined fraction is < 1e-9 (handles the edge
         case where fraction = 0, i.e. a pure end-member expressed as a
         mixture).

    Returns None if:
      * Fewer than 2 end-members are detected.
      * Fractions do not sum to ≈ 1 (tolerance 0.05).
      * Any end-member contains unknown cations.
      * Any end-member cannot be split into both A and B sites.

    Example
    -------
    ``"(Ca2Nb2O7)0.05(Gd2Zr2O7)0.95"``
      → ``({'Ca': 0.05, 'Gd': 0.95}, {'Nb': 0.05, 'Zr': 0.95})``

    ``"(Ca2Nb2O7)0(Gd2Zr2O7)1"``   (pure Gd2Zr2O7 written as mixture)
      → ``({'Gd': 1.0}, {'Zr': 1.0})``
    """
    pattern = r'\(([^)]+)\)([\d./]+)'
    matches = re.findall(pattern, formula)
    if len(matches) < 2:
        return None

    # Parse mixture fractions (support both decimal and ratio notation)
    fracs: list[float] = []
    for _, f in matches:
        if '/' in f:
            num, den = f.split('/', 1)
            fracs.append(float(num) / float(den))
        else:
            fracs.append(float(f))

    if abs(sum(fracs) - 1.0) > 0.05:
        log.debug(
            f"_expand_compound_mixture: fractions {fracs} sum to "
            f"{sum(fracs):.3f} ≠ 1 in '{formula}'"
        )
        return None

    a_combined: Dict[str, float] = {}
    b_combined: Dict[str, float] = {}

    for (sub_formula, _), frac in zip(matches, fracs):
        a_sub, b_sub, unknown = _parse_lit_formula(sub_formula)

        if unknown:
            log.debug(
                f"_expand_compound_mixture: unknown elements {unknown} "
                f"in end-member '{sub_formula}'"
            )
            return None
        if not a_sub or not b_sub:
            log.debug(
                f"_expand_compound_mixture: could not split '{sub_formula}' "
                f"into A and B sites"
            )
            return None

        # Normalise each end-member's site to fractions, then weight by
        # the mixture fraction.
        for elem, val in _comp_to_fractions(a_sub).items():
            a_combined[elem] = a_combined.get(elem, 0.0) + val * frac
        for elem, val in _comp_to_fractions(b_sub).items():
            b_combined[elem] = b_combined.get(elem, 0.0) + val * frac

    # Drop elements with negligible contribution (handles frac=0 end-members)
    a_combined = {e: v for e, v in a_combined.items() if v > 1e-9}
    b_combined = {e: v for e, v in b_combined.items() if v > 1e-9}

    if not a_combined or not b_combined:
        return None

    return a_combined, b_combined


# ---------------------------------------------------------------------------
# Pyrochlore classifier
# ---------------------------------------------------------------------------

def classify_compound(
    a_comp:    Dict[str, float],
    b_comp:    Dict[str, float],
    unknown:   Dict[str, float],
    lattice_a: Optional[float],
    phase:     str,
) -> str:
    """
    Return one of: ``'pristine'``, ``'high_entropy'``, ``'non_pyrochlore'``.

    A compound is ``non_pyrochlore`` if ANY of the following hold:
      • Phase not in {'P', 'SP', 'F+P'}
      • Unknown (unsupported) cations are present
      • Either site is empty after parsing

    Among valid pyrochlores:
      • pristine     → exactly 1 element on A-site AND 1 on B-site
      • high_entropy → ≥2 elements on A-site OR B-site (or both)
    """
    # Phase check
    if str(phase).strip() not in PYROCHLORE_PHASES:
        struct_err.append(str(phase).strip())
        return globals.NON_PYROCHLORE

    # Unknown cations
    if unknown:
        unknown_err.append([unknown, a_comp, b_comp])
        return globals.NON_PYROCHLORE

    # Both sites must be populated
    if not a_comp or not b_comp:
        stoic_err.append([a_comp, b_comp])
        return globals.NON_PYROCHLORE

    n_a = len(a_comp)
    n_b = len(b_comp)
    return globals.PRISTINE if (n_a == 1 and n_b == 1) else globals.HIGH_ENTROPY


# ---------------------------------------------------------------------------
# Canonical composition key (for deduplication)
# ---------------------------------------------------------------------------

def _composition_key(
    a_comp: Dict[str, float],
    b_comp: Dict[str, float],
) -> str:
    """
    Hashable string encoding the full fractional composition,
    rounded to 2 decimal places to merge near-identical entries.
    """
    a_frac = _comp_to_fractions(a_comp)
    b_frac = _comp_to_fractions(b_comp)
    a_part = '|'.join(f"{e}:{round(v,2):.2f}" for e, v in sorted(a_frac.items()))
    b_part = '|'.join(f"{e}:{round(v,2):.2f}" for e, v in sorted(b_frac.items()))
    return f"A[{a_part}]B[{b_part}]"


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_lit_ext(
    filepath:    str | Path | None = None,
    verbose:     bool = True,
    deduplicate: bool = False,
) -> pd.DataFrame:
    """
    Load and parse the literature-extraction pyrochlore CSV.

    Parameters
    ----------
    filepath    : path to ``Pyrochlore_literature_extraction.csv``
                  (defaults to ``data/raw/Pyrochlore_literature_extraction.csv``)
    verbose     : print a summary table
    deduplicate : average duplicate rows grouped by composition key

    Returns
    -------
    DataFrame in the canonical combined-dataset schema, containing only
    pyrochlore entries (pristine + high_entropy).

    Schema columns (all sources)
    ----------------------------
    Composition                  : canonical (A)2(B)2O7 standardized formula
    Sample A                     : comma-joined A-site element symbols
    Sample B                     : comma-joined B-site element symbols
    Thermal Conductivity W/m/K   : float or NaN
    Lattice Parameter (Å)        : float or NaN
    Relative Density %           : float or NaN
    Is Single Phase              : 'Yes'
    Synthesis Method             : string
    data_source                  : 'literature_extraction'
    b_o_distance / b_o_b_angle / oxygen_param_x : float or NaN

    Extra columns (this source only)
    ---------------------------------
    Standardized Composition     : same as Composition — explicit (A)2(B)2O7 string
    A-site                       : comma-joined sorted A-site element symbols
    B-site                       : comma-joined sorted B-site element symbols
    a_stoich_json                : JSON {element: mole_fraction} for A-site
    b_stoich_json                : JSON {element: mole_fraction} for B-site
    tc_ai_estimated              : True if TC was AI-estimated from a figure
    compound_type                : 'pristine' or 'high_entropy'
    source_doi                   : DOI of source publication
    n_lit_duplicates             : rows averaged into this entry after dedup
    _comp_key                    : composition fingerprint (dropped after dedup)
    """
    # --- locate file ---
    if filepath is None:
        _HERE    = Path(__file__).resolve().parent
        _PROJECT = _HERE.parent.parent
        filepath = _PROJECT / 'data' / 'raw' / 'Pyrochlore_literature_extraction.csv'
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}.\n"
            f"Place the file at data/raw/Pyrochlore_literature_extraction.csv "
            f"or pass the path explicitly."
        )

    df_raw = pd.read_csv(filepath)
    # df_raw['lattice_a'] = df_raw['Lattice constant (Å)'].apply(_parse_lattice)

    if verbose:
        log.info(f"Lit-extract: {len(df_raw)} raw rows loaded from {filepath.name}")

    records:          list = []
    n_non_pyro:       int  = 0
    n_mixture_failed: int  = 0

    for _, row in df_raw.iterrows():
        raw_formula = str(row.get('Composition', ''))

        # ── Step 1: compound-mixture expansion ───────────────────────────
        if _is_compound_mixture(raw_formula):
            expanded = _expand_compound_mixture(raw_formula)
            if expanded is None:
                n_mixture_failed += 1
                n_non_pyro += 1
                log.debug(f"Could not expand mixture: {raw_formula!r}")
                continue
            # a_comp / b_comp are already mole-fraction dicts (sum = 1)
            a_comp, b_comp = expanded
            unknown = {}

        # ── Step 2: normal formula — clean then parse ─────────────────────
        else:
            cleaned = _clean_formula(raw_formula)
            if cleaned is None:
                n_non_pyro += 1
                continue
            a_comp, b_comp, unknown = _parse_lit_formula(cleaned)

        # ── Step 3: pyrochlore classification ────────────────────────────
        latt, latt_ai = _parse_lattice(row.get('Lattice constant (Å)', np.nan))
        phase = str(row.get('Phase', '')).strip()
        ctype = classify_compound(
            a_comp, b_comp, unknown,
            latt,
            phase=phase,
        )
        if ctype == globals.NON_PYROCHLORE:
            n_non_pyro += 1
            continue

        # ── Step 4: derive canonical columns ─────────────────────────────
        a_frac      = _comp_to_fractions(a_comp)
        b_frac      = _comp_to_fractions(b_comp)
        std_formula = _to_standard_formula(a_comp, b_comp)

        tc_val, tc_ai   = _clean_and_ai(row.get('Thermal Conductivity (W/mK)', np.nan))
        rel_density,rai = _parse_relative_density(row.get('Relative density', np.nan))
        vol, vol_ai     = _clean_and_ai(row.get('Volume (nm^3)', np.nan))
        rA, rA_ai       = _clean_and_ai(row.get('rA (Å)', np.nan))
        rB, rB_ai       = _clean_and_ai(row.get('rB (Å)', np.nan))
        ab, ab_ai       = _clean_and_ai(row.get('rA/rB', np.nan))
        xo, xo_ai       = _clean_and_ai(row.get('xO48f', np.nan))
        sd, sd_ai       = _clean_and_ai(row.get('Size disorder (δ %)', np.nan))
        p, p_ai         = _clean_and_ai(row.get('Porosity (%)', np.nan))
        gs, gs_ai       = _clean_and_ai(row.get('Grain Size (μm)', np.nan))
        dens_m, dm_ai   = _clean_and_ai(row.get('ρmea (g/cm3)', np.nan))
        dens_c, dc_ai   = _clean_and_ai(row.get('ρcal (g/cm3)', np.nan))
        dbg, dbg_ai     = _clean_and_ai(row.get('Direct Band Gap', np.nan))
        ibg, ibg_ai     = _clean_and_ai(row.get('Indirect Band Gap', np.nan))
        bm, bm_ai       = _clean_and_ai(row.get('Bulk Modulus (GPa)', np.nan))
        sm, sm_ai       = _clean_and_ai(row.get('Shear Modulus (GPa)', np.nan))
        hv, hv_ai       = _clean_and_ai(row.get('Vickers Hardness (GPa)', np.nan))
        ym, ym_ai       = _clean_and_ai(row.get('Young\'s Modulus (GPa)', np.nan))
        ft, ft_ai       = _clean_and_ai(row.get('Fracture Toughness (Mpa*m^.5)', np.nan))
        cte, cte_ai     = _clean_and_ai(row.get('Thermal Expansion (K^-1)', np.nan))
        h_cte, h_cte_ai = _clean_and_ai(row.get('Thermal Expansion at high temp', np.nan))
        sh, sh_ai = _clean_and_ai(row.get('Specific Heat (Jg^−1K^−1', np.nan))
        td, td_ai       = _clean_and_ai(row.get('Thermal Diffusivity  (mm² s⁻¹)', np.nan))
        ae, ae_ai       = _clean_and_ai(row.get('Activation Energy (eV)', np.nan))
        pr, pr_ai       = _clean_and_ai(row.get('Poissons Ratio', np.nan))
        mm, mm_ai       = _clean_and_ai(row.get('Mole Mass (g/mol)', np.nan))

        records.append({
            # ── canonical schema ──────────────────────────────────────────
            'Composition':                          std_formula,
            'Sample A':                             _comp_to_str(a_comp),
            'Sample B':                             _comp_to_str(b_comp),
            'Oxidation State A':                   3.0,
            'Oxidation State B':                    4.0,
            'Thermal Conductivity (W/m/K)':         tc_val,
            'Lattice Parameter (Å)':                latt,
            'Vickers Hardness (GPa)':               hv,
            'CTE (K^-1)':                           cte,
            'Relative Density %':                   rel_density,
            'Density Measured':                     dens_m,
            'Density Calculated':                   dens_c,
            'Ionic Radius A (Å)':                   rA,
            'Ionic Radius B (Å)':                   rB,
            'rA/rB (Å)':                            ab,
            'oxygen_param_x':                       xo,
            'Size disorder (δ %)':                  sd,
            'Porosity (%)':                         p,
            'Grain Size (μm)':                      gs,
            'Molar Mass (g/mol)':                   mm,
            'Bulk Modulus (GPa)':                   bm,
            'Shear Modulus (GPa)':                  sm,
            'Youngs Modulus (GPa)':                 ym,
            'Poisson Ratio':                        pr,
            'Fracture Toughness (Mpa*m^.5)':        ft,
            'Specific Heat (Jg^−1K^−1)':            sh,
            'Thermal Diffusivity  (mm² s⁻¹)':       td,
            'High Temp CTE':                        h_cte,
            'Temperature (K)':                      np.nan,
            'Activation Energy (eV)':               ae,
            'Is Single Phase':                      np.nan,#'Yes',
            'Synthesis Method':                     str(row.get('Method', '')).strip(),
            'data_source':                          'literature_extraction',
            'b_o_distance':                         np.nan,
            'b_o_b_angle':                          np.nan,
            # ── site assignment columns ───────────────────────────────────
            'Standardized Composition':      std_formula,
            'A-site':                        _comp_to_str(a_comp),
            'B-site':                        _comp_to_str(b_comp),
            'a_stoich_json':                 json.dumps(a_frac),
            'b_stoich_json':                 json.dumps(b_frac),
            # ── provenance / metadata ─────────────────────────────────────
            'tc_ai_estimated':               tc_ai,
            'n_lit_duplicates':              1,
            'compound_type':                 ctype,
            'source_doi':                    str(row.get('Source DOI', '')).strip(),
            '_comp_key':                     _composition_key(a_comp, b_comp),
        })

    if verbose:
        log.info(
            f"Lit-extract: {n_non_pyro} entries excluded as non-pyrochlore "
            f"({len(records)} pyrochlore entries remain)"
        )
        if n_mixture_failed:
            log.warning(
                f"  {n_mixture_failed} compound-mixture rows could not be expanded "
                f"and were excluded"
            )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # ── optional deduplication ────────────────────────────────────────────
    if deduplicate:
        agg_rows = []
        for key, grp in df.groupby('_comp_key'):
            base = grp.iloc[0].copy()
            base['Lattice Parameter (Å)'] = grp['Lattice Parameter (Å)'].mean()
            base['Thermal Conductivity (W/m/K)']   = grp['Thermal Conductivity (W/m/K)'].mean()
            base['n_lit_duplicates']              = len(grp)
            base['source_doi']                   = '|'.join(
                grp['source_doi'].dropna().astype(str).tolist()
            )
            agg_rows.append(base)

        df = pd.DataFrame(agg_rows).reset_index(drop=True)

    df = df.drop(columns=['_comp_key'], errors='ignore')

    if verbose:
        pristine_n = (df['compound_type'] == globals.PRISTINE).sum()
        he_n       = (df['compound_type'] == globals.HIGH_ENTROPY).sum()
        n_expanded = df['Standardized Composition'].str.contains(r'\(', na=False).sum()

        print()
        print(f"  {'Compound type':<45} {'Count':>6}")
        print(f"  {'-'*53}")
        print(f"  {'Pristine':<45} {pristine_n:>6}")
        print(f"  {'High-entropy':<45} {he_n:>6}")
        print(f"  {'  of which multi-element (parenthetical)':<45} {n_expanded:>6}")
        print(f"  {'Non-pyrochlore (excluded)':<45} {n_non_pyro:>6}")
        if n_mixture_failed:
            print(f"  {'  of which unexpandable mixtures':<45} {n_mixture_failed:>6}")
        print(f"  {'-'*53}")
        print(f"  {'Total (raw)':<45} {len(df_raw):>6}")
        print()

    return df


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='  [%(levelname)s] %(message)s')
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else None
    result = load_lit_ext(filepath=fp, verbose=True, deduplicate=False)
    print(result[[
        'Standardized Composition', 'A-site', 'B-site',
        'a_stoich_json', 'b_stoich_json',
        'Lattice Parameter (Å)', 'compound_type',
    ]].head(20).to_string(index=False))
    # ]].to_string(index=False))
    print(f"\nTotal rows  : {len(result)}")
    print(f"Unknown err ({len(unknown_err)}): {unknown_err[:3]}")
    print(f"Phase err   ({len(struct_err)}): {struct_err[:5]}")
    print(f"Stoich err  ({len(stoic_err)}): {stoic_err[:3]}")
    print(f"Lattice err ({len(latt_err)}): {latt_err[:3]}")
