"""
make_combined_dataset.py
========================
Combines all four raw data sources into a single unified dataset:

  1. Sample_Properties_Safin_Feb_2026.csv  — HEC experimental data
  2. notebookLM_dataset.csv               — Binary pyrochlore lattice data
  3. parent_components.csv                — Literature family data
  4. HECPyrochlore_latt_data_ICSD.csv     — ICSD pyrochlore database

Every row is classified as one of:
  pristine       — exactly 1 element on the A-site AND 1 on the B-site
  high_entropy   — ≥2 elements on A-site OR B-site (or both)
  non_pyrochlore — fails pyrochlore sanity checks (excluded from training)

Only pristine and high_entropy rows are written to the processed dataset.

Canonical schema
----------------
  Composition | Sample A | Sample B | TPS Cond W/m/K
  | Lattice Parameter (Angstrom) | Relative Density %
  | Is Single Phase | Synthesis Method | data_source
  | b_o_distance | b_o_b_angle | oxygen_param_x
  | compound_type | a_stoich_json | b_stoich_json

Run directly:
  python src/data/make_combined_dataset.py
"""

from __future__ import annotations

import re
import json
import math
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from src import globals
from pymatgen.core import Composition

logging.basicConfig(level=logging.INFO, format='  [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent
RAW_DIR  = _PROJECT / 'data' / 'raw'
OUT_DIR  = _PROJECT / 'data' / 'processed'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUT_DIR / 'combined_pyrochlore.csv'
BASE_OUTPUT_FILE = OUT_DIR / 'pristine_pyrochlore.csv'
HEC_OUTPUT_FILE = OUT_DIR / 'hec_pyrochlore.csv'

# # ── element sets (shared with load_icsd.py) ───────────────────────────────────
# KNOWN_A: frozenset = frozenset({
#     'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
#     'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Y',
# })
# KNOWN_B: frozenset = frozenset({
#     'Ti', 'Zr', 'Hf', 'Sn', 'Ir', 'Nb',
# })
#
# # Pyrochlore stability window for r_A/r_B (Shannon ionic radii)
# # Outside this range → likely defect-fluorite or other polymorph
# _RA_RB_MIN = 1.40
# _RA_RB_MAX = 1.90

# ── canonical columns ─────────────────────────────────────────────────────────
CANONICAL_COLS = globals.CANONICAL_COLS
PRISTINE_COLS = globals.PRISTINE_COLS
HEC_COLS = globals.HEC_COLS

# # ── compound-type constants ───────────────────────────────────────────────────
# PRISTINE       = 'pristine'
# HIGH_ENTROPY   = 'high_entropy'
# NON_PYROCHLORE = 'non_pyrochlore'


# ── shared helpers ────────────────────────────────────────────────────────────

def _clean_element_list(raw: str) -> str:
    """Normalise element strings: strip whitespace, remove unicode superscripts."""
    if pd.isna(raw):
        return np.nan
    cleaned = re.sub(r'[⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉⁺⁻]+', '', str(raw))
    parts = [p.strip() for p in cleaned.split(',') if p.strip()]
    return ','.join(parts)


def _parse_sample_str(sample_str: str) -> Dict[str, float]:
    """
    Convert a comma-separated element string to an equiatomic fraction dict.
    e.g. "La,Gd,Lu" → {'La': 0.333, 'Gd': 0.333, 'Lu': 0.333}
    """
    if pd.isna(sample_str) or str(sample_str).strip() == '':
        return {}
    elems = [e.strip() for e in str(sample_str).split(',') if e.strip()]
    if not elems:
        return {}
    frac = 1.0 / len(elems)
    return {e: frac for e in elems}


def _comp_to_fractions_json(sample_str: str) -> str:
    """Return JSON string of equiatomic mole fractions for a Sample A/B string."""
    d = _parse_sample_str(sample_str)
    if d is None:
        return json.dumps({})
        # convert keys to str and values to native Python floats; skip NaN/inf
    safe = {}
    for k, v in (d.items() if isinstance(d, dict) else []):
        try:
            val = float(v)
            if math.isfinite(val):
                safe[str(k)] = val
        except Exception:
            # skip non-numeric values
            safe[str(k)] = str(v)
    return json.dumps(safe)


def classify_sample(sample_a: str, sample_b: str) -> str:
    """
    Classify a compound given its Sample A and Sample B strings.

    Rules
    -----
    non_pyrochlore  : any element in A-site is not in KNOWN_A, or any element
                      in B-site is not in KNOWN_B, or either site is empty
    pristine        : exactly 1 element on A-site AND 1 on B-site
    high_entropy    : ≥2 elements on either or both sites

    Parameters
    ----------
    sample_a : comma-separated A-site element symbols
    sample_b : comma-separated B-site element symbols

    Returns
    -------
    One of 'pristine', 'high_entropy', 'non_pyrochlore'
    """
    a_elems = [e.strip() for e in str(sample_a).split(',') if e.strip()] \
              if not pd.isna(sample_a) else []
    b_elems = [e.strip() for e in str(sample_b).split(',') if e.strip()] \
              if not pd.isna(sample_b) else []

    # Must have at least one element on each site
    if not a_elems or not b_elems:
        return globals.NON_PYROCHLORE

    # All A-site elements must be recognised rare-earth / Y cations
    if any(e not in globals.KNOWN_A for e in a_elems):
        return globals.NON_PYROCHLORE

    # All B-site elements must be recognised transition-metal cations
    if any(e not in globals.KNOWN_B for e in b_elems):
        return globals.NON_PYROCHLORE

    n_a = len(a_elems)
    n_b = len(b_elems)

    if n_a == 1 and n_b == 1:
        return globals.PRISTINE
    return globals.HIGH_ENTROPY


def _parse_tc_range(tc_str: str) -> float:
    """Parse strings like '<1.0', '0.8-1.2', '1.1' to a midpoint float."""
    if not tc_str or str(tc_str).strip() in ('nan', '-', ''):
        return np.nan
    tc_str = str(tc_str).replace('<', '').replace('>', '').strip()
    if '-' in tc_str:
        parts = tc_str.split('-')
        try:
            return (float(parts[0]) + float(parts[1])) / 2.0
        except ValueError:
            return np.nan
    try:
        return float(tc_str)
    except ValueError:
        return np.nan


# ── Source 1: Safin experimental data ────────────────────────────────────────

def load_safin(columns: List) -> pd.DataFrame:
    path = RAW_DIR / 'Sample_Properties_Safin_Feb_2026.csv'
    df = pd.read_csv(path, na_values=['NA', '', 'N/A'])
    log.info(f"Safin: {len(df)} rows from {path.name}")

    out = pd.DataFrame({c: np.nan for c in columns}, index=df.index)
    out['Composition']                  = df['ID'].fillna('').astype(str)
    out['Sample A']                     = df['Sample A'].apply(_clean_element_list)
    out['Sample B']                     = df['Sample B'].apply(_clean_element_list)
    out['Thermal Conductivity (W/m/K)'] = pd.to_numeric(df['TPS Cond W/m/K'], errors='coerce')
    out['Lattice Parameter (Angstrom)'] = pd.to_numeric(
                                            df['Lattice Parameter (Angstrom)'], errors='coerce')
    out['Relative Density %']           = pd.to_numeric(df['Relative Density %'], errors='coerce')
    out['Is Single Phase']              = df['Is Single Phase'].str.strip().str.lower().map(
                                            {'yes': 'Yes', 'no': 'No'})
    out['Synthesis Method']             = df['Synthesis Method'].fillna('')
    out['data_source']                  = 'safin_experimental'

    # Classify each row
    out['compound_type'] = out.apply(
        lambda r: classify_sample(r['Sample A'], r['Sample B']), axis=1)

    # Stoich JSON (equiatomic for Safin data)
    out['a_stoich_json'] = out['Sample A'].apply(_comp_to_fractions_json)
    out['b_stoich_json'] = out['Sample B'].apply(_comp_to_fractions_json)

    # out['Temperature'] = np.nan
    # out['density_calc'] = np.nan

    # print(out)
    # exit(0)

    n_excl = (out['compound_type'] == globals.NON_PYROCHLORE).sum()
    if n_excl:
        log.info(f"Safin: {n_excl} rows classified as non-pyrochlore (kept but flagged)")
    log.info(
        f"Safin: {out['Lattice Parameter (Angstrom)'].notna().sum()} lattice, "
        f"{out['Thermal Conductivity (W/m/K)'].notna().sum()} thermal values"
    )
    return out


# ── Source 2: notebookLM binary pyrochlore dataset ───────────────────────────

_NLM_FORMULA_MAP = {
    'lu2ti2o7':  ('Lu', 'Ti'),
    'nd2ir2o7':  ('Nd', 'Ir'),
    'pr2ir2o7':  ('Pr', 'Ir'),
    'gd2zr2o7':  ('Gd', 'Zr'),
}

def _parse_nlm_compound(compound: str) -> Tuple[str, str]:
    key = compound.strip().lower().replace(' ', '')
    for pattern, ab in _NLM_FORMULA_MAP.items():
        if key.startswith(pattern.split('(')[0]):
            return ab
    m = re.match(r'([a-z]+)2([a-z]+)2o7', key)
    if m:
        return m.group(1).capitalize(), m.group(2).capitalize()
    return np.nan, np.nan


def load_nlm() -> pd.DataFrame:
    path = RAW_DIR / 'notebookLM_dataset.csv'
    df = pd.read_csv(path, na_values=['-', '', 'NA'])
    log.info(f"NLM: {len(df)} rows from {path.name}")

    rows = []
    for _, row in df.iterrows():
        compound = str(row['Compound']).strip()
        a_str, b_str = _parse_nlm_compound(compound)

        # # Skip Bi/Pb — no ionic radius data
        # if isinstance(a_str, str) and any(e in a_str for e in ['Bi', 'Pb']):
        #     continue
        # if isinstance(b_str, str) and any(e in b_str for e in ['Bi', 'Pb']):
        #     continue

        ctype = classify_sample(a_str, b_str) if isinstance(a_str, str) else globals.NON_PYROCHLORE

        rows.append({
            'Composition':                  compound,
            'Sample A':                     a_str,
            'Sample B':                     b_str,
            'Thermal Conductivity (W/m/K)': np.nan,
            'Lattice Parameter (Angstrom)': pd.to_numeric(
                                              row.get('Lattice Parameter a (A)', np.nan),
                                              errors='coerce'),
            'Relative Density %':           np.nan,
            'Is Single Phase':              np.nan,
            'Synthesis Method':             '',
            'data_source':                  'notebookLM_literature',
            'b_o_distance':                 pd.to_numeric(
                                              row.get('B-O Distance (A)', np.nan), errors='coerce'),
            'b_o_b_angle':                  pd.to_numeric(
                                              row.get('B-O-B Angle (deg)', np.nan), errors='coerce'),
            'oxygen_param_x':               pd.to_numeric(
                                              row.get('Oxygen Parameter x', np.nan), errors='coerce'),
            'compound_type':                ctype,
            'a_stoich_json':                _comp_to_fractions_json(a_str),
            'b_stoich_json':                _comp_to_fractions_json(b_str),
            'Temperature':                  np.nan,
            'density_calc':                 np.nan,
        })

    out = pd.DataFrame(rows)
    out = out[out['Lattice Parameter (Angstrom)'].notna() | out['Sample A'].notna()]
    log.info(f"NLM: {out['Lattice Parameter (Angstrom)'].notna().sum()} lattice values parsed")
    return out


# ── Source 3: Parent components literature ────────────────────────────────────

def load_parent_components() -> pd.DataFrame:
    path = RAW_DIR / 'parent_components.csv'
    df = pd.read_csv(path, na_values=['', 'NA'])
    log.info(f"Parent components: {len(df)} rows from {path.name}")

    rows = []
    for _, row in df.iterrows():
        a_raw = _clean_element_list(str(row.get('A-site Cations', '')))
        b_raw = _clean_element_list(str(row.get('B-site Cations', '')))
        tc_val = _parse_tc_range(str(row.get('Thermal Conductivity (W/m·K)', '')))
        ctype = classify_sample(a_raw, b_raw)
        rows.append({
            'Composition':                  str(row.get('Parent Compound', '')).strip(),
            'Sample A':                     a_raw,
            'Sample B':                     b_raw,
            'Thermal Conductivity (W/m/K)': tc_val,
            'Lattice Parameter (Angstrom)': np.nan,
            'Relative Density %':           np.nan,
            'Is Single Phase':              'Yes',
            'Synthesis Method':             '',
            'data_source':                  'parent_components_literature',
            'b_o_distance':                 np.nan,
            'b_o_b_angle':                  np.nan,
            'oxygen_param_x':               np.nan,
            'compound_type':                ctype,
            'a_stoich_json':                _comp_to_fractions_json(a_raw),
            'b_stoich_json':                _comp_to_fractions_json(b_raw),
            'Temperature':                  np.nan,
            'density_calc':                 np.nan,
        })

    out = pd.DataFrame(rows)
    log.info(f"Parent components: {out['Thermal Conductivity (W/m/K)'].notna().sum()} thermal values")
    return out


# ── Source 4: ICSD database  ───────────────────────────────────────────────

def load_icsd_source() -> pd.DataFrame:
    """Thin wrapper that calls the dedicated ICSD loader."""
    try:
        from src.data.load_icsd import load_icsd
    except ImportError:
        import sys
        sys.path.insert(0, str(_PROJECT))
        from src.data.load_icsd import load_icsd

    icsd_path = RAW_DIR / 'HECPyrochlore_latt_data_ICSD.csv'
    df = load_icsd(filepath=icsd_path, verbose=True, deduplicate=False)
    log.info(
        f"ICSD: {len(df)} usable rows "
        f"({(df['compound_type']=='pristine').sum()} pristine, "
        f"{(df['compound_type']=='high_entropy').sum()} high-entropy)"
    )
    return df

# ── Source 5: Jordan Thermal Dataset ──────────────────────────────────────────

def load_jordan_source() -> pd.DataFrame:
    path = RAW_DIR / 'Jordan_pyrochlore_data.csv'
    df = pd.read_csv(path, na_values=['', 'NA'])
    log.info(f"Jordan components: {len(df)} rows from {path.name}")

    rows = []
    for _, row in df.iterrows():
        try:
            comp = Composition(str(row.get('Composition, ''')))
        except:
            comp = np.nan
        # a_raw = _clean_element_list(str(row.get('A-site Cations', '')))
        # b_raw = _clean_element_list(str(row.get('B-site Cations', '')))
        # tc_val = _parse_tc_range(str(row.get('Thermal Conductivity (W/m·K)', '')))
        # ctype = classify_sample(a_raw, b_raw)
        rows.append({
            'Composition':                  comp,
            'Sample A':                     np.nan, # a_raw,
            'Sample B':                     np.nan, # b_raw,
            'Thermal Conductivity (W/m/K)': np.nan, # tc_val,
            'Lattice Parameter (Angstrom)': np.nan,
            'Relative Density %':           np.nan,
            'Is Single Phase':              'Yes',
            'Synthesis Method':             '',
            'data_source':                  'Jordan_pyrochlore_data',
            'b_o_distance':                 np.nan,
            'b_o_b_angle':                  np.nan,
            'oxygen_param_x':               np.nan,
            'compound_type':                np.nan, # ctype,
            'a_stoich_json':                np.nan, # _comp_to_fractions_json(a_raw),
            'b_stoich_json':                np.nan, # _comp_to_fractions_json(b_raw),
            'Temperature':                  row.get('Start Temp'),
            'Thermal Expansion':            row.get('CTE Value', np.nan),
            'density_calc':                 np.nan,
        })

    out = pd.DataFrame(rows)
    log.info(f"Parent components: {out['TPS Cond W/m/K'].notna().sum()} thermal values")
    return out

# ── Source 6: Aflow Dataset ──────────────────────────────────────────

def load_aflow_source() -> pd.DataFrame:
    """Thin wrapper that calls the dedicated aflow loader."""
    try:
        from src.data.load_aflow import load_aflow
    except ImportError:
        import sys
        sys.path.insert(0, str(_PROJECT))
        from src.data.load_aflow import load_aflow

    aflow_path = RAW_DIR / 'aflow_pyrochlore_data_comb.csv'
    df = load_aflow(filepath=aflow_path, verbose=True)
    log.info(
        f"AFlow: {len(df)} usable rows "
        f"({(df['compound_type']=='pristine').sum()} pristine, "
        f"{(df['compound_type']=='high_entropy').sum()} high-entropy)"
    )
    return df

# ── Source 7: Materials Project Dataset ──────────────────────────────────────────

def load_mp_source() -> pd.DataFrame:
    """Thin wrapper that calls the dedicated materials project loader."""
    try:
        from src.data.load_mp import load_mp
    except ImportError:
        import sys
        sys.path.insert(0, str(_PROJECT))
        from src.data.load_mp import load_mp

    mp_path = RAW_DIR / 'mp_pyrochlore_query.csv'
    df = load_mp(filepath=mp_path, verbose=True)
    log.info(
        f"MP: {len(df)} usable rows "
        f"({(df['compound_type']=='pristine').sum()} pristine, "
        f"{(df['compound_type']=='high_entropy').sum()} high-entropy)"
    )
    return df

# ── Build combined dataset ────────────────────────────────────────────────────

def build_combined_dataset(save: bool = True) -> pd.DataFrame:
    """
    Load, classify, and combine all four sources.

    Pyrochlore classification summary is printed for each source.
    Only 'pristine' and 'high_entropy' rows are retained in the output.
    """
    print()
    print("=" * 66)
    print("  Building Combined Pyrochlore Dataset")
    print("=" * 66)

    frames = []

    for loader, arg, label in [
        (load_safin, CANONICAL_COLS,    'Safin experimental'),
        (load_nlm, None,                'notebookLM literature'),
        (load_parent_components, None,  'Parent components'),
        (load_icsd_source, None,        'ICSD database'),
    ]:
        try:
            if arg == None:
                frm = loader()
            else:
                frm = loader(arg)
            frames.append(frm)
        except FileNotFoundError as e:
            log.warning(f"Skipping {label}: {e}")

    if not frames:
        raise RuntimeError("No data sources found — check data/raw/ directory.")

    # Align to canonical columns (ICSD has extra columns; keep them)
    all_cols = CANONICAL_COLS + [c for frm in frames
                                  for c in frm.columns
                                  if c not in CANONICAL_COLS]
    combined = pd.concat(frames, ignore_index=True)

    # Ensure all canonical cols exist
    for col in CANONICAL_COLS:
        if col not in combined.columns:
            combined[col] = np.nan

    # Drop rows where BOTH Sample A and Sample B are missing
    combined = combined[combined['Sample A'].notna() | combined['Sample B'].notna()]

    # # Remove Bi/Pb entirely (incompatible ionic radius tables)
    # has_exotic = combined['Sample A'].fillna('').str.contains(r'\b(?:Bi|Pb)\b', regex=True)
    # combined = combined[~has_exotic]

    # ── Exclude non-pyrochlores from training data ────────────────────────────
    n_total   = len(combined)
    non_pyro  = combined[combined['compound_type'] == globals.NON_PYROCHLORE]
    combined  = combined[combined['compound_type'] != globals.NON_PYROCHLORE].reset_index(drop=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print(f"  {'Source':<38} {'Rows':>5}  {'Lattice':>8}  "
          f"{'Thermal':>8}  {'Pristine':>9}  {'HE':>5}")
    print("  " + "-" * 78)
    for src, grp in combined.groupby('data_source'):
        lat  = grp['Lattice Parameter (Angstrom)'].notna().sum()
        tc   = grp['Thermal Conductivity (W/m/K)'].notna().sum()
        pri  = (grp['compound_type'] == globals.PRISTINE).sum()
        he   = (grp['compound_type'] == globals.HIGH_ENTROPY).sum()
        print(f"  {src:<38} {len(grp):>5}  {lat:>8}  {tc:>8}  {pri:>9}  {he:>5}")
    print("  " + "-" * 78)
    lat_total = combined['Lattice Parameter (Angstrom)'].notna().sum()
    tc_total  = combined['Thermal Conductivity (W/m/K)'].notna().sum()
    pri_total = (combined['compound_type'] == globals.PRISTINE).sum()
    he_total  = (combined['compound_type'] == globals.HIGH_ENTROPY).sum()
    excl      = len(non_pyro)
    print(f"  {'TOTAL (pyrochlore)':<38} {len(combined):>5}  "
          f"{lat_total:>8}  {tc_total:>8}  {pri_total:>9}  {he_total:>5}")
    print(f"  {'Non-pyrochlore (excluded)':<38} {excl:>5}")
    print()

    # Save only canonical columns (drop ICSD-specific extras from CSV)
    out_cols = [c for c in CANONICAL_COLS if c in combined.columns]
    # Keep extra info columns if present
    extra = [c for c in combined.columns if c not in CANONICAL_COLS
             and c in ('compound_type', 'icsd_collection',
                        'n_icsd_duplicates', 'a_stoich_json', 'b_stoich_json')]
    out_cols = out_cols + extra

    if save:
        combined[out_cols].to_csv(OUTPUT_FILE, index=False)
        log.info(f"Saved combined dataset → {OUTPUT_FILE}")

    return combined

# ── Build single phase dataset ────────────────────────────────────────────────────

def build_single_phase_dataset(save: bool = True) -> pd.DataFrame:
    """
    Load, classify, and combine all pristine/single phase pyrochlores from all sources.

    Pyrochlore classification summary is printed for each source.
    Only 'pristine' rows are retained in the output.
    """
    print()
    print("=" * 66)
    print("  Building Single Phase Pyrochlore Dataset")
    print("=" * 66)

    frames = []

    for loader, arg, label in [
        # (load_safin, PRISTINE_COLS, 'Safin experimental'),
        # (load_nlm, None, 'notebookLM literature'),
        # (load_parent_components, None, 'Parent components'),
        (load_icsd_source, None, 'ICSD database'),
        (load_aflow_source, None, 'AFlow database'),
        (load_mp_source, None, 'Materials Project Database'),
        # (load_jordan_source, None, 'Jordan\'s data'),
    ]:
        try:
            if arg == None:
                frm = loader()
            else:
                frm = loader(arg)
            frames.append(frm)
        except FileNotFoundError as e:
            log.warning(f"Skipping {label}: {e}")

    if not frames:
        raise RuntimeError("No data sources found — check data/raw/ directory.")

    # Align to canonical columns
    # all_cols = PRISTINE_COLS + [c for frm in frames
    #                               for c in frm.columns
    #                               if c not in CANONICAL_COLS]
    combined = pd.concat(frames, ignore_index=True)

    # Ensure all single phase cols exist
    for col in PRISTINE_COLS:
        if col not in combined.columns:
            combined[col] = np.nan

    # Drop rows where BOTH Sample A and Sample B are missing
    combined = combined[combined['Sample A'].notna() | combined['Sample B'].notna()]

    # ── Exclude non-pyrochlores from training data ────────────────────────────
    n_total = len(combined)
    non_pyro = combined[combined['compound_type'] == globals.NON_PYROCHLORE]
    combined = combined[combined['compound_type'] != globals.NON_PYROCHLORE].reset_index(drop=True)

    # ── Summary table before filtering to pristine ────────────────────────────
    print()
    print(f"  {'Source':<38} {'Rows':>5}  {'Lattice':>8}  "
          f"{'Thermal':>8}  {'Pristine':>9}  {'HE':>5}")
    print("  " + "-" * 78)
    for src, grp in combined.groupby('data_source'):
        lat = grp['Lattice Parameter (Angstrom)'].notna().sum()
        tc = grp['Thermal Conductivity (W/m/K)'].notna().sum()
        pri = (grp['compound_type'] == globals.PRISTINE).sum()
        he = (grp['compound_type'] == globals.HIGH_ENTROPY).sum()
        print(f"  {src:<38} {len(grp):>5}  {lat:>8}  {tc:>8}  {pri:>9}  {he:>5}")
    print("  " + "-" * 78)
    lat_total = combined['Lattice Parameter (Angstrom)'].notna().sum()
    tc_total = combined['Thermal Conductivity (W/m/K)'].notna().sum()
    pri_total = (combined['compound_type'] == globals.PRISTINE).sum()
    he_total = (combined['compound_type'] == globals.HIGH_ENTROPY).sum()
    excl = len(non_pyro)
    print(f"  {'TOTAL (pyrochlore)':<38} {len(combined):>5}  "
          f"{lat_total:>8}  {tc_total:>8}  {pri_total:>9}  {he_total:>5}")
    print(f"  {'Non-pyrochlore (excluded)':<38} {excl:>5}")
    print()

    # ── Filter to PRISTINE only and select PRISTINE_COLS ────────────────────
    combined = combined[combined['compound_type'] == globals.PRISTINE].reset_index(drop=True)

    # Select only PRISTINE_COLS that exist in the dataframe
    out_cols = [c for c in PRISTINE_COLS if c in combined.columns]
    combined = combined[out_cols]

    # Checking the (LaY)(SnTi) type compositions
    # LaSn_df = combined[combined['Composition'] == 'La2Sn2O7']
    # LaTi_df = combined[combined['Composition'] == 'La2Ti2O7']
    # YSn_df = combined[combined['Composition'] == 'Y2Sn2O7']
    # YTi_df = combined[combined['Composition'] == 'Y2Ti2O7']
    # check_df = pd.concat([LaTi_df, LaSn_df, YTi_df, YSn_df])
    # print(check_df)
    # check_df.to_csv(RAW_DIR / 'check_df.csv')


    # ------- Deduplication ------------------------------------
    group_keys = ['Sample A', 'Sample B']

    # helper aggregators
    def concat_strings_col(series):
        vals = series.dropna().astype(str).unique()
        return ', '.join(vals) if len(vals) else np.nan

    def first_nonnull(series):
        nonnull = series.dropna()
        return nonnull.iloc[0] if len(nonnull) else np.nan

    # separate numeric and non-numeric (keep group keys out of non-numeric processing)
    numeric_cols = combined.select_dtypes(include='number').columns.tolist()
    other_cols = [c for c in combined.columns if c not in numeric_cols + group_keys]

    # we'll build aggregated rows per group manually to apply the 'DFT' logic
    out_rows = []
    for keys, grp in combined.groupby(group_keys, as_index=False):
        # keys is a tuple of group key values in same order as group_keys
        # decide which rows to use for averaging
        # has_dft = grp['Synthesis Method'].eq('DFT').any()
        # if has_dft:
        #     use_grp = grp[grp['Synthesis Method'].eq('DFT')]
        # else:
        #     use_grp = grp
        use_grp = grp
        agg_row = dict(zip(group_keys, keys))

        # numeric means (pandas will yield NaN if all NaN)
        for col in numeric_cols:
            agg_row[col] = use_grp[col].mean()

        # non-numeric processing
        for col in other_cols:
            if col == 'Band Gap Type' or col == 'data_source':
                agg_row[col] = concat_strings_col(use_grp[col])
            # elif col == 'Synthesis Method':
            #     # prefer 'DFT' if present, else first non-null, else NaN
            #     if has_dft:
            #         agg_row[col] = 'DFT'
            #     else:
            #         agg_row[col] = first_nonnull(use_grp[col])
            else:
                agg_row[col] = first_nonnull(use_grp[col])

        out_rows.append(agg_row)

    result = pd.DataFrame(out_rows)

    # Optional: ensure column order matches original
    combined = result[combined.columns.tolist()]

    # add ionic radius and electronegativity values
    from src.data.build_pristine import get_electronegativity, \
        get_ionic_radius_B, get_ionic_radius_A
    # ionic_a = get_ionic_radius_A(combined['Sample A'])
    # ionic_b = get_ionic_radius_B(combined['Sample B'])
    # en_a = get_electronegativity(combined['Sample A'])
    # en_b = get_electronegativity(combined['Sample B'])

    combined['Ionic Radius A (Angstrom)'] = combined['Sample A'].map(get_ionic_radius_A)
    combined['Ionic Radius B (Angstrom)'] = combined['Sample B'].map(get_ionic_radius_B)
    combined['Electronegativity A'] = combined['Sample A'].map(get_electronegativity)
    combined['Electronegativity B'] = combined['Sample B'].map(get_electronegativity)
    # combined.append({
    #     'Ionic Radius A (Angstrom)': ionic_a,
    #     'Ionic Radius B (Angstrom)': ionic_b,
    #     'Electronegativity A': en_a,
    #     'Electronegativity B': en_b,
    # })

    print(f"  Filtered to PRISTINE compounds: {len(combined)} rows")
    print(f"  Output columns: {len(out_cols)}")
    print()

    if save:
        combined.to_csv(BASE_OUTPUT_FILE, index=False)
        log.info(f"Saved pristine dataset → {BASE_OUTPUT_FILE}")

    return combined

# ── Build high entropy dataset ────────────────────────────────────────────────────

def build_high_entropy_dataset(save: bool = True) -> pd.DataFrame:
    """
    Load, classify, and combine all HEC pyrochlores from all sources.

    Pyrochlore classification summary is printed for each source.
    Only 'high entropy' rows are retained in the output.
    """
    print()
    print("=" * 66)
    print("  Building High Entorpy Pyrochlore Dataset")
    print("=" * 66)

    frames = []

    for loader, arg, label in [
        (load_safin, CANONICAL_COLS, 'Safin experimental'),
        (load_nlm, None, 'notebookLM literature'),
        # (load_parent_components, None, 'Parent components'),
        (load_icsd_source, None, 'ICSD database'),
        # (load_aflow_source, None, 'AFlow database'),
        # (load_mp_source, None, 'Materials Project Database'),
        # (load_jordan_source, None, 'Jordan\'s data'),
    ]:
        try:
            if arg == None:
                frm = loader()
            else:
                frm = loader(arg)
            frames.append(frm)
        except FileNotFoundError as e:
            log.warning(f"Skipping {label}: {e}")

    if not frames:
        raise RuntimeError("No data sources found — check data/raw/ directory.")

    # Align to canonical columns
    # all_cols = CANONICAL_COLS + [c for frm in frames
    #                               for c in frm.columns
    #                               if c not in CANONICAL_COLS]
    combined = pd.concat(frames, ignore_index=True)

    # Ensure all hec cols exist
    for col in HEC_COLS:
        if col not in combined.columns:
            combined[col] = np.nan

    # Drop rows where BOTH Sample A and Sample B are missing
    combined = combined[combined['Sample A'].notna() | combined['Sample B'].notna()]

    # ── Exclude non-pyrochlores from training data ────────────────────────────
    n_total = len(combined)
    non_pyro = combined[combined['compound_type'] == globals.NON_PYROCHLORE]
    combined = combined[combined['compound_type'] != globals.NON_PYROCHLORE].reset_index(drop=True)

    # ── Summary table before filtering to pristine ────────────────────────────
    print()
    print(f"  {'Source':<38} {'Rows':>5}  {'Lattice':>8}  "
          f"{'Thermal':>8}  {'Pristine':>9}  {'HE':>5}")
    print("  " + "-" * 78)
    for src, grp in combined.groupby('data_source'):
        lat = grp['Lattice Parameter (Angstrom)'].notna().sum()
        tc = grp['Thermal Conductivity (W/m/K)'].notna().sum()
        pri = (grp['compound_type'] == globals.PRISTINE).sum()
        he = (grp['compound_type'] == globals.HIGH_ENTROPY).sum()
        print(f"  {src:<38} {len(grp):>5}  {lat:>8}  {tc:>8}  {pri:>9}  {he:>5}")
    print("  " + "-" * 78)
    lat_total = combined['Lattice Parameter (Angstrom)'].notna().sum()
    tc_total = combined['Thermal Conductivity (W/m/K)'].notna().sum()
    pri_total = (combined['compound_type'] == globals.PRISTINE).sum()
    he_total = (combined['compound_type'] == globals.HIGH_ENTROPY).sum()
    excl = len(non_pyro)
    print(f"  {'TOTAL (pyrochlore)':<38} {len(combined):>5}  "
          f"{lat_total:>8}  {tc_total:>8}  {pri_total:>9}  {he_total:>5}")
    print(f"  {'Non-pyrochlore (excluded)':<38} {excl:>5}")
    print()

    # ── Filter to HIGH_ENTROPY only and select HEC_COLS ────────────────────
    combined = combined[combined['compound_type'] == globals.HIGH_ENTROPY].reset_index(drop=True)

    # Select only HEC_COLS that exist in the dataframe
    out_cols = [c for c in HEC_COLS if c in combined.columns]
    combined = combined[out_cols]

    '''
    # ------- Deduplication ------------------------------------
    group_keys = ['a_stoich_json', 'b_stoich_json']

    # helper aggregators
    def concat_strings_col(series):
        vals = series.dropna().astype(str).unique()
        return ', '.join(vals) if len(vals) else np.nan

    def first_nonnull(series):
        nonnull = series.dropna()
        return nonnull.iloc[0] if len(nonnull) else np.nan

    # separate numeric and non-numeric (keep group keys out of non-numeric processing)
    numeric_cols = combined.select_dtypes(include='number').columns.tolist()
    other_cols = [c for c in combined.columns if c not in numeric_cols + group_keys]

    # build aggregated rows per group
    out_rows = []
    for keys, grp in combined.groupby(group_keys, as_index=False):
        agg_row = dict(zip(group_keys, keys))

        # numeric means
        for col in numeric_cols:
            agg_row[col] = grp[col].mean()

        # non-numeric processing
        for col in other_cols:
            if col == 'data_source':
                agg_row[col] = concat_strings_col(grp[col])
            else:
                agg_row[col] = first_nonnull(grp[col])

        out_rows.append(agg_row)

    result = pd.DataFrame(out_rows)

    # Optional: ensure column order matches original
    combined = result[combined.columns.tolist()]
'''
    print(f"  Filtered to High Entropy compounds: {len(combined)} rows")
    print(f"  Output columns: {len(out_cols)}")
    print()

    if save:
        combined.to_csv(HEC_OUTPUT_FILE, index=False)
        log.info(f"Saved high entropy dataset → {HEC_OUTPUT_FILE}")

    return combined


if __name__ == '__main__':
    # df = build_combined_dataset(save=True)
    # df = build_single_phase_dataset(save=True)
    df = build_high_entropy_dataset(save=True)
    print("\nSample rows:")
    print(df[['Composition', 'Sample A', 'Sample B',
              'Thermal Conductivity (W/m/K)', 'Lattice Parameter (Angstrom)',
              'compound_type', 'data_source']].to_string(index=False))