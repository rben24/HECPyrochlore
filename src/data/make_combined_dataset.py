"""
make_combined_dataset.py
========================
Combines all three raw data sources into a single unified dataset:

  1. Sample_Properties_Safin_Feb_2026.csv  — HEC experimental data (A/B site,
       thermal conductivity, lattice param, density, synthesis)
  2. notebookLM_dataset.csv               — Binary pyrochlore lattice data
       (Compound formula → parsed A/B site, lattice param, structural params)
  3. parent_components.csv                — Literature family data
       (A/B cation lists, representative thermal conductivity)

Each source is parsed into a canonical schema before merging:

  Composition | Sample A | Sample B | TPS Cond W/m/K | Lattice Parameter (Angstrom)
  | Relative Density % | Is Single Phase | Synthesis Method | data_source
  | b_o_distance | b_o_b_angle | oxygen_param_x

Run directly:
  python src/data/make_combined_dataset.py
"""

import re
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='  [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent
RAW_DIR  = _PROJECT / 'data' / 'raw'
OUT_DIR  = _PROJECT / 'data' / 'processed'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUT_DIR / 'combined_pyrochlore.csv'

# ── canonical column schema ───────────────────────────────────────────────────
CANONICAL_COLS = [
    'Composition',
    'Sample A',
    'Sample B',
    'TPS Cond W/m/K',
    'Lattice Parameter (Angstrom)',
    'Relative Density %',
    'Is Single Phase',
    'Synthesis Method',
    'data_source',
    'b_o_distance',
    'b_o_b_angle',
    'oxygen_param_x',
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _clean_element_list(raw: str) -> str:
    """Normalise element strings: strip whitespace, remove subscripts."""
    if pd.isna(raw):
        return np.nan
    # remove numeric suffixes like ³⁺ or subscripts
    cleaned = re.sub(r'[⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉⁺⁻]+', '', str(raw))
    # split on comma, strip each element
    parts = [p.strip() for p in cleaned.split(',') if p.strip()]
    return ','.join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1: Experimental HEC data (Safin Feb 2026)
# ─────────────────────────────────────────────────────────────────────────────

def load_safin() -> pd.DataFrame:
    path = RAW_DIR / 'Sample_Properties_Safin_Feb_2026.csv'
    df = pd.read_csv(path, na_values=['NA', '', 'N/A'])
    log.info(f"Safin: {len(df)} rows loaded from {path.name}")

    out = pd.DataFrame()
    out['Composition']                = df['ID'].fillna('').astype(str)
    out['Sample A']                   = df['Sample A'].apply(_clean_element_list)
    out['Sample B']                   = df['Sample B'].apply(_clean_element_list)
    out['TPS Cond W/m/K']             = pd.to_numeric(df['TPS Cond W/m/K'], errors='coerce')
    out['Lattice Parameter (Angstrom)'] = pd.to_numeric(df['Lattice Parameter (Angstrom)'], errors='coerce')
    out['Relative Density %']         = pd.to_numeric(df['Relative Density %'], errors='coerce')
    out['Is Single Phase']            = df['Is Single Phase'].str.strip().str.lower().map(
                                            {'yes': 'Yes', 'no': 'No'})
    out['Synthesis Method']           = df['Synthesis Method'].fillna('')
    out['data_source']                = 'safin_experimental'
    out['b_o_distance']               = np.nan
    out['b_o_b_angle']                = np.nan
    out['oxygen_param_x']             = np.nan

    log.info(f"Safin: {out['Lattice Parameter (Angstrom)'].notna().sum()} lattice values, "
             f"{out['TPS Cond W/m/K'].notna().sum()} thermal values")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2: notebookLM binary pyrochlore dataset
# ─────────────────────────────────────────────────────────────────────────────

# Map from formula-style compound name → (A elements, B elements)
# Keys are lowercase stripped of spaces for matching
_NLM_FORMULA_MAP = {
    'lu2ti2o7':      ('Lu',        'Ti'),
    'bi2sn2o7':      ('Bi',        'Sn'),
    'nd2ir2o7':      ('Nd',        'Ir'),
    'pr2ir2o7':      ('Pr',        'Ir'),
    'bi2ir2o7':      ('Bi',        'Ir'),
    '(biy)ir2o7':    ('Bi,Y',      'Ir'),
    '(bipb)ir2o7':   ('Bi,Pb',     'Ir'),
    'pb2ir2o7':      ('Pb',        'Ir'),
    'gd2zr2o7':      ('Gd',        'Zr'),
    # high-entropy labelled entries from the combined df
    '#a3zo':         ('Nd,Sm,Eu,Gd,Dy',         'Zr'),   # 3-element A-site ZO
    '#a5zo':         ('La,Nd,Sm,Eu,Gd,Dy,Ho',   'Zr'),   # 5 (approximate)
    '#a7zo':         ('La,Nd,Sm,Eu,Gd,Dy,Ho',   'Zr'),   # 7-element
}

def _parse_nlm_compound(compound: str):
    """Return (a_str, b_str) from a compound name string."""
    key = compound.strip().lower().replace(' ', '').split('(')[0]
    # try direct map
    for pattern, ab in _NLM_FORMULA_MAP.items():
        if pattern in key or key.startswith(pattern.split('(')[0]):
            return ab
    # fallback: try to parse A2X2O7 style
    m = re.match(r'([a-z]+)2([a-z]+)2o7', key)
    if m:
        a_raw, b_raw = m.group(1).capitalize(), m.group(2).capitalize()
        return a_raw, b_raw
    return np.nan, np.nan


def load_nlm() -> pd.DataFrame:
    path = RAW_DIR / 'notebookLM_dataset.csv'
    df = pd.read_csv(path, na_values=['-', '', 'NA'])
    log.info(f"NLM: {len(df)} rows loaded from {path.name}")

    rows = []
    for _, row in df.iterrows():
        compound = str(row['Compound']).strip()
        a_str, b_str = _parse_nlm_compound(compound)

        lattice = pd.to_numeric(row.get('Lattice Parameter a (A)', np.nan), errors='coerce')
        bo_dist = pd.to_numeric(row.get('B-O Distance (A)', np.nan), errors='coerce')
        bob_ang = pd.to_numeric(row.get('B-O-B Angle (deg)', np.nan), errors='coerce')
        ox_x    = pd.to_numeric(row.get('Oxygen Parameter x', np.nan), errors='coerce')

        rows.append({
            'Composition':                  compound,
            'Sample A':                     a_str,
            'Sample B':                     b_str,
            'TPS Cond W/m/K':               np.nan,   # not in this source
            'Lattice Parameter (Angstrom)': lattice,
            'Relative Density %':           np.nan,
            'Is Single Phase':              np.nan,
            'Synthesis Method':             '',
            'data_source':                  'notebookLM_literature',
            'b_o_distance':                 bo_dist,
            'b_o_b_angle':                  bob_ang,
            'oxygen_param_x':               ox_x,
        })

    out = pd.DataFrame(rows)
    # Drop rows where we couldn't parse A or B (and have no lattice param either)
    out = out[out['Lattice Parameter (Angstrom)'].notna() | out['Sample A'].notna()]
    log.info(f"NLM: {out['Lattice Parameter (Angstrom)'].notna().sum()} lattice values parsed")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3: Parent components literature families
# ─────────────────────────────────────────────────────────────────────────────

def load_parent_components() -> pd.DataFrame:
    path = RAW_DIR / 'parent_components.csv'
    df = pd.read_csv(path, na_values=['', 'NA'])
    log.info(f"Parent components: {len(df)} rows loaded from {path.name}")

    rows = []
    for _, row in df.iterrows():
        # A-site: already a comma-separated element list
        a_raw = _clean_element_list(str(row.get('A-site Cations', '')))
        b_raw = _clean_element_list(str(row.get('B-site Cations', '')))

        # Thermal conductivity is given as a range string like "0.8-1.2" or "<1.0"
        # → take the midpoint / strip < sign
        tc_raw = str(row.get('Thermal Conductivity (W/m·K)', '')).strip()
        tc_val = _parse_tc_range(tc_raw)

        # Build a readable composition label
        comp_label = str(row.get('Parent Compound', '')).strip()

        rows.append({
            'Composition':                  comp_label,
            'Sample A':                     a_raw,
            'Sample B':                     b_raw,
            'TPS Cond W/m/K':               tc_val,
            'Lattice Parameter (Angstrom)': np.nan,   # not in this source
            'Relative Density %':           np.nan,
            'Is Single Phase':              'Yes',    # all listed as pyrochlore single-phase
            'Synthesis Method':             '',
            'data_source':                  'parent_components_literature',
            'b_o_distance':                 np.nan,
            'b_o_b_angle':                  np.nan,
            'oxygen_param_x':               np.nan,
        })

    out = pd.DataFrame(rows)
    log.info(f"Parent components: {out['TPS Cond W/m/K'].notna().sum()} thermal values parsed")
    return out


def _parse_tc_range(tc_str: str) -> float:
    """Parse strings like '<1.0', '0.8-1.2', '1.1-1.5' to a midpoint float."""
    if not tc_str or tc_str in ('nan', '-', ''):
        return np.nan
    tc_str = tc_str.replace('<', '').replace('>', '').strip()
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


# ─────────────────────────────────────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────────────────────────────────────

def build_combined_dataset(save: bool = True) -> pd.DataFrame:
    """Load, parse, and combine all three sources."""
    print()
    print("=" * 62)
    print("  Building Combined Pyrochlore Dataset")
    print("=" * 62)

    frames = []

    # Source 1
    try:
        frames.append(load_safin())
    except FileNotFoundError as e:
        log.warning(f"Skipping Safin data: {e}")

    # Source 2
    try:
        frames.append(load_nlm())
    except FileNotFoundError as e:
        log.warning(f"Skipping NLM data: {e}")

    # Source 3
    try:
        frames.append(load_parent_components())
    except FileNotFoundError as e:
        log.warning(f"Skipping parent components: {e}")

    if not frames:
        raise RuntimeError("No data sources found — check data/raw/ directory.")

    combined = pd.concat(frames, ignore_index=True)[CANONICAL_COLS]

    # Drop rows where BOTH Sample A and Sample B are missing
    combined = combined[combined['Sample A'].notna() | combined['Sample B'].notna()]

    # Remove entries with Bi or Pb (lone-pair active, incompatible feature table)
    has_exotic = combined['Sample A'].fillna('').str.contains(r'\b(Bi|Pb)\b', regex=True)
    n_before = len(combined)
    combined = combined[~has_exotic].reset_index(drop=True)
    n_dropped = n_before - len(combined)
    if n_dropped:
        log.info(f"Dropped {n_dropped} rows with Bi/Pb (no ionic radius data)")

    print()
    print(f"  {'Source':<35} {'Rows':>5}  {'Lattice':>8}  {'Thermal':>8}")
    print("  " + "-" * 60)
    for src, grp in combined.groupby('data_source'):
        lat = grp['Lattice Parameter (Angstrom)'].notna().sum()
        tc  = grp['TPS Cond W/m/K'].notna().sum()
        print(f"  {src:<35} {len(grp):>5}  {lat:>8}  {tc:>8}")
    print("  " + "-" * 60)
    lat_total = combined['Lattice Parameter (Angstrom)'].notna().sum()
    tc_total  = combined['TPS Cond W/m/K'].notna().sum()
    print(f"  {'TOTAL':<35} {len(combined):>5}  {lat_total:>8}  {tc_total:>8}")
    print()

    if save:
        combined.to_csv(OUTPUT_FILE, index=False)
        log.info(f"Saved combined dataset → {OUTPUT_FILE}")

    return combined


if __name__ == '__main__':
    df = build_combined_dataset(save=True)
    print("Sample rows:\n")
    print(df[['Composition', 'Sample A', 'Sample B',
              'TPS Cond W/m/K', 'Lattice Parameter (Angstrom)',
              'data_source']].to_string(index=False))
