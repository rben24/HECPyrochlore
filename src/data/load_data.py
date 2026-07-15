"""
Data loading and preprocessing for the HEC Pyrochlore project.

Changes vs. original
--------------------
* Passes ``a_stoich_json`` / ``b_stoich_json`` columns through to the feature
  builder so that non-equiatomic ICSD compositions are handled correctly.
* Reports a breakdown of compound types (pristine / high_entropy) in each
  dataset alongside the existing source breakdown.
* Filters training data so only pyrochlore entries (pristine + high_entropy)
  are used; non-pyrochlore rows are never included in X / y arrays.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path
from typing import Tuple, List
import logging
from src.features.build_features import add_engineered_features, FEATURE_COLS
from src.features.rule_of_mixture_cal import rom_from_dataframe
from src import globals

logging.basicConfig(level=logging.INFO, format='  [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent
DATA  = _PROJECT / 'data' / 'processed'
OUTPUT_FILE_L = DATA / 'engineered_pyrochlore_latt.csv'
OUTPUT_FILE_L_ROM = DATA / 'rom_hec_latt.csv'
OUTPUT_FILE_T_ROM = DATA / 'rom_hec_therm_cond.csv'
OUTPUT_FILE_V_ROM = DATA / 'rom_hec_vick.csv'
COMBINED_FILE = DATA / 'combined_pyrochlore.csv'
HEC_FILE = DATA / 'hec_pyrochlore.csv'

warnings.filterwarnings('ignore')

sys.path.insert(0, str(_PROJECT))



# Compound-type constants (mirror make_combined_dataset.py)
PRISTINE       = globals.PRISTINE
HIGH_ENTROPY   = globals.HIGH_ENTROPY
NON_PYROCHLORE = globals.NON_PYROCHLORE
_VALID_TYPES   = {PRISTINE, HIGH_ENTROPY}

DROP_COLUMNS = [
    'Relative Density %', 'Synthesis Method', 'b_o_distance',
    'b_o_b_angle', 'oxygen_param_x', 'icsd_collection',
    'n_icsd_duplicates', 'lattice_parameter', 'Thermal Conductivity (W/m/K)',
    'Is Single Phase',
]

MEAN_COLUMNS = [
    'lattice_volume', 'density_theoretical', 'density_calc',
    'Temperature',
]

FEATURE_COLS_ADD = [
    'Temperature', 'density_calc', 'lattice_volume',
    'density_theoretical'
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _ensure_combined(verbose: bool = True) -> None:
    if not COMBINED_FILE.exists():
        if verbose:
            print("[data] Combined dataset not found — building it now …")
        from src.data.make_combined_dataset import build_combined_dataset
        build_combined_dataset(save=True)
    elif verbose:
        print(f"[data] Using combined dataset: {COMBINED_FILE.name}")

def _ensure_hec(verbose: bool = True) -> None:
    if not HEC_FILE.exists():
        if verbose:
            print("[data] HEC dataset not found — building it now …")
        from src.data.make_combined_dataset import build_high_entropy_dataset
        build_high_entropy_dataset(save=True)
    elif verbose:
        print(f"[data] Using hec dataset: {HEC_FILE.name}")

def load_combined(verbose: bool = True) -> pd.DataFrame:
    _ensure_combined(verbose)
    df = pd.read_csv(COMBINED_FILE, na_values=['NA', '', 'N/A', 'nan'])

    num_cols = ['Thermal Conductivity (W/m/K)', 'Relative Density %',
                'Lattice Parameter (Å)', 'b_o_distance',
                'b_o_b_angle', 'oxygen_param_x']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if 'Is Single Phase' in df.columns:
        df['is_single_phase'] = (
            df['Is Single Phase'].astype(str).str.strip().str.lower() == 'yes'
        ).astype(float)

    # Ensure compound_type column exists; classify on-the-fly if absent
    if 'compound_type' not in df.columns:
        if verbose:
            print("[data] compound_type column missing — classifying now …")
        from src.data.make_combined_dataset import classify_sample
        df['compound_type'] = df.apply(
            lambda r: classify_sample(r['Sample A'], r['Sample B']), axis=1)

    if verbose:
        n_lat = df['Lattice Parameter (Å)'].notna().sum()
        n_tc  = df['Thermal Conductivity (W/m/K)'].notna().sum()
        n_pri = (df['compound_type'] == PRISTINE).sum()
        n_he  = (df['compound_type'] == HIGH_ENTROPY).sum()
        n_non = (df['compound_type'] == NON_PYROCHLORE).sum()
        print(f"[data] {len(df)} rows | {n_lat} lattice | {n_tc} thermal "
              f"| {n_pri} pristine | {n_he} high-entropy | {n_non} non-pyrochlore")
    return df

def load_hec(verbose: bool = True) -> pd.DataFrame:
    _ensure_hec(verbose)
    df = pd.read_csv(HEC_FILE, na_values=['NA', '', 'N/A', 'nan'])

    num_cols = ['Thermal Conductivity (W/m/K)', 'Lattice Parameter (Å)', 'Temperature',]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Ensure compound_type column exists; classify on-the-fly if absent
    if 'compound_type' not in df.columns:
        if verbose:
            print("[data] compound_type column missing — classifying now …")
        from src.data.make_combined_dataset import classify_sample
        df['compound_type'] = df.apply(
            lambda r: classify_sample(r['Sample A'], r['Sample B']), axis=1)

    if verbose:
        n_lat = df['Lattice Parameter (Å)'].notna().sum()
        n_tc = df['Thermal Conductivity (W/m/K)'].notna().sum()
        print(f"[data] {len(df)} rows | {n_lat} lattice | {n_tc} thermal ")
    return df

def clean_and_engineer(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = add_engineered_features(df)
    if verbose:
        print(f"[features] {df.shape[1]} total columns after feature engineering")
    return df

def clean_and_rom(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = rom_from_dataframe(df, auto_add_missing=False, auto_save=False)
    if verbose:
        print(f"[features] {df.shape[1]} total columns after feature engineering")
    return df

def _source_breakdown(df: pd.DataFrame, label: str) -> None:
    """Print per-source and per-compound-type row counts."""
    if 'data_source' not in df.columns:
        return
    print(f"  [{label}] breakdown by source:")
    for src, grp in df.groupby('data_source'):
        n_pri = (grp.get('compound_type', pd.Series(dtype=str)) == PRISTINE).sum()
        n_he  = (grp.get('compound_type', pd.Series(dtype=str)) == HIGH_ENTROPY).sum()
        print(f"    {src:<40} {len(grp):>3} rows  "
              f"(pristine: {n_pri}, high-entropy: {n_he})")


# ── Dataset builders ──────────────────────────────────────────────────────────

def get_lattice_dataset(
    verbose: bool = True,
    compound_types: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Return (X, y, feature_names) for the lattice-parameter model.

    Parameters
    ----------
    compound_types : which compound types to include.
                     Default: ['pristine', 'high_entropy']
                     Pass ['pristine'] to train on single-element compositions only.
    """
    if compound_types is None:
        compound_types = [PRISTINE, HIGH_ENTROPY]

    df = clean_and_engineer(load_combined(verbose=verbose), verbose=verbose)
    target = 'Lattice Parameter (Å)'

    # Restrict to valid compound types (always exclude non-pyrochlore)
    df_sub = df[df['compound_type'].isin(compound_types)].copy()
    df_sub = df_sub.dropna(subset=[target])
    df_sub = df_sub[df_sub['Sample A'].notna() & df_sub['Sample B'].notna()]

    # Deal with NaN columns
    df_sub = df_sub.drop(columns=DROP_COLUMNS)
    for col in MEAN_COLUMNS:
        if col in df_sub.columns:
            col_mean = df_sub[col].mean()
            df_sub.fillna(value={col: col_mean})

    df_sub.to_csv(OUTPUT_FILE_L, index=False)
    log.info(f"Saved combined dataset → {OUTPUT_FILE_L}")

    feat_mask = df_sub[FEATURE_COLS + FEATURE_COLS_ADD].notna().all(axis=1)
    df_sub = df_sub[feat_mask]

    X = df_sub[FEATURE_COLS + FEATURE_COLS_ADD].values.astype(float)
    y = df_sub[target].values.astype(float)

    if verbose:
        print(f"[lattice] {X.shape[0]} samples × {X.shape[1]} features")
        print(f"  compound types : {compound_types}")
        print(f"  target range   : [{y.min():.4f}, {y.max():.4f}] Å")
        _source_breakdown(df_sub, 'lattice')

    return X, y, FEATURE_COLS

def get_lattice_rom_dataset(
    verbose: bool = True,
    # compound_types: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Return (X, y, feature_names) for the lattice-parameter model.

    Parameters
    ----------
    # compound_types : which compound types to include.
    #                  Default: ['high_entropy']
    """
    # if compound_types is None:
    compound_types = [HIGH_ENTROPY]

    df = clean_and_rom(load_hec(verbose=verbose), verbose=verbose)
    target = 'Lattice Parameter (Å)'

    # Restrict to valid compound types (always exclude non-pyrochlore)
    df_sub = df[df['compound_type'].isin(compound_types)].copy()
    df_sub = df_sub.dropna(subset=[target])
    df_sub = df_sub[df_sub['Sample A'].notna() & df_sub['Sample B'].notna()]

    # Drop outlier columns given by plot_latt_vs_ROMlatt_wOutliers.py
    # df_sub = df_sub[~df_sub['Composition'].isin(globals.OUTLIER_COMPS)].reset_index(drop=True)

    # Combine Ionic Radius columns
    df_sub['Ionic Radius A (Å)'] = df_sub.apply(
        lambda row: (row['Ionic Radius A (Å)'] + row['ROM_Ionic_Radius_A']) / 2 \
            if pd.notna(row['Ionic Radius A (Å)']) and pd.notna(row['ROM_Ionic_Radius_A'])
        else row['Ionic Radius A (Å)'] if pd.notna(row['Ionic Radius A (Å)'])
        else row['ROM_Ionic_Radius_A'],
        axis=1
    )
    df_sub['Ionic Radius B (Å)'] = df_sub.apply(
        lambda row: (row['Ionic Radius B (Å)'] + row['ROM_Ionic_Radius_B']) / 2 \
            if pd.notna(row['Ionic Radius B (Å)']) and pd.notna(row['ROM_Ionic_Radius_B'])
        else row['Ionic Radius B (Å)'] if pd.notna(row['Ionic Radius B (Å)'])
        else row['ROM_Ionic_Radius_B'],
        axis=1
    )
    df_sub['rA/rB (Å)'] = df_sub.apply(
        lambda row: (row['rA/rB (Å)'] + row['ROM_Radius_Ratio_rA_rB']) / 2 \
            if pd.notna(row['rA/rB (Å)']) and pd.notna(row['ROM_Radius_Ratio_rA_rB'])
        else row['rA/rB (Å)'] if pd.notna(row['rA/rB (Å)'])
        else row['ROM_Radius_Ratio_rA_rB'],
        axis=1
    )

    # Deal with NaN columns
    # df_sub = df_sub.drop(columns=DROP_COLUMNS)
    df_sub.dropna(subset=['Ionic Radius A (Å)', 'Ionic Radius B (Å)'], inplace=True)
    threshold = 0.75
    min_non_na = int(threshold * len(df_sub))
    df_sub.dropna(axis=1, thresh=min_non_na, inplace=True)
    mean_cols = globals.ROM_THERM_COND_FEAT_COLS
    for col in mean_cols:
        if col in df_sub.columns:
            col_mean = df_sub[col].mean()
            # print(f'Mean of {col}: {col_mean}')
            df_sub[col] = df_sub[col].fillna(col_mean)
    print(df_sub.shape)

    df_sub.to_csv(OUTPUT_FILE_L_ROM, index=False)
    log.info(f"Saved combined dataset → {OUTPUT_FILE_L_ROM}")

    dropped = list(set(globals.ROM_LATT_FEAT_COLS) - set(df_sub.columns.to_list()))
    print(dropped)
    cols = [c for c in globals.ROM_LATT_FEAT_COLS if c not in dropped]
    feat_mask = df_sub[cols + [target]].notna().all(axis=1)
    df_sub = df_sub[feat_mask]

    X = df_sub[cols].values.astype(float)
    y = df_sub[target].values.astype(float)

    if verbose:
        print(f"[lattice] {X.shape[0]} samples × {X.shape[1]} features")
        print(f"  compound types : {compound_types}")
        print(f"  target range   : [{y.min():.4f}, {y.max():.4f}] Å")
        _source_breakdown(df_sub, 'lattice')

    return X, y, cols

def get_therm_cond_rom_dataset(
    verbose: bool = True,
    # compound_types: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Return (X, y, feature_names) for the thermal conductivity model.

    Parameters
    ----------
    # compound_types : which compound types to include.
    #                  Default: ['high_entropy']
    """
    # if compound_types is None:
    compound_types = [HIGH_ENTROPY]

    df = clean_and_rom(load_hec(verbose=verbose), verbose=verbose)
    target = 'Thermal Conductivity (W/m/K)'

    # Restrict to valid compound types (always exclude non-pyrochlore)
    df_sub = df[df['compound_type'].isin(compound_types)].copy()
    df_sub = df_sub.dropna(subset=[target])
    df_sub = df_sub[df_sub['Sample A'].notna() & df_sub['Sample B'].notna()]

    # Drop outlier columns given by plot_latt_vs_ROMlatt_wOutliers.py
    # df_sub = df_sub[~df_sub['Composition'].isin(globals.OUTLIER_COMPS)].reset_index(drop=True)

    # Combine Ionic Radius columns
    df_sub['Ionic Radius A (Å)'] = df_sub.apply(
        lambda row: (row['Ionic Radius A (Å)'] + row['ROM_Ionic_Radius_A']) / 2 \
            if pd.notna(row['Ionic Radius A (Å)']) and pd.notna(row['ROM_Ionic_Radius_A'])
        else row['Ionic Radius A (Å)'] if pd.notna(row['Ionic Radius A (Å)'])
        else row['ROM_Ionic_Radius_A'],
        axis=1
    )
    df_sub['Ionic Radius B (Å)'] = df_sub.apply(
        lambda row: (row['Ionic Radius B (Å)'] + row['ROM_Ionic_Radius_B']) / 2 \
            if pd.notna(row['Ionic Radius B (Å)']) and pd.notna(row['ROM_Ionic_Radius_B'])
        else row['Ionic Radius B (Å)'] if pd.notna(row['Ionic Radius B (Å)'])
        else row['ROM_Ionic_Radius_B'],
        axis=1
    )
    df_sub['rA/rB (Å)'] = df_sub.apply(
        lambda row: (row['rA/rB (Å)'] + row['ROM_Radius_Ratio_rA_rB']) / 2 \
            if pd.notna(row['rA/rB (Å)']) and pd.notna(row['ROM_Radius_Ratio_rA_rB'])
        else row['rA/rB (Å)'] if pd.notna(row['rA/rB (Å)'])
        else row['ROM_Radius_Ratio_rA_rB'],
        axis=1
    )

    # Deal with NaN columns
    # df_sub = df_sub.drop(columns=DROP_COLUMNS)
    df_sub.dropna(subset=['Ionic Radius A (Å)', 'Ionic Radius B (Å)'], inplace=True)
    threshold = 0.75
    min_non_na = int(threshold * len(df_sub))
    df_sub.dropna(axis=1, thresh=min_non_na, inplace=True)
    mean_cols = globals.ROM_THERM_COND_FEAT_COLS
    for col in mean_cols:
        if col in df_sub.columns:
            col_mean = df_sub[col].mean()
            # print(f'Mean of {col}: {col_mean}')
            df_sub[col] = df_sub[col].fillna(col_mean)
    print(df_sub.shape)

    df_sub.to_csv(OUTPUT_FILE_T_ROM, index=False)
    log.info(f"Saved combined dataset → {OUTPUT_FILE_T_ROM}")

    dropped = list(set(globals.ROM_THERM_COND_FEAT_COLS) - set(df_sub.columns.to_list()))
    print(dropped)
    cols = [c for c in globals.ROM_THERM_COND_FEAT_COLS if c not in dropped]
    feat_mask = df_sub[cols + [target]].notna().all(axis=1)
    df_sub = df_sub[feat_mask]

    X = df_sub[cols].values.astype(float)
    y = df_sub[target].values.astype(float)

    if verbose:
        print(f"[therm_cond] {X.shape[0]} samples × {X.shape[1]} features")
        print(f"  compound types : {compound_types}")
        print(f"  target range   : [{y.min():.4f}, {y.max():.4f}] W/m/K")
        _source_breakdown(df_sub, 'Thermal Conductivity')

    return X, y, cols

def get_vickers_rom_dataset(
    verbose: bool = True,
    # compound_types: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Return (X, y, feature_names) for the Vickers Hardness model.

    Parameters
    ----------
    # compound_types : which compound types to include.
    #                  Default: ['high_entropy']
    """
    # if compound_types is None:
    compound_types = [HIGH_ENTROPY]

    df = clean_and_rom(load_hec(verbose=verbose), verbose=verbose)
    target = 'Vickers Hardness (GPa)'

    # Restrict to valid compound types (always exclude non-pyrochlore)
    df_sub = df[df['compound_type'].isin(compound_types)].copy()
    df_sub = df_sub.dropna(subset=[target])
    df_sub = df_sub[df_sub['Sample A'].notna() & df_sub['Sample B'].notna()]

    # Drop outlier columns given by plot_latt_vs_ROMlatt_wOutliers.py
    # df_sub = df_sub[~df_sub['Composition'].isin(globals.OUTLIER_COMPS)].reset_index(drop=True)

    # Combine Ionic Radius columns
    df_sub['Ionic Radius A (Å)'] = df_sub.apply(
        lambda row: (row['Ionic Radius A (Å)'] + row['ROM_Ionic_Radius_A']) / 2 \
            if pd.notna(row['Ionic Radius A (Å)']) and pd.notna(row['ROM_Ionic_Radius_A'])
        else row['Ionic Radius A (Å)'] if pd.notna(row['Ionic Radius A (Å)'])
        else row['ROM_Ionic_Radius_A'],
        axis=1
    )
    df_sub['Ionic Radius B (Å)'] = df_sub.apply(
        lambda row: (row['Ionic Radius B (Å)'] + row['ROM_Ionic_Radius_B']) / 2 \
            if pd.notna(row['Ionic Radius B (Å)']) and pd.notna(row['ROM_Ionic_Radius_B'])
        else row['Ionic Radius B (Å)'] if pd.notna(row['Ionic Radius B (Å)'])
        else row['ROM_Ionic_Radius_B'],
        axis=1
    )
    df_sub['rA/rB (Å)'] = df_sub.apply(
        lambda row: (row['rA/rB (Å)'] + row['ROM_Radius_Ratio_rA_rB']) / 2 \
            if pd.notna(row['rA/rB (Å)']) and pd.notna(row['ROM_Radius_Ratio_rA_rB'])
        else row['rA/rB (Å)'] if pd.notna(row['rA/rB (Å)'])
        else row['ROM_Radius_Ratio_rA_rB'],
        axis=1
    )

    # Deal with NaN columns
    # df_sub = df_sub.drop(columns=DROP_COLUMNS)
    df_sub.dropna(subset=['Ionic Radius A (Å)', 'Ionic Radius B (Å)'], inplace=True)
    threshold = 0.75
    min_non_na = int(threshold * len(df_sub))
    df_sub.dropna(axis=1, thresh=min_non_na, inplace=True)
    mean_cols = globals.ROM_HARDNESS_FEAT_COLS
    for col in mean_cols:
        if col in df_sub.columns:
            col_mean = df_sub[col].mean()
            # print(f'Mean of {col}: {col_mean}')
            df_sub[col] = df_sub[col].fillna(col_mean)
    print(df_sub.shape)

    df_sub.to_csv(OUTPUT_FILE_V_ROM, index=False)
    log.info(f"Saved combined dataset → {OUTPUT_FILE_V_ROM}")

    dropped = list(set(globals.ROM_HARDNESS_FEAT_COLS) - set(df_sub.columns.to_list()))
    print(dropped)
    cols = [c for c in globals.ROM_HARDNESS_FEAT_COLS if c not in dropped]
    feat_mask = df_sub[cols + [target]].notna().all(axis=1)
    df_sub = df_sub[feat_mask]

    X = df_sub[cols].values.astype(float)
    y = df_sub[target].values.astype(float)

    if verbose:
        print(f"[vickers_hardness] {X.shape[0]} samples × {X.shape[1]} features")
        print(f"  compound types : {compound_types}")
        print(f"  target range   : [{y.min():.4f}, {y.max():.4f}] GPa")
        _source_breakdown(df_sub, 'Vickers Hardness')

    return X, y, cols

# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    # print("\n=== Lattice dataset ===")
    # X, y, names = get_lattice_dataset()
    # print(f"Shape: {X.shape}")

    # print("\n=== Lattice ROM dataset ===")
    # Xr, yr, names_r = get_lattice_rom_dataset()
    # print(f"Shape: {Xr.shape}")

    # print("\n=== Thermal dataset ===")
    # X2, y2, names2 = get_thermal_dataset()
    # print(f"Shape: {X2.shape}")

    print("\n=== Thermal Conductivity ROM dataset ===")
    Xt, yt, names_t = get_therm_cond_rom_dataset()
    print(f"Shape: {Xt.shape}")

    print("\n=== Vickers Hardness ROM dataset ===")
    Xv, yv, names_v = get_vickers_rom_dataset()
    print(f"Shape: {Xv.shape}")

