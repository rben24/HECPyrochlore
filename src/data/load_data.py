"""
Data loading and preprocessing for the HEC Pyrochlore project.
Uses the combined dataset built by make_combined_dataset.py.
Falls back to building it automatically if not yet generated.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Tuple, List

warnings.filterwarnings('ignore')

_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent
import sys
sys.path.insert(0, str(_PROJECT))

from src.features.build_features import add_engineered_features, FEATURE_COLS

COMBINED_FILE = _PROJECT / 'data' / 'processed' / 'combined_pyrochlore.csv'


def _ensure_combined(verbose: bool = True):
    if not COMBINED_FILE.exists():
        if verbose:
            print("[data] Combined dataset not found — building it now …")
        from src.data.make_combined_dataset import build_combined_dataset
        build_combined_dataset(save=True)
    elif verbose:
        print(f"[data] Using combined dataset: {COMBINED_FILE.name}")


def load_combined(verbose: bool = True) -> pd.DataFrame:
    _ensure_combined(verbose)
    df = pd.read_csv(COMBINED_FILE, na_values=['NA', '', 'N/A', 'nan'])

    num_cols = ['TPS Cond W/m/K', 'Relative Density %',
                'Lattice Parameter (Angstrom)', 'b_o_distance',
                'b_o_b_angle', 'oxygen_param_x']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if 'Is Single Phase' in df.columns:
        df['is_single_phase'] = (
            df['Is Single Phase'].astype(str).str.strip().str.lower() == 'yes'
        ).astype(float)

    if verbose:
        print(f"[data] {len(df)} rows | "
              f"{df['Lattice Parameter (Angstrom)'].notna().sum()} lattice | "
              f"{df['TPS Cond W/m/K'].notna().sum()} thermal")
    return df


def clean_and_engineer(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = add_engineered_features(df)
    if verbose:
        print(f"[features] {df.shape[1]} total columns after feature engineering")
    return df


def _source_breakdown(df: pd.DataFrame, label: str):
    if 'data_source' in df.columns:
        counts = df['data_source'].value_counts()
        print(f"  [{label}] source breakdown:")
        for src, n in counts.items():
            print(f"    {src:<40} {n:>3} rows")


def get_lattice_dataset(verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = clean_and_engineer(load_combined(verbose=verbose), verbose=verbose)
    target = 'Lattice Parameter (Angstrom)'
    df_sub = df.dropna(subset=[target])
    df_sub = df_sub[df_sub['Sample A'].notna() & df_sub['Sample B'].notna()]
    feat_mask = df_sub[FEATURE_COLS].notna().all(axis=1)
    df_sub = df_sub[feat_mask]
    X = df_sub[FEATURE_COLS].values.astype(float)
    y = df_sub[target].values.astype(float)
    if verbose:
        print(f"[lattice] {X.shape[0]} usable samples × {X.shape[1]} features")
        _source_breakdown(df_sub, 'lattice')
    return X, y, FEATURE_COLS


def get_thermal_dataset(verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    from src.features.build_features import THERMAL_EXTRA_FEATURES
    df = clean_and_engineer(load_combined(verbose=verbose), verbose=verbose)
    target = 'TPS Cond W/m/K'
    feat_cols = [c for c in THERMAL_EXTRA_FEATURES if c in df.columns]
    df_sub = df.dropna(subset=[target])
    df_sub = df_sub[df_sub['Sample A'].notna() & df_sub['Sample B'].notna()]
    base_feats = [c for c in feat_cols if c != 'lattice_parameter']
    feat_mask  = df_sub[base_feats].notna().all(axis=1)
    df_sub = df_sub[feat_mask]
    if 'lattice_parameter' in df_sub.columns:
        mean_lat = df_sub['lattice_parameter'].mean()
        df_sub = df_sub.copy()
        df_sub['lattice_parameter'] = df_sub['lattice_parameter'].fillna(mean_lat)
    X = df_sub[feat_cols].values.astype(float)
    y = df_sub[target].values.astype(float)
    if verbose:
        print(f"[thermal] {X.shape[0]} usable samples × {X.shape[1]} features")
        _source_breakdown(df_sub, 'thermal')
    return X, y, feat_cols


if __name__ == '__main__':
    X, y, names = get_lattice_dataset()
    print(f"\nLattice dataset: {X.shape}")
    X2, y2, names2 = get_thermal_dataset()
    print(f"Thermal dataset: {X2.shape}")
