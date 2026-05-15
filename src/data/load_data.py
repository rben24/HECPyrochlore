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
from src.globals import PRISTINE, HIGH_ENTROPY, NON_PYROCHLORE

warnings.filterwarnings('ignore')

_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT))

from src.features.build_features import add_engineered_features, FEATURE_COLS

COMBINED_FILE = _PROJECT / 'data' / 'processed' / 'combined_pyrochlore.csv'

# Compound-type constants (mirror make_combined_dataset.py)
# PRISTINE       = 'pristine'
# HIGH_ENTROPY   = 'high_entropy'
# NON_PYROCHLORE = 'non_pyrochlore'
_VALID_TYPES   = {PRISTINE, HIGH_ENTROPY}


# ── helpers ───────────────────────────────────────────────────────────────────

def _ensure_combined(verbose: bool = True) -> None:
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

    # Ensure compound_type column exists; classify on-the-fly if absent
    if 'compound_type' not in df.columns:
        if verbose:
            print("[data] compound_type column missing — classifying now …")
        from src.data.make_combined_dataset import classify_sample
        df['compound_type'] = df.apply(
            lambda r: classify_sample(r['Sample A'], r['Sample B']), axis=1)

    if verbose:
        n_lat = df['Lattice Parameter (Angstrom)'].notna().sum()
        n_tc  = df['TPS Cond W/m/K'].notna().sum()
        n_pri = (df['compound_type'] == PRISTINE).sum()
        n_he  = (df['compound_type'] == HIGH_ENTROPY).sum()
        n_non = (df['compound_type'] == NON_PYROCHLORE).sum()
        print(f"[data] {len(df)} rows | {n_lat} lattice | {n_tc} thermal "
              f"| {n_pri} pristine | {n_he} high-entropy | {n_non} non-pyrochlore")
    return df


def clean_and_engineer(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = add_engineered_features(df)
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
    target = 'Lattice Parameter (Angstrom)'

    # Restrict to valid compound types (always exclude non-pyrochlore)
    df_sub = df[df['compound_type'].isin(compound_types)].copy()
    df_sub = df_sub.dropna(subset=[target])
    df_sub = df_sub[df_sub['Sample A'].notna() & df_sub['Sample B'].notna()]

    feat_mask = df_sub[FEATURE_COLS].notna().all(axis=1)
    df_sub = df_sub[feat_mask]

    X = df_sub[FEATURE_COLS].values.astype(float)
    y = df_sub[target].values.astype(float)

    if verbose:
        print(f"[lattice] {X.shape[0]} samples × {X.shape[1]} features")
        print(f"  compound types : {compound_types}")
        print(f"  target range   : [{y.min():.4f}, {y.max():.4f}] Å")
        _source_breakdown(df_sub, 'lattice')

    return X, y, FEATURE_COLS


def get_thermal_dataset(
    verbose: bool = True,
    compound_types: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Return (X, y, feature_names) for the thermal conductivity model.

    Parameters
    ----------
    compound_types : which compound types to include.
                     Default: ['pristine', 'high_entropy']
    """
    if compound_types is None:
        compound_types = [PRISTINE, HIGH_ENTROPY]

    from src.features.build_features import THERMAL_EXTRA_FEATURES
    df = clean_and_engineer(load_combined(verbose=verbose), verbose=verbose)
    target   = 'TPS Cond W/m/K'
    feat_cols = [c for c in THERMAL_EXTRA_FEATURES if c in df.columns]

    df_sub = df[df['compound_type'].isin(compound_types)].copy()
    df_sub = df_sub.dropna(subset=[target])
    df_sub = df_sub[df_sub['Sample A'].notna() & df_sub['Sample B'].notna()]

    base_feats = [c for c in feat_cols if c != 'lattice_parameter']
    feat_mask  = df_sub[base_feats].notna().all(axis=1)
    df_sub     = df_sub[feat_mask]

    # Impute missing lattice parameter with dataset mean (fallback)
    if 'lattice_parameter' in df_sub.columns:
        mean_lat = df_sub['lattice_parameter'].mean()
        df_sub = df_sub.copy()
        df_sub['lattice_parameter'] = df_sub['lattice_parameter'].fillna(mean_lat)

    X = df_sub[feat_cols].values.astype(float)
    y = df_sub[target].values.astype(float)

    if verbose:
        print(f"[thermal] {X.shape[0]} samples × {X.shape[1]} features")
        print(f"  compound types : {compound_types}")
        print(f"  target range   : [{y.min():.4f}, {y.max():.4f}] W/m·K")
        _source_breakdown(df_sub, 'thermal')

    return X, y, feat_cols


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n=== Lattice dataset ===")
    X, y, names = get_lattice_dataset()
    print(f"Shape: {X.shape}")

    print("\n=== Thermal dataset ===")
    X2, y2, names2 = get_thermal_dataset()
    print(f"Shape: {X2.shape}")