"""
Data loading, cleaning and preprocessing for the HEC Pyrochlore project.
"""

import os
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Tuple, List, Optional

warnings.filterwarnings('ignore')

# Allow import from any working directory
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent
import sys
sys.path.insert(0, str(_PROJECT))

from src.features.build_features import add_engineered_features, FEATURE_COLS


RAW_FILE = _PROJECT / 'data' / 'raw' / 'Sample_Properties_Safin_Feb_2026.csv'


def load_raw(filepath: Optional[str] = None) -> pd.DataFrame:
    path = Path(filepath) if filepath else RAW_FILE
    df = pd.read_csv(path, na_values=['NA', '', 'N/A', 'nan'])
    # Standardise column: use ID as Composition label if present
    if 'ID' in df.columns:
        df.rename(columns={'ID': 'Composition'}, inplace=True)
    return df


def clean_and_engineer(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop useless columns, fix types, add engineered features."""
    drop_cols = ['Notes', 'Conductivity Type', 'Oxygen Parameter x',
                 'Anion Vacancy (delta)', 'B-O Distance (A)', 'B-O-B Angle (deg)']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Numeric coercion
    num_cols = ['TPS Cond W/m/K', 'Deviation', 'Relative Density %',
                'Corrected Cond W/m/K', 'Lattice Parameter (Angstrom)',
                'Lattice Parameter a (A)']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Consolidate lattice parameter columns
    if 'Lattice Parameter a (A)' in df.columns and 'Lattice Parameter (Angstrom)' in df.columns:
        df['Lattice Parameter (Angstrom)'] = df['Lattice Parameter (Angstrom)'].fillna(
            df['Lattice Parameter a (A)'])
        df.drop(columns=['Lattice Parameter a (A)'], inplace=True)

    # Binary target for single-phase
    if 'Is Single Phase' in df.columns:
        df['is_single_phase'] = (df['Is Single Phase'].str.strip().str.lower() == 'yes').astype(float)

    if verbose:
        print(f"[load] {len(df)} rows after cleaning")

    # Add features
    df = add_engineered_features(df)

    if verbose:
        print(f"[features] {df.shape[1]} total columns after feature engineering")

    return df


def get_lattice_dataset(filepath: Optional[str] = None,
                        verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (X, y, feature_names) for lattice-parameter prediction."""
    df = clean_and_engineer(load_raw(filepath), verbose=verbose)
    target = 'Lattice Parameter (Angstrom)'
    df = df.dropna(subset=[target] + FEATURE_COLS)
    # Remove rows where Sample A or B is empty
    df = df[df['Sample A'].notna() & df['Sample B'].notna()]
    X = df[FEATURE_COLS].values.astype(float)
    y = df[target].values.astype(float)
    if verbose:
        print(f"[lattice] {X.shape[0]} samples × {X.shape[1]} features")
    return X, y, FEATURE_COLS


def get_thermal_dataset(filepath: Optional[str] = None,
                        verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (X, y, feature_names) for thermal-conductivity prediction."""
    from src.features.build_features import THERMAL_EXTRA_FEATURES
    df = clean_and_engineer(load_raw(filepath), verbose=verbose)
    target = 'TPS Cond W/m/K'
    feat_cols = [c for c in THERMAL_EXTRA_FEATURES if c in df.columns]
    df = df.dropna(subset=[target] + feat_cols)
    df = df[df['Sample A'].notna() & df['Sample B'].notna()]
    X = df[feat_cols].values.astype(float)
    y = df[target].values.astype(float)
    if verbose:
        print(f"[thermal] {X.shape[0]} samples × {X.shape[1]} features")
    return X, y, feat_cols


if __name__ == '__main__':
    X, y, names = get_lattice_dataset()
    print("Lattice dataset ready:", X.shape)
    X2, y2, names2 = get_thermal_dataset()
    print("Thermal dataset ready:", X2.shape)
