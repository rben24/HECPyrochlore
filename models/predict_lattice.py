#!/usr/bin/env python3
"""
predict_lattice.py
==================
Interactive predictor for **lattice parameter (Å)** of a pyrochlore oxide.

Usage
-----
  python models/predict_lattice.py
  python models/predict_lattice.py --a "La,Gd,Lu" --b "Ti,Zr"
  python models/predict_lattice.py --train          # force re-train before predicting

The script will train the model automatically on first run and cache it in
  models/lattice_param/lattice_param_model.pkl
"""

import argparse
import sys
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')

# ── path setup ────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
sys.path.insert(0, str(_PROJECT))

from src.features.build_features import (
    build_features_for_row, FEATURE_COLS,
    IONIC_RADII_8, IONIC_RADII_6,
    parse_composition, mean_radius, a_site_b_site_radius_ratio,
)
from src.data.load_data import get_lattice_dataset
from src.build_models.train_model import train_and_evaluate, load_model

import pandas as pd

MODEL_DIR = _PROJECT / 'models' / 'lattice_param'
TASK = 'lattice_param'


# ── helpers ───────────────────────────────────────────────────────────────────

def _check_elements(comp_str: str, radii_table: dict, site: str):
    comp = parse_composition(comp_str)
    unknown = [e for e in comp if e not in radii_table]
    if unknown:
        print(f"  WARNING: Unknown {site}-site element(s): {unknown}")
        print(f"  Known elements: {sorted(radii_table.keys())}")
    return comp


def _print_banner():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║     HEC Pyrochlore — Lattice Parameter Predictor    ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def _ensure_model(force_train: bool = False):
    """Train if model file doesn't exist or force_train is True."""
    model_file = MODEL_DIR / f'{TASK}_model.pkl'
    if not model_file.exists() or force_train:
        print("[train] Training lattice parameter model …")
        X, y, feat_names = get_lattice_dataset(verbose=True)
        results = train_and_evaluate(
            X, y, feat_names,
            task_name=TASK,
            save_dir=MODEL_DIR,
            verbose=True,
        )
        print(f"[train] Model saved to {MODEL_DIR}")
        return results['best_model'], results['scaler']
    else:
        model, scaler, meta = load_model(TASK, MODEL_DIR)
        print(f"[load] Loaded cached model  ({meta['best_model']})")
        return model, scaler


def predict_from_composition(a_str: str, b_str: str,
                             model, scaler) -> float:
    """Build feature vector and return predicted lattice parameter."""
    row = pd.Series({
        'Sample A': a_str,
        'Sample B': b_str,
        'Lattice Parameter (Angstrom)': np.nan,
        'TPS Cond W/m/K': np.nan,
        'Relative Density %': np.nan,
        'Is Single Phase': np.nan,
    })
    feats = build_features_for_row(row)
    X_raw = np.array([feats.get(c, np.nan) for c in FEATURE_COLS], dtype=float)

    if np.any(np.isnan(X_raw)):
        missing = [FEATURE_COLS[i] for i, v in enumerate(X_raw) if np.isnan(v)]
        print(f"  WARNING: NaN features — {missing[:5]}{'…' if len(missing) > 5 else ''}")

    X_scaled = scaler.transform(X_raw.reshape(1, -1))
    return float(model.predict(X_scaled)[0])


def interactive_loop(model, scaler):
    """Prompt the user repeatedly until they quit."""
    print("Enter compositions as comma-separated element symbols.")
    print("  A-site: rare-earth cations  (e.g.  La,Gd,Lu  or  Pr,Nd,Gd,Yb,Lu)")
    print("  B-site: transition metal(s) (e.g.  Ti  or  Ti,Zr)")
    print("Type  'quit'  to exit.\n")

    while True:
        print("─" * 54)
        a = input("  A-site elements: ").strip()
        if a.lower() in ('quit', 'q', 'exit'):
            break
        b = input("  B-site elements: ").strip()
        if b.lower() in ('quit', 'q', 'exit'):
            break

        # Informational checks
        a_comp = _check_elements(a, IONIC_RADII_8, 'A')
        b_comp = _check_elements(b, IONIC_RADII_6, 'B')
        r_a = mean_radius(a_comp, IONIC_RADII_8)
        r_b = mean_radius(b_comp, IONIC_RADII_6)
        ratio = a_site_b_site_radius_ratio(r_a, r_b)

        pred = predict_from_composition(a, b, model, scaler)

        print()
        print(f"  Composition :  ({a})₂({b})₂O₇")
        print(f"  A-site radii:  r̄_A = {r_a:.4f} Å  (8-coord)")
        print(f"  B-site radii:  r̄_B = {r_b:.4f} Å  (6-coord)")
        print(f"  r_A/r_B ratio: {ratio:.4f}")
        print()
        print(f"  ► Predicted lattice parameter:  a = {pred:.5f} Å")
        print()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Predict pyrochlore lattice parameter from A/B-site composition.')
    parser.add_argument('--a', type=str, help='A-site elements (comma-separated)')
    parser.add_argument('--b', type=str, help='B-site elements (comma-separated)')
    parser.add_argument('--train', action='store_true',
                        help='Force re-train before predicting')
    args = parser.parse_args()

    _print_banner()
    model, scaler = _ensure_model(force_train=args.train)

    if args.a and args.b:
        # Single CLI prediction
        a_comp = _check_elements(args.a, IONIC_RADII_8, 'A')
        b_comp = _check_elements(args.b, IONIC_RADII_6, 'B')
        r_a = mean_radius(a_comp, IONIC_RADII_8)
        r_b = mean_radius(b_comp, IONIC_RADII_6)
        ratio = a_site_b_site_radius_ratio(r_a, r_b)
        pred = predict_from_composition(args.a, args.b, model, scaler)
        print(f"  Composition :  ({args.a})₂({args.b})₂O₇")
        print(f"  r̄_A = {r_a:.4f} Å  |  r̄_B = {r_b:.4f} Å  |  r_A/r_B = {ratio:.4f}")
        print(f"\n  ► Predicted lattice parameter:  a = {pred:.5f} Å\n")
    else:
        interactive_loop(model, scaler)


if __name__ == '__main__':
    main()
