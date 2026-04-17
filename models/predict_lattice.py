#!/usr/bin/env python3
"""
predict_lattice.py
==================
Interactive predictor for **lattice parameter (Å)** of a pyrochlore oxide.

Usage
-----
  python models/predict_lattice.py
  python models/predict_lattice.py --a "La,Gd,Lu" --b "Ti,Zr"
  python models/predict_lattice.py --target-lattice 10.35          # reverse mode
  python models/predict_lattice.py --target-lattice 10.35 --tol 0.05
  python models/predict_lattice.py --train          # force re-train before predicting

Forward mode  : given A/B composition → predict lattice parameter.
Reverse mode  : given a target lattice parameter → rank dataset compositions
                by closeness and suggest new compositions via grid search.

The script trains automatically on first run and caches the model in
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
from src.data.load_data import get_lattice_dataset, load_combined
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


# ── Reverse prediction ────────────────────────────────────────────────────────

def _reverse_lookup_dataset(target: float, tol: float, model, scaler):
    """
    Search the combined dataset for compositions whose *predicted* lattice
    parameter falls within `tol` Å of `target`.  Also reports the measured
    value when available, so you can see how close the model is.
    """
    df = load_combined(verbose=False)
    df = df[df['Sample A'].notna() & df['Sample B'].notna()].copy()

    results = []
    for _, row in df.iterrows():
        pred = predict_from_composition(
            str(row['Sample A']), str(row['Sample B']), model, scaler)
        diff = abs(pred - target)
        results.append({
            'Sample A':  row['Sample A'],
            'Sample B':  row['Sample B'],
            'Predicted': pred,
            'Measured':  row.get('Lattice Parameter (Angstrom)', np.nan),
            'Diff':      diff,
            'Source':    row.get('data_source', ''),
        })

    res_df = pd.DataFrame(results).sort_values('Diff')
    within = res_df[res_df['Diff'] <= tol]
    return res_df, within


def _reverse_grid_search(target: float, tol: float, model, scaler,
                         top_n: int = 10):
    """
    Enumerate equiatomic combinations of A-site (3–5 elements) and B-site
    (1–2 elements) from the known element tables, predict their lattice
    parameter, and return the closest matches.
    """
    from itertools import combinations

    a_elements = sorted(IONIC_RADII_8.keys())
    b_elements = sorted(IONIC_RADII_6.keys())

    candidates = []

    for n_a in range(3, 6):           # 3-, 4-, 5-element A-site
        for a_combo in combinations(a_elements, n_a):
            for n_b in range(1, 3):   # 1- or 2-element B-site
                for b_combo in combinations(b_elements, n_b):
                    a_str = ','.join(a_combo)
                    b_str = ','.join(b_combo)
                    pred = predict_from_composition(a_str, b_str, model, scaler)
                    diff = abs(pred - target)
                    if diff <= tol * 3:   # coarse pre-filter to save printing
                        candidates.append({
                            'A-site':    a_str,
                            'B-site':    b_str,
                            'Predicted': pred,
                            'Diff':      diff,
                        })

    if not candidates:
        return pd.DataFrame()

    return (pd.DataFrame(candidates)
              .sort_values('Diff')
              .head(top_n)
              .reset_index(drop=True))


def reverse_mode(target: float, tol: float, model, scaler):
    """Full reverse-prediction workflow."""
    print(f"  Target lattice parameter : {target:.4f} Å  (tolerance ± {tol:.4f} Å)")
    print()

    # ── 1. Dataset lookup ──────────────────────────────────────────────────
    print("  ── Dataset compositions ranked by closeness ──────────────────")
    all_res, within = _reverse_lookup_dataset(target, tol, model, scaler)

    if within.empty:
        print(f"  No dataset compositions predicted within ±{tol} Å.")
        print(f"  Showing top-5 closest instead:\n")
        show = all_res.head(5)
    else:
        print(f"  {len(within)} composition(s) within ±{tol} Å:\n")
        show = within

    for _, r in show.iterrows():
        meas_str = (f"  measured={r['Measured']:.5f} Å"
                    if not np.isnan(r['Measured']) else "  (no measured value)")
        flag = " ◄" if abs(r['Predicted'] - target) <= tol else ""
        print(f"  ({r['Sample A']})₂({r['Sample B']})₂O₇")
        print(f"    predicted={r['Predicted']:.5f} Å  |  Δ={r['Diff']:.5f} Å"
              f"{meas_str}  [{r['Source']}]{flag}")
        print()

    # ── 2. Grid search for novel compositions ─────────────────────────────
    print("  ── Novel composition search (grid over all element combos) ──")
    print("  Searching 3–5 A-site + 1–2 B-site combinations …")
    grid = _reverse_grid_search(target, tol, model, scaler, top_n=8)

    if grid.empty:
        print(f"  No novel compositions found within ±{tol*3:.3f} Å of target.")
    else:
        print(f"  Top suggestions (predicted closest to {target:.4f} Å):\n")
        for i, r in grid.iterrows():
            flag = " ◄ within tol" if r['Diff'] <= tol else ""
            print(f"  #{i+1:02d}  ({r['A-site']})₂({r['B-site']})₂O₇")
            print(f"       predicted={r['Predicted']:.5f} Å  |  Δ={r['Diff']:.5f} Å{flag}")
        print()


def interactive_loop(model, scaler):
    """Prompt the user repeatedly until they quit."""
    print("Mode options:")
    print("  [1] Forward  — enter A/B composition → get predicted lattice parameter")
    print("  [2] Reverse  — enter a target lattice parameter → find matching compositions")
    print("Type  'quit'  to exit.\n")

    while True:
        print("─" * 54)
        mode = input("  Mode [1/2]: ").strip()
        if mode.lower() in ('quit', 'q', 'exit'):
            break

        if mode == '2':
            # ── Reverse mode ──────────────────────────────────────────────
            lat_str = input("  Target lattice parameter (Å): ").strip()
            if lat_str.lower() in ('quit', 'q', 'exit'):
                break
            try:
                target = float(lat_str)
            except ValueError:
                print("  Please enter a numeric value.")
                continue
            tol_str = input("  Tolerance (Å) [default 0.03]: ").strip()
            tol = 0.03
            if tol_str:
                try:
                    tol = float(tol_str)
                except ValueError:
                    pass
            print()
            reverse_mode(target, tol, model, scaler)

        else:
            # ── Forward mode ──────────────────────────────────────────────
            a = input("  A-site elements (e.g. La,Gd,Lu): ").strip()
            if a.lower() in ('quit', 'q', 'exit'):
                break
            b = input("  B-site elements (e.g. Ti,Zr):    ").strip()
            if b.lower() in ('quit', 'q', 'exit'):
                break

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
        description='Predict pyrochlore lattice parameter from A/B-site composition, '
                    'or reverse-search compositions for a target lattice parameter.')
    parser.add_argument('--a', type=str, help='A-site elements (comma-separated)')
    parser.add_argument('--b', type=str, help='B-site elements (comma-separated)')
    parser.add_argument('--target-lattice', type=float, default=None,
                        metavar='Å',
                        help='Reverse mode: target lattice parameter in Å. '
                             'Ranks known & novel compositions by closeness.')
    parser.add_argument('--tol', type=float, default=0.03,
                        metavar='Å',
                        help='Tolerance for reverse mode (default: 0.03 Å)')
    parser.add_argument('--train', action='store_true',
                        help='Force re-train before predicting')
    args = parser.parse_args()

    _print_banner()
    model, scaler = _ensure_model(force_train=args.train)

    if args.target_lattice is not None:
        # ── Reverse mode (CLI) ────────────────────────────────────────────
        print()
        reverse_mode(args.target_lattice, args.tol, model, scaler)

    elif args.a and args.b:
        # ── Forward mode (CLI) ────────────────────────────────────────────
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
        # ── Interactive mode ──────────────────────────────────────────────
        interactive_loop(model, scaler)


if __name__ == '__main__':
    main()