#!/usr/bin/env python3
"""
predict_thermal.py
==================
Interactive predictor for **thermal conductivity (W/m·K)** of a pyrochlore oxide.

Usage
-----
  python models/predict_thermal.py
  python models/predict_thermal.py --a "Sm,Eu,Gd,Tb,Dy" --b "Ti" --lattice 10.18
  python models/predict_thermal.py --target-thermal 0.85         # reverse mode
  python models/predict_thermal.py --target-thermal 0.85 --tol 0.10
  python models/predict_thermal.py --train       # force re-train before predicting

Forward mode  : given A/B composition (+ optional lattice param) → predict κ.
Reverse mode  : given a target κ value → rank dataset compositions by closeness
                and suggest novel compositions via grid search.

If no lattice parameter is given in forward mode the script auto-predicts it
using the cached lattice model.

Model is cached in:  models/thermal_cond/thermal_cond_model.pkl
"""

import argparse
import sys
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')

_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
sys.path.insert(0, str(_PROJECT))

from src.features.build_features import (
    build_features_for_row, FEATURE_COLS, THERMAL_EXTRA_FEATURES,
    IONIC_RADII_8, IONIC_RADII_6,
    parse_composition, mean_radius, a_site_b_site_radius_ratio,
    configurational_entropy,
)
from src.data.load_data import get_thermal_dataset, load_combined
from src.build_models.train_model import train_and_evaluate, load_model

import pandas as pd

MODEL_DIR_THERMAL  = _PROJECT / 'models' / 'thermal_cond'
MODEL_DIR_LATTICE  = _PROJECT / 'models' / 'lattice_param'
TASK = 'thermal_cond'
TASK_LAT = 'lattice_param'


# ── helpers ───────────────────────────────────────────────────────────────────

def _print_banner():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   HEC Pyrochlore — Thermal Conductivity Predictor    ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def _check_elements(comp_str: str, radii_table: dict, site: str):
    comp = parse_composition(comp_str)
    unknown = [e for e in comp if e not in radii_table]
    if unknown:
        print(f"  WARNING: Unknown {site}-site element(s): {unknown}")
    return comp


def _ensure_thermal_model(force_train: bool = False):
    model_file = MODEL_DIR_THERMAL / f'{TASK}_model.pkl'
    if not model_file.exists() or force_train:
        print("[train] Training thermal conductivity model …")
        X, y, feat_names = get_thermal_dataset(verbose=True)
        results = train_and_evaluate(
            X, y, feat_names,
            task_name=TASK,
            save_dir=MODEL_DIR_THERMAL,
            verbose=True,
        )
        _, _, meta_dict = load_model(TASK, MODEL_DIR_THERMAL)
        return results['best_model'], results['scaler'], meta_dict['feature_names']
    else:
        model, scaler, meta = load_model(TASK, MODEL_DIR_THERMAL)
        print(f"[load] Loaded cached thermal model  ({meta['best_model']})")
        return model, scaler, meta['feature_names']


def _get_lattice_prediction(a_str: str, b_str: str) -> float:
    """Use the cached lattice model to predict lattice parameter."""
    lat_file = MODEL_DIR_LATTICE / f'{TASK_LAT}_model.pkl'
    if not lat_file.exists():
        from src.data.load_data import get_lattice_dataset
        X, y, feat_names = get_lattice_dataset(verbose=False)
        train_and_evaluate(X, y, feat_names, task_name=TASK_LAT,
                           save_dir=MODEL_DIR_LATTICE, verbose=False)
    lat_model, lat_scaler, _ = load_model(TASK_LAT, MODEL_DIR_LATTICE)

    row = pd.Series({'Sample A': a_str, 'Sample B': b_str,
                     'Lattice Parameter (Angstrom)': np.nan,
                     'TPS Cond W/m/K': np.nan})
    feats = build_features_for_row(row)
    X_raw = np.array([feats.get(c, np.nan) for c in FEATURE_COLS], dtype=float)
    X_s = lat_scaler.transform(X_raw.reshape(1, -1))
    return float(lat_model.predict(X_s)[0])


def predict_thermal(a_str: str, b_str: str,
                    lattice_a: float,
                    model, scaler,
                    feat_names) -> float:
    row = pd.Series({
        'Sample A': a_str,
        'Sample B': b_str,
        'Lattice Parameter (Angstrom)': lattice_a,
        'TPS Cond W/m/K': np.nan,
    })
    feats = build_features_for_row(row)
    X_raw = np.array([feats.get(c, np.nan) for c in feat_names], dtype=float)

    if np.any(np.isnan(X_raw)):
        missing = [feat_names[i] for i, v in enumerate(X_raw) if np.isnan(v)]
        print(f"  WARNING: NaN features — {missing[:5]}")

    X_s = scaler.transform(X_raw.reshape(1, -1))
    return float(model.predict(X_s)[0])


def _describe_composition(a_str: str, b_str: str, lattice_a: float):
    a_comp = parse_composition(a_str)
    b_comp = parse_composition(b_str)
    r_a = mean_radius(a_comp, IONIC_RADII_8)
    r_b = mean_radius(b_comp, IONIC_RADII_6)
    ratio = a_site_b_site_radius_ratio(r_a, r_b)
    s_a = configurational_entropy(a_comp)
    n_a = len(a_comp)
    return r_a, r_b, ratio, s_a, n_a


# ── Reverse prediction ────────────────────────────────────────────────────────

def _reverse_lookup_dataset(target: float, tol: float, model, scaler, feat_names):
    """
    Search the combined dataset for compositions whose *predicted* thermal
    conductivity falls within `tol` of `target`. Also shows measured values.
    """
    df = load_combined(verbose=False)
    df = df[df['Sample A'].notna() & df['Sample B'].notna()].copy()

    results = []
    for _, row in df.iterrows():
        a_str = str(row['Sample A'])
        b_str = str(row['Sample B'])
        # Auto-predict lattice if missing
        lattice_a = row.get('Lattice Parameter (Angstrom)', np.nan)
        if pd.isna(lattice_a):
            try:
                lattice_a = _get_lattice_prediction(a_str, b_str)
            except Exception:
                lattice_a = np.nan

        try:
            pred = predict_thermal(a_str, b_str, lattice_a, model, scaler, feat_names)
        except Exception:
            continue

        diff = abs(pred - target)
        results.append({
            'Sample A':  a_str,
            'Sample B':  b_str,
            'Predicted': pred,
            'Measured':  row.get('TPS Cond W/m/K', np.nan),
            'Lattice':   lattice_a,
            'Diff':      diff,
            'Source':    row.get('data_source', ''),
        })

    res_df = pd.DataFrame(results).sort_values('Diff')
    within = res_df[res_df['Diff'] <= tol]
    return res_df, within


def _reverse_grid_search(target: float, tol: float, model, scaler, feat_names,
                         top_n: int = 10):
    """
    Enumerate equiatomic A/B combinations, predict thermal conductivity for each,
    and return the closest matches to the target.
    """
    from itertools import combinations

    a_elements = sorted(IONIC_RADII_8.keys())
    b_elements = sorted(IONIC_RADII_6.keys())

    candidates = []
    for n_a in range(3, 6):
        for a_combo in combinations(a_elements, n_a):
            for n_b in range(1, 3):
                for b_combo in combinations(b_elements, n_b):
                    a_str = ','.join(a_combo)
                    b_str = ','.join(b_combo)
                    try:
                        lattice_a = _get_lattice_prediction(a_str, b_str)
                        pred = predict_thermal(a_str, b_str, lattice_a,
                                              model, scaler, feat_names)
                    except Exception:
                        continue
                    diff = abs(pred - target)
                    if diff <= tol * 4:
                        candidates.append({
                            'A-site':    a_str,
                            'B-site':    b_str,
                            'Predicted κ': pred,
                            'Lattice a': lattice_a,
                            'Diff':      diff,
                        })

    if not candidates:
        return pd.DataFrame()

    return (pd.DataFrame(candidates)
              .sort_values('Diff')
              .head(top_n)
              .reset_index(drop=True))


def reverse_mode(target: float, tol: float, model, scaler, feat_names):
    """Full reverse-prediction workflow for thermal conductivity."""
    print(f"  Target thermal conductivity : {target:.4f} W/m·K  (tolerance ± {tol:.4f})")
    print()

    # ── 1. Dataset lookup ──────────────────────────────────────────────────
    print("  ── Dataset compositions ranked by closeness ──────────────────")
    all_res, within = _reverse_lookup_dataset(target, tol, model, scaler, feat_names)

    if within.empty:
        print(f"  No dataset compositions predicted within ±{tol} W/m·K.")
        print("  Showing top-5 closest instead:\n")
        show = all_res.head(5)
    else:
        print(f"  {len(within)} composition(s) within ±{tol} W/m·K:\n")
        show = within

    for _, r in show.iterrows():
        meas_str = (f"  measured={r['Measured']:.4f} W/m·K"
                    if not np.isnan(r['Measured']) else "  (no measured κ)")
        flag = " ◄" if abs(r['Predicted'] - target) <= tol else ""
        print(f"  ({r['Sample A']})₂({r['Sample B']})₂O₇")
        print(f"    predicted κ={r['Predicted']:.4f}  |  Δ={r['Diff']:.4f}"
              f"  |  lattice={r['Lattice']:.4f} Å{meas_str}  [{r['Source']}]{flag}")
        print()

    # ── 2. Grid search for novel compositions ─────────────────────────────
    print("  ── Novel composition search (grid over all element combos) ──")
    print("  Searching 3–5 A-site + 1–2 B-site combinations …")
    print("  (lattice parameter auto-predicted for each composition)\n")
    grid = _reverse_grid_search(target, tol, model, scaler, feat_names, top_n=8)

    if grid.empty:
        print(f"  No novel compositions found within ±{tol*4:.3f} W/m·K of target.")
    else:
        print(f"  Top suggestions (predicted closest to {target:.4f} W/m·K):\n")
        for i, r in grid.iterrows():
            flag = " ◄ within tol" if r['Diff'] <= tol else ""
            print(f"  #{i+1:02d}  ({r['A-site']})₂({r['B-site']})₂O₇")
            print(f"       pred κ={r['Predicted κ']:.4f} W/m·K  |  "
                  f"Δ={r['Diff']:.4f}  |  lattice={r['Lattice a']:.4f} Å{flag}")
        print()


def interactive_loop(model, scaler, feat_names):
    print("Mode options:")
    print("  [1] Forward  — enter A/B composition → get predicted κ")
    print("  [2] Reverse  — enter a target κ → find matching compositions")
    print("Type  'quit'  to exit.\n")

    while True:
        print("─" * 54)
        mode = input("  Mode [1/2]: ").strip()
        if mode.lower() in ('quit', 'q', 'exit'):
            break

        if mode == '2':
            # ── Reverse mode ──────────────────────────────────────────────
            tc_str = input("  Target thermal conductivity (W/m·K): ").strip()
            if tc_str.lower() in ('quit', 'q', 'exit'):
                break
            try:
                target = float(tc_str)
            except ValueError:
                print("  Please enter a numeric value.")
                continue
            tol_str = input("  Tolerance (W/m·K) [default 0.10]: ").strip()
            tol = 0.10
            if tol_str:
                try:
                    tol = float(tol_str)
                except ValueError:
                    pass
            print()
            reverse_mode(target, tol, model, scaler, feat_names)

        else:
            # ── Forward mode ──────────────────────────────────────────────
            a = input("  A-site elements    : ").strip()
            if a.lower() in ('quit', 'q', 'exit'):
                break
            b = input("  B-site elements    : ").strip()
            if b.lower() in ('quit', 'q', 'exit'):
                break
            lat_in = input("  Lattice param (Å)  [Enter = auto]: ").strip()

            if lat_in == '':
                lattice_a = _get_lattice_prediction(a, b)
                print(f"  (Auto-predicted lattice parameter: {lattice_a:.5f} Å)")
            else:
                try:
                    lattice_a = float(lat_in)
                except ValueError:
                    print("  Invalid lattice parameter — please enter a number.")
                    continue

            r_a, r_b, ratio, s_a, n_a = _describe_composition(a, b, lattice_a)
            pred = predict_thermal(a, b, lattice_a, model, scaler, feat_names)

            print()
            print(f"  Composition     :  ({a})₂({b})₂O₇")
            print(f"  A-site elements :  {n_a}  |  S_config = {s_a:.4f} J/(mol·K)")
            print(f"  r̄_A = {r_a:.4f} Å  |  r̄_B = {r_b:.4f} Å  |  r_A/r_B = {ratio:.4f}")
            print(f"  Lattice a       :  {lattice_a:.5f} Å")
            print()
            print(f"  ► Predicted thermal conductivity:  κ = {pred:.4f} W/m·K")
            print()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Predict pyrochlore thermal conductivity from composition, '
                    'or reverse-search compositions for a target κ value.')
    parser.add_argument('--a',       type=str, help='A-site elements (comma-separated)')
    parser.add_argument('--b',       type=str, help='B-site elements (comma-separated)')
    parser.add_argument('--lattice', type=float, default=None,
                        help='Lattice parameter in Å (optional; auto-predicted if omitted)')
    parser.add_argument('--target-thermal', type=float, default=None,
                        metavar='κ',
                        help='Reverse mode: target thermal conductivity (W/m·K). '
                             'Ranks known & novel compositions by closeness.')
    parser.add_argument('--tol', type=float, default=0.10,
                        metavar='W/m·K',
                        help='Tolerance for reverse mode (default: 0.10 W/m·K)')
    parser.add_argument('--train', action='store_true',
                        help='Force re-train before predicting')
    args = parser.parse_args()

    _print_banner()
    model, scaler, feat_names = _ensure_thermal_model(force_train=args.train)

    if args.target_thermal is not None:
        # ── Reverse mode (CLI) ────────────────────────────────────────────
        print()
        reverse_mode(args.target_thermal, args.tol, model, scaler, feat_names)

    elif args.a and args.b:
        # ── Forward mode (CLI) ────────────────────────────────────────────
        lattice_a = args.lattice
        if lattice_a is None:
            lattice_a = _get_lattice_prediction(args.a, args.b)
            print(f"  (Auto-predicted lattice parameter: {lattice_a:.5f} Å)")

        r_a, r_b, ratio, s_a, n_a = _describe_composition(args.a, args.b, lattice_a)
        pred = predict_thermal(args.a, args.b, lattice_a, model, scaler, feat_names)

        print(f"  Composition :  ({args.a})₂({args.b})₂O₇")
        print(f"  S_config_A  :  {s_a:.4f} J/(mol·K)  |  n_A = {n_a}")
        print(f"  r̄_A = {r_a:.4f} Å  |  r̄_B = {r_b:.4f} Å  |  r_A/r_B = {ratio:.4f}")
        print(f"  Lattice a   :  {lattice_a:.5f} Å")
        print(f"\n  ► Predicted thermal conductivity:  κ = {pred:.4f} W/m·K\n")

    else:
        interactive_loop(model, scaler, feat_names)


if __name__ == '__main__':
    main()