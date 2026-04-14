#!/usr/bin/env python3
"""
predict_thermal.py
==================
Interactive predictor for **thermal conductivity (W/m·K)** of a pyrochlore oxide.

Usage
-----
  python models/predict_thermal.py
  python models/predict_thermal.py --a "Sm,Eu,Gd,Tb,Dy" --b "Ti" --lattice 10.18
  python models/predict_thermal.py --train       # force re-train before predicting

The model uses both compositional features AND lattice parameter (if provided).
If no lattice parameter is given the script will first predict it with the
lattice model, then use that value as a feature.

Model is cached in:  models/thermal_cond/thermal_cond_model.pkl
"""

import argparse
import sys
import warnings
from pathlib import Path
import numpy as np
import pickle, json

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
from src.data.load_data import get_thermal_dataset
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
    print("║   HEC Pyrochlore — Thermal Conductivity Predictor   ║")
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
        # quick train
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


def interactive_loop(model, scaler, feat_names):
    print("Enter compositions as comma-separated element symbols.")
    print("  A-site: rare-earth cations  (e.g.  Sm,Eu,Gd,Tb,Dy)")
    print("  B-site: transition metal(s) (e.g.  Ti  or  Ti,Zr)")
    print("  Lattice parameter: optional; press Enter to auto-predict it.")
    print("Type  'quit'  to exit.\n")

    while True:
        print("─" * 54)
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
        description='Predict pyrochlore thermal conductivity from composition.')
    parser.add_argument('--a',       type=str, help='A-site elements (comma-separated)')
    parser.add_argument('--b',       type=str, help='B-site elements (comma-separated)')
    parser.add_argument('--lattice', type=float, default=None,
                        help='Lattice parameter in Å (optional; auto-predicted if omitted)')
    parser.add_argument('--train', action='store_true',
                        help='Force re-train before predicting')
    args = parser.parse_args()

    _print_banner()
    model, scaler, feat_names = _ensure_thermal_model(force_train=args.train)

    if args.a and args.b:
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
