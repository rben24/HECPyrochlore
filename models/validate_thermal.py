#!/usr/bin/env python3
"""
validate_thermal.py
===================
Driver script — trains and validates the **thermal conductivity** model.

Outputs
-------
  • Console: CV scores, test-set metrics, top features
  • models/thermal_cond/thermal_parity.png
  • models/thermal_cond/thermal_cv_comparison.png
  • models/thermal_cond/thermal_feature_importance.png
  • models/thermal_cond/thermal_validation_report.txt

Usage
-----
  python models/validate_thermal.py
  python models/validate_thermal.py --splits 5
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

from src.data.load_data import get_thermal_dataset
from src.build_models.train_model import (
    train_and_evaluate, plot_feature_importance,
    plot_parity, plot_cv_comparison,
)

TASK = 'thermal_cond'
SAVE_DIR = _PROJECT / 'models' / TASK


def main():
    parser = argparse.ArgumentParser(
        description='Validate thermal conductivity ML model.')
    parser.add_argument('--splits', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Thermal Conductivity — Validation & Feature Report  ║")
    print("╚══════════════════════════════════════════════════════╝")

    # ── Load data ─────────────────────────────────────────────────────────────
    X, y, feat_names = get_thermal_dataset(verbose=True)
    print(f"\n  Samples : {len(y)}")
    print(f"  Features: {len(feat_names)}")
    print(f"  Target  : TPS Cond W/m·K  range [{y.min():.4f}, {y.max():.4f}]")

    # ── Train & evaluate ──────────────────────────────────────────────────────
    results = train_and_evaluate(
        X, y, feat_names,
        task_name=TASK,
        n_splits=args.splits,
        save_dir=SAVE_DIR,
        verbose=True,
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[plots] Generating figures …")
    plot_parity(
        results['y_test'], results['y_pred'],
        title='Thermal Conductivity — Parity Plot',
        ylabel='κ (W/m·K)',
        save_path=SAVE_DIR / 'thermal_parity.png',
    )
    plot_cv_comparison(
        results['cv_results'],
        title='Thermal Conductivity — Model Comparison (CV)',
        save_path=SAVE_DIR / 'thermal_cv_comparison.png',
    )
    plot_feature_importance(
        results['fi_df'],
        title=f"Thermal Conductivity — Feature Importance ({results['best_name']})",
        top_n=min(15, len(feat_names)),
        save_path=SAVE_DIR / 'thermal_feature_importance.png',
    )

    # ── Text report ───────────────────────────────────────────────────────────
    _write_report(results, feat_names, args.splits)

    print("\n[done] All outputs written to:", SAVE_DIR)


def _write_report(results, feat_names, n_splits):
    report_path = SAVE_DIR / 'thermal_validation_report.txt'
    fi = results['fi_df']
    cv = results['cv_results']

    lines = [
        "=" * 60,
        "  THERMAL CONDUCTIVITY — VALIDATION REPORT",
        "=" * 60,
        "",
        f"  Best model   : {results['best_name']}",
        f"  CV folds     : {n_splits}",
        f"  Features     : {len(feat_names)}",
        "",
        "── Test-set Metrics ────────────────────────────────────────",
    ]
    for k, v in results['test_metrics'].items():
        lines.append(f"  {k:6s}: {v:.5f}")

    lines += [
        "",
        "── Cross-Validation Summary (all models) ───────────────────",
    ]
    for name, cv_res in cv.items():
        r2   = cv_res['test_r2']
        rmse = -cv_res['test_neg_root_mean_squared_error']
        mae  = -cv_res['test_neg_mean_absolute_error']
        lines += [
            f"  {name}",
            f"    R²   : {r2.mean():.4f} ± {r2.std():.4f}",
            f"    RMSE : {rmse.mean():.5f} ± {rmse.std():.5f}",
            f"    MAE  : {mae.mean():.5f} ± {mae.std():.5f}",
        ]

    lines += [
        "",
        "── Top-15 Features by Permutation Importance ─────────────────",
        f"  {'Rank':<5} {'Feature':<35} {'Perm Imp':>10}  {'±':>2}  {'Std':>8}  {'Tree MDI':>10}",
        "  " + "-" * 72,
    ]
    for rank, row in fi.head(15).iterrows():
        tree_str = (f"{row['tree_importance']:.6f}"
                    if not np.isnan(row['tree_importance']) else "   N/A   ")
        lines.append(
            f"  {rank+1:<5} {row['feature']:<35} {row['perm_importance']:>10.6f}  ±  "
            f"{row['perm_std']:>8.6f}  {tree_str:>10}"
        )

    lines += [
        "",
        "── Physical Interpretation ──────────────────────────────────",
        "  Features are ranked by how much they reduce model R²",
        "  when randomly shuffled (permutation importance).",
        "",
        "  Key feature groups:",
    ]

    groups = {
        'A-site compositional': [f for f in feat_names if 'a_site' in f],
        'B-site compositional': [f for f in feat_names if 'b_site' in f],
        'Cross-site / lattice': [f for f in feat_names if f in
                                  ('a_b_radius_ratio', 'total_entropy',
                                   'total_delta', 'phonon_scattering_factor',
                                   'lattice_parameter', 'lattice_volume',
                                   'density_theoretical')],
    }
    for grp, cols in groups.items():
        sub = fi[fi['feature'].isin(cols)]
        if len(sub):
            mean_imp = sub['perm_importance'].mean()
            top_feat = sub.iloc[0]['feature']
            lines.append(f"  {grp:<25}: mean = {mean_imp:.5f}  | top = {top_feat}")

    lines += ["", "=" * 60]

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {report_path}")
    print('\n' + '\n'.join(lines))


if __name__ == '__main__':
    main()
