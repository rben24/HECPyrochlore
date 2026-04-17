#!/usr/bin/env python3
"""
validate_lattice.py
===================
Driver script — trains, validates, and diagnoses the **lattice parameter** model.

Outputs
-------
  models/lattice_param/
    lattice_parity.png                  — predicted vs actual
    lattice_cv_comparison.png           — model comparison
    lattice_feature_importance.png      — permutation + tree MDI
    lattice_residuals.png               — 4-panel residual diagnostics
    lattice_influence.png               — Cook's D + leverage plot
    lattice_target_distribution.png     — histogram, KDE, box-plot
    lattice_feature_vs_target.png       — top-6 features vs target
    lattice_correlation_heatmap.png     — feature–feature Pearson r
    lattice_validation_report.txt       — full text report

Usage
-----
  python models/validate_lattice.py
  python models/validate_lattice.py --splits 10
  python models/validate_lattice.py --no-validation   # skip data-validation plots
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

from src.data.load_data import get_lattice_dataset, load_combined, clean_and_engineer
from src.features.build_features import FEATURE_COLS
from src.build_models.train_model import (
    train_and_evaluate, plot_feature_importance,
    plot_parity, plot_cv_comparison,
)
from data_validation import run_data_validation   # same directory as this script

import pandas as pd

TASK     = 'lattice_param'
SAVE_DIR = _PROJECT / 'models' / TASK


def main():
    parser = argparse.ArgumentParser(
        description='Validate lattice parameter ML model.')
    parser.add_argument('--splits', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--no-validation', action='store_true',
                        help='Skip data-validation diagnostics (faster)')
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Lattice Parameter — Validation & Feature Report   ║")
    print("╚══════════════════════════════════════════════════════╝")

    # ── Load data ─────────────────────────────────────────────────────────────
    X, y, feat_names = get_lattice_dataset(verbose=True)
    print(f"\n  Samples : {len(y)}")
    print(f"  Features: {len(feat_names)}")
    print(f"  Target  : Lattice Parameter (Å)  "
          f"range [{y.min():.4f}, {y.max():.4f}]")

    # ── Train & evaluate ──────────────────────────────────────────────────────
    results = train_and_evaluate(
        X, y, feat_names,
        task_name=TASK,
        n_splits=args.splits,
        save_dir=SAVE_DIR,
        verbose=True,
    )

    # ── Standard plots ────────────────────────────────────────────────────────
    print("\n[plots] Generating standard figures …")
    plot_parity(
        results['y_test'], results['y_pred'],
        title='Lattice Parameter — Parity Plot',
        ylabel='Lattice Parameter (Å)',
        save_path=SAVE_DIR / 'lattice_parity.png',
    )
    plot_cv_comparison(
        results['cv_results'],
        title='Lattice Parameter — Model Comparison (CV)',
        save_path=SAVE_DIR / 'lattice_cv_comparison.png',
    )
    plot_feature_importance(
        results['fi_df'],
        title=f"Lattice Parameter — Feature Importance ({results['best_name']})",
        top_n=min(15, len(feat_names)),
        save_path=SAVE_DIR / 'lattice_feature_importance.png',
    )

    # ── Text report ───────────────────────────────────────────────────────────
    _write_report(results, feat_names, args.splits)

    # ── Data validation & diagnostic plots ────────────────────────────────────
    if not args.no_validation:
        print("\n[validation] Running data validation & diagnostic plots …")
        # Build a DataFrame of the full training data for validation
        df_full = _build_full_df(feat_names)
        # Sample labels for outlier annotation (composition strings)
        sample_labels = _get_sample_labels()

        run_data_validation(
            df=df_full,
            feat_cols=feat_names,
            target_col='Lattice Parameter (Angstrom)',
            task_label='Lattice Parameter',
            save_dir=SAVE_DIR,
            y_true=results['y_test'],
            y_pred=results['y_pred'],
            sample_labels=sample_labels,
        )

    print("\n[done] All outputs written to:", SAVE_DIR)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_full_df(feat_names):
    """Return the full engineered DataFrame used for the lattice dataset."""
    df = clean_and_engineer(load_combined(verbose=False), verbose=False)
    target = 'Lattice Parameter (Angstrom)'
    df_sub = df.dropna(subset=[target])
    df_sub = df_sub[df_sub['Sample A'].notna() & df_sub['Sample B'].notna()]
    feat_mask = df_sub[feat_names].notna().all(axis=1)
    return df_sub[feat_mask].reset_index(drop=True)


def _get_sample_labels():
    """Return composition ID strings for each row in the lattice dataset."""
    df = clean_and_engineer(load_combined(verbose=False), verbose=False)
    target = 'Lattice Parameter (Angstrom)'
    df_sub = df.dropna(subset=[target])
    df_sub = df_sub[df_sub['Sample A'].notna() & df_sub['Sample B'].notna()]
    feat_mask = df_sub[FEATURE_COLS].notna().all(axis=1)
    df_sub = df_sub[feat_mask].reset_index(drop=True)
    if 'Composition' in df_sub.columns:
        return df_sub['Composition'].fillna('').tolist()
    return [f"({a})₂({b})₂O₇"
            for a, b in zip(df_sub.get('Sample A', []), df_sub.get('Sample B', []))]


def _write_report(results, feat_names, n_splits):
    report_path = SAVE_DIR / 'lattice_validation_report.txt'
    fi = results['fi_df']
    cv = results['cv_results']

    lines = [
        "=" * 60,
        "  LATTICE PARAMETER — VALIDATION REPORT",
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
        "── Top-15 Features by Permutation Importance ────────────────",
        f"  {'Rank':<5} {'Feature':<35} {'Perm Imp':>10}  {'±':>2}  {'Std':>8}  {'Tree MDI':>10}",
        "  " + "-" * 72,
    ]
    for rank, row in fi.head(15).iterrows():
        tree_str = f"{row['tree_importance']:.6f}" if not np.isnan(row['tree_importance']) else "   N/A   "
        lines.append(
            f"  {rank+1:<5} {row['feature']:<35} {row['perm_importance']:>10.6f}  ±  "
            f"{row['perm_std']:>8.6f}  {tree_str:>10}"
        )

    lines += [
        "",
        "── Feature-Group Analysis ───────────────────────────────────",
    ]
    groups = {
        'A-site':    [f for f in feat_names if 'a_site' in f],
        'B-site':    [f for f in feat_names if 'b_site' in f],
        'Cross-site': [f for f in feat_names if f in
                       ('a_b_radius_ratio', 'total_entropy', 'total_delta',
                        'phonon_scattering_factor')],
    }
    for grp, cols in groups.items():
        sub = fi[fi['feature'].isin(cols)]
        if len(sub):
            mean_imp = sub['perm_importance'].mean()
            top_feat = sub.iloc[0]['feature'] if len(sub) > 0 else 'N/A'
            lines.append(f"  {grp:<12}: mean perm-imp = {mean_imp:.5f}  "
                         f"| top feature = {top_feat}")

    lines += [
        "",
        "── Diagnostic Plots Generated ───────────────────────────────",
        "  lattice_parity.png                — predicted vs actual",
        "  lattice_cv_comparison.png         — model CV comparison",
        "  lattice_feature_importance.png    — permutation + MDI importance",
        "  lattice_residuals.png             — 4-panel residual diagnostics",
        "  lattice_influence.png             — Cook's D + leverage",
        "  lattice_target_distribution.png   — histogram + KDE + box-plot",
        "  lattice_feature_vs_target.png     — top-6 features vs target",
        "  lattice_correlation_heatmap.png   — feature–feature Pearson r",
        "",
        "=" * 60,
    ]

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {report_path}")
    print('\n' + '\n'.join(lines))


if __name__ == '__main__':
    main()