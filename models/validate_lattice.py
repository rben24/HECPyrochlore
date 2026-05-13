#!/usr/bin/env python3
"""
validate_lattice.py
===================
Driver script — trains and validates the **lattice parameter** model.

Now uses the expanded dataset that includes ICSD literature entries
(pristine single-element and multi-element high-entropy pyrochlores).
Non-pyrochlore entries are automatically excluded by the data loader.

Outputs
-------
  • Console: CV scores, test-set metrics, compound-type breakdown, top features
  • models/lattice_param/lattice_parity.png
  • models/lattice_param/lattice_cv_comparison.png
  • models/lattice_param/lattice_feature_importance.png
  • models/lattice_param/lattice_validation_report.txt

Usage
-----
  python models/validate_lattice.py
  python models/validate_lattice.py --splits 10
  python models/validate_lattice.py --types pristine          # pristine only
  python models/validate_lattice.py --types high_entropy      # HEC only
  python models/validate_lattice.py --types pristine high_entropy  # both (default)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')

_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
sys.path.insert(0, str(_PROJECT))

from src.data.load_data import get_lattice_dataset
from src.build_models.train_model import (
    train_and_evaluate, plot_feature_importance,
    plot_parity, plot_cv_comparison,
)

TASK     = 'lattice_param'
SAVE_DIR = _PROJECT / 'models' / TASK


def main():
    parser = argparse.ArgumentParser(
        description='Validate lattice parameter ML model.')
    parser.add_argument('--splits', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--types', nargs='+',
                        default=['pristine', 'high_entropy'],
                        choices=['pristine', 'high_entropy'],
                        help='Compound types to include (default: both)')
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Lattice Parameter — Validation & Feature Report   ║")
    print("╚══════════════════════════════════════════════════════╝")

    X, y, feat_names = get_lattice_dataset(
        verbose=True, compound_types=args.types)

    print(f"\n  Samples  : {len(y)}")
    print(f"  Features : {len(feat_names)}")
    print(f"  Target   : Lattice Parameter (Å)  "
          f"range [{y.min():.4f}, {y.max():.4f}]")

    results = train_and_evaluate(
        X, y, feat_names,
        task_name=TASK,
        n_splits=args.splits,
        save_dir=SAVE_DIR,
        verbose=True,
        compound_types=args.types,
    )

    print("\n[plots] Generating figures …")
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
        top_n=min(20, len(feat_names)),
        save_path=SAVE_DIR / 'lattice_feature_importance.png',
    )

    _write_report(results, feat_names, args.splits, args.types)
    print("\n[done] All outputs written to:", SAVE_DIR)


def _write_report(results, feat_names, n_splits, compound_types):
    report_path = SAVE_DIR / 'lattice_validation_report.txt'
    fi = results['fi_df']
    cv = results['cv_results']

    lines = [
        "=" * 60,
        "  LATTICE PARAMETER — VALIDATION REPORT",
        "=" * 60,
        "",
        f"  Best model      : {results['best_name']}",
        f"  CV folds        : {n_splits}",
        f"  Features        : {len(feat_names)}",
        f"  Compound types  : {', '.join(compound_types)}",
        f"  Train samples   : {results.get('n_train', '?')}",
        f"  Test samples    : {results.get('n_test', '?')}",
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
        "── Top-20 Features by Permutation Importance ─────────────────",
        f"  {'Rank':<5} {'Feature':<35} {'Perm Imp':>10}  {'±':>2}  "
        f"{'Std':>8}  {'Tree MDI':>10}",
        "  " + "-" * 72,
    ]
    for rank, row in fi.head(20).iterrows():
        tree_str = (f"{row['tree_importance']:.6f}"
                    if not np.isnan(row['tree_importance']) else "   N/A   ")
        lines.append(
            f"  {rank+1:<5} {row['feature']:<35} {row['perm_importance']:>10.6f}  ±  "
            f"{row['perm_std']:>8.6f}  {tree_str:>10}"
        )

    lines += [
        "",
        "── Feature-Group Analysis ───────────────────────────────────",
    ]
    groups = {
        'A-site':          [f for f in feat_names if 'a_site' in f],
        'B-site':          [f for f in feat_names if 'b_site' in f],
        'Cross-site':      [f for f in feat_names if f in (
                            'a_b_radius_ratio', 'total_entropy', 'total_delta',
                            'phonon_scattering_factor', 'n_total_elements',
                            'site_asymmetry', 'en_site_contrast',
                            'mass_site_contrast')],
    }
    for grp, cols in groups.items():
        sub = fi[fi['feature'].isin(cols)]
        if len(sub):
            mean_imp = sub['perm_importance'].mean()
            top_feat = sub.iloc[0]['feature']
            lines.append(
                f"  {grp:<12}: mean perm-imp = {mean_imp:.5f}  "
                f"| top feature = {top_feat}"
            )

    lines += ["", "=" * 60]
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {report_path}")
    print('\n' + '\n'.join(lines))


if __name__ == '__main__':
    main()