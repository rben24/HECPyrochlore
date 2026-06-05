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
  python models/validate_rom_latt.py
  python models/validate_rom_latt.py --splits 10
  python models/validate_rom_latt.py --types pristine          # pristine only
  python models/validate_rom_latt.py --types high_entropy      # HEC only
  python models/validate_rom_latt.py --types pristine high_entropy  # both (default)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from time import sleep

import numpy as np

warnings.filterwarnings('ignore')

_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
sys.path.insert(0, str(_PROJECT))

from src.data.load_data import get_lattice_rom_dataset, HIGH_ENTROPY
from src.build_models.train_model import (
    train_and_evaluate, plot_feature_importance,
    plot_parity, plot_cv_comparison, plot_r2_vs_cv_folds,
    plot_r2_vs_feature_count,
)
from src.globals import HIGH_ENTROPY

TASK     = 'lattice_param_rom'
SAVE_DIR = _PROJECT / 'models' / TASK


def main():
    parser = argparse.ArgumentParser(
        description='Validate lattice parameter ML model.')
    parser.add_argument('--splits', type=int, default=6,
                        help='Number of CV folds (default: 6)')
    # parser.add_argument('--types', nargs='+',
    #                     default=['pristine', 'high_entropy'],
    #                     choices=['pristine', 'high_entropy'],
    #                     help='Compound types to include (default: both)')
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Lattice Parameter — Validation & Feature Report    ║")
    print("╚══════════════════════════════════════════════════════╝")

    X, y, feat_names = get_lattice_rom_dataset(verbose=True)

    print(f"\n  Samples  : {len(y)}")
    print(f"  Features : {len(feat_names)}")
    print(f"  Target   : Lattice Parameter (Å)  "
          f"range [{y.min():.4f}, {y.max():.4f}]")

    '''
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    # Convert to DataFrame for easier labeling
    X_df = pd.DataFrame(X, columns=feat_names)
    X_df['latt param'] = y

    # Set style
    sns.set_theme(style='whitegrid')
    sns.pairplot(X_df, diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'sns2.png')
    plt.show()

    # 1. Check distributions and outliers
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(X_df, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distributions')
    plt.show()

    sns.boxplot(data=X_df, ax=axes[0, 1])
    axes[0, 1].set_title('Outliers & Quartiles')
    axes[0, 1].tick_params(axis='x', rotation=45)
    plt.show()

    # 2. Check correlations
    corr = X_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[1, 0], fmt='.2f')
    axes[1, 0].set_title('Feature Correlations')
    plt.show()

    # 3. Target relationship
    y_series = pd.Series(y, name='target')
    sns.scatterplot(data=pd.concat([X_df.iloc[:, 0], y_series], axis=1),
                    x=X_df.columns[0], y='target', ax=axes[1, 1], alpha=0.6)
    axes[1, 1].set_title(f'{feat_names[0]} vs Target')

    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'sns.png')#, dpi=150, bbox_inches='tight')
    plt.show()

    sleep(120)
    exit(0)
    '''

    results = train_and_evaluate(
        X, y, feat_names,
        task_name=TASK,
        n_splits=args.splits,
        save_dir=SAVE_DIR,
        verbose=True,
        compound_types=[HIGH_ENTROPY],
        top_n_features=8,
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
    # plot_r2_vs_feature_count(
    #     X, y, feat_names,
    #     task_name='property',
    #     save_path=SAVE_DIR / 'r2_vs_features.png'
    # )
    # plot_r2_vs_cv_folds(
    #     X, y, feat_names,
    #     task_name='property',
    #     top_n_features=10,
    #     save_path=SAVE_DIR / 'r2_vs_folds.png'
    # )

    _write_report(results, feat_names, args.splits)
    print("\n[done] All outputs written to:", SAVE_DIR)


def _write_report(results, feat_names, n_splits):
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
        f"  Compound types  : {HIGH_ENTROPY}",
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

    num_rows = min(20, len(fi))
    lines += [
        "",
        f"── Top-{num_rows} Features by Permutation Importance ─────────────────",
        f"  {'Rank':<5} {'Feature':<35} {'Perm Imp':>10}  {'±':>2}  "
        f"{'Std':>8}  {'Tree MDI':>10}",
        "  " + "-" * 72,
    ]
    for rank, row in fi.head(num_rows).iterrows():
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