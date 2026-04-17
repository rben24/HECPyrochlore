"""
data_validation.py
==================
Shared data-validation and diagnostic-plot utilities for the HEC Pyrochlore
validate_lattice.py and validate_thermal.py scripts.

Exports
-------
  run_data_validation(df, feat_cols, target_col, task_label, save_dir)
      Prints a full validation report and saves diagnostic plots.

  plot_residuals(y_true, y_pred, labels, title, save_path)
  plot_influence(y_true, y_pred, labels, title, save_path)
  plot_correlation_heatmap(X, feat_names, title, save_path)
  plot_target_distribution(y, target_col, title, save_path)
  plot_feature_vs_target(X, y, feat_names, target_col, title, save_path)
  plot_pairplot_key_features(df_feat, target_col, top_features, save_path)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────

def _zscore_flags(series: pd.Series, threshold: float = 2.5) -> pd.Series:
    """Return boolean mask: True where |z-score| > threshold."""
    z = np.abs(stats.zscore(series.dropna()))
    flags = pd.Series(False, index=series.index)
    flags.loc[series.dropna().index] = z > threshold
    return flags


def _iqr_flags(series: pd.Series, k: float = 2.0) -> pd.Series:
    """Return boolean mask: True where value is outside [Q1-k*IQR, Q3+k*IQR]."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (series < lo) | (series > hi)


def _cooksd_approx(residuals: np.ndarray, n_features: int) -> np.ndarray:
    """
    Approximate Cook's distance using standardised residuals and hat-matrix
    diagonal approximation (h_ii ≈ 1/n for equal-leverage design).
    """
    n = len(residuals)
    p = n_features + 1
    mse = np.mean(residuals ** 2)
    h = np.full(n, 1.0 / n)          # equal-leverage approximation
    std_res = residuals / (np.sqrt(mse * (1 - h)) + 1e-12)
    cooksd = (std_res ** 2 * h) / (p * (1 - h + 1e-12))
    return cooksd


# ─────────────────────────────────────────────────────────────────────────────
# Main validation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_data_validation(df: pd.DataFrame,
                        feat_cols: List[str],
                        target_col: str,
                        task_label: str,
                        save_dir: Path,
                        y_true: Optional[np.ndarray] = None,
                        y_pred: Optional[np.ndarray] = None,
                        sample_labels: Optional[List[str]] = None):
    """
    Run a full data-validation sweep and save diagnostic plots.

    Parameters
    ----------
    df           : DataFrame with features and target (all rows used for training).
    feat_cols    : Feature column names present in df.
    target_col   : Name of the target column in df.
    task_label   : Short label, e.g. 'Lattice Parameter' — used in plot titles.
    save_dir     : Directory to save PNG plots.
    y_true       : (optional) test-set true values for residual plots.
    y_pred       : (optional) test-set predicted values for residual plots.
    sample_labels: (optional) composition IDs for labelling outlier points.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    prefix = save_dir.name  # e.g. 'lattice_param'

    print()
    print("=" * 60)
    print(f"  DATA VALIDATION — {task_label.upper()}")
    print("=" * 60)

    # ── 1. Target distribution ─────────────────────────────────────────────
    y_series = df[target_col].dropna()
    print(f"\n  Target  : {target_col}")
    print(f"  Samples : {len(y_series)}")
    print(f"  Mean    : {y_series.mean():.4f}   Std : {y_series.std():.4f}")
    print(f"  Min     : {y_series.min():.4f}   Max : {y_series.max():.4f}")
    q1, q3 = y_series.quantile(0.25), y_series.quantile(0.75)
    print(f"  Q1      : {q1:.4f}   Q3  : {q3:.4f}   IQR : {q3-q1:.4f}")

    plot_target_distribution(
        y_series.values, target_col,
        title=f"{task_label} — Target Distribution",
        save_path=save_dir / f"{prefix}_target_distribution.png",
    )

    # ── 2. Outlier detection on target ─────────────────────────────────────
    z_flags   = _zscore_flags(y_series, threshold=2.5)
    iqr_flags = _iqr_flags(y_series, k=2.0)
    outliers  = y_series[z_flags | iqr_flags]

    print(f"\n── Target Outlier Detection ────────────────────────────────")
    print(f"  Z-score  flagged (|z|>2.5) : {z_flags.sum()} samples")
    print(f"  IQR      flagged (k=2.0)   : {iqr_flags.sum()} samples")
    if len(outliers):
        print(f"\n  Flagged sample values:")
        for idx, val in outliers.items():
            lbl = sample_labels[idx] if (sample_labels and idx < len(sample_labels)) else f"row {idx}"
            z   = float(np.abs((val - y_series.mean()) / y_series.std()))
            print(f"    {lbl:35s}  {target_col}={val:.4f}   |z|={z:.2f}")
    else:
        print("  No target outliers detected.")

    # ── 3. Feature statistics & outliers ──────────────────────────────────
    feat_df = df[feat_cols].copy()
    print(f"\n── Feature Outlier Summary ({len(feat_cols)} features) ──────────────────")
    flagged_features = {}
    for col in feat_cols:
        s = feat_df[col].dropna()
        if len(s) < 3:
            continue
        zf  = _zscore_flags(s, threshold=2.5)
        iqrf = _iqr_flags(s, k=2.0)
        combined = zf | iqrf
        n_flagged = int(combined.sum())
        if n_flagged:
            flagged_features[col] = n_flagged
    if flagged_features:
        for col, n in sorted(flagged_features.items(), key=lambda x: -x[1]):
            print(f"  {col:<40} {n} flagged values")
    else:
        print("  No feature outliers detected.")

    # ── 4. Feature–target correlation ─────────────────────────────────────
    print(f"\n── Feature–Target Correlations (Pearson r) ──────────────────")
    corrs = {}
    for col in feat_cols:
        s = feat_df[col]
        paired = pd.concat([s, y_series], axis=1).dropna()
        if len(paired) > 2:
            r, p = stats.pearsonr(paired.iloc[:, 0], paired.iloc[:, 1])
            corrs[col] = (r, p)
    top_corrs = sorted(corrs.items(), key=lambda x: -abs(x[1][0]))
    print(f"  Top-10 features by |r| with {target_col}:")
    for col, (r, p) in top_corrs[:10]:
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "  ")
        print(f"  {col:<40} r={r:+.3f}  p={p:.3f} {sig}")

    # ── 5. Correlation heatmap ─────────────────────────────────────────────
    top10_feats = [c for c, _ in top_corrs[:10]]
    plot_correlation_heatmap(
        feat_df[top10_feats].values,
        top10_feats,
        title=f"{task_label} — Feature Correlation Matrix (top-10 by |r|)",
        save_path=save_dir / f"{prefix}_correlation_heatmap.png",
    )

    # ── 6. Feature vs target scatter grid ─────────────────────────────────
    top6 = [c for c, _ in top_corrs[:6]]
    plot_feature_vs_target(
        feat_df[top6].values,
        y_series.values if len(y_series) == len(feat_df) else feat_df.join(y_series)[target_col].values,
        top6, target_col,
        title=f"{task_label} — Top-6 Features vs Target",
        save_path=save_dir / f"{prefix}_feature_vs_target.png",
        df=df,
    )

    # ── 7. Residual diagnostics (if predictions provided) ─────────────────
    if y_true is not None and y_pred is not None:
        labels = sample_labels or [f"test_{i}" for i in range(len(y_true))]
        plot_residuals(
            y_true, y_pred, labels,
            title=f"{task_label} — Residual Diagnostics",
            save_path=save_dir / f"{prefix}_residuals.png",
        )
        plot_influence(
            y_true, y_pred, labels,
            n_features=len(feat_cols),
            title=f"{task_label} — Influence Plot (Cook's Distance approx.)",
            save_path=save_dir / f"{prefix}_influence.png",
        )

    print()
    print(f"  Validation plots saved to: {save_dir}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Individual plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_target_distribution(y: np.ndarray,
                              target_col: str,
                              title: str,
                              save_path: Optional[Path] = None):
    """Histogram + box-plot + KDE for the target variable."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Histogram + KDE
    ax = axes[0]
    ax.hist(y, bins=12, color='steelblue', edgecolor='white',
            alpha=0.8, density=True, label='Histogram')
    kde_x = np.linspace(y.min() - 0.5, y.max() + 0.5, 300)
    kde = stats.gaussian_kde(y)
    ax.plot(kde_x, kde(kde_x), 'r-', linewidth=1.5, label='KDE')
    ax.axvline(np.mean(y), color='k',      linestyle='--', linewidth=1, label=f'Mean={np.mean(y):.3f}')
    ax.axvline(np.median(y), color='gray', linestyle=':',  linewidth=1, label=f'Median={np.median(y):.3f}')
    # IQR fences
    q1, q3 = np.percentile(y, 25), np.percentile(y, 75)
    iqr = q3 - q1
    for fence, lbl in [(q1 - 2*iqr, 'IQR fence (k=2)'), (q3 + 2*iqr, None)]:
        ax.axvline(fence, color='orange', linestyle=':', linewidth=1,
                   label=lbl)
    ax.set_xlabel(target_col)
    ax.set_ylabel('Density')
    ax.set_title('Distribution & KDE')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Box plot
    ax2 = axes[1]
    bp = ax2.boxplot(y, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='steelblue', alpha=0.6),
                     medianprops=dict(color='red', linewidth=2),
                     flierprops=dict(marker='o', color='orange',
                                     markerfacecolor='orange', markersize=7))
    # Overlay individual points
    x_jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(y))
    ax2.scatter(1 + x_jitter, y, alpha=0.6, s=30, color='steelblue',
                edgecolors='k', linewidth=0.4, zorder=5)
    ax2.set_ylabel(target_col)
    ax2.set_title('Box Plot (fliers = outlier candidates)')
    ax2.set_xticks([])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   labels: List[str],
                   title: str,
                   save_path: Optional[Path] = None):
    """
    Four-panel residual diagnostic:
    1. Residuals vs Predicted
    2. Standardised residuals vs index (with outlier labels)
    3. Q-Q plot
    4. Scale-location (sqrt|std resid| vs fitted)
    """
    residuals = y_true - y_pred
    std_resid = (residuals - residuals.mean()) / (residuals.std() + 1e-12)
    threshold = 2.0   # label points beyond ±2 std

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    colours = np.where(np.abs(std_resid) > threshold, '#E24B4A', 'steelblue')

    # ── Panel 1: Residuals vs Fitted ────────────────────────────────────────
    ax = axes[0, 0]
    ax.scatter(y_pred, residuals, c=colours, alpha=0.75,
               edgecolors='k', linewidth=0.4, s=50, zorder=3)
    ax.axhline(0, color='r', linestyle='--', linewidth=1)
    for i, (xv, yv, sr) in enumerate(zip(y_pred, residuals, std_resid)):
        if abs(sr) > threshold:
            ax.annotate(labels[i] if i < len(labels) else str(i),
                        (xv, yv), fontsize=7, ha='left',
                        xytext=(4, 4), textcoords='offset points', color='#E24B4A')
    ax.set_xlabel('Fitted value')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Fitted')
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Standardised residuals vs index ───────────────────────────
    ax = axes[0, 1]
    idx = np.arange(len(std_resid))
    ax.scatter(idx, std_resid, c=colours, alpha=0.75,
               edgecolors='k', linewidth=0.4, s=50, zorder=3)
    ax.axhline(0,  color='k',      linestyle='-',  linewidth=0.8)
    ax.axhline( threshold, color='orange', linestyle='--', linewidth=1,
                label=f'±{threshold}σ')
    ax.axhline(-threshold, color='orange', linestyle='--', linewidth=1)
    for i, (xv, yv, sr) in enumerate(zip(idx, std_resid, std_resid)):
        if abs(sr) > threshold:
            ax.annotate(labels[i] if i < len(labels) else str(i),
                        (xv, yv), fontsize=7, ha='left',
                        xytext=(4, 4), textcoords='offset points', color='#E24B4A')
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Standardised residual')
    ax.set_title('Standardised Residuals  (|>2| = potential outlier)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Q-Q plot ──────────────────────────────────────────────────
    ax = axes[1, 0]
    (osm, osr), (slope, intercept, r) = stats.probplot(std_resid, dist='norm')
    ax.scatter(osm, osr, color='steelblue', alpha=0.75,
               edgecolors='k', linewidth=0.4, s=50)
    line_x = np.array([osm.min(), osm.max()])
    ax.plot(line_x, slope * line_x + intercept, 'r--', linewidth=1.2)
    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Sample quantiles')
    ax.set_title(f'Normal Q-Q Plot  (r={r:.3f})')
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Scale-location ────────────────────────────────────────────
    ax = axes[1, 1]
    sqrt_abs = np.sqrt(np.abs(std_resid))
    ax.scatter(y_pred, sqrt_abs, c=colours, alpha=0.75,
               edgecolors='k', linewidth=0.4, s=50, zorder=3)
    # Lowess-style smoothed line (simple rolling average)
    order = np.argsort(y_pred)
    win = max(3, len(y_pred) // 5)
    smooth = pd.Series(sqrt_abs[order]).rolling(win, center=True,
                                                  min_periods=1).mean().values
    ax.plot(y_pred[order], smooth, 'r-', linewidth=1.5)
    ax.set_xlabel('Fitted value')
    ax.set_ylabel('√|Standardised residual|')
    ax.set_title('Scale-Location  (homoscedasticity check)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_influence(y_true: np.ndarray, y_pred: np.ndarray,
                   labels: List[str],
                   n_features: int,
                   title: str,
                   save_path: Optional[Path] = None):
    """
    Two-panel influence plot:
    1. Cook's distance bar chart (approximate)
    2. Leverage vs standardised residual bubble chart
    """
    residuals = y_true - y_pred
    std_resid = (residuals - residuals.mean()) / (residuals.std() + 1e-12)
    cooksd    = _cooksd_approx(residuals, n_features)

    # Cook's D threshold: 4 / n  (common rule of thumb)
    thresh = 4.0 / len(y_true)
    high_influence = cooksd > thresh

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # ── Panel 1: Cook's D bar chart ────────────────────────────────────────
    ax = axes[0]
    colours = np.where(high_influence, '#E24B4A', 'steelblue')
    ax.bar(np.arange(len(cooksd)), cooksd, color=colours,
           edgecolor='white', linewidth=0.4)
    ax.axhline(thresh, color='orange', linestyle='--', linewidth=1.2,
               label=f"Threshold 4/n = {thresh:.3f}")
    for i, (cd, hi) in enumerate(zip(cooksd, high_influence)):
        if hi:
            ax.text(i, cd + thresh * 0.1,
                    labels[i] if i < len(labels) else str(i),
                    fontsize=7, ha='center', color='#E24B4A', rotation=45)
    ax.set_xlabel('Sample index')
    ax.set_ylabel("Cook's Distance (approx.)")
    ax.set_title("Cook's Distance — Influence of Each Point")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel 2: Leverage vs Std Residual bubble ───────────────────────────
    ax2 = axes[1]
    n = len(y_true)
    leverage = np.full(n, 1.0 / n)  # equal-leverage approximation
    sizes    = np.clip(cooksd / (thresh + 1e-12) * 40, 10, 300)
    sc = ax2.scatter(leverage, std_resid,
                     s=sizes, c=colours, alpha=0.7,
                     edgecolors='k', linewidth=0.4, zorder=3)
    ax2.axhline( 2, color='orange', linestyle='--', linewidth=1, label='±2σ')
    ax2.axhline(-2, color='orange', linestyle='--', linewidth=1)
    ax2.axhline( 0, color='k', linestyle='-', linewidth=0.8)
    for i, (lv, sr, hi) in enumerate(zip(leverage, std_resid, high_influence)):
        if hi or abs(sr) > 2:
            ax2.annotate(labels[i] if i < len(labels) else str(i),
                         (lv, sr), fontsize=7, ha='left',
                         xytext=(4, 4), textcoords='offset points', color='#E24B4A')
    ax2.set_xlabel('Leverage (approx. 1/n)')
    ax2.set_ylabel('Standardised Residual')
    ax2.set_title('Leverage vs Residual\n(bubble size ∝ Cook\'s D)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_correlation_heatmap(X: np.ndarray,
                              feat_names: List[str],
                              title: str,
                              save_path: Optional[Path] = None):
    """Pearson correlation heatmap for the given feature matrix."""
    df = pd.DataFrame(X, columns=feat_names).dropna()
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(max(8, len(feat_names) * 0.9),
                                    max(6, len(feat_names) * 0.8)))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    cmap = plt.get_cmap('RdBu_r')
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')

    n = len(feat_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(feat_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(feat_names, fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_feature_vs_target(X: np.ndarray,
                            y: np.ndarray,
                            feat_names: List[str],
                            target_col: str,
                            title: str,
                            save_path: Optional[Path] = None,
                            df: Optional[pd.DataFrame] = None):
    """
    Grid of scatter plots: each selected feature vs the target.
    Points are coloured by data_source when available.
    """
    n = len(feat_names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    axes = np.array(axes).flatten()

    # Colour by data source if available
    if df is not None and 'data_source' in df.columns:
        sources  = df['data_source'].fillna('unknown').values
        src_list = sorted(set(sources))
        pal      = cm.tab10(np.linspace(0, 1, len(src_list)))
        src_map  = {s: pal[i] for i, s in enumerate(src_list)}
        colours  = [src_map[s] for s in sources]
    else:
        colours = ['steelblue'] * len(y)

    for i, fname in enumerate(feat_names):
        ax = axes[i]
        xi = X[:, i]
        mask = ~(np.isnan(xi) | np.isnan(y))
        xm, ym = xi[mask], y[mask]
        cm_vals = [colours[j] for j in range(len(colours)) if mask[j]]

        ax.scatter(xm, ym, c=cm_vals, alpha=0.7, s=35,
                   edgecolors='k', linewidth=0.3)

        # Regression line
        if len(xm) > 2:
            slope, intercept, r, *_ = stats.linregress(xm, ym)
            xs = np.linspace(xm.min(), xm.max(), 100)
            ax.plot(xs, slope * xs + intercept, 'r--', linewidth=1.1)
            ax.set_title(f'{fname}\nr={r:.3f}', fontsize=9)
        else:
            ax.set_title(fname, fontsize=9)

        ax.set_xlabel(fname, fontsize=8)
        ax.set_ylabel(target_col, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Legend for data sources
    if df is not None and 'data_source' in df.columns:
        from matplotlib.patches import Patch
        legend_elems = [Patch(facecolor=src_map[s], label=s) for s in src_list]
        fig.legend(handles=legend_elems, loc='lower right',
                   fontsize=8, title='Data source', ncol=2)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)