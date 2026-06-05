"""
Model training, cross-validation, feature importance, and persistence.
All models use scikit-learn.

Changes vs. original
--------------------
* ``train_and_evaluate`` now accepts an optional ``compound_types`` list that
  is forwarded into the metadata JSON so reports can show which compound types
  were used for training.
* The feature-importance plot and report include the 4 new cross-site contrast
  features added in build_features.py.
* Everything else is backward-compatible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pickle
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any

from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

warnings.filterwarnings('ignore')

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent
MODELS_DIR = _PROJECT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── Model registry ───────────────────────────────────────────────────────────

def build_models(random_state: int = 42) -> Dict[str, Any]:
    return {
        'RandomForest': RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            max_features='sqrt', random_state=random_state, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, min_samples_leaf=2, random_state=random_state),
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            max_features='sqrt', random_state=random_state, n_jobs=-1),
        'Ridge': Ridge(alpha=10.0),
    }


# ── Metric helpers ───────────────────────────────────────────────────────────

def evaluate_predictions(y_true: np.ndarray,
                         y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE':  float(mean_absolute_error(y_true, y_pred)),
        'R2':   float(r2_score(y_true, y_pred)),
    }


def cross_validate_model(model: Any, X: np.ndarray, y: np.ndarray,
                         n_splits: int = 6,
                         random_state: int = 42) -> Dict[str, np.ndarray]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    return cross_validate(model, X, y, cv=kf, scoring=scoring,
                          return_train_score=True, n_jobs=1)


def print_cv_results(name: str, cv_res: Dict, n_splits: int = 5):
    print(f"\n  {'─'*54}")
    print(f"  {name}")
    print(f"  {'─'*54}")
    for label, key, sign in [
        ('RMSE', 'neg_root_mean_squared_error', -1),
        ('MAE',  'neg_mean_absolute_error',     -1),
        ('R²',   'r2',                           1),
    ]:
        test  = sign * cv_res[f'test_{key}']
        train = sign * cv_res[f'train_{key}']
        print(f"  {label:6s}  test : {test.mean():.4f} ± {test.std():.4f}"
              f"    train: {train.mean():.4f} ± {train.std():.4f}")


# ── Feature importance ───────────────────────────────────────────────────────

def compute_feature_importance(model: Any, X_val: np.ndarray, y_val: np.ndarray,
                               feature_names: List[str],
                               n_repeats: int = 30,
                               random_state: int = 42) -> pd.DataFrame:
    perm = permutation_importance(model, X_val, y_val,
                                  n_repeats=n_repeats, random_state=random_state,
                                  scoring='r2')
    df = pd.DataFrame({
        'feature':         feature_names,
        'perm_importance': perm.importances_mean,
        'perm_std':        perm.importances_std,
    })
    if hasattr(model, 'feature_importances_'):
        df['tree_importance'] = model.feature_importances_
    else:
        df['tree_importance'] = np.nan
    df.sort_values('perm_importance', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Main training pipeline ───────────────────────────────────────────────────

def train_and_evaluate(
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        task_name: str = 'property',
        n_splits: int = 6,
        test_size: float = 0.20,
        random_state: int = 42,
        save_dir: Optional[Path] = None,
        verbose: bool = True,
        compound_types: Optional[List[str]] = None,
        top_n_features: Optional[int] = None,
) -> Dict:
    """
    Full training pipeline with optional feature selection:
      1. Hold-out split (test_size)
      2. Select top N features (optional)
      3. K-fold CV on training portion for all 4 models
      4. Retrain best model on full training set
      5. Evaluate on hold-out test set
      6. Compute permutation + tree feature importances
      7. Persist model, scaler, and metadata JSON

    Parameters
    ----------
    top_n_features : int, optional (default=10)
        If set, only the top N features by importance are used for training.
        Set to None to use all features.
    compound_types : list of compound types used (for metadata / reporting only)
    """
    save_dir = save_dir or (MODELS_DIR / task_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    # scaler = StandardScaler()
    scaler = QuantileTransformer()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state)

    # ─── FEATURE SELECTION  ───────
    selected_feature_names = feature_names
    selected_feature_indices = np.arange(len(feature_names))

    if top_n_features is not None and top_n_features < len(feature_names):
        # Quick importance estimation on training set
        rf_temp = RandomForestRegressor(n_estimators=100, max_depth=10,
                                        random_state=random_state, n_jobs=-1)
        rf_temp.fit(X_tr, y_tr)

        # Get top N features
        importances = rf_temp.feature_importances_
        selected_feature_indices = np.argsort(importances)[-top_n_features:][::-1]

        selected_feature_names = [feature_names[i] for i in selected_feature_indices]
        X_tr = X_tr[:, selected_feature_indices]
        X_te = X_te[:, selected_feature_indices]

        if verbose:
            print(f"  Selected {top_n_features} top features:")
            for i, fname in enumerate(selected_feature_names, 1):
                print(f"    {i}. {fname}")

    if verbose:
        ctype_str = ', '.join(compound_types) if compound_types else 'all'
        print(f"\n{'=' * 58}")
        print(f"  Training pipeline : {task_name.upper()}")
        print(f"  Compound types    : {ctype_str}")
        print(f"  Features          : {len(selected_feature_names)}")
        print(f"  Train: {len(X_tr)}  |  Test: {len(X_te)}")
        print(f"  {n_splits}-fold CV on training set")
        print(f"{'=' * 58}")

    models = build_models(random_state)
    cv_results: Dict = {}
    best_name, best_r2, best_model = None, -np.inf, None

    for name, model in models.items():
        cv_res = cross_validate_model(model, X_tr, y_tr, n_splits, random_state)
        cv_results[name] = cv_res
        if verbose:
            print_cv_results(name, cv_res, n_splits)
        mean_r2 = cv_res['test_r2'].mean()
        if mean_r2 > best_r2:
            best_r2, best_name, best_model = mean_r2, name, model

    # Retrain best on full training set
    best_model.fit(X_tr, y_tr)
    y_pred = best_model.predict(X_te)
    test_metrics = evaluate_predictions(y_te, y_pred)

    if verbose:
        print(f"\n{'=' * 58}")
        print(f"  Best model : {best_name}")
        print(f"  Test-set metrics:")
        for k, v in test_metrics.items():
            print(f"    {k}: {v:.4f}")
        print(f"{'=' * 58}\n")

    # IMPORTANT: Pass selected_feature_names, not the original feature_names
    # fi_df = compute_feature_importance(best_model, X_te, y_te, selected_feature_names)
    fi_df = compute_feature_importance(best_model, X_te, y_te, selected_feature_names)

    # Persist
    pickle.dump(best_model, open(save_dir / f'{task_name}_model.pkl', 'wb'))
    pickle.dump(scaler, open(save_dir / f'{task_name}_scaler.pkl', 'wb'))

    meta = {
        'task': task_name,
        'best_model': best_name,
        'feature_names': selected_feature_names,
        'selected_indices': [int(i) for i in selected_feature_indices],  # Store indices for reference
        'compound_types': compound_types or ['pristine', 'high_entropy'],
        'n_train': int(len(X_tr)),
        'n_test': int(len(X_te)),
        'test_metrics': test_metrics,
        'cv_summary': {
            name: {
                'mean_r2': float(cv_results[name]['test_r2'].mean()),
                'std_r2': float(cv_results[name]['test_r2'].std()),
                'mean_rmse': float(-cv_results[name]
                ['test_neg_root_mean_squared_error'].mean()),
            }
            for name in cv_results
        },
    }
    json.dump(meta, open(save_dir / f'{task_name}_metadata.json', 'w'), indent=2)

    return {
        'best_model':           best_model,
        'best_name':            best_name,
        'scaler':               scaler,
        'cv_results':           cv_results,
        'test_metrics':         test_metrics,
        'fi_df':                fi_df,
        'X_test':               X_te,
        'y_test':               y_te,
        'y_pred':               y_pred,
        'feature_names':        selected_feature_names,
        'selected_indices':     selected_feature_indices,
        'save_dir':             save_dir,
        'n_train':              int(len(X_tr)),
        'n_test':               int(len(X_te)),
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_feature_importance(fi_df: pd.DataFrame, title: str,
                            top_n: int = 15,
                            save_path: Optional[Path] = None):
    top = fi_df.head(top_n).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, top_n * 0.45)))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    colours = cm.viridis(np.linspace(0.3, 0.9, len(top)))
    axes[0].barh(top['feature'], top['perm_importance'],
                 xerr=top['perm_std'], color=colours, capsize=3)
    axes[0].set_xlabel('Permutation Importance (ΔR²)')
    axes[0].set_title('Permutation Importance (validation set)')
    axes[0].axvline(0, color='k', linewidth=0.8, linestyle='--')

    if not top['tree_importance'].isna().all():
        colours2 = cm.plasma(np.linspace(0.3, 0.9, len(top)))
        axes[1].barh(top['feature'], top['tree_importance'], color=colours2)
        axes[1].set_xlabel('Mean Decrease in Impurity')
        axes[1].set_title('Tree-Based (MDI) Importance')
    else:
        axes[1].text(0.5, 0.5, 'Not available\n(linear model)',
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Tree-Based Importance')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_parity(y_true: np.ndarray, y_pred: np.ndarray,
                title: str, ylabel: str,
                save_path: Optional[Path] = None):
    metrics = evaluate_predictions(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.75, edgecolors='k', linewidth=0.5,
               color='steelblue', zorder=3)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'r--', linewidth=1.2)
    ax.set_xlabel(f'Actual {ylabel}')
    ax.set_ylabel(f'Predicted {ylabel}')
    ax.set_title(
        f'{title}\n'
        f'R²={metrics["R2"]:.3f}  RMSE={metrics["RMSE"]:.4f}  MAE={metrics["MAE"]:.4f}'
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_cv_comparison(cv_results: Dict, title: str,
                       save_path: Optional[Path] = None):
    names = list(cv_results.keys())
    r2_means = [cv_results[n]['test_r2'].mean() for n in names]
    r2_stds  = [cv_results[n]['test_r2'].std()  for n in names]
    rmse_means = [-cv_results[n]['test_neg_root_mean_squared_error'].mean()
                  for n in names]
    rmse_stds  = [cv_results[n]['test_neg_root_mean_squared_error'].std()
                  for n in names]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    colours = ['steelblue', 'darkorange', 'forestgreen', 'firebrick']

    axes[0].bar(x, r2_means, yerr=r2_stds, capsize=5,
                color=colours, edgecolor='k', linewidth=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20)
    axes[0].set_ylabel('CV R²')
    axes[0].set_title('Cross-Validation R²')
    axes[0].axhline(0, color='k', linewidth=0.8)
    axes[0].set_ylim(min(0, min(r2_means) - 0.1), 1.05)

    axes[1].bar(x, rmse_means, yerr=rmse_stds, capsize=5,
                color=colours, edgecolor='k', linewidth=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20)
    axes[1].set_ylabel('CV RMSE')
    axes[1].set_title('Cross-Validation RMSE')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)

# ── Extra plots for hyperparameter tuning ─────────────────────────────────────

def plot_r2_vs_feature_count(
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        task_name: str = 'property',
        test_size: float = 0.20,
        random_state: int = 42,
        save_path: Optional[Path] = None,
):
    """
    Train models with increasing numbers of top features.
    Plot R² (test, train, CV mean) vs. number of features.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    feature_names : List[str]
        Names of all features
    task_name : str
        Name of task (for plot title)
    test_size : float
        Test split size
    random_state : int
        Random seed
    save_path : Path, optional
        Where to save the figure
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state)

    # Get initial feature importance to sort features
    rf_temp = RandomForestRegressor(n_estimators=100, max_depth=10,
                                    random_state=random_state, n_jobs=-1)
    rf_temp.fit(X_tr, y_tr)
    importances = rf_temp.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    # Test different feature counts
    feature_counts = list(range(1, len(feature_names) + 1))
    results = {
        'RandomForest': {'test_r2': [], 'train_r2': [], 'cv_r2': []},
        'GradientBoosting': {'test_r2': [], 'train_r2': [], 'cv_r2': []},
        'ExtraTrees': {'test_r2': [], 'train_r2': [], 'cv_r2': []},
        'Ridge': {'test_r2': [], 'train_r2': [], 'cv_r2': []},
    }

    models = build_models(random_state)

    for n_feat in feature_counts:
        # Select top n features
        top_indices = sorted_indices[:n_feat]
        X_tr_subset = X_tr[:, top_indices]
        X_te_subset = X_te[:, top_indices]

        for model_name, model in models.items():
            # Train on full training set
            model.fit(X_tr_subset, y_tr)
            y_pred_train = model.predict(X_tr_subset)
            y_pred_test = model.predict(X_te_subset)

            train_r2 = r2_score(y_tr, y_pred_train)
            test_r2 = r2_score(y_te, y_pred_test)

            results[model_name]['train_r2'].append(train_r2)
            results[model_name]['test_r2'].append(test_r2)

            # CV on training set
            cv_res = cross_validate_model(model, X_tr_subset, y_tr,
                                          n_splits=5, random_state=random_state)
            cv_r2 = cv_res['test_r2'].mean()
            results[model_name]['cv_r2'].append(cv_r2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{task_name.upper()}: R² vs. Number of Features',
                 fontsize=14, fontweight='bold')

    colours = ['steelblue', 'darkorange', 'forestgreen', 'firebrick']
    model_names = list(results.keys())

    for idx, (ax, metric) in enumerate([
        (axes[0, 0], 'train_r2'),
        (axes[0, 1], 'test_r2'),
        (axes[1, 0], 'cv_r2'),
    ]):
        for color, name in zip(colours, model_names):
            ax.plot(feature_counts, results[name][metric],
                    marker='o', label=name, color=color, linewidth=2)

        ax.set_xlabel('Number of Features', fontsize=11)
        ax.set_ylabel('R²', fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, len(feature_counts) + 1, max(1, len(feature_counts) // 10)))

    # Combined view in bottom-right
    for color, name in zip(colours, model_names):
        axes[1, 1].plot(feature_counts, results[name]['test_r2'],
                        marker='o', label=f'{name} (test)',
                        color=color, linewidth=2, linestyle='-')
        axes[1, 1].plot(feature_counts, results[name]['cv_r2'],
                        marker='s', label=f'{name} (CV)',
                        color=color, linewidth=2, linestyle='--', alpha=0.7)

    axes[1, 1].set_xlabel('Number of Features', fontsize=11)
    axes[1, 1].set_ylabel('R²', fontsize=11)
    axes[1, 1].set_title('Test vs. CV R² Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].legend(loc='best', fontsize=8, ncol=2)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(range(0, len(feature_counts) + 1, max(1, len(feature_counts) // 10)))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_r2_vs_cv_folds(
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        task_name: str = 'property',
        test_size: float = 0.20,
        random_state: int = 42,
        save_path: Optional[Path] = None,
        top_n_features: int = 10,
):
    """
    Train models with increasing numbers of CV folds.
    Plot R² (CV mean and std) vs. number of folds.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    feature_names : List[str]
        Names of all features
    task_name : str
        Name of task (for plot title)
    test_size : float
        Test split size
    random_state : int
        Random seed
    save_path : Path, optional
        Where to save the figure
    top_n_features : int
        Number of top features to use (default 10)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state)

    # Feature selection
    rf_temp = RandomForestRegressor(n_estimators=100, max_depth=10,
                                    random_state=random_state, n_jobs=-1)
    rf_temp.fit(X_tr, y_tr)
    importances = rf_temp.feature_importances_
    top_indices = np.argsort(importances)[-top_n_features:][::-1]
    X_tr = X_tr[:, top_indices]
    X_te = X_te[:, top_indices]

    # Test different fold counts
    fold_counts = list(range(2, 21))  # 2 to 20 folds
    results = {
        'RandomForest': {'mean_r2': [], 'std_r2': []},
        'GradientBoosting': {'mean_r2': [], 'std_r2': []},
        'ExtraTrees': {'mean_r2': [], 'std_r2': []},
        'Ridge': {'mean_r2': [], 'std_r2': []},
    }

    models = build_models(random_state)

    for n_splits in fold_counts:
        for model_name, model in models.items():
            cv_res = cross_validate_model(model, X_tr, y_tr,
                                          n_splits=n_splits, random_state=random_state)
            cv_r2_mean = cv_res['test_r2'].mean()
            cv_r2_std = cv_res['test_r2'].std()

            results[model_name]['mean_r2'].append(cv_r2_mean)
            results[model_name]['std_r2'].append(cv_r2_std)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{task_name.upper()}: R² vs. Number of CV Folds (Top {top_n_features} Features)',
                 fontsize=14, fontweight='bold')

    colours = ['steelblue', 'darkorange', 'forestgreen', 'firebrick']
    model_names = list(results.keys())

    # Left: Mean R² with error bars
    for color, name in zip(colours, model_names):
        axes[0].errorbar(fold_counts, results[name]['mean_r2'],
                         yerr=results[name]['std_r2'],
                         marker='o', label=name, color=color, linewidth=2,
                         capsize=4, capthick=1.5)

    axes[0].set_xlabel('Number of CV Folds', fontsize=11)
    axes[0].set_ylabel('Mean R² ± Std Dev', fontsize=11)
    axes[0].set_title('Cross-Validation R² with Error Bars', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(fold_counts[::2])

    # Right: Stability (std dev)
    for color, name in zip(colours, model_names):
        axes[1].plot(fold_counts, results[name]['std_r2'],
                     marker='s', label=name, color=color, linewidth=2)

    axes[1].set_xlabel('Number of CV Folds', fontsize=11)
    axes[1].set_ylabel('Standard Deviation of R²', fontsize=11)
    axes[1].set_title('CV Stability (Lower is Better)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(fold_counts[::2])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(task_name: str, save_dir: Optional[Path] = None):
    d = save_dir or (MODELS_DIR / task_name)
    model  = pickle.load(open(d / f'{task_name}_model.pkl', 'rb'))
    scaler = pickle.load(open(d / f'{task_name}_scaler.pkl', 'rb'))
    meta   = json.load(open(d / f'{task_name}_metadata.json'))
    return model, scaler, meta