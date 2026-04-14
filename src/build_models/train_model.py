"""
Model training, cross-validation, feature importance, and persistence.
All models use scikit-learn so there are no heavy extra dependencies.
"""

import numpy as np
import pandas as pd
import pickle
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (KFold, cross_validate, train_test_split,
                                     GridSearchCV)
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


# ── Train / evaluate helpers ──────────────────────────────────────────────────

def evaluate_predictions(y_true: np.ndarray,
                         y_pred: np.ndarray) -> Dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


def cross_validate_model(model: Any, X: np.ndarray, y: np.ndarray,
                         n_splits: int = 5,
                         random_state: int = 42) -> Dict[str, np.ndarray]:
    """K-fold CV; returns dict with test and train score arrays."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    results = cross_validate(model, X, y, cv=kf, scoring=scoring,
                             return_train_score=True, n_jobs=1)
    return results


def print_cv_results(name: str, cv_res: Dict, n_splits: int = 5):
    print(f"\n  {'─'*54}")
    print(f"  {name}")
    print(f"  {'─'*54}")
    metrics = [('RMSE', 'neg_root_mean_squared_error', -1),
               ('MAE',  'neg_mean_absolute_error',     -1),
               ('R²',   'r2',                           1)]
    for label, key, sign in metrics:
        test  = sign * cv_res[f'test_{key}']
        train = sign * cv_res[f'train_{key}']
        print(f"  {label:6s}  test : {test.mean():.4f} ± {test.std():.4f}"
              f"    train: {train.mean():.4f} ± {train.std():.4f}")


# ── Feature importance ───────────────────────────────────────────────────────

def compute_feature_importance(model: Any, X_val: np.ndarray,
                               y_val: np.ndarray,
                               feature_names: List[str],
                               n_repeats: int = 30,
                               random_state: int = 42) -> pd.DataFrame:
    """
    Combine tree-based impurity importance (when available) with
    permutation importance on a held-out validation set.
    Returns a DataFrame sorted by permutation importance.
    """
    # Permutation importance (model-agnostic, robust)
    perm = permutation_importance(model, X_val, y_val,
                                  n_repeats=n_repeats,
                                  random_state=random_state,
                                  scoring='r2')
    df = pd.DataFrame({
        'feature':         feature_names,
        'perm_importance': perm.importances_mean,
        'perm_std':        perm.importances_std,
    })

    # Tree-based impurity importance (if available)
    if hasattr(model, 'feature_importances_'):
        df['tree_importance'] = model.feature_importances_
    else:
        df['tree_importance'] = np.nan

    df.sort_values('perm_importance', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Main training pipeline ───────────────────────────────────────────────────

def train_and_evaluate(X: np.ndarray, y: np.ndarray,
                       feature_names: List[str],
                       task_name: str = 'property',
                       n_splits: int = 5,
                       test_size: float = 0.20,
                       random_state: int = 42,
                       save_dir: Optional[Path] = None,
                       verbose: bool = True) -> Dict:
    """
    Full pipeline:
      1. Hold-out split (test)
      2. K-fold CV on the training portion for all models
      3. Retrain best model on full train set
      4. Evaluate on test set
      5. Compute feature importances
    Returns a results dict with all artefacts.
    """
    save_dir = save_dir or (MODELS_DIR / task_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hold-out split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state)

    if verbose:
        print(f"\n{'='*58}")
        print(f"  Training pipeline: {task_name.upper()}")
        print(f"  Train: {len(X_tr)}  |  Test: {len(X_te)}")
        print(f"  {n_splits}-fold CV on training set")
        print(f"{'='*58}")

    models = build_models(random_state)
    cv_results = {}
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
        print(f"\n{'='*58}")
        print(f"  Best model: {best_name}")
        print(f"  Test-set metrics:")
        for k, v in test_metrics.items():
            print(f"    {k}: {v:.4f}")
        print(f"{'='*58}\n")

    # Feature importance on test set
    fi_df = compute_feature_importance(best_model, X_te, y_te, feature_names)

    # Save artefacts
    pickle.dump(best_model, open(save_dir / f'{task_name}_model.pkl', 'wb'))
    pickle.dump(scaler,     open(save_dir / f'{task_name}_scaler.pkl', 'wb'))

    meta = {
        'task': task_name,
        'best_model': best_name,
        'feature_names': feature_names,
        'test_metrics': test_metrics,
        'cv_summary': {
            name: {
                'mean_r2': float(cv_results[name]['test_r2'].mean()),
                'std_r2':  float(cv_results[name]['test_r2'].std()),
                'mean_rmse': float(-cv_results[name]['test_neg_root_mean_squared_error'].mean()),
            }
            for name in cv_results
        }
    }
    json.dump(meta, open(save_dir / f'{task_name}_metadata.json', 'w'), indent=2)

    return {
        'best_model':   best_model,
        'best_name':    best_name,
        'scaler':       scaler,
        'cv_results':   cv_results,
        'test_metrics': test_metrics,
        'fi_df':        fi_df,
        'X_test':       X_te,
        'y_test':       y_te,
        'y_pred':       y_pred,
        'feature_names': feature_names,
        'save_dir':     save_dir,
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_feature_importance(fi_df: pd.DataFrame, title: str,
                            top_n: int = 15, save_path: Optional[Path] = None):
    top = fi_df.head(top_n).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Permutation importance
    colours = cm.viridis(np.linspace(0.3, 0.9, len(top)))
    axes[0].barh(top['feature'], top['perm_importance'],
                 xerr=top['perm_std'], color=colours, capsize=3)
    axes[0].set_xlabel('Permutation Importance (ΔR²)')
    axes[0].set_title('Permutation Importance (validation set)')
    axes[0].axvline(0, color='k', linewidth=0.8, linestyle='--')

    # Tree importance
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
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'r--', linewidth=1.2)
    ax.set_xlabel(f'Actual {ylabel}')
    ax.set_ylabel(f'Predicted {ylabel}')
    ax.set_title(f'{title}\nR²={metrics["R2"]:.3f}  RMSE={metrics["RMSE"]:.4f}  MAE={metrics["MAE"]:.4f}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_cv_comparison(cv_results: Dict, title: str,
                       save_path: Optional[Path] = None):
    names = list(cv_results.keys())
    r2_means = [-cv_results[n]['test_r2'].mean() * -1 for n in names]
    r2_stds  = [cv_results[n]['test_r2'].std() for n in names]
    rmse_means = [-cv_results[n]['test_neg_root_mean_squared_error'].mean() for n in names]
    rmse_stds  = [cv_results[n]['test_neg_root_mean_squared_error'].std() for n in names]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    axes[0].bar(x, r2_means, yerr=r2_stds, capsize=5,
                color=['steelblue', 'darkorange', 'forestgreen', 'firebrick'],
                edgecolor='k', linewidth=0.6)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=20)
    axes[0].set_ylabel('CV R²'); axes[0].set_title('Cross-Validation R²')
    axes[0].axhline(0, color='k', linewidth=0.8)
    axes[0].set_ylim(min(0, min(r2_means) - 0.1), 1.05)

    axes[1].bar(x, rmse_means, yerr=rmse_stds, capsize=5,
                color=['steelblue', 'darkorange', 'forestgreen', 'firebrick'],
                edgecolor='k', linewidth=0.6)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=20)
    axes[1].set_ylabel('CV RMSE'); axes[1].set_title('Cross-Validation RMSE')

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
