import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.inspection import permutation_importance
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
from sklearn.base import clone

def pca_features(X, n_components=0.95):
    pca = PCA(n_components=n_components, svd_solver="full")
    X_pca = pca.fit_transform(X)
    return pd.DataFrame(X_pca, index=X.index, columns=[f"pca_{i}" for i in range(X_pca.shape[1])]), pca

# LDA is not applicable to regression; provide a wrapper that returns None
def lda_features(*args, **kwargs):
    return None, None

def permutation_feature_importance(estimator, X, y, n_repeats=10, random_state=42, scoring=None):
    res = permutation_importance(estimator, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring)
    importances = res.importances_mean
    ranking = np.argsort(importances)[::-1]
    return importances, ranking

def sequential_feature_selection(estimator, X, y, k_features=10, forward=True, scoring="neg_mean_squared_error", cv=5, n_jobs=1):
    sfs = SFS(clone(estimator),
              k_features=k_features,
              forward=forward,
              scoring=scoring,
              cv=cv,
              n_jobs=n_jobs)
    sfs = sfs.fit(X.values, y.values)
    selected_idx = list(sfs.k_feature_idx_)
    return selected_idx, sfs

def evaluate_feature_methods(X_train, X_test, y_train, y_test, estimator=None, top_k=10):
    if estimator is None:
        estimator = LinearRegression()

    results = {}

    # PCA
    Xpca_train, pca = pca_features(X_train, n_components=0.95)
    Xpca_test = pca.transform(X_test)
    est_p = clone(estimator)
    est_p.fit(Xpca_train, y_train)
    results["pca"] = {"X_train": Xpca_train, "X_test": Xpca_test, "estimator": est_p}

    # LDA skipped for regression

    # Permutation importance
    est_base = clone(estimator)
    est_base.fit(X_train, y_train)
    importances, ranking = permutation_feature_importance(est_base, X_test, y_test, scoring="neg_mean_squared_error")
    top_idx = ranking[:min(top_k, X_train.shape[1])]
    feat_names = list(X_train.columns[top_idx])
    results["permutation"] = {"selected_features": feat_names, "importances": importances[top_idx]}

    # Sequential feature selection
    s_idx, sfs = sequential_feature_selection(estimator, X_train, y_train, k_features=min(top_k, X_train.shape[1]))
    s_feats = list(X_train.columns[s_idx])
    results["sfs"] = {"selected_features": s_feats, "sfs_obj": sfs}

    return results
