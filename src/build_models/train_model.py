"""
Training module for ML models
Handles model instantiation, training, and cross-validation
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import ModelConfig
from src.data.make_dataset import make_dataset



class ModelTrainer:
    """Train and validate ML models"""

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.scaler = StandardScaler()
        self.trained_models = {}
        self.cv_results = {}

    def prepare_data(self, df: pd.DataFrame,
                     target: str,
                     exclude_features: List[str] = None,
                     task_type: str = 'regression') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target for training
        """
        # Remove rows with missing target
        df_clean = df.dropna(subset=[target]).copy()

        # Select features
        feature_cols = [col for col in df_clean.columns
                        if col != target and col not in (exclude_features or [])]

        # Remove non-numeric features
        feature_cols = [col for col in feature_cols
                        if df_clean[col].dtype in [np.float64, np.float32, np.int64, np.int32]]

        X = df_clean[feature_cols].values
        y = df_clean[target].values

        # Remove rows with NaN in features
        valid_idx = ~np.isnan(X).any(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Target: {target}")
        print(f"Feature range: {X.min():.4f} to {X.max():.4f}")
        print(f"Target range: {y.min():.4f} to {y.max():.4f}")

        return X, y, feature_cols

    def train_test_split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Split data into train, validation, and test sets
        """
        # First split: 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size,
            random_state=self.config.random_state
        )

        # Second split: split train into train and val
        val_size_adjusted = self.config.val_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size_adjusted,
            random_state=self.config.random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\nData split:")
        print(f"  Train: {X_train_scaled.shape[0]} samples")
        print(f"  Validation: {X_val_scaled.shape[0]} samples")
        print(f"  Test: {X_test_scaled.shape[0]} samples")

        return (X_train_scaled, X_val_scaled, X_test_scaled,
                y_train, y_val, y_test)

    def build_model(self, model_name: str,
                    task_type: str = 'regression') -> Any:
        """
        Instantiate a model
        """
        if task_type == 'regression':
            if model_name == 'random_forest':
                return RandomForestRegressor(**self.config.random_forest_params)
            elif model_name == 'gradient_boosting':
                return GradientBoostingRegressor(**self.config.gradient_boosting_params)
            elif model_name == 'xgboost':
                return xgb.XGBRegressor(**self.config.xgboost_params)
            elif model_name == 'neural_network':
                return MLPRegressor(**self.config.neural_network_params)

        elif task_type == 'classification':
            if model_name == 'random_forest':
                params = self.config.random_forest_params.copy()
                params['n_estimators'] = 300
                return RandomForestClassifier(**params)
            elif model_name == 'gradient_boosting':
                return GradientBoostingClassifier(**self.config.gradient_boosting_params)
            elif model_name == 'xgboost':
                params = self.config.xgboost_params.copy()
                params['objective'] = 'binary:logistic'
                return xgb.XGBClassifier(**params)
            elif model_name == 'neural_network':
                return MLPClassifier(**self.config.neural_network_params)

        raise ValueError(f"Unknown model: {model_name}")

    def train_model(self, model: Any,
                    X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray = None, y_val: np.ndarray = None) -> Any:
        """
        Train a single model
        """
        if isinstance(model, (xgb.XGBRegressor, xgb.XGBClassifier)):
            # XGBoost with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)] if X_val is not None else None,
                early_stopping_rounds=20,
                verbose=False
            )
        else:
            # Standard training
            model.fit(X_train, y_train)

        return model

    def cross_validate_model(self, model: Any,
                             X: np.ndarray, y: np.ndarray,
                             model_name: str,
                             task_type: str = 'regression') -> Dict[str, np.ndarray]:
        """
        Perform k-fold cross-validation
        """
        kfold = KFold(n_splits=self.config.n_splits,
                      shuffle=True,
                      random_state=self.config.random_state)

        if task_type == 'regression':
            scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        else:
            scoring = ['accuracy', 'precision', 'recall', 'f1']

        cv_results = cross_validate(
            model, X, y, cv=kfold, scoring=scoring,
            return_train_score=True, n_jobs=self.config.n_jobs
        )

        # Print results
        print(f"\n{model_name.upper()} - Cross-Validation Results ({self.config.n_splits}-fold):")
        for metric in scoring:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            print(f"  {metric}:")
            print(f"    Test:  {test_scores.mean():.4f} (+/- {test_scores.std():.4f})")
            print(f"    Train: {train_scores.mean():.4f} (+/- {train_scores.std():.4f})")

        self.cv_results[model_name] = cv_results
        return cv_results

    def train_multiple_models(self, X: np.ndarray, y: np.ndarray,
                              model_names: List[str],
                              task_type: str = 'regression') -> Dict[str, Any]:
        """
        Train multiple models and return best one
        """
        print("\n" + "=" * 70)
        print("TRAINING MULTIPLE MODELS")
        print("=" * 70)

        X_train, X_val, X_test, y_train, y_val, y_test = self.train_test_split_data(X, y)

        trained_models = {}
        cv_results = {}

        for model_name in model_names:
            print(f"\n{'=' * 70}")
            print(f"Training: {model_name.upper()}")
            print(f"{'=' * 70}")

            # Build and train
            model = self.build_model(model_name, task_type=task_type)
            model = self.train_model(model, X_train, y_train, X_val, y_val)
            trained_models[model_name] = model

            # Cross-validate
            self.cross_validate_model(model, X_train, y_train, model_name, task_type)

        return {
            'models': trained_models,
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'scaler': self.scaler
        }

    def save_model(self, model: Any, filepath: Path):
        """Save trained model to disk"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to: {filepath}")

    def save_scaler(self, filepath: Path):
        """Save feature scaler"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to: {filepath}")


#----------------------------------------------------------
# src/build_models/train_model.py
from src.data.make_dataset import load_raw
from src.data.process_dataset import basic_process, get_train_test
from src.utils.globals import DATA_PATH, TARGET_COL, RANDOM_STATE, TEST_SIZE
from src.feat# src/build_models/train_model.py
from src.data.make_dataset import load_raw
from src.data.process_dataset import basic_process, get_train_test
from src.utils.globals import DATA_PATH, TARGET_COL, RANDOM_STATE, TEST_SIZE
from src.features.feature_extraction import evaluate_feature_methods
from src.build_models.evaluate_model import evaluate_regressor
from src.build_models.predict_model import predict_and_format

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def build_candidate_models(random_state=RANDOM_STATE):
    models = {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=random_state),
        "svr": SVR()
    }
    return models

def hyperparameter_tune(model, X, y, param_grid=None, cv=3, scoring="neg_mean_squared_error"):
    if not param_grid:
        return model
    gs = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    gs.fit(X, y)
    return gs.best_estimator_

def train_and_select_best(X_train, X_test, y_train, y_test, feature_candidates, models=None):
    if models is None:
        models = build_candidate_models()

    best_score = np.inf  # lower is better (RMSE)
    best = {"model_name": None, "estimator": None, "features": None, "metrics": None}

    for feat_key, feat_info in feature_candidates.items():
        if "X_train" in feat_info:
            Xtr = feat_info["X_train"]
            Xte = feat_info["X_test"]
            feat_names = list(Xtr.columns)
        elif "selected_features" in feat_info:
            feat_names = feat_info["selected_features"]
            Xtr = X_train[feat_names]
            Xte = X_test[feat_names]
        else:
            continue

        for model_name, estimator in models.items():
            est = estimator
            if model_name == "random_forest":
                est = hyperparameter_tune(estimator, Xtr, y_train,
                                          param_grid={"n_estimators": [100, 200]}, cv=3)
            est.fit(Xtr, y_train)
            metrics = evaluate_regressor(est, Xte, y_test)
            score = metrics.get("rmse", np.inf)
            if score < best_score:
                best_score = score
                best.update({"model_name": model_name, "estimator": est, "features": feat_names, "metrics": metrics})

    return best

def main():
    df = load_raw(DATA_PATH)
    X, y = basic_process(df, TARGET_COL)
    X_train, X_test, y_train, y_test = get_train_test(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    estimator_for_feat = LinearRegression()
    feature_candidates = evaluate_feature_methods(X_train, X_test, y_train, y_test, estimator=estimator_for_feat, top_k=10)

    best = train_and_select_best(X_train, X_test, y_train, y_test, feature_candidates)

    print("Best model:", best["model_name"])
    print("Features used (sample):", best["features"][:10] if best["features"] else None)
    print("Metrics:", best["metrics"])

    if best["estimator"] is not None and best["features"] is not None:
        X_test_sel = X_test[best["features"]]
        preds_df = predict_and_format(best["estimator"], X_test_sel)
        print("Predictions sample:\n", preds_df.head())

if __name__ == "__main__":
    main()
