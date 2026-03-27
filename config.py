"""
Configuration management for ML pipeline
Handles paths, model parameters, and caching logic
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

# Project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
SRC_DIR = PROJECT_ROOT / 'src'
UTILS_DIR = PROJECT_ROOT / 'utils'

# Create directories
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File paths
RAW_DATA_FILE = RAW_DATA_DIR / 'pyrochlore_dataset.csv'
RAW_COMPONENT_FILE = RAW_DATA_DIR / 'parent_components.csv'
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / 'processed_pyrochlore.csv'
METADATA_FILE = PROCESSED_DATA_DIR / 'preprocessing_metadata.json'
FEATURE_GROUPS_FILE = PROCESSED_DATA_DIR / 'feature_groups.json'


@dataclass
class ModelConfig:
    """Configuration for model training"""
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42
    n_splits: int = 5
    n_jobs: int = -1

    # Hyperparameters by model type
    random_forest_params: Dict = None
    gradient_boosting_params: Dict = None
    xgboost_params: Dict = None
    neural_network_params: Dict = None

    def __post_init__(self):
        if self.random_forest_params is None:
            self.random_forest_params = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': self.random_state,
                'n_jobs': self.n_jobs
            }

        if self.gradient_boosting_params is None:
            self.gradient_boosting_params = {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 5,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'subsample': 0.8,
                'random_state': self.random_state
            }

        if self.xgboost_params is None:
            self.xgboost_params = {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'verbosity': 0
            }

        if self.neural_network_params is None:
            self.neural_network_params = {
                'hidden_layer_sizes': (128, 64, 32),
                'activation': 'relu',
                'solver': 'adam',
                'learning_rate_init': 0.001,
                'max_iter': 500,
                'random_state': self.random_state,
                'early_stopping': True,
                'validation_fraction': 0.1
            }


# Default configuration
DEFAULT_MODEL_CONFIG = ModelConfig()

# Feature selection by target
FEATURE_CONFIGS = {
    'thermal_conductivity': {
        'target': 'TPS Cond W/m/K',
        'exclude_features': ['Composition', 'Sample A', 'Sample B', 'Synthesis Method',
                             'Corrected Cond W/m/K', 'thermal_conductivity_raw'],
        'models': ['random_forest', 'gradient_boosting', 'xgboost'],
        'primary_model': 'xgboost'
    },
    'density': {
        'target': 'Relative Density %',
        'exclude_features': ['Composition', 'Sample A', 'Sample B', 'Synthesis Method',
                             'relative_density'],
        'models': ['random_forest', 'gradient_boosting', 'xgboost'],
        'primary_model': 'xgboost'
    },
    'lattice_parameter': {
        'target': 'Lattice Parameter (Angstrom)',
        'exclude_features': ['Composition', 'Sample A', 'Sample B', 'Synthesis Method',
                             'lattice_volume', 'lattice_distortion_index'],
        'models': ['random_forest', 'xgboost'],
        'primary_model': 'xgboost'
    },
    'single_phase': {
        'target': 'Is Single Phase',
        'exclude_features': ['Composition', 'Sample A', 'Sample B', 'Synthesis Method'],
        'models': ['random_forest', 'gradient_boosting', 'xgboost'],
        'primary_model': 'xgboost',
        'task_type': 'classification'
    }
}