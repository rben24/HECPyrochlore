"""



"""

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import Simple, SimpleImputer
#
# def preprocess_data(file_path):
#     '''
#
#     :param file_path:
#     :return:
#     '''
#     df = pd.read_csv(file_path)
#
#     col_list = ["ID,Sample A","Sample B","TPS Cond W/m/K","Deviation","Synthesis Method","Relative Density %","Is Single Phase",
#                 "Corrected Cond W/m/K","Lattice Parameter (Angstrom)","Notes"]
#
#     # drop unnecessary columns
#     drop_col = ['ID', 'Notes']
#     for col in drop_col:
#         if col in df.columns:
#             df.drop(columns=(col), inplace=True)
#
#     # handle missing values
#     for col in df.columns:
#         if
#     df.fillna(, inplace=True)
#
#     X = df.drop(columns=['Lattice Parameter (Angstrom)'])
#     y = df['Lattice Parameter (Angstrom)']
#
#     # Use One Hot Encoding on categorical features
#     categorical_cols = ['Sample A', 'Sample B', 'Synthesis Method']
#     numerical_cols = X.columns.difference(categorical_cols).tolist()
#
#     # Preprocessing pipline
#     numeric_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='mean')),
#         ('scaler', StandardScaler()),
#     ])
#
#     categorical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')),
#     ])
#
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numeric_transformer, numerical_cols),
#             ('cat', categorical_transformer, categorical_cols),
#         ]
#     )
#
#     return preprocessor

"""
Data Preprocessing for Pyrochlore Oxide Dataset
Handles missing values, normalization, and data cleaning
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
import sys
from typing import List, Dict, Tuple

warnings.filterwarnings('ignore')

from features import build_features


class PyrochlorePreprocessor:
    """Preprocessing pipeline for pyrochlore oxide data"""

    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.categorical_mappings = {}

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load CSV data"""
        df = pd.read_csv(filepath)
        print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def handle_missing_values(self, df: pd.DataFrame,
                              strategy: str = 'mixed') -> pd.DataFrame:
        """
        Handle missing values with multiple strategies
        strategy: 'drop', 'mean', 'median', 'knn', 'mixed'
        """
        df_clean = df.copy()

        print("\n" + "=" * 60)
        print("MISSING VALUE ANALYSIS")
        print("=" * 60)

        missing_summary = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum(),
            'Missing_Percent': (df.isnull().sum() / len(df) * 100).round(2)
        })
        missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]

        if len(missing_summary) > 0:
            print(missing_summary.to_string(index=False))
        else:
            print("No missing values detected")

        if strategy == 'drop':
            df_clean = df.dropna()
            print(f"\nRows after dropping NaNs: {len(df_clean)}")

        elif strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            imputer = SimpleImputer(strategy='mean')
            df_clean[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            self.imputers['mean'] = imputer
            print(f"\nApplied mean imputation to {len(numeric_cols)} numeric columns")

        elif strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            imputer = SimpleImputer(strategy='median')
            df_clean[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            self.imputers['median'] = imputer
            print(f"\nApplied median imputation to {len(numeric_cols)} numeric columns")

        elif strategy == 'knn':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            imputer = KNNImputer(n_neighbors=5)
            df_clean[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            self.imputers['knn'] = imputer
            print(f"\nApplied KNN imputation (k=5) to {len(numeric_cols)} numeric columns")

        elif strategy == 'mixed':
            # Drop rows with critical missing values
            critical_cols = ['Composition', 'Sample A', 'Sample B', 'TPS Cond W/m/K']
            df_clean = df.dropna(subset=[col for col in critical_cols if col in df.columns])

            # For other numeric columns, use median
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            imputer = SimpleImputer(strategy='median')
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
            self.imputers['median'] = imputer

            print(f"\nApplied mixed strategy:")
            print(f"  - Dropped rows with missing critical columns")
            print(f"  - Applied median imputation to {len(numeric_cols)} numeric columns")
            print(f"  - Remaining rows: {len(df_clean)}")

        return df_clean

    def identify_outliers(self, df: pd.DataFrame,
                          method: str = 'iqr',
                          threshold: float = 1.5) -> pd.DataFrame:
        """
        Identify outliers using IQR or Z-score
        Returns dataframe with outlier flags
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_flags = pd.DataFrame(index=df.index)

        print("\n" + "=" * 60)
        print("OUTLIER DETECTION")
        print("=" * 60)

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outlier_flags[f'{col}_outlier'] = (
                        (df[col] < lower_bound) | (df[col] > upper_bound)
                )

            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_flags[f'{col}_outlier'] = z_scores > threshold

        # Summary
        outlier_summary = outlier_flags.sum()
        outlier_summary = outlier_summary[outlier_summary > 0]

        if len(outlier_summary) > 0:
            print(f"\nOutliers detected ({method}, threshold={threshold}):")
            print(outlier_summary)
        else:
            print("\nNo outliers detected")

        return outlier_flags

    def remove_outliers(self, df: pd.DataFrame,
                        outlier_flags: pd.DataFrame) -> pd.DataFrame:
        """Remove rows flagged as outliers"""
        any_outlier = outlier_flags.any(axis=1)
        df_clean = df[~any_outlier].copy()

        n_removed = any_outlier.sum()
        print(f"\nRemoved {n_removed} outlier rows")
        print(f"Remaining rows: {len(df_clean)}")

        return df_clean

    def encode_categorical(self, df: pd.DataFrame,
                           columns: List[str] = None) -> pd.DataFrame:
        """
        Encode categorical variables
        """
        df_encoded = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=['object']).columns.tolist()

        print("\n" + "=" * 60)
        print("CATEGORICAL ENCODING")
        print("=" * 60)

        for col in columns:
            if col in df.columns:
                unique_vals = df[col].unique()
                print(f"\n{col}: {len(unique_vals)} unique values")

                # One-hot encode if < 10 categories, else label encode
                if len(unique_vals) < 10:
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
                    self.categorical_mappings[col] = 'one_hot'
                    print(f"  Applied: One-hot encoding ({dummies.shape[1]} features)")
                else:
                    df_encoded[col] = pd.factorize(df[col])[0]
                    self.categorical_mappings[col] = 'label'
                    print(f"  Applied: Label encoding")

        return df_encoded

    def normalize_features(self, df: pd.DataFrame,
                           method: str = 'standard',
                           exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        Normalize numeric features
        method: 'standard', 'robust', 'minmax'
        """
        df_normalized = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if exclude_cols:
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        print("\n" + "=" * 60)
        print(f"FEATURE NORMALIZATION ({method})")
        print("=" * 60)

        if method == 'standard':
            scaler = StandardScaler()
            df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.scalers['standard'] = scaler
            print(f"Applied StandardScaler to {len(numeric_cols)} features")

        elif method == 'robust':
            scaler = RobustScaler()
            df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.scalers['robust'] = scaler
            print(f"Applied RobustScaler to {len(numeric_cols)} features")

        elif method == 'minmax':
            scaler = MinMaxScaler()
            df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.scalers['minmax'] = scaler
            print(f"Applied MinMaxScaler to {len(numeric_cols)} features")

        return df_normalized

    def create_feature_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Organize features into logical groups
        """
        feature_groups = {
            'composition': [col for col in df.columns if 'Sample' in col or 'Composition' in col],
            'a_site': [col for col in df.columns if 'a_site' in col],
            'b_site': [col for col in df.columns if 'b_site' in col],
            'lattice': [col for col in df.columns if 'lattice' in col or 'Lattice' in col],
            'density': [col for col in df.columns if 'density' in col or 'Density' in col],
            'thermal': [col for col in df.columns if 'thermal' in col or 'Thermal' in col or 'Cond' in col],
            'entropy': [col for col in df.columns if 'entropy' in col],
            'synthesis': [col for col in df.columns if 'Synthesis' in col or 'Method' in col],
            'quality': [col for col in df.columns if 'Phase' in col or 'Deviation' in col]
        }

        print("\n" + "=" * 60)
        print("FEATURE GROUPS")
        print("=" * 60)
        for group, cols in feature_groups.items():
            if cols:
                print(f"\n{group.upper()} ({len(cols)} features):")
                for col in cols:
                    print(f"  - {col}")

        return feature_groups

    def generate_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive summary statistics"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        summary = df[numeric_cols].describe().T
        summary['skewness'] = df[numeric_cols].skew()
        summary['kurtosis'] = df[numeric_cols].kurtosis()
        summary['cv'] = (df[numeric_cols].std() / df[numeric_cols].mean()).abs()  # Coefficient of variation

        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(summary.round(4))

        return summary

    def full_preprocessing_pipeline(self, filepath: str,
                                    handle_missing: str = 'mixed',
                                    remove_outliers: bool = True,
                                    outlier_method: str = 'iqr',
                                    normalize: str = 'standard',
                                    add_features: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete preprocessing pipeline
        """
        print("\n" + "=" * 70)
        print("PYROCHLORE OXIDE PREPROCESSING PIPELINE")
        print("=" * 70)

        # 1. Load data
        df = self.load_data(filepath)
        initial_shape = df.shape

        # 2. Add engineered features
        if add_features:
            print("\nAdding engineered features...")
            df = add_engineered_features(df)
            print(f"Features expanded: {initial_shape[1]} → {df.shape[1]} columns")

        # 3. Handle missing values
        df = self.handle_missing_values(df, strategy=handle_missing)

        # 4. Detect and remove outliers
        if remove_outliers:
            outlier_flags = self.identify_outliers(df, method=outlier_method)
            df = self.remove_outliers(df, outlier_flags)

        # 5. Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            df = self.encode_categorical(df, columns=categorical_cols)

        # 6. Generate statistics before normalization
        stats = self.generate_summary_statistics(df)

        # 7. Normalize features
        exclude_from_norm = ['Composition', 'is_single_phase', 'single_phase']
        df = self.normalize_features(df, method=normalize, exclude_cols=exclude_from_norm)

        # 8. Feature grouping
        feature_groups = self.create_feature_groups(df)

        # Summary
        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETE")
        print("=" * 70)
        print(f"Initial shape: {initial_shape}")
        print(f"Final shape: {df.shape}")
        print(f"Rows retained: {df.shape[0] / initial_shape[0] * 100:.1f}%")
        print(f"Features created: {df.shape[1] - initial_shape[1]}")

        metadata = {
            'initial_shape': initial_shape,
            'final_shape': df.shape,
            'feature_groups': feature_groups,
            'scalers': self.scalers,
            'imputers': self.imputers,
            'categorical_mappings': self.categorical_mappings,
            'statistics': stats
        }

        return df, metadata

    def save_processed_data(self, df: pd.DataFrame,
                            metadata: Dict,
                            output_dir: str = './processed_data/'):
        """Save processed data and metadata"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save processed dataframe
        df.to_csv(f'{output_dir}processed_pyrochlore.csv', index=False)
        print(f"\nSaved processed data to: {output_dir}processed_pyrochlore.csv")

        # Save metadata
        import json
        metadata_clean = {k: v for k, v in metadata.items()
                          if k not in ['scalers', 'imputers']}

        with open(f'{output_dir}preprocessing_metadata.json', 'w') as f:
            json.dump(metadata_clean, f, indent=2, default=str)
        print(f"Saved metadata to: {output_dir}preprocessing_metadata.json")

        # Save feature groups
        feature_groups = metadata['feature_groups']
        with open(f'{output_dir}feature_groups.json', 'w') as f:
            json.dump(feature_groups, f, indent=2)
        print(f"Saved feature groups to: {output_dir}feature_groups.json")

        return output_dir


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":


    # Configuration
    INPUT_FILE = 'pyrochlore_dataset.csv'  # Update with your filename
    OUTPUT_DIR = './processed_data/'

    # Initialize preprocessor
    preprocessor = PyrochlorePreprocessor()

    # Run full pipeline
    df_processed, metadata = preprocessor.full_preprocessing_pipeline(
        filepath=INPUT_FILE,
        handle_missing='mixed',
        remove_outliers=True,
        outlier_method='iqr',
        normalize='standard',
        add_features=True
    )

    # Save outputs
    preprocessor.save_processed_data(df_processed, metadata, OUTPUT_DIR)

    # Display sample of processed data
    print("\n" + "=" * 70)
    print("SAMPLE OF PROCESSED DATA (first 5 rows, selected columns)")
    print("=" * 70)

    sample_cols = [col for col in df_processed.columns
                   if any(x in col for x in ['entropy', 'radius', 'thermal', 'density'])]
    print(df_processed[sample_cols[:10]].head().to_string())
