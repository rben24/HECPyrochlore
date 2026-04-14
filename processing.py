# Run feature engineering standalone
from feature_engineering import add_engineered_features
import pandas as pd

df = pd.read_csv('pyrochlore_dataset.csv')
df_enhanced = add_engineered_features(df)
df_enhanced.to_csv('dataset_with_features.csv', index=False)

### **Step 2: Full Preprocessing Pipeline**

# Run complete preprocessing
from preprocessing import PyrochlorePreprocessor

preprocessor = PyrochlorePreprocessor()

df\_processed, metadata = preprocessor.full\_preprocessing\_pipeline(
    filepath='pyrochlore\_dataset.csv',
    handle\_missing='mixed',
    remove\_outliers=True,
    outlier\_method='iqr',
    normalize='standard',
    add\_features=True
)

# Save results
preprocessor.save\_processed\_data(df\_processed, metadata