import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
IN_DIR  = _PROJECT / 'data' / 'processed'
INPUT_FILE = IN_DIR / 'engineered_pyrochlore_latt.csv'

# Create sample dataframe with missing values

df = pd.read_csv(INPUT_FILE)


# Create heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=True, cmap='viridis', yticklabels=False)
plt.title('Missing Values Heatmap')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.tight_layout()
plt.show()

columns_with_nan = df.columns[df.isnull().any()].tolist()
print("Columns with NaN values:", columns_with_nan)
