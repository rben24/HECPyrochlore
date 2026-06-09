import pandas as pd
import json
import csv
import sys
from pathlib import Path

_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent
DATA_DIR  = _PROJECT / 'data' / 'raw'
SMALL_FILE = DATA_DIR / 'aflow_pyrochlore_data_thermal.csv'
LARGE_FILE = DATA_DIR / 'aflow_pyrochlore_data.csv'
EXTRA_FILE = DATA_DIR / 'aflow_pyrochlore_data_extra.csv'
OUTPUT_FILE = DATA_DIR / 'aflow_pyrochlore_data_comb.csv'

# Load your datasets
df_large = pd.read_csv(LARGE_FILE)
df_small = pd.read_csv(SMALL_FILE)
df_extra = pd.read_csv(EXTRA_FILE)

SAME_COLUMNS = [
    'compound', 'spin_atom', 'aflow_prototype_params_list_relax',
    'aflow_prototype_params_values_relax', 'auid', 'aurl',
]

SAME_EXTRA = [
    'valence_cell_iupac', 'spinD', 'Egap', 'Egap_type',
]

# Option 1: LEFT JOIN (keep all rows from larger dataset)
# This preserves all rows from the larger dataset and adds columns from smaller dataset
combined_df = df_extra.merge(
    df_small,
    on=SAME_EXTRA + SAME_COLUMNS,
    how='left'
)
combined_df_2 = df_large.merge(
    combined_df,
    on=SAME_COLUMNS,  # Replace with your actual common columns
    how='left'
)


# Save the result
combined_df_2.to_csv(OUTPUT_FILE, index=False)

