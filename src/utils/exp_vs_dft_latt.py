# Requires: pandas
# Python script to take two DataFrames (or CSVs) and return processed DataFrames
# containing only compositions present in both inputs, with averaged lattice parameters.

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from src.data.make_combined_dataset import (load_icsd_source, load_aflow_source,
                                                load_mp_source)
from src.build_models.train_model import plot_parity

_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent
SAVE_DIR  = _PROJECT / 'src' / 'utils'

def process_and_match(df1, df2,
                      composition_col='Composition',
                      lattice_col='Lattice Parameter (Angstrom)'):
    """
    Return two DataFrames (df1_out, df2_out) that:
      - contain only rows for compositions present in both input DataFrames
      - have one row per composition with the averaged lattice parameter
      - index is the composition column
    Parameters:
      df1, df2: pandas.DataFrame inputs (already loaded)
      composition_col: column name for composition (string)
      lattice_col: column name for lattice parameter (string)
    """
    # Make copies to avoid mutating inputs
    a = df1[[composition_col, lattice_col]].copy()
    b = df2[[composition_col, lattice_col]].copy()

    # Normalize composition strings (strip whitespace). Adjust if needed.
    a[composition_col] = a[composition_col].astype(str).str.strip()
    b[composition_col] = b[composition_col].astype(str).str.strip()

    # Drop rows with missing composition or lattice parameter
    a = a.dropna(subset=[composition_col, lattice_col])
    b = b.dropna(subset=[composition_col, lattice_col])

    # Convert lattice parameter to numeric (coerce errors to NaN then drop)
    a[lattice_col] = pd.to_numeric(a[lattice_col], errors='coerce')
    b[lattice_col] = pd.to_numeric(b[lattice_col], errors='coerce')
    a = a.dropna(subset=[lattice_col])
    b = b.dropna(subset=[lattice_col])

    # Group by composition and average lattice parameter (if duplicates exist)
    a_avg = a.groupby(composition_col, as_index=False)[lattice_col].mean()
    b_avg = b.groupby(composition_col, as_index=False)[lattice_col].mean()

    # Find intersection of compositions
    common = set(a_avg[composition_col]).intersection(b_avg[composition_col])

    # Filter to only common compositions and sort (optional)
    a_out = a_avg[a_avg[composition_col].isin(common)].sort_values(by=composition_col).reset_index(drop=True)
    b_out = b_avg[b_avg[composition_col].isin(common)].sort_values(by=composition_col).reset_index(drop=True)

    # Set composition as index for convenience
    a_out = a_out.set_index(composition_col)
    b_out = b_out.set_index(composition_col)

    return a_out, b_out


# Example usage:
if __name__ == '__main__':
    # Example: load from CSV files. Replace paths as needed.
    df1 = load_icsd_source()
    df2 = load_aflow_source()
    df3 = load_mp_source()

    # If your composition column has a different name, pass composition_col='YourColumnName'
    df1_proc, df2_proc = process_and_match(df1, df2,
                                           composition_col='Composition',
                                           lattice_col='Lattice Parameter (Angstrom)')
    # Save processed outputs (optional)
    # df1_proc.to_csv('dataset1_processed.csv')
    # df2_proc.to_csv('dataset2_processed.csv')

    latt_col = 'Lattice Parameter (Angstrom)'
    # Quick scatter plot of matched lattice parameters
    # plot_parity(df1_proc[latt_col], df2_proc[latt_col], 'ICSD vs AFLOW Lattice Parameters', 'AFLOW lattice param',
    #             SAVE_DIR / 'icsd_vs_aflow_latt.png', 'ICSD Lattice Parameter')
    #
    # df1_proc, df3_proc = process_and_match(df1, df3)
    # plot_parity(df1_proc[latt_col], df3_proc[latt_col], 'ICSD vs MP Lattice Parameters', 'MP lattice param',
    #             SAVE_DIR / 'icsd_vs_mp_latt.png', 'ICSD Lattice Parameter')

    model = LinearRegression()
    x = df1_proc[[latt_col]]
    model.fit(x, df2_proc[latt_col].values)
    y_pred = model.predict(x)
    r2 = r2_score(df2_proc[latt_col], y_pred)
    plt.scatter(df1_proc[latt_col], df2_proc[latt_col], color='blue', label='Data Points')  # Scatter plot
    plt.plot(df1_proc[latt_col], y_pred, color='red', label='Regression Line')  # Regression line
    plt.xlabel('ICSD Lattice Parameter')
    plt.ylabel('AFLOW lattice param')
    plt.title(f'ICSD vs AFLOW Lattice Parameters (r2={r2:.2f})')
    plt.legend()
    plt.savefig(SAVE_DIR / 'icsd_vs_aflow_r2.png')
    plt.show()
    # try:
    #     import matplotlib.pyplot as plt
    #     # Align indices to ensure same order
    #     df1_proc_aligned = df1_proc.sort_index()
    #     df2_proc_aligned = df2_proc.reindex(df1_proc_aligned.index)
    #
    #     plt.scatter(df1_proc_aligned['Lattice Parameter (Angstrom)'],
    #                 df2_proc_aligned['Lattice Parameter (Angstrom)'])
    #     plt.xlabel('Dataset1 Lattice Parameter (Angstrom)')
    #     plt.ylabel('Dataset2 Lattice Parameter (Angstrom)')
    #     plt.title('Matched compositions: lattice parameter comparison')
    #     plt.grid(True)
    #     plt.show()
    # except Exception:
    #     pass
