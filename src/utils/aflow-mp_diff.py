"""
compare_datasets.py
===================
Compares the pyrochlore compositions in the Materials Project and AFLOW
datasets by normalizing formulas via pymatgen and printing the symmetric
difference.

Only the formula columns are loaded from each CSV:
  - AFLOW : ``compound``
  - MP    : ``formula_pretty``

Usage
-----
    python compare_datasets.py
    python compare_datasets.py path/to/aflow.csv path/to/mp.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from pymatgen.core import Composition


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_reduced_formula(formula: str) -> str | None:
    """
    Return pymatgen's reduced_formula for *formula*, or None on parse failure.
    e.g. "Eu4O14Sn4" → "Eu2Sn2O7"
         "Tm2Pt2O7"  → "Tm2Pt2O7"
    """
    try:
        return Composition(str(formula)).reduced_formula
    except Exception:
        return None


# ── main comparison ───────────────────────────────────────────────────────────

def count_mp_not_in_aflow(
    aflow_path: str | Path,
    mp_path:    str | Path,
) -> Tuple[int, list[str]]:
    """
    Compare normalized pyrochlore formulas between the AFLOW and MP datasets.

    Parameters
    ----------
    aflow_path : path to ``aflow_pyrochlore_data_com.csv``
    mp_path    : path to ``mp_pyrochlore_query.csv``

    Returns
    -------
    (count, formulas)
      count    : number of MP formulas not found in AFLOW
      formulas : sorted list of those MP-only reduced formulas
    """
    # ── load only the formula columns ────────────────────────────────────────
    aflow_df = pd.read_csv(aflow_path, usecols=['compound'])
    mp_df    = pd.read_csv(mp_path,    usecols=['formula_pretty'])

    # ── normalize via pymatgen ────────────────────────────────────────────────
    aflow_df['reduced'] = aflow_df['compound'].apply(_safe_reduced_formula)
    mp_df['reduced']    = mp_df['formula_pretty'].apply(_safe_reduced_formula)

    # ── report parse failures ─────────────────────────────────────────────────
    aflow_bad = aflow_df[aflow_df['reduced'].isna()]
    mp_bad    = mp_df[mp_df['reduced'].isna()]

    if not aflow_bad.empty:
        print(f"[WARN] AFLOW: {len(aflow_bad)} formula(s) could not be parsed:")
        for f in aflow_bad['compound']:
            print(f"  {f}")

    if not mp_bad.empty:
        print(f"[WARN] MP: {len(mp_bad)} formula(s) could not be parsed:")
        for f in mp_bad['formula_pretty']:
            print(f"  {f}")

    # ── build sets (drop parse failures and duplicates) ───────────────────────
    aflow_set: set[str] = set(aflow_df['reduced'].dropna())
    mp_set:    set[str] = set(mp_df['reduced'].dropna())

    mp_only    = sorted(mp_set    - aflow_set)
    aflow_only = sorted(aflow_set - mp_set)
    common     = mp_set & aflow_set

    # ── print summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  AFLOW vs MP — pyrochlore formula comparison")
    print("=" * 60)
    print(f"  AFLOW unique formulas : {len(aflow_set):>5}")
    print(f"  MP    unique formulas : {len(mp_set):>5}")
    print(f"  Common to both        : {len(common):>5}")
    print(f"  In MP only            : {len(mp_only):>5}")
    print(f"  In AFLOW only         : {len(aflow_only):>5}")
    print("=" * 60)

    if mp_only:
        print(f"\n── In MP but NOT in AFLOW ({len(mp_only)}) ──────────────────")
        for i, f in enumerate(mp_only, 1):
            print(f"  {i:>4}. {f}")

    if aflow_only:
        print(f"\n── In AFLOW but NOT in MP ({len(aflow_only)}) ────────────────")
        for i, f in enumerate(aflow_only, 1):
            print(f"  {i:>4}. {f}")

    print()
    return len(mp_only), mp_only


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    _HERE    = Path(__file__).resolve().parent
    _PROJECT = _HERE.parent.parent

    aflow_default = _PROJECT / 'data' / 'raw' / 'aflow_pyrochlore_data_comb.csv'
    mp_default    = _PROJECT / 'data' / 'raw' / 'mp_pyrochlore_query.csv'

    aflow_path = Path(sys.argv[1]) if len(sys.argv) > 1 else aflow_default
    mp_path    = Path(sys.argv[2]) if len(sys.argv) > 2 else mp_default

    for p in (aflow_path, mp_path):
        if not p.exists():
            print(f"[ERROR] File not found: {p}")
            sys.exit(1)

    count, missing = count_mp_not_in_aflow(aflow_path, mp_path)
    print(f"Result: {count} MP composition(s) not found in AFLOW.")
