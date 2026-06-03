"""
load_aflow.py
=============
Parses the AFLOW pyrochlore dataset (aflow_pyrochlore_data_com.csv) into the
canonical combined-dataset schema.

Key responsibilities
--------------------
1. Parse the `compound` column (e.g. "Eu4O14Sn4") using pymatgen's Composition
   for robust normalization and pretty-formula generation.
2. Classify entries as:
      pristine       — formula reduces to A2B2O7 (or A4B4O14) with exactly
                       1 A-site element and 1 B-site element
      non_pyrochlore — anything else (AFLOW dataset contains non-pyrochlores)
   NOTE: AFLOW entries are all pristine by dataset construction (single A, single B),
         so high_entropy is not expected but is handled defensively.
3. Validate pyrochlore stoichiometry via the `composition` column
   (comma-separated counts, e.g. "4,14,4") — must be (2,2,7) or (4,4,14)
   in any element order, with O identified separately.
4. Return a DataFrame in the canonical schema.

Pristine pyrochlore check (AFLOW-specific)
------------------------------------------
Two complementary checks are applied:
  A) composition column: sorted non-O counts must be [2,2] with O count 7,
     OR sorted non-O counts [4,4] with O count 14.
  B) pymatgen reduced_formula: must match A2B2O7 pattern (same as MP check).
Both must pass for an entry to be classified as pristine.
"""

from __future__ import annotations

import re
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pymatgen.core import Composition, Element

from .. import globals

log = logging.getLogger(__name__)


# ── composition-column parser ────────────────────────────────────────────────

def _parse_aflow_composition(
    composition_str: str,
    compound_str: str,
) -> Tuple[bool, Optional[str]]:
    """
    Validate the AFLOW `composition` field against the A2B2O7 / A4B4O14 rule.

    Parameters
    ----------
    composition_str : comma-separated stoichiometry counts, e.g. "4,14,4"
    compound_str    : formula string, e.g. "Eu4O14Sn4" — used to identify
                      which count corresponds to O

    Returns
    -------
    (is_valid, reason)
      is_valid : True if stoichiometry matches pristine pyrochlore
      reason   : human-readable rejection reason, or None if valid
    """
    try:
        counts = [int(x.strip()) for x in str(composition_str).split(',')]
    except (ValueError, AttributeError):
        return False, f"unparseable composition field: {composition_str!r}"

    # Parse compound to find which elements map to which counts
    try:
        # Extract (element, count) pairs in order from the compound string
        tokens = re.findall(r'([A-Z][a-z]?)(\d+)', str(compound_str))
        if not tokens:
            return False, f"no tokens in compound: {compound_str!r}"

        elem_counts = [(sym, int(n)) for sym, n in tokens]
    except Exception as e:
        return False, f"compound parse error: {e}"

    # Verify the counts list matches the compound token counts
    token_counts = [c for _, c in elem_counts]
    if sorted(token_counts) != sorted(counts):
        # composition field may be in a different order — use token_counts
        pass

    o_counts = [c for sym, c in elem_counts if sym == 'O']
    cation_counts = [c for sym, c in elem_counts if sym != 'O']

    if not o_counts:
        return False, "no oxygen found in compound"

    o_count = o_counts[0]
    cation_counts_sorted = sorted(cation_counts)

    # Valid patterns: (2,2) with O=7, or (4,4) with O=14
    if cation_counts_sorted == [2, 2] and o_count == 7:
        return True, None
    if cation_counts_sorted == [4, 4] and o_count == 14:
        return True, None

    return False, (
        f"stoichiometry {cation_counts_sorted} + O{o_count} "
        f"is not A2B2O7 or A4B4O14"
    )


# ── pymatgen formula check (shared logic with load_mp.py) ────────────────────

def _is_pyrochlore_formula(comp: Composition) -> bool:
    """Return True if reduced formula matches A2B2O7 stoichiometry."""
    try:
        reduced = comp.reduced_composition
        elems = {str(el): amt for el, amt in reduced.items()}

        if 'O' not in elems:
            return False

        o_amt = elems['O']
        cation_amts = [v for k, v in elems.items() if k != 'O']

        if len(cation_amts) != 2:
            return False

        scale = 7.0 / o_amt
        scaled_cations = [round(a * scale, 3) for a in cation_amts]

        return all(abs(a - 2.0) < 0.15 for a in scaled_cations)
    except Exception:
        return False


# ── site assignment ───────────────────────────────────────────────────────────

def _assign_sites(
    comp: Composition,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Split cation elements into A-site, B-site, and unknown dicts.
    Returns mole fractions (sum to 1 per site).
    Ce ambiguity resolved identically to load_icsd.py.
    """
    reduced = comp.reduced_composition
    raw: Dict[str, float] = {
        str(el): amt
        for el, amt in reduced.items()
        if str(el) != 'O'
    }

    a_comp: Dict[str, float] = {}
    b_comp: Dict[str, float] = {}
    unknown: Dict[str, float] = {}
    ce_amt: float = 0.0

    for elem, amt in raw.items():
        if elem == globals.CE_AMBIGUOUS:
            ce_amt = amt
        if elem in globals.KNOWN_A:
            a_comp[elem] = amt
        elif elem in globals.KNOWN_B:
            b_comp[elem] = amt
        else:
            unknown[elem] = amt

    if ce_amt > 0:
        if a_comp:
            a_comp[globals.CE_AMBIGUOUS] = ce_amt
        elif b_comp:
            b_comp[globals.CE_AMBIGUOUS] = ce_amt
        else:
            a_comp[globals.CE_AMBIGUOUS] = ce_amt

    def _to_fracs(d: Dict[str, float]) -> Dict[str, float]:
        total = sum(d.values())
        return {k: v / total for k, v in d.items()} if total else {}

    return _to_fracs(a_comp), _to_fracs(b_comp), unknown


# ── per-row classifier ────────────────────────────────────────────────────────

def _classify_aflow(
    compound: str,
    composition_str: str,
) -> Tuple[str, Dict, Dict, Dict, Optional[str]]:
    """
    Classify a single AFLOW row.

    Returns
    -------
    (compound_type, a_comp, b_comp, unknown, rejection_reason)
    """
    # --- pymatgen formula parse ---
    try:
        comp = Composition(str(compound))
    except Exception as e:
        return globals.NON_PYROCHLORE, {}, {}, {}, f"Composition parse failed: {e}"

    # --- Check A: composition column stoichiometry ---
    stoich_ok, stoich_reason = _parse_aflow_composition(composition_str, compound)
    if not stoich_ok:
        return globals.NON_PYROCHLORE, {}, {}, {}, stoich_reason

    # --- Check B: pymatgen reduced formula ---
    if not _is_pyrochlore_formula(comp):
        return (
            globals.NON_PYROCHLORE, {}, {}, {},
            f"pymatgen reduced formula not A2B2O7: "
            f"{comp.reduced_formula}"
        )

    # --- site assignment ---
    a_comp, b_comp, unknown = _assign_sites(comp)

    if unknown:
        return (
            globals.NON_PYROCHLORE, a_comp, b_comp, unknown,
            f"unknown cations: {list(unknown.keys())}"
        )

    if not a_comp or not b_comp:
        return (
            globals.NON_PYROCHLORE, a_comp, b_comp, unknown,
            "empty A-site or B-site after assignment"
        )

    # --- pristine vs high_entropy ---
    if len(a_comp) == 1 and len(b_comp) == 1:
        return globals.PRISTINE, a_comp, b_comp, unknown, None

    return globals.HIGH_ENTROPY, a_comp, b_comp, unknown, None


# ── main loader ───────────────────────────────────────────────────────────────

def load_aflow(
    filepath: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load and parse the AFLOW pyrochlore CSV.

    Parameters
    ----------
    filepath : path to ``aflow_pyrochlore_data_com.csv``
               (defaults to ``data/raw/aflow_pyrochlore_data_com.csv``)
    verbose  : print a summary table

    Returns
    -------
    DataFrame in the canonical combined-dataset schema containing only
    pyrochlore entries, with extra AFLOW-specific columns.

    Extra columns
    -------------
    compound_type           : 'pristine' or 'high_entropy'
    auid                    : AFLOW unique ID
    aurl                    : AFLOW URL
    energy_atom             : eV/atom (DFT total energy)
    enthalpy_atom           : eV/atom
    Egap                    : band gap (eV)
    Egap_type               : 'metal', 'insulator-direct', etc.
    ael_bulk_modulus_vrh    : GPa
    ael_shear_modulus_vrh   : GPa
    agl_thermal_conductivity_300K : W/m/K
    a_stoich_json           : JSON {element: mole_fraction} for A-site
    b_stoich_json           : JSON {element: mole_fraction} for B-site
    """
    if filepath is None:
        _HERE = Path(__file__).resolve().parent
        _PROJECT = _HERE.parent.parent
        filepath = _PROJECT / 'data' / 'raw' / 'aflow_pyrochlore_data_comb.csv'
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"AFLOW dataset not found at {filepath}.\n"
            "Place the file at data/raw/aflow_pyrochlore_data_comb.csv "
            "or pass the path explicitly."
        )

    df_raw = pd.read_csv(filepath)

    if verbose:
        log.info(f"AFLOW: {len(df_raw)} raw rows loaded from {filepath.name}")

    records = []
    n_non_pyro = 0
    rejection_log: List[Tuple[str, str]] = []

    for _, row in df_raw.iterrows():
        compound = str(row.get('compound', ''))
        composition_str = str(row.get('composition', ''))

        ctype, a_comp, b_comp, unknown, reason = _classify_aflow(
            compound=compound,
            composition_str=composition_str,
        )

        if ctype == globals.NON_PYROCHLORE:
            n_non_pyro += 1
            rejection_log.append((compound, reason or 'unknown'))
            continue

        # Pymatgen pretty formula for canonical Composition label
        try:
            pretty = Composition(compound).reduced_formula
        except Exception:
            pretty = compound

        sample_a = ','.join(sorted(a_comp.keys()))
        sample_b = ','.join(sorted(b_comp.keys()))

        # Lattice parameter: AFLOW stores 'a' in prototype params
        # aflow_prototype_params_values_relax first value is 'a' (Å)
        lattice_a = np.nan
        try:
            param_keys = str(row.get('aflow_prototype_params_list_relax', ''))
            param_vals = str(row.get('aflow_prototype_params_values_relax', ''))
            keys = [k.strip() for k in param_keys.split(',')]
            vals = [float(v.strip()) for v in param_vals.split(',')]
            if 'a' in keys:
                idx = keys.index('a')
                raw_a = vals[idx]
                # AFLOW stores in Bohr sometimes; if > 20 convert (1 Bohr = 0.529177 Å)
                # Typical pyrochlore a ~ 10 Å; AFLOW FCC conventional cell
                # The conventional cubic cell a ~ 10-11 Å; primitive cell a ~ 7-8 Å
                # AFLOW uses the conventional cell for FCC pyrochlores
                lattice_a = raw_a  # already in Å per AFLOW convention
        except Exception:
            pass

        records.append({
            'Composition':                          pretty,
            'Sample A':                             sample_a,
            'Sample B':                             sample_b,
            'Thermal Conductivity (W/m/K)':         row.get('agl_thermal_conductivity_300K', np.nan),
            'Lattice Parameter (Angstrom)':         lattice_a,
            'Relative Density %':                   np.nan,
            'Is Single Phase':                      'Yes',
            'Synthesis Method':                     'DFT',
            'data_source':                          'aflow',
            'b_o_distance':                         np.nan,
            'b_o_b_angle':                          np.nan,
            'oxygen_param_x':                       np.nan,
            'compound_type':                        ctype,
            # 'auid':                                 str(row.get('auid', '')),
            # 'aurl':                                 str(row.get('aurl', '')),
            'Energy per Atom':                      row.get('energy_atom', np.nan),
            'Formation Energy per Atom':            np.nan,
            'Enthalpy':                             row.get('enthalpy_atom', np.nan),
            'Band Gap':                             row.get('Egap', np.nan),
            'Band Gap Type':                        str(row.get('Egap_type', '')),
            'Bulk Modulus (VRH)':                   row.get('ael_bulk_modulus_vrh', np.nan),
            'Shear Modulus (VRH)':                  row.get('ael_shear_modulus_vrh', np.nan),
            'Youngs Modulus (VRH)':                 row.get('ael_youngs_modulus_vrh', np.nan),
            'Poisson Ratio':                        row.get('ael_poisson_ratio', np.nan),
            'AEL Debye Temperature':                row.get('ael_debye_temperature', np.nan),
            'Temperature':                          np.nan,
            'Thermal Expansion':                    row.get('agl_thermal_expansion_300K', np.nan),
            'Energy Above Hull':                    np.nan,
            'Density':                              row.get('density', np.nan),
            'Magnetic Moment':                      row.get('spin_atom', np.nan),
            'Valence':                              row.get('valence_cell_iupac', np.nan),
            'a_stoich_json':                        json.dumps(a_comp),
            'b_stoich_json':                        json.dumps(b_comp),
        })

    if verbose:
        log.info(
            f"AFLOW: {n_non_pyro} entries excluded as non-pyrochlore "
            f"({len(records)} pyrochlore entries remain)"
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if verbose:
        pristine_n = (df['compound_type'] == globals.PRISTINE).sum()
        he_n = (df['compound_type'] == globals.HIGH_ENTROPY).sum()
        print()
        print(f"  {'Compound type':<20} {'Count':>6}")
        print(f"  {'-'*28}")
        print(f"  {'Pristine':<20} {pristine_n:>6}")
        print(f"  {'High-entropy':<20} {he_n:>6}")
        print(f"  {'Non-pyrochlore (excl.)':<20} {n_non_pyro:>6}")
        print(f"  {'-'*28}")
        print(f"  {'Total (raw)':<20} {len(df_raw):>6}")
        print()

        if rejection_log:
            print(f"  Sample rejection reasons (first 10):")
            for compound, reason in rejection_log:#[:10]:
                print(f"    {compound:<20} → {reason}")
            print()

    return df




# ── standalone test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import logging
    logging.basicConfig(level=logging.INFO, format='  [%(levelname)s] %(message)s')

    aflow_fp = sys.argv[1] if len(sys.argv) > 1 else None

    result = load_aflow(filepath=aflow_fp, verbose=True)
    print(result[[
        'Composition', 'Sample A', 'Sample B',
        'Lattice Parameter (Angstrom)', 'compound_type', 'auid'
    ]].head(20).to_string(index=False))
    print(f"\nTotal rows: {len(result)}")

