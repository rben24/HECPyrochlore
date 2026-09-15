"""
load_mp.py
==========
Parses the Materials Project pyrochlore query dataset
(mp_pyrochlore_query.csv) into the canonical combined-dataset schema.

Key responsibilities
--------------------
1. Use pymatgen's Composition to parse and normalize formula strings.
2. Assign A-site / B-site elements using KNOWN_A / KNOWN_B sets.
3. Classify every entry as pristine, high_entropy, or non_pyrochlore.
4. Return a DataFrame in the canonical schema ready for merge into
   combined_pyrochlore.csv.

Pyrochlore sanity checks
------------------------
  * Space group must be Fd-3m (No. 227)
  * Crystal system must be Cubic
  * Formula must reduce to A2B2O7 stoichiometry (after pymatgen normalization)
  * All cation elements must be in KNOWN_A or KNOWN_B (or Ce, handled separately)
  * Lattice parameter must be in [LATTICE_MIN, LATTICE_MAX] Å

MP composition note
-------------------
The `composition` column stores the primitive-cell formula (e.g. "Tm4 Pt4 O14"),
while `formula_pretty` stores the reduced formula (e.g. "Tm2Pt2O7").
We use pymatgen's Composition on `formula_pretty` for normalization and
element extraction.
"""

from __future__ import annotations

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from pymatgen.core import Composition, Element

from src import globals

log = logging.getLogger(__name__)

# ── pyrochlore check helpers ─────────────────────────────────────────────────

sg_err = []
latt_err = []
formula_err = []
a_b_err = []
err_c = []
unk_err = []
def _classify_mp(
    formula_pretty: str,
    space_group_number: int,
    crystal_system: str,
    lattice_a: float,
) -> Tuple[str, Dict, Dict, Dict, Tuple]:
    """
    Classify an MP entry and return (compound_type, a_comp, b_comp, unknown).
    """
    # --- space group / crystal system check ---
    try:
        sg = int(space_group_number)
    except (TypeError, ValueError):
        err_c.append(1)
        return globals.NON_PYROCHLORE, {}, {}, {}, ()

    if sg != 227 or str(crystal_system).strip().lower() != 'cubic':
        sg_err.append(sg)
        return globals.NON_PYROCHLORE, {}, {}, {}, ()

    # --- lattice parameter check ---
    try:
        a = float(lattice_a)
    except (TypeError, ValueError):
        err_c.append(2)
        return globals.NON_PYROCHLORE, {}, {}, {}, ()

    if not (globals.LATTICE_MIN <= a <= globals.LATTICE_MAX):
        latt_err.append(a)
        return globals.NON_PYROCHLORE, {}, {}, {}, ()

    # --- formula check via pymatgen ---
    try:
        comp = Composition(str(formula_pretty))
    except Exception:
        err_c.append(3)
        return globals.NON_PYROCHLORE, {}, {}, {}, ()

    if not globals.is_pyrochlore_formula(comp):
        formula_err.append(comp)
        return globals.NON_PYROCHLORE, {}, {}, {}, ()

    # --- site assignment ---
    # a_comp, b_comp, unknown = _assign_sites(comp)
    a_comp, b_comp, unknown, oxi_states = globals.assign_sites(comp)

    if unknown:
        unk_err.append(unknown)
        return globals.NON_PYROCHLORE, a_comp, b_comp, unknown, oxi_states

    if not a_comp or not b_comp:
        a_b_err.append(comp.reduced_composition)
        return globals.NON_PYROCHLORE, a_comp, b_comp, unknown, oxi_states

    # --- pristine vs high_entropy ---
    if len(a_comp) == 1 and len(b_comp) == 1:
        return globals.PRISTINE, a_comp, b_comp, unknown, oxi_states

    return globals.HIGH_ENTROPY, a_comp, b_comp, unknown, oxi_states


# ── main loader ──────────────────────────────────────────────────────────────

def load_mp(
    filepath: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load and parse the Materials Project pyrochlore CSV.

    Parameters
    ----------
    filepath : path to ``mp_pyrochlore_query.csv``
               (defaults to ``data/raw/mp_pyrochlore_query.csv``)
    verbose  : print a summary table

    Returns
    -------
    DataFrame in the canonical combined-dataset schema containing only
    pyrochlore entries (pristine + high_entropy).

    Extra columns
    -------------
    compound_type       : 'pristine' or 'high_entropy'
    mp_id               : Materials Project material_id
    band_gap            : eV (from MP)
    energy_above_hull   : eV/atom (from MP)
    formation_energy_per_atom : eV/atom (from MP)
    a_stoich_json       : JSON {element: mole_fraction} for A-site
    b_stoich_json       : JSON {element: mole_fraction} for B-site
    """
    if filepath is None:
        _HERE = Path(__file__).resolve().parent
        _PROJECT = _HERE.parent.parent
        filepath = _PROJECT / 'data' / 'raw' / 'mp_pyrochlore_query.csv'
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"MP dataset not found at {filepath}.\n"
            "Place the file at data/raw/mp_pyrochlore_query.csv "
            "or pass the path explicitly."
        )

    df_raw = pd.read_csv(filepath)

    if verbose:
        log.info(f"MP: {len(df_raw)} raw rows loaded from {filepath.name}")

    records = []
    n_non_pyro = 0

    for _, row in df_raw.iterrows():
        ctype, a_comp, b_comp, unknown, oxi_state = _classify_mp(
            formula_pretty=row.get('formula_pretty', ''),
            space_group_number=row.get('space_group_number', -1),
            crystal_system=row.get('crystal_system', ''),
            lattice_a=row.get('a_lattice', np.nan),
        )

        if ctype == globals.NON_PYROCHLORE:
            n_non_pyro += 1
            continue

        # get band gap type
        band_type = 'direct' if str(row.get('is_gap_direct')).upper == 'TRUE' else 'indirect'

        # get bulk and shear modulus VRH
        bulk = row.get('bulk_modulus')
        bulk_vrh = bulk.get('vrh') if isinstance(bulk, dict) else np.nan

        shear = row.get('shear_modulus')
        shear_vrh = shear.get('vrh') if isinstance(shear, dict) else np.nan

        # Canonical Sample A / B strings (sorted element symbols)
        sample_a = ','.join(sorted(a_comp.keys()))
        sample_b = ','.join(sorted(b_comp.keys()))

        # Oxidation states of A site and B site
        oxi_a = oxi_state[0] if oxi_state is not None else np.nan
        oxi_b = oxi_state[1] if oxi_state is not None else np.nan

        # Pretty formula from pymatgen for consistency
        try:
            pretty = Composition(str(row['formula_pretty'])).reduced_formula
        except Exception:
            pretty = str(row.get('formula_pretty', ''))

        records.append({
            'Composition':                      pretty,
            'Sample A':                         sample_a,
            'Sample B':                         sample_b,
            'Oxidation State A':                oxi_a,
            'Oxidation State B':                oxi_b,
            'Thermal Conductivity (W/m/K)':     np.nan,
            'Lattice Parameter (Å)':     float(row.get('a_lattice', np.nan)),
            'Relative Density %':               np.nan,
            'Is Single Phase':                  'Yes',
            'Synthesis Method':                 'DFT',
            'data_source':                      'mp',
            'b_o_distance':                     np.nan,
            'b_o_b_angle':                      np.nan,
            'oxygen_param_x':                   np.nan,
            'compound_type':                    ctype,
            # 'mp_id':                            str(row.get('material_id', '')),
            'Energy per Atom':                  row.get('energy_per_atom', np.nan),
            'Formation Energy per Atom':        row.get('formation_energy_per_atom', np.nan),
            'Enthalpy':                         row.get('enthalpy_atom', np.nan),
            'Band Gap':                         row.get('band_gap', np.nan),
            'Band Gap Type':                    band_type,
            'Bulk Modulus (VRH)':               bulk_vrh,
            'Shear Modulus (VRH)':              shear_vrh,
            'Youngs Modulus (VRH)':             np.nan,
            'Poisson Ratio':                    row.get('homogeneous_poisson', np.nan),
            'AEL Debye Temperature':            np.nan,
            'Temperature':                      np.nan,
            'Thermal Expansion':                np.nan,
            'Energy Above Hull':                row.get('energy_above_hull', np.nan),
            'Density':                          row.get('density', np.nan),
            'Magnetic Moment':                  row.get('total_magnetization', np.nan),
            'Valence':                          np.nan,
            'a_stoich_json':                    json.dumps(a_comp),
            'b_stoich_json':                    json.dumps(b_comp),
        })

    if verbose:
        print(f"Formula error({len(formula_err)}: {formula_err}")
        print(f"A B error({len(a_b_err)}: {a_b_err}")
        print(f"Lattice error({len(latt_err)}: {latt_err}")
        print(f"SG error({len(sg_err)}: {sg_err}")
        print(f"Unkown error({len(unk_err)}): {unk_err}")
        print(f"{len(err_c)} load err")

        log.info(
            f"MP: {n_non_pyro} entries excluded as non-pyrochlore "
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

    return df


# ── standalone test ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import logging
    logging.basicConfig(level=logging.INFO, format='  [%(levelname)s] %(message)s')
    fp = sys.argv[1] if len(sys.argv) > 1 else None
    result = load_mp(filepath=fp, verbose=True)
    print(result[[
        'Composition', 'Sample A', 'Sample B',
        'Lattice Parameter (Å)', 'compound_type', #'mp_id'
    ]].head(20).to_string(index=False))
    print(f"\nTotal rows: {len(result)}")
