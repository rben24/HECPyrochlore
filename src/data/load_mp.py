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

def _is_pyrochlore_formula(comp: Composition) -> bool:
    """
    Return True if the reduced formula matches A2B2O7 stoichiometry.

    Strategy: reduce to the smallest integer ratio, then check that:
      - exactly 3 distinct species are present
      - O has stoichiometry 7 in the reduced formula
      - the two cation species have stoichiometry 2 each
    """
    try:
        reduced = comp.reduced_composition
        elems = {str(el): amt for el, amt in reduced.items()}

        if 'O' not in elems:
            return False

        o_amt = elems['O']
        cation_amts = [v for k, v in elems.items() if k != 'O']

        if len(cation_amts) != 2:
            return False

        # Normalize so O → 7
        scale = 7.0 / o_amt
        scaled_cations = [round(a * scale, 3) for a in cation_amts]

        return all(abs(a - 2.0) < 0.15 for a in scaled_cations)
    except Exception:
        return False



def _assign_sites(
    comp: Composition,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]] | None:
    """
    Split cation elements into A-site (3+), B-site (4+), and unknown dicts.
    Stoichiometries are mole fractions (sum to 1 per site).

    Assignment priority (per element)
    ----------------------------------
    1. Oxygen is always skipped.
    2. Ce is held aside and resolved last (see Ce note below).
    3. For every other cation, pymatgen's ``Element.common_oxidation_states``
       is queried:
         - if +3 is the *only* common state, or +3 is present and +4 is not
           → A-site
         - if +4 is the *only* common state, or +4 is present and +3 is not
           → B-site
         - if *both* +3 and +4 are common (e.g. Mn, Ir, Ru, Os, …):
           → prefer the KNOWN_A / KNOWN_B hardcoded sets as a tiebreaker
           → if still ambiguous, fall back to the *most common* oxidation
             state (index 0 in pymatgen's tuple, which is sorted by
             prevalence)
         - if neither +3 nor +4 appears in common states
           → check KNOWN_A / KNOWN_B as a last resort
           → otherwise → unknown
    4. Ce resolution:
         - Ce has common states +3 and +4, so it is genuinely ambiguous.
         - If other A-site elements are already present → Ce goes to A-site.
         - If only B-site elements are present → Ce goes to B-site.
         - If Ce is the sole cation → default A-site (Ce³⁺ is more common
           in pyrochlore literature).

    Parameters
    ----------
    comp : pymatgen Composition (need not be reduced; reduced internally)

    Returns
    -------
    a_comp   : {element_symbol: mole_fraction}  A-site (sums to 1)
    b_comp   : {element_symbol: mole_fraction}  B-site (sums to 1)
    unknown  : {element_symbol: mole_fraction}  unassigned cations
               (mole fractions relative to *all* cations, not per-site)
    """
    '''reduced = comp.reduced_composition
    raw: Dict[str, float] = {
        str(el): amt
        for el, amt in reduced.items()
        if str(el) != 'O'
    }

    a_comp:  Dict[str, float] = {}
    b_comp:  Dict[str, float] = {}
    unknown: Dict[str, float] = {}
    ce_amt:  float = 0.0

    for sym, amt in raw.items():
        # ── Ce is always deferred ────────────────────────────────────────────
        if sym == globals.CE_AMBIGUOUS:
            ce_amt = amt
            continue

        try:
            el = Element(sym)
            ox_states: Tuple[int, ...] = el.common_oxidation_states  # e.g. (3,) or (2,3,4)
        except Exception:
            # pymatgen doesn't know this element → unknown
            unknown[sym] = amt
            continue

        has3 = 3 in ox_states
        has4 = 4 in ox_states

        if has3 and not has4:
            a_comp[sym] = amt

        elif has4 and not has3:
            b_comp[sym] = amt

        elif has3 and has4:
            # Tiebreaker 1: hardcoded sets
            in_a = sym in globals.KNOWN_A
            in_b = sym in globals.KNOWN_B
            if in_a and not in_b:
                a_comp[sym] = amt
            elif in_b and not in_a:
                b_comp[sym] = amt
            else:
                # Tiebreaker 2: most prevalent oxidation state
                # pymatgen orders common_oxidation_states by prevalence
                if ox_states[0] == 3:
                    a_comp[sym] = amt
                elif ox_states[0] == 4:
                    b_comp[sym] = amt
                else:
                    # Still ambiguous → unknown
                    unknown[sym] = amt

        else:
            # Neither +3 nor +4 in common states
            # Last resort: hardcoded sets
            if sym in globals.KNOWN_A:
                a_comp[sym] = amt
            elif sym in globals.KNOWN_B:
                b_comp[sym] = amt
            else:
                unknown[sym] = amt

    # ── Resolve Ce ───────────────────────────────────────────────────────────
    if ce_amt > 0:
        if a_comp:
            # Other lanthanides / 3+ cations present → Ce³⁺ on A-site
            a_comp[globals.CE_AMBIGUOUS] = ce_amt
        elif b_comp:
            # Only 4+ cations present → Ce⁴⁺ on B-site
            b_comp[globals.CE_AMBIGUOUS] = ce_amt
        else:
            # Ce is the sole cation → default Ce³⁺ A-site
            a_comp[globals.CE_AMBIGUOUS] = ce_amt
    '''
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
        if elem == globals.KNOWN_AMBIGUOUS:
            ce_amt = amt
        elif elem in globals.KNOWN_A:
            a_comp[elem] = amt
        elif elem in globals.KNOWN_B:
            b_comp[elem] = amt
        else:
            unknown[elem] = amt

    if ce_amt > 0:
        print(f"=====================================AMBIGUOUS")
        print(f"{ce_amt} ")
        if a_comp:
            a_comp[globals.KNOWN_AMBIGUOUS] = ce_amt
        elif b_comp:
            b_comp[globals.KNOWN_AMBIGUOUS] = ce_amt
        else:
            a_comp[globals.KNOWN_AMBIGUOUS] = ce_amt

    def _to_fracs(d: Dict[str, float]) -> Dict[str, float]:
        total = sum(d.values())
        return {k: v / total for k, v in d.items()} if total else {}

    return _to_fracs(a_comp), _to_fracs(b_comp), unknown

    '''
    # ── Normalise to per-site mole fractions ─────────────────────────────────
    def _to_fracs(d: Dict[str, float]) -> Dict[str, float]:
        total = sum(d.values())
        return {k: v / total for k, v in d.items()} if total else {}

    # unknown is returned as fractions of *all* cations for diagnostic use
    all_cation_total = sum(raw.values())
    unknown_fracs = (
        {k: v / all_cation_total for k, v in unknown.items()}
        if all_cation_total else {}
    )

    return _to_fracs(a_comp), _to_fracs(b_comp), unknown_fracs
    '''


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

    if not _is_pyrochlore_formula(comp):
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
