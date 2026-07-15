"""

Single-element site property lookups.
Accepts either a plain element symbol str (e.g. "Gd") or a pymatgen Element/Species.

Pymatgen is used opportunistically:
  - Element.X              → Pauling electronegativity (fallback to local table)
  - Species.ionic_radius   → ionic radius for a specific oxidation state (fallback to local table)

Calculates:
    Ionic Radius of Site A
    Ionic Radius of Site B
    Electronegativity of Site X


"""

# import re
# import json
from math import isnan
# import logging
import warnings
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from typing import Dict, Tuple, List, Optional
from src.globals import (IONIC_RADII_8_2, IONIC_RADII_8_3, IONIC_RADII_6_4,
                         IONIC_RADII_6_5, ELECTRONEGATIVITY)
# from src.data.make_combined_dataset import build_single_phase_dataset
try:
    from pymatgen.core import Element as PmgElement, Species as PmgSpecies
    _PYMATGEN = True
except ImportError:
    _PYMATGEN = False
    # PmgElement = None
    # PmgSpecies = None
from typing import Union

# logging.basicConfig(level=logging.INFO, format='  [%(levelname)s] %(message)s')
# log = logging.getLogger(__name__)

# _HERE    = Path(__file__).resolve().parent
# _PROJECT = _HERE.parent.parent
# RAW_DIR  = _PROJECT / 'data' / 'raw'
# OUT_DIR  = _PROJECT / 'data' / 'processed'
# OUTPUT_FILE = OUT_DIR / 'pristine_pyrochlore.csv'

# Type alias accepted by every function below
ElementLike = Union[str, "PmgElement", "PmgSpecies"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _symbol(element: ElementLike) -> str:
    """Extract the element symbol string from str / Element / Species."""
    if isinstance(element, str):
        return element
    # pymatgen Element and Species both expose .symbol
    return element.symbol


def _pmg_electronegativity(element: ElementLike) -> float | None:
    """Return Pauling electronegativity from pymatgen if available."""
    if not _PYMATGEN:
        return None
    try:
        el = element if isinstance(element, PmgElement) else PmgElement(_symbol(element))
        x = el.X          # float or None in pymatgen
        return float(x) if x is not None else None
    except Exception:
        return None


def _pmg_ionic_radius(element: ElementLike, oxidation: int, coord: int) -> float | None:
    """
    Return the ionic radius (Å) from pymatgen for the given oxidation state
    and coordination number, or None if not available.
    """
    if not _PYMATGEN:
        return None
    try:
        sp = PmgSpecies(_symbol(element), oxidation)
        radii = sp.get_crystal_field_spin_config  # not what we want — use ionic_radii dict
        # pymatgen stores per-coordination radii in Species.ionic_radii {CN: radius}
        r = sp.ionic_radii.get(coord)
        return float(r) if r is not None else None
    except Exception:
        return None


# ── Public functions ──────────────────────────────────────────────────────────

def get_ionic_radius_A(element: ElementLike, oxidation: int = 3) -> float | None:
    """
    Shannon ionic radius (Å) for an A-site cation at 8-fold coordination.

    Lookup order:
        1. IONIC_RADII_8 global table
        2. pymatgen Species (oxidation +3/+2, CN 8) — used if table misses

    Parameters
    ----------
    element : str | pymatgen Element | pymatgen Species
        e.g. "Gd", Element("Gd"), Species("Gd", 3)

    Returns
    -------
    float (Å) or None if not found in either source
    """
    sym = _symbol(element)
    if not isnan(oxidation): oxidation = int(oxidation)

    if oxidation == 3:
        r = IONIC_RADII_8_3.get(sym)
        if r is not None:
            return r
    elif oxidation == 2:
        r = IONIC_RADII_8_2.get(sym)
        if r is not None:
            return r

    # Fallback: pymatgen (assume +3 for lanthanides / typical A-site cation)
    r = _pmg_ionic_radius(element, oxidation=oxidation, coord=8)
    if r is not None:
        warnings.warn(
            f"[get_ionic_radius_A] '{sym}' not in IONIC_RADII_8; "
            f"using pymatgen value {r:.4f} Å (ox=+{oxidation}, CN=8)."
        )
        return r

    warnings.warn(f"[get_ionic_radius_A] '{sym}' not found in IONIC_RADII_8_{oxidation} or pymatgen.")
    return None


def get_ionic_radius_B(element: ElementLike, oxidation: int = 4) -> float | None:
    """
    Shannon ionic radius (Å) for a B-site cation at 6-fold coordination.

    Lookup order:
        1. IONIC_RADII_6 global table
        2. pymatgen Species (oxidation +4, CN 6) — used if table misses

    Parameters
    ----------
    element : str | pymatgen Element | pymatgen Species

    Returns
    -------
    float (Å) or None if not found in either source
    """
    sym = _symbol(element)
    if not isnan(oxidation): oxidation = int(oxidation)

    if oxidation == 4:
        r = IONIC_RADII_6_4.get(sym)
        if r is not None:
            return r
    elif oxidation == 5:
        r = IONIC_RADII_6_5.get(sym)
        if r is not None:
            return r

    # Fallback: pymatgen (assume +4 for typical B-site cation)
    r = _pmg_ionic_radius(element, oxidation=oxidation, coord=6)
    if r is not None:
        warnings.warn(
            f"[get_ionic_radius_B] '{sym}' not in IONIC_RADII_6; "
            f"using pymatgen value {r:.4f} Å (ox=+{oxidation}, CN=6)."
        )
        return r

    warnings.warn(f"[get_ionic_radius_B] '{sym}' not found in IONIC_RADII_6_{oxidation} or pymatgen.")
    return None


def get_electronegativity(element: ElementLike) -> float | None:
    """
    Pauling electronegativity for any element (A- or B-site).

    Lookup order:
        1. ELECTRONEGATIVITY global table
        2. pymatgen Element.X — used if table misses

    Parameters
    ----------
    element : str | pymatgen Element | pymatgen Species

    Returns
    -------
    float or None if not found in either source
    """
    sym = _symbol(element)

    chi = ELECTRONEGATIVITY.get(sym)
    if chi is not None:
        return chi

    # Fallback: pymatgen
    chi = _pmg_electronegativity(element)
    if chi is not None:
        warnings.warn(
            f"[get_electronegativity] '{sym}' not in ELECTRONEGATIVITY table; "
            f"using pymatgen value {chi:.4f}."
        )
        return chi

    warnings.warn(f"[get_electronegativity] '{sym}' not found in ELECTRONEGATIVITY or pymatgen.")
    return None


# def build_pristine_data(save: bool = True, file: str = OUTPUT_FILE) -> pd.DataFrame:
#     """
#     Combine all pristine/single phase pyrochlores values
#
#     Pyrochlore classification summary is printed for each source.
#     Only 'pristine' rows are retained in the output.
#     """
#     print()
#     print("=" * 66)
#     print("  Building Single Phase Pyrochlore Dataset")
#     print("=" * 66)
#
#     df = {}
#
#     if not OUTPUT_FILE.exists():
#         warnings.warn(f"[rule_of_mixtures] Pristine DB not found at:\n  {OUTPUT_FILE}\n")
#         df = build_single_phase_dataset(save=True)
#     else:
#         df = pd.read_csv(OUTPUT_FILE)
#
#     ionic_a = get_ionic_radius_A(df['Sample A'])
#     ionic_b = get_ionic_radius_B(df['Sample B'])
#     en_a = get_electronegativity(df['Sample A'])
#     en_b = get_electronegativity(df['Sample B'])
#
#     df.append({
#         'Ionic Radius A (Å)':    ionic_a,
#         'Ionic Radius B (Å)':    ionic_b,
#         'Electronegativity A':          en_a,
#         'Electronegativity B':          en_b,
#     })

