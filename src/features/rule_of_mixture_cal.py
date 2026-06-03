"""
Rule of Mixtures Calculator for High Entropy Pyrochlores
=========================================================
Composition format: (A1_x A2_y ...)(B1_p B2_q ...)O7
Example: (Sn0.3 Ho0.7)2 (Ti0.6 Hf0.4)2 O7

All single-phase pyrochlore values are stored in PYROCHLORE_DB.
Add or update entries as needed.
"""

from itertools import product
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math
import re
from pathlib import Path




# =============================================================================
# SINGLE-PHASE PYROCHLORE DATABASE
# Each entry: "A2B2O7" -> property dict
# Sources: literature values; update as needed
# =============================================================================

@dataclass
class PyrochloreProperties:
    """Properties for a single-phase A2B2O7 pyrochlore."""
    lattice_parameter: float = None      # Angstroms
    ionic_radius_A: float = None         # Angstroms (8-coord)
    ionic_radius_B: float = None         # Angstroms (6-coord)
    electronegativity_A: float = None    # Pauling scale
    electronegativity_B: float = None    # Pauling scale
    charge_A: int = 3                    # formal oxidation state
    charge_B: int = 4                    # formal oxidation state
    formation_enthalpy: float = None     # kJ/mol (if available)
    bulk_modulus: float = None           # GPa (if available)
    shear_modulus: float = None          # GPa (if available)
    thermal_conductivity: float = None   # W/m·K (if available)
    thermal_expansion: float = None      # °C⁻¹ (if available)


# fmt: off
PYROCHLORE_DB: Dict[str, PyrochloreProperties] = {
    # ---- Sn-A site pyrochlores ----
    "Sn2Ti2O7": PyrochloreProperties(
        lattice_parameter=9.926,
        ionic_radius_A=1.22,   # Sn2+ 8-coord (use Sn4+ if appropriate: 0.81)
        ionic_radius_B=0.605,  # Ti4+ 6-coord
        electronegativity_A=1.96,
        electronegativity_B=1.54,
        charge_A=2, charge_B=4,
    ),}
# fmt: on


# =============================================================================
# COMPOSITION PARSER
# Parses "(A1_x A2_y)(B1_p B2_q)O7" into structured dicts
# =============================================================================

def parse_composition(composition: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Parse a high-entropy pyrochlore composition string.

    Accepted formats:
        (Sn0.3Ho0.7)2(Ti0.6Hf0.4)2O7
        (Sn.3Ho.7)(Ti.6Hf.4)          <- fractions without leading zero also OK
        Sn0.3Ho0.7 / Ti0.6Hf0.4       <- slash-separated A/B

    Returns:
        a_site : dict  {element: fraction}   (fractions sum to ~1)
        b_site : dict  {element: fraction}
    """
    # Normalize: remove spaces, uppercase first letter of each element
    comp = composition.replace(" ", "")

    # Try parenthesis format first
    paren_pattern = re.findall(r'\(([^)]+)\)', comp)
    if len(paren_pattern) >= 2:
        a_str, b_str = paren_pattern[0], paren_pattern[1]
    elif '/' in comp:
        parts = comp.split('/')
        a_str, b_str = parts[0].strip('()'), parts[1].strip('()')
    else:
        raise ValueError(
            "Cannot parse composition. Use format: (A1xA2y)(B1pB2q) or A1x.../B1p..."
        )

    def parse_site(s: str) -> Dict[str, float]:
        # Match element symbol followed by optional decimal fraction
        tokens = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', s)
        site = {}
        for elem, frac in tokens:
            if elem:
                site[elem] = float(frac) if frac else 1.0
        # Normalize so fractions sum to 1
        total = sum(site.values())
        return {k: v / total for k, v in site.items()}

    return parse_site(a_str), parse_site(b_str)


# =============================================================================
# CORE RULE OF MIXTURES ENGINE
# =============================================================================

def rule_of_mixtures(
    a_site: Dict[str, float],
    b_site: Dict[str, float],
    prop_getter,                  # callable(PyrochloreProperties) -> float
    prop_name: str = "property",
    verbose: bool = False,
) -> float:
    """
    Core rule-of-mixtures calculation.

    For composition (A1_x A2_y)(B1_p B2_q)O7:
        ROM = Σ_{i,j} (x_i * p_j * val_ij) / Σ_{i,j} (x_i * p_j)

    Parameters
    ----------
    a_site      : {element: fraction} for A-site
    b_site      : {element: fraction} for B-site
    prop_getter : function that extracts the desired float from PyrochloreProperties
    prop_name   : label for verbose output
    verbose     : print contribution breakdown

    Returns
    -------
    Weighted average value (float), or None if no data found.
    """
    numerator = 0.0
    denominator = 0.0
    missing = []

    if verbose:
        print(f"\n  Rule of Mixtures — {prop_name}")
        print(f"  {'Phase':<20} {'Weight':>8}  {'Value':>10}  {'Contribution':>14}")
        print("  " + "-" * 58)

    for (a_elem, a_frac), (b_elem, b_frac) in product(
        a_site.items(), b_site.items()
    ):
        key = f"{a_elem}2{b_elem}2O7"
        weight = a_frac * b_frac

        if key not in PYROCHLORE_DB:
            missing.append(key)
            if verbose:
                print(f"  {key:<20} {weight:>8.4f}  {'MISSING':>10}")
            continue

        val = prop_getter(PYROCHLORE_DB[key])
        if val is None:
            missing.append(f"{key}({prop_name}=None)")
            if verbose:
                print(f"  {key:<20} {weight:>8.4f}  {'N/A':>10}")
            continue

        contribution = weight * val
        numerator += contribution
        denominator += weight

        if verbose:
            print(f"  {key:<20} {weight:>8.4f}  {val:>10.4f}  {contribution:>14.6f}")

    if missing:
        print(f"  ⚠  Missing DB entries / values: {missing}")

    if denominator == 0:
        return None

    result = numerator / denominator
    if verbose:
        print(f"  {'':20} {'':>8}  {'ROM =':>10}  {result:>14.6f}")

    return result


# =============================================================================
# HELPER: build composition label
# =============================================================================

def composition_label(a_site: Dict[str, float], b_site: Dict[str, float]) -> str:
    a_str = "".join(f"{e}{f:.2g}" for e, f in a_site.items())
    b_str = "".join(f"{e}{f:.2g}" for e, f in b_site.items())
    return f"({a_str})2({b_str})2O7"


# =============================================================================
# PROPERTY-SPECIFIC CALCULATION FUNCTIONS
# =============================================================================

def calc_lattice_parameter(
    a_site: Dict[str, float],
    b_site: Dict[str, float],
    verbose: bool = False,
) -> float:
    """Rule-of-mixtures lattice parameter (Å)."""
    return rule_of_mixtures(
        a_site, b_site,
        prop_getter=lambda p: p.lattice_parameter,
        prop_name="Lattice Parameter (Å)",
        verbose=verbose,
    )


def calc_ionic_radius_A(
    a_site: Dict[str, float],
    b_site: Dict[str, float],
    verbose: bool = False,
) -> float:
    """Rule-of-mixtures mean ionic radius on A-site (Å, 8-coord)."""
    return rule_of_mixtures(
        a_site, b_site,
        prop_getter=lambda p: p.ionic_radius_A,
        prop_name="Ionic Radius A-site (Å)",
        verbose=verbose,
    )


def calc_ionic_radius_B(
    a_site: Dict[str, float],
    b_site: Dict[str, float],
    verbose: bool = False,
) -> float:
    """Rule-of-mixtures mean ionic radius on B-site (Å, 6-coord)."""
    return rule_of_mixtures(
        a_site, b_site,
        prop_getter=lambda p: p.ionic_radius_B,
        prop_name="Ionic Radius B-site (Å)",
        verbose=verbose,
    )


def calc_electronegativity_A(
    a_site: Dict[str, float],
    b_site: Dict[str, float],
    verbose: bool = False,
) -> float:
    """Rule-of-mixtures mean Pauling electronegativity on A-site."""
    return rule_of_mixtures(
        a_site, b_site,
        prop_getter=lambda p: p.electronegativity_A,
        prop_name="Electronegativity A-site",
        verbose=verbose,
    )


def calc_electronegativity_B(
    a_site: Dict[str, float],
    b_site: Dict[str, float],
    verbose: bool = False,
) -> float:
    """Rule-of-mixtures mean Pauling electronegativity on B-site."""
    return rule_of_mixtures(
        a_site, b_site,
        prop_getter=lambda p: p.electronegativity_B,
        prop_name="Electronegativity B-site",
        verbose=verbose,
    )


def calc_radius_ratio(
    a_site: Dict[str, float],
    b_site: Dict[str, float],
    verbose: bool = False,
) -> float:
    """
    Pyrochlore stability criterion: r_A / r_B.
    Stable pyrochlore: 1.46 ≤ r_A/r_B ≤ 1.78
    """
    r_A = calc_ionic_radius_A(a_site, b_site, verbose=verbose)
    r_B = calc_ionic_radius_B(a_site, b_site, verbose=verbose)
    if r_A is None or r_B is None or r_B == 0:
        return None
    return r_A / r_B


def calc_lattice_distortion_A(
    a_site: Dict[str, float],
    b_site: Dict[str, float],
    verbose: bool = False,
) -> float:
    """
    A-site lattice distortion δ_A (Warren–Cowley-style RMS deviation):
        δ_A = sqrt( Σ x_i * (r_i - <r_A>)^2 ) / <r_A>

    Uses ROM ionic radii from the database, weighted by site fractions only.
    """
    # Get mean A-site radius from ROM
    r_A_mean = calc_ionic_radius_A(a_site, b_site, verbose=False)
    if r_A_mean is None:
        return None

    # Collect per-element A-site radii (average over B-site partners)
    variance = 0.0
    for a_elem, a_frac in a_site.items():
        # Average r_A for this element over all B partners
        r_vals = []
        for b_elem in b_site:
            key = f"{a_elem}2{b_elem}2O7"
            if key in PYROCHLORE_DB and PYROCHLORE_DB[key].ionic_radius_A is not None:
                r_vals.append(PYROCHLORE_DB[key].ionic_radius_A)
        if r_vals:
            r_i = sum(r_vals) / len(r_vals)
            variance += a_frac * (r_i - r_A_mean) ** 2

    delta = math.sqrt(variance) / r_A_mean if r_A_mean != 0 else None
    if verbose:
        print(f"\n  A-site lattice distortion δ_A = {delta:.6f}" if delta else "  δ_A: insufficient data")
    return delta


def calc_lattice_distortion_B(
    a_site: Dict[str, float],
    b_site: Dict[str, float],
    verbose: bool = False,
) -> float:
    """
    B-site lattice distortion δ_B:
        δ_B = sqrt( Σ p_j * (r_j - <r_B>)^2 ) / <r_B>
    """
    r_B_mean = calc_ionic_radius_B(a_site, b_site, verbose=False)
    if r_B_mean is None:
        return None

    variance = 0.0
    for b_elem, b_frac in b_site.items():
        r_vals = []
        for a_elem in a_site:
            key = f"{a_elem}2{b_elem}2O7"
            if key in PYROCHLORE_DB and PYROCHLORE_DB[key].ionic_radius_B is not None:
                r_vals.append(PYROCHLORE_DB[key].ionic_radius_B)
        if r_vals:
            r_j = sum(r_vals) / len(r_vals)
            variance += b_frac * (r_j - r_B_mean) ** 2

    delta = math.sqrt(variance) / r_B_mean if r_B_mean != 0 else None
    if verbose:
        print(f"\n  B-site lattice distortion δ_B = {delta:.6f}" if delta else "  δ_B: insufficient data")
    return delta


def calc_electronegativity_difference(
    a_site: Dict[str, float],
    b_site: Dict[str, float],
    verbose: bool = False,
) -> float:
    """Mean |χ_A - χ_B| electronegativity difference via ROM."""
    return rule_of_mixtures(
        a_site, b_site,
        prop_getter=lambda p: (
            abs(p.electronegativity_A - p.electronegativity_B)
            if p.electronegativity_A is not None and p.electronegativity_B is not None
            else None
        ),
        prop_name="|Δχ| A-B",
        verbose=verbose,
    )


def calc_bulk_modulus(
    a_site: Dict[str, float],
    b_site: Dict[str, float],
    verbose: bool = False,
) -> float:
    """Rule-of-mixtures bulk modulus (GPa), if available in DB."""
    return rule_of_mixtures(
        a_site, b_site,
        prop_getter=lambda p: p.bulk_modulus,
        prop_name="Bulk Modulus (GPa)",
        verbose=verbose,
    )

def calc_shear_modulus(
    a_site: Dict[str, float],
    b_site: Dict[str, float],
    verbose: bool = False,
) -> float:
    """Rule-of-mixtures shear modulus (GPa), if available in DB."""
    return rule_of_mixtures(
        a_site, b_site,
        prop_getter=lambda p: p.shear_modulus,
        prop_name="Shear Modulus (GPa)",
        verbose=verbose,
    )


def calc_thermal_conductivity(
    a_site: Dict[str, float],
    b_site: Dict[str, float],
    verbose: bool = False,
) -> float:
    """Rule-of-mixtures thermal conductivity (W/m·K), if available in DB."""
    return rule_of_mixtures(
        a_site, b_site,
        prop_getter=lambda p: p.thermal_conductivity,
        prop_name="Thermal Conductivity (W/m·K)",
        verbose=verbose,
    )


def calc_thermal_expansion(
    a_site: Dict[str, float],
    b_site: Dict[str, float],
    verbose: bool = False,
) -> float:
    """Rule-of-mixtures thermal expansion (°C⁻¹), if available in DB."""
    return rule_of_mixtures(
        a_site, b_site,
        prop_getter=lambda p: p.thermal_conductivity,
        prop_name="Thermal Expansion (°C⁻¹)",
        verbose=verbose,
    )

# =============================================================================
# FULL SUMMARY REPORT
# =============================================================================

def full_report(composition: str, verbose: bool = True) -> Dict[str, float]:
    """
    Parse composition and compute all available ROM properties.

    Parameters
    ----------
    composition : str   e.g. "(Sn0.3Ho0.7)(Ti0.6Hf0.4)"
    verbose     : print detailed breakdown per property

    Returns
    -------
    dict of {property_name: value}
    """
    a_site, b_site = parse_composition(composition)
    label = composition_label(a_site, b_site)

    print("=" * 62)
    print(f"  High-Entropy Pyrochlore — Rule of Mixtures Report")
    print(f"  Composition : {label}")
    print(f"  A-site : { {k: round(v,4) for k,v in a_site.items()} }")
    print(f"  B-site : { {k: round(v,4) for k,v in b_site.items()} }")
    print("=" * 62)

    results = {}

    calcs = [
        ("Lattice Parameter (Å)",           calc_lattice_parameter),
        ("Ionic Radius A-site (Å)",         calc_ionic_radius_A),
        ("Ionic Radius B-site (Å)",         calc_ionic_radius_B),
        ("r_A / r_B ratio",                 calc_radius_ratio),
        ("Electronegativity A-site",        calc_electronegativity_A),
        ("Electronegativity B-site",        calc_electronegativity_B),
        ("|Δχ| A-B",                        calc_electronegativity_difference),
        ("Lattice Distortion δ_A",          calc_lattice_distortion_A),
        ("Lattice Distortion δ_B",          calc_lattice_distortion_B),
        ("Bulk Modulus (GPa)",              calc_bulk_modulus),
        ("Shear Modulus (GPa)",             calc_shear_modulus),
        ("Thermal Conductivity (W/m·K)",    calc_thermal_conductivity),
        ("Thermal Expansion Coeff (°C⁻¹)",  calc_thermal_expansion),
    ]

    for name, func in calcs:
        val = func(a_site, b_site, verbose=verbose)
        results[name] = val

    print("\n" + "=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    for name, val in results.items():
        if val is not None:
            print(f"  {name:<38} {val:>10.4f}")
        else:
            print(f"  {name:<38} {'N/A':>10}")

    # Pyrochlore stability note
    ratio = results.get("r_A / r_B ratio")
    if ratio is not None:
        stable = 1.46 <= ratio <= 1.78
        status = "✓ Pyrochlore-stable" if stable else "✗ Outside pyrochlore stability window"
        print(f"\n  Stability (1.46 ≤ r_A/r_B ≤ 1.78): {status}")

    print("=" * 62 + "\n")
    return results


# =============================================================================
# CONVENIENCE: compute a single named property from a composition string
# =============================================================================

PROPERTY_MAP = {
    "lattice_parameter":        calc_lattice_parameter,
    "ionic_radius_A":           calc_ionic_radius_A,
    "ionic_radius_B":           calc_ionic_radius_B,
    "radius_ratio":             calc_radius_ratio,
    "electronegativity_A":      calc_electronegativity_A,
    "electronegativity_B":      calc_electronegativity_B,
    "electronegativity_diff":   calc_electronegativity_difference,
    "distortion_A":             calc_lattice_distortion_A,
    "distortion_B":             calc_lattice_distortion_B,
    "bulk_modulus":             calc_bulk_modulus,
    "thermal_conductivity":     calc_thermal_conductivity,
}


def get_property(composition: str, property_name: str, verbose: bool = False) -> float:
    """
    Compute a single ROM property for a given composition string.

    Parameters
    ----------
    composition   : e.g. "(Sn0.3Ho0.7)(Ti0.6Hf0.4)"
    property_name : one of the keys in PROPERTY_MAP
    verbose       : print breakdown table

    Returns
    -------
    float value or None
    """
    if property_name not in PROPERTY_MAP:
        raise ValueError(
            f"Unknown property '{property_name}'. "
            f"Choose from: {list(PROPERTY_MAP.keys())}"
        )
    a_site, b_site = parse_composition(composition)
    return PROPERTY_MAP[property_name](a_site, b_site, verbose=verbose)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":

    # --- Example 1: Full report for the composition from the problem statement ---
    full_report("(Sn0.3Ho0.7)(Ti0.6Hf0.4)")

    # --- Example 2: Single property query ---
    comp = "(Gd0.5Nd0.5)(Zr0.5Hf0.5)"
    lp = get_property(comp, "lattice_parameter", verbose=True)
    print(f"Lattice parameter for {comp}: {lp:.4f} Å\n")

    # --- Example 3: Multi-component A and B sites ---
    full_report("(Ho0.25Gd0.25Nd0.25Sm0.25)(Zr0.5Hf0.5)")

    # --- Example 4: Batch comparison ---
    compositions = [
        "(Ho0.5Gd0.5)(Zr1.0)",
        "(Ho0.5Nd0.5)(Zr0.5Hf0.5)",
        "(Gd0.5Sm0.5)(Ti0.5Zr0.5)",
    ]
    print(f"\n{'Composition':<40} {'a (Å)':>8} {'rA/rB':>8} {'δ_A':>8}")
    print("-" * 68)
    for c in compositions:
        a, b = parse_composition(c)
        lp_  = calc_lattice_parameter(a, b)
        rr   = calc_radius_ratio(a, b)
        dA   = calc_lattice_distortion_A(a, b)
        label = composition_label(a, b)
        print(f"{label:<40} {lp_ or 0:>8.4f} {rr or 0:>8.4f} {dA or 0:>8.5f}")
