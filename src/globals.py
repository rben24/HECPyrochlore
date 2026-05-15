"""
Globals will be declared here

"""

# ── element sets ────────────────────────────────────────────────────────────

# Rare-earth / Y cations that occupy the 8-coordinated A-site
KNOWN_A: frozenset[str] = frozenset({
    'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Y', 'Bi', 'Pb', 'Ca',
})

# Transition-metal cations that occupy the 6-coordinated B-site
KNOWN_B: frozenset[str] = frozenset({
    # Group IV
    'Ti', 'Zr', 'Hf', 'Sn',
    # Group V
    'V', 'Nb', 'Ta',
    # Group VI
    'Cr', 'Mo', 'W',
    # Group VII
    'Mn', 'Re',
    # Group VIII/IX (transition metals and 5d metals)
    'Fe', 'Co', 'Ni', 'Ru', 'Os', 'Rh', 'Ir',
    # Post-transition
    'Pb', 'Pt',
})

# # ── element sets (shared with load_icsd.py) ───────────────────────────────────
# KNOWN_A: frozenset = frozenset({
#     'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
#     'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Y',
# })
# KNOWN_B: frozenset = frozenset({
#     'Ti', 'Zr', 'Hf', 'Sn', 'Ir', 'Nb',
# })

# Ce can sit on either site depending on oxidation state; handled separately
CE_AMBIGUOUS = 'Ce'

# Pyrochlore structure-type identifiers used in the ICSD file
PYROCHLORE_STRUCTURE_TYPES: frozenset[str] = frozenset({
    'Ca2Nb2O7',   # standard Fd-3m pyrochlore prototype
    'Eu2Zr2O7',   # alternate ICSD label for the same structure
    'Bi2Ti2O7',
})

# Physical bounds
LATTICE_MIN = 9.5    # Å
LATTICE_MAX = 11.5   # Å
TEMP_MIN    = 285.0  # K
TEMP_MAX    = 305.0  # K
A_STOICH_RANGE = (1.5, 2.5)
B_STOICH_RANGE = (1.5, 2.5)

# Pyrochlore stability window for r_A/r_B (Shannon ionic radii)
# Outside this range → likely defect-fluorite or other polymorph
RA_RB_MIN = 1.40
RA_RB_MAX = 1.90

# ── Compound-type enum strings ───────────────────────────────────────────────

PRISTINE      = 'pristine'
HIGH_ENTROPY  = 'high_entropy'
NON_PYROCHLORE = 'non_pyrochlore'