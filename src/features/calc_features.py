"""
Pyrochlore Crystal Structure Feature Extraction using pymatgen
Processes compositional data, calculates bond geometry, polyhedral distortions,
and integrates elemental properties across multiple temperature points.
"""

import pandas as pd
import numpy as np
from pymatgen.core import Structure, Lattice, Composition, Element
#from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
#from pymatgen.analysis.structure_analyzer import RelaxationAnalyzer
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# ELEMENTAL PROPERTY DATABASE
# ============================================================================

def get_elemental_properties(element_symbol):
    """
    Retrieve elemental properties from pymatgen.

    Returns
    -------
    dict : Ionic radii, electronegativity, atomic mass, valence
    """
    try:
        elem = Element(element_symbol)
        return {
            'atomic_number': elem.Z,
            'mass': elem.atomic_mass,
            'electronegativity': elem.X,  # Pauling electronegativity
            'ionic_radius_6cn': elem.ionic_radius,  # Default 6-coordinate
        }
    except:
        return None


def build_elemental_database():
    """Build comprehensive elemental property database."""
    rare_earths = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Y']
    b_site = ['Ti', 'Zr', 'Hf', 'Ir', 'Sn', 'Pb', 'Bi']

    db = {}
    for elem_symbol in rare_earths + b_site:
        db[elem_symbol] = get_elemental_properties(elem_symbol)

    return db


ELEMENTAL_DB = build_elemental_database()


# ============================================================================
# PYROCHLORE STRUCTURE BUILDER
# ============================================================================

class PyrochloreBuilder:
    """
    Construct idealized pyrochlore A2B2O7 structures (Fd-3m, Space Group 227)
    and extract coordination polyhedra.
    """

    @staticmethod
    def build_structure(a_site_elements, b_site_elements, lattice_param,
                        oxygen_param_x=0.325, supercell=(2, 2, 2)):
        """
        Build pyrochlore structure from composition and lattice parameter.

        Parameters
        ----------
        a_site_elements : list
            A-site cation elements (e.g., ['Yb', 'Ho'])
        b_site_elements : list
            B-site cation elements (e.g., ['Ti'])
        lattice_param : float
            Cubic lattice parameter (Angstroms)
        oxygen_param_x : float
            Oxygen positional parameter (default 0.325 for pyrochlore)
        supercell : tuple
            Supercell multiplicity for structure analysis

        Returns
        -------
        Structure : pymatgen Structure object
        """

        # Create cubic lattice
        lattice = Lattice.cubic(lattice_param)

        # Wyckoff positions for Fd-3m (No. 227) conventional cubic cell
        # A-site (8c): (1/8, 1/8, 1/8)
        # B-site (16d): (1/2, 1/2, 1/2) + (0, 0, 0)
        # O-site (56h): (x, x, x) where x = oxygen parameter

        species = []
        coords = []

        # A-site (8-coordinate) - distribute elements equiatomically
        a_comp = Composition({elem: 1.0 for elem in a_site_elements})
        a_frac = {elem: a_comp[elem] for elem in a_site_elements}

        # Generate all A-site positions (8c Wyckoff)
        a_positions = [
            [0.125, 0.125, 0.125],
            [0.875, 0.875, 0.125],
            [0.875, 0.125, 0.875],
            [0.125, 0.875, 0.875],
        ]

        # Replicate for face-centered cubic
        all_a_pos = []
        for pos in a_positions:
            all_a_pos.append(pos)
            all_a_pos.append([pos[0] + 0.5, pos[1] + 0.5, pos[2]])
            all_a_pos.append([pos[0] + 0.5, pos[1], pos[2] + 0.5])
            all_a_pos.append([pos[0], pos[1] + 0.5, pos[2] + 0.5])

        # Add A-site atoms
        for a_pos in all_a_pos:
            # Normalize to [0, 1)
            a_pos = np.array(a_pos) % 1.0
            for elem in a_site_elements:
                species.append(elem)
                coords.append(a_pos)
                break  # Equiatomic: one element per position

        # B-site (6-coordinate octahedral) - distribute elements
        b_comp = Composition({elem: 1.0 for elem in b_site_elements})
        b_frac = {elem: b_comp[elem] for elem in b_site_elements}

        # Generate all B-site positions (16d Wyckoff)
        b_positions = [
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0],
        ]

        # Replicate for face-centered cubic
        all_b_pos = []
        for pos in b_positions:
            all_b_pos.append(pos)
            all_b_pos.append([pos[0] + 0.5, pos[1] + 0.5, pos[2]])
            all_b_pos.append([pos[0] + 0.5, pos[1], pos[2] + 0.5])
            all_b_pos.append([pos[0], pos[1] + 0.5, pos[2] + 0.5])

        # Add B-site atoms
        for b_pos in all_b_pos:
            b_pos = np.array(b_pos) % 1.0
            for elem in b_site_elements:
                species.append(elem)
                coords.append(b_pos)
                break

        # O-site (56h) - two sets of positions modulated by oxygen parameter
        o_positions = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # O at (x, x, x) type positions
                    o_positions.append([oxygen_param_x + i * 0.5, oxygen_param_x + j * 0.5, oxygen_param_x + k * 0.5])
                    # O at (x̄, x̄, x̄) type positions (complement)
                    o_positions.append(
                        [1 - oxygen_param_x + i * 0.5, 1 - oxygen_param_x + j * 0.5, 1 - oxygen_param_x + k * 0.5])

        # Add O-site atoms
        for o_pos in o_positions:
            o_pos = np.array(o_pos) % 1.0
            species.append('O')
            coords.append(o_pos)

        # Create structure
        structure = Structure(lattice, species, coords, validate_proximity=False)

        return structure

    @staticmethod
    def get_coordination_polyhedra(structure, site_index, cutoff_radius=3.0):
        """
        Extract coordination polyhedra around a given site.

        Parameters
        ----------
        structure : Structure
            Pymatgen Structure object
        site_index : int
            Index of central cation
        cutoff_radius : float
            Maximum bond distance (Angstroms)

        Returns
        -------
        dict : Coordinating anions with distances
        """
        central_site = structure[site_index]
        neighbors = []

        for i, site in enumerate(structure):
            if i == site_index:
                continue

            distance = central_site.distance(site)

            if distance < cutoff_radius:
                neighbors.append({
                    'index': i,
                    'element': site.species,
                    'distance': distance,
                    'coords': site.coords,
                })

        return neighbors


# ============================================================================
# GEOMETRIC ANALYSIS FUNCTIONS
# ============================================================================

def calculate_bond_statistics(neighbors):
    """
    Calculate bond length statistics for a coordination sphere.

    Parameters
    ----------
    neighbors : list
        List of neighbor dicts with 'distance' key

    Returns
    -------
    dict : Bond statistics (mean, std, variance, min, max)
    """
    if not neighbors:
        return {
            'n_bonds': 0,
            'bond_length_mean': np.nan,
            'bond_length_std': np.nan,
            'bond_length_variance': np.nan,
            'bond_length_min': np.nan,
            'bond_length_max': np.nan,
            'bond_length_range': np.nan,
        }

    distances = np.array([n['distance'] for n in neighbors])

    return {
        'n_bonds': len(distances),
        'bond_length_mean': np.mean(distances),
        'bond_length_std': np.std(distances),
        'bond_length_variance': np.var(distances),
        'bond_length_min': np.min(distances),
        'bond_length_max': np.max(distances),
        'bond_length_range': np.max(distances) - np.min(distances),
    }


def calculate_polyhedron_volume(neighbors, central_coords):
    """
    Calculate coordination polyhedron volume using convex hull.

    Parameters
    ----------
    neighbors : list
        Coordinating anions
    central_coords : ndarray
        Central cation coordinates

    Returns
    -------
    float : Polyhedral volume (Ų)
    """
    if len(neighbors) < 4:
        return np.nan

    try:
        from scipy.spatial import ConvexHull

        coords = np.array([n['coords'] for n in neighbors])
        hull = ConvexHull(coords)
        return hull.volume
    except:
        return np.nan


def calculate_bond_angle_distortion(neighbors, central_coords):
    """
    Calculate distortion metrics from M-O-M bond angles.

    Parameters
    ----------
    neighbors : list
        Coordinating anions
    central_coords : ndarray
        Central cation coordinates

    Returns
    -------
    dict : Angle statistics (variance, mean, std, min, max)
    """
    if len(neighbors) < 3:
        return {
            'angle_variance': np.nan,
            'angle_mean': np.nan,
            'angle_std': np.nan,
            'angle_min': np.nan,
            'angle_max': np.nan,
        }

    angles = []
    coords = np.array([n['coords'] for n in neighbors])

    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            v1 = coords[i] - central_coords
            v2 = coords[j] - central_coords

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180.0 / np.pi
            angles.append(angle)

    angles = np.array(angles)

    return {
        'angle_variance': np.var(angles),
        'angle_mean': np.mean(angles),
        'angle_std': np.std(angles),
        'angle_min': np.min(angles),
        'angle_max': np.max(angles),
    }


def calculate_octahedral_distortion_metrics(neighbors):
    """
    Calculate distortion parameters specific to octahedral (6-coordinate) geometry.

    Metrics include:
    - Bond length variance (ideal = 0)
    - Tetragonal distortion ratio
    - Trigonal distortion parameter
    """
    distances = np.array([n['distance'] for n in neighbors])

    if len(distances) != 6:
        print(f"Warning: Expected 6 bonds for octahedral, got {len(distances)}")

    # Tetragonal distortion: ratio of longest to shortest bond
    tetragonal_ratio = np.max(distances) / (np.min(distances) + 1e-10)

    return {
        'octahedral_bond_variance': np.var(distances),
        'octahedral_distortion_ratio': tetragonal_ratio,
        'octahedral_coordination_number': len(distances),
    }


def calculate_cubic_distortion_metrics(neighbors):
    """
    Calculate distortion parameters for cubic (8-coordinate) geometry.
    """
    distances = np.array([n['distance'] for n in neighbors])

    # Cubic distortion: ratio of longest to shortest bond
    cubic_ratio = np.max(distances) / (np.min(distances) + 1e-10)

    return {
        'cubic_bond_variance': np.var(distances),
        'cubic_distortion_ratio': cubic_ratio,
        'cubic_coordination_number': len(distances),
    }


# ============================================================================
# COMPOSITION PARSING & PROPERTY AGGREGATION
# ============================================================================

def parse_composition_string(comp_str):
    """
    Parse composition string like 'Yb,Ho' or 'Pr,Sm,Gd,Ho,Lu'.

    Returns
    -------
    list : Element symbols
    """
    if pd.isna(comp_str) or comp_str == '':
        return []

    elements = [e.strip() for e in str(comp_str).split(',')]
    return [e for e in elements if e]


def aggregate_elemental_properties(elements, coordinate_number=6):
    """
    Aggregate elemental properties for a site with multiple elements.
    Assumes equiatomic distribution.

    Parameters
    ----------
    elements : list
        Element symbols
    coordinate_number : int
        Coordination number (6 for B-site, 8 for A-site)

    Returns
    -------
    dict : Aggregated properties
    """
    if not elements:
        return {
            'n_elements': 0,
            'mean_mass': np.nan,
            'mean_electronegativity': np.nan,
            'mean_ionic_radius': np.nan,
            'electronegativity_variance': np.nan,
            'mass_variance': np.nan,
        }

    properties = []
    masses = []
    electroneg = []

    for elem in elements:
        if elem in ELEMENTAL_DB and ELEMENTAL_DB[elem] is not None:
            props = ELEMENTAL_DB[elem]
            properties.append(props)
            masses.append(props['mass'])
            electroneg.append(props['electronegativity'])

    if not properties:
        return {
            'n_elements': len(elements),
            'mean_mass': np.nan,
            'mean_electronegativity': np.nan,
            'mean_ionic_radius': np.nan,
            'electronegativity_variance': np.nan,
            'mass_variance': np.nan,
        }

    masses = np.array(masses)
    electroneg = np.array(electroneg)

    return {
        'n_elements': len(elements),
        'mean_mass': np.mean(masses),
        'mean_electronegativity': np.mean(electroneg),
        'electronegativity_variance': np.var(electroneg),
        'mass_variance': np.var(masses),
    }


# ============================================================================
# MAIN FEATURE EXTRACTION PIPELINE
# ============================================================================

def extract_compound_features(row, temperature_values=None):
    """
    Extract all geometric and compositional features for a single compound.
    Creates multiple rows for different temperatures.

    Parameters
    ----------
    row : pd.Series
        Single row from input dataset
    temperature_values : list
        Temperature points to replicate data across

    Returns
    -------
    list : List of feature dictionaries (one per temperature)
    """

    if temperature_values is None:
        temperature_values = [300, 500, 700, 1000]

    features_list = []

    # Extract composition
    a_elements = parse_composition_string(row['Sample A'])
    b_elements = parse_composition_string(row['Sample B'])

    # Handle missing data
    if not a_elements or not b_elements:
        return []

    # Get lattice parameter
    lattice_param = row['Lattice Parameter (Angstrom)']
    if pd.isna(lattice_param):
        lattice_param = row['Lattice Parameter a (A)']

    if pd.isna(lattice_param):
        return []

    # Get oxygen parameter (default to pyrochlore value)
    oxygen_param_x = row['Oxygen Parameter x'] if not pd.isna(row['Oxygen Parameter x']) else 0.325

    # Get anion vacancy
    anion_vacancy = row['Anion Vacancy (delta)'] if not pd.isna(row['Anion Vacancy (delta)']) else 0.0

    # Build pyrochlore structure
    try:
        structure = PyrochloreBuilder.build_structure(
            a_elements,
            b_elements,
            float(lattice_param),
            oxygen_param_x=float(oxygen_param_x)
        )
    except Exception as e:
        print(f"Error building structure for {row['Composition']}: {e}")
        return []

    # Extract polyhedra features for A-site and B-site
    a_site_features = extract_site_polyhedra_features(structure, a_elements, site_type='A')
    b_site_features = extract_site_polyhedra_features(structure, b_elements, site_type='B')

    # Aggregate elemental properties
    a_site_elem_props = aggregate_elemental_properties(a_elements, coordinate_number=8)
    b_site_elem_props = aggregate_elemental_properties(b_elements, coordinate_number=6)

    # Rename with site prefix
    a_site_elem_props = {f'a_site_{k}': v for k, v in a_site_elem_props.items()}
    b_site_elem_props = {f'b_site_{k}': v for k, v in b_site_elem_props.items()}

    # Combine base features (same for all temperatures)
    base_features = {
        'Composition': row['Composition'],
        'A_Elements': ','.join(a_elements),
        'B_Elements': ','.join(b_elements),
        'Lattice_Parameter_A': float(lattice_param),
        'Oxygen_Parameter_x': float(oxygen_param_x),
        'Anion_Vacancy': float(anion_vacancy),
        'TPS_Conductivity_W_m_K': row['TPS Cond W/m/K'] if not pd.isna(row['TPS Cond W/m/K']) else np.nan,
        'Corrected_Conductivity_W_m_K': row['Corrected Cond W/m/K'] if not pd.isna(
            row['Corrected Cond W/m/K']) else np.nan,
        'Relative_Density_percent': row['Relative Density %'] if not pd.isna(row['Relative Density %']) else np.nan,
        'Is_Single_Phase': row['Is Single Phase'] if not pd.isna(row['Is Single Phase']) else np.nan,
        'Synthesis_Method': row['Synthesis Method'],
        'Conductivity_Type': row['Conductivity Type'] if not pd.isna(row['Conductivity Type']) else 'Unknown',
    }

    # Combine all features
    combined_features = {
        **base_features,
        **a_site_features,
        **b_site_features,
        **a_site_elem_props,
        **b_site_elem_props,
    }

    # Create rows for each temperature
    for temp in temperature_values:
        row_features = combined_features.copy()
        row_features['Temperature_K'] = float(temp)
        features_list.append(row_features)

    return features_list


def extract_site_polyhedra_features(structure, site_elements, site_type='A', cutoff_radius=3.5):
    """
    Extract coordination polyhedra features for all sites of a given type.

    Parameters
    ----------
    structure : Structure
        Pymatgen Structure object
    site_elements : list
        Elements expected at this site
    site_type : str
        'A' for 8-coordinate A-site, 'B' for 6-coordinate B-site
    cutoff_radius : float
        Maximum bond distance for identifying neighbors

    Returns
    -------
    dict : Aggregated polyhedra features for the site
    """

    # Identify sites of interest
    site_indices = []
    for i, site in enumerate(structure):
        # Check if site element matches
        elem_symbol = list(site.species.keys())[0]
        if elem_symbol in site_elements:
            site_indices.append(i)

    if not site_indices:
        # Return empty features with appropriate prefixes
        prefix = 'a_site' if site_type == 'A' else 'b_site'
        return {
            f'{prefix}_n_sites': 0,
            f'{prefix}_avg_coordination_number': np.nan,
            f'{prefix}_avg_bond_length': np.nan,
            f'{prefix}_avg_bond_std': np.nan,
            f'{prefix}_avg_polyhedral_volume': np.nan,
            f'{prefix}_avg_angle_variance': np.nan,
            f'{prefix}_distortion_metric': np.nan,
        }

    # Extract features for each site
    bond_stats_list = []
    polyhedral_volumes = []
    angle_distortions = []
    coordination_numbers = []

    for site_idx in site_indices:
        neighbors = PyrochloreBuilder.get_coordination_polyhedra(
            structure,
            site_idx,
            cutoff_radius=cutoff_radius
        )

        if neighbors:
            # Filter for anions (oxygen)
            anion_neighbors = [n for n in neighbors if 'O' in str(n['element'])]

            if anion_neighbors:
                # Bond statistics
                bond_stats = calculate_bond_statistics(anion_neighbors)
                bond_stats_list.append(bond_stats)
                coordination_numbers.append(bond_stats['n_bonds'])

                # Polyhedral volume
                central_coords = structure[site_idx].coords
                volume = calculate_polyhedron_volume(anion_neighbors, central_coords)
                polyhedral_volumes.append(volume)

                # Angle distortion
                angle_dist = calculate_bond_angle_distortion(anion_neighbors, central_coords)
                angle_distortions.append(angle_dist)

    # Aggregate across sites
    prefix = 'a_site' if site_type == 'A' else 'b_site'

    if bond_stats_list:
        bond_lengths = np.array([bs['bond_length_mean'] for bs in bond_stats_list])
        bond_stds = np.array([bs['bond_length_std'] for bs in bond_stats_list])

        features = {
            f'{prefix}_n_sites': len(site_indices),
            f'{prefix}_avg_coordination_number': np.mean(coordination_numbers),
            f'{prefix}_avg_bond_length': np.mean(bond_lengths),
            f'{prefix}_avg_bond_std': np.mean(bond_stds),
            f'{prefix}_avg_bond_variance': np.mean([bs['bond_length_variance'] for bs in bond_stats_list]),
            f'{prefix}_avg_bond_range': np.mean([bs['bond_length_range'] for bs in bond_stats_list]),
        }

        # Polyhedral volumes
        valid_volumes = [v for v in polyhedral_volumes if not np.isnan(v)]
        if valid_volumes:
            features[f'{prefix}_avg_polyhedral_volume'] = float(np.mean(valid_volumes))
            features[f'{prefix}_polyhedral_volume_std'] = float(np.std(valid_volumes))
        else:
            features[f'{prefix}_avg_polyhedral_volume'] = np.nan
            features[f'{prefix}_polyhedral_volume_std'] = np.nan

        # Angle distortion
        valid_angle_vars = [ad['angle_variance'] for ad in angle_distortions if not np.isnan(ad['angle_variance'])]
        if valid_angle_vars:
            features[f'{prefix}_avg_angle_variance'] = float(np.mean(valid_angle_vars))
            features[f'{prefix}_avg_angle_std'] = float(np.mean(
                [ad['angle_std'] for ad in angle_distortions if not np.isnan(ad['angle_std'])]))
        else:
            features[f'{prefix}_avg_angle_variance'] = np.nan
            features[f'{prefix}_avg_angle_std'] = np.nan

        # Distortion metric (normalized bond length variance / mean bond length)
        distortion_metrics = [bs['bond_length_std'] / (bs['bond_length_mean'] + 1e-10) for bs in bond_stats_list]
        features[f'{prefix}_distortion_metric'] = np.mean(distortion_metrics)

    else:
        features = {
            f'{prefix}_n_sites': 0,
            f'{prefix}_avg_coordination_number': np.nan,
            f'{prefix}_avg_bond_length': np.nan,
            f'{prefix}_avg_bond_std': np.nan,
            f'{prefix}_avg_bond_variance': np.nan,
            f'{prefix}_avg_bond_range': np.nan,
            f'{prefix}_avg_polyhedral_volume': np.nan,
            f'{prefix}_polyhedral_volume_std': np.nan,
            f'{prefix}_avg_angle_variance': np.nan,
            f'{prefix}_avg_angle_std': np.nan,
            f'{prefix}_distortion_metric': np.nan,
        }

    return features


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_dataset(input_csv_path, output_csv_path, temperature_values=None):
    """
    Process entire dataset: extract features and create multi-temperature rows.

    Parameters
    ----------
    input_csv_path : str
        Path to input CSV file
    output_csv_path : str
        Path to output CSV file with extracted features
    temperature_values : list
        Temperature points to create rows for
    """

    if temperature_values is None:
        temperature_values = [300, 500, 700, 1000]

    print(f"Loading dataset from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)

    print(f"Dataset shape: {df.shape}")
    print(f"\nProcessing {len(df)} compounds...")

    all_features = []
    errors = []

    for idx, (_, row) in enumerate(df.iterrows()):
        compound_name = row['Composition']

        try:
            features = extract_compound_features(row, temperature_values=temperature_values)
            all_features.extend(features)

            if (idx + 1) % 5 == 0:
                print(f"  Processed {idx + 1}/{len(df)} compounds ({len(all_features)} total rows)")

        except Exception as e:
            error_msg = f"Compound {compound_name} (row {idx}): {str(e)}"
            errors.append(error_msg)
            print(f"  ERROR: {error_msg}")

    if not all_features:
        print("No features extracted! Check your input data.")
        return

    # Create output DataFrame
    output_df = pd.DataFrame(all_features)

    print(
        f"\n✓ Extracted features for {len(output_df)} samples ({len(all_features) // len(temperature_values)} compounds)")
    print(f"✓ Total rows: {len(output_df)} (including {len(temperature_values)} temperature points per compound)")

    # Reorder columns for readability
    col_order = [
        'Temperature_K',
        'Composition',
        'A_Elements',
        'B_Elements',
        'Lattice_Parameter_A',
        'Oxygen_Parameter_x',
        'Anion_Vacancy',
        'TPS_Conductivity_W_m_K',
        'Corrected_Conductivity_W_m_K',
        'Relative_Density_percent',
        'Is_Single_Phase',
        'Synthesis_Method',
        'Conductivity_Type',
    ]

    # Add geometric features
    geometric_cols = [col for col in output_df.columns if col not in col_order]
    col_order.extend(sorted(geometric_cols))

    output_df = output_df[[col for col in col_order if col in output_df.columns]]

    # Save to CSV
    print(f"\nSaving to {output_csv_path}...")
    output_df.to_csv(output_csv_path, index=False)

    # Print summary statistics
    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"\nOutput file: {output_csv_path}")
    print(f"Total samples: {len(output_df)}")
    print(f"Temperature points: {temperature_values}")
    print(f"Features extracted per sample: {len(output_df.columns)}")

    print("\nColumn categories:")
    print(f"  - Compositional: 3")
    print(f"  - Structural: 3")
    print(f"  - A-site polyhedra: {sum(1 for col in output_df.columns if 'a_site' in col)}")
    print(f"  - B-site polyhedra: {sum(1 for col in output_df.columns if 'b_site' in col)}")
    print(
        f"  - Physical properties: {sum(1 for col in output_df.columns if 'Conductivity' in col or 'Density' in col)}")

    print("\nFeature column names (A-site geometric):")
    a_site_cols = [col for col in output_df.columns if col.startswith('a_site')]
    for col in a_site_cols[:8]:
        print(f"  - {col}")
    if len(a_site_cols) > 8:
        print(f"  ... and {len(a_site_cols) - 8} more")

    print("\nFeature column names (B-site geometric):")
    b_site_cols = [col for col in output_df.columns if col.startswith('b_site')]
    for col in b_site_cols[:8]:
        print(f"  - {col}")
    if len(b_site_cols) > 8:
        print(f"  ... and {len(b_site_cols) - 8} more")

    print("\nData completeness:")
    print(output_df.isnull().sum().sum(), f"missing values out of {len(output_df) * len(output_df.columns)}")

    if errors:
        print(f"\n⚠ Encountered {len(errors)} errors during processing:")
        for error in errors[:5]:
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")

    return output_df


def generate_feature_summary(output_df):
    """
    Generate statistical summary of extracted features.

    Parameters
    ----------
    output_df : pd.DataFrame
        Processed feature dataframe
    """

    print("\n" + "=" * 80)
    print("FEATURE STATISTICS")
    print("=" * 80)

    # Numeric columns only
    numeric_df = output_df.select_dtypes(include=[np.number])

    summary_stats = numeric_df.describe()

    print("\nA-site (8-coordinate) features:")
    a_cols = [col for col in numeric_df.columns if 'a_site' in col]
    if a_cols:
        print(numeric_df[a_cols].describe().to_string())

    print("\n" + "-" * 80)
    print("\nB-site (6-coordinate) features:")
    b_cols = [col for col in numeric_df.columns if 'b_site' in col]
    if b_cols:
        print(numeric_df[b_cols].describe().to_string())
