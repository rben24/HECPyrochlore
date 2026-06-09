from mp_api.client import MPRester
import numpy as np
import pandas as pd
import time
from pymatgen.core import Structure, Element
from pathlib import Path

_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent
DATA_DIR  = _PROJECT / 'data' / 'raw'
OUTPUT_FILE = DATA_DIR / 'mp_pyrochlore_query.csv'

FIELDS = [
    'nsites', 'elements', 'composition', 'formula_pretty',
    'volume', 'density', 'symmetry', 'material_id', 'deprecated',
    'structure', 'energy_per_atom', 'energy_above_hull', 'band_gap',
    'cbm', 'vbm', 'is_gap_direct', 'total_magnetization', 'bulk_modulus',
    'shear_modulus', 'e_total', 'homogeneous_poisson', 'last_updated',
    'formation_energy_per_atom',
]

API_KEY = 'bmuIgvgjF3hvCwRTifO3LBgl8Pt1BzQb'


def fetch_material_ids(formula, spacegroup_number):
    """
    Step 1: Fetch only material IDs matching criteria
    """
    print("=" * 70)
    print("STEP 1: FETCHING MATERIAL IDs")
    print("=" * 70)

    try:
        with MPRester(API_KEY) as mpr:
            print(f"Searching for: formula={formula}, spacegroup={spacegroup_number}")

            results = mpr.materials.summary.search(
                formula=formula,
                spacegroup_number=spacegroup_number,
                fields=['material_id'],
            )

        material_ids = [result.material_id for result in results]
        print(f"✓ Found {len(material_ids)} materials\n")
        return material_ids

    except Exception as e:
        print(f"✗ Error fetching material IDs: {e}\n")
        return []


def fetch_material_properties(material_id, mpr):
    """
    Fetch all specific properties for a single material
    """
    material_data = {}

    try:
        summaries = mpr.materials.summary.search(material_ids=[material_id], fields=FIELDS)
        summary = summaries[0]
        print()
        # Basic identification
        material_data['material_id'] = summary.material_id if hasattr(summary, 'material_id') else 'N/A'
        material_data['formula_pretty'] = summary.formula_pretty if hasattr(summary, 'formula_pretty') else 'N/A'
        material_data['composition'] = str(summary.composition) if hasattr(summary, 'composition') else 'N/A'

        # Get structure for lattice parameters
        # structures = mpr.materials.search(material_ids=[material_id], fields=["structure", 'volume'])
        # structure = structures[0]
        structure = mpr.get_structure_by_material_id(material_id)
        if structure:
            # print(structure.lattice.abc)
            # exit(0)
            # Get lattice vectors (A, B, C) magnitudes from the lattice matrix
            lattice_matrix = structure.lattice.matrix # * np.sqrt(2)

            # Calculate magnitudes of lattice vectors
            A_vec = lattice_matrix[0] * np.sqrt(2)
            B_vec = lattice_matrix[1] * np.sqrt(2)
            C_vec = lattice_matrix[2] * np.sqrt(2)

            material_data['a_lattice'] = round(float(np.linalg.norm(A_vec)), 6)
            material_data['b_lattice'] = round(float(np.linalg.norm(B_vec)), 6)
            material_data['c_lattice'] = round(float(np.linalg.norm(C_vec)), 6)

            # Calculate angles between lattice vectors
            A_mag = np.linalg.norm(A_vec)
            B_mag = np.linalg.norm(B_vec)
            C_mag = np.linalg.norm(C_vec)

            # # Angle between B and C (alpha)
            # alpha = np.arccos(np.dot(B_vec, C_vec) / (B_mag * C_mag))
            # material_data['alpha'] = round(float(np.degrees(alpha)), 3)
            #
            # # Angle between A and C (beta)
            # beta = np.arccos(np.dot(A_vec, C_vec) / (A_mag * C_mag))
            # material_data['beta'] = round(float(np.degrees(beta)), 3)
            #
            # # Angle between A and B (gamma)
            # gamma = np.arccos(np.dot(A_vec, B_vec) / (A_mag * B_mag))
            # material_data['gamma'] = round(float(np.degrees(gamma)), 3)

            # print(A_mag, B_mag, C_mag, float(np.degrees(alpha)), float(np.degrees(beta)), gamma)
            # exit(0)
            material_data['volume'] = round(A_mag * B_mag * C_mag, 4)
            material_data['density'] = round(structure.density, 4)

        # Electronic properties
        material_data['band_gap'] = summary.band_gap if hasattr(summary, 'band_gap') else np.nan
        material_data['cbm'] = summary.cbm if hasattr(summary, 'cbm') else np.nan
        material_data['vbm'] = summary.vbm if hasattr(summary, 'vbm') else np.nan
        material_data['is_gap_direct'] = summary.is_gap_direct if hasattr(summary, 'is_gap_direct') else False

        # Magnetic properties
        material_data['total_magnetization'] = summary.total_magnetization if hasattr(summary,
                                                                                      'total_magnetization') else np.nan

        # Thermodynamic properties
        material_data['energy_above_hull'] = summary.energy_above_hull if hasattr(summary,
                                                                                  'energy_above_hull') else np.nan
        material_data['formation_energy_per_atom'] = summary.formation_energy_per_atom if hasattr(summary,
                                                                                                  'formation_energy_per_atom') else np.nan
        material_data['energy_per_atom'] = summary.energy_per_atom if hasattr(summary, 'energy_per_atom') else np.nan

        # Symmetry properties
        if hasattr(summary, 'symmetry') and summary.symmetry:
            sym = summary.symmetry
            material_data['crystal_system'] = sym.crystal_system if hasattr(sym, 'crystal_system') else 'N/A'
            material_data['space_group_symbol'] = sym.symbol if hasattr(sym, 'symbol') else 'N/A'
            material_data['space_group_number'] = sym.number if hasattr(sym, 'number') else np.nan
            material_data['point_group'] = sym.point_group if hasattr(sym, 'point_group') else 'N/A'
        else:
            material_data['crystal_system'] = 'N/A'
            material_data['space_group_symbol'] = 'N/A'
            material_data['space_group_number'] = np.nan
            material_data['point_group'] = 'N/A'

        # Composition info
        material_data['elements'] = ', '.join([str(e) for e in summary.elements]) if hasattr(summary,
                                                                                             'elements') else 'N/A'

        # Mechanical properties
        material_data['bulk_modulus'] = summary.bulk_modulus if hasattr(summary, 'bulk_modulus') else np.nan
        material_data['shear_modulus'] = summary.shear_modulus if hasattr(summary, 'shear_modulus') else np.nan
        material_data['homogeneous_poisson'] = summary.homogeneous_poisson if hasattr(summary,
                                                                                      'homogeneous_poisson') else np.nan

        # Dielectric properties
        material_data['dielectric_constant'] = summary.dielectric_constant if hasattr(summary,
                                                                                      'dielectric_constant') else np.nan

        # Update date
        material_data['last_updated'] = str(summary.last_updated) if hasattr(summary, 'last_updated') else 'N/A'
        material_data['deprecated'] = summary.deprecated if hasattr(summary, 'deprecated') else False

        return material_data

    except Exception as e:
        print(f"    Error fetching properties for {material_id}: {e}")
        return None


def fetch_all_material_properties(material_ids):
    """
    Step 2: Fetch detailed properties for each material ID
    """
    print("=" * 70)
    print("STEP 2: FETCHING MATERIAL PROPERTIES")
    print("=" * 70)

    all_materials = []

    with MPRester(API_KEY) as mpr:
        for i, material_id in enumerate(material_ids):
            try:
                print(f"  [{i + 1}/{len(material_ids)}] Fetching {material_id}...", end=" ")
                material_data = fetch_material_properties(str(material_id), mpr)

                if material_data:
                    all_materials.append(material_data)
                    print("✓")
                else:
                    print("✗")

                time.sleep(0.05)

            except Exception as e:
                print(f"✗ Error: {e}")
                continue

    print(f"\n✓ Successfully fetched {len(all_materials)}/{len(material_ids)} materials\n")
    return all_materials


def create_dataframe(materials):
    """
    Step 3: Create DataFrame directly from materials list
    """
    print("=" * 70)
    print("STEP 3: CREATING DATAFRAME")
    print("=" * 70)

    df = pd.DataFrame(materials)

    # Reorder columns for better readability
    column_order = [
        'material_id', 'formula_pretty', 'composition', 'elements',
        'a_lattice', 'b_lattice', 'c_lattice',
        'volume', 'density',
        'crystal_system', 'space_group_symbol', 'space_group_number', 'point_group',
        'band_gap', 'cbm', 'vbm', 'is_gap_direct',
        'total_magnetization',
        'energy_above_hull', 'formation_energy_per_atom', 'energy_per_atom',
        'bulk_modulus', 'shear_modulus', 'homogeneous_poisson',
        'dielectric_constant', 'last_updated', 'deprecated'
    ]

    # Keep only columns that exist
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]

    print(f"✓ Created DataFrame with {len(df)} materials and {len(df.columns)} properties")
    print(f"\nColumns: {list(df.columns)}\n")

    return df


def display_dataframe_info(df):
    """
    Display summary information about the DataFrame
    """
    print("=" * 70)
    print("DATAFRAME SUMMARY")
    print("=" * 70)
    print(f"Shape: {df.shape} (rows, columns)")
    print(f"\nFirst 5 entries:")
    print(df[['material_id', 'formula_pretty', 'band_gap', 'total_magnetization', 'energy_above_hull']].head())

    print(f"\nStatistical Summary:")
    print(
        df[['band_gap', 'total_magnetization', 'energy_above_hull', 'a_lattice', 'b_lattice', 'c_lattice']].describe())

    print(f"\nData Types:")
    print(df.dtypes)


def save_dataframe_to_csv(df, filename=OUTPUT_FILE):
    """
    Step 4: Save DataFrame to CSV
    """
    print("=" * 70)
    print("STEP 4: SAVING TO CSV")
    print("=" * 70)
    df.to_csv(filename, index=False)
    print(f"✓ Saved DataFrame to {filename}\n")


# Main execution
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("MATERIALS PROJECT DATA FETCHER - DIRECT TO CSV")
    print("=" * 70 + "\n")

    # Step 1: Get material IDs (or use hardcoded list)
    # material_ids = fetch_material_ids(
    #     formula=["*2*2O7"],
    #     spacegroup_number=["227"]
    # )
    # if not material_ids:
    #     print("No materials found. Exiting.")
    #     exit(1)

    material_ids = ["mp-1101257"]

    # Step 2: Fetch properties for each ID
    all_materials = fetch_all_material_properties(material_ids)

    if not all_materials:
        print("No materials were successfully fetched. Exiting.")
        exit(1)

    # Step 3: Create DataFrame directly
    df = create_dataframe(all_materials)

    # Display results
    display_dataframe_info(df)

    # Step 4: Save to CSV
    # save_dataframe_to_csv(df)

    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"✓ CSV file: {OUTPUT_FILE}")
    print(f"✓ Total materials: {len(df)}")
