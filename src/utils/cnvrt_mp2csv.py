import json
import pandas as pd
import numpy as np
from pymatgen.core import Structure
from pathlib import Path

_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent
DATA_DIR  = _PROJECT / 'data' / 'raw'
INPUT_FILE = DATA_DIR / 'mp_pyrochlore_query.json'
OUTPUT_FILE = DATA_DIR / 'mp_pyrochlore_query.csv'

def load_materials_dataframe(json_file=INPUT_FILE):
    """
    Load JSON file with material data and create a DataFrame
    with extracted lattice parameters, band gap, magnetization, etc.

    Args:
        json_file: Path to JSON file with material data

    Returns:
        DataFrame with material properties
    """

    # Load JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Ensure it's a list
    if isinstance(data, dict):
        data = [data]

    materials = []

    for entry in data:
        try:
            material = {}

            # Basic properties
            material['mp_id'] = entry.get('material_id', 'N/A')
            material['composition'] = entry.get('formula_pretty', 'N/A')
            material['reduced_composition'] = entry.get('formula_anonymous', 'N/A')

            # Extract lattice parameters from structure
            if 'structure' in entry:
                structure_data = entry['structure']

                # Reconstruct Structure object if it's a dict
                if isinstance(structure_data, dict):
                    try:
                        structure = Structure.from_dict(structure_data)
                    except:
                        structure = None
                else:
                    structure = structure_data

                if structure:
                    # Lattice parameters (a, b, c)
                    material['a_lattice'] = round(structure.lattice.a, 6)
                    material['b_lattice'] = round(structure.lattice.b, 6)
                    material['c_lattice'] = round(structure.lattice.c, 6)

                    # Lattice angles (alpha, beta, gamma)
                    material['alpha_angle'] = round(structure.lattice.alpha, 3)
                    material['beta_angle'] = round(structure.lattice.beta, 3)
                    material['gamma_angle'] = round(structure.lattice.gamma, 3)

                    # Volume
                    material['volume'] = round(structure.volume, 4)
                else:
                    material['a_lattice'] = np.nan
                    material['b_lattice'] = np.nan
                    material['c_lattice'] = np.nan
                    material['alpha_angle'] = np.nan
                    material['beta_angle'] = np.nan
                    material['gamma_angle'] = np.nan
                    material['volume'] = np.nan
            else:
                material['a_lattice'] = np.nan
                material['b_lattice'] = np.nan
                material['c_lattice'] = np.nan
                material['alpha_angle'] = np.nan
                material['beta_angle'] = np.nan
                material['gamma_angle'] = np.nan
                material['volume'] = np.nan

            # Electronic properties
            material['band_gap'] = entry.get('band_gap', np.nan)
            material['cbm'] = entry.get('cbm', np.nan)
            material['vbm'] = entry.get('vbm', np.nan)
            material['efermi'] = entry.get('efermi', np.nan)
            material['is_metal'] = entry.get('is_metal', False)

            # Magnetic properties
            material['total_magnetization'] = entry.get('total_magnetization', np.nan)
            material['total_magnetization_normalized_vol'] = entry.get('total_magnetization_normalized_vol', np.nan)
            material['is_magnetic'] = entry.get('is_magnetic', False)
            material['ordering'] = entry.get('ordering', 'N/A')
            material['num_magnetic_sites'] = entry.get('num_magnetic_sites', np.nan)

            # Thermodynamic properties
            material['energy_above_hull'] = entry.get('energy_above_hull', np.nan)
            material['formation_energy_per_atom'] = entry.get('formation_energy_per_atom', np.nan)
            material['energy_per_atom'] = entry.get('energy_per_atom', np.nan)
            material['uncorrected_energy_per_atom'] = entry.get('uncorrected_energy_per_atom', np.nan)
            material['is_stable'] = entry.get('is_stable', False)

            # Density
            material['density'] = entry.get('density', np.nan)
            material['density_atomic'] = entry.get('density_atomic', np.nan)

            # Symmetry properties
            if 'symmetry' in entry and entry['symmetry']:
                sym = entry['symmetry']
                material['crystal_system'] = sym.get('crystal_system', 'N/A')
                material['space_group'] = sym.get('symbol', 'N/A')
                material['space_group_number'] = sym.get('number', np.nan)
                material['point_group'] = sym.get('point_group', 'N/A')
            else:
                material['crystal_system'] = 'N/A'
                material['space_group'] = 'N/A'
                material['space_group_number'] = np.nan
                material['point_group'] = 'N/A'

            # Composition info
            material['num_sites'] = entry.get('nsites', np.nan)
            material['num_elements'] = entry.get('nelements', np.nan)
            material['elements'] = ', '.join(entry.get('elements', []))

            # Mechanical properties (if available)
            material['bulk_modulus'] = entry.get('bulk_modulus', np.nan)
            material['shear_modulus'] = entry.get('shear_modulus', np.nan)

            # Update date
            material['last_updated'] = entry.get('last_updated', 'N/A')
            material['deprecated'] = entry.get('deprecated', False)

            materials.append(material)

        except Exception as e:
            print(f"Error processing entry {entry.get('material_id', 'Unknown')}: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(materials)

    return df


def save_dataframe_csv(df, output_file=OUTPUT_FILE):
    """Save DataFrame to CSV"""
    df.to_csv(output_file, index=False)
    print(f"✓ Saved DataFrame to {output_file}")
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")


# Main execution
if __name__ == '__main__':
    # Load and process
    df = load_materials_dataframe(INPUT_FILE)

    # Display info
    print(f"\n{'=' * 60}")
    print(f"Materials DataFrame Summary")
    print(f"{'=' * 60}")
    print(f"Total materials: {len(df)}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nDataFrame Info:")
    print(df.info())
    print(f"\nFirst few entries:")
    print(df.head())

    # Show statistics for numerical columns
    print(f"\nStatistics:")
    print(df.describe())

    # Save to CSV
    save_dataframe_csv(df)
