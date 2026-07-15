from mp_api.client import MPRester
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from src.globals import KNOWN_A, KNOWN_B
import numpy as np
import pandas as pd
from pathlib import Path
import time
from dotenv import load_dotenv
import os

# ==================== Setup ====================
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent
DATA = _PROJECT / 'data' / 'raw'
OUTPUT_FILE = DATA / 'element_database.csv'
# OUTPUT_FILE = 'materials_properties.csv'
load_dotenv(_PROJECT / 'src' / '.env')
MP_API_KEY = os.environ.get('MP_API_KEY')

Element_basic = frozenset(KNOWN_A | KNOWN_B)
Element_list = Element_basic.union({'O'})
# Element_list = ['Ta']

print(f"Fetching data for {len(Element_list)} elements")

# ==================== Property Mapping ====================
# Maps standard property names to pymatgen keys
REQUIRED_PROPERTIES = [
    "amass",
    "electronegativity",
    "atomic_radius",
    "Melting point",
    "Coefficient of linear thermal expansion",
    "Thermal conductivity",
    "Vickers hardness",
    "Bulk modulus",
    "Youngs modulus",
    "Shear modulus",
    "Poissons ratio"
]

PROPERTY_MAPPING = {
    "Melting point": "Melting Point",
    "Coefficient of linear thermal expansion": "Thermal Expansion",
    "Thermal conductivity": "Thermal Conductivity",
    "Vickers hardness": "Vickers Hardness",
    "Bulk modulus": "Bulk Modulus",
    "Youngs modulus": "Youngs Modulus",
    "Shear modulus": "Shear Modulus",
    "Poissons ratio": "Poissons Ratio",
}


# ==================== Helper Functions ====================
def _extract_element_properties(el_obj, property_mapping):
    """Extract basic elemental properties using pymatgen Element class."""
    element_dict = {
        'Element': str(el_obj.symbol),
        'Atomic Mass': el_obj.atomic_mass,
        'Electronegativity': el_obj.X,
        'Atomic Radius': el_obj.atomic_radius,
        'Metallic Radius': el_obj.metallic_radius,
    }

    # Extract extended properties from Element.data dictionary
    extended_props = [
        "Melting point",
        "Coefficient of linear thermal expansion",
        "Thermal conductivity",
        "Vickers hardness",
        "Bulk modulus",
        "Youngs modulus",
        "Shear modulus",
        "Poissons ratio"
    ]

    for prop_name in extended_props:
        col_name = property_mapping.get(prop_name, prop_name)
        try:
            value = el_obj.data.get(prop_name)
            element_dict[col_name] = value if value is not None else np.nan
        except:
            element_dict[col_name] = np.nan

    return element_dict


def _calculate_youngs_modulus(bulk_modulus, shear_modulus):
    """
    Calculate Young's modulus from bulk and shear moduli.
    E = 9 * K * G / (3 * K + G)
    """
    if bulk_modulus and shear_modulus:
        try:
            E = (9 * bulk_modulus * shear_modulus) / (3 * bulk_modulus + shear_modulus)
            return E
        except:
            return np.nan
    return np.nan


# ==================== Query Materials Project ====================
Element_props = []

with MPRester(MP_API_KEY) as mpr:
    for ele in Element_list:
        print(f"\nProcessing {ele}...", end=" ")

        try:
            # Query for cubic phase (Fd-3m = space group 227) single elements
            docs = mpr.materials.summary.search(
                formula=ele,
                spacegroup_number=227,  # Fd-3m
                fields=[
                    "material_id",
                    "structure",
                    "density",
                    "volume",
                    "band_gap",
                    "bulk_modulus",
                    "shear_modulus",
                    "homogeneous_poisson",
                    "is_stable",
                    "energy_above_hull",
                    "is_magnetic",
                    "total_magnetization",
                ]
            )

            # If no Fd-3m phase, try most stable cubic phase
            if not docs:
                print(f"No Fd-3m phase found. Searching for most stable cubic phase...", end=" ")
                docs = mpr.materials.summary.search(
                    formula=ele,
                    crystal_system="cubic",
                    fields=[
                        "material_id",
                        "structure",
                        "density",
                        "volume",
                        "band_gap",
                        "bulk_modulus",
                        "shear_modulus",
                        "universal_anisotropy",
                        "homogeneous_poisson",
                        "is_stable",
                        "energy_above_hull",
                        "is_magnetic",
                        "total_magnetization",
                    ]
                )
                docs = sorted(docs, key=lambda x: x.energy_above_hull or float('inf'))

            if not docs:
                print(f"No cubic phase found in MP")
                # Fall back to Element properties only
                el_obj = Element(ele)
                element_dict = _extract_element_properties(el_obj, PROPERTY_MAPPING)
                Element_props.append(element_dict)
                continue

            # Use the most stable/first result
            doc = docs[0]
            print(f"✓ Found MP-{doc.material_id}")

            # Extract elemental properties
            el_obj = Element(ele)
            element_dict = _extract_element_properties(el_obj, PROPERTY_MAPPING)

            # Extract structure properties
            structure = doc.structure
            sga = SpacegroupAnalyzer(structure)

            element_dict.update({
                'Material_ID': str(doc.material_id),
                'Space_Group': sga.get_space_group_number(),
                'Space_Group_Symbol': sga.get_space_group_symbol(),
                'Density': doc.density,
                'Volume': doc.volume,
                'Lattice_Parameter_a': structure.lattice.a,
                # 'Band_Gap': doc.band_gap if doc.band_gap else np.nan,
            })

            if doc.band_gap is not None:
                element_dict['Band_Gap'] = doc.band_gap

            # Extract elastic properties
            if doc.bulk_modulus is not None:
                element_dict['Bulk Modulus'] = doc.bulk_modulus['vrh']


            if doc.shear_modulus is not None:
                    element_dict['Shear Modulus'] = doc.shear_modulus['vrh']

            if doc.bulk_modulus and doc.shear_modulus and not element_dict['Youngs Modulus']:
                youngs_modulus = _calculate_youngs_modulus(
                    doc.bulk_modulus,
                    doc.shear_modulus
                )
                if youngs_modulus:
                    element_dict['Youngs Modulus'] = youngs_modulus

                # Poisson ratio
            if doc.homogeneous_poisson is not None:
                element_dict['Poissons Ratio'] = doc.homogeneous_poisson

                # Anisotropy
            if doc.universal_anisotropy is not None:
                element_dict['Universal Anisotropy'] = doc.universal_anisotropy

                # Magnetic properties (useful for thermal conductivity prediction)
            if doc.is_magnetic:
                element_dict['Is Magnetic'] = True
                element_dict['Total Magnetization'] = doc.total_magnetization if doc.total_magnetization else np.nan
            else:
                element_dict['Is Magnetic'] = False
                element_dict['Total Magnetization'] = np.nan

            Element_props.append(element_dict)
            time.sleep(0.1)  # Rate limiting

        except Exception as e:
            print(f"Error: {e}")
            # Fall back to Element properties only
            try:
                el_obj = Element(ele)
                element_dict = _extract_element_properties(el_obj, PROPERTY_MAPPING)
                Element_props.append(element_dict)
            except:
                print(f"  Skipping {ele}")
                continue

# ==================== Create DataFrame ====================
Element_df = pd.DataFrame(Element_props)

# Reorder columns: Element first, then basic properties, then structure, then elasticity
column_order = [
    'Element', 'Material_ID', 'Atomic Mass', 'Electronegativity',
    'Atomic Radius',
    'Space_Group', 'Space_Group_Symbol', 'Density', 'Volume', 'Lattice_Parameter_a',
    'Band_Gap', 'Is Magnetic', 'Total Magnetization',
    'Melting Point', 'Thermal Expansion', 'Thermal Conductivity',
    'Vickers Hardness', 'Bulk Modulus', 'Youngs Modulus',
    'Shear Modulus', 'Poissons Ratio', 'Universal_Anisotropy'
]

# Add property columns that exist
for col in Element_df.columns:
    if col not in column_order:
        column_order.append(col)

Element_df = Element_df[[col for col in column_order if col in Element_df.columns]]

print("\n" + "=" * 80)
print("Element Database Summary")
print("=" * 80)
print(Element_df.to_string())
print("\nShape:", Element_df.shape)
print("Missing values per column:")
print(Element_df.isnull().sum())

# ==================== Export ====================
Element_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Exported to {OUTPUT_FILE}")


