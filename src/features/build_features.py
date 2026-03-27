"""
Feature Engineering for Pyrochlore Oxide Dataset
Calculates lattice parameters, thermal properties, and compositional features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Atomic radii (Shannon ionic radii in Angstroms, 6-coordination)
IONIC_RADII = {
    'La': 1.032, 'Ce': 1.010, 'Pr': 1.013, 'Nd': 0.983, 'Sm': 0.958,
    'Eu': 0.947, 'Gd': 0.938, 'Tb': 0.923, 'Dy': 0.912, 'Ho': 0.901,
    'Er': 0.890, 'Tm': 0.880, 'Yb': 0.868, 'Lu': 0.861, 'Y': 0.900,
    'Ti': 0.605, 'Zr': 0.72, 'Hf': 0.71, 'Sn': 0.69, 'Ir': 0.625,
    'O': 1.40
}

# Molar masses (g/mol)
MOLAR_MASSES = {
    'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24, 'Sm': 150.36,
    'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93, 'Dy': 162.50, 'Ho': 164.93,
    'Er': 167.26, 'Tm': 168.93, 'Yb': 173.04, 'Lu': 174.97, 'Y': 88.91,
    'Ti': 47.87, 'Zr': 91.22, 'Hf': 178.49, 'Sn': 118.71, 'Ir': 192.22,
    'O': 16.00
}

# Electronegativity (Pauling scale)
ELECTRONEGATIVITY = {
    'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14, 'Sm': 1.17,
    'Eu': 1.20, 'Gd': 1.20, 'Tb': 1.22, 'Dy': 1.23, 'Ho': 1.24,
    'Er': 1.24, 'Tm': 1.25, 'Yb': 1.10, 'Lu': 1.27, 'Y': 1.22,
    'Ti': 1.54, 'Zr': 1.33, 'Hf': 1.30, 'Sn': 1.96, 'Ir': 2.20,
    'O': 3.44
}


class PyrochloreFeatureEngine:
    """Calculate advanced features for pyrochlore oxide dataset"""

    def __init__(self):
        self.ionic_radii = IONIC_RADII
        self.molar_masses = MOLAR_MASSES
        self.electronegativity = ELECTRONEGATIVITY

    def parse_composition(self, comp_str: str) -> Dict[str, float]:
        """
        Parse composition string into element:fraction dictionary
        Handles formats: "La,Gd,Lu" (equiatomic), "Pr,Sm,Gd,Ho,Lu" (5-element)
        """
        if pd.isna(comp_str) or comp_str == '':
            return {}

        elements = [e.strip() for e in str(comp_str).split(',')]
        n_elements = len(elements)

        # Assume equiatomic distribution
        fractions = {elem: 1.0 / n_elements for elem in elements}
        return fractions

    def calculate_configurational_entropy(self, composition: Dict[str, float]) -> float:
        """
        Shannon entropy for A-site or B-site cations
        S_config = -R * Σ(x_i * ln(x_i))
        """
        if not composition:
            return np.nan

        fractions = list(composition.values())
        entropy = -np.sum([f * np.log(f) for f in fractions if f > 0])
        return entropy

    def calculate_mean_ionic_radius(self, composition: Dict[str, float]) -> float:
        """Weighted average ionic radius"""
        if not composition:
            return np.nan

        r_mean = sum(self.ionic_radii.get(elem, np.nan) * frac
                     for elem, frac in composition.items())
        return r_mean

    def calculate_ionic_radius_variance(self, composition: Dict[str, float]) -> float:
        """Variance in ionic radii (measure of lattice distortion)"""
        if not composition:
            return np.nan

        r_mean = self.calculate_mean_ionic_radius(composition)
        if np.isnan(r_mean):
            return np.nan

        variance = sum(frac * (self.ionic_radii.get(elem, np.nan) - r_mean) ** 2
                       for elem, frac in composition.items())
        return variance

    def calculate_octahedral_factor(self, composition: Dict[str, float]) -> float:
        """
        Octahedral tilting factor (tolerance factor variant for pyrochlores)
        ΔO = Σ|r_i - r_avg| / r_avg (normalized radius variance)
        """
        if not composition:
            return np.nan

        r_mean = self.calculate_mean_ionic_radius(composition)
        if np.isnan(r_mean) or r_mean == 0:
            return np.nan

        delta = sum(frac * abs(self.ionic_radii.get(elem, np.nan) - r_mean)
                    for elem, frac in composition.items()) / r_mean
        return delta

    def calculate_electronegativity_variance(self, composition: Dict[str, float]) -> float:
        """Variance in electronegativity"""
        if not composition:
            return np.nan

        en_mean = sum(self.electronegativity.get(elem, np.nan) * frac
                      for elem, frac in composition.items())

        en_var = sum(frac * (self.electronegativity.get(elem, np.nan) - en_mean) ** 2
                     for elem, frac in composition.items())
        return en_var

    def calculate_molar_mass(self, a_site: Dict[str, float],
                             b_site: Dict[str, float]) -> float:
        """
        Calculate molar mass of A₂B₂O₇ compound
        """
        a_mass = 2 * sum(self.molar_masses.get(elem, np.nan) * frac
                         for elem, frac in a_site.items())
        b_mass = 2 * sum(self.molar_masses.get(elem, np.nan) * frac
                         for elem, frac in b_site.items())
        o_mass = 7 * self.molar_masses['O']

        return a_mass + b_mass + o_mass

    def calculate_lattice_volume(self, lattice_param_a: float) -> float:
        """
        Volume of cubic pyrochlore unit cell
        V = a³ (in Angstrom³)
        """
        if pd.isna(lattice_param_a) or lattice_param_a <= 0:
            return np.nan

        return lattice_param_a ** 3

    def calculate_density_theoretical(self, a_site: Dict[str, float],
                                      b_site: Dict[str, float],
                                      lattice_param_a: float) -> float:
        """
        Theoretical density from lattice parameter
        ρ = (M * Z) / (V * N_A)
        where Z=8 formula units per unit cell
        """
        if pd.isna(lattice_param_a) or lattice_param_a <= 0:
            return np.nan

        M = self.calculate_molar_mass(a_site, b_site)
        V = self.calculate_lattice_volume(lattice_param_a) * 1e-30  # Convert Å³ to cm³
        N_A = 6.022e23
        Z = 8  # Pyrochlore cubic unit cell

        density = (M * Z) / (V * N_A)
        return density

    def calculate_lattice_distortion_index(self, lattice_param_a: float,
                                           reference_param: float = 10.2) -> float:
        """
        Lattice distortion from ideal cubic pyrochlore
        Normalized deviation from reference lattice parameter
        """
        if pd.isna(lattice_param_a):
            return np.nan

        return abs(lattice_param_a - reference_param) / reference_param

    def calculate_phonon_scattering_factor(self, entropy: float,
                                           radius_variance: float) -> float:
        """
        Composite factor for phonon scattering (predicts low thermal conductivity)
        Higher value = more phonon scattering = lower κ
        F_phonon = S_config * Δr²
        """
        if pd.isna(entropy) or pd.isna(radius_variance):
            return np.nan

        return entropy * radius_variance

    def calculate_a_site_features(self, a_composition_str: str) -> Dict[str, float]:
        """Calculate all A-site specific features"""
        a_comp = self.parse_composition(a_composition_str)

        return {
            'a_site_entropy': self.calculate_configurational_entropy(a_comp),
            'a_site_mean_radius': self.calculate_mean_ionic_radius(a_comp),
            'a_site_radius_variance': self.calculate_ionic_radius_variance(a_comp),
            'a_site_radius_std': np.sqrt(self.calculate_ionic_radius_variance(a_comp))
            if not pd.isna(self.calculate_ionic_radius_variance(a_comp)) else np.nan,
            'a_site_octahedral_factor': self.calculate_octahedral_factor(a_comp),
            'a_site_en_variance': self.calculate_electronegativity_variance(a_comp),
            'a_site_n_elements': len(a_comp)
        }

    def calculate_b_site_features(self, b_composition_str: str) -> Dict[str, float]:
        """Calculate all B-site specific features"""
        b_comp = self.parse_composition(b_composition_str)

        return {
            'b_site_entropy': self.calculate_configurational_entropy(b_comp),
            'b_site_mean_radius': self.calculate_mean_ionic_radius(b_comp),
            'b_site_radius_variance': self.calculate_ionic_radius_variance(b_comp),
            'b_site_radius_std': np.sqrt(self.calculate_ionic_radius_variance(b_comp))
            if not pd.isna(self.calculate_ionic_radius_variance(b_comp)) else np.nan,
            'b_site_octahedral_factor': self.calculate_octahedral_factor(b_comp),
            'b_site_en_variance': self.calculate_electronegativity_variance(b_comp),
            'b_site_n_elements': len(b_comp)
        }

    def calculate_all_features(self, row: pd.Series) -> Dict[str, float]:
        """
        Calculate all features for a single sample
        """
        a_comp = self.parse_composition(row['Sample A'])
        b_comp = self.parse_composition(row['Sample B'])

        features = {}

        # A-site features
        features.update(self.calculate_a_site_features(row['Sample A']))

        # B-site features
        features.update(self.calculate_b_site_features(row['Sample B']))

        # Combined features
        features['total_entropy'] = (features['a_site_entropy'] +
                                     features['b_site_entropy'])
        features['total_radius_variance'] = (features['a_site_radius_variance'] +
                                             features['b_site_radius_variance'])

        # Lattice features
        if not pd.isna(row.get('Lattice Parameter (Angstrom)')):
            lattice_a = row['Lattice Parameter (Angstrom)']
        elif not pd.isna(row.get('Lattice Parameter a (A)')):
            lattice_a = row['Lattice Parameter a (A)']
        else:
            lattice_a = np.nan

        features['lattice_volume'] = self.calculate_lattice_volume(lattice_a)
        features['density_theoretical'] = self.calculate_density_theoretical(
            a_comp, b_comp, lattice_a)
        features['lattice_distortion_index'] = self.calculate_lattice_distortion_index(
            lattice_a)

        # Thermal property predictors
        features['phonon_scattering_factor'] = self.calculate_phonon_scattering_factor(
            features['total_entropy'], features['total_radius_variance'])

        # Normalized features
        if not pd.isna(row.get('TPS Cond W/m/K')):
            features['thermal_conductivity_raw'] = row['TPS Cond W/m/K']

        if not pd.isna(row.get('Relative Density %')):
            features['relative_density'] = row['Relative Density %']

        return features


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered features to dataframe
    """
    engine = PyrochloreFeatureEngine()

    # Calculate features for each row
    feature_dicts = []
    for idx, row in df.iterrows():
        try:
            features = engine.calculate_all_features(row)
            feature_dicts.append(features)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            feature_dicts.append({})

    # Convert to dataframe and merge
    features_df = pd.DataFrame(feature_dicts)
    df_enhanced = pd.concat([df, features_df], axis=1)

    return df_enhanced


if __name__ == "__main__":
    # Example usage
    print("Pyrochlore Feature Engineering Module")
    print("=" * 60)

    engine = PyrochloreFeatureEngine()

    # Test with example composition
    test_a = "Pr,Sm,Gd,Ho,Lu"
    test_b = "Ti"

    a_features = engine.calculate_a_site_features(test_a)
    b_features = engine.calculate_b_site_features(test_b)

    print(f"\nA-site ({test_a}):")
    for key, val in a_features.items():
        if not pd.isna(val):
            print(f"  {key}: {val:.6f}")

    print(f"\nB-site ({test_b}):")
    for key, val in b_features.items():
        if not pd.isna(val):
            print(f"  {key}: {val:.6f}")