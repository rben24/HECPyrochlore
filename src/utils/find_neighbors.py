"""
Pyrochlore Structure Analyzer — pymatgen backend
=================================================
All distance calculations and PBC handling are delegated to pymatgen's
Structure.get_neighbors(), which is correct for any cell geometry.

Workflow
--------
1.  Build the structure from lattice parameters + fractional coordinates
    (or pass in an existing pymatgen Structure).
2.  Call suggest_cutoffs() to auto-detect bond cutoffs from the RDF.
3.  Call analyze() to get per-atom coordination numbers.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from pymatgen.core import Lattice, Structure


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PyrochloreAnalyzer:
    """
    Coordination-environment analyzer for pyrochlore (and related) structures.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        Fully initialised Structure object.  Use the class method
        ``from_parameters()`` to build one from raw lattice parameters and
        fractional coordinates.
    """

    def __init__(self, structure: Structure):
        self.structure   = structure
        # preserve insertion order of species
        self.species_list = list(dict.fromkeys(str(s.specie) for s in structure))
        self.atom_map: Dict[str, List[int]] = defaultdict(list)
        for i, site in enumerate(structure):
            self.atom_map[str(site.specie)].append(i)

    # ------------------------------------------------------------------
    # Constructor helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_parameters(cls,
                        lattice_params: Tuple[float, float, float,
                                              float, float, float],
                        atom_sites: Dict[str, List[List[float]]]) -> "PyrochloreAnalyzer":
        """
        Build an analyzer directly from lattice parameters and fractional
        coordinates — no pymatgen boilerplate needed in your script.

        Parameters
        ----------
        lattice_params : (a, b, c, alpha_deg, beta_deg, gamma_deg)
        atom_sites     : {'Species': [[x,y,z], ...], ...}  fractional coords
        """
        a, b, c, alpha, beta, gamma = lattice_params
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

        species, coords = [], []
        for sp, positions in atom_sites.items():
            for pos in positions:
                species.append(sp)
                coords.append(pos)

        structure = Structure(lattice, species, coords)
        return cls(structure)

    # ------------------------------------------------------------------
    # Distance collection (pymatgen backend — PBC always correct)
    # ------------------------------------------------------------------

    def get_all_distances(self,
                          center_sp:   str,
                          neighbor_sp: str,
                          r_max:       float = 8.0,
                          ) -> Tuple[np.ndarray, Dict[int, List[float]]]:
        """
        Collect all center→neighbor distances within r_max using pymatgen's
        neighbour search (handles arbitrary cell geometry and PBC).

        Returns
        -------
        all_dists    : 1-D array of every distance found
        per_site     : {site_index: [sorted distances], ...}  for each center atom
        """
        all_dists: List[float]              = []
        per_site:  Dict[int, List[float]]   = {}

        for i in self.atom_map[center_sp]:
            site      = self.structure[i]
            neighbors = self.structure.get_neighbors(site, r_max)
            dists     = sorted(n.nn_distance for n in neighbors
                               if str(n.specie) == neighbor_sp)
            all_dists.extend(dists)
            per_site[i] = dists

            print(per_site)
            exit(0)

        return np.array(all_dists), per_site

    # ------------------------------------------------------------------
    # RDF-based automatic cutoff suggestion
    # ------------------------------------------------------------------

    def suggest_cutoffs(self,
                        r_max:   float = 6.0,
                        n_bins:  int   = 300,
                        sigma:   float = 3.0,
                        pairs:   Optional[List[Tuple[str, str]]] = None,
                        plot:    bool  = False,
                        verbose: bool  = True,
                        ) -> Dict[Tuple[str, str], float]:
        """
        Auto-suggest bond cutoffs by finding the first valley in the smoothed
        pair distribution (RDF) for every requested species pair.

        Steps
        -----
        1. Histogram all pairwise distances up to r_max.
        2. Smooth with a Gaussian kernel of width `sigma` bins.
        3. Locate the first peak  → bonding shell.
        4. Locate the first valley after that peak → natural cutoff.
        5. Validate the valley: drop must be >10 % of the peak height;
           otherwise fall back to the midpoint between the peak and r_max.

        Parameters
        ----------
        r_max   : float  Scan distance (Å).  Set a bit above the expected 2nd shell.
        n_bins  : int    Histogram resolution (default 300).
        sigma   : float  Gaussian smoothing in bin units (default 3).
                         Increase (5–8) for noisy/defective structures.
        pairs   : list   Specific (center, neighbor) pairs.  None → all cross-pairs.
        plot    : bool   Show smoothed RDF with detected cutoffs (requires matplotlib).
        verbose : bool   Print suggested cutoffs to stdout.

        Returns
        -------
        dict : {(center_species, neighbor_species): cutoff_Å, ...}

        Tuning tips
        -----------
        - Cutoff too short / misses bonds  →  lower sigma or raise r_max.
        - Cutoff splits a single peak      →  raise sigma.
        - Still wrong                      →  use plot=True to inspect, then
                                              pass explicit cutoffs to analyze().
        """
        if pairs is None:
            pairs = [(c, n) for c in self.species_list
                            for n in self.species_list if c != n]

        bin_edges   = np.linspace(0.0, r_max, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        suggested: Dict[Tuple[str, str], float] = {}

        if plot:
            import matplotlib.pyplot as plt
            ncols = min(3, len(pairs))
            nrows = (len(pairs) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(5 * ncols, 3.5 * nrows))
            axes = np.array(axes).flatten()

        if verbose:
            print("=== Suggested cutoffs (Å) ===")

        for k, (center, neighbor) in enumerate(pairs):
            dists, _ = self.get_all_distances(center, neighbor, r_max)
            dists     = dists[(dists > 1e-6) & (dists <= r_max)]

            if len(dists) == 0:
                suggested[(center, neighbor)] = r_max
                continue

            counts, _ = np.histogram(dists, bins=bin_edges)
            smooth     = gaussian_filter1d(counts.astype(float), sigma=sigma)

            # --- first peak ---
            peaks, _ = find_peaks(smooth)
            if len(peaks) == 0:
                cutoff = float(np.median(dists))
                suggested[(center, neighbor)] = cutoff
                continue
            first_peak = peaks[0]

            # --- first valley after the first peak ---
            valleys, _ = find_peaks(-smooth[first_peak:])
            if len(valleys) > 0:
                valley_idx = valleys[0] + first_peak
                peak_val   = smooth[first_peak]
                valley_val = smooth[valley_idx]
                # accept only if there is a real dip (>10 % drop)
                if peak_val > 0 and (peak_val - valley_val) / peak_val > 0.10:
                    cutoff = float(bin_centers[valley_idx])
                else:
                    cutoff = float(0.5 * (bin_centers[first_peak] + r_max))
            else:
                cutoff = float(0.5 * (bin_centers[first_peak] + r_max))

            suggested[(center, neighbor)] = cutoff

            if verbose:
                print(f"  {center}-{neighbor}: {cutoff:.3f} Å  "
                      f"(1st peak @ {bin_centers[first_peak]:.3f} Å)")

            if plot:
                ax = axes[k]
                ax.plot(bin_centers, smooth, lw=1.5)
                ax.axvline(cutoff, color="red", ls="--", label=f"cutoff {cutoff:.2f} Å")
                ax.axvline(bin_centers[first_peak], color="green", ls=":",
                           label=f"peak {bin_centers[first_peak]:.2f} Å")
                ax.set_title(f"{center}–{neighbor}")
                ax.set_xlabel("Distance (Å)")
                ax.set_ylabel("Count")
                ax.legend(fontsize=7)

        if plot:
            for ax in axes[len(pairs):]:
                ax.set_visible(False)
            plt.tight_layout()
            plt.show()

        return suggested

    # ------------------------------------------------------------------
    # Coordination analysis
    # ------------------------------------------------------------------

    def get_coordination(self,
                         center_sp:   str,
                         neighbor_sp: str,
                         cutoff:      float,
                         ) -> Dict[int, int]:
        """
        Return {site_index: coordination_number} for every center atom,
        counting only neighbor_sp atoms within `cutoff` Å.
        """
        coord: Dict[int, int] = {}
        for i in self.atom_map[center_sp]:
            site      = self.structure[i]
            neighbors = self.structure.get_neighbors(site, cutoff)
            coord[i]  = sum(1 for n in neighbors if str(n.specie) == neighbor_sp)
        return coord

    def analyze(self,
                cutoffs: Optional[Dict[Tuple[str, str], float]] = None,
                r_max:   float = 6.0,
                verbose: bool  = True,
                ) -> Dict[Tuple[str, str], dict]:
        """
        Full coordination analysis for all cross-species pairs.

        Parameters
        ----------
        cutoffs : dict, optional
            {(center, neighbor): cutoff_Å}.  If None, cutoffs are suggested
            automatically from the RDF.
        r_max   : float  Passed to suggest_cutoffs() when cutoffs is None.
        verbose : bool   Print results.

        Returns
        -------
        dict : {(center, neighbor): {
                    'cutoff':               float,
                    'coordination_numbers': {site_idx: int, ...},
                    'mean':                 float,
                    'min':                  int,
                    'max':                  int,
                }}
        """
        if cutoffs is None:
            if verbose:
                print("No cutoffs provided — suggesting from RDF...\n")
            cutoffs = self.suggest_cutoffs(r_max=r_max, verbose=verbose)

        pairs = [(c, n) for c in self.species_list
                        for n in self.species_list if c != n]

        if verbose:
            print("\n=== Coordination Analysis ===")

        results: Dict[Tuple[str, str], dict] = {}

        for center, neighbor in pairs:
            key = (center, neighbor)
            if key not in cutoffs:
                continue

            cutoff   = cutoffs[key]
            coord    = self.get_coordination(center, neighbor, cutoff)
            values   = list(coord.values())

            results[key] = {
                "cutoff":               cutoff,
                "coordination_numbers": coord,
                "mean":                 float(np.mean(values)),
                "min":                  int(np.min(values)),
                "max":                  int(np.max(values)),
            }

            if verbose:
                print(f"\n{neighbor} neighbors around each {center}"
                      f"  (cutoff = {cutoff:.3f} Å)")
                print(f"  Per atom : {values}")
                print(f"  Mean / Min / Max : "
                      f"{np.mean(values):.1f} / {np.min(values)} / {np.max(values)}")

        return results

    # ------------------------------------------------------------------
    # Summary / diagnostics
    # ------------------------------------------------------------------

    def print_summary(self):
        """Print structure summary including Cartesian coordinates."""
        lp = self.structure.lattice
        print("=== Pyrochlore Structure ===")
        print(f"Lattice : a={lp.a:.4f}  b={lp.b:.4f}  c={lp.c:.4f} Å")
        print(f"          α={lp.alpha:.2f}°  β={lp.beta:.2f}°  γ={lp.gamma:.2f}°")
        print(f"Volume  : {lp.volume:.3f} Å³")

        print("\nComposition:")
        for sp in self.species_list:
            print(f"  {sp}: {len(self.atom_map[sp])} atoms")

        print(f"\n{'Element':>8} | {'Fractional coords':^36} | Cartesian coords (Å)")
        print("-" * 80)
        for site in self.structure:
            fc = site.frac_coords
            cc = site.coords
            print(f"{str(site.specie):>8} | "
                  f"[{fc[0]:7.4f}, {fc[1]:7.4f}, {fc[2]:7.4f}] | "
                  f"[{cc[0]:7.4f}, {cc[1]:7.4f}, {cc[2]:7.4f}]")


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    atom_sites = {
        # "La": [
        #     [0.5, 0.5, 0.5],
        #     [0.5, 0.5, 0.0],
        #     [0.0, 0.5, 0.5],
        #     [0.5, 0.0, 0.5],
        # ],
        # "Zr": [
        #     [0.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.5],
        #     [0.5, 0.0, 0.0],
        #     [0.0, 0.5, 0.0],
        # ],
        # "O": [
        #     [0.637, 0.113, 0.637],
        #     [0.113, 0.637, 0.113],
        #     [0.113, 0.637, 0.637],
        #     [0.637, 0.113, 0.113],
        #     [0.637, 0.637, 0.113],
        #     [0.113, 0.113, 0.637],
        #     [0.363, 0.887, 0.887],
        #     [0.887, 0.363, 0.363],
        #     [0.887, 0.363, 0.887],
        #     [0.363, 0.887, 0.363],
        #     [0.363, 0.363, 0.887],
        #     [0.887, 0.887, 0.363],
        #     [0.875, 0.875, 0.875],
        #     [0.125, 0.125, 0.125],
        # ],
        "Eu": [
            [0.5, 0.5, 0.5],
            [0.5, 0, 0],
            [0.5, 0, 0.5],
            [0, 0, 0.5],
        ],
        "O":  [
            [0.0873, 0.587, 0.663],
            [0.663, 0.163, 0.0873],
            [0.663, 0.587, 0.663],
            [0.0873, 0.163, 0.0873],
            [0.663, 0.587, 0.0873],
            [0.0873, 0.163, 0.663],
            [0.913, 0.837, 0.913],
            [0.337, 0.413, 0.337],
            [0.337, 0.837, 0.913],
            [0.913, 0.413, 0.337],
            [0.337, 0.413, 0.913],
            [0.913, 0.837, 0.337],
            [0.625, 0.125, 0.625],
            [0.375, 0.875, 0.375],
        ],
        "Sn": [
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0, 0.5, 0],
            [0.5, 0.5, 0],
        ],
    }

    # lattice_params = (7.653, 7.653, 7.653, 60.0, 60.0, 60.0)
    lattice_params = (7.608, 7.608, 7.608, 60.0, 60.0, 60.0)

    analyzer = PyrochloreAnalyzer.from_parameters(lattice_params, atom_sites)
    analyzer.print_summary()

    # --- Option A: auto cutoffs from RDF (may need sigma tuning) ---
    results = analyzer.analyze(r_max=6.0)

    # --- Option B: explicit cutoffs (most reliable for relaxed structures) ---
    # cutoffs = {
    #     ("La", "O"):  3.0,
    #     ("Zr", "O"):  2.6,
    #     ("La", "Zr"): 4.5,
    #     ("Zr", "La"): 4.5,
    #     ("O",  "La"): 3.0,
    #     ("O",  "Zr"): 2.6,
    # }
    # results = analyzer.analyze(cutoffs=cutoffs)
