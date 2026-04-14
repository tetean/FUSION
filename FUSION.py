import torch
import numpy as np
import os, sys
import yaml
import pickle
import time
import json
import scipy.optimize as optimize
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from models import ALIGNN
from models.alignn import Graph, load_alignn_data
from jarvis.core.atoms import Atoms
import pandas as pd
from tqdm import tqdm
import random
from dscribe.descriptors import SOAP
from ase.io import read
from ase import Atoms as ASEAtoms
from ase.data import chemical_symbols
import spglib
from pathlib import Path
import warnings
from sklearn.model_selection import train_test_split
from datetime import datetime
from utils import timer_and_log
from EpsilonSearcher import EpsilonSearcher

warnings.filterwarnings('ignore')


def ts():
    """Return current timestamp string in %Y%m%d-%H%M%S format."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class FUSION:
    """
    FUSION: Materials-Aware Pruning Strategy for identifying redundant crystal structures
    in materials datasets while preserving model performance using dynamic value functions and
    combinatorial optimization techniques.

    FUSION combines:
    - Dynamic value functions that consider sample relationships
    - Multi-objective optimization (quality, diversity, coverage)
    - Adaptive optimization strategies (beam search, enhanced SA)
    - Context-aware sample selection
    """

    def __init__(self, config_path, checkpoint_path, data_path, task, epsilon=0.1, gamma=0.1, Lambda=1, cache_dir="./fusion_cache"):
        """
        Initialize the FUSION pruner.

        Args:
            config_path (str): Path to the model configuration file
            checkpoint_path (str): Path to the model checkpoint
            data_path (str): Path to the dataset
            task (str): Task name (e.g., "dielectric", "formation_energy")
            epsilon (float): Threshold for uncertainty influence
            gamma (float): Hyperparameter controlling the influence of structural similarity
            Lambda (float): Hyperparameter for uncertainty computation
            cache_dir (str): Directory to store cached results
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.task = task
        self.epsilon = epsilon
        self.gamma = gamma
        self.cache_dir = cache_dir
        self.Lambda = Lambda

        os.makedirs(cache_dir, exist_ok=True)

        self.uncertainties_cache = os.path.join(cache_dir, f"{task}_uncertainties.pkl")
        self.soap_cache          = os.path.join(cache_dir, f"{task}_soap_features.pkl")
        self.influence_cache     = os.path.join(cache_dir, f"{task}_influence_scores.pkl")
        self.structures_cache    = os.path.join(cache_dir, f"{task}_structures.pkl")
        self.structure_paths_cache = os.path.join(cache_dir, f"{task}_structure_paths.pkl")

        with open(config_path, "r") as ymlfile:
            self.config = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.config["Models"] = self.config["Models"].get("ALIGNN")

        self.model = ALIGNN(evidential=self.config["Models"]["evidential"],
                            **(self.config["Models"]["model_setting"]))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(device)
        self.model.eval()
        self.device = device

        self.data_loader = load_alignn_data(data_path, task, self.config["Models"])

        self.structures      = []
        self.uncertainties   = []
        self.targets         = []
        self.structure_ids   = []
        self.structure_paths = []
        self.soap_features   = None
        self.influence_scores = None
        self.pruning_mask    = None

        # SOAP descriptor configuration.
        # NOTE: 'species' is intentionally absent here; it is supplied as the
        # global vocabulary argument to calculate_geometric_fingerprint() so
        # that every fingerprint is computed against the same species set and
        # therefore has a consistent, comparable dimension.
        self.soap_params = {
            'r_cut': 5.0,
            'n_max': 8,
            'l_max': 6,
            'sigma': 0.5,
            'periodic': True,
            'average': 'off',
            'sparse': False,
        }

        self.symprec = 1e-5

        self.distance_matrix     = None
        self.neighbor_distances  = None
        self.neighbor_indices    = None
        self.diversity_scores    = None
        self.influence_clusters  = None
        self.value_cache         = {}

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_from_cache(self, cache_path):
        """Load data from cache if it exists."""
        if os.path.exists(cache_path):
            print(f"[{ts()}] Loading cached data from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, data, cache_path):
        """Save data to cache."""
        print(f"[{ts()}] Saving data to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

    # ------------------------------------------------------------------
    # Uncertainty computation
    # ------------------------------------------------------------------

    def compute_uncertainties(self, force_recompute=False):
        """
        Compute uncertainties for all structures using deep evidential regression.
        Also extract CIF file paths from the dataset.

        Args:
            force_recompute (bool): Force recomputation even if cache exists
        """
        if not force_recompute:
            cached_data = self._load_from_cache(self.uncertainties_cache)
            if cached_data is not None:
                self.uncertainties   = cached_data['uncertainties']
                self.targets         = cached_data['targets']
                self.structure_ids   = cached_data['structure_ids']
                self.structures      = cached_data['structures']
                self.structure_paths = cached_data.get('structure_paths', [])
                print(f"[{ts()}] Loaded {len(self.uncertainties)} uncertainties from cache")
                print(f"[{ts()}] Uncertainties shape: {self.uncertainties.shape}")
                return

        print(f"[{ts()}] Computing uncertainties for all structures...")
        uncertainties_list    = []
        targets_list          = []
        structure_ids_list    = []
        structures_list       = []
        structure_paths_list  = []

        with torch.no_grad():
            for i, data in enumerate(tqdm(self.data_loader, desc="Computing uncertainties")):
                input_data = data[0]
                target     = data[1]

                if len(data) > 2:
                    structure_info = data[2]
                    if isinstance(structure_info, dict):
                        structure_id   = structure_info.get('jid', f"structure_{i}")
                        structure_path = structure_info.get('cif_path', '')
                    elif isinstance(structure_info, (list, tuple)):
                        structure_id   = structure_info[0] if len(structure_info) > 0 else f"structure_{i}"
                        structure_path = structure_info[1] if len(structure_info) > 1 else ''
                    else:
                        structure_id   = str(structure_info)
                        structure_path = ''
                else:
                    structure_id   = f"structure_{i}"
                    structure_path = ''

                input_var = [input_data[0].to(self.device), input_data[1].to(self.device)]
                output    = self.model(input_var)

                if isinstance(output, (list, tuple)) and len(output) >= 4:
                    mu    = output[0].cpu().numpy().flatten()
                    v     = output[1].cpu().numpy().flatten()
                    alpha = output[2].cpu().numpy().flatten() + 1.0
                    beta  = output[3].cpu().numpy().flatten()

                    aleatoric_uncertainty = beta / (alpha - 1)

                    if isinstance(aleatoric_uncertainty, np.ndarray):
                        aleatoric_uncertainty = torch.from_numpy(aleatoric_uncertainty).float()
                    if isinstance(v, np.ndarray):
                        v = torch.from_numpy(v).float()

                    total_uncertainty = torch.sigmoid(aleatoric_uncertainty * (-1 + self.Lambda * 1 / v))
                    total_uncertainty = total_uncertainty.detach().cpu().numpy()
                elif hasattr(output, 'shape') and len(output.shape) > 1:
                    total_uncertainty = np.ones(output.shape[0]) * 0.1
                else:
                    total_uncertainty = np.array([0.1])

                if total_uncertainty.size > 1:
                    total_uncertainty = total_uncertainty[0]

                target_np = target.cpu().numpy().flatten()
                if target_np.size > 1:
                    target_np = target_np[0]

                uncertainties_list.append(total_uncertainty)
                targets_list.append(target_np)
                structure_ids_list.append(structure_id)
                structures_list.append(input_data)
                structure_paths_list.append(structure_path)

        self.uncertainties   = np.array(uncertainties_list).flatten()
        self.targets         = np.array(targets_list).flatten()
        self.structure_ids   = structure_ids_list
        self.structures      = structures_list
        self.structure_paths = structure_paths_list

        cache_data = {
            'uncertainties':   self.uncertainties,
            'targets':         self.targets,
            'structure_ids':   self.structure_ids,
            'structures':      self.structures,
            'structure_paths': self.structure_paths,
        }
        self._save_to_cache(cache_data, self.uncertainties_cache)

        print(f"[{ts()}] Computed uncertainties for {len(self.uncertainties)} structures")
        print(f"[{ts()}] Uncertainties shape: {self.uncertainties.shape}")
        print(f"[{ts()}] Targets shape: {self.targets.shape}")

    # ------------------------------------------------------------------
    # Symmetry-aware SOAP fingerprint
    # ------------------------------------------------------------------

    def identify_equivalent_sites(self, structure, symprec=1e-5):
        """
        Identify equivalent sites in a crystal structure using spglib.

        Args:
            structure: ASE Atoms object
            symprec: Symmetry search precision

        Returns:
            Dictionary of equivalent site groups
        """
        lattice   = structure.get_cell()
        positions = structure.get_scaled_positions()
        numbers   = structure.get_atomic_numbers()
        cell      = (lattice, positions, numbers)

        symmetry_data = spglib.get_symmetry_dataset(cell, symprec=symprec)

        if symmetry_data is None:
            n_atoms = len(structure)
            return {
                i: {
                    "indices":      np.array([i]),
                    "multiplicity": 1,
                    "element":      structure.get_chemical_symbols()[i],
                }
                for i in range(n_atoms)
            }

        equivalent_atoms = symmetry_data["equivalent_atoms"]
        site_groups      = {}
        for i, site_id in enumerate(np.unique(equivalent_atoms)):
            atom_indices = np.where(equivalent_atoms == site_id)[0]
            site_groups[i] = {
                "indices":      atom_indices,
                "multiplicity": len(atom_indices),
                "element":      structure.get_chemical_symbols()[atom_indices[0]],
            }
        return site_groups

    def calculate_geometric_fingerprint(self, structure, species, r_cut=5.0, n_max=8, l_max=6, sigma=0.5):
        """
        Calculate the L2-normalized, symmetry-weighted SOAP fingerprint F_geom.

        Parameters
        ----------
        structure : ASE Atoms object
        species   : list of str
            Global species vocabulary (sorted, shared across all structures in the
            dataset).  Using a global vocabulary is mandatory: DScribe sizes the
            SOAP feature vector as
                (n_species*(n_species+1)/2) * n_max^2 * (l_max+1)
            so any per-structure local species list would produce a different
            output dimension for every structure, making cross-structure
            comparison undefined.
        r_cut, n_max, l_max, sigma : SOAP hyperparameters

        Returns
        -------
        F_geom : (soap_dim,) float64 array, L2-normalized
        """
        if not all(structure.pbc):
            structure.pbc = [True, True, True]

        site_groups = self.identify_equivalent_sites(structure, self.symprec)

        soap_params = self.soap_params.copy()
        soap_params.update({
            'species': species,   # global vocabulary -- fixes dimension consistency
            'r_cut':   r_cut,
            'n_max':   n_max,
            'l_max':   l_max,
            'sigma':   sigma,
        })
        soap = SOAP(**soap_params)

        soap_descriptors = soap.create(structure)   # (n_atoms, soap_dim)
        soap_dim         = soap_descriptors.shape[1]

        F_geom      = np.zeros(soap_dim, dtype=np.float64)
        total_atoms = 0

        for group_data in site_groups.values():
            indices      = group_data["indices"]
            multiplicity = group_data["multiplicity"]
            total_atoms += multiplicity

            site_soap = soap_descriptors[indices[0]]
            for idx in indices[1:]:
                if not np.allclose(site_soap, soap_descriptors[idx], rtol=1e-5, atol=1e-8):
                    site_soap = np.mean(soap_descriptors[indices], axis=0)
                    break

            F_geom += multiplicity * site_soap

        if total_atoms > 0:
            F_geom /= total_atoms

        norm = np.linalg.norm(F_geom)
        if norm > 0.0:
            F_geom /= norm

        return F_geom

    # ------------------------------------------------------------------
    # Structural similarity via SOAP
    # ------------------------------------------------------------------

    def compute_structural_similarity(self, force_recompute=False):
        """
        Compute structural similarities using symmetry-aware SOAP descriptors.

        Two-pass design
        ---------------
        Pass 1: read every CIF file once to collect the global species
                vocabulary.  All SOAP descriptors must be built with the
                same species list so that they have an identical, comparable
                output dimension.
        Pass 2: compute the L2-normalized, symmetry-weighted fingerprint for
                each structure using the global vocabulary established in
                Pass 1.

        This eliminates the root cause of the dimension inconsistency present
        in the original code, where each structure's SOAP was built from its
        own local species set, yielding a different vector length per
        structure.

        Args:
            force_recompute (bool): Force recomputation even if cache exists
        """
        if not force_recompute:
            cached_data = self._load_from_cache(self.soap_cache)
            if cached_data is not None:
                self.soap_features = cached_data
                print(f"[{ts()}] Loaded SOAP features from cache: shape {self.soap_features.shape}")
                return

        print(f"[{ts()}] Computing structural similarities using SOAP descriptors...")

        # ------------------------------------------------------------------
        # Locate CIF files and build id -> path mapping
        # ------------------------------------------------------------------
        data_dir = Path(self.data_path)
        cif_files = list(set(
            list((data_dir / self.task).glob("*.cif")) +
            list((data_dir / self.task).rglob("*.cif"))
        ))
        print(f"[{ts()}] Found {len(cif_files)} CIF files under {data_dir / self.task}")

        id_to_cif = {}
        for cif_file in cif_files:
            file_id = cif_file.stem
            id_to_cif[file_id] = cif_file
            if file_id.startswith("JVASP-"):
                id_to_cif[file_id[6:]] = cif_file
            else:
                id_to_cif[f"JVASP-{file_id}"] = cif_file

        # ------------------------------------------------------------------
        # Resolve the CIF path for every structure_id upfront
        # ------------------------------------------------------------------
        print(f"[{ts()}] Resolving CIF paths for {len(self.structure_ids)} structures...")
        resolved_cif_paths = []
        for i, structure_id in enumerate(self.structure_ids):
            cif_path = None
            if i < len(self.structure_paths) and self.structure_paths[i]:
                cif_path = Path(self.structure_paths[i])
            if not cif_path or not cif_path.exists():
                for potential_id in [
                    structure_id,
                    f"JVASP-{structure_id}",
                    f"{structure_id}.cif",
                    str(structure_id).replace("JVASP-", ""),
                ]:
                    if potential_id in id_to_cif:
                        cif_path = id_to_cif[potential_id]
                        break
            resolved_cif_paths.append(cif_path)

        # ------------------------------------------------------------------
        # Pass 1: collect global species vocabulary
        # All SOAP descriptors share this vocabulary, guaranteeing that every
        # fingerprint vector has the same dimension and that corresponding
        # dimensions encode the same pair-interaction channel.
        # ------------------------------------------------------------------
        print(f"[{ts()}] Pass 1: collecting global species vocabulary from all CIF files...")
        all_species = set()
        for cif_path in resolved_cif_paths:
            structure = read(str(cif_path))
            all_species.update(structure.get_chemical_symbols())
        global_species = sorted(all_species)
        print(f"[{ts()}] Global vocabulary: {len(global_species)} species: {global_species}")

        # Expected SOAP output dimension (informational):
        # (n_species*(n_species+1)/2) * n_max^2 * (l_max+1)
        n_sp   = len(global_species)
        n_max  = self.soap_params['n_max']
        l_max  = self.soap_params['l_max']
        expected_dim = (n_sp * (n_sp + 1) // 2) * n_max * n_max * (l_max + 1)
        print(f"[{ts()}] Expected SOAP fingerprint dimension: {expected_dim}")

        # ------------------------------------------------------------------
        # Pass 2: compute fingerprints with the global species vocabulary
        # ------------------------------------------------------------------
        print(f"[{ts()}] Pass 2: computing fingerprints for {len(resolved_cif_paths)} structures...")
        soap_features_list = []
        for i, cif_path in enumerate(tqdm(resolved_cif_paths, desc="Computing SOAP")):
            structure = read(str(cif_path))
            F_geom = self.calculate_geometric_fingerprint(
                structure,
                global_species,
                r_cut=self.soap_params['r_cut'],
                n_max=self.soap_params['n_max'],
                l_max=self.soap_params['l_max'],
                sigma=self.soap_params['sigma'],
            )
            soap_features_list.append(F_geom)
            if (i + 1) % 500 == 0:
                print(f"[{ts()}] Processed {i + 1}/{len(resolved_cif_paths)} structures")

        self.soap_features = np.array(soap_features_list, dtype=np.float64)
        self._save_to_cache(self.soap_features, self.soap_cache)

        print(f"[{ts()}] SOAP features computed: shape {self.soap_features.shape}")

    # ------------------------------------------------------------------
    # Weighting factors and influence scores
    # ------------------------------------------------------------------

    def compute_weighting_factors(self):
        """
        Compute weighting factors based on structural similarity.
        w(x_i) = exp(-gamma * min_{j != i} d(x_i, x_j))
        """
        print(f"[{ts()}] Computing weighting factors...")

        n_structures = len(self.soap_features)
        nbrs = NearestNeighbors(n_neighbors=min(5, n_structures), algorithm='auto').fit(self.soap_features)
        distances, indices = nbrs.kneighbors(self.soap_features)

        if distances.shape[1] > 1:
            min_distances = distances[:, 1]
        else:
            min_distances = np.ones(n_structures) * 0.1

        weighting_factors = np.exp(-self.gamma * min_distances)
        return weighting_factors

    def compute_uncertainty_influence(self, force_recompute=False):
        """
        Compute uncertainty influence scores: I_unc(x_i) = unc(x_i) / w(x_i)

        Args:
            force_recompute (bool): Force recomputation even if cache exists
        """
        if not force_recompute:
            cached_data = self._load_from_cache(self.influence_cache)
            if cached_data is not None:
                self.influence_scores = cached_data
                print(f"[{ts()}] Loaded influence scores from cache: shape {self.influence_scores.shape}")
                return

        print(f"[{ts()}] Computing uncertainty influence scores...")
        print(f"[{ts()}] Uncertainties shape: {self.uncertainties.shape}")
        print(f"[{ts()}] SOAP features shape: {self.soap_features.shape}")

        if len(self.uncertainties.shape) > 1:
            self.uncertainties = self.uncertainties.flatten()

        weighting_factors = self.compute_weighting_factors()
        print(f"[{ts()}] Weighting factors shape: {weighting_factors.shape}")

        if len(weighting_factors.shape) > 1:
            weighting_factors = weighting_factors.flatten()

        self.influence_scores = self.uncertainties / weighting_factors

        if len(self.influence_scores.shape) > 1:
            self.influence_scores = self.influence_scores.flatten()

        self._save_to_cache(self.influence_scores, self.influence_cache)

        print(f"[{ts()}] Influence scores shape: {self.influence_scores.shape}")
        print(f"[{ts()}] Influence scores min={self.influence_scores.min():.6f}  "
              f"max={self.influence_scores.max():.6f}  "
              f"mean={self.influence_scores.mean():.6f}  "
              f"std={self.influence_scores.std():.6f}")

    # ------------------------------------------------------------------
    # FUSION: structural relationship pre-computation
    # ------------------------------------------------------------------

    def _precompute_structural_relationships(self):
        """Pre-compute structural relationships for FUSION dynamic optimization."""
        print(f"[{ts()}] Pre-computing structural relationships for dynamic optimization...")

        self.distance_matrix = pairwise_distances(self.soap_features, metric='euclidean')

        k   = min(10, len(self.soap_features) - 1)
        knn = NearestNeighbors(n_neighbors=k, metric='precomputed')
        knn.fit(self.distance_matrix)
        self.neighbor_distances, self.neighbor_indices = knn.kneighbors(self.distance_matrix)

        self.diversity_scores = np.zeros(len(self.soap_features))
        for i in range(len(self.soap_features)):
            local_density = np.mean(self.neighbor_distances[i, 1:])
            self.diversity_scores[i] = 1.0 / (local_density + 1e-8)

        quantiles = np.quantile(self.influence_scores, [0.2, 0.4, 0.6, 0.8])
        self.influence_clusters = {
            'very_low':  np.where(self.influence_scores <= quantiles[0])[0],
            'low':       np.where((self.influence_scores > quantiles[0]) &
                                  (self.influence_scores <= quantiles[1]))[0],
            'medium':    np.where((self.influence_scores > quantiles[1]) &
                                  (self.influence_scores <= quantiles[2]))[0],
            'high':      np.where((self.influence_scores > quantiles[2]) &
                                  (self.influence_scores <= quantiles[3]))[0],
            'very_high': np.where(self.influence_scores > quantiles[3])[0],
        }

        print(f"[{ts()}] Distance matrix shape: {self.distance_matrix.shape}")
        print(f"[{ts()}] Diversity scores range: "
              f"[{self.diversity_scores.min():.3f}, {self.diversity_scores.max():.3f}]")
        print(f"[{ts()}] Influence cluster sizes: "
              f"{[len(v) for v in self.influence_clusters.values()]}")

    # ------------------------------------------------------------------
    # FUSION: dynamic value function
    # ------------------------------------------------------------------

    def fusion_dynamic_value_function(self, candidate_idx, current_selection):
        """
        FUSION Dynamic Value Function: context-aware sample value evaluation.

        Args:
            candidate_idx: Index of candidate sample
            current_selection: Current selection mask (boolean array)

        Returns:
            Dynamic value score (quality, diversity, redundancy, coverage)
        """
        selection_hash = hash(current_selection.tobytes())
        cache_key      = (candidate_idx, selection_hash)

        if cache_key in self.value_cache:
            return self.value_cache[cache_key]

        base_value = -self.influence_scores[candidate_idx]

        if np.any(current_selection):
            selected_indices = np.where(current_selection)[0]
            min_distance     = np.min(self.distance_matrix[candidate_idx, selected_indices])
            avg_distance     = np.mean(self.distance_matrix[candidate_idx, :])
            diversity_bonus  = (min_distance / (avg_distance + 1e-8)) * self.diversity_scores[candidate_idx]
        else:
            diversity_bonus = self.diversity_scores[candidate_idx]

        if np.any(current_selection):
            neighbor_mask       = np.isin(self.neighbor_indices[candidate_idx],
                                          np.where(current_selection)[0])
            redundancy_penalty  = np.sum(neighbor_mask) / len(self.neighbor_indices[candidate_idx])
        else:
            redundancy_penalty = 0.0

        candidate_cluster = None
        for cluster_name, cluster_indices in self.influence_clusters.items():
            if candidate_idx in cluster_indices:
                candidate_cluster = cluster_name
                break

        if candidate_cluster and np.any(current_selection):
            cluster_indices        = self.influence_clusters[candidate_cluster]
            selected_from_cluster  = np.sum(current_selection[cluster_indices])
            representation_ratio   = selected_from_cluster / len(cluster_indices)
            coverage_bonus         = 1.0 - representation_ratio
        else:
            coverage_bonus = 0.5

        selection_ratio = np.sum(current_selection) / len(current_selection)
        if selection_ratio < 0.2:
            w_base, w_diversity, w_redundancy, w_coverage = 0.6, 0.1, 0.1, 0.2
        elif selection_ratio < 0.5:
            w_base, w_diversity, w_redundancy, w_coverage = 0.4, 0.2, 0.2, 0.2
        else:
            w_base, w_diversity, w_redundancy, w_coverage = 0.2, 0.3, 0.3, 0.2

        dynamic_value = (w_base      * base_value
                       + w_diversity * diversity_bonus
                       + w_redundancy * (-redundancy_penalty)
                       + w_coverage  * coverage_bonus)

        self.value_cache[cache_key] = dynamic_value
        return dynamic_value

    def fusion_incremental_constraint_check(self, candidate_idx, current_selection):
        """
        FUSION Incremental Constraint Check: fast feasibility verification.

        Returns:
            (feasible, new_avg_influence)
        """
        selected_indices = np.where(current_selection)[0]

        if len(selected_indices) == 0:
            new_avg_influence = self.influence_scores[candidate_idx]
        else:
            current_sum       = np.sum(self.influence_scores[selected_indices])
            new_avg_influence = (current_sum + self.influence_scores[candidate_idx]) / (len(selected_indices) + 1)

        return new_avg_influence <= self.epsilon, new_avg_influence

    # ------------------------------------------------------------------
    # FUSION optimization algorithms
    # ------------------------------------------------------------------

    def fusion_fast_dynamic_greedy(self, max_iterations=None):
        """
        FUSION Fast Dynamic Greedy: efficient greedy with dynamic value functions.

        Returns:
            Pruning ratio
        """
        print(f"[{ts()}] Running FUSION Fast Dynamic Greedy optimization...")

        if self.distance_matrix is None:
            self._precompute_structural_relationships()

        current_selection = np.zeros(len(self.influence_scores), dtype=bool)

        if max_iterations is None:
            max_iterations = min(1000, len(self.influence_scores) // 2)

        for iteration in tqdm(range(max_iterations), desc="FUSION Dynamic Greedy"):
            candidates = np.where(~current_selection)[0]
            if len(candidates) == 0:
                break

            best_candidate = None
            best_value     = -np.inf

            for candidate in candidates:
                feasible, _ = self.fusion_incremental_constraint_check(candidate, current_selection)
                if not feasible:
                    continue
                value = self.fusion_dynamic_value_function(candidate, current_selection)
                if value > best_value:
                    best_value     = value
                    best_candidate = candidate

            if best_candidate is None:
                break

            current_selection[best_candidate] = True

            if len(self.value_cache) > 10000:
                self.value_cache.clear()

        self.pruning_mask = current_selection
        ratio = np.sum(current_selection) / len(current_selection)
        print(f"[{ts()}] FUSION Fast Dynamic Greedy completed. Ratio: {ratio:.4f}")
        return ratio

    def fusion_adaptive_beam_search(self, beam_width=3, max_iterations=500):
        """
        FUSION Adaptive Beam Search: maintain multiple partial solutions.

        Returns:
            Pruning ratio
        """
        print(f"[{ts()}] Running FUSION Adaptive Beam Search (beam_width={beam_width})...")

        if self.distance_matrix is None:
            self._precompute_structural_relationships()

        beam        = [np.zeros(len(self.influence_scores), dtype=bool)]
        beam_scores = [0.0]

        best_solution = None
        best_score    = -np.inf
        best_ratio    = 0.0

        for iteration in tqdm(range(max_iterations), desc="FUSION Beam Search"):
            new_beam   = []
            new_scores = []

            for sol_idx, current_solution in enumerate(beam):
                current_score = beam_scores[sol_idx]
                candidates    = np.where(~current_solution)[0]

                candidate_values = []
                for candidate in candidates:
                    feasible, _ = self.fusion_incremental_constraint_check(candidate, current_solution)
                    if feasible:
                        value = self.fusion_dynamic_value_function(candidate, current_solution)
                        candidate_values.append((candidate, value))

                if candidate_values:
                    candidate_values.sort(key=lambda x: x[1], reverse=True)

                    n_selected = np.sum(current_solution)
                    n_expand   = min(beam_width, len(candidate_values)) if n_selected < 10 \
                                 else min(beam_width // 2, len(candidate_values))

                    for i in range(n_expand):
                        candidate, value = candidate_values[i]
                        new_solution     = current_solution.copy()
                        new_solution[candidate] = True
                        new_score = current_score + value
                        new_beam.append(new_solution)
                        new_scores.append(new_score)
                        if new_score > best_score:
                            best_solution = new_solution.copy()
                            best_score    = new_score
                            best_ratio    = np.sum(new_solution) / len(new_solution)

            if new_beam:
                combined = sorted(zip(new_beam, new_scores), key=lambda x: x[1], reverse=True)
                beam        = [sol   for sol, score in combined[:beam_width]]
                beam_scores = [score for sol, score in combined[:beam_width]]
            else:
                break

            if iteration % 50 == 0:
                self.value_cache.clear()

        if best_solution is not None:
            self.pruning_mask = best_solution
            print(f"[{ts()}] FUSION Beam Search completed. Best ratio: {best_ratio:.4f}")
            return best_ratio
        else:
            print(f"[{ts()}] No feasible solution found.")
            self.pruning_mask = np.zeros(len(self.influence_scores), dtype=bool)
            return 0.0

    def fusion_enhanced_simulated_annealing(self, max_iterations=5000):
        """
        FUSION Enhanced Simulated Annealing: global optimization with dynamic value functions.

        Returns:
            Pruning ratio
        """
        print(f"[{ts()}] Running FUSION Enhanced Simulated Annealing...")

        if self.distance_matrix is None:
            self._precompute_structural_relationships()

        temp_epsilon  = self.epsilon
        self.epsilon  = self.epsilon * 1.1
        self.fusion_fast_dynamic_greedy(max_iterations=100)
        current_mask  = self.pruning_mask.copy()
        self.epsilon  = temp_epsilon

        current_count    = np.sum(current_mask)
        current_influence = np.mean(self.influence_scores[current_mask]) if current_count > 0 else 0.0

        best_mask  = current_mask.copy()
        best_count = current_count

        temp_initial = 10.0
        temp_final   = 0.01

        print(f"[{ts()}] Initial solution: {current_count} structures, "
              f"mean influence: {current_influence:.6f}")

        for iteration in tqdm(range(max_iterations), desc="FUSION Enhanced SA"):
            progress = iteration / max_iterations
            temp     = temp_final * (temp_initial / temp_final) ** (1 - progress)

            neighbor_mask = current_mask.copy()

            if np.random.rand() < 0.7:
                candidates = np.where(~current_mask)[0]
                if len(candidates) > 0:
                    values = []
                    for candidate in candidates:
                        feasible, _ = self.fusion_incremental_constraint_check(candidate, current_mask)
                        if feasible:
                            value = self.fusion_dynamic_value_function(candidate, current_mask)
                            values.append((candidate, value))
                    if values:
                        values.sort(key=lambda x: x[1], reverse=True)
                        top_candidates = values[:min(5, len(values))]
                        weights        = np.exp([v[1] / (temp + 1e-8) for v in top_candidates])
                        weights       /= np.sum(weights)
                        chosen_idx     = np.random.choice(len(top_candidates), p=weights)
                        neighbor_mask[top_candidates[chosen_idx][0]] = True
            else:
                selected_indices = np.where(current_mask)[0]
                if len(selected_indices) > 0:
                    selected_influences = self.influence_scores[selected_indices]
                    removal_probs       = selected_influences / np.sum(selected_influences)
                    chosen_idx          = np.random.choice(len(selected_indices), p=removal_probs)
                    neighbor_mask[selected_indices[chosen_idx]] = False

            neighbor_count = np.sum(neighbor_mask)
            if neighbor_count > 0:
                neighbor_influence = np.mean(self.influence_scores[neighbor_mask])
                feasible           = neighbor_influence <= self.epsilon
            else:
                neighbor_influence = 0.0
                feasible           = True

            if feasible:
                delta_e = neighbor_count - current_count
                if delta_e > 0 or (temp > temp_final and np.random.rand() < np.exp(delta_e / temp)):
                    current_mask      = neighbor_mask
                    current_count     = neighbor_count
                    current_influence = neighbor_influence
                    if current_count > best_count:
                        best_mask  = current_mask.copy()
                        best_count = current_count

            if iteration % 1000 == 0:
                self.value_cache.clear()

        self.pruning_mask = best_mask
        ratio = best_count / len(self.influence_scores)
        print(f"[{ts()}] FUSION Enhanced SA completed. Best ratio: {ratio:.4f}")
        return ratio

    # ------------------------------------------------------------------
    # Legacy optimization methods (for compatibility)
    # ------------------------------------------------------------------

    def optimize_pruning_mask(self, indices=None):
        """
        Legacy greedy optimization (for compatibility).

        Args:
            indices (list): Optional list of indices to consider for pruning.
        """
        print(f"[{ts()}] Running legacy greedy optimization...")

        if indices is not None:
            mask             = np.zeros(len(self.influence_scores), dtype=bool)
            mask[indices]    = True
            influence_scores = self.influence_scores[mask]
        else:
            influence_scores = self.influence_scores

        print(f"[{ts()}] Influence scores shape: {influence_scores.shape}")
        print(f"[{ts()}] Number of structures: {len(influence_scores)}")

        if len(influence_scores.shape) > 1:
            influence_scores = influence_scores.flatten()

        n_structures  = len(influence_scores)
        sorted_indices = np.argsort(influence_scores)
        sorted_scores  = influence_scores[sorted_indices]

        max_index      = 0
        cumulative_sum = 0.0

        print(f"[{ts()}] Finding optimal pruning (epsilon={self.epsilon})...")

        for i in range(n_structures):
            cumulative_sum  += float(sorted_scores[i])
            cumulative_mean  = cumulative_sum / (i + 1)
            if cumulative_mean <= self.epsilon:
                max_index = i + 1
            else:
                break
            if i % 1000 == 0:
                print(f"[{ts()}] Processed {i + 1}/{n_structures}, current mean: {cumulative_mean:.6f}")

        if indices is None:
            self.pruning_mask = np.zeros(n_structures, dtype=bool)
            self.pruning_mask[sorted_indices[:max_index]] = True
        else:
            self.pruning_mask = np.zeros(len(self.influence_scores), dtype=bool)
            for i in range(max_index):
                self.pruning_mask[indices[sorted_indices[i]]] = True

        pruning_ratio = max_index / n_structures
        print(f"[{ts()}] Legacy greedy completed: {max_index}/{n_structures} structures selected for removal")
        print(f"[{ts()}] Pruning ratio: {pruning_ratio:.4f}")
        return pruning_ratio

    def simulated_annealing_optimization(self, max_iterations=10000, indices=None):
        """Legacy simulated annealing optimization (for compatibility)."""
        print(f"[{ts()}] Using legacy simulated annealing (delegates to FUSION enhanced SA)...")
        return self.fusion_enhanced_simulated_annealing(max_iterations // 2)

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def run(self, optimization_method='fusion_dynamic_greedy', force_recompute=False, indices=None, **kwargs):
        """
        Execute the complete FUSION pipeline.

        Args:
            optimization_method (str): One of
                'fusion_dynamic_greedy', 'fusion_beam_search', 'fusion_enhanced_sa',
                'greedy', 'sa'
            force_recompute (bool): Force recomputation of all steps
            indices (list): Optional list of indices to consider for pruning
            **kwargs: Additional arguments for the chosen optimization method

        Returns:
            dict: Results containing pruning mask, pruning ratio, and other metrics
        """
        self.compute_uncertainties(force_recompute=force_recompute)
        self.compute_structural_similarity(force_recompute=force_recompute)
        self.compute_uncertainty_influence(force_recompute=force_recompute)

        print(f"[{ts()}] Using optimization method: {optimization_method}")

        if optimization_method == 'fusion_dynamic_greedy':
            pruning_ratio = self.fusion_fast_dynamic_greedy(**kwargs)
        elif optimization_method == 'fusion_beam_search':
            pruning_ratio = self.fusion_adaptive_beam_search(**kwargs)
        elif optimization_method == 'fusion_enhanced_sa':
            pruning_ratio = self.fusion_enhanced_simulated_annealing(**kwargs)
        elif optimization_method == 'greedy':
            pruning_ratio = self.optimize_pruning_mask(indices=indices)
        elif optimization_method == 'sa':
            pruning_ratio = self.simulated_annealing_optimization(indices=indices, **kwargs)
        else:
            print(f"[{ts()}] Unknown optimization method '{optimization_method}', "
                  f"falling back to fusion_dynamic_greedy")
            pruning_ratio = self.fusion_fast_dynamic_greedy(**kwargs)

        return {
            'structure_ids':    self.structure_ids,
            'pruning_mask':     self.pruning_mask,
            'uncertainties':    self.uncertainties,
            'influence_scores': self.influence_scores,
            'pruning_ratio':    pruning_ratio,
            'targets':          self.targets,
            'soap_features':    self.soap_features,
            'structure_paths':  self.structure_paths,
            'optimization_method': optimization_method,
        }

    # ------------------------------------------------------------------
    # JSON split generation
    # ------------------------------------------------------------------

    def generate_json_splits(self, epsilon_values, output_dir="./fusion_output",
                             test_size=0.15, val_size=0.15, random_state=42,
                             optimization_method='fusion_dynamic_greedy'):
        """
        Generate train, val, and test JSON documents with data splits using FUSION.
        Output files are written to {output_dir}/{YYYYMMDD-HHMMSS}/.
        """
        run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir   = os.path.join(output_dir, run_stamp)
        os.makedirs(run_dir, exist_ok=True)
        print(f"[{ts()}] Output directory: {run_dir}")

        n_structures    = len(self.structure_ids)
        all_indices     = np.arange(n_structures)

        train_val_indices, test_indices = train_test_split(
            all_indices, test_size=test_size, random_state=random_state
        )
        val_size_adjusted = val_size / (1 - test_size)
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_size_adjusted, random_state=random_state
        )

        print(f"[{ts()}] Dataset split:")
        print(f"[{ts()}]   Total structures: {n_structures}")
        print(f"[{ts()}]   Training set    : {len(train_indices)} ({len(train_indices)/n_structures:.2%})")
        print(f"[{ts()}]   Validation set  : {len(val_indices)}   ({len(val_indices)/n_structures:.2%})")
        print(f"[{ts()}]   Test set        : {len(test_indices)}  ({len(test_indices)/n_structures:.2%})")

        json_docs = {"train": {}, "val": {}, "test": {}}
        actual_pruning_ratios = {}

        json_docs["train"]["0"] = train_indices.tolist()
        json_docs["val"]["0"]   = val_indices.tolist()
        json_docs["test"]["0"]  = test_indices.tolist()
        actual_pruning_ratios[0] = 0.0

        for model_num, epsilon in epsilon_values.items():
            if model_num == 0:
                continue

            model_str  = str(model_num)
            self.epsilon = epsilon
            print(f"[{ts()}] Processing model {model_num} with epsilon={epsilon} "
                  f"using {optimization_method}")

            combined_indices = np.concatenate([train_indices, val_indices])
            results          = self.run(optimization_method=optimization_method,
                                        force_recompute=False, indices=combined_indices)

            actual_pruning_ratios[model_num] = results['pruning_ratio']

            retained_indices = np.where(~results['pruning_mask'])[0]
            retained_indices = retained_indices[np.isin(retained_indices, combined_indices)]
            retained_train   = np.intersect1d(retained_indices, train_indices)
            retained_val     = np.intersect1d(retained_indices, val_indices)

            json_docs["train"][model_str] = retained_train.tolist()
            json_docs["val"][model_str]   = retained_val.tolist()
            json_docs["test"][model_str]  = test_indices.tolist()

            print(f"[{ts()}] Model {model_num}: "
                  f"train={len(retained_train)}  val={len(retained_val)}  "
                  f"test={len(test_indices)}  "
                  f"removed={len(combined_indices)-len(retained_train)-len(retained_val)}")

        for split, data in json_docs.items():
            out_path = os.path.join(run_dir, f"{split}.json")
            with open(out_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[{ts()}] Saved {out_path}")

        ratios_path = os.path.join(run_dir, "pruning_ratios.json")
        with open(ratios_path, 'w') as f:
            json.dump(actual_pruning_ratios, f, indent=2)
        print(f"[{ts()}] Saved {ratios_path}")

        mapping = {
            "metadata": {
                "generated_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "optimization_method": optimization_method,
                "total_models":        len(actual_pruning_ratios),
                "task":                self.task,
                "lambda":              self.Lambda,
            },
            "model_data": {
                str(m): {"epsilon": epsilon_values.get(m, "N/A"),
                          "pruning_ratio": actual_pruning_ratios[m]}
                for m in sorted(actual_pruning_ratios)
            },
        }
        mapping_path = os.path.join(run_dir, "epsilon_ratio_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"[{ts()}] Saved {mapping_path}")

        print(f"[{ts()}] Summary:")
        for m in sorted(actual_pruning_ratios):
            print(f"[{ts()}]   model {m}: epsilon={epsilon_values.get(m,'N/A')}  "
                  f"pruning_ratio={actual_pruning_ratios[m]:.4f}")

        return json_docs

    def generate_json_splits_from_predefined(self, epsilon_values, predefined_splits,
                                             output_dir="./fusion_output",
                                             optimization_method='fusion_dynamic_greedy'):
        """
        Generate pruned JSON documents from predefined train/val/test splits.
        Output files are written to {output_dir}/{YYYYMMDD-HHMMSS}/.
        """
        run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir   = os.path.join(output_dir, run_stamp)
        os.makedirs(run_dir, exist_ok=True)
        print(f"[{ts()}] Output directory: {run_dir}")

        original_train_indices = np.array(predefined_splits['train'])
        original_val_indices   = np.array(predefined_splits['val'])
        test_indices           = np.array(predefined_splits['test'])

        n_structures = len(self.structure_ids)
        print(f"[{ts()}] Predefined splits: total={n_structures}  "
              f"train={len(original_train_indices)}  "
              f"val={len(original_val_indices)}  "
              f"test={len(test_indices)}  "
              f"method={optimization_method}")

        json_docs = {"train": {}, "val": {}, "test": {}}
        actual_pruning_ratios = {}

        json_docs["train"]["0"] = original_train_indices.tolist()
        json_docs["val"]["0"]   = original_val_indices.tolist()
        json_docs["test"]["0"]  = test_indices.tolist()
        actual_pruning_ratios[0] = 0.0

        for model_num, epsilon in epsilon_values.items():
            if model_num == 0:
                continue

            model_str    = str(model_num)
            self.epsilon = epsilon
            print(f"[{ts()}] Processing model {model_num} with epsilon={epsilon} "
                  f"using {optimization_method}")

            combined_indices = np.concatenate([original_train_indices, original_val_indices])
            results          = self.run(optimization_method=optimization_method,
                                        force_recompute=False, indices=combined_indices)

            actual_pruning_ratios[model_num] = results['pruning_ratio']

            retained_indices = np.where(~results['pruning_mask'])[0]
            retained_indices = retained_indices[np.isin(retained_indices, combined_indices)]
            retained_train   = np.intersect1d(retained_indices, original_train_indices)
            retained_val     = np.intersect1d(retained_indices, original_val_indices)

            json_docs["train"][model_str] = retained_train.tolist()
            json_docs["val"][model_str]   = retained_val.tolist()
            json_docs["test"][model_str]  = test_indices.tolist()

            print(f"[{ts()}] Model {model_num}: "
                  f"train={len(retained_train)}  val={len(retained_val)}  "
                  f"test={len(test_indices)}  "
                  f"removed={len(combined_indices)-len(retained_train)-len(retained_val)}")

        for split, data in json_docs.items():
            out_path = os.path.join(run_dir, f"{split}.json")
            with open(out_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[{ts()}] Saved {out_path}")

        ratios_path = os.path.join(run_dir, "pruning_ratios.json")
        with open(ratios_path, 'w') as f:
            json.dump(actual_pruning_ratios, f, indent=2)
        print(f"[{ts()}] Saved {ratios_path}")

        return json_docs


# Alias for backward compatibility
MAPS = FUSION


# ------------------------------------------------------------------
# Standalone helpers
# ------------------------------------------------------------------

def load_splits_from_json_files(train_file, val_file, test_file, model_key="0"):
    """Load train/val/test splits from existing JSON files."""
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    with open(val_file, 'r') as f:
        val_data = json.load(f)
    with open(test_file, 'r') as f:
        test_data = json.load(f)

    predefined_splits = {
        'train': train_data.get(model_key, []),
        'val':   val_data.get(model_key,   []),
        'test':  test_data.get(model_key,  []),
    }

    print(f"[{ts()}] Loaded predefined splits: "
          f"train={len(predefined_splits['train'])}  "
          f"val={len(predefined_splits['val'])}  "
          f"test={len(predefined_splits['test'])}")

    return predefined_splits


# ------------------------------------------------------------------
# Entry points
# ------------------------------------------------------------------

def main_with_predefined_splits():
    """Main function for running FUSION with predefined splits."""
    for model_key in ["0", "1", "2", "3", "4"]:
        config_path      = "config.yml"
        checkpoint_path  = "/home/wangxiean/PycharmProjects/MAPS/surrogate/jdft2d/ALIGNN/model_best.pth.tar"
        data_path        = "/data1/tanliqin/uq-ood-mat/data/"
        cache_dir        = "./fusion_cache_lambda0"
        task             = "jdft2d"
        json_output_dir  = f"./FUSION_output/{task}_lambda0_ood/model_key_{model_key}"

        fusion = FUSION(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            data_path=data_path,
            task=task,
            epsilon=0.01,
            gamma=0.1,
            cache_dir=cache_dir,
            Lambda=0,
        )

        fusion.soap_params.update({
            'r_cut': 5.0, 'n_max': 8, 'l_max': 6,
            'sigma': 0.5, 'periodic': True, 'average': 'off', 'sparse': False,
        })
        fusion.symprec = 1e-5

        fusion.compute_uncertainties(force_recompute=False)
        fusion.compute_structural_similarity(force_recompute=False)
        fusion.compute_uncertainty_influence(force_recompute=False)

        searcher         = EpsilonSearcher(fusion)
        target_ratios    = [0, 0.2, 0.4, 0.6, 0.8]
        optimal_epsilons = searcher.optimize_epsilon_for_target_ratios(target_ratios)
        epsilon_values   = {i: eps for i, eps in enumerate(optimal_epsilons.values())}

        predefined_splits = load_splits_from_json_files(
            train_file="/data1/tanliqin/uq-ood-mat/folds/jdft2d_folds/train/SOAP_jdft2d_LOCO_target_clusters40_train.json",
            val_file="/data1/tanliqin/uq-ood-mat/folds/jdft2d_folds/val/SOAP_jdft2d_LOCO_target_clusters40_val.json",
            test_file="/data1/tanliqin/uq-ood-mat/folds/jdft2d_folds/test/SOAP_jdft2d_LOCO_target_clusters40_test.json",
            model_key=model_key,
        )

        print(f"[{ts()}] Generating JSON documents with FUSION optimization...")
        json_docs = fusion.generate_json_splits_from_predefined(
            epsilon_values=epsilon_values,
            predefined_splits=predefined_splits,
            output_dir=json_output_dir,
            optimization_method='fusion_dynamic_greedy',
        )

        print(f"[{ts()}] Summary of FUSION-generated JSON documents:")
        for split, model_data in json_docs.items():
            print(f"[{ts()}] {split.upper()} set:")
            for model, indices in model_data.items():
                print(f"[{ts()}]   Model {model}: {len(indices)} structures")


def main():
    """
    Main function for running FUSION from scratch.
    """
    config_path = "config.yml"
    checkpoint_paths = [
        "surrogate/perovskites/ALIGNN/example.pth.tar",
    ]
    data_path = "/data1/tanliqin/uq-ood-mat/data/"
    tasks = ["perovskites",]

    # Test different optimization methods
    optimization_methods = [
        "greedy",
        # 'fusion_dynamic_greedy',  # Fastest, good quality
        # 'fusion_beam_search',  # Best balance of speed/quality
        # 'fusion_enhanced_sa'  # Best quality, slower
    ]

    # for lambda_value in range(-5, 6):
    for lambda_value in [0]:
        cache_dir = f"./fusion_cache/lambda{lambda_value}"
        for task_idx in range(len(tasks)):
            start_time = time.time()
            for seed in range(1):
                checkpoint_path = checkpoint_paths[task_idx]
                task_name = tasks[task_idx]

                for opt_method in optimization_methods:
                    json_output_dir = f"./FUSION_output/{task_name}/lambda_{lambda_value}_{opt_method}/seed{seed}"

                    # Initialize FUSION
                    fusion = FUSION(
                        config_path=config_path,
                        checkpoint_path=checkpoint_path,
                        data_path=data_path,
                        task=task_name,
                        epsilon=0,
                        gamma=0.1,
                        cache_dir=cache_dir,
                        Lambda=lambda_value
                    )

                    # Update SOAP parameters
                    fusion.soap_params.update({
                        'r_cut': 5.0,
                        'n_max': 8,
                        'l_max': 6,
                        'sigma': 0.5,
                        'periodic': True,
                        'average': 'off',
                        'sparse': False
                    })
                    fusion.symprec = 1e-5

                    # Compute features
                    fusion.compute_uncertainties(force_recompute=False)
                    fusion.compute_structural_similarity(force_recompute=False)
                    fusion.compute_uncertainty_influence(force_recompute=False)

                    # Get epsilon values
                    searcher = EpsilonSearcher(fusion)
                    target_ratios = [0, 0.20, 0.40, 0.60, 0.80]
                    optimal_epsilons = searcher.optimize_epsilon_for_target_ratios(target_ratios)
                    epsilon_values = {i: eps for i, eps in enumerate(optimal_epsilons.values())}

                    # Generate JSON documents
                    print(f"\nGenerating JSON documents using {opt_method}...")
                    json_docs = fusion.generate_json_splits(
                        epsilon_values=epsilon_values,
                        output_dir=json_output_dir,
                        test_size=0.15,
                        val_size=0.15,
                        random_state=seed,
                        optimization_method=opt_method
                    )

                    print(f"\nSummary for {opt_method}:")
                    for split, model_data in json_docs.items():
                        print(f"{split.upper()} set:")
                        for model, indices in model_data.items():
                            print(f"  Model {model}: {len(indices)} structures")

            duration = time.time() - start_time
            print(f"\nTotal time for task '{task_name}' with lambda={lambda_value}: {duration:.2f} seconds")


if __name__ == "__main__":
    # main_with_predefined_splits()
    main()
