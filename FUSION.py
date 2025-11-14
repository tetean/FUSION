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
from EpsilonSearcher import EpsilonSearcher

warnings.filterwarnings('ignore')


class FUSION:
    def __init__(self, config_path, checkpoint_path, data_path, task, epsilon=0.1, gamma=0.1, Lambda=1,
                 cache_dir="./fusion_cache"):
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
        self.Lambda = Lambda  # Hyperparameter for uncertainty

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Cache file paths
        self.uncertainties_cache = os.path.join(cache_dir, f"{task}_uncertainties.pkl")
        self.soap_cache = os.path.join(cache_dir, f"{task}_soap_features.pkl")
        self.influence_cache = os.path.join(cache_dir, f"{task}_influence_scores.pkl")
        self.structures_cache = os.path.join(cache_dir, f"{task}_structures.pkl")
        self.structure_paths_cache = os.path.join(cache_dir, f"{task}_structure_paths.pkl")

        # Load model configuration
        with open(config_path, "r") as ymlfile:
            self.config = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.config["Models"] = self.config["Models"].get("ALIGNN")

        # Initialize model with evidential regression
        self.model = ALIGNN(evidential=self.config["Models"]["evidential"],
                            **(self.config["Models"]["model_setting"]))

        # Load model checkpoint
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(device)
        self.model.eval()  # Set model to evaluation mode
        self.device = device

        # Load dataset
        self.data_loader = load_alignn_data(data_path, task, self.config["Models"])

        # Initialize tracking containers
        self.structures = []
        self.uncertainties = []
        self.targets = []
        self.structure_ids = []
        self.structure_paths = []  # Store CIF file paths
        self.soap_features = None
        self.influence_scores = None
        self.pruning_mask = None

        # SOAP descriptor configuration (based on the reference code)
        self.soap_params = {
            'r_cut': 5.0,
            'n_max': 8,
            'l_max': 6,
            'sigma': 0.5,
            'periodic': True,
            'average': 'off',  # We'll compute weighted average manually
            'sparse': False
        }

        # Symmetry precision for equivalent sites identification
        self.symprec = 1e-5

        # FUSION-specific attributes for dynamic optimization
        self.distance_matrix = None
        self.neighbor_distances = None
        self.neighbor_indices = None
        self.diversity_scores = None
        self.influence_clusters = None
        self.value_cache = {}

    def _load_from_cache(self, cache_path):
        """Load data from cache if it exists."""
        if os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, data, cache_path):
        """Save data to cache."""
        print(f"Saving data to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

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
                self.uncertainties = cached_data['uncertainties']
                self.targets = cached_data['targets']
                self.structure_ids = cached_data['structure_ids']
                self.structures = cached_data['structures']
                self.structure_paths = cached_data.get('structure_paths', [])
                print(f"Loaded {len(self.uncertainties)} uncertainties from cache")
                print(f"Uncertainties shape: {self.uncertainties.shape}")
                return

        print("Computing uncertainties for all structures...")
        uncertainties_list = []
        targets_list = []
        structure_ids_list = []
        structures_list = []
        structure_paths_list = []

        with torch.no_grad():
            for i, data in enumerate(tqdm(self.data_loader, desc="Computing uncertainties")):
                input_data = data[0]
                target = data[1]

                # Extract structure ID and path
                if len(data) > 2:
                    structure_info = data[2]
                    if isinstance(structure_info, dict):
                        structure_id = structure_info.get('jid', f"structure_{i}")
                        structure_path = structure_info.get('cif_path', '')
                    elif isinstance(structure_info, (list, tuple)):
                        # Handle case where structure_info is a list
                        structure_id = structure_info[0] if len(structure_info) > 0 else f"structure_{i}"
                        structure_path = structure_info[1] if len(structure_info) > 1 else ''
                    else:
                        structure_id = str(structure_info)
                        structure_path = ''
                else:
                    structure_id = f"structure_{i}"
                    structure_path = ''

                # Move data to device
                input_var = [input_data[0].to(self.device), input_data[1].to(self.device)]

                # Get model output (evidential parameters)
                output = self.model(input_var)

                # Extract evidential parameters
                if isinstance(output, (list, tuple)) and len(output) >= 4:
                    mu = output[0].cpu().numpy().flatten()
                    v = output[1].cpu().numpy().flatten()  # Precision parameter
                    alpha = output[2].cpu().numpy().flatten() + 1.0  # Ensure alpha > 1
                    beta = output[3].cpu().numpy().flatten()

                    # Compute aleatoric uncertainty: beta/(alpha-1)
                    aleatoric_uncertainty = beta / (alpha - 1)

                    if isinstance(aleatoric_uncertainty, np.ndarray):
                        aleatoric_uncertainty = torch.from_numpy(aleatoric_uncertainty).float()
                    if isinstance(v, np.ndarray):
                        v = torch.from_numpy(v).float()

                    # Compute total uncertainty: aleatoric * (-1 + lambda * 1/v)
                    total_uncertainty = torch.sigmoid(aleatoric_uncertainty * (-1 + self.Lambda * 1 / v))
                    total_uncertainty = total_uncertainty.detach().cpu().numpy()
                elif hasattr(output, 'shape') and len(output.shape) > 1:
                    # If output is a tensor, take the first column as mean
                    total_uncertainty = np.ones(output.shape[0]) * 0.1
                else:
                    # Fallback if evidential output is different
                    total_uncertainty = np.array([0.1])

                # Ensure we have scalar values
                if total_uncertainty.size > 1:
                    total_uncertainty = total_uncertainty[0]

                target_np = target.cpu().numpy().flatten()
                if target_np.size > 1:
                    target_np = target_np[0]

                # Store results
                uncertainties_list.append(total_uncertainty)
                targets_list.append(target_np)
                structure_ids_list.append(structure_id)
                structures_list.append(input_data)
                structure_paths_list.append(structure_path)

        # Convert to numpy arrays and ensure they are 1D
        self.uncertainties = np.array(uncertainties_list).flatten()
        self.targets = np.array(targets_list).flatten()
        self.structure_ids = structure_ids_list
        self.structures = structures_list
        self.structure_paths = structure_paths_list

        # Cache the results
        cache_data = {
            'uncertainties': self.uncertainties,
            'targets': self.targets,
            'structure_ids': self.structure_ids,
            'structures': self.structures,
            'structure_paths': self.structure_paths
        }
        self._save_to_cache(cache_data, self.uncertainties_cache)

        print(f"Computed uncertainties for {len(self.uncertainties)} structures")
        print(f"Final uncertainties shape: {self.uncertainties.shape}")
        print(f"Final targets shape: {self.targets.shape}")

    def identify_equivalent_sites(self, structure, symprec=1e-5):
        """
        Identify equivalent sites in crystal structure.
        Based on the reference code.

        Args:
            structure: ASE Atoms object
            symprec: Symmetry search precision

        Returns:
            Dictionary of equivalent site groups
        """
        # Create spglib cell tuple (lattice, positions, numbers)
        lattice = structure.get_cell()
        positions = structure.get_scaled_positions()
        numbers = structure.get_atomic_numbers()
        cell = (lattice, positions, numbers)

        # Get symmetry data
        symmetry_data = spglib.get_symmetry_dataset(cell, symprec=symprec)

        if symmetry_data is None:
            # Fallback: treat each atom as a separate site
            n_atoms = len(structure)
            site_groups = {}
            for i in range(n_atoms):
                site_groups[i] = {
                    "indices": np.array([i]),
                    "multiplicity": 1,
                    "element": structure.get_chemical_symbols()[i]
                }
            return site_groups

        # Equivalent atoms table
        equivalent_atoms = symmetry_data["equivalent_atoms"]

        # Group equivalent sites
        site_groups = {}
        unique_sites = np.unique(equivalent_atoms)

        for i, site_id in enumerate(unique_sites):
            # Get indices of all atoms belonging to this equivalent site
            atom_indices = np.where(equivalent_atoms == site_id)[0]

            # Store equivalent site group and its multiplicity
            site_groups[i] = {
                "indices": atom_indices,
                "multiplicity": len(atom_indices),
                "element": structure.get_chemical_symbols()[atom_indices[0]]
            }

        return site_groups

    def calculate_geometric_fingerprint(self, structure, r_cut=5.0, n_max=8, l_max=6, sigma=0.5):
        """
        Calculate geometric feature fingerprint F_geom as described in FUSION paper.
        Based on the reference code implementation.

        Args:
            structure: ASE Atoms object
            r_cut: Cutoff radius (Å)
            n_max: Maximum radial basis function index
            l_max: Maximum spherical harmonics angular index
            sigma: Gaussian width (Å)

        Returns:
            F_geom: Geometric feature vector
        """
        # Ensure structure has periodic boundary conditions
        if not all(structure.pbc):
            structure.pbc = [True, True, True]

        # Identify equivalent sites
        site_groups = self.identify_equivalent_sites(structure, self.symprec)

        # Define chemical species list
        species = list(set(structure.get_chemical_symbols()))

        # Set up SOAP descriptor with updated parameters
        soap_params = self.soap_params.copy()
        soap_params.update({
            'species': species,
            'r_cut': r_cut,
            'n_max': n_max,
            'l_max': l_max,
            'sigma': sigma
        })

        soap = SOAP(**soap_params)

        # Compute SOAP descriptors for all atoms
        soap_descriptors = soap.create(structure)

        # Get SOAP descriptor dimension
        soap_dim = soap_descriptors.shape[1]

        # Compute weighted average
        F_geom = np.zeros(soap_dim)
        total_atoms = 0

        for group_id, group_data in site_groups.items():
            indices = group_data["indices"]
            multiplicity = group_data["multiplicity"]
            total_atoms += multiplicity

            # Get SOAP descriptor for this equivalent site group
            site_soap = soap_descriptors[indices[0]]

            # Verify that SOAP descriptors within the same equivalent site group are consistent
            # (they should be identical due to symmetry)
            for idx in indices[1:]:
                if not np.allclose(site_soap, soap_descriptors[idx], rtol=1e-5, atol=1e-8):
                    # If not identical, use the average
                    site_soap = np.mean(soap_descriptors[indices], axis=0)
                    break

            # Weighted contribution
            F_geom += multiplicity * site_soap

        # Normalize
        if total_atoms > 0:
            F_geom /= total_atoms

        # L2 normalization
        if np.linalg.norm(F_geom) > 0:
            F_geom_norm = F_geom / np.linalg.norm(F_geom)
        else:
            F_geom_norm = F_geom

        return F_geom_norm

    def compute_structural_similarity(self, force_recompute=False):
        """
        Compute structural similarities using SOAP descriptors from CIF files.
        Uses the improved implementation based on the reference code.

        Args:
            force_recompute (bool): Force recomputation even if cache exists
        """
        if not force_recompute:
            cached_data = self._load_from_cache(self.soap_cache)
            if cached_data is not None:
                self.soap_features = cached_data
                print(f"Loaded SOAP features from cache: {self.soap_features.shape}")
                return

        print("Computing structural similarities using SOAP descriptors from CIF files...")

        soap_features_list = []
        successful_computations = 0

        # Look for CIF files in the data directory structure
        data_dir = Path(self.data_path)
        possible_cif_dirs = [
            data_dir / self.task,
        ]

        # Find all CIF files
        cif_files = []
        for dir_path in possible_cif_dirs:
            if dir_path.exists():
                cif_files.extend(list(dir_path.glob("*.cif")))
                # Also look recursively
                cif_files.extend(list(dir_path.rglob("*.cif")))

        # Remove duplicates
        cif_files = list(set(cif_files))
        print(f"Found {len(cif_files)} CIF files in data directory")

        # Create a mapping from structure IDs to CIF files
        id_to_cif = {}
        for cif_file in cif_files:
            # Extract ID from filename (you may need to adjust this logic)
            file_id = cif_file.stem
            id_to_cif[file_id] = cif_file

            # Also try without extension and with common prefixes
            if file_id.startswith("JVASP-"):
                id_to_cif[file_id[6:]] = cif_file
            elif not file_id.startswith("JVASP-"):
                id_to_cif[f"JVASP-{file_id}"] = cif_file

        print("Computing SOAP descriptors for structures...")

        for i, structure_id in enumerate(tqdm(self.structure_ids, desc="Computing SOAP")):
            try:
                # First try to use stored CIF path
                cif_path = None
                if i < len(self.structure_paths) and self.structure_paths[i]:
                    cif_path = Path(self.structure_paths[i])

                # If no stored path, try to find it by ID
                if not cif_path or not cif_path.exists():
                    # Try different variations of the structure ID
                    potential_ids = [
                        structure_id,
                        f"JVASP-{structure_id}",
                        f"{structure_id}.cif",
                        str(structure_id).replace("JVASP-", "")
                    ]

                    for potential_id in potential_ids:
                        if potential_id in id_to_cif:
                            cif_path = id_to_cif[potential_id]
                            break

                if cif_path and cif_path.exists():
                    # Read structure from CIF file
                    structure = read(str(cif_path))

                    # Calculate geometric fingerprint
                    F_geom = self.calculate_geometric_fingerprint(
                        structure,
                        r_cut=self.soap_params['r_cut'],
                        n_max=self.soap_params['n_max'],
                        l_max=self.soap_params['l_max'],
                        sigma=self.soap_params['sigma']
                    )

                    soap_features_list.append(F_geom)
                    successful_computations += 1

                else:
                    # Fallback: try to extract from graph data
                    print(f"Warning: No CIF file found for {structure_id}, using fallback method")

                    # Extract features from ALIGNN graph data
                    if i < len(self.structures):
                        graph_data = self.structures[i]
                        # Use a simple feature extraction as fallback
                        if hasattr(graph_data[0], 'cpu'):
                            node_features = graph_data[0].cpu().numpy()
                        else:
                            node_features = graph_data[0]

                        # Create a simple averaged feature vector
                        if len(node_features.shape) > 1:
                            avg_features = np.mean(node_features, axis=0)
                        else:
                            avg_features = node_features

                        # Pad or truncate to match expected SOAP dimension
                        if len(soap_features_list) > 0:
                            target_dim = len(soap_features_list[0])
                            if len(avg_features) > target_dim:
                                avg_features = avg_features[:target_dim]
                            elif len(avg_features) < target_dim:
                                padded = np.zeros(target_dim)
                                padded[:len(avg_features)] = avg_features
                                avg_features = padded

                        soap_features_list.append(avg_features)
                    else:
                        # Last resort: zero vector
                        if len(soap_features_list) > 0:
                            soap_features_list.append(np.zeros_like(soap_features_list[0]))
                        else:
                            soap_features_list.append(np.zeros(100))  # Default dimension

            except Exception as e:
                print(f"Error processing structure {structure_id}: {e}")
                # Use fallback
                if len(soap_features_list) > 0:
                    soap_features_list.append(np.zeros_like(soap_features_list[0]))
                else:
                    soap_features_list.append(np.zeros(100))  # Default dimension

        # Convert to numpy array and ensure all features have the same dimension
        if len(soap_features_list) > 0:
            # Find the maximum dimension
            max_dim = max(len(f) for f in soap_features_list)

            # Pad all features to the same dimension
            padded_features = []
            for features in soap_features_list:
                if len(features) < max_dim:
                    padded = np.zeros(max_dim)
                    padded[:len(features)] = features
                    padded_features.append(padded)
                else:
                    padded_features.append(features[:max_dim])

            self.soap_features = np.array(padded_features)
        else:
            # Fallback: create dummy features
            print("Warning: No SOAP features computed, creating dummy features")
            self.soap_features = np.random.randn(len(self.structure_ids), 100)

        # Cache the results
        self._save_to_cache(self.soap_features, self.soap_cache)

        print(f"Computed SOAP features with shape: {self.soap_features.shape}")
        print(f"Successfully computed SOAP for {successful_computations} out of {len(self.structure_ids)} structures")

    def compute_weighting_factors(self):
        """
        Compute weighting factors based on structural similarity.
        The weighting factor ω(x_i) accounts for the structure's position in the materials space.
        """
        print("Computing weighting factors...")

        # Use k-nearest neighbors to find the most similar structure for each structure
        n_structures = len(self.soap_features)
        nbrs = NearestNeighbors(n_neighbors=min(5, n_structures), algorithm='auto').fit(self.soap_features)

        # For each structure, find the distance to its nearest neighbor
        distances, indices = nbrs.kneighbors(self.soap_features)

        # The first neighbor is the structure itself, so take the second one
        if distances.shape[1] > 1:
            min_distances = distances[:, 1]
        else:
            min_distances = np.ones(n_structures) * 0.1  # Fallback

        # Compute weighting factor: ω(x_i) = exp(-γ · min_{j≠i} d(x_i, x_j))
        weighting_factors = np.exp(-self.gamma * min_distances)

        return weighting_factors

    def compute_uncertainty_influence(self, force_recompute=False):
        """
        Compute uncertainty influence scores for each structure.
        Uncertainty influence I_unc(x_i) = unc(x_i) / ω(x_i)

        Args:
            force_recompute (bool): Force recomputation even if cache exists
        """
        if not force_recompute:
            cached_data = self._load_from_cache(self.influence_cache)
            if cached_data is not None:
                self.influence_scores = cached_data
                print(f"Loaded influence scores from cache")
                print(f"Influence scores shape: {self.influence_scores.shape}")
                return

        print("Computing uncertainty influence scores...")

        # Add debugging information
        print(f"Uncertainties shape: {self.uncertainties.shape}")
        print(f"SOAP features shape: {self.soap_features.shape}")

        # Ensure arrays are 1D
        if len(self.uncertainties.shape) > 1:
            print("Warning: Flattening uncertainties array")
            self.uncertainties = self.uncertainties.flatten()

        # Compute weighting factors based on structural similarity
        weighting_factors = self.compute_weighting_factors()
        print(f"Weighting factors shape: {weighting_factors.shape}")

        # Ensure weighting factors are 1D
        if len(weighting_factors.shape) > 1:
            print("Warning: Flattening weighting factors array")
            weighting_factors = weighting_factors.flatten()

        # Calculate uncertainty influence: I_unc(x_i) = unc(x_i) / ω(x_i)
        self.influence_scores = self.uncertainties / weighting_factors

        # Ensure influence scores are 1D
        if len(self.influence_scores.shape) > 1:
            print("Warning: Flattening influence scores array")
            self.influence_scores = self.influence_scores.flatten()

        # Cache the results
        self._save_to_cache(self.influence_scores, self.influence_cache)

        print(f"Final influence scores shape: {self.influence_scores.shape}")
        print(f"Uncertainty influence scores statistics:")
        print(f"  Min: {self.influence_scores.min():.6f}")
        print(f"  Max: {self.influence_scores.max():.6f}")
        print(f"  Mean: {self.influence_scores.mean():.6f}")
        print(f"  Std: {self.influence_scores.std():.6f}")

    # ===== FUSION-SPECIFIC METHODS: Dynamic Value Functions =====

    def _precompute_structural_relationships(self):
        """Pre-compute structural relationships for FUSION dynamic optimization"""
        print("Pre-computing structural relationships for FUSION dynamic optimization...")

        # Compute pairwise distance matrix
        self.distance_matrix = pairwise_distances(self.soap_features, metric='euclidean')

        # Build k-nearest neighbor graph
        k = min(10, len(self.soap_features) - 1)
        knn = NearestNeighbors(n_neighbors=k, metric='precomputed')
        knn.fit(self.distance_matrix)
        self.neighbor_distances, self.neighbor_indices = knn.kneighbors(self.distance_matrix)

        # Compute diversity scores (inverse of local density)
        self.diversity_scores = np.zeros(len(self.soap_features))
        for i in range(len(self.soap_features)):
            local_density = np.mean(self.neighbor_distances[i, 1:])  # Skip self
            self.diversity_scores[i] = 1.0 / (local_density + 1e-8)

        # Compute influence clusters for coverage bonus
        quantiles = np.quantile(self.influence_scores, [0.2, 0.4, 0.6, 0.8])
        self.influence_clusters = {
            'very_low': np.where(self.influence_scores <= quantiles[0])[0],
            'low': np.where((self.influence_scores > quantiles[0]) &
                            (self.influence_scores <= quantiles[1]))[0],
            'medium': np.where((self.influence_scores > quantiles[1]) &
                               (self.influence_scores <= quantiles[2]))[0],
            'high': np.where((self.influence_scores > quantiles[2]) &
                             (self.influence_scores <= quantiles[3]))[0],
            'very_high': np.where(self.influence_scores > quantiles[3])[0]
        }

        print(f"Structural relationships computed:")
        print(f"  Distance matrix shape: {self.distance_matrix.shape}")
        print(f"  Diversity scores range: [{self.diversity_scores.min():.3f}, {self.diversity_scores.max():.3f}]")
        print(f"  Influence clusters: {[len(cluster) for cluster in self.influence_clusters.values()]}")

    def fusion_dynamic_value_function(self, candidate_idx, current_selection):
        """
        FUSION Dynamic Value Function: Context-aware sample value evaluation

        Args:
            candidate_idx: Index of candidate sample
            current_selection: Current selection mask (boolean array)

        Returns:
            Dynamic value score considering quality, diversity, redundancy, and coverage
        """
        # Cache key for memoization
        selection_hash = hash(current_selection.tobytes())
        cache_key = (candidate_idx, selection_hash)

        if cache_key in self.value_cache:
            return self.value_cache[cache_key]

        # 1. Base influence score (negative because we want low influence)
        base_value = -self.influence_scores[candidate_idx]

        # 2. Diversity bonus: reward selecting structurally diverse samples
        if np.any(current_selection):
            selected_indices = np.where(current_selection)[0]
            min_distance = np.min(self.distance_matrix[candidate_idx, selected_indices])
            avg_distance = np.mean(self.distance_matrix[candidate_idx, :])
            diversity_bonus = (min_distance / (avg_distance + 1e-8)) * self.diversity_scores[candidate_idx]
        else:
            diversity_bonus = self.diversity_scores[candidate_idx]

        # 3. Redundancy penalty: penalize selecting similar samples
        if np.any(current_selection):
            neighbor_mask = np.isin(self.neighbor_indices[candidate_idx],
                                    np.where(current_selection)[0])
            redundancy_penalty = np.sum(neighbor_mask) / len(self.neighbor_indices[candidate_idx])
        else:
            redundancy_penalty = 0.0

        # 4. Coverage bonus: ensure balanced representation across influence ranges
        candidate_cluster = None
        for cluster_name, cluster_indices in self.influence_clusters.items():
            if candidate_idx in cluster_indices:
                candidate_cluster = cluster_name
                break

        if candidate_cluster and np.any(current_selection):
            cluster_indices = self.influence_clusters[candidate_cluster]
            selected_from_cluster = np.sum(current_selection[cluster_indices])
            representation_ratio = selected_from_cluster / len(cluster_indices)
            coverage_bonus = 1.0 - representation_ratio
        else:
            coverage_bonus = 0.5

        # 5. Adaptive weights based on selection progress
        selection_ratio = np.sum(current_selection) / len(current_selection)
        if selection_ratio < 0.2:
            # Early stage: prioritize base quality
            w_base, w_diversity, w_redundancy, w_coverage = 0.6, 0.1, 0.1, 0.2
        elif selection_ratio < 0.5:
            # Middle stage: balance all factors
            w_base, w_diversity, w_redundancy, w_coverage = 0.4, 0.2, 0.2, 0.2
        else:
            # Late stage: prioritize diversity and coverage
            w_base, w_diversity, w_redundancy, w_coverage = 0.2, 0.3, 0.3, 0.2

        # 6. Combine all factors
        dynamic_value = (w_base * base_value +
                         w_diversity * diversity_bonus +
                         w_redundancy * (-redundancy_penalty) +
                         w_coverage * coverage_bonus)

        # Cache the result
        self.value_cache[cache_key] = dynamic_value

        return dynamic_value

    def fusion_incremental_constraint_check(self, candidate_idx, current_selection):
        """
        FUSION Incremental Constraint Check: Fast feasibility verification

        Args:
            candidate_idx: Index of candidate sample
            current_selection: Current selection mask

        Returns:
            (feasible, new_avg_influence)
        """
        selected_indices = np.where(current_selection)[0]

        if len(selected_indices) == 0:
            new_avg_influence = self.influence_scores[candidate_idx]
        else:
            current_sum = np.sum(self.influence_scores[selected_indices])
            new_sum = current_sum + self.influence_scores[candidate_idx]
            new_avg_influence = new_sum / (len(selected_indices) + 1)

        feasible = new_avg_influence <= self.epsilon
        return feasible, new_avg_influence

    # ===== FUSION OPTIMIZATION ALGORITHMS =====

    def fusion_fast_dynamic_greedy(self, max_iterations=None):
        """
        FUSION Fast Dynamic Greedy: Efficient greedy with dynamic value functions

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            Pruning ratio
        """
        print("Running FUSION Fast Dynamic Greedy optimization...")

        # Pre-compute relationships if not done
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
            best_value = -np.inf

            # Evaluate all candidates using dynamic value function
            for candidate in candidates:
                # Check feasibility
                feasible, _ = self.fusion_incremental_constraint_check(candidate, current_selection)
                if not feasible:
                    continue

                # Compute dynamic value
                value = self.fusion_dynamic_value_function(candidate, current_selection)

                if value > best_value:
                    best_value = value
                    best_candidate = candidate

            if best_candidate is None:
                break

            # Add best candidate to selection
            current_selection[best_candidate] = True

            # Periodically clear cache to manage memory
            if len(self.value_cache) > 10000:
                self.value_cache.clear()

        self.pruning_mask = current_selection
        ratio = np.sum(current_selection) / len(current_selection)

        print(f"FUSION Fast Dynamic Greedy completed. Ratio: {ratio:.4f}")
        return ratio

    def fusion_adaptive_beam_search(self, beam_width=3, max_iterations=500):
        """
        FUSION Adaptive Beam Search: Maintain multiple partial solutions for better exploration

        Args:
            beam_width: Number of partial solutions to maintain
            max_iterations: Maximum number of iterations

        Returns:
            Pruning ratio
        """
        print(f"Running FUSION Adaptive Beam Search (beam_width={beam_width})...")

        # Pre-compute relationships if not done
        if self.distance_matrix is None:
            self._precompute_structural_relationships()

        # Initialize beam with empty selection
        beam = [np.zeros(len(self.influence_scores), dtype=bool)]
        beam_scores = [0.0]

        best_solution = None
        best_score = -np.inf
        best_ratio = 0.0

        for iteration in tqdm(range(max_iterations), desc="FUSION Beam Search"):
            new_beam = []
            new_scores = []

            # Expand each solution in current beam
            for sol_idx, current_solution in enumerate(beam):
                current_score = beam_scores[sol_idx]
                candidates = np.where(~current_solution)[0]

                # Evaluate candidates for this solution
                candidate_values = []
                for candidate in candidates:
                    feasible, _ = self.fusion_incremental_constraint_check(candidate, current_solution)
                    if feasible:
                        value = self.fusion_dynamic_value_function(candidate, current_solution)
                        candidate_values.append((candidate, value))

                # Select top candidates for expansion
                if candidate_values:
                    candidate_values.sort(key=lambda x: x[1], reverse=True)

                    # Adaptive expansion based on current selection size
                    n_selected = np.sum(current_solution)
                    if n_selected < 10:
                        n_expand = min(beam_width, len(candidate_values))
                    else:
                        n_expand = min(beam_width // 2, len(candidate_values))

                    for i in range(n_expand):
                        candidate, value = candidate_values[i]

                        # Create new solution
                        new_solution = current_solution.copy()
                        new_solution[candidate] = True
                        new_score = current_score + value

                        new_beam.append(new_solution)
                        new_scores.append(new_score)

                        # Update best solution
                        if new_score > best_score:
                            best_solution = new_solution.copy()
                            best_score = new_score
                            best_ratio = np.sum(new_solution) / len(new_solution)

            # Select top solutions for next iteration
            if new_beam:
                combined = list(zip(new_beam, new_scores))
                combined.sort(key=lambda x: x[1], reverse=True)

                beam = [sol for sol, score in combined[:beam_width]]
                beam_scores = [score for sol, score in combined[:beam_width]]
            else:
                # No more feasible expansions
                break

            # Clear cache periodically
            if iteration % 50 == 0:
                self.value_cache.clear()

        if best_solution is not None:
            self.pruning_mask = best_solution
            print(f"FUSION Beam Search completed. Best ratio: {best_ratio:.4f}")
            return best_ratio
        else:
            print("No feasible solution found!")
            self.pruning_mask = np.zeros(len(self.influence_scores), dtype=bool)
            return 0.0

    def fusion_enhanced_simulated_annealing(self, max_iterations=5000):
        """
        FUSION Enhanced Simulated Annealing: Global optimization with dynamic value functions

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            Pruning ratio
        """
        print("Running FUSION Enhanced Simulated Annealing...")

        # Pre-compute relationships if not done
        if self.distance_matrix is None:
            self._precompute_structural_relationships()

        # Initialize with fast greedy solution
        current_mask = np.zeros(len(self.influence_scores), dtype=bool)
        temp_epsilon = self.epsilon
        self.epsilon = self.epsilon * 1.1  # Slightly relax constraint for initial solution
        self.fusion_fast_dynamic_greedy(max_iterations=100)
        current_mask = self.pruning_mask.copy()
        self.epsilon = temp_epsilon  # Restore original constraint

        current_count = np.sum(current_mask)
        current_influence = np.mean(self.influence_scores[current_mask]) if current_count > 0 else 0

        # Best solution tracking
        best_mask = current_mask.copy()
        best_count = current_count

        # Temperature parameters
        temp_initial = 10.0
        temp_final = 0.01

        print(f"Initial solution: {current_count} structures, influence: {current_influence:.6f}")

        for iteration in tqdm(range(max_iterations), desc="FUSION Enhanced SA"):
            # Temperature update
            progress = iteration / max_iterations
            temp = temp_final * (temp_initial / temp_final) ** (1 - progress)

            # Intelligent neighbor generation
            neighbor_mask = current_mask.copy()

            # Decide operation based on temperature and current state
            if np.random.rand() < 0.7:  # 70% probability to add
                candidates = np.where(~current_mask)[0]
                if len(candidates) > 0:
                    # Evaluate candidates with dynamic value function
                    values = []
                    for candidate in candidates:
                        feasible, _ = self.fusion_incremental_constraint_check(candidate, current_mask)
                        if feasible:
                            value = self.fusion_dynamic_value_function(candidate, current_mask)
                            values.append((candidate, value))

                    if values:
                        # Temperature-based probabilistic selection
                        values.sort(key=lambda x: x[1], reverse=True)
                        top_candidates = values[:min(5, len(values))]
                        weights = np.exp([v[1] / (temp + 1e-8) for v in top_candidates])
                        weights /= np.sum(weights)

                        chosen_idx = np.random.choice(len(top_candidates), p=weights)
                        chosen_candidate = top_candidates[chosen_idx][0]
                        neighbor_mask[chosen_candidate] = True
            else:  # 30% probability to remove
                selected_indices = np.where(current_mask)[0]
                if len(selected_indices) > 0:
                    # Prefer removing high influence samples
                    selected_influences = self.influence_scores[selected_indices]
                    removal_probs = selected_influences / np.sum(selected_influences)

                    chosen_idx = np.random.choice(len(selected_indices), p=removal_probs)
                    neighbor_mask[selected_indices[chosen_idx]] = False

            # Evaluate neighbor solution
            neighbor_count = np.sum(neighbor_mask)
            if neighbor_count > 0:
                neighbor_influence = np.mean(self.influence_scores[neighbor_mask])
                feasible = neighbor_influence <= self.epsilon
            else:
                neighbor_influence = 0
                feasible = True

            # Acceptance criterion
            if feasible:
                delta_e = neighbor_count - current_count
                if delta_e > 0 or (temp > temp_final and np.random.rand() < np.exp(delta_e / temp)):
                    current_mask = neighbor_mask
                    current_count = neighbor_count
                    current_influence = neighbor_influence

                    if current_count > best_count:
                        best_mask = current_mask.copy()
                        best_count = current_count

            # Clear cache periodically
            if iteration % 1000 == 0:
                self.value_cache.clear()

        self.pruning_mask = best_mask
        ratio = best_count / len(self.influence_scores)
        print(f"FUSION Enhanced SA completed. Best ratio: {ratio:.4f}")
        return ratio

    # ===== LEGACY OPTIMIZATION METHODS (for compatibility) =====

    def optimize_pruning_mask(self, indices=None):
        """
        Legacy greedy optimization (for compatibility)

        Args:
            indices (list): Optional list of indices to consider for pruning.
                           If None, all structures are considered.
        """
        print("Running legacy greedy optimization...")

        # If indices are provided, create a mask for them
        if indices is not None:
            mask = np.zeros(len(self.influence_scores), dtype=bool)
            mask[indices] = True
            influence_scores = self.influence_scores[mask]
            # Map original indices to subset indices
            idx_map = {orig_idx: subset_idx for subset_idx, orig_idx in enumerate(indices)}
        else:
            influence_scores = self.influence_scores
            idx_map = None

        # Add debugging information
        print(f"Influence scores shape: {influence_scores.shape}")
        print(f"Number of structures: {len(influence_scores)}")

        # Ensure influence_scores is a 1D array
        if len(influence_scores.shape) > 1:
            print("Warning: Flattening influence scores array")
            influence_scores = influence_scores.flatten()

        n_structures = len(influence_scores)
        print(f"Processing {n_structures} structures")

        # Sort structures by their influence scores (ascending order)
        try:
            sorted_indices = np.argsort(influence_scores)
            sorted_scores = np.array([influence_scores[i] for i in sorted_indices])
        except Exception as e:
            print(f"Error during sorting: {e}")
            sorted_indices = np.arange(n_structures)
            sorted_scores = influence_scores.copy()

        # Find the maximum number of structures to remove while keeping average influence below epsilon
        max_index = 0
        cumulative_sum = 0.0

        print(f"Finding optimal pruning (epsilon={self.epsilon})...")

        for i in range(n_structures):
            cumulative_sum += float(sorted_scores[i])
            cumulative_mean = cumulative_sum / (i + 1)

            if cumulative_mean <= self.epsilon:
                max_index = i + 1
            else:
                break

            # Print progress for large datasets
            if i % 1000 == 0:
                print(f"  Processed {i + 1}/{n_structures}, current mean: {cumulative_mean:.6f}")

        # Initialize pruning mask (False = keep, True = remove)
        if indices is None:
            self.pruning_mask = np.zeros(n_structures, dtype=bool)
            for i in range(max_index):
                idx = sorted_indices[i]
                self.pruning_mask[idx] = True
        else:
            self.pruning_mask = np.zeros(len(self.influence_scores), dtype=bool)
            for i in range(max_index):
                subset_idx = sorted_indices[i]
                orig_idx = indices[subset_idx]
                self.pruning_mask[orig_idx] = True

        pruning_ratio = max_index / n_structures
        print(f"Legacy greedy completed: {max_index} out of {n_structures} structures selected for removal")
        print(f"Pruning ratio: {pruning_ratio}")

        return pruning_ratio

    def simulated_annealing_optimization(self, max_iterations=10000, indices=None):
        """
        Legacy simulated annealing optimization (for compatibility)
        """
        print("Using legacy simulated annealing algorithm...")

        # This is a simplified version - use FUSION enhanced SA for better results
        return self.fusion_enhanced_simulated_annealing(max_iterations // 2)

    # ===== MAIN EXECUTION METHODS =====

    def run(self, optimization_method='fusion_dynamic_greedy', force_recompute=False, indices=None, **kwargs):
        """
        Execute the complete FUSION pipeline.

        Args:
            optimization_method (str): Optimization method to use
                - 'fusion_dynamic_greedy': Fast dynamic greedy (recommended for large datasets)
                - 'fusion_beam_search': Adaptive beam search (best quality/speed trade-off)
                - 'fusion_enhanced_sa': Enhanced simulated annealing (best quality, slower)
                - 'greedy': Original greedy method
                - 'sa': Original simulated annealing
            force_recompute (bool): Force recomputation of all steps
            indices (list): Optional list of indices to consider for pruning
            **kwargs: Additional arguments for optimization methods

        Returns:
            dict: Results containing pruning mask, pruning ratio, and other metrics
        """
        # Step 1: Compute uncertainties using deep evidential regression
        self.compute_uncertainties(force_recompute=force_recompute)

        # Step 2: Compute structural similarities from CIF files
        self.compute_structural_similarity(force_recompute=force_recompute)

        # Step 3: Compute uncertainty influence scores
        self.compute_uncertainty_influence(force_recompute=force_recompute)

        # Step 4: Choose optimization method
        print(f"Using optimization method: {optimization_method}")

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
            print(f"Warning: Unknown optimization method '{optimization_method}', using fusion_dynamic_greedy")
            pruning_ratio = self.fusion_fast_dynamic_greedy(**kwargs)

        # Prepare results
        results = {
            'structure_ids': self.structure_ids,
            'pruning_mask': self.pruning_mask,
            'uncertainties': self.uncertainties,
            'influence_scores': self.influence_scores,
            'pruning_ratio': pruning_ratio,
            'targets': self.targets,
            'soap_features': self.soap_features,
            'structure_paths': self.structure_paths,
            'optimization_method': optimization_method
        }

        return results

    def generate_json_splits(self, epsilon_values, output_dir="./fusion_output", test_size=0.15, val_size=0.15,
                             random_state=42, optimization_method='fusion_dynamic_greedy'):
        """
        Generate train, val, and test JSON documents with data splits using FUSION.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Get total number of structures
        n_structures = len(self.structure_ids)
        all_indices = np.arange(n_structures)

        # First split: separate test set
        train_val_indices, test_indices = train_test_split(
            all_indices, test_size=test_size, random_state=random_state
        )

        # Second split: separate validation set from training set
        val_size_adjusted = val_size / (1 - test_size)
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_size_adjusted, random_state=random_state
        )

        print(f"Dataset split:")
        print(f"  Total structures: {n_structures}")
        print(f"  Training set: {len(train_indices)} structures ({len(train_indices) / n_structures:.2%})")
        print(f"  Validation set: {len(val_indices)} structures ({len(val_indices) / n_structures:.2%})")
        print(f"  Test set: {len(test_indices)} structures ({len(test_indices) / n_structures:.2%})")

        # Create the JSON documents
        json_docs = {
            "train": {},
            "val": {},
            "test": {}
        }

        # Store actual pruning ratios
        actual_pruning_ratios = {}

        # Model 0 contains the complete dataset without pruning
        json_docs["train"]["0"] = train_indices.tolist()
        json_docs["val"]["0"] = val_indices.tolist()
        json_docs["test"]["0"] = test_indices.tolist()
        actual_pruning_ratios[0] = 0.0000

        # Process each model with different epsilon values
        for model_num, epsilon in epsilon_values.items():
            if model_num == 0:
                continue

            model_str = str(model_num)
            print(f"\nProcessing Model {model_num} with epsilon={epsilon} using {optimization_method}")
            self.epsilon = epsilon

            # Run FUSION optimization only on train+val set
            combined_indices = np.concatenate([train_indices, val_indices])
            results = self.run(optimization_method=optimization_method,
                               force_recompute=False, indices=combined_indices)

            # Calculate actual pruning ratio
            actual_pruning_ratio = results['pruning_ratio']
            actual_pruning_ratios[model_num] = actual_pruning_ratio

            # Get indices of retained structures (not pruned)
            retained_indices = np.where(~results['pruning_mask'])[0]
            retained_indices = retained_indices[np.isin(retained_indices, combined_indices)]

            # Split retained structures into train and validation sets
            retained_train = np.intersect1d(retained_indices, train_indices)
            retained_val = np.intersect1d(retained_indices, val_indices)

            # Test set remains the same for fair comparison
            json_docs["train"][model_str] = retained_train.tolist()
            json_docs["val"][model_str] = retained_val.tolist()
            json_docs["test"][model_str] = test_indices.tolist()

            print(f"Model {model_num} after FUSION pruning:")
            print(f"  Training set: {len(retained_train)} structures")
            print(f"  Validation set: {len(retained_val)} structures")
            print(f"  Test set: {len(test_indices)} structures (same as model 0)")
            print(f"  FUSION removed {len(combined_indices) - len(retained_train) - len(retained_val)} structures")

        # Generate timestamp and save results
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON documents
        for split, data in json_docs.items():
            output_path = os.path.join(output_dir, f"{split}_{timestamp}.json")
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {split}.json to {output_path}")

        # Save pruning ratios
        pruning_ratios_file = os.path.join(output_dir, f"fusion_pruning_ratios_{timestamp}.json")
        with open(pruning_ratios_file, 'w') as f:
            json.dump(actual_pruning_ratios, f, indent=2)
        print(f"FUSION pruning ratios saved to {pruning_ratios_file}")

        # Save detailed mapping
        fusion_mapping = {
            "metadata": {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "optimization_method": optimization_method,
                "total_models": len(actual_pruning_ratios),
                "task": self.task,
                "lambda": self.Lambda
            },
            "model_data": {}
        }

        for model_num, epsilon in epsilon_values.items():
            if model_num in actual_pruning_ratios:
                fusion_mapping["model_data"][str(model_num)] = {
                    "epsilon": epsilon,
                    "pruning_ratio": actual_pruning_ratios[model_num]
                }

        mapping_file = os.path.join(output_dir, f"fusion_epsilon_ratio_mapping_{timestamp}.json")
        with open(mapping_file, 'w') as f:
            json.dump(fusion_mapping, f, indent=2)
        print(f"FUSION epsilon-ratio mapping saved to {mapping_file}")

        print(f"\nFUSION Summary:")
        for model_num in sorted(actual_pruning_ratios.keys()):
            epsilon = epsilon_values.get(model_num, "N/A")
            ratio = actual_pruning_ratios[model_num]
            print(f"  Model {model_num}: epsilon={epsilon}, pruning_ratio={ratio:.4f}")

        return json_docs

    def generate_json_splits_from_predefined(self, epsilon_values, predefined_splits,
                                             output_dir="./fusion_output",
                                             optimization_method='fusion_dynamic_greedy'):
        """
        Generate pruned JSON documents based on predefined train/val/test splits using FUSION.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Extract indices from predefined splits
        original_train_indices = np.array(predefined_splits['train'])
        original_val_indices = np.array(predefined_splits['val'])
        test_indices = np.array(predefined_splits['test'])

        n_structures = len(self.structure_ids)

        print(f"Using predefined dataset splits with FUSION:")
        print(f"  Total structures: {n_structures}")
        print(f"  Training set: {len(original_train_indices)} structures")
        print(f"  Validation set: {len(original_val_indices)} structures")
        print(f"  Test set: {len(test_indices)} structures")
        print(f"  Optimization method: {optimization_method}")

        # Create JSON documents
        json_docs = {
            "train": {},
            "val": {},
            "test": {}
        }

        # Store actual pruning ratios
        actual_pruning_ratios = {}

        # Model 0 contains complete dataset (no pruning)
        json_docs["train"]["0"] = original_train_indices.tolist()
        json_docs["val"]["0"] = original_val_indices.tolist()
        json_docs["test"]["0"] = test_indices.tolist()
        actual_pruning_ratios[0] = 0.0000

        # Process each model with different epsilon values
        for model_num, epsilon in epsilon_values.items():
            if model_num == 0:
                continue

            model_str = str(model_num)

            print(f"\nProcessing Model {model_num} with epsilon={epsilon} using FUSION")
            self.epsilon = epsilon

            # Apply FUSION pruning only to train+val set
            combined_indices = np.concatenate([original_train_indices, original_val_indices])

            # Run FUSION optimization
            results = self.run(optimization_method=optimization_method,
                               force_recompute=False, indices=combined_indices)

            # Calculate actual pruning ratio
            actual_pruning_ratio = results['pruning_ratio']
            actual_pruning_ratios[model_num] = actual_pruning_ratio

            # Get retained structure indices (not pruned)
            retained_indices = np.where(~results['pruning_mask'])[0]
            retained_indices = retained_indices[np.isin(retained_indices, combined_indices)]

            # Redistribute retained structures to train and val sets
            retained_train = np.intersect1d(retained_indices, original_train_indices)
            retained_val = np.intersect1d(retained_indices, original_val_indices)

            # Test set remains unchanged
            json_docs["train"][model_str] = retained_train.tolist()
            json_docs["val"][model_str] = retained_val.tolist()
            json_docs["test"][model_str] = test_indices.tolist()

            print(f"Model {model_num} after FUSION pruning:")
            print(f"  Training set: {len(retained_train)} structures")
            print(f"  Validation set: {len(retained_val)} structures")
            print(f"  Test set: {len(test_indices)} structures (unchanged)")
            print(f"  FUSION removed {len(combined_indices) - len(retained_train) - len(retained_val)} structures")

        # Generate timestamp and save results
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON documents
        for split, data in json_docs.items():
            output_path = os.path.join(output_dir, f"{split}_{timestamp}.json")
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {split}.json to {output_path}")

        # Save pruning ratios
        pruning_ratios_file = os.path.join(output_dir, f"fusion_pruning_ratios_{timestamp}.json")
        with open(pruning_ratios_file, 'w') as f:
            json.dump(actual_pruning_ratios, f, indent=2)
        print(f"FUSION pruning ratios saved to {pruning_ratios_file}")

        return json_docs


# Alias for backward compatibility
MAPS = FUSION


def main_with_predefined_splits():
    """
    Main function for running FUSION with predefined splits.
    """
    for model_key in ["0", "1", "2", "3", "4"]:
        # Configuration
        config_path = "config.yml"
        checkpoint_path = "/home/wangxiean/PycharmProjects/FUSION/surrogate/jdft2d/ALIGNN/model_best.pth.tar"
        data_path = "/data1/tanliqin/uq-ood-mat/data/"
        cache_dir = "./fusion_cache_lambda0"
        task = "jdft2d"
        json_output_dir = f"./FUSION_output/{task}_lambda0_ood/model_key_{model_key}"

        # Initialize FUSION
        fusion = FUSION(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            data_path=data_path,
            task=task,
            epsilon=0.01,
            gamma=0.1,
            cache_dir=cache_dir,
            Lambda=0
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

        # Get optimal epsilons
        searcher = EpsilonSearcher(fusion)
        target_ratios = [0, 0.2, 0.4, 0.6, 0.8]
        optimal_epsilons = searcher.optimize_epsilon_for_target_ratios(target_ratios)
        epsilon_values = {i: eps for i, eps in enumerate(optimal_epsilons.values())}

        # Load predefined splits
        predefined_splits = load_splits_from_json_files(
            train_file="/data1/tanliqin/uq-ood-mat/folds/jdft2d_folds/train/SOAP_jdft2d_LOCO_target_clusters40_train.json",
            val_file="/data1/tanliqin/uq-ood-mat/folds/jdft2d_folds/val/SOAP_jdft2d_LOCO_target_clusters40_val.json",
            test_file="/data1/tanliqin/uq-ood-mat/folds/jdft2d_folds/test/SOAP_jdft2d_LOCO_target_clusters40_test.json",
            model_key=model_key
        )

        # Generate JSON documents using FUSION
        print("\nGenerating JSON documents with FUSION optimization...")
        json_docs = fusion.generate_json_splits_from_predefined(
            epsilon_values=epsilon_values,
            predefined_splits=predefined_splits,
            output_dir=json_output_dir,
            optimization_method='fusion_dynamic_greedy'  # or 'fusion_beam_search', 'fusion_enhanced_sa'
        )

        print("\nSummary of FUSION-generated JSON documents:")
        for split, model_data in json_docs.items():
            print(f"\n{split.upper()} set:")
            for model, indices in model_data.items():
                print(f"  Model {model}: {len(indices)} structures")


def load_splits_from_json_files(train_file, val_file, test_file, model_key="0"):
    """
    Load train/val/test splits from existing JSON files
    """
    predefined_splits = {}

    try:
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        predefined_splits['train'] = train_data.get(model_key, [])

        with open(val_file, 'r') as f:
            val_data = json.load(f)
        predefined_splits['val'] = val_data.get(model_key, [])

        with open(test_file, 'r') as f:
            test_data = json.load(f)
        predefined_splits['test'] = test_data.get(model_key, [])

        print(f"Successfully loaded predefined splits:")
        print(f"  Training set: {len(predefined_splits['train'])} structures")
        print(f"  Validation set: {len(predefined_splits['val'])} structures")
        print(f"  Test set: {len(predefined_splits['test'])} structures")

    except Exception as e:
        print(f"Error loading splits files: {e}")
        raise

    return predefined_splits


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
    main()