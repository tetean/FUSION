import numpy as np
import json
import os, sys
from datetime import datetime
from tqdm import tqdm
from FUSION import FUSION


class EpsilonSearcher:
    """
    Fast epsilon value searcher for FUSION.
    Uses pre-computed influence scores to quickly determine pruning ratios
    for different epsilon values without re-running the optimization.
    """

    def __init__(self, FUSION_instance):
        """
        Initialize with a FUSION instance that has computed influence scores.

        Args:
            FUSION_instance: FUSION instance with computed influence_scores
        """
        self.FUSION = FUSION_instance
        if FUSION_instance.influence_scores is None:
            raise ValueError("FUSION instance must have computed influence scores first")

        self.influence_scores = FUSION_instance.influence_scores.copy()
        self.n_structures = len(self.influence_scores)

        # Sort influence scores for efficient searching
        self.sorted_indices = np.argsort(self.influence_scores)
        self.sorted_scores = self.influence_scores[self.sorted_indices]

        # Pre-compute cumulative statistics for fast lookup
        self._precompute_statistics()

    def _precompute_statistics(self):
        """Pre-compute cumulative statistics for fast epsilon-to-ratio mapping."""
        print("Pre-computing statistics for fast epsilon search...")

        # Cumulative sum and mean for each possible subset
        self.cumulative_sums = np.cumsum(self.sorted_scores)
        self.cumulative_means = self.cumulative_sums / np.arange(1, self.n_structures + 1)

        # For each position, store the maximum number of structures that can be removed
        # while maintaining average influence <= epsilon
        self.max_removable = np.zeros(self.n_structures)

        for i in range(self.n_structures):
            # Find maximum k such that cumulative_means[k-1] <= sorted_scores[i]
            epsilon_val = self.sorted_scores[i]
            valid_positions = np.where(self.cumulative_means <= epsilon_val)[0]
            if len(valid_positions) > 0:
                self.max_removable[i] = valid_positions[-1] + 1
            else:
                self.max_removable[i] = 0

        print("Statistics pre-computation completed.")

    def epsilon_to_pruning_ratio(self, epsilon):
        """
        Quickly convert epsilon value to expected pruning ratio.

        Args:
            epsilon (float): Epsilon threshold value

        Returns:
            float: Expected pruning ratio (0-1)
        """
        if epsilon <= 0:
            return 0.0

        # Find the maximum number of structures that can be removed
        # while keeping average influence <= epsilon
        valid_positions = np.where(self.cumulative_means <= epsilon)[0]

        if len(valid_positions) > 0:
            max_removable = valid_positions[-1] + 1
            return min(max_removable / self.n_structures, 1.0)
        else:
            return 0.0

    def pruning_ratio_to_epsilon(self, target_ratio):
        """
        Find epsilon value that achieves approximately the target pruning ratio.

        Args:
            target_ratio (float): Target pruning ratio (0-1)

        Returns:
            float: Epsilon value that achieves the target ratio
        """
        if target_ratio <= 0:
            return 0.0
        if target_ratio >= 1:
            return float('inf')

        target_count = int(target_ratio * self.n_structures)
        target_count = min(target_count, self.n_structures - 1)

        if target_count <= 0:
            return 0.0

        # The epsilon should be the cumulative mean at the target position
        return self.cumulative_means[target_count - 1]

    def search_epsilon_range(self, min_ratio=0.0, max_ratio=0.95, num_points=100):
        """
        Search for epsilon values in a range of pruning ratios.

        Args:
            min_ratio (float): Minimum pruning ratio
            max_ratio (float): Maximum pruning ratio  
            num_points (int): Number of points to sample

        Returns:
            dict: Mapping from pruning ratios to epsilon values
        """
        ratios = np.linspace(min_ratio, max_ratio, num_points)
        epsilon_mapping = {}

        for ratio in tqdm(ratios, desc="Searching epsilon range"):
            epsilon = self.pruning_ratio_to_epsilon(ratio)
            epsilon_mapping[ratio] = epsilon

        return epsilon_mapping

    def binary_search_epsilon(self, target_ratio, tolerance=0.01):
        """
        Use binary search to find epsilon for a specific target pruning ratio.

        Args:
            target_ratio (float): Target pruning ratio
            tolerance (float): Acceptable tolerance for the ratio

        Returns:
            float: Epsilon value
        """
        if target_ratio <= 0:
            return 0.0
        if target_ratio >= 1:
            return float('inf')

        # Binary search bounds
        low_epsilon = 0.0
        high_epsilon = self.sorted_scores.max() * 2

        best_epsilon = 0.0
        best_diff = float('inf')

        for _ in range(50):  # Max iterations
            mid_epsilon = (low_epsilon + high_epsilon) / 2
            achieved_ratio = self.epsilon_to_pruning_ratio(mid_epsilon)

            diff = abs(achieved_ratio - target_ratio)
            if diff < best_diff:
                best_diff = diff
                best_epsilon = mid_epsilon

            if diff < tolerance:
                break

            if achieved_ratio < target_ratio:
                low_epsilon = mid_epsilon
            else:
                high_epsilon = mid_epsilon

        return best_epsilon

    def generate_epsilon_schedule(self, num_models, min_ratio=0.0, max_ratio=0.9,
                                  distribution='linear'):
        """
        Generate a schedule of epsilon values for multiple models.

        Args:
            num_models (int): Number of models to generate
            min_ratio (float): Minimum pruning ratio
            max_ratio (float): Maximum pruning ratio
            distribution (str): 'linear', 'log', or 'quadratic'

        Returns:
            dict: Model number to epsilon mapping
        """
        epsilon_schedule = {}

        if distribution == 'linear':
            ratios = np.linspace(min_ratio, max_ratio, num_models)
        elif distribution == 'log':
            # Logarithmic spacing in ratio space
            log_min = np.log10(max(min_ratio, 1e-6))
            log_max = np.log10(max_ratio)
            ratios = np.logspace(log_min, log_max, num_models)
        elif distribution == 'quadratic':
            # Quadratic spacing - more dense at lower ratios
            x = np.linspace(0, 1, num_models)
            ratios = min_ratio + (max_ratio - min_ratio) * x ** 2
        else:
            raise ValueError("Distribution must be 'linear', 'log', or 'quadratic'")

        for i, ratio in enumerate(ratios):
            epsilon = self.pruning_ratio_to_epsilon(ratio)
            epsilon_schedule[i] = epsilon

        return epsilon_schedule

    def optimize_epsilon_for_target_ratios(self, target_ratios):
        """
        Optimize epsilon values for specific target pruning ratios.

        Args:
            target_ratios (list): List of target pruning ratios

        Returns:
            dict: Mapping from target ratio to optimal epsilon
        """
        epsilon_mapping = {}

        for ratio in tqdm(target_ratios, desc="Optimizing epsilon values"):
            epsilon = self.binary_search_epsilon(ratio)
            epsilon_mapping[ratio] = epsilon

        return epsilon_mapping

    def save_epsilon_mapping(self, epsilon_mapping, output_dir, filename_prefix="epsilon_search"):
        """
        Save epsilon mapping results to files.

        Args:
            epsilon_mapping (dict): Epsilon mapping to save
            output_dir (str): Output directory
            filename_prefix (str): Prefix for filenames
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        json_path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.json")
        json_serializable = {str(k): float(v) for k, v in epsilon_mapping.items()}

        with open(json_path, 'w') as f:
            json.dump(json_serializable, f, indent=2)
        print(f"Epsilon mapping saved to {json_path}")

        # Save as Python dict
        py_path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.py")
        with open(py_path, 'w') as f:
            f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("epsilon_values = {\n")
            for k, v in epsilon_mapping.items():
                f.write(f"    {k}: {v:.6e},\n")
            f.write("}\n")
        print(f"Python dict saved to {py_path}")

    def quick_analysis(self):
        """
        Perform quick analysis of the influence score distribution.

        Returns:
            dict: Analysis results
        """
        results = {
            'total_structures': self.n_structures,
            'influence_stats': {
                'min': float(self.influence_scores.min()),
                'max': float(self.influence_scores.max()),
                'mean': float(self.influence_scores.mean()),
                'std': float(self.influence_scores.std()),
                'median': float(np.median(self.influence_scores)),
                'q25': float(np.percentile(self.influence_scores, 25)),
                'q75': float(np.percentile(self.influence_scores, 75))
            }
        }

        # Sample some key pruning ratios and their epsilon values
        key_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        epsilon_for_ratios = {}

        for ratio in key_ratios:
            epsilon = self.pruning_ratio_to_epsilon(ratio)
            epsilon_for_ratios[ratio] = epsilon

        results['epsilon_for_key_ratios'] = epsilon_for_ratios

        return results


