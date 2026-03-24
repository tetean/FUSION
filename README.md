<h1>
<p align="center">
    <img src="properties/FUSION.png" alt="FUSION logo" width="800"/>
</p>
</h1>

<h4 align="center">

[![Static Badge](https://img.shields.io/badge/PYTHON-3.9%2B-gray?style=for-the-badge&labelColor=blue)](https://python.org/downloads) 
&nbsp;&nbsp;&nbsp;&nbsp;
[![Static Badge](https://img.shields.io/badge/2026-AAAI?style=for-the-badge&label=AAAI&labelColor=purple&color=gray)](https://aaai.org/conference/aaai/aaai-26/)
&nbsp;&nbsp;&nbsp;&nbsp;
[![Static Badge](https://img.shields.io/badge/PAPER-red?style=for-the-badge&labelColor=blue)](https://ojs.aaai.org/index.php/AAAI/article/view/39863)

</h4>

---
# FUSION

## Installation

1. Clone the Repository:
```bash
git clone https://github.com/tetean/FUSION.git
cd FUSION
```
2.  Install Dependencies:

### Some Important Dependencies
| Package | Description |
|---------|-------------|
| **PyTorch** | Deep learning framework |
| **DGL** | Deep Graph Library for GNNs |
| **pymatgen** | Crystal structure manipulation |
| **jarvis-tools** | Materials science toolkit |
| **ase** | Atomic Simulation Environment |
| **dscribe** | SOAP descriptor computation |
| **spglib** | Symmetry analysis |
| **numpy** | Scientific computing |
| **scipy** | Scientific computing |
| **scikit-learn** | Machine learning utilities |
| **pandas** | Data processing |
| **tqdm** | Progress bars |
| **pyyaml** | Configuration management |


```bash
conda env create -f environment.yml -n FUSION
conda activate FUSION
```

## Quick Start

### Basic Usage

```python
from FUSION import FUSION

# Initialize FUSION
fusion = FUSION(
    config_path="config.yml",
    checkpoint_path="surrogate/perovskites/ALIGNN/example.pth.tar",
    data_path="./data",
    task="perovskites",
    epsilon=0.499,
    gamma=0.1,
    Lambda=0,
    cache_dir="./fusion_cache"
)

# Run FUSION pipeline
results = fusion.run(
    optimization_method='greedy',
    force_recompute=False
)

# Access results
pruning_mask = results['pruning_mask']
pruning_ratio = results['pruning_ratio']
print(f"Pruning ratio: {pruning_ratio:.2%}")
```

### Generate Train/Val/Test Splits

```python
from EpsilonSearcher import EpsilonSearcher
from FUSION import FUSION

# Initialize FUSION
fusion = FUSION(
    config_path="config.yml",
    checkpoint_path="surrogate/perovskites/ALIGNN/example.pth.tar",
    data_path="./data",
    task="perovskites",
    epsilon=0.01,
    gamma=0.1,
    Lambda=0,
    cache_dir="./fusion_cache"
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

# Find optimal epsilon values for target pruning ratios
searcher = EpsilonSearcher(fusion)
target_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]
optimal_epsilons = searcher.optimize_epsilon_for_target_ratios(target_ratios)

# Generate JSON splits
epsilon_values = {i: eps for i, eps in enumerate(optimal_epsilons.values())}
json_docs = fusion.generate_json_splits(
    epsilon_values=epsilon_values,
    output_dir="./fusion_output",
    test_size=0.15,
    val_size=0.15,
    random_state=42,
    optimization_method='greedy'
)
```
> It should be noted that the Epsilon Searcher algorithm provides exact solutions for the greedy optimization strategy, where samples are selected strictly in ascending order of influence scores. However, when employing FUSION's advanced optimization methods (dynamic greedy, beam search, or enhanced simulated annealing), the algorithm serves as an approximation tool. For critical applications requiring precise pruning ratios, we recommend using the epsilon searcher as an initial estimation.

## Data Structure
```python
data/
└── perovskites/
    ├── 0.cif
    ├── 1.cif
    ├── 2.cif
    ├── 3.cif
    ├── 4.cif
    ├── 5.cif
    ├── 6.cif
    ├── ...
    └── 18927.cif
```

Meanwhile, for fast reading and to save storage space, we have provided a preprocessed Perovskites data `pkl` file that is packaged and ready for use by ALIGNN as a demonstration:
```python
data/
└── perovskites/
    └── alignn_data.pkl
```

## JSON File Structure

FUSION generates three JSON files (`train.json`, `val.json`, `test.json`), each containing multiple models with different pruning levels. **Importantly, the test set remains identical across all models to ensure fair evaluation.**

### train.json
```python
{
  "0": [0, 1, 2, 3, 5, 8, 12, 15, ..., 18920],      // Model 0: Full training set
  "1": [0, 5, 8, 15, 23, 45, ..., 18915],           // Model 1: ~20% pruned
  "2": [0, 8, 23, 67, 89, ..., 18900],              // Model 2: ~40% pruned
  "3": [5, 23, 89, 145, ..., 18850],                // Model 3: ~60% pruned
  "4": [8, 67, 145, 234, ..., 18800]                // Model 4: ~80% pruned
}
```

### val.json
```python
{
  "0": [4, 6, 7, 9, 11, 16, 20, ..., 18925],        // Model 0: Full validation set
  "1": [4, 7, 11, 20, 28, ..., 18922],              // Model 1: ~20% pruned
  "2": [4, 11, 28, 56, ..., 18918],                 // Model 2: ~40% pruned
  "3": [7, 28, 78, ..., 18910],                     // Model 3: ~60% pruned
  "4": [11, 56, 123, ..., 18905]                    // Model 4: ~80% pruned
}
```

### test.json
```python
{
  "0": [10, 13, 14, 17, 19, 21, ..., 18927],        // Model 0: Full test set
  "1": [10, 13, 14, 17, 19, 21, ..., 18927],        // Model 1: IDENTICAL to Model 0
  "2": [10, 13, 14, 17, 19, 21, ..., 18927],        // Model 2: IDENTICAL to Model 0
  "3": [10, 13, 14, 17, 19, 21, ..., 18927],        // Model 3: IDENTICAL to Model 0
  "4": [10, 13, 14, 17, 19, 21, ..., 18927]         // Model 4: IDENTICAL to Model 0
}
```

## Poster
<p align="center">
    <img src="properties/FUSION-poster.png" alt="FUSION poster" width="800"/>
</p>

## Citation

If you use FUSION in your research, please cite:

```
@inproceedings{wang2026fusion,
  title={FUSION: Dataset Pruning via Fusing Uncertainty with Structural Information for Optimal Neural Training in Crystal Property Prediction},
  author={Wang, Xiean and Chen, Pin and Tan, Liqin and Lu, Yutong and Zou, Qingsong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={31},
  pages={26553--26561},
  year={2026}
}
```

---
