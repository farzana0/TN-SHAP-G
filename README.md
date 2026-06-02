# TN-SHAP-G: Graph-Structured Tensor Network Surrogates for Shapley Values and Interactions

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **TN-SHAP-G** from the paper:

> **TN-SHAP-G: Graph-Structured Tensor Network Surrogates for Shapley Values and Interactions**
> Farzaneh Heidari, Guillaume Rabusseau.
> arXiv:2606.01540 — <https://arxiv.org/abs/2606.01540>

## Overview

TN-SHAP-G computes **deterministic Shapley values** and **interaction indices** for graph-structured data by:

1. **Training a tensor network surrogate** whose topology mirrors the input graph
2. **Computing Shapley values** via the **diagonal derivative integral** using Vandermonde interpolation
3. **Computing O2 interactions** by extending the derivative method to mixed partials

### Key Features

- ✅ **Deterministic**: No Monte Carlo sampling needed for Shapley computation
- ✅ **Graph-aligned**: TN structure respects graph topology
- ✅ **Efficient**: O(n·m) TN contractions for n nodes and m interpolation points
- ✅ **Interactions**: Supports O2 (pairwise) Shapley interaction indices

## Method Overview

### Masked Graph Game

For a graph instance (G, X) with baseline X₀, a coalition S defines a masked input:
```
X_S[i] = X[i]   if i ∈ S
X_S[i] = X₀[i]  if i ∉ S
```

The game value is `v(S) = f(G, X_S)` where f is the teacher model.

### Graph-Aligned Tensor Network

Each node u has a tensor core A^(u) with:
- Physical dimension 2 (coalition membership: excluded/included)
- Bond dimension χ for each incident edge

The TN computes: `ν̂(z) = contract(A^(1), ..., A^(n), z₁, ..., zₙ)`

### Diagonal Derivative Integral

Shapley values are computed as:
```
φ_u = ∫₀¹ (∂ν̂/∂z_u)(t·1) dt
```

where the derivative is evaluated using:
- **Bernoulli vectors** b(t) = [1-t, t] at nodes v ≠ u
- **Derivative vectors** b'(t) = [-1, 1] at node u

The integral is computed exactly via polynomial interpolation at Chebyshev nodes.

## Installation

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/farzana0/TN-SHAP-G.git
cd TN-SHAP-G

# Install in development mode
pip install -e .

# Or with all optional dependencies
pip install -e ".[all]"
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 1.10
- NumPy, SciPy, NetworkX
- (Optional) opt_einsum for optimized contractions

## Quickstart

### One-Command Demo

Run the complete toy experiment in ~3-5 minutes:

```bash
cd TN-SHAP-G
python scripts/run_toy_experiment.py
```

This will:
1. Create a synthetic graph (n=12 nodes)
2. Train a TN surrogate from 500 coalition samples
3. Compute deterministic Shapley values
4. Compare against exact enumeration
5. Report cosine similarity (expected: >0.95)

### Python API

```python
from tnshapg import (
    GraphAlignedTN,
    MaskedGame,
    MLPTeacher,
    train_surrogate,
    compute_diagonal_shapley,
    compute_o2_interactions,
)
import torch

# 1. Setup: graph, features, teacher model
n_nodes = 10
X = torch.randn(n_nodes, 7)
baseline = torch.zeros_like(X)
teacher = MLPTeacher(n_features=7, hidden_dims=[32, 16])

# 2. Create masked game
game = MaskedGame(X=X, baseline=baseline, teacher=teacher)

# 3. Create graph-aligned TN (matches graph topology)
import networkx as nx
G = nx.random_tree(n_nodes)
tn = GraphAlignedTN(n_nodes, graph_structure=G, bond_dim=4)

# 4. Train surrogate
result = train_surrogate(tn, game, config=TrainingConfig(n_train_samples=500))
print(f"Test R²: {result.final_val_r2:.4f}")

# 5. Compute Shapley values (deterministic!)
phi = compute_diagonal_shapley(tn, n_nodes, m=16)
print(f"Shapley values: {phi}")

# 6. Compute O2 interactions on edges
edge_pairs = list(G.edges())
phi2 = compute_o2_interactions(tn, n_nodes, edge_pairs=edge_pairs)
```

### Jupyter Notebook

See `notebooks/00_quickstart_tnshapg.ipynb` for an interactive walkthrough.

## Project Structure

```
TN-SHAP-G/
├── src/tnshapg/
│   ├── __init__.py              # Public API exports
│   ├── masked_game.py           # Coalition masking & teacher models
│   ├── coalition_sampler.py     # Balanced coalition sampling
│   ├── graph_aligned_tn.py      # Graph-aligned tensor network
│   ├── training.py              # Training loop with early stopping
│   ├── diagonal_shapley.py      # Deterministic Shapley computation
│   ├── interactions.py          # O2 Shapley interactions
│   ├── exact_shapley.py         # Exact enumeration (sanity check)
│   └── utils.py                 # Utility functions
├── scripts/
│   ├── run_toy_experiment.py    # One-command reproducibility
│   ├── train_surrogate.py       # Train TN surrogate CLI
│   ├── compute_shapley.py       # Compute Shapley values CLI
│   └── compute_interactions.py  # Compute O2 interactions CLI
├── notebooks/
│   └── 00_quickstart_tnshapg.ipynb
├── configs/
│   └── toy_demo.yaml
├── tests/
│   └── test_tnshapg.py
├── pyproject.toml
├── LICENSE
└── README.md
```

## Reproducing Paper Results

### Toy Experiment

```bash
python scripts/run_toy_experiment.py
```

Expected output:
- Surrogate test R² > 0.95
- O1 Shapley cosine similarity > 0.98
- O2 interaction cosine similarity > 0.95

### Custom Dataset

```bash
# Train surrogate
python scripts/train_surrogate.py \
    --n_nodes 20 \
    --bond_dim 8 \
    --n_samples 1000 \
    --output_dir outputs/custom

# Compute Shapley
python scripts/compute_shapley.py \
    --model outputs/custom/tn_surrogate.pt \
    --m 20

# Compute O2 interactions
python scripts/compute_interactions.py \
    --model outputs/custom/tn_surrogate.pt \
    --edges_only
```

## Runtime Estimates

| Graph Size | Training | Shapley (O1) | Interactions (O2, edges) |
|------------|----------|--------------|--------------------------|
| n=10       | ~30s     | ~2s          | ~5s                      |
| n=20       | ~60s     | ~5s          | ~30s                     |
| n=50       | ~3min    | ~30s         | ~2min                    |

Runtimes on CPU. GPU provides 2-5x speedup for training.

## Adding Custom Teacher Models

Implement a callable that takes masked features and returns a scalar:

```python
class MyTeacher(torch.nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Your model here
    
    def forward(self, X: torch.Tensor, edge_index=None) -> torch.Tensor:
        """
        Args:
            X: Node features [n_nodes, n_features]
            edge_index: Optional edge connectivity [2, num_edges]
        Returns:
            Scalar prediction
        """
        # Your computation here
        return prediction

# Use with MaskedGame
teacher = MyTeacher(...)
game = MaskedGame(X=X, baseline=baseline, teacher=teacher, edge_index=edge_index)
```

## Tests

Run unit tests:

```bash
cd TN-SHAP-G
pip install -e ".[dev]"
pytest tests/ -v
```

## Citation

If you use TN-SHAP-G in your research, please cite:

> **TN-SHAP-G: Graph-Structured Tensor Network Surrogates for Shapley Values and Interactions**
> Farzaneh Heidari, Guillaume Rabusseau.
> arXiv:2606.01540 — <https://arxiv.org/abs/2606.01540>

```bibtex
@article{heidari2026tnshapg,
  title         = {TN-SHAP-G: Graph-Structured Tensor Network Surrogates for Shapley Values and Interactions},
  author        = {Heidari, Farzaneh and Rabusseau, Guillaume},
  journal       = {arXiv preprint arXiv:2606.01540},
  year          = {2026},
  eprint        = {2606.01540},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2606.01540}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This work builds on concepts from:
- Shapley value theory in cooperative game theory
- Tensor network methods from quantum physics
- Graph neural network explainability research
