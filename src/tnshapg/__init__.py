"""
TN-SHAP-G: Graph-Structured Tensor Network Surrogates for Shapley Values and Interactions.

A publication-quality implementation for computing deterministic Shapley values
and interaction indices using graph-aligned tensor network surrogates.

Main components:
- masked_game: Coalition masking and teacher model queries
- coalition_sampler: Balanced sampling of coalitions for training
- graph_aligned_tn: Graph-aligned tensor network surrogate
- training: Training loop for TN surrogate distillation
- diagonal_shapley: Deterministic Shapley values via Vandermonde interpolation
- interactions: Order-2 Shapley interaction indices
- exact_shapley: Exact enumeration for sanity checks

Example usage:
    from tnshapg import MaskedGame, GraphAlignedTN, train_surrogate
    from tnshapg import compute_diagonal_shapley, compute_o2_interactions
    
    # Create masked game from graph and teacher
    game = MaskedGame(X, edge_index, baseline_X, teacher_model)
    
    # Train TN surrogate
    tn = GraphAlignedTN(n_nodes, graph_structure, bond_dim=4)
    train_surrogate(tn, game, n_samples=500, epochs=100)
    
    # Compute Shapley values deterministically
    shapley_values = compute_diagonal_shapley(tn, n_nodes)
    
    # Compute O2 interactions on edges
    o2_interactions = compute_o2_interactions(tn, n_nodes, edge_pairs)
"""

__version__ = "0.1.0"

# Core components
from .masked_game import MaskedGame, SimpleTeacher, MLPTeacher
from .coalition_sampler import (
    CoalitionSampler,
    BalancedSizeSampler,
    sample_binary_coalitions,
)
from .graph_aligned_tn import (
    GraphAlignedTN,
    bernoulli_vector,
    derivative_vector,
)
from .training import (
    train_surrogate,
    TrainingConfig,
)
from .diagonal_shapley import (
    compute_diagonal_shapley,
    chebyshev_nodes_01,
    vandermonde_matrix,
    stable_polyfit,
    shapley_weights,
)
from .interactions import (
    compute_o2_interactions,
    compute_o2_exact,
)
from .exact_shapley import (
    exact_shapley_enumeration,
    exact_shapley_fast,
    efficiency_check,
)
from .utils import set_seed, cosine_similarity, max_abs_error

__all__ = [
    # Version
    "__version__",
    # Masked game
    "MaskedGame",
    "SimpleTeacher",
    "MLPTeacher",
    # Sampling
    "CoalitionSampler",
    "BalancedSizeSampler",
    "sample_binary_coalitions",
    # Tensor network
    "GraphAlignedTN",
    "bernoulli_vector",
    "derivative_vector",
    # Training
    "train_surrogate",
    "TrainingConfig",
    # Shapley
    "compute_diagonal_shapley",
    "chebyshev_nodes_01",
    "vandermonde_matrix",
    "stable_polyfit",
    "shapley_weights",
    # Interactions
    "compute_o2_interactions",
    "compute_o2_exact",
    # Exact enumeration
    "exact_shapley_enumeration",
    "exact_shapley_fast",
    "efficiency_check",
    # Utils
    "set_seed",
    "cosine_similarity",
    "max_abs_error",
]
