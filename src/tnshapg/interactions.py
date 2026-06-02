"""
Interactions: Order-2 (O2) Shapley interaction indices.

Extends the diagonal derivative method to compute pairwise interactions.
For pairs (u, v), the O2 interaction is:

    φ_{uv} = ∫_0^1 (∂²ν̂/∂z_u∂z_v)(t·1) dt

This is computed by placing derivative vectors at BOTH nodes u and v,
and Bernoulli vectors at all other nodes.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .graph_aligned_tn import GraphAlignedTN, bernoulli_vector, derivative_vector
from .diagonal_shapley import chebyshev_nodes_01, shapley_weights, stable_polyfit, integrate_polynomial


def compute_o2_interactions(
    tn: GraphAlignedTN,
    n_nodes: Optional[int] = None,
    edge_pairs: Optional[List[Tuple[int, int]]] = None,
    m: int = 16,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute O2 Shapley interaction indices.
    
    For each pair (u, v), computes:
        φ_{uv} = ∫_0^1 g_{uv}(t) dt
    
    where g_{uv}(t) = ∂²ν̂/∂z_u∂z_v evaluated at z = t·1.
    
    The second derivative is computed by TN contraction with:
        - Derivative vectors b'(t) = [-1, 1] at nodes u and v
        - Bernoulli vectors b(t) = [1-t, t] at all other nodes
    
    Args:
        tn: Trained GraphAlignedTN surrogate
        n_nodes: Number of nodes (inferred from tn if None)
        edge_pairs: List of (u, v) pairs to compute. If None, computes all pairs.
                   For graphs, typically pass graph edges only for efficiency.
        m: Number of Chebyshev interpolation nodes
        device: Torch device
        verbose: Print progress
        
    Returns:
        O2 interaction matrix [n_nodes, n_nodes] (symmetric, zero diagonal)
        Only pairs in edge_pairs are computed (if specified).
    """
    if n_nodes is None:
        n_nodes = tn.n_nodes
    
    if device is None:
        device = next(tn.parameters()).device
    
    # Get Chebyshev nodes and integration weights
    nodes = chebyshev_nodes_01(m)
    weights = shapley_weights(m)
    
    y_mean = getattr(tn, 'y_mean', 0.0)
    y_std = getattr(tn, 'y_std', 1.0)
    
    # Initialize interaction matrix
    phi2 = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    
    # Determine which pairs to compute
    if edge_pairs is None:
        # All pairs
        pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    else:
        # Normalize to (i, j) with i < j
        pairs = []
        for u, v in edge_pairs:
            if u > v:
                u, v = v, u
            pairs.append((u, v))
        pairs = list(set(pairs))  # Remove duplicates
    
    if verbose:
        print(f"Computing O2 interactions for {len(pairs)} pairs...")
    
    iterator = pairs
    if verbose:
        iterator = tqdm(pairs, desc="Computing O2 interactions")
    
    for u, v in iterator:
        # Evaluate g_{uv}(t_j) at each Chebyshev node
        g_values = np.zeros(m, dtype=np.float64)
        
        for j, t in enumerate(nodes):
            # Build vectors: derivative at u and v, Bernoulli at others
            vectors = []
            for w in range(n_nodes):
                if w == u or w == v:
                    vectors.append(derivative_vector(device))
                else:
                    vectors.append(bernoulli_vector(t, device))
            
            # Contract TN
            with torch.no_grad():
                g_val = tn.contract_with_vectors(vectors)
                g_values[j] = g_val.item()
        
        # Denormalize (second derivative scales by y_std)
        g_values = g_values * y_std
        
        # Integrate using weights
        phi2[u, v] = float(np.dot(weights, g_values))
        phi2[v, u] = phi2[u, v]  # Symmetric
    
    return phi2


def compute_o2_exact(
    game_value_fn,
    n_nodes: int,
    edge_pairs: Optional[List[Tuple[int, int]]] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute exact O2 interactions by enumeration (for small n).
    
    The exact O2 Shapley interaction is:
        φ_{ij} = Σ_{S⊆N\{i,j}} c(|S|) · Δ²v(S; i, j)
    
    where:
        c(s) = s! (n-s-2)! / (n-1)!
        Δ²v(S; i, j) = v(S∪{i,j}) - v(S∪{i}) - v(S∪{j}) + v(S)
    
    Args:
        game_value_fn: Function mapping coalition set -> value
        n_nodes: Number of nodes
        edge_pairs: List of pairs to compute (if None, all pairs)
        verbose: Print progress
        
    Returns:
        O2 interaction matrix [n_nodes, n_nodes]
    """
    from math import factorial
    from itertools import combinations
    
    phi2 = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    
    # Determine pairs
    if edge_pairs is None:
        pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    else:
        pairs = [(min(u, v), max(u, v)) for u, v in edge_pairs]
        pairs = list(set(pairs))
    
    n = n_nodes
    
    iterator = pairs
    if verbose:
        iterator = tqdm(pairs, desc="Exact O2 interactions")
    
    for i, j in iterator:
        # Sum over all coalitions not containing i, j
        others = [k for k in range(n) if k != i and k != j]
        
        phi_ij = 0.0
        
        for s in range(len(others) + 1):
            # Coefficient: s! (n-s-2)! / (n-1)!
            coeff = factorial(s) * factorial(n - s - 2) / factorial(n - 1)
            
            for S_tuple in combinations(others, s):
                S = set(S_tuple)
                
                # Discrete second derivative
                v_ij = game_value_fn(S | {i, j})
                v_i = game_value_fn(S | {i})
                v_j = game_value_fn(S | {j})
                v_0 = game_value_fn(S)
                
                delta2 = v_ij - v_i - v_j + v_0
                phi_ij += coeff * delta2
        
        phi2[i, j] = phi_ij
        phi2[j, i] = phi_ij
    
    return phi2


def interaction_matrix_to_dict(
    phi2: np.ndarray,
    edge_pairs: Optional[List[Tuple[int, int]]] = None,
) -> dict:
    """
    Convert interaction matrix to dictionary format.
    
    Args:
        phi2: Interaction matrix [n, n]
        edge_pairs: If provided, only include these pairs
        
    Returns:
        Dictionary mapping "i,j" -> value
    """
    n = phi2.shape[0]
    result = {}
    
    if edge_pairs is not None:
        pairs = [(min(u, v), max(u, v)) for u, v in edge_pairs]
        pairs = sorted(set(pairs))
    else:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for i, j in pairs:
        key = f"{i},{j}"
        result[key] = float(phi2[i, j])
    
    return result


def dict_to_interaction_matrix(
    d: dict,
    n_nodes: int,
) -> np.ndarray:
    """
    Convert dictionary format back to matrix.
    
    Args:
        d: Dictionary mapping "i,j" -> value
        n_nodes: Number of nodes
        
    Returns:
        Interaction matrix [n_nodes, n_nodes]
    """
    phi2 = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    
    for key, val in d.items():
        i, j = map(int, key.split(","))
        phi2[i, j] = val
        phi2[j, i] = val
    
    return phi2
