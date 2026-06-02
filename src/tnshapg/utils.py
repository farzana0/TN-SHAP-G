"""
Utility functions for TN-SHAP-G.

Provides common helpers for random seed management, metrics, and data handling.
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity in [-1, 1]
    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))


def max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute maximum absolute error between two arrays.
    
    Args:
        a: First array
        b: Second array
        
    Returns:
        Maximum absolute difference
    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    return float(np.max(np.abs(a - b)))


def mean_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute mean absolute error between two arrays.
    
    Args:
        a: First array
        b: Second array
        
    Returns:
        Mean absolute difference
    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    return float(np.mean(np.abs(a - b)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination) score.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        R² score (1.0 is perfect, can be negative for bad fits)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot < 1e-10:
        return 1.0 if ss_res < 1e-10 else 0.0
    
    return float(1.0 - ss_res / ss_tot)


def augment_with_ones(X: torch.Tensor) -> torch.Tensor:
    """
    Augment feature tensor with trailing 1s for affine transformation.
    
    This allows TN to learn affine functions by having a constant
    feature that can represent bias terms.
    
    Args:
        X: Feature tensor of shape [..., n, d]
        
    Returns:
        Augmented tensor of shape [..., n, d+1] with 1s appended
    """
    ones = torch.ones(*X.shape[:-1], 1, device=X.device, dtype=X.dtype)
    return torch.cat([X, ones], dim=-1)


def edge_index_to_networkx(edge_index: torch.Tensor, n_nodes: int):
    """
    Convert PyG edge_index to NetworkX graph.
    
    Args:
        edge_index: [2, num_edges] tensor
        n_nodes: Number of nodes
        
    Returns:
        NetworkX Graph
    """
    import networkx as nx
    
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    
    edge_index = edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        if u < v:  # Avoid adding edges twice
            G.add_edge(u, v)
    
    return G


def networkx_to_edge_index(G) -> torch.Tensor:
    """
    Convert NetworkX graph to PyG edge_index.
    
    Args:
        G: NetworkX graph
        
    Returns:
        edge_index: [2, 2*num_edges] tensor (both directions)
    """
    edges = list(G.edges())
    if len(edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    
    # Add both directions
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    
    return torch.tensor([src, dst], dtype=torch.long)


def get_edge_pairs_from_graph(G) -> list:
    """
    Extract unique edge pairs from a NetworkX graph.
    
    Args:
        G: NetworkX graph
        
    Returns:
        List of (i, j) tuples where i < j
    """
    edges = []
    for u, v in G.edges():
        if u > v:
            u, v = v, u
        edges.append((u, v))
    return sorted(set(edges))
