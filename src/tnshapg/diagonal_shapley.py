"""
Diagonal Shapley: Deterministic Shapley values via Vandermonde interpolation.

This module implements the diagonal derivative integral method for computing
Shapley values deterministically from a trained TN surrogate.

Key insight: For multilinear functions, the Shapley value is:
    φ_u = ∫_0^1 (∂ν̂/∂z_u)(t·1) dt

where (t·1) means evaluating at the diagonal point z = (t, t, ..., t).

The derivative ∂ν̂/∂z_u is computed by TN contraction with:
    - Bernoulli vectors b(t) = [1-t, t] at all nodes except u
    - Derivative vector b'(t) = [-1, 1] at node u

The integral is evaluated using polynomial interpolation:
    1. Evaluate g_u(t_j) at Chebyshev nodes {t_j}
    2. Fit polynomial of degree ≤ n-1 through the points
    3. Integrate the polynomial exactly on [0, 1]

This gives exact Shapley values (up to TN approximation error) in O(n·m)
TN contractions, where m is the number of interpolation nodes.
"""

from typing import Optional, Tuple

import numpy as np
import torch

from .graph_aligned_tn import GraphAlignedTN, bernoulli_vector, derivative_vector


def chebyshev_nodes_01(m: int) -> np.ndarray:
    """
    Compute m Chebyshev nodes mapped to [0, 1].
    
    Standard Chebyshev nodes on [-1, 1]:
        t_j = cos((2j - 1)π / (2m)),  j = 1, ..., m
    
    Mapped to [0, 1]:
        x_j = (1 + t_j) / 2
    
    Args:
        m: Number of Chebyshev nodes
        
    Returns:
        Sorted array of m nodes in [0, 1]
    """
    j = np.arange(1, m + 1)
    t = np.cos((2 * j - 1) * np.pi / (2 * m))
    x = (1 + t) / 2
    return np.sort(x).astype(np.float64)


def vandermonde_matrix(nodes: np.ndarray, degree: Optional[int] = None) -> np.ndarray:
    """
    Build Vandermonde matrix V[i, k] = nodes[i]^k.
    
    Args:
        nodes: Array of interpolation nodes
        degree: Number of polynomial terms (default: len(nodes))
        
    Returns:
        Vandermonde matrix [m, degree]
    """
    m = len(nodes)
    if degree is None:
        degree = m
    
    V = np.zeros((m, degree), dtype=np.float64)
    for k in range(degree):
        V[:, k] = nodes ** k
    
    return V


def stable_polyfit(
    nodes: np.ndarray,
    values: np.ndarray,
    degree: Optional[int] = None,
) -> np.ndarray:
    """
    Fit polynomial coefficients using stable least squares.
    
    For values y_j at nodes t_j, finds coefficients c_k such that
    p(t) = Σ_k c_k t^k ≈ y interpolates/fits the data.
    
    Args:
        nodes: Interpolation nodes [m]
        values: Function values at nodes [m]
        degree: Polynomial degree (default: m-1)
        
    Returns:
        Polynomial coefficients [degree+1] for p(t) = Σ c_k t^k
    """
    m = len(nodes)
    if degree is None:
        degree = m - 1
    
    V = vandermonde_matrix(nodes, degree + 1)
    
    # Use least squares for stability (handles m > degree or ill-conditioning)
    coeffs, residuals, rank, s = np.linalg.lstsq(V, values, rcond=None)
    
    return coeffs


def integrate_polynomial(coeffs: np.ndarray) -> float:
    """
    Integrate polynomial on [0, 1].
    
    For p(t) = Σ_k c_k t^k, computes:
        ∫_0^1 p(t) dt = Σ_k c_k / (k + 1)
    
    Args:
        coeffs: Polynomial coefficients [d+1]
        
    Returns:
        Integral value
    """
    k = np.arange(len(coeffs))
    return float(np.sum(coeffs / (k + 1)))


def shapley_weights(m: int) -> np.ndarray:
    """
    Compute Shapley integration weights for Chebyshev nodes.
    
    These weights w satisfy: w · f(nodes) ≈ ∫_0^1 f(t) dt
    for polynomials f of degree ≤ m-1.
    
    Uses: w = h @ V⁻¹ where h_k = 1/(k+1) and V is Vandermonde.
    
    Args:
        m: Number of Chebyshev nodes
        
    Returns:
        Integration weights [m]
    """
    nodes = chebyshev_nodes_01(m)
    V = vandermonde_matrix(nodes, m)
    
    # Integration weights for monomials: ∫_0^1 t^k dt = 1/(k+1)
    h = np.array([1.0 / (k + 1) for k in range(m)], dtype=np.float64)
    
    # Solve V^T @ w = h for weights
    # This gives w such that w · f(nodes) = ∫ f for polynomials
    w, _, _, _ = np.linalg.lstsq(V.T, h, rcond=None)
    
    return w


def compute_diagonal_shapley(
    tn: GraphAlignedTN,
    n_nodes: Optional[int] = None,
    m: int = 16,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute Shapley values using diagonal derivative integral.
    
    For each node u, computes:
        φ_u = ∫_0^1 g_u(t) dt
    
    where g_u(t) = ∂ν̂/∂z_u evaluated at z = t·1 (all nodes at t).
    
    The derivative is computed by TN contraction with:
        - Bernoulli vectors b(t) = [1-t, t] at nodes v ≠ u
        - Derivative vector b'(t) = [-1, 1] at node u
    
    Args:
        tn: Trained GraphAlignedTN surrogate
        n_nodes: Number of nodes (inferred from tn if None)
        m: Number of Chebyshev interpolation nodes
        device: Torch device
        verbose: Whether to print progress
        
    Returns:
        Shapley values [n_nodes]
    """
    if n_nodes is None:
        n_nodes = tn.n_nodes
    
    if device is None:
        device = next(tn.parameters()).device
    
    # Get Chebyshev nodes and integration weights
    nodes = chebyshev_nodes_01(m)
    weights = shapley_weights(m)
    
    # Apply denormalization from training
    y_mean = getattr(tn, 'y_mean', 0.0)
    y_std = getattr(tn, 'y_std', 1.0)
    
    phi = np.zeros(n_nodes, dtype=np.float64)
    
    # For each node u, evaluate derivative at all interpolation points
    iterator = range(n_nodes)
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Computing Shapley values")
    
    for u in iterator:
        # Evaluate g_u(t_j) at each Chebyshev node
        g_values = np.zeros(m, dtype=np.float64)
        
        for j, t in enumerate(nodes):
            # Build vectors: derivative at u, Bernoulli at others
            vectors = []
            for v in range(n_nodes):
                if v == u:
                    vectors.append(derivative_vector(device))
                else:
                    vectors.append(bernoulli_vector(t, device))
            
            # Contract TN
            with torch.no_grad():
                g_val = tn.contract_with_vectors(vectors)
                g_values[j] = g_val.item()
        
        # Denormalize derivative values (derivative wrt normalized output)
        # Since output is normalized by y_std, derivative scales by y_std
        g_values = g_values * y_std
        
        # Integrate using weights (or polynomial fit)
        phi[u] = float(np.dot(weights, g_values))
    
    return phi


def compute_diagonal_shapley_polynomial(
    tn: GraphAlignedTN,
    n_nodes: Optional[int] = None,
    m: int = 16,
    poly_degree: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Compute Shapley values using polynomial interpolation.
    
    This is an alternative to the weighted sum approach, using
    explicit polynomial fitting and integration.
    
    Args:
        tn: Trained GraphAlignedTN surrogate
        n_nodes: Number of nodes
        m: Number of interpolation nodes (use m > n for overdetermined)
        poly_degree: Polynomial degree (default: n-1)
        device: Torch device
        verbose: Print progress
        
    Returns:
        Tuple of (shapley_values [n], debug_info dict)
    """
    if n_nodes is None:
        n_nodes = tn.n_nodes
    
    if poly_degree is None:
        poly_degree = n_nodes - 1
    
    if device is None:
        device = next(tn.parameters()).device
    
    # Use more interpolation points for stability
    if m < n_nodes:
        m = max(n_nodes, 2 * poly_degree)
    
    nodes = chebyshev_nodes_01(m)
    
    y_mean = getattr(tn, 'y_mean', 0.0)
    y_std = getattr(tn, 'y_std', 1.0)
    
    phi = np.zeros(n_nodes, dtype=np.float64)
    debug_info = {"polynomials": {}}
    
    iterator = range(n_nodes)
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Computing Shapley (polynomial)")
    
    for u in iterator:
        # Evaluate g_u at interpolation nodes
        g_values = np.zeros(m, dtype=np.float64)
        
        for j, t in enumerate(nodes):
            vectors = []
            for v in range(n_nodes):
                if v == u:
                    vectors.append(derivative_vector(device))
                else:
                    vectors.append(bernoulli_vector(t, device))
            
            with torch.no_grad():
                g_val = tn.contract_with_vectors(vectors)
                g_values[j] = g_val.item() * y_std  # Denormalize
        
        # Fit polynomial
        coeffs = stable_polyfit(nodes, g_values, degree=poly_degree)
        
        # Integrate
        phi[u] = integrate_polynomial(coeffs)
        
        debug_info["polynomials"][u] = {
            "coeffs": coeffs.tolist(),
            "nodes": nodes.tolist(),
            "values": g_values.tolist(),
        }
    
    return phi, debug_info


def verify_efficiency(
    phi: np.ndarray,
    tn: GraphAlignedTN,
    device: Optional[torch.device] = None,
    tol: float = 1e-4,
) -> dict:
    """
    Verify Shapley efficiency axiom: Σ φ_i = ν̂(1) - ν̂(0).
    
    Args:
        phi: Computed Shapley values [n]
        tn: TN surrogate
        device: Torch device
        tol: Tolerance for efficiency check
        
    Returns:
        Dictionary with verification results
    """
    if device is None:
        device = next(tn.parameters()).device
    
    n = len(phi)
    
    y_mean = getattr(tn, 'y_mean', 0.0)
    y_std = getattr(tn, 'y_std', 1.0)
    
    # Evaluate at full coalition (all 1s) and empty (all 0s)
    z_full = torch.ones(n, device=device)
    z_empty = torch.zeros(n, device=device)
    
    with torch.no_grad():
        v_full = tn.forward(z_full).item() * y_std + y_mean
        v_empty = tn.forward(z_empty).item() * y_std + y_mean
    
    expected_sum = v_full - v_empty
    actual_sum = np.sum(phi)
    diff = abs(actual_sum - expected_sum)
    
    passed = diff <= tol
    
    return {
        "passed": passed,
        "expected_sum": expected_sum,
        "actual_sum": actual_sum,
        "difference": diff,
        "v_full": v_full,
        "v_empty": v_empty,
        "tolerance": tol,
    }
