"""
Exact Shapley: Brute-force enumeration for sanity checks.

Computes exact Shapley values by enumerating all 2^n coalitions.
This is only feasible for small n (n ≤ 15-20) but provides ground
truth for validating the TN-SHAP-G method.
"""

from itertools import combinations
from math import factorial
from typing import Callable, Dict, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm


def powerset(n: int, include_empty: bool = True, include_full: bool = True):
    """
    Generate all subsets of {0, 1, ..., n-1}.
    
    Args:
        n: Number of elements
        include_empty: Include empty set
        include_full: Include full set
        
    Yields:
        Sets representing coalitions
    """
    start = 0 if include_empty else 1
    end = n + 1 if include_full else n
    
    for r in range(start, end):
        for combo in combinations(range(n), r):
            yield set(combo)


def compute_all_coalition_values(
    value_fn: Callable[[Set[int]], float],
    n: int,
    verbose: bool = True,
) -> Dict[frozenset, float]:
    """
    Compute value for all 2^n coalitions.
    
    Args:
        value_fn: Function mapping coalition set -> value
        n: Number of players
        verbose: Show progress bar
        
    Returns:
        Dictionary mapping frozenset(coalition) -> value
    """
    if n > 25:
        raise ValueError(f"n={n} too large for exact enumeration")
    
    values = {}
    total = 2 ** n
    
    iterator = powerset(n)
    if verbose:
        iterator = tqdm(iterator, total=total, desc="Computing all coalitions")
    
    for S in iterator:
        v = value_fn(S)
        values[frozenset(S)] = float(v)
    
    return values


def exact_shapley_enumeration(
    value_fn: Callable[[Set[int]], float],
    n: int,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute exact Shapley values by enumeration.
    
    The Shapley value for player i is:
        φ_i = Σ_{S ⊆ N\{i}} w(|S|) · [v(S ∪ {i}) - v(S)]
    
    where w(s) = s!(n-s-1)!/n!
    
    Args:
        value_fn: Function mapping coalition set -> value
        n: Number of players
        verbose: Show progress
        
    Returns:
        Shapley values [n]
    """
    # First compute all coalition values
    values = compute_all_coalition_values(value_fn, n, verbose=verbose)
    
    # Then compute Shapley values
    phi = np.zeros(n, dtype=np.float64)
    n_fact = factorial(n)
    
    iterator = range(n)
    if verbose:
        iterator = tqdm(iterator, desc="Computing Shapley values")
    
    for i in iterator:
        others = [j for j in range(n) if j != i]
        
        for s in range(n):  # Coalition sizes without i
            # Shapley weight
            w = factorial(s) * factorial(n - s - 1) / n_fact
            
            for S_tuple in combinations(others, s):
                S = frozenset(S_tuple)
                S_with_i = frozenset(S | {i})
                
                v_with = values.get(S_with_i, 0.0)
                v_without = values.get(S, 0.0)
                
                phi[i] += w * (v_with - v_without)
    
    return phi


def exact_shapley_fast(
    value_fn: Callable[[Set[int]], float],
    n: int,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[frozenset, float]]:
    """
    Compute exact Shapley values (returns coalition values too).
    
    Args:
        value_fn: Function mapping coalition set -> value
        n: Number of players
        verbose: Show progress
        
    Returns:
        Tuple of (shapley_values [n], coalition_values dict)
    """
    values = compute_all_coalition_values(value_fn, n, verbose=verbose)
    phi = _shapley_from_values(values, n, verbose=verbose)
    return phi, values


def _shapley_from_values(
    values: Dict[frozenset, float],
    n: int,
    verbose: bool = False,
) -> np.ndarray:
    """Compute Shapley from precomputed coalition values."""
    phi = np.zeros(n, dtype=np.float64)
    n_fact = factorial(n)
    
    for i in range(n):
        others = [j for j in range(n) if j != i]
        
        for s in range(n):
            w = factorial(s) * factorial(n - s - 1) / n_fact
            
            for S_tuple in combinations(others, s):
                S = frozenset(S_tuple)
                S_with_i = frozenset(S | {i})
                
                v_with = values.get(S_with_i, 0.0)
                v_without = values.get(S, 0.0)
                
                phi[i] += w * (v_with - v_without)
    
    return phi


def efficiency_check(
    phi: np.ndarray,
    values: Optional[Dict[frozenset, float]] = None,
    value_fn: Optional[Callable] = None,
    n: Optional[int] = None,
    tol: float = 1e-6,
) -> dict:
    """
    Check Shapley efficiency axiom: Σ φ_i = v(N) - v(∅).
    
    Args:
        phi: Shapley values [n]
        values: Coalition values dict (optional)
        value_fn: Value function (used if values not provided)
        n: Number of players
        tol: Tolerance
        
    Returns:
        Dictionary with check results
    """
    if n is None:
        n = len(phi)
    
    # Get v(N) and v(∅)
    if values is not None:
        v_full = values.get(frozenset(range(n)), 0.0)
        v_empty = values.get(frozenset(), 0.0)
    elif value_fn is not None:
        v_full = value_fn(set(range(n)))
        v_empty = value_fn(set())
    else:
        raise ValueError("Provide either values dict or value_fn")
    
    expected_sum = v_full - v_empty
    actual_sum = float(np.sum(phi))
    diff = abs(actual_sum - expected_sum)
    
    passed = diff <= tol
    
    return {
        "passed": passed,
        "expected_sum": expected_sum,
        "actual_sum": actual_sum,
        "difference": diff,
        "v_grand_coalition": v_full,
        "v_empty_coalition": v_empty,
        "tolerance": tol,
    }


def symmetry_check(
    phi: np.ndarray,
    values: Dict[frozenset, float],
    n: int,
    tol: float = 1e-6,
) -> list:
    """
    Find player pairs that violate symmetry.
    
    Two players i, j are symmetric if they have the same
    marginal contribution to every coalition. If symmetric,
    they should have equal Shapley values.
    
    Args:
        phi: Shapley values
        values: Coalition values
        n: Number of players
        tol: Tolerance
        
    Returns:
        List of (i, j) pairs that should be symmetric but have different φ
    """
    violations = []
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check if i, j are symmetric
            symmetric = True
            others = [k for k in range(n) if k != i and k != j]
            
            for S_tuple in powerset(len(others)):
                S = frozenset(others[k] for k in S_tuple if k < len(others))
                
                S_i = frozenset(S | {i})
                S_j = frozenset(S | {j})
                
                mc_i = values.get(S_i, 0) - values.get(S, 0)
                mc_j = values.get(S_j, 0) - values.get(S, 0)
                
                if abs(mc_i - mc_j) > tol:
                    symmetric = False
                    break
            
            if symmetric and abs(phi[i] - phi[j]) > tol:
                violations.append((i, j))
    
    return violations


def null_player_check(
    phi: np.ndarray,
    values: Dict[frozenset, float],
    n: int,
    tol: float = 1e-6,
) -> list:
    """
    Find null players (zero marginal contribution everywhere).
    
    A null player should have Shapley value 0.
    
    Args:
        phi: Shapley values
        values: Coalition values
        n: Number of players
        tol: Tolerance
        
    Returns:
        List of null players whose φ is not ~0
    """
    violations = []
    
    for i in range(n):
        is_null = True
        others = [j for j in range(n) if j != i]
        
        for S in powerset(n):
            if i not in S:
                S_with_i = frozenset(S | {i})
                mc = values.get(S_with_i, 0) - values.get(frozenset(S), 0)
                
                if abs(mc) > tol:
                    is_null = False
                    break
        
        if is_null and abs(phi[i]) > tol:
            violations.append(i)
    
    return violations
