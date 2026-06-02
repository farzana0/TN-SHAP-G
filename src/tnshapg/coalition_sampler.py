"""
Coalition Sampler: Sampling strategies for TN surrogate training.

Provides balanced coalition sampling where the coalition size is
first sampled uniformly, then a random subset of that size is drawn.
This ensures good coverage across all coalition sizes, which is
important for learning the multilinear extension accurately.

The key insight is that uniform random sampling of binary vectors
concentrates around size n/2 (binomial distribution), missing
small and large coalitions that are important for Shapley values.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch

from .utils import set_seed


def sample_binary_coalitions(
    n: int,
    m: int,
    method: str = "balanced",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample binary coalition vectors.
    
    Args:
        n: Number of players/nodes
        m: Number of coalitions to sample
        method: Sampling method
            - 'balanced': Sample size uniformly, then sample subset
            - 'uniform': Each node independently included with p=0.5
            - 'stratified': Equal samples per size (when possible)
        seed: Random seed
        
    Returns:
        Binary coalition matrix [m, n]
    """
    if seed is not None:
        set_seed(seed)
    
    if method == "balanced":
        return _sample_balanced(n, m)
    elif method == "uniform":
        return _sample_uniform(n, m)
    elif method == "stratified":
        return _sample_stratified(n, m)
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def _sample_balanced(n: int, m: int) -> torch.Tensor:
    """Sample with uniform distribution over coalition sizes."""
    rng = np.random.default_rng()
    coalitions = np.zeros((m, n), dtype=np.float32)
    
    for i in range(m):
        # Sample size uniformly from 0 to n
        k = rng.integers(0, n + 1)
        # Sample k indices without replacement
        if k > 0:
            indices = rng.choice(n, size=k, replace=False)
            coalitions[i, indices] = 1.0
    
    return torch.from_numpy(coalitions)


def _sample_uniform(n: int, m: int) -> torch.Tensor:
    """Sample with each node independently included with p=0.5."""
    return (torch.rand(m, n) > 0.5).float()


def _sample_stratified(n: int, m: int) -> torch.Tensor:
    """Sample equal number of coalitions per size (approximately)."""
    rng = np.random.default_rng()
    coalitions = []
    
    # Number of samples per size (n+1 possible sizes: 0, 1, ..., n)
    n_sizes = n + 1
    samples_per_size = m // n_sizes
    remainder = m % n_sizes
    
    for k in range(n_sizes):
        n_samples = samples_per_size + (1 if k < remainder else 0)
        for _ in range(n_samples):
            coalition = np.zeros(n, dtype=np.float32)
            if k > 0:
                indices = rng.choice(n, size=k, replace=False)
                coalition[indices] = 1.0
            coalitions.append(coalition)
    
    # Shuffle to avoid ordering by size
    rng.shuffle(coalitions)
    return torch.from_numpy(np.array(coalitions[:m]))


class CoalitionSampler:
    """
    Base class for coalition samplers.
    
    Provides an iterator interface for generating coalitions.
    """
    
    def __init__(self, n: int, seed: Optional[int] = None):
        """
        Initialize sampler.
        
        Args:
            n: Number of players/nodes
            seed: Random seed
        """
        self.n = n
        self.seed = seed
        self._rng = np.random.default_rng(seed)
    
    def sample(self, m: int) -> torch.Tensor:
        """
        Sample m coalitions.
        
        Args:
            m: Number of coalitions
            
        Returns:
            Coalition matrix [m, n]
        """
        raise NotImplementedError
    
    def __call__(self, m: int) -> torch.Tensor:
        """Alias for sample()."""
        return self.sample(m)


class BalancedSizeSampler(CoalitionSampler):
    """
    Balanced coalition sampler with uniform size distribution.
    
    This sampler first draws the coalition size uniformly from {0, 1, ..., n},
    then draws a random subset of that size. This ensures:
    
    1. All coalition sizes are equally represented
    2. Boundary cases (empty and full coalitions) are well covered
    3. The multilinear extension is learned accurately at all scales
    
    Attributes:
        n: Number of players
        include_endpoints: Whether to always include empty/full coalitions
    """
    
    def __init__(
        self,
        n: int,
        seed: Optional[int] = None,
        include_endpoints: bool = True,
    ):
        """
        Initialize balanced sampler.
        
        Args:
            n: Number of players
            seed: Random seed
            include_endpoints: If True, always include empty and full coalitions
        """
        super().__init__(n, seed)
        self.include_endpoints = include_endpoints
    
    def sample(self, m: int) -> torch.Tensor:
        """
        Sample m coalitions with balanced sizes.
        
        Args:
            m: Number of coalitions to sample
            
        Returns:
            Binary coalition matrix [m, n]
        """
        coalitions = np.zeros((m, self.n), dtype=np.float32)
        
        idx = 0
        
        # Optionally include endpoints
        if self.include_endpoints and m >= 2:
            # Empty coalition
            coalitions[0, :] = 0.0
            # Full coalition
            coalitions[1, :] = 1.0
            idx = 2
        
        # Sample remaining coalitions with balanced sizes
        while idx < m:
            # Uniform size
            k = self._rng.integers(0, self.n + 1)
            # Random subset of size k
            if k == 0:
                pass  # Already zeros
            elif k == self.n:
                coalitions[idx, :] = 1.0
            else:
                indices = self._rng.choice(self.n, size=k, replace=False)
                coalitions[idx, indices] = 1.0
            idx += 1
        
        return torch.from_numpy(coalitions)
    
    def sample_with_size(self, size: int) -> torch.Tensor:
        """
        Sample a single coalition of specific size.
        
        Args:
            size: Coalition size (0 to n)
            
        Returns:
            Binary coalition vector [n]
        """
        coalition = np.zeros(self.n, dtype=np.float32)
        if size > 0:
            indices = self._rng.choice(self.n, size=size, replace=False)
            coalition[indices] = 1.0
        return torch.from_numpy(coalition)


class InterpolationSampler(CoalitionSampler):
    """
    Sampler for continuous interpolation training.
    
    Instead of binary coalitions, samples continuous selector vectors
    z ∈ [0, 1]^n that interpolate between baseline and original features.
    
    This is useful for training TN to learn the multilinear extension
    directly, enabling the diagonal derivative trick for Shapley computation.
    """
    
    def __init__(
        self,
        n: int,
        seed: Optional[int] = None,
        t_distribution: str = "chebyshev",
    ):
        """
        Initialize interpolation sampler.
        
        Args:
            n: Number of players
            seed: Random seed
            t_distribution: Distribution for interpolation values
                - 'uniform': Uniform on [0, 1]
                - 'chebyshev': Chebyshev nodes for better interpolation
                - 'beta': Beta(0.5, 0.5) for emphasis on boundaries
        """
        super().__init__(n, seed)
        self.t_distribution = t_distribution
    
    def sample(self, m: int) -> torch.Tensor:
        """
        Sample m continuous selector vectors.
        
        Args:
            m: Number of samples
            
        Returns:
            Selector matrix [m, n] with values in [0, 1]
        """
        if self.t_distribution == "uniform":
            return torch.from_numpy(
                self._rng.random((m, self.n)).astype(np.float32)
            )
        elif self.t_distribution == "chebyshev":
            # Sample t from Chebyshev distribution
            # Use arccosine transform: X = 0.5 * (1 + cos(π * U))
            u = self._rng.random((m, self.n))
            t = 0.5 * (1 + np.cos(np.pi * u))
            return torch.from_numpy(t.astype(np.float32))
        elif self.t_distribution == "beta":
            # Beta(0.5, 0.5) = arcsine distribution
            t = self._rng.beta(0.5, 0.5, size=(m, self.n))
            return torch.from_numpy(t.astype(np.float32))
        else:
            raise ValueError(f"Unknown distribution: {self.t_distribution}")


def generate_all_coalitions(n: int) -> torch.Tensor:
    """
    Generate all 2^n coalitions (for small n only).
    
    Args:
        n: Number of players (should be ≤ 20 for memory)
        
    Returns:
        Binary coalition matrix [2^n, n]
    """
    if n > 20:
        raise ValueError(f"n={n} too large for exhaustive enumeration (2^{n} coalitions)")
    
    num_coalitions = 2 ** n
    coalitions = np.zeros((num_coalitions, n), dtype=np.float32)
    
    for i in range(num_coalitions):
        for j in range(n):
            if (i >> j) & 1:
                coalitions[i, j] = 1.0
    
    return torch.from_numpy(coalitions)


def create_training_dataset(
    game,
    n_samples: int,
    method: str = "balanced",
    seed: Optional[int] = None,
    include_corners: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create training dataset from masked game.
    
    Args:
        game: MaskedGame instance
        n_samples: Number of coalition samples
        method: Sampling method ('balanced', 'uniform', 'stratified')
        seed: Random seed
        include_corners: Whether to include empty and full coalitions
        
    Returns:
        coalitions: [n_samples, n] binary masks
        values: [n_samples] game values
    """
    n = game.n_nodes
    
    # Sample coalitions
    coalitions = sample_binary_coalitions(n, n_samples, method=method, seed=seed)
    
    # Optionally ensure endpoints are included
    if include_corners and n_samples >= 2:
        coalitions[0, :] = 0.0  # Empty
        coalitions[1, :] = 1.0  # Full
    
    # Query game values
    values = game.query_batch(coalitions)
    
    return coalitions, values
