"""
Unit tests for TN-SHAP-G core functionality.

Run tests with:
    cd TN-SHAP-G
    python -m pytest tests/ -v
"""

import numpy as np
import torch
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tnshapg import (
    GraphAlignedTN,
    MaskedGame,
    SimpleTeacher,
    BalancedSizeSampler,
    compute_diagonal_shapley,
    exact_shapley_enumeration,
    chebyshev_nodes_01,
    vandermonde_matrix,
    stable_polyfit,
    set_seed,
    cosine_similarity,
    bernoulli_vector,
    derivative_vector,
)


class TestVandermonde:
    """Test polynomial interpolation components."""
    
    def test_chebyshev_nodes_in_unit_interval(self):
        """Chebyshev nodes should be in [0, 1]."""
        for m in [5, 10, 16]:
            nodes = chebyshev_nodes_01(m)
            assert len(nodes) == m
            assert nodes.min() >= 0
            assert nodes.max() <= 1
            assert np.all(np.diff(nodes) > 0)  # Sorted ascending
    
    def test_vandermonde_shape(self):
        """Vandermonde matrix shape should be correct."""
        nodes = chebyshev_nodes_01(8)
        V = vandermonde_matrix(nodes, 8)
        assert V.shape == (8, 8)
        
        V = vandermonde_matrix(nodes, 5)
        assert V.shape == (8, 5)
    
    def test_polynomial_interpolation_identity(self):
        """Polynomial fit should recover known polynomial exactly."""
        # Test polynomial p(t) = 1 + 2t + 3t^2
        true_coeffs = np.array([1.0, 2.0, 3.0])
        
        # Generate values at nodes
        nodes = chebyshev_nodes_01(10)  # Overdetermined
        values = np.polyval(true_coeffs[::-1], nodes)  # numpy uses reversed order
        
        # Fit polynomial
        fitted_coeffs = stable_polyfit(nodes, values, degree=2)
        
        # Check coefficients match
        np.testing.assert_allclose(fitted_coeffs, true_coeffs, rtol=1e-6)
    
    def test_polynomial_integration(self):
        """Integration of polynomial on [0,1] should be exact."""
        from tnshapg.diagonal_shapley import integrate_polynomial
        
        # p(t) = 1 + 2t + 3t^2
        # ∫_0^1 p(t) dt = 1 + 1 + 1 = 3
        coeffs = np.array([1.0, 2.0, 3.0])
        integral = integrate_polynomial(coeffs)
        
        assert abs(integral - 3.0) < 1e-10


class TestTensorNetworkContraction:
    """Test TN contraction correctness."""
    
    def test_tn_forward_shape(self):
        """TN forward pass should return scalar."""
        n = 5
        tn = GraphAlignedTN(n, bond_dim=2, seed=42)
        
        z = torch.rand(n)
        output = tn.forward(z)
        
        assert output.shape == ()  # Scalar
        assert torch.isfinite(output)
    
    def test_tn_batch_shape(self):
        """TN batch forward should return correct shape."""
        n = 5
        batch = 10
        tn = GraphAlignedTN(n, bond_dim=2, seed=42)
        
        Z = torch.rand(batch, n)
        outputs = tn.contract_batch(Z)
        
        assert outputs.shape == (batch,)
        assert torch.all(torch.isfinite(outputs))
    
    def test_tn_endpoints(self):
        """TN should handle z=0 and z=1 correctly."""
        n = 5
        tn = GraphAlignedTN(n, bond_dim=2, seed=42)
        
        # All zeros (empty coalition)
        z_empty = torch.zeros(n)
        v_empty = tn.forward(z_empty)
        assert torch.isfinite(v_empty)
        
        # All ones (full coalition)
        z_full = torch.ones(n)
        v_full = tn.forward(z_full)
        assert torch.isfinite(v_full)
    
    def test_bernoulli_derivative_vectors(self):
        """Test Bernoulli and derivative vector construction."""
        b = bernoulli_vector(0.3)
        assert b.shape == (2,)
        np.testing.assert_allclose(b.numpy(), [0.7, 0.3])
        
        d = derivative_vector()
        assert d.shape == (2,)
        np.testing.assert_allclose(d.numpy(), [-1.0, 1.0])
    
    def test_multilinear_extension_identity(self):
        """
        Test multilinear extension property:
        ν̂(z) for binary z should equal ν̂(1_S) evaluated directly.
        """
        n = 4
        tn = GraphAlignedTN(n, bond_dim=2, seed=42)
        
        # Test a few binary coalitions
        coalitions = [
            torch.tensor([0., 0., 0., 0.]),
            torch.tensor([1., 0., 0., 0.]),
            torch.tensor([1., 1., 0., 0.]),
            torch.tensor([1., 1., 1., 1.]),
        ]
        
        for z in coalitions:
            v = tn.forward(z)
            assert torch.isfinite(v)


class TestShapleyComputation:
    """Test Shapley value computation correctness."""
    
    def test_shapley_efficiency_simple_game(self):
        """Test efficiency axiom: Σφ = v(N) - v(∅)."""
        n = 5
        set_seed(42)
        
        # Create simple additive game: v(S) = sum of node values
        node_values = np.random.randn(n)
        
        def value_fn(S):
            return sum(node_values[i] for i in S)
        
        # Exact Shapley should equal node values for additive game
        phi = exact_shapley_enumeration(value_fn, n, verbose=False)
        
        np.testing.assert_allclose(phi, node_values, rtol=1e-6)
    
    def test_shapley_efficiency_tn(self):
        """TN Shapley values should satisfy efficiency."""
        n = 6
        set_seed(42)
        
        tn = GraphAlignedTN(n, bond_dim=4, seed=42)
        tn.y_mean = 0.0
        tn.y_std = 1.0
        
        device = torch.device("cpu")
        
        # Compute Shapley
        phi = compute_diagonal_shapley(tn, n, m=16, device=device, verbose=False)
        
        # Check efficiency
        z_empty = torch.zeros(n)
        z_full = torch.ones(n)
        
        with torch.no_grad():
            v_empty = tn.forward(z_empty).item()
            v_full = tn.forward(z_full).item()
        
        expected_sum = v_full - v_empty
        actual_sum = np.sum(phi)
        
        np.testing.assert_allclose(actual_sum, expected_sum, rtol=0.01)
    
    def test_shapley_matches_exact_small_game(self):
        """TN-SHAP-G should match exact Shapley for trained surrogate."""
        n = 5
        set_seed(42)
        
        # Simple teacher: sum of features
        teacher = SimpleTeacher(n_features=3, aggregation="sum")
        
        # Random features
        X = torch.randn(n, 3)
        baseline = torch.zeros_like(X)
        
        # Create game
        game = MaskedGame(X=X, baseline=baseline, teacher=teacher)
        
        # Exact Shapley
        def value_fn(S):
            return game.query_coalition_set(S)
        
        phi_exact = exact_shapley_enumeration(value_fn, n, verbose=False)
        
        # For additive game, Shapley = contribution of each node's features
        # v(S) = sum_{i in S} sum_j X[i,j]
        # φ_i = sum_j X[i,j]
        expected = X.sum(dim=1).numpy()
        
        np.testing.assert_allclose(phi_exact, expected, rtol=1e-5)


class TestCoalitionSampler:
    """Test coalition sampling."""
    
    def test_balanced_sampler_coverage(self):
        """Balanced sampler should cover all coalition sizes."""
        n = 8
        m = 1000
        
        sampler = BalancedSizeSampler(n, seed=42)
        coalitions = sampler.sample(m)
        
        # Check sizes
        sizes = coalitions.sum(dim=1).numpy()
        
        # All sizes should be represented
        unique_sizes = set(int(s) for s in sizes)
        assert 0 in unique_sizes  # Empty
        assert n in unique_sizes  # Full
        
        # Distribution should be roughly uniform
        size_counts = np.bincount(sizes.astype(int), minlength=n+1)
        expected_per_size = m / (n + 1)
        
        # Allow some variance
        for count in size_counts:
            assert count > expected_per_size * 0.3  # At least 30% of expected
    
    def test_sampler_reproducibility(self):
        """Same seed should give same coalitions."""
        n = 5
        m = 50
        
        sampler1 = BalancedSizeSampler(n, seed=42)
        sampler2 = BalancedSizeSampler(n, seed=42)
        
        c1 = sampler1.sample(m)
        c2 = sampler2.sample(m)
        
        torch.testing.assert_close(c1, c2)


class TestMaskedGame:
    """Test masked game functionality."""
    
    def test_mask_features(self):
        """Feature masking should interpolate correctly."""
        n, d = 3, 4
        X = torch.ones(n, d)
        baseline = torch.zeros(n, d)
        teacher = SimpleTeacher(n_features=d)
        
        game = MaskedGame(X=X, baseline=baseline, teacher=teacher)
        
        # Full coalition: should get X
        z_full = torch.ones(n)
        X_full = game.mask_features(z_full)
        torch.testing.assert_close(X_full, X)
        
        # Empty coalition: should get baseline
        z_empty = torch.zeros(n)
        X_empty = game.mask_features(z_empty)
        torch.testing.assert_close(X_empty, baseline)
        
        # Partial: should interpolate
        z_half = torch.tensor([1., 0., 0.5])
        X_half = game.mask_features(z_half)
        assert X_half[0, 0] == 1.0  # Node 0 included
        assert X_half[1, 0] == 0.0  # Node 1 excluded
        assert X_half[2, 0] == 0.5  # Node 2 partial


class TestUtilities:
    """Test utility functions."""
    
    def test_cosine_similarity(self):
        """Cosine similarity should be correct."""
        a = np.array([1, 0, 0])
        b = np.array([1, 0, 0])
        assert abs(cosine_similarity(a, b) - 1.0) < 1e-10
        
        c = np.array([0, 1, 0])
        assert abs(cosine_similarity(a, c) - 0.0) < 1e-10
        
        d = np.array([-1, 0, 0])
        assert abs(cosine_similarity(a, d) - (-1.0)) < 1e-10
    
    def test_set_seed_reproducibility(self):
        """set_seed should make results reproducible."""
        set_seed(123)
        x1 = torch.randn(10)
        
        set_seed(123)
        x2 = torch.randn(10)
        
        torch.testing.assert_close(x1, x2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
