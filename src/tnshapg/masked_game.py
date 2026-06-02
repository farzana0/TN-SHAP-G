"""
Masked Game: Coalition masking and teacher model queries.

This module implements the masked graph game that serves as the
oracle for training tensor network surrogates. For a graph (G, X)
with baseline X₀, coalition S masks nodes not in S by replacing
their features with the baseline, then queries the teacher model.

The game value is: v(S) = f(G, X_S) where
    X_S[i] = X[i]   if i ∈ S
    X_S[i] = X₀[i]  if i ∉ S

This implements feature masking (vs. graph masking where edges would
also be removed for excluded nodes).
"""

from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn


class MaskedGame:
    """
    Masked graph game for Shapley value computation.
    
    Given a graph with node features X, baseline X₀, and a teacher model f,
    this class computes game values v(S) = f(X_S) where X_S masks nodes
    not in coalition S with baseline features.
    
    Attributes:
        X: Original node features [n, d]
        baseline: Baseline features [n, d] (typically mean or zero)
        teacher: Teacher model that produces scalar predictions
        n_nodes: Number of nodes in the graph
        device: Torch device
    
    Example:
        >>> game = MaskedGame(X, baseline, teacher_model)
        >>> coalition = torch.tensor([1, 0, 1, 0])  # Include nodes 0 and 2
        >>> value = game(coalition)  # Returns f(X_masked)
    """
    
    def __init__(
        self,
        X: torch.Tensor,
        baseline: torch.Tensor,
        teacher: Callable[[torch.Tensor], torch.Tensor],
        edge_index: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the masked game.
        
        Args:
            X: Node features [n, d]
            baseline: Baseline features [n, d] for masking
            teacher: Teacher model f: [n, d] -> scalar
            edge_index: Optional edge connectivity [2, E] (for GNN teachers)
            device: Torch device (inferred from X if not provided)
        """
        if X.ndim != 2:
            raise ValueError(f"X should be 2D [n, d], got shape {X.shape}")
        if baseline.shape != X.shape:
            raise ValueError(f"baseline shape {baseline.shape} != X shape {X.shape}")
        
        self.device = device or X.device
        self.X = X.to(self.device)
        self.baseline = baseline.to(self.device)
        self.teacher = teacher
        self.edge_index = edge_index.to(self.device) if edge_index is not None else None
        self.n_nodes = X.shape[0]
        self.n_features = X.shape[1]
        
        # Precompute delta for efficiency
        self._delta = self.X - self.baseline
    
    def mask_features(self, coalition: torch.Tensor) -> torch.Tensor:
        """
        Apply coalition mask to features.
        
        X_S = baseline + S * (X - baseline)
            = baseline + S * delta
        
        Args:
            coalition: Binary mask [n] or [batch, n] or selector in [0, 1]
            
        Returns:
            Masked features [n, d] or [batch, n, d]
        """
        coalition = coalition.to(self.device)
        
        if coalition.ndim == 1:
            # Single coalition [n]
            z = coalition.view(-1, 1).float()  # [n, 1]
            return self.baseline + z * self._delta
        else:
            # Batch of coalitions [batch, n]
            z = coalition.unsqueeze(-1).float()  # [batch, n, 1]
            return self.baseline.unsqueeze(0) + z * self._delta.unsqueeze(0)
    
    def __call__(
        self,
        coalition: torch.Tensor,
        return_masked: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Evaluate game value for a coalition.
        
        Args:
            coalition: Binary mask [n] (single) or [batch, n] (batch)
            return_masked: If True, also return masked features
            
        Returns:
            Game value(s): scalar if single coalition, [batch] if batched
            If return_masked=True: tuple of (values, masked_features)
        """
        X_masked = self.mask_features(coalition)
        
        with torch.no_grad():
            if X_masked.ndim == 2:
                # Single coalition
                value = self.teacher(X_masked, self.edge_index)
                if isinstance(value, torch.Tensor):
                    value = value.squeeze()
            else:
                # Batch of coalitions [batch, n, d]
                batch_size = X_masked.shape[0]
                values = []
                for i in range(batch_size):
                    v = self.teacher(X_masked[i], self.edge_index)
                    if isinstance(v, torch.Tensor):
                        v = v.squeeze()
                    values.append(v)
                value = torch.stack(values)
        
        if return_masked:
            return value, X_masked
        return value
    
    def query_coalition_set(self, coalition_set: set) -> float:
        """
        Query game value for a coalition specified as a set of indices.
        
        Args:
            coalition_set: Set of node indices to include
            
        Returns:
            Game value as float
        """
        mask = torch.zeros(self.n_nodes, device=self.device)
        for i in coalition_set:
            mask[i] = 1.0
        value = self(mask)
        return float(value.item() if isinstance(value, torch.Tensor) else value)
    
    def query_batch(
        self,
        coalitions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate game for a batch of coalitions.
        
        Args:
            coalitions: [batch, n] binary masks
            
        Returns:
            Game values [batch]
        """
        return self(coalitions)
    
    def grand_coalition_value(self) -> float:
        """Value of the grand coalition (all nodes included)."""
        mask = torch.ones(self.n_nodes, device=self.device)
        return float(self(mask).item())
    
    def empty_coalition_value(self) -> float:
        """Value of the empty coalition (all nodes masked)."""
        mask = torch.zeros(self.n_nodes, device=self.device)
        return float(self(mask).item())


class SimpleTeacher(nn.Module):
    """
    Simple teacher model for testing: sum of node features.
    
    f(X) = sum(X) or weighted sum.
    """
    
    def __init__(self, n_features: int, aggregation: str = "sum"):
        """
        Initialize simple teacher.
        
        Args:
            n_features: Number of input features
            aggregation: 'sum', 'mean', or 'weighted'
        """
        super().__init__()
        self.aggregation = aggregation
        
        if aggregation == "weighted":
            self.weights = nn.Parameter(torch.randn(n_features))
        else:
            self.weights = None
    
    def forward(
        self,
        X: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute prediction.
        
        Args:
            X: Node features [n, d]
            edge_index: Ignored for this simple model
            
        Returns:
            Scalar prediction
        """
        if self.aggregation == "sum":
            return X.sum()
        elif self.aggregation == "mean":
            return X.mean()
        elif self.aggregation == "weighted":
            return (X * self.weights).sum()
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


class MLPTeacher(nn.Module):
    """
    MLP teacher model that aggregates node features and applies MLP.
    
    This provides a simple but non-trivial teacher model that:
    1. Aggregates node features (sum/mean)
    2. Applies a small MLP to produce scalar output
    
    No graph structure is used (pure feature-based prediction).
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_dims: list = [32, 16],
        aggregation: str = "sum",
        seed: Optional[int] = None,
    ):
        """
        Initialize MLP teacher.
        
        Args:
            n_features: Number of input features per node
            hidden_dims: List of hidden layer dimensions
            aggregation: 'sum' or 'mean' for node aggregation
            seed: Random seed for initialization
        """
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
        
        self.aggregation = aggregation
        
        # Build MLP layers
        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        X: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute prediction.
        
        Args:
            X: Node features [n, d]
            edge_index: Ignored for this model
            
        Returns:
            Scalar prediction
        """
        # Aggregate node features
        if self.aggregation == "sum":
            x_agg = X.sum(dim=0)  # [d]
        elif self.aggregation == "mean":
            x_agg = X.mean(dim=0)  # [d]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Apply MLP
        return self.mlp(x_agg).squeeze()


def create_baseline(
    X: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """
    Create baseline features for masking.
    
    Args:
        X: Original features [n, d]
        mode: 'zero', 'mean', 'node_mean'
            - 'zero': All zeros baseline
            - 'mean': Mean across all nodes (same for all positions)
            - 'node_mean': Per-feature mean (broadcast to all nodes)
            
    Returns:
        Baseline features [n, d]
    """
    n, d = X.shape
    
    if mode == "zero":
        return torch.zeros_like(X)
    elif mode == "mean":
        mean_val = X.mean(dim=0, keepdim=True)  # [1, d]
        return mean_val.expand(n, d).clone()
    elif mode == "node_mean":
        return X.mean(dim=0, keepdim=True).expand(n, d).clone()
    else:
        raise ValueError(f"Unknown baseline mode: {mode}")
