"""
Graph-Aligned Tensor Network: Core TN surrogate implementation.

This module implements a tensor network (TN) whose structure mirrors
the input graph. Each node u has a tensor core A^(u) with:
  - Physical dimension 2 (coalition membership: 0=excluded, 1=included)
  - Bond dimension χ for each edge incident to u

The TN represents a multilinear function over binary/continuous selectors:
    ν̂(z) = contract(A^(1), ..., A^(n), z₁, ..., zₙ)

For Shapley computation, we use:
  - Bernoulli vectors b(t) = [1-t, t] at most nodes
  - Derivative vectors b'(t) = [-1, 1] at the node(s) of interest
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import networkx as nx

# Try to import contraction optimization backends
try:
    import opt_einsum as oe
    from opt_einsum import contract, get_symbol
    OPT_EINSUM_AVAILABLE = True
except ImportError:
    OPT_EINSUM_AVAILABLE = False
    def get_symbol(i: int) -> str:
        """Generate einsum index labels."""
        if i < 52:
            return chr(ord('a') + i) if i < 26 else chr(ord('A') + i - 26)
        return f"idx{i}"


def bernoulli_vector(t: float, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create Bernoulli probability vector b(t) = [1-t, t].
    
    This represents the probability distribution over {excluded, included}
    when each node is included independently with probability t.
    
    Args:
        t: Inclusion probability in [0, 1]
        device: Torch device
        
    Returns:
        Bernoulli vector [2]
    """
    return torch.tensor([1.0 - t, t], dtype=torch.float32, device=device)


def derivative_vector(device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create derivative vector b'(t) = [-1, 1].
    
    This is the derivative of bernoulli_vector(t) with respect to t.
    Used for computing partial derivatives of the TN output.
    
    Args:
        device: Torch device
        
    Returns:
        Derivative vector [2]
    """
    return torch.tensor([-1.0, 1.0], dtype=torch.float32, device=device)


class GraphAlignedTN(nn.Module):
    """
    Graph-aligned tensor network surrogate.
    
    The TN topology mirrors the input graph structure:
    - Each graph node u has a tensor core A^(u)
    - Each graph edge (u, v) corresponds to a bond between cores
    - Physical dimension represents coalition membership (binary)
    
    The TN computes:
        ν̂(z) = Σ_{bond indices} Π_u A^(u)_{bonds_u, z_u}
    
    where z is a binary/continuous selector vector indicating coalition membership.
    
    Attributes:
        n_nodes: Number of nodes
        bond_dim: Bond dimension χ (higher = more expressive)
        graph: NetworkX graph defining TN structure
        tn_parameters: Learnable tensor cores
    """
    
    def __init__(
        self,
        n_nodes: int,
        graph_structure: Optional[nx.Graph] = None,
        bond_dim: int = 4,
        seed: Optional[int] = None,
        init_scale: float = 0.1,
    ):
        """
        Initialize graph-aligned TN.
        
        Args:
            n_nodes: Number of nodes in the graph
            graph_structure: NetworkX graph defining connections.
                            If None, uses a chain/path graph.
            bond_dim: Bond dimension for all edges
            seed: Random seed for initialization
            init_scale: Scale for random initialization
        """
        super().__init__()
        
        self.n_nodes = n_nodes
        self.bond_dim = bond_dim
        self.physical_dim = 2  # Binary: excluded/included
        self.seed = seed
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Build graph structure
        if graph_structure is not None:
            self.graph = nx.Graph(graph_structure)
            # Ensure all nodes exist
            self.graph.add_nodes_from(range(n_nodes))
        else:
            # Default: path graph (simple chain)
            self.graph = nx.path_graph(n_nodes)
        
        # Store per-edge bond dimensions (can be customized later)
        self._edge_bond_dims: Dict[Tuple[int, int], int] = {}
        for u, v in self.graph.edges():
            key = (min(u, v), max(u, v))
            self._edge_bond_dims[key] = bond_dim
        
        # Initialize tensor cores as parameters
        self.tn_parameters = nn.ParameterDict()
        self._initialize_cores(init_scale)
        
        # Precompute contraction structure (for efficiency)
        self._contraction_info = None
        self._build_contraction_info()
    
    def _get_edge_bond_dim(self, u: int, v: int) -> int:
        """Get bond dimension for edge (u, v)."""
        key = (min(u, v), max(u, v))
        return self._edge_bond_dims.get(key, self.bond_dim)
    
    def _initialize_cores(self, init_scale: float):
        """
        Initialize tensor cores with numerically stable values.
        
        Each core A^(u) has shape [χ_1, χ_2, ..., χ_deg(u), 2]
        where the last dimension is the physical (coalition) index.
        """
        for node in range(self.n_nodes):
            neighbors = sorted(list(self.graph.neighbors(node)))
            
            if len(neighbors) == 0:
                # Isolated node: just physical dimension
                shape = [self.physical_dim]
            else:
                # Bond dimensions for each neighbor + physical dimension
                bond_dims = [self._get_edge_bond_dim(node, nb) for nb in neighbors]
                shape = bond_dims + [self.physical_dim]
            
            # Xavier-like initialization scaled by degree
            fan_in = np.prod(shape[:-1]) if len(shape) > 1 else 1
            fan_out = shape[-1]
            scale = init_scale / np.sqrt(max(1, fan_in + fan_out))
            
            tensor_data = torch.randn(*shape) * scale
            
            # Set canonical entry for stability (at physical index 0)
            # This ensures the "all excluded" output is well-defined
            if len(shape) > 1:
                idx = tuple([0] * (len(shape) - 1) + [0])
                tensor_data[idx] = 1.0
            else:
                tensor_data[0] = 1.0
            
            param_name = f"core_{node}"
            self.tn_parameters[param_name] = nn.Parameter(tensor_data)
    
    def _build_contraction_info(self):
        """
        Precompute the einsum contraction structure.
        
        This creates the index labels and equation for efficient contraction.
        """
        edge_to_label = {}
        label_counter = 0
        
        # Assign labels to each edge (bond)
        for u, v in self.graph.edges():
            key = (min(u, v), max(u, v))
            if key not in edge_to_label:
                edge_to_label[key] = get_symbol(label_counter)
                label_counter += 1
        
        # Assign labels for physical indices (input selectors)
        physical_labels = []
        for node in range(self.n_nodes):
            physical_labels.append(get_symbol(label_counter))
            label_counter += 1
        
        # Build index lists for each tensor
        core_indices = []
        for node in range(self.n_nodes):
            neighbors = sorted(list(self.graph.neighbors(node)))
            indices = []
            
            # Bond indices (in order of sorted neighbors)
            for nb in neighbors:
                key = (min(node, nb), max(node, nb))
                indices.append(edge_to_label[key])
            
            # Physical index (last)
            indices.append(physical_labels[node])
            
            core_indices.append("".join(indices))
        
        # Input selectors: each has one physical index
        input_indices = [physical_labels[i] for i in range(self.n_nodes)]
        
        # Output: scalar (no output indices)
        output_indices = ""
        
        # Full einsum equation
        all_inputs = core_indices + input_indices
        einsum_eq = ",".join(all_inputs) + "->" + output_indices
        
        self._contraction_info = {
            "einsum_eq": einsum_eq,
            "core_indices": core_indices,
            "input_indices": input_indices,
            "physical_labels": physical_labels,
            "edge_to_label": edge_to_label,
        }
    
    def get_core(self, node: int) -> torch.Tensor:
        """Get tensor core for a node."""
        return self.tn_parameters[f"core_{node}"]
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Evaluate TN at selector vector z.
        
        Args:
            z: Selector vector [n] with values in [0, 1]
               0 = excluded (use baseline), 1 = included (use original)
               Can also be binary.
               
        Returns:
            Scalar TN output
        """
        if z.ndim != 1 or z.shape[0] != self.n_nodes:
            raise ValueError(f"z should be [n_nodes], got {z.shape}")
        
        return self._contract_with_selectors(z)
    
    def _contract_with_selectors(self, z: torch.Tensor) -> torch.Tensor:
        """
        Contract TN cores with selector vectors.
        
        Each selector z_i ∈ [0, 1] is converted to Bernoulli vector [1-z_i, z_i].
        """
        device = z.device
        
        # Build Bernoulli vectors for each node
        # b_i = [1 - z_i, z_i]
        bernoulli_vecs = []
        for i in range(self.n_nodes):
            b = torch.stack([1.0 - z[i], z[i]])  # [2]
            bernoulli_vecs.append(b)
        
        # Collect all tensors: cores + bernoulli vectors
        tensors = []
        for node in range(self.n_nodes):
            tensors.append(self.get_core(node).to(device))
        for b in bernoulli_vecs:
            tensors.append(b)
        
        # Contract using einsum
        einsum_eq = self._contraction_info["einsum_eq"]
        
        if OPT_EINSUM_AVAILABLE:
            result = oe.contract(einsum_eq, *tensors, backend='torch')
        else:
            result = torch.einsum(einsum_eq, *tensors)
        
        return result
    
    def contract_batch(
        self,
        Z: torch.Tensor,
        detach_inputs: bool = False,
    ) -> torch.Tensor:
        """
        Evaluate TN for a batch of selector vectors.
        
        Args:
            Z: Selector matrix [batch, n_nodes]
            detach_inputs: Whether to detach from autograd (for inference)
            
        Returns:
            Output values [batch]
        """
        if Z.ndim != 2 or Z.shape[1] != self.n_nodes:
            raise ValueError(f"Z should be [batch, n_nodes], got {Z.shape}")
        
        batch_size = Z.shape[0]
        results = []
        
        for i in range(batch_size):
            z = Z[i]
            if detach_inputs:
                z = z.detach()
            results.append(self._contract_with_selectors(z))
        
        return torch.stack(results)
    
    def contract_with_vectors(
        self,
        vectors: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Contract TN with arbitrary 2D vectors at each node.
        
        This is the core routine for Shapley computation:
        - Use Bernoulli vectors b(t) = [1-t, t] for most nodes
        - Use derivative vectors b'(t) = [-1, 1] for differentiated nodes
        
        Args:
            vectors: List of n_nodes vectors, each of shape [2]
            
        Returns:
            Scalar contraction result
        """
        if len(vectors) != self.n_nodes:
            raise ValueError(f"Expected {self.n_nodes} vectors, got {len(vectors)}")
        
        device = vectors[0].device
        
        # Collect all tensors: cores + input vectors
        tensors = []
        for node in range(self.n_nodes):
            tensors.append(self.get_core(node).to(device))
        for v in vectors:
            tensors.append(v)
        
        # Contract
        einsum_eq = self._contraction_info["einsum_eq"]
        
        if OPT_EINSUM_AVAILABLE:
            result = oe.contract(einsum_eq, *tensors, backend='torch')
        else:
            result = torch.einsum(einsum_eq, *tensors)
        
        return result
    
    def evaluate_derivative_at_t(
        self,
        t: float,
        derivative_nodes: List[int],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Evaluate partial derivative of TN output at diagonal point t·1.
        
        This computes ∂^k ν̂ / ∂z_{i1}...∂z_{ik} evaluated at z = t·1
        by placing derivative vectors at the specified nodes.
        
        Args:
            t: Evaluation point in [0, 1]
            derivative_nodes: List of node indices for differentiation
            device: Torch device
            
        Returns:
            Derivative value (scalar)
        """
        if device is None:
            device = next(self.parameters()).device
        
        vectors = []
        for node in range(self.n_nodes):
            if node in derivative_nodes:
                # Derivative vector b'(t) = [-1, 1]
                vectors.append(derivative_vector(device))
            else:
                # Bernoulli vector b(t) = [1-t, t]
                vectors.append(bernoulli_vector(t, device))
        
        return self.contract_with_vectors(vectors)
    
    def get_num_parameters(self) -> int:
        """Count total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_graph_info(self) -> dict:
        """Get information about the TN graph structure."""
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.graph.number_of_edges(),
            "bond_dim": self.bond_dim,
            "num_parameters": self.get_num_parameters(),
            "degrees": dict(self.graph.degree()),
        }
    
    def __repr__(self) -> str:
        info = self.get_graph_info()
        return (
            f"GraphAlignedTN(n_nodes={info['n_nodes']}, "
            f"n_edges={info['n_edges']}, bond_dim={info['bond_dim']}, "
            f"params={info['num_parameters']})"
        )


def create_tn_from_graph(
    edge_index: torch.Tensor,
    n_nodes: int,
    bond_dim: int = 4,
    seed: Optional[int] = None,
) -> GraphAlignedTN:
    """
    Create GraphAlignedTN from PyG-style edge_index.
    
    Args:
        edge_index: [2, num_edges] tensor (both directions included)
        n_nodes: Number of nodes
        bond_dim: Bond dimension
        seed: Random seed
        
    Returns:
        GraphAlignedTN instance
    """
    from .utils import edge_index_to_networkx
    
    G = edge_index_to_networkx(edge_index, n_nodes)
    return GraphAlignedTN(n_nodes, graph_structure=G, bond_dim=bond_dim, seed=seed)


def create_chain_tn(
    n_nodes: int,
    bond_dim: int = 4,
    seed: Optional[int] = None,
) -> GraphAlignedTN:
    """
    Create TN with chain/path topology.
    
    This is the simplest structure and works well for small graphs.
    Equivalent to a Matrix Product State (MPS).
    
    Args:
        n_nodes: Number of nodes
        bond_dim: Bond dimension
        seed: Random seed
        
    Returns:
        GraphAlignedTN with chain structure
    """
    G = nx.path_graph(n_nodes)
    return GraphAlignedTN(n_nodes, graph_structure=G, bond_dim=bond_dim, seed=seed)


def create_tree_tn(
    n_nodes: int,
    bond_dim: int = 4,
    seed: Optional[int] = None,
) -> GraphAlignedTN:
    """
    Create TN with random tree topology.
    
    Trees allow exact contraction in O(n) and are a good default
    when the graph structure is unknown.
    
    Args:
        n_nodes: Number of nodes
        bond_dim: Bond dimension
        seed: Random seed
        
    Returns:
        GraphAlignedTN with tree structure
    """
    if seed is not None:
        np.random.seed(seed)
    
    G = nx.random_tree(n_nodes, seed=seed)
    return GraphAlignedTN(n_nodes, graph_structure=G, bond_dim=bond_dim, seed=seed)
