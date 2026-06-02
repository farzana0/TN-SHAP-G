#!/usr/bin/env python3
"""
Compute Shapley values from trained TN surrogate.

Example:
    python compute_shapley.py --model outputs/tn_surrogate.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tnshapg import (
    GraphAlignedTN,
    compute_diagonal_shapley,
    set_seed,
)
from tnshapg.diagonal_shapley import verify_efficiency


def main():
    parser = argparse.ArgumentParser(description="Compute Shapley values from TN surrogate")
    
    parser.add_argument("--model", type=str, required=True, help="Path to trained TN model")
    parser.add_argument("--m", type=int, default=16, help="Number of interpolation nodes")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device)
    
    n_nodes = checkpoint["n_nodes"]
    bond_dim = checkpoint["bond_dim"]
    graph_edges = checkpoint.get("graph_edges", None)
    
    # Reconstruct graph structure
    import networkx as nx
    if graph_edges:
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from(graph_edges)
    else:
        G = None
    
    # Create TN
    tn = GraphAlignedTN(
        n_nodes=n_nodes,
        graph_structure=G,
        bond_dim=bond_dim,
    )
    tn.load_state_dict(checkpoint["state_dict"])
    tn.y_mean = checkpoint.get("y_mean", 0.0)
    tn.y_std = checkpoint.get("y_std", 1.0)
    tn = tn.to(device)
    tn.eval()
    
    print(f"Loaded TN with {n_nodes} nodes, bond_dim={bond_dim}")
    
    # Compute Shapley values
    print(f"\nComputing Shapley values with m={args.m} interpolation nodes...")
    phi = compute_diagonal_shapley(tn, n_nodes=n_nodes, m=args.m, device=device, verbose=True)
    
    # Verify efficiency
    print("\nVerifying efficiency axiom...")
    eff_result = verify_efficiency(phi, tn, device=device)
    
    print(f"\nEfficiency check: {'PASSED' if eff_result['passed'] else 'FAILED'}")
    print(f"  Σφ = {eff_result['actual_sum']:.6f}")
    print(f"  v(N) - v(∅) = {eff_result['expected_sum']:.6f}")
    print(f"  Difference = {eff_result['difference']:.6e}")
    
    # Print Shapley values
    print(f"\nShapley values:")
    for i, phi_i in enumerate(phi):
        print(f"  φ[{i}] = {phi_i:+.6f}")
    
    print(f"\nSum: {np.sum(phi):.6f}")
    
    # Save output
    output_path = args.output or args.model.replace(".pt", "_shapley.json")
    results = {
        "shapley_values": phi.tolist(),
        "n_nodes": n_nodes,
        "m_interpolation_nodes": args.m,
        "efficiency_check": eff_result,
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
