#!/usr/bin/env python3
"""
Compute O2 Shapley interaction indices from trained TN surrogate.

Example:
    python compute_interactions.py --model outputs/tn_surrogate.pt
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
    compute_o2_interactions,
    set_seed,
)
from tnshapg.interactions import interaction_matrix_to_dict


def main():
    parser = argparse.ArgumentParser(description="Compute O2 interactions from TN surrogate")
    
    parser.add_argument("--model", type=str, required=True, help="Path to trained TN model")
    parser.add_argument("--m", type=int, default=16, help="Number of interpolation nodes")
    parser.add_argument("--edges_only", action="store_true", help="Compute only for graph edges")
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
    
    # Determine which pairs to compute
    if args.edges_only and graph_edges:
        edge_pairs = [(min(u, v), max(u, v)) for u, v in graph_edges]
        edge_pairs = list(set(edge_pairs))
        print(f"Computing O2 for {len(edge_pairs)} graph edges only")
    else:
        edge_pairs = None
        n_pairs = n_nodes * (n_nodes - 1) // 2
        print(f"Computing O2 for all {n_pairs} pairs")
    
    # Compute O2 interactions
    print(f"\nComputing O2 interactions with m={args.m} interpolation nodes...")
    phi2 = compute_o2_interactions(
        tn,
        n_nodes=n_nodes,
        edge_pairs=edge_pairs,
        m=args.m,
        device=device,
        verbose=True,
    )
    
    # Print top interactions
    print("\nTop 10 interactions (by magnitude):")
    phi2_dict = interaction_matrix_to_dict(phi2, edge_pairs=edge_pairs)
    sorted_items = sorted(phi2_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    for key, val in sorted_items[:10]:
        print(f"  φ[{key}] = {val:+.6f}")
    
    # Save output
    output_path = args.output or args.model.replace(".pt", "_o2_interactions.json")
    results = {
        "interactions": phi2_dict,
        "matrix": phi2.tolist(),
        "n_nodes": n_nodes,
        "m_interpolation_nodes": args.m,
        "edges_only": args.edges_only,
        "n_pairs_computed": len(phi2_dict),
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
