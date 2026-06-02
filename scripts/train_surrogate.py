#!/usr/bin/env python3
"""
Train TN surrogate from command line.

Example:
    python train_surrogate.py --config configs/toy_demo.yaml
    python train_surrogate.py --n_nodes 10 --n_samples 500 --bond_dim 4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tnshapg import (
    GraphAlignedTN,
    MaskedGame,
    MLPTeacher,
    train_surrogate,
    TrainingConfig,
    set_seed,
)
from tnshapg.utils import edge_index_to_networkx


def create_synthetic_graph(n_nodes: int, edge_prob: float = 0.3, seed: int = 42):
    """Create a random synthetic graph for testing."""
    import networkx as nx
    
    np.random.seed(seed)
    
    # Create Erdos-Renyi graph
    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)
    
    # Ensure connected
    if not nx.is_connected(G):
        # Connect components
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = list(components[i])[0]
            v = list(components[i + 1])[0]
            G.add_edge(u, v)
    
    return G


def main():
    parser = argparse.ArgumentParser(description="Train TN-SHAP-G surrogate")
    
    # Config file (overrides command line args)
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    
    # Graph parameters
    parser.add_argument("--n_nodes", type=int, default=10, help="Number of nodes")
    parser.add_argument("--n_features", type=int, default=7, help="Feature dimension")
    parser.add_argument("--edge_prob", type=float, default=0.3, help="Edge probability for random graph")
    
    # TN parameters
    parser.add_argument("--bond_dim", type=int, default=4, help="TN bond dimension")
    
    # Training parameters
    parser.add_argument("--n_samples", type=int, default=500, help="Training samples")
    parser.add_argument("--n_val_samples", type=int, default=100, help="Validation samples")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        # Override with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create synthetic graph
    print(f"\nCreating synthetic graph with {args.n_nodes} nodes...")
    G = create_synthetic_graph(args.n_nodes, args.edge_prob, args.seed)
    print(f"Graph has {G.number_of_edges()} edges")
    
    # Create random node features
    X = torch.randn(args.n_nodes, args.n_features, dtype=torch.float32)
    baseline = torch.zeros_like(X)  # Zero baseline
    
    # Create teacher model
    print("\nCreating MLP teacher model...")
    teacher = MLPTeacher(
        n_features=args.n_features,
        hidden_dims=[32, 16],
        aggregation="sum",
        seed=args.seed,
    )
    teacher = teacher.to(device)
    teacher.eval()
    
    # Create masked game
    game = MaskedGame(
        X=X,
        baseline=baseline,
        teacher=teacher,
        device=device,
    )
    
    # Create graph-aligned TN
    print(f"\nCreating GraphAlignedTN with bond_dim={args.bond_dim}...")
    tn = GraphAlignedTN(
        n_nodes=args.n_nodes,
        graph_structure=G,
        bond_dim=args.bond_dim,
        seed=args.seed,
    )
    print(f"TN has {tn.get_num_parameters()} parameters")
    
    # Training config
    train_config = TrainingConfig(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_train_samples=args.n_samples,
        n_val_samples=args.n_val_samples,
        seed=args.seed,
        verbose=True,
    )
    
    # Train
    print("\nTraining TN surrogate...")
    result = train_surrogate(tn, game, config=train_config, device=device)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(args.output_dir, "tn_surrogate.pt")
    torch.save({
        "state_dict": tn.state_dict(),
        "n_nodes": args.n_nodes,
        "bond_dim": args.bond_dim,
        "y_mean": tn.y_mean,
        "y_std": tn.y_std,
        "graph_edges": list(G.edges()),
    }, model_path)
    print(f"\nSaved model to {model_path}")
    
    # Save training results
    results_path = os.path.join(args.output_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "final_train_r2": result.final_train_r2,
            "final_val_r2": result.final_val_r2,
            "n_epochs": result.n_epochs,
        }, f, indent=2)
    print(f"Saved results to {results_path}")
    
    print(f"\n{'='*50}")
    print("Training complete!")
    print(f"Final validation R²: {result.final_val_r2:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
