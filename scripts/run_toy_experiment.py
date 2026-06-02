#!/usr/bin/env python3
"""
Run complete TN-SHAP-G toy experiment: one-command reproducibility.

This script demonstrates the full TN-SHAP-G pipeline on a small synthetic graph:
1. Create synthetic graph with n~12 nodes
2. Create MLP teacher model
3. Train TN surrogate
4. Compute deterministic Shapley values via diagonal derivative
5. Compute O2 interactions on edges
6. Compare against exact enumeration (sanity check)
7. Report metrics: R², cosine similarity, max error

Expected runtime: ~3-5 minutes on CPU

Example:
    cd TN-SHAP-G
    python scripts/run_toy_experiment.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tnshapg import (
    # Core components
    GraphAlignedTN,
    MaskedGame,
    MLPTeacher,
    # Training
    train_surrogate,
    TrainingConfig,
    # Shapley computation
    compute_diagonal_shapley,
    compute_o2_interactions,
    # Exact enumeration
    exact_shapley_fast,
    efficiency_check,
    # O2 exact
    compute_o2_exact,
    # Utilities
    set_seed,
    cosine_similarity,
    max_abs_error,
)
from tnshapg.training import compute_surrogate_r2


def create_toy_graph(n_nodes: int = 12, seed: int = 42):
    """Create a small synthetic graph for the toy demo."""
    import networkx as nx
    
    np.random.seed(seed)
    
    # Use a random tree (guaranteed connected, simple structure)
    G = nx.random_tree(n_nodes, seed=seed)
    
    # Add a few extra edges to make it more interesting
    # but keep it sparse for tractable exact enumeration
    nodes = list(G.nodes())
    n_extra = min(3, n_nodes // 4)
    for _ in range(n_extra):
        u, v = np.random.choice(nodes, 2, replace=False)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
    
    return G


def main():
    print("=" * 60)
    print("TN-SHAP-G Toy Experiment")
    print("=" * 60)
    
    # Configuration
    N_NODES = 12          # Small enough for exact enumeration
    N_FEATURES = 7        # Feature dimension
    BOND_DIM = 4          # TN bond dimension
    N_TRAIN_SAMPLES = 500 # Training coalitions
    N_VAL_SAMPLES = 100   # Validation coalitions
    EPOCHS = 100          # Training epochs
    M_INTERP = 16         # Interpolation nodes for Shapley
    SEED = 42
    
    # Output directory
    output_dir = Path(__file__).parent.parent / "outputs" / "toy_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    set_seed(SEED)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # =========================================================================
    # Step 1: Create synthetic graph
    # =========================================================================
    print(f"\n[Step 1] Creating synthetic graph with {N_NODES} nodes...")
    start_time = time.time()
    
    G = create_toy_graph(N_NODES, seed=SEED)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create random node features
    X = torch.randn(N_NODES, N_FEATURES, dtype=torch.float32)
    baseline = torch.zeros_like(X)  # Zero baseline
    
    print(f"  Features: X shape = {X.shape}")
    print(f"  Time: {time.time() - start_time:.2f}s")
    
    # =========================================================================
    # Step 2: Create teacher model
    # =========================================================================
    print(f"\n[Step 2] Creating MLP teacher model...")
    start_time = time.time()
    
    teacher = MLPTeacher(
        n_features=N_FEATURES,
        hidden_dims=[32, 16],
        aggregation="sum",
        seed=SEED,
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
    
    # Check game values
    v_empty = game.empty_coalition_value()
    v_full = game.grand_coalition_value()
    print(f"  v(∅) = {v_empty:.4f}")
    print(f"  v(N) = {v_full:.4f}")
    print(f"  Time: {time.time() - start_time:.2f}s")
    
    # =========================================================================
    # Step 3: Train TN surrogate
    # =========================================================================
    print(f"\n[Step 3] Training TN surrogate...")
    print(f"  Bond dimension: {BOND_DIM}")
    print(f"  Training samples: {N_TRAIN_SAMPLES}")
    start_time = time.time()
    
    tn = GraphAlignedTN(
        n_nodes=N_NODES,
        graph_structure=G,
        bond_dim=BOND_DIM,
        seed=SEED,
    )
    print(f"  TN parameters: {tn.get_num_parameters()}")
    
    train_config = TrainingConfig(
        lr=0.01,
        epochs=EPOCHS,
        batch_size=32,
        n_train_samples=N_TRAIN_SAMPLES,
        n_val_samples=N_VAL_SAMPLES,
        patience=15,
        seed=SEED,
        verbose=False,  # Suppress per-epoch output
    )
    
    result = train_surrogate(tn, game, config=train_config, device=device)
    
    print(f"  Epochs trained: {result.n_epochs}")
    print(f"  Best validation loss: {result.best_val_loss:.6f}")
    print(f"  Final train R²: {result.final_train_r2:.4f}")
    print(f"  Final val R²: {result.final_val_r2:.4f}")
    print(f"  Time: {time.time() - start_time:.2f}s")
    
    # Test on held-out samples
    test_metrics = compute_surrogate_r2(tn, game, n_samples=100, seed=SEED + 1000, device=device)
    print(f"  Test R²: {test_metrics['r2']:.4f}")
    
    # =========================================================================
    # Step 4: Compute TN-SHAP-G Shapley values
    # =========================================================================
    print(f"\n[Step 4] Computing TN-SHAP-G Shapley values...")
    print(f"  Interpolation nodes: {M_INTERP}")
    start_time = time.time()
    
    phi_tnshap = compute_diagonal_shapley(
        tn, n_nodes=N_NODES, m=M_INTERP, device=device, verbose=False
    )
    
    print(f"  Time: {time.time() - start_time:.2f}s")
    print(f"  Shapley values: min={phi_tnshap.min():.4f}, max={phi_tnshap.max():.4f}")
    
    # =========================================================================
    # Step 5: Compute exact Shapley values (ground truth)
    # =========================================================================
    print(f"\n[Step 5] Computing exact Shapley values via enumeration...")
    print(f"  (This may take a moment for 2^{N_NODES} = {2**N_NODES} coalitions)")
    start_time = time.time()
    
    def game_value_fn(S):
        """Query game value for a set S."""
        return game.query_coalition_set(S)
    
    phi_exact, coalition_values = exact_shapley_fast(game_value_fn, N_NODES, verbose=False)
    
    print(f"  Time: {time.time() - start_time:.2f}s")
    
    # Verify efficiency of exact values
    eff_check = efficiency_check(phi_exact, values=coalition_values, n=N_NODES)
    print(f"  Efficiency check: {'PASSED' if eff_check['passed'] else 'FAILED'}")
    
    # =========================================================================
    # Step 6: Compare TN-SHAP-G vs Exact
    # =========================================================================
    print(f"\n[Step 6] Comparing TN-SHAP-G vs Exact Shapley...")
    
    cos_sim = cosine_similarity(phi_tnshap, phi_exact)
    max_err = max_abs_error(phi_tnshap, phi_exact)
    mae = np.mean(np.abs(phi_tnshap - phi_exact))
    
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max absolute error: {max_err:.6f}")
    print(f"  Mean absolute error: {mae:.6f}")
    
    # Print side-by-side comparison
    print(f"\n  Node-by-node comparison:")
    print(f"  {'Node':>6} | {'TN-SHAP-G':>12} | {'Exact':>12} | {'Diff':>12}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    for i in range(N_NODES):
        diff = phi_tnshap[i] - phi_exact[i]
        print(f"  {i:>6} | {phi_tnshap[i]:>+12.6f} | {phi_exact[i]:>+12.6f} | {diff:>+12.6f}")
    
    # =========================================================================
    # Step 7: Compute O2 interactions on edges
    # =========================================================================
    print(f"\n[Step 7] Computing O2 interactions on graph edges...")
    start_time = time.time()
    
    edge_pairs = [(min(u, v), max(u, v)) for u, v in G.edges()]
    
    phi2_tnshap = compute_o2_interactions(
        tn, n_nodes=N_NODES, edge_pairs=edge_pairs, m=M_INTERP, device=device, verbose=False
    )
    
    print(f"  Computed {len(edge_pairs)} edge interactions")
    print(f"  Time: {time.time() - start_time:.2f}s")
    
    # =========================================================================
    # Step 8: Compute exact O2 on edges (for comparison)
    # =========================================================================
    print(f"\n[Step 8] Computing exact O2 interactions on edges...")
    start_time = time.time()
    
    phi2_exact = compute_o2_exact(game_value_fn, N_NODES, edge_pairs=edge_pairs, verbose=False)
    
    print(f"  Time: {time.time() - start_time:.2f}s")
    
    # Compare O2
    phi2_tnshap_flat = np.array([phi2_tnshap[u, v] for u, v in edge_pairs])
    phi2_exact_flat = np.array([phi2_exact[u, v] for u, v in edge_pairs])
    
    cos_sim_o2 = cosine_similarity(phi2_tnshap_flat, phi2_exact_flat)
    max_err_o2 = max_abs_error(phi2_tnshap_flat, phi2_exact_flat)
    
    print(f"  O2 Cosine similarity: {cos_sim_o2:.6f}")
    print(f"  O2 Max absolute error: {max_err_o2:.6f}")
    
    # =========================================================================
    # Save results
    # =========================================================================
    print(f"\n[Saving results to {output_dir}]")
    
    results = {
        "config": {
            "n_nodes": N_NODES,
            "n_features": N_FEATURES,
            "bond_dim": BOND_DIM,
            "n_train_samples": N_TRAIN_SAMPLES,
            "m_interpolation": M_INTERP,
            "seed": SEED,
        },
        "training": {
            "epochs_trained": result.n_epochs,
            "best_val_loss": result.best_val_loss,
            "final_train_r2": result.final_train_r2,
            "final_val_r2": result.final_val_r2,
            "test_r2": test_metrics["r2"],
        },
        "shapley_o1": {
            "tnshap": phi_tnshap.tolist(),
            "exact": phi_exact.tolist(),
            "cosine_similarity": cos_sim,
            "max_abs_error": max_err,
            "mean_abs_error": mae,
        },
        "shapley_o2": {
            "n_edges": len(edge_pairs),
            "cosine_similarity": cos_sim_o2,
            "max_abs_error": max_err_o2,
        },
    }
    
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save model
    model_path = output_dir / "tn_surrogate.pt"
    torch.save({
        "state_dict": tn.state_dict(),
        "n_nodes": N_NODES,
        "bond_dim": BOND_DIM,
        "y_mean": tn.y_mean,
        "y_std": tn.y_std,
        "graph_edges": list(G.edges()),
    }, model_path)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Graph: {N_NODES} nodes, {G.number_of_edges()} edges")
    print(f"  TN parameters: {tn.get_num_parameters()}")
    print(f"  Surrogate test R²: {test_metrics['r2']:.4f}")
    print(f"  O1 Shapley cosine similarity: {cos_sim:.6f}")
    print(f"  O2 Interaction cosine similarity: {cos_sim_o2:.6f}")
    print(f"\n  Results saved to: {output_dir}")
    print("=" * 60)
    
    # Return success code based on quality
    if cos_sim > 0.95 and test_metrics['r2'] > 0.9:
        print("\n✓ Experiment PASSED: High-quality Shapley approximation achieved!")
        return 0
    else:
        print("\n⚠ Experiment completed but with lower quality than expected.")
        print("  Consider: increasing training samples, bond dimension, or epochs.")
        return 1


if __name__ == "__main__":
    exit(main())
