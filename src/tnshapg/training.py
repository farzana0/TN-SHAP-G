"""
Training: TN surrogate training loop.

Trains the GraphAlignedTN to approximate the masked game by minimizing
MSE loss on sampled coalitions. The training loop includes:
- Coalition sampling (balanced by size)
- MSE loss computation
- Early stopping based on validation loss
- Learning rate scheduling
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .utils import set_seed, r2_score
from .coalition_sampler import BalancedSizeSampler


@dataclass
class TrainingConfig:
    """Configuration for TN surrogate training."""
    
    # Optimization
    lr: float = 0.01
    weight_decay: float = 1e-5
    epochs: int = 100
    batch_size: int = 32
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-5
    
    # Learning rate scheduling
    lr_scheduler: str = "plateau"  # "plateau", "cosine", "none"
    lr_factor: float = 0.5
    lr_patience: int = 5
    
    # Data
    n_train_samples: int = 500
    n_val_samples: int = 100
    val_fraction: float = 0.2
    
    # Reproducibility
    seed: int = 42
    
    # Logging
    verbose: bool = True
    log_every: int = 10


@dataclass
class TrainingResult:
    """Results from training."""
    
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    final_train_r2: float = 0.0
    final_val_r2: float = 0.0
    n_epochs: int = 0


def train_surrogate(
    tn: nn.Module,
    game: Callable,
    config: Optional[TrainingConfig] = None,
    device: Optional[torch.device] = None,
) -> TrainingResult:
    """
    Train TN surrogate to approximate the masked game.
    
    The training process:
    1. Sample coalitions with balanced size distribution
    2. Query game values for all coalitions
    3. Train TN to minimize MSE on (coalition, value) pairs
    4. Use early stopping based on validation loss
    
    Args:
        tn: GraphAlignedTN instance
        game: MaskedGame instance or callable (coalition -> value)
        config: Training configuration
        device: Torch device
        
    Returns:
        TrainingResult with loss history and metrics
    """
    if config is None:
        config = TrainingConfig()
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seed(config.seed)
    
    # Move TN to device
    tn = tn.to(device)
    
    # Get number of nodes from TN
    n_nodes = tn.n_nodes
    
    # Sample training and validation coalitions
    sampler = BalancedSizeSampler(n_nodes, seed=config.seed)
    
    n_total = config.n_train_samples + config.n_val_samples
    all_coalitions = sampler.sample(n_total)
    
    # Query game values
    if config.verbose:
        print(f"Querying {n_total} game values...")
    
    all_values = _query_game_values(game, all_coalitions, device)
    
    # Split into train/val
    train_coalitions = all_coalitions[:config.n_train_samples].to(device)
    train_values = all_values[:config.n_train_samples].to(device)
    val_coalitions = all_coalitions[config.n_train_samples:].to(device)
    val_values = all_values[config.n_train_samples:].to(device)
    
    # Normalize targets for stable training
    y_mean = train_values.mean().item()
    y_std = train_values.std().item()
    if y_std < 1e-6:
        y_std = 1.0
    
    train_values_norm = (train_values - y_mean) / y_std
    val_values_norm = (val_values - y_mean) / y_std
    
    # Store normalization constants on TN for later use
    tn.y_mean = y_mean
    tn.y_std = y_std
    
    if config.verbose:
        print(f"Training data: {len(train_coalitions)} samples")
        print(f"Validation data: {len(val_coalitions)} samples")
        print(f"Target mean: {y_mean:.4f}, std: {y_std:.4f}")
    
    # Setup optimizer
    optimizer = optim.Adam(
        tn.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    
    # Setup scheduler
    if config.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.lr_factor,
            patience=config.lr_patience,
            verbose=config.verbose,
        )
    elif config.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    else:
        scheduler = None
    
    # Training loop
    result = TrainingResult()
    best_state = None
    patience_counter = 0
    
    for epoch in range(config.epochs):
        # Training step
        train_loss = _train_epoch(
            tn, train_coalitions, train_values_norm,
            optimizer, config.batch_size, device,
        )
        result.train_losses.append(train_loss)
        
        # Validation step
        val_loss = _validate(tn, val_coalitions, val_values_norm, device)
        result.val_losses.append(val_loss)
        
        # Learning rate scheduling
        if scheduler is not None:
            if config.lr_scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Early stopping check
        if val_loss < result.best_val_loss - config.min_delta:
            result.best_val_loss = val_loss
            result.best_epoch = epoch
            best_state = {k: v.clone() for k, v in tn.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Logging
        if config.verbose and (epoch + 1) % config.log_every == 0:
            print(
                f"Epoch {epoch + 1:3d}/{config.epochs}: "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
            )
        
        # Early stopping
        if patience_counter >= config.patience:
            if config.verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break
    
    result.n_epochs = epoch + 1
    
    # Restore best model
    if best_state is not None:
        tn.load_state_dict(best_state)
    
    # Compute final R² scores
    with torch.no_grad():
        train_pred = tn.contract_batch(train_coalitions, detach_inputs=True)
        val_pred = tn.contract_batch(val_coalitions, detach_inputs=True)
        
        result.final_train_r2 = r2_score(
            train_values_norm.cpu().numpy(),
            train_pred.cpu().numpy(),
        )
        result.final_val_r2 = r2_score(
            val_values_norm.cpu().numpy(),
            val_pred.cpu().numpy(),
        )
    
    if config.verbose:
        print(f"\nTraining complete!")
        print(f"Best epoch: {result.best_epoch + 1}")
        print(f"Best val loss: {result.best_val_loss:.6f}")
        print(f"Final train R²: {result.final_train_r2:.4f}")
        print(f"Final val R²: {result.final_val_r2:.4f}")
    
    return result


def _query_game_values(
    game: Callable,
    coalitions: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Query game values for all coalitions."""
    values = []
    
    for i in range(len(coalitions)):
        coalition = coalitions[i].to(device)
        value = game(coalition)
        if isinstance(value, torch.Tensor):
            value = value.item()
        values.append(value)
    
    return torch.tensor(values, dtype=torch.float32)


def _train_epoch(
    tn: nn.Module,
    coalitions: torch.Tensor,
    values: torch.Tensor,
    optimizer: optim.Optimizer,
    batch_size: int,
    device: torch.device,
) -> float:
    """Single training epoch."""
    tn.train()
    
    n_samples = len(coalitions)
    indices = torch.randperm(n_samples)
    
    total_loss = 0.0
    n_batches = 0
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        
        batch_z = coalitions[batch_idx]
        batch_y = values[batch_idx]
        
        optimizer.zero_grad()
        
        # Forward pass
        pred = tn.contract_batch(batch_z, detach_inputs=False)
        
        # MSE loss
        loss = nn.functional.mse_loss(pred, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def _validate(
    tn: nn.Module,
    coalitions: torch.Tensor,
    values: torch.Tensor,
    device: torch.device,
) -> float:
    """Compute validation loss."""
    tn.eval()
    
    with torch.no_grad():
        pred = tn.contract_batch(coalitions, detach_inputs=True)
        loss = nn.functional.mse_loss(pred, values)
    
    return loss.item()


def compute_surrogate_r2(
    tn: nn.Module,
    game: Callable,
    n_samples: int = 100,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate surrogate quality on held-out samples.
    
    Args:
        tn: Trained GraphAlignedTN
        game: MaskedGame instance
        n_samples: Number of test samples
        seed: Random seed
        device: Torch device
        
    Returns:
        Dictionary with R², MSE, and MAE metrics
    """
    if device is None:
        device = next(tn.parameters()).device
    
    set_seed(seed + 1000)  # Different seed from training
    
    n_nodes = tn.n_nodes
    sampler = BalancedSizeSampler(n_nodes, seed=seed + 1000)
    
    test_coalitions = sampler.sample(n_samples).to(device)
    test_values = _query_game_values(game, test_coalitions, device).to(device)
    
    # Get normalization from training
    y_mean = getattr(tn, 'y_mean', 0.0)
    y_std = getattr(tn, 'y_std', 1.0)
    
    test_values_norm = (test_values - y_mean) / y_std
    
    with torch.no_grad():
        pred = tn.contract_batch(test_coalitions, detach_inputs=True)
    
    pred_np = pred.cpu().numpy()
    true_np = test_values_norm.cpu().numpy()
    
    mse = float(np.mean((pred_np - true_np) ** 2))
    mae = float(np.mean(np.abs(pred_np - true_np)))
    r2 = r2_score(true_np, pred_np)
    
    return {
        "r2": r2,
        "mse": mse,
        "mae": mae,
        "n_samples": n_samples,
    }
