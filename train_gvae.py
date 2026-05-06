

"""Training utilities for the graph-level VAE in Mini-project 3.

This module handles end-to-end training and validation of the deep generative
model defined in `model_gvae.py`.

Main responsibilities:
- build reproducible dataloaders from the matrix-based MUTAG dataset,
- collate `GraphSample` objects into batch tensors,
- train the graph-level VAE using an ELBO-style objective,
- evaluate on a validation split,
- save checkpoints, configs, and training history.

The code is intentionally compact and easy to reuse from `run_project.py`.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from logging_config import get_logger
from data import DataBundle, GraphSample, build_data_bundle
from model_gvae import GraphLevelVAE, GraphVAEConfig, build_graph_vae
from utils_io import ensure_parent_dir, get_device, save_json


# Initialize module logger
logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_TRAIN_SEED = 42
DEFAULT_DEVICE = get_device()
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_BETA = 1.0
DEFAULT_BETA_START = 0.0
DEFAULT_BETA_WARMUP_EPOCHS = 50
DEFAULT_GRAD_CLIP_NORM = 5.0
DEFAULT_USE_EARLY_STOPPING = True
DEFAULT_PATIENCE = 30

DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_CHECKPOINT_DIR = DEFAULT_OUTPUT_DIR / "checkpoints"
DEFAULT_HISTORY_PATH = DEFAULT_OUTPUT_DIR / "tables" / "gvae_train_history.json"
DEFAULT_TRAIN_CONFIG_PATH = DEFAULT_OUTPUT_DIR / "tables" / "gvae_train_config.json"
DEFAULT_MODEL_PATH = DEFAULT_CHECKPOINT_DIR / "best_gvae.pt"


# -----------------------------------------------------------------------------
# Config containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """Configuration for training the graph-level VAE."""

    seed: int = DEFAULT_TRAIN_SEED
    device: str = DEFAULT_DEVICE
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    beta: float = DEFAULT_BETA
    beta_start: float = DEFAULT_BETA_START
    beta_warmup_epochs: int = DEFAULT_BETA_WARMUP_EPOCHS
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM
    use_early_stopping: bool = DEFAULT_USE_EARLY_STOPPING
    patience: int = DEFAULT_PATIENCE
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    checkpoint_dir: str = str(DEFAULT_CHECKPOINT_DIR)
    history_path: str = str(DEFAULT_HISTORY_PATH)
    train_config_path: str = str(DEFAULT_TRAIN_CONFIG_PATH)
    model_path: str = str(DEFAULT_MODEL_PATH)

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative.")
        if self.beta_start < 0.0:
            raise ValueError("beta_start must be non-negative.")
        if self.beta_warmup_epochs < 0:
            raise ValueError("beta_warmup_epochs must be non-negative.")
        if self.grad_clip_norm < 0.0:
            raise ValueError("grad_clip_norm must be non-negative.")
        if self.patience < 0:
            raise ValueError("patience must be non-negative.")
        if not self.output_dir:
            raise ValueError("output_dir must be a non-empty string.")
        if not self.model_path:
            raise ValueError("model_path must be a non-empty string.")


# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)  # type: ignore[attr-defined]
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Collate function
# -----------------------------------------------------------------------------


def collate_graph_samples(batch: list[GraphSample]) -> dict[str, torch.Tensor]:
    """Collate `GraphSample` objects into a mini-batch.

    Returns:
        Dictionary with keys:
        - adjacency: (B, N, N)
        - node_features: (B, N, F)
        - node_mask: (B, N)
        - num_nodes: (B,)
        - labels: (B,)
        - graph_indices: (B,)
    """
    adjacency = torch.stack([sample.adjacency for sample in batch], dim=0)
    node_features = torch.stack([sample.node_features for sample in batch], dim=0)
    node_mask = torch.stack([sample.node_mask for sample in batch], dim=0)
    num_nodes = torch.tensor([sample.num_nodes for sample in batch], dtype=torch.long)
    labels = torch.tensor([sample.label for sample in batch], dtype=torch.long)
    graph_indices = torch.tensor([sample.graph_index for sample in batch], dtype=torch.long)

    return {
        "adjacency": adjacency,
        "node_features": node_features,
        "node_mask": node_mask,
        "num_nodes": num_nodes,
        "labels": labels,
        "graph_indices": graph_indices,
    }


# -----------------------------------------------------------------------------
# Dataloader helpers
# -----------------------------------------------------------------------------


def build_dataloaders(data_bundle: DataBundle, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test dataloaders."""
    train_loader = DataLoader(
        data_bundle.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graph_samples,
    )
    val_loader = DataLoader(
        data_bundle.val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graph_samples,
    )
    test_loader = DataLoader(
        data_bundle.test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graph_samples,
    )
    return train_loader, val_loader, test_loader


# -----------------------------------------------------------------------------
# Batch utilities
# -----------------------------------------------------------------------------


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move all tensor values in a batch dictionary to the target device."""
    return {key: value.to(device) for key, value in batch.items()}


# -----------------------------------------------------------------------------
# Epoch routines
# -----------------------------------------------------------------------------


def train_one_epoch(
    model: GraphLevelVAE,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
    grad_clip_norm: float | None = None,
) -> dict[str, float]:
    """Train the model for one epoch and return averaged metrics."""
    model.train()

    totals = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "kl_loss": 0.0,
    }
    n_graphs = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for batch in progress_bar:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.loss(
            node_features=batch["node_features"],
            adjacency=batch["adjacency"],
            node_mask=batch["node_mask"],
            beta=beta,
        )
        loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        batch_size = batch["adjacency"].shape[0]
        n_graphs += batch_size
        for key in totals:
            totals[key] += metrics[key] * batch_size

        progress_bar.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            recon=f"{metrics['recon_loss']:.4f}",
            kl=f"{metrics['kl_loss']:.4f}",
        )

    return {f"train_{key}": value / max(n_graphs, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_one_epoch(
    model: GraphLevelVAE,
    data_loader: DataLoader,
    device: torch.device,
    beta: float,
    split_name: str = "val",
) -> dict[str, float]:
    """Evaluate the model on a given split and return averaged metrics."""
    model.eval()

    totals = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "kl_loss": 0.0,
    }
    n_graphs = 0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        _, metrics = model.loss(
            node_features=batch["node_features"],
            adjacency=batch["adjacency"],
            node_mask=batch["node_mask"],
            beta=beta,
        )

        batch_size = batch["adjacency"].shape[0]
        n_graphs += batch_size
        for key in totals:
            totals[key] += metrics[key] * batch_size

    return {f"{split_name}_{key}": value / max(n_graphs, 1) for key, value in totals.items()}


# -----------------------------------------------------------------------------
# Main training API
# -----------------------------------------------------------------------------


def train_graph_vae(
    model_config: GraphVAEConfig,
    train_config: TrainConfig | None = None,
    data_bundle: DataBundle | None = None,
) -> tuple[GraphLevelVAE, list[dict[str, float]], DataBundle]:
    """Train the graph-level VAE and return the best model.

    Args:
        model_config: Configuration of the graph-level VAE.
        train_config: Training configuration. Uses defaults if omitted.
        data_bundle: Optionally provide a pre-built data bundle.

    Returns:
        best_model: Model loaded with the best validation checkpoint.
        history: List of epoch-level metric dictionaries.
        data_bundle: The data bundle used for training.
    """
    train_config = train_config if train_config is not None else TrainConfig()
    set_seed(train_config.seed)

    device = torch.device(train_config.device)
    data_bundle = data_bundle if data_bundle is not None else build_data_bundle(seed=train_config.seed)
    train_loader, val_loader, _ = build_dataloaders(data_bundle, batch_size=train_config.batch_size)

    model = build_graph_vae(model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    model_path = Path(train_config.model_path)
    history_path = Path(train_config.history_path)
    train_config_path = Path(train_config.train_config_path)
    ensure_parent_dir(model_path)
    ensure_parent_dir(history_path)
    ensure_parent_dir(train_config_path)

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    logger.info("Graph-level VAE training setup")
    logger.info(f"Device: {train_config.device}")
    logger.info(f"Epochs: {train_config.epochs}")
    logger.info(f"Batch size: {train_config.batch_size}")
    logger.info(f"Learning rate: {train_config.learning_rate}")
    logger.info(f"Beta: {train_config.beta}")
    logger.info(f"Train/Val sizes: {len(data_bundle.train_dataset)}/{len(data_bundle.val_dataset)}")
    logger.info(f"Max nodes: {data_bundle.max_nodes}")
    logger.info(f"Num node features: {data_bundle.num_node_features}")

    for epoch in range(train_config.epochs):
        # Linear KL annealing: ramp beta from beta_start to beta over the first
        # beta_warmup_epochs epochs, then hold at beta.
        if train_config.beta_warmup_epochs > 0:
            progress = epoch / train_config.beta_warmup_epochs
            current_beta = train_config.beta_start + (train_config.beta - train_config.beta_start) * min(progress, 1.0)
        else:
            current_beta = train_config.beta

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            beta=current_beta,
            grad_clip_norm=train_config.grad_clip_norm,
        )
        val_metrics = evaluate_one_epoch(
            model=model,
            data_loader=val_loader,
            device=device,
            beta=current_beta,
            split_name="val",
        )

        epoch_metrics = {"epoch": epoch + 1, "beta": round(current_beta, 6), **train_metrics, **val_metrics}
        history.append(epoch_metrics)

        logger.info(
            f"Epoch {epoch + 1:03d}/{train_config.epochs:03d} | "
            f"train_loss={epoch_metrics['train_loss']:.4f} | "
            f"val_loss={epoch_metrics['val_loss']:.4f} | "
            f"train_recon={epoch_metrics['train_recon_loss']:.4f} | "
            f"val_recon={epoch_metrics['val_recon_loss']:.4f} | "
            f"train_kl={epoch_metrics['train_kl_loss']:.4f} | "
            f"val_kl={epoch_metrics['val_kl_loss']:.4f}"
        )

        if epoch_metrics["val_loss"] < best_val_loss:
            best_val_loss = epoch_metrics["val_loss"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1

        if train_config.use_early_stopping and epochs_without_improvement >= train_config.patience:
            logger.info(
                f"Early stopping triggered after epoch {epoch + 1}. "
                f"Best validation loss was {best_val_loss:.4f} at epoch {best_epoch}."
            )
            break

    save_json({"history": history}, history_path)
    save_json(
        {
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        },
        train_config_path,
    )

    logger.info(f"Best model saved to: {model_path}")
    logger.info(f"Training history saved to: {history_path}")
    logger.info(f"Training config saved to: {train_config_path}")

    # Reload best checkpoint before returning.
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    logger.info("Training complete.")

    return model, history, data_bundle


# -----------------------------------------------------------------------------
# Quick manual check
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    data_bundle = build_data_bundle(seed=DEFAULT_TRAIN_SEED)
    model_config = GraphVAEConfig(
        num_node_features=data_bundle.num_node_features,
        max_nodes=data_bundle.max_nodes,
    )
    train_config = TrainConfig(epochs=10)

    model, history, _ = train_graph_vae(
        model_config=model_config,
        train_config=train_config,
        data_bundle=data_bundle,
    )

    print("Training finished.")
    print(f"Number of logged epochs: {len(history)}")