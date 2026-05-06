

"""Graph generation utilities for Mini-project 3.

This module contains the sampling pipeline for the deep generative model.
It is responsible for:
- sampling graph sizes from the empirical training-size distribution,
- sampling latent variables from the GraphVAE prior,
- decoding adjacency probabilities or binary adjacency samples,
- trimming generated graphs to the sampled number of nodes,
- cleaning the generated adjacency matrices so they represent simple
  undirected graphs,
- saving generated samples for later evaluation.

The module is designed to work with the graph-level VAE from `model_gvae.py`
and the empirical size model from `baseline_er.py`.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from baseline_er import (
    DEFAULT_BASELINE_SEED,
    EmpiricalSizeDensityModel,
    fit_empirical_size_density_model,
    extract_train_graphs,
)
from data import DataBundle, build_data_bundle
from model_gvae import GraphLevelVAE, GraphVAEConfig, build_graph_vae
from utils_graph import clean_simple_undirected_adjacency
from utils_io import ensure_parent_dir, get_device, save_json


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_NUM_GENERATED_SAMPLES = 1000
DEFAULT_GENERATION_SEED = DEFAULT_BASELINE_SEED
DEFAULT_GENERATION_DEVICE = get_device()
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_GENERATED_SAMPLES_PATH = DEFAULT_OUTPUT_DIR / "samples" / "gvae_generated_samples.pt"
DEFAULT_GENERATION_METADATA_PATH = DEFAULT_OUTPUT_DIR / "tables" / "gvae_generation_metadata.json"
DEFAULT_MODEL_CHECKPOINT_PATH = DEFAULT_OUTPUT_DIR / "checkpoints" / "best_gvae.pt"


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GeneratedGraphSample:
    """One generated graph sample from the deep generative model.

    Attributes:
        adjacency: Binary cleaned adjacency matrix of shape (N, N).
        adjacency_probabilities: Probability matrix of shape (N, N).
        num_nodes: Number of valid nodes in the generated graph.
    """

    adjacency: torch.Tensor
    adjacency_probabilities: torch.Tensor
    num_nodes: int


# -----------------------------------------------------------------------------
# Size sampling helpers
# -----------------------------------------------------------------------------


def build_empirical_size_model(data_bundle: DataBundle) -> EmpiricalSizeDensityModel:
    """Fit the empirical graph-size model from the training split.

    We reuse the same graph-size distribution as the baseline so the comparison
    between baseline and deep model is aligned.
    """
    train_graphs = extract_train_graphs(data_bundle)
    return fit_empirical_size_density_model(train_graphs)



def sample_graph_sizes(
    size_model: EmpiricalSizeDensityModel,
    num_samples: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample graph sizes from the empirical training-size distribution."""
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")
    if len(size_model.size_values) == 0 or len(size_model.size_probabilities) == 0:
        raise ValueError("Size model must contain at least one size value and probability.")

    sampled = rng.choice(size_model.size_values, size=num_samples, p=size_model.size_probabilities)
    return [int(value) for value in sampled]


# -----------------------------------------------------------------------------
# Model loading helpers
# -----------------------------------------------------------------------------


def load_trained_gvae(
    model_config: GraphVAEConfig,
    checkpoint_path: str | Path = DEFAULT_MODEL_CHECKPOINT_PATH,
    device: str | torch.device = DEFAULT_GENERATION_DEVICE,
) -> GraphLevelVAE:
    """Load a trained graph-level VAE checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}.")

    device = torch.device(device)
    model = build_graph_vae(model_config).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Graph decoding helpers
# -----------------------------------------------------------------------------


def trim_probability_matrix(adjacency_probabilities: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Trim a padded probability matrix to its valid graph size."""
    if adjacency_probabilities.dim() != 2 or adjacency_probabilities.shape[0] != adjacency_probabilities.shape[1]:
        raise ValueError("adjacency_probabilities must be a square matrix.")
    if num_nodes < 0 or num_nodes > adjacency_probabilities.shape[0]:
        raise ValueError("num_nodes must be between 0 and the matrix dimension.")
    return adjacency_probabilities[:num_nodes, :num_nodes]


def sample_binary_adjacency(
    adjacency_probabilities: torch.Tensor,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Sample a binary adjacency matrix from Bernoulli edge probabilities.

    Args:
        adjacency_probabilities: Tensor of shape (N, N).
        rng: Optional NumPy RNG. If omitted, torch Bernoulli sampling is used.

    Returns:
        Binary adjacency matrix of shape (N, N).
    """
    if adjacency_probabilities.dim() != 2 or adjacency_probabilities.shape[0] != adjacency_probabilities.shape[1]:
        raise ValueError("adjacency_probabilities must be a square matrix.")
    if not torch.all((adjacency_probabilities >= 0.0) & (adjacency_probabilities <= 1.0)):
        raise ValueError("adjacency_probabilities must lie in [0, 1].")

    if rng is None:
        sampled = torch.bernoulli(adjacency_probabilities)
    else:
        probs_np = adjacency_probabilities.detach().cpu().numpy()
        sampled_np = rng.binomial(n=1, p=probs_np).astype(np.float32)
        sampled = torch.from_numpy(sampled_np).to(adjacency_probabilities.device)

    sampled = clean_simple_undirected_adjacency(sampled, threshold=0.5)
    return sampled


# -----------------------------------------------------------------------------
# Main generation API
# -----------------------------------------------------------------------------


def generate_graph_samples(
    model: GraphLevelVAE,
    size_model: EmpiricalSizeDensityModel,
    num_samples: int = DEFAULT_NUM_GENERATED_SAMPLES,
    seed: int = DEFAULT_GENERATION_SEED,
    device: str | torch.device = DEFAULT_GENERATION_DEVICE,
    sample_binary: bool = True,
) -> list[GeneratedGraphSample]:
    """Generate graph samples from the trained graph-level VAE.

    Args:
        model: Trained graph-level VAE.
        size_model: Empirical graph-size model fit on the training split.
        num_samples: Number of graphs to generate.
        seed: Random seed for reproducible sampling.
        device: Torch device.
        sample_binary: If True, sample Bernoulli edges. If False, threshold at 0.5.

    Returns:
        List of generated graph samples.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")

    device = torch.device(device)
    rng = np.random.default_rng(seed)

    sampled_sizes = sample_graph_sizes(size_model=size_model, num_samples=num_samples, rng=rng)

    # Pass the sampled sizes to the decoder so it can condition on graph size.
    sampled_sizes_tensor = torch.tensor(sampled_sizes, dtype=torch.float32, device=device)
    adjacency_probabilities = model.sample_adjacency_probs(
        num_samples=num_samples,
        num_nodes=sampled_sizes_tensor,
        device=device,
    )

    generated_samples: list[GeneratedGraphSample] = []

    for idx in range(num_samples):
        num_nodes = sampled_sizes[idx]
        prob_matrix = trim_probability_matrix(adjacency_probabilities[idx], num_nodes=num_nodes)

        if sample_binary:
            binary_adjacency = sample_binary_adjacency(prob_matrix, rng=rng)
        else:
            binary_adjacency = clean_simple_undirected_adjacency(prob_matrix, threshold=0.5)

        generated_samples.append(
            GeneratedGraphSample(
                adjacency=binary_adjacency.detach().cpu(),
                adjacency_probabilities=prob_matrix.detach().cpu(),
                num_nodes=num_nodes,
            )
        )

    return generated_samples


# -----------------------------------------------------------------------------
# Serialization helpers
# -----------------------------------------------------------------------------


def save_generated_samples(
    samples: list[GeneratedGraphSample],
    output_path: str | Path = DEFAULT_GENERATED_SAMPLES_PATH,
) -> None:
    """Save generated graph samples to disk using torch.save."""
    output_path = Path(output_path)
    ensure_parent_dir(output_path)

    serializable_samples = [
        {
            "adjacency": sample.adjacency,
            "adjacency_probabilities": sample.adjacency_probabilities,
            "num_nodes": sample.num_nodes,
        }
        for sample in samples
    ]
    torch.save(serializable_samples, output_path)



def save_generation_metadata(
    size_model: EmpiricalSizeDensityModel,
    num_samples: int,
    metadata_path: str | Path = DEFAULT_GENERATION_METADATA_PATH,
) -> None:
    """Save generation metadata as JSON."""
    metadata_path = Path(metadata_path)
    payload = {
        "num_samples": num_samples,
        "size_values": size_model.size_values.tolist(),
        "size_probabilities": size_model.size_probabilities.tolist(),
    }
    save_json(payload, metadata_path)


# -----------------------------------------------------------------------------
# Convenience pipeline
# -----------------------------------------------------------------------------


def generate_from_checkpoint(
    model_config: GraphVAEConfig,
    checkpoint_path: str | Path = DEFAULT_MODEL_CHECKPOINT_PATH,
    data_bundle: DataBundle | None = None,
    num_samples: int = DEFAULT_NUM_GENERATED_SAMPLES,
    seed: int = DEFAULT_GENERATION_SEED,
    device: str | torch.device = DEFAULT_GENERATION_DEVICE,
) -> tuple[list[GeneratedGraphSample], EmpiricalSizeDensityModel]:
    """Load a trained model, fit the empirical size model, and generate samples."""
    data_bundle = data_bundle if data_bundle is not None else build_data_bundle(seed=seed)
    size_model = build_empirical_size_model(data_bundle)
    model = load_trained_gvae(
        model_config=model_config,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    samples = generate_graph_samples(
        model=model,
        size_model=size_model,
        num_samples=num_samples,
        seed=seed,
        device=device,
        sample_binary=True,
    )
    return samples, size_model


# -----------------------------------------------------------------------------
# Quick manual check
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    data_bundle = build_data_bundle(seed=DEFAULT_GENERATION_SEED)
    model_config = GraphVAEConfig(
        num_node_features=data_bundle.num_node_features,
        max_nodes=data_bundle.max_nodes,
    )

    checkpoint_path = DEFAULT_MODEL_CHECKPOINT_PATH
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train the GVAE first."
        )

    samples, size_model = generate_from_checkpoint(
        model_config=model_config,
        checkpoint_path=checkpoint_path,
        data_bundle=data_bundle,
        num_samples=5,
        seed=DEFAULT_GENERATION_SEED,
        device=DEFAULT_GENERATION_DEVICE,
    )

    print("Graph generation sanity check")
    print(f"Observed size values: {size_model.size_values.tolist()}")
    for idx, sample in enumerate(samples):
        print(
            f"Sample {idx}: N={sample.num_nodes}, "
            f"adj_shape={tuple(sample.adjacency.shape)}, "
            f"prob_shape={tuple(sample.adjacency_probabilities.shape)}"
        )