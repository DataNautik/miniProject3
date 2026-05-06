

"""Erdos-Renyi baseline for Mini-project 3.

This module implements the baseline required in the project description.
For each sampled graph, the baseline works as follows:

1. Sample the number of nodes N from the empirical distribution of graph sizes
   in the training set.
2. Estimate the edge probability r using the average graph density among the
   training graphs with exactly N nodes.
3. Sample an undirected Erdos-Renyi graph with N nodes and edge probability r.

The implementation is kept deliberately simple and reproducible. It uses the
matrix-based dataset prepared in `data.py` and the graph helpers defined in
`utils_graph.py`.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data import DataBundle, GraphSample, build_data_bundle
from utils_graph import clean_simple_undirected_adjacency, graph_density
from utils_io import ensure_parent_dir, save_json


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_BASELINE_SEED = 42
DEFAULT_NUM_SAMPLES = 1000
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_BASELINE_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "samples" / "baseline_er_samples.pt"
DEFAULT_BASELINE_METADATA_PATH = DEFAULT_OUTPUT_DIR / "tables" / "baseline_er_metadata.json"


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BaselineSample:
    """One sampled graph from the Erdos-Renyi baseline.

    Attributes:
        adjacency: Binary adjacency matrix of shape (N, N).
        num_nodes: Number of nodes in the sampled graph.
        edge_probability: Estimated conditional probability r used for sampling.
    """

    adjacency: torch.Tensor
    num_nodes: int
    edge_probability: float


@dataclass(frozen=True)
class EmpiricalSizeDensityModel:
    """Empirical distribution over graph sizes and conditional densities.

    Attributes:
        size_values: Sorted graph sizes observed in the training set.
        size_probabilities: Empirical probability of each graph size.
        density_by_size: Average graph density conditioned on graph size.
    """

    size_values: np.ndarray
    size_probabilities: np.ndarray
    density_by_size: dict[int, float]


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def extract_train_graphs(data_bundle: DataBundle) -> list[GraphSample]:
    """Materialize the training split as a list of GraphSample objects."""
    return [data_bundle.train_dataset[idx] for idx in range(len(data_bundle.train_dataset))]



def fit_empirical_size_density_model(train_graphs: list[GraphSample]) -> EmpiricalSizeDensityModel:
    """Estimate the empirical size distribution and density conditioned on size.

    This follows the baseline specification in the project handout.
    """
    if len(train_graphs) == 0:
        raise ValueError("Training graph list is empty.")

    counts_by_size: dict[int, int] = defaultdict(int)
    densities_by_size: dict[int, list[float]] = defaultdict(list)

    for sample in train_graphs:
        num_nodes = int(sample.num_nodes)
        adjacency = sample.adjacency[:num_nodes, :num_nodes]
        counts_by_size[num_nodes] += 1
        densities_by_size[num_nodes].append(graph_density(adjacency))

    size_values = np.array(sorted(counts_by_size.keys()), dtype=np.int64)
    size_counts = np.array([counts_by_size[size] for size in size_values], dtype=np.float64)
    size_probabilities = size_counts / size_counts.sum()

    density_by_size = {
        int(size): float(np.mean(densities_by_size[int(size)]))
        for size in size_values
    }

    return EmpiricalSizeDensityModel(
        size_values=size_values,
        size_probabilities=size_probabilities,
        density_by_size=density_by_size,
    )



def sample_num_nodes(model: EmpiricalSizeDensityModel, rng: np.random.Generator) -> int:
    """Sample the graph size N from the empirical size distribution."""
    sampled_size = rng.choice(model.size_values, p=model.size_probabilities)
    return int(sampled_size)



def sample_erdos_renyi_adjacency(
    num_nodes: int,
    edge_probability: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Sample one undirected Erdos-Renyi adjacency matrix.

    Args:
        num_nodes: Number of nodes in the sampled graph.
        edge_probability: Edge probability r.
        rng: NumPy random generator.

    Returns:
        Binary adjacency matrix of shape (num_nodes, num_nodes).
    """
    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError(f"edge_probability must be in [0, 1], got {edge_probability}.")

    upper_triangle_mask = np.triu(np.ones((num_nodes, num_nodes), dtype=np.float32), k=1)
    bernoulli_draws = rng.binomial(n=1, p=edge_probability, size=(num_nodes, num_nodes)).astype(np.float32)
    adjacency = bernoulli_draws * upper_triangle_mask
    adjacency = adjacency + adjacency.T

    adjacency_tensor = torch.from_numpy(adjacency)
    adjacency_tensor = clean_simple_undirected_adjacency(adjacency_tensor, threshold=0.5)
    return adjacency_tensor


# -----------------------------------------------------------------------------
# Main baseline API
# -----------------------------------------------------------------------------


def sample_baseline_graph(
    model: EmpiricalSizeDensityModel,
    rng: np.random.Generator,
) -> BaselineSample:
    """Sample one graph from the Erdos-Renyi baseline."""
    num_nodes = sample_num_nodes(model, rng=rng)
    edge_probability = model.density_by_size[num_nodes]
    adjacency = sample_erdos_renyi_adjacency(
        num_nodes=num_nodes,
        edge_probability=edge_probability,
        rng=rng,
    )

    return BaselineSample(
        adjacency=adjacency,
        num_nodes=num_nodes,
        edge_probability=edge_probability,
    )



def generate_baseline_samples(
    data_bundle: DataBundle,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = DEFAULT_BASELINE_SEED,
) -> tuple[list[BaselineSample], EmpiricalSizeDensityModel]:
    """Generate multiple baseline samples from the training split."""
    rng = np.random.default_rng(seed)
    train_graphs = extract_train_graphs(data_bundle)
    model = fit_empirical_size_density_model(train_graphs)

    samples = [sample_baseline_graph(model=model, rng=rng) for _ in range(num_samples)]
    return samples, model


# -----------------------------------------------------------------------------
# Serialization helpers
# -----------------------------------------------------------------------------


def save_baseline_samples(samples: list[BaselineSample], output_path: str | Path = DEFAULT_BASELINE_OUTPUT_PATH) -> None:
    """Save sampled baseline graphs to disk using torch.save."""
    output_path = Path(output_path)
    ensure_parent_dir(output_path)

    serializable_samples = [
        {
            "adjacency": sample.adjacency,
            "num_nodes": sample.num_nodes,
            "edge_probability": sample.edge_probability,
        }
        for sample in samples
    ]
    torch.save(serializable_samples, output_path)



def save_baseline_metadata(
    model: EmpiricalSizeDensityModel,
    metadata_path: str | Path = DEFAULT_BASELINE_METADATA_PATH,
) -> None:
    """Save the empirical size-density model metadata as JSON."""
    metadata_path = Path(metadata_path)
    payload = {
        "size_values": model.size_values.tolist(),
        "size_probabilities": model.size_probabilities.tolist(),
        "density_by_size": {str(k): float(v) for k, v in model.density_by_size.items()},
    }
    save_json(payload, metadata_path)


# -----------------------------------------------------------------------------
# Quick manual check
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    data_bundle = build_data_bundle()
    samples, model = generate_baseline_samples(data_bundle=data_bundle, num_samples=5, seed=DEFAULT_BASELINE_SEED)

    print("Erdos-Renyi baseline sanity check")
    print(f"Observed graph sizes: {model.size_values.tolist()}")
    print(f"Size probabilities: {np.round(model.size_probabilities, 4).tolist()}")
    print(f"Density by size: {model.density_by_size}")

    for idx, sample in enumerate(samples):
        print(
            f"Sample {idx}: N={sample.num_nodes}, "
            f"r={sample.edge_probability:.4f}, "
            f"adjacency_shape={tuple(sample.adjacency.shape)}"
        )