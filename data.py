

"""Data loading and preprocessing utilities for Mini-project 3.

This module is responsible for preparing the MUTAG dataset for both the
baseline and the deep generative model.

Main responsibilities:
- load MUTAG from PyTorch Geometric,
- create reproducible train/validation/test splits,
- convert each graph into a padded adjacency matrix,
- preserve node features for the encoder,
- compute masks and metadata such as the number of valid nodes,
- provide helper access to both PyG graphs and matrix-based representations.

The project requires generating graph adjacency matrices rather than node
features, but node features may still be used in the encoder. Therefore, this
module keeps both views available.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from utils_graph import canonicalize_pyg_graph


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATA_ROOT = Path("data")
DATASET_NAME = "MUTAG"
DEFAULT_SEED = 42
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphSample:
    """Matrix-based representation of one graph.

    Attributes:
        adjacency: Binary padded adjacency matrix of shape (max_nodes, max_nodes).
        node_features: Padded node feature matrix of shape (max_nodes, num_features).
        node_mask: Binary mask of shape (max_nodes,) marking valid nodes.
        num_nodes: Number of valid nodes before padding.
        label: Graph-level target label.
        graph_index: Original index in the full MUTAG dataset.
    """

    adjacency: torch.Tensor
    node_features: torch.Tensor
    node_mask: torch.Tensor
    num_nodes: int
    label: int
    graph_index: int


@dataclass(frozen=True)
class DataSplits:
    """Container with raw split indices."""

    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]


@dataclass(frozen=True)
class DataBundle:
    """Container with all prepared dataset objects and metadata."""

    pyg_dataset: TUDataset
    train_dataset: "MutagMatrixDataset"
    val_dataset: "MutagMatrixDataset"
    test_dataset: "MutagMatrixDataset"
    splits: DataSplits
    max_nodes: int
    num_node_features: int
    num_classes: int


# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------------
# Graph helpers
# -----------------------------------------------------------------------------



def pad_square_matrix(matrix: torch.Tensor, target_size: int) -> torch.Tensor:
    """Pad a square matrix with zeros to a target size."""
    current_size = matrix.shape[0]
    if current_size > target_size:
        raise ValueError(f"Current size {current_size} exceeds target size {target_size}.")

    padded = torch.zeros((target_size, target_size), dtype=matrix.dtype)
    padded[:current_size, :current_size] = matrix
    return padded



def pad_feature_matrix(features: torch.Tensor, target_size: int) -> torch.Tensor:
    """Pad a node feature matrix with zeros along the node dimension."""
    current_size, num_features = features.shape
    if current_size > target_size:
        raise ValueError(f"Current size {current_size} exceeds target size {target_size}.")

    padded = torch.zeros((target_size, num_features), dtype=features.dtype)
    padded[:current_size] = features
    return padded



def build_node_mask(num_nodes: int, target_size: int) -> torch.Tensor:
    """Create a binary mask that marks valid node positions."""
    mask = torch.zeros(target_size, dtype=torch.float32)
    mask[:num_nodes] = 1.0
    return mask



def pyg_graph_to_sample(graph: Data, graph_index: int, max_nodes: int) -> GraphSample:
    """Convert a PyG graph to the padded matrix representation used in the project.

    Applies a deterministic BFS-based canonical node ordering so that the same
    graph always maps to the same adjacency matrix, regardless of the original
    node numbering in the dataset.
    """
    adjacency_np, node_features_np, _ = canonicalize_pyg_graph(graph)
    num_nodes = adjacency_np.shape[0]

    adjacency = torch.from_numpy(adjacency_np.copy())
    node_features = torch.from_numpy(node_features_np.copy())

    padded_adjacency = pad_square_matrix(adjacency, target_size=max_nodes)
    padded_node_features = pad_feature_matrix(node_features, target_size=max_nodes)
    node_mask = build_node_mask(num_nodes=num_nodes, target_size=max_nodes)

    label = int(graph.y.item()) if graph.y is not None else -1

    return GraphSample(
        adjacency=padded_adjacency,
        node_features=padded_node_features,
        node_mask=node_mask,
        num_nodes=num_nodes,
        label=label,
        graph_index=graph_index,
    )


# -----------------------------------------------------------------------------
# Dataset wrapper
# -----------------------------------------------------------------------------

class MutagMatrixDataset(Dataset):
    """Dataset wrapper that exposes padded matrix-based MUTAG samples."""

    def __init__(self, pyg_dataset: TUDataset, indices: Iterable[int], max_nodes: int) -> None:
        self.pyg_dataset = pyg_dataset
        self.indices = list(indices)
        self.max_nodes = max_nodes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> GraphSample:
        graph_index = self.indices[idx]
        graph = self.pyg_dataset[graph_index]
        return pyg_graph_to_sample(graph=graph, graph_index=graph_index, max_nodes=self.max_nodes)


# -----------------------------------------------------------------------------
# Split helpers
# -----------------------------------------------------------------------------


def make_splits(
    n_graphs: int,
    seed: int = DEFAULT_SEED,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> DataSplits:
    """Create reproducible train/validation/test splits.

    The split is graph-level and uses shuffled indices.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1).")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be smaller than 1.")

    indices = np.arange(n_graphs)

    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
    )

    relative_val_ratio = val_ratio / (1.0 - test_ratio)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=relative_val_ratio,
        random_state=seed,
        shuffle=True,
    )

    return DataSplits(
        train_indices=train_indices.tolist(),
        val_indices=val_indices.tolist(),
        test_indices=test_indices.tolist(),
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def load_mutag_dataset(root: str | Path = DATA_ROOT, seed: int = DEFAULT_SEED) -> TUDataset:
    """Load the MUTAG dataset from PyTorch Geometric."""
    set_seed(seed)
    return TUDataset(root=str(root), name=DATASET_NAME)



def get_max_nodes(pyg_dataset: TUDataset) -> int:
    """Return the maximum number of nodes across all graphs in the dataset."""
    return max(int(graph.num_nodes) for graph in pyg_dataset)



def build_data_bundle(
    root: str | Path = DATA_ROOT,
    seed: int = DEFAULT_SEED,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> DataBundle:
    """Load MUTAG, create splits, and build matrix-based dataset wrappers."""
    pyg_dataset = load_mutag_dataset(root=root, seed=seed)
    max_nodes = get_max_nodes(pyg_dataset)
    splits = make_splits(
        n_graphs=len(pyg_dataset),
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_dataset = MutagMatrixDataset(pyg_dataset, splits.train_indices, max_nodes=max_nodes)
    val_dataset = MutagMatrixDataset(pyg_dataset, splits.val_indices, max_nodes=max_nodes)
    test_dataset = MutagMatrixDataset(pyg_dataset, splits.test_indices, max_nodes=max_nodes)

    return DataBundle(
        pyg_dataset=pyg_dataset,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        splits=splits,
        max_nodes=max_nodes,
        num_node_features=pyg_dataset.num_features,
        num_classes=pyg_dataset.num_classes,
    )


# -----------------------------------------------------------------------------
# Quick manual check
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    bundle = build_data_bundle()
    sample = bundle.train_dataset[0]

    print("MUTAG data bundle loaded successfully.")
    print(f"Total graphs: {len(bundle.pyg_dataset)}")
    print(f"Train/Val/Test: {len(bundle.train_dataset)}/{len(bundle.val_dataset)}/{len(bundle.test_dataset)}")
    print(f"Max nodes: {bundle.max_nodes}")
    print(f"Num node features: {bundle.num_node_features}")
    print(f"Adjacency shape: {tuple(sample.adjacency.shape)}")
    print(f"Node feature shape: {tuple(sample.node_features.shape)}")
    print(f"Node mask shape: {tuple(sample.node_mask.shape)}")
    print(f"Graph label: {sample.label}")