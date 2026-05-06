"""Tests for generate module validation helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from baseline_er import EmpiricalSizeDensityModel
from generate import trim_probability_matrix, sample_binary_adjacency, sample_graph_sizes


def test_trim_probability_matrix_requires_square_matrix() -> None:
    tensor = torch.randn(3, 4)

    with pytest.raises(ValueError, match="adjacency_probabilities must be a square matrix"):
        trim_probability_matrix(tensor, num_nodes=2)


def test_trim_probability_matrix_rejects_invalid_num_nodes() -> None:
    tensor = torch.randn(4, 4)

    with pytest.raises(ValueError, match="num_nodes must be between 0 and the matrix dimension"):
        trim_probability_matrix(tensor, num_nodes=5)


def test_sample_binary_adjacency_rejects_invalid_probabilities() -> None:
    tensor = torch.tensor([[0.3, -0.1], [0.1, 0.0]])

    with pytest.raises(ValueError, match=r"adjacency_probabilities must lie in \[0, 1\]"):
        sample_binary_adjacency(tensor)


def test_sample_graph_sizes_rejects_empty_size_model() -> None:
    size_model = EmpiricalSizeDensityModel(
        size_values=np.array([], dtype=int),
        size_probabilities=np.array([], dtype=float),
        density_by_size={},
    )

    with pytest.raises(ValueError, match="Size model must contain at least one size value"):
        sample_graph_sizes(size_model=size_model, num_samples=1, rng=np.random.default_rng(0))
