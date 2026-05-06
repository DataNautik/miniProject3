"""Pytest configuration and fixtures for Mini-project 3 tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np
from torch_geometric.data import Data

from data import GraphSample


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_graph_sample():
    """Create a simple 3-node graph sample for testing."""
    adjacency = torch.tensor([
        [0., 1., 1.],
        [1., 0., 0.],
        [1., 0., 0.]
    ])
    node_features = torch.ones((3, 7))
    node_mask = torch.ones(3)
    
    return GraphSample(
        adjacency=adjacency,
        node_features=node_features,
        node_mask=node_mask,
        num_nodes=3,
        label=1,
        graph_index=0
    )


@pytest.fixture
def small_batch():
    """Create a small batch of graph samples for testing."""
    batch_size = 4
    max_nodes = 10
    num_features = 7
    
    samples = []
    for i in range(batch_size):
        num_nodes = np.random.randint(3, 8)
        adjacency = torch.zeros((max_nodes, max_nodes))
        node_features = torch.randn((max_nodes, num_features))
        node_mask = torch.zeros(max_nodes)
        node_mask[:num_nodes] = 1.0
        
        samples.append(GraphSample(
            adjacency=adjacency,
            node_features=node_features,
            node_mask=node_mask,
            num_nodes=num_nodes,
            label=i % 2,
            graph_index=i
        ))
    
    return samples


@pytest.fixture
def dummy_pyg_graph():
    """Create a dummy PyTorch Geometric graph for testing."""
    x = torch.randn(5, 7)  # 5 nodes, 7 features
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])  # 4 edges
    y = torch.tensor([1])  # class label
    
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=5)
