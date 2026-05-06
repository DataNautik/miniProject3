"""Tests for data module."""

from __future__ import annotations

import torch
import numpy as np
import pytest

from data import (
    GraphSample,
    pad_square_matrix,
    pad_feature_matrix,
    build_node_mask,
    MutagMatrixDataset,
)


class TestPadding:
    """Test padding functions."""
    
    def test_pad_square_matrix(self):
        """Test square matrix padding."""
        matrix = torch.ones((3, 3))
        padded = pad_square_matrix(matrix, target_size=5)
        
        assert padded.shape == (5, 5)
        assert torch.allclose(padded[:3, :3], matrix)
        assert torch.allclose(padded[3:, :], torch.zeros(2, 5))
        assert torch.allclose(padded[:, 3:], torch.zeros(5, 2))
    
    def test_pad_square_matrix_no_padding_needed(self):
        """Test when no padding is needed."""
        matrix = torch.ones((5, 5))
        padded = pad_square_matrix(matrix, target_size=5)
        
        assert torch.allclose(padded, matrix)
    
    def test_pad_square_matrix_error(self):
        """Test error when matrix is too large."""
        matrix = torch.ones((6, 6))
        
        with pytest.raises(ValueError):
            pad_square_matrix(matrix, target_size=5)
    
    def test_pad_feature_matrix(self):
        """Test feature matrix padding."""
        features = torch.randn(3, 7)
        padded = pad_feature_matrix(features, target_size=5)
        
        assert padded.shape == (5, 7)
        assert torch.allclose(padded[:3, :], features)
        assert torch.allclose(padded[3:, :], torch.zeros(2, 7))
    
    def test_pad_feature_matrix_error(self):
        """Test error when matrix is too large."""
        features = torch.randn(6, 7)
        
        with pytest.raises(ValueError):
            pad_feature_matrix(features, target_size=5)


class TestNodeMask:
    """Test node mask building."""
    
    def test_build_node_mask(self):
        """Test node mask creation."""
        mask = build_node_mask(num_nodes=3, target_size=5)
        
        assert mask.shape == (5,)
        assert torch.allclose(mask[:3], torch.ones(3))
        assert torch.allclose(mask[3:], torch.zeros(2))
    
    def test_build_node_mask_full(self):
        """Test when all nodes are valid."""
        mask = build_node_mask(num_nodes=5, target_size=5)
        
        assert torch.all(mask == 1.0)


class TestGraphSample:
    """Test GraphSample dataclass."""
    
    def test_graph_sample_creation(self):
        """Test creating a GraphSample."""
        adjacency = torch.ones((5, 5))
        node_features = torch.randn(5, 7)
        node_mask = torch.ones(5)
        
        sample = GraphSample(
            adjacency=adjacency,
            node_features=node_features,
            node_mask=node_mask,
            num_nodes=5,
            label=1,
            graph_index=0
        )
        
        assert sample.num_nodes == 5
        assert sample.label == 1
        assert sample.graph_index == 0
        assert sample.adjacency.shape == (5, 5)
        assert sample.node_features.shape == (5, 7)


class TestMutagMatrixDataset:
    """Test MutagMatrixDataset wrapper."""
    
    def test_dataset_len(self):
        """Test dataset length."""
        # We need to mock a PyG dataset for testing
        # For now, we'll skip this if we can't load the real data
        pytest.skip("Requires PyTorch Geometric MUTAG dataset")
    
    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        pytest.skip("Requires PyTorch Geometric MUTAG dataset")
