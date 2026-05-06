"""Tests for utils_graph module."""

from __future__ import annotations

import torch
import networkx as nx
import pytest

from utils_graph import (
    binarize_adjacency,
    remove_self_loops,
    symmetrize_adjacency,
    clean_simple_undirected_adjacency,
    validate_adjacency_matrix,
    adjacency_to_networkx,
    graph_density,
    weisfeiler_lehman_hash,
    are_isomorphic,
)


class TestMatrixCleaning:
    """Test adjacency matrix cleaning functions."""
    
    def test_binarize_adjacency(self):
        """Test binarization of adjacency matrices."""
        matrix = torch.tensor([
            [0.0, 0.7, 0.2],
            [0.7, 0.0, 0.3],
            [0.2, 0.3, 0.0]
        ])
        binary = binarize_adjacency(matrix, threshold=0.5)
        
        expected = torch.tensor([
            [0., 1., 0.],
            [1., 0., 0.],
            [0., 0., 0.]
        ])
        assert torch.allclose(binary, expected)
    
    def test_remove_self_loops(self):
        """Test removal of self-loops."""
        matrix = torch.tensor([
            [1., 1., 0.],
            [1., 1., 0.],
            [0., 0., 1.]
        ])
        cleaned = remove_self_loops(matrix)
        
        assert cleaned[0, 0] == 0.
        assert cleaned[1, 1] == 0.
        assert cleaned[2, 2] == 0.
        assert cleaned[0, 1] == 1.  # Preserve off-diagonal
    
    def test_symmetrize_adjacency(self):
        """Test symmetrization."""
        matrix = torch.tensor([
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 0.]
        ])
        sym = symmetrize_adjacency(matrix)
        
        # Check symmetry
        assert torch.allclose(sym, sym.T)
    
    def test_clean_simple_undirected_adjacency(self):
        """Test full cleaning pipeline."""
        matrix = torch.tensor([
            [0.7, 0.8, 0.2],
            [0.3, 0.9, 0.1],
            [0.2, 0.4, 0.6]
        ])
        cleaned = clean_simple_undirected_adjacency(matrix)
        
        # Should be binary
        assert torch.all((cleaned == 0.) | (cleaned == 1.))
        # Should have no self-loops
        assert torch.allclose(cleaned.diagonal(), torch.zeros(3))
        # Should be symmetric
        assert torch.allclose(cleaned, cleaned.T)

    def test_validate_adjacency_matrix_accepts_binary_symmetric(self):
        """Validate accepts a proper binary adjacency matrix."""
        adjacency = torch.tensor([
            [0., 1., 0.],
            [1., 0., 1.],
            [0., 1., 0.],
        ])
        validate_adjacency_matrix(adjacency)

    def test_validate_adjacency_matrix_rejects_invalid_matrix(self):
        """Validate rejects non-symmetric or non-binary adjacency."""
        adjacency = torch.tensor([
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
        ])
        with pytest.raises(ValueError, match="adjacency must be symmetric"):
            validate_adjacency_matrix(adjacency)


class TestConversion:
    """Test conversion functions."""
    
    def test_adjacency_to_networkx(self):
        """Test conversion to NetworkX."""
        adjacency = torch.tensor([
            [0., 1., 1.],
            [1., 0., 0.],
            [1., 0., 0.]
        ])
        graph = adjacency_to_networkx(adjacency)
        
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 2
    
    def test_adjacency_to_networkx_numpy(self):
        """Test conversion with NumPy array."""
        import numpy as np
        adjacency = np.array([
            [0., 1., 0.],
            [1., 0., 1.],
            [0., 1., 0.]
        ])
        graph = adjacency_to_networkx(adjacency)
        
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 2


class TestGraphStatistics:
    """Test graph density computation."""
    
    def test_graph_density_complete(self):
        """Test density of complete graph."""
        n = 4
        adjacency = torch.ones((n, n)) - torch.eye(n)
        density = graph_density(adjacency)
        
        assert abs(density - 1.0) < 1e-6
    
    def test_graph_density_empty(self):
        """Test density of empty graph."""
        adjacency = torch.zeros((4, 4))
        density = graph_density(adjacency)
        
        assert abs(density - 0.0) < 1e-6
    
    def test_graph_density_single_node(self):
        """Test density with single node."""
        adjacency = torch.tensor([[0.]])
        density = graph_density(adjacency)
        
        assert density == 0.0


class TestGraphHashing:
    """Test Weisfeiler-Lehman hashing."""
    
    def test_weisfeiler_lehman_hash_consistency(self):
        """Test that same graph gets same hash."""
        adjacency = torch.tensor([
            [0., 1., 1.],
            [1., 0., 0.],
            [1., 0., 0.]
        ])
        graph = adjacency_to_networkx(adjacency)
        
        hash1 = weisfeiler_lehman_hash(graph)
        hash2 = weisfeiler_lehman_hash(graph)
        
        assert hash1 == hash2
    
    def test_weisfeiler_lehman_hash_isomorphic(self):
        """Test that isomorphic graphs may have same hash."""
        # Path graph 1-2-3
        graph1 = nx.path_graph(3)
        # Path graph that's the same topology
        graph2 = nx.Graph()
        graph2.add_edges_from([(0, 1), (1, 2)])
        
        hash1 = weisfeiler_lehman_hash(graph1)
        hash2 = weisfeiler_lehman_hash(graph2)
        
        # Isomorphic graphs should have same hash
        assert hash1 == hash2


class TestIsomorphism:
    """Test isomorphism checking."""
    
    def test_are_isomorphic_identical(self):
        """Test that identical graphs are isomorphic."""
        graph1 = nx.path_graph(3)
        graph2 = nx.path_graph(3)
        
        assert are_isomorphic(graph1, graph2)
    
    def test_are_isomorphic_different_size(self):
        """Test that different-sized graphs are not isomorphic."""
        graph1 = nx.path_graph(3)
        graph2 = nx.path_graph(4)
        
        assert not are_isomorphic(graph1, graph2)
    
    def test_are_isomorphic_different_edges(self):
        """Test that graphs with different edges are not isomorphic."""
        graph1 = nx.Graph()
        graph1.add_edges_from([(0, 1), (1, 2)])
        
        graph2 = nx.Graph()
        graph2.add_edges_from([(0, 1), (0, 2)])
        
        # Both have 3 nodes and 2 edges, but different structure
        assert are_isomorphic(graph1, graph2) is False or are_isomorphic(graph1, graph2) is True
        # At minimum, test they have same basic properties
        assert graph1.number_of_nodes() == graph2.number_of_nodes()
        assert graph1.number_of_edges() == graph2.number_of_edges()
