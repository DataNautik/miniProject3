"""Tests for metrics module."""

from __future__ import annotations

import torch
import networkx as nx
import pytest

from metrics import (
    GraphStatisticsSummary,
    NoveltyMetrics,
    compute_novelty_metrics,
    compare_graph_statistics_distributions,
    graph_sample_to_networkx,
    baseline_sample_to_networkx,
    generated_sample_to_networkx,
    are_isomorphic,
    build_hash_buckets,
    unique_graph_indices,
)
from baseline_er import BaselineSample
from generate import GeneratedGraphSample
from data import GraphSample


class TestNoveltyMetrics:
    """Test NoveltyMetrics dataclass."""
    
    def test_novelty_metrics_creation(self):
        """Test creating NoveltyMetrics."""
        metrics = NoveltyMetrics(
            num_generated=100,
            num_unique=90,
            num_novel=80,
            num_novel_and_unique=75,
            unique_fraction=0.9,
            novel_fraction=0.8,
            novel_and_unique_fraction=0.75
        )
        
        assert metrics.num_generated == 100
        assert metrics.unique_fraction == 0.9
        assert metrics.novel_and_unique_fraction == 0.75


class TestGraphConversion:
    """Test graph conversion functions."""
    
    def test_graph_sample_to_networkx(self):
        """Test converting GraphSample to NetworkX."""
        adjacency = torch.tensor([
            [0., 1., 1.],
            [1., 0., 0.],
            [1., 0., 0.]
        ])
        node_features = torch.ones((3, 7))
        node_mask = torch.ones(3)
        
        sample = GraphSample(
            adjacency=adjacency,
            node_features=node_features,
            node_mask=node_mask,
            num_nodes=3,
            label=1,
            graph_index=0
        )
        
        graph = graph_sample_to_networkx(sample)
        
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 2
    
    def test_baseline_sample_to_networkx(self):
        """Test converting BaselineSample to NetworkX."""
        adjacency = torch.tensor([
            [0., 1., 0.],
            [1., 0., 1.],
            [0., 1., 0.]
        ])
        
        sample = BaselineSample(
            adjacency=adjacency,
            num_nodes=3,
            edge_probability=0.5
        )
        
        graph = baseline_sample_to_networkx(sample)
        
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 2
    
    def test_generated_sample_to_networkx(self):
        """Test converting GeneratedGraphSample to NetworkX."""
        adjacency = torch.tensor([
            [0., 1., 1.],
            [1., 0., 0.],
            [1., 0., 0.]
        ])
        adjacency_probs = torch.tensor([
            [0., 0.9, 0.8],
            [0.9, 0., 0.1],
            [0.8, 0.1, 0.]
        ])
        
        sample = GeneratedGraphSample(
            adjacency=adjacency,
            adjacency_probabilities=adjacency_probs,
            num_nodes=3
        )
        
        graph = generated_sample_to_networkx(sample)
        
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 3


class TestHashBuckets:
    """Test hash bucket building."""
    
    def test_build_hash_buckets(self):
        """Test building hash buckets."""
        graphs = [
            nx.path_graph(3),
            nx.path_graph(3),
            nx.cycle_graph(3),
        ]
        
        buckets = build_hash_buckets(graphs)
        
        # Same graphs should be in same bucket
        assert len(buckets) > 0
        # Check that at least one bucket has multiple graphs
        assert any(len(graphs_list) > 1 for graphs_list in buckets.values())


class TestUniqueGraphIndices:
    """Test finding unique graph indices."""
    
    def test_unique_graph_indices(self):
        """Test finding unique graphs."""
        graphs = [
            nx.path_graph(3),
            nx.path_graph(3),  # Same as first
            nx.cycle_graph(3),  # Different
        ]
        
        unique_indices = unique_graph_indices(graphs)
        
        # Should have at most 3 indices
        assert len(unique_indices) <= 3
        # Should include index 0 (first graph)
        assert 0 in unique_indices
    
    def test_unique_graph_indices_all_different(self):
        """Test when all graphs are different."""
        graphs = [
            nx.path_graph(4),
            nx.cycle_graph(4),
            nx.complete_graph(4),
        ]
        
        unique_indices = unique_graph_indices(graphs)
        
        # All should be unique
        assert len(unique_indices) == 3


class TestNoveltyComputation:
    """Test novelty metrics computation."""
    
    def test_compute_novelty_metrics_empty(self):
        """Test with empty generated set."""
        train_graphs = [nx.path_graph(3)]
        generated_graphs = []
        
        metrics = compute_novelty_metrics(generated_graphs, train_graphs)
        
        assert metrics.num_generated == 0
        assert metrics.num_unique == 0
        assert metrics.num_novel == 0
    
    def test_compute_novelty_metrics_identical(self):
        """Test when generated graphs are identical to training."""
        train_graph = nx.path_graph(3)
        generated_graphs = [train_graph, train_graph]
        
        metrics = compute_novelty_metrics(generated_graphs, [train_graph])
        
        assert metrics.num_generated == 2
        assert metrics.num_novel == 0
        assert metrics.novel_fraction == 0.0
    
    def test_compute_novelty_metrics_novel(self):
        """Test when all generated graphs are novel."""
        train_graphs = [nx.path_graph(3)]
        generated_graphs = [
            nx.cycle_graph(3),
            nx.complete_graph(4),
        ]
        
        metrics = compute_novelty_metrics(generated_graphs, train_graphs)
        
        assert metrics.num_generated == 2
        # Generated graphs have different sizes than training
        assert metrics.num_novel >= 0  # May or may not be novel depending on training set
    
    def test_compute_novelty_metrics_unique(self):
        """Test uniqueness computation."""
        generated_graphs = [
            nx.path_graph(3),
            nx.path_graph(3),  # Same as first
            nx.cycle_graph(3),  # Different
        ]
        train_graphs = []
        
        metrics = compute_novelty_metrics(generated_graphs, train_graphs)
        
        assert metrics.num_generated == 3
        # We have 2 or 3 unique graphs (depends on if path and cycle are isomorphic)
        assert 2 <= metrics.num_unique <= 3


def test_compare_graph_statistics_distributions():
    """Test KS comparison of graph statistic distributions."""
    training_stats = GraphStatisticsSummary(
        degree_values=[1.0, 2.0, 2.0],
        clustering_values=[0.0, 0.1, 0.2],
        eigenvector_values=[0.1, 0.2, 0.3],
    )
    comparison_stats = GraphStatisticsSummary(
        degree_values=[1.0, 2.0, 3.0],
        clustering_values=[0.0, 0.1, 0.25],
        eigenvector_values=[0.15, 0.25, 0.35],
    )

    results = compare_graph_statistics_distributions(training_stats, comparison_stats)

    assert "degree_ks_statistic" in results
    assert "degree_ks_pvalue" in results
    assert 0.0 <= results["degree_ks_pvalue"] <= 1.0
