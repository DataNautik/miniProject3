

"""Evaluation metrics for Mini-project 3.

This module implements the project-level evaluation for both the Erdos-Renyi
baseline and the deep generative model.

Main responsibilities:
- convert saved/generated samples into NetworkX graphs,
- compute novelty, uniqueness, and novelty+uniqueness,
- compare graphs using Weisfeiler-Lehman hashing and exact isomorphism checks,
- extract graph statistics used in the report:
  degree, clustering coefficient, and eigenvector centrality,
- aggregate statistics across collections of graphs.

The implementation uses WL hashing as a fast first-stage filter and then falls
back to exact isomorphism checks when needed.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any

import networkx as nx
import numpy as np
import torch
from scipy.stats import ks_2samp

from baseline_er import BaselineSample
from data import DataBundle, GraphSample, build_data_bundle
from generate import GeneratedGraphSample
from utils_graph import (
    adjacency_to_networkx,
    clustering_coefficient_values,
    degree_values,
    eigenvector_centrality_values,
    weisfeiler_lehman_hash,
)
from utils_io import ensure_parent_dir, save_json


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_METRICS_PATH = DEFAULT_OUTPUT_DIR / "tables" / "evaluation_metrics.json"
DEFAULT_STATS_PATH = DEFAULT_OUTPUT_DIR / "tables" / "graph_statistics.json"


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class NoveltyMetrics:
    """Novelty and uniqueness summary for a generated sample set.

    Attributes:
        num_generated: Number of generated graphs evaluated.
        num_unique: Number of unique generated graphs.
        num_novel: Number of generated graphs not isomorphic to any train graph.
        num_novel_and_unique: Number of generated graphs that are both novel and
            unique within the generated set.
        unique_fraction: num_unique / num_generated.
        novel_fraction: num_novel / num_generated.
        novel_and_unique_fraction: num_novel_and_unique / num_generated.
    """

    num_generated: int
    num_unique: int
    num_novel: int
    num_novel_and_unique: int
    unique_fraction: float
    novel_fraction: float
    novel_and_unique_fraction: float


@dataclass(frozen=True)
class GraphStatisticsSummary:
    """Flattened graph-statistics summary for a collection of graphs."""

    degree_values: list[float]
    clustering_values: list[float]
    eigenvector_values: list[float]


# -----------------------------------------------------------------------------
# Filesystem helpers
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Conversion helpers
# -----------------------------------------------------------------------------


def graph_sample_to_networkx(sample: GraphSample) -> nx.Graph:
    """Convert a matrix-based training sample into a NetworkX graph."""
    adjacency = sample.adjacency[:sample.num_nodes, :sample.num_nodes]
    return adjacency_to_networkx(adjacency)



def baseline_sample_to_networkx(sample: BaselineSample) -> nx.Graph:
    """Convert a baseline sample into a NetworkX graph."""
    return adjacency_to_networkx(sample.adjacency)



def generated_sample_to_networkx(sample: GeneratedGraphSample) -> nx.Graph:
    """Convert a generated deep-model sample into a NetworkX graph."""
    return adjacency_to_networkx(sample.adjacency)



def build_training_graphs(data_bundle: DataBundle) -> list[nx.Graph]:
    """Convert the training split into a list of NetworkX graphs."""
    return [graph_sample_to_networkx(data_bundle.train_dataset[idx]) for idx in range(len(data_bundle.train_dataset))]


# -----------------------------------------------------------------------------
# Isomorphism helpers
# -----------------------------------------------------------------------------


def are_isomorphic(graph_a: nx.Graph, graph_b: nx.Graph) -> bool:
    """Check graph isomorphism.

    The current project generates adjacency matrices only and does not generate
    node labels, so the comparison is structural only.
    """
    if graph_a.number_of_nodes() != graph_b.number_of_nodes():
        return False
    if graph_a.number_of_edges() != graph_b.number_of_edges():
        return False
    return nx.is_isomorphic(graph_a, graph_b)



def build_hash_buckets(graphs: Iterable[nx.Graph]) -> dict[str, list[nx.Graph]]:
    """Group graphs by Weisfeiler-Lehman hash for faster lookup."""
    buckets: dict[str, list[nx.Graph]] = {}
    for graph in graphs:
        graph_hash = weisfeiler_lehman_hash(graph)
        buckets.setdefault(graph_hash, []).append(graph)
    return buckets



def is_graph_novel(graph: nx.Graph, train_buckets: dict[str, list[nx.Graph]]) -> bool:
    """Return True if a graph is not isomorphic to any train graph."""
    graph_hash = weisfeiler_lehman_hash(graph)
    candidates = train_buckets.get(graph_hash, [])

    for candidate in candidates:
        if are_isomorphic(graph, candidate):
            return False
    return True



def unique_graph_indices(graphs: list[nx.Graph]) -> list[int]:
    """Return indices of graphs that are unique within the generated set.

    A graph is kept if it is the first representative of its isomorphism class
    among the generated graphs.
    """
    unique_indices: list[int] = []
    seen_buckets: dict[str, list[nx.Graph]] = {}

    for idx, graph in enumerate(graphs):
        graph_hash = weisfeiler_lehman_hash(graph)
        candidates = seen_buckets.get(graph_hash, [])

        is_new = True
        for candidate in candidates:
            if are_isomorphic(graph, candidate):
                is_new = False
                break

        if is_new:
            unique_indices.append(idx)
            seen_buckets.setdefault(graph_hash, []).append(graph)

    return unique_indices


# -----------------------------------------------------------------------------
# Main novelty / uniqueness metrics
# -----------------------------------------------------------------------------


def compute_novelty_metrics(
    generated_graphs: list[nx.Graph],
    train_graphs: list[nx.Graph],
) -> NoveltyMetrics:
    """Compute novelty, uniqueness, and novelty+uniqueness.

    Definitions:
    - Unique: graph is the first representative of its isomorphism class within
      the generated sample set.
    - Novel: graph is not isomorphic to any training graph.
    - Novel + Unique: both conditions hold.
    """
    num_generated = len(generated_graphs)
    if num_generated == 0:
        return NoveltyMetrics(
            num_generated=0,
            num_unique=0,
            num_novel=0,
            num_novel_and_unique=0,
            unique_fraction=0.0,
            novel_fraction=0.0,
            novel_and_unique_fraction=0.0,
        )

    train_buckets = build_hash_buckets(train_graphs)
    unique_indices = set(unique_graph_indices(generated_graphs))

    novel_flags = [is_graph_novel(graph, train_buckets) for graph in generated_graphs]

    num_unique = len(unique_indices)
    num_novel = int(sum(novel_flags))
    num_novel_and_unique = sum(1 for idx, is_novel in enumerate(novel_flags) if is_novel and idx in unique_indices)

    return NoveltyMetrics(
        num_generated=num_generated,
        num_unique=num_unique,
        num_novel=num_novel,
        num_novel_and_unique=num_novel_and_unique,
        unique_fraction=num_unique / num_generated,
        novel_fraction=num_novel / num_generated,
        novel_and_unique_fraction=num_novel_and_unique / num_generated,
    )


# -----------------------------------------------------------------------------
# Graph statistics
# -----------------------------------------------------------------------------


def compute_graph_statistics_summary(graphs: list[nx.Graph]) -> GraphStatisticsSummary:
    """Aggregate node-level graph statistics over a collection of graphs."""
    degree_list: list[float] = []
    clustering_list: list[float] = []
    eigenvector_list: list[float] = []

    for graph in graphs:
        degree_list.extend(float(x) for x in degree_values(graph))
        clustering_list.extend(float(x) for x in clustering_coefficient_values(graph))
        eigenvector_list.extend(float(x) for x in eigenvector_centrality_values(graph))

    return GraphStatisticsSummary(
        degree_values=degree_list,
        clustering_values=clustering_list,
        eigenvector_values=eigenvector_list,
    )


def compare_graph_statistics_distributions(
    reference_stats: GraphStatisticsSummary,
    comparison_stats: GraphStatisticsSummary,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Compare two graph-statistic distributions with KS tests."""
    def _trace(values: list[float]) -> np.ndarray:
        return np.asarray(values, dtype=float)

    results: dict[str, float] = {}
    for stat_name in ("degree", "clustering", "eigenvector"):
        ref_values = _trace(getattr(reference_stats, f"{stat_name}_values"))
        comp_values = _trace(getattr(comparison_stats, f"{stat_name}_values"))

        if ref_values.size == 0 or comp_values.size == 0:
            results[f"{stat_name}_ks_statistic"] = 0.0
            results[f"{stat_name}_ks_pvalue"] = 1.0
            continue

        ks_result = ks_2samp(ref_values, comp_values, alternative=alternative)
        results[f"{stat_name}_ks_statistic"] = float(ks_result.statistic)
        results[f"{stat_name}_ks_pvalue"] = float(ks_result.pvalue)

    return results


def compare_training_and_generated_distributions(
    training_stats: GraphStatisticsSummary,
    baseline_stats: GraphStatisticsSummary,
    deep_stats: GraphStatisticsSummary,
    alternative: str = "two-sided",
) -> dict[str, dict[str, float]]:
    """Compute pairwise KS test results for the three graph distributions."""
    return {
        "training_vs_baseline": compare_graph_statistics_distributions(
            training_stats, baseline_stats, alternative=alternative
        ),
        "training_vs_deep": compare_graph_statistics_distributions(
            training_stats, deep_stats, alternative=alternative
        ),
        "baseline_vs_deep": compare_graph_statistics_distributions(
            baseline_stats, deep_stats, alternative=alternative
        ),
    }


# -----------------------------------------------------------------------------
# Serialization helpers
# -----------------------------------------------------------------------------


def novelty_metrics_to_dict(metrics: NoveltyMetrics) -> dict[str, Any]:
    """Convert NoveltyMetrics to a JSON-serializable dictionary."""
    return {
        "num_generated": metrics.num_generated,
        "num_unique": metrics.num_unique,
        "num_novel": metrics.num_novel,
        "num_novel_and_unique": metrics.num_novel_and_unique,
        "unique_fraction": metrics.unique_fraction,
        "novel_fraction": metrics.novel_fraction,
        "novel_and_unique_fraction": metrics.novel_and_unique_fraction,
    }



def graph_statistics_to_dict(stats: GraphStatisticsSummary) -> dict[str, Any]:
    """Convert GraphStatisticsSummary to a JSON-serializable dictionary."""
    return {
        "degree_values": stats.degree_values,
        "clustering_values": stats.clustering_values,
        "eigenvector_values": stats.eigenvector_values,
    }


# -----------------------------------------------------------------------------
# Convenience pipelines
# -----------------------------------------------------------------------------


def evaluate_against_training_set(
    generated_graphs: list[nx.Graph],
    data_bundle: DataBundle,
) -> NoveltyMetrics:
    """Compute novelty metrics against the training split of a data bundle."""
    train_graphs = build_training_graphs(data_bundle)
    return compute_novelty_metrics(generated_graphs=generated_graphs, train_graphs=train_graphs)


# -----------------------------------------------------------------------------
# Quick manual check
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    data_bundle = build_data_bundle()
    train_graphs = build_training_graphs(data_bundle)

    # Simple sanity check: compare part of the training set against itself.
    subset_graphs = train_graphs[:10]
    novelty = compute_novelty_metrics(generated_graphs=subset_graphs, train_graphs=train_graphs)
    stats = compute_graph_statistics_summary(subset_graphs)

    print("Metrics sanity check")
    print(f"Novelty metrics: {novelty_metrics_to_dict(novelty)}")
    print(
        "Statistics sizes:",
        len(stats.degree_values),
        len(stats.clustering_values),
        len(stats.eigenvector_values),
    )