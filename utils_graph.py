

"""Graph utility functions for Mini-project 3.

This module contains reusable helpers that sit between raw graph data,
matrix-based graph generation, and evaluation.

Main responsibilities:
- convert between PyG, adjacency matrices, and NetworkX graphs,
- enforce clean undirected simple-graph constraints,
- apply a deterministic node ordering for matrix-based models,
- extract standard graph statistics used in the project evaluation,
- provide helpers for graph hashing and isomorphism-friendly comparison.

These utilities are shared by the baseline generator, the deep generative
model, the evaluation code, and the plotting code.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from collections import deque
from typing import Iterable

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


# -----------------------------------------------------------------------------
# Basic matrix cleaning helpers
# -----------------------------------------------------------------------------


def binarize_adjacency(adjacency: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert an adjacency-like tensor into a binary matrix.

    Args:
        adjacency: Tensor of shape (N, N).
        threshold: Threshold used to binarize non-binary values.

    Returns:
        Binary float tensor of shape (N, N).
    """
    return (adjacency >= threshold).to(torch.float32)



def remove_self_loops(adjacency: torch.Tensor) -> torch.Tensor:
    """Remove self-loops from an adjacency matrix."""
    cleaned = adjacency.clone()
    cleaned.fill_diagonal_(0.0)
    return cleaned



def symmetrize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Force an adjacency matrix to be symmetric."""
    return torch.maximum(adjacency, adjacency.T)



def clean_simple_undirected_adjacency(
    adjacency: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Convert an arbitrary matrix into a clean simple undirected adjacency.

    Steps:
    1. Binarize.
    2. Remove self-loops.
    3. Symmetrize.
    4. Binarize again for safety.
    """
    cleaned = binarize_adjacency(adjacency, threshold=threshold)
    cleaned = remove_self_loops(cleaned)
    cleaned = symmetrize_adjacency(cleaned)
    cleaned = (cleaned > 0).to(torch.float32)
    return cleaned


def validate_adjacency_matrix(
    adjacency: torch.Tensor,
    allow_self_loops: bool = False,
    allow_non_binary: bool = False,
) -> None:
    """Validate that a tensor represents a valid adjacency matrix.

    Args:
        adjacency: Tensor of shape (N, N).
        allow_self_loops: If False, require zero diagonal.
        allow_non_binary: If False, require values to be exactly 0 or 1.
    """
    if adjacency.dim() != 2:
        raise ValueError("adjacency must be a 2D tensor.")
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square.")
    if not torch.allclose(adjacency, adjacency.T):
        raise ValueError("adjacency must be symmetric.")
    if not allow_self_loops and not torch.allclose(adjacency.diagonal(), torch.zeros(adjacency.shape[0], device=adjacency.device)):
        raise ValueError("adjacency must have zero diagonal when self-loops are not allowed.")
    if not allow_non_binary:
        binary_mask = (adjacency == 0.0) | (adjacency == 1.0)
        if not torch.all(binary_mask):
            raise ValueError("adjacency must contain only binary values 0 or 1.")


def adjacency_to_networkx(adjacency: np.ndarray | torch.Tensor) -> nx.Graph:
    """Convert an adjacency matrix into a simple undirected NetworkX graph."""
    if isinstance(adjacency, torch.Tensor):
        adjacency_np = adjacency.detach().cpu().numpy()
    else:
        adjacency_np = np.asarray(adjacency)

    adjacency_np = (adjacency_np > 0).astype(np.int64)
    adjacency_np = np.maximum(adjacency_np, adjacency_np.T)
    np.fill_diagonal(adjacency_np, 0)

    return nx.from_numpy_array(adjacency_np)



def networkx_to_adjacency(graph: nx.Graph) -> np.ndarray:
    """Convert a NetworkX graph into a dense binary adjacency matrix."""
    adjacency = nx.to_numpy_array(graph, dtype=np.float32)
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    adjacency = (adjacency > 0).astype(np.float32)
    return adjacency



def pyg_to_networkx(graph: Data) -> nx.Graph:
    """Convert a PyG graph into a simple undirected NetworkX graph.

    Node labels are stored under the node attribute `label`, if available.
    """
    nx_graph = nx.Graph()
    num_nodes = int(graph.num_nodes)

    for node_idx in range(num_nodes):
        label = None
        if graph.x is not None:
            label = int(torch.argmax(graph.x[node_idx]).item())
        nx_graph.add_node(node_idx, label=label)

    edge_index = graph.edge_index.detach().cpu()
    for col in range(edge_index.shape[1]):
        u = int(edge_index[0, col].item())
        v = int(edge_index[1, col].item())
        if u != v:
            nx_graph.add_edge(u, v)

    return nx_graph


# -----------------------------------------------------------------------------
# Node ordering helpers
# -----------------------------------------------------------------------------


def _bfs_order(graph: nx.Graph, start_node: int) -> list[int]:
    """Return a deterministic BFS order starting from a given node."""
    visited: set[int] = set()
    order: list[int] = []
    queue: deque[int] = deque([start_node])

    while queue:
        node = queue.popleft()
        if node in visited:
            continue

        visited.add(node)
        order.append(node)

        neighbors = sorted(graph.neighbors(node))
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append(neighbor)

    # If the graph is disconnected, append the remaining connected components.
    for node in sorted(graph.nodes()):
        if node not in visited:
            queue.append(node)
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                order.append(current)
                neighbors = sorted(graph.neighbors(current))
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)

    return order



def canonical_node_order(graph: nx.Graph) -> list[int]:
    """Compute a deterministic node order for matrix-based generative modeling.

    The order is based on simple structural cues and intended as a practical,
    lightweight approximation to a canonical labeling. It is not guaranteed to
    produce the same order for all isomorphic graphs, but it is deterministic.

    Strategy:
    1. Pick a start node using the lexicographically smallest tuple
       (-degree, label, node_id).
    2. Run BFS from that node.
    3. Refine the BFS order by sorting nodes using
       (bfs_position, -degree, label, node_id).
    """
    if graph.number_of_nodes() == 0:
        return []

    node_keys = []
    for node in graph.nodes():
        degree = graph.degree[node]
        label = graph.nodes[node].get("label", -1)
        node_keys.append((node, -degree, label, node))

    start_node = sorted(node_keys, key=lambda item: (item[1], item[2], item[3]))[0][0]
    bfs_order = _bfs_order(graph, start_node)
    bfs_position = {node: idx for idx, node in enumerate(bfs_order)}

    ordered_nodes = sorted(
        graph.nodes(),
        key=lambda node: (
            bfs_position[node],
            -graph.degree[node],
            graph.nodes[node].get("label", -1),
            node,
        ),
    )
    return ordered_nodes



def reorder_adjacency(adjacency: np.ndarray, order: list[int]) -> np.ndarray:
    """Reorder an adjacency matrix using a node order."""
    return adjacency[np.ix_(order, order)]



def reorder_node_features(node_features: np.ndarray, order: list[int]) -> np.ndarray:
    """Reorder a node feature matrix using a node order."""
    return node_features[order]



def canonicalize_pyg_graph(graph: Data) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Convert a PyG graph to canonically ordered adjacency and node features.

    Returns:
        adjacency: Binary adjacency matrix in canonical node order.
        node_features: Node features in canonical node order.
        order: Applied node order.
    """
    nx_graph = pyg_to_networkx(graph)
    order = canonical_node_order(nx_graph)

    adjacency = networkx_to_adjacency(nx_graph)
    adjacency = reorder_adjacency(adjacency, order)

    if graph.x is None:
        raise ValueError("Expected node features in MUTAG graph, but graph.x is None.")

    node_features = graph.x.detach().cpu().numpy().astype(np.float32)
    node_features = reorder_node_features(node_features, order)
    return adjacency, node_features, order


# -----------------------------------------------------------------------------
# Graph statistics for evaluation
# -----------------------------------------------------------------------------


def degree_values(graph: nx.Graph) -> list[int]:
    """Return node degree values for a graph."""
    return [int(degree) for _, degree in graph.degree()]



def clustering_coefficient_values(graph: nx.Graph) -> list[float]:
    """Return local clustering coefficient values for all nodes."""
    coefficients = nx.clustering(graph)
    return [float(value) for value in coefficients.values()]



def eigenvector_centrality_values(graph: nx.Graph) -> list[float]:
    """Return eigenvector centrality values for all nodes.

    For empty graphs or numerically problematic graphs, fall back to zeros.
    """
    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_edges() == 0:
        return [0.0 for _ in graph.nodes()]

    try:
        centrality = nx.eigenvector_centrality_numpy(graph)
        return [float(value) for value in centrality.values()]
    except Exception:
        return [0.0 for _ in graph.nodes()]


def graph_density(adjacency: torch.Tensor) -> float:
    """Compute the density of a simple undirected adjacency matrix."""
    num_nodes = adjacency.shape[0]
    if num_nodes <= 1:
        return 0.0

    num_possible_edges = num_nodes * (num_nodes - 1) / 2.0
    num_edges = float(torch.triu(adjacency, diagonal=1).sum().item())
    return num_edges / num_possible_edges


# -----------------------------------------------------------------------------
# Graph comparison helpers
# -----------------------------------------------------------------------------


def weisfeiler_lehman_hash(graph: nx.Graph) -> str:
    """Compute a Weisfeiler-Lehman graph hash using NetworkX.

    If all nodes contain a `label` attribute, include it in the hash.
    Otherwise, fall back to a purely structural hash.
    """
    has_node_labels = all("label" in data for _, data in graph.nodes(data=True))

    if has_node_labels:
        return nx.weisfeiler_lehman_graph_hash(graph, node_attr="label")

    return nx.weisfeiler_lehman_graph_hash(graph)


def are_isomorphic(graph_a: nx.Graph, graph_b: nx.Graph) -> bool:
    """Check structural isomorphism between two NetworkX graphs."""
    if graph_a.number_of_nodes() != graph_b.number_of_nodes():
        return False
    if graph_a.number_of_edges() != graph_b.number_of_edges():
        return False
    return nx.is_isomorphic(graph_a, graph_b)



# -----------------------------------------------------------------------------
# Sampling helpers
# -----------------------------------------------------------------------------


def upper_triangle_to_adjacency(upper_triangle: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Convert a flattened upper-triangle vector into a full adjacency matrix.

    Args:
        upper_triangle: Tensor of shape (num_nodes * (num_nodes - 1) // 2,).
        num_nodes: Number of valid nodes.

    Returns:
        Symmetric adjacency matrix of shape (num_nodes, num_nodes).
    """
    expected_size = num_nodes * (num_nodes - 1) // 2
    if upper_triangle.numel() != expected_size:
        raise ValueError(
            f"Expected upper triangle of size {expected_size}, got {upper_triangle.numel()}."
        )

    adjacency = torch.zeros((num_nodes, num_nodes), dtype=upper_triangle.dtype, device=upper_triangle.device)
    row_idx, col_idx = torch.triu_indices(num_nodes, num_nodes, offset=1, device=upper_triangle.device)
    adjacency[row_idx, col_idx] = upper_triangle
    adjacency[col_idx, row_idx] = upper_triangle
    return adjacency



def adjacency_to_upper_triangle(adjacency: torch.Tensor) -> torch.Tensor:
    """Flatten the strict upper triangle of an adjacency matrix."""
    num_nodes = adjacency.shape[0]
    row_idx, col_idx = torch.triu_indices(num_nodes, num_nodes, offset=1, device=adjacency.device)
    return adjacency[row_idx, col_idx]


# -----------------------------------------------------------------------------
# Quick manual check
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Small toy example for a quick sanity check.
    adjacency = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )

    graph = adjacency_to_networkx(adjacency)
    print("Graph utility sanity check")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"WL hash: {weisfeiler_lehman_hash(graph)}")
    print(f"Degrees: {degree_values(graph)}")
    print(f"Clustering: {clustering_coefficient_values(graph)}")
    print(f"Eigenvector centrality: {eigenvector_centrality_values(graph)}")