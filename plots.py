

"""Plotting utilities for Mini-project 3.

This module creates the report-ready figures required by the project.
In particular, it generates the 3x3 histogram grid comparing:
- the training distribution,
- the Erdos-Renyi baseline,
- the deep generative model,

across the three required graph statistics:
- degree,
- clustering coefficient,
- eigenvector centrality.

The key requirement from the project statement is that each statistic must use
shared bin edges across the three compared distributions. This module therefore
computes bins jointly per statistic and applies them consistently.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from metrics import GraphStatisticsSummary, graph_statistics_to_dict
from utils_io import ensure_parent_dir, save_json


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_FIGURE_PATH = DEFAULT_OUTPUT_DIR / "figures" / "graph_statistics_3x3.png"
DEFAULT_TRAINING_HISTORY_FIGURE_PATH = DEFAULT_OUTPUT_DIR / "figures" / "training_history.png"
DEFAULT_BINS_METADATA_PATH = DEFAULT_OUTPUT_DIR / "tables" / "graph_statistics_bins.json"
DEFAULT_NUM_BINS = 20


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class StatisticBins:
    """Container with shared histogram bin edges for the three statistics."""

    degree_bins: np.ndarray
    clustering_bins: np.ndarray
    eigenvector_bins: np.ndarray


# -----------------------------------------------------------------------------
# Filesystem helpers
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Bin computation helpers
# -----------------------------------------------------------------------------


def _safe_range(values: np.ndarray) -> tuple[float, float]:
    """Return a safe numeric range for histogram bins.

    If all values are identical, widen the range slightly so matplotlib can
    build a visible histogram.
    """
    if values.size == 0:
        return 0.0, 1.0

    vmin = float(np.min(values))
    vmax = float(np.max(values))

    if np.isclose(vmin, vmax):
        delta = 0.5 if vmin == 0.0 else 0.05 * abs(vmin)
        return vmin - delta, vmax + delta

    return vmin, vmax



def _compute_shared_bins(
    arrays: list[np.ndarray],
    num_bins: int,
) -> np.ndarray:
    """Compute shared histogram bins across multiple arrays."""
    non_empty = [arr for arr in arrays if arr.size > 0]
    if not non_empty:
        return np.linspace(0.0, 1.0, num_bins + 1)

    all_values = np.concatenate(non_empty, axis=0)
    vmin, vmax = _safe_range(all_values)
    return np.linspace(vmin, vmax, num_bins + 1)



def compute_statistic_bins(
    training_stats: GraphStatisticsSummary,
    baseline_stats: GraphStatisticsSummary,
    deep_stats: GraphStatisticsSummary,
    num_bins: int = DEFAULT_NUM_BINS,
) -> StatisticBins:
    """Compute shared bins for each statistic across the three distributions."""
    degree_bins = _compute_shared_bins(
        arrays=[
            np.asarray(training_stats.degree_values, dtype=float),
            np.asarray(baseline_stats.degree_values, dtype=float),
            np.asarray(deep_stats.degree_values, dtype=float),
        ],
        num_bins=num_bins,
    )
    clustering_bins = _compute_shared_bins(
        arrays=[
            np.asarray(training_stats.clustering_values, dtype=float),
            np.asarray(baseline_stats.clustering_values, dtype=float),
            np.asarray(deep_stats.clustering_values, dtype=float),
        ],
        num_bins=num_bins,
    )
    eigenvector_bins = _compute_shared_bins(
        arrays=[
            np.asarray(training_stats.eigenvector_values, dtype=float),
            np.asarray(baseline_stats.eigenvector_values, dtype=float),
            np.asarray(deep_stats.eigenvector_values, dtype=float),
        ],
        num_bins=num_bins,
    )

    return StatisticBins(
        degree_bins=degree_bins,
        clustering_bins=clustering_bins,
        eigenvector_bins=eigenvector_bins,
    )


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------


def _plot_single_histogram(
    ax: plt.Axes,
    values: list[float],
    bins: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str = "Count",
) -> None:
    """Plot a single histogram on a provided axis."""
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)



def plot_graph_statistics_grid(
    training_stats: GraphStatisticsSummary,
    baseline_stats: GraphStatisticsSummary,
    deep_stats: GraphStatisticsSummary,
    bins: StatisticBins,
    figure_path: str | Path = DEFAULT_FIGURE_PATH,
) -> None:
    """Create the required 3x3 histogram grid.

    Layout:
        Rows   = training / baseline / deep model
        Cols   = degree / clustering / eigenvector centrality
    """
    figure_path = Path(figure_path)
    ensure_parent_dir(figure_path)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Row 1: training distribution
    _plot_single_histogram(
        ax=axes[0, 0],
        values=training_stats.degree_values,
        bins=bins.degree_bins,
        title="Training distribution — Degree",
        xlabel="Degree",
    )
    _plot_single_histogram(
        ax=axes[0, 1],
        values=training_stats.clustering_values,
        bins=bins.clustering_bins,
        title="Training distribution — Clustering coefficient",
        xlabel="Clustering coefficient",
    )
    _plot_single_histogram(
        ax=axes[0, 2],
        values=training_stats.eigenvector_values,
        bins=bins.eigenvector_bins,
        title="Training distribution — Eigenvector centrality",
        xlabel="Eigenvector centrality",
    )

    # Row 2: baseline
    _plot_single_histogram(
        ax=axes[1, 0],
        values=baseline_stats.degree_values,
        bins=bins.degree_bins,
        title="Erdos-Renyi baseline — Degree",
        xlabel="Degree",
    )
    _plot_single_histogram(
        ax=axes[1, 1],
        values=baseline_stats.clustering_values,
        bins=bins.clustering_bins,
        title="Erdos-Renyi baseline — Clustering coefficient",
        xlabel="Clustering coefficient",
    )
    _plot_single_histogram(
        ax=axes[1, 2],
        values=baseline_stats.eigenvector_values,
        bins=bins.eigenvector_bins,
        title="Erdos-Renyi baseline — Eigenvector centrality",
        xlabel="Eigenvector centrality",
    )

    # Row 3: deep model
    _plot_single_histogram(
        ax=axes[2, 0],
        values=deep_stats.degree_values,
        bins=bins.degree_bins,
        title="Deep generative model — Degree",
        xlabel="Degree",
    )
    _plot_single_histogram(
        ax=axes[2, 1],
        values=deep_stats.clustering_values,
        bins=bins.clustering_bins,
        title="Deep generative model — Clustering coefficient",
        xlabel="Clustering coefficient",
    )
    _plot_single_histogram(
        ax=axes[2, 2],
        values=deep_stats.eigenvector_values,
        bins=bins.eigenvector_bins,
        title="Deep generative model — Eigenvector centrality",
        xlabel="Eigenvector centrality",
    )

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Serialization helpers
# -----------------------------------------------------------------------------


def bins_to_dict(bins: StatisticBins) -> dict[str, Any]:
    """Convert StatisticBins to a JSON-serializable dictionary."""
    return {
        "degree_bins": bins.degree_bins.tolist(),
        "clustering_bins": bins.clustering_bins.tolist(),
        "eigenvector_bins": bins.eigenvector_bins.tolist(),
    }


# -----------------------------------------------------------------------------
# Convenience pipeline
# -----------------------------------------------------------------------------


def save_plot_and_bins(
    training_stats: GraphStatisticsSummary,
    baseline_stats: GraphStatisticsSummary,
    deep_stats: GraphStatisticsSummary,
    num_bins: int = DEFAULT_NUM_BINS,
    figure_path: str | Path = DEFAULT_FIGURE_PATH,
    bins_metadata_path: str | Path = DEFAULT_BINS_METADATA_PATH,
) -> StatisticBins:
    """Compute shared bins, generate the 3x3 grid, and save metadata."""
    bins = compute_statistic_bins(
        training_stats=training_stats,
        baseline_stats=baseline_stats,
        deep_stats=deep_stats,
        num_bins=num_bins,
    )

    plot_graph_statistics_grid(
        training_stats=training_stats,
        baseline_stats=baseline_stats,
        deep_stats=deep_stats,
        bins=bins,
        figure_path=figure_path,
    )

    save_json(bins_to_dict(bins), Path(bins_metadata_path))
    return bins


def plot_training_history(
    history: list[dict[str, float]],
    figure_path: str | Path = DEFAULT_TRAINING_HISTORY_FIGURE_PATH,
) -> None:
    """Plot training and validation metrics from a model training history."""
    if not history:
        raise ValueError("Training history must not be empty.")

    epochs = [int(entry["epoch"]) for entry in history]
    figure_path = Path(figure_path)
    ensure_parent_dir(figure_path)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    def _plot_series(ax: plt.Axes, metric_keys: list[str], legend_title: str) -> None:
        for key in metric_keys:
            values = [float(entry[key]) for entry in history if key in entry]
            if values:
                ax.plot(epochs[: len(values)], values, marker="o", label=key)
        ax.set_ylabel(legend_title)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.25)

    _plot_series(axes[0], ["train_loss", "val_loss"], "Loss")
    _plot_series(
        axes[1],
        ["train_recon_loss", "val_recon_loss", "train_kl_loss", "val_kl_loss"],
        "Component Loss",
    )

    axes[1].set_xlabel("Epoch")
    fig.suptitle("Training History")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_training_history_plot(
    history: list[dict[str, float]],
    figure_path: str | Path = DEFAULT_TRAINING_HISTORY_FIGURE_PATH,
) -> None:
    """Save a plot of training history to disk."""
    plot_training_history(history, figure_path)


# -----------------------------------------------------------------------------
# Quick manual check
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Small synthetic sanity check.
    training_stats = GraphStatisticsSummary(
        degree_values=[1, 2, 2, 3, 4, 2],
        clustering_values=[0.0, 0.2, 0.5, 0.3],
        eigenvector_values=[0.1, 0.2, 0.35, 0.4],
    )
    baseline_stats = GraphStatisticsSummary(
        degree_values=[0, 1, 1, 2, 3],
        clustering_values=[0.0, 0.0, 0.1, 0.2],
        eigenvector_values=[0.05, 0.1, 0.15, 0.2],
    )
    deep_stats = GraphStatisticsSummary(
        degree_values=[1, 2, 3, 3, 4],
        clustering_values=[0.1, 0.2, 0.25, 0.4],
        eigenvector_values=[0.08, 0.18, 0.28, 0.38],
    )

    bins = save_plot_and_bins(
        training_stats=training_stats,
        baseline_stats=baseline_stats,
        deep_stats=deep_stats,
    )

    print("Plotting sanity check completed.")
    print(f"Saved figure to: {DEFAULT_FIGURE_PATH}")
    print(f"Saved bins to: {DEFAULT_BINS_METADATA_PATH}")
    print(f"Degree bins: {bins.degree_bins.tolist()}")