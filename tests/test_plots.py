"""Tests for plotting utilities."""

from __future__ import annotations

from pathlib import Path

from plots import GraphStatisticsSummary, save_training_history_plot


def test_save_training_history_plot(tmp_path: Path) -> None:
    history = [
        {
            "epoch": 1,
            "train_loss": 1.0,
            "val_loss": 1.2,
            "train_recon_loss": 0.8,
            "val_recon_loss": 0.9,
            "train_kl_loss": 0.2,
            "val_kl_loss": 0.3,
        },
        {
            "epoch": 2,
            "train_loss": 0.9,
            "val_loss": 1.1,
            "train_recon_loss": 0.7,
            "val_recon_loss": 0.85,
            "train_kl_loss": 0.2,
            "val_kl_loss": 0.25,
        },
    ]

    figure_path = tmp_path / "training_history.png"
    save_training_history_plot(history, figure_path)

    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
