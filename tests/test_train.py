"""Tests for training configuration validation."""

from __future__ import annotations

import pytest

from train_gvae import TrainConfig


def test_train_config_rejects_invalid_epochs() -> None:
    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        TrainConfig(epochs=0)


def test_train_config_rejects_negative_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        TrainConfig(learning_rate=0.0)


def test_train_config_rejects_negative_beta() -> None:
    with pytest.raises(ValueError, match="beta must be non-negative"):
        TrainConfig(beta=-1.0)
