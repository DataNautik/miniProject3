"""Shared I/O and device utilities for Mini-project 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory of a file path if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Save a dictionary to JSON with indentation."""
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2))


def get_device() -> str:
    """Return the best available device string (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
