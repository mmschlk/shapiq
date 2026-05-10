"""Docstring."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from .base import GraphModel


def validate_graph_model(model: torch.nn.Module) -> list[GraphModel]:
    """Docstring."""
    # TO DO
    return [model]
