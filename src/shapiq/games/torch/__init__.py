"""Torch game adapters."""

from __future__ import annotations

from shapiq.games.torch._callable import TorchCallableGame
from shapiq.games.torch._convert import to_jax
from shapiq.games.torch._masker import BaselineMasker

__all__ = ["BaselineMasker", "TorchCallableGame", "to_jax"]
