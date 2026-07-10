"""Torch execution-policy adapters.

Masking is backend-general and lives in ``shapiq.games``; this package owns
the torch call policy — one adapter per entry style: ``TorchCallableGame``
for raw coalition callables and ``ChunkedMaskedPredictor`` for masked
models — plus the ``to_jax`` value boundary.
"""

from __future__ import annotations

from shapiq.games.torch._callable import TorchCallableGame
from shapiq.games.torch._chunked import ChunkedMaskedPredictor
from shapiq.games.torch._convert import to_jax

__all__ = [
    "ChunkedMaskedPredictor",
    "TorchCallableGame",
    "to_jax",
]
