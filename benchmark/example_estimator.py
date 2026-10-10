"""Minimal local adapter; replace the constructor with a paper's implementation."""

from __future__ import annotations

from shapiq.approximator import KernelSHAP


def factory(n: int, index: str, order: int, seed: int) -> KernelSHAP:
    """Return an object with approximate(budget, game) -> InteractionValues."""
    if index != "SV" or order != 1:
        msg = "This example adapter supports Shapley values only."
        raise ValueError(msg)
    return KernelSHAP(n=n, random_state=seed)
