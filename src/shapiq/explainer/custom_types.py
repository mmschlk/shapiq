"""Types for the shapiq explainer module."""

from __future__ import annotations

from typing import Literal

ExplainerIndices = Literal[
    "SV", "SII", "k-SII", "STII", "FSII", "BV", "BII", "FBII", "Moebius", "Fourier"
]
# the explainers estimating with approximators have no route to the Moebius and Fourier transforms
ValidApproximationExplainerIndices = Literal[
    "SV", "SII", "k-SII", "STII", "FSII", "BV", "BII", "FBII"
]
ValidNNExplainerIndices = Literal["SV"]
ValidProductKernelExplainerIndices = Literal["SV"]
