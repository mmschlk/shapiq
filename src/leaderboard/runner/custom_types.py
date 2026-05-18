"""Custom types for the leaderboard runner."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from shapiq import InteractionValues

InteractionIndex = Literal[
    "SV",
    "BV",
    "SII",
    "BII",
    "k-SII",
    "STII",
    "FBII",
    "FSII",
    "kADD-SHAP",
    "CHII",
]

MetricFunction = Callable[[InteractionValues, InteractionValues], float]
