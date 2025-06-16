"""This module contains all custom types used in the shapiq package."""

from __future__ import annotations

from typing import Literal, TypeVar

# Model type for all machine learning models
Model = TypeVar("Model")
IndexType = Literal[
    "SII",
    "BII",
    "CHII",
    "Co-Moebius",
    "SGV",
    "BGV",
    "CHGV",
    "IGV",
    "EGV",
    "k-SII",
    "STII",
    "FSII",
    "kADD-SHAP",
    "FBII",
    "SV",
    "BV",
    "JointSV",
    "Moebius",
    "ELC",
    "EC",
]
