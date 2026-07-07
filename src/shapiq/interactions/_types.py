from __future__ import annotations

from typing import Literal

type InteractionIndexName = Literal[
    "SV",
    "BV",
    "SII",
    "BII",
    "CHII",
    "k-SII",
    "STII",
    "FSII",
    "FBII",
    "kADD-SHAP",
    "SGV",
    "BGV",
    "CHGV",
    "IGV",
    "EGV",
    "JointSV",
    "Moebius",
    "Co-Moebius",
]
type InteractionOrientation = Literal["undirected", "directed"]
type Interaction = tuple[int, ...]
type OrderSemantics = Literal["coverage", "identity"]
