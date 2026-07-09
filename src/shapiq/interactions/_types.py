from __future__ import annotations

from typing import Literal

type InteractionIndexName = Literal[
    "SV",
    "BV",
    "WeightedBV",
    "SII",
    "BII",
    "WeightedBII",
    "CHII",
    "k-SII",
    "STII",
    "FSII",
    "FBII",
    "WeightedFBII",
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
