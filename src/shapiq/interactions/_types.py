from __future__ import annotations

from typing import Literal

type InteractionIndexName = Literal["SV", "BV", "SII", "BII", "STII", "FSII"]
type InteractionOrientation = Literal["undirected", "directed"]
type Interaction = tuple[int, ...]
type OrderSemantics = Literal["coverage", "identity"]
