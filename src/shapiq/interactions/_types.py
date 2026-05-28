from __future__ import annotations

from typing import Literal

type InteractionIndexName = Literal["SV", "SII", "k-SII", "STII", "FSII"]
type InteractionOrientation = Literal["undirected", "directed"]
type Interaction = tuple[int, ...]
