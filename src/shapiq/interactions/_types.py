from __future__ import annotations

from typing import Literal

# Explanations record the index name as open metadata: shipped indices use the
# names SV, BV, SII, BII, CHII, k-SII, STII, FSII, FBII, kADD-SHAP, SGV, BGV,
# CHGV, IGV, EGV, JointSV, Moebius, and Co-Moebius, while formalism-defined
# indices introduce their own names.
type InteractionIndexName = str
type InteractionOrientation = Literal["undirected", "directed"]
type Interaction = tuple[int, ...]
type OrderSemantics = Literal["coverage", "identity"]
