from __future__ import annotations

from abc import ABC

from shapiq.explainers._base import Explainer
from shapiq.games import Game


class ExactExplainer[ValueT, GameT: Game](Explainer[ValueT, GameT], ABC):
    """Marker base for non-sampling explainers."""
