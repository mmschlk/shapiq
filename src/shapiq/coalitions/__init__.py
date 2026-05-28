"""Coalition array abstractions."""

from __future__ import annotations

from shapiq.coalitions._base import CoalitionArray
from shapiq.coalitions._dense import DenseCoalitionArray

__all__ = ["CoalitionArray", "DenseCoalitionArray"]
