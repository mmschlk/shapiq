"""Games of tree models with exact tree-based ground truth."""

from .interventional import InterventionalTreeGame
from .path_dependent import PathDependentTreeGame

__all__ = ["InterventionalTreeGame", "PathDependentTreeGame"]
