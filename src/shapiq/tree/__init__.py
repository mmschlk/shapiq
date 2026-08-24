"""Tree-based explanation algorithms, including TreeSHAP-IQ.

Provides :class:`TreeSHAPIQ` for computing exact Shapley interaction values on tree ensembles,
along with supporting data structures and algorithm variants.
"""

from .base import TreeModel
from .interventional import InterventionalGame, InterventionalTreeSHAPIQ
from .linear import LinearTreeSHAP
from .precision import TreeNumericalPrecisionError, TreeNumericalPrecisionWarning
from .quadrature import QuadratureTreeSHAP
from .treeshapiq import TreeSHAPIQ

__all__ = [
    "TreeSHAPIQ",
    "TreeModel",
    "InterventionalTreeSHAPIQ",
    "InterventionalGame",
    "LinearTreeSHAP",
    "QuadratureTreeSHAP",
    "TreeNumericalPrecisionError",
    "TreeNumericalPrecisionWarning",
]


# This function is used to lazily import the TreeExplainer class (and its warning class) when
# accessed as an attribute of the module.
def __getattr__(name: str) -> object:
    if name == "TreeExplainer":
        from .explainer import TreeExplainer

        return TreeExplainer
    if name == "WoodelfNotAvailableWarning":
        from .explainer import WoodelfNotAvailableWarning

        return WoodelfNotAvailableWarning
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
