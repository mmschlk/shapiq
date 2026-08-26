"""Tree-based explanation algorithms for shapiq.

Provides :class:`TreeExplainer`, the user-facing explainer for tree ensembles, together with
the computation algorithms it builds on: :class:`QuadratureTreeSHAP` (the numerically exact
path-dependent default), :class:`TreeSHAPIQ`, :class:`LinearTreeSHAP`, and
:class:`InterventionalTreeSHAPIQ`.
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
