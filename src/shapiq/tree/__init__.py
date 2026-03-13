"""Implementation of TreeSHAPIQ and the TreeExplainer."""

from .base import TreeModel
from .interventional import InterventionalGame, InterventionalTreeExplainer
from .linear import LinearTreeSHAP
from .treeshapiq import TreeSHAPIQ

__all__ = [
    "TreeSHAPIQ",
    "TreeModel",
    "InterventionalTreeExplainer",
    "InterventionalGame",
    "LinearTreeSHAP",
]


# This function is used to lazily import the TreeExplainer class when it is accessed as an attribute of the module.
def __getattr__(name: str) -> object:
    if name == "TreeExplainer":
        from .explainer import TreeExplainer

        return TreeExplainer
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
