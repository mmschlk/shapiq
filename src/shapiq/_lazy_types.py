"""Fully-qualified type names for lazy backend dispatch.

The names feed ``flextype`` string registrations, which match a class's
``module.qualname`` along its MRO, so optional backends are recognized
without ever being imported by shapiq itself.
"""

from __future__ import annotations

TORCH_TENSOR = "torch.Tensor"

SKLEARN_DECISION_TREE = "sklearn.tree._classes.BaseDecisionTree"
SKLEARN_FOREST = "sklearn.ensemble._forest.BaseForest"
