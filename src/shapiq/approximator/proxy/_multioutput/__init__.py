"""Experimental multi-output (vector-valued leaf) tree support for :class:`ProxySHAP`.

This private subpackage is an experimental "code addendum" that extends shapiq's
``ProxySHAP`` approximator towards *multivariate* value functions (output vector in
``R^c``). It is deliberately kept separate from the shared shapiq tree code
(``shapiq.tree.base.TreeModel``, the C parsers, the existing explainers): nothing in
this package modifies shared shapiq source.

The current contents cover Phase 1 only: a pure-Python parser that turns a fitted
``XGBRegressor(multi_strategy="multi_output_tree")`` into a list of
:class:`MultiOutputTreeModel` containers whose ``predict`` reproduces
``model.predict``.
"""

from __future__ import annotations

from .tree import (
    MultiOutputTreeModel,
    convert_multioutput_xgboost,
    convert_multioutput_xgboost_with_base_score,
    predict_multioutput,
)

__all__ = [
    "MultiOutputTreeModel",
    "convert_multioutput_xgboost",
    "convert_multioutput_xgboost_with_base_score",
    "predict_multioutput",
]
