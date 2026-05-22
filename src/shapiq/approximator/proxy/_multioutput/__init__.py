"""Experimental multi-output (vector-valued leaf) tree support for :class:`ProxySHAP`.

This private subpackage is an experimental "code addendum" that extends shapiq's
``ProxySHAP`` approximator towards *multivariate* value functions (output vector in
``R^c``). It is deliberately kept separate from the shared shapiq tree code
(``shapiq.tree.base.TreeModel``, the C parsers, the existing explainers): nothing in
this package modifies shared shapiq source.

Contents:

* :mod:`tree` -- a pure-Python parser that turns a fitted
  ``XGBRegressor(multi_strategy="multi_output_tree")`` into a list of
  :class:`MultiOutputTreeModel` containers whose ``predict`` reproduces
  ``model.predict``.
* :mod:`explainer` -- :class:`MultiOutputInterventionalTreeExplainer`, the
  multivariate counterpart of
  :class:`shapiq.tree.interventional.explainer.InterventionalTreeExplainer`,
  driven by the fused multi-output C kernel.
* :mod:`game` -- :class:`MultiOutputMarginalGame`, a self-contained marginal
  (interventional) multivariate value function for a multiclass classifier.
* :mod:`proxyshap` -- :class:`MultiOutputProxySHAP`, the non-adjustment
  ProxySHAP approximator for multivariate value functions.
"""

from __future__ import annotations

from .explainer import (
    MultiOutputInterventionalTreeExplainer,
    build_offset_to_tuple_map,
)
from .game import MultiOutputMarginalGame
from .proxyshap import MultiOutputProxySHAP
from .tree import (
    MultiOutputTreeModel,
    convert_multioutput_xgboost,
    convert_multioutput_xgboost_with_base_score,
    predict_multioutput,
)

__all__ = [
    "MultiOutputInterventionalTreeExplainer",
    "MultiOutputMarginalGame",
    "MultiOutputProxySHAP",
    "MultiOutputTreeModel",
    "build_offset_to_tuple_map",
    "convert_multioutput_xgboost",
    "convert_multioutput_xgboost_with_base_score",
    "predict_multioutput",
]
