"""High-level interfaces for explaining ML model predictions.

The main entry point is :class:`Explainer`, which auto-selects the best
explanation method for a given model type. For more control, use one of the
specialised explainers directly:

- :class:`TabularExplainer` — tabular data via imputation-based games
- :class:`TabPFNExplainer` — fast in-context learning with TabPFN
- :class:`AgnosticExplainer` — any callable model
- :class:`TreeExplainer` — optimised for tree-based models

.. seealso::
   :doc:`Examples & Tutorials </auto_examples/index>` for end-to-end examples.
"""

from shapiq.tree.explainer import TreeExplainer

from .agnostic import AgnosticExplainer
from .base import Explainer
from .tabpfn import TabPFNExplainer
from .tabular import TabularExplainer

__all__ = ["Explainer", "TabularExplainer", "TabPFNExplainer", "AgnosticExplainer", "TreeExplainer"]
