"""High-level interfaces for explaining ML model predictions.

The main entry point is :class:`Explainer`, which auto-selects the best
explanation method for a given model type. For more control, use one of the
specialised explainers directly:

- :class:`TabularExplainer` — tabular data via imputation-based games
- :class:`TabPFNExplainer` — fast in-context learning with TabPFN
- :class:`AgnosticExplainer` — any callable model
- :class:`TreeExplainer` — optimised for tree-based models
- :class:`ProductKernelExplainer` — exact for Gaussian processes and support vector machines
- :class:`KNNExplainer`, :class:`WeightedKNNExplainer`, :class:`ThresholdNNExplainer` — exact data
  valuation of nearest-neighbor classifiers

.. seealso::
   :doc:`Examples & Tutorials </examples>` for end-to-end examples.
"""

from shapiq.tree.explainer import TreeExplainer

from .agnostic import AgnosticExplainer
from .base import Explainer
from .nn.knn import KNNExplainer
from .nn.threshold_nn import ThresholdNNExplainer
from .nn.weighted_knn import WeightedKNNExplainer
from .product_kernel import ProductKernelExplainer
from .tabpfn import TabPFNExplainer
from .tabular import TabularExplainer

__all__ = [
    "Explainer",
    "TabularExplainer",
    "TabPFNExplainer",
    "AgnosticExplainer",
    "TreeExplainer",
    "KNNExplainer",
    "WeightedKNNExplainer",
    "ThresholdNNExplainer",
    "ProductKernelExplainer",
]
