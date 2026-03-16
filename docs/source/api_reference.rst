.. _api_reference:

=============
API Reference
=============

This is the API reference for the ``shapiq`` package. Use the summary tables below
to find what you need, then click through to the detailed documentation on each
category page.

.. toctree::
   :hidden:
   :maxdepth: 2

   api/core
   api/explainers
   api/approximators
   api/imputers
   api/plotting
   api/datasets
   api/utilities


Core
----

Fundamental data structures and the cooperative game interface.
See :doc:`api/core` for full details.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class / Function
     - Description
   * - :class:`~shapiq.Game`
     - Base class wrapping any callable as a cooperative game.
   * - :class:`~shapiq.InteractionValues`
     - Central output container for interaction scores.
   * - :class:`~shapiq.ExactComputer`
     - Exact computation of all interaction indices.


Explainers
----------

High-level interfaces for explaining ML model predictions.
See :doc:`api/explainers` for full details.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :class:`~shapiq.Explainer`
     - Main entry point — auto-selects the best explanation method.
   * - :class:`~shapiq.TabularExplainer`
     - Explainer for tabular data using imputation-based games.
   * - :class:`~shapiq.TabPFNExplainer`
     - Explainer leveraging TabPFN for fast in-context learning.
   * - :class:`~shapiq.AgnosticExplainer`
     - Model-agnostic explainer for any callable model.
   * - :class:`~shapiq.TreeExplainer`
     - Optimised explainer for tree-based models.


Approximators
-------------

Algorithms for approximating Shapley values and interaction indices.
See :doc:`api/approximators` for full details.

**Permutation-based**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :class:`~shapiq.PermutationSamplingSV`
     - Permutation sampling for Shapley values.
   * - :class:`~shapiq.PermutationSamplingSII`
     - Permutation sampling for the Shapley Interaction Index.
   * - :class:`~shapiq.PermutationSamplingSTII`
     - Permutation sampling for the Shapley-Taylor Interaction Index.

**Regression-based**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :class:`~shapiq.KernelSHAP`
     - Weighted least-squares approximation of Shapley values.
   * - :class:`~shapiq.UnbiasedKernelSHAP`
     - Unbiased variant of KernelSHAP.
   * - :class:`~shapiq.kADDSHAP`
     - k-additive SHAP regression approximation.
   * - :class:`~shapiq.KernelSHAPIQ`
     - Weighted least-squares for Shapley interaction indices.
   * - :class:`~shapiq.InconsistentKernelSHAPIQ`
     - Faster but inconsistent variant of KernelSHAPIQ.
   * - :class:`~shapiq.RegressionFSII`
     - Regression-based Faithful Shapley Interaction Index.
   * - :class:`~shapiq.RegressionFBII`
     - Regression-based Faithful Banzhaf Interaction Index.

**Monte Carlo**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :class:`~shapiq.SHAPIQ`
     - ShapIQ algorithm for any-order interaction indices.
   * - :class:`~shapiq.SVARM`
     - SVARM algorithm for Shapley value estimation.
   * - :class:`~shapiq.SVARMIQ`
     - SVARM-IQ algorithm for interaction indices.

**Marginal Sampling**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :class:`~shapiq.OwenSamplingSV`
     - Owen sampling for Shapley values.
   * - :class:`~shapiq.StratifiedSamplingSV`
     - Stratified sampling for Shapley values.

**Sparse**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :class:`~shapiq.SPEX`
     - Sparse explanation approximation.
   * - :class:`~shapiq.ProxySPEX`
     - Proxy-based sparse explanation approximation.

**Proxy**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :class:`~shapiq.approximator.ProxySHAP`
     - Proxy-model-based SHAP approximation.
   * - :class:`~shapiq.approximator.MSRBiased`
     - Biased MSR approximation via proxy models.


Imputers
--------

Imputation strategies for handling missing coalitions in tabular explanations.
See :doc:`api/imputers` for full details.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :class:`~shapiq.MarginalImputer`
     - Marginal distribution imputer.
   * - :class:`~shapiq.GenerativeConditionalImputer`
     - Conditional imputer using a generative model.
   * - :class:`~shapiq.BaselineImputer`
     - Imputer using a fixed baseline value.
   * - :class:`~shapiq.TabPFNImputer`
     - Imputer leveraging TabPFN for in-context learning.
   * - :class:`~shapiq.GaussianImputer`
     - Imputer assuming Gaussian feature distributions.
   * - :class:`~shapiq.GaussianCopulaImputer`
     - Imputer using a Gaussian copula model.


Plotting
--------

Functions for visualizing interaction values and Shapley interactions.
See :doc:`api/plotting` for full details.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`~shapiq.bar_plot`
     - Bar chart of interaction values.
   * - :func:`~shapiq.beeswarm_plot`
     - Beeswarm plot of interaction values.
   * - :func:`~shapiq.force_plot`
     - Force plot showing feature contributions.
   * - :func:`~shapiq.network_plot`
     - Network graph of pairwise interactions.
   * - :func:`~shapiq.sentence_plot`
     - Sentence-level visualization for text data.
   * - :func:`~shapiq.si_graph_plot`
     - Shapley interaction graph plot.
   * - :func:`~shapiq.stacked_bar_plot`
     - Stacked bar chart of interaction values.
   * - :func:`~shapiq.upset_plot`
     - UpSet plot for higher-order interactions.
   * - :func:`~shapiq.waterfall_plot`
     - Waterfall plot of feature contributions.
   * - :func:`~shapiq.plot.abbreviate_feature_names`
     - Shorten feature names for display.


Datasets
--------

Built-in datasets for benchmarking and examples.
See :doc:`api/datasets` for full details.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`~shapiq.load_bike_sharing`
     - Load the bike sharing dataset.
   * - :func:`~shapiq.load_adult_census`
     - Load the adult census dataset.
   * - :func:`~shapiq.load_california_housing`
     - Load the California housing dataset.


Utilities
---------

Low-level utility functions for working with subsets, coalitions, and modules.
See :doc:`api/utilities` for full details.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`~shapiq.powerset`
     - Generate all subsets of a given set.
   * - :func:`~shapiq.get_explicit_subsets`
     - Get explicit subsets from interaction indices.
   * - :func:`~shapiq.split_subsets_budget`
     - Split a budget across subset sizes.
   * - :func:`~shapiq.safe_isinstance`
     - Safe isinstance check with lazy imports.
