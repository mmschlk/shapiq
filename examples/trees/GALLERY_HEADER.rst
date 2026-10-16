Tree Ensembles
==============

.. container:: shapiq-gallery-hero

   .. image:: /_static/gallery/trees.svg
      :alt: A decision tree with the path of one prediction highlighted
      :class: shapiq-gallery-hero-image

   .. container:: shapiq-gallery-hero-text

      For decision trees and tree ensembles, :class:`~shapiq.explainer.TreeExplainer` computes
      Shapley values and interactions of any order exactly: it works on the trees themselves
      instead of evaluating coalitions, so it needs no budget and no sampling. It supports
      scikit-learn's decision trees, random forests, and gradient boosting, XGBoost, LightGBM, and
      CatBoost, for regression and classification, and :class:`~shapiq.explainer.Explainer` picks
      it for these models.

      Two modes define what an absent feature means. In the default path-dependent mode, it
      follows both children of a split, weighted by the training data stored in the tree (indices
      ``SV``, ``SII``, ``k-SII``, ``BV``, ``BII``, ``Moebius``, ``Fourier``). In the interventional
      mode, it takes its values from a reference dataset (additionally ``STII``, ``FSII``,
      ``FBII``).

.. rst-class:: shapiq-gallery-api

**Key API:** :class:`~shapiq.explainer.TreeExplainer` · :class:`~shapiq.tree.TreeModel` (custom
trees) · :class:`~shapiq.tree.QuadratureTreeSHAP` · :class:`~shapiq.tree.InterventionalTreeSHAPIQ`

.. rst-class:: shapiq-gallery-next

**Next:** plot the explanations of many instances in :doc:`/auto_examples/visualization/index`, or
benchmark approximators on tree games in :doc:`/auto_examples/benchmarking/index`.
