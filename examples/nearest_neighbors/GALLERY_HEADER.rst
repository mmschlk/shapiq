Nearest Neighbors
=================

.. container:: shapiq-gallery-hero

   .. image:: /_static/gallery/nearest_neighbors.svg
      :alt: A query point and its nearest neighbors
      :class: shapiq-gallery-hero-image

   .. container:: shapiq-gallery-hero-text

      For nearest-neighbor classifiers, ``shapiq`` values the *training data*: the players are the
      training points, and the Shapley value of a point measures how much it contributes to the
      model's probability of a class at a test point. These values are computed exactly, without
      enumerating coalitions, even for large training sets.

      Use them for data valuation, for example to find mislabeled or redundant training points.
      :class:`~shapiq.explainer.KNNExplainer` explains a ``KNeighborsClassifier`` with uniform
      weights, :class:`~shapiq.explainer.WeightedKNNExplainer` one with distance weights, and
      :class:`~shapiq.explainer.ThresholdNNExplainer` a ``RadiusNeighborsClassifier``;
      :class:`~shapiq.explainer.Explainer` picks the right one.

.. rst-class:: shapiq-gallery-api

**Key API:** :class:`~shapiq.explainer.KNNExplainer` ·
:class:`~shapiq.explainer.WeightedKNNExplainer` ·
:class:`~shapiq.explainer.ThresholdNNExplainer` · :class:`~shapiq_games.KNNGame` (the game, for
benchmarks)
