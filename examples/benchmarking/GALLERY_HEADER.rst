Games and Benchmarks
====================

.. container:: shapiq-gallery-hero

   .. image:: /_static/gallery/benchmarking.svg
      :alt: Approximation error versus budget for three approximators
      :class: shapiq-gallery-hero-image

   .. container:: shapiq-gallery-hero-text

      ``shapiq_games`` collects cooperative games to test methods on: synthetic games with known
      values, games of tree, nearest-neighbor, kernel, and graph neural network models with exact
      values for many players, and explanation, data valuation, feature selection, vision, and language games.
      ``shapiq_benchmark`` pairs a game with a computer of its exact values, runs approximators
      over budgets and seeds, and scores them with error, ranking, and faithfulness metrics.

      Use these examples to find the approximator that suits your game and budget, or to evaluate
      a new approximator against the existing ones.

.. rst-class:: shapiq-gallery-api

**Key API:** :class:`~shapiq_benchmark.Benchmark` · :func:`~shapiq_benchmark.run` ·
:func:`~shapiq_benchmark.metrics.compare` · :class:`~shapiq_games.SOUM` ·
:class:`~shapiq_games.TabularLocalExplanation`
