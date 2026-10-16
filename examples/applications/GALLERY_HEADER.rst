Applications
============

.. container:: shapiq-gallery-hero

   .. image:: /_static/gallery/applications.svg
      :alt: Image patches, text tokens, and a table
      :class: shapiq-gallery-hero-image

   .. container:: shapiq-gallery-hero-text

      Shapley values and interactions for foundation models and real tasks: the tabular foundation
      model TabPFN, a sentiment classifier whose players are tokens, and the attribution of
      confounding bias in a causal effect estimate to the covariates.

      The games come from ``shapiq_games``, and some examples download a pretrained model on first
      use. They show how to choose the players and the value of a coalition for a new kind of data.

.. rst-class:: shapiq-gallery-api

**Key API:** :class:`~shapiq.explainer.TabPFNExplainer` · :class:`~shapiq_games.SentimentAnalysis` ·
:class:`~shapiq_games.GlobalConfoundingXAI` · :func:`~shapiq.plot.sentence_plot`
