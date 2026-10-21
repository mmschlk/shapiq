Getting Started
===============

.. container:: shapiq-gallery-hero

   .. image:: /_static/gallery/basics.svg
      :alt: Three players sharing a payout
      :class: shapiq-gallery-hero-image

   .. container:: shapiq-gallery-hero-text

      New to Shapley values or to ``shapiq``? Start here. The examples introduce cooperative games
      and the Shapley value with a small cooking game, explain a scikit-learn model with Shapley
      values and pairwise Shapley interactions, and show how to write a game of your own, from a
      toy game to a game around any machine learning model.

      :class:`~shapiq.explainer.Explainer` picks a method for your model: an exact algorithm for
      trees, nearest neighbors, and kernel methods, and an approximator for any other model. Every
      result is an :class:`~shapiq.interaction_values.InteractionValues` object to inspect and
      plot.

.. rst-class:: shapiq-gallery-api

**Key API:** :class:`~shapiq.explainer.Explainer` · :class:`~shapiq.explainer.TabularExplainer` ·
:class:`~shapiq.game.Game` · :class:`~shapiq.game_theory.ExactComputer` ·
:class:`~shapiq.interaction_values.InteractionValues`

.. rst-class:: shapiq-gallery-next

**Next:** plot explanations in :doc:`/auto_examples/visualization/index`, or compute every index of
a small game in :doc:`/auto_examples/exact_computation/index`.
