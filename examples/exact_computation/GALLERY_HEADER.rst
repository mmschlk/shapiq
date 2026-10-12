Exact Computation
=================

.. container:: shapiq-gallery-hero

   .. image:: /_static/gallery/exact_computation.svg
      :alt: The lattice of all coalitions of three players
      :class: shapiq-gallery-hero-image

   .. container:: shapiq-gallery-hero-text

      Shapley values and interactions are weighted averages over all coalitions of players, so
      they can be computed exactly by evaluating the game on all :math:`2^n` coalitions.
      :class:`~shapiq.game_theory.ExactComputer` evaluates them once and derives every concept
      from the stored values: the Möbius transform, the Shapley and Banzhaf values, Shapley
      interactions (SII, k-SII, STII, FSII, FBII, and more), generalized values, and the
      egalitarian least core.

      Use it for small games of up to about 14 players, as every player doubles the number of
      coalitions: to get ground truth for approximators, to compare interaction indices, or to
      study a game in detail. Trees, nearest neighbors, and kernel methods have exact algorithms
      that need no enumeration and scale to many players.

.. rst-class:: shapiq-gallery-api

**Key API:** :class:`~shapiq.game_theory.ExactComputer` ·
:class:`~shapiq.game_theory.MoebiusConverter` · :class:`~shapiq.game.Game`

.. rst-class:: shapiq-gallery-next

**Next:** estimate the same values for larger games in :doc:`/auto_examples/approximators/index`,
or get them exactly for a model class in :doc:`/auto_examples/trees/index`,
:doc:`/auto_examples/nearest_neighbors/index`, and :doc:`/auto_examples/kernel_methods/index`.
