Visualization
=============

.. container:: shapiq-gallery-hero

   .. image:: /_static/gallery/visualization.svg
      :alt: A force plot, a bar plot, and an interaction graph
      :class: shapiq-gallery-hero-image

   .. container:: shapiq-gallery-hero-text

      Every explanation in ``shapiq`` is an :class:`~shapiq.interaction_values.InteractionValues`
      object, and its plots show it from different angles. Local plots explain one prediction:
      the force and waterfall plots split it into contributions, the network and SI graph plots
      draw the interactions between features, the stacked bar plot splits the contribution of
      every feature by interaction order, and the UpSet plot ranks the strongest interactions.
      Global plots summarize the explanations of many instances: the bar, beeswarm, and scatter
      plots.

      Start with the object itself and the catalog of all plots, then go into the details of a
      single plot.

.. rst-class:: shapiq-gallery-api

**Key API:** :class:`~shapiq.interaction_values.InteractionValues` ·
:func:`~shapiq.plot.force_plot` · :func:`~shapiq.plot.waterfall_plot` ·
:func:`~shapiq.plot.network_plot` · :func:`~shapiq.plot.si_graph_plot` ·
:func:`~shapiq.plot.stacked_bar_plot` · :func:`~shapiq.plot.upset_plot` ·
:func:`~shapiq.plot.bar_plot` · :func:`~shapiq.plot.beeswarm_plot` ·
:func:`~shapiq.plot.scatter_plot` · :func:`~shapiq.plot.sentence_plot`
