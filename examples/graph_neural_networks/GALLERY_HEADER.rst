Graph Neural Networks
=====================

.. container:: shapiq-gallery-hero

   .. image:: /_static/gallery/graph_neural_networks.svg
      :alt: A graph with the receptive field of one node highlighted
      :class: shapiq-gallery-hero-image

   .. container:: shapiq-gallery-hero-text

      For graph neural networks, the players are the nodes of a graph.
      :class:`~shapiq.graph.GraphExplainer` computes their Shapley values and interactions of any
      order exactly with GraphSHAP-IQ: a node's embedding depends only on its receptive field, the
      nodes within as many hops as the network has message passing layers, so the network is
      evaluated only on subsets of receptive fields instead of on all coalitions.

      The values are exact for networks that pool the node embeddings and read them out
      linearly, explained on their logits. :class:`~shapiq.graph.GraphSHAPIQ` computes the values
      of any game that is local on a graph, without ``torch``.

.. rst-class:: shapiq-gallery-api

**Key API:** :class:`~shapiq.graph.GraphExplainer` · :class:`~shapiq.graph.GraphGame` (the game,
also for benchmarks) · :class:`~shapiq.graph.GraphSHAPIQ`

.. rst-class:: shapiq-gallery-next

**Next:** benchmark approximators against the exact values of graph games in
:doc:`/auto_examples/benchmarking/index`.
