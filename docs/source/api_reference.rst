.. _api_reference:

======
shapiq
======

The API of the ``shapiq`` package. The names you need most are importable from the package
itself, e.g. ``shapiq.Explainer`` or ``shapiq.KernelSHAPIQ``; their pages are listed by module
below. The games of :mod:`shapiq_games` and the benchmark :mod:`shapiq_benchmark` ship with
``shapiq`` and have their own pages.

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: shapiq-cards

   .. grid-item-card:: :octicon:`search` Explainers
      :link: generated/shapiq.explainer
      :link-type: doc
      :class-card: shapiq-card

      Explain a model's predictions; ``Explainer`` picks the explainer for the model.

      +++
      shapiq.explainer

   .. grid-item-card:: :octicon:`workflow` Tree Ensembles
      :link: generated/shapiq.tree
      :link-type: doc
      :class-card: shapiq-card

      Exact values and interactions of decision trees and tree ensembles.

      +++
      shapiq.tree

   .. grid-item-card:: :octicon:`share-android` Graph Neural Networks
      :link: generated/shapiq.graph
      :link-type: doc
      :class-card: shapiq-card

      Exact values and interactions of the nodes of graph neural networks with GraphSHAP-IQ.

      +++
      shapiq.graph

   .. grid-item-card:: :octicon:`zap` Approximators
      :link: generated/shapiq.approximator
      :link-type: doc
      :class-card: shapiq-card

      Estimate the values of any game within a budget of evaluations.

      +++
      shapiq.approximator

   .. grid-item-card:: :octicon:`check-circle` Game Theory
      :link: generated/shapiq.game_theory
      :link-type: doc
      :class-card: shapiq-card

      Exact computation, the Möbius transform, and the interaction indices.

      +++
      shapiq.game_theory

   .. grid-item-card:: :octicon:`apps` Games
      :link: generated/shapiq.game
      :link-type: doc
      :class-card: shapiq-card

      The base class of every cooperative game.

      +++
      shapiq.game

   .. grid-item-card:: :octicon:`columns` Imputers
      :link: generated/shapiq.imputer
      :link-type: doc
      :class-card: shapiq-card

      Turn a model and data into a game by removing features.

      +++
      shapiq.imputer

   .. grid-item-card:: :octicon:`table` Interaction Values
      :link: generated/shapiq.interaction_values
      :link-type: doc
      :class-card: shapiq-card

      The result of every explainer, approximator, and computer.

      +++
      shapiq.interaction_values

   .. grid-item-card:: :octicon:`graph` Plots
      :link: generated/shapiq.plot
      :link-type: doc
      :class-card: shapiq-card

      Force, waterfall, bar, beeswarm, scatter, 2-D scatter, network, SI graph, and UpSet
      plots.

      +++
      shapiq.plot

   .. grid-item-card:: :octicon:`tools` Utilities
      :link: generated/shapiq.utils
      :link-type: doc
      :class-card: shapiq-card

      Subsets, coalitions, and other helpers.

      +++
      shapiq.utils


Where to Start
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 60
   :class: shapiq-table

   * - To ...
     - use
   * - explain a model
     - :class:`~shapiq.explainer.Explainer`, which picks one of the explainers below
   * - explain a tree ensemble exactly
     - :class:`~shapiq.explainer.TreeExplainer`
   * - explain a graph neural network exactly
     - :class:`~shapiq.graph.GraphExplainer`
   * - explain any model on tabular data
     - :class:`~shapiq.explainer.TabularExplainer`, with an imputer such as
       :class:`~shapiq.imputer.MarginalImputer`
   * - explain TabPFN
     - :class:`~shapiq.explainer.TabPFNExplainer`
   * - compute every index of a small game exactly
     - :class:`~shapiq.game_theory.ExactComputer`
   * - estimate the values of a large game
     - an approximator, e.g. :class:`~shapiq.approximator.KernelSHAPIQ` or
       :class:`~shapiq.approximator.ProxySPEX`
   * - define your own game
     - a subclass of :class:`~shapiq.game.Game`, or any function of coalitions
   * - read, combine, and save results
     - :class:`~shapiq.interaction_values.InteractionValues`
   * - plot results
     - the methods ``plot_*`` of :class:`~shapiq.interaction_values.InteractionValues`, or
       the functions of :mod:`shapiq.plot`


Modules
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   shapiq.interaction_values
   shapiq.game
   shapiq.explainer
   shapiq.approximator
   shapiq.imputer
   shapiq.game_theory
   shapiq.plot
   shapiq.utils
   shapiq.tree
   shapiq.graph
