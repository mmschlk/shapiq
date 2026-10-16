Examples & Tutorials
====================

.. rst-class:: shapiq-examples-lead

Runnable examples of ``shapiq``, ``shapiq_games``, and ``shapiq_benchmark``, from a first
explanation to exact algorithms, approximators, plots, and benchmarks. Every example is a Python
script that you can download or open as a Jupyter notebook.

Where to start
--------------

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: shapiq-paths

   .. grid-item-card:: :octicon:`rocket;1.1em` Explain a model
      :class-card: shapiq-path-card

      Explain the predictions of your model with Shapley values and interactions, and plot them.

      .. rst-class:: shapiq-path-steps

      :doc:`Getting Started <auto_examples/basics/index>`
      :doc:`Visualization <auto_examples/visualization/index>`
      :doc:`Applications <auto_examples/applications/index>`

   .. grid-item-card:: :octicon:`zap;1.1em` Exact values for my model
      :class-card: shapiq-path-card

      Trees, nearest neighbors, and kernel methods have exact algorithms: no sampling, no budget.

      .. rst-class:: shapiq-path-steps

      :doc:`Tree Ensembles <auto_examples/trees/index>`
      :doc:`Nearest Neighbors <auto_examples/nearest_neighbors/index>`
      :doc:`Kernel Methods <auto_examples/kernel_methods/index>`

   .. grid-item-card:: :octicon:`mortar-board;1.1em` Learn the game theory
      :class-card: shapiq-path-card

      Cooperative games, the Shapley value, interaction indices, and the core, computed exactly.

      .. rst-class:: shapiq-path-steps

      :doc:`Getting Started <auto_examples/basics/index>`
      :doc:`Exact Computation <auto_examples/exact_computation/index>`

   .. grid-item-card:: :octicon:`graph;1.1em` Compare or develop approximators
      :class-card: shapiq-path-card

      Choose an approximator for an index and a budget, or benchmark a new one on exact values.

      .. rst-class:: shapiq-path-steps

      :doc:`Approximators <auto_examples/approximators/index>`
      :doc:`Games and Benchmarks <auto_examples/benchmarking/index>`

All examples by topic
---------------------

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: shapiq-groups

   .. grid-item-card:: Getting Started
      :img-top: /_static/gallery/basics.svg
      :img-alt: Three players sharing a payout
      :link: auto_examples/basics/index
      :link-type: doc
      :class-card: shapiq-group-card

      Your first Shapley values and interactions: a small game, a scikit-learn model, and games of
      your own.

   .. grid-item-card:: Exact Computation
      :img-top: /_static/gallery/exact_computation.svg
      :img-alt: The lattice of all coalitions of three players
      :link: auto_examples/exact_computation/index
      :link-type: doc
      :class-card: shapiq-group-card

      Every index of a small game from all its coalitions, and the core.

   .. grid-item-card:: Approximators
      :img-top: /_static/gallery/approximators.svg
      :img-alt: Estimates converging to the true value
      :link: auto_examples/approximators/index
      :link-type: doc
      :class-card: shapiq-group-card

      All approximators of Shapley values and interactions, and which one to choose.

   .. grid-item-card:: Tree Ensembles
      :img-top: /_static/gallery/trees.svg
      :img-alt: A decision tree with the path of one prediction highlighted
      :link: auto_examples/trees/index
      :link-type: doc
      :class-card: shapiq-group-card

      Exact values and interactions of decision trees, random forests, and gradient boosting.

   .. grid-item-card:: Nearest Neighbors
      :img-top: /_static/gallery/nearest_neighbors.svg
      :img-alt: A query point and its nearest neighbors
      :link: auto_examples/nearest_neighbors/index
      :link-type: doc
      :class-card: shapiq-group-card

      Exact data valuation of the training points of nearest-neighbor classifiers.

   .. grid-item-card:: Gaussian Processes and Kernel Methods
      :img-top: /_static/gallery/kernel_methods.svg
      :img-alt: A Gaussian-process mean with its uncertainty band
      :link: auto_examples/kernel_methods/index
      :link-type: doc
      :class-card: shapiq-group-card

      Exact Shapley values of Gaussian processes and support vector machines.

   .. grid-item-card:: Visualization
      :img-top: /_static/gallery/visualization.svg
      :img-alt: A force plot, a bar plot, and an interaction graph
      :link: auto_examples/visualization/index
      :link-type: doc
      :class-card: shapiq-group-card

      Force, waterfall, network, SI graph, UpSet, bar, beeswarm, and scatter plots.

   .. grid-item-card:: Applications
      :img-top: /_static/gallery/applications.svg
      :img-alt: Image patches, text tokens, and a table
      :link: auto_examples/applications/index
      :link-type: doc
      :class-card: shapiq-group-card

      Vision transformers, language models, TabPFN, and confounding in causal effects.

   .. grid-item-card:: Games and Benchmarks
      :img-top: /_static/gallery/benchmarking.svg
      :img-alt: Approximation error versus budget for three approximators
      :link: auto_examples/benchmarking/index
      :link-type: doc
      :class-card: shapiq-group-card

      The games of ``shapiq_games``, and benchmarks of approximators with ``shapiq_benchmark``.

   .. grid-item-card:: Technicalities
      :img-top: /_static/gallery/technicalities.svg
      :img-alt: A masked feature vector imputed from background data
      :link: auto_examples/technicalities/index
      :link-type: doc
      :class-card: shapiq-group-card

      Imputation of absent features, and parallel computation.

.. toctree::
   :hidden:
   :maxdepth: 2

   auto_examples/basics/index
   auto_examples/exact_computation/index
   auto_examples/approximators/index
   auto_examples/trees/index
   auto_examples/nearest_neighbors/index
   auto_examples/kernel_methods/index
   auto_examples/visualization/index
   auto_examples/applications/index
   auto_examples/benchmarking/index
   auto_examples/technicalities/index
