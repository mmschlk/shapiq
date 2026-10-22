Gaussian Processes and Kernel Methods
=====================================

.. container:: shapiq-gallery-hero

   .. image:: /_static/gallery/kernel_methods.svg
      :alt: A Gaussian-process mean with its uncertainty band
      :class: shapiq-gallery-hero-image

   .. container:: shapiq-gallery-hero-text

      Models that are weighted sums of a product kernel, such as the RBF kernel, have exact
      Shapley values that are computed in polynomial time instead of over all coalitions.
      :class:`~shapiq.explainer.ProductKernelExplainer` computes them for scikit-learn's
      Gaussian-process regressors, support vector regressors, and binary support vector
      classifiers with an RBF kernel, and :class:`~shapiq.explainer.Explainer` picks it for these
      models.

      The examples explain Gaussian processes and kernel support vector machines. To benchmark
      approximators on these models, :class:`~shapiq_games.ProductKernelGame` turns one into a
      game whose exact Shapley values are known.

.. rst-class:: shapiq-gallery-api

**Key API:** :class:`~shapiq.explainer.ProductKernelExplainer` ·
:class:`~shapiq.explainer.Explainer` · :class:`~shapiq_games.ProductKernelGame`
