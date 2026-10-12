Approximators
=============

.. container:: shapiq-gallery-hero

   .. image:: /_static/gallery/approximators.svg
      :alt: Estimates converging to the true value
      :class: shapiq-gallery-hero-image

   .. container:: shapiq-gallery-hero-text

      Most games are too large to evaluate on all :math:`2^n` coalitions. An approximator
      estimates Shapley values or interactions from a *budget* of game evaluations, which are
      usually model calls and the costly part of an explanation. All approximators share one
      interface: create one with the number of players ``n`` (and, where it applies, the ``index``
      and the ``max_order``), and ``approximate(budget, game)`` returns an
      :class:`~shapiq.interaction_values.InteractionValues` object.

      Every example runs one approximator on the same synthetic game of eight players with a
      budget of 200 evaluations, so it runs in about a second and the results are comparable.
      The thumbnails show the family of each method.

.. rst-class:: shapiq-gallery-api

**Key API:** :class:`~shapiq.approximator.Approximator` ·
:class:`~shapiq.explainer.TabularExplainer` · :class:`~shapiq.explainer.AgnosticExplainer` ·
:class:`~shapiq.game_theory.ExactComputer` (ground truth for small games)

Which approximator?
-------------------

.. container:: shapiq-gallery-notes

   With ``approximator="auto"``, the explainers choose an approximator for the index; the last
   column lists the other approximators of the index.

.. list-table::
   :header-rows: 1
   :widths: 26 22 52
   :class: shapiq-approximator-guide

   * - Index
     - ``"auto"`` chooses
     - Also supported by
   * - Shapley value (``SV``)
     - :class:`~shapiq.approximator.KernelSHAP`
     - :class:`~shapiq.approximator.LeverageSHAP`, :class:`~shapiq.approximator.kADDSHAP`,
       :class:`~shapiq.approximator.PermutationSamplingSV`,
       :class:`~shapiq.approximator.StratifiedSamplingSV`,
       :class:`~shapiq.approximator.OwenSamplingSV`, :class:`~shapiq.approximator.SVARM`,
       :class:`~shapiq.approximator.UnbiasedKernelSHAP`,
       :class:`~shapiq.approximator.RegressionMSR`, :class:`~shapiq.approximator.OddSHAP`,
       :class:`~shapiq.approximator.ShaplEIG`
   * - Banzhaf value (``BV``)
     - :class:`~shapiq.approximator.RegressionFBII` at order 1
     - :class:`~shapiq.approximator.SVARM`, :class:`~shapiq.approximator.RegressionMSR`,
       :class:`~shapiq.approximator.SHAPIQ`, :class:`~shapiq.approximator.SVARMIQ`
   * - Shapley interactions (``SII``, ``k-SII``)
     - :class:`~shapiq.approximator.KernelSHAPIQ`
     - :class:`~shapiq.approximator.InconsistentKernelSHAPIQ`,
       :class:`~shapiq.approximator.PermutationSamplingSII`, :class:`~shapiq.approximator.SHAPIQ`,
       :class:`~shapiq.approximator.SVARMIQ`
   * - Shapley-Taylor interactions (``STII``)
     - :class:`~shapiq.approximator.SVARMIQ`
     - :class:`~shapiq.approximator.PermutationSamplingSTII`, :class:`~shapiq.approximator.SHAPIQ`
   * - Faithful interactions (``FSII``, ``FBII``)
     - :class:`~shapiq.approximator.RegressionFSII`, :class:`~shapiq.approximator.RegressionFBII`
     - :class:`~shapiq.approximator.SHAPIQ` and :class:`~shapiq.approximator.SVARMIQ` (top order
       only)
   * - Banzhaf and chaining interactions (``BII``, ``CHII``)
     - :class:`~shapiq.approximator.SVARMIQ`
     - :class:`~shapiq.approximator.SHAPIQ`
   * - Interactions among many players
     - :class:`~shapiq.approximator.SPEX` beyond 64 players at order 2, 32 at order 3, and 16 at
       higher orders
     - :class:`~shapiq.approximator.ProxySPEX`

.. container:: shapiq-gallery-notes

   :class:`~shapiq.approximator.ProxySHAP`, :class:`~shapiq.approximator.SPEX`, and
   :class:`~shapiq.approximator.ProxySPEX` support every index of the table but ``CHII``. A few
   rules of thumb about the budget:

   - With a budget of :math:`2^n` or more, the regression approximators, SHAP-IQ, SVARM, SVARM-IQ,
     Unbiased KernelSHAP, and the surrogates with their residual correction evaluate every
     coalition and return the exact values. The exception is
     :class:`~shapiq.approximator.InconsistentKernelSHAPIQ`, which does not converge to them.
   - At small budgets, :class:`~shapiq.approximator.InconsistentKernelSHAPIQ` often estimates SII
     and k-SII better than :class:`~shapiq.approximator.KernelSHAPIQ`.
   - :class:`~shapiq.approximator.ProxySHAP` and :class:`~shapiq.approximator.RegressionMSR` learn
     a surrogate of the game from the budget and correct its error with a Monte Carlo estimate on
     the residuals (``ProxySHAP`` with ``adjustment=True``). :class:`~shapiq.approximator.ShaplEIG`
     chooses every coalition it evaluates by its expected information gain and also returns the
     variances of its estimates.
   - :class:`~shapiq.approximator.SPEX` needs the ``sparse`` extra and
     :class:`~shapiq.approximator.ShaplEIG` the ``shapleig`` extra
     (``pip install shapiq[sparse]``, ``pip install shapiq[shapleig]``).
   - To compare approximators on your own game, see :doc:`/auto_examples/benchmarking/index`.

How they work
-------------

.. grid:: 2 3 4 4
   :gutter: 2
   :class-container: shapiq-family-grid

   .. grid-item-card:: Regression
      :img-top: /_static/gallery/approximators_regression.svg
      :img-alt: A weighted least-squares fit
      :class-card: shapiq-family-card

      Weighted least squares on sampled coalitions: KernelSHAP, KernelSHAP-IQ, the faithful
      regressions, kADD-SHAP.

   .. grid-item-card:: Monte Carlo
      :img-top: /_static/gallery/approximators.svg
      :img-alt: Estimates converging to the true value
      :class-card: shapiq-family-card

      Every sampled coalition updates all estimates: SHAP-IQ, Unbiased KernelSHAP.

   .. grid-item-card:: Stratified sampling
      :img-top: /_static/gallery/approximators_stratified.svg
      :img-alt: Samples stratified by coalition size
      :class-card: shapiq-family-card

      Samples per coalition size: SVARM, SVARM-IQ, stratified sampling.

   .. grid-item-card:: Permutations
      :img-top: /_static/gallery/approximators_permutation.svg
      :img-alt: Random permutations of the players
      :class-card: shapiq-family-card

      Marginal contributions along random orders of the players.

   .. grid-item-card:: Owen sampling
      :img-top: /_static/gallery/approximators_owen.svg
      :img-alt: An integral over the inclusion probability
      :class-card: shapiq-family-card

      An integral over the probability that a player joins a coalition.

   .. grid-item-card:: Surrogates
      :img-top: /_static/gallery/approximators_surrogate.svg
      :img-alt: A tree surrogate fitted to sampled game values
      :class-card: shapiq-family-card

      A model of the game, read off exactly: ProxySHAP, RegressionMSR.

   .. grid-item-card:: Sparse
      :img-top: /_static/gallery/approximators_sparse.svg
      :img-alt: A sparse spectrum with a few large coefficients
      :class-card: shapiq-family-card

      A sparse Fourier representation of the game: SPEX, ProxySPEX.
