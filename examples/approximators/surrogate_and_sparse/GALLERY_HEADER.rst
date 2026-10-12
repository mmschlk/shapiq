Surrogate and Sparse Methods
----------------------------

.. container:: shapiq-gallery-notes

   Approximators that learn a model of the game. :class:`~shapiq.approximator.RegressionMSR` and
   :class:`~shapiq.approximator.ProxySHAP` fit a surrogate to the sampled coalition values, read the
   values off it exactly, and can correct its error with a Monte Carlo estimate on the residuals.
   :class:`~shapiq.approximator.SPEX` and :class:`~shapiq.approximator.ProxySPEX` recover a sparse
   Fourier representation of the game, which lets them explain interactions among many players.
