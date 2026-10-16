Shapley Values
--------------

.. container:: shapiq-gallery-notes

   Estimators of the Shapley value; :class:`~shapiq.approximator.SVARM` also estimates the Banzhaf
   value. :class:`~shapiq.approximator.KernelSHAP`, the default of the explainers,
   :class:`~shapiq.approximator.kADDSHAP`, :class:`~shapiq.approximator.LeverageSHAP`, and
   :class:`~shapiq.approximator.OddSHAP` solve a weighted regression; permutation, stratified, and
   Owen sampling average marginal contributions; :class:`~shapiq.approximator.SVARM` and
   :class:`~shapiq.approximator.UnbiasedKernelSHAP` update the estimates of all players with every
   sampled coalition; and :class:`~shapiq.approximator.ShaplEIG` also reports the uncertainty of its
   estimates.
