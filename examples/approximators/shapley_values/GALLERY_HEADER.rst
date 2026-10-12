Shapley Values
--------------

.. container:: shapiq-gallery-notes

   Estimators of the Shapley value; :class:`~shapiq.approximator.SVARM` also estimates the Banzhaf
   value. :class:`~shapiq.approximator.KernelSHAP`, the default of the explainers, and
   :class:`~shapiq.approximator.kADDSHAP` solve a weighted regression; permutation, stratified, and
   Owen sampling average marginal contributions; :class:`~shapiq.approximator.SVARM` and
   :class:`~shapiq.approximator.UnbiasedKernelSHAP` update the estimates of all players with every
   sampled coalition. Further estimators without an example here are
   :class:`~shapiq.approximator.LeverageSHAP`, :class:`~shapiq.approximator.OddSHAP`, and
   :class:`~shapiq.approximator.ShaplEIG`.
