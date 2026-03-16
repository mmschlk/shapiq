Approximators
=============

Algorithms for approximating Shapley values and interaction indices.

.. currentmodule:: shapiq

.. autosummary::
   :nosignatures:

   PermutationSamplingSV
   PermutationSamplingSII
   PermutationSamplingSTII
   KernelSHAP
   UnbiasedKernelSHAP
   kADDSHAP
   KernelSHAPIQ
   InconsistentKernelSHAPIQ
   RegressionFSII
   RegressionFBII
   SHAPIQ
   SVARM
   SVARMIQ
   OwenSamplingSV
   StratifiedSamplingSV
   SPEX
   ProxySPEX
   approximator.ProxySHAP
   approximator.MSRBiased

Permutation-based
~~~~~~~~~~~~~~~~~

.. autoclass:: PermutationSamplingSV
   :members:
   :no-private-members:
   :show-inheritance:

.. autoclass:: PermutationSamplingSII
   :members:
   :no-private-members:
   :show-inheritance:

.. autoclass:: PermutationSamplingSTII
   :members:
   :no-private-members:
   :show-inheritance:

Regression-based
~~~~~~~~~~~~~~~~

.. autoclass:: KernelSHAP
   :members:
   :no-private-members:
   :show-inheritance:

.. autoclass:: UnbiasedKernelSHAP
   :members:
   :no-private-members:
   :show-inheritance:

.. autoclass:: kADDSHAP
   :members:
   :no-private-members:
   :show-inheritance:

.. autoclass:: KernelSHAPIQ
   :members:
   :no-private-members:
   :show-inheritance:

.. autoclass:: InconsistentKernelSHAPIQ
   :members:
   :no-private-members:
   :show-inheritance:

.. autoclass:: RegressionFSII
   :members:
   :no-private-members:
   :show-inheritance:

.. autoclass:: RegressionFBII
   :members:
   :no-private-members:
   :show-inheritance:

Monte Carlo
~~~~~~~~~~~

.. autoclass:: SHAPIQ
   :members:
   :no-private-members:
   :show-inheritance:

.. autoclass:: SVARM
   :members:
   :no-private-members:
   :show-inheritance:

.. autoclass:: SVARMIQ
   :members:
   :no-private-members:
   :show-inheritance:

Marginal Sampling
~~~~~~~~~~~~~~~~~

.. autoclass:: OwenSamplingSV
   :members:
   :no-private-members:
   :show-inheritance:

.. autoclass:: StratifiedSamplingSV
   :members:
   :no-private-members:
   :show-inheritance:

Sparse
~~~~~~

.. autoclass:: SPEX
   :members:
   :no-private-members:
   :show-inheritance:

.. autoclass:: ProxySPEX
   :members:
   :no-private-members:
   :show-inheritance:

Proxy
~~~~~

.. currentmodule:: shapiq.approximator

.. autoclass:: ProxySHAP
   :members:
   :no-private-members:
   :show-inheritance:

.. autoclass:: MSRBiased
   :members:
   :no-private-members:
   :show-inheritance:

.. currentmodule:: shapiq

Approximator Groups
~~~~~~~~~~~~~~~~~~~

.. autodata:: shapiq.approximator.SV_APPROXIMATORS
.. autodata:: shapiq.approximator.SI_APPROXIMATORS
.. autodata:: shapiq.approximator.SII_APPROXIMATORS
.. autodata:: shapiq.approximator.STII_APPROXIMATORS
.. autodata:: shapiq.approximator.FSII_APPROXIMATORS
.. autodata:: shapiq.approximator.FBII_APPROXIMATORS
