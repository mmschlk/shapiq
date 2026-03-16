Approximators
=============

Algorithms for approximating Shapley values and interaction indices.

.. currentmodule:: shapiq.approximator

.. autosummary::
   :nosignatures:

   ~shapiq.approximator.Approximator
   ~shapiq.approximator.regression.Regression
   ~shapiq.approximator.montecarlo.MonteCarlo
   ~shapiq.approximator.sparse.Sparse
   ~shapiq.approximator.permutation.PermutationSamplingSV
   ~shapiq.approximator.permutation.PermutationSamplingSII
   ~shapiq.approximator.permutation.PermutationSamplingSTII
   ~shapiq.approximator.regression.KernelSHAP
   ~shapiq.approximator.montecarlo.UnbiasedKernelSHAP
   ~shapiq.approximator.regression.kADDSHAP
   ~shapiq.approximator.regression.KernelSHAPIQ
   ~shapiq.approximator.regression.InconsistentKernelSHAPIQ
   ~shapiq.approximator.regression.RegressionFSII
   ~shapiq.approximator.regression.RegressionFBII
   ~shapiq.approximator.montecarlo.SHAPIQ
   ~shapiq.approximator.montecarlo.SVARM
   ~shapiq.approximator.montecarlo.SVARMIQ
   ~shapiq.approximator.marginals.OwenSamplingSV
   ~shapiq.approximator.marginals.StratifiedSamplingSV
   ~shapiq.approximator.sparse.SPEX
   ~shapiq.approximator.sparse.ProxySPEX
   ~shapiq.approximator.proxy.ProxySHAP
   ~shapiq.approximator.proxy.MSRBiased

Base Classes
~~~~~~~~~~~~

.. autoclass:: shapiq.approximator.Approximator
   :members:
   :no-private-members:
   :show-inheritance:

.. autoclass:: shapiq.approximator.regression.Regression
   :members:
   :no-private-members:
   :show-inheritance:

.. autoclass:: shapiq.approximator.montecarlo.MonteCarlo
   :members:
   :no-private-members:
   :show-inheritance:

.. autoclass:: shapiq.approximator.sparse.Sparse
   :members:
   :no-private-members:
   :show-inheritance:

Permutation-based
~~~~~~~~~~~~~~~~~

.. automodule:: shapiq.approximator.permutation
   :members:
   :no-private-members:
   :show-inheritance:

Regression-based
~~~~~~~~~~~~~~~~

.. automodule:: shapiq.approximator.regression
   :members:
   :no-private-members:
   :show-inheritance:
   :exclude-members: Regression

Monte Carlo
~~~~~~~~~~~

.. automodule:: shapiq.approximator.montecarlo
   :members:
   :no-private-members:
   :show-inheritance:
   :exclude-members: MonteCarlo

Marginal Sampling
~~~~~~~~~~~~~~~~~

.. automodule:: shapiq.approximator.marginals
   :members:
   :no-private-members:
   :show-inheritance:

Sparse
~~~~~~

.. automodule:: shapiq.approximator.sparse
   :members:
   :no-private-members:
   :show-inheritance:
   :exclude-members: Sparse

Proxy
~~~~~

.. automodule:: shapiq.approximator.proxy
   :members:
   :no-private-members:
   :show-inheritance:

Approximator Groups
~~~~~~~~~~~~~~~~~~~

.. autodata:: shapiq.approximator.SV_APPROXIMATORS
.. autodata:: shapiq.approximator.SI_APPROXIMATORS
.. autodata:: shapiq.approximator.SII_APPROXIMATORS
.. autodata:: shapiq.approximator.STII_APPROXIMATORS
.. autodata:: shapiq.approximator.FSII_APPROXIMATORS
.. autodata:: shapiq.approximator.FBII_APPROXIMATORS
