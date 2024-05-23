API Reference
=============

This page contains the API reference for public objects and functions in ``shapiq``.


.. autosummary::
    :toctree: api/
    :caption: Development
    :recursive:

    shapiq.explainer
    shapiq.approximator
    shapiq.plot
    shapiq.utils

Approximators
-------------
.. autosummary::
    :nosignatures:

    shapiq.approximator.ShapIQ
    shapiq.approximator.PermutationSamplingSII
    shapiq.approximator.PermutationSamplingSTII
    shapiq.approximator.RegressionFSII

Plotting
--------
.. autosummary::
    :nosignatures:

    shapiq.plot.network

Utils
-----

.. autosummary::
    :nosignatures:

    shapiq.utils.powerset
    shapiq.utils.get_explicit_subsets
    shapiq.utils.split_subsets_budget
    shapiq.utils.get_conditional_sample_weights
    shapiq.utils.get_parent_array
