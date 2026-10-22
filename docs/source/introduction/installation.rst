Installation
============

``shapiq`` needs Python 3.12 or newer. Install the latest release from
`PyPI <https://pypi.org/project/shapiq>`_:

.. tab-set::
   :sync-group: installer

   .. tab-item:: pip
      :sync: pip

      .. code-block:: bash

         pip install shapiq

   .. tab-item:: uv
      :sync: uv

      .. code-block:: bash

         uv add shapiq

   .. tab-item:: Development version
      :sync: git

      The latest version from `GitHub <https://github.com/mmschlk/shapiq>`_:

      .. code-block:: bash

         pip install git+https://github.com/mmschlk/shapiq

      This compiles the C++ extensions of ``shapiq`` and needs a C++17 compiler; the releases
      on PyPI come with compiled wheels.

The core installation covers the explainers, the exact algorithms, most approximators, and the
plots. The games of :mod:`shapiq_games` and the benchmark :mod:`shapiq_benchmark` ship with
``shapiq`` too.


Optional Extras
---------------

Some features need packages that the core installation leaves out. Install them with an extra,
or several at once:

.. tab-set::
   :sync-group: installer

   .. tab-item:: pip
      :sync: pip

      .. code-block:: bash

         pip install "shapiq[tree,proxy]"

   .. tab-item:: uv
      :sync: uv

      .. code-block:: bash

         uv add "shapiq[tree,proxy]"

   .. tab-item:: Development version
      :sync: git

      .. code-block:: bash

         pip install "shapiq[tree,proxy] @ git+https://github.com/mmschlk/shapiq"

.. list-table::
   :header-rows: 1
   :widths: 20 80
   :class: shapiq-table

   * - Extra
     - Adds
   * - ``tree``
     - The Woodelf backend of :class:`~shapiq.explainer.TreeExplainer`, which explains large
       interventional inputs much faster (``woodelf-explainer``).
   * - ``sparse``
     - The :class:`~shapiq.approximator.SPEX` approximator (``sparse-transform``, ``galois``).
   * - ``proxy``
     - :class:`~shapiq.approximator.ProxySHAP`, :class:`~shapiq.approximator.ProxySPEX`,
       :class:`~shapiq.approximator.RegressionMSR`, and :class:`~shapiq.approximator.OddSHAP`
       with their proxy models (XGBoost, LightGBM, CatBoost, SMAC).
   * - ``shapleig``
     - The :class:`~shapiq.approximator.ShaplEIG` approximator (PyTorch, GPyTorch, BoTorch).
   * - ``games``
     - The games of :mod:`shapiq_games` that need deep learning or foundation models: image,
       language, and causal games (PyTorch, transformers, TabPFN, scikit-image, XGBoost).
   * - ``benchmark``
     - The datasets and models of :mod:`shapiq_benchmark` (OpenML, UCI, XGBoost, LightGBM,
       CatBoost, Optuna); includes ``games``.

.. tip::

   When a package of an extra is missing, the feature that needs it tells you which extra to
   install.


Development Setup
-----------------

To work on ``shapiq`` itself, clone the repository and create the environment with
`uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   git clone https://github.com/mmschlk/shapiq.git
   cd shapiq
   uv sync --all-extras --group docs

``uv sync`` installs ``shapiq`` in editable mode with its development tools (tests, linters);
``--all-extras`` adds every extra and ``--group docs`` the tools that build this documentation.
The development and documentation tools are dependency groups of ``pyproject.toml``, not extras.

The `contributing guide <https://github.com/mmschlk/shapiq/blob/main/.github/CONTRIBUTING.md>`_
explains the pre-commit hooks, the tests, and how to build the documentation.
