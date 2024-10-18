The ``shapiq`` Python package
================================

Shapley Interaction Quantification (``shapiq``) is a Python package for (1) approximating any-order Shapley interactions, (2) benchmarking game-theoretical algorithms for machine learning, (3) explaining feature interactions of model predictions. ``shapiq`` extends the well-known `shap <https://github.com/shap/shap>`_ package for both researchers working on game theory in machine learning, as well as the end-users explaining models. SHAP-IQ extends individual Shapley values by quantifying the **synergy** effect between entities (aka **players** in the jargon of game theory) like explanatory features, data points, or weak learners in ensemble models. Synergies between players give a more comprehensive view of machine learning models.

If you use the ``shapiq`` package, please cite our `NeurIPS paper <https://arxiv.org/abs/2410.01649>`_:

.. code::

   @inproceedings{muschalik2024shapiq,
      title     = {shapiq: Shapley Interactions for Machine Learning},
      author    = {Maximilian Muschalik and Hubert Baniecki and Fabian Fumagalli and
                   Patrick Kolpaczki and Barbara Hammer and Eyke H\"{u}llermeier},
      booktitle = {NeurIPS},
      year      = {2024}
   }


Contents
~~~~~~~~

.. toctree::
   :maxdepth: 1
   :caption: OVERVIEW

   installation
   start

.. toctree::
   :maxdepth: 1
   :caption: TUTORIALS

   notebooks/shapiq_scikit_learn
   notebooks/treeshapiq_lightgbm
   notebooks/language_model_game
   notebooks/vision_transformer
   notebooks/conditional_imputer
   notebooks/parallel_computation
   notebooks/benchmark_approximators
   notebooks/core

.. toctree::
   :maxdepth: 2
   :caption: API REFERENCE

   api

.. toctree::
   :maxdepth: 1
   :caption: BIBLIOGRAPHY

   references
