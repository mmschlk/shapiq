================
shapiq_benchmark
================

.. grid:: 1 2 3 3
   :gutter: 2
   :class-container: shapiq-cards shapiq-cards-compact

   .. grid-item-card:: :octicon:`trophy` Benchmarks
      :link: benchmark-benchmarks
      :link-type: ref
      :class-card: shapiq-card

      A game and its exact values; run and score approximators

   .. grid-item-card:: :octicon:`check-circle` Computers
      :link: benchmark-computers
      :link-type: ref
      :class-card: shapiq-card

      The exact values of a game

   .. grid-item-card:: :octicon:`graph` Metrics
      :link: benchmark-metrics
      :link-type: ref
      :class-card: shapiq-card

      Error, ranking, and faithfulness

   .. grid-item-card:: :octicon:`gear` Setups
      :link: benchmark-setups
      :link-type: ref
      :class-card: shapiq-card

      Games built from datasets, models, and seeds

   .. grid-item-card:: :octicon:`database` Datasets
      :link: benchmark-datasets
      :link-type: ref
      :class-card: shapiq-card

      Downloaded on first use, cached locally

   .. grid-item-card:: :octicon:`cpu` Models
      :link: benchmark-models
      :link-type: ref
      :class-card: shapiq-card

      A registry of seeded, tuned models

.. automodule:: shapiq_benchmark
   :no-members:

.. currentmodule:: shapiq_benchmark

.. _benchmark-benchmarks:

Benchmarks
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Benchmark
   run
   save_results
   runner.build_approximator

.. _benchmark-computers:

Computers
---------

.. automodule:: shapiq_benchmark.computers
   :no-members:

.. currentmodule:: shapiq_benchmark.computers

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Computer
   BruteForceComputer
   MoebiusComputer
   PathDependentTreeComputer
   InterventionalTreeComputer
   NearestNeighborComputer
   ProductKernelComputer
   default_computer
   UnsupportedComputationError

.. _benchmark-metrics:

Metrics
-------

.. automodule:: shapiq_benchmark.metrics
   :no-members:

.. currentmodule:: shapiq_benchmark.metrics

.. autosummary::
   :toctree: generated/
   :nosignatures:

   compare
   error_metrics
   ranking_metrics
   faithfulness
   faithfulness_sample
   RANK_TOLERANCE

.. _benchmark-setups:

Setups
------

.. automodule:: shapiq_benchmark.setups
   :no-members:

.. currentmodule:: shapiq_benchmark.setups

Setups of the Model-Specific Games
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   PathDependentTreeSetup
   InterventionalTreeSetup
   KNNSetup
   WeightedKNNSetup
   ThresholdNNSetup
   ProductKernelSetup

Setups of the Machine Learning Games
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   TabularLocalExplanationSetup
   TabularGlobalExplanationSetup
   TabularUncertaintyExplanationSetup
   FeatureSelectionSetup
   DataValuationSetup
   DatasetValuationSetup
   EnsembleSelectionSetup
   RandomForestEnsembleSelectionSetup
   ClusterExplanationSetup
   UnsupervisedDataSetup
   GlobalConfoundingSetup
   LocalConfoundingSetup

Setups of the Image and Language Games
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ImageClassifierSetup
   ImageTextSimilaritySetup
   SentimentAnalysisSetup

Base Classes and Helpers
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Setup
   DatasetSetup
   TabularSetup
   ModelSetup
   setup_from_dict
   runtime_field
   SETUPS
   RECIPE_VERSION
   DEFAULT_MEMBER_POOL

.. _benchmark-datasets:

Datasets
--------

.. automodule:: shapiq_benchmark.datasets
   :no-members:

.. currentmodule:: shapiq_benchmark.datasets

.. autosummary::
   :toctree: generated/
   :nosignatures:

   load_dataset
   list_datasets
   get_dataset_spec
   register_dataset
   get_data_dir
   Dataset
   DatasetSplit
   DatasetSpec
   DatasetKind
   TABARENA_DATASETS
   load_imagenette
   ImageDataset
   IMAGENETTE_CLASSES
   ImagenetteSplit
   ImagenetteSize
   load_curthvds_synthetic
   CausalSetting

.. _benchmark-models:

Models
------

.. automodule:: shapiq_benchmark.models
   :no-members:

.. currentmodule:: shapiq_benchmark.models

.. autosummary::
   :toctree: generated/
   :nosignatures:

   build_model
   fit_model
   tuned_params
   ModelName
   TreeModelName
   Preset
   MODEL_NAMES
   TUNED_PRESETS
   MISSING_VALUE_MODELS
