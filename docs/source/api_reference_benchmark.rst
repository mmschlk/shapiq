================
shapiq_benchmark
================

.. automodule:: shapiq_benchmark
   :no-members:

.. currentmodule:: shapiq_benchmark

Benchmarks
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Benchmark
   run
   save_results
   runner.build_approximator

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
