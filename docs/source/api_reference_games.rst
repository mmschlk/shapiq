============
shapiq_games
============

.. grid:: 2 2 4 4
   :gutter: 2
   :class-container: shapiq-cards shapiq-cards-compact

   .. grid-item-card:: :octicon:`beaker` Synthetic
      :link: games-synthetic
      :link-type: ref
      :class-card: shapiq-card

      Values known in closed form

   .. grid-item-card:: :octicon:`workflow` Model-Specific
      :link: games-model-specific
      :link-type: ref
      :class-card: shapiq-card

      Trees, neighbors, kernels

   .. grid-item-card:: :octicon:`search` Explanation
      :link: games-explanation
      :link-type: ref
      :class-card: shapiq-card

      Local, global, uncertainty

   .. grid-item-card:: :octicon:`law` Valuation and Selection
      :link: games-valuation
      :link-type: ref
      :class-card: shapiq-card

      Data, features, ensembles

   .. grid-item-card:: :octicon:`dot-fill` Unsupervised
      :link: games-unsupervised
      :link-type: ref
      :class-card: shapiq-card

      Clusters and data

   .. grid-item-card:: :octicon:`git-branch` Causal
      :link: games-causal
      :link-type: ref
      :class-card: shapiq-card

      Confounding

   .. grid-item-card:: :octicon:`image` Image and Language
      :link: games-image-language
      :link-type: ref
      :class-card: shapiq-card

      Vision and text models

   .. grid-item-card:: :octicon:`tools` Utilities
      :link: games-utilities
      :link-type: ref
      :class-card: shapiq-card

      Helpers, constants, and types

.. automodule:: shapiq_games
   :no-members:

.. currentmodule:: shapiq_games

.. _games-synthetic:

Synthetic Games
---------------

Games whose interaction values are known in closed form.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   DummyGame
   UnanimityGame
   SOUM
   RandomTableGame

.. _games-model-specific:

Model-Specific Games
--------------------

Games of models with an exact explainer in ``shapiq``, so their values are known for many
players.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   PathDependentTreeGame
   InterventionalTreeGame
   KNNGame
   WeightedKNNGame
   ThresholdNNGame
   ProductKernelGame

.. _games-explanation:

Explanation Games
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   TabularLocalExplanation
   TabularGlobalExplanation
   TabularUncertaintyExplanation

.. _games-valuation:

Valuation and Selection Games
-----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   FeatureSelection
   DataValuation
   DatasetValuation
   EnsembleSelection
   RandomForestEnsembleSelection

.. _games-unsupervised:

Unsupervised Games
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ClusterExplanation
   UnsupervisedData

.. _games-causal:

Causal Games
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GlobalConfoundingXAI
   LocalConfoundingXAI

.. _games-image-language:

Image and Language Games
------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ImageClassifier
   ImageTextSimilarity
   SentimentAnalysis

.. _games-utilities:

Utilities
---------

The helpers and constants the games use, and the types of their arguments.

Helpers
~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   causal.tabpfn_regressor
   language.SENTIMENT_MODEL_ID
   unsupervised.total_correlation
   vision.get_superpixels
   vision.grid_regions

.. _games-types:

Types
~~~~~

.. automodule:: shapiq_games.typing
   :no-members:

.. currentmodule:: shapiq_games.typing

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Task
   PredictFunction
   Metric
   MetricName
   Loss
   LossName
   ImputerName
   Uncertainty
   PlayerSizes
   ClusterMethod
   ClusterScore
   ConfoundingMode
   MaskStrategy
   ImageModel
   ImageTextModel
   Fill
