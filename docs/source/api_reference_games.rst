============
shapiq_games
============

.. automodule:: shapiq_games
   :no-members:

.. currentmodule:: shapiq_games

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

Explanation Games
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   TabularLocalExplanation
   TabularGlobalExplanation
   TabularUncertaintyExplanation

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

Unsupervised Games
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ClusterExplanation
   UnsupervisedData
   unsupervised.total_correlation

Causal Games
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GlobalConfoundingXAI
   LocalConfoundingXAI
   causal.tabpfn_regressor

Image and Language Games
------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ImageClassifier
   ImageTextSimilarity
   SentimentAnalysis
   vision.grid_regions
   vision.get_superpixels
   language.SENTIMENT_MODEL_ID

Types
-----

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
