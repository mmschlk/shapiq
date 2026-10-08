"""A collection of cooperative games for shapiq.

Every game is a :class:`shapiq.Game` defined by its value function. Build one from your own
objects and evaluate coalitions:

>>> import numpy as np
>>> from sklearn.datasets import make_regression
>>> from sklearn.ensemble import RandomForestRegressor
>>> import shapiq_games as sg
>>> X, y = make_regression(n_samples=200, n_features=5, random_state=0)
>>> model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
>>> game = sg.LocalExplanation(model, data=X[:50], x=X[0])
>>> values = game(np.array([[1, 0, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=bool))

All games follow the same contract (see :mod:`shapiq_games._base`): their values are
deterministic given their arguments, and the explained point and class are explicit.

The games are definitions: they know how to model a problem as a cooperative game, not where
data and models come from. Benchmarks build them from names (datasets, models, seeds) with the
setups of :mod:`shapiq_benchmark.setups`, e.g.
``LocalExplanationSetup(dataset="breast_cancer", model="xgboost").build()``.

Game families:

- synthetic games with analytic values: :class:`DummyGame`, :class:`UnanimityGame`,
  :class:`SOUM`, :class:`RandomTableGame`
- model-specific games with exact explainers: :class:`PathDependentTreeGame`,
  :class:`InterventionalTreeGame`, :class:`KNNGame`, :class:`WeightedKNNGame`,
  :class:`ThresholdNNGame`, :class:`ProductKernelGame`
- machine learning games: :class:`LocalExplanation`, :class:`GlobalExplanation`,
  :class:`FeatureSelection`, :class:`DataValuation`, :class:`DatasetValuation`,
  :class:`EnsembleSelection`, :class:`RandomForestEnsembleSelection`,
  :class:`UncertaintyExplanation`, :class:`ClusterExplanation`, :class:`UnsupervisedData`,
  :class:`GlobalConfoundingXAI`, :class:`LocalConfoundingXAI`, :class:`ImageClassifier`,
  :class:`ImageTextSimilarity`, :class:`SentimentAnalysis`
"""

from .causal import GlobalConfoundingXAI, LocalConfoundingXAI
from .clustering import ClusterExplanation
from .ensemble_selection import EnsembleSelection, RandomForestEnsembleSelection
from .feature_selection import FeatureSelection
from .global_xai import GlobalExplanation
from .kernel import ProductKernelGame
from .language import SentimentAnalysis
from .local_xai import LocalExplanation
from .nn import KNNGame, ThresholdNNGame, WeightedKNNGame
from .synthetic import SOUM, DummyGame, RandomTableGame, UnanimityGame
from .tree import InterventionalTreeGame, PathDependentTreeGame
from .uncertainty import UncertaintyExplanation
from .unsupervised import UnsupervisedData
from .valuation import DatasetValuation, DataValuation
from .vision import ImageClassifier, ImageTextSimilarity

__all__ = [
    # synthetic
    "DummyGame",
    "RandomTableGame",
    "SOUM",
    "UnanimityGame",
    # model-specific
    "InterventionalTreeGame",
    "KNNGame",
    "PathDependentTreeGame",
    "ProductKernelGame",
    "ThresholdNNGame",
    "WeightedKNNGame",
    # machine learning
    "ClusterExplanation",
    "DataValuation",
    "DatasetValuation",
    "EnsembleSelection",
    "FeatureSelection",
    "GlobalConfoundingXAI",
    "GlobalExplanation",
    "ImageClassifier",
    "ImageTextSimilarity",
    "LocalConfoundingXAI",
    "LocalExplanation",
    "RandomForestEnsembleSelection",
    "SentimentAnalysis",
    "UncertaintyExplanation",
    "UnsupervisedData",
]
