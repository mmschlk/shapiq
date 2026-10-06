"""A curated collection of cooperative games for shapiq.

Every game is a :class:`shapiq.Game` that follows the same contract (see
:mod:`shapiq_games._base`): its values are deterministic given its arguments, the explained point
and class are explicit, and games configured from strings with ``from_config`` carry a stable
``fingerprint``. Datasets are downloaded on first use and cached locally
(:mod:`shapiq_games.datasets`); no data ships with the package.

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
  :class:`SentimentAnalysis`
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
from .vision import ImageClassifier

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
    "LocalConfoundingXAI",
    "LocalExplanation",
    "RandomForestEnsembleSelection",
    "SentimentAnalysis",
    "UncertaintyExplanation",
    "UnsupervisedData",
]
