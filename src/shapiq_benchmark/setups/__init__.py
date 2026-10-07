"""Setups: typed, named recipes that build the games of :mod:`shapiq_games` for benchmarks.

The games of :mod:`shapiq_games` are built from objects: a model, data, a point, an image, a text.
A setup describes such a game with plain values instead (a dataset name, a model name, seeds, the
index of the explained point), so it can be stored, compared, and used as a cache key. Every game
except the synthetic ones has a setup; their fields differ with the kind of game.

:meth:`shapiq_benchmark.Benchmark.from_setup` builds the game of a setup and caches its exact
values under the setup's :attr:`~Setup.key`. A setup also converts to and from a dictionary, the
form in which run specifications are stored (:meth:`Setup.to_dict`, :func:`setup_from_dict`).

Examples:
    >>> from shapiq_benchmark import Benchmark
    >>> from shapiq_benchmark.setups import LocalExplanationSetup
    >>> setup = LocalExplanationSetup(dataset="xor", model="decision_tree", x=3)
    >>> game = setup.build()  # a plain shapiq_games.LocalExplanation
    >>> benchmark = Benchmark.from_setup(setup)  # its exact values are cached under setup.key
    >>> setup.to_dict()["setup"]
    'local_explanation'

Setups by kind of game:

- tree, nearest-neighbor, and kernel games: :class:`PathDependentTreeSetup`,
  :class:`InterventionalTreeSetup`, :class:`KNNSetup`, :class:`WeightedKNNSetup`,
  :class:`ThresholdNNSetup`, :class:`ProductKernelSetup`
- machine learning games on tabular data: :class:`LocalExplanationSetup`,
  :class:`GlobalExplanationSetup`, :class:`FeatureSelectionSetup`, :class:`DataValuationSetup`,
  :class:`DatasetValuationSetup`, :class:`EnsembleSelectionSetup`,
  :class:`RandomForestEnsembleSelectionSetup`, :class:`UncertaintyExplanationSetup`,
  :class:`ClusterExplanationSetup`, :class:`UnsupervisedDataSetup`
- causal games: :class:`GlobalConfoundingSetup`, :class:`LocalConfoundingSetup`
- image and text games: :class:`ImageClassifierSetup`, :class:`SentimentAnalysisSetup`
"""

from ._base import SETUPS, ModelSetup, Setup, TabularSetup, runtime_field, setup_from_dict
from ._causal import GlobalConfoundingSetup, LocalConfoundingSetup
from ._language import SentimentAnalysisSetup
from ._ml_games import (
    DEFAULT_MEMBER_POOL,
    ClusterExplanationSetup,
    DatasetValuationSetup,
    DataValuationSetup,
    EnsembleSelectionSetup,
    FeatureSelectionSetup,
    GlobalExplanationSetup,
    LocalExplanationSetup,
    RandomForestEnsembleSelectionSetup,
    UncertaintyExplanationSetup,
    UnsupervisedDataSetup,
)
from ._model_games import (
    InterventionalTreeSetup,
    KNNSetup,
    PathDependentTreeSetup,
    ProductKernelSetup,
    ThresholdNNSetup,
    WeightedKNNSetup,
)
from ._vision import ImageClassifierSetup

__all__ = [
    "DEFAULT_MEMBER_POOL",
    "SETUPS",
    "ClusterExplanationSetup",
    "DataValuationSetup",
    "DatasetValuationSetup",
    "EnsembleSelectionSetup",
    "FeatureSelectionSetup",
    "GlobalConfoundingSetup",
    "GlobalExplanationSetup",
    "ImageClassifierSetup",
    "InterventionalTreeSetup",
    "KNNSetup",
    "LocalConfoundingSetup",
    "LocalExplanationSetup",
    "ModelSetup",
    "PathDependentTreeSetup",
    "ProductKernelSetup",
    "RandomForestEnsembleSelectionSetup",
    "SentimentAnalysisSetup",
    "Setup",
    "TabularSetup",
    "ThresholdNNSetup",
    "UncertaintyExplanationSetup",
    "UnsupervisedDataSetup",
    "WeightedKNNSetup",
    "runtime_field",
    "setup_from_dict",
]
