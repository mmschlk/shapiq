"""This module contains all pre-defined benchmark games."""

# data and dataset valuation games
from .data_valuation.base import DataValuation
from .data_valuation.benchmark import (
    AdultCensus as AdultCensusDataValuation,
    BikeSharing as BikeSharingDataValuation,
    CaliforniaHousing as CaliforniaHousingDataValuation,
)
from .dataset_valuation.base import DatasetValuation
from .dataset_valuation.benchmark import (
    AdultCensus as AdultCensusDatasetValuation,
    BikeSharing as BikeSharingDatasetValuation,
    CaliforniaHousing as CaliforniaHousingDatasetValuation,
)

# ensemble selection and random forest ensemble selection games
from .ensemble_selection.base import EnsembleSelection, RandomForestEnsembleSelection
from .ensemble_selection.benchmark import (
    AdultCensus as AdultCensusEnsembleSelection,
    BikeSharing as BikeSharingEnsembleSelection,
    CaliforniaHousing as CaliforniaHousingEnsembleSelection,
)
from .ensemble_selection.benchmark_random_forest import (
    AdultCensus as AdultCensusRandomForestEnsembleSelection,
    BikeSharing as BikeSharingRandomForestEnsembleSelection,
    CaliforniaHousing as CaliforniaHousingRandomForestEnsembleSelection,
)

# feature selection games
from .feature_selection.base import FeatureSelection
from .feature_selection.benchmark import (
    AdultCensus as AdultCensusFeatureSelection,
    BikeSharing as BikeSharingFeatureSelection,
    CaliforniaHousing as CaliforniaHousingFeatureSelection,
)

# global explanation games
from .global_xai.base import GlobalExplanation
from .global_xai.benchmark_tabular import (
    AdultCensus as AdultCensusGlobalXAI,
    BikeSharing as BikeSharingGlobalXAI,
    CaliforniaHousing as CaliforniaHousingGlobalXAI,
)

# local explanation games
from .local_xai.base import LocalExplanation
from .local_xai.benchmark_image import ImageClassifier as ImageClassifierLocalXAI
from .local_xai.benchmark_language import SentimentAnalysis as SentimentAnalysisLocalXAI
from .local_xai.benchmark_tabular import (
    AdultCensus as AdultCensusLocalXAI,
    BikeSharing as BikeSharingLocalXAI,
    CaliforniaHousing as CaliforniaHousingLocalXAI,
)

# synthetic games
from .synthetic.dummy import DummyGame
from .synthetic.random_game import RandomGame
from .synthetic.soum import SOUM, UnanimityGame

# treeshap-iq explanation games
from .treeshapiq_xai.base import TreeSHAPIQXAI
from .treeshapiq_xai.benchmark import (
    AdultCensus as AdultCensusTreeSHAPIQXAI,
    BikeSharing as BikeSharingTreeSHAPIQXAI,
    CaliforniaHousing as CaliforniaHousingTreeSHAPIQXAI,
    SynthData as SynthDataTreeSHAPIQXAI,
)

# uncertainty explanation games
from .uncertainty.base import UncertaintyExplanation
from .uncertainty.benchmark import AdultCensus as AdultCensusUncertaintyExplanation

# cluster explanation games
from .unsupervised_cluster.base import ClusterExplanation
from .unsupervised_cluster.benchmark import (
    AdultCensus as AdultCensusClusterExplanation,
    BikeSharing as BikeSharingClusterExplanation,
    CaliforniaHousing as CaliforniaHousingClusterExplanation,
)

# unsupervised data games
from .unsupervised_data.base import UnsupervisedData
from .unsupervised_data.benchmark import (
    AdultCensus as AdultCensusUnsupervisedData,
    BikeSharing as BikeSharingUnsupervisedData,
    CaliforniaHousing as CaliforniaHousingUnsupervisedData,
)

__all__ = [
    # local_xai games
    "LocalExplanation",
    "AdultCensusLocalXAI",
    "BikeSharingLocalXAI",
    "CaliforniaHousingLocalXAI",
    "ImageClassifierLocalXAI",
    "SentimentAnalysisLocalXAI",
    # dataset_valuation games
    "DatasetValuation",
    "AdultCensusDatasetValuation",
    "BikeSharingDatasetValuation",
    "CaliforniaHousingDatasetValuation",
    # data_valuation games
    "DataValuation",
    "AdultCensusDataValuation",
    "BikeSharingDataValuation",
    "CaliforniaHousingDataValuation",
    # feature_selection games
    "FeatureSelection",
    "AdultCensusFeatureSelection",
    "BikeSharingFeatureSelection",
    "CaliforniaHousingFeatureSelection",
    # global_xai games
    "GlobalExplanation",
    "AdultCensusGlobalXAI",
    "BikeSharingGlobalXAI",
    "CaliforniaHousingGlobalXAI",
    # cluster explanation games
    "ClusterExplanation",
    "AdultCensusClusterExplanation",
    "BikeSharingClusterExplanation",
    "CaliforniaHousingClusterExplanation",
    # ensemble selection games
    "EnsembleSelection",
    "AdultCensusEnsembleSelection",
    "BikeSharingEnsembleSelection",
    "CaliforniaHousingEnsembleSelection",
    # RandomForestEnsembleSelection
    "RandomForestEnsembleSelection",
    "AdultCensusRandomForestEnsembleSelection",
    "BikeSharingRandomForestEnsembleSelection",
    "CaliforniaHousingRandomForestEnsembleSelection",
    # uncertainty explanation games
    "UncertaintyExplanation",
    "AdultCensusUncertaintyExplanation",
    # unspervised data
    "UnsupervisedData",
    "AdultCensusUnsupervisedData",
    "BikeSharingUnsupervisedData",
    "CaliforniaHousingUnsupervisedData",
    # treeshapiq_xai games
    "TreeSHAPIQXAI",
    "AdultCensusTreeSHAPIQXAI",
    "BikeSharingTreeSHAPIQXAI",
    "CaliforniaHousingTreeSHAPIQXAI",
    "SynthDataTreeSHAPIQXAI",
    # synthetic games
    "DummyGame",
    "SOUM",
    "UnanimityGame",
    "RandomGame",
]
