"""Benchmark games."""

# data and dataset valuation games
from .data_valuation.base import DataValuation
from .data_valuation.benchmark import AdultCensus as AdultCensusDataValuation
from .data_valuation.benchmark import BikeSharing as BikeSharingDataValuation
from .data_valuation.benchmark import CaliforniaHousing as CaliforniaHousingDataValuation
from .dataset_valuation.base import DatasetValuation
from .dataset_valuation.benchmark import AdultCensus as AdultCensusDatasetValuation
from .dataset_valuation.benchmark import BikeSharing as BikeSharingDatasetValuation
from .dataset_valuation.benchmark import CaliforniaHousing as CaliforniaHousingDatasetValuation

# ensemble selection and random forest ensemble selection games
from .ensemble_selection.base import EnsembleSelection, RandomForestEnsembleSelection
from .ensemble_selection.benchmark import AdultCensus as AdultCensusEnsembleSelection
from .ensemble_selection.benchmark import BikeSharing as BikeSharingEnsembleSelection
from .ensemble_selection.benchmark import CaliforniaHousing as CaliforniaHousingEnsembleSelection
from .ensemble_selection.benchmark_random_forest import (
    AdultCensus as AdultCensusRandomForestEnsembleSelection,
)
from .ensemble_selection.benchmark_random_forest import (
    BikeSharing as BikeSharingRandomForestEnsembleSelection,
)
from .ensemble_selection.benchmark_random_forest import (
    CaliforniaHousing as CaliforniaHousingRandomForestEnsembleSelection,
)

# feature selection games
from .feature_selection.base import FeatureSelection
from .feature_selection.benchmark import AdultCensus as AdultCensusFeatureSelection
from .feature_selection.benchmark import BikeSharing as BikeSharingFeatureSelection
from .feature_selection.benchmark import CaliforniaHousing as CaliforniaHousingFeatureSelection

# global explanation games
from .global_xai.base import GlobalExplanation
from .global_xai.benchmark_tabular import AdultCensus as AdultCensusGlobalXAI
from .global_xai.benchmark_tabular import BikeSharing as BikeSharingGlobalXAI
from .global_xai.benchmark_tabular import CaliforniaHousing as CaliforniaHousingGlobalXAI

# local explanation games
from .local_xai.base import LocalExplanation
from .local_xai.benchmark_image import ImageClassifier as ImageClassifierLocalXAI
from .local_xai.benchmark_language import SentimentAnalysis as SentimentAnalysisLocalXAI
from .local_xai.benchmark_tabular import AdultCensus as AdultCensusLocalXAI
from .local_xai.benchmark_tabular import BikeSharing as BikeSharingLocalXAI
from .local_xai.benchmark_tabular import CaliforniaHousing as CaliforniaHousingLocalXAI

# metrics
from .metrics import (
    get_all_metrics,
)

# precompute util functions
from .precompute import (
    SHAPIQ_DATA_DIR,
    get_game_files,
    pre_compute_and_store,
    pre_compute_and_store_from_list,
)

# util functions
from .run import run_benchmark, save_results

# synthetic games
from .synthetic.dummy import DummyGame
from .synthetic.soum import SOUM, UnanimityGame

# treeshap-iq explanation games
from .treeshapiq_xai.base import TreeSHAPIQXAI
from .treeshapiq_xai.benchmark import AdultCensus as AdultCensusTreeSHAPIQXAI
from .treeshapiq_xai.benchmark import BikeSharing as BikeSharingTreeSHAPIQXAI
from .treeshapiq_xai.benchmark import CaliforniaHousing as CaliforniaHousingTreeSHAPIQXAI
from .treeshapiq_xai.benchmark import SynthData as SynthDataTreeSHAPIQXAI

# uncertainty explanation games
from .uncertainty.base import UncertaintyExplanation
from .uncertainty.benchmark import AdultCensus as AdultCensusUncertaintyExplanation

# cluster explanation games
from .unsupervised_cluster.base import ClusterExplanation
from .unsupervised_cluster.benchmark import AdultCensus as AdultCensusClusterExplanation
from .unsupervised_cluster.benchmark import BikeSharing as BikeSharingClusterExplanation
from .unsupervised_cluster.benchmark import CaliforniaHousing as CaliforniaHousingClusterExplanation

# unsupervised data games
from .unsupervised_data.base import UnsupervisedData
from .unsupervised_data.benchmark import AdultCensus as AdultCensusUnsupervisedData
from .unsupervised_data.benchmark import BikeSharing as BikeSharingUnsupervisedData
from .unsupervised_data.benchmark import CaliforniaHousing as CaliforniaHousingUnsupervisedData

__all__ = [
    # util functions
    "run_benchmark",
    "save_results",
    "pre_compute_and_store",
    "pre_compute_and_store_from_list",
    "SHAPIQ_DATA_DIR",
    "get_game_files",
    # all metrics
    "get_all_metrics",
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
]
