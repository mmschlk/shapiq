"""This module contains all benchmark games."""

# dataset valuation games
from .dataset_valuation.base import DatasetValuation
from .dataset_valuation.benchmark import AdultCensus as AdultCensusDatasetValuation
from .dataset_valuation.benchmark import BikeSharing as BikeSharingDatasetValuation
from .dataset_valuation.benchmark import CaliforniaHousing as CaliforniaHousingDatasetValuation

# feature selection games
from .feature_selection.base import FeatureSelectionGame
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

# synthetic games
from .synthetic.dummy import DummyGame
from .synthetic.soum import SOUM, UnanimityGame

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
    # feature_selection games
    "FeatureSelectionGame",
    "AdultCensusFeatureSelection",
    "BikeSharingFeatureSelection",
    "CaliforniaHousingFeatureSelection",
    # global_xai games
    "GlobalExplanation",
    "AdultCensusGlobalXAI",
    "BikeSharingGlobalXAI",
    "CaliforniaHousingGlobalXAI",
    # synthetic games
    "DummyGame",
    "SOUM",
    "UnanimityGame",
]
