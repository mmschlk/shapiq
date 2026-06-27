"""Configuration Constants and Whitelists.

This module contains all hardcoded whitelists and type definitions for the benchmark configuration system.
"""

from __future__ import annotations

from typing import Literal

import shapiq_games.benchmark.global_xai.benchmark_tabular as global_tabular
import shapiq_games.benchmark.local_xai.benchmark_image as benchmark_image
import shapiq_games.benchmark.local_xai.benchmark_tabular as local_tabular

# --- Index Type Definition ---
VALID_INDICES = Literal[
    "SV", "BV", "SII", "BII", "k-SII", "STII", "FBII", "FSII", "kADD-SHAP", "CHII"
]

# --- Supported Games Whitelist ---
SUPPORTED_GAMES = [
    "SOUM",
    "BikeSharing",
    "CaliforniaHousing",
    "AdultCensus",
    "Mushroom",
    "Soybean",
    "Thyroid",
    "Annealing",
    "Arrhythmia",
    "BreastCancer",
    "Hepatitis",
    "Ionosphere",
    "Nursery",
    "Zoo",
]
GAME_PLAYER_COUNTS = {
    "CaliforniaHousing": 8,
    "Nursery": 8,
    "BikeSharing": 12,
    "AdultCensus": 14,
    "Zoo": 16,
    "Hepatitis": 19,
    "Thyroid": 21,
    "Mushroom": 22,
    "BreastCancer": 30,
    "Ionosphere": 34,
    "Soybean": 35,
    "Annealing": 38,
    "SOUM": 10,
    "Arrhythmia": 279,
}
# --- Supported Imputers Whitelist ---
SUPPORTED_IMPUTERS = ["marginal", "conditional"]

# --- Family specific game lists ---
# local_xai games (many tabular datasets)
LOCAL_GAMES = {
    "AdultCensus",
    "Annealing",
    "Arrhythmia",
    "BikeSharing",
    "BreastCancer",
    "CaliforniaHousing",
    "Hepatitis",
    "Ionosphere",
    "Mushroom",
    "Nursery",
    "Soybean",
    "Thyroid",
    "Zoo",
    "ImageClassifier",
}

# global_xai games (currently implemented global explanations)
GLOBAL_GAMES = {"AdultCensus", "BikeSharing", "CaliforniaHousing"}

# local_xai games (many tabular datasets)
LOCAL_GAME_REGISTRY = {
    "BikeSharing": local_tabular.BikeSharing,
    "CaliforniaHousing": local_tabular.CaliforniaHousing,
    "AdultCensus": local_tabular.AdultCensus,
    "Mushroom": local_tabular.Mushroom,
    "Soybean": local_tabular.Soybean,
    "Thyroid": local_tabular.Thyroid,
    "Annealing": local_tabular.Annealing,
    "Arrhythmia": local_tabular.Arrhythmia,
    "BreastCancer": local_tabular.BreastCancer,
    "Hepatitis": local_tabular.Hepatitis,
    "Ionosphere": local_tabular.Ionosphere,
    "Nursery": local_tabular.Nursery,
    "Zoo": local_tabular.Zoo,
    "ImageClassifier": benchmark_image.ImageClassifier,
}

# global_xai games (currently implemented global explanations)
GLOBAL_GAME_REGISTRY = {
    "BikeSharing": global_tabular.BikeSharing,
    "CaliforniaHousing": global_tabular.CaliforniaHousing,
    "AdultCensus": global_tabular.AdultCensus,
}
REGRESSION_GAMES = {
    "BikeSharing",
    "CaliforniaHousing",
}

CLASSIFICATION_GAMES = {
    "AdultCensus",
    "Mushroom",
    "Soybean",
    "Thyroid",
    "Annealing",
    "Arrhythmia",
    "BreastCancer",
    "Hepatitis",
    "Ionosphere",
    "Nursery",
    "Zoo",
    "ImageClassifier",
}
# --- Supported Approximators Whitelist ---
# Extracted from shapiq/approximator/__init__.py
ALL_SUPPORTED_APPROXIMATORS = [
    # Sampling-based SV methods
    "OwenSamplingSV",
    "StratifiedSamplingSV",
    "PermutationSamplingSV",
    # Sampling-based Interaction methods
    "PermutationSamplingSII",
    "PermutationSamplingSTII",
    # Monte Carlo / Stochastic methods
    "SVARM",
    "SVARMIQ",
    "SHAPIQ",
    "UnbiasedKernelSHAP",
    # Regression methods
    "KernelSHAP",
    "KernelSHAPIQ",
    "InconsistentKernelSHAPIQ",
    "RegressionFSII",
    "RegressionFBII",
    "kADDSHAP",
    # Proxy / Sparse methods
    "SPEX",
    "ProxySPEX",
    "ProxySHAP",
    "MSRBiased",
]
