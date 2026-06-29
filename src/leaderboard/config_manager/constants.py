"""Configuration Constants and Whitelists.

This module contains all hardcoded whitelists and type definitions for the benchmark configuration system.
"""

from __future__ import annotations

from typing import Literal

import shapiq_games.benchmark.global_xai.benchmark_tabular as global_tabular
import shapiq_games.benchmark.local_xai.benchmark_tabular as local_tabular
from shapiq_games.benchmark.local_xai import benchmark_image

# --- Index Type Definition ---
VALID_INDICES = Literal[
    "SV", "BV", "SII", "BII", "k-SII", "STII", "FBII", "FSII", "kADD-SHAP", "CHII"
]

# --- 🎯 SINGLE SOURCE OF TRUTH: Registries First ---

# local_xai games registry (tabular datasets + visual games)
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

# global_xai games registry (currently implemented global explanations)
GLOBAL_GAME_REGISTRY = {
    "BikeSharing": global_tabular.BikeSharing,
    "CaliforniaHousing": global_tabular.CaliforniaHousing,
    "AdultCensus": global_tabular.AdultCensus,
}

# --- 🧠 DYNAMIC DERIVATION: Automatically Synced Collections ---

# Dynamically derive family specific game sets directly from registries!
LOCAL_GAMES = set(LOCAL_GAME_REGISTRY.keys())
GLOBAL_GAMES = set(GLOBAL_GAME_REGISTRY.keys())

# Dynamically combine registries to build the supported games whitelist (+ synthetic SOUM)
SUPPORTED_GAMES = ["SOUM", *list(LOCAL_GAMES | GLOBAL_GAMES)]

# visual / image-local games
VISUAL_GAMES = {"ImageClassifier"}

# Supported model backends for visual games.
SUPPORTED_VISUAL_MODELS = [
    "vit_16_patches",
    "vit_9_patches",
    "resnet_18",
]

# Task classification mapping
REGRESSION_GAMES = {
    "BikeSharing",
    "CaliforniaHousing",
}

# Classification games are derived dynamically by filtering out regressions from the supported pool
CLASSIFICATION_GAMES = {
    game for game in SUPPORTED_GAMES if game not in REGRESSION_GAMES and game != "SOUM"
}

# --- Hardcoded Dataset Properties (Bypasses instantiation requirement) ---
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

# --- Supported Approximators Whitelist ---
ALL_SUPPORTED_APPROXIMATORS = [
    "OwenSamplingSV",
    "StratifiedSamplingSV",
    "PermutationSamplingSV",
    "PermutationSamplingSII",
    "PermutationSamplingSTII",
    "SVARM",
    "SVARMIQ",
    "SHAPIQ",
    "UnbiasedKernelSHAP",
    "KernelSHAP",
    "KernelSHAPIQ",
    "InconsistentKernelSHAPIQ",
    "RegressionFSII",
    "RegressionFBII",
    "kADDSHAP",
    "SPEX",
    "ProxySPEX",
    "ProxySHAP",
    "MSRBiased",
]
