"""Configuration Constants and Whitelists.

This module contains all hardcoded whitelists and type definitions for the benchmark configuration system.
"""

from __future__ import annotations

from typing import Literal

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

# --- Supported Imputers Whitelist ---
SUPPORTED_IMPUTERS = ["marginal", "conditional", "baseline"]

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
}

# global_xai games (currently implemented global explanations)
GLOBAL_GAMES = {"AdultCensus", "BikeSharing", "CaliforniaHousing"}

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
