"""
Configuration Constants and Whitelists.

This module contains all hardcoded whitelists and type definitions for the benchmark configuration system.
"""

from typing import Literal

# --- Index Type Definition ---
VALID_INDICES = Literal[
    "SV", "BV", "SII", "BII", "k-SII", "STII", "FBII", "FSII", "kADD-SHAP", "CHII"
]

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
