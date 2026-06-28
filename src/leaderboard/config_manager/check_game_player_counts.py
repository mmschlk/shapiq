"""Configuration Data Models and Validators.

This module contains all Pydantic models and their associated validators for the benchmark configuration system.
"""

from __future__ import annotations

import logging

# Import the dataset loader functions from your shapiq_games package
import shapiq_games.datasets._all as loaders

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# game located in src/shapiq_games/datasets/data
# =======================================================
# Game Name                 | Actual Player Count (n)
# =======================================================
# CaliforniaHousing         | 8
# Nursery                   | 8
# BikeSharing               | 12
# AdultCensus               | 14
# Zoo                       | 16
# Hepatitis                 | 19
# Thyroid                   | 21
# Mushroom                  | 22
# BreastCancer              | 30
# Ionosphere                | 34
# Soybean                   | 35

# Annealing                 | 38
# ERROR - Failed to check player count for Annealing: Cannot perform reduction 'median' with string dtype

# Arrhythmia                | 279
# =======================================================


def main() -> None:
    """Check exact player (feature) counts for all tabular benchmark games."""
    # Dictionary mapping game names to their respective loader function names
    game_loaders = {
        "CaliforniaHousing": "load_california_housing",
        "Nursery": "load_nursery",
        "BikeSharing": "load_bike_sharing",
        "AdultCensus": "load_adult_census",
        "Zoo": "load_zoo",
        "Hepatitis": "load_hepatitis",
        "Thyroid": "load_thyroid",
        "Mushroom": "load_mushroom",
        "BreastCancer": "load_breast_cancer",
        "Ionosphere": "load_ionosphere",
        "Soybean": "load_soybean",
        "Annealing": "load_annealing",
        "Arrhythmia": "load_arrhythmia",
    }

    for game_name, loader_name in game_loaders.items():
        try:
            # Dynamically fetch the loader function from _all.py
            loader_fn = getattr(loaders, loader_name)

            # X is the feature matrix, y is the target labels
            X, _ = loader_fn()

            # The number of columns in X is the exact number of players (n_players)
            _ = X.shape[1]

        except Exception:
            logger.exception("Failed to check player count for %s", game_name)


if __name__ == "__main__":
    main()
