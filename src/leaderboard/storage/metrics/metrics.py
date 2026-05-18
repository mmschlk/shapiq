from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np

from leaderboard.storage.connection import MongoDBClient
from leaderboard.storage.data_classes import RunConfig


class MetricsLoader:
    """
    Collects and aggregates the ``metrics`` sub-document across runs.

    Parameters
    ----------
    db:
        An open ``MongoDBClient`` instance.
    """

    def __init__(self, db: MongoDBClient) -> None:
        self.db = db

    def load_for_config(self, config: RunConfig) -> Dict[str, List[float]]:
        """
        Return raw metric values for every successful run matching *config*.

        Returns
        -------
        dict
            ``{metric_name: [value_seed_1, value_seed_2, ...]}``.
        """
        runs = self.db.get_by_config(config)
        metrics: Dict[str, List[float]] = defaultdict(list)

        for run in runs:
            if run.get("run_failed", False):
                continue
            for key, value in run.get("metrics", {}).items():
                metrics[key].append(value)

        return dict(metrics)

    def aggregate(self, config: RunConfig) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for each metric over all seeds.

        Returns
        -------
        dict
            ``{metric_name: {"mean": …, "std": …, "min": …, "max": …, "count": …}}``.
        """
        aggregated = {}

        for key, values in self.load_for_config(config).items():
            arr = np.array(values, dtype=float)
            aggregated[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "count": int(len(arr)),
            }

        return aggregated