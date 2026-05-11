from typing import Dict, List
from collections import defaultdict
import numpy as np

from config import RunConfig
from database import MongoDBClient


class MetricsLoader:
    def __init__(self, db: MongoDBClient):
        self.db = db

    def load_metrics_for_config(self, config: RunConfig) -> Dict[str, List[float]]:
        """
        Collect metrics across all runs with the same config (different seeds).
        Returns: {metric_name: [values]}
        """
        runs = self.db.get_runs_by_config(config)
        metrics_dict = defaultdict(list)

        for run in runs:
            if run.get("run_failed", False):
                continue

            metrics = run.get("metrics", {})
            for key, value in metrics.items():
                metrics_dict[key].append(value)

        return dict(metrics_dict)

    def aggregate_metrics(self, config: RunConfig) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for each metric.
        Returns:
        {
            "mse": {"mean": ..., "std": ..., "min": ..., "max": ..., "count": ...},
            ...
        }
        """
        metrics = self.load_metrics_for_config(config)
        aggregated = {}

        for key, values in metrics.items():
            arr = np.array(values)
            aggregated[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "count": int(len(arr)),
            }

        return aggregated