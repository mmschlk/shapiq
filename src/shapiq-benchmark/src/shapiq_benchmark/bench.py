"""Class to run benchmarks for interaction value approximation methods."""
import random
import pandas as pd
import numpy as np

from collections.abc import Callable

from shapiq.approximator.base import Approximator
from shapiq.interaction_values import InteractionValues
from shapiq.typing import IndexType, Model
from shapiq_benchmark.configuration import DATASET_NAME_TO_GAME_MAPPING
from shapiq_benchmark.metrics import Metric, get_all_metrics
from shapiq_games.benchmark.local_xai.base import LocalExplanation
from .approximators import get_approximators
from .run import (
    run_benchmark,
    run_benchmark_from_configuration,
)


class IntBench:
    def __init__(
        self,
        model: str | Model | Callable[[np.ndarray], np.ndarray],
        data: str | np.ndarray,
        x_explain: int | np.ndarray | None,
        benchmark_name: str = "benchmark",
        random_state: int = 42,
        *args,
        **kwargs,
    ) -> None:
        self.model = model
        self.data = data

        if isinstance(data, np.ndarray):
            if x_explain is None:
                self.x_explain = random.sample(list(data), 1)[0]
            elif isinstance(x_explain, int):
                self.x_explain = data[x_explain]
            else:
                self.x_explain = x_explain
        else:
            self.game_class = DATASET_NAME_TO_GAME_MAPPING[data.lower()]

        if isinstance(model, str):
            if not isinstance(data, str):
                raise ValueError("If model is given as string, data must also be given as string.")
            self.config = {"model_name": model.lower(), "imputer": "marginal"} #TODO add more options
            self.game = self.game_class(**self.config)
        else:
            if isinstance(data, str):
                raise ValueError("If model is given as a model, data must be given as a np.ndarray.")
            self.game = LocalExplanation(
                data=self.data, model=self.model.predict, x=self.x_explain
            )

        self.random_state = random_state
        self.benchmark_name = benchmark_name

    def run(
        self,
        index: IndexType,
        order: int,
        budget: int,
        approximation_methods: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run the benchmark with the given index, order, budget, and approximation methods for the game instantiated with model and data."""
        return run_benchmark(
            index=index,
            order=order,
            games=[self.game],
            approximators=approximation_methods,
            max_budget=budget,
        )

    def run_from_game_class_config(
        self,
        index: IndexType,
        order: int,
        budget: int,
        approximation_methods: (
            list[Approximator] | list[Approximator.__class__] | list[str] | None
        ) = None,
    ):
        """Run the benchmark with the given index, order, budget, approximation methods for a game class given as string."""
        return run_benchmark_from_configuration(
            index=index,
            order=order,
            game_class=self.game_class,
            game_configuration=self.config,
            approximators=approximation_methods,
            max_budget=budget,
        )

    def approximate_values(
        self,
        index: IndexType,
        order: int,
        budget: int,
        approximation_methods: list[str] | None = None,
    ):
        """Compute the approximate interaction values for a given index and order."""
        approximators = get_approximators(
            APPROXIMATORS=approximation_methods,
            NPLAYERS=self.game.n_players,
            RANDOMSTATE=self.random_state,
            INDEX=index,
            MAXORDER=order,
            PAIRING=False,
        )
        results = {}
        for approximator in approximators:
            result = approximator.approximate(budget=budget, game=self.game)
            results[approximator.name] = result
        return results

    def exact_values(self, index: IndexType, order: int) -> InteractionValues:
        """Compute the exact interaction values for a given index and order."""
        return self.game.exact_values(index=index, order=order)

    def compare(
        self, index: IndexType, order: int, prediction_values: InteractionValues
    ) -> list[Metric]:
        """Compare the computed interaction values with the exact values."""
        ground_truth = self.exact_values(index=index, order=order)
        return get_all_metrics(
            ground_truth=ground_truth,
            estimated=prediction_values,
            estimated_game=self.game,
        )
