"""This module contains the LeverageSHAP regression approximator for estimating the SV."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scipy.special

from shapiq.interaction_values import InteractionValues

from .base import Regression

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.game import Game
    from shapiq.typing import FloatVector

ValidRegressionLeverageSHAPIndices = Literal["SV"]


class LeverageSHAP(Regression[ValidRegressionLeverageSHAPIndices]):
    """The LeverageSHAP regression approximator for estimating the Shapley values.

    LeverageSHAP improves on KernelSHAP by using leverage scores of the regression design matrix
    to guide coalition sampling. Coalitions with high leverage scores are more influential for
    estimating the Shapley values and are prioritized during sampling, yielding lower-variance
    estimates for the same evaluation budget.

    See Also:
        - :class:`~shapiq.approximator.regression.kernelshap.KernelSHAP`: The standard KernelSHAP
            approximator for the Shapley value.

    """

    valid_indices: tuple[ValidRegressionLeverageSHAPIndices, ...] = ("SV",)

    def __init__(
        self,
        n: int,
        *,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the LeverageSHAP approximator.

        Args:
            n: The number of players.

            pairing_trick: Not used. LeverageSHAP uses its own sampling scheme via
                :meth:`_sample`. Kept for interface compatibility.

            sampling_weights: Not used. LeverageSHAP uses its own sampling scheme via
                :meth:`_sample`. Kept for interface compatibility.

            random_state: The random state of the estimator. Defaults to ``None``.

            **kwargs: Additional keyword arguments (not used, only for compatibility).
        """
        super().__init__(
            n,
            max_order=1,
            index="SV",
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )

    def approximate(
        self,
        budget: int,
        game: Game | Callable[[np.ndarray], np.ndarray],
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Approximate the Shapley values using leverage-score-guided coalition sampling.

        Args:
            budget: The number of game evaluations available.
            game: The game to approximate.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The estimated Shapley values as an :class:`~shapiq.InteractionValues` object.
        """
        # Same four-step structure as KernelSHAP (Regression.approximate in base.py),
        # but with custom sampling and regression instead of CoalitionSampler + regression_routine.
        Z, proposal_probs = self._sample(budget)  # step 1: sample coalitions
        game_values: FloatVector = game(Z)  # step 2: query the black-box game
        v0 = float(game_values[np.sum(Z, axis=1) == 0][0])
        sv = self._solve(Z, game_values, proposal_probs, v0)  # step 3: solve for Shapley values
        return InteractionValues(  # step 4: package result
            values=sv,
            index=self.approximation_index,
            interaction_lookup=self.interaction_lookup,
            baseline_value=v0,
            min_order=self.min_order,
            max_order=self.max_order,
            n_players=self.n,
            estimated=not budget >= 2**self.n,
            estimation_budget=budget,
            target_index=self.index,
        )

    def _sample(self, budget: int) -> tuple[np.ndarray, FloatVector]:
        """Sample coalitions for the regression.

        Custom sampler bypassing CoalitionSampler.
        Uniform size sampling over {1,...,n-1} + paired sampling (S and complement) + Bernoulli without replacement.
        Returns coalition matrix Z, counts, proposal probs.

        Args:
            budget: The number of game evaluations available.

        Returns:
            Z: Boolean coalition matrix of shape ``(n_coalitions, n)``.
            proposal_probs: Per-coalition proposal probabilities of shape ``(n_coalitions,)``,
                used as importance weights in the regression.
        """
        if budget < 2:
            msg = "Budget must be at least 2 to evaluate baseline and grand coalition."
            raise ValueError(msg)

        seen_coalitions = set()
        Z_list = []
        probs_list = []

        z_empty = np.zeros(self.n, dtype=bool)
        seen_coalitions.add(tuple(z_empty))
        Z_list.append(z_empty)
        probs_list.append(1.0)

        z_grand = np.ones(self.n, dtype=bool)
        seen_coalitions.add(tuple(z_grand))
        Z_list.append(z_grand)
        probs_list.append(1.0)

        current_budget = 2

        if self.n > 2:
            while current_budget < budget:
                s = self._rng.integers(1, self.n)
                active_indices = self._rng.choice(self.n, size=s, replace=False)

                z_current = np.zeros(self.n, dtype=bool)
                z_current[active_indices] = True
                z_tuple = tuple(z_current)

                if z_tuple in seen_coalitions:
                    continue

                seen_coalitions.add(z_tuple)
                Z_list.append(z_current)

                prob = 1.0 / scipy.special.binom(self.n, s)
                probs_list.append(prob)
                current_budget += 1

                # pairing_trick is inherited from the Regression base class via super().__init__;
                # We have to use getattr to satisfy static type checkers (e.g. ruff)
                if getattr(self, "pairing_trick", False) and current_budget < budget:
                    z_paired = (
                        ~z_current
                    )  # The '~' operator flips all Trues to Falses and vice versa
                    z_paired_tuple = tuple(z_paired)

                    if z_paired_tuple not in seen_coalitions:
                        seen_coalitions.add(z_paired_tuple)
                        Z_list.append(z_paired)

                        probs_list.append(prob)
                        current_budget += 1

        # Finally, convert our Python lists into NumPy arrays (faster) for the Solver to use
        Z = np.array(Z_list)
        proposal_probs = np.array(probs_list, dtype=float)

        return Z, proposal_probs

    def _solve(
        self,
        Z: np.ndarray,
        game_values: FloatVector,
        proposal_probs: FloatVector,
        v0: float,
    ) -> FloatVector:
        """Solve the weighted regression to estimate Shapley values.

        Build target y (shifted by v0). Compute A=ZP via row-centering trick (avoid materializing P).
        Form b=y-Z1. Solve via np.linalg.lstsq. Apply Efficiency offset, return InteractionValues.

        Args:
            Z: Boolean coalition matrix of shape ``(n_coalitions, n)``.
            game_values: Game values for each coalition in ``Z``.
            proposal_probs: Per-coalition proposal probabilities from ``_sample``.
            v0: Value of the empty coalition (baseline).

        Returns:
            Array of Shapley values including the empty-coalition entry, matching the shape
            expected by :attr:`~shapiq.approximator.base.Approximator.interaction_lookup`.
        """
        raise NotImplementedError
