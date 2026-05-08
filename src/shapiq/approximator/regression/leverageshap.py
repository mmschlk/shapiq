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
        Z, _ = self._sample(budget)  # step 1: sample coalitions; proposal_probs not needed (IS weights cancel analytically)
        game_values: FloatVector = game(Z)  # step 2: query the black-box game
        v0 = float(game_values[np.sum(Z, axis=1) == 0][0])
        sv = self._solve(Z, game_values, v0)  # step 3: solve for Shapley values
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
        # We need at least budget 2 to include the empty and grand coalitions.
        # Without these two, we cannot compute the baseline or the total model output.
        if budget < 2:
            msg = "Budget must be at least 2 to evaluate baseline and grand coalition."
            raise ValueError(msg)

        # We use a set to keep track of the teams we have already picked to avoid duplicates (sampling without replacement).
        seen_coalitions = set()
        Z_list = []
        probs_list = []

        # Manually add the empty coalition (no features active -> all False)
        z_empty = np.zeros(self.n, dtype=bool)
        seen_coalitions.add(
            tuple(z_empty)
        )  # Convert to tuple because sets cannot store lists/arrays
        Z_list.append(z_empty)
        probs_list.append(1.0)

        # Manually add the grand coalition (all features active -> all True)
        z_grand = np.ones(self.n, dtype=bool)
        seen_coalitions.add(tuple(z_grand))
        Z_list.append(z_grand)
        probs_list.append(1.0)

        # We have used 2 out of our total budget so far.
        current_budget = 2

        # Leverage score sampling loop: We continue sampling until we exhaust our budget.
        # We only need to randomly sample if there are more than 2 features.
        # Cap budget at 2^n: once all 2^n coalitions are in seen_coalitions every draw
        # would be rejected and the loop would spin forever.
        budget = min(budget, 2**self.n)
        if self.n > 2:
            while current_budget < budget:
                # Uniformly pick a coalition size 's' (Ref: Musco & Witter (2024) Section 3.2.)
                # To sample proportional to leverage scores, we pick a subset size uniformly at random.
                s = self._rng.integers(1, self.n)

                # Pick a random subset of size 's'
                # We randomly select 's' indices (features) to be turned on.
                # Note: The original repo (https://github.com/rtealwitter/leverageshap/blob/main/leverageshap/estimators/sampling.py)
                # uses a complex _combination_generator here. Using np.random.choice is computationally simpler.
                active_indices = self._rng.choice(self.n, size=s, replace=False)

                # Create an array of all False, then flip the chosen indices to True
                z_current = np.zeros(self.n, dtype=bool)
                z_current[active_indices] = True
                z_tuple = tuple(z_current)

                # Bernoulli check (no duplicates) (Ref: Musco & Witter (2024) Section 1.1 / Algorithm 1)
                # If we have already drawn this exact team before, skip the rest of this loop
                # (= sampling without replacement) and try again without increasing the budget.
                if z_tuple in seen_coalitions:
                    continue

                # Record the successfully drawn sample
                seen_coalitions.add(z_tuple)
                Z_list.append(z_current)

                # Leverage Score Calculation (Ref: Musco & Witter (2024) Lemma 3.2.)
                # The mathematical leverage score for any team of size 's' is exactly 1 / binom(n, s).
                prob = 1.0 / scipy.special.binom(self.n, s)
                probs_list.append(prob)
                current_budget += 1

                # Paired Sampling (Yin-Yang balancing) (Ref: Musco & Witter (2024) Section 1.1 / Section 3.2)
                # If we draw a team, we also draw its exact opposite to reduce statistical variance.
                # This matches the 'pairing_trick' flag in the original repo's CoalitionSampler.
                # Variable 'pairing_trick' is inherited from the Regression base class via super().__init__;
                # We have to use getattr to satisfy static type checkers (e.g. ruff)
                if getattr(self, "pairing_trick", False) and current_budget < budget:
                    z_paired = (
                        ~z_current
                    )  # The '~' operator flips all Trues to Falses and vice versa
                    z_paired_tuple = tuple(z_paired)

                    # Make sure the opposite team hasn't been drawn before either
                    if z_paired_tuple not in seen_coalitions:
                        seen_coalitions.add(z_paired_tuple)
                        Z_list.append(z_paired)

                        # The opposite team has size (n - s).
                        # Mathematically, binom(n, s) is identical to binom(n, n-s).
                        # Therefore, the probability remains exactly the same!
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
        v0: float,
    ) -> FloatVector:
        """Solve the weighted regression to estimate Shapley values.

        Build target y (shifted by v0). Compute A=ZP via row-centering trick (avoid materializing P).
        Form b=y-Z1. Solve via np.linalg.lstsq. Apply Efficiency offset, return InteractionValues.

        Args:
            Z: Boolean coalition matrix of shape ``(n_coalitions, n)``.
            game_values: Game values for each coalition in ``Z``.
            v0: Value of the empty coalition (baseline).

        Returns:
            Array of Shapley values including the empty-coalition entry, matching the shape
            expected by :attr:`~shapiq.approximator.base.Approximator.interaction_lookup`.
        """
    

        n = self.n # total number of features
        coalition_sizes = Z.sum(axis=1) # Calculates the size of each coalitio


        # Calculates the total payout of game (difference between model's prediction using all features and baseline prediction with no features). -> this is "efficiency gap" we need to distribute among players bc sum of all feature contributions must equal the total payout.
        # Grand coalition value pins efficiency axiom: SUM phi = v(N) − v(sigma).
        v_grand = float(game_values[coalition_sizes == n][0]) # Finds models prediction for grand coalition(where all features are present)

        efficiency_shift = (v_grand - v0) / n # Calculates total prize of game (`v_grand - v0`) && divides it equally among all n features

        # Removes the two trivial coalitions (0 < |S| < n) -> Empty and grand coalitions define v0 and v_grand are excluded ->  not regression rows.
        interior = (coalition_sizes > 0) & (coalition_sizes < n) # Creates a boolean mask of interior coalitions
        Z_int = Z[interior].astype(float) # selects only interior coal rows from Z
        v_int = game_values[interior] # selects only interior coal values from game_values
        s_int = coalition_sizes[interior] # selects only interior coal sizes from coalition_sizes

        # cheks if list of interior coalitions (Z_int) is empty.
        if len(Z_int) == 0:
            # happens when e.g. n ≤ 2 at minimum budget -> sip main regression case because there are only 1 or two featres (v0, v_grand) in coalistions
            return np.concatenate([[v0], np.full(n, efficiency_shift)]) # array where every feature gets same efficiency_shit value (no distingush between importanc)

        # --- A = Z·P  (Lemma 3.1) ---
        # transforms the coalition matrix Z_int into a new matrix A bc regression has efficiency contraint -> normalyl slow -> transformation allows for unconstrained regression problem -> faster solvable -> Not O(n^2) bc row-centering instead of matrix multiplication with P. 
        A = Z_int - (s_int / n)[:, np.newaxis]  # shape (m, n)

        # --- Centred target vector b ---
        # c the input matrix `Z_int` was adjusted to get `A`, the output values (`v_int`) must be adjusted to match -> New target vector for unconstrained reg problem -> NECESSARY BC WE NEED TO SOLVE FOR THE NEW TRANSFORMED PROBLEM defined by lemma 3.1, not the original one. -> b = y - Z·1·(v(N)−v(sigma))/n, where y_j = v(z_j) − v(sigma).
        b = (v_int - v0) - efficiency_shift * s_int  # shape (m,)

        # --- Importance-sampling (IS) weights ---
        # Now calculating weight for each coalition in regression.
        # Trick: in KernelSHAP weights are very complex and are huge numbers -> numerically unstable
        # bc of smart sampeling the complex terms cancle out -> simple formula with no huge binomial coefficients anywhere -> numerically stable.
        # Leverage-score sampling draws each z with probability p = 1 / binom(n, s)
        # (Lemma 3.2), so the IS-corrected WLS weight is  w(s) / p = 1 / (s·(n−s)). -> formula independent of the sampling probabilities
        w_is = 1.0 / (s_int * (n - s_int))  # shape (m,)

        # --- Solve  min_x ‖W^{1/2}·A·x − W^{1/2}·b‖₂²  via weighted least squares ---
        # find best x
        # A = Z·P always has ones in its null space (each row sums to 0, so A·1 = 0) -> Gram matrix rank-deficient (a technical property resulting from row-centering trick) -> uses np.linalg.lstsq because of this
        W_sqrt = np.sqrt(w_is)
        phi_perp = np.linalg.lstsq(
            W_sqrt[:, np.newaxis] * A, W_sqrt * b, rcond=None
        )[0]

        # --- Efficiency correction  (Algorithm 1, line 13) ---
        # takes efficiency_shift calculated in Step 1 and adds it to every value in phi_perp -> shifts the entire solution so that it now satisfies the Efficiency rule
        sv = phi_perp + efficiency_shift

        return np.concatenate([[v0], sv])
