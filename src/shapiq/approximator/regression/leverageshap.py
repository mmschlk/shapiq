"""LeverageSHAP regression approximator (Algorithm 1 of Musco & Witter, 2024)."""

from __future__ import annotations

import math
import random as _py_random
import sys
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from shapiq.approximator.regression.base import solve_regression
from shapiq.interaction_values import InteractionValues

from .base import Regression

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.game import Game
    from shapiq.typing import FloatVector

ValidRegressionLeverageSHAPIndices = Literal["SV"]


class LeverageSHAP(Regression[ValidRegressionLeverageSHAPIndices]):
    """The LeverageSHAP regression approximator for estimating Shapley values.

    Faithful implementation of Algorithm 1 from Musco & Witter (2024). The algorithm:

    1. Finds an oversampling parameter ``c`` via binary search such that
       ``m - 2 = sum_{s=1}^{n-1} min(C(n,s), 2c)`` (Equation 12).
    2. Draws coalition pairs ``(z, z̄)`` via Bernoulli sampling without replacement
       (Algorithm 2): for each size ``s``, ``m_s ~ Binomial(C(n,s), min(1, 2c/C(n,s)))``
       pairs are included. Subsets of small sizes (where ``C(n,s) <= 2c``) are
       included deterministically; only larger sizes are subsampled.
    3. Reweights each row by ``w(||z||) / min(1, 2c·l_z)`` where
       ``w(s) = (s-1)!(n-s-1)!/n!`` is the Shapley kernel weight and
       ``l_z = 1/C(n,s)`` is its leverage score.
    4. Solves the unconstrained centered regression (Lemma 3.1) and adds the
       efficiency offset.

    Note:
        The number of game evaluations is a random variable concentrated around
        ``budget``; small overshoots and undershoots are expected. The paper notes
        a deterministic variant (``m_s := E[Binomial(...)]``) is also valid; this
        implementation uses the random Binomial form to match the paper's main
        figures.

    Example:
        >>> from shapiq.approximator.regression import LeverageSHAP
        >>> from shapiq_games.synthetic import DummyGame
        >>> n = 5
        >>> game = DummyGame(n=n, interaction=(1, 2))
        >>> approximator = LeverageSHAP(n=n, random_state=42)
        >>> sv_estimates = approximator.approximate(budget=100, game=game)
        >>> print(sv_estimates.values)
        [0.  0.2 0.7 0.7 0.2 0.2]
    """

    valid_indices: tuple[ValidRegressionLeverageSHAPIndices, ...] = ("SV",)

    def __init__(
        self,
        n: int,
        *,
        pairing_trick: bool = True,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the LeverageSHAP approximator.

        Args:
            n: The number of players.

            pairing_trick: Kept for interface compatibility. Algorithm 1 always
                samples pairs ``(z, z̄)`` together.

            sampling_weights: Kept for interface compatibility. LeverageSHAP uses
                its own leverage-score-based sampling scheme.

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
        """Approximate the Shapley values via leverage-score-guided sampling.

        Args:
            budget: Target number of game evaluations (Algorithm 1 input ``m``).
            game: The game to approximate.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            The estimated Shapley values as an :class:`~shapiq.InteractionValues` object.
        """
        Z, weights = self._sample(budget)
        game_values: FloatVector = game(Z)
        if not np.all(np.isfinite(game_values)):
            msg = "Game returned NaN or Inf values. LeverageSHAP requires finite game values."
            raise ValueError(msg)
        v0 = float(game_values[np.sum(Z, axis=1) == 0][0])

        n = self.n
        coalition_sizes = Z.sum(axis=1)
        v_grand = float(game_values[coalition_sizes == n][0])
        efficiency_shift = (v_grand - v0) / n

        interior = (coalition_sizes > 0) & (coalition_sizes < n)
        Z_int = Z[interior].astype(float)
        v_int = game_values[interior]
        s_int = coalition_sizes[interior]
        w_is = weights[interior]

        if len(Z_int) == 0:
            sv = np.concatenate([[v0], np.full(n, efficiency_shift)])
        else:
            # Keep all LeverageSHAP-specific math here: the shared solver only handles
            # the weighted least-squares step.
            A = Z_int - (s_int / n)[:, np.newaxis]
            b = (v_int - v0) - efficiency_shift * s_int
            phi_perp = solve_regression(
                X=A,
                y=b,
                kernel_weights=w_is,
                use_svd=True,
            )
            sv = np.concatenate([[v0], phi_perp + efficiency_shift])
        return InteractionValues(
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

    def _sample(self, budget: int) -> tuple[np.ndarray, np.ndarray]:
        r"""Algorithm 1, lines 1-7: BernoulliSample plus IS reweighting.

        This method implements the custom Bernoulli sampling logic required by
        LeverageSHAP, bypassing the generic ``CoalitionSampler``. This is necessary
        to strictly enforce the $2c$ threshold boundaries (Equation 12). By explicitly
        oversampling and deterministically evaluating low-cardinality subsets
        (where \binom{n}{s} \le 2c), this algorithm optimally captures high-leverage
        main effects.

        Args:
            budget: Target number of evaluations ``m``.

        Returns:
            Z: Boolean coalition matrix of shape ``(n_coalitions, n)`` containing
                the empty coalition, the grand coalition, and the BernoulliSample
                pairs.
            weights: Per-coalition IS weights ``w(s) / min(1, 2c·l_z)`` with
                arbitrary positive scale (only relative weights matter for lstsq).
                Empty/grand coalitions get weight 0 (they enter via the efficiency
                shift, not the regression).
        """
        if budget < 2:  # need at least empty + grand coalition
            msg = "Budget must be at least 2 to evaluate baseline and grand coalition."
            raise ValueError(msg)

        n = self.n  # number of players
        m = min(budget, 2**n)  # cap budget at full enumeration (2^n)

        z_empty = np.zeros(n, dtype=bool)  # empty coalition (no players)
        z_grand = np.ones(n, dtype=bool)  # grand coalition (all players)

        c = self._find_c(n, m)  # oversampling parameter from Eq. 12 of paper

        Z_pairs, sizes = self._bernoulli_sample(n, c)  # draw the (z, z̄) coalition pairs

        # IS weights (Algorithm 1 line 7)
        if Z_pairs.shape[0] > 0:  # if any pairs were sampled
            weights_pairs = np.empty(Z_pairs.shape[0], dtype=float)

            # Precompute factorial of n to save cycles, Python handles this as huge int
            fact_n = math.factorial(n)

            for i, s in enumerate(sizes):
                # Exact Shapley kernel weight: w(s) = (s-1)!(n-s-1)! / n!
                # Computed entirely in native Python big-int space before converting to float
                w_s = (math.factorial(s - 1) * math.factorial(n - s - 1)) / fact_n

                # Leverage score: l_z = 1 / C(n, s)
                l_z = 1.0 / math.comb(n, s)

                # Cap probability at 1
                p = min(1.0, 2.0 * c * l_z)

                # IS weight = w(s) / min(1, 2c * l_z)
                weights_pairs[i] = w_s / p

            Z = np.vstack(
                [z_empty[None, :], z_grand[None, :], Z_pairs]
            )  # stack empty + grand + pairs
        else:
            weights_pairs = np.empty(0, dtype=float)  # no pairs → no weights
            Z = np.vstack([z_empty[None, :], z_grand[None, :]])  # only empty + grand

        weights = np.concatenate(
            [[0.0, 0.0], weights_pairs]
        )  # empty/grand get weight 0 (excluded from lstsq)
        return Z, weights

    @staticmethod
    def _find_c(n: int, m: int) -> float:
        """Algorithm 1, line 2: binary search for ``c`` solving Eq. 12.

        ``m - 2 = sum_{s=1}^{n-1} min(C(n,s), 2c)``.
        """
        MAX_BISECT_ITER = 200

        if n < 2:
            return 0.0  # trivial case: nothing to sample
        target = m - 2  # budget minus empty + grand
        if target <= 0:
            return 0.0  # nothing left to sample beyond empty + grand

        binoms = [math.comb(n, s) for s in range(1, n)]  # keep as int to avoid float overflow

        def total(c_: float) -> float:  # expected sample count for a given c
            two_c = 2.0 * c_
            return float(sum((min(b, two_c)) for b in binoms))

        # Find an upper bound without relying on float(max_binom), which can overflow for large n.
        hi = 1.0
        while total(hi) < target:
            hi *= 2.0
        lo = 0.0  # lower bound for binary search
        for _ in range(MAX_BISECT_ITER):  # bisect up to MAX_BISECT_ITER iterations
            mid = 0.5 * (lo + hi)  # midpoint
            if total(mid) >= target:
                hi = mid  # too big, shrink upper bound
            else:
                lo = mid  # too small, raise lower bound
            if hi - lo < 1e-12 * max(1.0, hi):
                break  # converged
        return 0.5 * (lo + hi)  # final c estimate

    def _bernoulli_sample(self, n: int, c: float) -> tuple[np.ndarray, np.ndarray]:
        """Algorithm 2 (BernoulliSample) of Musco & Witter (2024).

        For each size ``s in {1, ..., floor(n/2)}`` draws ``m_s ~ Binomial`` pairs
        ``(z, z̄)`` without replacement. The middle size (when ``n`` is even and
        ``s = n/2``) is partitioned by fixing ``z_n = 1`` so each unordered pair
        is sampled at most once.

        Returns:
            Z_pairs: Boolean coalition matrix with both ``z`` and ``z̄`` appended
                consecutively for each pair.
            sizes: Cardinality of each row of ``Z_pairs``.
        """
        if n < 2 or c <= 0.0:
            return np.zeros((0, n), dtype=bool), np.zeros(0, dtype=int)  # nothing to sample

        z_list: list[np.ndarray] = []  # collected coalition vectors
        sizes_list: list[int] = []  # their sizes

        # Convert numpy seed to a Python random seed so randrange supports
        # arbitrary-precision integers (needed for large n where C(n, n/2) overflows int64).
        py_seed = int(self._rng.integers(0, 2**32))  # reproducible seed for python RNG
        py_rng = _py_random.Random(py_seed)  # noqa: S311 - reproducible, non-crypto sampling

        two_c = 2.0 * c  # cached for the loop
        for s in range(1, n // 2 + 1):  # iterate sizes 1..⌊n/2⌋ (rest covered via complement z̄)
            is_middle = (n % 2 == 0) and (s == n // 2)  # special case: pair would self-complement
            full_count = math.comb(n, s)  # C(n, s) total subsets of this size
            prob = (
                1.0 if full_count <= two_c else two_c / full_count
            )  # inclusion probability l_z*2c

            # Number of distinct unordered pairs at this size.
            pool_size = (
                math.comb(n - 1, s - 1) if is_middle else full_count
            )  # # of distinct unordered pairs

            if prob >= 1.0:
                m_s = pool_size  # include all pairs deterministically
            elif pool_size > 2**31 - 1:
                # Pool overflows C long → fall back to Poisson(pool_size·prob).
                # In this regime pool_size·prob = 2c is bounded and prob → 0, so
                # Poisson is exact in the limit (n large, p small, np fixed).
                m_s = min(int(self._rng.poisson(pool_size * prob)), pool_size)
            else:
                # Pseudocode samples Binomial(C(n,s), prob) then halves for middle;
                # equivalent (and exact) to Binomial(pool_size, prob) here.
                m_s = int(self._rng.binomial(pool_size, prob))  # random count of pairs to draw

            if m_s == 0:
                continue  # skip this size

            indices = self._sample_without_replacement(
                pool_size, m_s, py_rng
            )  # pick m_s unique indices

            for idx in indices:  # build each sampled coalition
                if is_middle:
                    # Sample over n-1 items with size s-1, then fix z_n = 1.
                    z_partial = self._combo(n - 1, s - 1, idx)  # combo over n-1 items
                    z = np.zeros(n, dtype=bool)
                    z[: n - 1] = z_partial  # copy partial pattern
                    z[n - 1] = True  # force last player → ensures unique unordered pair
                else:
                    z = self._combo(n, s, idx)  # idx-th lexicographic combination
                z_bar = ~z  # complement (paired sampling)
                z_list.append(z)  # add z
                z_list.append(z_bar)  # add z̄
                sizes_list.append(int(z.sum()))  # |z|
                sizes_list.append(int(z_bar.sum()))  # |z̄| = n - |z|

        if z_list:
            return np.array(z_list), np.array(sizes_list, dtype=int)  # stack into arrays
        return np.zeros((0, n), dtype=bool), np.zeros(0, dtype=int)  # nothing got sampled

    @staticmethod
    def _sample_without_replacement(total: int, k: int, py_rng: _py_random.Random) -> list[int]:
        """Sample ``k`` distinct integers from ``[0, total)`` without replacement.

        ``total`` may be an arbitrary-precision Python int (for large ``n``).
        """
        if k >= total:
            return list(range(total))  # asking for everything → return all indices

        # Fallback for astronomically large binomial pools (e.g. n=101) where
        # range(total) exceeds C ssize_t (sys.maxsize) and crashes random.sample.
        # Since k is tiny compared to total, rejection collisions are practically impossible.
        if total > sys.maxsize:
            seen: set[int] = set()
            while len(seen) < k:
                seen.add(py_rng.randrange(total))
            return list(seen)

        # random.sample supports range objects efficiently without materializing them and is
        # robust even when k is a large fraction of total.
        return py_rng.sample(range(total), k)

    @staticmethod
    def _combo(n: int, s: int, i: int) -> np.ndarray:
        """Algorithm 3: ``i``-th lexicographic combination of size ``s`` from ``n`` items.

        Returns a boolean vector of length ``n`` with exactly ``s`` True entries.
        ``i`` is 0-indexed.
        """
        z = np.zeros(n, dtype=bool)  # output coalition vector
        if s == 0:
            return z  # empty combination
        k = s  # remaining slots to fill
        j = 0  # current position
        while k > 0 and j < n:
            # Number of combinations that include j (and choose k-1 from remaining n-j-1).
            count = math.comb(n - j - 1, k - 1)  # # combinations with position j included
            if i < count:
                z[j] = True  # include position j in the coalition
                k -= 1  # one fewer slot to fill
            else:
                i -= count  # skip this branch, adjust i
            j += 1  # advance to next position
        return z  # boolean vector with exactly s True entries
