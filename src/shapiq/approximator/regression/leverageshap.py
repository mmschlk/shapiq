"""LeverageSHAP regression approximator (Algorithm 1 of Musco & Witter, 2024)."""

from __future__ import annotations

import math
import random as _py_random
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

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
    3. Reweights each row by ``w(||z||) / min(1, 2c·ℓ_z)`` where
       ``w(s) = (s-1)!(n-s-1)!/n!`` is the Shapley kernel weight and
       ``ℓ_z = 1/C(n,s)`` is its leverage score.
    4. Solves the unconstrained centered regression (Lemma 3.1) and adds the
       efficiency offset.

    Note:
        The number of game evaluations is a random variable concentrated around
        ``budget``; small overshoots and undershoots are expected. The paper notes
        a deterministic variant (``m_s := E[Binomial(...)]``) is also valid; this
        implementation uses the random Binomial form to match the paper's main
        figures.

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
        v0 = float(game_values[np.sum(Z, axis=1) == 0][0])
        sv = self._solve(Z, game_values, v0, weights)
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
        """Algorithm 1, lines 1-7: BernoulliSample plus IS reweighting.

        Args:
            budget: Target number of evaluations ``m``.

        Returns:
            Z: Boolean coalition matrix of shape ``(n_coalitions, n)`` containing
                the empty coalition, the grand coalition, and the BernoulliSample
                pairs.
            weights: Per-coalition IS weights ``w(s) / min(1, 2c·ℓ_z)`` with
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

        # IS weights (Algorithm 1 line 7) computed in log space to stay numerically
        # stable for large n where C(n, n/2) easily exceeds 1e300.
        if Z_pairs.shape[0] > 0:  # if any pairs were sampled
            log_2c = math.log(2.0 * c) if c > 0 else -math.inf  # log(2c) for log-space math
            log_weights = np.empty(Z_pairs.shape[0], dtype=float)  # buffer for log IS weights
            for i, s in enumerate(sizes):  # for each sampled coalition of size s
                log_w = (
                    math.lgamma(s) + math.lgamma(n - s) - math.lgamma(n + 1)
                )  # log Shapley kernel w(s)
                log_C = (
                    math.lgamma(n + 1) - math.lgamma(s + 1) - math.lgamma(n - s + 1)
                )  # log C(n,s)
                log_p = log_2c - log_C  # log(2c · ℓ_z) = log(2c / C(n,s))
                log_min_p = min(0.0, log_p)  # cap probability at 1 (log 1 = 0)
                log_weights[i] = log_w - log_min_p  # IS weight = w(s) / min(1, 2c·ℓ_z)
            log_weights -= log_weights.max()  # shift so exp doesn't overflow
            weights_pairs = np.exp(log_weights)  # back to linear space
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
        if n < 2:
            return 0.0  # trivial case: nothing to sample
        target = m - 2  # budget minus empty + grand
        if target <= 0:
            return 0.0  # nothing left to sample beyond empty + grand

        binoms = [float(math.comb(n, s)) for s in range(1, n)]  # C(n,s) for each interior size
        max_binom = max(binoms)  # largest binomial coefficient (≈ middle size)

        def total(c_: float) -> float:  # expected sample count for a given c
            two_c = 2.0 * c_
            return sum(min(b, two_c) for b in binoms)  # sum of min(C(n,s), 2c)

        # Upper bound: 2c >= max_binom guarantees full inclusion of every size.
        hi = max_binom / 2.0 + 1.0  # upper bound: covers every size fully
        if total(hi) < target:
            return hi  # even max c can't reach target → return it
        lo = 0.0  # lower bound for binary search
        for _ in range(200):  # bisect up to 200 iterations
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
        py_rng = _py_random.Random(py_seed)  # python RNG (handles big ints)

        two_c = 2.0 * c  # cached for the loop
        for s in range(1, n // 2 + 1):  # iterate sizes 1..⌊n/2⌋ (rest covered via complement z̄)
            is_middle = (n % 2 == 0) and (s == n // 2)  # special case: pair would self-complement
            full_count = math.comb(n, s)  # C(n, s) total subsets of this size
            prob = (
                1.0 if full_count <= two_c else two_c / full_count
            )  # inclusion probability ℓ_z·2c

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
        if total < 10**6:
            return py_rng.sample(range(total), k)  # small pool: use built-in sampler
        seen: set[int] = set()  # track unique picks
        # Rejection sampling: collisions are rare when total >> k.
        while len(seen) < k:
            seen.add(py_rng.randrange(total))  # pick a random index, dedupe via set
        return list(seen)  # return as list

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

    def _solve(
        self,
        Z: np.ndarray,
        game_values: FloatVector,
        v0: float,
        weights: np.ndarray,
    ) -> FloatVector:
        """Algorithm 1, lines 8-13: weighted least squares with provided IS weights.

        Builds A = Z·P (row-centering trick from Lemma 3.1), centered target
        ``b = (y - v0·1) - efficiency_shift · Z·1`` and solves
        ``argmin_x ||W^{1/2} A x - W^{1/2} b||_2`` via lstsq, then adds the
        efficiency offset.

        Args:
            Z: Coalition matrix (rows include empty and grand coalitions).
            game_values: ``v(z)`` for each row of ``Z``.
            v0: Empty-coalition value (baseline).
            weights: Per-row IS weights from ``_sample``; entries for empty/grand
                are zero and are filtered out by the interior mask.

        Returns:
            ``[v0, phi_1, ..., phi_n]``.
        """
        n = self.n  # number of players
        coalition_sizes = Z.sum(axis=1)  # |z| for each row

        v_grand = float(game_values[coalition_sizes == n][0])  # value of grand coalition v(N)
        efficiency_shift = (
            v_grand - v0
        ) / n  # average per-player share (used to re-center problem)

        interior = (coalition_sizes > 0) & (coalition_sizes < n)  # rows excluding empty + grand
        Z_int = Z[interior].astype(float)  # interior coalition matrix
        v_int = game_values[interior]  # interior game values v(z)
        s_int = coalition_sizes[interior]  # interior sizes |z|
        w_is = weights[interior]  # interior IS weights

        if len(Z_int) == 0:
            return np.concatenate(
                [[v0], np.full(n, efficiency_shift)]
            )  # fallback: split value evenly

        # A = Z·P with P = I - 1/n · 11^T → row-center each Z row by its size mean.
        A = Z_int - (s_int / n)[:, np.newaxis]  # row-center: applies projection P from Lemma 3.1
        b = (v_int - v0) - efficiency_shift * s_int  # centered target vector

        W_sqrt = np.sqrt(w_is)  # sqrt of IS weights for weighted lstsq
        phi_perp = np.linalg.lstsq(W_sqrt[:, np.newaxis] * A, W_sqrt * b, rcond=None)[
            0
        ]  # solve weighted least squares

        sv = phi_perp + efficiency_shift  # add efficiency offset back to recover Shapley values
        return np.concatenate([[v0], sv])  # prepend baseline → [v0, φ₁, ..., φₙ]
