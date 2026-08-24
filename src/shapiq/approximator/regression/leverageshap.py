"""LeverageSHAP regression approximator (Algorithm 1 of Musco and Witter, 2025)."""

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
    r"""Leverage SHAP approximator for Shapley values.

    Leverage SHAP (Musco and Witter, 2025 :cite:t:`Musco.2025`) recovers Shapley
    values as the solution of a weighted least-squares problem over sampled
    coalitions, like KernelSHAP, but samples coalitions proportional to their
    statistical *leverage scores*, which have the closed form ``l_z = 1/C(n, ||z||)``
    (Lemma 3.2). Implementation of Algorithm 1:

    1. For the deterministic default, normalize the budget to an even number without
       exceeding it. Solve for the oversampling parameter ``c`` so that
       ``m - 2 = sum_{s=1}^{n-1} min(C(n, s), 2c)`` (Eq. 12); two evaluations are
       reserved for the empty and grand coalitions.
    2. Draw coalition pairs ``(z, z̄)`` without replacement (Algorithm 2). By default
       (``deterministic_counts=True``) each size's pair count is fixed to the
       expectation of the Binomial draw, largest-remainder rounded so the total is
       exact -- stratification in the spirit of SVARM :cite:t:`Kolpaczki.2024a`; with
       ``deterministic_counts=False`` it is drawn at random as Algorithm 2 states.
       Sizes whose layer fits within ``2c`` are enumerated exhaustively.
    3. Reweight each row by the inverse of its inclusion probability. For the
       deterministic default this probability is the realized per-size count divided
       by ``C(n, s)``; for the Binomial variant it is ``min(1, 2c * l_z)``.
    4. Project out the efficiency constraint (Lemma 3.1), solve by weighted least
       squares, and add the efficiency offset back.

    Note:
        The deterministic default follows the fixed-per-size design used by the
        paper's released implementation and reported experiments. To preserve
        shapiq's hard budget ceiling, an odd budget is rounded down rather than up as
        in that implementation; largest-remainder ties can also select a different
        size. The evaluation count is exactly
        ``2 + 2 * ((min(budget, 2**n) - 2) // 2)``. The paper's accuracy theorem is
        proved for the Binomial ``deterministic_counts=False`` variant only
        (Musco and Witter, 2025, end of Sec. 4).

    Example:
        >>> from shapiq.approximator import LeverageSHAP
        >>> from shapiq_games.synthetic import DummyGame
        >>> n = 5
        >>> game = DummyGame(n=n, interaction=(1, 2))
        >>> approximator = LeverageSHAP(n=n, random_state=42)
        >>> sv_estimates = approximator.approximate(budget=100, game=game)
        >>> print(sv_estimates.values)
        [0.  0.2 0.7 0.7 0.2 0.2]

    See Also:
        - :class:`~shapiq.approximator.regression.kernelshap.KernelSHAP`: The original
          KernelSHAP approximator that Leverage SHAP refines.
    """

    valid_indices: tuple[ValidRegressionLeverageSHAPIndices, ...] = ("SV",)

    def __init__(
        self,
        n: int,
        *,
        pairing_trick: bool = True,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
        deterministic_counts: bool = True,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the LeverageSHAP approximator.

        Args:
            n: The number of players.

            pairing_trick: If ``True`` (default), every sampled coalition's complement
                is also included (Algorithm 1's design). If ``False``, the same
                per-size counts are drawn independently instead -- the paper's
                "without paired sampling" ablation. Both modes sample without
                replacement.

            sampling_weights: Inert; kept only for interface compatibility.

            random_state: The random state of the estimator. Defaults to ``None``.

            deterministic_counts: If ``True`` (default), fix each size's pair count to
                the expectation of Algorithm 2's Binomial draw (largest-remainder
                rounded, exact total); if ``False``, draw it at random. See the class
                docstring's Note.

            **kwargs: Additional keyword arguments (not used, only for compatibility).
        """
        self.deterministic_counts = deterministic_counts
        self.pairing_trick = pairing_trick
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
            The estimated Shapley values as an :class:`~shapiq.InteractionValues`
            object. ``estimation_budget`` reports the realized number of evaluations:
            exact with ``deterministic_counts=True`` (see the class Note), a random
            draw that concentrates around the budget otherwise.

        Raises:
            ValueError: If ``budget`` is less than ``2`` (the empty and grand coalitions
                must both be evaluated), or if the game returns non-finite (NaN/Inf) values.
        """
        Z, weights = self._sample(budget)
        game_values: FloatVector = game(Z)
        n_evaluations = int(Z.shape[0])
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
            estimation_budget=n_evaluations,
            target_index=self.index,
        )

    def _sample(self, budget: int) -> tuple[np.ndarray, np.ndarray]:
        r"""Algorithm 1, lines 1-7: BernoulliSample plus IS reweighting.

        Bypasses the generic ``CoalitionSampler`` to enforce the ``2c`` thresholds of
        Eq. 12 exactly; extreme sizes fit within ``2c`` and are enumerated
        exhaustively.

        Args:
            budget: Target number of evaluations ``m``.

        Returns:
            Z: Boolean coalition matrix (empty and grand coalition first).
            weights: Per-coalition IS weights (relative scale only), computed from
                each layer's realized fixed-count probability in deterministic mode
                and its Binomial inclusion probability otherwise. Empty/grand get 0
                because they enter via the efficiency shift, not the regression.
        """
        if budget < 2:
            msg = "Budget must be at least 2 to evaluate baseline and grand coalition."
            raise ValueError(msg)

        n = self.n
        m = min(budget, 2**n)
        if self.deterministic_counts:
            # Paired sampling can only realize an even total. Unlike the released
            # implementation, shapiq treats ``budget`` as a hard ceiling.
            m = 2 + 2 * ((m - 2) // 2)

        z_empty = np.zeros(n, dtype=bool)
        z_grand = np.ones(n, dtype=bool)

        c = self._find_c(n, m)
        if self.deterministic_counts:
            Z_pairs, sizes = self._bernoulli_sample_deterministic(n, c, m)
        else:
            Z_pairs, sizes = self._bernoulli_sample(n, c)

        if Z_pairs.shape[0] > 0:
            weights_pairs = np.empty(Z_pairs.shape[0], dtype=float)
            two_c = 2.0 * c
            fact_n = math.factorial(n)
            unique_sizes, size_counts = np.unique(sizes, return_counts=True)
            realized = dict(zip(unique_sizes.tolist(), size_counts.tolist(), strict=True))

            for i, s in enumerate(sizes):
                if self.deterministic_counts:
                    # A fixed-size uniform draw of k_s rows has marginal inclusion
                    # probability k_s/C(n,s). Dividing the kernel weight by it cancels
                    # C(n,s), including when the layer is fully enumerated.
                    weights_pairs[i] = 1.0 / (s * (n - s) * realized[int(s)])
                else:
                    full_count = math.comb(n, s)
                    if full_count <= two_c:
                        weights_pairs[i] = (
                            math.factorial(s - 1) * math.factorial(n - s - 1) / fact_n
                        )
                    else:
                        # w(s)/p collapses to 1/(s*(n-s)*2c).
                        weights_pairs[i] = 1.0 / (s * (n - s) * two_c)

            Z = np.vstack([z_empty[None, :], z_grand[None, :], Z_pairs])
        else:
            weights_pairs = np.empty(0, dtype=float)
            Z = np.vstack([z_empty[None, :], z_grand[None, :]])

        weights = np.concatenate([[0.0, 0.0], weights_pairs])
        return Z, weights

    @staticmethod
    def _find_c(n: int, m: int) -> float:
        """Algorithm 1, line 2: binary search for ``c`` solving Eq. 12.

        ``m - 2 = sum_{s=1}^{n-1} min(C(n,s), 2c)``.
        """
        MAX_BISECT_ITER = 200

        if n < 2:
            return 0.0
        target = m - 2
        if target <= 0:
            return 0.0

        binoms = [math.comb(n, s) for s in range(1, n)]  # int, avoids float overflow

        def total(c_: float) -> float:
            two_c = 2.0 * c_
            return float(sum(min(b, two_c) for b in binoms))

        # Double instead of starting at float(max binom), which overflows for large n.
        hi = 1.0
        while total(hi) < target:
            hi *= 2.0
        lo = 0.0
        for _ in range(MAX_BISECT_ITER):
            mid = 0.5 * (lo + hi)
            if total(mid) >= target:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-12 * max(1.0, hi):
                break
        # Return ``hi``, not the midpoint: total(hi) >= target is the bisection
        # invariant, so at budget == 2**n every layer is guaranteed exhaustive.
        return hi

    def _bernoulli_sample(self, n: int, c: float) -> tuple[np.ndarray, np.ndarray]:
        """Algorithm 2 (BernoulliSample) of Musco and Witter (2025) :cite:t:`Musco.2025`.

        Draws ``m_s ~ Binomial`` per half-size (the ``deterministic_counts=False``
        path). For the self-complementary middle layer, Algorithm 2 draws against the
        full layer and then halves the result. Row construction is delegated to
        ``_build_rows``.

        Returns:
            Z_pairs: Boolean coalition matrix.
            sizes: Cardinality of each row of ``Z_pairs``.
        """
        if n < 2 or c <= 0.0:
            return np.zeros((0, n), dtype=bool), np.zeros(0, dtype=int)

        # Python RNG: randrange/sample handle arbitrary-precision ints (large-n pools).
        py_seed = int(self._rng.integers(0, 2**32))
        py_rng = _py_random.Random(py_seed)  # noqa: S311 - reproducible, non-crypto sampling

        counts: dict[int, int] = {}
        two_c = 2.0 * c
        for s in range(1, n // 2 + 1):  # sizes 1..⌊n/2⌋ (the rest are covered via complement z̄)
            is_middle = (n % 2 == 0) and (s == n // 2)  # pair would self-complement here
            full_count = math.comb(n, s)
            pool_size = math.comb(n - 1, s - 1) if is_middle else full_count

            if full_count <= two_c:
                m_s = pool_size
            elif full_count > np.iinfo(np.int64).max:
                # NumPy cannot represent the Binomial trial count. Here p is tiny and
                # the mean is fixed, so Poisson is the limiting distribution.
                raw_count = int(self._rng.poisson(two_c))
                m_s = raw_count // 2 if is_middle else raw_count
                m_s = min(m_s, pool_size)
            else:
                prob = two_c / full_count
                raw_count = int(self._rng.binomial(full_count, prob))
                m_s = raw_count // 2 if is_middle else raw_count

            counts[s] = m_s

        return self._build_rows(n, counts, py_rng, paired=self.pairing_trick)

    def _bernoulli_sample_deterministic(
        self, n: int, c: float, m: int
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Deterministic analogue of ``_bernoulli_sample``'s Binomial draw.

        Fixes each size's pair count to the Binomial expectation, largest-remainder
        rounded so the total matches the target exactly, following the approach used
        by the paper's released implementation. Row construction is delegated to
        ``_build_rows``.

        Returns:
            Same shape/semantics as ``_bernoulli_sample``: (Z_pairs, sizes).
        """
        if n < 2 or c <= 0.0:
            return np.zeros((0, n), dtype=bool), np.zeros(0, dtype=int)

        target_pairs = (m - 2) // 2
        two_c = 2.0 * c
        half_sizes = list(range(1, n // 2 + 1))

        m_s: dict[int, int] = {}
        frac: dict[int, float] = {}
        pool_size_of: dict[int, int] = {}

        for s in half_sizes:
            is_middle = (n % 2 == 0) and (s == n // 2)
            full_count = math.comb(n, s)
            pool_size = math.comb(n - 1, s - 1) if is_middle else full_count
            pool_size_of[s] = pool_size

            if full_count <= two_c:
                m_s[s] = pool_size
                frac[s] = 0.0
            else:
                mu = two_c * s / n if is_middle else two_c
                floor_mu = int(mu)
                m_s[s] = min(floor_mu, pool_size)
                frac[s] = mu - floor_mu

        # Largest-remainder fill over non-full sizes until the target is met.
        shortfall = target_pairs - sum(m_s.values())
        fill_order = sorted(half_sizes, key=lambda s: -frac[s])
        while shortfall > 0:
            open_sizes = [s for s in fill_order if m_s[s] < pool_size_of[s]]
            if not open_sizes:
                break
            for s in open_sizes:
                if shortfall <= 0:
                    break
                m_s[s] += 1
                shortfall -= 1

        py_seed = int(self._rng.integers(0, 2**32))
        py_rng = _py_random.Random(py_seed)  # noqa: S311

        return self._build_rows(n, m_s, py_rng, paired=self.pairing_trick)

    def _build_rows(
        self,
        n: int,
        counts: dict[int, int],
        py_rng: _py_random.Random,
        *,
        paired: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Turn per-half-size pair counts into sampled coalition rows.

        If ``paired``, each drawn coalition's complement is appended next to it (the
        even-n middle size partitions its restricted pool ``C(n - 1, s - 1)`` by
        fixing one player). If not, sizes ``s`` and ``n - s`` are drawn independently
        -- the paper's unpaired ablation. Both modes sample without replacement.

        Returns:
            Z_pairs: Boolean coalition matrix.
            sizes: Cardinality of each row of ``Z_pairs``.
        """
        z_list: list[np.ndarray] = []
        sizes_list: list[int] = []

        for s, count in counts.items():
            if count == 0:
                continue
            is_middle = (n % 2 == 0) and (s == n // 2)

            if is_middle:
                if paired:
                    pool_size = math.comb(n - 1, s - 1)
                    indices = self._sample_without_replacement(pool_size, count, py_rng)
                    for idx in indices:
                        z_partial = self._combo(n - 1, s - 1, idx)
                        z = np.zeros(n, dtype=bool)
                        z[: n - 1] = z_partial
                        z[n - 1] = True
                        z_bar = ~z
                        z_list.append(z)
                        z_list.append(z_bar)
                        sizes_list.append(s)
                        sizes_list.append(s)
                else:
                    pool_size = math.comb(n, s)
                    indices = self._sample_without_replacement(pool_size, 2 * count, py_rng)
                    for idx in indices:
                        z_list.append(self._combo(n, s, idx))
                        sizes_list.append(s)
            else:
                pool_size = math.comb(n, s)
                indices = self._sample_without_replacement(pool_size, count, py_rng)
                if paired:
                    for idx in indices:
                        z = self._combo(n, s, idx)
                        z_bar = ~z
                        z_list.append(z)
                        z_list.append(z_bar)
                        sizes_list.append(s)
                        sizes_list.append(n - s)
                else:
                    for idx in indices:
                        z_list.append(self._combo(n, s, idx))
                        sizes_list.append(s)
                    # Independent draw for size n - s (same pool size C(n, s)).
                    indices_complement = self._sample_without_replacement(pool_size, count, py_rng)
                    for idx in indices_complement:
                        z_list.append(self._combo(n, n - s, idx))
                        sizes_list.append(n - s)

        if z_list:
            return np.array(z_list), np.array(sizes_list, dtype=int)
        return np.zeros((0, n), dtype=bool), np.zeros(0, dtype=int)

    @staticmethod
    def _sample_without_replacement(total: int, k: int, py_rng: _py_random.Random) -> list[int]:
        """Sample ``k`` distinct integers from ``[0, total)`` without replacement.

        ``total`` may be an arbitrary-precision Python int (for large ``n``).
        """
        if k >= total:
            return list(range(total))

        # range(total) beyond sys.maxsize crashes random.sample; there k << total, so
        # rejection sampling is collision-free in practice.
        if total > sys.maxsize:
            seen: set[int] = set()
            while len(seen) < k:
                seen.add(py_rng.randrange(total))
            return list(seen)

        return py_rng.sample(range(total), k)

    @staticmethod
    def _combo(n: int, s: int, i: int) -> np.ndarray:
        """Algorithm 3: ``i``-th lexicographic combination of size ``s`` from ``n`` items.

        Returns a boolean vector of length ``n`` with exactly ``s`` True entries.
        ``i`` is 0-indexed.
        """
        z = np.zeros(n, dtype=bool)
        if s == 0:
            return z
        k = s  # remaining slots to fill
        j = 0  # current position
        while k > 0 and j < n:
            # If i falls in the block of combinations that include position j, j is in
            # the combination; otherwise skip past the block.
            count = math.comb(n - j - 1, k - 1)
            if i < count:
                z[j] = True
                k -= 1
            else:
                i -= count
            j += 1
        return z
