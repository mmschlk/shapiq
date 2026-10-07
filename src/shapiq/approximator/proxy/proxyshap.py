"""ProxySHAP approximator class."""

from __future__ import annotations

import math
from functools import cached_property, reduce
from operator import add
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import gammaln

from shapiq.approximator.base import Approximator
from shapiq.approximator.proxy._models import (
    ProxyLiteral,
    ProxyModel,
    ProxyModelWithHPO,
    _select_base_proxy_via_string,
    _wrap_in_default_hpo,
)
from shapiq.approximator.proxy._routes import (
    ValidProxySHAPIndices,
    _extract_proxy_interactions,
    fit_proxy,
    predict_proxy,
)
from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import generate_interaction_lookup, log_binom

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.game import Game
    from shapiq.typing import CoalitionMatrix, FloatVector


def _log_discrete_derivative_weight(
    index: str, *, n: int, max_order: int, coalition_size: int, interaction_size: int
) -> float:
    r"""Natural log of the (non-negative) discrete-derivative weight of a computation index.

    Self-contained counterpart of the per-index ``_log_*_weight`` methods of the MonteCarlo
    approximators, restricted to the computation indices ProxySHAP's user-facing indices map to
    (``SII`` also covers ``k-SII``/``SV``, ``BII`` covers ``BV``; ``FSII``/``FBII`` are defined
    for top-order interactions only, ``STII`` weights lower orders with an indicator on
    ``T ⊆ S``). Computed in log-space so it stays finite for many players.

    Args:
        index: The computation index (``"SII"``, ``"BII"``, ``"STII"``, ``"FSII"``, or
            ``"FBII"``).
        n: The number of players.
        max_order: The maximum interaction order of the approximation.
        coalition_size: The size of the coalition *outside* the interaction, i.e. ``|T \ S|``.
        interaction_size: The size ``|S|`` of the interaction.

    Returns:
        The log of the discrete-derivative weight.

    Raises:
        ValueError: If the index is not supported by the MSR residual adjustment.
    """
    if index == "SII":
        return float(
            -np.log(n - interaction_size + 1) - log_binom(n - interaction_size, coalition_size)
        )
    if index == "BII":
        return float(-(n - interaction_size) * np.log(2))
    if index == "STII":
        if interaction_size == max_order:
            return float(np.log(max_order) - np.log(n) - log_binom(n - 1, coalition_size))
        # Lower orders are weighted by an indicator on ``T ⊆ S`` (i.e. ``|T \ S| == 0``).
        return 0.0 if coalition_size == 0 else -np.inf
    if index == "FSII" and interaction_size == max_order:
        return float(
            gammaln(2 * max_order)
            - 2 * gammaln(max_order)
            + gammaln(n - coalition_size)
            + gammaln(coalition_size + max_order)
            - gammaln(n + max_order)
        )
    if index == "FBII" and interaction_size == max_order:
        return float(-(n - interaction_size) * np.log(2))
    msg = f"The computation index {index} is not supported by the MSR residual adjustment."
    raise ValueError(msg)


def _standard_form_log_weights(
    index: str, *, n: int, min_order: int, max_order: int
) -> tuple[np.ndarray, np.ndarray]:
    r"""Sign and log-magnitude of the standard form weights, stable for large ``n``.

    The interaction index is re-written from discrete derivatives to standard form (Theorem 1 of
    `Fumagalli et al. (2023) <https://doi.org/10.48550/arXiv.2303.01179>`_): the weight of a
    coalition ``T`` for an interaction ``S`` is ``(-1) ** (|S| - |T ∩ S|) * w(|T \ S|, |S|)``.
    The non-negative magnitude is kept in log-space, with the sign tracked separately, so the
    caller can cancel it against the (log) sampling-adjustment weight before exponentiating.

    Args:
        index: The computation index (see :func:`_log_discrete_derivative_weight`).
        n: The number of players.
        min_order: The minimum interaction order of the approximation.
        max_order: The maximum interaction order of the approximation.

    Returns:
        A tuple ``(sign_weights, log_abs_weights)`` of arrays of shape
        ``(max_order + 1, n + 1, max_order + 1)`` indexed by interaction order, coalition size,
        and intersection size. Unfilled entries have sign ``0`` and log ``-inf`` (i.e. weight
        ``0``).
    """
    shape = (max_order + 1, n + 1, max_order + 1)
    sign_weights = np.zeros(shape)
    log_abs_weights = np.full(shape, -np.inf)
    for order in range(min_order, max_order + 1):
        for coalition_size in range(n + 1):
            for intersection_size in range(
                max(0, order + coalition_size - n),
                min(order, coalition_size) + 1,
            ):
                sign_weights[order, coalition_size, intersection_size] = (-1) ** (
                    order - intersection_size
                )
                log_abs_weights[order, coalition_size, intersection_size] = (
                    _log_discrete_derivative_weight(
                        index,
                        n=n,
                        max_order=max_order,
                        coalition_size=coalition_size - intersection_size,
                        interaction_size=order,
                    )
                )
    return sign_weights, log_abs_weights


def _interaction_members(
    interactions: list[tuple[int, ...]], n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sizes and binary membership matrix of the given interactions, in order.

    Args:
        interactions: The interactions to encode.
        n: The number of players.

    Returns:
        A tuple ``(sizes, binary)`` where ``sizes`` has shape ``(n_interactions,)`` and
        ``binary`` is the ``(n_interactions, n)`` one-hot matrix of interaction members.
    """
    n_interactions = len(interactions)
    sizes = np.fromiter((len(it) for it in interactions), dtype=np.int64, count=n_interactions)
    binary = np.zeros((n_interactions, n), dtype=np.int64)
    row_index = np.repeat(np.arange(n_interactions), sizes)
    col_index = np.fromiter(
        (player for interaction in interactions for player in interaction),
        dtype=np.int64,
        count=int(sizes.sum()),
    )
    binary[row_index, col_index] = 1
    return sizes, binary


class ProxySHAP(Approximator[ValidProxySHAPIndices]):
    """ProxySHAP is a proxy-based approximator that uses a regression model to approximate the value function and can correct the proxy's error with an MSR residual adjustment.

    The regression proxy is trained on the sampled coalitions, and interaction values are read out
    of the fitted model exactly. Optionally (``adjustment=True``), the proxy's residuals
    (true game values minus proxy predictions) are estimated with a self-contained, fully
    vectorized MSR (maximum sample reuse) Monte Carlo routine ( unstratified SHAP-IQ :cite:t:`Fumagalli.2023`).
    Depending on ``k_folds`` the adjustment is computed in-sample (``k_folds=1``) or out-of-fold (``k_folds>1``) on the same sampled coalitions, and added to the proxy's interactions.

    Example:
        >>> from shapiq_games.synthetic import DummyGame
        >>> from shapiq.approximator import ProxySHAP
        >>> game = DummyGame(n=5, interaction=(1, 2))
        >>> approximator = ProxySHAP(n=5, max_order=2, index="k-SII")
        >>> approximator.approximate(budget=100, game=game)
        InteractionValues(
            index=k-SII, max_order=2, min_order=0, estimated=False, estimation_budget=32,
            n_players=5, baseline_value=0.0
        )
    """

    def __init__(
        self,
        n: int,
        *,
        max_order: int = 2,
        index: ValidProxySHAPIndices = "k-SII",
        proxy_model: ProxyModel | ProxyModelWithHPO | ProxyLiteral = "xgboost",
        hpo: bool = False,
        adjustment: bool = False,
        k_folds: int = 1,
        sampling_weights: FloatVector | None = None,
        pairing_trick: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initialize the ProxySHAP approximator.

        Args:
            n: Number of features (players).
            max_order: Maximum order of interactions to consider.
            index: Index of the instance to explain.
            proxy_model: Optional proxy model to use for approximating the value function. If None, a default XGBoost regressor will be used.
                We support HPO of tree-models, via sklearn's GridSearchCV, RandomizedSearchCV, and HalvingGridSearchCV. In this case, the ``.best_estimator_`` will be used as the proxy model for interaction extraction and residual adjustment.
            hpo: If ``True``, wrap a string-resolved gradient-boosting proxy (``"xgboost"`` /
                ``"lightgbm"``) in its default grid search (the HPO-informed proxy). Defaults to
                ``False`` (a bare estimator). Has no effect when ``proxy_model`` is a passed-in
                estimator/wrapper, or for the ``"tree"`` / ``"linear"`` tags.
            adjustment: If ``True``, the MSR residual adjustment is applied to the proxy's
                interactions, covering the complete interaction lattice up to ``max_order``.
                Defaults to ``False`` (no adjustment). Note the lattice grows as
                ``O(n**max_order)``, so the adjustment is infeasible for high orders on
                high-dimensional games; extraction-only runs (the default) are not affected.
                For ``FSII``/``FBII`` only the top order is corrected (see ``top_order`` below), so
                their lower orders are returned as the uncorrected proxy readout.
            k_folds: Number of folds the sampled coalitions are split into. With
                the default ``1``, a single proxy is fit on all sampled coalitions and its
                residuals are computed in-sample. For values ``> 1`` (cross-fitting), the sampled
                coalitions are split into folds within each size; one proxy is fit per fold on
                all coalitions but its held-out part, and its residuals are used on all sampled
                coalitions: training coalitions count only for themselves, held-out ones stand in
                for all coalitions of their size the proxy never saw. This keeps the correction
                unbiased for any proxy and exact at full budget; the per-fold results are
                averaged.
            sampling_weights: Optional array of weights for the sampling procedure. The weights must be of shape (n + 1,) and are used to determine the probability of sampling a coalition. Defaults to None.
                `None` means uniform sampling by size and uniform within each size.
            pairing_trick: If True, the pairing trick is applied to the sampling procedure. Defaults to True.
            random_state: The random state of the estimator. Defaults to None.
        """
        if sampling_weights is None:
            # Sample uniformly by size and uniformly within each size as default.
            sampling_weights = np.ones(n + 1, dtype=np.float64)
        super().__init__(
            n=n,
            min_order=0,
            max_order=max_order,
            index=index,
            # FSII/FBII discrete-derivative weights exist only at top order, so the adjustment
            # lattice is restricted there and their lower orders stay uncorrected.
            top_order=index in ("FSII", "FBII") and adjustment,
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
            random_state=random_state,
            # The interaction lookup is only needed for the MSR adjustment, which is optional. Will be generated on first use if needed.
            initialize_dict=False,
        )
        self.adjustment = adjustment
        self.k_folds = k_folds
        if isinstance(proxy_model, ProxyModel):
            self.proxy_model: ProxyModel | ProxyModelWithHPO = proxy_model
        else:
            resolved = _select_base_proxy_via_string(proxy_model, random_state)
            # ``hpo`` wraps a resolved boosting backend in its default grid search (the
            # HPO-informed proxy); a DecisionTree fallback is left unwrapped by the helper.
            self.proxy_model = _wrap_in_default_hpo(resolved) if hpo else resolved

    @cached_property
    def _msr_weight_tables(self) -> tuple[np.ndarray, np.ndarray]:
        """Sign and log-magnitude standard-form weight tables for the computation index."""
        return _standard_form_log_weights(
            self.approximation_index, n=self.n, min_order=self.min_order, max_order=self.max_order
        )

    def _lazy_interaction_lookup(self) -> dict[tuple[int, ...], int]:
        """Return the full interaction lookup, generating and caching it on first use.

        ``__init__`` defers the lattice via ``initialize_dict=False`` since only the MSR
        adjustment needs it; extraction-only runs never pay its ``O(n**max_order)`` cost.

        Returns:
            The interaction lookup the residual estimate is aligned with.
        """
        if not self._interaction_lookup:
            self._interaction_lookup = generate_interaction_lookup(
                self.n, self.min_order, self.max_order
            )
        return self._interaction_lookup

    def _msr_routine(
        self,
        residuals: np.ndarray,
        coalition_indices: np.ndarray,
        coalitions_matrix: CoalitionMatrix,
        interaction_lookup: dict[tuple[int, ...], int],
        log_coalition_weights: np.ndarray,
    ) -> np.ndarray:
        """Vectorized MSR (unstratified SHAP-IQ) estimate of all interactions at once.

        The estimator is the standard form of :cite:t:`Fumagalli.2023` without stratification,
        which makes the sampling-adjustment weight interaction-independent. This allows estimating
        *all* interactions in a single matrix product instead of the per-interaction loop of the
        generic MonteCarlo routine.

        The per-coalition weight says for how many coalitions of the population each residual
        counts: the sampler's inverse inclusion probability for the in-sample estimate, or the
        cross-fitting weights of :meth:`_cross_fitting_log_weights` for a fold.

        Args:
            residuals: Residual values for the selected coalitions, of shape ``(m,)``, normalized
                to ``0`` at the empty coalition.
            coalition_indices: Row indices into ``coalitions_matrix`` the residuals belong to, of
                shape ``(m,)``.
            coalitions_matrix: The full binary coalition matrix of shape ``(n_coalitions, n)``.
            interaction_lookup: The interactions to estimate. Any subset of the lattice works, but
                its values must be the interactions' positions in iteration order (as
                :func:`~shapiq.utils.sets.generate_interaction_lookup` returns), since the result
                is filled positionally and read back by lookup value.
            log_coalition_weights: Log of the per-coalition weight of each residual, of shape
                ``(m,)``.

        Returns:
            The estimated interaction values as an array aligned with ``interaction_lookup``.
        """
        sign_table, log_abs_table = self._msr_weight_tables
        interactions = list(interaction_lookup)

        # float64 so the intersection matrix product below runs on BLAS (numpy computes integer
        # matmuls without it, an order of magnitude slower); the products/sums are small integers,
        # exact in float64.
        coalitions = coalitions_matrix[coalition_indices].astype(np.float64)
        coalition_sizes = coalitions.sum(axis=1).astype(np.int64)[:, None]

        # Process the interactions in blocks so the work buffers stay bounded (~1 GB each, and the loop body holds about five of them) however large the lattice grows.
        chunk_size = max(1, min(2**27 // max(len(coalition_indices), 1), 2**27 // self.n))
        estimates = np.empty(len(interactions))
        for start in range(0, len(interactions), chunk_size):
            block = interactions[start : start + chunk_size]
            interaction_sizes, interaction_binary = _interaction_members(block, self.n)
            # (m, block) matrix of intersection sizes |T ∩ S| in one matrix product. We do type conversion, to float64, so the matmul runs on BLAS (numpy computes integer matmuls without it, an order of magnitude slower).
            intersection_sizes = (coalitions @ interaction_binary.T.astype(np.float64)).astype(
                np.int64
            )

            # Gather the standard-form weights for every (coalition, interaction) pair and
            # contract the residuals:
            # estimate_S = sum_T r_T * sign(S,T) * exp(log|w|(S,T) + log_weight_T).
            signs = sign_table[interaction_sizes[None, :], coalition_sizes, intersection_sizes]
            log_weights = log_abs_table[
                interaction_sizes[None, :], coalition_sizes, intersection_sizes
            ]
            estimates[start : start + len(block)] = residuals @ (
                signs * np.exp(log_weights + log_coalition_weights[:, None])
            )

        if () in interaction_lookup:
            # The empty interaction is the residual game's baseline, which is 0 by normalization.
            estimates[interaction_lookup[()]] = 0.0
        return estimates

    def _cross_fitting_folds(
        self, coalitions_matrix: CoalitionMatrix
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        r"""Split the sampled coalitions into folds with conditionally unbiased residual weights.

        Fold ``l``'s proxy is trained on all sampled coalitions except its held-out part, and its
        residuals are used on *all* sampled coalitions. The weight says for how many coalitions of
        the population each residual counts:

        - a training coalition counts only for itself (weight ``1``).
        - a held-out coalition of size ``s`` stands in for all coalitions of that size the proxy
          never saw, i.e. weight ``(N_s - tr_s) / t_s`` with ``N_s = binom(n, s)``, ``tr_s`` the
          fold's training and ``t_s`` its held-out coalitions of size ``s``.

        The held-out part is drawn uniformly within each size, and the sampler draws uniformly
        within each size, so given the training set the held-out coalitions are a uniform subset
        of the unseen ones. The correction is therefore unbiased conditionally on the fitted
        proxy, however closely it fits its training coalitions. A fully enumerated size gets weight
        ``1`` everywhere, which keeps the estimate exact once the sampler enumerates. Under the
        pairing trick a coalition and its complement form one unit and are held out together, so
        the training set does not reveal which unseen coalitions were sampled.

        Args:
            coalitions_matrix: The binary coalition matrix of the sampled coalitions.

        Returns:
            One ``(train_index, residual_index, log_weights)`` tuple per fold: the rows the proxy
            is trained on, the rows its residuals are used on, and the log weight of each of them.
        """
        n_players = self.n
        sizes = coalitions_matrix.sum(axis=1).astype(np.int64)
        n_rows = len(sizes)
        n_sampled_per_size = np.bincount(sizes, minlength=n_players + 1)
        # math.comb is exact for any n, and math.log accepts arbitrarily large ints, so the
        # population sizes never overflow.
        population_sizes = [math.comb(n_players, size) for size in range(n_players + 1)]

        # Units of the split: a coalition, or a coalition together with its sampled complement.
        unit_of_row = np.arange(n_rows)
        stratum_of_row = sizes
        if self._sampler.pairing_trick:
            is_member = coalitions_matrix.astype(bool)
            row_of_coalition = {row.tobytes(): i for i, row in enumerate(is_member)}
            for i, row in enumerate(is_member):
                j = row_of_coalition.get((~row).tobytes())
                if j is not None:
                    unit_of_row[i] = min(i, j)
            stratum_of_row = np.minimum(sizes, n_players - sizes)

        # Split the units of every stratum into k parts. A fold whose part is empty holds out one
        # random unit instead, so the unseen coalitions of that size stay represented -- unless
        # the stratum is fully enumerated, in which case nothing is unseen.
        rng = np.random.default_rng(self._random_state)
        held_out_units: list[list[int]] = [[] for _ in range(self.k_folds)]
        for stratum in np.unique(stratum_of_row):
            in_stratum = stratum_of_row == stratum
            units = np.unique(unit_of_row[in_stratum])
            is_enumerated = all(
                n_sampled_per_size[size] == population_sizes[size]
                for size in np.unique(sizes[in_stratum])
            )
            parts = np.array_split(rng.permutation(units), self.k_folds)
            for part, fold in zip(parts, rng.permutation(self.k_folds), strict=True):
                if len(part) == 0:
                    if is_enumerated:
                        continue
                    part = rng.choice(units, size=1)  # noqa: PLW2901
                held_out_units[fold].extend(part.tolist())

        folds = []
        for units in held_out_units:
            is_held_out = np.isin(unit_of_row, units)
            train_index = np.flatnonzero(~is_held_out)
            test_index = np.flatnonzero(is_held_out)
            n_held_out_per_size = np.bincount(sizes[test_index], minlength=n_players + 1)
            log_weight_per_size = np.zeros(n_players + 1)
            for size in np.flatnonzero(n_held_out_per_size):
                n_unseen = population_sizes[size] - (
                    int(n_sampled_per_size[size])
                    - int(
                        n_held_out_per_size[size]
                    )  # Remove training coalitions of that size from the population count
                )
                log_weight_per_size[size] = math.log(n_unseen) - math.log(
                    int(n_held_out_per_size[size])
                )
            folds.append(
                (
                    train_index,
                    np.concatenate((train_index, test_index)),
                    np.concatenate(
                        (
                            np.zeros(len(train_index)),
                            log_weight_per_size[sizes[test_index]],
                        )  # log-weights: log(0) for training coalitions, log((N_s - tr_s) / t_s) for held-out ones
                    ),
                )
            )
        return folds

    def _apply_msr_adjustment(
        self,
        residuals: np.ndarray,
        coalition_indices: np.ndarray,
        coalitions_matrix: CoalitionMatrix,
        proxy_interactions: InteractionValues,
        log_coalition_weights: np.ndarray,
    ) -> InteractionValues:
        """Apply the MSR residual adjustment to the proxy's interactions.

        Args:
            residuals: Residual values for the selected coalitions, normalized to ``0`` at the
                empty coalition.
            coalition_indices: Row indices into ``coalitions_matrix`` the residuals belong to.
            coalitions_matrix: The full binary coalition matrix.
            proxy_interactions: The interactions extracted from the fitted proxy.
            log_coalition_weights: Log of the per-coalition weight of each residual (see
                :meth:`_msr_routine`).

        Returns:
            The proxy interactions with the estimated residual interactions added.
        """
        n_samples = coalitions_matrix.shape[0]
        interaction_lookup = self._lazy_interaction_lookup()
        residual_adjustment = self._msr_routine(
            residuals,
            coalition_indices,
            coalitions_matrix,
            interaction_lookup,
            log_coalition_weights,
        )
        return proxy_interactions + InteractionValues(
            residual_adjustment,
            index=self.approximation_index,
            n_players=self.n,
            interaction_lookup=interaction_lookup,
            min_order=self.min_order,
            max_order=self.max_order,
            baseline_value=0.0,  # residuals are normalized to 0 at the empty coalition
            estimated=n_samples < 2**self.n,
            estimation_budget=n_samples,
            target_index=self.index,
        )

    def approximate(
        self,
        budget: int,
        game: Game | Callable[[np.ndarray], np.ndarray],
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Approximate interaction values, dispatching on the proxy's base estimator type.

        The proxy is fit by :func:`fit_proxy` (which selects the feature transform from the base
        estimator type and unwraps any HPO wrapper). Interactions are then read out of the *fitted*
        model by :func:`_extract_proxy_interactions`, which dispatches on its type: linear models
        route to :func:`_extract_linear`, registered tree models to :func:`_extract_tree`. If
        enabled, the proxy's residuals are estimated with the vectorized MSR routine.
        Depending on ``k_folds``, the residuals are corrected either in-sample (``k_folds=1``)
        or cross-fitted (``k_folds>1``, see :meth:`_cross_fitting_folds`) and added to the
        proxy's interactions.
        For ``k_folds>1``, the final interaction values are the average of the per-fold results, and the baseline is fixed to the empty-coalition value of the game.

        Args:
            budget: Number of coalition evaluations to draw.
            game: Coalition game (a :class:`shapiq.game.Game` or any callable
                accepting a binary coalition matrix and returning game values).
            **kwargs: Ignored; present for interface compatibility.

        Returns:
            :class:`~shapiq.interaction_values.InteractionValues` for orders 0
            through ``self.max_order``.
        """
        # 1. Sample coalitions and evaluate the game once; the proxy fit and the MSR residual
        # adjustment both reuse these evaluations.
        self._sampler.sample(int(budget))
        coalitions_matrix = self._sampler.coalitions_matrix
        empty_index = self._sampler.empty_coalition_index
        game_values = game(coalitions_matrix)
        baseline_value = float(game_values[empty_index])
        coalition_values = game_values - baseline_value
        n_samples, n_players = coalitions_matrix.shape

        # 2. Split the coalitions into folds. With a single fold the proxy trains on all coalitions
        # and its residuals carry the sampler's weights; with cross-fitting each fold's proxy
        # corrects with the weights of ``_cross_fitting_folds``.
        if self.k_folds > 1:
            folds = self._cross_fitting_folds(coalitions_matrix)
        else:
            all_index = np.arange(n_samples)
            folds = [(all_index, all_index, self._sampler.log_sampling_adjustment_weights)]

        # 3. Per fold: fit the proxy, read interactions out of the fitted model (dispatch on its
        # type), and record its weighted residuals on all sampled coalitions (every fold's
        # residual set is a permutation of all rows, stored here in row order).
        fold_results: list[InteractionValues] = []
        fold_residuals = np.zeros((len(folds), n_samples))
        fold_log_weights = np.full((len(folds), n_samples), -np.inf)
        for fold, (train_index, residual_index, log_coalition_weights) in enumerate(folds):
            fitted = fit_proxy(
                self.proxy_model,
                coalitions_matrix[train_index],
                coalition_values[train_index],
                max_order=self.max_order,
            )
            fold_results.append(
                _extract_proxy_interactions(
                    fitted,
                    baseline_value=baseline_value,
                    max_order=self.max_order,
                    approximation_index=self.approximation_index,
                    target_index=self.index,
                    budget=n_samples,
                    n_players=n_players,
                )
            )
            if self.adjustment:
                # Normalize the residuals to 0 at the empty coalition (the *centered* game value
                # there is 0, so its residual is subtracted from all others).
                predictions = predict_proxy(fitted, coalitions_matrix, max_order=self.max_order)
                residuals = coalition_values - predictions
                residuals -= residuals[empty_index]
                fold_residuals[fold, residual_index] = residuals[residual_index]
                fold_log_weights[fold, residual_index] = log_coalition_weights

        # 4. Average the fold results and fix the empty-coalition/baseline value.
        proxy_interactions = reduce(add, fold_results) * (1.0 / len(fold_results))
        if self.adjustment:
            # Apply the MSR residual adjustment to the combined residuals of all folds, with the log weights of each fold's residuals.
            # The log weights are shifted by their max to avoid overflow.
            max_log_val = fold_log_weights.max(axis=0)
            combined_residuals = (fold_residuals * np.exp(fold_log_weights - max_log_val)).sum(
                axis=0
            ) / len(folds)
            proxy_interactions = self._apply_msr_adjustment(
                combined_residuals,
                np.arange(n_samples),
                coalitions_matrix,
                proxy_interactions,
                max_log_val,
            )
        proxy_interactions.baseline_value = baseline_value
        proxy_interactions.interactions[()] = baseline_value  # Ensure empty coalition is correct
        return proxy_interactions
