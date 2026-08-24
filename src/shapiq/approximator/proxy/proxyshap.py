"""ProxySHAP approximator class."""

from __future__ import annotations

from functools import cached_property, reduce
from operator import add
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import gammaln
from sklearn.model_selection import KFold

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
    of the fitted model exactly. Optionally (``apply_msr_adjustment=True``), the proxy's residuals
    (true game values minus proxy predictions) are estimated with a self-contained, fully
    vectorized MSR (maximum sample reuse) Monte Carlo routine ( unstratified SHAP-IQ :cite:t:`Fumagalli.2023`).
    Depending on `kfold` the adjustment is computed in-sample (``kfolds=1``) or out-of-fold (``kfolds>1``) on the same sampled coalitions, and added to the proxy's interactions.

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
        kfolds: int = 1,
        apply_msr_adjustment: bool = False,
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
            kfolds: Number of folds the sampled coalitions are split into. With
                the default ``1``, a single proxy is fit on all sampled coalitions and its
                residuals are computed in-sample. For values ``> 1``, one proxy is fit per fold
                (KFold) on the training split, its interactions are extracted, and its residuals
                are computed on the held-out split only; the per-fold results are averaged.
            apply_msr_adjustment: If ``True``, the MSR residual adjustment is applied to the proxy's
                interactions, covering the complete interaction lattice up to ``max_order``.
                Defaults to ``False`` (no adjustment). Note the lattice grows as
                ``O(n**max_order)``, so the adjustment is infeasible for high orders on
                high-dimensional games; extraction-only runs (the default) are not affected.
                For ``FSII``/``FBII`` only the top order is corrected (see ``top_order`` below), so
                their lower orders are returned as the uncorrected proxy readout.
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
            top_order=index in ("FSII", "FBII") and apply_msr_adjustment,
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
            random_state=random_state,
            # The interaction lookup is only needed for the MSR adjustment, which is optional. Will be generated on first use if needed.
            initialize_dict=False,
        )
        self.kfolds = kfolds
        self.apply_msr_adjustment = apply_msr_adjustment
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
    ) -> np.ndarray:
        """Vectorized MSR (unstratified SHAP-IQ) estimate of all interactions at once.

        The estimator is the standard form of :cite:t:`Fumagalli.2023` without stratification,
        which makes the sampling-adjustment weight interaction-independent. This allows estimating
        *all* interactions in a single matrix product instead of the per-interaction loop of the
        generic MonteCarlo routine.

        When only a subset of the sampled coalitions carries residuals (a held-out
        cross-validation fold), the estimate is rescaled by the inverse inclusion probability
        ``n_coalitions / len(coalition_indices)`` (Horvitz-Thompson). Since the folds partition
        the coalitions, the fold-average of these subset estimates is exactly the full-sample MSR
        estimate of the assembled out-of-fold residuals.

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

        # Per-coalition sampling adjustment, plus the Horvitz-Thompson factor for fold subsets.
        log_adjustment = self._sampler.log_sampling_adjustment_weights[coalition_indices] + np.log(
            coalitions_matrix.shape[0] / len(coalition_indices)
        )

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
            # estimate_S = sum_T r_T * sign(S,T) * exp(log|w|(S,T) + log_adj_T).
            signs = sign_table[interaction_sizes[None, :], coalition_sizes, intersection_sizes]
            log_weights = log_abs_table[
                interaction_sizes[None, :], coalition_sizes, intersection_sizes
            ]
            estimates[start : start + len(block)] = residuals @ (
                signs * np.exp(log_weights + log_adjustment[:, None])
            )

        if () in interaction_lookup:
            # The empty interaction is the residual game's baseline, which is 0 by normalization.
            estimates[interaction_lookup[()]] = 0.0
        return estimates

    def _apply_msr_adjustment(
        self,
        residuals: np.ndarray,
        coalition_indices: np.ndarray,
        coalitions_matrix: CoalitionMatrix,
        proxy_interactions: InteractionValues,
    ) -> InteractionValues:
        """Apply the MSR residual adjustment to the proxy's interactions.

        Args:
            residuals: Residual values for the selected coalitions, normalized to ``0`` at the
                empty coalition.
            coalition_indices: Row indices into ``coalitions_matrix`` the residuals belong to.
            coalitions_matrix: The full binary coalition matrix.
            proxy_interactions: The interactions extracted from the fitted proxy.

        Returns:
            The proxy interactions with the estimated residual interactions added.
        """
        n_samples = coalitions_matrix.shape[0]
        interaction_lookup = self._lazy_interaction_lookup()
        residual_adjustment = self._msr_routine(
            residuals, coalition_indices, coalitions_matrix, interaction_lookup
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
        Depending on ``kfolds``, the residuals are corrected either in-sample (``kfolds=1``)
        or out-of-fold (``kfolds>1``) and added to the proxy's interactions.
        For ``kfolds>1``, the final interaction values are the average of the per-fold results, and the baseline is fixed to the empty-coalition value of the game.

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

        # 2. Split the coalitions into folds. With a single fold the proxy trains and predicts on
        # all coalitions; with cross-validation each fold's proxy predicts its held-out split.
        if self.kfolds > 1:
            folder = KFold(
                n_splits=self.kfolds,
                shuffle=True,
                random_state=self._random_state,
            )
            folds = list(folder.split(coalitions_matrix, coalition_values))
        else:
            folds = [(np.arange(n_samples), np.arange(n_samples))]  # single fold, train on all

        # 3. Per fold: fit the proxy, read interactions out of the fitted model (dispatch on its
        # type), and optionally add the MSR estimate of the fold's held-out residuals.
        fold_results: list[InteractionValues] = []
        for train_index, test_index in folds:
            fitted = fit_proxy(
                self.proxy_model,
                coalitions_matrix[train_index],
                coalition_values[train_index],
                max_order=self.max_order,
            )
            fold_interactions = _extract_proxy_interactions(
                fitted,
                baseline_value=baseline_value,
                max_order=self.max_order,
                approximation_index=self.approximation_index,
                target_index=self.index,
                budget=n_samples,
                n_players=n_players,
            )
            if self.apply_msr_adjustment:
                # Normalize the residuals to 0 at the empty coalition (the *centered* game value
                # there is 0, and the empty coalition may not be part of the held-out split, so
                # its residual is computed explicitly).
                empty_residual = (
                    coalition_values[empty_index]
                    - predict_proxy(
                        fitted,
                        coalitions_matrix[empty_index].reshape(1, -1),
                        max_order=self.max_order,
                    )[0]
                )
                residuals = coalition_values[test_index] - predict_proxy(
                    fitted, coalitions_matrix[test_index], max_order=self.max_order
                )
                residuals -= empty_residual
                fold_interactions = self._apply_msr_adjustment(
                    residuals, test_index, coalitions_matrix, fold_interactions
                )
            fold_results.append(fold_interactions)

        # 4. Average the fold results and fix the empty-coalition/baseline value.
        proxy_interactions = reduce(add, fold_results) * (1.0 / len(fold_results))
        proxy_interactions.baseline_value = baseline_value
        proxy_interactions.interactions[()] = baseline_value  # Ensure empty coalition is correct
        return proxy_interactions
