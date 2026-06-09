"""ProxySPEX approximator for sparse higher-order interactions."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV

from shapiq.approximator.base import Approximator
from shapiq.approximator.proxy._models import (
    ProxyLiteral,
    ProxyModel,
    ProxyModelWithHPO,
    _select_base_proxy_via_string,
)
from shapiq.game_theory.moebius_converter import MoebiusConverter, ValidMoebiusConverterIndices
from shapiq.interaction_values import InteractionValues
from shapiq.tree.conversion import convert_tree_model
from shapiq.utils.modules import safe_isinstance
from shapiq.utils.sets import powerset

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.game import Game
    from shapiq.tree.base import TreeModel
    from shapiq.typing import Model


ValidProxySPEXIndices = ValidMoebiusConverterIndices

# Hyperparameter grid for ProxySPEX's reference HPO-informed LightGBM proxy (Butler et al., 2025).
_LIGHTGBM_DECODER_GRID = {
    "max_depth": [3, 5],
    "max_iter": [500, 1000],
    "learning_rate": [0.01, 0.1],
}


class ProxySPEX(Approximator[ValidProxySPEXIndices]):
    """ProxySPEX (SParse EXplainer) via Fourier transform sampling.

    An approximator for cardinal interaction indices using Fourier transform sampling to efficiently
    compute sparse higher-order interactions. ProxySPEX is presented in :cite:t:`Butler.2025`.
    """

    def __init__(
        self,
        *,
        n: int,
        max_order: int = 2,
        index: ValidProxySPEXIndices = "k-SII",
        proxy_model: Model | ProxyLiteral | ProxyModelWithHPO = "lightgbm",
        sampling_weights: np.ndarray | None = None,
        pairing_trick: bool = False,
        top_order: bool = False,
        random_state: int | None = None,
    ) -> None:
        """Initialize the ProxySPEX approximator.

        Args:
            n: Number of players (features).

            max_order: Maximum interaction order to consider.

            index: The Interaction index to use. All indices supported by shapiq's
                :class:`~shapiq.game_theory.moebius_converter.MoebiusConverter` are supported.

            top_order: If ``True``, only reports interactions of exactly order ``max_order``.
                Otherwise, reports all interactions up to order ``max_order``. Defaults to
                ``False``.

            pairing_trick: If `True`, the pairing trick is applied to the sampling procedure.
                Defaults to ``False``.

            sampling_weights: An optional array of weights for the sampling procedure. The weights
                must be of shape ``(n + 1,)`` and are used to determine the probability of sampling
                a coalition of a certain size. Defaults to ``None``.

            proxy_model: Proxy model used to approximate the value function. ProxySPEX reads
                interactions off the proxy's tree structure, so only **tree** proxies are
                supported. May be:

                * a string identifier (``"lightgbm"`` (default), ``"xgboost"``, ``"tree"``)
                  selecting a tree estimator; ``"linear"`` is rejected since ProxySPEX is
                  tree-only. The ``"lightgbm"`` tag yields the reference HPO-informed LightGBM
                  proxy -- the selected ``LGBMRegressor`` wrapped in a grid search over
                  :data:`_LIGHTGBM_DECODER_GRID`, as described in :cite:t:`Butler.2025`. The
                  other tags select a bare estimator. ProxySPEX does not require the optional
                  gradient-boosting backends: if the requested package is unavailable the tag
                  warns and falls back to a bare scikit-learn ``DecisionTreeRegressor`` (which
                  is therefore *not* wrapped in the LightGBM grid search).
                * a fitted-on-call estimator implementing the scikit-learn regressor interface,
                  or a hyperparameter-search wrapper exposing ``best_estimator_`` (e.g.
                  :class:`~sklearn.model_selection.GridSearchCV` or the
                  :class:`~shapiq.approximator.proxy._models.ProxyModelWithHPO` wrappers).

            random_state: Seed for random number generator. Defaults to ``None``.


        """
        if sampling_weights is None:
            sampling_weights = np.array([math.comb(n, i) for i in range(n + 1)], dtype=float)
        if isinstance(proxy_model, str):
            if proxy_model == "linear":
                msg = "ProxySPEX only supports tree-based proxy models; 'linear' is not available."
                raise ValueError(msg)
            self.proxy_model = _select_base_proxy_via_string(proxy_model, random_state)
            if proxy_model == "lightgbm" and safe_isinstance(
                self.proxy_model, "lightgbm.LGBMRegressor"
            ):
                self.proxy_model: ProxyModel | ProxyModelWithHPO = GridSearchCV(
                    estimator=self.proxy_model,
                    param_grid=_LIGHTGBM_DECODER_GRID,
                    scoring="r2",
                    cv=5,
                    verbose=0,
                    n_jobs=1,
                )
        else:
            self.proxy_model: ProxyModel | ProxyModelWithHPO = proxy_model
        super().__init__(
            n=n,
            max_order=max_order,
            index=index,
            top_order=top_order,
            pairing_trick=pairing_trick,
            random_state=random_state,
            sampling_weights=sampling_weights,
            initialize_dict=False,
        )

    def approximate(
        self,
        budget: int,
        game: Game | Callable[[np.ndarray], np.ndarray],
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Approximates the interaction values using a sparse transform approach.

        Args:
            budget: The budget for the approximation.
            game: The game function that returns the values for the coalitions.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            The approximated Shapley interaction values.
        """
        # Take the budget amount of uniform samples
        self._sampler.sample(budget)

        coalitions_matrix = self._sampler.coalitions_matrix
        coalition_values = game(coalitions_matrix)

        # Fit the model on the training data
        self.proxy_model.fit(coalitions_matrix, coalition_values)

        if isinstance(self.proxy_model, ProxyModelWithHPO):
            final_model = self.proxy_model.best_estimator_
        else:
            final_model = self.proxy_model
        # Obtain TreeModel(s). convert_tree_model returns a single TreeModel for single-tree
        # proxies (e.g. a DecisionTreeRegressor) and a list for ensembles; normalize to a list.
        tree_models = convert_tree_model(final_model)
        if not isinstance(tree_models, list):
            tree_models = [tree_models]
        # Obtain fourier coefficients
        unrefined_fourier = self._sklearn_to_fourier(tree_models=tree_models)
        # Refine the Fourier coefficients using the training data
        refined_fourier = self._refine(
            unrefined_fourier,
            coalitions_matrix,
            coalition_values,
        )
        # Convert the Fourier coefficients to the Moebius transform
        moebius_transform = self.fourier_to_moebius(refined_fourier)
        # Convert the Moebius transform to the desired index
        result = self._process_moebius(moebius_transform=moebius_transform)
        # Filter the output as needed
        if self.top_order:
            result = self._filter_order(result)
        # finalize the interactions
        return InteractionValues(
            values=result,
            index=self.approximation_index,
            min_order=self.min_order,
            max_order=self.max_order,
            n_players=self.n,
            interaction_lookup=self.interaction_lookup,
            estimated=True,
            estimation_budget=budget,
            baseline_value=result[self.interaction_lookup[()]]
            if () in self.interaction_lookup
            else 0.0,
            target_index=self.index,
        )

    def fourier_to_moebius(
        self, four_dict: dict[tuple[int, ...], float]
    ) -> dict[tuple[int, ...], float]:
        """Converts a Fourier representation of a function to its Moebius representation."""
        moebius_dict = defaultdict(float)
        for four_interaction, four_coef in four_dict.items():
            for moebius_interaction in powerset(four_interaction):
                moebius_dict[moebius_interaction] += four_coef * (-2) ** (len(moebius_interaction))
        return dict(moebius_dict)

    def _sklearn_to_fourier(self, tree_models: list[TreeModel]) -> dict[tuple[int, ...], float]:
        """Extracts the aggregated Fourier coefficients from a list of sklearn tree models.

        This method iterates over all trees in the ensemble, computes the Fourier coefficients
        for each individual tree using the `_sklearn_tree_to_fourier` helper method, and then
        sums these coefficients to get the final Fourier representation of the complete model.

        Args:
            tree_models: A list of `TreeModel` instances representing the tree using the sklearn.tree structure.

        Returns:
            A dictionary that maps interaction tuples (representing Fourier frequencies) to their aggregated Fourier coefficients.
        """
        aggregated_coeffs = defaultdict(float)

        for tree_model in tree_models:
            tree_coeffs = self._sklearn_tree_to_fourier(tree_model)
            for interaction, value in tree_coeffs.items():
                aggregated_coeffs[interaction] += value

        # Convert defaultdict to a standard dict, removing zero-valued coefficients
        return {k: v for k, v in aggregated_coeffs.items() if v != 0.0}

    def _sklearn_tree_to_fourier(self, tree_model: TreeModel) -> dict[tuple[int, ...], float]:
        """Recursively extracts Fourier coefficients from a single sklearn decision tree.

        This method traverses the sklearn decision tree structure defined in `shapiq.tree.base`.
        It computes the Fourier representation of the piecewise-constant function that the tree defines.

        Args:
            tree_model: A fitted sklearn decision tree model.
        """

        def _combine_coeffs(
            left_coeffs: dict[tuple[int, ...], float],
            right_coeffs: dict[tuple[int, ...], float],
            feature_idx: int,
        ) -> dict[tuple[int, ...], float]:
            """Combines Fourier coefficients from the left and right children of a split node."""
            combined_coeffs = {}
            all_interactions = set(left_coeffs.keys()) | set(right_coeffs.keys())

            for interaction in all_interactions:
                left_val = left_coeffs.get(interaction, 0.0)
                right_val = right_coeffs.get(interaction, 0.0)
                combined_coeffs[interaction] = (left_val + right_val) / 2

                new_interaction = tuple(sorted(set(interaction) | {feature_idx}))
                combined_coeffs[new_interaction] = (left_val - right_val) / 2
            return combined_coeffs

        def _dfs_traverse(node_id: int) -> dict[tuple[int, ...], float]:
            """Performs a depth-first traversal of the tree to compute coefficients."""
            # Base case: if the node is a leaf, its function is a constant.
            if tree_model.children_left[node_id] == -1 and tree_model.children_right[node_id] == -1:
                # The only non-zero coefficient is for the empty interaction (the bias term).
                return {(): tree_model.values[node_id]}
            # Recursive step: if the node is a split node.
            left_child_id = tree_model.children_left[node_id]
            right_child_id = tree_model.children_right[node_id]
            left_coeffs = _dfs_traverse(left_child_id)
            right_coeffs = _dfs_traverse(right_child_id)
            feature_idx = tree_model.features[node_id]
            return _combine_coeffs(left_coeffs, right_coeffs, feature_idx)

        return _dfs_traverse(0)  # Start traversal from the root node (id=0)

    def _refine(
        self,
        four_dict: dict[tuple[int, ...], float],
        train_X: np.ndarray,
        train_y: np.ndarray,
    ) -> dict[tuple[int, ...], float]:
        """Refines the estimated Fourier coefficients using a Ridge regression model.

        This method takes an initial set of estimated Fourier coefficients and refines them to
        better fit the observed game values. It first identifies the most significant
        coefficients by keeping those that contribute to 95% of the total "energy" (sum of
        squared Fourier coefficients, excluding the baseline). Then, it constructs a new feature matrix
        based on the Fourier basis functions corresponding to these significant interactions.
        Finally, it fits a `RidgeCV` model to re-estimate the values of these coefficients,
        effectively fine-tuning them against the training data.

        Args:
            four_dict: A dictionary mapping interaction tuples to their initial estimated
                Fourier coefficient values.
            train_X: The training data matrix where rows are coalitions (binary vectors) and
                columns are players.
            train_y: The corresponding game values for each coalition in `train_X`.

        Returns:
            A dictionary containing the refined Fourier coefficients for the most significant
            interactions.
        """
        n = train_X.shape[1]
        four_items = list(four_dict.items())
        if len(four_items) <= self.n:
            return four_dict
        list_keys = [item[0] for item in four_items]
        four_coefs = np.array([item[1] for item in four_items])

        nfc_idx = list_keys.index(()) if () in list_keys else None

        four_coefs_for_energy = np.copy(four_coefs)
        if nfc_idx is not None:
            four_coefs_for_energy[nfc_idx] = 0
        four_coefs_sq = four_coefs_for_energy**2
        tot_energy = np.sum(four_coefs_sq)
        if tot_energy == 0:
            return four_dict
        sorted_four_coefs_sq = np.sort(four_coefs_sq)[::-1]
        cumulative_energy_ratio = np.cumsum(sorted_four_coefs_sq / tot_energy)
        thresh_idx_95 = np.argmin(cumulative_energy_ratio < 0.95) + 1
        thresh = np.sqrt(sorted_four_coefs_sq[thresh_idx_95])

        four_dict_trunc = {
            tuple(int(i in k) for i in range(n)): v for k, v in four_dict.items() if abs(v) > thresh
        }
        support = np.array(list(four_dict_trunc.keys()))

        # Construct the fourier basis coefficient matrix for the training data
        X = np.real(np.exp(train_X @ (1j * np.pi * support.T)))
        # Solve the regression problem to obtain refined Fourier coefficients
        reg = RidgeCV(alphas=np.logspace(-6, 6, 100), fit_intercept=False).fit(X, train_y)

        regression_coefs = dict(
            zip([tuple(s.astype(int)) for s in support], reg.coef_, strict=False)
        )
        return {tuple(i for i, x in enumerate(k) if x): v for k, v in regression_coefs.items()}

    def _process_moebius(self, moebius_transform: dict[tuple, float]) -> np.ndarray:
        """Convert the Moebius transform into the desired index.

        Args:
            moebius_transform: The Moebius transform to process as a dict mapping tuples to float
                values.

        Returns:
            np.ndarray: The converted interaction values based on the specified index.
            The function also updates the internal _interaction_lookup dictionary.
        """
        moebius_interactions = InteractionValues(
            values=moebius_transform,
            index="Moebius",
            min_order=self.min_order,
            max_order=self.max_order,
            n_players=self.n,
            estimated=True,
            baseline_value=moebius_transform.get((), 0.0),
        )
        autoconverter = MoebiusConverter(moebius_coefficients=moebius_interactions)
        converted_interaction_values = autoconverter(index=self.index, order=self.max_order)
        self._interaction_lookup = converted_interaction_values.interaction_lookup
        return converted_interaction_values.values

    def _filter_order(self, result: np.ndarray) -> np.ndarray:
        """Filters the interactions to keep only those of the maximum order.

        This method is used when top_order=True to filter out all interactions that are not
        of exactly the maximum order (self.max_order).

        Args:
            result: Array of interaction values.

        Returns:
            Filtered array containing only interaction values of the maximum order.
            The method also updates the internal _interaction_lookup dictionary.
        """
        filtered_interactions = {}
        filtered_results = []
        i = 0
        for j, key in enumerate(self.interaction_lookup):
            if len(key) == self.max_order:
                filtered_interactions[key] = i
                filtered_results.append(result[j])
                i += 1
        self._interaction_lookup = filtered_interactions
        return np.array(filtered_results)
