"""Base Sparse approximator for fourier-based interaction computation."""

from __future__ import annotations

import collections.abc
import copy
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from sparse_transform.qsft.qsft import transform as sparse_fourier_transform
from sparse_transform.qsft.signals.input_signal_subsampled import (
    SubsampledSignal as SubsampledSignalFourier,
)
from sparse_transform.qsft.utils.general import fourier_to_mobius as fourier_to_moebius
from sparse_transform.qsft.utils.query import get_bch_decoder

from shapiq.approximator.base import Approximator
from shapiq.approximator.sampling import CoalitionSampler
from shapiq.game_theory.moebius_converter import MoebiusConverter, ValidMoebiusConverterIndices
from shapiq.interaction_values import InteractionValues

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor

    from shapiq.game import Game

ValidSparseIndices = ValidMoebiusConverterIndices


class Sparse(Approximator):
    """Approximator interface using sparse transformation techniques.

    This class implements a sparse approximation method for computing various interaction indices
    using sparse Fourier transforms. It efficiently estimates interaction values with a limited
    sample budget by leveraging sparsity in the Fourier domain. The notion of sparse approximation
    is described in [Kan25]_ and further improved in [But25]_.

    See Also:
        - :class:`~shapiq.approximator.sparse.SPEX` for a specific implementation of the
            sparse approximation using Fourier transforms described in [Kan25]_.
        - :class:`~shapiq.approximator.sparse.ProxySPEX` for a specific implementation of the
            sparse approximation using Fourier transforms described in [But25]_.

    Attributes:
        transform_type: Type of transform used (currently only ``"fourier"`` is supported).

        degree_parameter: A parameter that controls the maximum degree of the interactions to
            extract during execution of the algorithm. Note that this is a soft limit, and in
            practice, the algorithm may extract interactions of any degree. We typically find
            that there is little value going beyond ``5``. Defaults to ``5``. Note that
            increasing this parameter will need more ``budget`` in the :meth:`approximate`
            method.

        query_args: Parameters for querying the signal.

        decoder_args: Parameters for decoding the transform.

    Raises:
        ValueError: If transform_type is not "fourier" or if decoder_type is not "soft" or "hard".

    References:
        .. [Kan25] Kang, J.S., Butler, L., Agarwal. A., Erginbas, Y.E., Pedarsani, R., Ramchandran, K., Yu, Bin (2025). SPEX: Scaling Feature Interaction Explanations for LLMs https://arxiv.org/abs/2502.13870
        .. [But25] Butler, L., Kang, J.S., Agarwal. A., Erginbas, Y.E., Yu, Bin, Ramchandran, K. (2025). ProxySPEX: Inference-Efficient Interpretability via Sparse Feature Interactions in LLMs https://arxiv.org/pdf/2505.17495
    """

    valid_indices: tuple[ValidSparseIndices] = tuple(get_args(ValidSparseIndices))
    """The valid indices for the SPEX approximator."""

    def __init__(
        self,
        n: int,
        index: ValidSparseIndices,
        *,
        max_order: int | None = None,
        top_order: bool = False,
        random_state: int | None = None,
        transform_type: Literal["fourier"] = "fourier",
        decoder_type: Literal["soft", "hard", "proxyspex"] | None = "proxyspex",
        degree_parameter: int = 5,
    ) -> None:
        """Initialize the Sparse approximator.

        Args:
            n: Number of players (features).

            max_order: Maximum interaction order to consider. Defaults to ``None``, which means
                that all orders up to ``n`` will be considered.

            index: The Interaction index to use. All indices supported by shapiq's
                :class:`~shapiq.game_theory.moebius_converter.MoebiusConverter` are supported.

            top_order: If ``True``, only reports interactions of exactly order ``max_order``.
                Otherwise, reports all interactions up to order ``max_order``. Defaults to
                ``False``.

            random_state: Seed for random number generator. Defaults to ``None``.

            transform_type: Type of transform to use. Currently only "fourier" is supported.

            decoder_type: Type of decoder to use, either "soft", "hard", or "proxyspex". Defaults to "proxyspex".

            degree_parameter: A parameter that controls the maximum degree of the interactions to
                extract during execution of the algorithm. Note that this is a soft limit, and in
                practice, the algorithm may extract interactions of any degree. We typically find
                that there is little value going beyond ``5``. Defaults to ``5``. Note that
                increasing this parameter will need more ``budget`` in the :meth:`approximate`
                method.

        """
        if transform_type.lower() not in ["fourier"]:
            msg = "transform_type must be 'fourier'"
            raise ValueError(msg)
        self.transform_type = transform_type.lower()
        self.degree_parameter = degree_parameter
        max_order = n if max_order is None else max_order
        self.decoder_type = "proxyspex" if decoder_type is None else decoder_type.lower()
        if self.decoder_type not in ["soft", "hard", "proxyspex"]:
            msg = "decoder_type must be 'soft', 'hard', or 'proxyspex'"
            raise ValueError(msg)
        # The sampling parameters for the Fourier transform
        self.query_args = {
            "query_method": "complex",
            "num_subsample": 3,
            "delays_method_source": "joint-coded",
            "subsampling_method": "qsft",
            "delays_method_channel": "identity-siso",
            "num_repeat": 1,
            "t": self.degree_parameter,
        }
        if self.decoder_type == "proxyspex":
            self.decoder_args = {
                "max_depth": [3, max_order],
                "max_iter": [500],
                "learning_rate": [0.01, 0.1],
            }
            self._uniform_sampler = CoalitionSampler(
                n_players=n,
                sampling_weights=np.array([math.comb(n, i) for i in range(n + 1)]),
                pairing_trick=False,
                random_state=random_state,
            )
        else:
            self.decoder_args = {
                "num_subsample": 3,
                "num_repeat": 1,
                "reconstruct_method_source": "coded",
                "peeling_method": "multi-detect",
                "reconstruct_method_channel": "identity-siso"
                if self.decoder_type == "soft"
                else "identity",
                "regress": "lasso",
                "res_energy_cutoff": 0.9,
                "source_decoder": get_bch_decoder(n, self.degree_parameter, self.decoder_type),
            }
        super().__init__(
            n=n,
            max_order=max_order,
            index=index,
            top_order=top_order,
            random_state=random_state,
            initialize_dict=False,  # Important for performance
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
        if self.decoder_type == "proxyspex":
            # Take a budget amount of uniform samples
            used_budget = budget
            self._uniform_sampler.sample(budget)
            train_y = game(self._uniform_sampler.coalitions_matrix)
            train_X = pd.DataFrame(
                self._uniform_sampler.coalitions_matrix, columns=[f"f{i}" for i in range(self.n)]
            )

            # Train a proxy model with GridSearchCV to find best hyperparameters
            base_model = HistGradientBoostingRegressor(
                random_state=0, categorical_features=[True] * self.n, max_bins=32
            )
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=self.decoder_args,
                scoring="r2",
                cv=3,
                verbose=0,
                n_jobs=-1,
            )
            grid_search.fit(train_X, train_y)

            # Extract and refine Fourier coefficients from the best model
            best_model = grid_search.best_estimator_
            initial_transform = self._refine(
                self._histgb_to_fourier(best_model, best_model._baseline_prediction.item()),  # noqa: SLF001
                self._uniform_sampler.coalitions_matrix,
                train_y,
            )

        else:
            # Find the max value of b that fits within the given sample budget and get the used budget
            used_budget = self._set_transform_budget(budget)
            signal = SubsampledSignalFourier(
                func=lambda inputs: game(inputs.astype(bool)),
                n=self.n,
                q=2,
                query_args=self.query_args,
            )
            # Extract the coefficients of the original transform
            initial_transform = {
                tuple(np.nonzero(key)[0]): np.real(value)
                for key, value in sparse_fourier_transform(signal, **self.decoder_args).items()
            }
        # If we are using the fourier transform, we need to convert it to a Moebius transform
        moebius_transform = fourier_to_moebius(initial_transform)
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
            interaction_lookup=copy.deepcopy(self.interaction_lookup),
            estimated=True,
            estimation_budget=used_budget,
            baseline_value=self.interaction_lookup.get((), 0.0),
            target_index=self.index,
        )

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
            values=np.array([moebius_transform[key] for key in moebius_transform]),
            index="Moebius",
            min_order=self.min_order,
            max_order=self.max_order,
            n_players=self.n,
            interaction_lookup={key: i for i, key in enumerate(moebius_transform.keys())},
            estimated=True,
            baseline_value=moebius_transform.get((), 0.0),
        )
        autoconverter = MoebiusConverter(moebius_coefficients=moebius_interactions)
        converted_interaction_values = autoconverter(index=self.index, order=self.max_order)
        self._interaction_lookup = converted_interaction_values.interaction_lookup
        return converted_interaction_values.values  # noqa: PD011

    def _set_transform_budget(self, budget: int) -> int:
        """Sets the appropriate transform budget parameters based on the given sample budget.

        This method calculates the maximum possible 'b' parameter (number of bits to subsample)
        that fits within the provided budget, then configures the query and decoder arguments
        accordingly. The actual number of samples that will be used is returned.

        Args:
            budget: The maximum number of samples allowed for the approximation.

        Returns:
            int: The actual number of samples that will be used, which is less than or equal to the
                budget.

        Raises:
            ValueError: If the budget is too low to compute the transform with acceptable parameters.
        """
        b = SubsampledSignalFourier.get_b_for_sample_budget(
            budget, self.n, self.degree_parameter, 2, self.query_args
        )
        used_budget = SubsampledSignalFourier.get_number_of_samples(
            self.n, b, self.degree_parameter, 2, self.query_args
        )

        if b <= 2:
            while self.degree_parameter > 2:
                self.degree_parameter -= 1
                self.query_args["t"] = self.degree_parameter

                # Recalculate 'b' with the updated 't'
                b = SubsampledSignalFourier.get_b_for_sample_budget(
                    budget, self.n, self.degree_parameter, 2, self.query_args
                )

                # Compute the used budget
                used_budget = SubsampledSignalFourier.get_number_of_samples(
                    self.n, b, self.degree_parameter, 2, self.query_args
                )

                # Break if 'b' is now sufficient
                if b > 2:
                    self.decoder_args["source_decoder"] = get_bch_decoder(
                        self.n, self.degree_parameter, self.decoder_type
                    )
                    break

            # If 'b' is still too low, raise an error
            if b <= 2:
                msg = (
                    "Insufficient budget to compute the transform. Increase the budget or use a "
                    "different approximator."
                )
                raise ValueError(msg)
        # Store the final 'b' value
        self.query_args["b"] = b
        self.decoder_args["b"] = b
        return used_budget

    def _histgb_to_fourier(
        self, model: HistGradientBoostingRegressor, baseline: float
    ) -> dict[tuple[int, ...], float]:
        """Extracts the aggregated Fourier coefficients from a HistGradientBoostingRegressor model.

        This method iterates through all the trees in a trained scikit-learn booster model,
        extracts the Fourier coefficients from each individual tree using the
        `_histgb_tree_to_fourier` helper method, and aggregates them.

        Args:
            model: The trained `HistGradientBoostingRegressor` model.
            baseline: The baseline prediction value of the model.

        Returns:
            A dictionary mapping interaction tuples to their aggregated Fourier coefficient values.
        """
        aggregated_coeffs = defaultdict(float)

        # Scikit-learn's trees are stored in an internal `_predictors` attribute.
        all_tree_predictors = (
            [p for sublist in model._predictors for p in sublist]  # noqa: SLF001
            if model._predictors and isinstance(model._predictors[0], collections.abc.Sequence)  # noqa: SLF001
            else model._predictors  # noqa: SLF001
        )
        for tree_predictor in all_tree_predictors:
            tree_coeffs = self._histgb_tree_to_fourier(tree_predictor)
            for interaction, value in tree_coeffs.items():
                aggregated_coeffs[interaction] += value
        aggregated_coeffs[()] += baseline

        return {k: v for k, v in aggregated_coeffs.items() if v != 0.0}

    def _histgb_tree_to_fourier(
        self, tree_predictor: TreePredictor
    ) -> dict[tuple[int, ...], float]:
        """Recursively extracts the Fourier coefficients from a single scikit-learn tree.

        This method traverses a single decision tree from a HistGradientBoostingRegressor model
        and computes its exact Fourier transform. It uses a recursive, depth-first approach
        adapted to scikit-learn's internal tree representation (a flattened NumPy array).

        Args:
            tree_predictor: An internal `TreePredictor` object from the trained model.

        Returns:
            A dictionary mapping interaction tuples to their Fourier coefficients for the given tree.
        """

        def _combine_coeffs(
            left_coeffs: dict[tuple[int, ...], float],
            right_coeffs: dict[tuple[int, ...], float],
            feature_idx: int,
        ) -> dict[tuple[int, ...], float]:
            # Combines Fourier coefficients from the left and right children of a split node.
            combined = {}

            base_interactions = set(left_coeffs.keys()) | set(right_coeffs.keys())
            for interaction in base_interactions:
                # This check is technically not needed if the logic is right, but is good for safety
                if feature_idx in interaction:
                    continue

                left_val = left_coeffs.get(interaction, 0.0)
                right_val = right_coeffs.get(interaction, 0.0)

                # Formula for interactions NOT containing the split feature
                combined[interaction] = (left_val + right_val) / 2

                # Formula for interactions that DO contain the split feature
                new_interaction = tuple(sorted(set(interaction) | {feature_idx}))
                combined[new_interaction] = (left_val - right_val) / 2

            return combined

        def _dfs_traverse(nodes: np.ndarray, node_idx: int) -> dict[tuple[int, ...], float]:
            # Performs a depth-first traversal of the tree array to compute coefficients.

            # Define column indices for clarity. This mapping corresponds to the structure
            # of the TreePredictor's `nodes` array in your scikit-learn version.
            VALUE_IDX = 0
            FEATURE_IDX = 2
            LEFT_CHILD_IDX = 5
            RIGHT_CHILD_IDX = 6
            LEAF_IDX = 9

            current_node = nodes[node_idx]

            if current_node[LEAF_IDX]:
                # The leaf value is at index 0.
                return {(): current_node[VALUE_IDX]}
            # Recursive step: process left and right children using indices
            left_coeffs = _dfs_traverse(nodes, int(current_node[LEFT_CHILD_IDX]))
            right_coeffs = _dfs_traverse(nodes, int(current_node[RIGHT_CHILD_IDX]))
            feature_idx = int(current_node[FEATURE_IDX])

            # The _combine_coeffs helper function doesn't need to change.
            return _combine_coeffs(left_coeffs, right_coeffs, feature_idx)

        return _dfs_traverse(tree_predictor.nodes, 0)

    def _refine(
        self,
        four_dict: dict[tuple[int, ...], float],
        train_X: np.ndarray[bool],
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
        list_keys = [item[0] for item in four_items]
        four_coefs = np.array([item[1] for item in four_items])

        nfc_idx = list_keys.index(()) if () in list_keys else None

        four_coefs_for_energy = np.copy(four_coefs)
        if nfc_idx is not None:
            four_coefs_for_energy[nfc_idx] = 0
        four_coefs_sq = four_coefs_for_energy**2
        tot_energy = np.sum(four_coefs_sq)
        sorted_four_coefs_sq = np.sort(four_coefs_sq)[::-1]
        cumulative_energy_ratio = np.cumsum(sorted_four_coefs_sq / tot_energy)
        thresh_idx_95 = np.argmin(cumulative_energy_ratio < 0.95) + 1
        thresh = np.sqrt(sorted_four_coefs_sq[thresh_idx_95])

        four_dict_trunc = {
            tuple(int(i in k) for i in range(n)): v for k, v in four_dict.items() if abs(v) > thresh
        }
        support = np.array(list(four_dict_trunc.keys()))

        X = np.real(np.exp(train_X @ (1j * np.pi * support.T)))
        reg = RidgeCV(alphas=np.logspace(-6, 6, 100), fit_intercept=False).fit(X, train_y)

        regression_coefs = dict(
            zip([tuple(s.astype(int)) for s in support], reg.coef_, strict=False)
        )
        return {tuple(i for i, x in enumerate(k) if x): v for k, v in regression_coefs.items()}
