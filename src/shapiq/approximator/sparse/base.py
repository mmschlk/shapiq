"""Base Sparse approximator for fourier-based interaction computation."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, cast, get_args

import numpy as np
import pandas as pd
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

    from shapiq.game import Game

ValidSparseIndices = ValidMoebiusConverterIndices


class Sparse(Approximator[ValidSparseIndices]):
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

    valid_indices: tuple[ValidSparseIndices, ...] = tuple(get_args(ValidSparseIndices))  # type: ignore[assignment]
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
        if self.decoder_type == "proxyspex":
            try:
                import lightgbm as lgb  # noqa: F401
            except ImportError as err:
                msg = (
                    "The 'lightgbm' package is required when decoder_type is 'proxyspex' but it is "
                    "not installed. Please see the installation instructions at "
                    "https://github.com/microsoft/LightGBM/tree/master/python-package."
                )
                raise ImportError(msg) from err
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
                "max_depth": [3, 5],
                "max_iter": [500, 1000],
                "learning_rate": [0.01, 0.1],
            }
            self._uniform_sampler = CoalitionSampler(
                n_players=n,
                sampling_weights=np.array([math.comb(n, i) for i in range(n + 1)], dtype=float),
                pairing_trick=True,
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
            import lightgbm as lgb

            used_budget = budget

            # Take the budget amount of uniform samples
            self._uniform_sampler.sample(budget)

            train_X = pd.DataFrame(
                self._uniform_sampler.coalitions_matrix,
                columns=np.array([f"f{i}" for i in range(self.n)]),
            )
            train_y = game(self._uniform_sampler.coalitions_matrix)

            base_model = lgb.LGBMRegressor(verbose=-1, n_jobs=1, random_state=self._random_state)

            # Set up GridSearchCV with cross-validation
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=self.decoder_args,
                scoring="r2",
                cv=5,
                verbose=0,
                n_jobs=1,
            )

            # Fit the model on the training data
            grid_search.fit(train_X, train_y)

            best_model = grid_search.best_estimator_

            initial_transform = self._refine(
                self._lgboost_to_fourier(best_model.booster_.dump_model()),
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
            baseline_value=result[self.interaction_lookup[()]]
            if () in self.interaction_lookup
            else 0.0,
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
        converted_interaction_values = autoconverter(
            index=cast(ValidMoebiusConverterIndices, self.index), order=self.max_order
        )
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

    def _lgboost_to_fourier(self, model_dict: dict[str, Any]) -> dict[tuple[int, ...], float]:
        """Extracts the aggregated Fourier coefficients from an LGBoost model dictionary.

        This method iterates over all trees in the LightGBM ensemble, computes the
        Fourier coefficients for each individual tree using the `_lgboost_tree_to_fourier`
        helper method, and then sums these coefficients to get the final Fourier
        representation of the complete model.

        Args:
        model_dict: A dictionary representing the trained LGBoost model, as
            produced by `model.booster_.dump_model()`.

        Returns:
            A dictionary that maps interaction tuples (representing Fourier frequencies)
            to their aggregated Fourier coefficients.
        """
        aggregated_coeffs = defaultdict(float)

        for tree_info in model_dict["tree_info"]:
            tree_coeffs = self._lgboost_tree_to_fourier(tree_info)
            for interaction, value in tree_coeffs.items():
                aggregated_coeffs[interaction] += value

        # Convert defaultdict to a standard dict, removing zero-valued coefficients
        return {k: v for k, v in aggregated_coeffs.items() if v != 0.0}

    def _lgboost_tree_to_fourier(self, tree_info: dict[str, Any]) -> dict[tuple[int, ...], float]:
        """Recursively strips the Fourier coefficients from a single LGBoost tree.

        This method traverses a tree's structure, as provided by LightGBM's `dump_model`
        method, and computes the Fourier representation of the piecewise-constant
        function that the tree defines. The logic is adapted from the work by Gorji et al. (2024).

        Args:
            tree_info: A dictionary representing a single decision tree from an LGBM model.

        Returns:
            A dictionary mapping interaction tuples to their corresponding coefficients for
            the single tree.

        References:
            Gorji, Ali, Andisheh Amrollahi, and Andreas Krause.
            "SHAP values via sparse Fourier representation"
            arXiv preprint arXiv:2410.06300 (2024).
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

        def _dfs_traverse(node: dict[str, Any]) -> dict[tuple[int, ...], float]:
            """Performs a depth-first traversal of the tree to compute coefficients."""
            # Base case: if the node is a leaf, its function is a constant.
            if "leaf_value" in node:
                # The only non-zero coefficient is for the empty interaction (the bias term).
                return {(): node["leaf_value"]}
            # Recursive step: if the node is a split node.
            left_coeffs = _dfs_traverse(node["left_child"])
            right_coeffs = _dfs_traverse(node["right_child"])
            feature_idx = node["split_feature"]
            return _combine_coeffs(left_coeffs, right_coeffs, feature_idx)

        return _dfs_traverse(tree_info["tree_structure"])

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
