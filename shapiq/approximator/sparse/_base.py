from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
from sparse_transform.qsft.qsft import transform as sparse_fourier_transform
from sparse_transform.qsft.signals.input_signal_subsampled import (
    SubsampledSignal as SubsampledSignalFourier,
)
from sparse_transform.qsft.utils.general import fourier_to_mobius as fourier_to_moebius
from sparse_transform.qsft.utils.query import get_bch_decoder

from shapiq.approximator._base import Approximator
from shapiq.game_theory.moebius_converter import MoebiusConverter
from shapiq.interaction_values import InteractionValues, finalize_computed_interactions

if TYPE_CHECKING:
    from collections.abc import Callable


class Sparse(Approximator):
    """Approximator for interaction values using sparse transformation techniques.

    This class implements a sparse approximation method for computing various interaction indices
    using sparse Fourier transforms. It efficiently estimates interaction values with a limited
    sample budget by leveraging sparsity in the Fourier domain.

    Attributes:
        transform_type: Type of transform used (currently only "fourier" is supported).
        transform_error: Error tolerance parameter for the sparse Fourier transform.
        query_args: Parameters for querying the signal.
        decoder_args: Parameters for decoding the transform.

    Args:
        n: Number of players/features.
        index: Type of interaction index to compute (e.g., "STII", "FBII", "FSII").
        max_order: Maximum interaction order to compute. It is not suggested to use this parameter
            since sparse approximation dynamically and implicitly adjusts the order based on the
            budget and function.
        top_order: If True, only compute interactions of exactly max_order.
            If False, compute interactions up to max_order. Defaults to False.
        random_state: Random seed for reproducibility. Defaults to None.
        transform_type: Type of transform to use. Currently only "fourier"
            is supported. Defaults to "fourier".
        decoder_type: Type of decoder to use, either "soft" or "hard".
            Defaults to "soft"
        transform_error: Error tolerance parameter for the sparse Fourier transform.
            Higher values increase accuracy but require more samples. Defaults to 5.

    Raises:
        ValueError: If transform_type is not "fourier" or if decoder_type is not "soft" or "hard".
    """

    def __init__(
        self,
        n: int,
        index: str,
        max_order: int | None = None,
        top_order: bool = False,
        random_state: int | None = None,
        transform_type: str = "fourier",
        decoder_type: str = "soft",
        transform_error: int = 5,
    ) -> None:
        if transform_type.lower() not in ["fourier"]:
            msg = "transform_type must be 'fourier'"
            raise ValueError(msg)
        self.transform_type = transform_type.lower()
        self.transform_error = transform_error
        self.decoder_type = "hard" if decoder_type is None else decoder_type.lower()
        if self.decoder_type not in ["soft", "hard"]:
            msg = "decoder_type must be 'soft' or 'hard'"
            raise ValueError(msg)
        # The sampling parameters for the Fourier transform
        self.query_args = {
            "query_method": "complex",
            "num_subsample": 3,
            "delays_method_source": "joint-coded",
            "subsampling_method": "qsft",
            "delays_method_channel": "identity-siso",
            "num_repeat": 1,
            "t": self.transform_error,
        }
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
            "source_decoder": get_bch_decoder(n, self.transform_error, self.decoder_type),
        }
        super().__init__(
            n=n,
            max_order=n if max_order is None else max_order,
            index=index,
            top_order=top_order,
            random_state=random_state,
            initialize_dict=False,  # Important for performance
        )

    def approximate(
        self,
        budget: int,
        game: Callable[[np.ndarray], np.ndarray],
    ) -> InteractionValues:
        """Approximates the interaction values using a sparse transform approach.

        Args:
            budget: The budget for the approximation.
            game: The game function that returns the values for the coalitions.

        Returns:
            The approximated Shapley interaction values.
        """
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
        output = InteractionValues(
            values=result,
            index=self.approximation_index,
            min_order=self.min_order,
            max_order=self.max_order,
            n_players=self.n,
            interaction_lookup=copy.deepcopy(self.interaction_lookup),
            estimated=True,
            estimation_budget=used_budget,
            baseline_value=self.interaction_lookup.get((), 0.0),
        )
        # finalize the interactions
        return finalize_computed_interactions(output, target_index=self.index)

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
        return converted_interaction_values.values

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
            budget, self.n, self.transform_error, 2, self.query_args
        )
        used_budget = SubsampledSignalFourier.get_number_of_samples(
            self.n, b, self.transform_error, 2, self.query_args
        )

        if b <= 2:
            while self.transform_error > 2:
                self.transform_error -= 1
                self.query_args["t"] = self.transform_error

                # Recalculate 'b' with the updated 't'
                b = SubsampledSignalFourier.get_b_for_sample_budget(
                    budget, self.n, self.transform_error, 2, self.query_args
                )

                # Compute the used budget
                used_budget = SubsampledSignalFourier.get_number_of_samples(
                    self.n, b, self.transform_error, 2, self.query_args
                )

                # Break if 'b' is now sufficient
                if b > 2:
                    self.decoder_args["source_decoder"] = get_bch_decoder(
                        self.n, self.transform_error, self.decoder_type
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
