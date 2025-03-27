from collections.abc import Callable
from .._base import Approximator
from ...interaction_values import InteractionValues
from ...game_theory.moebius_converter import MoebiusConverter
from sparse_transform.qsft.qsft import transform as sparse_fourier_transform
from sparse_transform.qsft.codes.BCH import BCH
from sparse_transform.qsft.signals.input_signal_subsampled import SubsampledSignal as SubsampledSignalFourier
from ...utils.sets import powerset
from functools import partial
import numpy as np

class Sparse(Approximator):
    """Approximator for interaction values using sparse transformation techniques.

    This class implements a sparse approximation method for computing various interaction indices
    using sparse Fourier transforms. It efficiently estimates interaction values with a limited
    sample budget by leveraging sparsity in the Fourier domain.

    Attributes:
        transform_type (str): Type of transform used (currently only "fourier" is supported).
        t (int): Error parameter for the sparse Fourier transform.
        signal_class: Class for representing the subsampled signal.
        initial_transformer: Function for performing the initial transform.
        query_args (dict): Parameters for querying the signal.
        decoder_args (dict): Parameters for decoding the transform.

    Args:
        n (int): Number of players/features.
        index (str): Type of interaction index to compute (e.g., "STII", "FBII", "FSII").
        max_order (int, optional): Maximum interaction order to compute. It is not suggested to use this parameter
            since sparse approximation dynamically adjusts the order based on the budget.
        top_order (bool, optional): If True, only compute interactions of exactly max_order.
            If False, compute interactions up to max_order. Defaults to False.
        random_state (int, optional): Random seed for reproducibility. Defaults to None.
        transform_type (str, optional): Type of transform to use. Currently only "fourier" 
            is supported. Defaults to "fourier".
        decoder_type (str, optional): Type of decoder to use, either "soft" or "hard".
            Defaults to "soft"

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
    ) -> None:
        if  transform_type.lower() not in ["fourier"]:
            raise ValueError("transform_type must be 'fourier'")
        self.transform_type = transform_type.lower()
        self.t = 5 # 5 could be a parameter
        self.signal_class = SubsampledSignalFourier
        self.initial_transformer = sparse_fourier_transform
        decoder_type = 'hard' if decoder_type is None else decoder_type.lower()
        if decoder_type not in ["soft", "hard"]:
            raise ValueError("decoder_type must be either 'soft' or 'hard'")
        # The sampling parameters for the Fourier transform
        self.query_args = {
        "query_method": "complex",
        "num_subsample": 3,
        "delays_method_source": "joint-coded",
        "subsampling_method": "qsft",
        "delays_method_channel": "identity-siso",
        "num_repeat": 1,
        "t": self.t,
    }
        self.decoder_args = {
            "num_subsample": 3,
            "num_repeat": 1,
            "reconstruct_method_source": "coded",
            "peeling_method": "multi-detect",
            "reconstruct_method_channel": "identity-siso" if decoder_type != "hard" else "identity",
            "noise_sd": 0,
            "regress": 'lasso',
            "res_energy_cutoff": 0.9,
            "trap_exit": True,
            "verbosity": 0,
            "report": False,
            "peel_average": True,
        }
        # deal with the decoder type
        if decoder_type == "hard":
            source_decoder = BCH(n, self.t).parity_decode
        else:
            source_decoder = partial(BCH(n, self.t).parity_decode_2chase_t2_max_likelihood,
                                     chase_depth=2 * self.t)
        self.decoder_args["source_decoder"] = source_decoder
        super().__init__(
            n=n,
            max_order= n if max_order is None else max_order,
            index=index,
            top_order=top_order,
            random_state=random_state,
            initialize_dict=False,
        )

    def approximate(
        self, budget: int, game: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> InteractionValues:
        """Approximates the interaction values using a sparse transform approach.

        Args:
            budget: The budget for the approximation.
            game: The game function that returns the values for the coalitions.

        Returns:
            The approximated Shapley interaction values.
        """
        # Find the maximum value of b that fits within the given sample budget and get the used budget
        used_budget = self._set_transform_budget(budget)
        signal = self.signal_class(func=game, n=self.n, q=2, query_args=self.query_args)
        # Extract the coefficients of the original transform
        initial_transform = {tuple(np.nonzero(key)[0]): np.real(value) for key, value in
                            self.initial_transformer(signal, **self.decoder_args).items()}
        # If we are using the fourier transform, we need to convert it to a Moebius transform
        moebius_transform = Sparse._fourier_to_moebius(initial_transform)  # TODO replace with sparse_transform.qsft.utils.general.fourier_to_mobius
        result = self._process_moebius(moebius_transform=moebius_transform)
        # Filter the output as needed
        if self.top_order:
            result = self._filter_order(result)
        return self._finalize_result(result=result,
                                     baseline_value=self.interaction_lookup.get((), 0.0),
                                     estimated=True,
                                     budget=used_budget)

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
        """Processes the Moebius transform to extract the desired index.

        Args:
            moebius_transform: The Moebius transform to process (dict mapping tuples to float values).

        Returns:
            np.ndarray: The converted interaction values based on the specified index.
            The function also updates the internal _interaction_lookup dictionary.
        """
        moebius_interactions = InteractionValues(
            values=np.array([moebius_transform[key] for key in moebius_transform.keys()]),
            index="Moebius",
            min_order=self.min_order,
            max_order=self.max_order,
            n_players=self.n,
            interaction_lookup={key: i for i, key in enumerate(moebius_transform.keys())},
            estimated=True,
            baseline_value=moebius_transform.get((), 0.0)
        )
        #TODO check that the following code doesn't do anything inefficient
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
            int: The actual number of samples that will be used, which is less than or equal to the budget.

        Raises:
            ValueError: If the budget is too low to compute the transform with acceptable parameters.
        """
        #TODO replace with static functions in SubsampledSignalFourier
        b = Sparse.get_b_for_sample_budget_fourier(budget, self.n, self.t, 2, self.query_args)
        used_budget = Sparse.get_number_of_samples_fourier(self.n, b, self.t, 2, self.query_args)
        # TODO Should we try to decrease t to get b to at least 3?
        if b <= 2: #TODO Better to re-route than to throw an error?
            raise ValueError("Budget is too low to compute the transform, consider increasing the budget or  using a "
                             "different approximator.")
        self.query_args['b'] = b
        self.decoder_args['b'] = b
        return used_budget

    ##########################################################################
    # These functions will be replaced in the next release of sparse-transform
    ##########################################################################
    @staticmethod
    def _fourier_to_moebius(fourier_transform):
        moebius_dict = {}
        for loc, coef in fourier_transform.items():
            for subset in powerset(loc):
                scaling_factor = np.power(-2.0, len(subset))
                if subset in moebius_dict:
                    moebius_dict[subset] += coef * scaling_factor
                else:
                    moebius_dict[subset] = coef * scaling_factor
        return moebius_dict

    @staticmethod
    def get_number_of_samples_fourier(n, b, t, q, query_args):
        """
        Computes the number of vector-wise calls to self.func for the given query_args, n, t, and b.
        """
        num_subsample = query_args.get("num_subsample", 1)
        num_rows_per_D = Sparse._get_delay_overhead_fourier(n, t, query_args)
        samples_per_row = q ** b
        total_samples = num_subsample * num_rows_per_D * samples_per_row  # upper bound
        return total_samples

    @staticmethod
    def get_b_for_sample_budget_fourier(budget, n, t, q, query_args):
        """
        Find the maximum value of b that fits within the given sample budget.

        Parameters:
        budget (int): The maximum number of samples allowed.
        n (int): Number of rows.
        t (int): Error parameter.
        q (int): Base of the transform.
        query_args (dict): Additional query arguments.

        Returns:
        int: The maximum value of b that keeps the total samples within budget.
        """
        num_subsample = query_args.get("num_subsample", 1)
        num_rows_per_D = Sparse._get_delay_overhead_fourier(n, t, query_args)
        largest_b = np.floor(np.log(budget / (num_rows_per_D * num_subsample)) / np.log(q))
        return int(largest_b)

    @staticmethod
    def _get_delay_overhead_fourier(n, t, query_args):  # TODO depends on q in general
        """
        Returns the overhead of the delays in terms of the number of samples
        """
        delays_method_source = query_args.get("delays_method_source", "identity")
        if delays_method_source == "identity":
            num_rows_per_D = n + 1
        elif delays_method_source == "joint-coded":
            from sparse_transform.qsft.codes.BCH import BCH
            nc, kc = BCH.parameter_search(n, t)
            num_rows_per_D = nc - kc + 1  # BCH parity length + 1 (for zero row)
        elif delays_method_source == "random":
            # For random delays, the number is specified or defaulted
            num_rows_per_D = query_args.get("num_delays", n)
        else:
            # For other delay methods, assume default behavior
            num_rows_per_D = n + 1

        if query_args.get("delays_method_channel") == "nso":
            num_repeat = query_args.get("num_repeat", 1)
        else:
            num_repeat = 1
        return num_rows_per_D * num_repeat

    @staticmethod
    def get_number_of_samples_moebius(n, b, query_args):
        """
        Computes the number of vector-wise calls to self.func for the given query_args, n, and b.
        """
        # Get number of subsampling matrices
        num_subsample = query_args.get("num_subsample", 1)
        num_rows_per_D = Sparse._get_delay_overhead_mobius(n, query_args)
        # Calculate samples per row (2^b for binary case)
        samples_per_row = 2 ** b

        # Calculate total samples
        total_samples = num_subsample * num_rows_per_D * samples_per_row
        return total_samples


    @staticmethod
    def get_b_for_sample_budget_moebius(budget, n, query_args):
        """
        Find the maximum value of b that fits within the given sample budget.
        """
        num_subsample = query_args.get("num_subsample", 1)
        num_rows_per_D = Sparse._get_delay_overhead_mobius(n, query_args)
        largest_b = np.floor(np.log(budget / (num_rows_per_D * num_subsample))/np.log(2))
        return int(largest_b)


    @staticmethod
    def _get_delay_overhead_moebius(n, query_args, t = None):
        # Calculate number of rows in each delay matrix
        delays_method_source = query_args.get("delays_method_source", "identity")
        if delays_method_source == "identity":
            num_rows_per_D = n + 1
        elif delays_method_source == "random":
            num_rows_per_D = query_args.get("num_delays", n) + 1
        else:
            # For other delay methods in SMT, default to n+1
            num_rows_per_D = n + 1

        # Account for channel method and num_repeat
        if query_args.get("delays_method_channel") == "nso":
            num_repeat = query_args.get("num_repeat", 1)
        else:
            num_repeat = 1
        return num_rows_per_D * num_repeat