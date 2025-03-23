from collections.abc import Callable
from .._base import Approximator
from ...interaction_values import InteractionValues
from ...game_theory.moebius_converter import MoebiusConverter
from sparse_transform.qsft.qsft import transform as sparse_fourier_transform
from sparse_transform.qsft.codes.BCH import BCH
from sparse_transform.qsft.signals.input_signal_subsampled import SubsampledSignal as SubsampledSignalFourier
from sparse_transform.smt.smt import transform as sparse_moebius_transform
from sparse_transform.smt.input_signal_subsampled import SubsampledSignal as SubsampledSignalMoebius
from ...utils.sets import powerset
from functools import partial
import numpy as np

class Sparse(Approximator):

    def __init__(
        self,
        n: int,
        max_order: int,
        index: str,
        top_order: bool = False,
        random_state: int | None = None,
        transform_type: str = "fourier",
        decoder_type: str | None = None, # TODO JK Do we want to do this?
    ) -> None:
        if  transform_type.lower() not in ["fourier", 'mobius']:
            raise ValueError("transform_type must be either 'fourier' or mobius'")
        self.transform_type = transform_type.lower()

        self.t = 5 if transform_type == 'fourier' else max(5, max_order) # 5 could be a parameter

        if self.transform_type == "fourier":
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
            self.decoder_args = { #TODO set defaults correctly in sparse-transform and remove some of this
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

        elif self.transform_type == "mobius":
            if decoder_type is not None:
                raise ValueError("decoder_type is not supported for Mobius transform")
            self.query_args = {
                "query_method": "simple",
                "num_subsample": 3,
                "delays_method_source": "identity",
                "delays_method_channel": "identity",
                "subsampling_method": "smt",
                "num_repeat": 1,
            }
            self.decoder_args = {
                "num_subsample": 3,
                "num_repeat": 1,
                "reconstruct_method_source": "simple",
                "reconstruct_method_channel": "simple",
                "noise_sd": 0,
                "source_decoder": None
            }
        super().__init__(
            n=n,
            max_order=max_order,
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
        used_budget = self._set_transform_budget(budget)
        if self.transform_type == "fourier":
            signal = SubsampledSignalFourier(func=game, n=self.n, q=2, query_args=self.query_args)
            initial_transformer = sparse_fourier_transform
        elif self.transform_type == "mobius":
            signal = SubsampledSignalMoebius(func=game, n=self.n, q=2, query_args=self.query_args)
            initial_transformer = sparse_moebius_transform

        initial_transform = {tuple(np.nonzero(key)[0]): np.real(value) for key, value in
                            initial_transformer(signal, **self.decoder_args).items()}

        if self.transform_type == "fourier":
            moebius_transform = Sparse._fourier_to_moebius(initial_transform) # TODO add max order
            # TODO replace with sparse_transform.qsft.utils.general.fourier_to_mobius
        elif self.transform_type == "mobius":
            moebius_transform = initial_transform

        result = self._process_mobius(mobius_transform=moebius_transform)
        return self._finalize_result(result=result,
                                     baseline_value=0.0,
                                     estimated=True,
                                     budget=used_budget,
                                     )

    def _process_mobius(self, mobius_transform: dict[tuple, float]) -> np.ndarray:
        """Processes the Mobius transform to extract interaction values.

        Args:
            mobius_transform: The Mobius transform to process.

        Returns:
            A dictionary of interaction values.
        """
        moebius_interactions = list(mobius_transform.keys())
        self._interaction_lookup = {key: i for i, key in enumerate(moebius_interactions)}
        values = np.array([mobius_transform[key] for key in moebius_interactions])
        mobius_interactions = InteractionValues(
            values=values,
            index="Moebius",
            min_order=self.min_order,
            max_order=self.max_order,
            n_players=self.n,
            interaction_lookup=self._interaction_lookup,
            estimated=True,
            baseline_value=0.0, # TODO verify
        )
        autoconverter = MoebiusConverter(moebius_coefficients=mobius_interactions)
        converted_interaction_values = autoconverter(index=self.index, order=self.max_order)
        self._interaction_lookup = converted_interaction_values.interaction_lookup
        return converted_interaction_values.values

    def _set_transform_budget(self, budget: int) -> int:
        if self.transform_type == "fourier":
            #TODO replace with static functions in SubsampledSignalFourier
            b = Sparse.get_b_for_sample_budget_fourier(budget, self.n, self.t, 2, self.query_args)
            used_budget = Sparse.get_number_of_samples_fourier(self.n, b, self.t, 2, self.query_args)
        elif self.transform_type == "mobius":
            # TODO replace with static functions in SubsampledSignalMoebius
            b = Sparse.get_b_for_sample_budget_mobius(budget, self.n, self.query_args)
            used_budget = Sparse.get_number_of_samples_mobius(self.n, b, self.query_args)
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
    def get_number_of_samples_mobius(n, b, query_args):
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
    def get_b_for_sample_budget_mobius(budget, n, query_args):
        """
        Find the maximum value of b that fits within the given sample budget.
        """
        num_subsample = query_args.get("num_subsample", 1)
        num_rows_per_D = Sparse._get_delay_overhead_mobius(n, query_args)
        largest_b = np.floor(np.log(budget / (num_rows_per_D * num_subsample))/np.log(2))
        return int(largest_b)


    @staticmethod
    def _get_delay_overhead_mobius(n, query_args, t = None):
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