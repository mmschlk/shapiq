"""The GraphExplainer class for the shapiq package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

import joblib
import numpy as np
from torch_geometric.data import Data
from tqdm.auto import tqdm

from shapiq.explainer.base import Explainer
from shapiq.interaction_values import InteractionValues

from .base import GraphGame
from .graphshapiq import GraphSHAPIQ

if TYPE_CHECKING:
    from torch import nn

    from shapiq.game_theory.moebius_converter import ValidMoebiusConverterIndices

SPARSIFY_THRESHOLD = 1e-8


class GraphExplainer(Explainer):
    """The GraphExplainer class for graph-based models.

    The explainer for graph-based models using the
    :class:`~shapiq.graph.graphshapiq.GraphSHAPIQ` algorithm.
    The algorithm is described in:footcite:t:`muschalik2025exactcomputationanyordershapley`.

    GraphSHAP-IQ is an algorithm for computing Shapley Interaction values for graph-based models.
    It is based on the GraphSHAPIQ algorithm by `Muschalik et al. (2025)` [Mus25]_, which efficiently
    calculates any order Shapley Interactions for GNN-based predictions, by utilizing the unique
    characteristics of GNNs, where only nodes in node i's l-hop neighborhood are considered for coalitions,
    thus drastically reducing the total amount of coalitions to evaluate.

    The GraphExplainer can be used with a variety of graph-based models, including
    GCNs, GATs and GINS.
    """

    def __init__(
        self,
        model: nn.Module,
        index: ValidMoebiusConverterIndices = "k-SII",
        baseline_strategy: str = "average",
        max_order: int = 2,
        class_index: int | None = None,
        *,
        efficiency_routine: bool = True,
        normalize: bool = False,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initializes the GraphExplainer.

        Args:
            model: A Graph-based model to explain.
            index: The type of Shapley interaction index to use. Defaults to "k-SII",
                which computes the k-Shapley Interaction Index. If max_order is set
                to 1, this corresponds to the Shapley value (index="SV"). Options are:
                "SV": Shapley value
                "k-SII": k-Shapley Interaction Index
                "FSII": Faithful Shapley Interaction Index
                "FBII": Faithful Banzhaf Interaction Index (becomes BV for order 1)
                "STII": Shapley Taylor Interaction Index
                "SII": Shapley Interaction Index
            class_index: The class index of the model to explain. Defaults to None,
                which will set the class index to 1 per default for classification
                models and is ignored for regression models. Note, it is important
                to specify the class index for your classification model.
            max_order: The maximum interaction order to be computed.
                Defaults to 2. Set to 1 for no interactions (single feature attribution).
            normalize: Whether to normalize the GraphSHAP-IQ algorithm.
            baseline_strategy: The node masking strategy.
            efficiency_routine: Whether to enforce the efficiency axiom during the
                GraphSHAP-IQ computation. Defaults to True.
            **kwargs: Additional keyword arguments are ignored.
        """
        super().__init__(
            model,
            class_index=class_index,
            index=index,
            max_order=max_order
        )
        self._model: nn.Module = model
        self._class_index = class_index
        self._baseline_strategy = baseline_strategy
        self._normalize = normalize
        self._efficiency_routine = efficiency_routine

    @override
    def explain_X(
        self,
        X: np.ndarray | list[Data],
        *,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> list[InteractionValues]:
        """Explain multiple graph predictions at once.

        Args:
            X: A list of :class:`~torch_geometric.data.Data` objects to be explained.
                Passing a NumPy array (as accepted by the base class) is not supported
                and will raise a ``TypeError``.
            n_jobs: Number of jobs for ``joblib.Parallel``. Defaults to ``None`` (no
                parallelization). Set to ``-1`` to use all available cores.
            random_state: Unused; kept for API compatibility with the base class.
            verbose: Whether to show a progress bar. Defaults to ``False``.
            **kwargs: Additional keyword-only arguments forwarded to ``explain_function``.

        Returns:
            A list of interaction values, one per graph in ``X``.

        Raises:
            TypeError: If ``X`` is a NumPy array instead of a list of ``Data`` objects.
        """
        if isinstance(X, np.ndarray):
            msg = (
                "GraphExplainer.explain_X expects a list of torch_geometric.data.Data objects, "
                "not a NumPy array."
            )
            raise TypeError(msg)

        if n_jobs:
            parallel = joblib.Parallel(n_jobs=n_jobs)
            return cast(
                "list[InteractionValues]",
                parallel(joblib.delayed(self.explain)(x_graph, **kwargs) for x_graph in X),
            )
        pbar = tqdm(total=len(X), desc="Explaining") if verbose else None
        ivs: list[InteractionValues] = []
        for x_graph in X:
            ivs.append(self.explain(x_graph, **kwargs))
            if pbar is not None:
                pbar.update(1)
        return ivs

    @override
    def explain_function(
        self,
        x: np.ndarray | Data | None,
        *args: Any,
        max_interaction_size: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> InteractionValues:
        """Computes the Shapley Interaction values for a single instance.

        Args:
            x: The input graph to explain.
            *args: Unused; present only to match the base class signature.
            max_interaction_size: Maximum k-hop neighbourhood size for the L-Shapley
                approximation. When ``None`` the full neighbourhood size reported by
                :class:`~shapiq.graph.graphshapiq.GraphSHAPIQ` is used. Ignored when
                *l_shapley* is ``False``.
            verbose: Whether to print debug information for the underlying game and
                GraphSHAP-IQ explainer. Defaults to ``False``.
            **kwargs: Allows for passing and overriding ``index``, ``efficiency_routine``,
                and ``max_subset_size`` arguments.

        Returns:
            The interaction values for the instance.
        """
        if not isinstance(x, Data):
            msg = (
                f"GraphExplainer requires a torch_geometric.data.Data object, "
                f"got {type(x).__name__!r}."
            )
            raise TypeError(msg)

        # Allow per-call overrides of explainer-level settings via kwargs.
        index: ValidMoebiusConverterIndices = kwargs.get("index", self._index)
        efficiency_routine: bool = kwargs.get("efficiency_routine", self._efficiency_routine)
        max_subset_size: int | None = kwargs.get("max_subset_size")

        game = GraphGame(
            model=self._model,
            x_graph=x,
            class_index=self._class_index,
            baseline_strategy=self._baseline_strategy,
            normalize=self._normalize,
            verbose=verbose,
        )
        explainer = GraphSHAPIQ(game=game, verbose=verbose)

        return self._run_graph_shapiq_approximation(
            explainer,
            index,
            efficiency_routine=efficiency_routine,
            max_subset_size=max_subset_size,
        )

    @override
    def explain(
            self,
            x: np.ndarray | Data | None = None,
            **kwargs: Any
    ) -> InteractionValues:
        """Explain a single graph prediction.

        Args:
            x: The input graph to explain as a
                :class:`~torch_geometric.data.Data` object.
            **kwargs: Additional keyword-only arguments forwarded to ``explain_function``.

        Returns:
            The interaction values for the graph.
        """
        if not isinstance(x, Data):
            msg = f"GraphExplainer requires a torch_geometric.data.Data object, got {type(x).__name__!r}."
            raise TypeError(msg)
        return self.explain_function(x=x, **kwargs)

    def _run_graph_shapiq_approximation(
        self,
        explainer: GraphSHAPIQ,
        index: ValidMoebiusConverterIndices = "k-SII",
        *,
        efficiency_routine: bool = True,
        max_subset_size: int | None = None,
    ) -> InteractionValues:
        """Approximate Shapley Interactions using GraphSHAP-IQ.

        Args:
            explainer: The GraphSHAPIQ explainer initialised from the game.
            index: The type of Shapley interaction index to compute.
            efficiency_routine: Whether to enforce the efficiency axiom. Defaults to ``True``.
            max_subset_size: Maximum subset size for the Möbius transform. When ``None``,
                the full neighbourhood size is used (i.e. ``explainer.max_size_neighbors``).
                Passed per-call via ``explain(**kwargs)``; not stored on the explainer since
                the appropriate value is graph-instance-specific.
        """
        moebius, interactions = explainer.explain(
            max_subset_size=max_subset_size,
            order=self._max_order,
            efficiency_routine=efficiency_routine,
            index=index,
        )

        if not isinstance(moebius, InteractionValues):
            err_msg = f"Expected InteractionValues, got {type(moebius)}"
            raise TypeError(err_msg)

        moebius.estimation_budget = explainer.last_n_model_calls
        moebius.estimated = False
        moebius.sparsify(threshold=SPARSIFY_THRESHOLD)
        interactions.estimation_budget = explainer.last_n_model_calls
        interactions.estimated = False
        interactions.sparsify(threshold=SPARSIFY_THRESHOLD)

        return interactions
