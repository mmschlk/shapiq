"""Two-feature scatter plot for pairwise :class:`~shapiq.InteractionValues`.

Places the values of two features on the x- and y-axis and colors each sample by
the value of their pairwise interaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shapiq.interaction_values import aggregate_interaction_values

from .beeswarm import _get_red_blue_cmap
from .scatter import _resolve_feature
from .utils import abbreviate_feature_names

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from shapiq.interaction_values import InteractionValues


__all__ = ["scatter_2d_plot"]


def _resolve_pair(
    interaction: tuple[int, int] | tuple[str, str] | None,
    interaction_values_list: list[InteractionValues],
    name_to_idx: dict[str, int],
    n_players: int,
) -> tuple[int, int]:
    """Resolves an ``interaction`` argument to a sorted pair of feature indices."""
    if interaction is None:
        agg = aggregate_interaction_values(
            [abs(iv) for iv in interaction_values_list], aggregation="mean"
        )
        candidates = [(k, v) for k, v in agg.interactions.items() if len(k) == 2]
        if candidates:
            candidates.sort(key=lambda kv: kv[1], reverse=True)
            first, second = candidates[0][0]
            return first, second
        # first-order explanations: use the two features with the largest mean absolute value
        singletons = [(k[0], v) for k, v in agg.interactions.items() if len(k) == 1]
        if len(singletons) < 2:
            error_message = "No pairwise interactions or first-order values available to plot."
            raise ValueError(error_message)
        singletons.sort(key=lambda kv: kv[1], reverse=True)
        first, second = sorted((singletons[0][0], singletons[1][0]))
        return first, second

    if not isinstance(interaction, tuple):
        error_message = f"interaction must be a tuple of two features or None. Got {type(interaction).__name__}."
        raise TypeError(error_message)
    resolved = tuple(sorted({_resolve_feature(f, name_to_idx, n_players) for f in interaction}))
    if len(resolved) != 2:
        error_message = (
            f"interaction must contain exactly two different features. Got {interaction}."
        )
        raise ValueError(error_message)
    first, second = resolved
    return first, second


def scatter_2d_plot(
    interaction_values_list: list[InteractionValues],
    data: pd.DataFrame | np.ndarray,
    interaction: tuple[int, int] | tuple[str, str] | None = None,
    *,
    include_main_effects: bool = False,
    feature_names: list[str] | None = None,
    abbreviate: bool = True,
    alpha: float = 0.8,
    dot_size: float = 16,
    ax: Axes | None = None,
    show: bool = True,
) -> Axes | None:
    """Plots two features against each other, colored by their pairwise interaction value.

    Each point is one sample of ``data``: its x-coordinate is the value of the first feature
    of ``interaction``, its y-coordinate is the value of the second feature and its color is
    the sample's interaction value for the pair. The colormap is centered at zero, so positive
    values are red and negative values are blue.

    For first-order explanations (for example Shapley values), which have no pairwise
    interactions, the color is the sum of the two features' values. For second-order
    explanations, ``include_main_effects=True`` adds both features' first-order values to the
    pairwise interaction, so the color shows the joint contribution of the pair.

    Args:
        interaction_values_list: A non-empty list of :class:`~shapiq.InteractionValues` objects,
            one per sample row of ``data``.
        data: The feature values for the samples, as a ``pandas.DataFrame`` or 2D ``numpy`` array.
            Must have the same number of rows as ``interaction_values_list``.
        interaction: The pair of features to plot, as a tuple of feature indices like ``(0, 2)``
            or of feature names like ``("MedInc", "Latitude")``. The lower feature index goes on
            the x-axis. If ``None``, the pairwise interaction with the highest mean absolute value
            is selected (for first-order explanations, the two features with the highest mean
            absolute value). Defaults to ``None``.
        include_main_effects: For second-order explanations, whether to add the first-order
            values of both features to the pairwise interaction. Has no effect for first-order
            explanations, which are always shown as the sum of the two first-order values.
            Defaults to ``False``.
        feature_names: Names of the features. Defaults to ``["F0", "F1", ...]``.
        abbreviate: Whether to abbreviate feature names for axis labels. Defaults to ``True``.
        alpha: Transparency of the points, in ``(0, 1]``. Defaults to ``0.8``.
        dot_size: Size of the scatter points. Defaults to ``16``.
        ax: ``matplotlib`` ``Axes`` object to plot on. If ``None``, a new figure and axes are
            created.
        show: Whether to call ``plt.show()`` at the end. If ``False``, returns the axes instead.
            Defaults to ``True``.

    Returns:
        The ``Axes`` object if ``show=False``, otherwise ``None``.

    Raises:
        ValueError: If inputs are inconsistent (empty list, length mismatch, unknown feature
            names or indices, an interaction that is not a pair or is absent from every
            sample's lookup, or invalid numeric parameters).
        TypeError: If ``data`` is not a DataFrame or ndarray, or if ``interaction`` or a feature
            identifier has an unsupported type.

    """
    if not isinstance(interaction_values_list, list) or len(interaction_values_list) == 0:
        error_message = "interaction_values_list must be a non-empty list."
        raise ValueError(error_message)
    if not isinstance(data, pd.DataFrame) and not isinstance(data, np.ndarray):
        error_message = f"data must be a pandas DataFrame or a numpy array. Got: {type(data)}."
        raise TypeError(error_message)
    if len(interaction_values_list) != len(data):
        error_message = "Length of interaction_values_list must match number of rows in data."
        raise ValueError(error_message)
    if alpha <= 0 or alpha > 1:
        error_message = "alpha must be between 0 and 1."
        raise ValueError(error_message)
    if dot_size <= 0:
        error_message = "dot_size must be a positive value."
        raise ValueError(error_message)

    n_players = interaction_values_list[0].n_players

    if feature_names is None:
        feature_names_full = [f"F{i}" for i in range(n_players)]
    else:
        if len(feature_names) != n_players:
            error_message = "Length of feature_names must match n_players."
            raise ValueError(error_message)
        feature_names_full = list(feature_names)

    feature_names_display = (
        abbreviate_feature_names(feature_names_full) if abbreviate else list(feature_names_full)
    )
    name_to_idx = {n: i for i, n in enumerate(feature_names_full)}

    x_idx, y_idx = _resolve_pair(interaction, interaction_values_list, name_to_idx, n_players)
    first_order = all(iv.max_order < 2 for iv in interaction_values_list)
    if first_order:
        for feature in (x_idx, y_idx):
            if not any((feature,) in iv.interaction_lookup for iv in interaction_values_list):
                error_message = f"Feature {feature} not found in InteractionValues lookup."
                raise ValueError(error_message)
    elif not any((x_idx, y_idx) in iv.interaction_lookup for iv in interaction_values_list):
        error_message = f"Interaction {(x_idx, y_idx)} not found in InteractionValues lookup."
        raise ValueError(error_message)

    x_numpy = data.to_numpy(dtype=float) if isinstance(data, pd.DataFrame) else data.astype(float)
    x_vals = x_numpy[:, x_idx]
    y_vals = x_numpy[:, y_idx]
    if first_order:
        c_vals = np.array(
            [iv[(x_idx,)] + iv[(y_idx,)] for iv in interaction_values_list], dtype=float
        )
    elif include_main_effects:
        c_vals = np.array(
            [iv[(x_idx,)] + iv[(y_idx,)] + iv[(x_idx, y_idx)] for iv in interaction_values_list],
            dtype=float,
        )
    else:
        c_vals = np.array([iv[(x_idx, y_idx)] for iv in interaction_values_list], dtype=float)

    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, 5))
    fig: Figure = ax.get_figure()  # type: ignore[assignment]

    limit = float(np.nanmax(np.abs(c_vals))) if np.isfinite(c_vals).any() else 0.0
    if limit == 0:
        limit = 1e-9
    sc = ax.scatter(
        x_vals,
        y_vals,
        c=c_vals,
        cmap=_get_red_blue_cmap(),
        vmin=-limit,
        vmax=limit,
        s=dot_size,
        alpha=alpha,
        linewidth=0,
        rasterized=len(x_vals) > 500,
    )

    index_name = interaction_values_list[0].index
    x_name, y_name = feature_names_display[x_idx], feature_names_display[y_idx]
    main_effects_label = f"{index_name}({x_name}) + {index_name}({y_name})"
    pair_label = f"{index_name}({x_name}, {y_name})"
    if first_order:
        color_label = main_effects_label
    elif include_main_effects:
        color_label = f"{main_effects_label} + {pair_label}"
    else:
        color_label = pair_label
    cb = fig.colorbar(sc, ax=ax, aspect=80)
    cb.set_label(color_label, size=11, labelpad=0)
    cb.ax.tick_params(labelsize=10, length=0)
    cb.outline.set_visible(False)

    ax.set_xlabel(feature_names_display[x_idx], fontsize=12)
    ax.set_ylabel(feature_names_display[y_idx], fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if not show:
        return ax
    plt.show()
    return None
