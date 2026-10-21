"""Two-feature scatter plot for pairwise :class:`~shapiq.InteractionValues`.

Places the values of two features on the x- and y-axis and colors each sample by
the value of their pairwise interaction.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .beeswarm import _get_red_blue_cmap
from .scatter import _resolve_feature
from .utils import _as_batch, abbreviate_feature_names

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from shapiq.interaction_values import InteractionValues, InteractionValuesBatch


__all__ = ["scatter_2d_plot"]


def _resolve_pair(
    interaction: tuple[int, int] | tuple[str, str] | None,
    batch: InteractionValuesBatch,
    name_to_idx: dict[str, int],
) -> tuple[int, int]:
    """Resolves an ``interaction`` argument to the feature indices on the x- and y-axis."""
    if interaction is None:
        values = batch.values  # noqa: PD011
        importance = {k: float(np.mean(np.abs(v))) for k, v in values.items()}
        candidates = [k for k in importance if len(k) == 2]
        if not candidates:
            # first-order explanations: use the two features with the largest mean absolute value
            candidates = sorted(
                [k for k in importance if len(k) == 1], key=importance.__getitem__, reverse=True
            )
            if len(candidates) < 2:
                error_message = "No pairwise interactions or first-order values available to plot."
                raise ValueError(error_message)
            first, second = sorted((candidates[0][0], candidates[1][0]))
            return first, second
        first, second = max(candidates, key=importance.__getitem__)
        return first, second

    if not isinstance(interaction, tuple):
        error_message = f"interaction must be a tuple of two features or None. Got {type(interaction).__name__}."
        raise TypeError(error_message)
    resolved = [_resolve_feature(f, name_to_idx, batch.n_players) for f in interaction]
    if len(resolved) != 2 or resolved[0] == resolved[1]:
        error_message = (
            f"interaction must contain exactly two different features. Got {interaction}."
        )
        raise ValueError(error_message)
    first, second = resolved
    return first, second


def scatter_2d_plot(
    interaction_values_list: InteractionValuesBatch | Sequence[InteractionValues],
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
    values are red and negative values are blue. Points with larger absolute values are drawn on
    top.

    For first-order explanations (for example Shapley values), which have no pairwise
    interactions, the color is the sum of the two features' values. For second-order
    explanations, ``include_main_effects=True`` adds both features' first-order values to the
    pairwise interaction, so the color shows the joint contribution of the pair.

    Args:
        interaction_values_list: The explanations of the samples in ``data``, as an
            :class:`~shapiq.InteractionValuesBatch` (for example from
            :meth:`~shapiq.Explainer.explain_X`) or a list of
            :class:`~shapiq.InteractionValues` objects, one per sample row of ``data``.
        data: The feature values for the samples, as a ``pandas.DataFrame`` or 2D ``numpy`` array.
            Must have the same number of rows as ``interaction_values_list``. Only the two
            plotted columns need to be numeric.
        interaction: The pair of features to plot, as a tuple of feature indices like ``(0, 2)``
            or of feature names like ``("MedInc", "Latitude")``. The first feature goes on the
            x-axis and the second on the y-axis. If ``None``, the pairwise interaction with the
            highest mean absolute value is selected (for first-order explanations, the two
            features with the highest mean absolute value), with the lower feature index on the
            x-axis. Defaults to ``None``.
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
        ValueError: If inputs are inconsistent (no explanations, length mismatch, unknown feature
            names or indices, an interaction that is not a pair or is absent from the
            explanations, or invalid numeric parameters).
        TypeError: If ``interaction_values_list`` or ``data`` has an unsupported type, or if
            ``interaction`` or a feature identifier has an unsupported type.

    """
    batch = _as_batch(interaction_values_list)
    if not isinstance(data, pd.DataFrame) and not isinstance(data, np.ndarray):
        error_message = f"data must be a pandas DataFrame or a numpy array. Got: {type(data)}."
        raise TypeError(error_message)
    if len(batch) != len(data):
        error_message = "Length of interaction_values_list must match number of rows in data."
        raise ValueError(error_message)
    if alpha <= 0 or alpha > 1:
        error_message = "alpha must be between 0 and 1."
        raise ValueError(error_message)
    if dot_size <= 0:
        error_message = "dot_size must be a positive value."
        raise ValueError(error_message)

    n_players = batch.n_players

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

    x_idx, y_idx = _resolve_pair(interaction, batch, name_to_idx)
    values = batch.values  # noqa: PD011
    pair = (min(x_idx, y_idx), max(x_idx, y_idx))
    first_order = batch.max_order < 2
    if first_order:
        for feature in pair:
            if (feature,) not in values:
                error_message = f"Feature {feature} not found in InteractionValues lookup."
                raise ValueError(error_message)
    elif pair not in values:
        error_message = f"Interaction {pair} not found in InteractionValues lookup."
        raise ValueError(error_message)
    if include_main_effects and not first_order and batch.min_order > 1:
        warnings.warn(
            "include_main_effects=True has no effect, since the explanations contain no "
            "first-order values (min_order > 1).",
            stacklevel=2,
        )
        include_main_effects = False

    zeros = np.zeros(len(batch))
    main_effects = values.get((x_idx,), zeros) + values.get((y_idx,), zeros)
    if first_order:
        c_vals = main_effects
    elif include_main_effects:
        c_vals = main_effects + values[pair]
    else:
        c_vals = values[pair]
    c_vals = np.asarray(c_vals, dtype=float)

    columns = data.to_numpy() if isinstance(data, pd.DataFrame) else data
    x_vals = columns[:, x_idx].astype(float)
    y_vals = columns[:, y_idx].astype(float)

    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, 5))
    fig: Figure = ax.get_figure()  # type: ignore[assignment]

    limit = float(np.nanmax(np.abs(c_vals))) if np.isfinite(c_vals).any() else 0.0
    if limit == 0:
        limit = 1e-9
    # draw the points with the largest absolute values last, so they are not hidden
    order = np.argsort(np.abs(c_vals))
    sc = ax.scatter(
        x_vals[order],
        y_vals[order],
        c=c_vals[order],
        cmap=_get_red_blue_cmap(),
        vmin=-limit,
        vmax=limit,
        s=dot_size,
        alpha=alpha,
        linewidth=0,
        rasterized=len(x_vals) > 500,
    )

    index_name = batch.index
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
    cb.outline.set_visible(False)  # type: ignore[union-attr]

    ax.set_xlabel(x_name, fontsize=12)
    ax.set_ylabel(y_name, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if not show:
        return ax
    plt.show()
    return None
