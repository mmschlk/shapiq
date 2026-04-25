"""Scatter (a.k.a. dependence) plot for :class:`~shapiq.InteractionValues`.

Plots the per-sample interaction value of a chosen interaction tuple against
the value of one feature. For first-order interactions this matches
``shap.plots.scatter``; for higher-order interactions the x-axis is restricted
to a single feature (selected from the interaction tuple).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shapiq.interaction_values import InteractionValues, aggregate_interaction_values

from .beeswarm import _get_red_blue_cmap
from .utils import abbreviate_feature_names, format_labels

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


__all__ = ["scatter_plot"]


def _resolve_feature(
    feature: int | str | np.integer,
    name_to_idx: dict[str, int],
    n_players: int,
) -> int:
    """Resolves a feature identifier (index or name) to an integer index."""
    if isinstance(feature, (int, np.integer)) and not isinstance(feature, bool):
        idx = int(feature)
        if not 0 <= idx < n_players:
            error_message = f"Feature index {idx} out of range [0, {n_players})."
            raise ValueError(error_message)
        return idx
    if isinstance(feature, str):
        if feature not in name_to_idx:
            error_message = f"Unknown feature name: {feature!r}."
            raise ValueError(error_message)
        return name_to_idx[feature]
    error_message = f"Feature identifier must be int or str, got {type(feature).__name__}."
    raise TypeError(error_message)


def _resolve_interaction(
    interaction: tuple[int, ...] | tuple[str, ...] | int | str | None,
    interaction_values_list: list[InteractionValues],
    name_to_idx: dict[str, int],
    n_players: int,
) -> tuple[int, ...]:
    """Resolves an ``interaction`` argument to a sorted tuple of feature indices."""
    if interaction is None:
        agg = aggregate_interaction_values(
            [abs(iv) for iv in interaction_values_list], aggregation="mean"
        )
        candidates = [(k, v) for k, v in agg.interactions.items() if len(k) >= 1]
        if not candidates:
            error_message = "No non-empty interactions available to plot."
            raise ValueError(error_message)
        candidates.sort(key=lambda kv: kv[1], reverse=True)
        return candidates[0][0]

    if isinstance(interaction, (int, np.integer, str)):
        return (_resolve_feature(interaction, name_to_idx, n_players),)

    if isinstance(interaction, tuple):
        resolved = tuple(
            sorted(_resolve_feature(f, name_to_idx, n_players) for f in interaction)
        )
        if len(resolved) == 0:
            error_message = "interaction tuple must contain at least one feature."
            raise ValueError(error_message)
        return resolved

    error_message = (
        f"interaction must be a tuple, int, str, or None. Got {type(interaction).__name__}."
    )
    raise TypeError(error_message)


def scatter_plot(
    interaction_values_list: list[InteractionValues],
    data: pd.DataFrame | np.ndarray,
    interaction: tuple[int, ...] | tuple[str, ...] | int | str | None = None,
    *,
    x_feature: int | str | None = None,
    color: int | str | None = None,
    feature_names: list[str] | None = None,
    abbreviate: bool = True,
    alpha: float = 0.8,
    dot_size: float = 16,
    jitter: float = 0.0,
    ax: Axes | None = None,
    show: bool = True,
) -> Axes | None:
    """Plots a scatter (dependence) plot of an interaction's per-sample value against one feature.

    Inspired by `SHAP's <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/scatter.html>`_
    ``shap.plots.scatter``. For a first-order interaction ``(i,)`` the x-axis is feature ``i``'s
    value across samples and the y-axis is its Shapley value. For higher-order interactions like
    ``(i, j)`` the x-axis is the value of a single feature in the interaction (selected via
    ``x_feature``, defaulting to the first feature in the sorted tuple) and the y-axis is the
    higher-order interaction value.

    Args:
        interaction_values_list: A non-empty list of :class:`~shapiq.InteractionValues` objects,
            one per sample row of ``data``.
        data: The feature values for the samples, as a ``pandas.DataFrame`` or 2D ``numpy`` array.
            Must have the same number of rows as ``interaction_values_list``.
        interaction: Identifies the interaction to plot. Accepts an ``int`` or ``str`` (treated
            as a main effect single-element tuple), a tuple of feature indices like ``(0, 2)``,
            or a tuple of feature names like ``("MedInc", "Latitude")``. If ``None``, the
            globally most important interaction (by mean absolute aggregated value) is selected.
            Defaults to ``None``.
        x_feature: For higher-order interactions, which feature in ``interaction`` to place on
            the x-axis. Must be a member of ``interaction``. Ignored for first-order
            interactions. Defaults to the first feature in the sorted interaction tuple.
        color: Feature index or name used to color the points (with a red-blue colormap and a
            colorbar). If ``None`` (default), all points are drawn in a neutral color and no
            colorbar is shown. ``NaN`` color values render gray.
        feature_names: Names of the features. Defaults to ``["F0", "F1", ...]``.
        abbreviate: Whether to abbreviate feature names for axis labels. Defaults to ``True``.
        alpha: Transparency of the points, in ``(0, 1]``. Defaults to ``0.8``.
        dot_size: Size of the scatter points. Defaults to ``16``.
        jitter: If positive, adds Gaussian jitter to the plotted x-values, scaled to
            ``jitter * std(x_vals)``. Useful for categorical or integer-valued features.
            Defaults to ``0.0`` (disabled).
        ax: ``matplotlib`` ``Axes`` object to plot on. If ``None``, a new figure and axes are
            created.
        show: Whether to call ``plt.show()`` at the end. If ``False``, returns the axes instead.
            Defaults to ``True``.

    Returns:
        The ``Axes`` object if ``show=False``, otherwise ``None``.

    Raises:
        ValueError: If inputs are inconsistent (empty list, length mismatch, unknown feature
            names or indices, an interaction tuple not present in the lookup, an out-of-tuple
            ``x_feature``, or invalid numeric parameters).
        TypeError: If ``data`` is not a DataFrame or ndarray, or if a feature identifier has an
            unsupported type.

    """
    if not isinstance(interaction_values_list, list) or len(interaction_values_list) == 0:
        error_message = "interaction_values_list must be a non-empty list."
        raise ValueError(error_message)
    if not isinstance(data, pd.DataFrame) and not isinstance(data, np.ndarray):
        error_message = f"data must be a pandas DataFrame or a numpy array. Got: {type(data)}."
        raise TypeError(error_message)
    if len(interaction_values_list) != len(data):
        error_message = (
            "Length of interaction_values_list must match number of rows in data."
        )
        raise ValueError(error_message)
    if alpha <= 0 or alpha > 1:
        error_message = "alpha must be between 0 and 1."
        raise ValueError(error_message)
    if dot_size <= 0:
        error_message = "dot_size must be a positive value."
        raise ValueError(error_message)
    if jitter < 0:
        error_message = "jitter must be non-negative."
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
    display_mapping = dict(enumerate(feature_names_display))

    interaction_tuple = _resolve_interaction(
        interaction, interaction_values_list, name_to_idx, n_players
    )
    if interaction_tuple not in interaction_values_list[0].interaction_lookup:
        error_message = (
            f"Interaction {interaction_tuple} not found in InteractionValues lookup."
        )
        raise ValueError(error_message)

    if len(interaction_tuple) == 1:
        x_idx = interaction_tuple[0]
    elif x_feature is None:
        x_idx = interaction_tuple[0]
    else:
        x_idx = _resolve_feature(x_feature, name_to_idx, n_players)
        if x_idx not in interaction_tuple:
            error_message = (
                f"x_feature {x_feature!r} must be a member of interaction {interaction_tuple}."
            )
            raise ValueError(error_message)

    color_idx: int | None = None
    if color is not None:
        color_idx = _resolve_feature(color, name_to_idx, n_players)

    x_numpy = (
        data.to_numpy(dtype=float) if isinstance(data, pd.DataFrame) else data.astype(float)
    )
    x_vals = x_numpy[:, x_idx]
    y_vals = np.array(
        [iv.dict_values[interaction_tuple] for iv in interaction_values_list], dtype=float
    )

    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, 5))
    fig: Figure = ax.get_figure()  # type: ignore[assignment]

    x_plot = x_vals
    if jitter > 0:
        std = float(np.nanstd(x_vals))
        if std > 0:
            rng = np.random.default_rng(0)
            x_plot = x_vals + rng.normal(0.0, jitter * std, size=x_vals.shape)

    n_samples = len(x_vals)
    if color_idx is None:
        ax.scatter(
            x_plot,
            y_vals,
            color="#1f77b4",
            s=dot_size,
            alpha=alpha,
            linewidth=0,
            rasterized=n_samples > 500,
        )
    else:
        c_vals = x_numpy[:, color_idx]
        nan_mask = np.isnan(c_vals)
        valid_mask = ~nan_mask

        if nan_mask.any():
            ax.scatter(
                x_plot[nan_mask],
                y_vals[nan_mask],
                color="#777777",
                s=dot_size,
                alpha=alpha * 0.5,
                linewidth=0,
                rasterized=n_samples > 500,
            )

        if valid_mask.any():
            valid_color_vals = c_vals[valid_mask]
            vmin = float(np.min(valid_color_vals))
            vmax = float(np.max(valid_color_vals))
            if vmin == vmax:
                vmin -= 1e-9
                vmax += 1e-9
            sc = ax.scatter(
                x_plot[valid_mask],
                y_vals[valid_mask],
                c=valid_color_vals,
                cmap=_get_red_blue_cmap(),
                vmin=vmin,
                vmax=vmax,
                s=dot_size,
                alpha=alpha,
                linewidth=0,
                rasterized=n_samples > 500,
            )
            cb = fig.colorbar(sc, ax=ax, aspect=80)
            cb.set_label(display_mapping[color_idx], size=11, labelpad=0)
            cb.ax.tick_params(labelsize=10, length=0)
            cb.outline.set_visible(False)  # type: ignore[union-attr]

    ax.axhline(0, color="#999999", linestyle="-", linewidth=1, zorder=1)
    ax.set_xlabel(display_mapping[x_idx], fontsize=12)
    if len(interaction_tuple) == 1:
        ax.set_ylabel("SHAP value", fontsize=12)
    else:
        label = format_labels(display_mapping, interaction_tuple)
        ax.set_ylabel(f"Interaction value: {label}", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if not show:
        return ax
    plt.show()
    return None
