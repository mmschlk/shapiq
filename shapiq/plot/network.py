"""This module contains the network plots for the shapiq package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from PIL.Image import Image

    from shapiq.interaction_values import InteractionValues

__all__ = ["network_plot"]


def network_plot(
    interaction_values: InteractionValues | None = None,
    *,
    feature_names: list[Any] | dict[int, Any] | None = None,
    feature_image_patches: dict[int, Image] | list[Image] | None = None,
    feature_image_patches_size: float | dict[int, float] | None = 0.2,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes] | None:
    """Draws the interaction network plot[1]_.

    An interaction network is a graph where the nodes represent the features and the edges represent
    the interactions. The edge width is proportional to the interaction value. The color of the edge
    is red if the interaction value is positive and blue if the interaction value is negative. The
    network plot has been used to visualize local Shapley interaction values[1]_ and is a variation
    of the graph plots presented by Inglis et al. (2022)[2]_. Below is an example of an interaction
    network with an image in the center.

    .. image:: /_static/network_example.png
        :width: 400
        :align: center

    Args:
        interaction_values: The interaction values as an interaction object.

        feature_names: The feature names used for plotting. List/dict mapping index of the player as
            index/key to name. If no feature names are provided, the feature indices are used
            instead. Defaults to ``None``.

        feature_image_patches: A dictionary/list containing the image patches to be displayed
            instead of the feature labels in the network. The keys/indices of the list are the
            feature indices and the values are the feature images. If explicit feature names are
            provided, they are displayed on top of the image. Defaults to ``None``.

        feature_image_patches_size: The size of the feature image patches. Defaults to ``0.2``.

        show: Whether to show the plot. Defaults to ``False``. If ``False``, the figure and the axis
            containing the plot are returned, otherwise ``None``.

    Returns:
        The figure and the axis containing the plot if ``show=False``.

    References:
        .. [1] Muschalik, M., Fumagalli, F., Hammer, B., & Hüllermeier, E. (2024). Beyond TreeSHAP: Efficient Computation of Any-Order Shapley Interactions for Tree Ensembles. Proceedings of the AAAI Conference on Artificial Intelligence, 38(13), 14388-14396. https://doi.org/10.1609/aaai.v38i13.29352

        .. [2] Inglis, A.; Parnell, A.; and Hurley, C. B. 2022. Visualizing Variable Importance and Variable Interaction Effects in Machine Learning Models. Journal of Computational and Graphical Statistics, 31(3): 766-778.

    """
    from . import si_graph_plot

    fig, ax = si_graph_plot(
        interaction_values=interaction_values,
        feature_names=feature_names,
        feature_image_patches=feature_image_patches,
        feature_image_patches_size=feature_image_patches_size,
        show=False,
        min_max_order=(1, 2),
    )
    if not show:
        return fig, ax
    plt.show()
    return None
