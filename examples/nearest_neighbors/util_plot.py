"""Helper functions for plotting to be used by the notebooks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from matplotlib.axes import Axes

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_datasets(
    ax: Axes,
    X_train: npt.NDArray[np.floating],
    y_train: npt.NDArray[np.floating],
    X_test: npt.NDArray[np.floating] | None = None,
    y_test: npt.NDArray[np.floating] | None = None,
    title: str | None = None,
) -> None:
    """Plots train and test datasets in the same figure."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if title is not None:
        ax.set_title(title)
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=[colors[i] for i in y_train],
        label="Training Points",
        marker="o",
    )
    if X_test is not None and y_test is not None:
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=[colors[i] for i in y_test],
            label="Test Points",
            marker="x",
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[i],
            markersize=10,
            label=f"Class {i} (Train)",
        )
        for i in set(y_train)
    ]
    if y_test is not None:
        handles += [
            Line2D(
                [0],
                [0],
                marker="x",
                linewidth=0,
                color=colors[i],
                markerfacecolor=colors[i],
                markersize=10,
                label=f"Class {i} (Test)",
            )
            for i in set(y_train)
        ]
    ax.legend(handles=handles, loc="upper right", title="Data Points")

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
