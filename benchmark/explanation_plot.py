"""Tests the new explanation_plot function."""

import numpy as np
import matplotlib.pyplot as plt

from plot.explanation_graph import explanation_graph
from shapiq.interaction_values import InteractionValues

if __name__ == "__main__":

    example_values = InteractionValues(
        n_players=4,
        values=np.array(
            [
                0.0,  # ()
                -0.2,  # (1)
                0.2,  # (2)
                0.2,  # (3)
                -0.1,  # (4)
                0.2,  # (1, 2)
                -0.2,  # (1, 3)
                0.2,  # (1, 4)
                0.2,  # (2, 3)
                -0.2,  # (2, 4)
                0.2,  # (3, 4)
                0.4,  # (1, 2, 3)
                0.0,  # (1, 2, 4)
                0.0,  # (1, 3, 4)
                0.0,  # (2, 3, 4)
                -0.1,  # (1, 2, 3, 4)
            ],
            dtype=float,
        ),
        index="k-SII",
        interaction_lookup={
            (): 0,
            (1,): 1,
            (2,): 2,
            (3,): 3,
            (4,): 4,
            (1, 2): 5,
            (1, 3): 6,
            (1, 4): 7,
            (2, 3): 8,
            (2, 4): 9,
            (3, 4): 10,
            (1, 2, 3): 11,
            (1, 2, 4): 12,
            (1, 3, 4): 13,
            (2, 3, 4): 14,
            (1, 2, 3, 4): 15,
        },
        baseline_value=0,
        min_order=0,
        max_order=4,
    )

    fig, ax = explanation_graph(
        example_values, random_seed=1, size_factor=0.7, plot_explanation=False, weight_factor=1.1
    )
    plt.show()
