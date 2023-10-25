from typing import Union

import numpy as np


class DummyGame:

    """Dummy game for testing purposes. When called, it returns the size of the coalition divided by
    the number of players.

    Args:
        n: The number of players.
        interaction: The interaction of the game as a tuple of player indices. Defaults to an empty
            tuple.

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        interaction: The interaction of the game as a tuple of player indices.
    """

    def __init__(self, n: int, interaction: Union[set, tuple] = ()):
        self.n = n
        self.N = set(range(self.n))
        self.interaction: tuple = tuple(sorted(interaction))

    def __call__(
            self,
            coalition: Union[set, tuple, np.ndarray, list[set], list[tuple]]
    ) -> Union[float, np.ndarray[float]]:
        """Returns the size of the coalition divided by the number of players plus the interaction
        term."""

        def _worth(coal):
            worth = len(coal) / self.n
            if set(self.interaction).issubset(set(coal)):
                worth += 1
            return worth

        singular_value = True if isinstance(coalition, (set, tuple)) else False
        # check if coalition is an array
        if isinstance(coalition, np.ndarray):
            # if it is an array the array has the player indices as entries this needs to be converted to a list of sorted tuples for example: [[0,1],[3,2]] -> [(0, 1), (2, 3)]
            coalition = [tuple(sorted(coal)) for coal in coalition]
        if singular_value:
            coalition = tuple(sorted(coalition))
            return _worth(coalition)
        else:
            values = []
            for coal in coalition:
                coal = tuple(sorted(coal))
                values.append(_worth(coal))
        return np.asarray(values)

    def __repr__(self):
        return f"DummyGame(n={self.n}, interaction={self.interaction})"

    def __str__(self):
        return f"DummyGame(n={self.n}, interaction={self.interaction})"

    def __eq__(self, other):
        return self.n == other.n and self.interaction == other.interaction

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.n, self.interaction))

    def __copy__(self):
        return DummyGame(n=self.n, interaction=self.interaction)

    def __deepcopy__(self, memo):
        return DummyGame(n=self.n, interaction=self.interaction)

    def __getstate__(self):
        return {'n': self.n, 'interaction': self.interaction}

    def __setstate__(self, state):
        self.n = state['n']
        self.interaction = state['interaction']
        self.N = set(range(self.n))
