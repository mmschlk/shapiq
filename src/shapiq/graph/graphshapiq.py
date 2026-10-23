"""GraphSHAP-IQ: the exact Shapley interactions of games that are local on a graph."""

from __future__ import annotations

import itertools
from functools import cached_property
from typing import TYPE_CHECKING, Literal, NamedTuple, get_args

import numpy as np
from scipy import sparse

from shapiq.game_theory.moebius_converter import MoebiusConverter, ValidMoebiusConverterIndices
from shapiq.interaction_values import InteractionValues

if TYPE_CHECKING:
    from shapiq.game import Game

    from .base import GraphModel

__all__ = ["GraphSHAPIQ", "GraphSHAPIQIndices"]

GraphSHAPIQIndices = Literal["Moebius", ValidMoebiusConverterIndices]
"""The indices GraphSHAP-IQ computes: the Möbius transform and the indices of
:class:`~shapiq.game_theory.moebius_converter.MoebiusConverter`."""

MAX_RECEPTIVE_FIELD_SIZE = 62
"""The largest receptive field GraphSHAP-IQ enumerates (its subsets are 64-bit masks)."""

_CHUNK_SIZE = 2**16
"""The number of coalitions packed, or evaluated by the game, at once."""


class _Block(NamedTuple):
    """The evaluated subsets of a maximal receptive field."""

    nodes: np.ndarray
    """The nodes of the receptive field, sorted."""
    masks: np.ndarray
    """The evaluated subsets as sorted bitmasks over ``nodes``."""
    rows: np.ndarray
    """The rows of the subsets in the coalition matrix."""


class _Gap(NamedTuple):
    """A coalition of the efficiency routine: a receptive field or an intersection of them."""

    row: int
    """The row of the coalition in the coalition matrix."""
    block: int
    """A block containing the coalition."""
    mask: int
    """The coalition as a bitmask over the nodes of the block."""
    smaller: list[int]
    """The positions of the smaller gaps contained in the coalition."""


def _subset_masks(n_bits: int, max_size: int) -> np.ndarray:
    """Return the sorted bitmasks of all subsets of ``n_bits`` bits with at most ``max_size``."""
    if max_size >= n_bits:
        return np.arange(1 << n_bits, dtype=np.int64)
    masks = [
        np.fromiter(
            (
                sum(1 << bit for bit in subset)
                for subset in itertools.combinations(range(n_bits), s)
            ),
            dtype=np.int64,
        )
        for s in range(max_size + 1)
    ]
    return np.sort(np.concatenate(masks))


def _pack(nodes: np.ndarray, masks: np.ndarray, n_players: int) -> np.ndarray:
    """Return the subsets ``masks`` of ``nodes`` as rows of bits over all players."""
    packed = np.empty((len(masks), (n_players + 7) // 8), dtype=np.uint8)
    shifts = np.arange(len(nodes))
    for start in range(0, len(masks), _CHUNK_SIZE):
        chunk = masks[start : start + _CHUNK_SIZE]
        rows = np.zeros((len(chunk), n_players), dtype=bool)
        rows[:, nodes] = (chunk[:, None] >> shifts) & 1
        packed[start : start + _CHUNK_SIZE] = np.packbits(rows, axis=1)
    return packed


def _membership(sets: list[tuple[int, ...]], n_players: int) -> sparse.csr_array:
    """Return the sets as rows of a sparse 0/1 matrix over the players."""
    lengths = [len(s) for s in sets]
    return sparse.csr_array(
        (
            np.ones(sum(lengths), dtype=np.int64),
            np.fromiter((node for s in sets for node in s), dtype=np.int64, count=sum(lengths)),
            np.concatenate([[0], np.cumsum(lengths)]),
        ),
        shape=(len(sets), n_players),
    )


def _subset_pairs(sets: list[tuple[int, ...]], n_players: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the positions ``(i, j)``, ``i != j``, of all pairs with ``sets[i]`` in ``sets[j]``."""
    membership = _membership(sets, n_players)
    overlap = sparse.coo_array(membership @ membership.T)
    lengths = np.array([len(s) for s in sets])
    is_subset = (overlap.row != overlap.col) & (overlap.data == lengths[overlap.row])
    return overlap.row[is_subset], overlap.col[is_subset]


def _intersection_closure(
    sets: list[tuple[int, ...]], size: int, n_players: int
) -> list[tuple[int, ...]]:
    """Add the intersections of more than ``size`` nodes of the sets until none is new.

    Returns:
        The sets and their intersections, sorted, smallest first.
    """
    closed = list(dict.fromkeys(sets))
    known = set(closed)
    new = closed
    while new:
        overlap = sparse.coo_array(_membership(new, n_players) @ _membership(closed, n_players).T)
        large = overlap.data > size  # the number of common nodes
        found = []
        for a, b in zip(overlap.row[large], overlap.col[large], strict=True):
            common = tuple(sorted(set(new[a]).intersection(closed[b])))
            if common not in known:
                known.add(common)
                found.append(common)
        closed += found
        new = found
    return sorted(closed, key=lambda s: (len(s), s))


def _moebius_transform(values: np.ndarray, masks: np.ndarray, n_bits: int) -> None:
    """Replace the values of the subsets ``masks`` by their Möbius transform, in place.

    ``masks`` is sorted and closed under taking subsets. Yates' algorithm subtracts, bit after bit,
    the value of every subset without the bit: ``n_bits * len(masks)`` operations instead of the
    ``3**n_bits`` of summing over all subsets of every subset.
    """
    if len(masks) == 1 << n_bits:  # all subsets, masks[i] == i
        for bit in range(n_bits):
            pairs = values.reshape(-1, 2, 1 << bit)
            pairs[:, 1] -= pairs[:, 0]
        return
    for bit in range(n_bits):
        has_bit = ((masks >> bit) & 1).astype(bool)
        values[has_bit] -= values[np.searchsorted(masks, masks[has_bit] ^ (1 << bit))]


class GraphSHAPIQ:
    r"""GraphSHAP-IQ computes the exact Shapley interactions of games that are local on a graph.

    The players of the game are the nodes of a :class:`~shapiq.graph.GraphModel`. The game is local
    if its Möbius transform vanishes on every coalition that is not contained in the receptive
    field of a single node, the nodes within ``n_hops`` hops. The output of a graph neural network with ``n_hops`` message
    passing layers, a linear global pooling, and a linear readout is local
    :cite:p:`Muschalik.2025`: the embedding of a node depends only on its receptive field.

    GraphSHAP-IQ evaluates the game on the subsets of the receptive fields only, computes their
    Möbius coefficients with Yates' algorithm, and converts them with
    :class:`~shapiq.game_theory.moebius_converter.MoebiusConverter` into any of its indices. This
    takes at most :math:`\sum_{i} 2^{|\mathcal{N}(i)|}` evaluations instead of :math:`2^n`.

    With ``max_interaction_size`` :math:`\lambda`, only the subsets of at most :math:`\lambda`
    nodes are evaluated, and the Möbius coefficients of larger coalitions are taken as zero. The
    efficiency routine also evaluates the receptive fields of more than :math:`\lambda` nodes and
    their intersections of more than :math:`\lambda` nodes. Each of them, in increasing size, gets
    as Möbius coefficient the part of its value that its subsets do not explain. Every larger
    coalition's coefficient is so assigned to exactly one of them, the smallest containing it,
    which keeps the values of a local game efficient. With the efficiency routine, the values are
    still exact for :math:`\lambda` one less than the largest receptive field (:attr:`exact`).

    Attributes:
        game: The game, whose players are the nodes.
        graph: The graph, which defines the receptive fields.
        max_interaction_size: The largest coalitions whose Möbius coefficients are computed.
        efficiency_routine: Whether the receptive fields beyond ``max_interaction_size`` are
            evaluated to keep the values efficient.

    Examples:
        >>> import numpy as np
        >>> from shapiq import Game
        >>> from shapiq.graph import GraphModel
        >>> class EdgeGame(Game):  # the number of edges with both of their nodes present
        ...     def value_function(self, coalitions):
        ...         return (coalitions[:, :-1] & coalitions[:, 1:]).sum(axis=1).astype(float)
        >>> graph = GraphModel(edge_index=np.array([[0, 1, 2], [1, 2, 3]]), n_hops=1)  # 0-1-2-3
        >>> graphshapiq = GraphSHAPIQ(EdgeGame(n_players=4, normalize=False), graph)
        >>> graphshapiq.budget  # the subsets of {0, 1, 2} and {1, 2, 3}, instead of 2**4
        12
        >>> shapley_values = graphshapiq("SV", order=1)
        >>> float(shapley_values[(0,)]), float(shapley_values[(1,)])
        (0.5, 1.0)
    """

    def __init__(
        self,
        game: Game,
        graph: GraphModel,
        *,
        max_interaction_size: int | None = None,
        efficiency_routine: bool = True,
    ) -> None:
        """Initialize GraphSHAP-IQ.

        Args:
            game: A game whose players are the nodes of the graph.
            graph: The graph, whose edges and ``n_hops`` define the receptive fields.
            max_interaction_size: The largest coalitions whose Möbius coefficients are computed.
                Defaults to ``None``, the largest receptive field, which computes exact values.
            efficiency_routine: Whether to evaluate the receptive fields larger than
                ``max_interaction_size`` to keep the values efficient. Defaults to ``True``.

        Raises:
            ValueError: If the graph has other nodes than the game has players, or
                ``max_interaction_size`` is not positive.
        """
        if graph.n_nodes != game.n_players:
            msg = f"The graph has {graph.n_nodes} nodes, the game {game.n_players} players."
            raise ValueError(msg)
        if max_interaction_size is not None and max_interaction_size < 1:
            msg = f"max_interaction_size must be positive or None, got {max_interaction_size}."
            raise ValueError(msg)
        self.game = game
        self.graph = graph
        self.max_interaction_size = max_interaction_size
        self.efficiency_routine = efficiency_routine

    @property
    def n_players(self) -> int:
        """The number of players (nodes)."""
        return self.game.n_players

    @property
    def max_receptive_field_size(self) -> int:
        """The number of nodes of the largest receptive field."""
        return self.graph.max_receptive_field_size

    @property
    def interaction_size(self) -> int:
        """The size of the largest coalitions whose Möbius coefficients are computed."""
        if self.max_interaction_size is None:
            return self.max_receptive_field_size
        return min(self.max_interaction_size, self.max_receptive_field_size)

    @property
    def exact(self) -> bool:
        """Whether the computed values are exact (for a game that is local on the graph)."""
        largest = self.max_receptive_field_size
        size = self.interaction_size
        return size >= largest or (self.efficiency_routine and size >= largest - 1)

    @property
    def budget(self) -> int:
        """The number of coalitions GraphSHAP-IQ evaluates."""
        return len(self._coalitions)

    @cached_property
    def _plan(self) -> tuple[np.ndarray, list[_Block], list[_Gap]]:
        """Enumerate the coalitions to evaluate, without evaluating the game.

        Returns:
            The distinct coalitions as a matrix, the blocks of the maximal receptive fields, and
            the gaps of the efficiency routine, smallest first.
        """
        n_players, size = self.n_players, self.interaction_size
        distinct = {tuple(field.tolist()) for field in self.graph.receptive_fields}
        fields = sorted(distinct, key=lambda field: (len(field), field))
        contained, _ = _subset_pairs(fields, n_players)
        maximal = [fields[i] for i in sorted(set(range(len(fields))) - set(contained.tolist()))]
        if (largest := max(len(field) for field in maximal)) > MAX_RECEPTIVE_FIELD_SIZE:
            msg = (
                f"GraphSHAPIQ supports receptive fields of up to {MAX_RECEPTIVE_FIELD_SIZE} "
                f"nodes, the graph has one of {largest} nodes."
            )
            raise ValueError(msg)
        gap_sets = []
        if self.efficiency_routine:
            gap_sets = _intersection_closure([f for f in fields if len(f) > size], size, n_players)

        nodes = [np.asarray(field, dtype=np.int64) for field in maximal]
        masks = [_subset_masks(len(block_nodes), size) for block_nodes in nodes]
        packed = [_pack(n, m, n_players) for n, m in zip(nodes, masks, strict=True)]
        packed += [_pack(np.asarray(g), np.array([(1 << len(g)) - 1]), n_players) for g in gap_sets]
        rows = np.concatenate(packed)
        keys = np.ascontiguousarray(rows).view(np.dtype((np.void, rows.shape[1]))).ravel()
        _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        coalitions = np.unpackbits(rows[first], axis=1, count=n_players).astype(bool)

        offsets = np.cumsum([0, *(len(m) for m in masks)])
        blocks = [
            _Block(n, m, inverse[offsets[b] : offsets[b + 1]])
            for b, (n, m) in enumerate(zip(nodes, masks, strict=True))
        ]
        n_blocks = len(blocks)
        container: dict[int, int] = {}
        smaller: list[list[int]] = [[] for _ in gap_sets]
        for small, large in zip(*_subset_pairs(maximal + gap_sets, n_players), strict=True):
            if small >= n_blocks and large < n_blocks:
                container.setdefault(int(small) - n_blocks, int(large))
            elif small >= n_blocks:
                smaller[int(large) - n_blocks].append(int(small) - n_blocks)
        gaps = []
        for position, gap_set in enumerate(gap_sets):
            block = container[position]
            local = np.searchsorted(blocks[block].nodes, np.asarray(gap_set))
            gaps.append(
                _Gap(
                    row=int(inverse[offsets[-1] + position]),
                    block=block,
                    mask=sum(1 << int(bit) for bit in local),
                    smaller=smaller[position],
                )
            )
        return coalitions, blocks, gaps

    @property
    def _coalitions(self) -> np.ndarray:
        return self._plan[0]

    @cached_property
    def moebius(self) -> InteractionValues:
        """The Möbius transform of the game on all evaluated coalitions.

        The game is evaluated on first access.
        """
        coalitions, blocks, gaps = self._plan
        values = np.concatenate(
            [
                np.asarray(self.game(coalitions[start : start + _CHUNK_SIZE]), dtype=float)
                for start in range(0, len(coalitions), _CHUNK_SIZE)
            ]
        ).reshape(-1)
        moebius = np.zeros(len(coalitions))
        for block in blocks:
            coefficients = values[block.rows]
            _moebius_transform(coefficients, block.masks, len(block.nodes))
            moebius[block.rows] = coefficients
        gap_values = np.zeros(len(gaps))
        for position, gap in enumerate(gaps):
            block = blocks[gap.block]
            inside = (block.masks & ~gap.mask) == 0
            explained = moebius[block.rows[inside]].sum() + gap_values[gap.smaller].sum()
            gap_values[position] = values[gap.row] - explained
            moebius[gap.row] = gap_values[position]
        sizes = coalitions.sum(axis=1)
        members = np.nonzero(coalitions)[1].tolist()  # row by row, sorted within a row
        starts = (np.cumsum(sizes) - sizes).tolist()
        lookup = {
            tuple(members[start : start + size]): i
            for i, (start, size) in enumerate(zip(starts, sizes.tolist(), strict=True))
        }
        return InteractionValues(
            values=moebius,
            index="Moebius",
            max_order=self.n_players,
            min_order=0,
            n_players=self.n_players,
            interaction_lookup=lookup,
            estimated=not self.exact,
            estimation_budget=len(coalitions),
            baseline_value=float(moebius[lookup[()]]),
        )

    def __call__(self, index: GraphSHAPIQIndices = "k-SII", order: int = 2) -> InteractionValues:
        """Compute the interaction values of the game.

        Args:
            index: The interaction index, ``"Moebius"`` or an index of
                :class:`~shapiq.game_theory.moebius_converter.MoebiusConverter`. Defaults to
                ``"k-SII"``.
            order: The highest interaction order. Defaults to ``2``.

        Returns:
            The interaction values, flagged as estimated unless they are :attr:`exact`.

        Raises:
            ValueError: If the index is not supported or the order is not positive.
        """
        if index not in get_args(GraphSHAPIQIndices):
            msg = f"GraphSHAPIQ supports the indices {get_args(GraphSHAPIQIndices)}, got {index!r}."
            raise ValueError(msg)
        if order < 1:
            msg = f"order must be positive, got {order}."
            raise ValueError(msg)
        if index == "Moebius":
            moebius = self.moebius
            values = moebius.get_n_order(min_order=0, max_order=order)
        else:
            values = MoebiusConverter(self.moebius)(index, order)
        values.estimated = not self.exact
        values.estimation_budget = self.budget
        return values
