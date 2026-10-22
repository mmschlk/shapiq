"""Graph classification datasets of the TUDataset collection, for the graph games."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

from shapiq_games._optional import require

from ._cache import get_data_dir

if TYPE_CHECKING:
    from torch_geometric.data import Data

__all__ = ["GraphDatasetName", "load_graph_dataset"]

GraphDatasetName = Literal["MUTAG", "BZR", "COX2", "DHFR", "AIDS", "PROTEINS", "ENZYMES"]
"""The graph classification datasets of :func:`load_graph_dataset`: molecules (``"MUTAG"``,
``"BZR"``, ``"COX2"``, ``"DHFR"``, ``"AIDS"``) and proteins (``"PROTEINS"``, ``"ENZYMES"``)."""


def load_graph_dataset(name: GraphDatasetName) -> list[Data]:
    """Load a graph classification dataset of the TUDataset collection :cite:p:`Morris.2020`.

    The first call downloads the dataset with ``torch_geometric`` from the collection's site and
    caches it in ``<data dir>/datasets/tudataset/`` (see :func:`get_data_dir`). The node features
    are the one-hot node labels (e.g. the atom types of a molecule), the targets the graph classes.

    Args:
        name: The name of the dataset.

    Returns:
        The graphs with node features ``x``, edges ``edge_index``, and class ``y`` of shape
        ``(1,)``.

    Raises:
        ValueError: If the dataset is unknown.
        ImportError: If ``torch_geometric`` is not installed.
    """
    if name not in get_args(GraphDatasetName):
        msg = f"Unknown graph dataset {name!r}. Available: {', '.join(get_args(GraphDatasetName))}."
        raise ValueError(msg)
    datasets = require("torch_geometric.datasets", purpose="the graph datasets", extra="benchmark")
    dataset = datasets.TUDataset(root=str(get_data_dir() / "datasets" / "tudataset"), name=name)
    return [dataset[i] for i in range(len(dataset))]
