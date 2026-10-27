"""Train the GraphSHAP-IQ demo's GNNs: one GIN per task of config.toml.

Every task's GNN is B-XAIC's GIN (see molecules.py), trained as B-XAIC's ``train_model.py``
trains it: Adam with a learning rate of 0.001 and a weight decay of 0.0001, batches of 64
molecules drawn with class-balanced weights, gradients clipped to a norm of 2, and early stopping
after 10 epochs (after 10 warm-up epochs) without a better weighted F1 score on the validation
molecules, for at most 100 epochs. The B-XAIC tasks use B-XAIC's first split (``split_0``), the
Mutagenicity task a random split of 80/10/10 (see molecules.py).

Usage::

    python train.py                  # every task of config.toml
    python train.py --tasks indole

The models go to ``results/models/<task>.pt``, their scores to ``results/models/<task>.json``;
models that exist are not trained again. A B-XAIC task takes 5 to 10 minutes on four CPU cores,
Mutagenicity a minute.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from molecules import GNN, load_bxaic, load_mutagenicity
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.loader import DataLoader

if TYPE_CHECKING:
    from torch_geometric.data import Data

HERE = Path(__file__).resolve().parent
SEED = 123  # B-XAIC's


@torch.no_grad()
def score(model: GNN, graphs: list[Data]) -> dict[str, float]:
    """Return the weighted F1 score and the accuracy of the model on the molecules."""
    model.eval()
    labels, predictions = [], []
    for batch in DataLoader(graphs, batch_size=512):
        predictions.append(model(batch.x, batch.edge_index, batch.batch).argmax(dim=1))
        labels.append(batch.y)
    y, y_hat = torch.cat(labels).numpy(), torch.cat(predictions).numpy()
    return {"f1": f1_score(y, y_hat, average="weighted"), "accuracy": accuracy_score(y, y_hat)}


def train(task: dict[str, Any], data: Path, models: Path) -> None:
    """Train a task's GNN on its dataset and save it with its scores."""
    torch.manual_seed(SEED)  # the GNN's weights and the batches
    if task["dataset"] == "B-XAIC":
        graphs, splits = load_bxaic(task["id"], data)
    else:
        graphs, splits = load_mutagenicity(data)
    split = {name: [g for g, s in zip(graphs, splits, strict=True) if s == name] for name in splits}
    labels = torch.cat([graph.y for graph in split["train"]])
    weights = (1.0 / torch.bincount(labels).float())[labels]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
    loader = DataLoader(split["train"], batch_size=64, sampler=sampler)

    model = GNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    best, best_state, waited = -1.0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, 101):
        start = time.time()
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            torch.nn.functional.cross_entropy(logits, batch.y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        valid = score(model, split["valid"])["f1"]
        if valid > best:
            best, best_state, waited = valid, copy.deepcopy(model.state_dict()), 0
        elif epoch > 10:
            waited += 1
        print(
            f"{task['id']}, epoch {epoch}: validation F1 {valid:.4f}, {time.time() - start:.0f} s"
        )
        if waited >= 10:
            break
    model.load_state_dict(best_state)
    scores = {name: score(model, split[name]) for name in ("train", "valid", "test")}
    scores["sizes"] = {name: len(split[name]) for name in ("train", "valid", "test")}
    models.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models / f"{task['id']}.pt")
    (models / f"{task['id']}.json").write_text(json.dumps(scores, indent=2))
    print(
        f"{task['id']}: test F1 {scores['test']['f1']:.3f}, accuracy {scores['test']['accuracy']:.3f}"
    )


def main() -> None:
    """Train the GNNs of the tasks of config.toml that are not trained yet."""
    config = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tasks", nargs="+", default=[t["id"] for t in config["tasks"]])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    torch.set_num_threads(4)
    for task in (t for t in config["tasks"] if t["id"] in args.tasks):
        if not (args.results / "models" / f"{task['id']}.pt").exists():
            train(task, args.results / "data", args.results / "models")


if __name__ == "__main__":
    main()
