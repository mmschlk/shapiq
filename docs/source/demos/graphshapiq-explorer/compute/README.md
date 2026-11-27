# How the figures of the GraphSHAP-IQ explorer's page were computed

The explorer's own data (`../assets/data.js`: the GNNs, the molecules and their explanations) was
computed by the scripts in [`../../graphshapiq-common/compute`](../../graphshapiq-common/compute/README.md).
The scripts in this folder compute what the figures of the page show beyond it, and write it to
`../assets/figures-data.js`. They are run by hand, once; the docs only show their result.

They need neither torch nor the datasets nor a trained model. They run the explorer's three GNNs
from `data.js` in numpy (the GNNs' batch norms are folded into their linear layers there), mask
atoms as the explorer does (a masked atom takes the mean atom of the question's training set), and
explain them with shapiq's `GraphSHAPIQ`.

| File | What it does |
|---|---|
| `gnn.py` | What the scripts share: the GNNs in numpy, the graph game as a shapiq `Game` (`GraphBaselineImputer` and `GraphModel`), GraphSHAP-IQ on it, the game on all 2^n maskings and its Möbius transform by brute force, and a molecule from a SMILES as the demos draw it. Run it to check it against `data.js`. |
| `pack_figures.py` | Merges every part in `parts/` into `../assets/figures-data.js`. |
| `parts/<name>.json` | What one figure script computed, as the figure `<name>` reads it (`window.SHAPIQ_GRAPHSHAPIQ_EXPLORER_FIGURES["<name>"]`). |

## Run

From this folder, with `shapiq` installed from the repository (numpy and scipy are enough):

```bash
pip install -e "../../../../.."

python gnn.py           # the check, about 15 seconds
python <figure>.py      # each writes parts/<figure>.json
python pack_figures.py  # writes ../assets/figures-data.js from all parts
```

`gnn.py` explains every molecule for every question with GraphSHAP-IQ and compares the result with
`data.js`: the same budgets, the log-odds within 1e-4, the Shapley values within 2e-4, and the
prediction on the all-baseline graph within 1e-3 (`data.js` was computed with the unfolded GNNs in
PyTorch, and rounds to four decimals). It also evaluates the GNN on all 2^n maskings of two
molecules and checks that the brute-force Möbius interactions equal GraphSHAP-IQ's inside the
receptive fields and vanish outside them (up to float32 noise, 4e-6).

`gnn.from_smiles` needs RDKit (`pip install rdkit`), for a molecule that is not one of the
explorer's. It draws the molecule as `../../graphshapiq-common/compute/pack.py` does, with its atoms
in RDKit's order for the SMILES, which is the order of the players.

## Add the data of a figure

1. Write `<figure>.py` in this folder. It imports what it needs from `gnn.py`, uses fixed seeds,
   and writes `parts/<figure>.json` with rounded values.
2. Run it, then `python pack_figures.py`.
3. The figure reads its part in `../assets/figures.js` as `c.part`.

The data of the docs was computed with Python 3.13, numpy 2.5, scipy 1.18 and shapiq from the
repository.
