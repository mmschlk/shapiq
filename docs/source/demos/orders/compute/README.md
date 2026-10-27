# How the orders demo was computed

These scripts computed the data of the [orders demo](https://shapiq.readthedocs.io/en/latest/demos/orders/index.html)
(`../assets/data.js`): one prediction explained at every order of interaction, from the Shapley
values to the Möbius transform. They are run by hand, once; the docs only show their result.

| File | What it does |
|---|---|
| `molecule.py` | Trains a GCN on Mutagenicity and explains the molecule of GraphSHAP-IQ's Figure 1 exactly, at every order, with each order's R^2: `results/molecule.json`. |
| `fixlip.py` | The illustrative explanation of the FIxLIP demo's elephant, from a toy game: `results/fixlip.json`. |
| `pack.py` | Writes the demo's data, `../assets/data.js`: the orders 1 to 4 and the Möbius transform, with the strongest interactions of each. |

## Run

From this folder, with `shapiq` installed from the repository, with `torch`, `torch_geometric` and
RDKit (for the drawing of the molecule):

```bash
pip install -e "../../../../..[graph]" rdkit

python molecule.py  # downloads Mutagenicity; trains the GCN in about four minutes on four CPU cores
python fixlip.py
python pack.py
```

`molecule.py` saves the trained GCN to `results/gcn.pt` and does not train it again. The
intermediate results (`results/`) are not part of the repository.

The data of the docs was computed with Python 3.12, shapiq from the repository (after release
1.8.0), torch 2.14, torch_geometric 2.8 and RDKit 2026.09. The GCN classifies 78.2 % of the test
molecules correctly; GraphSHAP-IQ needs 7,693 evaluations of it.

The photo of the FIxLIP example is the FIxLIP demo's (`../../fixlip/assets/elephant.jpg`): the
demo loads it from there.
