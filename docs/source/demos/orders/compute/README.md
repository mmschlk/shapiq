# How the orders demo was computed

These scripts computed the data of the [orders demo](https://shapiq.readthedocs.io/en/latest/demos/orders/index.html)
(`../assets/data.js`): one prediction explained at every order of interaction, from the Shapley
values to the full decomposition. They are run by hand, once; the docs only show their result.

| File | What it does |
|---|---|
| `molecule.py` | Trains a GCN on Mutagenicity and explains the molecule of GraphSHAP-IQ's Figure 1 exactly, at every order, with each order's R^2: `results/molecule.json`. |
| `housing.py` | Trains an XGBoost model of depth 5 on California housing and explains one house in Los Angeles and 150 random houses exactly (interventional), at every order: `results/house.json`, `results/houses.json`. |
| `house_tabpfn.py` | Explains the same house with TabPFN, a black box, exactly over all 256 subsets of its features with shapiq's `TabPFNImputer` (remove and contextualize), at every order: `results/house_tabpfn.json`. TabPFN makes one pass without randomness, so that the game is the same on every run. |
| `fixlip.py` | Explains how SigLIP 2 matches the FIxLIP demo's elephant and caption with ProxySHAP (an XGBoost proxy of depth 6, no adjustment) from 10,000 coalitions, every player present with probability 1/2 as in FIxLIP, at every order, scored on 512 held-out coalitions: `results/fixlip.json`, the evaluations in `results/fixlip_coalitions.npz`. |
| `pack.py` | Writes the demo's data, `../assets/data.js`: the orders 1 to 4 and the full decomposition (order n), with the 300 strongest interactions of each (for the molecule, with the centers of its groups where shapiq's SI graph plot places them; for the 150 houses, the strongest rows of their heatmap), and the R^2 of every order. |

## Run

From this folder, with `shapiq` installed from the repository, with `torch`, `torch_geometric` and
RDKit (for the drawing of the molecule), XGBoost, scikit-learn and Woodelf (for the houses), and
the dependencies of the FIxLIP demo's scripts (for SigLIP 2, which `fixlip.py` imports from
`../../fixlip/compute`):

```bash
pip install -e "../../../../..[graph,tree,games]" rdkit xgboost scikit-learn tabpfn

python molecule.py  # downloads Mutagenicity; trains the GCN in about four minutes on four CPU cores
python housing.py   # downloads California housing; explains 4,128 houses in about three minutes
python house_tabpfn.py  # downloads TabPFN's weights; evaluates the house's 256 subsets in four minutes
python fixlip.py    # downloads SigLIP 2; evaluates 10,512 coalitions in about ten minutes
python pack.py
```

`molecule.py` saves the trained GCN to `results/gcn.pt` and does not train it again. The
intermediate results (`results/`) are not part of the repository.

The data of the docs was computed with Python 3.12, shapiq from the repository (after release
1.8.0), torch 2.14, torch_geometric 2.8, RDKit 2026.09, XGBoost 3.4, scikit-learn 1.9 and
woodelf-explainer 0.4. The GCN classifies 78.2 % of the test molecules correctly; GraphSHAP-IQ needs
7,693 evaluations of it. The XGBoost model reaches R^2 = 0.848 on the test houses.

The photo of the FIxLIP example is the FIxLIP demo's (`../../fixlip/assets/elephant.jpg`): the
demo loads it from there. `fixlip.py` keeps the game's evaluations, so a run again only fits the
proxy; it picked the proxy's depth, 6, as the best of 3 to 8 in a 5-fold cross-validation on the
10,000 coalitions.
