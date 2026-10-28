# How the orders demo was computed

These scripts computed the data of the [orders demo](https://shapiq.readthedocs.io/en/latest/demos/orders/index.html)
(`../assets/data.js`): one prediction explained at every order of interaction, from the Shapley
values to the full decomposition. They are run by hand, once; the docs only show their result.

Every order is FSII, the Faithful Shapley Interactions (Tsai et al., 2023), which shapiq's
`MoebiusConverter` computes from each example's Möbius transform: FSII of order 1 is the Shapley
values, FSII of the highest order the Möbius transform. Every order is scored by its R^2 under the
Shapley distribution, the one FSII is faithful to: a coalition size s from 1 to n - 1 with
probability proportional to 1 / (s (n - s)), the coalitions of a size equally likely, and no weight
on the empty and the full coalition, which FSII reproduces exactly. Under it, the R^2 does not fall
with the order.

| File | What it does |
|---|---|
| `faithfulness.py` | What the other scripts share: the Shapley distribution (its sizes' probabilities, all its coalitions of 8 players with their probabilities, and coalitions drawn from it), an explanation's prediction of a coalition (the sum of its subsets' values, the empty one's included), the weighted R^2, and FSII of an order. |
| `molecule.py` | Trains a GCN on Mutagenicity and explains the molecule of GraphSHAP-IQ's Figure 1 exactly, at every order, with each order's R^2 on 10,000 coalitions drawn from the Shapley distribution (their values from the exact Möbius transform): `results/molecule.json`. |
| `housing.py` | Trains an XGBoost model of depth 5 on California housing and explains one house in Los Angeles (test house 837) and 150 random houses exactly (interventional), at every order, with each order's R^2 exactly over all 254 coalitions but the empty and the full one, weighted by the Shapley distribution: `results/house.json`, `results/houses.json`. |
| `house_tabpfn.py` | Explains the same house with TabPFN, a black box, exactly over all 256 subsets of its features with shapiq's `TabPFNImputer` (remove and contextualize), at every order, with the R^2 of `housing.py`: `results/house_tabpfn.json`. TabPFN makes one pass without randomness, so that the game is the same on every run. |
| `fixlip.py` | Explains how SigLIP 2 matches the FIxLIP demo's elephant and caption with ProxySHAP (an XGBoost proxy of depth 6, no adjustment) from 10,000 coalitions drawn from the Shapley distribution, at every order, scored on 512 more drawn from it: `results/fixlip.json`, the evaluations in `results/fixlip_coalitions.npz`. |
| `pack.py` | Writes the demo's data, `../assets/data.js`: the index (FSII), the orders 1 to 4 and the full decomposition (order n), with the 300 strongest interactions of each (for the molecule, with the centers of its groups where shapiq's SI graph plot places them; for the 150 houses, the strongest rows of their heatmap), and the R^2 of every order. |

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
