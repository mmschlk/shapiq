# How the orders demo was computed

These scripts computed the data of the [orders demo](https://shapiq.readthedocs.io/en/latest/introduction/how-much-explained.html)
(`../assets/data.js`): one prediction explained at every order of interaction, from the Shapley
values to the full decomposition. They are run by hand, once; the docs only show their result.

Every order is FSII, the Faithful Shapley Interactions (Tsai et al., 2023), which shapiq's
`MoebiusConverter` computes from each example's Möbius transform: FSII of order 1 is the Shapley
values, FSII of the highest order the Möbius transform. Every order is scored by its R^2 beyond size
under the Shapley distribution, the one FSII is faithful to: a coalition size s from 1 to n - 1
with probability proportional to 1 / (s (n - s)), the coalitions of a size equally likely, and no
weight on the empty and the full coalition, which FSII reproduces exactly. R^2 beyond size weighs
the explanation's errors against the spread of the game's values among coalitions of the same size,
not against the spread of all of them: how much of what the number of players alone does not
predict the explanation reproduces (`faithfulness.py`). FSII of a higher order fits the game at
least as well under it: for the examples explained exactly, the R^2 does not fall with the order. SigLIP 2 is explained through a proxy, to which FSII is faithful, and scored against the
model itself: its R^2 rises to 0.91 at order 4 and stays there, up to the full decomposition,
which falls short of 1 by the estimation gap the demo shades.

| File | What it does |
|---|---|
| `faithfulness.py` | What the other scripts share: the Shapley distribution (its sizes' probabilities, all its coalitions of 8 players with their probabilities, and coalitions drawn from it), an explanation's prediction of a coalition (the sum of its subsets' values, the empty one's included), the weighted R^2 beyond size, and FSII of an order. |
| `molecule.py` | Trains a GCN on Mutagenicity and explains the molecule of GraphSHAP-IQ's Figure 1 exactly, at every order, with each order's R^2 on 10,000 coalitions drawn from the Shapley distribution (their values from the exact Möbius transform): `results/molecule.json`. |
| `housing.py` | Trains an XGBoost model of depth 5 on California housing and explains one house in Los Angeles (test house 837) and 2,000 random test houses exactly (interventional), at every order, with each order's R^2 exactly over all 254 coalitions but the empty and the full one, weighted by the Shapley distribution: `results/house.json`, `results/houses.json` (with the 2,000 houses' features, of which the demo's map uses the latitude and the longitude). |
| `house_tabpfn.py` | Explains the same house with TabPFN, a black box, exactly over all 256 subsets of its features with shapiq's `TabPFNImputer` (remove and contextualize), at every order, with the R^2 of `housing.py`: `results/house_tabpfn.json`. TabPFN makes one pass without randomness, so that the game is the same on every run. |
| `fixlip.py` | Explains how SigLIP 2 matches the FIxLIP demo's elephant and caption with ProxySHAP (an XGBoost proxy of depth 6, no adjustment, read off its trees as a Möbius transform) from 10,000 coalitions drawn from the Shapley distribution, at every order, scored on 512 more drawn from it: `results/fixlip.json`, the evaluations in `results/fixlip_coalitions.npz`. With `--photo`, another photo of the FIxLIP demo, in the same way (the demo's giraffe and sign): `results/fixlip_<photo>.json`. |
| `pack.py` | Writes the demo's data, `../assets/data.js`: the index (FSII), the orders 1 to 4 and the full decomposition (order n), with the 300 strongest interactions of each (for the molecule, with the centers of its groups where shapiq's SI graph plot places them; for the 2,000 houses, where each is on the map of California, and the 12 strongest features and groups of features of every order, each with its value for every house, as text, and the mean of its absolute values), and the R^2 of every order. It also writes the examples' words and their notes: what the demo's tip says, in a sentence, of some players and groups. With `--examples`, it packs only the examples named, and keeps the others as the data has them. |
| `california.json` | The outline of California on the houses' map: the state's mainland from Natural Earth's 1:50m states and provinces (public domain), simplified to 114 points. |

## Run

From this folder, with `shapiq` installed from the repository, with `torch`, `torch_geometric` and
RDKit (for the drawing of the molecule), XGBoost, scikit-learn and Woodelf (for the houses), and
the dependencies of the FIxLIP demo's scripts (for SigLIP 2, which `fixlip.py` imports from
`../../fixlip/compute`):

```bash
pip install -e "../../../../..[graph,tree,games]" rdkit xgboost scikit-learn tabpfn

python molecule.py  # downloads Mutagenicity; trains the GCN in about four minutes on four CPU cores
python housing.py   # downloads California housing; explains 4,128 houses in under a minute
python house_tabpfn.py  # downloads TabPFN's weights; evaluates the house's 256 subsets in four minutes
python fixlip.py    # downloads SigLIP 2; evaluates 10,512 coalitions in about nine minutes
python fixlip.py --photo giraffe-tree  # the other photos of "Describing images", as long each
python fixlip.py --photo juice-theory
python pack.py
```

To change one example without computing the others again, run its script and pack it alone: the
other examples stay as `../assets/data.js` has them (their words, the tabs' names and the models',
and their notes are written anew). The examples are `molecule`, `house`, `tabpfn`, `houses`,
`fixlip` (the elephant), and the other photos of its tab, `giraffe` and `sign`. The houses' model is the same on every run with the versions below, to the last
digit of the data:

```bash
python housing.py
python pack.py --examples house houses
```

`molecule.py` saves the trained GCN to `results/gcn.pt` and does not train it again; with it, a run
takes about 20 seconds. The intermediate results (`results/`) are not part of the repository.

The data of the docs was computed with Python 3.12, shapiq from the repository (after release
1.8.0), torch 2.14, torch_geometric 2.8, RDKit 2026.09, XGBoost 3.4, scikit-learn 1.9 and
woodelf-explainer 0.4. The GCN classifies 78.2 % of the test molecules correctly; GraphSHAP-IQ needs
7,693 evaluations of it. The XGBoost model reaches R^2 = 0.848 on the test houses.

The giraffe and the sign (`fixlip.py --photo`) were computed later, with torch 2.9 and transformers
5.3, with the elephant's proxy: its depth was not chosen anew for them. Their full decompositions
fit SigLIP 2 with R^2 = 0.78 and 0.76.

The photo of the FIxLIP example is the FIxLIP demo's (`../../fixlip/assets/elephant.jpg`): the
demo loads it from there. `fixlip.py` keeps the game's evaluations, so a run again only fits the
proxy. The proxy's depth, 6, is the best of 4 to 12 in a 5-fold cross-validation on the 10,000
coalitions by R^2 beyond size (0.917), which is not part of the script; the explanation keeps the
proxy's own value of the empty coalition, -0.008, where ProxySHAP gives the game's, 0. With the game's value, every order
would be off the proxy by this constant, and the R^2 would peak at order 3 and fall after it.
