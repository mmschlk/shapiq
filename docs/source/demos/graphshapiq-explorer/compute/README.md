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

**`pipeline.py`, `nitro.py`, `maskings.py`** (figures `pipeline`, `nitro`, `maskings`; each for
every molecule and every question, under 15 seconds together). `pipeline.py` runs each GNN on the
unmasked molecule and writes, for one atom per molecule (`FOCUS`), its node features and its
embedding after each layer, and every atom's term of the log-odds with the readout's bias (they sum
to the log-odds). `nitro.py` takes the pair of atoms with the strongest 2-SII and measures the
changes of the log-odds when either atom or both are added, with all other atoms masked (there the
discrete derivative is the pair's Möbius interaction) and averaged over 256 random maskings of the
other atoms (seed 0; the number of atoms kept drawn uniformly, then the atoms, as in the SII, so
the average estimates the pair's 2-SII). `maskings.py` evaluates all atoms, no atoms, a
substructure removed (`SUBSTRUCTURE`, per molecule) and a random masking (seed 3).

**`order.py`, `arrangement.py` and `orders.py`** (figures `order`, `arrangement`, `orders`; nothing
random). `order.py` explains every molecule for every question with GraphSHAP-IQ and writes, for the
k-SII of order 1 to 4 and the Möbius interactions, the number of non-zero contributions, their sum
and the sums of the positive and the negative ones; it checks that every order sums to the log-odds
minus the prediction with no atoms (efficiency, to 1e-5). `orders.py` sums the positive and the
negative Möbius interactions of each molecule (its own question) by the size of their set (1, 2, 3,
4 or more), with the same check, and the same for the 4-SII. `arrangement.py` builds catechol,
resorcinol and hydroquinone from their SMILES with `gnn.from_smiles`, turns them so that their rings
lie alike, and writes the PAINS GNN's probability and the complete 2-SII of each; it needs RDKit
(run it where RDKit is installed).

**`sparsity.py`, `complexity.py`, `approximation.py`, `paper_approximation.py`** (figures
`sparsity`, `complexity`, `approximation`, `paper-approximation`). `sparsity.py` evaluates each GNN on
all 2^n maskings of every molecule, in float32 and in float64 (the same weights, cast), takes the
Möbius transform of both, marks every set as inside or outside the set I of Proposition 3.6 (the
subsets of the receptive fields of three layers) and writes the absolute values binned by the size
of the set and a quarter of a decade, the largest values inside and outside I, and the rounding of
float32 (the largest difference of an MI between float32 and float64); it checks that GraphSHAP-IQ's
MIs agree with the float64 brute force (to 7e-5) and that |I| is GraphSHAP-IQ's budget (about four
minutes, three of them indigo). `complexity.py` counts |I| for one to four layers for the explorer's
seven molecules and the four of `../../graphshapiq-exact/assets/data.js`, and checks the counts
against the budgets of both demos (a second). `approximation.py` evaluates the graph game once on
all 2^n maskings and runs, at every budget of GraphSHAP-IQ's approximation variant
(`max_interaction_size` lambda = 1, 2, ... until it is exact), shapiq's `KernelSHAPIQ`, `SVARMIQ`,
`SHAPIQ` and `PermutationSamplingSII` (k-SII, order 2, seeds 0, 1, 2), with the MSE over all 2-SII
of one and two atoms against exact GraphSHAP-IQ; it also writes, for the record, the paper's
interaction-informed variant of the three Monte Carlo methods (the SII of pairs outside I set to
zero before the aggregation). The molecules with their own questions take about 25 minutes, almost
all of it SVARM-IQ on indigo; `--all` adds the other questions (about an hour more).
`paper_approximation.py` reads Figure 4 (left) of the paper from the vector graphic on page 9 of its
PDF with PyMuPDF (`pip install pymupdf`; `python paper_approximation.py <the PDF>`): the axes are
calibrated on their tick marks, the curves and bands are told apart by their colors and dashes.

**`algorithm.py`** (figure `algorithm`; nothing random, a second). For the explorer's seven
molecules and one to four layers, it counts what each line of Algorithm 1 produces: |I|, the sets of
line 1 and the model calls and MIs of lines 2 and 3, from `GraphSHAPIQ(...).budget` on a game that
is never evaluated, and the SIs of at most k atoms of line 4 for k = 1 to 4. It also writes the
largest receptive field and its atom, which the figure draws. Everything depends on the graph only,
so the counts for other numbers of layers than the demo's three need no GNN. It checks |I| against an
enumeration of the union of the power sets of the receptive fields and against
`parts/complexity.json`.
