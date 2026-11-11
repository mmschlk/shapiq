# How the GraphSHAP-IQ demos were computed

These scripts computed the data of the four GraphSHAP-IQ demos (`../../graphshapiq-*/assets/data.js`):
the [explorer](../../graphshapiq-explorer/index.rst), the [lab](../../graphshapiq-lab/index.rst), the
[exact](../../graphshapiq-exact/index.rst) demo and the [orders](../../graphshapiq-orders/index.rst)
demo. They are run by hand, once; the docs only show their result.

| File | What it does |
|---|---|
| `config.toml` | The questions (tasks), the explorer's molecules as SMILES with their texts, and the molecules of the orders, lab and exact demos. |
| `molecules.py` | What the scripts share: molecules as graphs, the datasets, and the GNN (B-XAIC's GIN). |
| `train.py` | Trains a GNN per task: `results/models/<task>.pt`, with its scores. |
| `explain.py` | Explains every molecule for every task with GraphSHAP-IQ: `results/<molecule>_<task>.json` (all Möbius interactions, k-SII of orders 1 to 4). |
| `pack.py` | Writes the four demos' data: GNNs folded into affine maps (checked against PyTorch), drawings, compact explanations, FSII of every order with its R² (orders), maskings per depth (exact). |
| `check_lab.py` | Checks the lab's in-browser GraphSHAP-IQ (the worker in `../../graphshapiq-lab/assets/demo.js`, run with Node) against `shapiq` on a few molecules. |

## Run

From this folder, with `shapiq` installed from the repository with its `graph` extra (torch and
torch_geometric), RDKit, scikit-learn and networkx:

```bash
pip install -e "../../../../..[graph]" rdkit scikit-learn networkx

python train.py      # 3 GNNs: 5 to 10 minutes each for B-XAIC, a minute for Mutagenicity
python explain.py    # 24 explanations, at most 15 seconds each (4 minutes in all)
python pack.py       # all four demos (a minute); or e.g. `python pack.py --demos lab exact`
python check_lab.py  # optional, needs Node: the lab's numbers against shapiq's
```

The first run downloads B-XAIC's molecules from [Hugging Face](https://huggingface.co/datasets/mproszewska/B-XAIC)
(6 MB) and TUDataset's Mutagenicity (1 MB). `train.py` and `explain.py` skip what exists in
`results/`, so a new molecule costs only its own explanations. The intermediate results
(`results/`) are not part of the repository.

The data of the docs was computed with Python 3.13, shapiq from the repository, torch 2.15,
torch_geometric 2.8 and RDKit 2026.09, on four CPU cores. The GNNs score a test F1 of 0.98
(*Indole?*), 0.79 (*Mutagenic?*) and 0.94 (*PAINS?*).

## Change a demo's molecules

- **Explorer:** add a `[[molecules]]` entry to `config.toml` (SMILES with at most 31 heavy atoms, a
  name, a blurb, its question as `task`, and optionally a `takeaway`), then run `explain.py` and
  `pack.py`. A takeaway quotes numbers of the data: check them after `explain.py` ran again.
- **Orders:** list explorer molecules of at most 16 heavy atoms in `[orders_demo]` (their R² is
  computed over all 2^n coalitions), then run `pack.py --demos orders`.
- **Exact:** add an `[[exact_demo]]` entry, then run `pack.py --demos exact` (no GNN is evaluated).
- **Lab:** its groups and examples are in `../../graphshapiq-lab/assets/demo.js`; its GNNs are those
  of `[lab_demo]`, written by `pack.py --demos lab`.
