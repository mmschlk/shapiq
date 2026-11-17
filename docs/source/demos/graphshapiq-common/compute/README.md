# How the GraphSHAP-IQ demos were computed

These scripts computed the data of the GraphSHAP-IQ demos (`../../graphshapiq-*/assets/data.js`):
the [explorer](../../graphshapiq-explorer/index.rst) and
the figure of receptive fields on the explorer's page (`../../graphshapiq-exact`). They are run by hand, once; the docs only show their result.

| File | What it does |
|---|---|
| `config.toml` | The questions (tasks) with their plain words, the explorer's molecules as SMILES with their texts, and the molecules of the figure of receptive fields (exact). |
| `molecules.py` | What the scripts share: molecules as graphs, the datasets, and the GNN (B-XAIC's GIN). |
| `train.py` | Trains a GNN per task: `results/models/<task>.pt`, with its scores. |
| `explain.py` | Explains every molecule for every task with GraphSHAP-IQ: `results/<molecule>_<task>.json` (all Möbius interactions, k-SII of orders 1 to 4). |
| `pack.py` | Writes the demos' data: GNNs folded into affine maps (checked against PyTorch), drawings, compact explanations, maskings per depth (exact). |
| `check_worker.py` | Checks the explorer's in-browser GraphSHAP-IQ (the worker of *Build your own* in `../../graphshapiq-explorer/assets/demo.js`, run with Node) against shapiq's explanations in the explorer's `data.js`. Needs neither a GNN nor `results/`. |

## Run

From this folder, with `shapiq` installed from the repository with its `graph` extra (torch and
torch_geometric), RDKit, scikit-learn and networkx:

```bash
pip install -e "../../../../..[graph]" rdkit scikit-learn networkx

python train.py      # 3 GNNs: 5 to 10 minutes each for B-XAIC, a minute for Mutagenicity
python explain.py    # 24 explanations, at most 15 seconds each (4 minutes in all)
python pack.py       # all of them (a minute); or e.g. `python pack.py --demos exact`
python pack.py --texts  # only config.toml's texts, into the data that is there (needs no results/)
python check_worker.py  # optional, needs Node: the browser's GraphSHAP-IQ against shapiq's
```

The first run downloads B-XAIC's molecules from [Hugging Face](https://huggingface.co/datasets/mproszewska/B-XAIC)
(6 MB) and TUDataset's Mutagenicity (1 MB). `train.py` and `explain.py` skip what exists in
`results/`, so a new molecule costs only its own explanations. The intermediate results
(`results/`) are not part of the repository.

The data of the docs was computed with Python 3.13, shapiq from the repository, torch 2.15,
torch_geometric 2.8 and RDKit 2026.09, on four CPU cores. The GNNs score a test F1 of 0.98
(*Indole?*), 0.79 (*Mutagenic?*) and 0.94 (*PAINS?*).

## Change a text

The questions' plain words (`label`, `question`, `meaning`, and `motif`, the pattern the question
card draws), the molecules' blurbs and takeaways are in `config.toml`. `python pack.py --texts`
writes them into the explorer's `data.js` as it is: the GNNs and the explanations
stay, so that it needs neither `results/` nor a GNN (RDKit draws the patterns).

## Change a demo's molecules

- **Explorer:** add a `[[molecules]]` entry to `config.toml` (SMILES with at most 31 heavy atoms, a
  name, a blurb, its question as `task`, and optionally a `takeaway`), then run `explain.py` and
  `pack.py`. A takeaway quotes numbers of the data: check them after `explain.py` ran again.
- **Exact:** add an `[[exact_demo]]` entry, then run `pack.py --demos exact` (no GNN is evaluated).
- **Build your own:** the groups a reader can place and the molecules to start from are in
  `../../graphshapiq-explorer/assets/demo.js` (`GROUPS`, `STARTS`). Its GNNs are the explorer's.
