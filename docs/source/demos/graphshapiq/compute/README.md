# How the GraphSHAP-IQ demo was computed

These scripts computed the data of the [GraphSHAP-IQ demo](https://shapiq.readthedocs.io/en/latest/demos/graphshapiq/index.html)
(`../assets/data.js`). They are run by hand, once; the docs only show their result.

| File | What it does |
|---|---|
| `config.toml` | The questions (tasks), the molecules as SMILES, and the texts the demo shows. |
| `molecules.py` | What the scripts share: molecules as graphs, the datasets, and the GNN (B-XAIC's GIN). |
| `train.py` | Trains a GNN per task: `results/models/<task>.pt`, with its scores. |
| `explain.py` | Explains every molecule for every task with GraphSHAP-IQ: `results/<molecule>_<task>.json`. |
| `pack.py` | Writes the demo's data, `../assets/data.js`: the GNNs, the drawings, the explanations. |

## Run

From this folder, with `shapiq` installed from the repository with its `graph` extra (torch and
torch_geometric), RDKit, scikit-learn and networkx:

```bash
pip install -e "../../../../..[graph]" rdkit scikit-learn networkx

python train.py    # 3 GNNs: 5 to 10 minutes each for B-XAIC, a minute for Mutagenicity
python explain.py  # 24 explanations, less than 20 seconds each
python pack.py
```

The first run downloads B-XAIC's molecules from [Hugging Face](https://huggingface.co/datasets/mproszewska/B-XAIC)
(6 MB) and TUDataset's Mutagenicity (1 MB). `train.py` and `explain.py` skip what exists in
`results/`, so a new molecule costs only its own explanations. The intermediate results
(`results/`) are not part of the repository.

The data of the docs was computed with Python 3.13, shapiq from the repository, torch 2.14,
torch_geometric 2.8 and RDKit 2026.09, on four CPU cores.

## Add a molecule

1. Add a `[[molecules]]` entry to `config.toml`: its SMILES (at most 31 heavy atoms), its name and
   the sentence the demo shows under it.
2. Run `explain.py`, then `pack.py`.
