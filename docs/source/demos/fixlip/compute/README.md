# How the FIxLIP demo was computed

These scripts computed the data of the [FIxLIP demo](https://shapiq.readthedocs.io/en/latest/demos/fixlip/index.html)
(`../assets/data.js`). They are run by hand, once; the docs only show their result.

| File | What it does |
|---|---|
| `config.toml` | The models, the photos with their captions, and the texts the demo shows. |
| `explain.py` | Explains every photo and model with ProxySHAP: `results/<photo>_<model>.json`. |
| `norms.py` | The scale of the similarities, on 200 COCO photos: `results/norms.json`. |
| `pack.py` | Writes the demo's data, `../assets/data.js`, and copies the photos next to it. |

## Run

From this folder, with `shapiq` installed from the repository, with the dependencies of
`shapiq_games` (torch, transformers, xgboost and the SigLIP tokenizer's):

```bash
pip install -e "../../../../..[games]"

python explain.py  # 10 explanations, 5 to 10 minutes each on four CPU cores
python norms.py    # downloads COCO's annotations (241 MB) and 200 photos
python pack.py
```

`explain.py` skips the explanations that exist in `results/`, so a new photo costs only its own
explanations. The intermediate results (`results/`) are not part of the repository.

The data of the docs was computed with Python 3.13, shapiq from the repository (after release 1.8.0),
torch 2.14, transformers 5.20 and xgboost 3.4.

## Add a photo

1. Add a `[[photos]]` entry to `config.toml`: its COCO id (of the train2017 split), the square
   crop, the caption, and the texts the demo shows (title, alt text, excerpt and credit; the credit
   and license of a COCO photo are on [cocodataset.org](https://cocodataset.org/#explore)).
2. Run `explain.py`, then `pack.py`.
