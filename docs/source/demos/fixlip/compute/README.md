# How the FIxLIP demo was computed

These scripts computed the data of the [FIxLIP demo](https://shapiq.readthedocs.io/en/latest/demos/fixlip/index.html)
(`../assets/data.js`) and the figures of its page (`../figures/`). They are run by hand, once; the
docs only show their result.

| File | What it does |
|---|---|
| `config.toml` | The models, the photos with their captions, and the texts the demo shows. |
| `explain.py` | Explains every photo and model with ProxySHAP: `results/<photo>_<model>.json`. |
| `norms.py` | The scale of the similarities, on 200 COCO photos: `results/norms.json`. |
| `pack.py` | Writes the demo's data, `../assets/data.js`, and copies the photos next to it. |
| `evaluate.py` | Evaluates SigLIP 2 on the masks that the page's figures show: `figures.json`. |
| `figures.py` | Draws the page's figures in both themes: `../figures/<name>-<theme>.webp`. |

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

The figures of the page explain the demo on its first photo, for SigLIP 2. They are drawn from the
demo's data and from the model's predictions for the masks they show:

```bash
python evaluate.py  # SigLIP 2 on about 3,500 masks of the elephant, 4 minutes on four CPU cores
python figures.py   # needs matplotlib, scipy and the Inter typeface, and no model
```

`figures.json` is part of the repository, so `figures.py` runs on its own, e.g. after a change to
a figure's layout. Run both again when the elephant's explanation in `data.js` changes.

The data of the docs was computed with Python 3.13, shapiq from the repository (after release 1.8.0),
torch 2.14, transformers 5.20 and xgboost 3.4.

## Add a photo

1. Add a `[[photos]]` entry to `config.toml`: its COCO id (of the train2017 split), the square
   crop, the caption, and the texts the demo shows (title, alt text, excerpt and credit; the credit
   and license of a COCO photo are on [cocodataset.org](https://cocodataset.org/#explore)).
2. Run `explain.py`, then `pack.py`.
