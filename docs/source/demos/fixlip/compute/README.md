# How the FIxLIP demo was computed

These scripts computed the data of the [FIxLIP demo](https://shapiq.readthedocs.io/en/latest/demos/fixlip/index.html)
(`../assets/data.js`) and the data of the figures of its page (`../assets/figures-data.js`). They
are run by hand, once; the docs only show their result.

| File | What it does |
|---|---|
| `config.toml` | The models, the photos with their captions, and the texts the demo shows. |
| `explain.py` | Explains every photo and model with ProxySHAP: `results/<photo>_<model>.json`. |
| `norms.py` | The scale of the similarities, on 200 COCO photos: `results/norms.json`. |
| `pack.py` | Writes the demo's data, `../assets/data.js`, and copies the photos next to it. |
| `evaluate.py` | Evaluates SigLIP 2 on the masks that the page's figures show, for every photo. |
| `paper.py` | Reads the data of the paper's plots that the page draws again from the paper's figures. |
| `pointing.py` | Plays the paper's pointing game for SigLIP 2 on four photos of the demo. |
| `baselines.py` | Explains every photo for CLIP with GAME, Grad-ECLIP and exCLIP, as in the paper's Figure 2. |

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

The figures of the page are drawn in the browser by `../assets/figures.js`, for the photo chosen
in the demo. Most of them only read the demo's data. The predictions of SigLIP 2 for the masks they
show, the data of the paper's plots, and the explanations of CLIP by the paper's baselines are in
`../assets/figures-data.js`, which these scripts write:

```bash
python evaluate.py [photo ...]  # SigLIP 2 on about 3,500 masks of each photo, 4 minutes each on four CPU cores
python paper.py <figures>       # needs PyMuPDF and the folder "figures" of the paper's LaTeX sources
python pointing.py              # one explanation of four photos in one image, about 10 minutes
python baselines.py             # GAME, Grad-ECLIP and exCLIP for CLIP, about 45 seconds a photo
```

`evaluate.py` needs scipy and writes the part `images` of the file, one entry for each photo. It
reads the two objects of each photo from `config.toml` (`objects`, with their patches and the words
that name them). Run it again for a photo when its explanation for SigLIP 2 in `data.js` changes.
`paper.py` writes the part `paper`. It reads the boxes and the curves of the paper's vector figures
(Figures 3, 4 and 5) with their exact coordinates, and holds the scores of Table 1. `pointing.py`
writes the part `pointing` and the image `../assets/pointing.jpg`. It combines four photos of the
demo into one image, explains it with a text that names the four photos as `explain.py` does, and
keeps the interactions between names and patches with their pointing game recognition.
`baselines.py` writes the part `baselines`. It explains every photo of `config.toml` for CLIP
(ViT-B/32) with the three baselines of the paper's Figure 2, with the settings that FIxLIP's
repository used for the paper: GAME from the last attention block of both encoders, Grad-ECLIP
from the last block of the vision encoder and the last eight blocks of the text encoder, and
exCLIP with 20 steps from a reference (an image of the normalization mean and an empty text) at
the input of the last block. It keeps a value for every patch and every token (GAME, Grad-ECLIP)
and for every pair of a patch and a token (exCLIP). The fourth method of the figure, FIxLIP, is the
demo's explanation in `data.js`.

The data of the docs was computed with Python 3.13, shapiq from the repository (after release 1.8.0),
torch 2.14, transformers 5.20 and xgboost 3.4. `baselines.py` ran with torch 2.9 and
transformers 5.3.

## Add a photo

1. Add a `[[photos]]` entry to `config.toml`: its COCO id (of the train2017 split), the square
   crop, the caption, and the texts the demo shows (title, alt text, excerpt and credit; the credit
   and license of a COCO photo are on [cocodataset.org](https://cocodataset.org/#explore)).
2. Run `explain.py`, then `pack.py`.
3. For the figures of the page, give the photo two `objects` in `config.toml` and run
   `evaluate.py <photo>`.
