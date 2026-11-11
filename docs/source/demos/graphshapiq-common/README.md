# The GraphSHAP-IQ demos: notes for whoever iterates on them

Two demos and a figure explain graph neural networks (GNNs) on molecules with GraphSHAP-IQ
(Muschalik et al., 2025), computed with `shapiq.graph`. They are a counterpart to the FIxLIP demo:
the same kind of page (a demo, then what to see, what it shows and how it was computed), for
graphs. This folder is not a page: it holds what they share, a library of code and styles
(`assets/`) and the scripts that computed all their data (`compute/`).

| Demo | Folder | What a reader does | What they should learn |
|---|---|---|---|
| Explorer | `../graphshapiq-explorer` | Picks one of 8 molecules and one of 3 questions, and reads on the question's card what it asks and what the GNN answers. Then switches between *Hide atoms* (click atoms to mask them, the GNN reruns in the browser and answers again), *Atoms alone* (Shapley values) and *Atoms together* (k-SII of order 2, 3 or 4, or Möbius). | Interactions show what the Shapley values hide. A nitro group's oxygens matter only with their nitrogen, and no atom of a catechol is a PAINS alone. Also the reference demo. |
| Lab | `../graphshapiq-lab` | Puts groups (OH, NH₂, NO₂, CH₃, Cl, OMe, COOH) on the 6 positions of a benzene ring. A Web Worker then computes the GNN's answer and the **exact** Shapley values and 2-SII over all 2ⁿ maskings, with a progress bar. | The interactions are computed, not looked up: catechol is >99% PAINS, resorcinol 11%, hydroquinone 25%. Same atoms, other arrangement, so the GNN learned a pair. |
| Exact (a figure on the explorer's page, under "Exact, and still feasible"; no page or card of its own) | `../graphshapiq-exact` | Picks a molecule (benzene, caffeine, nitropyrene, cholesterol) and an atom, and sees its receptive field pulse out bond by bond. Compares the maskings GraphSHAP-IQ needs at 1 to 4 layers with 2ⁿ. | Why GraphSHAP-IQ is exact *and* feasible. A node's output depends only on its receptive field, so cholesterol needs 129,664 maskings instead of 2²⁸ (2,070× fewer). |

Each demo's `index.rst` is its page. For now a page holds its demos only: the write-up (what you
see, what it shows, how it was computed, with the claims a reader can check in the demo) follows
once the demos are settled. The last write-up is in the repository's history, the `index.rst` files
at commit 20c526c. The cards on `../index.rst` use `_static/images/hero_graphshapiq_<demo>_demo.png`,
which still show the demos before the question card: retake them when the look is settled.

## How it fits together

```
graphshapiq-common/
├── assets/demo.css        shared styles, all classes gsq-*: tokens, controls, the molecule scene, panels
├── assets/demo.js         window.ShapiqGraph: the molecule scene, helpers and mount()
└── compute/               config.toml, train.py, explain.py, pack.py, check_lab.py (see its README)
graphshapiq-<demo>/
├── index.rst              the page
└── assets/
    ├── requires.txt       "graphshapiq-common": the page loads the library before the demo
    ├── demo.js / demo.css the demo itself, built on window.ShapiqGraph
    └── data.js            window.SHAPIQ_GRAPHSHAPIQ_<DEMO> = {...}, written by compute/pack.py
```

- **Shared library.** `requires.txt` is a feature of the `shapiq-demo` directive
  (`docs/source/_ext/shapiq_demos.py`, documented in `../README.md`). Every page that embeds a
  GraphSHAP-IQ demo gets `demos/graphshapiq-common/demo.css` and `demo.js` before the demo's own,
  with Sphinx's checksums, and is rebuilt when they change. Do not copy library code into a demo:
  add it to the library.
- **`window.ShapiqGraph`** (`assets/demo.js`):
  - `createScene(plot, {duration})` builds the persistent, animated SVG molecule. Its
    `setMolecule(m)` tweens to a new molecule. Its `update(state)` tweens rings (own effects),
    bands and shapes (interactions), masking, focus, the ground-truth outline and dimming, with
    state keys `rings`, `ringLargest`, `groups`, `groupLargest`, `masked`, `focus`, `truth`,
    `truthKey`, `truthShown` and `dim`. It also has `atomAt(event)`, `keyFocus(k)`, `hits()` and
    `geom`.
  - `predict(symbols, m, task, masked)` runs a folded GNN (affine maps and ReLU, float32 from
    base64) in the browser.
  - `geom`, `skeleton` and `tile` draw molecules and thumbnails. `legend`, `row`, `formula`,
    `groupName` and `atomName` build the panels. `colorOf`, `signed`, `percent` and `bits` format
    values.
  - `question(tasks, pick)` builds the question in plain words, which the explorer and the lab
    show over the molecule: a switch of the questions (`label`), and a card of the one asked with
    the question in full (`question`), what its term means and when the answer is yes (`meaning`),
    the pattern drawn as a small skeletal formula (`motif(m)`), whether the molecule has the pattern
    (the ground truth, outlined in the drawing), and the GNN's answer. Its
    `set({task, subject, has, p, note})` shows a question for a molecule.
  - `switcher(label, options, pick)`, `picker(label, molecules, pick)` and `hinter(frame)` build
    the controls every demo has, in the look of the FIxLIP and orders demos: a switch whose pill
    slides to the option chosen, the molecules as tiles, and the dark hint over the molecule
    ("Hover an atom", or "Tap an atom" without a mouse: `hovers()`). Each returns its `node` and a
    `set(k)` (the hint: `show(text)` and `hide()`).
  - `mount(className, global, start)` finds every container of a demo and adds the class `gsq`.
    It uses the data the page loaded, or loads `data.js` from `data-root`. Then it calls
    `start(root, data)`. If the demo fails, it falls back to the `<noscript>` image and adds the
    class `gsq-failed`.
- **Styles.** Every class is named `gsq-*` (as the FIxLIP demo's are `fixlip-demo-*`), so that no
  rule of the theme or of another demo applies to them by accident; a demo's own rules start with
  its container's class (`.graphshapiq-lab-demo .gsq-chip`). Everything is built on the docs' tokens
  (`--color-foreground-primary`, `--shapiq-surface`, `--shapiq-surface-raised`, `--shapiq-border`,
  `--color-brand-content`). Dark mode follows Furo: `body[data-theme="dark"]` and
  `prefers-color-scheme` unless `body[data-theme="light"]`. The colors are shapiq's: red
  `#FF0D57` raises the odds of *yes*, blue `#1E88E5` lowers them, and violet is the receptive
  field. The layout breaks at a container width of 42rem, not at the viewport.
- **Data.** `compute/config.toml` is the single source of the molecules, the questions and their
  texts (the questions' plain words, blurbs and takeaways). `pack.py --texts` writes changed texts
  into the `data.js` files as they are, without `compute/results/`. `explain.py` writes the full GraphSHAP-IQ results to
  `compute/results/`, which is gitignored and not committed. `pack.py` turns them into the
  `data.js` files. The GNN weights are in the data as folded affine maps, which `pack_gnn` checks
  against PyTorch (≤ 1e-3).
- **The lab computes, it does not look up.** Its worker (the `WORKER` string in
  `graphshapiq-lab/assets/demo.js`, run from a Blob URL) evaluates the GNN on all 2ⁿ maskings. It
  then computes Möbius, Shapley values and SII up to order 2, and turns them into 2-SII. It is
  limited to 15 heavy atoms (`MAX_ATOMS`, up to 32,768 maskings).

## Preview and verify

From the repository root, with the docs' dependency group installed (`uv sync --group docs`, or
`pip install --group docs`):

```bash
sphinx-build -b html -D plot_gallery=0 -j 4 docs/source /tmp/html   # about a minute
python -m http.server -d /tmp/html 8000                              # then open /demos/
```

`-D plot_gallery=0` skips running the examples. The demos do not need them. The pages to check are
`demos/graphshapiq-{explorer,lab}/index.html` (the explorer's has the exact figure too) and
`demos/index.html` (the cards).

Before you push, check every demo you changed:

1. **No console errors or warnings.** Load the page with Playwright (preinstalled in cloud
   sessions: `require("playwright")`, `chromium.launch()`) and collect `console` and `pageerror`
   events.
2. **Light, dark and phone.** Toggle Furo's theme with `document.body.dataset.theme = "dark"`, and
   use a 390 px wide viewport for the phone. `document.documentElement.scrollWidth` must not exceed
   the viewport: no horizontal scroll.
3. **Keyboard.** Every control is a button or radio group. In the explorer, the molecule's atoms
   take focus with Tab and show it as a drawn ring (`keyFocus`). Focus must stay visible.
4. **Reduced motion.** With `prefers-reduced-motion: reduce`, the scene must not tween
   (`ShapiqGraph.reducedMotion`).
5. **Numbers.** If you change the data or a text that quotes it, check the number against
   `data.js`. If you change the lab's worker, run `python compute/check_lab.py`. It runs the worker
   with Node and compares it with `shapiq`'s `GraphSHAPIQ`. It currently agrees to 3e-6.
6. **Card images.** These are 1952 × 1252 screenshots: the demo at 2× in a white frame with a 2 px
   `#e2e6f3` edge and 26 px corners, as the other cards (see `../orders/README.md`). Retake them
   if a demo's look changes.

## Verified numbers (do not change a text without the data)

- Explorer, 1,3-dinitrobenzene, *Mutagenic?*: Shapley values N +1.86 each and O +0.07. 2-SII: an
  oxygen alone −0.70, with its nitrogen +1.83. GNN: 98%, log-odds +3.98, exact over 4,096 of 4,096
  maskings.
- Explorer, *Explain it yourself* snippet: 43,968 of 2¹⁶ maskings; O–O −1.72, N–O +1.32.
- Lab, *PAINS?*: catechol >99%, resorcinol 11%, hydroquinone 25%.
- Exact: cholesterol 2²⁸ → 129,664 (2,070×), 5,509 at 2 layers; benzene 64 = 2⁶.
- GNN test F1: *Indole?* 0.98, *Mutagenic?* 0.79, *PAINS?* 0.94.

## Design decisions (and what was tried and dropped)

- **Masking keeps bonds** and gives the atom the average atom of the GNN's training set. That is
  `GraphBaselineImputer` with a mean baseline, as in the paper.
- **The GNN sees elements and links only**, not bond orders. That is why indigo fools the
  *Indole?* GNN, which the explorer turns into a story.
- **A lay reader must know what is asked.** *Indole?*, *Mutagenic?* and *PAINS?* on a bare switch
  told a reader neither what they mean nor when a molecule is one. The question card says it in
  plain words, draws the pattern, says whether this molecule has it, and puts the GNN's answer next
  to it. The lab finds the pattern in the reader's molecule itself (`patternOf` in its `demo.js`).
  The card says "one kind of PAINS is a catechol" and "a known warning sign is a nitro group": both
  patterns are signs, not definitions (2-aminofluorene is a mutagen without a nitro group).
- **Questions with a ground truth.** B-XAIC's indole and PAINS (catechol) labels follow from
  substructures, and the nitro group is a known Ames alert, so a reader can see whether the
  explanation finds the truth (the thin outline).
- **Neutral, scientific chemistry.** The molecules are drugs, dyes, metabolites and textbook
  aromatics. Keep it that way: no hazardous or weapon-related compounds or framings.
- **Dropped:** a water-network setting (the paper's other graph example) and a scent question
  ("why does it smell fruity?"), for lack of a fitting dataset. The first version drew green
  circles around atoms; the user disliked them, and they were replaced by the shapiq-colored rings
  and bands.

## Open issues and ideas

- **Blob worker.** The lab creates its Web Worker from a Blob URL. This works in the local build.
  Check it on Read the Docs, in case a Content-Security-Policy blocks `blob:` workers. If it does,
  move the worker to `assets/worker.js` and load it from `data-root`.
- **Explorer data size.** `graphshapiq-explorer/assets/data.js` is about 215 kB: 130 kB of
  explanations and 94 kB of the three folded GNNs. `OVERALL` and `PER_ATOM` in `pack.py` trade
  the explanations' size for detail.
- **The lab's scaffold** is a benzene ring only. Other rings (pyridine, naphthalene) or free
  editing would make it more of a lab. `assemble()` and `GROUPS` in its `demo.js` are where to
  start.
- **Merge the lab into the explorer** as a fourth view? The two share the GNNs and the scene, and
  a single demo would be simpler for readers. Decide after feedback.
- **Mobile polish.** All demos fit 390 px, but the explorer's molecule strip is dense there.
- **Exact demo:** an animation of message passing itself (features flowing along bonds, layer by
  layer) would show *why* the receptive field grows, not only that it does.
- **More molecules / questions:** add them in `compute/config.toml` (see `compute/README.md`).
  Every number a text quotes must be checked again afterwards.

## History

The demos grew in one session. It started as a single explorer replicating a docs page, iterated
for colors (shapiq's), accessibility (keyboard) and visuals (rings and bands instead of circles).
Then a long autonomous run explored four concepts, published as artifacts and then turned into
these four demos:

1. Explorer: breadth, animated.
2. Lab: exact GraphSHAP-IQ live in the browser.
3. Exact: receptive fields and maskings.
4. Orders: every order and its faithfulness. It was first cut for the Möbius-truncation R² above,
   and brought back with FSII.

A second pass aligned them with the FIxLIP and orders demos: class names `gsq-*`, the switches'
sliding pill, the dark hint, and one molecule picker for all four (the lab's examples are tiles
too). It also gave the orders demo a real curve (axes, an order per click, the R² itself where it is
below 0) and the exact demo a fourth layer and the numbers at its bars.

A third pass made the questions understandable (the question card above), moved the GNN's answer
from the panel onto that card in both demos, renamed the explorer's *Prediction* view to *Hide
atoms*, and took the write-ups off the pages until the demos are settled.

The suggestion from that run was to lead with the explorer (the safe reference) and feature the
lab (the wow), with the exact and orders demos as explainer pages.

The maintainer then decided what goes into the docs: the explorer and the lab as demos; the exact
demo as a figure on the explorer's page ("Exact, and still feasible"), since it explains the
algorithm rather than what interactions show; and no orders demo, which overlapped with the guide
"From Shapley values to the full decomposition" (`../orders`), whose first example is a molecule
and a GNN at every order. The orders demo is in the history of the repository (it was removed
after commit 47bf168): FSII of every order from GraphSHAP-IQ's Möbius interactions, with its R²
beyond size. Its finding, should the guide want it: the Shapley values scored below 0 for all five
molecules; 1,3-dinitrobenzene reached 0.95 with pairs, dopamine 0.89 only at order 3.
