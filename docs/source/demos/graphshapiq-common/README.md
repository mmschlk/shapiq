# The GraphSHAP-IQ demos: notes for whoever iterates on them

A demo and a figure explain graph neural networks (GNNs) on molecules with GraphSHAP-IQ
(Muschalik et al., 2025), computed with `shapiq.graph`. They are a counterpart to the FIxLIP demo:
the same kind of page (a demo, then what to see, what it shows and how it was computed), for
graphs. This folder is not a page: it holds what they share, a library of code and styles
(`assets/`) and the scripts that computed all their data (`compute/`).

| Demo | Folder | What a reader does | What they should learn |
|---|---|---|---|
| Explorer | `../graphshapiq-explorer` | Picks one of 8 molecules and one of 3 questions, and reads on the question's card what it asks and what the GNN answers. Then switches between *Hide atoms* (click atoms to mask them, the GNN reruns in the browser and answers again), *Atoms alone* (Shapley values) and *Atoms together* (k-SII of order 2, 3 or 4, or Möbius). The ninth tile, *Build your own*, puts groups (OH, NH₂, NO₂, CH₃, Cl, OMe, COOH) on the 6 positions of a benzene ring in a fourth view, *Build*: a click on a position opens a wheel of the groups. The browser then explains the molecule itself, exactly, with a progress bar. | Interactions show what the Shapley values hide. A nitro group's oxygens matter only with their nitrogen, and no atom of a catechol is a PAINS alone. The interactions are computed, not looked up: catechol is >99% PAINS, resorcinol 11%, hydroquinone 25%. Same atoms, other arrangement, so the GNN learned a pair. |
| Exact (a figure on the explorer's page, under "Exact, and still feasible"; no page or card of its own) | `../graphshapiq-exact` | Picks a molecule (benzene, caffeine, nitropyrene, cholesterol) and an atom, and sees its receptive field pulse out bond by bond. Compares the maskings GraphSHAP-IQ needs at 1 to 4 layers with 2ⁿ. | Why GraphSHAP-IQ is exact *and* feasible. A node's output depends only on its receptive field, so cholesterol needs 129,664 maskings instead of 2²⁸ (2,070× fewer). |

Each demo's `index.rst` is its page. For now a page holds its demos only: the write-up (what you
see, what it shows, how it was computed, with the claims a reader can check in the demo) follows
once the demos are settled. The last write-up is in the repository's history, the `index.rst` files
at commit 20c526c. The card on `../index.rst` uses `_static/images/hero_graphshapiq_explorer_demo.png`,
which still shows the explorer before the question card: retake it when the look is settled.

## How it fits together

```
graphshapiq-common/
├── assets/demo.css        shared styles, all classes gsq-*: tokens, controls, the molecule scene, panels
├── assets/demo.js         window.ShapiqGraph: the molecule scene, helpers and mount()
└── compute/               config.toml, train.py, explain.py, pack.py, check_worker.py (see its README)
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
  - `question(tasks, pick)` builds the question in plain words, which the explorer shows over
    the molecule: a switch of the questions (`label`), and a card of the one asked with
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
  its container's class (`.graphshapiq-explorer-demo .gsq-start`). Everything is built on the docs' tokens
  (`--color-foreground-primary`, `--shapiq-surface`, `--shapiq-surface-raised`, `--shapiq-border`,
  `--color-brand-content`). Dark mode follows Furo: `body[data-theme="dark"]` and
  `prefers-color-scheme` unless `body[data-theme="light"]`. The colors are shapiq's: red
  `#FF0D57` raises the odds of *yes*, blue `#1E88E5` lowers them, and violet is the receptive
  field. The layout breaks at a container width of 42rem, not at the viewport.
- **Data.** `compute/config.toml` is the single source of the molecules, the questions and their
  texts (the questions' plain words, blurbs and takeaways; the demos show neither blurbs nor takeaways since the seventh pass, they wait for the write-up). `pack.py --texts` writes changed texts
  into the `data.js` files as they are, without `compute/results/`. `explain.py` writes the full GraphSHAP-IQ results to
  `compute/results/`, which is gitignored and not committed. `pack.py` turns them into the
  `data.js` files. The GNN weights are in the data as folded affine maps, which `pack_gnn` checks
  against PyTorch (≤ 1e-3).
- **The explorer's ninth molecule is the reader's own** (*Build your own*, with a fourth view,
  *Build*): groups on the positions of a benzene ring. A click on a position (a + or a group)
  opens a wheel of the groups that can go there, around the position (`openWheel`; a group that
  would pass the 15 atoms is dimmed). The wheel is a ring of frosted glass: the molecule shows
  through it, and the position through its hub. Escape, the hub or a click beside it close it. A
  pill along the bottom of the frame offers two molecules to start from, as drawings without text (`STARTS`: an empty ring and catechol, named in their tooltips), in the view *Build* only. The molecule is then drawn above it (`.gsq-building`), and the wheel is placed through the plot's transform.
  Building has a color of its own, amber (`--gsq-build`): the tile, which stands apart from the
  eight molecules and stays in sight where the row scrolls, the position chosen, and the progress.
  Nothing of the molecule is in `data.js`. The page runs
  the GNN for the answer at once (`predict`), and a Web Worker (the `WORKER` string in
  `graphshapiq-explorer/assets/demo.js`, run from a Blob URL) explains it with GraphSHAP-IQ proper:
  it evaluates the GNN on the maskings inside the receptive fields only, computes the Möbius
  interactions, and from them the k-SII of orders 1 to 4. Its result has the shape of an entry of
  `data.explanations`, so that every view works for it as for the other molecules. It is limited to
  15 heavy atoms (`MAX_ATOMS`, at most 32,768 maskings, about five seconds). `compute/check_worker.py`
  runs the worker with Node on the explorer's own molecules and compares it with shapiq's numbers
  in `data.js`: the same maskings (GraphSHAP-IQ's budget), and values equal to a few 1e-3.

## Preview and verify

From the repository root, with the docs' dependency group installed (`uv sync --group docs`, or
`pip install --group docs`):

```bash
sphinx-build -b html -D plot_gallery=0 -j 4 docs/source /tmp/html   # about a minute
python -m http.server -d /tmp/html 8000                              # then open /demos/
```

`-D plot_gallery=0` skips running the examples. The demos do not need them. The pages to check are
`demos/graphshapiq-explorer/index.html` (it has the exact figure too) and
`demos/index.html` (the cards).

Before you push, check every demo you changed:

1. **No console errors or warnings.** Load the page with Playwright (preinstalled in cloud
   sessions: `require("playwright")`, `chromium.launch()`) and collect `console` and `pageerror`
   events.
2. **Light, dark and phone.** Toggle Furo's theme with `document.body.dataset.theme = "dark"`, and
   use viewports of 390 and 320 px for phones. No control may reach past the demo's container
   (compare each switch's bounding box with the container's). `documentElement.scrollWidth` is not
   enough: Furo's `.content` scrolls on its own.
3. **Keyboard.** Every control is a button or radio group. In the explorer, the molecule's atoms
   take focus with Tab and show it as a drawn ring (`keyFocus`). Focus must stay visible.
4. **Reduced motion.** With `prefers-reduced-motion: reduce`, the scene must not tween
   (`ShapiqGraph.reducedMotion`).
5. **Numbers.** If you change the data or a text that quotes it, check the number against
   `data.js`. If you change the explorer's worker, run `python compute/check_worker.py`. It runs
   the worker with Node and compares it with shapiq's explanations in `data.js`. It currently
   agrees on the maskings, and on the values to 1.6e-3 of the largest.
6. **Card images.** These are 1952 × 1252 screenshots: the demo at 2× in a white frame with a 2 px
   `#e2e6f3` edge and 26 px corners, as the other cards (see `../orders/README.md`). Retake them
   if a demo's look changes.

## Verified numbers (do not change a text without the data)

- Explorer, 1,3-dinitrobenzene, *Mutagenic?*: Shapley values N +1.86 each and O +0.07. 2-SII: an
  oxygen alone −0.70, with its nitrogen +1.83. GNN: 98%, log-odds +3.98, exact over 4,096 of 4,096
  maskings.
- Explorer, *Explain it yourself* snippet: 43,968 of 2¹⁶ maskings; O–O −1.72, N–O +1.32.
- Explorer, *Build your own*, *PAINS?*: catechol >99%, resorcinol 11%, hydroquinone 25%.
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
  to it. For the reader's own molecule, the explorer finds the pattern itself (`patternOf` in its
  `demo.js`).
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

- **Blob worker.** The explorer creates its Web Worker (*Build your own*) from a Blob URL. This works in the local build.
  Check it on Read the Docs, in case a Content-Security-Policy blocks `blob:` workers. If it does,
  move the worker to `assets/worker.js` and load it from `data-root`.
- **Explorer data size.** `graphshapiq-explorer/assets/data.js` is about 215 kB: 130 kB of
  explanations and 94 kB of the three folded GNNs. `OVERALL` and `PER_ATOM` in `pack.py` trade
  the explanations' size for detail.
- **The scaffold of *Build your own*** is a benzene ring only. Other rings (pyridine, naphthalene)
  or free editing would make it more of a lab. `assemble()` and `GROUPS` in the explorer's `demo.js` are where to
  start.
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

A fourth pass folded the lab into the explorer, as its ninth molecule (*Build your own*): the two
demos were the same instrument (card, GNNs, drawing, views), and the lab's catechol story answers
the explorer's question of when a molecule is a PAINS. The lab's page and card went. Its last
version is in the repository's history (`docs/source/demos/graphshapiq-lab` at commit 2f980a2).
The built molecule gained what only the explorer had: hiding atoms, and the orders beyond pairs.

A fifth pass made building direct: the wheel of groups at the position clicked replaced a palette
of chips in the panel, the tiles took the demo's whole width, and the tile of *Build your own* its
accent. A sixth made that accent amber (blue is a value's color here), the wheel see-through, and
cut the molecules to start from to two, on a selector at the bottom of the frame.

A seventh pass cut the text the demos do not need to be understood: the questions' meanings to two
short sentences, the molecules' formulas, blurbs and takeaways, the panels' subtitles and
paragraphs, the "which atoms" line, and the receptive-field figure's sentences to a few words.

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
