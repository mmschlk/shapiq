# The GraphSHAP-IQ demos: notes for whoever iterates on them

A demo and a figure explain graph neural networks (GNNs) on molecules with GraphSHAP-IQ
(Muschalik et al., 2025), computed with `shapiq.graph`. They are a counterpart to the FIxLIP demo:
the same kind of page (a demo, then what to see, what it shows and how it was computed), for
graphs. This folder is not a page: it holds what they share, a library of code and styles
(`assets/`) and the scripts that computed all their data (`compute/`).

| Demo | Folder | What a reader does | What they should learn |
|---|---|---|---|
| Explorer | `../graphshapiq-explorer` | Picks one of 7 molecules and one of 3 questions, and reads on the question's card what it asks and what the GNN answers. Then switches between *Input* (click atoms to mask them, the GNN reruns in the browser and answers again) and *Explanation*, with a switch of every explanation (Shapley values, 2-, 3- and 4-SII, Möbius) drawn as an SI graph. The ninth tile, *Build your own*, puts groups (OH, NH₂, NO₂, CH₃, Cl, OMe, COOH) on the 6 positions of a benzene ring, in every view: a + opens a wheel of the groups, a × on a group removes it. The browser then explains the molecule itself, exactly, with a progress bar. | Interactions show what the Shapley values hide. A nitro group's oxygens matter only with their nitrogen, and no atom of a catechol is a PAINS alone. The interactions are computed, not looked up: catechol is >99% PAINS, resorcinol 11%, hydroquinone 25%. Same atoms, other arrangement, so the GNN learned a pair. |
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
    `setMolecule(m)` tweens to a new molecule. Its `update(state)` tweens halos (own effects),
    hyperedges as shapiq's `si_graph_plot` draws them (interactions), masking, focus, the ground-truth outline and dimming, with
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
- **The explorer's ninth molecule is the reader's own** (*Build your own*): groups on the positions
  of a benzene ring, built in every view (`slots`). A light-amber + on a free position opens a wheel
  of the groups that can go there, around the position (`openWheel`; a group that would pass the 15
  atoms is dimmed). The wheel is a ring of frosted glass: the molecule shows through it, and the
  position through its hub. Escape, the hub or a click beside it close it. A small × in the middle
  of a group's bond to the ring removes the group. The + and the × stay off the atoms, so that the
  atoms can still be hidden and pointed at. A pill beside the views offers two molecules to start
  from, as drawings without text (`STARTS`: an empty ring and catechol, named in their tooltips).
  The views and the pill form one menu, centred along the top of the frame (`.gsq-viewbar`, a flex
  row that wraps only if it must). The frame is a flex column, so the molecule takes all the room
  under the menu. The hint sits at the bottom. The wheel is placed through the plot's transform.
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
  oxygen's own share −0.70, with its nitrogen +1.83. GNN: 98%, log-odds +3.98, exact over 4,096 of 4,096
  maskings.
- Explorer, *Explain it yourself* snippet: 43,968 of 2¹⁶ maskings; O–O −1.72, N–O +1.32.
- Explorer, *Build your own*, *PAINS?*: catechol >99%, resorcinol 11%, hydroquinone 25%.
- Exact: cholesterol 2²⁸ → 129,664 (2,070×), 5,509 at 2 layers; benzene 64 = 2⁶.
- GNN test F1: *Indole?* 0.98, *Mutagenic?* 0.79, *PAINS?* 0.94.

## Design decisions (and what was tried and dropped)

- **Masking keeps bonds** and gives the atom the average atom of the GNN's training set. That is
  `GraphBaselineImputer` with a mean baseline. The paper takes the mean over the explained graph
  itself. The demos take the training set's mean, so that baselines agree across molecules,
  the reader's own included.
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
Two reviews followed (code, and a first-time visitor). They removed the caption under the molecule
(the card names it), the panels' titles that repeated the view, the card's note while hiding atoms,
and the panel's sentence on the 15-atom limit (the wheel says it). A new molecule starts at pairs.

An eighth pass moved the views into the frame, at its top (*Input*, formerly *Hide atoms*, *Atoms
alone*, *Atoms together*). *Build* is a separate amber tab after a rule, which comes in for the
reader's own molecule without moving the three. The "i" sits right of the questions. The progress
bar shows only for a job that takes longer than 350 ms (a short one flickered), and the
receptive-field figure took the amber of building (`--gsq-build`, now in the shared stylesheet).
A ninth pass dropped the *Build* tab again: the reader builds in every view, with the + and a × to
remove a group.
A tenth pass moved the molecules to start from up beside the views and centred that menu, so that
the molecule gets the room below (the atoms had become small). The menu fits one row down to
320 px phones.
An eleventh pass removed the buttons under *Input* (hide the pattern, hide all, show all). The
reader hides atoms by clicking them only. The ground truth stays outlined in every view, and the
*Input* panel shows its key too.
A twelfth pass:
- The tiles are grouped by question, in the questions' order (mutagenic, indole, PAINS), with the
  molecules that have the pattern first. The question switch takes the same order.
- The selected tile is amber. *Build your own* is light amber until it is selected.
- Two views remain, *Input* and *Explanation*. The switch at the top of the panel picks the
  explanation (`KINDS`).
- The scene draws an SI graph as shapiq's `si_graph_plot` does. A halo is an atom's own value. An
  interaction is a hyperedge, a band that narrows halfway from each of its atoms to a hub (a pair
  runs from one atom to the other). Halos and hyperedges share one scale. The bands and shapes
  (convex hulls) are gone.
- The ground truth's key is a shaded patch in the frame's bottom left.
- `patternOf` (the ground truth of the reader's own molecule) agrees with RDKit and the SMARTS of
  `compute/config.toml` on all 154,245 molecules of at most 15 atoms the builder can make.
A thirteenth pass made what is selected white on a darker, yellow amber (`--gsq-build-strong`, #c98500, 3.1:1): the
tiles and the starts. The ground-truth outline is slimmer and light amber (`--gsq-lasso`).

A fourteenth pass followed four reviews (correctness, UI/UX, accessibility and mobile, fidelity
to shapiq and the paper). The worker agrees with `ExactComputer` on every value. The fixes:
- Interactions reach **twice the layers** (two atoms meet in a third atom's field). The "i" text
  says six bonds, and the receptive-field figure counts partners up to 2L bonds away.
- The captions no longer say "alone" for k-SII singletons. A Shapley halo is a share, with each
  interaction split evenly.
- One scale per explanation, over all its values, so halos do not change size on hover.
- Hubs never land on an atom (none of the 2,982 groups of 3+ atoms in `data.js`).
- Cut-offs keep ties together, and the drawing shows every group the list shows.
- The row of molecules fades where it scrolls on (phones), before the pinned tile.
- Keyboard: the focus stays on a row through re-renders and when leaving the wheel. Escape and a
  click outside close the "i". Rows read their atoms with numbers. The live region names the
  molecule and says when a built molecule's interactions are ready.
- Touch: invisible hit circles around the + and ×, wider layer options in the second figure.
- The hint sits under the menu, off the molecule. The progress bar comes back with the molecule.
- Hidden atoms keep a faint letter. One worker script URL for all workers. "Catechol is the
  catechol itself" instead of "contains a catechol".

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

A sixteenth pass named the questions as the field does: Mutagenic, Indole and PAINS (they were
"Damages DNA?", "Has an indole?" and "Fools lab tests?"). Each card's explainer says what the term
means in one more sentence. On a phone the question row reads "Q", and it fits one row with the
"i" down to 320 px. The explanations' switch takes the panel's width under the molecule.

A seventeenth pass renamed the answer box "GNN prediction" and added a line under it, whether the
GNN is right. The true answers (`labels` in `data.js`, by `pack.py`): indole by its SMARTS, PAINS by
RDKit's PAINS filters, mutagenic by the Ames test (`ames` in `config.toml`, only where well
established, so no line for serotonin, indigo and dopamine). Indigo is a PAINS of another
kind (`ene_five_het_E`), and the GNN says yes (94%). For the reader's own molecule, PAINS is the
catechol or RDKit's `anil_no_alk` (`anilNoAlk`, checked on all 154,245 molecules of the builder),
and mutagenic has no line. The line goes while atoms are hidden. The maskings line and the band
captions of k-SII and Möbius are gone, and a panel lists at most six rows (`ROWS`).

An eighteenth pass changed the molecules after a review of their story. LSD went (it retold
serotonin's story on 24 atoms, and its Möbius values cancel). Caffeine (still in the receptive-field
figure) gave way to benzimidazole, an indole with a second nitrogen. Each nitrogen alone looks like
an indole's (+4.12 in 2-SII), the pair says no (−3.35), and the GNN is right (3.5%). It is the
set's only negative pair that explains a right no. Its explanations were computed by the
explorer's worker under Node (it agrees with shapiq to 1.6e-03), not by explain.py. A full
`pack.py` run after `explain.py` recomputes it with shapiq.

A nineteenth pass made the mark the same for every question, "✓ Correct" or "✗ Wrong" on the
label's line (the reasons of the seventeenth pass are gone), so the answer is no taller than the
card's text. A panel lists up to seven rows, the Shapley caption is gone too, and beside the
molecule the key of the colors sits at the bottom, level with the frame's.

A twentieth pass shortened the questions' explainers to two lines on a desktop, with the picture as
high as the text (6.5rem). The explanations' switch keeps its width, centred (up to 26rem when the
panel is under the molecule), and the explanation picked stays when the reader picks another
molecule.

A twenty-first pass filled the panel of the view Input. The question's card (picture, question,
explainer, the GNN's answer) is now that panel (`ask.card`), and the answer changes beside the atoms
the reader hides, with the whole molecule's answer as a faint ring on the scale. The line "N points
less than with all atoms" is gone. Above the molecule only the questions' switch stays. In the view
Explanation, and in Input when the card is under the molecule, the frame's bottom right shows the
answer small ("GNN 98% mutagenic ✓", `predKey`). The demo opens on Input.

A twenty-second pass dropped hiding atoms from the view Input (and the faint ring of the whole
molecule's answer). The reader changes a molecule only in *Build your own*, by adding and removing
groups.

A twenty-third pass fixed the receptive-field figure. Its layers' switch showed the wrong number in
the pill (the pill's copies of the labels were sized as content boxes, the buttons as border boxes:
`.gsq-opt` is now `box-sizing: border-box` everywhere). Its panel is built once and updated in place,
so hovering an atom no longer fades the panel in again, and the bars glide. On a phone the row of
molecules scrolls to the one shown.

A twenty-fourth pass took the docs' amber of the FIxLIP page (`--shapiq-amber` #f5a300, its ink
`--shapiq-amber-ink` #8c5800 and #ffc24d in dark, from `tokens.css` on the FIxLIP branch, with these
values as fallbacks here). `--gsq-build` and `--gsq-build-text` follow it, and the selected amber
is mixed from it. Red and blue were the same already.

A twenty-fifth pass compared the demos with the FIxLIP page. The score's chip and fill are amber
(`--gsq-lavender` now takes `--shapiq-amber`). The "i" is as tall as a switch, and its note opens
under the bar in the page's flow, with the FIxLIP page's growing rows. It closes with the "i", its
× or Escape (only while the focus is in the note or on the "i"), not on a click elsewhere, and is
inert while closed. Hover is gated by `(hover: hover)`, so it does not stick after a tap. On a phone
the switches keep their size and lose padding. Where the row of molecules scrolls, dots under it
(as the FIxLIP figures' dots) show the molecule shown and pick one, and both edges fade.
