# The orders demo: maintainer notes

This file explains how the orders demo is built, embedded and changed. It is for maintainers and
coding agents, not readers of the docs. It is not a page of the docs (`exclude_patterns` in
`docs/source/conf.py`). The demo has no page of its own: what it means is on the guide
`introduction/how-much-explained.rst`, which builds it up step by step and ends with how it was
computed. How its data was computed is in `compute/README.md`.

The demo explains one prediction at every order of interaction, from the Shapley values (order 1)
to the full decomposition (order n). The explanations are drawn as layers floating above the
example's input, with a plot of each order's fit (R^2) beside them. Above the layers, the
"problem" is drawn as the input `x`, wired through the model `f` (a chip) to the prediction `f(x)`.
A card shows the explanation that is pointed at. The demo has five examples: a molecule (GCN), one
house (XGBoost), the same house (TabPFN), 150 houses as a heatmap, and an image and caption
(SigLIP 2).

The demo is plain JavaScript and CSS, with no build step and no dependencies. A single
self-contained script fills every container on a page.

## Files

| Path | What it is |
|---|---|
| `assets/demo.js` | The whole demo. A single IIFE that finds every `.orders-demo` on the page, loads `data.js` once, and builds each demo. The header comment describes the behavior; the comments inside describe the geometry. |
| `assets/demo.css` | All styles, scoped under `.orders-demo`. Colors are custom properties (`--orders-*`) on `.orders-demo`, with dark-mode overrides. |
| `assets/data.js` | The data: `window.SHAPIQ_ORDERS = {...}`, written by `compute/pack.py`. Never edit it by hand. |
| `compute/` | The scripts that computed the data, and `compute/README.md` on how to run them. They are run by hand, not by the docs build. |
| `../index.rst` | The Demos gallery: one card per demo. This demo's card shows `/_static/images/hero_orders_demo.png` and opens the guide below. |
| `../../index.rst` | The landing page, which embeds the full demo. |
| `../../introduction/how-much-explained.rst` | The guide, and the demo's page: it builds the figure up step by step (five demos, one per `:step:`, the last one all of it), and after a line, "Further Information": how the fit is measured, the examples, how they were computed, and how to run it again. |
| `../../_static/images/hero_orders.png` | The static figure. It is shown without JavaScript, and when the demo cannot start. |
| `../../_static/images/hero_orders_demo.png` | The gallery card's image: a screenshot of the wide-screen layout, 1952 x 1252, the same size as the FIxLIP card's. |
| `../../_ext/shapiq_demos.py` | The Sphinx extension behind `.. shapiq-demo::`. It is shared by all demos (see `../README.md`). |

The FIxLIP example loads its photo from the FIxLIP demo's assets (`image: "../fixlip/elephant.jpg"`
in `data.js`, relative to `data-root`). The extension copies every demo's `assets/` into the built
docs, so the photo is always there. A page outside these docs needs both folders.

## Embed it in a page

```rst
.. shapiq-demo:: orders
   :fallback: images/hero_orders.png
   :alt: What the static figure shows.
```

All options are optional:

| Option | HTML attribute | Values | Effect |
|---|---|---|---|
| `:anchor:` | `id` | any id | An id, to link to the demo. |
| `:fallback:` | `<noscript><img>` | a path under `_static/` | The image shown without JavaScript, and when the demo fails. Without it, the demo shows nothing then. (The write-up's first four steps have none, on purpose: the last step shows the whole figure.) |
| `:alt:` | the image's `alt` | text | The fallback image's alt text. |
| `:step:` | `data-step` | `problem`, `mechanism`, `shapley`, `interactions`, `full` | Shows one step of the step-by-step figure (see below). Without it, the demo shows everything. |
| `:select:` | `data-select` | `none`, `mechanism`, `full`, or an order (`1`, `2`, ...) | What the demo points at while the reader points at nothing. |

The directive writes the container below, and adds `demo.css` and `demo.js` to the page once:

```html
<div class="orders-demo" data-root="../_static/demos/orders/" data-version="<crc32 of data.js>"
     data-step="..." data-select="...">
  <noscript><img class="orders-demo-fallback" src="..." alt="..."></noscript>
</div>
```

Two more attributes are read by the script, but have no directive option. To use them, add them to
`DemoDirective.option_spec` and `_visit_demo` in `_ext/shapiq_demos.py`, or write the container as
raw HTML:

- `data-reveal="open"`: start already explained. By default, a demo without a step starts with
  the problem alone, and a click on the chip explains it. (The `problem` step always shows the
  problem alone; the other steps start explained.)
- `data-plan="above"`: on a wide screen, keep the explanation card above the layers instead of
  beside them. The script watches this attribute, so it can be changed live.

The directive also adds `data.js` to the page, deferred, just before `demo.js`, so that both load
at once. When the script runs and `window.SHAPIQ_ORDERS` is already defined, it loads nothing.
Otherwise (a page written by hand) it loads the data itself from `data-root` of the first
container, with `?v=<data-version>` to get past caches; `data-root` must end with `/`. Every demo
then starts in a task of its own (`setTimeout`), so that the page can respond and paint between
them.

### Steps

`:step:` is for a post that builds the figure up step by step, one demo per step. Each step shows
what the previous one showed, plus:

| Step | Shows | Points at, at rest |
|---|---|---|
| `problem` | The problem alone: `x`, `f`, `f(x)`. | Nothing. |
| `mechanism` | The input's card, and the mechanism (the model) below it. | The mechanism. |
| `shapley` | The Shapley values above, and the plot of their fit. | Order 1. |
| `interactions` | The explanations of orders 1 to 4. | Order 2. |
| `full` | Everything, as with no step. | Nothing. |

### Several demos on one page

All containers on a page share one data load. They also show the same example: choosing an
example in any of them chooses it in all. The page can choose an example, and hear every choice,
through a `CustomEvent` on `document`:

```js
// choose an example in every demo of the page
document.dispatchEvent(new CustomEvent("orders-demo-example", { detail: { example: "house" } }));
// hear every choice (detail.example is the example's id)
document.addEventListener("orders-demo-example", (event) => console.log(event.detail.example));
```

The example ids, in the order of their tabs, are `molecule`, `house`, `tabpfn`, `houses` and
`fixlip`.

## Layout

The layout follows the container's width, through container queries and `ResizeObserver`, not the
window's width. The container must have a definite width from its parent. It sets `clear: both`
and vertical margins of its own.

| Container width | Layout |
|---|---|
| below 40rem | Phone: the problem's parts are compact, and the card is above the layers. |
| below 45rem | The examples are behind a stepper (arrows, dots below the layers, and a swipe across them) instead of tabs. |
| 640px and up | `orders-demo-wide`. If the demo shows everything (no step, or `full`), has been explained, and has no `data-plan="above"`, it also gets `orders-demo-beside`: the card stands left of the layers (below). Otherwise the card is above the layers. |

With the docs' 16px root font, 40rem is 640px, so the phone layout and the wide one meet there.

The stage (the layers, the slider left of them and the plot right of them) has a narrow variant
below 560px of its own width (`orders-demo-narrow`): smaller gaps, and a narrower plot.

The window's height also bounds the layers, in every layout. It is read as
`document.documentElement.clientHeight` (`viewportHeight()`), not `innerHeight`, so that a phone's
toolbars coming and going as the page scrolls do not resize the stack. (On a page without a
doctype, in quirks mode, the root's height is the whole page's: there it reads `innerHeight`.) With the card above the
layers, the stack is at most the window's height minus 96px (`WINDOW_ROOM`), which matters on a
phone on its side. `relayout()` runs again when the stage's width or the window's height
changes.

### Beside the layers (`orders-demo-beside`)

On a wide screen, `fitColumns()` in `demo.js` sizes everything to the window:

- **The layers** are as large as the window's height allows, below what is above them in the
  demo, minus 24px (`WINDOW_MARGIN`). A layer's side is clamped to 110 to 360px.
- **The card** grows and shrinks with the layers. Its face is 0.8 of the layers' width
  (`FACE_BESIDE`). Its column is as wide as the face, but never narrower than 14rem
  (`WORDS_BESIDE`), so that its words stay legible. The words scale with the column, at most 1.3
  times their size (`settleCard()`).
- **When the demo is too narrow** for both, the layers and the card shrink together. The largest
  size that fits is found by bisection.

The CSS grid has four columns: `1fr`, the card, the stage, `1fr`. The script sets three custom
properties on the container:

- `--orders-why-column`: the card's column, including the 1.25rem gap before the stage, which is
  the card's right margin.
- `--orders-why-face`: the face's width.
- `--orders-stage-column`: the stage's column.

The width left over goes to the two outer columns. The tabs and the problem span all four, so they
keep the demo's full width. Do not let the tabs wrap: a taller bar pushes the stage down, and the
layers shrink to fit the window.

### Theme and motion

- **Colors** come from the docs' tokens (`--color-foreground-primary`, `--shapiq-surface`, ... from
  Furo and `_static/css/tokens.css`), with fallbacks, so the demo also works on a page without
  them.
- **Dark mode** follows `body[data-theme="dark"]`, and `prefers-color-scheme: dark` unless
  `body[data-theme="light"]` (Furo's three modes). The script re-inks the chip when `data-theme`
  changes.
- **Motion**: `prefers-reduced-motion: reduce` turns off every transition and animation in the demo
  (CSS, pseudo-elements included), and the script's own movement.
- **Browsers**: container queries and `cqw` need Safari 16 and Firefox 110; `color-mix()` needs
  Safari 16.2 and Firefox 113. Where `color-mix()` is missing, the chip falls back to its solid
  color (an `@supports` rule), so its words stay legible.

## Input and states

- **Pointing** at an explanation is how the reader explores. It can be done by hovering a layer
  or the plot, by keyboard focus on a row of the plot, by clicking or tapping a layer or the plot
  (a tap pins the explanation, and another tap lets go), or with the slider.
- **The slider**: with a mouse, a press anywhere on its column goes to the stop there and drags
  on. On a touch screen, a drag starts only at the handle (its hit area is 44px), and a tap on the
  column goes to the stop there. A touch that scrolls the page, starting on the column or on the
  plot, changes nothing. Touch pointers are captured implicitly by what they land on, so the drag
  is tracked by its own pointer id, not by `hasPointerCapture()`.
- **Touch gestures**: a vertical drag scrolls the page and a pinch zooms it, everywhere on the
  demo (`touch-action: pan-y pinch-zoom`; `auto` in Safari, which does not support
  `pinch-zoom`). A horizontal swipe across the layers steps through the examples, unless the page
  is zoomed in. Tap highlights are off.
- **Keyboard**: the chip explains with Enter or Space. The plot's rows are buttons: ArrowUp and
  ArrowDown move between them, and Escape lets go. The slider's handle moves with the arrow keys,
  Home and End.

- **Screen readers**: a visually hidden status line (`.orders-demo-status`, `aria-live`) says the
  example chosen, once, in the demo where it was chosen. The problem's parts are not a live region,
  since they flip through random letters to the next example. After the chip is explained from
  the keyboard, focus moves to the slider's handle.

The script toggles state classes on the container, which are useful in tests:

| Class | Meaning |
|---|---|
| `orders-demo-ready` | The demo has started. |
| `orders-demo-failed` | The data did not load, or the demo threw while starting. The fallback is shown: the container keeps no other class and no inline style, and the demo's observers and its window and document listeners are removed (`start()` ties them to an `AbortSignal`, `life`). |
| `orders-demo-veiled` | Not explained yet: the problem alone. |
| `orders-demo-revealing` | The build-up after the click on the chip. |
| `orders-demo-wide`, `orders-demo-beside`, `orders-demo-narrow` | The layouts above. |

The chip is `.orders-demo-wire-chip`. Before it has been clicked, it also has
`.orders-demo-chip-call`. To explain a demo in a test, click it: `chip.click()`.

## Data (`window.SHAPIQ_ORDERS`)

Written by `compute/pack.py`. Shape, in short:

```js
{
  index: "FSII",                       // the interaction index of every order
  examples: [{
    id, label, detail,                 // the tab: "molecule", "Molecule", "GCN"
    kind,                              // "graph" | "tree" | "heatmap" | "image-text": how faces are drawn
    model: { icon, title, detail, method, how, short },  // the chip and the card's "Explained with"
    problem: { input, output, value }, // the problem's parts: x, f(x)
    sheets: [{                         // the layers, top to bottom: orders 1..4, then the full one
      order, full,                     // full: true for the full decomposition (order n)
      r2,                              // the fit, R^2 beyond size
      first, groups, count,            // the main effects, the strongest groups, how many there are
    }],                                // (heatmap: count, scale, rows instead)
    between,                           // optional: the blank orders' explanations, by order
    fits,                              // the plot: every order's R^2
    largest,                           // the highest order with an interaction (the trees' depth,
                                       // the network's receptive field, the proxy's depth): every
                                       // order above it equals the full decomposition
    // and, by kind: atoms, bonds, coords (graph); features, values, layout (tree); image, grid,
    // tokens (image-text); features, predictions, quartiles (heatmap)
  }],
}
```

To change the examples, change and run the scripts in `compute/`, then `pack.py`. Adding an example
of a new `kind` also needs a face for it in `demo.js` (next to `moleculeFace`, `treeFace`,
`heatmapFace` and `imageTextFace`), its players' words (`PLAYERS`), and its input on the
problem's card.

## Working on it

1. **Build** the docs (from the repository's root):

   ```bash
   python -m sphinx -b html -D plot_gallery=0 docs/source docs/build/html
   ```

   `-D plot_gallery=0` skips running the examples. The demo is on `index.html` and
   `introduction/how-much-explained.html`.

2. **Iterate**: after a full build once, copy `assets/demo.js` and `assets/demo.css` into
   `docs/build/html/_static/demos/orders/` and reload the page. A change to `data.js` changes
   `data-version`, so rebuild then.

3. **Check** in a browser (Playwright works well), at least at these sizes:

   | Viewport | Page | What it covers |
   |---|---|---|
   | 1440 x 900, and a short window such as 1440 x 700 | `index.html` | Beside the layers: the stack and the card fit the window, and the card scales with the stack. |
   | 1280 x 800 | `introduction/how-much-explained.html` | The docs' content width, beside the layers (the last step). |
   | 390 x 844, 360 x 740 | both pages | Phones: the stepper, no horizontal overflow. |
   | any | `introduction/how-much-explained.html` | The five steps. |

   For each, explain the demo (click the chip), step through the examples, point at layers, and
   check for console errors, horizontal overflow (`document.documentElement.scrollWidth >
   innerWidth`), and text that collides along the slider.

4. **Keep the static figure current**: if the look changes a lot, take new screenshots for
   `hero_orders.png` (the fallback) and `hero_orders_demo.png` (the card). The card's image is
   the landing page's demo in a 1440 x 652 window, explained, with the mouse on the plot's row of
   order 2. At that height the demo is 976 x 628 CSS px, nearly the FIxLIP card's aspect. It is
   captured at a device scale factor of 1.879, so that it fits, unscaled, inside a 1952 x 1252
   frame like `hero_fixlip_demo.png`'s: white, a `#e2e6f3` edge of 2 image px (1 CSS px at 2x),
   corners of 26 image px (13 CSS px), the demo in its middle. The FIxLIP card's image is the
   FIxLIP demo, 976 CSS px wide, at a device scale factor of 1.918 (the giraffe, "Clique", CLIP,
   6 players, its invitation dismissed), in the same frame.

### Things that are easy to break

- **`place()`** runs on every pointer move over the layers and every step of a slider drag. Do not
  read layout in it (`offsetHeight`, `getBoundingClientRect`); cache measurements, as
  `labelLength()` does.
- **Stable sizes**: the card keeps its size whatever it shows, in every example. `settleCard()`
  measures the most it can say, out of view, and fixes its width and height. When you add words to
  the card, add them to `wordsOf()`, so that they are measured too.
- **Re-layout**: a new layout state must reach `relayout()`. It runs on the container's
  `ResizeObserver`, on window resize, and on `data-plan` changes.
- **No globals**: everything is scoped under `.orders-demo` (CSS) and inside the IIFE (JS). The
  only globals are `window.SHAPIQ_ORDERS`, and the `orders-demo-example` event on `document`.

### Performance

The demo is meant to cost nothing while nobody uses it. Keep it that way:

- **Endless animations** move only `transform` and `opacity`, which the compositor runs off the
  main thread. The wire's flowing dashes are a strip in a clipping box, moved by a transform
  (`.orders-demo-wire-dashes::before`); a moving `background-position` was repainted every frame.
  Out of view, the container gets `orders-demo-away`, and every endless animation pauses. The
  invitation stops bobbing once it is gone. Measured idle cost: about 2 ms/s in view and 0
  offscreen, down from about 110 ms/s (450 ms/s with a 4x throttled CPU).
- **No layout reads on pointing**: the demo is a size container, so `getComputedStyle()` or a
  size read lays it all out. `inkChip()` reads the chip's colors only when its kind or the theme
  changes, and the chip's note is refit only when its shape, not just its digits, changes.
- **`settleCard()`** puts every explanation's words into the probe at once and reads them after
  one layout, instead of laying out once per explanation (about 60 times).
- **Offscreen demos**: an example chosen in another demo is drawn in a demo out of view only once
  it comes within a screen of the view, or when it is laid out anew (`stale`).
- **Steps** draw only the sheets they show (`sheetsOf()`, `drawAhead()`).
- **Nothing per frame**: there is no `requestAnimationFrame` loop that runs without changing
  something. The light out of the chip is redrawn by `relayout()` and the plan's
  `ResizeObserver`.
