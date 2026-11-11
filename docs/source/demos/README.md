# Demos

The [Demos](https://shapiq.readthedocs.io/en/latest/demos/index.html) of the docs: what `shapiq`
explains on real models and data, computed once and shown on a page. Unlike the examples
(`examples/`), which run with every build of the docs and show how to use the API, a demo is not
run with the docs: its data is computed by hand and committed.

## One folder per demo

```
demos/
├── index.rst          the Demos page: a card per demo, and the toctree
└── <name>/
    ├── index.rst      the demo's page: the demo (`.. shapiq-demo:: <name>`) and its write-up (the
    │                  orders demo has none: its card opens the guide that builds it up)
    ├── README.md      optional: notes for maintainers on how the demo is built and embedded
    ├── assets/        what the page loads: demo.js, demo.css and the demo's data
    ├── figures/       optional: the static images of the write-up (the FIxLIP demo has one)
    └── compute/       the scripts that computed the data, with a README on how to run them
```

The `shapiq-demo` directive (`_ext/shapiq_demos.py`) embeds a demo in a page, its own or another
one (the landing page embeds the FIxLIP demo):

```rst
.. shapiq-demo:: fixlip
   :anchor: fixlip-demo
   :fallback: images/hero_siglip2.png
   :alt: What the fallback image shows.
```

It writes `<div class="<name>-demo" data-root="..." data-version="...">`, where `data-root` is the
URL of the demo's assets relative to the page and `data-version` a checksum of `assets/data.js`,
and adds `assets/demo.css`, `assets/data.js` (if there is one) and `assets/demo.js` to the page,
the data deferred just before the script, so that both load at once; the build copies `assets/` to
`_static/demos/<name>/` before it writes the pages, so that their links to the assets carry
checksums too, and writes the pages with a demo again when its assets change. The script finds its
container by the class, uses the data the page loaded (or, where it is missing, loads it from
`data-root`, with `?v=` and the version, past browsers' caches), and fills the container. Without
JavaScript, the container shows the fallback image, a file of `_static/` (all options are
optional). Builders other than HTML leave demos out.

Three more options are passed on to the demo's script, as `data-step`, `data-select` and
`data-examples`, for the demos that read them (the orders demo does, the FIxLIP demo does not):
`:step:` shows one step of a figure built up step by step, `:select:` what it points at at rest,
and `:examples:` only the examples it names, in that order (`:examples: tabpfn`, or
`house, houses`). `orders/README.md` lists their values.

A write-up's figures are drawn in the browser, so that they follow the theme of the docs, the
width of the page, and what is chosen in the demo. The `shapiq-demo-figure` directive names the
demo and the figure, with a description for screen readers and the caption as its content:

```rst
.. shapiq-demo-figure:: fixlip masks
   :alt: What the figure shows.

   The caption.
```

It writes `<figure class="<name>-figure" data-figure="..." data-root="...">` with an empty stage
and the caption, and adds `assets/figures.css`, `assets/figures-data.js` (if there is one) and
`assets/figures.js` to the page, next to the demo's data. The script draws every figure into its
stage. Without JavaScript, the stage shows the description. The figures of the FIxLIP demo show
the image–text pair chosen in the demo. Each of them has its own selector, and the demo and the
figures tell each other about a choice with the event `fixlip-demo-image` on the document.
Builders other than HTML keep only the caption.

On a page with such figures, the extension numbers the captions in the order of the page. The
figures of the directive and the `figure` directives with a caption become Figure 1, Figure 2, and
so on, and the tables with a title become Table 1, and so on. The figures are narrower than the
text column and sit in its middle (`--fixlip-figure-width` in `figures.css`). The text refers to a
figure or a table with the role `shapiq-figure`, which becomes its number and a link to it. The
target is the name of a demo's figure, or the `:name:` of a `figure` or a table:

```rst
:shapiq-figure:`masks` shows eight masks.
```

Demos can share code. A folder of `demos/` without an `index.rst`, whose `assets/` hold a
`demo.css` and a `demo.js` too, is a library: a demo names the libraries it needs in its
`assets/requires.txt`, one per line (`#` starts a comment), and every page that embeds the demo gets
their stylesheets and scripts before its own, with checksums, and is written again when they change.
The four GraphSHAP-IQ demos share `graphshapiq-common`, whose `compute/` computes all their data
(its README has notes for working on them).

The folders `assets/` and `compute/` and this README are not pages of the docs (`exclude_patterns`
in `conf.py`).

## Add a demo

1. Make the folder `demos/<name>/` with `index.rst`, `assets/demo.js`, `assets/demo.css` and the
   data, and `compute/` with the scripts that computed the data and a README on how to run them.
2. Add a card for it to `demos/index.rst` and its page to the toctree there. The cards' images
   (`_static/images/hero_<name>_demo.png`) are screenshots of the demos, 1952 x 1252: the demo at
   about 2x, in the middle of a white frame with a 2 px `#e2e6f3` edge and 26 px corners, so that
   the cards match (see `orders/README.md`).
