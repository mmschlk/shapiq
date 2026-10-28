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
    ├── index.rst      the demo's page: the demo (`.. shapiq-demo:: <name>`) and its write-up
    ├── assets/        what the page loads: demo.js, demo.css and the demo's data
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
and adds `assets/demo.css` and `assets/demo.js` to the page; the build copies `assets/` to
`_static/demos/<name>/` before it writes the pages, so that their links to the assets carry
checksums too, and writes the pages with a demo again when its assets change. The script finds its
container by the class, loads its data from `data-root` (with `?v=` and the version, past browsers'
caches), and fills the container. Without JavaScript, the container shows the fallback image, a
file of `_static/` (all options are optional). Builders other than HTML leave demos out.

The folders `assets/` and `compute/` and this README are not pages of the docs (`exclude_patterns`
in `conf.py`).

## Add a demo

1. Make the folder `demos/<name>/` with `index.rst`, `assets/demo.js`, `assets/demo.css` and the
   data, and `compute/` with the scripts that computed the data and a README on how to run them.
2. Add a card for it to `demos/index.rst` and its page to the toctree there.
