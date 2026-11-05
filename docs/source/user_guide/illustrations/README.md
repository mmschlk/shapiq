# Illustrations

The interactive figures of the user guide: each explains one idea, where a demo (`demos/`) shows a
result. They are built with the page's own HTML, SVG, CSS, and JavaScript, without libraries, and
they work for docs opened from disk.

## One folder per illustration

```
user_guide/illustrations/
├── README.md
└── <name>/
    ├── assets/        what the page loads: illustration.js, illustration.css, and data.js if any
    └── compute/       the script that computed data.js, if any, with a README on how to run it
```

The `shapiq-illustration` directive (`_ext/shapiq_illustrations.py`) embeds an illustration in a
page; its content is the caption:

````markdown
```{shapiq-illustration} vocabulary
:anchor: illustration-vocabulary
:alt: What the illustration shows, for readers without scripts or eyes.

The caption, with **markup**, $math$, and {cite:p}`Key`.
```
````

It writes `<figure class="shapiq-illustration" data-illustration="<name>" data-root="...">` with an
empty stage, and adds the shared files of all illustrations (`_static/js/illustrations.js`,
`_static/css/illustrations.css`) and the illustration's own to the page; the build copies `assets/`
to `_static/illustrations/<name>/`. The folders `assets/` and `compute/` and this README are not
pages of the docs (`exclude_patterns` in `conf.py`).

## The illustrations

| Name | Page | Shows |
| --- | --- | --- |
| `vocabulary` | 2.1 Feature-Based Explanations | features are the universal language of models: four inputs split into features, which one model reads |
| `units-and-forms` | 2.1 Feature-Based Explanations | Definition 2.1 at work: the units and the explanation space of an explanation of one house |

## Add an illustration

1. Make the folder `<name>/assets/` with `illustration.js` and `illustration.css`. The script calls
   `window.shapiqIllustrations?.define("<name>", (stage, kit) => { ... })` and builds the
   illustration in `stage`; `kit` has the URL of the assets (`kit.root`), `kit.loadScript` for
   `data.js`, tweens that respect reduced motion, a visibility watcher, and the segmented control
   that the illustrations share (see `_static/js/illustrations.js`).
2. If it shows values, compute them with a script in `<name>/compute/` that writes
   `assets/data.js`, and commit both: no invented numbers.
3. Embed it with the directive, and add it to the table above.

The rules for illustrations (colors, motion, themes, widths) are in `docs/STYLE_GUIDE.md`.
