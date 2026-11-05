# Writing the user guide

Notes for the authors of the user guide. This file is not a page of the docs (`exclude_patterns` in
`docs/source/conf.py`). The rules that pages follow (colors, references, notation, structure) are in
`docs/STYLE_GUIDE.md`.

## Structure

One folder per part, one file per chapter. The numbers are automatic (`:numbered:` in `index.md`):
the part pages are 1, 2, 3, their chapters 1.1, 1.2, ..., and the sections of a chapter 1.1.1, ...

A part page (`<part>/index.md`) holds a short introduction, the sentence "This part is organized as
follows:", and the toctree of its chapters. It has no `##` sections: they would be numbered
together with the chapters.

## Math

Inline math is `$...$`, displayed math `$$...$$`. A label after the closing `$$` numbers the
equation and makes it referable:

```markdown
$$
F_S(x) = (\mask \model)(x, S), \qquad S \subseteq \players.
$$ (eq-masking)

As {eq}`eq-masking` shows, ...
```

The notation is defined once, as MathJax macros in `docs/source/conf.py` (`_NOTATION` and
`mathjax3_config`), so a symbol can change in one place. The symbols of `_NOTATION` link to their
row of the notation page (`notation.md`). A new symbol needs both: its macro in `_NOTATION`, and a
row in `notation.md` that starts with `<span id="notation-<macro>"></span>`.

| Macro | Renders | Meaning |
| --- | --- | --- |
| `\players` | $D$ | the players (features) |
| `\nplayers` | $d$ | the number of players |
| `\featurespace` | $\mathcal{X}$ | the feature space |
| `\outputspace` | $\mathcal{Y}$ | the output space |
| `\model` | $f$ | the model |
| `\units` | $\mathcal{S}$ | the units of an explanation |
| `\explspace` | $E$ | the explanation space |
| `\expl` | $e$ | an explanation |
| `\mask` | $\mathcal{M}$ | the masking operator |
| `\behav` | $\mathcal{B}$ | the behavior operator |
| `\inter` | $\mathcal{I}$ | the interaction operator |
| `\game` | $\nu$ | the explanation game |
| `\E`, `\Var`, `\R` | $\mathbb{E}$, $\mathbb{V}$, $\mathbb{R}$ | expectation, variance, real numbers |

## Illustrations

Interactive figures go in with the `shapiq-illustration` directive; how they work and how to
add one is in `illustrations/README.md`.

## Citations

Keys are `Author.Year` (e.g. `Fumagalli.2025`, `Herbinger.2026`). `` {cite:p}`Fumagalli.2025` ``
cites in parentheses, `` {cite:t}`Fumagalli.2025` `` in the text. A note after the year goes in
braces after the key: `` {cite:t}`Muschalik.2026{Def. 2.1}` `` gives "Muschalik [2026, Def. 2.1]".
Several keys are separated by commas. New entries go into
`docs/source/references.bib`; the pre-commit hook `format-bib` formats them.

## Comments and markers

Lines starting with `%` are comments and do not render. Markers for Claude in the text:
`[cite: GRANITE]` (find the key and cite it) and `TODO eq: ...` (typeset the equation).

## Building

The full build runs the examples of the gallery and takes long. For writing, skip them:

```bash
cd docs && make html SPHINXOPTS="-D plot_gallery=0"
```

The pages are then in `docs/build/html/user_guide/`.
