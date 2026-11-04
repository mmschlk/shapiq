# Writing the user guide

Notes for the authors of the user guide. This file is not a page of the docs (`exclude_patterns` in
`docs/source/conf.py`).

## Structure

One folder per part, one file per chapter, sections as headings within a chapter. The parts are
numbered automatically (`:numbered:` in `index.md`): Part II, chapter 3, section 2 shows as "2.3.2".
Part index pages hold only a title, an intent comment and their toctree; sections on them would mix
into the numbering of the chapters.

| Folder | Part |
| --- | --- |
| `perspectives/` | I. Three perspectives on post-hoc explanations |
| `framework/` | II. The unified framework |
| `methods/` | III. Explanation methods |
| `computation/` | IV. Computing explanations |
| `results/` | V. Working with explanations |
| `guidance/` | VI. Guidance |
| `beyond/` | VII. Beyond feature-based explanations |

## Comments

MyST comments start a line with `%` and do not render. The scaffold uses them for:

- `% intent:` what the section is for
- `% sources:` where the content comes from (keys of `references.bib`, sections of papers or of the thesis)
- `% card:` the coordinates of a method in the framework (Part III)
- `% neighbors:` methods that differ in one operator
- `% v1:` / `% v2:` the shapiq objects of the current release and of shapiq v2
- `% open:` a decision that is still open
- `% TODO eq:` / `% TODO fig:` / `% TODO table` an equation, figure, or table to add

Markers for Claude in the text itself: `[cite: GRANITE]` (find the key and cite it) and
`TODO eq: ...` (typeset the equation).

## Math

Inline math is `$...$`, displayed math `$$...$$`. A label after the closing `$$` numbers the
equation and makes it referable:

```markdown
$$
F_S(x) = (\mask \model)(x, S), \qquad S \subseteq \players.
$$ (eq-masking)

As {eq}`eq-masking` shows, ...
```

Use the macros of `conf.py` (`mathjax3_config`) for the notation, so that it can change in one place.
They are listed in `notation.md`.

## Citations

Keys are `Author.Year` (e.g. `Fumagalli.2025`, `Herbinger.2026`). `` {cite:p}`Fumagalli.2025` ``
cites in parentheses, `` {cite:t}`Fumagalli.2025` `` in the text. New entries go into
`docs/source/references.bib`; the pre-commit hook `format-bib` formats them.

## The method card

Every method of Part III gets the same card. Its first row is the method's place in the framework,
the same columns as the map of methods:

```markdown
### Partial dependence plot (PDP)

| Masking | Behavior | Interaction | Unit |
| --- | --- | --- | --- |
| marginal | local, averaged over the data | pure | feature or group |

**Question.** How does the prediction change, on average, with the value of a feature?

**Definition.** The explanation in the operators of Part II, e.g.
$\phi(S) = (\inter(\behav(\mask \model)))(S)$ with ...

**Neighbors.** M-plot (conditional masking), ICE (full effect, baseline masking).

**Estimation.** How it is computed or estimated; link to Part IV.

**In shapiq.** Code.

**Pitfalls.** ...

**References.** {cite:p}`Friedman.2001`
```

## Building

The full build runs the examples of the gallery and takes long. For writing, skip them:

```bash
cd docs && make html SPHINXOPTS="-D plot_gallery=0"
```

The pages are then in `docs/build/html/user_guide/`.
