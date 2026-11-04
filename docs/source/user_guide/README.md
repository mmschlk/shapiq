# Writing the user guide

Notes for the authors of the user guide. This file is not a page of the docs (`exclude_patterns` in
`docs/source/conf.py`).

## Math

Inline math is `$...$`, displayed math `$$...$$`. A label after the closing `$$` numbers the
equation and makes it referable:

```markdown
$$
F_S(x) = (\mask \model)(x, S), \qquad S \subseteq \players.
$$ (eq-masking)

As {eq}`eq-masking` shows, ...
```

The notation is defined once, as MathJax macros in `docs/source/conf.py` (`mathjax3_config`), so a
symbol can change in one place:

| Macro | Renders | Meaning |
| --- | --- | --- |
| `\players` | $N$ | the players (features) |
| `\nplayers` | $n$ | the number of players |
| `\model` | $F$ | the model |
| `\mask` | $\mathcal{M}$ | the masking operator |
| `\behav` | $\mathcal{B}$ | the behavior operator |
| `\inter` | $\mathcal{I}$ | the interaction operator |
| `\game` | $\nu$ | the explanation game |
| `\E`, `\Var`, `\R` | $\mathbb{E}$, $\mathbb{V}$, $\mathbb{R}$ | expectation, variance, real numbers |

## Citations

Keys are `Author.Year` (e.g. `Fumagalli.2025`, `Herbinger.2026`). `` {cite:p}`Fumagalli.2025` ``
cites in parentheses, `` {cite:t}`Fumagalli.2025` `` in the text. New entries go into
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
