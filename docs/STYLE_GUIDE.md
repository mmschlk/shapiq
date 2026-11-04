# Style Guide of the Docs

Rules for the docs of shapiq, in particular for the user guide. Follow them when you write or change
a page. To break a rule, change it here first.

How to write pages (math, citations, comments, building) is in `source/user_guide/README.md`.

## Colors

- **Red and blue carry meaning.** They are the colors of attributions: red is positive, blue is
  negative, as in the logo and in the plots of shapiq. Use them with this meaning in figures and
  examples, not as decoration. The theme's own colors (the blue of links, the red of the current page
  in the navigation) stay as they are.
- **Amber marks references** to numbered elements of the docs: equations, figures, tables, and the
  symbols of the notation. Use amber for nothing else.
- Take colors from the tokens of `source/_static/css/tokens.css` (`--shapiq-red`, `--shapiq-blue`,
  `--shapiq-amber`, ...), not as hex values in a rule.

## References

- **References to numbered elements are underlined in amber**, and more pronounced on hover: a
  thicker underline and an amber offset shadow. `custom.css` styles references made with `{eq}`
  (equations) and `{numref}` (figures, tables) this way. Refer to a numbered element with these
  roles, never with a plain link or as plain text such as "(1)".
- **Symbols of the notation are amber only on hover**, so that equations stay clean. Each links to
  its row of the notation page.
- Links to pages (`{doc}`, `{ref}`) and citations keep the theme's link color.

## Notation

- Write each symbol of the notation with its macro (`\players`, `\model`, `\mask`, ...), never as the
  plain letter: the macro links the symbol to the notation page, and a symbol changes in one place.
- A new symbol needs both its macro in `_NOTATION` (`source/conf.py`) and a row on the notation page
  (`source/user_guide/notation.md`).

## Structure of the User Guide

- **Parts and chapters, as in scikit-learn.** A part page (`<part>/index.md`) holds a short
  introduction, the sentence "This part is organized as follows:", and the toctree of its chapters.
  The first page with content is a chapter: 1.1, not 1.
- **Part pages have no sections**: their `##` headings would be numbered among the chapters.
- **One page per chapter**, with its sections on the page. No deeper levels of pages: content is at
  most two clicks away from the user guide's start page.

## Writing

- **Title Case** for the titles of pages and sections.
- **Titles name their subject; they are not questions**: "Feature-Based Explanations", not "What Is
  a Feature-Based Explanation?". A chapter does not repeat the title of its part.
- **Cite the source** of every definition, result, and method, with the keys of `references.bib`
  (`Author.Year`). A note after the key points into the source: `` {cite:t}`Muschalik.2026{Def. 2.1}` ``.
- **Label every displayed equation the text refers to**, and refer to it with `{eq}`.
- Notes for authors (sources, open decisions) go into comments (`%`), which do not render.

## Checking a Change

- Build with the gallery skipped (`make html SPHINXOPTS="-D plot_gallery=0"` in `docs`): the user
  guide builds without warnings.
- Look at a changed page in the light and in the dark theme.
