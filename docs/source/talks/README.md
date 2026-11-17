# Talks: maintainer notes

This file is for maintainers and coding agents, not readers of the docs. It is not a page of the
docs (`exclude_patterns` in `docs/source/conf.py`).

A talk is a page of slides, `talks/<name>/index.rst`, listed on `talks/index.rst`. The extension
`_ext/shapiq_talks.py` (its docstring has the details) gives it three directives:

- `shapiq-slide:: <title>` makes a slide, with `:layout:` (`title`, `section`, `wide`, or none),
  `:steps:` and `:name:`. Slides get the ids `slide-1`, `slide-2`, and so on.
- `shapiq-notes::` inside a slide holds what to say. It shows below the slide on the page and with
  the key N in the deck.
- `shapiq-talk-figure:: <name>` adds a figure that `talks/<name>/assets/figures.js` draws.

A part of a slide with the class `shapiq-from-<k>` (`.. rst-class:: shapiq-from-1`, or a
`container`) shows from step k on. Demos go on a slide with `shapiq-demo`, as on any page.

`_static/js/talk.js` and `_static/css/talk.css` are the deck. On the page, the slides read like a
handout, every step shown. "Present" or P moves the slides into a deck that fills the window. The
keys are listed at the top of `talk.js`. The address follows the slide (`#slide-3`), and
`?present` opens a page as a deck. The deck stays on after a reload (sessionStorage).

The text size of a slide follows the window (`font-size` of `.shapiq-deck`). Slides of the layout
`wide` keep the docs' text size for the demos. A slide whose content is taller than the window
scrolls.
