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

## As a PDF

"PDF", beside "Present" and in the corner of the deck (and Ctrl+P or ⌘P in the deck), lays the
slides out for print and opens the browser's print dialog. "Save as PDF" there makes a PDF of one
slide a page, 16:9. Not on a phone. It is for venues where the deck cannot run. Nothing else is
needed for it.

- Every slide is laid out at 1600 x 900 px, the size of its page, as the deck shows it in a window
  of that size (the docs' text at 16px). The rules that size the deck by the window (`vw`, `vh`)
  are copied with the page's sizes while it is laid out (`pageSheet()` in `talk.js`), so the
  demos measure on the screen the room they get on paper, whatever the window.
- A slide shows its last step, in the light theme, without notes, "Try it", the corner or
  anything that moves (no transitions or animations). A slide too tall is scaled down, never cut
  off.
- The demos and figures hear `shapiq-talk-print` (detail: `printing`), and the root has the class
  `shapiq-talk-print`. The orders demo explains itself, its layers apart, and leaves out its tabs,
  its picker of photos and its button in the corner. The FIxLIP demo leaves out its "i" and hint.
- A slide with a demo or a figure has a line at its foot, "Live" and a link to the slide in the
  deck, and "Demo" and a link to the demo's own page (`data-page` of the demo's container, from
  `_ext/shapiq_demos.py`). The canonical URL of the page, where there is one, else its address.
- `?print` lays the slides out without the dialog, to look at or to test.
  `html[data-talk-print="ready"]` tells when all of it has settled. Esc goes back to the page.

To make the PDF in a test (Playwright): open `index.html?print`, wait for
`html[data-talk-print="ready"]`, then `page.emulateMedia({media: "print"})` and
`page.pdf({printBackground: true, preferCSSPageSize: true})`.

## Text size

The text size of a slide follows the window (`font-size` of `.shapiq-deck`). Slides of the layout
`wide` keep the docs' text size for the demos. A slide whose content is taller than the window
scrolls.

## For readers during and after the talk

No backend: everything is static. Below a slide, on the page and in the deck on a phone:

- `shapiq-try::` code that reproduces the slide ("Try it in shapiq"). Run each snippet before you
  commit it.
- `:cite:` of a slide lists its references from `references.bib`, each with a button that copies
  its BibTeX.
- `:recording: m:ss` of a slide is where it starts in the recording. Once the talk names its
  recording (`.. shapiq-talk::` with `:recording: <url>` and `:event:`), each becomes a link that
  adds `t=<seconds>` to the URL. Without a recording, the slots stay hidden.

The title slide shows `assets/qr-deck.svg`, a QR code of the talk's page on Read the Docs with
`?present`, so that it opens as a deck. It is a static file. To make it again for another address:

```bash
npx qrcode@1.5.4 -t svg -e M -q 2 -d 1b1e27 -l ffffff -o assets/qr-deck.svg "<url>?present"
```

The probly slide shows `assets/probly/qr-probly.svg`, made the same way from
`"https://github.com/pwhofman/probly"`. The last slide shows `assets/qr-shapiq.svg`, of
`"https://github.com/mmschlk/shapiq"`. Remove the XML prolog (`<?xml ...?><!DOCTYPE ...>`) that
`qrcode` writes, as `qr-deck.svg` has none.
