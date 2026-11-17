AI Involvement
==============

.. rst-class:: shapiq-examples-lead

Some pages of this documentation show a thin line left of their text. The line reports how much
AI was involved in each part of the page. The people who maintain the page report it themselves.
It is meant to make readers aware, not to certify anything.

Why a Line
----------

Language models now write much of the text, code and figures of documentation. A reader cannot
tell from the result how much of it a person wrote, thought through or checked. This matters most
where a page makes claims, explains a method or reports numbers. The line gives this information
*part by part*, right where the reader is. A reader can then decide how far to trust a part and
where to look closer.

The Levels
----------

Six levels describe a part. They answer two questions. Who wrote the part, a person or AI? And how
closely did a person check what AI wrote? At levels 0 to 2 a person wrote the part, alone or
together with AI. At levels 3 to 5 AI wrote it, and the levels differ in how closely a person
checked it.

.. shapiq-ai-levels::

Only the writing of the page counts. The models that a page explains, such as SigLIP 2 in the
FIxLIP demo, and the data it shows are its subject and do not count. A part that reproduces the
work of other people, such as a definition from a paper, has the level of that work. The card of
the part names the source.

Reading the Line
----------------

The line runs along the whole page. Its position and its color give the level. Level 0 is green
and on the far left. Level 5 is an intense orange and next to the text. The shade under the line
grows with the level, so a part written mostly by AI carries more color. Where the level changes,
the line curves into another lane. A part inside a section, such as a figure or a definition, can
have its own level. The line then swings out and back for the length of the part.

A small tag sticks to the top of the line while the page scrolls. It names the level of the part
being read. Pointing at the line shows a dot on it and tints the part on the page. A click opens a
card with the details of the part, and a click on the tag opens a card for the whole page, with the
share of the page at each level. Each card lists the following.

.. list-table::
   :header-rows: 1
   :widths: 1 4

   * - Field
     - What it says
   * - Written by
     - The person or the AI that wrote the part.
   * - Models
     - The language models that worked on the part, as the tools reported them at the time.
   * - Human role
     - What a person did with the part, e.g. set its structure, asked for changes or approved it.
   * - Sources
     - The work that the part builds on or reproduces, and where its values come from.
   * - Last checked
     - When a person last checked the level of the part.

Limitations
-----------

The levels are self-reported. Nobody verifies them, and they cannot be verified from the page. A
level is also coarse. A section at level 3 may contain a sentence that a person wrote word for
word. A part can change after its level was set, which is why each card gives the date of the last
check. Finally, the names of the models are those that the tools reported. A long piece of work may
have involved other models at times. The line is meant as an honest summary, not as a record of
every edit.

Adding the Line to a Page
-------------------------

A page reports its parts in a TOML file next to it and names it once, anywhere on the page.

.. code-block:: rst

   .. shapiq-ai-involvement:: ai-involvement.toml

The file has a table for the page as a whole, defaults for all parts, and one table per part. A part
names its element by a CSS selector, usually the label of a section.

.. code-block:: toml

   [page]
   level = 3
   summary = "Claude wrote most of this page. A person reviewed every section."

   [[parts]]
   target = "#fixlip-sec-introduction"
   level = 1
   written = "Max"
   human = "Max wrote it. Claude changed a few words where he asked."

The levels themselves are defined in ``docs/source/_data/ai-involvement.toml``, and the line is
drawn by ``docs/source/_static/js/ai-involvement.js``.
