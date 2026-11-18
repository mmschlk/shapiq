AI Involvement
==============

.. rst-class:: shapiq-examples-lead

Some pages of this documentation show a thin line left of their text. The line reports how much
AI was involved in each part of the page. The people who maintain the page report it themselves.
It is meant to make readers aware, not to certify anything.

.. shapiq-ai-figure:: page

   **The line beside a page.** AI wrote this section and a person reviewed it. A figure inside it
   was only checked, and a definition comes from a person's own text. The line swings out for both.

Why a Line
----------

Language models now write much of the text, code and figures of documentation. A reader cannot
tell from the result how much of it a person wrote, thought through or checked. This matters most
where a page makes claims, explains a method or reports numbers. The line gives this information
*part by part*, right where the reader is. A reader can then decide how far to trust a part and
where to look closer.

The Levels
----------

Each part has two inputs, what a person did and what AI did. Neither is a judgment. A person may
have written the part, reviewed every line, only checked the result, or done nothing. AI may have
written the part, drafted pieces of it, only fixed its wording, or done nothing. The two inputs
together give one of six levels, from *Handmade* to *Autopilot*.

.. shapiq-ai-figure:: levels

   **From two inputs to one place.** Each level is a pair of inputs on the grid. The line runs at
   their balance, which stays the same along each diagonal of the grid. A person or AI wrote every
   part, so the middle lane stays empty.

.. shapiq-ai-levels::

Only the writing of the page counts. The models that a page explains, such as SigLIP 2 in the
FIxLIP demo, and the data it shows are its subject and do not count. A part that reproduces the
work of other people, such as a definition from a paper, has the level of that work. The card of
the part names the source.

Reading the Line
----------------

The line runs along the whole page. Its place is the balance of the two inputs. It runs on the
far left where a person did everything and next to the text where AI did everything. Its color
follows its place, and the faint shade under it grows towards the text. Where a part changes, the
line curves into its new place. A part inside a section, such as a figure or a definition, can have
its own place. The line then swings out and back for the length of the part.

The tag at the top of the line names the level of the part being read. A click on the tag opens a
card for the whole page, with the share of the page at each level. A click on the line opens the
card of a part. A card can also hide the line, and the tag then shows it again. Each card lists the following.

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
