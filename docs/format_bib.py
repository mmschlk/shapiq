r"""Format BibTeX files in the style of the documentation's bibliography (references.bib).

The pre-commit hook ``format-bib`` (``.pre-commit-config.yaml``) runs this script on the ``.bib``
files of a commit. Like ``ruff format``, it rewrites a file and fails if the file changed, so that
the formatted file can be reviewed and staged. Problems it cannot fix without guessing, such as an
unknown field or a duplicate key, fail with a message and leave the file as it is.

The format, described at the top of ``docs/source/references.bib``:

- the comment before the first section stays as it is; sections start with a line
  ``% -- Name ---...``; articles, books, chapters and theses go to the sections of their type and
  are sorted by key, all other entries stay in their section (e.g. ``% -- Conference: ICML``) and
  are sorted by year and key
- fields indented by a tab as ``name = {value}``, in the order of ``FIELDS``
- titles in double braces, names as "First Last" with plain hyphens, accents as LaTeX commands
  (``{\"u}``, ``{\'\i}``, ``{\v{S}}``) so that the file is ASCII, page ranges with ``--``, dois
  without ``https://doi.org/`` and in lowercase, dois and urls without escaped underscores

Usage::

    python docs/format_bib.py docs/source/references.bib  # format in place
    python docs/format_bib.py --check docs/source/references.bib  # only report
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

FIELDS = (
    "title",
    "author",
    "editor",
    "year",
    "journal",
    "booktitle",
    "edition",
    "chapter",
    "school",
    "institution",
    "publisher",
    "address",
    "series",
    "volume",
    "number",
    "pages",
    "archiveprefix",
    "eprint",
    "primaryclass",
    "isbn",
    "doi",
    "url",
    "note",
    "shorthand",
)

# the sections that entries of these types belong to, sorted by key (other entries by year and key)
TYPE_SECTIONS = {
    "article": "Journal articles",
    "book": "Books",
    "incollection": "Chapters in books",
    "inbook": "Chapters in books",
    "phdthesis": "Theses",
    "mastersthesis": "Theses",
}
OTHER_TYPES = {"inproceedings", "misc", "techreport"}
HEADER = re.compile(r"[%#]\s*--\s+(?P<name>.*?)\s+-+\s*")
RULE_WIDTH = 100

# accents as LaTeX commands, by the combining mark of the Unicode decomposition
ACCENTS = {
    "\u0300": "`",  # grave
    "\u0301": "'",  # acute
    "\u0302": "^",  # circumflex
    "\u0303": "~",  # tilde
    "\u0304": "=",  # macron
    "\u0306": "u",  # breve
    "\u0307": ".",  # dot above
    "\u0308": '"',  # diaeresis
    "\u030a": "r",  # ring above
    "\u030b": "H",  # double acute
    "\u030c": "v",  # caron
    "\u0327": "c",  # cedilla
    "\u0328": "k",  # ogonek
}
# letters without a decomposition, and punctuation that LaTeX writes in ASCII
LETTERS = {
    "\u0142": "{\\l}",  # l with stroke
    "\u0141": "{\\L}",
    "\u00f8": "{\\o}",  # o with stroke
    "\u00d8": "{\\O}",
    "\u00df": "{\\ss}",  # sharp s
    "\u00e6": "{\\ae}",
    "\u00c6": "{\\AE}",
    "\u0153": "{\\oe}",
    "\u0152": "{\\OE}",
    "\u00e5": "{\\aa}",  # a with ring
    "\u00c5": "{\\AA}",
    "\u0131": "{\\i}",  # dotless i
    "\u2013": "--",  # en dash
    "\u2014": "---",  # em dash
    "\u2018": "`",  # quotation marks
    "\u2019": "'",
    "\u201c": "``",
    "\u201d": "''",
    "\u00a0": " ",  # no-break space
}


class BibError(Exception):
    """A problem in a bib file that the formatter does not fix by itself."""


@dataclass
class Entry:
    """An entry of a bib file: its type, key, fields and the section it stands in."""

    kind: str
    key: str
    section: str | None
    fields: dict[str, str] = field(default_factory=dict)


def _braced(text: str, start: int) -> int:
    """Return the index after the brace group that opens at ``start``."""
    depth = 0
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return i + 1
    msg = f"unbalanced braces after line {text.count(chr(10), 0, start) + 1}"
    raise BibError(msg)


def parse(text: str) -> tuple[str, list[str], list[Entry]]:
    """Return the preamble, the names of the sections in order, and the entries of a bib file."""
    preamble, sections, entries = [], [], []
    section, i = None, 0
    while i < len(text):
        line_end = text.find("\n", i)
        line_end = len(text) if line_end == -1 else line_end
        line = text[i:line_end].strip()
        if not line:
            if section is None:
                preamble.append("")
            i = line_end + 1
        elif line[0] in "%#":
            if header := HEADER.fullmatch(line):
                section = header.group("name")
                sections.append(section)
            elif section is None:
                preamble.append(line)
            else:
                msg = f"a comment between entries is not kept: {line!r} (put notes at the top)"
                raise BibError(msg)
            i = line_end + 1
        elif line[0] == "@":
            entry, i = _parse_entry(text, i, section)
            entries.append(entry)
        else:
            msg = f"unexpected text outside of an entry: {line!r}"
            raise BibError(msg)
    return "\n".join(preamble).strip(), sections, entries


def _parse_entry(text: str, i: int, section: str | None) -> tuple[Entry, int]:
    """Return the entry that starts at ``i`` and the index after it."""
    head = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,").match(text, i)
    if not head:
        msg = f"cannot read the entry at {text[i : i + 40]!r}"
        raise BibError(msg)
    entry = Entry(head.group(1).lower(), head.group(2), section)
    i = head.end()
    while True:
        name = re.compile(r"\s*(\w+)\s*=\s*").match(text, i)
        if not name:
            close = re.compile(r"\s*,?\s*\}").match(text, i)
            if not close:
                msg = f"{entry.key}: cannot read the fields at {text[i : i + 40]!r}"
                raise BibError(msg)
            return entry, close.end()
        key, i = name.group(1).lower(), name.end()
        if text[i] == "{":
            end = _braced(text, i)
            value = text[i + 1 : end - 1]
        elif text[i] == '"':
            end = text.index('"', i + 1)
            value, end = text[i + 1 : end], end + 1
        else:
            bare = re.compile(r"[\w.-]+").match(text, i)
            if not bare:
                msg = f"{entry.key}: cannot read the value of {key!r}"
                raise BibError(msg)
            value, end = bare.group(0), bare.end()
        if key in entry.fields:
            msg = f"{entry.key}: the field {key!r} appears twice"
            raise BibError(msg)
        entry.fields[key] = value
        comma = re.compile(r"\s*,").match(text, end)
        i = comma.end() if comma else end


def latex(value: str, key: str) -> str:
    r"""Return the value in ASCII with one spelling per accent: {\"u}, {\'\i}, {\v{S}}."""
    out = []
    for char in value:
        if char.isascii():
            out.append(char)
        elif char in LETTERS:
            out.append(LETTERS[char])
        else:
            base, *marks = unicodedata.normalize("NFD", char)
            if len(marks) != 1 or marks[0] not in ACCENTS or not base.isascii():
                msg = f"{key}: no LaTeX spelling for {char!r} (U+{ord(char):04X}): write it as a command"
                raise BibError(msg)
            out.append(_accent(ACCENTS[marks[0]], base))
    value = "".join(out)
    value = re.sub(r"\{\{(\\[\"'`^~=.]\\?[a-zA-Z])\}\}", r"{\1}", value)  # {{\"a}} -> {\"a}
    value = re.sub(r"\{(\\[\"'`^~=.])\{(\\?[a-zA-Z])\}\}", r"{\1\2}", value)  # {\"{u}} -> {\"u}
    value = re.sub(r"(?<!\{)(\\[\"'`^~=.])\{(\\?[a-zA-Z])\}", r"{\1\2}", value)  # \"{u} -> {\"u}
    return re.sub(r"(?<![{\\])(\\[\"'`^~=.])([a-zA-Z])", r"{\1\2}", value)  # M\"obius -> M{\"o}bius


def _accent(command: str, base: str) -> str:
    r"""Return the LaTeX command of an accented letter, e.g. {\'e}, {\'\i} or {\v{S}}."""
    letter = {"i": "\\i", "j": "\\j"}.get(base, base)
    if command.isalpha():
        return f"{{\\{command}{{{letter}}}}}"
    return f"{{\\{command}{letter}}}"


def _split(value: str, separator: str) -> list[str]:
    """Split at a separator outside of braces."""
    parts, depth, start = [], 0, 0
    for i, char in enumerate(value):
        depth += {"{": 1, "}": -1}.get(char, 0)
        if depth == 0 and value.startswith(separator, i):
            parts.append(value[start:i])
            start = i + len(separator)
    return [*parts, value[start:]]


def _person(name: str) -> str:
    """Return a name as "First Last" with plain hyphens."""
    name = name.replace("{-}", "-").strip()
    parts = _split(name, ",")
    if len(parts) == 2:  # "Last, First"; "Last, Jr, First" stays as it is
        name = f"{parts[1].strip()} {parts[0].strip()}"
    return name


def _wrapped(value: str) -> bool:
    """Return whether the brace group that opens the value closes at its end."""
    return value.startswith("{") and _braced(value, 0) == len(value)


def _venue(value: str) -> str:
    """Return a journal or booktitle as "Name (ACRONYM)", without protecting braces."""
    value = re.sub(r",?:? \(\{?([^(){}]+)\}?\)$", r" (\1)", value)  # ", ({CVPR})" -> " (CVPR)"
    value = re.sub(r" \{\(([^(){}]+)\)\}$", r" (\1)", value)  # " {(xAI)}" -> " (xAI)"
    return re.sub(r"\{([A-Za-z]+)\}", r"\1", value)  # {IEEE} -> IEEE


def format_value(name: str, value: str, key: str) -> str:
    """Return a field's value in the format of the bibliography."""
    value = " ".join(value.split())
    if name in {"doi", "url"}:
        value = value.replace("\\_", "_")
        if name == "doi":
            value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value).lower()
        return value
    value = latex(value, key)
    if name == "title" and not _wrapped(value):
        value = "{" + value + "}"
    if name in {"author", "editor"}:
        value = " and ".join(_person(part) for part in _split(value, " and "))
    if name == "pages":
        value = re.sub(r"^(\S+?)\s*-{1,3}\s*(\S+)$", r"\1--\2", value)  # 1-8, 60:1 -- 60:21
    if name in {"journal", "booktitle"}:
        value = _venue(value)
    return value


def format_entry(entry: Entry) -> str:
    """Return the formatted entry, or raise a BibError for a problem the format cannot fix."""
    if entry.kind not in TYPE_SECTIONS and entry.kind not in OTHER_TYPES:
        msg = f"{entry.key}: the entry type @{entry.kind} is not part of the format"
        raise BibError(msg)
    if unknown := sorted(set(entry.fields) - set(FIELDS)):
        msg = f"{entry.key}: the fields {unknown} are not part of the format, which has {list(FIELDS)}"
        raise BibError(msg)
    if not {"title", "year"} <= set(entry.fields) or not {"author", "editor"} & set(entry.fields):
        msg = f"{entry.key}: an entry needs a title, a year, and an author or editor"
        raise BibError(msg)
    if not re.fullmatch(r"\d{4}", entry.fields["year"].strip("{} ")):
        msg = f"{entry.key}: the year {entry.fields['year']!r} is not a year"
        raise BibError(msg)
    lines = [f"@{entry.kind}{{{entry.key},"]
    lines += [
        f"\t{name} = {{{format_value(name, entry.fields[name], entry.key)}}},"
        for name in FIELDS
        if name in entry.fields
    ]
    lines[-1] = lines[-1].removesuffix(",")
    return "\n".join([*lines, "}"])


def format_bib(text: str) -> str:
    """Return the formatted text of a bib file."""
    preamble, written_sections, entries = parse(text)
    if duplicates := sorted({e.key for e in entries if [x.key for x in entries].count(e.key) > 1}):
        msg = f"duplicate keys {duplicates}"
        raise BibError(msg)
    type_sections = list(dict.fromkeys(TYPE_SECTIONS.values()))
    order = [s for s in type_sections if s not in written_sections] + written_sections
    sections: dict[str, list[tuple[object, str]]] = {name: [] for name in order}
    for entry in entries:
        section = TYPE_SECTIONS.get(entry.kind, entry.section)
        if section is None or (entry.kind in OTHER_TYPES and section in type_sections):
            msg = (
                f"{entry.key}: put the @{entry.kind} entry into a section of its venue, e.g. "
                "'% -- Conference: ICML ---', 'Preprints' or 'Other conferences and workshops'"
            )
            raise BibError(msg)
        formatted = format_entry(entry)  # checks the entry, e.g. that it has a year
        year = int(entry.fields["year"].strip("{} "))
        sort_key = (
            entry.key.casefold() if section in type_sections else (year, entry.key.casefold())
        )
        sections[section].append((sort_key, formatted))
    out = [preamble + "\n"] if preamble else []
    for name, items in sections.items():
        if items:
            rule = f"% -- {name} "
            out.append("\n" + rule + "-" * (RULE_WIDTH - len(rule)) + "\n\n")
            out.append("\n".join(entry for _, entry in sorted(items)) + "\n")
    return "".join(out).lstrip("\n")


def main(argv: list[str] | None = None) -> int:
    """Format the given bib files; return 1 if one changed or has a problem, else 0."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true", help="report, but do not rewrite files")
    args = parser.parse_args(argv)
    status = 0
    for path in args.files:
        text = path.read_text(encoding="utf-8")
        try:
            formatted = format_bib(text)
        except BibError as error:
            print(f"{path}: {error}", file=sys.stderr)
            status = 1
            continue
        if formatted != text:
            status = 1
            if args.check:
                print(f"{path}: would be reformatted", file=sys.stderr)
            else:
                path.write_text(formatted, encoding="utf-8")
                print(f"{path}: reformatted", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
