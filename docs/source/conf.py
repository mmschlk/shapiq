"""shapiq documentation build configuration file."""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

import functools
import importlib
import inspect
import os
import pkgutil
import re
import sys
import unicodedata
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, TypeAliasType, override

from docutils import nodes
from pybtex.backends.plaintext import Backend as PlainTextBackend
from pybtex.database import Person, parse_file
from pybtex.plugin import register_plugin
from pybtex.richtext import Text
from pybtex.style.formatting.alpha import Style as AlphaStyle
from pybtex.style.labels.alpha import LabelStyle as AlphaLabelStyle
from pybtex.style.sorting.author_year_title import SortingStyle as AuthorYearTitleSortingStyle
from pybtex.style.template import field, first_of, href, join, node, optional, sentence, words
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.errors import PycodeError
from sphinx.pycode import ModuleAnalyzer
from sphinxcontrib.bibtex.style.referencing import basic_author_year, extra_year

if TYPE_CHECKING:
    from pybtex.database import Entry
    from pybtex.style.template import Node

root = Path(__file__).resolve().parents[2]  # ../../ from this file
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "shapiq"))
sys.path.insert(0, str(root / "examples"))
sys.path.insert(0, str(root / "src"))  # get the shapiq package

# The examples download public Hugging Face models without a token; the hub's notice that a token
# raises its rate limits would otherwise show up in an example's rendered output.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")


# -- Read the Docs ---------------------------------------------------------------------------------
master_doc = "index"

# -- Project information ---------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "shapiq"
copyright = "2024, Muschalik et al."
author = "Muschalik et al."
release = version("shapiq")
version = version("shapiq")

# -- General configuration -------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.duration",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
]

# -- Timeline of Shapley-based methods (timeline.rst) ----------------------------------------------
# The ``shapiq-timeline`` directive of ``_ext/shapiq_timeline.py`` renders ``_data/timeline.toml``
# and adds ``_static/css/timeline.css`` and ``_static/js/timeline.js`` to the pages that use it.
sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))
extensions.append("shapiq_timeline")

# -- Demos (demos/) --------------------------------------------------------------------------------
# One folder per demo: its page, its assets (copied to ``_static/demos/<name>/``) and the scripts
# that computed its data. The ``shapiq-demo`` directive of ``_ext/shapiq_demos.py`` embeds a demo in
# a page and adds its stylesheet and script there; see ``demos/README.md``.
extensions.append("shapiq_demos")

# -- Sphinx-Gallery -------------------------------------------------------------------------------
# One gallery per group of examples (``examples/<group>``), each with its own page; the landing
# page ``examples.rst`` links them. Groups may have subsections (subfolders with a header).
EXAMPLE_GROUPS = (
    "basics",
    "exact_computation",
    "approximators",
    "trees",
    "graph_neural_networks",
    "nearest_neighbors",
    "kernel_methods",
    "visualization",
    "applications",
    "benchmarking",
    "technicalities",
)
# The examples of each gallery (and subsection) in learning order, by folder below ``examples``.
# Examples that are not listed follow by file name, so a new example appears without an entry.
EXAMPLE_ORDER = {
    "basics": (
        "plot_sv_calculation.py",
        "plot_shapiq_scikit_learn.py",
        "plot_custom_games.py",
        "plot_ml_games.py",
    ),
    "exact_computation": (
        "plot_exact_computer.py",
        "plot_interaction_indices.py",
        "plot_moebius_conversion.py",
        "plot_core.py",
    ),
    "approximators/shapley_values": (
        "plot_kernelshap.py",
        "plot_leverageshap.py",
        "plot_permutation_sampling_sv.py",
        "plot_svarm.py",
        "plot_stratified_sampling_sv.py",
        "plot_owen_sampling_sv.py",
        "plot_unbiased_kernelshap.py",
        "plot_kadd_shap.py",
        "plot_polyshap.py",
        "plot_oddshap.py",
        "plot_shapleig.py",
    ),
    "approximators/shapley_interactions": (
        "plot_kernelshapiq.py",
        "plot_inconsistent_kernelshapiq.py",
        "plot_regression_fsii.py",
        "plot_regression_fbii.py",
        "plot_shapiq_approximator.py",
        "plot_svarmiq.py",
        "plot_permutation_sampling_sii.py",
        "plot_permutation_sampling_stii.py",
    ),
    "approximators/surrogate_and_sparse": (
        "plot_regression_msr.py",
        "plot_proxyshap.py",
        "plot_spex.py",
        "plot_proxyspex.py",
    ),
    "trees": (
        "plot_treeshapiq_lightgbm.py",
        "plot_tree_libraries.py",
        "plot_tree_pathdependent_vs_interventional.py",
        "plot_treeshapiq_custom_tree.py",
    ),
    "graph_neural_networks": ("plot_graphshapiq.py",),
    "nearest_neighbors": (
        "plot_nn_explainers_exact.py",
        "plot_nn_data_valuation.py",
        "plot_nn_noisy_labels.py",
    ),
    "kernel_methods": ("plot_gaussian_process_exact.py", "plot_kernel_svm_scaling.py"),
    "visualization": (
        "plot_interaction_values.py",
        "plot_visualizing_interactions.py",
        "plot_si_graph.py",
        "plot_upset.py",
        "plot_beeswarm.py",
        "plot_scatter.py",
        "plot_shapley_values.py",
    ),
    "applications": (
        "plot_explaining_tabpfn.py",
        "plot_sentiment_analysis.py",
        "plot_confounding_xai.py",
    ),
    "benchmarking": ("plot_games_tour.py", "plot_benchmark_approximators.py"),
    "technicalities": (
        "plot_imputers.py",
        "plot_conditional_imputer.py",
        "plot_causal_shapley_values.py",
        "plot_budgets_and_seeds.py",
        "plot_parallel_computation.py",
    ),
}


class ExampleOrder:
    """Sort key for sphinx-gallery: the examples of ``EXAMPLE_ORDER`` first, then the others."""

    def __init__(self, src_dir: str) -> None:
        """Look up the order of the gallery folder ``src_dir`` (none for an unknown folder)."""
        folder, examples = Path(src_dir).resolve(), root / "examples"
        known = bool(src_dir) and folder.is_relative_to(examples)
        self.order = EXAMPLE_ORDER.get(folder.relative_to(examples).as_posix(), ()) if known else ()

    def __call__(self, filename: str) -> tuple[int, str]:
        """Return the position of an example (a file name or path) and its name."""
        name = Path(filename).name
        position = self.order.index(name) if name in self.order else len(self.order)
        return position, name

    def __repr__(self) -> str:
        """Show the order (sphinx-gallery prints the sort key in its messages)."""
        return f"ExampleOrder({self.order})"


sphinx_gallery_conf = {
    "examples_dirs": [str(root / "examples" / group) for group in EXAMPLE_GROUPS],
    "gallery_dirs": [f"auto_examples/{group}" for group in EXAMPLE_GROUPS],
    # subsections as paths relative to this file; the other subsections are sorted by name
    "subsection_order": [
        *(
            f"../../examples/approximators/{subsection}"
            for subsection in ("shapley_values", "shapley_interactions", "surrogate_and_sparse")
        ),
        "*",
    ],
    "within_subsection_order": ExampleOrder,
    # thumbnails of examples without a telling figure: `# sphinx_gallery_thumbnail_path = ...`
    "remove_config_comments": True,
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("shapiq", "shapiq_games", "shapiq_benchmark"),
    "reference_url": {"shapiq": None, "shapiq_games": None, "shapiq_benchmark": None},
    "filename_pattern": r"plot_.*\.py",
    "ignore_pattern": r"util_.*\.py",
    "plot_gallery": True,
    "download_all_examples": False,
    "show_signature": False,
    "run_stale_examples": True,
    "abort_on_example_error": True,
    "default_thumb_file": str(
        root / "docs" / "source" / "_static" / "logo" / "logo_shapiq_light.svg"
    ),
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "auto_examples/**.ipynb",
    # stale sphinx-gallery artifact at source root (real copy lives in auto_examples/)
    "sg_execution_times.rst",
    # the demos' notes for maintainers, assets and scripts, which are not pages
    "demos/README.md",
    "demos/*/assets",
    "demos/*/compute",
]
bibtex_bibfiles = ["references.bib"]


def _unescape_latex(text: str) -> str:
    r"""Undo the LaTeX escaping of underscores, e.g. in DOIs such as ``10.1007/...-1\_23``."""
    return text.replace("\\_", "_")


_GENERATIONS = {"Jr", "Jr.", "Sr", "Sr.", "II", "III", "IV"}
# rendering by the backend's name (render_as("text")) looks it up in the entry points of every
# installed package, about 0.1 s per name part: it took over a minute of every docs build
_PLAIN_TEXT = PlainTextBackend()


@functools.cache
def _plain_words(part: str) -> tuple[str, ...]:
    """Return the words of a LaTeX name part as plain text without accents and generations."""
    text = unicodedata.normalize("NFD", Text.from_latex(part).render(_PLAIN_TEXT))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return tuple(word for word in text.split() if word not in _GENERATIONS)


def _plain_person(person: Person) -> Person:
    r"""Return a person whose names are plain text without accents and generations.

    ``alpha`` keeps a name part whole unless it consists of letters only, so LaTeX accents and
    braces (``H{\"{u}}llermeier``, ``{Banzhaf III}``) put whole names into the labels, and they
    sort after "Z".
    """

    def plain(parts: list[str]) -> list[str]:
        return [word for part in parts for word in _plain_words(part)]

    result = Person()
    result.first_names = plain(person.first_names)
    result.middle_names = plain(person.middle_names)
    result.prelast_names = plain(person.prelast_names)
    result.last_names = plain(person.last_names)
    return result


class _LabelStyle(AlphaLabelStyle):
    """The ``alpha`` labels from the plain names: ``[PKH26]``, not ``[PKHullermeier26]``."""

    @override
    def format_lab_names(self, persons: list[Person]) -> str:
        return super().format_lab_names([_plain_person(person) for person in persons])


class _SortingStyle(AuthorYearTitleSortingStyle):
    """Sort by the plain names, so that ``Štrumbelj`` sorts with S and ``{Banzhaf III}`` with B."""

    @override
    def person_key(self, person: Person) -> str:
        return super().person_key(_plain_person(person))


class _BibStyle(AlphaStyle):
    r"""The ``alpha`` style, but titles keep the capitalization of ``references.bib``.

    ``references.bib`` is written for LaTeX styles that print titles as written, so its titles are
    not all brace-protected; ``alpha`` would lowercase them (e.g. "Shapley" -> "shapley"). Links
    from the ``doi`` and ``url`` fields are unescaped (``\_`` -> ``_``) so that they resolve.
    Labels and order come from the plain names (see ``_plain_person``).
    """

    default_label_style = "shapiq_alpha"
    default_sorting_style = "shapiq_author_year_title"

    @override
    def format_title(self, e: Entry, which_field: str, *, as_sentence: bool = True) -> Node:
        title = field(which_field)
        return sentence[title] if as_sentence else title

    @override
    def format_url(self, e: Entry) -> Node:
        url = field("url", raw=True, apply_func=_unescape_latex)
        return words["URL:", href(url)[url]]

    @override
    def format_doi(self, e: Entry) -> Node:
        doi = field("doi", raw=True, apply_func=_unescape_latex)
        return href(join["https://doi.org/", doi])[join["doi:", doi]]


register_plugin("pybtex.style.labels", "shapiq_alpha", _LabelStyle)
register_plugin("pybtex.style.sorting", "shapiq_author_year_title", _SortingStyle)
register_plugin("pybtex.style.formatting", "shapiq_alpha", _BibStyle)


def _year_letters(bibfile: Path) -> dict[str, str]:
    """Return the letter after the year of works that the text would cite alike.

    The text cites by author and year ("Kolpaczki et al., 2024"); works with the same first author,
    the same second author or "et al.", and the same year get "a", "b", ... in the order of their
    keys, so that ``Kolpaczki.2024`` to ``Kolpaczki.2024b`` read as 2024a to 2024c.
    """
    works = defaultdict(list)
    for entry in parse_file(bibfile).entries.values():
        persons = entry.persons.get("author") or entry.persons.get("editor") or []
        surnames = [" ".join(_plain_person(person).last_names) for person in persons[:2]]
        cited_as = (*surnames[:1], surnames[1] if len(persons) == 2 else len(persons) > 2)
        works[cited_as, entry.fields.get("year", "")].append(entry)
    return {
        entry.key: letter
        for group in works.values()
        if len(group) > 1
        for letter, entry in zip(
            "abcdefghijklmnopqrstuvwxyz", sorted(group, key=lambda e: e.key), strict=False
        )
    }


_YEAR_LETTERS = _year_letters(root / "docs" / "source" / "references.bib")


@node
def _year(children, data):  # noqa: ANN202
    """The year of a citation, with a letter if other works would be cited alike (2024a, 2024b)."""
    assert not children  # noqa: S101
    letter = _YEAR_LETTERS.get(data["entry"].key, "")
    return join[first_of[optional[field("year")], "n.d."], letter].format_data(data)


# the author-year templates of sphinxcontrib-bibtex look up ``year`` when they format a citation
basic_author_year.year = extra_year.year = _year
# every citation of the docs (docstrings, examples, timeline) uses :cite:, so the bibliography page
# lists every cited work; the text cites by author and year, the bibliography lists alpha labels
bibtex_default_style = "shapiq_alpha"
bibtex_reference_style = "author_year"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

intersphinx_mapping = {
    "python3": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- Options for HTML output -----------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "furo"
html_title = "shapiq documentation"
html_static_path = ["_static"]
html_css_files = [
    "css/tokens.css",  # colors, radii, shadows and motion shared by the custom pages
    "css/custom.css",
    "css/gallery.css",  # the examples landing page, the group pages and their thumbnails
]
html_js_files = [
    ("js/sidebar.js", {"defer": "defer"}),  # keeps the place in the sidebar
    ("js/logo.js", {"defer": "defer"}),  # the players of the landing page's logo are springs
]


def _inline_logo(theme: str) -> str:
    """Return the logo of a theme as inline SVG markup for ``_templates/sidebar/brand.html``.

    An ``<img>`` cannot be animated per player, and fetching the file at runtime fails for docs
    opened from disk, so the sidebar carries the SVG itself and ``_static/js/logo.js`` copies it to
    the landing page, where its players move. Both themes' logos are on every page: the ids go
    (only the gradient of the dark logo keeps a unique one), and CSS sizes the logo.
    """
    svg = (root / "docs" / "source" / "_static" / "logo" / f"logo_shapiq_{theme}.svg").read_text()
    svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
    svg = re.sub(r'\s+id="(?!hexGlow")[^"]*"', "", svg).replace("hexGlow", "shapiq-logo-glow")
    for size in ("width", "height"):
        svg = re.sub(rf'(<svg\b[^>]*?)\s+{size}="[^"]*"', r"\1", svg, count=1)
    return svg.replace(
        "<svg",
        f'<svg class="sidebar-logo shapiq-logo only-{theme}" role="img" aria-label="shapiq"',
        1,
    )


html_context = {
    "shapiq_logo_light": _inline_logo("light"),
    "shapiq_logo_dark": _inline_logo("dark"),
}
html_favicon = "_static/logo/shapiq.ico"
pygments_style = "github-light"
pygments_dark_style = "github-dark"

# The footer icons are the octicons "mark-github" and "package" (as bundled with sphinx-design).
_GITHUB_ICON = (
    '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">'
    '<path fill-rule="evenodd" d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55'
    "-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31"
    "-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2"
    " .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86"
    " 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27"
    ".38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995"
    ' 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"></path></svg>'
)
_PYPI_ICON = (
    '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">'
    '<path d="m8.878.392 5.25 3.045c.54.314.872.89.872 1.514v6.098a1.75 1.75 0 0 1-.872 1.514l-5.25'
    " 3.045a1.75 1.75 0 0 1-1.756 0l-5.25-3.045A1.75 1.75 0 0 1 1 11.049V4.951c0-.624.332-1.201.872"
    "-1.514L7.122.392a1.75 1.75 0 0 1 1.756 0ZM7.875 1.69l-4.63 2.685L8 7.133l4.755-2.758-4.63-2.685a"
    ".248.248 0 0 0-.25 0ZM2.5 5.677v5.372c0 .09.047.171.125.216l4.625 2.683V8.432Zm6.25 8.271"
    ' 4.625-2.683a.25.25 0 0 0 .125-.216V5.677L8.75 8.432Z"></path></svg>'
)
html_theme_options = {
    "sidebar_hide_name": True,  # the logo (inline, see _inline_logo) carries the name
    # "view" and "edit" buttons; _templates/components maps the gallery pages to their scripts
    # and leaves them out on the generated API pages, which have no source file in the repository
    "source_repository": "https://github.com/mmschlk/shapiq/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "footer_icons": [
        {"name": "GitHub", "url": "https://github.com/mmschlk/shapiq", "html": _GITHUB_ICON},
        {"name": "PyPI", "url": "https://pypi.org/project/shapiq/", "html": _PYPI_ICON},
    ],
    # brand colors of the probly and shapiq family (see _static/css/tokens.css), darkened or
    # lightened for contrast
    "light_css_variables": {
        "color-brand-primary": "#d9004e",
        "color-brand-content": "#285bff",
        "color-brand-visited": "#285bff",
        "color-sidebar-link-text--top-level": "var(--color-foreground-primary)",
        "color-highlighted-background": "#ffe9ef",
        "color-inline-code-background": "#f3f4f9",
        "color-code-background": "var(--shapiq-surface-raised)",
    },
    "dark_css_variables": {
        "color-brand-primary": "#ff426d",
        "color-brand-content": "#608fff",
        "color-brand-visited": "#608fff",
        "color-sidebar-link-text--top-level": "var(--color-foreground-primary)",
        "color-highlighted-background": "#3d1724",
        "color-inline-code-background": "#1d2027",
        "color-code-background": "var(--shapiq-surface-raised)",
    },
}

html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ],
}

# -- Autosectionlabel -----------------------------------------------------------------------------
# Prefix labels with the document path to avoid collisions when multiple documents
# share identical section headings (e.g. "General Use", "Feature Names").
autosectionlabel_prefix_document = True

# -- Napoleon (Google-style docstrings) ------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True

# -- Autodoc ---------------------------------------------------------------------------------------
autosummary_generate = True
autosummary_ignore_module_all = False
autodoc_default_options = {
    "show-inheritance": True,
    "members": True,
    "private-members": False,
    "member-order": "groupwise",
    "special-members": "__call__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autoclass_content = "both"
autodoc_inherit_docstrings = True
autodoc_member_order = "groupwise"
# sphinx-autodoc-typehints adds the types to the documented parameters; Sphinx's own type hints
# ("both" or "description") would add a second, types-only parameter list to every class page
autodoc_typehints = "none"

# -- Re-exported constants -------------------------------------------------------------------------
# A package may define a constant in a private module and re-export it, e.g.
# shapiq_benchmark.datasets.TABARENA_DATASETS. Autodoc reads the docstring of a constant from the
# source of the module it documents, finds none there, and shows the docstring of its type (dict,
# int) instead. Take the docstring from the module that defines it.


def _defining_module_docstring(package_name: str, attribute: str, obj: object) -> list[str] | None:
    """Return the docstring of ``attribute`` in the submodule of a package that defines ``obj``."""
    package = sys.modules.get(package_name)
    if package is None or not hasattr(package, "__path__"):
        return None
    for submodule in pkgutil.iter_modules(package.__path__, f"{package_name}."):
        module = importlib.import_module(submodule.name)
        if getattr(module, attribute, None) is not obj:
            continue
        docs = ModuleAnalyzer.for_module(submodule.name).find_attr_docs().get(("", attribute))
        if docs:
            return list(docs)
    return None


def _use_the_defining_module_docstring(app, what, name, obj, options, lines) -> None:  # noqa: ARG001
    if what != "data":
        return
    package_name, _, attribute = name.rpartition(".")
    try:
        if ("", attribute) in ModuleAnalyzer.for_module(package_name).find_attr_docs():
            return  # the documenting module has its docstring
    except PycodeError:  # no source to read
        return
    lines[:] = _defining_module_docstring(package_name, attribute, obj) or []


# -- Type aliases ----------------------------------------------------------------------------------
# The games and the benchmark declare their choices as type aliases (``type Task = Literal[...]``),
# which sphinx-autodoc-typehints shows as "TypeAliasType". Show the alias, linked to its page (an
# alias of a private module is documented under the package that re-exports it).


def _format_type_alias(annotation: object, config: object) -> str | None:  # noqa: ARG001
    if not isinstance(annotation, TypeAliasType):
        return None  # the default formatting
    package = annotation.__module__.split(".")[0]
    if package == "numpy":  # e.g. ArrayLike, documented by numpy under numpy.typing
        return f":py:obj:`~numpy.typing.{annotation.__name__}`"
    if package not in _PACKAGES:
        return None
    module = ".".join(part for part in annotation.__module__.split(".") if not part.startswith("_"))
    return f":py:type:`~{module}.{annotation.__name__}`"


typehints_formatter = _format_type_alias


_PACKAGES = ("shapiq", "shapiq_games", "shapiq_benchmark")


_MISSING = object()


@functools.cache
def _import_path(path: str) -> object:
    """Return the object at a dotted path such as ``shapiq.KernelSHAP``, or ``_MISSING``."""
    parts = path.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        for attribute in parts[split:]:
            obj = getattr(obj, attribute, _MISSING)
            if obj is _MISSING:
                return _MISSING
        return getattr(obj, "__func__", obj)  # a classmethod is a new bound method on every access
    return _MISSING


def _identifies(obj: object) -> bool:
    """Whether an object stands for one documented object: a module, class, function, or property.

    A value such as ``None`` or ``0`` is shared by unrelated attributes, so it identifies nothing.
    """
    return inspect.ismodule(obj) or callable(obj) or isinstance(obj, property)


def _link_reexported_names(app, env, node, contnode):  # noqa: ANN202
    """Link a re-exported name (``shapiq.KernelSHAP``) to the page that documents its object.

    The API reference documents each object once, under its subpackage (e.g.
    ``shapiq.approximator.KernelSHAP``); references to the shorter public alias, or to a bare name
    outside its module (``MoebiusConverter``), would otherwise render as plain text.
    """
    target = node.get("reftarget", "")
    if node.get("refdomain") != "py":
        return None
    domain = env.domains.python_domain
    documented = [d for d in domain.objects if d.split(".")[0] in _PACKAGES]
    package = target.split(".")[0]
    if package in _PACKAGES:  # a public alias: the same name for the same object
        name = target.rpartition(".")[2]
        candidates = [d for d in documented if d.rpartition(".")[2] == name]
        obj = _import_path(target)
        if _identifies(obj):
            candidates = [c for c in candidates if _import_path(c) is obj]
        else:  # a value (e.g. a constant) links only to its one namesake in the same package
            candidates = [c for c in candidates if c.split(".")[0] == package]
            candidates = candidates if obj is not _MISSING and len(candidates) == 1 else []
    else:  # a bare or partial name such as MoebiusConverter or ExactComputer.moebius_transform
        candidates = [d for d in documented if d.endswith(f".{target}")]
        if len(candidates) > 1:  # several documented names: all of them must be this one object
            obj = _import_path(candidates[0])
            if not _identifies(obj) or any(_import_path(c) is not obj for c in candidates):
                return None
    if not candidates:
        return None
    return domain.resolve_xref(
        env, node["refdoc"], app.builder, node["reftype"], candidates[0], node, contnode
    )


def _short_navigation_titles(app, doctree) -> None:  # noqa: ARG001
    """Name the API pages by their last component in the sidebar and the previous/next links.

    The sidebar already shows where a page sits (``shapiq`` > ``approximator`` > ``KernelSHAP``), so
    full dotted names only repeat it and overflow the sidebar. The page itself keeps its full name
    as its heading and browser title.
    """
    docname = app.env.docname
    if not docname.startswith("generated/") or docname not in app.env.titles:
        return
    short = app.env.titles[docname].astext().rpartition(".")[2]
    app.env.titles[docname] = nodes.title(short, short)  # previous/next links
    toc = app.env.tocs.get(docname)  # the sidebar entry: the first reference of the page's toc
    title_reference = next(toc.findall(nodes.reference), None) if toc is not None else None
    if title_reference is not None:
        title_reference.children = [nodes.Text(short)]


# -- Redirects from moved pages ---------------------------------------------------------------------
# The examples moved into one gallery per group. Their old addresses (and those of the old groups)
# keep working: after the build, every old page that no new page replaces gets a small page that
# redirects to the new one. Add an entry here when a page moves.

_REDIRECTS = {
    "auto_examples/index": "examples",
    "auto_examples/causal/index": "auto_examples/applications/index",
    "auto_examples/game_theory/index": "auto_examples/exact_computation/index",
    "auto_examples/language/index": "auto_examples/applications/index",
    "auto_examples/nearest_neighbor/index": "auto_examples/nearest_neighbors/index",
    "auto_examples/tabular/index": "examples",
    "auto_examples/vision/index": "auto_examples/applications/index",
    "auto_examples/causal/plot_confounding_xai": "auto_examples/applications/plot_confounding_xai",
    "auto_examples/tabular/plot_explaining_tabpfn": "auto_examples/applications/plot_explaining_tabpfn",
    "auto_examples/language/plot_sentiment_analysis": (
        "auto_examples/applications/plot_sentiment_analysis"
    ),
    # the vision transformer example was removed (it is to be reworked)
    "auto_examples/vision/plot_vision_transformer": "auto_examples/applications/index",
    "auto_examples/applications/plot_vision_transformer": "auto_examples/applications/index",
    "auto_examples/tabular/plot_shapiq_scikit_learn": "auto_examples/basics/plot_shapiq_scikit_learn",
    "auto_examples/game_theory/plot_core": "auto_examples/exact_computation/plot_core",
    "auto_examples/approximators/plot_exact_computer": (
        "auto_examples/exact_computation/plot_exact_computer"
    ),
    "auto_examples/nearest_neighbor/plot_nn_data_valuation": (
        "auto_examples/nearest_neighbors/plot_nn_data_valuation"
    ),
    "auto_examples/tabular/plot_conditional_imputer": (
        "auto_examples/technicalities/plot_conditional_imputer"
    ),
    "auto_examples/basics/plot_parallel_computation": (
        "auto_examples/technicalities/plot_parallel_computation"
    ),
    **{
        f"auto_examples/approximators/{name}": f"auto_examples/approximators/{group}/{name}"
        for group, names in {
            "shapley_values": [
                "plot_kadd_shap",
                "plot_kernelshap",
                "plot_owen_sampling_sv",
                "plot_permutation_sampling_sv",
                "plot_stratified_sampling_sv",
                "plot_svarm",
                "plot_unbiased_kernelshap",
            ],
            "shapley_interactions": [
                "plot_inconsistent_kernelshapiq",
                "plot_kernelshapiq",
                "plot_permutation_sampling_sii",
                "plot_permutation_sampling_stii",
                "plot_regression_fbii",
                "plot_regression_fsii",
                "plot_shapiq_approximator",
                "plot_svarmiq",
            ],
            "surrogate_and_sparse": [
                "plot_proxyshap",
                "plot_proxyspex",
                "plot_regression_msr",
                "plot_spex",
            ],
        }.items()
        for name in names
    },
}

_REDIRECT_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Moved</title>
  <meta http-equiv="refresh" content="0; url={target}">
  <link rel="canonical" href="{target}">
</head>
<body><p>This page moved to <a href="{target}">{target}</a>.</p></body>
</html>
"""


def _write_redirects(app, exception) -> None:
    """Write a redirect page for every moved page of ``_REDIRECTS`` that no new page replaces."""
    if exception is not None or app.builder.format != "html":
        return
    outdir = Path(app.outdir)
    for old, new in _REDIRECTS.items():
        page = outdir / f"{old}.html"
        if page.exists():
            continue
        page.parent.mkdir(parents=True, exist_ok=True)
        target = Path(os.path.relpath(outdir / f"{new}.html", page.parent)).as_posix()
        page.write_text(_REDIRECT_PAGE.format(target=target), encoding="utf-8")


def setup(app) -> None:
    """Connect the docstring handler (before napoleon, which formats the docstrings)."""
    app.connect("autodoc-process-docstring", _use_the_defining_module_docstring, priority=400)
    app.connect("missing-reference", _link_reexported_names)
    app.connect("build-finished", _write_redirects)
    app.connect("doctree-read", _short_navigation_titles, priority=900)  # after the title collector


# -- Suppress warnings -----------------------------------------------------------------------------
suppress_warnings = [
    "misc.highlighting_failure",  # should be fixed in the future
    "config.cache",  # typehints_formatter is a function, which Sphinx cannot pickle with the config
]

# -- Images ----------------------------------------------------------------------------------------
StandaloneHTMLBuilder.supported_image_types = [
    "image/svg+xml",
    "image/gif",
    "image/png",
    "image/jpeg",
]
# -- Copy Paste Button -----------------------------------------------------------------------------
# Ignore >>> when copying code
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
