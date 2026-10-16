/*
 * Timeline of Shapley-based methods (timeline.rst).
 *
 * _ext/shapiq_timeline.py renders every entry as a card in a list grouped by year: the page
 * without JavaScript and the view on phones. This script adds filters (lanes, "in shapiq only",
 * search), collapsible cards, a swimlane view for wide screens that shows the open card in a
 * drawer below its lane, a table grouped by lane, deep links (timeline.html#kernelshap) and a
 * scroll-reveal animation.
 */
(() => {
  "use strict";

  const root = document.querySelector(".shapiq-timeline");
  if (!root) return;

  const lanes = JSON.parse(root.dataset.lanes);
  const columns = JSON.parse(root.dataset.columns);
  const list = root.querySelector(".tl-list");
  const cards = [...list.querySelectorAll(".tl-card")];
  const cardById = new Map(cards.map((card) => [card.id, card]));
  const wideScreen = window.matchMedia("(min-width: 50em)");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const VIEW_KEY = "shapiq-timeline-view";
  const VIEWS = { lanes: "Lanes", list: "List", table: "Table" };

  const ICONS = {
    search:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
    lanes:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M3 6h7M14 6h7M3 12h4M11 12h10M3 18h11M18 18h3"/></svg>',
    list: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M8 6h13M8 12h13M8 18h13"/><circle cx="4" cy="6" r="1"/><circle cx="4" cy="12" r="1"/><circle cx="4" cy="18" r="1"/></svg>',
    table:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M3 15h18M9 10v10"/></svg>',
    sort: '<svg class="tl-sort-icon" viewBox="0 0 12 18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path class="tl-sort-up" d="m3 7 3-3 3 3"/><path class="tl-sort-down" d="m3 11 3 3 3-3"/></svg>',
    left: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m15 6-6 6 6 6"/></svg>',
    right:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9 6 6 6-6 6"/></svg>',
    close:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>',
  };

  const state = {
    terms: [],
    lanes: new Set(), // the selected lanes; none selected shows all
    implementedOnly: false,
    view: loadView(),
    open: null, // the id of the card in a drawer of the swimlane view
    sort: { key: "year", direction: 1 }, // the order of the table within each lane
  };

  // -- helpers ----------------------------------------------------------------------------------

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function button(className, label) {
    const node = el("button", className, label);
    node.type = "button";
    return node;
  }

  // lowercase without accents, so that "mobius" finds "Möbius" and "strumbelj" "Štrumbelj"
  function normalize(text) {
    return text.normalize("NFD").replace(/\p{M}/gu, "").toLowerCase();
  }

  function loadView() {
    try {
      const view = window.localStorage.getItem(VIEW_KEY);
      return view in VIEWS ? view : "lanes";
    } catch {
      return "lanes";
    }
  }

  function saveView(view) {
    try {
      window.localStorage.setItem(VIEW_KEY, view);
    } catch {
      // the view is a convenience; without storage it is not remembered
    }
  }

  // "instant", not "auto": furo makes the page scroll smoothly by default
  function scrollBehavior() {
    return reducedMotion.matches ? "instant" : "smooth";
  }

  function currentView() {
    return wideScreen.matches ? state.view : "list";
  }

  // the id in the address (timeline.html#kernelshap), or "" if there is none or it is malformed
  function hashId() {
    try {
      return decodeURIComponent(window.location.hash.slice(1));
    } catch {
      return "";
    }
  }

  function setHash(id) {
    const url = id ? `#${id}` : window.location.pathname + window.location.search;
    window.history.replaceState(null, "", url);
  }

  // -- cards of the list view: collapsible, searchable ----------------------------------------

  const searchText = new Map();
  for (const card of cards) {
    const lane = lanes.find((candidate) => candidate.id === card.dataset.lane);
    searchText.set(card, normalize(`${card.textContent} ${lane.name}`));
    const name = card.querySelector(".tl-card-name");
    const details = card.querySelector(".tl-card-details");
    const toggle = button("tl-card-toggle");
    toggle.setAttribute("aria-expanded", "false");
    toggle.setAttribute("aria-controls", details.id);
    name.replaceWith(toggle);
    toggle.append(name);
    details.inert = true;
    toggle.addEventListener("click", () => setCardOpen(card, !card.classList.contains("is-open")));
    card.querySelector(".tl-card-header").addEventListener("click", (event) => {
      if (!event.target.closest("a, button")) toggle.click();
    });
  }

  function setCardOpen(card, open) {
    card.classList.toggle("is-open", open);
    card.querySelector(".tl-card-toggle").setAttribute("aria-expanded", String(open));
    card.querySelector(".tl-card-details").inert = !open;
    if (open) setHash(card.id);
    else if (window.location.hash === `#${card.id}`) setHash(null);
  }

  // -- toolbar ----------------------------------------------------------------------------------

  const toolbar = el("div", "tl-toolbar");
  toolbar.setAttribute("role", "search");
  toolbar.setAttribute("aria-label", "Filter the timeline");

  const search = el("label", "tl-search");
  search.innerHTML = ICONS.search;
  const searchInput = el("input");
  searchInput.type = "search";
  searchInput.placeholder = "Search methods, authors, classes…";
  searchInput.setAttribute("aria-label", "Search the timeline");
  searchInput.autocomplete = "off";
  search.append(searchInput);

  const implementedSwitch = button("tl-switch");
  implementedSwitch.setAttribute("role", "switch");
  implementedSwitch.setAttribute("aria-checked", "false");
  const track = el("span", "tl-switch-track");
  track.setAttribute("aria-hidden", "true");
  implementedSwitch.append(track, "In shapiq only");

  const views = el("div", "tl-views");
  views.setAttribute("role", "group");
  views.setAttribute("aria-label", "View");
  const viewButtons = Object.entries(VIEWS).map(([view, label]) => {
    const viewButton = button("");
    viewButton.dataset.view = view;
    viewButton.innerHTML = ICONS[view];
    viewButton.append(label);
    views.append(viewButton);
    return viewButton;
  });

  const filters = el("div", "tl-filters");
  filters.setAttribute("role", "group");
  filters.setAttribute("aria-label", "Show lanes");
  const filterButtons = lanes.map((lane) => {
    const filter = button("tl-filter");
    filter.dataset.lane = lane.id;
    filter.dataset.slot = lane.slot;
    filter.title = lane.description;
    filter.setAttribute("aria-pressed", "false");
    const count = cards.filter((card) => card.dataset.lane === lane.id).length;
    filter.append(lane.name, el("span", "tl-filter-count", String(count)));
    filters.append(filter);
    return filter;
  });
  const reset = button("tl-reset", "Reset filters");
  reset.hidden = true;
  filters.append(reset);

  const status = el("p", "tl-status");
  status.setAttribute("role", "status");
  const key = el("ul", "tl-key");
  key.setAttribute("aria-label", "Key");
  key.innerHTML =
    '<li><span class="tl-key-swatch" aria-hidden="true"></span>In shapiq</li>' +
    '<li><span class="tl-key-swatch is-dashed" aria-hidden="true"></span>Not in shapiq</li>';

  const rows = [[search, implementedSwitch, views], [filters], [status, key]].map(
    (items, index) => {
      const row = el("div", index === 2 ? "tl-toolbar-row tl-status-row" : "tl-toolbar-row");
      row.append(...items);
      return row;
    },
  );
  toolbar.append(...rows);

  const empty = el("p", "tl-empty");
  empty.hidden = true;
  const emptyReset = button("tl-reset", "Reset filters");
  empty.append("No entries match the filters. ", emptyReset);

  // -- swimlane view ----------------------------------------------------------------------------

  const lanesView = el("div", "tl-lanes");
  const lanesBar = el("div", "tl-lanes-bar");
  const hint = el(
    "span",
    "tl-scroll-hint",
    "Click a card for details. Scroll sideways for more years.",
  );
  const previous = button("tl-scroll-button");
  previous.innerHTML = ICONS.left;
  previous.setAttribute("aria-label", "Scroll to earlier years");
  const next = button("tl-scroll-button");
  next.innerHTML = ICONS.right;
  next.setAttribute("aria-label", "Scroll to later years");
  lanesBar.append(hint, previous, next);

  const scrollWrap = el("div", "tl-scroll-wrap");
  const scroller = el("div", "tl-scroll");
  const grid = el("div", "tl-grid");
  grid.style.setProperty("--tl-columns", columns.length);
  const isEra = (column) => column.includes("–");

  function axis(className) {
    const row = el("div", className);
    row.setAttribute("aria-hidden", "true");
    row.append(el("div", "tl-axis-corner", "Lane / year"));
    columns.forEach((column, index) =>
      row.append(place(el("div", isEra(column) ? "tl-axis-era" : "", column), index)),
    );
    row.append(fade());
    return row;
  }

  // the cells are placed explicitly, so that the fade can overlap the last one
  function place(node, index) {
    node.style.gridColumn = String(index + 2);
    return node;
  }

  function fade() {
    const node = el("div", "tl-fade");
    node.setAttribute("aria-hidden", "true");
    return node;
  }

  const chipById = new Map();
  const laneRows = new Map();
  grid.append(axis("tl-axis"));
  for (const lane of lanes) {
    const row = el("section", "tl-lane");
    row.dataset.lane = lane.id;
    row.dataset.slot = lane.slot;
    row.setAttribute("aria-label", lane.name);
    const label = el("div", "tl-lane-label");
    label.title = lane.description;
    const count = el("span", "tl-lane-count");
    label.append(el("span", "tl-lane-name", lane.name), count);
    row.append(label);

    const drawer = el("div", "tl-drawer");
    drawer.id = `tl-drawer-${lane.id}`;
    drawer.setAttribute("role", "region");
    drawer.setAttribute("aria-label", `${lane.name}: details`);
    drawer.inert = true;
    const panel = el("div", "tl-drawer-panel");
    panel.tabIndex = -1; // focused when a card is opened from the keyboard
    const inner = el("div", "tl-drawer-inner");
    inner.append(panel);
    drawer.append(inner);

    const laneCards = cards.filter((card) => card.dataset.lane === lane.id);
    for (const [columnIndex, column] of columns.entries()) {
      const cell = place(el("div", "tl-cell"), columnIndex);
      laneCards
        .filter((card) => card.dataset.column === column)
        .forEach((card, index) => {
          const chip = button("tl-chip");
          chip.dataset.card = card.id;
          chip.dataset.implemented = card.dataset.implemented;
          chip.setAttribute("aria-expanded", "false");
          chip.setAttribute("aria-controls", drawer.id);
          chip.title = card.querySelector(".tl-card-summary").textContent.trim();
          chip.style.setProperty("--tl-index", String(index));
          const name = card.querySelector(".tl-card-name").textContent;
          const extra = card.dataset.implemented === "true" ? ", in shapiq" : "";
          chip.append(
            el("span", "tl-chip-year", card.dataset.year),
            el("span", "tl-chip-name", name),
            el("span", "tl-sr-only", extra),
          );
          chip.addEventListener("click", (event) => {
            if (state.open === card.id) {
              closeDrawer();
              return;
            }
            showInDrawer(card.id);
            // from the keyboard (no pointer: detail 0), go on reading in the details
            if (event.detail === 0) laneRows.get(lane.id).panel.focus({ preventScroll: true });
          });
          chipById.set(card.id, chip);
          cell.append(chip);
        });
      row.append(cell);
    }
    row.append(fade(), drawer);
    grid.append(row);
    laneRows.set(lane.id, { row, count, drawer, panel });
  }
  grid.append(axis("tl-axis tl-axis-bottom"));
  scroller.append(grid);
  scrollWrap.append(scroller);
  lanesView.append(lanesBar, scrollWrap);

  function cloneForDrawer(card) {
    const clone = card.cloneNode(true);
    clone.hidden = false;
    clone.classList.add("is-open");
    for (const node of [clone, ...clone.querySelectorAll("[id]")]) node.removeAttribute("id");
    for (const node of clone.querySelectorAll("[inert]")) node.inert = false;
    const toggle = clone.querySelector(".tl-card-toggle");
    toggle.replaceWith(...toggle.childNodes);
    const close = button("tl-drawer-close");
    close.innerHTML = ICONS.close;
    close.setAttribute("aria-label", "Close the details");
    close.addEventListener("click", () => closeDrawer({ focus: true }));
    clone.append(close);
    return clone;
  }

  function setDrawerOpen(laneRow, open, instant) {
    if (instant) laneRow.drawer.classList.add("is-instant");
    laneRow.drawer.classList.toggle("is-open", open);
    laneRow.drawer.inert = !open;
    laneRow.row.classList.toggle("is-active", open);
    if (instant) {
      laneRow.drawer.getBoundingClientRect(); // apply the change without a transition
      laneRow.drawer.classList.remove("is-instant");
    }
  }

  function showInDrawer(id, { instant = false } = {}) {
    const card = cardById.get(id);
    const laneRow = laneRows.get(card.dataset.lane);
    const chip = chipById.get(id);
    if (state.open) {
      const openRow = laneRows.get(cardById.get(state.open).dataset.lane);
      chipById.get(state.open).setAttribute("aria-expanded", "false");
      if (openRow !== laneRow) {
        // close the other drawer at once and keep the clicked card where it is on the screen
        const before = chip.getBoundingClientRect().top;
        setDrawerOpen(openRow, false, true);
        window.scrollBy({ top: chip.getBoundingClientRect().top - before, behavior: "instant" });
      }
    }
    laneRow.panel.replaceChildren(cloneForDrawer(card));
    chip.setAttribute("aria-expanded", "true");
    laneRow.row.classList.add("is-revealed");
    setDrawerOpen(laneRow, true, instant);
    state.open = id;
    setHash(id);
  }

  function closeDrawer({ focus = false } = {}) {
    if (!state.open) return;
    const id = state.open;
    const chip = chipById.get(id);
    chip.setAttribute("aria-expanded", "false");
    setDrawerOpen(laneRows.get(cardById.get(id).dataset.lane), false, false);
    state.open = null;
    if (window.location.hash === `#${id}`) setHash(null);
    if (focus) chip.focus({ preventScroll: true });
  }

  function updateScroll() {
    const max = scroller.scrollWidth - scroller.clientWidth;
    scrollWrap.classList.toggle("can-scroll-left", scroller.scrollLeft > 2);
    scrollWrap.classList.toggle("can-scroll-right", scroller.scrollLeft < max - 2);
    previous.disabled = scroller.scrollLeft <= 2;
    next.disabled = scroller.scrollLeft >= max - 2;
    previous.hidden = next.hidden = max <= 2;
    hint.textContent =
      max > 2
        ? "Click a card for details. Scroll sideways for more years."
        : "Click a card for details.";
    grid.style.setProperty("--tl-view-width", `${scroller.clientWidth}px`);
  }

  function scrollByPage(direction) {
    const labelWidth = grid.querySelector(".tl-lane-label").offsetWidth;
    const page = (scroller.clientWidth - labelWidth) * 0.8;
    scroller.scrollBy({ left: direction * page, behavior: scrollBehavior() });
  }

  scroller.addEventListener("scroll", updateScroll, { passive: true });
  previous.addEventListener("click", () => scrollByPage(-1));
  next.addEventListener("click", () => scrollByPage(1));
  if ("ResizeObserver" in window) new ResizeObserver(updateScroll).observe(scroller);

  // -- table view -------------------------------------------------------------------------------

  const COLUMN_FIELDS = ["Reference", "References", "In shapiq"]; // the fields with a column

  // the label of a field of a card's details ("Reference", "In shapiq", ...)
  function fieldLabel(term) {
    return term.textContent.replace(/:\s*$/, "").trim();
  }

  // the body of a field of a card's details, or null
  function field(card, labels) {
    const terms = [...card.querySelectorAll(".tl-card-fields > dt")];
    return terms.find((term) => labels.includes(fieldLabel(term)))?.nextElementSibling ?? null;
  }

  // a copy of the links of a field (citations, API objects), without the ids of the original
  function copyOf(body) {
    const copy = document.createDocumentFragment();
    for (const node of (body.querySelector("p") ?? body).childNodes) copy.append(node.cloneNode(true));
    for (const node of copy.querySelectorAll("[id]")) node.removeAttribute("id");
    return copy;
  }

  // let long names such as ExactComputer.probabilistic_value wrap after a dot or an underscore
  function breakAfterDots(fragment) {
    for (const name of fragment.querySelectorAll("code .pre")) {
      const parts = name.textContent.split(/(?<=[._])/);
      name.replaceChildren(...parts.flatMap((part, i) => (i ? [el("wbr"), part] : [part])));
    }
    return fragment;
  }

  // the details of a card below its row: the description and the fields without a column
  function detailsForRow(card) {
    const details = card.querySelector(".tl-card-details-inner").cloneNode(true);
    for (const node of [details, ...details.querySelectorAll("[id]")]) node.removeAttribute("id");
    for (const term of [...details.querySelectorAll(".tl-card-fields > dt")]) {
      if (COLUMN_FIELDS.includes(fieldLabel(term))) {
        term.nextElementSibling.remove();
        term.remove();
      }
    }
    const fields = details.querySelector(".tl-card-fields");
    if (fields && !fields.children.length) fields.remove();
    return details;
  }

  const tableView = el("div", "tl-table-view");
  const tableBar = el("div", "tl-lanes-bar");
  tableBar.append(
    el("span", "tl-scroll-hint", "Click a method for details and a column heading to sort each lane."),
  );
  const table = el("table", "tl-table");
  table.append(el("caption", "tl-sr-only", "The entries of the timeline"));
  const head = el("thead");
  const headRow = el("tr");
  const SORTS = { year: "Year", name: "Method", cite: "Reference", shapiq: "In shapiq" };
  const sortHeaders = Object.entries(SORTS).map(([key, label]) => {
    const header = el("th", `tl-col-${key}`);
    header.scope = "col";
    const sortButton = button("tl-sort");
    sortButton.append(label);
    sortButton.insertAdjacentHTML("beforeend", ICONS.sort);
    sortButton.addEventListener("click", () => {
      const direction = state.sort.key === key ? -state.sort.direction : 1;
      state.sort = { key, direction };
      sortTable({ animate: true });
    });
    header.dataset.sort = key;
    header.append(sortButton);
    headRow.append(header);
    return header;
  });
  head.append(headRow);
  table.append(head);

  // one group per lane, as in a literature overview: a heading row, then the lane's entries
  const groups = new Map(
    lanes.map((lane) => {
      const group = el("tbody", "tl-group");
      group.dataset.slot = lane.slot;
      const heading = el("th");
      heading.scope = "rowgroup";
      heading.colSpan = Object.keys(SORTS).length;
      const count = el("span", "tl-group-count");
      heading.append(
        el("span", "tl-group-name", lane.name),
        count,
        el("span", "tl-group-description", lane.description),
      );
      const row = el("tr");
      row.append(heading);
      group.append(row);
      return [lane.id, { group, count }];
    }),
  );

  const rowById = new Map();
  for (const [index, card] of cards.entries()) {
    const lane = lanes.find((candidate) => candidate.id === card.dataset.lane);
    const name = card.querySelector(".tl-card-name").textContent;
    const entry = el("tbody", "tl-entry");
    entry.dataset.slot = card.dataset.slot;
    const row = el("tr", "tl-row");

    const nameCell = el("th", "tl-col-name");
    nameCell.scope = "row";
    const toggle = button("tl-row-toggle");
    toggle.setAttribute("aria-expanded", "false");
    toggle.setAttribute("aria-controls", `${card.id}-row`);
    toggle.append(el("span", "tl-row-name", name));
    toggle.insertAdjacentHTML("beforeend", ICONS.right);
    const summary = card.querySelector(".tl-card-summary").textContent.trim();
    nameCell.append(toggle, el("span", "tl-row-summary", summary));

    const citeCell = el("td", "tl-col-cite");
    const references = field(card, ["Reference", "References"]);
    if (references) citeCell.append(copyOf(references));
    const shapiqCell = el("td", "tl-col-shapiq");
    const api = field(card, ["In shapiq"]);
    if (api) shapiqCell.append(breakAfterDots(copyOf(api)));
    else shapiqCell.append(el("span", "tl-row-none", "–"), el("span", "tl-sr-only", "not in shapiq"));
    row.append(el("td", "tl-col-year", card.dataset.year), nameCell, citeCell, shapiqCell);

    const detailsRow = el("tr", "tl-row-details");
    const detailsCell = el("td");
    detailsCell.colSpan = Object.keys(SORTS).length;
    const drawer = el("div", "tl-row-drawer");
    drawer.id = `${card.id}-row`;
    drawer.inert = true;
    const inner = el("div", "tl-row-drawer-inner");
    drawer.append(inner);
    detailsCell.append(drawer);
    detailsRow.append(detailsCell);
    entry.append(row, detailsRow);
    table.append(entry);

    toggle.addEventListener("click", () => setRowOpen(card.id, !entry.classList.contains("is-open")));
    row.addEventListener("click", (event) => {
      if (!event.target.closest("a, button") && !window.getSelection().toString()) toggle.click();
    });
    rowById.set(card.id, {
      entry,
      toggle,
      drawer,
      inner,
      lane: lane.id,
      keys: {
        year: Number(card.dataset.year),
        name: normalize(name),
        cite: references ? normalize(references.textContent.replace(/^\W+/, "")) : null,
        shapiq: api ? 0 : 1,
      },
      index, // the order of the cards: year, lane, name
    });
  }
  tableView.append(tableBar, table);

  function setRowOpen(id, open) {
    const row = rowById.get(id);
    if (open && !row.inner.firstChild) row.inner.append(detailsForRow(cardById.get(id)));
    row.entry.classList.toggle("is-open", open);
    row.toggle.setAttribute("aria-expanded", String(open));
    row.drawer.inert = !open;
    if (open) setHash(id);
    else if (window.location.hash === `#${id}`) setHash(null);
  }

  function compareRows(a, b) {
    const { key, direction } = state.sort;
    const [x, y] = [a.keys[key], b.keys[key]];
    if (x !== y) {
      if (x === null || y === null) return x === null ? 1 : -1; // no reference: always last
      return (x < y ? -1 : 1) * direction;
    }
    return a.index - b.index;
  }

  // sort the rows of every lane; with animate, every row glides from its old place to its new one
  function sortTable({ animate = false } = {}) {
    const rows = [...rowById.values()];
    const glide = animate && !reducedMotion.matches && "animate" in Element.prototype;
    const before = new Map(glide ? rows.map((row) => [row, row.entry.getBoundingClientRect().top]) : []);
    rows.sort(compareRows);
    for (const [lane, { group }] of groups) {
      table.append(group, ...rows.filter((row) => row.lane === lane).map((row) => row.entry));
    }
    for (const header of sortHeaders) {
      const sorted = header.dataset.sort === state.sort.key;
      const order = state.sort.direction > 0 ? "ascending" : "descending";
      if (sorted) header.setAttribute("aria-sort", order);
      else header.removeAttribute("aria-sort");
    }
    if (!glide) return;
    table.classList.add("is-sorted"); // all rows are shown: no scroll reveal after a sort
    const height = window.innerHeight;
    const onScreen = (top) => top > -height * 0.25 && top < height;
    for (const row of rows) {
      if (row.entry.hidden) continue;
      const top = row.entry.getBoundingClientRect().top;
      const shift = before.get(row) - top;
      if (!shift || !onScreen(top)) continue;
      // a row from far away fades in over a short way instead of flying across the screen
      const keyframes = onScreen(before.get(row))
        ? [{ transform: `translateY(${shift}px)` }, { transform: "none" }]
        : [{ transform: `translateY(${Math.sign(shift) * 24}px)`, opacity: 0 }, { transform: "none", opacity: 1 }];
      row.entry.animate(keyframes, { duration: 420, easing: "cubic-bezier(0.2, 0.8, 0.2, 1)" });
    }
  }

  // the table's head sticks below furo's header, which only shows on narrower screens
  const mobileHeader = document.querySelector(".mobile-header");
  function updateStickyTop() {
    const visible = mobileHeader && window.getComputedStyle(mobileHeader).display !== "none";
    table.style.setProperty("--tl-sticky-top", `${visible ? mobileHeader.offsetHeight : 0}px`);
  }
  window.addEventListener("resize", updateStickyTop, { passive: true });

  // -- filters ----------------------------------------------------------------------------------

  function matches(card) {
    if (state.lanes.size && !state.lanes.has(card.dataset.lane)) return false;
    if (state.implementedOnly && card.dataset.implemented !== "true") return false;
    const text = searchText.get(card);
    return state.terms.every((term) => text.includes(term));
  }

  function isFiltering() {
    return state.lanes.size > 0 || state.implementedOnly || state.terms.length > 0;
  }

  function applyFilters() {
    let shown = 0;
    for (const card of cards) {
      const visible = matches(card);
      card.hidden = !visible;
      chipById.get(card.id).hidden = !visible;
      rowById.get(card.id).entry.hidden = !visible;
      shown += visible;
    }
    for (const year of list.querySelectorAll(".tl-year")) {
      year.hidden = !year.querySelector(".tl-card:not([hidden])");
    }
    for (const [lane, laneRow] of laneRows) {
      const visible = laneRow.row.querySelectorAll(".tl-chip:not([hidden])").length;
      laneRow.row.hidden = visible === 0;
      laneRow.count.textContent = visible === 1 ? "1 entry" : `${visible} entries`;
      const { group, count } = groups.get(lane);
      group.hidden = visible === 0;
      count.textContent = String(visible);
    }
    for (const filter of filterButtons) {
      filter.setAttribute("aria-pressed", String(state.lanes.has(filter.dataset.lane)));
    }
    filters.classList.toggle("is-filtering", state.lanes.size > 0);
    implementedSwitch.setAttribute("aria-checked", String(state.implementedOnly));
    reset.hidden = !isFiltering();
    empty.hidden = shown > 0;
    status.textContent =
      shown === cards.length
        ? `Showing all ${cards.length} entries`
        : `Showing ${shown} of ${cards.length} entries`;
    if (state.open && !matches(cardById.get(state.open))) closeDrawer();
    updateScroll();
  }

  function resetFilters() {
    state.terms = [];
    state.lanes.clear();
    state.implementedOnly = false;
    searchInput.value = "";
    applyFilters();
  }

  let searchTimer;
  searchInput.addEventListener("input", () => {
    window.clearTimeout(searchTimer);
    searchTimer = window.setTimeout(() => {
      state.terms = normalize(searchInput.value).split(/\s+/).filter(Boolean);
      applyFilters();
    }, 80);
  });
  implementedSwitch.addEventListener("click", () => {
    state.implementedOnly = !state.implementedOnly;
    applyFilters();
  });
  for (const filter of filterButtons) {
    filter.addEventListener("click", () => {
      const lane = filter.dataset.lane;
      if (state.lanes.has(lane)) state.lanes.delete(lane);
      else state.lanes.add(lane);
      applyFilters();
    });
  }
  reset.addEventListener("click", resetFilters);
  emptyReset.addEventListener("click", resetFilters);

  // -- views ------------------------------------------------------------------------------------

  let shownView = null;

  function applyView() {
    const view = currentView();
    for (const name of Object.keys(VIEWS)) root.classList.toggle(`tl-view-${name}`, view === name);
    views.hidden = !wideScreen.matches;
    key.hidden = view !== "lanes"; // the key explains the swimlanes' cards only
    for (const viewButton of viewButtons) {
      viewButton.setAttribute("aria-pressed", String(viewButton.dataset.view === state.view));
    }
    if (shownView && shownView !== view) {
      // keep the open entry open in the other view
      const id = state.open || hashId();
      if (view !== "lanes") closeDrawer();
      if (cardById.has(id)) {
        if (view === "list") setCardOpen(cardById.get(id), true);
        else if (view === "table") setRowOpen(id, true);
        else if (matches(cardById.get(id))) showInDrawer(id, { instant: true });
      }
    }
    shownView = view;
    if (view === "lanes") updateScroll();
    if (view === "table") updateStickyTop();
  }

  for (const viewButton of viewButtons) {
    viewButton.addEventListener("click", () => {
      state.view = viewButton.dataset.view;
      saveView(state.view);
      applyView();
    });
  }
  wideScreen.addEventListener("change", applyView);

  // -- deep links -------------------------------------------------------------------------------

  function scrollToEntry(id, behavior) {
    if (currentView() === "list") {
      cardById.get(id).scrollIntoView({ block: "start", behavior });
      return;
    }
    if (currentView() === "table") {
      const below = head.getBoundingClientRect().height + parseFloat(table.style.getPropertyValue("--tl-sticky-top") || "0");
      const top = window.scrollY + rowById.get(id).entry.getBoundingClientRect().top - below - 12;
      window.scrollTo({ top, behavior });
      return;
    }
    const chip = chipById.get(id);
    const labelWidth = grid.querySelector(".tl-lane-label").offsetWidth;
    const chipBox = chip.getBoundingClientRect();
    const scrollerBox = scroller.getBoundingClientRect();
    const free = scroller.clientWidth - labelWidth - chipBox.width;
    const left = scroller.scrollLeft + chipBox.left - scrollerBox.left - labelWidth - free / 2;
    scroller.scrollTo({ left, behavior });
    const top = window.scrollY + chipBox.top - Math.min(140, window.innerHeight * 0.2);
    window.scrollTo({ top, behavior });
  }

  function openFromHash({ initial = false } = {}) {
    const id = hashId();
    const card = cardById.get(id);
    if (!card) return;
    if (!matches(card)) resetFilters();
    if (currentView() === "list") {
      card.classList.add("is-revealed");
      setCardOpen(card, true);
    } else if (currentView() === "table") {
      rowById.get(id).entry.classList.add("is-revealed");
      setRowOpen(id, true);
    } else if (state.open !== id) {
      showInDrawer(id, { instant: initial });
    }
    if (!initial) {
      scrollToEntry(id, scrollBehavior());
      const targets = {
        list: card.querySelector(".tl-card-toggle"),
        table: rowById.get(id).toggle,
        lanes: chipById.get(id),
      };
      targets[currentView()].focus({ preventScroll: true });
    } else if (document.readyState === "complete") {
      scrollToEntry(id, "instant");
    } else {
      // the browser scrolled to the card before the cards collapsed (or the list was hidden):
      // scroll again once the layout is final (images, fonts)
      window.addEventListener("load", () => scrollToEntry(id, "instant"), { once: true });
    }
  }

  window.addEventListener("hashchange", () => openFromHash());
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.open && currentView() === "lanes") {
      closeDrawer({ focus: lanesView.contains(document.activeElement) });
    }
  });

  // -- scroll reveal ----------------------------------------------------------------------------

  if ("IntersectionObserver" in window && !reducedMotion.matches) {
    root.classList.add("tl-animate");
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          entry.target.classList.add("is-revealed");
          observer.unobserve(entry.target);
        }
      },
      { rootMargin: "0px 0px -6% 0px" },
    );
    const tableRows = [
      ...[...groups.values()].map(({ group }) => group),
      ...[...rowById.values()].map((row) => row.entry),
    ];
    for (const node of [...cards, ...grid.querySelectorAll(".tl-lane"), ...tableRows]) {
      observer.observe(node);
    }
  }

  // -- start --------------------------------------------------------------------------------------

  root.prepend(toolbar, lanesView, tableView);
  list.before(empty);
  root.classList.add("tl-js");
  sortTable();
  applyFilters();
  applyView();
  if (window.location.hash) openFromHash({ initial: true });
})();
