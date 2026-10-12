/*
 * Timeline of Shapley-based methods (timeline.rst).
 *
 * _ext/shapiq_timeline.py renders every entry as a card in a list grouped by year: the page
 * without JavaScript and the view on phones. This script adds filters (lanes, "in shapiq only",
 * search), collapsible cards, a swimlane view for wide screens that shows the open card in a
 * drawer below its lane, deep links (timeline.html#kernelshap) and a scroll-reveal animation.
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

  const ICONS = {
    search:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
    lanes:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M3 6h7M14 6h7M3 12h4M11 12h10M3 18h11M18 18h3"/></svg>',
    list: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M8 6h13M8 12h13M8 18h13"/><circle cx="4" cy="6" r="1"/><circle cx="4" cy="12" r="1"/><circle cx="4" cy="18" r="1"/></svg>',
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
      return window.localStorage.getItem(VIEW_KEY) === "list" ? "list" : "lanes";
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
  const viewButtons = ["lanes", "list"].map((view) => {
    const viewButton = button("");
    viewButton.dataset.view = view;
    viewButton.innerHTML = ICONS[view];
    viewButton.append(view === "lanes" ? "Lanes" : "List");
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
    '<li class="tl-key-lanes-only"><span class="tl-key-swatch" aria-hidden="true"></span>In shapiq</li>' +
    '<li class="tl-key-lanes-only"><span class="tl-key-swatch is-dashed" aria-hidden="true"></span>Not in shapiq</li>' +
    '<li><span class="tl-key-star" aria-hidden="true">★</span>Thesis contribution</li>';

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
          chip.dataset.thesis = card.dataset.thesis;
          chip.setAttribute("aria-expanded", "false");
          chip.setAttribute("aria-controls", drawer.id);
          chip.title = card.querySelector(".tl-card-summary").textContent.trim();
          chip.style.setProperty("--tl-index", String(index));
          const name = card.querySelector(".tl-card-name").textContent;
          const extra = card.dataset.implemented === "true" ? ", in shapiq" : "";
          const thesis = card.dataset.thesis === "true" ? ", thesis contribution" : "";
          chip.append(
            el("span", "tl-chip-year", card.dataset.year),
            el("span", "tl-chip-name", name),
            el("span", "tl-sr-only", `${extra}${thesis}`),
          );
          if (thesis) {
            const star = el("span", "tl-chip-star", "★");
            star.setAttribute("aria-hidden", "true");
            chip.append(star);
          }
          chip.addEventListener("click", () => {
            if (state.open === card.id) closeDrawer();
            else showInDrawer(card.id);
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
      shown += visible;
    }
    for (const year of list.querySelectorAll(".tl-year")) {
      year.hidden = !year.querySelector(".tl-card:not([hidden])");
    }
    for (const laneRow of laneRows.values()) {
      const visible = laneRow.row.querySelectorAll(".tl-chip:not([hidden])").length;
      laneRow.row.hidden = visible === 0;
      laneRow.count.textContent = visible === 1 ? "1 entry" : `${visible} entries`;
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
    root.classList.toggle("tl-view-lanes", view === "lanes");
    root.classList.toggle("tl-view-list", view === "list");
    views.hidden = !wideScreen.matches;
    for (const viewButton of viewButtons) {
      viewButton.setAttribute("aria-pressed", String(viewButton.dataset.view === state.view));
    }
    if (shownView && shownView !== view) {
      // keep the open entry open in the other view
      const id = state.open || window.location.hash.slice(1);
      if (view === "list") {
        closeDrawer();
        if (cardById.has(id)) setCardOpen(cardById.get(id), true);
      } else if (cardById.has(id) && matches(cardById.get(id))) {
        showInDrawer(id, { instant: true });
      }
    }
    shownView = view;
    if (view === "lanes") updateScroll();
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
    const id = decodeURIComponent(window.location.hash.slice(1));
    const card = cardById.get(id);
    if (!card) return;
    if (!matches(card)) resetFilters();
    if (currentView() === "list") {
      card.classList.add("is-revealed");
      setCardOpen(card, true);
    } else if (state.open !== id) {
      showInDrawer(id, { instant: initial });
    }
    if (!initial) {
      scrollToEntry(id, scrollBehavior());
      const target =
        currentView() === "list" ? card.querySelector(".tl-card-toggle") : chipById.get(id);
      target.focus({ preventScroll: true });
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
    for (const node of [...cards, ...grid.querySelectorAll(".tl-lane")]) observer.observe(node);
  }

  // -- start --------------------------------------------------------------------------------------

  root.prepend(toolbar, lanesView);
  list.before(empty);
  root.classList.add("tl-js");
  applyFilters();
  applyView();
  if (window.location.hash) openFromHash({ initial: true });
})();
