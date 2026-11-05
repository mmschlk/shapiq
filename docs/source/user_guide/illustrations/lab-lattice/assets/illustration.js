// The illustration "lab-lattice" (chapter 2.1 of the user guide): Definition 2.1, e : 𝒮 → E, as a
// lit lattice. The 16 subsets of four features of one hour of the bike sharing data (hour, working
// day, temperature, humidity) stand in the Hasse diagram of 2^D, by size: the empty set on top, D
// at the bottom, a line from each set to the sets with one feature more. Choosing the units 𝒮 (a
// switch, or clicking sets) lights the sets of 𝒮 up; choosing the explanation space E (numbers or
// curves) decides what hangs on each lit set; the explanation e(S) is the object on the lit set S,
// and the whole explanation is the lit lattice. A panel below shows one set larger, with its axes.
//
// Which explanation e each choice is (the data: ../compute/compute.py, marginal masking):
// - Features, 𝒮 = {{1}, …, {4}}: Shapley values; ICE curves (the prediction as one feature varies).
// - Groups, 𝒮 = {{1, 2}, {3, 4}} (time and weather): the Shapley values of the groups, each group
//   one player; the ICE curves of the groups (the prediction as both of its features vary).
// - Pairs, the single features and the pairs: Shapley interactions up to pairs (2-SII); the
//   components of the functional decomposition, main effects and pure pair interactions.
// - All, 𝒮 = 2^D: the Möbius transform (the empty set holds the average prediction); the
//   functional decomposition f = Σ f_S, whose component f_S at this hour is the Möbius value of S.
// - Sets picked by the reader: the Möbius transform and the functional decomposition, as for all
//   sets, since the pure effect of a set does not depend on which other sets are picked.
//
// Red is positive and blue negative; amber marks the features, the lit sets, and values that are
// neither, such as a prediction (docs/STYLE_GUIDE.md).
window.shapiqIllustrations?.define("lab-lattice", async (stage, kit) => {
  const { el, svg, tween, ease, reducedMotion, segmented, loadScript, watchVisibility } = kit;
  // the data that the page loaded before this script, or, where it is missing, loaded here
  if (!window.SHAPIQ_LAB_LATTICE) await loadScript(`${kit.root}data.js?v=${kit.version}`);
  const data = window.SHAPIQ_LAB_LATTICE;
  const features = data.features;
  const n = features.length;
  const binary = new Set(data.binary);
  const uid = `llt${Math.random().toString(36).slice(2, 8)}`;
  let ids = 0;
  const hover = window.matchMedia("(hover: hover)");

  // -- sets of features: keys of their 0-based indices ("" is the empty set, "0123" is D) ------------

  const combinations = (size, from = 0) =>
    size === 0
      ? [""]
      : Array.from({ length: Math.max(0, n - from) }, (_, k) => from + k).flatMap((i) =>
          combinations(size - 1, i + 1).map((rest) => `${i}${rest}`),
        );
  const LEVELS = Array.from({ length: n + 1 }, (_, size) => combinations(size));
  const ALL = LEVELS.flat();
  const idsOf = (key) => [...key].map(Number);
  const setText = (key) => (key ? `{${idsOf(key).map((i) => i + 1).join(", ")}}` : "∅");
  const name = (i) => features[i].label.toLowerCase();
  const listing = (words) =>
    words.length < 2 ? words.join("") : `${words.slice(0, -1).join(", ")} and ${words.at(-1)}`;
  const namesOf = (key) => (key ? listing(idsOf(key).map(name)) : "no feature");

  // -- numbers: rentals per hour, whole ---------------------------------------------------------------

  const minus = "−";
  const whole = (w) => (w < 0 ? `${minus}${-w}` : `${w}`);
  const signed = (w) => (w > 0 ? `+${w}` : whole(w));
  const sign = (w) => (w > 0 ? "llt-pos" : w < 0 ? "llt-neg" : "llt-zero");
  const valueText = (i, v = features[i].value) => {
    if (binary.has(i)) return v >= 0.5 ? "yes" : "no";
    const r = Math.round(v);
    const unit = features[i].unit;
    if (unit === ":00") return `${r}:00`;
    if (unit === "%") return `${r}%`;
    return `${r} ${unit}`;
  };
  const [, predicted] = data.prediction;
  const [, average] = data.baseline;

  // -- the choices, and the explanation they give -------------------------------------------------------

  const PRESETS = {
    features: { label: "Features", sets: LEVELS[1] },
    groups: { label: "Groups", sets: data.groups.map((g) => g.join("")) },
    pairs: { label: "Pairs", sets: [...LEVELS[1], ...LEVELS[2]] },
    all: { label: "All", sets: ALL },
  };
  const SPACES = { numbers: "Numbers", curves: "Curves" };
  const NUMBERS = { features: "features", groups: "groups", pairs: "pairs", all: "all", custom: "all" };
  const CURVES = { features: "ice", groups: "ice", pairs: "components", all: "components", custom: "components" };
  // the explanation of each choice: its name, and for sets of the reader's own, why it is this one
  const METHODS = {
    numbers: {
      features: ["Shapley values"],
      groups: ["Shapley values of the groups"],
      pairs: ["Shapley interactions (2-SII)"],
      all: ["Möbius transform"],
      custom: ["Möbius transform", "for sets of your own: the value of a set does not depend on which others you pick"],
    },
    curves: {
      features: ["ICE curves"],
      groups: ["ICE curves of the groups"],
      pairs: ["Functional decomposition, up to pairs"],
      all: ["Functional decomposition"],
      custom: ["Functional decomposition", "for sets of your own: the function of a set does not depend on which others you pick"],
    },
  };
  // the set the panel shows first for each choice of units
  const SHOWCASE = { features: "0", groups: "01", pairs: "01", all: "01" };
  const state = { mode: "features", space: "numbers", members: new Set(PRESETS.features.sets), focus: "0" };

  // -- the scales of the curves, shared by all sets of a kind, so that their sizes compare -------------

  const shapeOf = (c) => {
    if (!c.axes.length) return "constant";
    if (c.axes.length === 1) return binary.has(c.axes[0]) ? "steps" : "line";
    return c.axes.some((i) => binary.has(i)) ? "lines" : "map";
  };
  const RANGES = {};
  for (const [source, curves] of Object.entries(data.curves)) {
    const lines = [];
    const maps = [];
    for (const c of Object.values(curves)) {
      const shape = shapeOf(c);
      if (shape === "constant") continue;
      (shape === "map" ? maps : lines).push(...c.z.flat());
    }
    if (source === "ice") {
      const hi = Math.max(...lines, ...maps);
      RANGES.ice = { lo: 0, hi: hi * 1.04, reach: hi };
    } else {
      const lo = Math.min(0, ...lines);
      const hi = Math.max(0, ...lines);
      const pad = (hi - lo) * 0.06;
      RANGES[source] = { lo: lo - pad, hi: hi + pad, reach: Math.max(...maps.map(Math.abs)) };
    }
  }

  // -- the layout -----------------------------------------------------------------------------------

  const root = el("div", "llt");
  const label = (symbol, words) => {
    const node = el("p", "llt-label");
    node.innerHTML = `<span class="llt-sym">${symbol}</span><span class="llt-label-words">${words}</span>`;
    return node;
  };
  const head = el("div", "llt-head");
  const unitsSwitch = segmented(
    "Units",
    Object.entries(PRESETS).map(([id, p]) => [id, p.label]),
    (id, byUser) => byUser && choosePreset(id),
  );
  unitsSwitch.node.classList.add("llt-units-switch");
  const spaceSwitch = segmented(
    "Explanation space",
    Object.entries(SPACES),
    (id, byUser) => byUser && chooseSpace(id),
  );
  const unitsChoice = el("div", "llt-choice");
  const unitsValue = el("p", "llt-choice-value");
  unitsChoice.append(label('<span class="llt-cal">𝒮</span>', "the units"), unitsSwitch.node, unitsValue);
  const spaceChoice = el("div", "llt-choice");
  const spaceValue = el("p", "llt-choice-value");
  spaceChoice.append(label("<i>E</i>", "the explanation space"), spaceSwitch.node, spaceValue);
  const explChoice = el("div", "llt-choice llt-choice-e");
  const methodName = el("p", "llt-method");
  const methodNote = el("p", "llt-method-note");
  explChoice.append(label("<i>e</i>", "the explanation"), methodName, methodNote);
  head.append(unitsChoice, spaceChoice, explChoice);

  // the features D of this hour, numbered as in the vocabulary
  const chipsRow = el("div", "llt-features");
  const chips = features.map((f, i) => {
    const chip = el("span", "shapiq-ill-chip llt-chip");
    chip.appendChild(el("span", "shapiq-ill-chip-index", String(i + 1)));
    const words = el("span", "shapiq-ill-chip-text", f.label);
    words.append(" ", el("span", "shapiq-ill-chip-value", valueText(i)));
    chip.appendChild(words);
    return chip;
  });
  const dLabel = label("<i>D</i>", "the features of this hour");
  const chipList = el("div", "llt-chip-list");
  chipList.append(...chips);
  chipsRow.append(dLabel, chipList);

  // the lattice: the lines between the sets, the sets, and two notes in its empty corners
  const lattice = el("div", "llt-lattice");
  lattice.setAttribute("role", "group");
  lattice.setAttribute("aria-label", "All 16 sets of the features, by size; the lit ones are the units");
  const edges = svg("svg", { class: "llt-edges", "aria-hidden": "true" });
  lattice.appendChild(edges);
  const cornerLeft = el("p", "llt-corner llt-corner-left");
  cornerLeft.innerHTML =
    '<span class="llt-corner-big">2<sup><i>D</i></sup></span><span>all 16 sets of the features, by size</span>';
  const cornerRight = el("p", "llt-corner llt-corner-right");
  cornerRight.innerHTML =
    '<span class="llt-key"><span class="llt-swatch is-lit"></span><span>a unit <i>S</i> ∈ <span class="llt-cal">𝒮</span></span></span>' +
    '<span class="llt-key"><span class="llt-swatch"></span><span>not in <span class="llt-cal">𝒮</span></span></span>' +
    `<span class="llt-key llt-key-hint">${hover.matches ? "click" : "tap"} a set to add or remove it</span>`;
  lattice.append(cornerLeft, cornerRight);

  const nodes = new Map();
  for (const key of ALL) {
    const node = el("button", key ? "llt-node" : "llt-node llt-node-empty");
    node.type = "button";
    const set = el("span", "llt-set", setText(key));
    const body = el("span", "llt-body");
    node.append(set, body);
    lattice.appendChild(node);
    const entry = { key, node, set, body, current: null, armed: undefined, lit: false };
    nodes.set(key, entry);
    node.addEventListener("pointerenter", (event) => {
      if (event.pointerType === "mouse") setFocus(key);
    });
    // a tap shows a set first and picks it on the next; a click (or a key) picks it at once
    node.addEventListener("pointerdown", (event) => {
      entry.armed = event.pointerType === "mouse" || state.focus === key;
    });
    node.addEventListener("focus", () => setFocus(key));
    // one set in the tab order; the arrow keys move along a row and to the nearest set of the next
    node.addEventListener("keydown", (event) => {
      const step = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[event.key];
      if (!step) return;
      event.preventDefault();
      const level = key.length;
      const row = LEVELS[level];
      let next = key;
      if (step[0] === 0) next = row[(row.indexOf(key) + step[1] + row.length) % row.length];
      else if (LEVELS[level + step[0]]) {
        const [x] = geom.centers.get(key);
        next = LEVELS[level + step[0]].reduce((best, k) =>
          Math.abs(geom.centers.get(k)[0] - x) < Math.abs(geom.centers.get(best)[0] - x) ? k : best,
        );
      }
      nodes.get(next).node.focus();
    });
    node.addEventListener("click", () => {
      const armed = entry.armed;
      entry.armed = undefined;
      if (armed === false) return setFocus(key);
      toggle(key);
    });
  }

  // the panel: the set in focus, what its explanation says, and its curve larger, with axes
  const panel = el("div", "llt-panel");
  const panelText = el("div", "llt-panel-text");
  const panelSet = el("p", "llt-panel-set");
  const panelEquation = el("p", "llt-panel-equation");
  const panelMeaning = el("p", "llt-panel-meaning");
  const panelHint = el("p", "llt-panel-hint");
  panelText.append(panelSet, panelEquation, panelMeaning, panelHint);
  const panelFigure = el("div", "llt-panel-figure");
  const panelReadout = el("p", "llt-panel-readout");
  panelReadout.setAttribute("aria-live", "polite");
  const panelGraphic = el("div", "llt-panel-graphic");
  panelFigure.append(panelGraphic, panelReadout);
  panel.append(panelText, panelFigure);

  const sum = el("p", "llt-sum");
  const foot = el("p", "llt-foot");
  const day = new Date(`${data.hour.day}T08:00:00`);
  const dayText = `${data.hour.weekday}, ${day.getDate()} ${day.toLocaleString("en", { month: "long" })} ${day.getFullYear()}`;
  foot.textContent =
    `A gradient boosting model on four features of the UCI bike sharing data (R² ` +
    `${data.r2.toFixed(2)} on held-out hours) predicts ${predicted} rentals for ${dayText}, ` +
    `${valueText(0)} (${data.hour.actual} were rented), against ${average} on average over ` +
    `${data.background.toLocaleString("en")} hours of the training data.`;

  root.append(head, chipsRow, lattice, panel, sum, foot);
  stage.appendChild(root);

  // -- the geometry of the lattice ------------------------------------------------------------------

  let geom = null;
  // a lit set is a card with its explanation; a set that is not lit, a slim pill with its name,
  // both centered on the set's place, so that the lines between the places run behind them
  const sizeOf = (key, lit) => {
    const [cx, cy] = geom.centers.get(key);
    const width = geom.widths[key.length];
    const height = lit ? geom.h : geom.slim;
    return { left: Math.round(cx - width / 2), top: Math.round(cy - height / 2), width, height };
  };
  const objWidth = (key) => geom.widths[key.length] - 2 * geom.pad - 3;
  const placeNode = (key, lit) => {
    const box = sizeOf(key, lit);
    Object.assign(nodes.get(key).node.style, {
      left: `${box.left}px`,
      top: `${box.top}px`,
      width: `${box.width}px`,
      height: `${box.height}px`,
    });
  };
  function layout() {
    const width = lattice.clientWidth;
    const gap = width >= 480 ? 10 : 5;
    const margin = 4; // room for the ring of the set in focus
    const room = width - 2 * margin;
    const w = Math.floor(Math.min(124, (room - 5 * gap) / 6));
    const big = w >= 76;
    const pad = big ? 6 : 3;
    const setH = big ? 20 : 15;
    const objH = Math.round(Math.max(24, Math.min(42, w * 0.36)));
    const h = pad + setH + 2 + objH + pad + 3;
    const slim = setH + 2 * pad + 4;
    const rowGap = Math.round(Math.max(16, Math.min(28, width * 0.04)));
    const pitch = { 1: 0, 4: Math.min((room - w) / 3, (w + gap) * 1.45), 6: w + gap };
    // the rows of four and of one have room for wider sets, which narrow figures use for their names
    const widths = LEVELS.map((keys) =>
      big || keys.length === 6 ? w : Math.floor(Math.min(Math.max(w, keys.length === 1 ? 66 : 56), pitch[4] - gap)),
    );
    const height = (n + 1) * h + n * rowGap;
    geom = { width, big, w, widths, h, slim, pad, setH, objH, rowGap, barHalf: (w - 2 * pad - 3) / 2 - 3, centers: new Map() };
    lattice.style.height = `${height}px`;
    lattice.classList.toggle("is-small", !big);
    LEVELS.forEach((keys, level) => {
      const cy = level * (h + rowGap) + h / 2;
      keys.forEach((key, k) => {
        geom.centers.set(key, [width / 2 + (k - (keys.length - 1) / 2) * pitch[keys.length], cy]);
      });
    });
    // the names of the sets: compact where the figure is narrow
    for (const [key, entry] of nodes) entry.set.textContent = big || !key ? setText(key) : setText(key).replaceAll(", ", ",");
    for (const [key, entry] of nodes) placeNode(key, entry.lit);
    // the corners beside the empty set: as wide as the room beside it
    const corner = Math.max(0, (width - widths[0]) / 2 - gap * 2);
    for (const node of [cornerLeft, cornerRight]) node.style.width = `${corner}px`;
    cornerLeft.style.display = cornerRight.style.display = corner >= 70 ? "" : "none";
    // a line from each set to each set with one feature more, from center to center
    edges.setAttribute("viewBox", `0 0 ${width} ${height}`);
    edges.replaceChildren();
    for (const key of ALL) {
      const [x1, y1] = geom.centers.get(key);
      for (let i = 0; i < n; i++) {
        if (key.includes(String(i))) continue;
        const child = [...idsOf(key), i].sort((a, b) => a - b).join("");
        const [x2, y2] = geom.centers.get(child);
        svg("line", { x1, y1, x2, y2, class: "llt-edge", "data-from": key, "data-to": child }, edges);
      }
    }
  }

  // -- drawing a curve, a pair of curves, or a surface ------------------------------------------------

  // the color of a point of a surface: a prediction in amber, from the lowest to the highest of its
  // surface (lo, hi); an effect in red or blue, on the scale of all surfaces of its kind
  const color = (z, source, lo = 0, hi = 1) => {
    if (source === "ice") {
      const t = Math.max(0, Math.min(1, (z - lo) / (hi - lo || 1)));
      return `color-mix(in oklab, var(--shapiq-amber) ${Math.round(8 + t * 92)}%, transparent)`;
    }
    const t = Math.min(1, Math.abs(z) / RANGES[source].reach);
    return `color-mix(in oklab, var(${z >= 0 ? "--shapiq-red" : "--shapiq-blue"}) ${Math.round(t * 100)}%, transparent)`;
  };
  const pathOf = (points) => points.map(([x, y], k) => `${k ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join("");
  const tickText = (i, v) => (binary.has(i) ? (v >= 0.5 ? "yes" : "no") : valueText(i, v));

  // draws the curve c of the source ("ice" or "components") into the box of the svg node; with
  // `axes`, with ticks and names. Returns a function from a point of the node to words, for the readout.
  function drawCurve(node, c, source, width, height, { axes = false } = {}) {
    const shape = shapeOf(c);
    const m = axes ? { left: 40, right: 10, top: 22, bottom: 30 } : { left: 1, right: 1, top: 2, bottom: 2 };
    if (axes && shape === "map") m.right = 46;
    const x0 = m.left;
    const x1 = width - m.right;
    const y0 = m.top;
    const y1 = height - m.bottom;
    const range = RANGES[source];
    const Y = (z) => y1 - ((z - range.lo) / (range.hi - range.lo)) * (y1 - y0);
    const signedSource = source !== "ice";
    const text = (x, y, content, cls, anchor = "middle") => {
      const t = svg("text", { x, y, class: cls, "text-anchor": anchor }, node);
      t.textContent = content;
      return t;
    };
    // the dot is this hour; the curve grows out of it when it enters (place)
    const dot = (cx, cy) => {
      Object.assign(node.dataset, { dotX: cx, dotY: cy });
      return svg("circle", { cx, cy, r: axes ? 4 : 2.6, class: "llt-dot" }, node);
    };

    if (shape === "constant") {
      const y = (y0 + y1) / 2 + (axes ? 6 : 4);
      svg("line", { x1: x0, x2: x1, y1: y, y2: y, class: "llt-line llt-line-amber" }, node);
      text((x0 + x1) / 2, y - (axes ? 8 : 5), whole(average), axes ? "llt-axis-value" : "llt-mini-value");
      if (axes) text(x0, height - 8, "a constant: the average prediction", "llt-axis-name", "start");
      return () => `f_∅ = ${average} rentals, the average prediction, whatever the features`;
    }

    // a signed line: red above zero, blue below; an amber line for a prediction
    const drawLine = (points, { faint = false } = {}) => {
      const d = pathOf(points);
      const base = signedSource ? Y(0) : y1;
      const area = `${d}L${points.at(-1)[0].toFixed(1)} ${base.toFixed(1)}L${points[0][0].toFixed(1)} ${base.toFixed(1)}Z`;
      const g = svg("g", { class: faint ? "llt-faint" : "" }, node);
      if (!signedSource) {
        if (!faint) svg("path", { d: area, class: "llt-area llt-area-amber" }, g);
        svg("path", { d, class: `llt-line llt-line-amber${faint ? " llt-dashed" : ""}` }, g);
        return;
      }
      const id = `${uid}-${ids++}`;
      const defs = svg("defs", {}, g);
      const up = svg("clipPath", { id: `${id}-up` }, defs);
      svg("rect", { x: 0, y: 0, width, height: Math.max(0, base) }, up);
      const down = svg("clipPath", { id: `${id}-down` }, defs);
      svg("rect", { x: 0, y: base, width, height: Math.max(0, height - base) }, down);
      for (const [part, cls] of [["up", "pos"], ["down", "neg"]]) {
        if (!faint) svg("path", { d: area, class: `llt-area llt-area-${cls}`, "clip-path": `url(#${id}-${part})` }, g);
        svg("path", { d, class: `llt-line llt-line-${cls}${faint ? " llt-dashed" : ""}`, "clip-path": `url(#${id}-${part})` }, g);
      }
    };
    const zeroLine = () => {
      if (signedSource) svg("line", { x1: x0, x2: x1, y1: Y(0), y2: Y(0), class: "llt-zero-line" }, node);
    };
    const yTicks = () => {
      if (!axes) return;
      text(0, 11, signedSource ? "↑ effect on rentals" : "↑ rentals", "llt-axis-name", "start");
      const ticks = signedSource ? [range.lo / 1.06, 0, range.hi / 1.06] : [0, range.reach];
      for (const v of ticks) {
        const r = Math.round(v / 10) * 10;
        text(x0 - 6, Y(r) + 4, signedSource ? signed(r) : whole(r), "llt-axis-tick", "end");
        svg("line", { x1: x0 - 3, x2: x0, y1: Y(r), y2: Y(r), class: "llt-axis-line" }, node);
      }
    };
    const xTicks = (i, X, grid) => {
      if (!axes) return;
      const values = i === 0 ? [0, 6, 12, 18, 23] : [grid[0], grid[Math.round((grid.length - 1) / 2)], grid.at(-1)];
      values.forEach((v, k) => {
        const anchor = k === 0 ? "start" : k === values.length - 1 ? "end" : "middle";
        text(X(v), y1 + 14, tickText(i, v), "llt-axis-tick", i === 0 ? "middle" : anchor);
      });
      text(x1, height - 2, `${features[i].label} →`, "llt-axis-name", "end");
    };
    const wordsAt = (point) => {
      const words = point.map(([i, v]) => `${name(i)} ${tickText(i, v)}`);
      return words.join(", ");
    };
    const valueWords = (z, isHour) =>
      signedSource ? `${signed(Math.round(z))}${isHour ? " (this hour)" : ""}` : `${whole(Math.round(z))} rentals${isHour ? " (this hour)" : ""}`;

    if (shape === "line") {
      const [i] = c.axes;
      const grid = c.grids[0];
      const X = (v) => x0 + ((v - grid[0]) / (grid.at(-1) - grid[0])) * (x1 - x0);
      zeroLine();
      yTicks();
      drawLine(grid.map((v, k) => [X(v), Y(c.z[k])]));
      xTicks(i, X, grid);
      dot(X(grid[c.at[0]]), Y(c.z[c.at[0]]));
      return (px) => {
        const v = grid[0] + ((px - x0) / (x1 - x0)) * (grid.at(-1) - grid[0]);
        let k = 0;
        grid.forEach((g, j) => {
          if (Math.abs(g - v) < Math.abs(grid[k] - v)) k = j;
        });
        return `${wordsAt([[i, grid[k]]])}: ${valueWords(c.z[k], k === c.at[0])}`;
      };
    }

    if (shape === "steps") {
      const [i] = c.axes;
      const X = (v) => x0 + (0.25 + 0.5 * v) * (x1 - x0);
      const half = (x1 - x0) * 0.16;
      zeroLine();
      yTicks();
      c.grids[0].forEach((v, k) => {
        const g = svg("g", { class: k === c.at[0] ? "" : "llt-faint" }, node);
        const z = c.z[k];
        svg("line", { x1: X(v) - half, x2: X(v) + half, y1: Y(z), y2: Y(z), class: `llt-line ${signedSource ? `llt-line-${z >= 0 ? "pos" : "neg"}` : "llt-line-amber"}` }, g);
        if (axes) text(X(v), y1 + 14, tickText(i, v), "llt-axis-tick");
      });
      if (axes) text(x1, height - 2, `${features[i].label}`, "llt-axis-name", "end");
      dot(X(c.grids[0][c.at[0]]), Y(c.z[c.at[0]]));
      return (px) => {
        const k = px < (x0 + x1) / 2 ? 0 : 1;
        return `${wordsAt([[i, c.grids[0][k]]])}: ${valueWords(c.z[k], k === c.at[0])}`;
      };
    }

    if (shape === "lines") {
      // one curve per value of the binary feature, over the other: the hour's own value solid
      const b = binary.has(c.axes[0]) ? 0 : 1;
      const o = 1 - b;
      const i = c.axes[o];
      const grid = c.grids[o];
      const X = (v) => x0 + ((v - grid[0]) / (grid.at(-1) - grid[0])) * (x1 - x0);
      const zOf = (bv, k) => (b === 0 ? c.z[bv][k] : c.z[k][bv]);
      const mine = c.at[b];
      zeroLine();
      yTicks();
      for (const bv of [1 - mine, mine]) {
        drawLine(grid.map((v, k) => [X(v), Y(zOf(bv, k))]), { faint: bv !== mine });
      }
      xTicks(i, X, grid);
      if (axes) {
        const j = c.axes[b];
        const legend = text(x1, 11, "", "llt-axis-legend", "end");
        const solid = svg("tspan", { class: "llt-legend-solid" }, legend);
        solid.textContent = `— ${name(j)}: ${tickText(j, c.grids[b][mine])}`;
        const dashed = svg("tspan", { class: "llt-legend-dashed", dx: 10 }, legend);
        dashed.textContent = `- - ${tickText(j, c.grids[b][1 - mine])}`;
      }
      dot(X(grid[c.at[o]]), Y(zOf(mine, c.at[o])));
      return (px, py) => {
        const v = grid[0] + ((px - x0) / (x1 - x0)) * (grid.at(-1) - grid[0]);
        let k = 0;
        grid.forEach((g, j) => {
          if (Math.abs(g - v) < Math.abs(grid[k] - v)) k = j;
        });
        // the curve nearer to the pointer
        const bv = Math.abs(Y(zOf(0, k)) - py) < Math.abs(Y(zOf(1, k)) - py) ? 0 : 1;
        const j = c.axes[b];
        return `${wordsAt([[i, grid[k]], [j, c.grids[b][bv]]])}: ${valueWords(zOf(bv, k), k === c.at[o] && bv === mine)}`;
      };
    }

    // a surface: the first feature across, the second up
    const [i, j] = c.axes;
    const [gi, gj] = c.grids;
    const cw = (x1 - x0) / gi.length;
    const ch = (y1 - y0) / gj.length;
    const cells = svg("g", { class: "llt-cells", "shape-rendering": "crispEdges" }, node);
    const flat = c.z.flat();
    const zLo = Math.min(...flat);
    const zHi = Math.max(...flat);
    c.z.forEach((row, a) => {
      row.forEach((z, b) => {
        // on whole pixels, so that neighbors meet without seams
        const left = Math.round(x0 + a * cw);
        const top = Math.round(y1 - (b + 1) * ch);
        const rect = svg("rect", { x: left, y: top, width: Math.round(x0 + (a + 1) * cw) - left, height: Math.round(y1 - b * ch) - top }, cells);
        rect.style.fill = color(z, source, zLo, zHi);
      });
    });
    svg("rect", { x: x0, y: y0, width: x1 - x0, height: y1 - y0, class: "llt-frame" }, node);
    dot(x0 + (c.at[0] + 0.5) * cw, y1 - (c.at[1] + 0.5) * ch);
    if (axes) {
      const X = (v) => x0 + ((v - gi[0]) / (gi.at(-1) - gi[0])) * (x1 - x0 - cw) + cw / 2;
      xTicks(i, X, gi);
      text(x0 - 6, y1 - ch / 2 + 4, tickText(j, gj[0]), "llt-axis-tick", "end");
      text(x0 - 6, y0 + ch / 2 + 4, tickText(j, gj.at(-1)), "llt-axis-tick", "end");
      text(0, 11, `↑ ${features[j].label}`, "llt-axis-name", "start");
      // the scale of the colors, beside the map
      const sx = x1 + 12;
      const steps = 24;
      const lo = signedSource ? -range.reach : zLo;
      const hi = signedSource ? range.reach : zHi;
      for (let k = 0; k < steps; k++) {
        const z = lo + ((k + 0.5) / steps) * (hi - lo);
        const rect = svg("rect", { x: sx, y: y1 - ((k + 1) / steps) * (y1 - y0), width: 8, height: (y1 - y0) / steps + 0.5 }, node);
        rect.style.fill = color(z, source, zLo, zHi);
      }
      svg("rect", { x: sx, y: y0, width: 8, height: y1 - y0, class: "llt-frame" }, node);
      const r = (v) => Math.round(v);
      text(sx + 11, y0 + 8, signedSource ? signed(r(hi)) : whole(r(hi)), "llt-axis-tick", "start");
      text(sx + 11, y1, signedSource ? signed(r(lo)) : whole(r(lo)), "llt-axis-tick", "start");
    }
    return (px, py) => {
      const a = Math.max(0, Math.min(gi.length - 1, Math.floor((px - x0) / cw)));
      const b = Math.max(0, Math.min(gj.length - 1, Math.floor((y1 - py) / ch)));
      const fixed = c.fixed.map((f) => [f, features[f].value]);
      const isHour = a === c.at[0] && b === c.at[1];
      return `${wordsAt([[i, gi[a]], [j, gj[b]], ...fixed])}: ${valueWords(c.z[a][b], isHour)}`;
    };
  }

  // -- what hangs on a set: a number or a curve ---------------------------------------------------------

  let stops = [];
  const motion = () => !reducedMotion();
  const want = (key) => {
    if (!state.members.has(key)) return null;
    if (state.space === "numbers") {
      const source = NUMBERS[state.mode];
      const [v, w] = data.numbers[source][key];
      return { kind: "number", id: `n:${source}:${key}`, v, w, neutral: key === "" };
    }
    const source = CURVES[state.mode];
    return { kind: "curve", id: `c:${source}:${key}`, source, c: data.curves[source][key] };
  };
  const numberScale = () => {
    const values = [...state.members].filter((k) => k).map((k) => Math.abs(want(k)?.v ?? 0));
    return Math.max(1, ...values);
  };

  // a number: its value, and a bar from the middle on the scale of the lit sets
  function makeNumber(entry, target, scale) {
    const { objH: H, big } = geom;
    const W = objWidth(entry.key);
    const node = svg("svg", { class: "llt-e", width: W, height: H, viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true" });
    const text = svg("text", { x: W / 2, y: big ? H * 0.52 : H * 0.56, "text-anchor": "middle", class: "llt-num" }, node);
    const out = { kind: "number", node, v: 0, w: 0, target };
    if (target.neutral) {
      text.classList.add("llt-num-neutral");
      text.textContent = whole(target.w);
      if (big) {
        const t = svg("text", { x: W / 2, y: H - 3, "text-anchor": "middle", class: "llt-num-note" }, node);
        t.textContent = "average";
      }
      out.set = () => {};
      out.v = target.v;
      out.w = target.w;
      return out;
    }
    const barY = H - (big ? 9 : 7);
    const barH = big ? 5 : 4;
    svg("line", { x1: W / 2, x2: W / 2, y1: barY - 2, y2: barY + barH + 2, class: "llt-zero-line" }, node);
    const bar = svg("rect", { y: barY, height: barH, rx: barH / 2 }, node);
    const half = geom.barHalf;
    out.set = (v, w, s) => {
      const length = (Math.abs(v) / s) * half;
      bar.setAttribute("x", v >= 0 ? W / 2 : W / 2 - length);
      bar.setAttribute("width", Math.max(0, length));
      bar.setAttribute("class", `llt-bar ${sign(v)}`);
      text.textContent = signed(w);
      text.setAttribute("class", `llt-num ${sign(w)}`);
    };
    out.set(0, 0, scale);
    return out;
  }

  function place(entry, target, scale, animated, delay) {
    const { body } = entry;
    if (target.kind === "number") {
      const made = makeNumber(entry, target, scale);
      body.appendChild(made.node);
      entry.current = made;
      if (target.neutral) {
        if (animated) made.node.animate([{ opacity: 0, transform: "scale(0.6)" }, { opacity: 1, transform: "none" }], { duration: 420, delay, easing: "cubic-bezier(0.2, 0.7, 0.2, 1)", fill: "backwards" });
        return;
      }
      retarget(made, target, scale, animated, delay, 0, 0);
      return;
    }
    const W = objWidth(entry.key);
    const node = svg("svg", { class: "llt-e", width: W, height: geom.objH, viewBox: `0 0 ${W} ${geom.objH}`, "aria-hidden": "true" });
    drawCurve(node, target.c, target.source, W, geom.objH);
    body.appendChild(node);
    entry.current = { kind: "curve", node, target };
    if (animated) {
      // the curve grows out of its dot, this hour, to all the values of its features: across for a
      // curve, across and up and down for a surface
      const H = geom.objH;
      const dx = Number(node.dataset.dotX ?? W / 2);
      const dy = Number(node.dataset.dotY ?? H / 2);
      const map = shapeOf(target.c) === "map";
      const from = map ? `inset(${dy}px ${W - dx}px ${H - dy}px ${dx}px)` : `inset(0 ${W - dx}px 0 ${dx}px)`;
      node.animate([{ clipPath: from }, { clipPath: "inset(0 0 0 0)" }], {
        duration: 750,
        delay,
        easing: "cubic-bezier(0.4, 0, 0.2, 1)",
        fill: "backwards",
      });
    }
  }

  // a number moves to its next value: the bar grows or shrinks, the digits count
  function retarget(made, target, scale, animated, delay, fromV = made.v, fromW = made.w) {
    made.target = target;
    const toV = target.v;
    const toW = target.w;
    made.v = toV;
    made.w = toW;
    stops.push(
      tween(
        animated ? 700 : 0,
        (p) => made.set(fromV + (toV - fromV) * p, p >= 1 ? toW : Math.round(fromW + (toW - fromW) * p), scale),
        { easing: ease.spring, delay },
      ),
    );
  }

  function drop(entry, animated) {
    const old = entry.current;
    entry.current = null;
    if (!old) return;
    if (!animated) return old.node.remove();
    old.node.classList.add("is-leaving");
    old.node.animate([{ opacity: 1 }, { opacity: 0, transform: "scale(0.85)" }], { duration: 180, easing: "ease-in" }).finished.then(
      () => old.node.remove(),
      () => old.node.remove(),
    );
  }

  // -- the whole picture --------------------------------------------------------------------------------

  let seen = false;
  let ready = reducedMotion(); // the units light up when the lattice has built itself
  function render({ animated = true, rebuild = false, stagger = false } = {}) {
    for (const stop of stops) stop();
    stops = [];
    animated = animated && seen && motion();
    const scale = state.space === "numbers" ? numberScale() : 1;
    for (const [key, entry] of nodes) {
      const target = ready ? want(key) : null;
      const lit = Boolean(target);
      const wasLit = entry.lit;
      entry.lit = lit;
      entry.node.classList.toggle("is-lit", lit);
      if (lit !== wasLit) placeNode(key, lit);
      entry.node.setAttribute("aria-pressed", String(lit));
      entry.node.setAttribute("aria-label", ariaOf(key, target));
      const delay = stagger ? 120 + key.length * 110 : 0;
      if (lit && !wasLit && animated) {
        entry.node.classList.remove("is-lighting");
        void entry.node.offsetWidth; // to start the glow again
        entry.node.classList.add("is-lighting");
      }
      if (!target) {
        drop(entry, animated);
        continue;
      }
      const current = entry.current;
      if (!rebuild && current?.kind === "number" && target.kind === "number" && !current.target.neutral && !target.neutral) {
        retarget(current, target, scale, animated, delay);
      } else if (!rebuild && current?.target.id === target.id) {
        // the same object: it stays
      } else {
        drop(entry, animated && !rebuild);
        place(entry, target, scale, animated, delay + (current && !rebuild ? 120 : 0));
      }
    }
    renderText();
    renderPanel();
  }

  const ariaOf = (key, target) => {
    const what = `${setText(key)}, ${namesOf(key)}`;
    if (!target) return `${what}: not in the units`;
    if (target.kind === "number") return `${what}: ${target.neutral ? whole(target.w) : signed(target.w)} rentals`;
    return `${what}: a ${shapeOf(target.c) === "map" ? "surface" : "curve"}`;
  };

  function renderText() {
    const [method, note = ""] = METHODS[state.space][state.mode];
    methodNote.textContent = note;
    if (methodName.textContent !== method && seen && motion()) {
      methodName.animate([{ opacity: 0, transform: "translateY(4px)" }, { opacity: 1, transform: "none" }], { duration: 300, easing: "ease-out" });
      explChoice.classList.remove("is-changed");
      void explChoice.offsetWidth;
      explChoice.classList.add("is-changed");
    }
    methodName.textContent = method;
    const cal = '<span class="llt-cal">𝒮</span>';
    const members = ALL.filter((k) => state.members.has(k));
    const listed = `{${members.map(setText).join(", ")}}`;
    unitsValue.innerHTML = `${cal} = ${
      {
        features: listed,
        groups: listed,
        pairs: "{<i>S</i> : |<i>S</i>| = 1 or 2}",
        all: "2<sup><i>D</i></sup>, all 16 sets",
      }[state.mode] ?? (members.length <= 3 ? listed : `your ${members.length} sets`)
    }`;
    spaceValue.innerHTML = state.space === "numbers" ? "<i>E</i> = ℝ" : "<i>E</i> = functions";

    // where the explanation adds up, the line that adds it up
    const shown = [...state.members].map((k) => data.numbers[NUMBERS[state.mode]][k]?.[1] ?? 0);
    const parts = shown.reduce((a, b) => a + b, 0);
    const amount = (w, cls = sign(w)) => `<span class="${cls}">${w}</span>`;
    if (state.space === "curves") {
      sum.innerHTML =
        CURVES[state.mode] === "ice"
          ? `Each dot is this hour: ${amount(predicted, "llt-amber")} rentals predicted.`
          : state.mode === "all"
            ? `Each dot is this hour. There, the 16 functions add up to ${amount(predicted, "llt-amber")} rentals, this hour's prediction: their values are the numbers of the Möbius transform.`
            : "Each dot is this hour; there, the value of a function is the number of the Möbius transform for its set.";
    } else if (state.mode === "all") {
      sum.innerHTML = `They add up: the 16 values, the average ${average} on ∅ among them, give ${amount(predicted, "llt-amber")} rentals, this hour's prediction.`;
    } else if (state.mode === "custom") {
      sum.innerHTML = state.members.size
        ? `Your ${state.members.size} ${state.members.size === 1 ? "set adds" : "sets add"} up to ${amount(parts, "llt-amber")}; all 16 sets add up to ${predicted}, this hour's prediction.`
        : `No set is lit: ${hover.matches ? "click" : "tap"} sets to pick your own units.`;
    } else {
      const total = predicted - average;
      sum.innerHTML = `They add up: average ${average} <span class="${sign(total)}">${total >= 0 ? "+" : minus} ${Math.abs(total)}</span> = ${amount(predicted, "llt-amber")} rentals, this hour's prediction.`;
    }
  }

  // -- the panel: the set in focus, larger ----------------------------------------------------------------

  let readFrom = null;
  function setFocus(key) {
    if (state.focus === key) return;
    state.focus = key;
    renderPanel(true);
  }

  // what the explanation of a set says, in words
  const capital = (words) => `${words[0].toUpperCase()}${words.slice(1)}`;
  function meaning(key, target) {
    const names = namesOf(key);
    const size = key.length;
    const gap = `${signed(predicted - average)} rentals between the average and this hour's prediction`;
    if (target.kind === "number") {
      if (target.neutral) return "The average prediction: what the model predicts knowing no feature of this hour.";
      switch (NUMBERS[state.mode]) {
        case "features":
          return `The share of ${names} in the ${gap}.`;
        case "groups": {
          const k = data.groups.findIndex((g) => g.join("") === key);
          return `${data.groupNames[k]}, ${names} as one player: its share in the ${gap}.`;
        }
        case "pairs":
          return size === 1
            ? `The share of ${names} in the ${gap}, its interactions aside.`
            : `What ${names} add together beyond each alone, as their share in the ${gap}.`;
        default:
          return size === 1
            ? `What ${names} does alone, beyond the average.`
            : `What ${names} do together that no smaller set of them does.`;
      }
    }
    if (target.source === "ice") {
      return size === 1
        ? `This hour's prediction as ${names} varies, the other features as they are. The dot is this hour.`
        : `This hour's prediction as ${names} vary, the other features as they are. The dot is this hour.`;
    }
    const dot = ` At the dot, this hour, it is ${signed(data.numbers.all[key][1])}: the number of the Möbius transform.`;
    if (!size) return "A constant: the average prediction, whatever the features.";
    if (size === 1) return `The main effect of ${names}: the average prediction as ${names} varies, minus the average.${dot}`;
    const slice = target.c.fixed.length ? ` A slice: ${listing(target.c.fixed.map(name))} as at this hour.` : "";
    return `${size === 2 ? "The pure interaction of" : "What"} ${names}${size === 2 ? ": what the two do together beyond each alone." : " do together beyond all smaller sets."}${slice}${dot}`;
  }

  function renderPanel(changed = false) {
    const key = state.focus;
    const target = want(key);
    for (const [k, entry] of nodes) {
      entry.node.classList.toggle("is-focus", k === key);
      entry.node.tabIndex = k === key ? 0 : -1;
    }
    chips.forEach((chip, i) => chip.classList.toggle("is-selected", key.includes(String(i))));
    for (const line of edges.children) {
      line.classList.toggle("is-focus", line.dataset.from === key || line.dataset.to === key);
    }
    const verb = hover.matches ? "Click" : "Tap";
    panelSet.innerHTML = `<i>S</i> = <span class="llt-set-inline">${setText(key)}</span> ${key ? capital(namesOf(key)) : "the empty set, no feature"}`;
    if (!target) {
      panelEquation.innerHTML = `not in <span class="llt-cal">𝒮</span>: not a unit, no explanation`;
      panelMeaning.textContent = "";
      panelHint.textContent = `${verb} the set to add it to the units.`;
    } else {
      if (target.kind === "number") {
        const value = target.neutral ? whole(target.w) : signed(target.w);
        panelEquation.innerHTML = `<i>e</i>(<i>S</i>) = <strong class="${target.neutral ? "llt-amber" : sign(target.w)}">${value}</strong> <span class="llt-unit">rentals per hour</span>`;
      } else {
        const shape = shapeOf(target.c);
        panelEquation.innerHTML = `<i>e</i>(<i>S</i>) = ${shape === "constant" ? "a constant" : shape === "map" ? "a surface" : shape === "lines" ? "two curves" : "a curve"}`;
      }
      panelMeaning.textContent = meaning(key, target);
      panelHint.textContent = `${verb} the set to take it out of the units.`;
    }

    // a curve, larger: with its axes, and a readout of the point under the pointer
    panelGraphic.replaceChildren();
    panelReadout.textContent = "";
    const curve = target?.kind === "curve";
    panel.classList.toggle("is-curve", curve);
    if (!curve) return;
    const width = Math.max(240, Math.min(460, panelGraphic.clientWidth || 300));
    const height = shapeOf(target.c) === "map" ? 150 : 128;
    const node = svg("svg", { class: "llt-panel-svg", width, height, viewBox: `0 0 ${width} ${height}` }, panelGraphic);
    node.setAttribute("aria-hidden", "true");
    readFrom = drawCurve(node, target.c, target.source, width, height, { axes: true });
    const read = (event) => {
      const box = node.getBoundingClientRect();
      panelReadout.textContent = readFrom(
        ((event.clientX - box.left) / box.width) * width,
        ((event.clientY - box.top) / box.height) * height,
      );
    };
    node.addEventListener("pointermove", read);
    node.addEventListener("pointerdown", read);
    if (changed && motion()) node.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 220, easing: "ease-out" });
  }

  // -- the choices --------------------------------------------------------------------------------------

  function choosePreset(id) {
    state.mode = id;
    state.members = new Set(PRESETS[id].sets);
    unitsSwitch.node.classList.remove("is-custom");
    if (!state.members.has(state.focus) || id !== "features") state.focus = SHOWCASE[id];
    render();
  }

  function chooseSpace(id) {
    state.space = id;
    render();
  }

  // a set clicked: in or out of the units; a collection that is one of the switch's is that one
  function toggle(key) {
    const next = new Set(state.members);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    state.members = next;
    state.focus = key;
    const preset = Object.keys(PRESETS).find(
      (id) => PRESETS[id].sets.length === next.size && PRESETS[id].sets.every((k) => next.has(k)),
    );
    state.mode = preset ?? "custom";
    unitsSwitch.set(state.mode);
    unitsSwitch.node.classList.toggle("is-custom", !preset);
    if (!preset) unitsSwitch.node.querySelector(".shapiq-ill-segment").tabIndex = 0;
    render();
  }

  // -- start: the lattice builds itself when first seen, then the units light up ------------------------

  unitsSwitch.set("features");
  spaceSwitch.set("numbers");
  layout();
  render({ animated: false });
  lattice.classList.add("is-waiting");
  let lastWidth = lattice.clientWidth;
  new ResizeObserver(() => {
    if (Math.abs(lattice.clientWidth - lastWidth) < 2) return;
    lastWidth = lattice.clientWidth;
    layout();
    render({ animated: false, rebuild: true });
  }).observe(lattice);

  let stopWatching = null;
  stopWatching = watchVisibility(stage, (visible) => {
    if (!visible || seen) return;
    seen = true;
    stopWatching?.();
    lattice.classList.remove("is-waiting");
    if (!motion()) {
      ready = true;
      render({ animated: false });
      return;
    }
    // the sets appear by size, the lines between them draw, then the units light up
    for (const [key, entry] of nodes) {
      entry.node.animate([{ opacity: 0, transform: "scale(0.7)" }, { opacity: 1, transform: "none" }], {
        duration: 380,
        delay: key.length * 110,
        easing: "cubic-bezier(0.2, 0.7, 0.2, 1.2)",
        fill: "backwards",
      });
    }
    for (const line of edges.children) {
      const length = Math.hypot(line.x2.baseVal.value - line.x1.baseVal.value, line.y2.baseVal.value - line.y1.baseVal.value);
      line.style.strokeDasharray = `${length}`;
      line.animate([{ strokeDashoffset: length }, { strokeDashoffset: 0 }], {
        duration: 360,
        delay: 80 + line.dataset.from.length * 110,
        easing: "ease-out",
        fill: "backwards",
      }).finished.then(() => line.style.removeProperty("stroke-dasharray"), () => {});
    }
    setTimeout(() => {
      ready = true;
      render({ stagger: true });
    }, 640);
  });
});
