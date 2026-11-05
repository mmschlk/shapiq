// The illustration "lab-waterfall" (chapter 2.1 of the user guide): the same prediction, cut into
// different units. A waterfall climbs from the average prediction to the prediction for one house
// of the California housing data; each brick is one unit S of the collection 𝒮, and its height is
// the explanation e(S), a number: E = ℝ, the price axis. The reader switches 𝒮 between four
// explanations that all add up to the same total (they are efficient), and the bricks split and
// merge in place:
//
// - features: each feature alone, their Shapley values;
// - groups: the house's location (latitude and longitude) as one unit, the Shapley values of the
//   parts; the two bricks of the location merge into one;
// - pairs: every feature and every pair, the Shapley interactions of order 2 (k-SII);
// - all subsets: every non-empty subset, the Möbius transform.
//
// Under the bricks, a matrix of dots names each unit: one column per set S, a dot per feature in S.
// The reference lines (the average and this house) and the total never move: the units change what
// is explained, the total stays. Values are computed once by ../compute/compute.py into data.js; red
// is positive and blue negative, amber marks features and prices (docs/STYLE_GUIDE.md).
window.shapiqIllustrations?.define("lab-waterfall", async (stage, kit) => {
  const { el, svg, tween, ease, reducedMotion, segmented, loadScript, watchVisibility } = kit;
  if (!window.SHAPIQ_LAB_WATERFALL) await loadScript(`${kit.root}data.js?v=${kit.version}`);
  const data = window.SHAPIQ_LAB_WATERFALL;
  const features = data.features;
  const n = features.length;

  // -- words and numbers: values are in $1,000s, shown whole, so that the bricks add up -------------

  // a brick's key: its set of features; the location as a part of the partition is a brick of its
  // own, which splits into latitude and longitude and does not turn into their pair
  const keyOf = (view, ids) => `${view === "groups" && ids.length > 1 ? "part:" : ""}${ids.join("-")}`;
  const setOf = (ids) => `{${ids.map((k) => k + 1).join(", ")}}`;
  const money = (k) => (k === 0 ? "$0k" : `${k > 0 ? "+" : "−"}$${Math.abs(k)}k`);
  const compact = (k) => (k === 0 ? "0" : `${k > 0 ? "+" : "−"}${Math.abs(k)}`);
  const price = (k) => `$${k}k`;
  const listOf = (words) =>
    words.length < 3 ? words.join(" and ") : `${words.slice(0, -1).join(", ")}, and ${words.at(-1)}`;
  // the names of a unit's features; short, as in the matrix, for three or more
  const namesOf = (ids) =>
    ids.length > 2
      ? ids.map((k) => features[k].short).join(", ")
      : listOf(ids.map((k, j) => (j ? features[k].label.toLowerCase() : features[k].label)));

  const VIEWS = {
    features: {
      label: "Features",
      method: "Shapley values",
      blocks: false,
      featured: (units) => units[0],
    },
    groups: {
      label: "Groups",
      method: "Shapley values of the parts",
      blocks: false,
      // the unit that is new here: the location
      featured: (units) => units.find((ids) => ids.length > 1),
    },
    pairs: {
      label: "Pairs",
      short: "Pairs",
      method: "Shapley interactions (k‑SII)",
      blocks: true,
      featured: (units, values) => largest(units, values, 2),
    },
    all: {
      label: "All subsets",
      short: "All",
      method: "Möbius transform",
      blocks: true,
      featured: (units, values) => largest(units, values, 3),
    },
  };
  // the unit of at least `size` features with the largest value, in size
  function largest(units, values, size) {
    let best = null;
    units.forEach((ids, k) => {
      if (ids.length >= size && (best === null || Math.abs(values[k]) > Math.abs(values[best]))) best = k;
    });
    return units[best];
  }

  // the axis: from below the average to above the highest level any explanation reaches
  let lowest = data.baseline;
  let highest = data.prediction;
  for (const view of Object.values(data.views)) {
    let level = data.baseline;
    for (const v of view.values) {
      level += v;
      lowest = Math.min(lowest, level);
      highest = Math.max(highest, level);
    }
  }
  const span = highest - lowest;
  const lo = Math.floor((lowest - 0.06 * span) / 10) * 10;
  const hi = highest + 0.02 * span;

  // -- the layout: a head with the switch, the plot (bricks over the matrix of units), a readout -----

  const root = el("div", "lwf");
  const head = el("div", "lwf-head");
  const pick = el("div", "lwf-pick");
  const pickSymbol = el("span", "lwf-cal lwf-pick-symbol", "𝒮");
  pickSymbol.setAttribute("aria-hidden", "true");
  const unitSwitch = segmented(
    "The units 𝒮",
    Object.entries(VIEWS).map(([id, view]) => [id, view.label]),
    (id, byUser) => {
      if (byUser) show(id);
    },
  );
  // a short label where the figure is narrow (the pill holds a copy of the labels)
  unitSwitch.node.querySelectorAll(".shapiq-ill-segment").forEach((node) => {
    const view = Object.values(VIEWS).find((v) => v.label === node.textContent);
    if (!view?.short || view.short === view.label) return;
    node.replaceChildren(el("span", "lwf-long", view.label), el("span", "lwf-short", view.short));
  });
  pick.append(pickSymbol, el("span", "lwf-pick-equals", "="), unitSwitch.node);
  const method = el("p", "lwf-method");
  head.append(pick, method);

  const plot = el("div", "lwf-plot");
  const chart = svg("svg", { class: "lwf-svg", role: "group" });
  chart.setAttribute("tabindex", "0");
  const tag = el("div", "lwf-tag");
  tag.setAttribute("aria-hidden", "true");
  plot.append(chart, tag);
  const readout = el("p", "lwf-readout");
  readout.setAttribute("aria-live", "polite");
  const foot = el("p", "lwf-foot");
  foot.textContent =
    `A gradient boosting model on six features of the California housing data (R² ` +
    `${data.r2.toFixed(2)} on held-out houses) predicts ${price(data.shown.prediction)} for ` +
    `this house, against ${price(data.shown.baseline)} on average. Every explanation adds up to ` +
    `the difference.`;
  root.append(head, plot, readout, foot);
  stage.appendChild(root);

  // the layers of the plot, back to front
  const layer = (cls) => svg("g", { class: cls }, chart);
  const gridLayer = layer("lwf-grid");
  const bandLayer = layer("lwf-bands");
  const band = svg("rect", { class: "lwf-band", rx: 5 }, bandLayer);
  const guides = [0, 1].map(() => svg("line", { class: "lwf-guide" }, bandLayer));
  const refLayer = layer("lwf-refs");
  const brickLayer = layer("lwf-bricks");
  const dotLayer = layer("lwf-dots");
  const gutterLayer = layer("lwf-gutter");
  const projection = svg("rect", { class: "lwf-projection", width: 4, rx: 2 }, gutterLayer);

  // -- the sizes, which follow the figure's width ------------------------------------------------------

  let size = null;
  function measure() {
    const W = Math.max(260, Math.round(plot.clientWidth));
    const wide = W >= 540;
    const T = 46; // room above the bricks for the tag
    const P = wide ? 250 : 210; // the plot's height
    const M = wide ? 22 : 24; // from the plot to the matrix
    const rowH = wide ? 17 : 14;
    const top = T + P + M;
    return {
      W,
      wide,
      G: wide ? 108 : 48, // the left gutter: the axis, the matrix's rows
      R: wide ? 96 : 60, // the right gutter: the total
      T,
      P,
      rowH,
      matrixTop: top,
      H: top + n * rowH + 8,
      y: (v) => T + ((hi - v) / (hi - lo)) * P,
    };
  }

  // the geometry of each brick of a view: its column (x, w, the gap to the next), its levels (y0
  // from, y1 to, in $1,000s), and what it shows
  function layout(id) {
    const view = data.views[id];
    const { blocks } = VIEWS[id];
    const units = view.units;
    const m = units.length;
    const left = size.G + 4;
    const right = size.W - size.R - 6;
    const breaks = blocks ? units.filter((ids, k) => k && ids.length > units[k - 1].length).length : 0;
    const gapSlots = 0.9;
    const pitch = (right - left) / (m + breaks * gapSlots);
    const fill = m > 30 ? 0.76 : m > 10 ? 0.7 : 0.6;
    const w = Math.min(pitch * fill, 76);
    const labels = pitch >= 27 ? 1 : 0;
    const out = new Map();
    let x = left;
    let level = data.baseline;
    units.forEach((ids, k) => {
      if (blocks && k && ids.length > units[k - 1].length) x += gapSlots * pitch;
      const v = view.values[k];
      out.set(keyOf(id, ids), {
        ids,
        index: k,
        x: x + (pitch - w) / 2,
        w,
        pitch,
        y0: level,
        y1: level + v,
        shown: view.shown[k],
        labels,
        o: 1,
      });
      level += v;
      x += pitch;
    });
    // the gap to the next brick, for the connector at the brick's end level
    const list = [...out.values()];
    list.forEach((g, k) => {
      g.gap = k + 1 < list.length ? list[k + 1].x - g.x - g.w : right + 4 - g.x - g.w;
    });
    return out;
  }

  // -- the bricks: one per unit, kept while a unit stays, so that it moves (FLIP) ------------------------

  const bricks = new Map(); // key → { ids, g, rect, conn, label, dots, cur, from, to, delay, leaving }
  const FIELDS = ["x", "w", "pitch", "y0", "y1", "gap", "labels", "o", "shown"];
  const lerp = (a, b, t) => {
    const out = { ...b };
    for (const f of FIELDS) out[f] = a[f] + (b[f] - a[f]) * t;
    return out;
  };

  function makeBrick(key, ids) {
    const g = svg("g", { class: "lwf-brick" }, brickLayer);
    const conn = svg("line", { class: "lwf-conn" }, g);
    const rect = svg("rect", { class: "lwf-rect" }, g);
    const label = svg("text", { class: "lwf-value", "text-anchor": "middle" }, g);
    const dots = svg("g", { class: "lwf-unit" }, dotLayer);
    const stem = svg("line", { class: "lwf-stem" }, dots);
    const circles = features.map((_, k) =>
      svg("circle", { class: ids.includes(k) ? "lwf-dot is-in" : "lwf-dot" }, dots),
    );
    const brick = { key, ids, g, rect, conn, label, dots, stem, circles, cur: null, from: null, to: null };
    bricks.set(key, brick);
    return brick;
  }

  function paintBrick(brick) {
    const { cur, ids } = brick;
    const { y } = size;
    const top = y(Math.max(cur.y0, cur.y1));
    const bottom = y(Math.min(cur.y0, cur.y1));
    const up = cur.y1 >= cur.y0;
    const height = Math.max(1.2, bottom - top);
    const hot = brick.key === hotKey;
    brick.g.setAttribute("class", `lwf-brick ${up ? "lwf-pos" : "lwf-neg"}${hot ? " is-hot" : ""}`);
    brick.g.style.opacity = cur.o;
    brick.rect.setAttribute("x", cur.x.toFixed(2));
    brick.rect.setAttribute("width", Math.max(0.5, cur.w).toFixed(2));
    brick.rect.setAttribute("y", (bottom - top < 1.2 ? (top + bottom) / 2 - 0.6 : top).toFixed(2));
    brick.rect.setAttribute("height", height.toFixed(2));
    brick.rect.setAttribute("rx", Math.min(3, cur.w / 4).toFixed(2));
    const end = y(cur.y1);
    brick.conn.setAttribute("x1", (cur.x + cur.w).toFixed(2));
    brick.conn.setAttribute("x2", (cur.x + cur.w + Math.max(0, cur.gap)).toFixed(2));
    brick.conn.setAttribute("y1", end.toFixed(2));
    brick.conn.setAttribute("y2", end.toFixed(2));
    // its value above it (or below, where it goes down), where the column has room; the tag
    // replaces it on the brick it names
    const shown = Math.round(cur.shown);
    brick.label.textContent = cur.pitch >= 52 ? money(shown) : compact(shown);
    brick.label.setAttribute("x", (cur.x + cur.w / 2).toFixed(2));
    brick.label.setAttribute("y", (up ? top - 5 : bottom + 12).toFixed(2));
    labelOpacity(brick);
    // its unit in the matrix: a dot per feature, filled for the features of S
    const cx = cur.x + cur.w / 2;
    const rIn = Math.max(1.4, Math.min(4.6, cur.pitch * 0.3));
    const rOut = Math.max(0.8, Math.min(2, cur.pitch * 0.13));
    brick.dots.style.opacity = cur.o;
    brick.dots.classList.toggle("is-hot", hot);
    brick.circles.forEach((circle, k) => {
      circle.setAttribute("cx", cx.toFixed(2));
      circle.setAttribute("cy", rowY(k).toFixed(2));
      circle.setAttribute("r", (ids.includes(k) ? rIn : rOut).toFixed(2));
    });
    brick.stem.setAttribute("x1", cx.toFixed(2));
    brick.stem.setAttribute("x2", cx.toFixed(2));
    brick.stem.setAttribute("y1", rowY(Math.min(...ids)).toFixed(2));
    brick.stem.setAttribute("y2", rowY(Math.max(...ids)).toFixed(2));
    brick.stem.style.strokeWidth = Math.max(1, Math.min(2.4, cur.pitch * 0.12)).toFixed(2);
  }

  const rowY = (k) => size.matrixTop + (k + 0.5) * size.rowH;

  // -- the frame: the axis E = ℝ, the reference lines, the total, and the rows of the matrix -----------

  let refs = null;
  function drawFrame() {
    const { W, H, G, R, T, P, y, wide } = size;
    chart.setAttribute("viewBox", `0 0 ${W} ${H}`);
    chart.setAttribute("width", W);
    chart.setAttribute("height", H);
    gridLayer.replaceChildren();
    refLayer.replaceChildren();
    for (const node of [...gutterLayer.children]) if (node !== projection) node.remove();
    const text = (parent, x, yy, content, cls, anchor = "start") => {
      const node = svg("text", { x, y: yy, class: cls, "text-anchor": anchor }, parent);
      node.textContent = content;
      return node;
    };
    // the axis: E = ℝ, a brick is a stretch of it
    const axisX = G - 2;
    svg("line", { x1: axisX, x2: axisX, y1: T - 6, y2: T + P, class: "lwf-axis" }, gridLayer);
    for (let v = Math.ceil(lo / 100) * 100; v <= hi; v += 100) {
      svg("line", { x1: axisX, x2: W - R, y1: y(v), y2: y(v), class: "lwf-gridline" }, gridLayer);
      svg("line", { x1: axisX - 4, x2: axisX, y1: y(v), y2: y(v), class: "lwf-axis" }, gridLayer);
      text(gridLayer, axisX - 7, y(v) + 4, `$${v}k`, "lwf-tick", "end");
    }
    chart.classList.toggle("lwf-narrow", !wide);
    const axisName = text(gutterLayer, wide ? 4 : 0, T - 24, "", "lwf-name");
    axisName.innerHTML = '<tspan class="lwf-math">E</tspan> = ℝ';
    if (wide) text(gutterLayer, 4, T - 9, "explanation space", "lwf-name-note");
    // the average and this house: prices, amber, and the total between them, which never moves
    const yAvg = y(data.baseline);
    const yPred = y(data.prediction);
    svg("line", { x1: axisX, x2: W - R + 8, y1: yAvg, y2: yAvg, class: "lwf-ref" }, refLayer);
    svg("line", { x1: axisX, x2: W - R + 8, y1: yPred, y2: yPred, class: "lwf-ref" }, refLayer);
    const bx = W - R + (wide ? 14 : 8);
    const total = svg("g", { class: "lwf-total" }, gutterLayer);
    svg("path", { d: `M${bx - 4} ${yPred}H${bx}V${yAvg}H${bx - 4}`, class: "lwf-bracket" }, total);
    const tx = bx + (wide ? 7 : 5);
    const nx = wide ? tx - 7 : bx - 6; // the prices and their notes
    const mid = (yAvg + yPred) / 2;
    const totalValue = text(total, tx, mid - 2, money(data.shown.total), "lwf-total-value lwf-pos-ink");
    const sum = text(total, tx, mid + 14, "", "lwf-total-note");
    sum.innerHTML = 'Σ <tspan class="lwf-math">e</tspan>(<tspan class="lwf-math">S</tspan>)';
    text(gutterLayer, nx, yPred - 21, "this house", "lwf-ref-note");
    text(gutterLayer, nx, yPred - 7, price(data.shown.prediction), "lwf-ref-value");
    text(gutterLayer, nx, yAvg + 16, price(data.shown.baseline), "lwf-ref-value");
    text(gutterLayer, nx, yAvg + 30, "average", "lwf-ref-note");
    refs = { total: totalValue };
    // the rows of the matrix: one per feature, numbered as everywhere, named where there is room
    features.forEach((feature, k) => {
      const yy = rowY(k);
      svg("line", { x1: G + 2, x2: W - R - 4, y1: yy, y2: yy, class: "lwf-row" }, gridLayer);
      const badge = svg("g", { class: "lwf-badge", transform: `translate(${wide ? 14 : G - 16} ${yy})` }, gutterLayer);
      svg("circle", { r: Math.min(8, size.rowH / 2 - 0.5) }, badge);
      svg("text", { y: 0.5 }, badge).textContent = String(k + 1);
      if (wide) text(gutterLayer, 28, yy + 4, feature.short, "lwf-row-name");
    });
    const unitsName = text(gutterLayer, wide ? 4 : G - 16, size.matrixTop - 7, "", "lwf-name", wide ? "start" : "middle");
    unitsName.innerHTML = '<tspan class="lwf-cal">𝒮</tspan>';
    if (wide) text(gutterLayer, 22, size.matrixTop - 7, "the units", "lwf-name-note");
  }

  // -- the unit in focus: its band, its stretch of the axis, its tag e(S), the readout -------------------

  let hotKey = null;
  let tagShown = false;
  let tagBox = null; // where the tag is: the values it covers fade
  function labelOpacity(brick) {
    const { cur, label } = brick;
    let o = cur.labels;
    if (tagShown && brick.key === hotKey) o = 0;
    else if (tagShown && tagBox && o > 0) {
      const x = cur.x + cur.w / 2;
      const half = label.textContent.length * 3.4 + 2;
      const y = Number(label.getAttribute("y"));
      const apart = x + half < tagBox.left || x - half > tagBox.right || y < tagBox.top || y - 11 > tagBox.bottom;
      if (!apart) o = 0;
    }
    label.style.opacity = o;
  }
  function paintFocus() {
    const brick = bricks.get(hotKey);
    if (!brick || !brick.cur) {
      band.style.opacity = 0;
      projection.style.opacity = 0;
      for (const guide of guides) guide.style.opacity = 0;
      tag.style.opacity = 0;
      return;
    }
    const { cur } = brick;
    const { y, H } = size;
    const top = y(Math.max(cur.y0, cur.y1));
    const bottom = y(Math.min(cur.y0, cur.y1));
    const bandWidth = Math.max(1, Math.min(cur.pitch, cur.w + 14));
    band.setAttribute("x", (cur.x + cur.w / 2 - bandWidth / 2).toFixed(2));
    band.setAttribute("width", bandWidth.toFixed(2));
    band.setAttribute("y", (top - 5).toFixed(2));
    band.setAttribute("height", Math.max(0, H - top + 3).toFixed(2));
    // its stretch of the axis E = ℝ: the brick's ends, carried over to the axis
    for (const [guide, level] of [[guides[0], top], [guides[1], bottom]]) {
      guide.setAttribute("x1", size.G - 2);
      guide.setAttribute("x2", cur.x.toFixed(2));
      guide.setAttribute("y1", level.toFixed(2));
      guide.setAttribute("y2", level.toFixed(2));
      guide.style.opacity = cur.o * (tagShown ? 1 : 0);
    }
    band.style.opacity = cur.o * (tagShown ? 1 : 0);
    projection.setAttribute("x", size.G - 4);
    projection.setAttribute("y", top.toFixed(2));
    projection.setAttribute("height", Math.max(1.5, bottom - top).toFixed(2));
    projection.setAttribute("class", `lwf-projection ${cur.y1 >= cur.y0 ? "lwf-pos" : "lwf-neg"}`);
    projection.style.opacity = cur.o * (tagShown ? 1 : 0);
    // the tag above the brick, inside the figure, its tip on the brick
    const shown = Math.round(cur.shown);
    const text = `<span class="lwf-math">e</span>(${setOf(brick.ids)}) = <b class="${shown >= 0 ? "lwf-pos-ink" : "lwf-neg-ink"}">${money(shown)}</b>`;
    if (tag.dataset.text !== text) {
      tag.innerHTML = text;
      tag.dataset.text = text;
    }
    const width = tag.offsetWidth;
    const center = cur.x + cur.w / 2;
    // within the plot: not over the axis, nor over the prices on the right
    const left = Math.max(size.G - 6, Math.min(size.W - size.R + (size.wide ? 8 : -4) - width, center - width / 2));
    const tagTop = top - tag.offsetHeight - 9;
    tag.style.transform = `translate(${left.toFixed(1)}px, ${tagTop.toFixed(1)}px)`;
    tagBox = { left: left - 3, right: left + width + 3, top: tagTop - 2, bottom: tagTop + tag.offsetHeight + 6 };
    for (const other of bricks.values()) if (other.cur) labelOpacity(other);
    tag.style.setProperty("--lwf-tip", `${(center - left).toFixed(1)}px`);
    tag.style.opacity = cur.o * (tagShown ? 1 : 0);
  }

  function describe() {
    const brick = bricks.get(hotKey);
    if (!brick || !brick.to) return;
    const { ids, to } = brick;
    const what =
      ids.length === 1
        ? features[ids[0]].label
        : state.view === "groups"
          ? `${namesOf(ids)}: the location`
          : namesOf(ids);
    // the definition's S ↦ e(S), for the unit in focus
    readout.innerHTML =
      `<span class="lwf-read-unit"><span class="lwf-math">S</span> = ${setOf(ids)} ` +
      `<span class="lwf-read-names">${what}</span></span>` +
      `<span class="lwf-read-value"><span class="lwf-read-to">↦</span> <span class="lwf-math">e</span>(<span class="lwf-math">S</span>) = ` +
      `<b class="${to.shown >= 0 ? "lwf-pos-ink" : "lwf-neg-ink"}">${money(to.shown)}</b></span>`;
  }

  function focus(key) {
    if (!bricks.get(key)?.to || bricks.get(key).leaving) return;
    if (key === hotKey) return;
    hotKey = key;
    for (const brick of bricks.values()) if (brick.cur) paintBrick(brick);
    paintFocus();
    describe();
  }

  // the brick under the pointer: the nearest column
  const order = () => [...bricks.values()].filter((b) => !b.leaving && b.to).sort((a, b) => a.to.index - b.to.index);
  function pointAt(event) {
    const box = chart.getBoundingClientRect();
    const x = ((event.clientX - box.left) / box.width) * size.W;
    let best = null;
    for (const brick of order()) {
      const d = Math.abs(brick.to.x + brick.to.w / 2 - x);
      if (!best || d < best[0]) best = [d, brick.key];
    }
    if (best && best[0] <= Math.max(14, bricks.get(best[1]).to.pitch)) focus(best[1]);
  }
  chart.addEventListener("pointermove", pointAt);
  chart.addEventListener("pointerdown", pointAt);
  chart.addEventListener("keydown", (event) => {
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
    const list = order();
    if (!step && event.key !== "Home" && event.key !== "End") return;
    event.preventDefault();
    const at = list.findIndex((b) => b.key === hotKey);
    const next = event.key === "Home" ? 0 : event.key === "End" ? list.length - 1 : (at + step + list.length) % list.length;
    focus(list[next].key);
  });

  // -- switching the units: every brick goes from where it is to where it belongs ---------------------

  const state = { view: "features" };
  let stopTween = null;
  let seen = false;

  // the rectangle of several bricks together, as one brick: from the first's start to the last's end
  const joined = (geoms) => {
    const sorted = [...geoms].sort((a, b) => a.x - b.x);
    const first = sorted[0];
    const last = sorted.at(-1);
    return {
      ...last,
      x: first.x,
      w: last.x + last.w - first.x,
      y0: first.y0,
      y1: last.y1,
      shown: geoms.reduce((s, g) => s + g.shown, 0),
    };
  };
  // the k-th of m equal pieces of a brick, in its order
  const piece = (geom, k, m) => ({
    ...geom,
    x: geom.x + (k * geom.w) / m,
    w: geom.w / m,
    y0: geom.y0 + ((geom.y1 - geom.y0) * k) / m,
    y1: geom.y0 + ((geom.y1 - geom.y0) * (k + 1)) / m,
    shown: geom.shown / m,
  });
  const within = (a, b) => a.every((k) => b.includes(k)); // a ⊆ b

  function show(id, { animate = true } = {}) {
    stopTween?.();
    if (seen) tagShown = true; // also where a switch cuts the first climb short
    const before = state.view;
    state.view = id;
    const next = layout(id);
    const changed = before !== id;
    const moving = animate && seen && changed && !reducedMotion();
    const old = [...bricks.values()].filter((b) => b.cur && !b.leaving);

    // the bricks that come: from the bricks they merge (their union) or split from (a piece)
    for (const [key, to] of next) {
      let brick = bricks.get(key);
      if (brick?.cur) {
        brick.from = { ...brick.cur };
      } else {
        brick ??= makeBrick(key, to.ids);
        const parts = old.filter((b) => b.ids.length < to.ids.length && within(b.ids, to.ids));
        const whole = old.find((b) => b.ids.length > to.ids.length && within(to.ids, b.ids));
        const covered = parts.flatMap((b) => b.ids).length === to.ids.length;
        if (parts.length > 1 && covered && id === "groups") {
          brick.from = { ...joined(parts.map((b) => b.cur)), o: 0.2 };
        } else if (whole && before === "groups") {
          const siblings = [...next.values()].filter((g) => g.ids.length < whole.ids.length && within(g.ids, whole.ids));
          brick.from = { ...piece(whole.cur, siblings.indexOf(to), siblings.length), o: 0.2 };
        } else {
          // a new kind of unit: it grows from nothing at its place
          brick.from = { ...to, y1: to.y0, shown: 0, o: 0, labels: 0 };
        }
      }
      brick.to = { ...to };
      brick.leaving = false;
    }
    // the bricks that go: into the brick they merge into, or into the pieces they split into
    for (const brick of bricks.values()) {
      if (next.has(brick.key) || !brick.cur) continue;
      const into = [...next.values()].find((g) => g.ids.length > brick.ids.length && within(brick.ids, g.ids));
      const pieces = [...next.values()].filter((g) => g.ids.length < brick.ids.length && within(g.ids, brick.ids));
      brick.from = { ...brick.cur };
      if (into && id === "groups") {
        const siblings = old
          .filter((b) => b.ids.length < into.ids.length && within(b.ids, into.ids))
          .sort((a, b) => a.cur.x - b.cur.x);
        brick.to = { ...piece(into, Math.max(0, siblings.indexOf(brick)), siblings.length), o: 0, labels: 0 };
      } else if (pieces.length > 1 && before === "groups") {
        brick.to = { ...joined(pieces), o: 0, labels: 0 };
      } else {
        brick.to = { ...brick.cur, y1: brick.cur.y0, shown: 0, o: 0, labels: 0 };
      }
      brick.leaving = true;
    }

    // the unit in focus: the one this view adds (the location, the largest pair, the largest of more)
    const view = data.views[id];
    const featured = VIEWS[id].featured(view.units, view.values);
    hotKey = keyOf(id, featured);
    const m = view.units.length;
    method.innerHTML =
      `<b>${VIEWS[id].method}</b><span class="lwf-method-rest"> · ${m} unit${m > 1 ? "s" : ""}</span>`;
    chart.setAttribute(
      "aria-label",
      `${VIEWS[id].method}: ${m} bricks from the average ${price(data.shown.baseline)} to this ` +
        `house's ${price(data.shown.prediction)}, together ${money(data.shown.total)}. The arrow keys ` +
        "step through the bricks.",
    );

    const settle = () => {
      for (const brick of [...bricks.values()]) {
        if (brick.leaving) {
          brick.g.remove();
          brick.dots.remove();
          bricks.delete(brick.key);
        } else {
          brick.cur = { ...brick.to };
          paintBrick(brick);
        }
      }
      paintFocus();
      describe();
    };
    if (!moving) {
      for (const brick of bricks.values()) brick.cur = { ...brick.to };
      settle();
      return;
    }
    // each brick on its own clock, a little after the one before it
    const list = [...bricks.values()];
    const stagger = Math.min(40, 520 / Math.max(1, m));
    for (const brick of list) {
      brick.delay = (brick.to.index ?? brick.from.index ?? 0) * stagger;
      if (brick.leaving) brick.delay = 0;
    }
    const duration = 620;
    const total = duration + Math.max(...list.map((b) => b.delay));
    describe();
    stopTween = tween(
      total,
      (p) => {
        const now = p * total;
        for (const brick of list) {
          const t = ease.inOut(Math.min(1, Math.max(0, (now - brick.delay) / duration)));
          brick.cur = lerp(brick.from, brick.to, t);
          paintBrick(brick);
        }
        paintFocus();
      },
      { easing: (t) => t, done: settle },
    );
    pulse(refs.total);
  }

  // the total glows once after a switch: it stays the same
  function pulse(node) {
    if (reducedMotion() || !node) return;
    node.animate([{ transform: "scale(1)" }, { transform: "scale(1.18)", offset: 0.3 }, { transform: "scale(1)" }], {
      duration: 700,
      delay: 250,
      easing: "ease-in-out",
    });
  }

  // -- the first time the figure is seen: the bricks climb from the average to this house ----------------

  function build() {
    const list = order();
    const m = list.length;
    if (reducedMotion()) {
      tagShown = true;
      for (const brick of list) paintBrick(brick);
      paintFocus();
      return;
    }
    const grow = 420;
    const stagger = 150;
    const total = grow + stagger * (m - 1) + 260;
    for (const brick of list) {
      brick.from = { ...brick.to, y1: brick.to.y0, shown: 0, labels: 0 };
      brick.cur = { ...brick.from };
      paintBrick(brick);
    }
    stopTween = tween(
      total,
      (p) => {
        const now = p * total;
        list.forEach((brick, k) => {
          const t = Math.min(1, Math.max(0, (now - k * stagger) / grow));
          brick.cur = lerp(brick.from, brick.to, ease.out(t));
          paintBrick(brick);
        });
      },
      {
        easing: (t) => t,
        done: () => {
          tagShown = true;
          for (const brick of list) paintBrick(brick);
          paintFocus();
          tag.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300 });
          pulse(refs.total);
        },
      },
    );
  }

  // -- start, and redraw when the width changes ------------------------------------------------------

  let lastWidth = 0;
  function redraw() {
    stopTween?.();
    if (seen) tagShown = true;
    size = measure();
    lastWidth = size.W;
    drawFrame();
    const next = layout(state.view);
    for (const brick of [...bricks.values()]) {
      if (!next.has(brick.key)) {
        brick.g.remove();
        brick.dots.remove();
        bricks.delete(brick.key);
      }
    }
    for (const [key, to] of next) {
      const brick = bricks.get(key) ?? makeBrick(key, to.ids);
      brick.to = { ...to };
      brick.cur = { ...to };
      brick.leaving = false;
      paintBrick(brick);
    }
    paintFocus();
  }

  unitSwitch.set("features");
  size = measure();
  drawFrame();
  show("features", { animate: false });
  redraw();
  // until it is seen, the bricks wait at the average
  for (const brick of bricks.values()) {
    brick.cur = { ...brick.to, y1: brick.to.y0, shown: 0, labels: 0 };
    paintBrick(brick);
  }
  paintFocus();
  new ResizeObserver(() => {
    if (Math.abs(plot.clientWidth - lastWidth) > 2) {
      redraw();
      if (!seen) {
        for (const brick of bricks.values()) {
          brick.cur = { ...brick.to, y1: brick.to.y0, shown: 0, labels: 0 };
          paintBrick(brick);
        }
      }
    }
  }).observe(plot);
  const stopWatching = watchVisibility(stage, (visible) => {
    if (visible && !seen) {
      seen = true;
      build();
      stopWatching?.();
    }
  });
});
