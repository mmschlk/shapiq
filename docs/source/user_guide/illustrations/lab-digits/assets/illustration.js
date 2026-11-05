// The illustration "lab-digits" (chapter 2.1 of the user guide): Definition 2.1, e : 𝒮 → E, on an
// image model. A handwritten 7 of scikit-learn's digits, 8 × 8 pixels, is cut into 16 patches of
// 2 × 2 pixels, the features 1, …, 16; a neural network (an MLP) reads it and says 7. On the left,
// the units 𝒮 cut the image (each patch alone, the four rows of patches, or pairs of patches); on the
// right, the explanation space E (a line of numbers, or a frame of curves); between them, the
// explanation e takes each unit S to its explanation e(S), painted on the image and placed in E.
//
// - patches: their Shapley values; their ICE curves (the probability as the ink of the patch goes
//   from blank to black);
// - rows, each row of four patches one unit: the Shapley values of the rows, each row one player;
//   their ICE curves;
// - pairs: the pairwise Shapley interactions (k-SII), the strongest drawn as arcs; for those, the
//   pure interaction surfaces (what the ink of both does beyond what each does alone).
//
// A patch is removed by erasing it (its pixels set to 0). When the illustration is first seen, the
// parts come one by one: the digit, the units that cut it, the space, and the explanation, whose
// values fly from their units into E. Values are computed once by ../compute/compute.py into
// data.js; red is positive and blue negative, amber marks the features (docs/STYLE_GUIDE.md).
window.shapiqIllustrations?.define("lab-digits", async (stage, kit) => {
  const { el, svg, tween, ease, reducedMotion, segmented, loadScript, watchVisibility } = kit;
  // the data that the page loaded before this script, or, where it is missing, loaded here
  if (!window.SHAPIQ_LAB_DIGITS) await loadScript(`${kit.root}data.js?v=${kit.version}`);
  const data = window.SHAPIQ_LAB_DIGITS;
  const GRID = 4; // patches per side
  const N = GRID * GRID;
  const CLASS = String(data.target);

  // -- numbers: probabilities in percent, values in percentage points, to one decimal ----------------

  const tenths = (p) => Math.round(p * 1000); // a probability in tenths of a percent
  const percent = (p) => `${(tenths(p) / 10).toFixed(1)}%`;
  const points = (t) => `${t > 0 ? "+" : t < 0 ? "−" : ""}${(Math.abs(t) / 10).toFixed(1)}`;
  const sign = (v) => (v > 0 ? "pos" : v < 0 ? "neg" : "zero");
  // the prediction minus the erased digit, as the difference of the shown probabilities
  const total = tenths(data.prediction) - tenths(data.empty);
  // values in tenths that add up to `sum`: each rounded down, the rest given to those that lost most
  const roundTo = (values, sum) => {
    const raw = values.map((v) => v * 1000);
    const shown = raw.map(Math.floor);
    const order = raw.map((v, k) => [v - Math.floor(v), k]).sort((a, b) => b[0] - a[0]);
    let rest = sum - shown.reduce((a, b) => a + b, 0);
    for (let j = 0; rest > 0; j++, rest--) shown[order[j % order.length][1]] += 1;
    for (let j = order.length - 1; rest < 0; j--, rest++) shown[order[(j + order.length) % order.length][1]] -= 1;
    return shown;
  };

  // -- the units 𝒮, the spaces E, and the explanations they give -------------------------------------

  const keyOf = (ids) => [...ids].sort((a, b) => a - b).join("-");
  const setOf = (ids) => `{${ids.map((k) => k + 1).join(", ")}}`;
  const allPairs = [];
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) allPairs.push([i, j]);
  const pairValues = allPairs.map((ids) => data.pairs[keyOf(ids)]);
  const strongest = Object.keys(data.surfaces); // the pairs that get a surface, strongest first
  const UNITS = {
    patches: {
      label: "Patches",
      sets: Array.from({ length: N }, (_, k) => [k]),
      values: data.shapley,
      shown: roundTo(data.shapley, total),
      curves: data.curves,
      notation: "{{1}, {2}, …, {16}}",
      gloss: "each patch alone",
      what: (ids) => `patch ${ids[0] + 1}`,
    },
    rows: {
      label: "Rows",
      sets: data.groups,
      values: data.rows,
      shown: roundTo(data.rows, total),
      curves: data.row_curves,
      notation: "{{1, …, 4}, …, {13, …, 16}}",
      gloss: "each row of four patches as one",
      what: (ids) => `row ${Math.floor(ids[0] / GRID) + 1}`,
    },
    pairs: {
      label: "Pairs",
      sets: allPairs,
      values: pairValues,
      shown: pairValues.map(tenths),
      notation: "{{<i>i</i>, <i>j</i>} : <i>i</i> &lt; <i>j</i>}",
      gloss: `every pair of patches, ${allPairs.length} units`,
      what: (ids) => `patches ${ids[0] + 1} and ${ids[1] + 1}`,
    },
  };
  const SPACES = {
    numbers: { label: "Numbers", notation: "ℝ", gloss: "a number per unit, in points of probability" },
    curves: { label: "Curves", notation: "curves", gloss: "a curve per unit: the probability as its ink goes from blank to black" },
  };
  const SURFACE_GLOSS = "a surface per pair: what the ink of both does beyond each alone";
  const METHODS = {
    "patches-numbers": "Shapley values",
    "patches-curves": "ICE curves",
    "rows-numbers": "Shapley values of the rows",
    "rows-curves": "ICE curves of the rows",
    "pairs-numbers": "Shapley interactions",
    "pairs-curves": "interaction surfaces",
  };
  const state = { units: "patches", space: "numbers" };
  // the units a view shows: in "pairs", only the strongest pairs, which have surfaces, as arcs
  const isSurface = () => state.units === "pairs" && state.space === "curves";

  // -- the layout: the units on the image, the explanation e, and the space E ----------------------

  const root = el("div", "dig");
  const part = (cls, symbol, name) => {
    const node = el("div", `dig-part ${cls}`);
    const head = el("p", "dig-part-head");
    head.innerHTML = `<span class="dig-symbol">${symbol}</span><span class="dig-part-name">${name}</span>`;
    node.appendChild(head);
    return node;
  };
  const unitsPart = part("dig-part-units", "𝒮", "the units");
  const spacePart = part("dig-part-space", "<i>E</i>", "the explanation space");
  const choose = (key, id, byUser) => {
    if (!byUser) return;
    state[key] = id;
    pointed = null;
    render({ changed: key });
  };
  const unitsSwitch = segmented("Units", Object.entries(UNITS).map(([id, u]) => [id, u.label]), (id, byUser) =>
    choose("units", id, byUser),
  );
  const spaceSwitch = segmented("Explanation space", Object.entries(SPACES).map(([id, s]) => [id, s.label]), (id, byUser) =>
    choose("space", id, byUser),
  );
  const unitsNotation = el("p", "dig-notation");
  const spaceNotation = el("p", "dig-notation");
  unitsPart.append(unitsSwitch.node, unitsNotation);
  spacePart.append(spaceSwitch.node, spaceNotation);

  // the image: the digit's pixels, cut into its units, and the explanation painted on them
  const mapBox = el("div", "dig-map-box");
  const map = svg("svg", { viewBox: "0 0 400 400", class: "dig-map", role: "img" });
  map.setAttribute("aria-label", `A handwritten ${CLASS}, 8 × 8 pixels, cut into 16 patches numbered 1 to 16`);
  mapBox.appendChild(map);
  unitsPart.appendChild(mapBox);

  // the explanation e between them: its symbol on an arrow, and the method it is
  const arrow = el("div", "dig-arrow");
  arrow.innerHTML =
    '<span class="dig-symbol"><i>e</i></span><span class="dig-arrow-line" aria-hidden="true"></span>' +
    '<span class="dig-part-name">the explanation</span><span class="dig-method"></span>';
  const method = arrow.querySelector(".dig-method");

  // the space E: a line of numbers, or a frame of curves
  const spaceBox = el("div", "dig-space-box");
  const space = svg("svg", { class: "dig-space", role: "img" });
  spaceBox.appendChild(space);
  const spaceGloss = el("p", "dig-space-gloss");
  spacePart.append(spaceBox, spaceGloss);

  const body = el("div", "dig-body");
  const fly = el("div", "dig-fly");
  fly.setAttribute("aria-hidden", "true");
  body.append(unitsPart, arrow, spacePart, fly);

  const readout = el("p", "dig-readout");
  readout.setAttribute("aria-live", "polite");
  const equation = el("p", "dig-equation");
  const foot = el("p", "dig-foot");
  foot.innerHTML =
    `A neural network (an MLP, ${(data.accuracy * 100).toFixed(1)}% right on ${data.test} held-out digits) ` +
    `says this handwritten digit is a <strong>${CLASS}</strong>, with probability ${percent(data.prediction)}. ` +
    `To remove a patch, we <strong>erase</strong> it, its pixels blank; with all erased, the probability of ` +
    `${CLASS} is ${percent(data.empty)}.`;
  root.append(body, readout, equation, foot);
  stage.appendChild(root);

  // -- the image -----------------------------------------------------------------------------------------

  const P = 100; // a patch in the image's units
  const PX = P / 2; // a pixel
  const corner = (k) => [(k % GRID) * P, Math.floor(k / GRID) * P];
  const center = (k) => corner(k).map((v) => v + P / 2);
  const defs = svg("defs", {}, map);
  const clipId = `dig-clip-${Math.random().toString(36).slice(2, 8)}`;
  const clip = svg("clipPath", { id: clipId }, defs);
  svg("rect", { width: 400, height: 400, rx: 14 }, clip);
  const picture = svg("g", { "clip-path": `url(#${clipId})` }, map);
  svg("rect", { width: 400, height: 400, class: "dig-paper" }, picture);
  const fills = Array.from({ length: N }, (_, k) => {
    const [x, y] = corner(k);
    return svg("rect", { x, y, width: P, height: P, class: "dig-fill" }, picture);
  });
  // the digit: its pixels, as dark as their ink (--a, from 0 to 1); the explanation colors them
  const inkLayer = svg("g", { class: "dig-ink" }, picture);
  const pixels = [];
  data.pixels.forEach((ink, p) => {
    if (!ink) return;
    const [r, c] = [Math.floor(p / 8), p % 8];
    const pixel = svg("rect", { x: c * PX, y: r * PX, width: PX, height: PX, rx: 5 }, inkLayer);
    pixel.style.setProperty("--a", (ink / 16).toFixed(3));
    pixels.push({ node: pixel, patch: Math.floor(r / 2) * GRID + Math.floor(c / 2) });
  });
  // the cuts between the patches: wide between units, a hairline inside one
  const cuts = [];
  for (let k = 0; k < N; k++) {
    const [x, y] = corner(k);
    if (k % GRID < GRID - 1) cuts.push({ a: k, b: k + 1, line: [x + P, y, x + P, y + P] });
    if (k < N - GRID) cuts.push({ a: k, b: k + GRID, line: [x, y + P, x + P, y + P] });
  }
  const cutLayer = svg("g", { class: "dig-cuts" }, picture);
  for (const cut of cuts) {
    const [x1, y1, x2, y2] = cut.line;
    cut.node = svg("line", { x1, y1, x2, y2 }, cutLayer);
  }
  const content = svg("g", { class: "dig-content" }, picture);
  const arcs = svg("g", { class: "dig-arcs" }, map);
  svg("rect", { x: 0.5, y: 0.5, width: 399, height: 399, rx: 14, class: "dig-frame" }, map);
  const badges = Array.from({ length: N }, (_, k) => {
    const [x, y] = corner(k);
    const group = svg("g", { class: "dig-badge", transform: `translate(${x + 17} ${y + 17})` }, map);
    svg("circle", { r: 12 }, group);
    svg("text", {}, group).textContent = String(k + 1);
    return group;
  });
  const focusRing = svg("path", { class: "dig-focus" }, map);
  const hits = svg("g", { class: "dig-hits" }, map);

  // the outline of a set of patches (a patch or a row: a rectangle)
  const outline = (ids, inset = 4) => {
    const xs = ids.map((k) => corner(k)[0]);
    const ys = ids.map((k) => corner(k)[1]);
    const [x0, y0] = [Math.min(...xs) + inset, Math.min(...ys) + inset];
    const [x1, y1] = [Math.max(...xs) + P - inset, Math.max(...ys) + P - inset];
    const r = 10;
    return `M${x0 + r} ${y0}H${x1 - r}Q${x1} ${y0} ${x1} ${y0 + r}V${y1 - r}Q${x1} ${y1} ${x1 - r} ${y1}H${x0 + r}Q${x0} ${y1} ${x0} ${y1 - r}V${y0 + r}Q${x0} ${y0} ${x0 + r} ${y0}Z`;
  };
  // an arc between the centers of two patches, bowed to one side
  const arcPath = ([i, j], bow = 0.3) => {
    const [x1, y1] = center(i);
    const [x2, y2] = center(j);
    const [mx, my] = [(x1 + x2) / 2, (y1 + y2) / 2];
    const [dx, dy] = [x2 - x1, y2 - y1];
    const [cx, cy] = [mx - dy * bow, my + dx * bow];
    return { d: `M${x1} ${y1}Q${cx} ${cy} ${x2} ${y2}`, mid: [(x1 + 2 * cx + x2) / 4, (y1 + 2 * cy + y2) / 4] };
  };

  // -- painting -----------------------------------------------------------------------------------------

  const fillOf = (v, reach, strength = 1) => {
    if (!v) return "var(--shapiq-surface)";
    const share = Math.round((10 + 62 * Math.min(1, Math.abs(v) / reach)) * strength);
    return `color-mix(in oklab, var(${v > 0 ? "--shapiq-red" : "--shapiq-blue"}) ${share}%, var(--shapiq-surface))`;
  };
  let stops = [];
  let entering = false;
  const animate = (duration, update, options = {}) => {
    const still = !entering || reducedMotion();
    stops.push(tween(still ? 0 : duration, update, still ? {} : options));
  };
  const text = (parent, x, y, words, cls, anchor = "middle") => {
    const node = svg("text", { x, y, class: cls, "text-anchor": anchor }, parent);
    node.textContent = words;
    return node;
  };

  // the units of the view: [ids, value, shown] for each
  const unitsOf = () => {
    const units = UNITS[state.units];
    return units.sets.map((ids, k) => ({ ids, value: units.values[k], shown: units.shown[k], index: k }));
  };
  // which unit a patch belongs to, where the units cut the image (patches, rows)
  const unitOfPatch = (k) =>
    state.units === "rows" ? Math.floor(k / GRID) : state.units === "patches" ? k : null;

  // a small curve in a box: the probability (0 to 100%) as the ink goes from 0 to 16
  const curvePath = (ys, x0, y0, w, h) =>
    ys
      .map((p, t) => `${t ? "L" : "M"}${(x0 + (t / (ys.length - 1)) * w).toFixed(1)} ${(y0 + (1 - p) * h).toFixed(1)}`)
      .join("");

  // the strongest pairs, drawn as arcs: the bow alternates so that arcs of one patch part
  const drawnPairs = () => strongest.map((key) => key.split("-").map(Number));

  function paintMap() {
    content.replaceChildren();
    arcs.replaceChildren();
    hits.replaceChildren();
    const units = unitsOf();
    const numbers = state.space === "numbers";
    const reach = Math.max(...units.map((u) => Math.abs(u.value)));
    const cutting = state.units !== "pairs";
    // the cuts: wide between two units, a hairline inside a unit
    for (const cut of cuts) {
      const same = cutting && unitOfPatch(cut.a) === unitOfPatch(cut.b);
      cut.node.classList.toggle("is-inside", same);
    }
    root.classList.toggle("dig-pairs", !cutting);
    root.classList.toggle("dig-numbers", numbers);
    // in numbers, each patch and its ink take the color of its unit's value: the paper a tint, the
    // ink the color itself (gray where the value is small); in curves and for pairs, the ink is gray
    const painted = numbers && cutting;
    root.classList.toggle("dig-quiet", !painted);
    fills.forEach((rect, k) => {
      const unit = unitOfPatch(k);
      rect.style.fill = painted ? fillOf(units[unit].value, reach, 0.3) : "var(--shapiq-surface)";
    });
    for (const { node, patch } of pixels) {
      const v = painted ? units[unitOfPatch(patch)].value : 0;
      node.style.fill = painted && v
        ? `color-mix(in oklab, var(${v > 0 ? "--shapiq-red" : "--shapiq-blue"}) ${Math.round(30 + 70 * Math.min(1, Math.abs(v) / reach))}%, var(--dig-ink))`
        : "var(--dig-ink)";
    }
    if (cutting) {
      units.forEach((unit, u) => {
        const [x, y] = corner(unit.ids[0]);
        const width = unit.ids.length === 1 ? P : 400;
        const group = svg("g", { class: "dig-unit" }, content);
        if (numbers) {
          // the value on a pill of paper, in the middle of the unit
          const label = svg("g", { class: `dig-label dig-${sign(unit.shown)}`, transform: `translate(${x + width / 2} ${y + 58})` }, group);
          const words = points(unit.shown);
          const half = 9 + 5.6 * words.length;
          svg("rect", { x: -half, y: -15, width: 2 * half, height: 30, rx: 15 }, label);
          text(label, 0, 6.5, words, "dig-value");
          animate(500, (p) => (label.style.opacity = p), { delay: 140 + 40 * u });
        } else {
          const ys = UNITS[state.units].curves[u];
          const box = unit.ids.length === 1 ? [x + 12, y + 34, P - 24, P - 46] : [x + 44, y + 22, 400 - 88, P - 34];
          svg("rect", { x: box[0], y: box[1], width: box[2], height: box[3], class: "dig-curve-box" }, group);
          svg("line", { x1: box[0], x2: box[0] + box[2], y1: box[1] + box[3], y2: box[1] + box[3], class: "dig-curve-base" }, group);
          const line = svg("path", { d: curvePath(ys, ...box), class: "dig-curve" }, group);
          if (entering && !reducedMotion()) {
            const length = line.getTotalLength();
            line.style.strokeDasharray = `${length}`;
            line.animate([{ strokeDashoffset: length }, { strokeDashoffset: 0 }], {
              duration: 700,
              delay: 120 + 40 * u,
              easing: "cubic-bezier(0.4, 0, 0.2, 1)",
              fill: "backwards",
            });
          }
        }
        const hit = svg("path", { d: outline(unit.ids, 0), class: "dig-hit" }, hits);
        hit.addEventListener("pointerenter", () => point(u));
        hit.addEventListener("pointerdown", () => pin(u));
      });
    } else {
      // pairs: the strongest as arcs between their patches; in numbers, red or blue and as wide as
      // their value, with the value on the one pointed at; in curves, amber, their surfaces in E
      const drawn = drawnPairs();
      drawn.forEach((ids, a) => {
        const u = allPairs.findIndex((pair) => keyOf(pair) === keyOf(ids));
        const unit = units[u];
        const { d, mid } = arcPath(ids, a % 2 ? -0.3 : 0.3);
        const group = svg("g", { class: "dig-pair" }, arcs);
        group.dataset.unit = u;
        const width = numbers ? 3 + 9 * (Math.abs(unit.value) / reach) : 4;
        const path = svg("path", { d, class: `dig-arc dig-${numbers ? sign(unit.value) : "curve"}`, "stroke-width": width }, group);
        for (const k of ids) svg("circle", { cx: center(k)[0], cy: center(k)[1], r: 6, class: "dig-arc-end" }, group);
        if (numbers) {
          const label = svg("g", { class: "dig-pill", transform: `translate(${mid[0]} ${mid[1]})` }, group);
          svg("rect", { x: -27, y: -14, width: 54, height: 28, rx: 14 }, label);
          text(label, 0, 5.5, points(unit.shown), `dig-pill-text dig-${sign(unit.shown)}`);
        }
        if (entering && !reducedMotion()) {
          const length = path.getTotalLength();
          path.style.strokeDasharray = `${length}`;
          path.animate([{ strokeDashoffset: length }, { strokeDashoffset: 0 }], {
            duration: 600,
            delay: 160 + 80 * a,
            easing: "cubic-bezier(0.4, 0, 0.2, 1)",
            fill: "backwards",
          });
        }
        const hit = svg("path", { d, class: "dig-hit dig-hit-arc" }, hits);
        hit.addEventListener("pointerenter", () => point(u));
        hit.addEventListener("pointerdown", () => pin(u));
      });
    }
  }

  // a surface: cells of the pure interaction, the first patch's ink across, the second's up
  function drawSurface(parent, grid, x, y, size, reach, cls = "dig-surface") {
    const group = svg("g", { class: cls }, parent);
    const cells = grid.length;
    const cell = size / cells;
    grid.forEach((row, a) =>
      row.forEach((z, b) => {
        const rect = svg("rect", {
          x: x + a * cell,
          y: y + (cells - 1 - b) * cell,
          width: cell + 0.3,
          height: cell + 0.3,
        }, group);
        rect.style.fill = fillOf(Math.abs(z) < 0.0005 ? 0 : z, reach, 1.25);
      }),
    );
    svg("rect", { x, y, width: size, height: size, class: "dig-surface-frame" }, group);
    return group;
  }

  // -- the space E --------------------------------------------------------------------------------------

  let spaceMarks = []; // per unit, the mark in E: a dot or a curve
  let spaceSize = [0, 0];
  function paintSpace() {
    const width = Math.max(200, Math.round(spaceBox.clientWidth));
    const height = Math.max(150, Math.round(spaceBox.clientHeight));
    spaceSize = [width, height];
    space.setAttribute("viewBox", `0 0 ${width} ${height}`);
    space.replaceChildren();
    spaceMarks = [];
    const units = unitsOf();
    if (state.space === "numbers") {
      // the number line: E = ℝ, each unit a dot at its value, dodged up and down so that none hides
      // another; the ticks below, with faint lines up through the swarm
      const values = units.map((u) => u.value * 100);
      const lo = Math.min(0, ...values);
      const hi = Math.max(0, ...values);
      const step = hi - lo > 60 ? 20 : hi - lo > 24 ? 10 : 5;
      const [from, to] = [Math.floor(lo / step) * step, Math.ceil(hi / step) * step];
      const left = 22;
      const right = width - 22;
      const x = (v) => left + ((v - from) / (to - from)) * (right - left);
      const bottom = height - 40;
      const lineY = Math.round((bottom + 8) / 2);
      const axis = svg("g", { class: "dig-axis" }, space);
      for (let t = from; t <= to + 1e-9; t += step) {
        svg("line", { x1: x(t), x2: x(t), y1: 8, y2: bottom, class: t ? "dig-axis-grid" : "dig-axis-zero" }, axis);
        text(axis, x(t), bottom + 16, t ? `${t > 0 ? "+" : "−"}${Math.abs(t)}` : "0", "dig-axis-label");
      }
      svg("line", { x1: left - 10, x2: right + 10, y1: lineY, y2: lineY, class: "dig-axis-line" }, axis);
      text(axis, (left + right) / 2, bottom + 33, "points of probability", "dig-axis-caption");
      const many = units.length > 20;
      const r = many ? 3.6 : state.units === "rows" ? 10 : 7.5;
      const gap = 2 * r + (many ? 0.6 : 1.5);
      const room = (lineY - 10 - r) * (many ? 0.6 : 1);
      const order = units.map((_, k) => k).sort((a, b) => Math.abs(values[b]) - Math.abs(values[a]));
      const placed = [];
      for (const k of order) {
        const cx = x(values[k]);
        // the height nearest the line where it touches no dot placed before
        const near = placed.filter((q) => Math.abs(q.cx - cx) < gap);
        const candidates = [0];
        for (const q of near) {
          const rise = Math.sqrt(Math.max(0, gap ** 2 - (q.cx - cx) ** 2));
          candidates.push(q.cy + rise, q.cy - rise);
        }
        candidates.sort((a, b) => Math.abs(a) - Math.abs(b) || b - a);
        const cy = candidates.find((c) => near.every((q) => (q.cx - cx) ** 2 + (q.cy - c) ** 2 >= gap ** 2 - 0.5)) ?? 0;
        placed.push({ cx, cy, k });
      }
      // a swarm taller than the room is pressed flat, its dots then overlap a little
      const press = Math.min(1, room / Math.max(1, ...placed.map((q) => Math.abs(q.cy))));
      for (const q of placed) q.cy *= press;
      const dots = svg("g", { class: "dig-dots" }, space);
      spaceMarks = units.map(() => null);
      for (const { cx, cy, k } of placed) {
        const dot = svg("circle", { cx, cy: lineY - cy, r, class: `dig-dot dig-${sign(units[k].shown)}` }, dots);
        dot.addEventListener("pointerenter", () => point(k));
        dot.addEventListener("pointerdown", () => pin(k));
        spaceMarks[k] = dot;
      }
      spaceMarks.label = text(space, 0, 0, "", "dig-dot-label");
    } else if (isSurface()) {
      // the surfaces of the strongest pairs, side by side: the pure interaction as the ink of both
      // patches goes from blank to black, the first patch's ink across, the second's up
      const reach = Math.max(...Object.values(data.surfaces).flat(2).map(Math.abs));
      const count = strongest.length;
      const columns = width >= 250 ? 3 : 2;
      const lines = Math.ceil(count / columns);
      const caption = 30;
      const label = 18;
      const size = Math.floor(Math.min((width - 16) / columns - 16, (height - 12 - caption) / lines - label - 8));
      const cellW = (width - 16) / columns;
      const cellH = size + label + 8;
      const top = Math.max(8, (height - caption - lines * cellH) / 2);
      spaceMarks = allPairs.map(() => null);
      strongest.forEach((key, a) => {
        const u = allPairs.findIndex((pair) => keyOf(pair) === key);
        const [col, line] = [a % columns, Math.floor(a / columns)];
        const x0 = 8 + col * cellW + (cellW - size) / 2;
        const y0 = top + line * cellH;
        const group = svg("g", { class: "dig-multiple" }, space);
        drawSurface(group, data.surfaces[key], x0, y0, size, reach);
        text(group, x0 + size / 2, y0 + size + 14, setOf(allPairs[u]), "dig-multiple-label");
        svg("rect", { x: x0 - 3, y: y0 - 3, width: size + 6, height: size + 6, rx: 4, class: "dig-multiple-ring" }, group);
        group.addEventListener("pointerenter", () => point(u));
        group.addEventListener("pointerdown", () => pin(u));
        if (entering && !reducedMotion()) {
          group.animate([{ opacity: 0, transform: "scale(0.85)" }, { opacity: 1, transform: "none" }], {
            duration: 420,
            delay: 160 + 70 * a,
            easing: "cubic-bezier(0.3, 1.4, 0.5, 1)",
            fill: "backwards",
          });
        }
        spaceMarks[u] = group;
      });
      text(space, width / 2, height - 12, "ink of the first patch →, of the second ↑", "dig-axis-caption");
    } else {
      // a frame of curves: the probability (0 to 100%) as the ink of a unit goes from 0 to 16
      const left = 42;
      const right = width - 16;
      const top = 30;
      const bottom = height - 40;
      const axis = svg("g", { class: "dig-axis" }, space);
      text(axis, 10, 16, `probability of ${CLASS}`, "dig-axis-caption", "start");
      for (const p of [0, 0.5, 1]) {
        const y = bottom - p * (bottom - top);
        svg("line", { x1: left, x2: right, y1: y, y2: y, class: p ? "dig-axis-grid" : "dig-axis-line" }, axis);
        text(axis, left - 7, y + 4, `${p * 100}%`, "dig-axis-label", "end");
      }
      text(axis, left, bottom + 17, "blank", "dig-axis-label", "start");
      text(axis, right, bottom + 17, "black", "dig-axis-label", "end");
      text(axis, (left + right) / 2, bottom + 17, `ink of the ${state.units === "rows" ? "row" : "patch"} →`, "dig-axis-caption");
      const curves = UNITS[state.units].curves;
      const lines = svg("g", { class: "dig-lines" }, space);
      spaceMarks = curves.map((ys, k) => {
        const line = svg("path", { d: curvePath(ys, left, top, right - left, bottom - top), class: "dig-line" }, lines);
        const hit = svg("path", { d: line.getAttribute("d"), class: "dig-line-hit" }, lines);
        hit.addEventListener("pointerenter", () => point(k));
        hit.addEventListener("pointerdown", () => pin(k));
        if (entering && !reducedMotion()) {
          const length = line.getTotalLength();
          line.style.strokeDasharray = `${length}`;
          line.animate([{ strokeDashoffset: length }, { strokeDashoffset: 0 }], {
            duration: 800,
            delay: 200 + 30 * k,
            easing: "cubic-bezier(0.4, 0, 0.2, 1)",
            fill: "backwards",
          });
        }
        return line;
      });
      spaceMarks.label = text(space, 0, 0, "", "dig-dot-label");
      spaceMarks.frame = [left, right, top, bottom];
    }
  }

  // -- pointing at a unit: on the image, in E, and in words -----------------------------------------------

  let pointed = null; // the unit under the pointer
  let pinned = null; // the unit tapped or chosen with the keys
  // the unit shown when none is pointed at: the strongest of the view
  const strongestUnit = () => {
    if (state.units === "pairs") return allPairs.findIndex((pair) => keyOf(pair) === strongest[0]);
    const values = UNITS[state.units].values;
    return values.reduce((best, v, k) => (Math.abs(v) > Math.abs(values[best]) ? k : best), 0);
  };
  const currentUnit = () => pointed ?? pinned ?? strongestUnit();
  const point = (k) => {
    if (isSurface() && !strongest.includes(keyOf(allPairs[k]))) return;
    pointed = k;
    highlight();
  };
  const pin = (k) => {
    pinned = k;
    point(k);
  };
  map.addEventListener("pointerleave", () => {
    pointed = null;
    highlight();
  });
  space.addEventListener("pointerleave", () => {
    pointed = null;
    highlight();
  });

  function highlight() {
    const k = currentUnit();
    const units = UNITS[state.units];
    const ids = units.sets[k];
    // on the image: the unit's outline, or its arc
    if (state.units === "pairs") {
      focusRing.setAttribute("d", "");
      let drawn = false;
      for (const group of arcs.children) {
        const on = Number(group.dataset.unit) === k;
        group.classList.toggle("is-active", on);
        if (on) arcs.appendChild(group); // on top
        drawn ||= on;
      }
      arcs.querySelector(".dig-pair-ghost")?.remove();
      if (!drawn) {
        // a pair without an arc gets one while it is pointed at
        const { d, mid } = arcPath(ids, 0.3);
        const ghost = svg("g", { class: "dig-pair dig-pair-ghost is-active" }, arcs);
        svg("path", { d, class: `dig-arc dig-${sign(units.shown[k])}`, "stroke-width": 3 }, ghost);
        for (const p of ids) svg("circle", { cx: center(p)[0], cy: center(p)[1], r: 6, class: "dig-arc-end" }, ghost);
        const label = svg("g", { class: "dig-pill", transform: `translate(${mid[0]} ${mid[1]})` }, ghost);
        svg("rect", { x: -27, y: -14, width: 54, height: 28, rx: 14 }, label);
        text(label, 0, 5.5, points(units.shown[k]), `dig-pill-text dig-${sign(units.shown[k])}`);
      }
    } else {
      focusRing.setAttribute("d", outline(ids));
    }
    badges.forEach((badge, p) => badge.classList.toggle("is-active", ids.includes(p)));
    // in E: its dot, its curve, or its surface
    if (state.space === "numbers") {
      spaceMarks.forEach((dot, j) => dot?.classList.toggle("is-active", j === k));
      const dot = spaceMarks[k];
      if (dot && spaceMarks.label) {
        dot.parentNode.appendChild(dot); // on top
        const [cx, cy] = [Number(dot.getAttribute("cx")), Number(dot.getAttribute("cy"))];
        const label = spaceMarks.label;
        label.textContent = setOf(ids);
        label.setAttribute("x", Math.min(spaceSize[0] - 34, Math.max(34, cx)));
        label.setAttribute("y", Math.max(14, cy - Number(dot.getAttribute("r")) - 8));
      }
    } else if (isSurface()) {
      spaceMarks.forEach((group, j) => group?.classList.toggle("is-active", j === k));
    } else {
      spaceMarks.forEach((line, j) => line.classList.toggle("is-active", j === k));
      const line = spaceMarks[k];
      if (line) {
        line.parentNode.appendChild(line);
        const ys = units.curves[k];
        const [, right, top, bottom] = spaceMarks.frame;
        const label = spaceMarks.label;
        label.textContent = setOf(ids);
        const end = bottom - ys.at(-1) * (bottom - top);
        label.setAttribute("x", right - 2);
        label.setAttribute("y", ys.at(-1) > 0.5 ? end + 17 : end - 9);
        label.style.textAnchor = "end";
      }
    }
    // in words: e(S) = …
    const S = `<span class="dig-set">${setOf(ids)}</span>`;
    const head = `<span class="dig-e"><i>e</i>(${S})</span>`;
    const what = units.what(ids);
    if (state.space === "numbers") {
      const t = units.shown[k];
      const word =
        state.units === "pairs"
          ? `what ${what} add together beyond what each adds alone`
          : `${what} ${t >= 0 ? "adds" : "takes"} ${Math.abs(t / 10).toFixed(1)} points ${t >= 0 ? "to" : "from"} the probability of ${CLASS}`;
      readout.innerHTML = `${head} = <strong class="dig-${sign(t)}">${points(t)}</strong> <span class="dig-readout-gloss">${word}</span>`;
    } else if (isSurface()) {
      const flat = data.surfaces[keyOf(ids)].flat();
      const [lo, hi] = [Math.min(...flat), Math.max(...flat)];
      readout.innerHTML =
        `${head} = <strong>its surface</strong> <span class="dig-readout-gloss">what the ink of ${what} does ` +
        `together beyond each alone: from ${points(tenths(lo))} to ${points(tenths(hi))} points</span>`;
    } else {
      const ys = units.curves[k];
      readout.innerHTML =
        `${head} = <strong>its curve</strong> <span class="dig-readout-gloss">the probability of ${CLASS}: ` +
        `${percent(ys[0])} with ${what} blank, ${percent(ys.at(-1))} with it black</span>`;
    }
  }

  // keys move between the units of the image
  map.tabIndex = 0;
  map.addEventListener("keydown", (event) => {
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
    if (!step) return;
    event.preventDefault();
    const choices =
      state.units === "pairs"
        ? strongest.map((key) => allPairs.findIndex((pair) => keyOf(pair) === key))
        : UNITS[state.units].sets.map((_, k) => k);
    const at = choices.indexOf(currentUnit());
    pin(choices[(at + step + choices.length) % choices.length]);
  });

  // -- the values fly from their units into E ---------------------------------------------------------

  const flyValues = (delay) => {
    if (reducedMotion() || state.space !== "numbers") return;
    const base = body.getBoundingClientRect();
    const box = map.getBoundingClientRect();
    const scale = box.width / 400;
    const units = unitsOf();
    const many = units.length > 20;
    let last = 0;
    units.forEach((unit, k) => {
      const dot = spaceMarks[k];
      if (!dot) return;
      // from the middle of its unit on the image (for a pair, between its patches)
      const from = unit.ids.map(center).reduce(([x, y], [a, b]) => [x + a / unit.ids.length, y + b / unit.ids.length], [0, 0]);
      const to = dot.getBoundingClientRect();
      const r = to.width / 2;
      const start = [box.left + from[0] * scale - base.left, box.top + from[1] * scale - base.top];
      const end = [to.left + r - base.left, to.top + r - base.top];
      const spark = el("span", `dig-spark dig-${sign(unit.shown)}`);
      spark.style.width = spark.style.height = `${2 * r}px`;
      spark.style.left = `${end[0] - r}px`;
      spark.style.top = `${end[1] - r}px`;
      fly.appendChild(spark);
      const [dx, dy] = [start[0] - end[0], start[1] - end[1]];
      const wait = delay + (many ? 3 : 45) * k;
      dot.animate([{ opacity: 0 }, { opacity: 0, offset: 0.99 }, { opacity: 1 }], { duration: wait + 620, fill: "backwards" });
      last = Math.max(last, wait + 620);
      spark
        .animate(
          [
            { transform: `translate(${dx}px, ${dy}px) scale(0.4)`, opacity: 0 },
            { transform: `translate(${dx * 0.55}px, ${dy * 0.55 - 30}px) scale(1.25)`, opacity: 1, offset: 0.45 },
            { transform: "none", opacity: 1 },
          ],
          { duration: 620, delay: wait, easing: "cubic-bezier(0.45, 0, 0.25, 1)", fill: "backwards" },
        )
        .finished.then(() => spark.remove(), () => spark.remove());
    });
    // the name of the unit pointed at comes when its value has landed
    spaceMarks.label?.animate([{ opacity: 0 }, { opacity: 0, offset: 0.99 }, { opacity: 1 }], { duration: last, fill: "backwards" });
  };

  // -- a view: the two choices, painted ---------------------------------------------------------------

  let lastWidth = 0;
  let seen = false;
  function render({ changed = null } = {}) {
    for (const stop of stops) stop();
    stops = [];
    fly.replaceChildren();
    entering = changed !== null && changed !== "resize" && seen;
    const key = `${state.units}-${state.space}`;
    const units = UNITS[state.units];
    unitsNotation.innerHTML = `<span class="dig-cal">𝒮</span> = ${units.notation}`;
    spaceNotation.innerHTML = `<i>E</i> = ${isSurface() ? "surfaces" : SPACES[state.space].notation}`;
    spaceGloss.textContent = isSurface() ? SURFACE_GLOSS : SPACES[state.space].gloss;
    method.textContent = METHODS[key];
    lastWidth = root.clientWidth;
    paintMap();
    paintSpace();
    highlight();
    if (entering) {
      for (const node of [changed === "units" ? unitsPart : spacePart, arrow]) glow(node);
      flyValues(120);
    }
    // where the explanation adds up, the line that adds it up
    equation.innerHTML =
      state.space === "numbers" && state.units !== "pairs"
        ? `They add up: all erased ${percent(data.empty)} ` +
          `<span class="dig-${sign(total)}">${total >= 0 ? "+" : "−"} ${(Math.abs(total) / 10).toFixed(1)}</span> ` +
          `= this digit ${percent(data.prediction)}`
        : state.units === "pairs" && state.space === "numbers"
          ? `The ${strongest.length} strongest of the ${allPairs.length} pairs as arcs; point at a dot for any other.`
          : "";
  }
  const glow = (node) => {
    if (reducedMotion()) return;
    node.classList.remove("is-lit");
    void node.offsetWidth; // to start the glow again
    node.classList.add("is-lit");
  };

  // -- the first time it is seen: the digit, the units, the space, then the explanation -------------------

  unitsSwitch.set("patches");
  spaceSwitch.set("numbers");
  render();

  const intro = () => {
    if (reducedMotion()) {
      render();
      return;
    }
    root.classList.add("dig-intro");
    // the digit: its pixels, row by row
    [...inkLayer.children].forEach((pixel, p) =>
      pixel.animate([{ opacity: 0, transform: "scale(0.4)" }, { opacity: pixel.style.getPropertyValue("--a") }], {
        duration: 380,
        delay: 18 * p,
        easing: "cubic-bezier(0.3, 1.4, 0.5, 1)",
        fill: "backwards",
      }),
    );
    // the units: the cuts open, and the patches get their numbers
    const t1 = 900;
    setTimeout(() => {
      glow(unitsPart);
      root.classList.add("dig-cut");
    }, t1);
    badges.forEach((badge, k) =>
      badge.animate(
        [
          { opacity: 0, transform: `translate(${corner(k)[0] + 17}px, ${corner(k)[1] + 17}px) scale(0.2)` },
          { opacity: 1, transform: `translate(${corner(k)[0] + 17}px, ${corner(k)[1] + 17}px) scale(1.25)`, offset: 0.6 },
          { opacity: 1, transform: `translate(${corner(k)[0] + 17}px, ${corner(k)[1] + 17}px)` },
        ],
        { duration: 420, delay: t1 + 40 * k, easing: "cubic-bezier(0.3, 1.4, 0.5, 1)", fill: "backwards" },
      ),
    );
    // the space E
    const t2 = t1 + 900;
    setTimeout(() => glow(spacePart), t2);
    spaceBox.animate([{ opacity: 0, transform: "translateY(8px)" }, { opacity: 1, transform: "none" }], {
      duration: 500,
      delay: t2,
      easing: "cubic-bezier(0.2, 0.7, 0.2, 1)",
      fill: "backwards",
    });
    // the explanation: the image takes its colors, and the values fly into E
    const t3 = t2 + 700;
    setTimeout(() => {
      root.classList.remove("dig-intro", "dig-cut");
      glow(arrow);
      entering = true;
      paintMap();
      entering = false;
      flyValues(0);
    }, t3);
    for (const node of [readout, equation]) {
      node.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 400, delay: t3 + 1100, fill: "backwards" });
    }
  };

  new ResizeObserver(() => {
    if (Math.abs(root.clientWidth - lastWidth) > 2 || spaceBox.clientHeight !== spaceSize[1]) {
      render({ changed: "resize" });
    }
  }).observe(root);

  // until it is seen, the explanation waits as the bare digit
  if (!reducedMotion()) root.classList.add("dig-intro");
  const stopWatching = watchVisibility(stage, (visible) => {
    if (visible && !seen) {
      seen = true;
      stopWatching?.();
      intro();
    }
  });
});
