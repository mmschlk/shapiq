// The illustration "lab-hulls" (chapter 2.1 of the user guide): Definition 2.1, e : 𝒮 → E, drawn
// on one patient of the diabetes data. The features are amber dots, numbered 1, …, d; a unit S ∈ 𝒮
// is a shape drawn around the features it contains: a ring around one dot, a capsule around two.
// Its explanation e(S) ∈ E lives on the shape: a tint in red or blue and a tag with the number, or a
// small curve or surface. Two switches choose the units 𝒮 (single features, groups, or all pairs)
// and the explanation space E (numbers or curves); the shapes merge, split, and slide from one
// collection to the next, over the same dots: the units are a choice of sets of features.
//
// - single features: their Shapley values; their ICE curves (the prediction as one feature varies);
// - groups, BMI and blood pressure as one unit (the body), HDL and log triglycerides as one (the
//   lipids), age and glucose alone: the Shapley values of the units, each unit one player; the
//   curves of the single features and the surfaces of the groups (the prediction as both vary);
// - pairs: the pairwise Shapley interactions (k-SII); the pure interaction surfaces (what a pair
//   does together beyond what each feature does alone).
//
// Pointing at a shape (or a feature, for pairs) reads S ↦ e(S) out below the stage. Values are
// computed by ../compute/compute.py into data.js; red is positive and blue negative, amber marks
// the features (docs/STYLE_GUIDE.md).
window.shapiqIllustrations?.define("lab-hulls", async (stage, kit) => {
  const { el, svg, tween, ease, reducedMotion, segmented, loadScript, watchVisibility } = kit;
  if (!window.SHAPIQ_LAB_HULLS) await loadScript(`${kit.root}data.js?v=${kit.version}`);
  const data = window.SHAPIQ_LAB_HULLS;
  const features = data.features;
  const d = features.length;

  // -- numbers: on the scale of the target, to one decimal -------------------------------------------

  const signed = (v) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(1)}`;
  const plain = (v) => v.toFixed(1);
  const signClass = (v) => (v >= 0 ? "is-pos" : "is-neg");
  const keyOf = (ids) => ids.join("-");
  const setOf = (ids) => `{${ids.map((k) => k + 1).join(", ")}}`;
  const valueOf = (k, v = features[k].value) =>
    `${v.toFixed(features[k].decimals)}${features[k].unit ? ` ${features[k].unit}` : ""}`;
  const listOf = (words) =>
    words.length < 3 ? words.join(" and ") : `${words.slice(0, -1).join(", ")}, and ${words.at(-1)}`;
  const namesOf = (ids) => listOf(ids.map((k, n) => (n ? lower(features[k].label) : features[k].label)));
  const lower = (label) => (label === label.toUpperCase() || /^[A-Z]{2}/.test(label) ? label : label.toLowerCase());
  const total = data.prediction - data.baseline;

  // -- the units, the spaces, and the explanation they give ---------------------------------------------

  const pairs = [];
  for (let i = 0; i < d; i++) for (let j = i + 1; j < d; j++) pairs.push([i, j]);
  const COLLECTIONS = {
    features: {
      label: "Features",
      sets: features.map((_, k) => [k]),
      notation: `{{1}, {2}, …, {${d}}}`,
      values: data.shapley_shown,
    },
    groups: {
      label: "Groups",
      sets: data.partition,
      notation: `{${data.partition.map(setOf).join(", ")}}`,
      values: data.partition_shown,
    },
    pairs: {
      label: "Pairs",
      sets: pairs,
      notation: "{{<i>i</i>, <i>j</i>} : <i>i</i> &lt; <i>j</i>}",
      values: pairs.map((ids) => Math.round(data.interactions[keyOf(ids)] * 10) / 10),
    },
  };
  const SPACES = {
    numbers: { label: "Numbers", notation: "ℝ" },
    curves: { label: "Curves", notation: "curves" },
  };
  const METHODS = {
    "features-numbers": "Shapley values",
    "groups-numbers": "Shapley values of the units",
    "pairs-numbers": "Shapley interactions",
    "features-curves": "ICE curves",
    "groups-curves": "ICE curves and surfaces",
    "pairs-curves": "pure interaction surfaces",
  };
  const groupName = (ids) => {
    const k = data.partition.findIndex((part) => keyOf(part) === keyOf(ids));
    return k >= 0 ? data.group_names[k] : "";
  };
  // what e(S) says for this patient, in words
  const gloss = (collection, space, ids, v) => {
    const who = namesOf(ids);
    if (space === "numbers") {
      const way = v >= 0 ? "raise" : "lower";
      if (collection === "pairs") {
        return `Together, ${who} ${v >= 0 ? "add" : "take away"} ${plain(Math.abs(v))} beyond what each does alone.`;
      }
      if (ids.length > 1) return `${who}, as one unit, ${way} the predicted progression by ${plain(Math.abs(v))}.`;
      return `${who} ${way}s the predicted progression by ${plain(Math.abs(v))}${collection === "groups" ? ", as one of four units" : ""}.`;
    }
    const axes = ids.length > 1 ? ` (${lower(features[ids[0]].label)} up, ${lower(features[ids[1]].label)} across)` : "";
    if (collection === "pairs") return `What ${who} do together beyond what each does alone, as both vary${axes}; the dot is this patient.`;
    if (ids.length > 1) return `The prediction as ${who} vary together${axes}; the dot is this patient.`;
    return `The prediction for this patient as ${lower(features[ids[0]].label)} varies; the dot is the patient.`;
  };

  // -- the layout of the figure ----------------------------------------------------------------------

  const root = el("div", "lh");
  const head = el("div", "lh-head");
  const control = (symbol, name) => {
    const node = el("div", "lh-control");
    const label = el("p", "lh-control-label");
    label.innerHTML = `<span class="lh-sym">${symbol}</span><span class="lh-name">${name}</span>`;
    node.appendChild(label);
    head.appendChild(node);
    return node;
  };
  const state = { collection: "features", space: "numbers" };
  const unitsControl = control('<span class="lh-cal">𝒮</span>', "the units");
  const spaceControl = control("<i>E</i>", "the explanation space");
  const unitsSwitch = segmented("Units", Object.entries(COLLECTIONS).map(([k, c]) => [k, c.label]), (k, byUser) => {
    if (byUser) change({ collection: k });
  });
  const spaceSwitch = segmented("Explanation space", Object.entries(SPACES).map(([k, s]) => [k, s.label]), (k, byUser) => {
    if (byUser) change({ space: k });
  });
  unitsControl.appendChild(unitsSwitch.node);
  spaceControl.appendChild(spaceSwitch.node);

  // the definition, filled in: 𝒮 = ..., E = ..., e = the method
  const definition = el("p", "lh-definition");
  const defUnits = el("span", "lh-def lh-def-units");
  const defSpace = el("span", "lh-def lh-def-space");
  const defMethod = el("span", "lh-def lh-def-method");
  definition.append(defUnits, defSpace, defMethod);

  const figure = svg("svg", { class: "lh-stage", role: "img", tabindex: "0" });
  const layers = {};
  for (const name of ["shapes", "dots", "labels", "tags"]) layers[name] = svg("g", { class: `lh-${name}` }, figure);

  // S ↦ e(S) for the unit in focus, and what it says
  const map = el("div", "lh-map");
  map.setAttribute("aria-live", "polite");
  const mapRow = el("div", "lh-map-row");
  const mapGloss = el("p", "lh-map-gloss");
  map.append(mapRow, mapGloss);
  const sum = el("p", "lh-sum");
  const foot = el("p", "lh-foot");
  foot.textContent =
    `A gradient boosting model on six features of the diabetes data (${data.patients} patients; ` +
    `R² ${data.r2.toFixed(2)} on held-out patients, ${data.folds}-fold cross-validation) predicts a ` +
    `disease progression of ${plain(data.prediction)} one year after baseline for this patient, ` +
    `against ${plain(data.baseline)} on average.`;
  root.append(head, definition, figure, map, sum, foot);
  stage.appendChild(root);

  // -- geometry: the features on an ellipse, the units as the hull of their dots, rounded --------------

  const N = 64; // points on the outline of a shape, one per direction of its normal
  const DIRS = Array.from({ length: N }, (_, k) => [Math.cos((2 * Math.PI * k) / N), Math.sin((2 * Math.PI * k) / N)]);
  // A shape is the hull of a few anchors, grown by r: a ring around one, a capsule around two. Its
  // outline: for each direction, the anchor farthest in it, moved by r. Between two collections the
  // anchors slide from dot to dot, so a ring stretches into a capsule, a capsule swings to another
  // pair or shrinks into a ring.
  const pathOf = (anchors, r) => {
    let text = "";
    DIRS.forEach(([c, s], k) => {
      let best = -Infinity;
      let x = 0;
      let y = 0;
      for (const [ax, ay] of anchors) {
        const reach = ax * c + ay * s;
        if (reach > best) [best, x, y] = [reach, ax, ay];
      }
      text += `${k ? "L" : "M"}${(x + r * c).toFixed(1)} ${(y + r * s).toFixed(1)}`;
    });
    return `${text}Z`;
  };
  const lerp = (a, b, t) => a + (b - a) * t;
  // the anchors of one shape matched to those of the next, as two lists of the same length, with
  // the features each anchor goes to: shared features stay, the others pair up in turn, and an
  // anchor without a partner comes from (or goes to) the nearest one
  const nearest = (point, points) =>
    points.reduce((a, b) => (Math.hypot(b[0] - point[0], b[1] - point[1]) < Math.hypot(a[0] - point[0], a[1] - point[1]) ? b : a));
  function pairUp(fromIds, fromPts, toIds) {
    const toPts = toIds.map((k) => geo.P[k]);
    const [from, to, ids] = [[], [], []];
    const restFrom = fromIds.map((_, i) => i);
    const restTo = [];
    toIds.forEach((id, j) => {
      const i = restFrom.findIndex((at) => fromIds[at] === id);
      if (i < 0) return restTo.push(j);
      from.push(fromPts[restFrom[i]]);
      to.push(toPts[j]);
      ids.push(id);
      restFrom.splice(i, 1);
    });
    while (restFrom.length || restTo.length) {
      const i = restFrom.shift();
      const j = restTo.shift();
      const start = i !== undefined ? fromPts[i] : nearest(toPts[j], fromPts);
      const end = j !== undefined ? toPts[j] : nearest(start, toPts);
      from.push(start);
      to.push(end);
      ids.push(toIds[toPts.indexOf(end)]);
    }
    return { from, to, ids };
  }

  let geo = null;
  const ANGLES = [180, 116, 64, 0, -64, -116]; // degrees, counterclockwise from the right
  function geometry(width) {
    const narrow = width < 540;
    const height = narrow ? 392 : 350;
    const cx = width / 2;
    const cy = narrow ? 194 : 172;
    const rx = narrow ? Math.min(width / 2 - 56, 120) : Math.min(width / 2 - 150, 230);
    const ry = narrow ? 126 : 112;
    const P = ANGLES.map((a) => [cx + rx * Math.cos((a * Math.PI) / 180), cy - ry * Math.sin((a * Math.PI) / 180)]);
    return {
      width,
      height,
      narrow,
      cx,
      cy,
      P,
      r: narrow ? 22 : 27,
      rPair: narrow ? 13 : 16,
      dot: narrow ? 10 : 11,
    };
  }
  const radiusOf = (collection) => (collection === "pairs" ? geo.rPair : geo.r);

  // -- the features: amber dots with their numbers, and their names and values -------------------------

  const dots = features.map((f, k) => {
    const g = svg("g", { class: "lh-dot" }, layers.dots);
    const inner = svg("g", {}, g);
    svg("circle", { r: 11 }, inner);
    svg("text", { "text-anchor": "middle", dy: "0.35em" }, inner).textContent = String(k + 1);
    const label = svg("text", { class: "lh-label" }, layers.labels);
    const name = svg("tspan", { class: "lh-label-name" }, label);
    const value = svg("tspan", { class: "lh-label-value" }, label);
    return { g, inner, label, name, value };
  });
  function placeDots() {
    dots.forEach(({ g, inner, label, name, value }, k) => {
      const [x, y] = geo.P[k];
      g.setAttribute("transform", `translate(${x} ${y})`);
      inner.querySelector("circle").setAttribute("r", geo.dot);
      const f = features[k];
      name.textContent = geo.narrow ? f.short : f.label;
      value.textContent = valueOf(k);
      // beside the dot at the left and right, else above or below it; on a narrow stage, below
      // the dots at the sides
      const angle = ANGLES[k];
      const gap = geo.r + 8;
      let anchor = "middle";
      let lx = x;
      let ly;
      if (Math.abs(angle) === 180 || angle === 0) {
        if (geo.narrow) {
          ly = y + gap + 10;
        } else {
          anchor = angle === 0 ? "start" : "end";
          lx = x + (angle === 0 ? gap : -gap);
          ly = y - 4;
        }
      } else if (angle > 0) {
        ly = y - gap - 18;
      } else {
        ly = y + gap + 10;
      }
      label.setAttribute("text-anchor", anchor);
      name.setAttribute("x", lx);
      name.setAttribute("y", ly);
      value.setAttribute("x", lx);
      value.setAttribute("y", ly + 15);
    });
    // where the labels are, for the tags to keep clear of them
    labelBoxes = dots.map(({ label }) => {
      try {
        const b = label.getBBox();
        return [b.x + b.width / 2, b.y + b.height / 2, b.width, b.height];
      } catch {
        return [0, 0, 0, 0];
      }
    });
  }
  let labelBoxes = [];

  // -- the explanations: number tags and curve glyphs ------------------------------------------------

  const allCurves = [...data.curves.flatMap((c) => c.y), ...Object.values(data.surfaces).flatMap((z) => z.flat())];
  const curveLo = Math.min(...allCurves);
  const curveHi = Math.max(...allCurves);
  const pureReach = Math.max(...Object.values(data.pure).flatMap((g) => g.flat().map(Math.abs)));
  // the patient's prediction as one feature varies, on one scale for all; the dot is the patient
  function drawCurve(parent, k, w, h, pad = 3) {
    const { x: xs, y: ys } = data.curves[k];
    const x = (v) => pad + ((v - xs[0]) / (xs.at(-1) - xs[0])) * (w - 2 * pad);
    const y = (v) => h - pad - ((v - curveLo) / (curveHi - curveLo)) * (h - 2 * pad);
    const line = xs.map((v, i) => `${i ? "L" : "M"}${x(v).toFixed(1)} ${y(ys[i]).toFixed(1)}`).join("");
    svg("path", { d: `${line}L${x(xs.at(-1)).toFixed(1)} ${h}L${x(xs[0]).toFixed(1)} ${h}Z`, class: "lh-area" }, parent);
    const path = svg("path", { d: line, class: "lh-line" }, parent);
    svg("circle", { cx: x(features[k].value), cy: y(data.prediction), r: Math.max(2.2, h / 14), class: "lh-marker" }, parent);
    return path;
  }
  // the prediction (amber) or the pure interaction (red and blue, deeper by the square root of its
  // size, on one scale for all pairs) as two features vary: the first up, the second across; the dot
  // is the patient
  function drawSurface(parent, [i, j], kind, size) {
    const key = `${i}-${j}`;
    const grid = kind === "pure" ? data.pure[key] : data.surfaces[key];
    const cells = grid.length;
    const cell = size / cells;
    const heat = svg("g", { class: "lh-heat" }, parent);
    grid.forEach((row, a) => {
      row.forEach((z, b) => {
        const rect = svg("rect", { x: (b * cell).toFixed(2), y: ((cells - 1 - a) * cell).toFixed(2), width: (cell + 0.35).toFixed(2), height: (cell + 0.35).toFixed(2) }, heat);
        rect.style.fill =
          kind === "pure"
            ? `color-mix(in oklab, var(${z >= 0 ? "--shapiq-red" : "--shapiq-blue"}) ${Math.round(Math.sqrt(Math.abs(z) / pureReach) * 100)}%, var(--lh-paper))`
            : `color-mix(in oklab, var(--shapiq-amber) ${Math.round(6 + ((z - curveLo) / (curveHi - curveLo)) * 94)}%, var(--lh-paper))`;
      });
    });
    svg("rect", { x: 0, y: 0, width: size, height: size, class: "lh-frame" }, parent);
    // the grids pass through the patient's values: the patient sits on the middle of a cell
    const at = (k) => {
      const [lo, hi] = data.axes[k];
      return ((features[k].value - lo) / (hi - lo)) * (size - cell) + cell / 2;
    };
    svg("circle", { cx: at(j), cy: size - at(i), r: Math.max(2, size / 16), class: "lh-marker" }, parent);
  }

  // a tag: the number of a unit, or its curve or surface, on the edge of its shape
  function makeTag(collection, space, ids, v) {
    const g = svg("g", { class: "lh-tag" }, layers.tags);
    const body = svg("g", {}, g);
    let w;
    let h;
    if (space === "numbers") {
      const text = signed(v);
      w = 12 + 7.1 * text.length;
      h = 21;
      body.classList.add(signClass(v));
      svg("rect", { x: -w / 2, y: -h / 2, width: w, height: h, rx: h / 2, class: "lh-tag-box" }, body);
      const label = svg("text", { "text-anchor": "middle", dy: "0.35em", class: "lh-tag-text" }, body);
      label.textContent = text;
      return { g, body, w, h, label, v };
    }
    const small = geo.narrow;
    if (ids.length === 1) {
      [w, h] = small ? [50, 28] : [62, 34];
      svg("rect", { x: -w / 2, y: -h / 2, width: w, height: h, rx: 6, class: "lh-tag-box lh-tag-card" }, body);
      const inner = svg("g", { transform: `translate(${-w / 2 + 3} ${-h / 2 + 3})` }, body);
      const line = drawCurve(inner, ids[0], w - 6, h - 6, 2);
      return { g, body, w, h, line };
    }
    const size = small ? 28 : 34;
    [w, h] = [size + 6, size + 6];
    svg("rect", { x: -w / 2, y: -h / 2, width: w, height: h, rx: 6, class: "lh-tag-box lh-tag-card" }, body);
    const inner = svg("g", { transform: `translate(${-size / 2} ${-size / 2})` }, body);
    drawSurface(inner, ids, collection === "pairs" ? "pure" : "surface", size);
    return { g, body, w, h };
  }

  // where a tag sits: on the edge of its shape, toward the middle of the stage (along the axis
  // nearest to that direction); for a pair, on the line between its two dots
  function tagSpot(unit, tag, t = 0.5, side = 0) {
    const ids = unit.ids;
    const P = geo.P;
    if (ids.length === 2 && state.collection === "pairs") {
      // t along the line between the dots, and side across it, toward its edge
      const [a, b] = [P[ids[0]], P[ids[1]]];
      const length = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
      const [nx, ny] = [-(b[1] - a[1]) / length, (b[0] - a[0]) / length];
      return [a[0] + (b[0] - a[0]) * t + side * nx, a[1] + (b[1] - a[1]) * t + side * ny];
    }
    const cx = ids.reduce((s, k) => s + P[k][0], 0) / ids.length;
    const cy = ids.reduce((s, k) => s + P[k][1], 0) / ids.length;
    const [dx, dy] = [geo.cx - cx, geo.cy - cy];
    const r = radiusOf(state.collection);
    const overlap = 7;
    if (Math.abs(dx) > Math.abs(dy) * 1.2) {
      const edge = dx > 0 ? Math.max(...ids.map((k) => P[k][0])) + r : Math.min(...ids.map((k) => P[k][0])) - r;
      return [edge + Math.sign(dx) * (tag.w / 2 - overlap), cy];
    }
    const edge = dy > 0 ? Math.max(...ids.map((k) => P[k][1])) + r : Math.min(...ids.map((k) => P[k][1])) - r;
    return [cx, edge + Math.sign(dy) * (tag.h / 2 - overlap)];
  }

  // -- the units on the stage --------------------------------------------------------------------------

  let units = []; // { ids, key, value, g, fill, line, anchors, anchorIds, r, tint, tag, at }
  let dying = []; // the shapes of the last collection that still fade
  let focus = null; // { kind: "unit", key } or { kind: "feature", k }, from the pointer or the keys
  let pinned = false; // a tap keeps the focus until the next tap
  let stops = [];
  const stopAll = () => {
    for (const stop of stops) stop();
    stops = [];
  };

  // the tint of a shape: red or blue by the sign of its number, deeper for a larger one (on one scale
  // for all the numbers, by its square root); for curves, a light amber
  const maxNumber = Math.max(...Object.values(COLLECTIONS).flatMap((c) => c.values.map(Math.abs)));
  const tintOf = (space, v) => (space === "numbers" ? { sign: v >= 0 ? "pos" : "neg", mix: 7 + 33 * Math.sqrt(Math.abs(v) / maxNumber) } : { sign: "none", mix: 100 });
  const paint = (unit, tint) => {
    unit.fill.classList.toggle("is-pos", tint.sign === "pos");
    unit.fill.classList.toggle("is-neg", tint.sign === "neg");
    unit.fill.classList.toggle("is-none", tint.sign === "none");
    unit.fill.style.setProperty("--lh-mix", `${tint.mix.toFixed(1)}%`);
    unit.tint = tint;
  };
  const makeUnit = (ids, value) => {
    const unit = { ids, key: keyOf(ids), value };
    // its tint and its outline together, so that a stronger unit lies over a weaker one, as paper
    unit.g = svg("g", { class: "lh-unit" }, layers.shapes);
    unit.fill = svg("path", { class: "lh-fill" }, unit.g);
    unit.line = svg("path", { class: "lh-outline" }, unit.g);
    return unit;
  };
  // a shape from its anchors (and the features they stand on) and its radius
  const setShape = (unit, anchors, r, anchorIds = unit.anchorIds) => {
    Object.assign(unit, { anchors, r, anchorIds });
    const path = pathOf(anchors, r);
    unit.fill.setAttribute("d", path);
    unit.line.setAttribute("d", path);
  };
  const atRest = (unit, r) => setShape(unit, unit.ids.map((k) => geo.P[k]), r, unit.ids);

  // which tags show: all, but for pairs only the strongest few and those in focus, placed so that
  // they do not cover one another
  function placeTags() {
    const placed = [];
    // how much a tag at (x, y) would cover the tags placed before it and the dots
    const cover = (ax, ay, aw, ah, bx, by, bw, bh) =>
      Math.max(0, (aw + bw) / 2 - Math.abs(ax - bx)) * Math.max(0, (ah + bh) / 2 - Math.abs(ay - by));
    const overlap = ([x, y], tag) =>
      placed.reduce((a, [px, py, pw, ph]) => a + cover(x, y, tag.w + 6, tag.h + 6, px, py, pw, ph), 0) +
      geo.P.reduce((a, [dx, dy]) => a + cover(x, y, tag.w, tag.h, dx, dy, 2 * geo.dot + 4, 2 * geo.dot + 4), 0) +
      labelBoxes.reduce((a, [lx, ly, lw, lh]) => a + 0.5 * cover(x, y, tag.w, tag.h, lx, ly, lw, lh), 0);
    const ranked = [...units].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    const shown = new Set(
      state.collection !== "pairs"
        ? units
        : focus
          ? units.filter((u) => (focus.kind === "unit" ? u.key === focus.key : u.ids.includes(focus.k)))
          : ranked.slice(0, geo.narrow ? 3 : 4),
    );
    for (const unit of [...ranked].sort((a, b) => shown.has(b) - shown.has(a))) {
      const visible = shown.has(unit);
      let spot = tagSpot(unit, unit.tag);
      if (state.collection === "pairs" && visible) {
        // along the line between its dots, where it covers the least, the middle if it can; on a
        // short pair, off to the edge of its capsule
        let least = Infinity;
        const sides = [0, geo.rPair, -geo.rPair];
        for (const [t, side] of [0.5, 0.42, 0.58, 0.34, 0.66, 0.27, 0.73].flatMap((t) => sides.map((side) => [t, side]))) {
          const at = tagSpot(unit, unit.tag, t, side);
          const covered = overlap(at, unit.tag) + (side ? 40 : 0);
          if (covered < least - 1) [spot, least] = [at, covered];
        }
      }
      if (visible) placed.push([...spot, unit.tag.w, unit.tag.h]);
      unit.at = spot;
      unit.tag.visible = visible;
      unit.tag.g.setAttribute("transform", `translate(${spot[0].toFixed(1)} ${spot[1].toFixed(1)})`);
      unit.tag.g.classList.toggle("is-hidden", !visible);
      // a pair without its tag draws back: thinner, lighter, under the others
      unit.g.classList.toggle("is-quiet", state.collection === "pairs" && !visible);
    }
  }

  // the stage at rest: shapes, tags, dots, all in place
  function draw() {
    const r = radiusOf(state.collection);
    figure.setAttribute("viewBox", `0 0 ${geo.width} ${geo.height}`);
    figure.setAttribute("height", geo.height);
    placeDots();
    for (const unit of units) atRest(unit, r);
    placeTags();
    showFocus();
  }

  // the units of a collection, with their numbers and tags, ordered so the strongest lies on top
  function build(collection) {
    const c = COLLECTIONS[collection];
    return c.sets
      .map((ids, k) => ({ ids, value: c.values[k] }))
      .sort((a, b) => Math.abs(a.value) - Math.abs(b.value));
  }

  // -- focus: the unit (or, for pairs, the feature) under the pointer --------------------------------

  const distance = (x, y, ids) => {
    const P = geo.P;
    if (ids.length === 1) return Math.hypot(x - P[ids[0]][0], y - P[ids[0]][1]);
    const [[ax, ay], [bx, by]] = [P[ids[0]], P[ids[1]]];
    const t = Math.max(0, Math.min(1, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)));
    return Math.hypot(x - ax - t * (bx - ax), y - ay - t * (by - ay));
  };
  function focusAt(x, y) {
    if (state.collection === "pairs") {
      const k = geo.P.findIndex(([px, py]) => Math.hypot(x - px, y - py) < geo.dot + 9);
      if (k >= 0) return { kind: "feature", k };
    }
    let best = null;
    let near = radiusOf(state.collection) + 10;
    for (const unit of units) {
      const gap = distance(x, y, unit.ids);
      if (gap < near) [best, near] = [unit, gap];
    }
    // tags count as their shapes
    for (const unit of units) {
      if (!unit.tag.visible || !unit.at) continue;
      if (Math.abs(x - unit.at[0]) <= unit.tag.w / 2 && Math.abs(y - unit.at[1]) <= unit.tag.h / 2) best = unit;
    }
    return best ? { kind: "unit", key: best.key } : null;
  }
  const sameFocus = (a, b) => (a && b ? a.kind === b.kind && a.key === b.key && a.k === b.k : a === b);
  function setFocus(next) {
    if (sameFocus(next, focus)) return;
    focus = next;
    if (state.collection === "pairs") placeTags();
    showFocus();
  }
  // the unit whose S ↦ e(S) the row below shows: the one in focus, else the strongest
  const strongest = () => [...units].sort((a, b) => Math.abs(b.value) - Math.abs(a.value))[0];
  function showFocus() {
    const lit = (unit) => !focus || (focus.kind === "unit" ? unit.key === focus.key : unit.ids.includes(focus.k));
    for (const unit of units) {
      const on = lit(unit);
      unit.g.classList.toggle("is-dim", !on);
      unit.g.classList.toggle("is-lit", Boolean(focus) && on);
      unit.tag.g.classList.toggle("is-dim", !on);
    }
    // the units in focus lie on top; without focus, the strongest
    const order = [...units].sort((a, b) => lit(a) - lit(b) || Math.abs(a.value) - Math.abs(b.value));
    for (const unit of order) layers.shapes.appendChild(unit.g);
    const inFocus = focus?.kind === "unit" ? units.find((u) => u.key === focus.key) : null;
    // without focus, the row below reads out the strongest unit: its outline a little darker
    const shown = focus ? null : strongest();
    for (const unit of units) {
      unit.g.classList.toggle("is-default", unit === shown);
      unit.tag.g.classList.toggle("is-focus", unit === inFocus);
    }
    dots.forEach(({ g, label }, k) => {
      const on = !inFocus || inFocus.ids.includes(k);
      g.classList.toggle("is-dim", !on);
      label.classList.toggle("is-dim", !on);
      g.classList.toggle("is-focus", focus?.kind === "feature" && focus.k === k);
    });
    writeMap(inFocus ?? (focus?.kind === "feature" ? null : strongest()), focus?.kind === "feature" ? focus.k : null);
  }

  // the row S ↦ e(S) under the stage
  function writeMap(unit, feature) {
    mapRow.replaceChildren();
    if (feature !== null && feature !== undefined) {
      const mine = units.filter((u) => u.ids.includes(feature)).sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
      const chip = el("span", "lh-set", String(feature + 1));
      mapRow.append(chip, el("span", "lh-map-names", `${features[feature].label}: its ${mine.length} pairs`));
      mapGloss.textContent =
        state.space === "numbers"
          ? `With ${mine.map((u) => `${lower(features[u.ids.find((k) => k !== feature)].label)} ${signed(u.value)}`).join(", ")}.`
          : "Its pure interaction surfaces with each other feature.";
      return;
    }
    if (!unit) return;
    const name = groupName(unit.ids);
    const left = el("span", "lh-map-s");
    left.innerHTML = "<span><i>S</i> =</span>";
    left.appendChild(el("span", "lh-set", setOf(unit.ids)));
    const names = el("span", "lh-map-names", `${state.collection === "groups" && name ? `${name}: ` : ""}${namesOf(unit.ids)}`);
    const right = el("span", "lh-map-e");
    right.innerHTML = '<span class="lh-map-to">↦</span><span><i>e</i>(<i>S</i>) =</span>';
    if (state.space === "numbers") {
      right.appendChild(el("span", `lh-map-value ${signClass(unit.value)}`, signed(unit.value)));
    } else {
      right.appendChild(bigGlyph(unit.ids));
    }
    const unitSide = el("span", "lh-map-unit");
    unitSide.append(left, names);
    mapRow.append(unitSide, right);
    mapGloss.textContent = gloss(state.collection, state.space, unit.ids, unit.value);
  }
  // the curve or surface of a unit, larger, with its ranges
  function bigGlyph(ids) {
    const box = el("span", "lh-map-glyph");
    const narrow = geo.narrow;
    if (ids.length === 1) {
      const [w, h] = narrow ? [120, 40] : [150, 44];
      const node = svg("svg", { width: w, height: h + 14, viewBox: `0 0 ${w} ${h + 14}`, "aria-hidden": "true" });
      drawCurve(node, ids[0], w, h, 4);
      const { x } = data.curves[ids[0]];
      const f = ids[0];
      const lo = svg("text", { x: 4, y: h + 12, class: "lh-axis" }, node);
      lo.textContent = x[0].toFixed(features[f].decimals);
      const hi = svg("text", { x: w - 4, y: h + 12, class: "lh-axis", "text-anchor": "end" }, node);
      hi.textContent = x.at(-1).toFixed(features[f].decimals);
      box.appendChild(node);
      const ys = data.curves[f].y;
      box.appendChild(el("span", "lh-map-range", `${plain(Math.min(...ys))} to ${plain(Math.max(...ys))}`));
      return box;
    }
    const size = narrow ? 44 : 48;
    const node = svg("svg", { width: size, height: size, viewBox: `0 0 ${size} ${size}`, "aria-hidden": "true" });
    const pure = state.collection === "pairs";
    drawSurface(node, ids, pure ? "pure" : "surface", size);
    box.appendChild(node);
    const grid = pure ? data.pure[keyOf(ids)] : data.surfaces[keyOf(ids)];
    const flat = grid.flat();
    const [lo, hi] = [Math.min(...flat), Math.max(...flat)];
    box.appendChild(el("span", "lh-map-range", pure ? `${signed(lo)} to ${signed(hi)}` : `${plain(lo)} to ${plain(hi)}`));
    return box;
  }

  figure.addEventListener("pointermove", (event) => {
    if (pinned && event.pointerType !== "mouse") return;
    const box = figure.getBoundingClientRect();
    const x = ((event.clientX - box.left) / box.width) * geo.width;
    const y = ((event.clientY - box.top) / box.height) * geo.height;
    setFocus(focusAt(x, y));
  });
  figure.addEventListener("pointerdown", (event) => {
    if (event.pointerType === "mouse") return;
    const box = figure.getBoundingClientRect();
    const next = focusAt(((event.clientX - box.left) / box.width) * geo.width, ((event.clientY - box.top) / box.height) * geo.height);
    pinned = Boolean(next);
    setFocus(next);
  });
  figure.addEventListener("pointerleave", (event) => {
    if (event.pointerType === "mouse") setFocus(null);
  });
  // the keys step through the units
  figure.addEventListener("keydown", (event) => {
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
    if (event.key === "Escape") return setFocus(null);
    if (!step) return;
    event.preventDefault();
    const order = units.map((u) => u.key).sort();
    const at = focus?.kind === "unit" ? order.indexOf(focus.key) : -1;
    setFocus({ kind: "unit", key: order[(at + step + order.length) % order.length] });
  });
  figure.addEventListener("blur", () => setFocus(null));

  // -- changes: the shapes morph from one collection to the next --------------------------------------

  // For each new unit, the old one it grows out of: the one that shares the most features with it,
  // a part of it where there is one, and the one that has given the fewest so far (so that the
  // rings spread their pairs evenly). An old unit that no new one grows out of shrinks or merges into
  // the new unit that shares the most with it, and fades.
  function plan(old, next) {
    const share = (a, b) => a.ids.filter((k) => b.ids.includes(k)).length;
    const given = new Map(old.map((u) => [u, 0]));
    const sources = next.map((n) => {
      let best = null;
      let score = -Infinity;
      for (const o of old) {
        const s = share(o, n) * 100 + (o.key === n.key ? 1000 : 0) + (o.ids.every((k) => n.ids.includes(k)) ? 10 : 0) - given.get(o);
        if (s > score) [best, score] = [o, s];
      }
      if (best) given.set(best, given.get(best) + 1);
      return best;
    });
    const fates = old
      .filter((o) => !sources.includes(o))
      .map((o) => {
        let best = next[0];
        let score = -Infinity;
        for (const n of next) {
          const s = share(o, n) * 100 - n.ids.length;
          if (s > score) [best, score] = [n, s];
        }
        return [o, best];
      });
    return { sources, fates };
  }

  function change(next) {
    const before = { ...state };
    Object.assign(state, next);
    focus = null;
    pinned = false;
    stopAll();
    const r = radiusOf(state.collection);
    const moving = !reducedMotion();

    if (state.collection === before.collection) {
      // a new space: the tints turn, and the tags flip over, from numbers to curves or back
      const order = [...units].sort((a, b) => a.ids[0] - b.ids[0] || (a.ids[1] ?? 0) - (b.ids[1] ?? 0));
      for (const unit of units) {
        const old = unit.tag;
        unit.tag = makeTag(state.collection, state.space, unit.ids, unit.value);
        const from = unit.tint;
        const to = tintOf(state.space, unit.value);
        stops.push(tween(moving ? 500 : 0, (p) => paint(unit, mixTint(from, to, p)), { easing: ease.inOut }));
        const delay = 35 * order.indexOf(unit);
        if (!moving || old.g.classList.contains("is-hidden")) {
          old.g.remove();
        } else {
          old.body.animate([{ transform: "scaleX(1)" }, { transform: "scaleX(0)" }], { duration: 170, delay, easing: "ease-in", fill: "forwards" }).onfinish = () => old.g.remove();
        }
        if (moving) {
          unit.tag.body.animate([{ transform: "scaleX(0)" }, { transform: "scaleX(1)" }], { duration: 260, delay: delay + 170, easing: "cubic-bezier(0.2, 0.7, 0.2, 1)", fill: "backwards" });
          drawIn(unit.tag, delay + 300);
        }
      }
      placeTags();
      showFocus();
      writeDefinition(true);
      return;
    }

    // a new collection: the shapes morph (what still fades from a change before goes at once)
    for (const o of dying) o.g.remove();
    const old = units;
    const made = build(state.collection).map(({ ids, value }) => ({ ids, value, key: keyOf(ids) }));
    const { sources, fates } = plan(old, made);
    const kept = new Set();
    units = made.map((m, k) => {
      const source = sources[k];
      const continued = source && !kept.has(source) && sources.indexOf(source) === k;
      const path = source ? pairUp(source.anchorIds, source.anchors, m.ids) : pairUp(m.ids, m.ids.map((i) => geo.P[i]), m.ids);
      const fromR = source ? source.r : 0;
      let unit;
      if (continued) {
        kept.add(source);
        unit = Object.assign(source, { ids: m.ids, key: m.key, value: m.value });
      } else {
        // a new shape grows out of its source, untinted at first
        unit = makeUnit(m.ids, m.value);
        paint(unit, source ? { ...source.tint, mix: 0 } : { ...tintOf(state.space, m.value), mix: 0 });
      }
      Object.assign(unit, { path, fromR });
      unit.fromTint = unit.tint;
      unit.toTint = tintOf(state.space, m.value);
      const oldTag = unit.tag;
      unit.tag = makeTag(state.collection, state.space, m.ids, m.value);
      if (oldTag) retire(oldTag);
      return unit;
    });
    // a shape without a successor slides into the one that takes its place, and fades
    for (const [o, target] of fates) {
      retire(o.tag);
      Object.assign(o, { path: pairUp(o.anchorIds, o.anchors, target.ids), fromR: o.r });
    }
    placeTags();
    for (const unit of units) unit.tag.g.classList.add("is-waiting");
    showFocus();
    writeDefinition(true);
    dying = fates.map(([o]) => o);
    const shapeAt = (unit, p) =>
      setShape(
        unit,
        unit.path.from.map(([x, y], i) => [lerp(x, unit.path.to[i][0], p), lerp(y, unit.path.to[i][1], p)]),
        lerp(unit.fromR, r, p),
        unit.path.ids,
      );
    stops.push(
      tween(
        moving ? 760 : 0,
        (p) => {
          for (const unit of units) {
            shapeAt(unit, p);
            paint(unit, mixTint(unit.fromTint, unit.toTint, p));
          }
          for (const o of dying) {
            shapeAt(o, p);
            o.g.style.opacity = String(1 - p);
          }
        },
        {
          easing: ease.inOut,
          done: () => {
            for (const o of dying) o.g.remove();
            dying = [];
            for (const unit of units) atRest(unit, r);
          },
        },
      ),
    );
    popTags(moving ? 560 : 0);
  }

  // between two tints: through no tint at all where the sign changes (red and blue never mix)
  function mixTint(a, b, p) {
    if (a.sign === b.sign) return { sign: b.sign, mix: a.mix + (b.mix - a.mix) * p };
    return p < 0.5 ? { sign: a.sign, mix: a.mix * (1 - 2 * p) } : { sign: b.sign, mix: b.mix * (2 * p - 1) };
  }
  // an old tag shrinks away
  function retire(tag) {
    if (!tag) return;
    const g = tag.g;
    if (reducedMotion()) return g.remove();
    g.animate([{ opacity: g.classList.contains("is-hidden") ? 0 : 1 }, { opacity: 0 }], { duration: 160, fill: "forwards" }).onfinish = () => g.remove();
  }
  // the new tags pop in, one after another, and their numbers count up
  function popTags(delay) {
    const order = [...units].sort((a, b) => a.ids[0] - b.ids[0] || (a.ids[1] ?? 0) - (b.ids[1] ?? 0));
    order.forEach((unit, k) => {
      const tag = unit.tag;
      tag.g.classList.remove("is-waiting");
      if (reducedMotion()) return;
      tag.body.animate([{ transform: "scale(0.2)", opacity: 0 }, { transform: "scale(1.08)", opacity: 1, offset: 0.7 }, { transform: "scale(1)", opacity: 1 }], {
        duration: 420,
        delay: delay + 40 * k,
        easing: "cubic-bezier(0.2, 0.7, 0.2, 1)",
        fill: "backwards",
      });
      if (tag.label) {
        const v = tag.v;
        stops.push(tween(520, (p) => (tag.label.textContent = signed(v * p)), { delay: delay + 40 * k, easing: ease.out }));
      }
      drawIn(tag, delay + 40 * k + 120);
    });
  }
  // a curve draws itself from left to right
  function drawIn(tag, delay) {
    const length = tag.line?.getTotalLength?.() || 0;
    if (!length || reducedMotion()) return;
    tag.line.style.strokeDasharray = `${length}`;
    tag.line.animate([{ strokeDashoffset: length }, { strokeDashoffset: 0 }], { duration: 600, delay, easing: "ease-out", fill: "backwards" });
  }

  // the definition, filled in by the two choices
  function writeDefinition(flash) {
    const c = COLLECTIONS[state.collection];
    defUnits.innerHTML = `<span class="lh-cal">𝒮</span> = ${c.notation}`;
    defSpace.innerHTML = `<i>E</i> = ${SPACES[state.space].notation}`;
    defMethod.innerHTML = `<i>e</i> = <b>${METHODS[`${state.collection}-${state.space}`]}</b>`;
    if (flash && !reducedMotion()) {
      defMethod.animate([{ backgroundColor: "var(--shapiq-amber-soft)" }, { backgroundColor: "transparent" }], { duration: 900, easing: "ease-out" });
    }
    const adds = state.space === "numbers" && state.collection !== "pairs";
    sum.innerHTML = adds
      ? `They add up: average ${plain(data.baseline)} <span class="${signClass(total)}">${total < 0 ? "−" : "+"} ${plain(Math.abs(total))}</span> = this patient ${plain(data.prediction)}`
      : state.collection === "pairs"
        ? "Point at (or tap) a feature for all its pairs."
        : "";
    figure.setAttribute(
      "aria-label",
      `${c.label}: ${units.map((u) => `${setOf(u.ids)} ${namesOf(u.ids)}${state.space === "numbers" ? ` ${signed(u.value)}` : ""}`).join("; ")}`,
    );
  }

  // -- the start: the dots, then the shapes around them, then their explanations ----------------------

  geo = geometry(Math.max(260, root.clientWidth));
  units = build(state.collection).map(({ ids, value }) => {
    const unit = makeUnit(ids, value);
    unit.tag = makeTag(state.collection, state.space, ids, value);
    paint(unit, tintOf(state.space, value));
    return unit;
  });
  unitsSwitch.set(state.collection);
  spaceSwitch.set(state.space);
  writeDefinition(false);
  draw();

  let seen = false;
  let lastWidth = root.clientWidth;
  new ResizeObserver(() => {
    const width = root.clientWidth;
    if (Math.abs(width - lastWidth) < 2) return;
    lastWidth = width;
    stopAll();
    const narrow = geo.narrow;
    geo = geometry(Math.max(260, width));
    if (narrow !== geo.narrow) {
      // the tags change size with the stage
      for (const unit of units) {
        unit.tag.g.remove();
        unit.tag = makeTag(state.collection, state.space, unit.ids, unit.value);
      }
    }
    for (const unit of units) paint(unit, tintOf(state.space, unit.value));
    for (const node of [...layers.shapes.children]) {
      if (!units.some((u) => u.g === node)) node.remove();
    }
    dying = [];
    draw();
  }).observe(root);

  function intro() {
    if (reducedMotion()) return;
    const r = geo.r;
    dots.forEach(({ inner, label }, k) => {
      inner.animate([{ transform: "scale(0)" }, { transform: "scale(1.15)", offset: 0.6 }, { transform: "scale(1)" }], {
        duration: 520,
        delay: 80 * k,
        easing: "cubic-bezier(0.2, 0.7, 0.2, 1)",
        fill: "backwards",
      });
      label.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 400, delay: 80 * k + 150, fill: "backwards" });
    });
    // the rings grow out of the dots
    const start = 650;
    [...units].sort((a, b) => a.ids[0] - b.ids[0]).forEach((unit, k) => {
      const tint = unit.tint;
      const anchors = unit.ids.map((k) => geo.P[k]);
      setShape(unit, anchors, 0, unit.ids);
      paint(unit, { ...tint, mix: 0 });
      stops.push(
        tween(
          640,
          (p) => {
            setShape(unit, anchors, r * p);
            paint(unit, { ...tint, mix: tint.mix * Math.max(0, Math.min(1, (p - 0.4) / 0.6)) });
          },
          { delay: start + 70 * k, easing: ease.spring },
        ),
      );
    });
    for (const node of [defUnits, defSpace, defMethod]) node.style.opacity = "0";
    defUnits.animate([{ opacity: 0, transform: "translateY(4px)" }, { opacity: 1, transform: "none" }], { duration: 400, delay: start, fill: "forwards" }).onfinish = () => (defUnits.style.opacity = "");
    for (const node of [defSpace, defMethod]) {
      node.animate([{ opacity: 0, transform: "translateY(4px)" }, { opacity: 1, transform: "none" }], { duration: 400, delay: start + 750, fill: "forwards" }).onfinish = () => (node.style.opacity = "");
    }
    popTags(start + 650);
    // and the row S ↦ e(S) under it, when the tags are there
    for (const node of [map, sum]) {
      node.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 500, delay: start + 650 + 40 * units.length + 250, fill: "backwards" });
    }
  }

  root.classList.add("is-waiting");
  let stopWatching = null;
  stopWatching = watchVisibility(stage, (visible) => {
    if (!visible || seen) return;
    seen = true;
    root.classList.remove("is-waiting");
    intro();
    stopWatching?.();
  });
});
