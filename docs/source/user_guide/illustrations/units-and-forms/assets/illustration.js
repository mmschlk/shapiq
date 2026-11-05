// The illustration "units-and-forms" (chapter 2.1 of the user guide): Definition 2.1,
// e : 𝒮 → E, at work on one house of the California housing data. Three cards name the three parts
// of the definition: the units 𝒮 (single features, groups, or pairs, a switch), the explanation
// space E (numbers or curves, a switch), and the explanation e that the two choices give (Shapley
// values, ICE curves, ...). Under them, the explanation itself: one row per unit S of 𝒮, the unit on
// the left and its explanation e(S) on the right.
//
// - single features: their Shapley values; their ICE curves (the prediction as one feature varies);
// - groups, the house's location (latitude and longitude) as one unit and the other features alone:
//   the Shapley values of the units, each unit one player; the curves of the single features and
//   the surface of the location (the prediction as both vary);
// - pairs: the pairwise Shapley interactions (k-SII); the pure interaction surfaces (what a pair does
//   together beyond what each feature does alone).
//
// Values are computed once by ../compute/compute.py into data.js; red is positive and blue negative,
// amber marks the features (docs/STYLE_GUIDE.md).
window.shapiqIllustrations?.define("units-and-forms", async (stage, kit) => {
  const { el, svg, tween, ease, reducedMotion, segmented, loadScript } = kit;
  // the data that the page loaded before this script, or, where it is missing, loaded here
  if (!window.SHAPIQ_UNITS_AND_FORMS) await loadScript(`${kit.root}data.js?v=${kit.version}`);
  const data = window.SHAPIQ_UNITS_AND_FORMS;
  const features = data.features;
  const n = features.length;

  // -- numbers: values are in $100,000, shown in whole $1,000s ---------------------------------------

  const value = (k, v = features[k].value) => v.toFixed(features[k].decimals).replace("-", "−");
  const thousands = (v) => Math.round(v * 100);
  const price = (v) => `$${thousands(v)}k`;
  const signedK = (k) => `${k >= 0 ? "+" : "−"}$${Math.abs(k)}k`;
  const signed = (v) => signedK(thousands(v));
  const sign = (v) => (v >= 0 ? "uaf-pos" : "uaf-neg");
  const keyOf = (ids) => [...ids].sort((a, b) => a - b).join("-");
  // the names of a unit's features; short, as in the table of the vocabulary, for the many pairs
  const names = (ids, short = false) => ids.map((k) => features[k][short ? "short" : "label"]).join(", ");
  // the prediction minus the average, as the difference of the shown prices, so that a line adds up
  const total = thousands(data.prediction) - thousands(data.baseline);
  // values in whole $1,000s that add up to `sum`: each rounded down, and the rest given to those
  // that lost the most (or, the other way, taken from those that lost the least)
  const roundTo = (values, sum) => {
    const raw = values.map((v) => v * 100);
    const shown = raw.map(Math.floor);
    const order = raw.map((v, k) => [v - Math.floor(v), k]).sort((a, b) => b[0] - a[0]);
    let rest = sum - shown.reduce((a, b) => a + b, 0);
    for (let j = 0; rest > 0; j++, rest--) shown[order[j % order.length][1]] += 1;
    for (let j = order.length - 1; rest < 0; j--, rest++) shown[order[(j + order.length) % order.length][1]] -= 1;
    return shown;
  };

  // -- the two choices, and the explanation they give ------------------------------------------------

  const pairs = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) pairs.push([i, j]);
  const setOf = (ids) => `{${ids.map((k) => k + 1).join(", ")}}`;
  const UNITS = {
    features: {
      label: "Features",
      sets: features.map((_, k) => [k]),
      notation: `{{1}, …, {${n}}}`,
      gloss: "each feature alone",
    },
    groups: {
      label: "Groups",
      sets: data.partition,
      notation: `{${data.partition.map(setOf).join(", ")}}`,
      gloss: "the location, latitude and longitude, as one unit",
    },
    pairs: {
      label: "Pairs",
      sets: pairs,
      notation: "{{<i>i</i>, <i>j</i>} : <i>i</i> &lt; <i>j</i>}",
      gloss: `every pair of features, ${pairs.length} units`,
    },
  };
  const SPACES = {
    numbers: { label: "Numbers", notation: "ℝ", gloss: "a number per unit" },
    curves: {
      label: "Curves",
      notation: "curves",
      gloss: "a curve per unit, a surface for two features",
    },
  };
  const METHODS = {
    "features-numbers": [
      "Shapley values",
      "How much each feature moves this house's price away from the average.",
    ],
    "features-curves": [
      "ICE curves",
      "The prediction for this house as one feature varies and the others stay, all on one scale. The dot is the house.",
    ],
    "groups-numbers": [
      "Shapley values of the units",
      "Each unit one player: the location's value is what latitude and longitude do as one.",
    ],
    "groups-curves": [
      "ICE curves and a surface",
      "The location, two features, has a surface: the prediction as latitude and longitude vary together. The dot is the house.",
    ],
    "pairs-numbers": [
      "Shapley interactions (k\u2011SII)",
      "What the two features of a pair do together beyond what each does alone.",
    ],
    "pairs-curves": [
      "pure interaction surfaces",
      "For each pair, the change in the prediction as both vary together, minus the changes each makes alone, on one scale: red raises the price, blue lowers it.",
    ],
  };
  const state = { units: "features", space: "numbers" };

  // -- the layout: the three parts, then the explanation, one row per unit ---------------------------

  const root = el("div", "uaf");
  const parts = el("div", "uaf-parts");
  const card = (cls, symbol, name) => {
    const node = el("div", `uaf-part ${cls}`);
    const head = el("p", "uaf-part-head");
    head.innerHTML = `<span class="uaf-part-symbol">${symbol}</span><span class="uaf-part-name">${name}</span>`;
    node.appendChild(head);
    parts.appendChild(node);
    return node;
  };
  const unitsCard = card("uaf-part-units", '<span class="uaf-cal">𝒮</span>', "the units");
  const spaceCard = card("uaf-part-space", "<i>E</i>", "the explanation space");
  const explCard = card("uaf-part-expl", "<i>e</i>", "the explanation");
  const choose = (key, id, byUser) => {
    if (!byUser) return;
    state[key] = id;
    render({ changed: key });
  };
  const unitsSwitch = segmented("Units", Object.entries(UNITS).map(([id, u]) => [id, u.label]), (id, byUser) =>
    choose("units", id, byUser),
  );
  const spaceSwitch = segmented("Explanation space", Object.entries(SPACES).map(([id, s]) => [id, s.label]), (id, byUser) =>
    choose("space", id, byUser),
  );
  const unitsValue = el("p", "uaf-part-value");
  const unitsGloss = el("p", "uaf-part-gloss");
  unitsCard.append(unitsSwitch.node, unitsValue, unitsGloss);
  const spaceValue = el("p", "uaf-part-value");
  const spaceGloss = el("p", "uaf-part-gloss");
  spaceCard.append(spaceSwitch.node, spaceValue, spaceGloss);
  const explValue = el("p", "uaf-part-value");
  explValue.innerHTML = '<i>e</i> : <span class="uaf-cal">𝒮</span> → <i>E</i>, &nbsp;<i>S</i> ↦ <i>e</i>(<i>S</i>)';
  const explMethod = el("p", "uaf-part-method");
  explCard.append(explValue, explMethod);

  const map = el("div", "uaf-map");
  const rows = el("div", "uaf-rows");
  rows.setAttribute("role", "list");
  map.appendChild(rows);
  const equation = el("p", "uaf-equation");
  const readout = el("p", "uaf-readout");
  readout.setAttribute("aria-live", "polite");
  const note = el("p", "uaf-note");
  const foot = el("p", "uaf-foot");
  foot.textContent =
    `A gradient boosting model on six features of the California housing data ` +
    `(R² ${data.r2.toFixed(2)} on held-out houses) predicts ${price(data.prediction)} for this ` +
    `house, against ${price(data.baseline)} on average over ${data.background} houses of the ` +
    `training data.`;
  root.append(parts, map, equation, readout, note, foot);
  stage.appendChild(root);

  // -- the explanation of one unit: a bar, a curve, or a surface ----------------------------------------

  const text = (parent, x, y, content, cls, anchor = "start") => {
    const node = svg("text", { x, y, class: cls, "text-anchor": anchor }, parent);
    node.textContent = content;
    return node;
  };
  let stops = [];
  let entering = false; // a view enters with an animation; a resize redraws it in place
  const animate = (duration, update, options = {}) => {
    const still = !entering || reducedMotion();
    stops.push(tween(still ? 0 : duration, update, still ? {} : options));
  };
  // a readout of what the pointer is on; a tap shows it too, and it stays until the next
  const readFrom = (target, describe) => {
    target.onpointermove = (event) => {
      const words = describe(event);
      if (words) readout.textContent = words;
    };
    target.onpointerdown = target.onpointermove;
    target.onpointerleave = (event) => {
      if (event.pointerType === "mouse") readout.textContent = "";
    };
  };

  // a bar from zero, on the scale of all the rows, with its value beside it
  function drawNumber(node, width, v, shown, scale, k) {
    const height = 26;
    node.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const zero = scale(0);
    svg("line", { x1: zero, x2: zero, y1: 0, y2: height, class: "uaf-zero" }, node);
    const bar = svg("rect", { y: 7, height: 12, rx: 3, class: `uaf-bar ${sign(v)}` }, node);
    const label = text(node, 0, 17.5, signedK(shown), `uaf-value ${sign(v)}`, v >= 0 ? "start" : "end");
    animate(
      700,
      (p) => {
        const end = scale(v * p);
        bar.setAttribute("x", Math.min(zero, end));
        bar.setAttribute("width", Math.abs(end - zero));
        label.setAttribute("x", v >= 0 ? end + 6 : end - 6);
        label.style.opacity = p;
      },
      { easing: ease.spring, delay: 45 * k },
    );
  }

  // the house's prediction as one feature varies, on the scale of all the curves; the dot is the house
  const allCurves = data.curves.flatMap((c) => c.y);
  const curveLo = Math.min(...allCurves) * 0.97;
  const curveHi = Math.max(...allCurves) * 1.03;
  function drawCurve(node, width, f, k) {
    const height = 40;
    const pad = 5;
    node.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const { x: xs, y: ys } = data.curves[f];
    const x = (v) => pad + ((v - xs[0]) / (xs.at(-1) - xs[0])) * (width - 2 * pad);
    const y = (v) => height - 2 - ((v - curveLo) / (curveHi - curveLo)) * (height - 4);
    const points = xs.map((v, i) => [x(v), y(ys[i])]);
    const d = points.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)} ${py.toFixed(1)}`).join("");
    svg("path", { d: `${d}L${points.at(-1)[0]} ${height}L${points[0][0]} ${height}Z`, class: "uaf-area" }, node);
    const line = svg("path", { d, class: "uaf-line" }, node);
    const dot = svg("circle", { cx: x(features[f].value), cy: y(data.prediction), r: 3.5, class: "uaf-marker-dot" }, node);
    if (entering && !reducedMotion()) {
      const length = line.getTotalLength();
      line.style.strokeDasharray = `${length}`;
      line.animate([{ strokeDashoffset: length }, { strokeDashoffset: 0 }], {
        duration: 700,
        delay: 45 * k,
        easing: "cubic-bezier(0.4, 0, 0.2, 1)",
        fill: "backwards",
      });
      dot.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300, delay: 45 * k + 550, fill: "backwards" });
    }
    readFrom(node, (event) => {
      const box = node.getBoundingClientRect();
      const at = xs[0] + (((event.clientX - box.left) / box.width) * width - pad) / (width - 2 * pad) * (xs.at(-1) - xs[0]);
      let i = 0; // the point nearest to the pointer
      xs.forEach((v, j) => {
        if (Math.abs(v - at) < Math.abs(xs[i] - at)) i = j;
      });
      return `${features[f].label} ${value(f, xs[i])}: ${price(ys[i])}`;
    });
  }

  // the prediction (kind "surface", amber) or the pure interaction (kind "pure", red and blue) as two
  // features vary, a small map: the first feature up, the second to the right; the dot is the house
  function drawSurface(node, width, [i, j], kind, reach, k) {
    const height = 48;
    node.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const key = `${i}-${j}`;
    const surface = data.surfaces[key];
    const grid = kind === "pure" ? data.pure[key] : surface.z; // grid[a][b]: feature i at a, j at b
    const cells = grid.length;
    const size = height;
    const cell = size / cells;
    const flat = grid.flat();
    const lo = Math.min(...flat);
    const hi = Math.max(...flat);
    const fill = (z) =>
      kind === "pure"
        ? `color-mix(in oklab, var(${z >= 0 ? "--shapiq-red" : "--shapiq-blue"}) ${Math.round((Math.abs(z) / reach) * 100)}%, var(--shapiq-surface))`
        : `color-mix(in oklab, var(--shapiq-amber) ${Math.round(8 + ((z - lo) / (hi - lo)) * 92)}%, var(--shapiq-surface))`;
    const heat = svg("g", { class: "uaf-heat" }, node);
    grid.forEach((row, a) => {
      row.forEach((z, b) => {
        const rect = svg("rect", { x: b * cell, y: (cells - 1 - a) * cell, width: cell + 0.4, height: cell + 0.4, class: "uaf-cell" }, heat);
        rect.style.fill = fill(z);
        rect.dataset.a = a;
        rect.dataset.b = b;
      });
    });
    svg("rect", { x: 0, y: 0, width: size, height: size, class: "uaf-frame" }, node);
    const at = (v, axis) => ((v - axis[0]) / (axis.at(-1) - axis[0])) * (size - cell) + cell / 2;
    svg("circle", { cx: at(features[j].value, surface.y), cy: size - at(features[i].value, surface.x), r: 2.6, class: "uaf-marker-dot" }, node);
    // its range beside it, on one line where there is room (with which feature runs which way, where
    // there is more), else on two
    const [low, high] = kind === "pure" ? [signed(lo), signed(hi)] : [price(lo), price(hi)];
    const room = width - size - 10;
    const axes = `${features[i].label.toLowerCase()} up, ${features[j].label.toLowerCase()} across: `;
    if (room >= 130) {
      text(node, size + 10, height / 2 + 4, `${kind === "surface" && room >= 250 ? axes : ""}from ${low} to ${high}`, "uaf-surface-label");
    } else if (room >= 44) {
      text(node, size + 8, height / 2 - 3, `${low} to`, "uaf-surface-label");
      text(node, size + 8, height / 2 + 11, high, "uaf-surface-label");
    }
    if (entering && !reducedMotion()) {
      heat.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 500, delay: 45 * k, fill: "backwards" });
    }
    readFrom(heat, (event) => {
      const target = event.target.closest("rect");
      if (target?.dataset.a === undefined) return "";
      const [a, b] = [Number(target.dataset.a), Number(target.dataset.b)];
      const z = grid[a][b];
      return (
        `${features[i].label} ${value(i, surface.x[a])}, ${features[j].label.toLowerCase()} ` +
        `${value(j, surface.y[b])}: ${kind === "pure" ? signed(z) : price(z)}`
      );
    });
  }

  // -- the explanation: one row per unit -------------------------------------------------------------

  // the numbers of a view, and the values shown, which add up where the explanation adds up
  const numbersOf = (units) => {
    if (units === "features") return [data.shapley, roundTo(data.shapley, total)];
    if (units === "groups") return [data.partition_shapley, roundTo(data.partition_shapley, total)];
    const values = pairs.map((ids) => data.interactions[keyOf(ids)]);
    return [values, values.map(thousands)];
  };

  // a unit: its set of features, by their numbers, and their names
  const unitCell = (ids, short) => {
    const cell = el("div", "uaf-unit");
    cell.append(el("span", "uaf-set", setOf(ids)), el("span", "uaf-unit-names", names(ids, short)));
    return cell;
  };

  let lastWidth = 0;
  function render({ changed = null, animated = true } = {}) {
    for (const stop of stops) stop();
    stops = [];
    const key = `${state.units}-${state.space}`;
    const units = UNITS[state.units];
    const space = SPACES[state.space];
    const [method, words] = METHODS[key];
    entering = animated && changed !== "resize";

    // the three parts
    unitsValue.innerHTML = `<span class="uaf-cal">𝒮</span> = ${units.notation}`;
    unitsGloss.textContent = units.gloss;
    spaceValue.innerHTML = `<i>E</i> = ${space.notation}`;
    spaceGloss.textContent = space.gloss;
    explMethod.textContent = method;
    note.textContent = words;
    if (changed && changed !== "resize" && !reducedMotion()) {
      for (const node of [changed === "units" ? unitsCard : spaceCard, explCard]) {
        node.classList.remove("is-changed");
        void node.offsetWidth; // to start the flash again
        node.classList.add("is-changed");
      }
    }

    // the rows: the unit, ↦, and its explanation
    const before = map.getBoundingClientRect().height;
    const columns = state.units === "pairs" && root.clientWidth >= 640 ? 2 : 1;
    rows.style.setProperty("--uaf-columns", columns);
    rows.classList.toggle("uaf-rows-pairs", state.units === "pairs");
    rows.replaceChildren();
    readout.textContent = "";
    for (let c = 0; c < columns; c++) {
      const head = el("div", "uaf-row uaf-row-head");
      head.setAttribute("aria-hidden", "true");
      head.innerHTML =
        '<span><i>S</i> ∈ <span class="uaf-cal">𝒮</span></span><span class="uaf-to"></span>' +
        "<span><i>e</i>(<i>S</i>) ∈ <i>E</i></span>";
      rows.appendChild(head);
    }
    const [numbers, shown] = numbersOf(state.units);
    const made = units.sets.map((ids, k) => {
      const row = el("div", "uaf-row");
      row.setAttribute("role", "listitem");
      const graphic = svg("svg", { class: "uaf-e", "aria-hidden": "true" });
      row.append(unitCell(ids, state.units === "pairs"), el("span", "uaf-to", "↦"), graphic);
      if (state.space === "numbers") {
        row.setAttribute("aria-label", `${names(ids)}: ${signedK(shown[k])}`);
        row.addEventListener("pointerenter", () => {
          readout.textContent = `${names(ids)}: ${signedK(shown[k])}`;
        });
      } else {
        row.setAttribute(
          "aria-label",
          ids.length === 1
            ? `${names(ids)}: the prediction from ${price(Math.min(...data.curves[ids[0]].y))} to ${price(Math.max(...data.curves[ids[0]].y))} as it varies`
            : `${names(ids)}: a surface, as both vary`,
        );
      }
      rows.appendChild(row);
      return [row, graphic, ids];
    });
    const width = Math.max(120, Math.round(made[0][1].getBoundingClientRect().width));
    lastWidth = root.clientWidth;
    if (state.space === "numbers") {
      const lo = Math.min(0, ...numbers);
      const hi = Math.max(0, ...numbers);
      const x0 = lo < 0 ? 52 : 4;
      const x1 = width - (hi > 0 ? 54 : 4);
      const scale = (v) => x0 + ((v - lo) / (hi - lo)) * (x1 - x0);
      made.forEach(([, graphic], k) => drawNumber(graphic, width, numbers[k], shown[k], scale, k));
    } else {
      const reach = Math.max(...pairs.flatMap((ids) => data.pure[keyOf(ids)].flat().map(Math.abs)));
      made.forEach(([, graphic, ids], k) => {
        if (ids.length === 1) drawCurve(graphic, width, ids[0], k);
        else drawSurface(graphic, width, ids, state.units === "pairs" ? "pure" : "surface", reach, k);
      });
    }
    if (entering && !reducedMotion()) {
      made.forEach(([row], k) => {
        row.animate([{ opacity: 0, transform: "translateX(-6px)" }, { opacity: 1, transform: "none" }], {
          duration: 320,
          delay: 30 * k,
          easing: "cubic-bezier(0.2, 0.7, 0.2, 1)",
          fill: "backwards",
        });
      });
      const after = map.getBoundingClientRect().height;
      if (before && Math.abs(after - before) > 1) {
        map.animate([{ height: `${before}px` }, { height: `${after}px` }], {
          duration: 320,
          easing: "cubic-bezier(0.2, 0.7, 0.2, 1)",
        });
      }
    }

    // where the explanation adds up, the line that adds it up
    equation.innerHTML =
      state.space === "numbers" && state.units !== "pairs"
        ? `They add up: average ${price(data.baseline)} ` +
          `<span class="${total >= 0 ? "uaf-pos" : "uaf-neg"}">${total >= 0 ? "+" : "−"} $${Math.abs(total)}k</span> ` +
          `= this house ${price(data.prediction)}`
        : "";
  }

  unitsSwitch.set("features");
  spaceSwitch.set("numbers");
  new ResizeObserver(() => {
    if (seen && Math.abs(root.clientWidth - lastWidth) > 2) render({ changed: "resize" });
  }).observe(root);

  // the explanation enters when the illustration is first seen; until then, it waits unseen
  let seen = false;
  render({ changed: "resize" });
  map.style.visibility = "hidden";
  let stopWatching = null;
  stopWatching = kit.watchVisibility(stage, (visible) => {
    if (visible && !seen) {
      seen = true;
      map.style.visibility = "";
      render();
      stopWatching?.();
    }
  });
});
