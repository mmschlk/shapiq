// The illustration "units-and-forms" (chapter 2.1 of the user guide): Definition 2.1 at work.
// A feature-based explanation picks its units (single features, a group of features, or pairs of
// features that interact) and its explanation space (a number or a curve). The two controls make
// these two choices for one house of the California housing data, and the view shows the
// explanation that they give:
//
// - single features and numbers: the Shapley value of each feature, a bar from zero;
// - single features and curves: the house's prediction as one feature varies (its ICE curve);
// - a group and a number: the Shapley value of the group taken as one player, beside the values of
//   its features one by one;
// - a group and a curve: the prediction as the two features of the group vary, a surface;
// - interactions and numbers: the pairwise Shapley interactions (k-SII), a network of the features;
// - interactions and curves: the part of a pair's surface that neither feature explains alone.
//
// The chips of the house's features pick the feature, the group, or the pair. Values are computed
// once by ../compute/compute.py into data.js; red is positive and blue negative, amber marks the
// features (docs/STYLE_GUIDE.md).
window.shapiqIllustrations?.define("units-and-forms", async (stage, kit) => {
  const { el, svg, tween, ease, reducedMotion, segmented, loadScript } = kit;
  // the data that the page loaded before this script, or, where it is missing, loaded here
  if (!window.SHAPIQ_UNITS_AND_FORMS) await loadScript(`${kit.root}data.js?v=${kit.version}`);
  const data = window.SHAPIQ_UNITS_AND_FORMS;
  const features = data.features;
  const n = features.length;

  // values are in $100,000, shown in whole $1,000s
  const value = (k, v = features[k].value) => v.toFixed(features[k].decimals).replace("-", "−");
  const thousands = (v) => Math.round(v * 100);
  const price = (v) => `$${thousands(v)}k`;
  const signedK = (k) => `${k >= 0 ? "+" : "−"}$${Math.abs(k)}k`;
  const signed = (v) => signedK(thousands(v));
  // a term of a sum of whole $1,000s, in its color: the first with its sign, the next ones with
  // their operation, as in +$157k + $25k − $4k
  const term = (k, first = false) =>
    `<span class="${k >= 0 ? "uaf-pos" : "uaf-neg"}">` +
    `${first ? signedK(k) : `${k >= 0 ? "+" : "−"} $${Math.abs(k)}k`}</span>`;
  const terms = (ks) => ks.map((k, j) => term(k, j === 0)).join(" ");
  const sign = (v) => (v >= 0 ? "uaf-pos" : "uaf-neg");
  const keyOf = (ids) => [...ids].sort((a, b) => a - b).join("-");
  const names = (ids) => [...ids].sort((a, b) => a - b).map((k) => features[k].label).join(", ");
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const lerp = (a, b, t) => a + (b - a) * t;

  // the pair of features it starts with, for the group and the interactions: the house's location
  const state = { units: "features", form: "numbers", feature: 0, group: new Set(data.pair), pair: [...data.pair] };

  // -- the layout ----------------------------------------------------------------------------------

  const root = el("div", "uaf");
  const controls = el("div", "uaf-controls");
  const control = (label, options, key) => {
    const wrap = el("div", "uaf-control");
    const text = el("span", "uaf-control-label");
    text.innerHTML = label;
    const seg = segmented(label.replace(/<[^>]+>/g, ""), options, (v) => {
      state[key] = v;
      render();
    });
    wrap.append(text, seg.node);
    controls.appendChild(wrap);
    return seg;
  };
  const unitsControl = control(
    'units <span class="uaf-cal">𝒮</span>',
    [["features", "Single features"], ["groups", "A group"], ["interactions", "Interactions"]],
    "units",
  );
  const formControl = control(
    "space <i>E</i>",
    [["numbers", "Numbers"], ["curves", "Curves"]],
    "form",
  );

  const body = el("div", "uaf-body");
  const side = el("div", "uaf-side");
  side.appendChild(el("p", "uaf-column-label", "the house"));
  const chips = el("div", "uaf-chips");
  const chipNodes = features.map((feature, k) => {
    const chip = el("button", "shapiq-ill-chip uaf-chip");
    chip.type = "button";
    chip.appendChild(el("span", "shapiq-ill-chip-index", String(k + 1)));
    const text = el("span", "shapiq-ill-chip-text", feature.label);
    text.append(" ", el("span", "shapiq-ill-chip-value", value(k)));
    chip.appendChild(text);
    chip.addEventListener("click", () => pick(k));
    chip.addEventListener("pointerenter", () => highlight([k]));
    chip.addEventListener("pointerleave", () => highlight(null));
    chips.appendChild(chip);
    return chip;
  });
  const hint = el("p", "uaf-hint");
  side.append(chips, hint);

  const view = el("div", "uaf-view");
  const title = el("div", "uaf-title");
  const chartBox = el("div", "uaf-chart");
  const chart = svg("svg", { class: "uaf-svg", role: "img" });
  chartBox.appendChild(chart);
  // what a view cannot draw, in words
  const message = el("p", "uaf-message");
  message.hidden = true;
  // the sum a view adds up, and the value pointed at, announced as they change
  const equation = el("p", "uaf-equation");
  equation.setAttribute("aria-live", "polite");
  const readout = el("p", "uaf-readout");
  readout.setAttribute("aria-live", "polite");
  const note = el("p", "uaf-note");
  view.append(title, chartBox, message, equation, readout, note);
  body.append(side, view);

  const foot = el("p", "uaf-foot");
  foot.textContent =
    `A gradient boosting model on six features of the California housing data ` +
    `(R² ${data.r2.toFixed(2)} on held-out houses) predicts ${price(data.prediction)} for this ` +
    `house, against ${price(data.baseline)} on average over ${data.background} houses of the ` +
    `training data.`;
  root.append(controls, body, foot);
  stage.appendChild(root);

  // -- picking features --------------------------------------------------------------------------

  // a group keeps at least one feature: the last one shakes its head
  const pick = (k) => {
    if (state.units === "features") {
      if (state.feature === k) return;
      state.feature = k;
    } else if (state.units === "groups") {
      if (state.group.has(k)) {
        if (state.group.size === 1) {
          if (!reducedMotion()) {
            chipNodes[k].animate(
              [{ transform: "translateX(0)" }, { transform: "translateX(-3px)" }, { transform: "translateX(3px)" }, { transform: "translateX(0)" }],
              { duration: 260 },
            );
          }
          return;
        }
        state.group.delete(k);
      } else state.group.add(k);
    } else {
      if (state.pair.includes(k)) return;
      state.pair = [state.pair[1], k];
    }
    render();
  };

  const selection = () =>
    state.units === "features" ? [state.feature] : state.units === "groups" ? [...state.group] : state.pair;

  // lights up chips (and what the view draws for them) while pointing; null restores the selection
  let pointing = null;
  const highlight = (ids) => {
    pointing = ids;
    const lit = new Set(ids ?? []);
    chipNodes.forEach((chip, k) => chip.classList.toggle("is-active", lit.has(k)));
    for (const node of chart.querySelectorAll("[data-ids]")) {
      const own = node.dataset.ids.split("-").map(Number);
      const on = own.length === 1 ? lit.has(own[0]) : lit.size === 2 ? own.every((k) => lit.has(k)) : own.some((k) => lit.has(k));
      node.classList.toggle("is-lit", ids !== null && on);
    }
    chart.classList.toggle("uaf-pointing", ids !== null);
  };

  // -- drawing helpers ---------------------------------------------------------------------------

  // A view enters with an animation when it is shown; a new selection, or a resize, redraws it in
  // place (only a curve morphs into the next).
  let stops = [];
  let entering = false;
  const still = () => !entering || reducedMotion();
  const animate = (duration, update, options) =>
    stops.push(tween(still() ? 0 : duration, update, still() ? {} : options));
  const morphing = (duration, update, options) =>
    stops.push(tween(reducedMotion() ? 0 : duration, update, reducedMotion() ? {} : options));
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
  const text = (parent, x, y, content, cls, anchor = "start") => {
    const node = svg("text", { x, y, class: cls, "text-anchor": anchor }, parent);
    node.textContent = content;
    return node;
  };
  const niceTicks = (lo, hi, count) => {
    const step = 10 ** Math.floor(Math.log10((hi - lo) / count));
    const err = (hi - lo) / count / step;
    const nice = step * (err >= 5 ? 10 : err >= 2 ? 5 : err >= 1.5 ? 2 : 1);
    const ticks = [];
    for (let t = Math.ceil(lo / nice) * nice; t <= hi + 1e-9; t += nice) ticks.push(t);
    ticks.decimals = Math.max(0, -Math.floor(Math.log10(nice) + 1e-9));
    return ticks;
  };
  const ticksOf = (lo, hi, count) => {
    const ticks = niceTicks(lo, hi, count);
    return ticks.map((t) => [t, t.toFixed(ticks.decimals).replace("-", "−")]);
  };

  // -- the views ---------------------------------------------------------------------------------

  function drawBars(width) {
    const values = data.shapley;
    const labelWidth = Math.min(122, width * 0.34);
    const lo = Math.min(0, ...values);
    const hi = Math.max(0, ...values);
    const x0 = labelWidth + 56; // room for the labels of negative values
    const x1 = width - 62; // and of positive ones
    const x = (v) => x0 + ((v - lo) / (hi - lo)) * (x1 - x0);
    const row = 30;
    const height = n * row + 10;
    chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg("line", { x1: x(0), x2: x(0), y1: 4, y2: n * row + 4, class: "uaf-zero" }, chart);
    values.forEach((v, k) => {
      const y = 6 + k * row;
      const group = svg("g", { class: "uaf-row", "data-ids": String(k) }, chart);
      svg("rect", { x: 0, y: y - 2, width, height: row - 4, rx: 7, class: "uaf-row-back" }, group);
      text(group, 8, y + row / 2 - 2, features[k].label, "uaf-label");
      const bar = svg("rect", { y: y + 5, height: row - 14, rx: 3, class: `uaf-bar ${sign(v)}` }, group);
      const label = text(group, 0, y + row / 2 - 2, signed(v), `uaf-value ${sign(v)}`, v >= 0 ? "start" : "end");
      animate(
        700,
        (p) => {
          const end = x(v * p);
          bar.setAttribute("x", Math.min(x(0), end));
          bar.setAttribute("width", Math.abs(end - x(0)));
          label.setAttribute("x", v >= 0 ? end + 6 : end - 6);
          label.style.opacity = p;
        },
        { easing: ease.spring, delay: 50 * k },
      );
      group.addEventListener("pointerenter", () => highlight([k]));
      group.addEventListener("pointerleave", () => highlight(null));
      group.addEventListener("click", () => {
        if (state.feature === k) return;
        state.feature = k;
        render();
      });
      if (k === state.feature) group.classList.add("is-selected");
    });
    // the Shapley values add up to the prediction minus the average, shown as the difference of
    // the shown prices, so that the line adds up
    const total = thousands(data.prediction) - thousands(data.baseline);
    equation.innerHTML = `average ${price(data.baseline)} ${term(total)} = this house ${price(data.prediction)}`;
    chart.setAttribute(
      "aria-label",
      `Bars of the Shapley values of the features: ${features.map((f, k) => `${f.label} ${signed(values[k])}`).join(", ")}.`,
    );
    return "The Shapley values of the features: how much each moves this house's price away from the average.";
  }

  let lastCurve = null; // the points of the curve on screen, to morph from
  function drawCurve(width, k, morph) {
    const height = 250;
    const m = { left: 50, right: 16, top: 14, bottom: 42 };
    const curve = data.curves[k];
    const all = data.curves.flatMap((c) => c.y);
    const yLo = Math.min(...all) * 0.96;
    const yHi = Math.max(...all) * 1.03;
    const xs = curve.x;
    const x = (v) => m.left + ((v - xs[0]) / (xs.at(-1) - xs[0])) * (width - m.left - m.right);
    const y = (v) => height - m.bottom - ((v - yLo) / (yHi - yLo)) * (height - m.top - m.bottom);
    chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
    for (const t of niceTicks(yLo, yHi, 4)) {
      svg("line", { x1: m.left, x2: width - m.right, y1: y(t), y2: y(t), class: "uaf-grid" }, chart);
      text(chart, m.left - 8, y(t) + 4, price(t), "uaf-tick", "end");
    }
    for (const [t, label] of ticksOf(xs[0], xs.at(-1), 4)) {
      text(chart, x(t), height - m.bottom + 17, label, "uaf-tick", "middle");
    }
    text(chart, (m.left + width - m.right) / 2, height - 6, `${features[k].label} (${features[k].unit})`, "uaf-axis", "middle");
    const target = xs.map((v, i) => [x(v), y(curve.y[i])]);
    const from = morph && lastCurve ? lastCurve : target.map(([px]) => [px, height - m.bottom]);
    const area = svg("path", { class: "uaf-area" }, chart);
    const line = svg("path", { class: "uaf-line" }, chart);
    const hx = clamp(x(features[k].value), m.left, width - m.right);
    const guide = svg("line", { x1: hx, x2: hx, y1: m.top, y2: height - m.bottom, class: "uaf-guide" }, chart);
    const marker = svg("g", { class: "uaf-marker", transform: `translate(${hx} ${y(data.prediction)})` }, chart);
    svg("circle", { r: 9, class: "uaf-marker-halo" }, marker);
    svg("circle", { r: 4.5, class: "uaf-marker-dot" }, marker);
    const label = text(chart, hx + (hx > width * 0.7 ? -8 : 8), m.top + 10, `this house: ${value(k)}`, "uaf-marker-label", hx > width * 0.7 ? "end" : "start");
    const draw = (p) => {
      const points = target.map(([tx, ty], i) => [lerp(from[i][0], tx, p), lerp(from[i][1], ty, p)]);
      const d = points.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)} ${py.toFixed(1)}`).join("");
      line.setAttribute("d", d);
      area.setAttribute("d", `${d}L${points.at(-1)[0]} ${height - m.bottom}L${points[0][0]} ${height - m.bottom}Z`);
      lastCurve = points;
    };
    if (morph && lastCurve) morphing(600, draw, { easing: ease.inOut });
    else animate(800, draw, { easing: ease.out });
    for (const node of [guide, marker, label]) {
      if (!still()) node.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 400, delay: 300, fill: "backwards" });
    }
    readFrom(chart, (event) => {
      const box = chart.getBoundingClientRect();
      const px = ((event.clientX - box.left) / box.width) * width;
      const at = xs[0] + ((px - m.left) / (width - m.left - m.right)) * (xs.at(-1) - xs[0]);
      let i = 0; // the point nearest to the pointer
      xs.forEach((v, j) => {
        if (Math.abs(v - at) < Math.abs(xs[i] - at)) i = j;
      });
      return `${features[k].label} ${value(k, xs[i])}: ${price(curve.y[i])}`;
    });
    chart.setAttribute(
      "aria-label",
      `The prediction for this house as ${features[k].label.toLowerCase()} varies from ${value(k, xs[0])} to ` +
        `${value(k, xs.at(-1))}: between ${price(Math.min(...curve.y))} and ${price(Math.max(...curve.y))}; ` +
        `at the house's own value, ${value(k)}, ${price(data.prediction)}.`,
    );
    return `The prediction for this house as ${features[k].label.toLowerCase()} varies and its other features stay (an ICE curve).`;
  }

  function drawGroup(width, ids) {
    const members = [...ids].sort((a, b) => a - b);
    const together = members.length === 1 ? data.shapley[members[0]] : data.groups[keyOf(members)];
    const parts = members.map((k) => data.shapley[k]);
    const sum = parts.reduce((a, b) => a + b, 0);
    // the scale spans zero, the group's value, and the stacked values of its features, with room
    // for a value's label on either side
    const up = parts.filter((v) => v > 0).reduce((a, b) => a + b, 0);
    const down = parts.filter((v) => v < 0).reduce((a, b) => a + b, 0);
    const lo = Math.min(0, together, down);
    const hi = Math.max(0, together, up);
    const labelWidth = Math.min(98, width * 0.26);
    const x0 = labelWidth + (lo < 0 ? 64 : 8);
    const x1 = width - (hi > 0 ? 66 : 8);
    const x = (v) => x0 + ((v - lo) / Math.max(hi - lo, 0.05)) * (x1 - x0);
    const height = 140;
    chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg("line", { x1: x(0), x2: x(0), y1: 14, y2: 128, class: "uaf-zero" }, chart);
    text(chart, 4, 48, "together", "uaf-label");
    text(chart, 4, 108, "one by one", "uaf-label");
    const bar = svg("rect", { y: 30, height: 30, rx: 5, class: `uaf-bar uaf-bar-group ${sign(together)}` }, chart);
    const big = text(chart, 0, 50, signed(together), `uaf-value uaf-value-big ${sign(together)}`, together >= 0 ? "start" : "end");
    animate(
      800,
      (p) => {
        const end = x(together * p);
        bar.setAttribute("x", Math.min(x(0), end));
        bar.setAttribute("width", Math.abs(end - x(0)));
        big.setAttribute("x", together >= 0 ? end + 8 : end - 8);
        big.style.opacity = p;
      },
      { easing: ease.spring },
    );
    let above = 0;
    let below = 0;
    parts.forEach((v, j) => {
      const k = members[j];
      const start = v >= 0 ? above : below;
      if (v >= 0) above += v;
      else below += v;
      const segment = svg("rect", { y: 94, height: 22, rx: 3, class: `uaf-bar uaf-segment ${sign(v)}`, "data-ids": String(k) }, chart);
      const tag = text(chart, 0, 109, String(k + 1), `uaf-segment-label ${v >= 0 ? "uaf-on-pos" : "uaf-on-neg"}`, "middle");
      animate(
        600,
        (p) => {
          const a = x(start * p);
          const b = x((start + v) * p);
          segment.setAttribute("x", Math.min(a, b));
          segment.setAttribute("width", Math.abs(b - a));
          tag.setAttribute("x", (a + b) / 2);
          tag.style.opacity = Math.abs(b - a) > 16 ? p : 0;
        },
        { easing: ease.out, delay: 250 + 90 * j },
      );
      segment.addEventListener("pointerenter", () => highlight([k]));
      segment.addEventListener("pointerleave", () => highlight(null));
    });
    const tick = svg("g", { class: "uaf-sum-tick" }, chart);
    svg("path", { d: "M0 -7 L5 0 L0 7 L-5 0 Z" }, tick);
    animate(500, (p) => tick.setAttribute("transform", `translate(${x(sum * p)} 128) scale(${p})`), { delay: 450, easing: ease.spring });
    // the sum of the shown values, so that the line adds up
    const shown = members.map((k) => thousands(data.shapley[k]));
    const added = shown.reduce((a, b) => a + b, 0);
    equation.innerHTML =
      members.length === 1
        ? `${term(added, true)} alone, as one feature`
        : `one by one ${terms(shown)} = ${term(added, true)}, together ${term(thousands(together), true)}`;
    chart.setAttribute(
      "aria-label",
      `The group ${names(members)}: ${signed(together)} together, ${signedK(added)} one by one.`,
    );
    return "The Shapley value of the group, its features taken as one player, beside their Shapley values one by one.";
  }

  function drawHeatmap(width, pair, kind) {
    const [i, j] = [...pair].sort((a, b) => a - b);
    const key = `${i}-${j}`;
    const surface = data.surfaces[key];
    const grid = kind === "pure" ? data.pure[key] : surface.z; // grid[a][b]: feature i at a, j at b
    const cells = grid.length;
    const size = Math.max(150, Math.min(width - 96, 290));
    const cell = size / cells;
    const ox = Math.max(56, (width - size) / 2);
    const oy = 8;
    const lx = ox + size + 12;
    const beside = lx + 62 <= width; // the legend beside the map, or under it
    const height = size + (beside ? 58 : 92);
    chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const flat = grid.flat();
    const lo = Math.min(...flat);
    const hi = Math.max(...flat);
    const reach = Math.max(Math.abs(lo), Math.abs(hi));
    const fill = (z) => {
      if (kind === "pure") {
        const share = Math.round((Math.abs(z) / reach) * 100);
        return `color-mix(in oklab, var(${z >= 0 ? "--shapiq-red" : "--shapiq-blue"}) ${share}%, var(--shapiq-surface))`;
      }
      const share = Math.round(8 + ((z - lo) / (hi - lo)) * 92);
      return `color-mix(in oklab, var(--shapiq-amber) ${share}%, var(--shapiq-surface))`;
    };
    const cellsGroup = svg("g", {}, chart);
    grid.forEach((row, a) => {
      row.forEach((z, b) => {
        const rect = svg("rect", {
          x: ox + b * cell,
          y: oy + (cells - 1 - a) * cell,
          width: cell + 0.5,
          height: cell + 0.5,
          class: "uaf-cell",
        }, cellsGroup);
        rect.style.fill = fill(z);
        rect.dataset.a = a;
        rect.dataset.b = b;
        if (!still()) {
          rect.animate([{ opacity: 0, transform: "scale(0.4)" }, { opacity: 1, transform: "none" }], {
            duration: 360,
            delay: 14 * (a + b),
            easing: "cubic-bezier(0.2, 0.8, 0.3, 1.2)",
            fill: "backwards",
          });
        }
      });
    });
    svg("rect", { x: ox, y: oy, width: size, height: size, class: "uaf-frame" }, chart);
    const xs = surface.y; // the x axis is feature j
    const ys = surface.x; // the y axis is feature i
    const px = (v) => ox + ((v - xs[0]) / (xs.at(-1) - xs[0])) * (size - cell) + cell / 2;
    const py = (v) => oy + size - (((v - ys[0]) / (ys.at(-1) - ys[0])) * (size - cell) + cell / 2);
    for (const [t, label] of ticksOf(xs[0], xs.at(-1), 4)) text(chart, px(t), oy + size + 15, label, "uaf-tick", "middle");
    for (const [t, label] of ticksOf(ys[0], ys.at(-1), 4)) text(chart, ox - 6, py(t) + 4, label, "uaf-tick", "end");
    text(chart, ox + size / 2, oy + size + 34, features[j].label, "uaf-axis", "middle");
    const yLabel = text(chart, 0, 0, features[i].label, "uaf-axis", "middle");
    yLabel.setAttribute("transform", `translate(${ox - 40} ${oy + size / 2}) rotate(-90)`);
    const hx = clamp(px(features[j].value), ox, ox + size);
    const hy = clamp(py(features[i].value), oy, oy + size);
    const marker = svg("g", { class: "uaf-marker", transform: `translate(${hx} ${hy})` }, chart);
    svg("circle", { r: 9, class: "uaf-marker-halo" }, marker);
    svg("circle", { r: 4.5, class: "uaf-marker-dot" }, marker);
    if (!still()) marker.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300, delay: 14 * 2 * cells, fill: "backwards" });
    const legend = svg("g", {}, chart);
    const [top, bottom] = kind === "pure" ? [signed(reach), signed(-reach)] : [price(hi), price(lo)];
    for (let s = 0; s < 24; s++) {
      const z = kind === "pure" ? reach - (2 * reach * s) / 23 : hi - ((hi - lo) * s) / 23;
      const r = beside
        ? svg("rect", { x: lx, y: oy + (size / 24) * s, width: 8, height: size / 24 + 0.5 }, legend)
        : svg("rect", { x: ox + size - (size / 24) * (s + 1), y: oy + size + 48, width: size / 24 + 0.5, height: 8 }, legend);
      r.style.fill = fill(z);
    }
    if (beside) {
      text(legend, lx + 12, oy + 9, top, "uaf-tick");
      text(legend, lx + 12, oy + size, bottom, "uaf-tick");
    } else {
      text(legend, ox, oy + size + 72, bottom, "uaf-tick", "start");
      text(legend, ox + size, oy + size + 72, top, "uaf-tick", "end");
    }
    const describe = (a, b) =>
      `${features[i].label} ${value(i, surface.x[a])}, ${features[j].label.toLowerCase()} ${value(j, surface.y[b])}: ` +
      (kind === "pure" ? signed(grid[a][b]) : price(grid[a][b]));
    readout.textContent = "";
    readFrom(cellsGroup, (event) => {
      const target = event.target.closest("rect");
      return target?.dataset.a === undefined ? "" : describe(Number(target.dataset.a), Number(target.dataset.b));
    });
    const [first, second] = [features[i].label.toLowerCase(), features[j].label.toLowerCase()];
    chart.setAttribute(
      "aria-label",
      kind === "pure"
        ? `The pure interaction of ${first} and ${second}: from ${signed(lo)} to ${signed(hi)}, zero where either has the house's own value.`
        : `The prediction for this house as ${first} and ${second} vary: from ${price(lo)} to ${price(hi)}; ${price(data.prediction)} at the house's own values.`,
    );
    return kind === "pure"
      ? `The change in the prediction for this house as ${first} and ${second} vary together, minus the changes each makes alone: red where the pair raises the price beyond them, blue where it lowers it.`
      : `The prediction for this house as ${first} and ${second} vary together; the dot is the house.`;
  }

  function drawNetwork(width) {
    const height = Math.min(310, Math.max(250, width * 0.62));
    chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width / 2 - 84, height / 2 - 34); // room for the labels beside
    const at = (k) => {
      const angle = -Math.PI / 2 + (2 * Math.PI * k) / n;
      return [cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)];
    };
    const pairs = Object.entries(data.interactions).map(([key, v]) => [key.split("-").map(Number), v]);
    const strongest = Math.max(...pairs.map(([, v]) => Math.abs(v)));
    pairs.sort((a, b) => Math.abs(a[1]) - Math.abs(b[1]));
    const chosen = keyOf(state.pair);
    const [ca, cb] = state.pair.map(at);
    svg("line", { x1: ca[0], y1: ca[1], x2: cb[0], y2: cb[1], class: "uaf-edge-halo" }, chart);
    const edges = svg("g", {}, chart);
    const hits = svg("g", {}, chart); // wide, invisible lines over the edges, to point at them
    pairs.forEach(([[a, b], v], rank) => {
      const [x1, y1] = at(a);
      const [x2, y2] = at(b);
      const strength = Math.abs(v) / strongest;
      const edge = svg("line", {
        x1, y1, x2, y2,
        class: `uaf-edge ${sign(v)}${keyOf([a, b]) === chosen ? " is-selected" : ""}`,
        "data-ids": `${a}-${b}`,
        "stroke-width": 1.2 + 9 * strength,
      }, edges);
      edge.style.opacity = 0.18 + 0.82 * strength;
      const length = Math.hypot(x2 - x1, y2 - y1);
      if (!still()) {
        edge.style.strokeDasharray = `${length}`;
        edge.animate([{ strokeDashoffset: length }, { strokeDashoffset: 0 }], {
          duration: 520,
          delay: 40 * rank,
          easing: "cubic-bezier(0.4, 0, 0.2, 1)",
          fill: "backwards",
        });
      }
      const hit = svg("line", { x1, y1, x2, y2, class: "uaf-edge-hit" }, hits);
      hit.addEventListener("pointerenter", () => {
        highlight([a, b]);
        readout.textContent = `${features[a].label} × ${features[b].label}: ${signed(v)}`;
      });
      hit.addEventListener("pointerleave", (event) => {
        highlight(null);
        if (event.pointerType === "mouse") readout.textContent = "";
      });
      hit.addEventListener("click", () => {
        if (keyOf([a, b]) === chosen) return;
        state.pair = [a, b];
        render();
      });
    });
    features.forEach((feature, k) => {
      const [x, y] = at(k);
      const node = svg("g", { class: "uaf-node", "data-ids": String(k), transform: `translate(${x} ${y})` }, chart);
      svg("circle", { r: 15 }, node);
      text(node, 0, 4.5, String(k + 1), "uaf-node-index", "middle");
      const outward = x < cx - 5 ? "end" : x > cx + 5 ? "start" : "middle";
      const dx = outward === "end" ? -21 : outward === "start" ? 21 : 0;
      const dy = Math.abs(x - cx) < 5 ? (y < cy ? -22 : 30) : 4;
      text(node, dx, dy, feature.label, "uaf-node-label", outward);
      if (!still()) {
        node.querySelector("circle").animate([{ transform: "scale(0)" }, { transform: "scale(1)" }], {
          duration: 420,
          delay: 60 * k,
          easing: "cubic-bezier(0.3, 1.5, 0.5, 1)",
          fill: "backwards",
        });
      }
      node.addEventListener("pointerenter", () => highlight([k]));
      node.addEventListener("pointerleave", () => highlight(null));
    });
    const [a, b] = state.pair;
    const v = data.interactions[keyOf([a, b])];
    readout.textContent = "";
    title.querySelector(".uaf-picked").innerHTML = `{${features[a].label}, ${features[b].label}}: <span class="${sign(v)}">${signed(v)}</span>`;
    const strongest3 = [...pairs].reverse().slice(0, 3);
    chart.setAttribute(
      "aria-label",
      `A network of the features, an edge for each pair, as thick as their Shapley interaction; the strongest: ` +
        `${strongest3.map(([[p, q], w]) => `${features[p].label} and ${features[q].label} ${signed(w)}`).join(", ")}.`,
    );
    return "Shapley interactions of order two (k-SII): what two features do together beyond what each does alone. Thicker lines are stronger.";
  }

  function drawMessage(words) {
    chartBox.hidden = true;
    message.hidden = false;
    message.textContent = words;
    if (!still()) message.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300 });
    return "";
  }

  // -- the title of each view: the units and the space of Definition 2.1 -------------------------------

  const TITLES = {
    "features-numbers": [`{{1}, …, {${n}}}`, "ℝ", "one number per feature"],
    "features-curves": [`{{1}, …, {${n}}}`, "curves", "one curve per feature"],
    "groups-numbers": ["{<i>S</i>}", "ℝ", "one number for the group"],
    "groups-curves": ["{<i>S</i>}", "surfaces", "one surface for the group"],
    "interactions-numbers": ["{{<i>i</i>, <i>j</i>} : <i>i</i> ≠ <i>j</i>}", "ℝ", "one number per pair"],
    "interactions-curves": ["{{<i>i</i>, <i>j</i>} : <i>i</i> ≠ <i>j</i>}", "surfaces", "one surface per pair"],
  };
  const HINTS = {
    features: "Pick a feature.",
    groups: "Pick the features of the group.",
    interactions: "Pick two features.",
  };

  // a group of one feature has a curve, of two a surface, and of more a function the page cannot draw
  const titleOf = (key) => {
    const size = state.group.size;
    if (key !== "groups-curves" || size === 2) return TITLES[key];
    return size === 1
      ? ["{<i>S</i>}", "curves", "one curve for the group"]
      : ["{<i>S</i>}", `functions of ${size} features`, "one function for the group"];
  };
  const shapeOf = (key) => (key === "groups-curves" ? `${key}-${Math.min(state.group.size, 3)}` : key);

  let lastView = null;
  let lastWidth = 0;
  function render({ animated = true } = {}) {
    for (const stop of stops) stop();
    stops = [];
    const key = `${state.units}-${state.form}`;
    const changed = shapeOf(key) !== lastView;
    entering = animated && changed;
    const width = Math.max(260, Math.round(chartBox.clientWidth));
    lastWidth = width;
    const ids = selection();
    chipNodes.forEach((chip, k) => {
      chip.classList.toggle("is-selected", ids.includes(k));
      chip.setAttribute("aria-pressed", String(ids.includes(k)));
    });
    hint.textContent = HINTS[state.units];
    const [units, space, words] = titleOf(key);
    title.innerHTML =
      `<span class="uaf-def"><span class="uaf-cal">𝒮</span> = ${units}, <i>E</i> = ${space}</span>` +
      `<span class="uaf-words">${words}</span><span class="uaf-picked"></span>`;
    chart.replaceChildren();
    chart.onpointermove = null;
    chart.onpointerdown = null;
    chart.onpointerleave = null;
    chartBox.hidden = false;
    message.hidden = true;
    equation.innerHTML = "";
    readout.textContent = "";
    const morph = !changed && animated;
    let said; // the note under the view
    if (key === "features-numbers") said = drawBars(width);
    else if (key === "features-curves") said = drawCurve(width, state.feature, morph);
    else if (key === "groups-numbers") said = drawGroup(width, state.group);
    else if (key === "groups-curves") {
      const members = [...state.group];
      if (members.length === 1) said = drawCurve(width, members[0], morph);
      else if (members.length === 2) said = drawHeatmap(width, members, "surface");
      else {
        said = drawMessage(
          `With ${members.length} features, the explanation is a function of ${members.length} features, ` +
            "more than a page can draw. Remove features until two are left.",
        );
      }
    } else if (key === "interactions-numbers") said = drawNetwork(width);
    else said = drawHeatmap(width, state.pair, "pure");
    note.textContent = said;
    if (key !== "interactions-numbers" && key !== "features-numbers") {
      title.querySelector(".uaf-picked").innerHTML =
        state.units === "features"
          ? `{${features[state.feature].label}}`
          : state.units === "groups"
            ? `{${names(state.group)}}`
            : `{${names(state.pair)}}`;
    }
    if (entering && !reducedMotion()) {
      for (const node of [chart, title, equation, note]) {
        node.animate([{ opacity: 0, transform: "translateY(5px)" }, { opacity: 1, transform: "none" }], {
          duration: 300,
          easing: "cubic-bezier(0.2, 0.7, 0.2, 1)",
        });
      }
    }
    if (pointing) highlight(pointing);
    lastView = shapeOf(key);
    if (!(key === "features-curves" || (key === "groups-curves" && state.group.size === 1))) lastCurve = null;
  }

  unitsControl.set("features");
  formControl.set("numbers");
  new ResizeObserver(() => {
    const width = Math.round(chartBox.clientWidth);
    if (lastView && Math.abs(width - lastWidth) > 2) render({ animated: false });
  }).observe(chartBox);

  // the first view draws itself when the illustration is first seen
  let seen = false;
  let stopWatching = null;
  stopWatching = kit.watchVisibility(stage, (visible) => {
    if (visible && !seen) {
      seen = true;
      lastView = null;
      render();
      stopWatching?.();
    }
  });
});
