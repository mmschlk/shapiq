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
// once by ../compute/compute.py into data.js; red is positive and blue negative, violet marks the
// features (docs/STYLE_GUIDE.md).
window.shapiqIllustrations?.define("units-and-forms", async (stage, kit) => {
  const { el, svg, tween, ease, reducedMotion, segmented, loadScript } = kit;
  // the data that the page loaded before this script, or, where it is missing, loaded here
  if (!window.SHAPIQ_UNITS_AND_FORMS) await loadScript(`${kit.root}data.js?v=${kit.version}`);
  const data = window.SHAPIQ_UNITS_AND_FORMS;
  const features = data.features;
  const n = features.length;
  const DECIMALS = [2, 0, 2, 0, 2, 2];

  const value = (k, v = features[k].value) => v.toFixed(DECIMALS[k]).replace("-", "−");
  const price = (v) => `$${Math.round(v * 100)}k`;
  const signed = (v) => `${v >= 0 ? "+" : "−"}$${Math.abs(Math.round(v * 100))}k`;
  const sign = (v) => (v >= 0 ? "uaf-pos" : "uaf-neg");
  const keyOf = (ids) => [...ids].sort((a, b) => a - b).join("-");
  const names = (ids) => [...ids].sort((a, b) => a - b).map((k) => features[k].label).join(", ");
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const lerp = (a, b, t) => a + (b - a) * t;

  const state = { units: "features", form: "numbers", feature: 0, group: new Set([4, 5]), pair: [4, 5] };

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
  const readout = el("p", "uaf-readout");
  const note = el("p", "uaf-note");
  view.append(title, chartBox, readout, note);
  body.append(side, view);

  const foot = el("p", "uaf-foot");
  foot.textContent =
    `A gradient boosting model on six features of the California housing data ` +
    `(R² ${data.r2.toFixed(2)} on held-out houses) predicts ${price(data.prediction)} for this ` +
    `house, against ${price(data.baseline)} on average.`;
  root.append(controls, body, foot);
  stage.appendChild(root);

  // -- picking features --------------------------------------------------------------------------

  const pick = (k) => {
    if (state.units === "features") state.feature = k;
    else if (state.units === "groups") {
      if (state.group.has(k)) {
        if (state.group.size === 1) {
          chipNodes[k].animate(
            [{ transform: "translateX(0)" }, { transform: "translateX(-3px)" }, { transform: "translateX(3px)" }, { transform: "translateX(0)" }],
            { duration: 260 },
          );
          return;
        }
        state.group.delete(k);
      } else state.group.add(k);
    } else if (!state.pair.includes(k)) state.pair = [state.pair[1], k];
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

  let stops = [];
  let instant = false; // a redraw without animation, after a resize
  const still = () => instant || reducedMotion();
  const animate = (duration, update, options) =>
    stops.push(tween(instant ? 0 : duration, update, instant ? {} : options));
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
    const height = n * row + 46;
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
        state.feature = k;
        render();
      });
      if (k === state.feature) group.classList.add("is-selected");
    });
    const total = values.reduce((a, b) => a + b, 0);
    const sum = text(chart, width / 2, n * row + 34, "", "uaf-sum", "middle");
    sum.innerHTML =
      `average ${price(data.baseline)} <tspan class="${sign(total)}">${signed(total)}</tspan>` +
      ` = this house ${price(data.prediction)}`;
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
    animate(morph && lastCurve ? 600 : 800, draw, { easing: morph ? ease.inOut : ease.out });
    for (const node of [guide, marker, label]) {
      if (!still()) node.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 400, delay: 300, fill: "backwards" });
    }
    chart.onpointermove = (event) => {
      const box = chart.getBoundingClientRect();
      const px = ((event.clientX - box.left) / box.width) * width;
      const i = clamp(Math.round(((px - m.left) / (width - m.left - m.right)) * (xs.length - 1)), 0, xs.length - 1);
      readout.textContent = `${features[k].label} ${value(k, xs[i])}: ${price(curve.y[i])}`;
    };
    chart.onpointerleave = () => {
      readout.textContent = "";
    };
    return `The prediction for this house as ${features[k].label.toLowerCase()} varies and its other features stay (an ICE curve).`;
  }

  function drawGroup(width, ids) {
    const members = [...ids].sort((a, b) => a - b);
    const together = members.length === 1 ? data.shapley[members[0]] : data.groups[keyOf(members)];
    const parts = members.map((k) => data.shapley[k]);
    const sum = parts.reduce((a, b) => a + b, 0);
    const reach = Math.max(Math.abs(together), parts.reduce((a, b) => a + Math.abs(b), 0), 0.05) * 1.18;
    const labelWidth = Math.min(98, width * 0.26);
    const x = (v) => labelWidth + ((v + reach) / (2 * reach)) * (width - labelWidth - 12);
    const height = 170;
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
    let up = 0;
    let down = 0;
    parts.forEach((v, j) => {
      const k = members[j];
      const start = v >= 0 ? up : down;
      if (v >= 0) up += v;
      else down += v;
      const segment = svg("rect", { y: 94, height: 22, rx: 3, class: `uaf-bar uaf-segment ${sign(v)}`, "data-ids": String(k) }, chart);
      const tag = text(chart, 0, 109, String(k + 1), "uaf-segment-label", "middle");
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
    const summary = text(chart, width / 2, 158, "", "uaf-sum", "middle");
    summary.innerHTML =
      `${members.map((k) => `<tspan class="${sign(data.shapley[k])}">${signed(data.shapley[k])}</tspan>`).join(" + ")}` +
      ` = <tspan class="${sign(sum)}">${signed(sum)}</tspan> one by one`;
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
    const height = size + 58;
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
      return `color-mix(in oklab, var(--shapiq-violet) ${share}%, var(--shapiq-surface))`;
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
    const lx = ox + size + 12;
    if (lx + 62 <= width) {
      for (let s = 0; s < 24; s++) {
        const z = kind === "pure" ? reach - (2 * reach * s) / 23 : hi - ((hi - lo) * s) / 23;
        const r = svg("rect", { x: lx, y: oy + (size / 24) * s, width: 8, height: size / 24 + 0.5 }, legend);
        r.style.fill = fill(z);
      }
      const [top, bottom] = kind === "pure" ? [signed(reach), signed(-reach)] : [price(hi), price(lo)];
      text(legend, lx + 12, oy + 9, top, "uaf-tick");
      text(legend, lx + 12, oy + size, bottom, "uaf-tick");
    }
    const describe = (a, b) =>
      `${features[i].label} ${value(i, surface.x[a])}, ${features[j].label.toLowerCase()} ${value(j, surface.y[b])}: ` +
      (kind === "pure" ? signed(grid[a][b]) : price(grid[a][b]));
    readout.textContent = "";
    cellsGroup.onpointermove = (event) => {
      const target = event.target.closest("rect");
      if (target?.dataset.a !== undefined) readout.textContent = describe(Number(target.dataset.a), Number(target.dataset.b));
    };
    cellsGroup.onpointerleave = () => {
      readout.textContent = "";
    };
    return kind === "pure"
      ? `The prediction for this house as ${features[i].label.toLowerCase()} and ${features[j].label.toLowerCase()} vary, minus what each does alone: red where the pair raises the price together, blue where it lowers it.`
      : `The prediction for this house as ${features[i].label.toLowerCase()} and ${features[j].label.toLowerCase()} vary together; the dot is the house.`;
  }

  function drawNetwork(width) {
    const height = Math.min(310, Math.max(250, width * 0.62));
    chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width / 2 - 70, height / 2 - 34);
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
    pairs.forEach(([[a, b], v], rank) => {
      const [x1, y1] = at(a);
      const [x2, y2] = at(b);
      const strength = Math.abs(v) / strongest;
      const edge = svg("line", {
        x1, y1, x2, y2,
        class: `uaf-edge ${sign(v)}${keyOf([a, b]) === chosen ? " is-selected" : ""}`,
        "data-ids": `${a}-${b}`,
        "stroke-width": 1.2 + 9 * strength,
      }, chart);
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
      edge.addEventListener("pointerenter", () => {
        highlight([a, b]);
        readout.textContent = `${features[a].label} × ${features[b].label}: ${signed(v)}`;
      });
      edge.addEventListener("pointerleave", () => {
        highlight(null);
        readout.textContent = "";
      });
      edge.addEventListener("click", () => {
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
    return "Shapley interactions of order two (k-SII): what two features do together beyond what each does alone. Thicker lines are stronger.";
  }

  function drawMessage(width, message) {
    chart.setAttribute("viewBox", `0 0 ${width} 120`);
    const node = text(chart, width / 2, 64, message, "uaf-message", "middle");
    if (!still()) node.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300 });
    return "";
  }

  // -- the title of each view: the units and the space of Definition 2.1 -------------------------------

  const TITLES = {
    "features-numbers": ['{{1}, …, {6}}', "ℝ", "one number per feature"],
    "features-curves": ['{{1}, …, {6}}', "curves", "one curve per feature"],
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

  let lastView = null;
  let lastWidth = 0;
  function render({ animated = true } = {}) {
    instant = !animated;
    for (const stop of stops) stop();
    stops = [];
    const key = `${state.units}-${state.form}`;
    const changed = key !== lastView;
    const width = Math.max(260, Math.round(chartBox.clientWidth));
    lastWidth = width;
    const ids = selection();
    chipNodes.forEach((chip, k) => {
      chip.classList.toggle("is-selected", ids.includes(k));
      chip.setAttribute("aria-pressed", String(ids.includes(k)));
    });
    hint.textContent = HINTS[state.units];
    const [units, space, words] = TITLES[key];
    title.innerHTML =
      `<span class="uaf-def"><span class="uaf-cal">𝒮</span> = ${units}, <i>E</i> = ${space}</span>` +
      `<span class="uaf-words">${words}</span><span class="uaf-picked"></span>`;
    chart.replaceChildren();
    chart.onpointermove = null;
    chart.onpointerleave = null;
    readout.textContent = "";
    const morph = !changed && animated;
    let message;
    if (key === "features-numbers") message = drawBars(width);
    else if (key === "features-curves") message = drawCurve(width, state.feature, morph);
    else if (key === "groups-numbers") message = drawGroup(width, state.group);
    else if (key === "groups-curves") {
      const members = [...state.group];
      if (members.length === 1) message = drawCurve(width, members[0], morph);
      else if (members.length === 2) message = drawHeatmap(width, members, "surface");
      else message = drawMessage(width, `With ${members.length} features, the explanation is a function of ${members.length} features. Pick two to see it.`);
    } else if (key === "interactions-numbers") message = drawNetwork(width);
    else message = drawHeatmap(width, state.pair, "pure");
    note.textContent = message;
    if (key !== "interactions-numbers" && key !== "features-numbers") {
      title.querySelector(".uaf-picked").innerHTML =
        state.units === "features"
          ? `{${features[state.feature].label}}`
          : state.units === "groups"
            ? `{${names(state.group)}}`
            : `{${names(state.pair)}}`;
    }
    if (changed && animated && !reducedMotion()) {
      for (const node of [chart, title, note]) {
        node.animate([{ opacity: 0, transform: "translateY(5px)" }, { opacity: 1, transform: "none" }], {
          duration: 300,
          easing: "cubic-bezier(0.2, 0.7, 0.2, 1)",
        });
      }
    }
    if (pointing) highlight(pointing);
    lastView = key;
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
  kit.watchVisibility(stage, (visible) => {
    if (visible && !seen) {
      seen = true;
      lastView = null;
      render();
    }
  });
});
