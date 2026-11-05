// The illustration "lab-stepper" (chapter 2.1 of the user guide): Definition 2.1, e : 𝒮 → E, told
// as three questions and then the answer, on one passenger of the Titanic. A step bar (1 · 2 · 3 · 4,
// with Back and Next) drives one picture that changes a little at each step:
//
// 1. the passenger x: the five features the model sees, and its prediction, her chance to survive;
// 2. the units 𝒮: which sets of features we ask about (single features, groups, or pairs, a switch);
//    each unit drops from the features as a row of dots, one per feature of the set;
// 3. the explanation space E: what one answer looks like (a number or a curve, a switch), shown for
//    one unit, while the others wait with a question mark;
// 4. the explanation e: every unit gets its answer, the whole map S ↦ e(S). From here on, both
//    switches stay, and a unit can be picked to read its answer.
//
// The explanations: Shapley values (single features), Shapley values of groups (each group one
// player), Shapley interactions (k-SII, pairs); her ICE curves (single features), the curves of a
// group (one per value of its first feature), and pure interaction surfaces (pairs). Values are
// computed once by ../compute/compute.py into data.js, in percentage points (pp) of her predicted
// chance to survive; red is positive, blue negative, amber marks the features (docs/STYLE_GUIDE.md).
window.shapiqIllustrations?.define("lab-stepper", async (stage, kit) => {
  const { el, svg, tween, ease, reducedMotion, segmented, loadScript, watchVisibility } = kit;
  if (!window.SHAPIQ_LAB_STEPPER) await loadScript(`${kit.root}data.js?v=${kit.version}`);
  const data = window.SHAPIQ_LAB_STEPPER;
  const features = data.features;
  const n = features.length;
  const moving = () => !reducedMotion();

  // -- numbers: probabilities in percent, explanations in percentage points (pp) ---------------------

  const MINUS = "−";
  const percent = (v) => `${Math.round(v)}%`;
  const signedPP = (k, unit = true) => `${k > 0 ? "+" : k < 0 ? MINUS : ""}${Math.abs(k)}${unit ? " pp" : ""}`;
  const sign = (v) => (v >= 0 ? "lst-pos" : "lst-neg");
  // a value of a feature, as the passenger's table would show it
  const valueText = (k, v = features[k].value) => {
    const f = features[k];
    if (f.categories) return f.categories[data.curves[k].x.indexOf(v)] ?? String(v);
    if (f.name === "fare") return `£${v.toFixed(2)}`;
    return Number.isInteger(v) ? String(v) : v.toFixed(1);
  };
  // the same, rounded, for the ends of an axis
  const tickText = (k, v) => (features[k].name === "fare" ? `£${Math.round(v)}` : valueText(k, v));
  const setText = (ids) => `{${ids.map((k) => k + 1).join(", ")}}`;
  const namesOf = (ids) => {
    const names = ids.map((k) => features[k].label);
    return names.length < 3 ? names.join(" and ") : `${names.slice(0, -1).join(", ")}, and ${names.at(-1)}`;
  };
  // integers that add up to `sum`: each rounded down, and the rest given to those that lost the most
  const roundTo = (values, sum) => {
    const shown = values.map(Math.floor);
    const order = values.map((v, k) => [v - Math.floor(v), k]).sort((a, b) => b[0] - a[0]);
    let rest = sum - shown.reduce((a, b) => a + b, 0);
    for (let j = 0; rest > 0; j++, rest--) shown[order[j % order.length][1]] += 1;
    for (let j = order.length - 1; rest < 0; j--, rest++) shown[order[(j + order.length) % order.length][1]] -= 1;
    return shown;
  };
  // "an 18%", "a 41%": as the number is spoken
  const article = (k) => (k === 8 || k === 11 || k === 18 || (k >= 80 && k < 90) ? "an" : "a");
  const prediction = Math.round(data.prediction);
  const baseline = Math.round(data.baseline);
  const total = prediction - baseline; // shown parts add up to the shown difference

  // -- the two choices, and the explanation they give ------------------------------------------------

  const pairs = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) pairs.push([i, j]);
  const UNITS = {
    features: {
      label: "Features",
      sets: features.map((_, k) => [k]),
      notation: `{${features.map((_, k) => setText([k])).join(", ")}}`,
      gloss: () => `${n} units: each feature alone.`,
      values: data.shapley,
      shown: roundTo(data.shapley, total),
    },
    groups: {
      label: "Groups",
      sets: data.groups.map((g) => g.ids),
      notation: `{${data.groups.map((g) => setText(g.ids)).join(", ")}}`,
      gloss: () =>
        `${data.groups.length} units: ${data.groups.map((g, k) => `${k === data.groups.length - 1 ? "and " : ""}${g.label} ${setText(g.ids)}`).join(", ")}.`,
      values: data.group_shapley,
      shown: roundTo(data.group_shapley, total),
    },
    pairs: {
      label: "Pairs",
      sets: pairs,
      notation: `{${setText(pairs[0])}, ${setText(pairs[1])}, …, ${setText(pairs.at(-1))}}`,
      gloss: () => `${pairs.length} units: every pair of features.`,
      values: pairs.map(([i, j]) => data.interactions[`${i}-${j}`]),
      shown: pairs.map(([i, j]) => Math.round(data.interactions[`${i}-${j}`])),
    },
  };
  const SPACES = {
    numbers: { label: "Numbers", notation: "<i>E</i> = ℝ" },
    curves: { label: "Curves", notation: "<i>E</i> = curves" },
  };
  const METHODS = {
    "features-numbers": "Shapley values",
    "features-curves": "ICE curves",
    "groups-numbers": "Shapley values of the groups",
    "groups-curves": "ICE curves of the groups",
    "pairs-numbers": "Shapley interactions (k‑SII)",
    "pairs-curves": "pure interaction surfaces",
  };
  const additive = (units, space) => space === "numbers" && units !== "pairs";
  // the unit to show first, at step 3: the one with the largest answer
  const largest = (values) => values.reduce((best, v, k) => (Math.abs(v) > Math.abs(values[best]) ? k : best), 0);
  const state = {
    step: 1,
    units: "features",
    space: "numbers",
    focus: Object.fromEntries(Object.entries(UNITS).map(([id, u]) => [id, largest(u.values)])),
  };

  const STEPS = [
    {
      symbol: "<i>x</i>",
      label: "Passenger",
      text: () =>
        `The model sees one passenger as five features, and gives her ${article(prediction)} ${prediction}% chance to survive.`,
    },
    {
      symbol: '<span class="lst-cal">𝒮</span>',
      label: "Units",
      text: () => "Which sets of features do we ask about? These are the <strong>units</strong>.",
    },
    {
      symbol: "<i>E</i>",
      label: "Space",
      text: () => "What does one answer look like? This is the <strong>explanation space</strong>.",
    },
    {
      symbol: "<i>e</i>",
      label: "Explanation",
      text: () =>
        `The <strong>explanation</strong> answers every unit, <span class="lst-nowrap"><i>S</i> ↦ <i>e</i>(<i>S</i>)</span>: here, ${METHODS[`${state.units}-${state.space}`]}.`,
    },
  ];

  // -- the layout: the steps, the question, the choices, and the picture ------------------------------

  const root = el("div", "lst");
  const nav = el("div", "lst-nav");
  const stepList = el("ol", "lst-steps");
  stepList.setAttribute("aria-label", "Steps");
  const stepButtons = STEPS.map((s, k) => {
    const item = el("li", "lst-steps-item");
    const button = el("button", "lst-step");
    button.type = "button";
    button.innerHTML =
      `<span class="lst-step-dot">${k + 1}</span>` +
      `<span class="lst-step-label"><span class="lst-step-symbol">${s.symbol}</span> ${s.label}</span>`;
    button.setAttribute("aria-label", `Step ${k + 1}: ${s.label}`);
    button.addEventListener("click", () => go(k + 1));
    button.addEventListener("keydown", (event) => {
      const delta = { ArrowRight: 1, ArrowLeft: -1, Home: -9, End: 9 }[event.key];
      if (!delta) return;
      event.preventDefault();
      const to = Math.min(4, Math.max(1, state.step + delta));
      go(to);
      stepButtons[to - 1].focus();
    });
    item.appendChild(button);
    stepList.appendChild(item);
    return button;
  });
  const back = el("button", "lst-back");
  back.type = "button";
  back.innerHTML = '<span aria-hidden="true">←</span><span class="lst-button-text"> Back</span>';
  back.setAttribute("aria-label", "Back");
  back.addEventListener("click", () => go(state.step - 1));
  const next = el("button", "lst-next");
  next.type = "button";
  next.addEventListener("click", () => go(state.step === 4 ? 1 : state.step + 1));
  const buttons = el("div", "lst-buttons");
  buttons.append(back, next);
  nav.append(stepList, buttons);

  const question = el("div", "lst-question");
  const symbol = el("span", "lst-symbol");
  symbol.setAttribute("aria-hidden", "true");
  const sentence = el("p", "lst-sentence");
  sentence.setAttribute("aria-live", "polite");
  question.append(symbol, sentence);

  // the two choices, as the heads of the two sides of the map: S ∈ 𝒮 = … over the units, and
  // e(S) ∈ E = … over the answers, each with its switch under it
  const choice = (key, options, name) => {
    const node = el("div", `lst-choice lst-choice-${key}`);
    const control = segmented(name, Object.entries(options).map(([id, o]) => [id, o.label]), (id, byUser) => {
      if (!byUser) return;
      state[key] = id;
      sync({ changed: key });
    });
    const notation = el("p", "lst-notation");
    node.append(notation, control.node);
    return { node, control, notation };
  };
  const unitsChoice = choice("units", UNITS, "Units");
  const spaceChoice = choice("space", SPACES, "Explanation space");

  // the picture: the passenger's features as the heads of five columns, the model's prediction,
  // then one row per unit, its features marked by dots in their columns, ↦, and its answer
  const board = el("div", "lst-board");
  const head = el("div", "lst-head lst-grid");
  const featureCells = features.map((f, k) => {
    const cell = el("div", "lst-feature");
    cell.title = `${f.label}: ${valueText(k)} (${f.description})`;
    cell.append(
      el("span", "shapiq-ill-chip-index lst-index", String(k + 1)),
      el("span", "lst-feature-label", f.short),
      el("span", "lst-feature-value", valueText(k)),
    );
    head.appendChild(cell);
    return cell;
  });
  const out = el("div", "lst-out");
  const modelChip = el("span", "lst-model");
  modelChip.title = "The model f: gradient boosting, an ensemble of decision trees";
  modelChip.innerHTML =
    '<svg viewBox="0 0 32 32" aria-hidden="true">' +
    '<rect x="11" y="3" width="10" height="6" rx="1.5"/><path d="M16 9v3.5M7 16v-3.5h17V16"/>' +
    '<circle class="lst-leaf" cx="7" cy="19" r="3"/><rect x="19" y="16" width="10" height="6" rx="1.5"/>' +
    '<path d="M24 22v3M19.5 27v-2h9v2"/><circle class="lst-leaf" cx="19.5" cy="29" r="2"/>' +
    '<circle class="lst-leaf" cx="28.5" cy="29" r="2"/></svg>';
  const outText = el("div", "lst-out-text");
  const outValue = el("span", "lst-out-value");
  outValue.innerHTML = `<span class="lst-out-symbol"><i>f</i>(<i>x</i>)</span> <span class="lst-out-number">${prediction}%</span>`;
  const gauge = el("span", "lst-gauge");
  gauge.setAttribute("aria-hidden", "true");
  const gaugeFill = el("span", "lst-gauge-fill");
  const gaugeMark = el("span", "lst-gauge-mark");
  gaugeMark.style.left = `${baseline}%`;
  gauge.append(gaugeFill, gaugeMark);
  const outLabel = el("span", "lst-out-label");
  outLabel.innerHTML = `<span>chance to survive</span><span class="lst-out-sep">; </span><span>average ${baseline}%</span>`;
  outText.append(outValue, gauge, outLabel);
  out.append(modelChip, el("span", "lst-out-arrow", "→"), outText);
  head.appendChild(out);

  const caption = el("div", "lst-caption lst-grid");
  const captionTo = el("span", "lst-caption-to", "↦");
  captionTo.setAttribute("aria-hidden", "true");
  caption.append(unitsChoice.node, captionTo, spaceChoice.node);

  const rowsBox = el("div", "lst-rows");
  rowsBox.setAttribute("role", "list");
  const sum = el("p", "lst-sum");
  board.append(head, caption, rowsBox, sum);

  const readout = el("p", "lst-readout");
  readout.setAttribute("aria-live", "polite");
  const foot = el("p", "lst-foot");
  foot.textContent =
    `A gradient boosting classifier on ${data.passengers.toLocaleString("en")} passengers of the ` +
    `Titanic with known age (AUC ${data.auc.toFixed(2)} on ${data.test} held-out passengers). ` +
    `Answers in percentage points (pp) of her chance to survive; the average is the model's mean ` +
    `prediction for the ${data.train} training passengers.`;
  root.append(nav, question, board, readout, foot);
  stage.appendChild(root);

  // -- the rows: one per unit -----------------------------------------------------------------------

  let rows = []; // the rows shown: { node, ids, k, answer, drawn }
  let generation = 0; // a newer change cancels what an older one still waits to do
  const later = (ms, run) => {
    const mine = generation;
    if (ms <= 0) return run();
    setTimeout(() => mine === generation && run(), ms);
  };

  const unitNames = (ids) =>
    state.units === "groups" && ids.length > 1
      ? `${data.groups.find((g) => g.ids.join() === ids.join()).label} (${namesOf(ids)})`
      : namesOf(ids);

  function buildRows() {
    const units = UNITS[state.units];
    rowsBox.dataset.kind = `${state.units}-${state.space}`;
    rows = units.sets.map((ids, k) => {
      const node = el("div", "lst-row lst-grid");
      node.setAttribute("role", "listitem");
      node.tabIndex = -1;
      const dots = [];
      for (let f = 0; f < n; f++) {
        const cell = el("span", "lst-cell");
        const member = ids.includes(f);
        const dot = el("span", member ? "lst-dot" : "lst-dot lst-ghost");
        cell.style.gridColumn = String(f + 1);
        cell.appendChild(dot);
        node.appendChild(cell);
        dots.push(dot);
      }
      let link = null;
      if (ids.length > 1) {
        link = el("span", "lst-link");
        link.style.gridColumn = `${ids[0] + 1} / ${ids.at(-1) + 2}`;
        node.appendChild(link);
      }
      const set = el("span", "lst-set", setText(ids));
      const to = el("span", "lst-to", "↦");
      const answer = el("div", "lst-answer");
      node.append(set, to, answer);
      node.addEventListener("click", () => pick(k));
      node.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          pick(k);
        }
        const delta = { ArrowDown: 1, ArrowUp: -1 }[event.key];
        if (delta && rows[k + delta]) {
          event.preventDefault();
          rows[k + delta].node.focus();
          if (state.step === 4) pick(k + delta);
        }
      });
      node.addEventListener("pointerenter", () => {
        light(ids);
        if (state.step >= 3) describe(k);
      });
      node.addEventListener("pointerleave", (event) => {
        if (event.pointerType !== "mouse") return;
        light(state.step >= 3 ? rows[focusOf()]?.ids ?? [] : []);
        describe(focusOf());
      });
      node.addEventListener("focus", () => {
        light(ids);
        if (state.step >= 3) describe(k);
      });
      rowsBox.appendChild(node);
      return { node, ids, k, dots, link, set, to, answer, drawn: false };
    });
  }

  // the units drop from the features: each dot falls from its feature's column head, then the line
  // that joins the features of a set is drawn, and the set appears. Returns when each row lands.
  function dropRows(delay, quick = false) {
    const stagger = quick ? 40 : rows.length > 6 ? 60 : 95;
    return rows.map((row, r) => {
      const wait = delay + stagger * r;
      if (!moving()) return 0;
      row.ids.forEach((f) => {
        const from = featureCells[f].getBoundingClientRect();
        const dot = row.dots[f].getBoundingClientRect();
        const dy = from.bottom - 10 - (dot.top + dot.height / 2);
        row.dots[f].animate(
          [
            { transform: `translateY(${dy}px) scale(0.5)`, opacity: 0 },
            { transform: `translateY(${dy * 0.7}px) scale(0.8)`, opacity: 1, offset: 0.25 },
            { transform: "none", opacity: 1 },
          ],
          { duration: 560, delay: wait, easing: "cubic-bezier(0.3, 0.9, 0.35, 1.12)", fill: "backwards" },
        );
        featureCells[f].querySelector(".lst-index").animate(
          [{ transform: "scale(1)" }, { transform: "scale(1.28)", offset: 0.35 }, { transform: "scale(1)" }],
          { duration: 420, delay: wait },
        );
      });
      row.dots.forEach((dot, f) => {
        if (!row.ids.includes(f)) dot.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300, delay: wait + 200, fill: "backwards" });
      });
      row.link?.animate([{ transform: "scaleX(0)" }, { transform: "scaleX(1)" }], {
        duration: 280,
        delay: wait + 430,
        easing: "cubic-bezier(0.2, 0.7, 0.2, 1)",
        fill: "backwards",
      });
      for (const part of [row.set, row.to]) {
        part.animate([{ opacity: 0, transform: "translateX(-4px)" }, { opacity: 1, transform: "none" }], {
          duration: 260,
          delay: wait + 480,
          fill: "backwards",
        });
      }
      return wait + 480;
    });
  }

  // the units rise back into the features, and leave
  function liftRows() {
    const leaving = rows;
    rows = [];
    if (!leaving.length) return 0;
    if (!moving()) {
      for (const row of leaving) row.node.remove();
      return 0;
    }
    const duration = 260;
    leaving.forEach((row, r) => {
      row.node.classList.add("is-leaving");
      row.node.classList.remove("is-focus");
      const wait = 25 * (leaving.length - 1 - r);
      row.ids.forEach((f) => {
        const from = featureCells[f].getBoundingClientRect();
        const dot = row.dots[f].getBoundingClientRect();
        const dy = from.bottom - 10 - (dot.top + dot.height / 2);
        row.dots[f].animate([{ transform: "none" }, { transform: `translateY(${dy}px) scale(0.5)`, opacity: 0 }], {
          duration,
          delay: wait,
          easing: "cubic-bezier(0.5, 0, 0.75, 0)",
          fill: "forwards",
        });
      });
      for (const part of [...row.node.children]) {
        if (!part.classList.contains("lst-cell")) part.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 160, delay: wait, fill: "forwards" });
      }
      row.dots.forEach((dot, f) => {
        if (!row.ids.includes(f)) dot.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 160, delay: wait, fill: "forwards" });
      });
    });
    const end = duration + 25 * leaving.length;
    setTimeout(() => leaving.forEach((row) => row.node.remove()), end);
    return end;
  }

  // -- the answers: a bar, a curve, or a surface ------------------------------------------------------

  const focusOf = () => state.focus[state.units];
  let geometry = null; // the width of the answers, and the scale of the bars, shared by the rows
  const measure = () => {
    const width = Math.max(60, Math.round(rows[0]?.answer.getBoundingClientRect().width || 200));
    const values = UNITS[state.units].values;
    const short = width < 170; // without "pp" beside the bars, as the caption says it
    const room = short ? (width < 110 ? 25 : 30) : 48;
    const lo = Math.min(0, ...values);
    const hi = Math.max(0, ...values);
    const x0 = lo < 0 ? room : 4;
    const x1 = width - (hi > 0 ? room : 4);
    geometry = { width, short, scale: (v) => x0 + ((v - lo) / (hi - lo || 1)) * (x1 - x0) };
  };

  const text = (parent, x, y, content, cls, anchor = "start") => {
    const node = svg("text", { x, y, class: cls, "text-anchor": anchor }, parent);
    node.textContent = content;
    return node;
  };

  function drawNumber(row, animate, delay) {
    const { width, scale, short } = geometry;
    const height = 26;
    const v = UNITS[state.units].values[row.k];
    const shown = UNITS[state.units].shown[row.k];
    const node = svg("svg", { class: "lst-e", viewBox: `0 0 ${width} ${height}`, width, height, "aria-hidden": "true" });
    const zero = scale(0);
    svg("line", { x1: zero, x2: zero, y1: 1, y2: height - 1, class: "lst-zero" }, node);
    const bar = svg("rect", { y: 7, height: 12, rx: 3, class: `lst-bar ${sign(v)}` }, node);
    const label = text(node, 0, 17.5, signedPP(shown, !short), `lst-value ${shown ? sign(v) : "lst-nil"}`, v >= 0 ? "start" : "end");
    const place = (p) => {
      const end = scale(v * p);
      bar.setAttribute("x", Math.min(zero, end));
      bar.setAttribute("width", Math.max(0, Math.abs(end - zero)));
      label.setAttribute("x", v >= 0 ? end + 5 : end - 5);
      label.style.opacity = Math.min(1, p * 1.5);
    };
    if (animate && moving()) {
      place(0);
      tween(720, place, { easing: ease.spring, delay });
    } else place(1);
    return node;
  }

  // a curve of her chance (0 to 100%) over the values of a feature, which are categories or numbers
  const CHANCE = { lo: 0, hi: 100 };
  function plotFrame(width, height) {
    const room = width >= 230 ? 74 : 0; // the range beside the plot, where there is room
    const left = 4;
    const right = width - room - 6;
    const top = 4;
    const bottom = height - 13;
    const y = (v) => bottom - ((v - CHANCE.lo) / (CHANCE.hi - CHANCE.lo)) * (bottom - top);
    return { room, left, right, top, bottom, y };
  }
  function xScale(feature, frame) {
    const xs = data.curves[feature].x;
    if (features[feature].categories) {
      const m = xs.length;
      return (v) => frame.left + ((xs.indexOf(v) + 0.5) / m) * (frame.right - frame.left);
    }
    return (v) => frame.left + ((v - xs[0]) / (xs.at(-1) - xs[0])) * (frame.right - frame.left);
  }
  const pathOf = (points) => points.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)} ${py.toFixed(1)}`).join("");

  function drawLine(node, points, cls, animate, delay) {
    const line = svg("path", { d: pathOf(points), class: cls }, node);
    if (animate && moving()) {
      const length = line.getTotalLength();
      line.style.strokeDasharray = `${length}`;
      line.animate([{ strokeDashoffset: length }, { strokeDashoffset: 0 }], {
        duration: 650,
        delay,
        easing: "cubic-bezier(0.4, 0, 0.2, 1)",
        fill: "backwards",
      });
      line.addEventListener("animationend", () => line.style.removeProperty("stroke-dasharray"));
    }
    return line;
  }

  function drawCurve(row, animate, delay) {
    const { width } = geometry;
    const height = 74;
    const node = svg("svg", { class: "lst-e", viewBox: `0 0 ${width} ${height}`, width, height, "aria-hidden": "true" });
    const frame = plotFrame(width, height);
    const ids = row.ids;
    svg("line", { x1: frame.left, x2: frame.right, y1: frame.bottom, y2: frame.bottom, class: "lst-axis" }, node);
    const avg = frame.y(data.baseline);
    svg("line", { x1: frame.left, x2: frame.right, y1: avg, y2: avg, class: "lst-average" }, node);
    let lo = Infinity;
    let hi = -Infinity;
    let her = null;
    let hover = null;
    if (ids.length === 1) {
      const f = ids[0];
      const { x: xs, y: ys } = data.curves[f];
      const x = xScale(f, frame);
      const points = xs.map((v, i) => [x(v), frame.y(ys[i])]);
      const area = svg("path", { d: `${pathOf(points)}L${points.at(-1)[0]} ${frame.bottom}L${points[0][0]} ${frame.bottom}Z`, class: "lst-area" }, node);
      if (animate && moving()) area.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 500, delay: delay + 300, fill: "backwards" });
      drawLine(node, points, "lst-line", animate, delay);
      if (features[f].categories) {
        xs.forEach((v, i) => {
          svg("circle", { cx: x(v), cy: frame.y(ys[i]), r: 2.6, class: "lst-point" }, node);
          text(node, x(v), height - 2, features[f].categories[i], "lst-tick", "middle");
        });
      } else {
        text(node, frame.left, height - 2, tickText(f, xs[0]), "lst-tick");
        text(node, frame.right, height - 2, tickText(f, xs.at(-1)), "lst-tick", "end");
      }
      lo = Math.min(...ys);
      hi = Math.max(...ys);
      her = [x(features[f].value), frame.y(data.prediction)];
      hover = (px) => {
        let i = 0;
        xs.forEach((v, j) => {
          if (Math.abs(x(v) - px) < Math.abs(x(xs[i]) - px)) i = j;
        });
        return `${features[f].label} ${valueText(f, xs[i])}: her chance ${percent(ys[i])}`;
      };
    } else {
      // a group of two: one curve over the second feature per value of the first, hers in amber
      const [a, b] = ids;
      const g = data.group_curves[`${a}-${b}`];
      const x = xScale(b, frame);
      const mine = g.by.indexOf(features[a].value);
      g.y.forEach((ys, c) => {
        if (c === mine) return;
        drawLine(node, g.x.map((v, i) => [x(v), frame.y(ys[i])]), "lst-line lst-line-other", animate, delay);
      });
      drawLine(node, g.x.map((v, i) => [x(v), frame.y(g.y[mine][i])]), "lst-line", animate, delay + 120);
      text(node, frame.left, height - 2, tickText(b, g.x[0]), "lst-tick");
      text(node, frame.right, height - 2, tickText(b, g.x.at(-1)), "lst-tick", "end");
      // the name of each curve at its end, hers first; curves that end together share a name
      if (frame.room) {
        const ends = g.y.map((ys, c) => ({ y: frame.y(ys.at(-1)), names: [valueText(a, g.by[c])], mine: c === mine }));
        const merged = [];
        for (const end of ends.sort((p, q) => p.y - q.y)) {
          const close = merged.find((m) => Math.abs(m.y - end.y) < 11 && !m.mine && !end.mine);
          if (close) close.names.push(...end.names);
          else merged.push({ ...end });
        }
        for (let k = 1; k < merged.length; k++) merged[k].y = Math.max(merged[k].y, merged[k - 1].y + 11);
        for (const end of merged) {
          text(node, frame.right + 8, end.y + 3.5, end.names.join(", "), end.mine ? "lst-end lst-end-mine" : "lst-end");
        }
      }
      const flat = g.y.flat();
      lo = Math.min(...flat);
      hi = Math.max(...flat);
      her = [x(features[b].value), frame.y(data.prediction)];
      hover = (px, py) => {
        let i = 0;
        g.x.forEach((v, j) => {
          if (Math.abs(x(v) - px) < Math.abs(x(g.x[i]) - px)) i = j;
        });
        let c = mine;
        g.y.forEach((ys, j) => {
          if (Math.abs(frame.y(ys[i]) - py) < Math.abs(frame.y(g.y[c][i]) - py)) c = j;
        });
        return `${features[a].label} ${valueText(a, g.by[c])}, ${features[b].label.toLowerCase()} ${valueText(b, g.x[i])}: her chance ${percent(g.y[c][i])}`;
      };
    }
    const dot = svg("circle", { cx: her[0], cy: her[1], r: 3.6, class: "lst-her" }, node);
    if (frame.room && ids.length === 1) text(node, frame.right + 10, height / 2 + 1, `${percent(lo)} to ${percent(hi)}`, "lst-range");
    if (animate && moving()) dot.animate([{ opacity: 0, transform: "scale(0)" }, { opacity: 1, transform: "none" }], { duration: 300, delay: delay + 560, fill: "backwards" });
    readFrom(node, width, hover);
    return node;
  }

  // the pure interaction of a pair as both features vary: a small map, red where the two together
  // raise her chance beyond what each does alone, blue where they lower it, on one scale for all
  // pairs (its intensity to the power 0.6, so that weak interactions still show); the dot is her
  const reach = Math.max(...Object.values(data.pure).flatMap((p) => p.z.flat().map(Math.abs)));
  function drawSurface(row, animate, delay) {
    const { width } = geometry;
    const height = 40;
    const node = svg("svg", { class: "lst-e", viewBox: `0 0 ${width} ${height}`, width, height, "aria-hidden": "true" });
    const [i, j] = row.ids;
    const p = data.pure[`${i}-${j}`];
    const room = width >= 230 ? 84 : 0;
    const w = Math.min(width - room - 6, 150);
    const rowsN = p.x.length;
    const colsN = p.y.length;
    const ch = (height - 4) / rowsN;
    const cw = w / colsN;
    const heat = svg("g", { class: "lst-heat" }, node);
    p.z.forEach((line, a) => {
      line.forEach((z, b) => {
        const cell = svg("rect", { x: 2 + b * cw, y: 2 + (rowsN - 1 - a) * ch, width: cw + 0.3, height: ch + 0.3, class: "lst-cell-rect" }, heat);
        cell.style.fill = `color-mix(in oklab, var(${z >= 0 ? "--shapiq-red" : "--shapiq-blue"}) ${Math.round(Math.min(1, Math.abs(z) / reach) ** 0.6 * 100)}%, var(--shapiq-surface))`;
      });
    });
    svg("rect", { x: 2, y: 2, width: w, height: height - 4, class: "lst-frame" }, node);
    const nearest = (axis, v) => axis.reduce((best, u, k) => (Math.abs(u - v) < Math.abs(axis[best] - v) ? k : best), 0);
    const ai = nearest(p.x, features[i].value);
    const bj = nearest(p.y, features[j].value);
    const dot = svg("circle", { cx: 2 + (bj + 0.5) * cw, cy: 2 + (rowsN - 1 - ai + 0.5) * ch, r: 3, class: "lst-her" }, node);
    const flat = p.z.flat();
    const lo = Math.round(Math.min(...flat));
    const hi = Math.round(Math.max(...flat));
    if (room) text(node, w + 12, height / 2 + 4, `${signedPP(lo, false)} to ${signedPP(hi)}`, "lst-range");
    if (animate && moving()) {
      heat.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 480, delay, fill: "backwards" });
      dot.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 240, delay: delay + 380, fill: "backwards" });
    }
    readFrom(node, width, (px, py) => {
      if (px > w + 2) return "";
      const b = Math.min(colsN - 1, Math.max(0, Math.floor((px - 2) / cw)));
      const a = rowsN - 1 - Math.min(rowsN - 1, Math.max(0, Math.floor((py - 2) / ch)));
      return `${features[i].label} ${valueText(i, p.x[a])}, ${features[j].label.toLowerCase()} ${valueText(j, p.y[b])}: ${signedPP(Math.round(p.z[a][b]))} together, beyond each alone`;
    });
    return node;
  }

  // a readout of the point under the pointer, in place of the unit's words while it is there
  function readFrom(node, width, describePoint) {
    node.addEventListener("pointermove", (event) => {
      const box = node.getBoundingClientRect();
      const scale = width / box.width;
      const words = describePoint((event.clientX - box.left) * scale, (event.clientY - box.top) * scale);
      if (words) readout.textContent = words;
    });
  }

  // where its answer will be: at the zero of the bars, or at the start of the curves
  function placeholder(animate, delay) {
    const mark = el("span", "lst-q", "?");
    mark.setAttribute("aria-hidden", "true");
    if (state.space === "numbers") mark.style.marginLeft = `${Math.max(0, geometry.scale(0) - 14)}px`;
    if (animate && moving()) mark.animate([{ opacity: 0, transform: "scale(0.6)" }, { opacity: 1, transform: "none" }], { duration: 260, delay, fill: "backwards" });
    return mark;
  }

  // a row's answer: drawn, or waiting with a question mark, or nothing yet (before step 3)
  function setAnswer(row, mode, { animate = false, delay = 0 } = {}) {
    const draw = { numbers: drawNumber, curves: row.ids.length === 2 && state.units === "pairs" ? drawSurface : drawCurve }[state.space];
    const old = [...row.answer.children];
    const swap = () => {
      if (mode === "drawn") row.answer.replaceChildren(draw(row, animate, 0));
      else if (mode === "waiting") row.answer.replaceChildren(placeholder(animate, 0));
      else row.answer.replaceChildren();
    };
    row.drawn = mode === "drawn";
    row.mode = mode;
    if (animate && moving() && old.length) {
      for (const node of old) node.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 160, delay, fill: "forwards" });
      later(delay + 160, swap);
    } else if (animate && moving() && delay) {
      later(delay, swap);
    } else swap();
  }

  // -- the step: what each part of the picture shows ----------------------------------------------------

  let lastStep = 0;
  function showStep(animate) {
    const s = STEPS[state.step - 1];
    const changedStep = lastStep !== state.step;
    root.dataset.step = state.step;
    symbol.innerHTML = s.symbol;
    sentence.innerHTML = s.text();
    if (animate && changedStep && moving()) {
      for (const [node, dx] of [[symbol, 0], [sentence, 8]]) {
        node.animate([{ opacity: 0, transform: `translateX(${dx}px)` }, { opacity: 1, transform: "none" }], {
          duration: 320,
          easing: "cubic-bezier(0.2, 0.7, 0.2, 1)",
        });
      }
    }
    stepButtons.forEach((button, k) => {
      if (k + 1 === state.step) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
      button.closest("li").classList.toggle("is-done", k + 1 < state.step);
      button.tabIndex = k + 1 === state.step ? 0 : -1;
    });
    back.disabled = state.step === 1;
    next.innerHTML =
      state.step === 4
        ? '<span aria-hidden="true">↺</span><span class="lst-next-long">Start over</span><span class="lst-next-short">Again</span>'
        : '<span>Next</span><span aria-hidden="true">→</span>';
    // the choices, each from the step that asks for it
    for (const [part, from] of [[unitsChoice, 2], [spaceChoice, 3]]) {
      const shown = state.step >= from;
      const was = !part.node.classList.contains("is-off");
      part.node.classList.toggle("is-off", !shown);
      part.node.inert = !shown;
      if (shown && !was && animate && moving()) {
        part.node.animate([{ opacity: 0, transform: "translateY(-6px)" }, { opacity: 1, transform: "none" }], {
          duration: 360,
          easing: "cubic-bezier(0.2, 0.7, 0.2, 1)",
        });
      }
    }
    const units = UNITS[state.units];
    unitsChoice.notation.innerHTML = `<i>S</i> ∈ <span class="lst-cal">𝒮</span> = ${units.notation}`;
    spaceChoice.notation.innerHTML =
      `<i>e</i>(<i>S</i>) ∈ ${SPACES[state.space].notation}` +
      (state.space === "numbers" ? '<span class="lst-notation-unit">, in pp</span>' : "");
    lastStep = state.step;
  }

  // the sum, where the explanation adds up
  function showSum(animate, delay) {
    const shown = state.step === 4 && additive(state.units, state.space);
    if (!shown) {
      sum.innerHTML = "";
      return;
    }
    sum.innerHTML =
      `They add up: <span class="lst-nowrap">average ${baseline}% <span class="${total >= 0 ? "lst-pos" : "lst-neg"}">${total >= 0 ? "+" : MINUS}&nbsp;${Math.abs(total)}&#8239;pp</span> ` +
      `= her&nbsp;${prediction}%</span>`;
    if (animate && moving()) sum.animate([{ opacity: 0, transform: "translateY(-4px)" }, { opacity: 1, transform: "none" }], { duration: 360, delay, fill: "backwards" });
  }

  // what the focused (or pointed) unit's answer says, in words
  function describe(k) {
    if (state.step === 2) {
      readout.textContent = UNITS[state.units].gloss();
      return;
    }
    if (state.step < 3 || !rows[k]) {
      readout.textContent = "";
      return;
    }
    const ids = rows[k].ids;
    const names = unitNames(ids);
    const set = `<i>e</i>(${setText(ids)})`;
    if (state.space === "numbers") {
      const shown = UNITS[state.units].shown[k];
      const verb = shown >= 0 ? (state.units === "pairs" ? "raise" : "raises") : state.units === "pairs" ? "lower" : "lowers";
      const what =
        state.units === "pairs"
          ? `${names} together, beyond what each does alone, ${verb} her chance by ${Math.abs(shown)} points.`
          : `${names[0].toUpperCase()}${names.slice(1)} ${verb} her chance by ${Math.abs(shown)} points against the average.`;
      const nothing =
        state.units === "pairs"
          ? `${names} do little together beyond what each does alone.`
          : `${names[0].toUpperCase()}${names.slice(1)} barely moves her chance.`;
      readout.innerHTML = `${set} = <span class="${shown ? sign(shown) : "lst-nil"}">${signedPP(shown)}</span>: ${shown ? what : nothing}`;
    } else if (state.units === "pairs") {
      readout.innerHTML = `${set}: what ${names} do together beyond each alone, as both vary. The dot is her.`;
    } else if (ids.length === 2) {
      readout.innerHTML = `${set}: her chance as ${namesOf(ids).toLowerCase()} vary together, one curve per ${features[ids[0]].label.toLowerCase()}, hers in amber. The dot is her.`;
    } else {
      readout.innerHTML = `${set}: her chance as ${features[ids[0]].label.toLowerCase()} varies and all else stays. The dot is her.`;
    }
  }

  function markFocus() {
    rows.forEach((row) => row.node.classList.toggle("is-focus", state.step >= 3 && row.k === focusOf()));
    light(state.step >= 3 && rows[focusOf()] ? rows[focusOf()].ids : []);
  }

  // the features of the unit pointed at, or picked, light up in the passenger
  function light(ids) {
    featureCells.forEach((cell, f) => cell.classList.toggle("is-on", ids.includes(f)));
  }

  // -- bring the picture to the state: the step, the units, the space ----------------------------------

  let seen = false;
  function sync({ changed = "step", from = state.step } = {}) {
    generation += 1;
    const animate = seen && changed !== "resize";
    showStep(animate);
    const wantRows = state.step >= 2;
    let wait = 0;
    const heightBefore = rowsBox.getBoundingClientRect().height;

    if (changed === "units" && rows.length) wait = liftRows();
    if (!wantRows && rows.length) liftRows();

    const answerMode = (row) =>
      state.step <= 2 ? "none" : state.step === 3 && row.k !== focusOf() ? "waiting" : "drawn";

    // the answers that change, each after its delay: delayOf(row, k), k counting the rows that get
    // their answer now
    const fill = (delayOf) => {
      measure();
      let k = 0;
      let last = 0;
      const redraw = changed === "space" || changed === "resize" || changed === "focus";
      for (const row of rows) {
        const mode = answerMode(row);
        if (mode !== row.mode || (redraw && mode === "drawn")) {
          const delay = delayOf(row, mode === "drawn" && row.mode !== "drawn" ? k++ : 0);
          last = Math.max(last, delay);
          setAnswer(row, mode, { animate: animate && changed !== "resize", delay });
        }
      }
      return last;
    };

    const build = () => {
      buildRows();
      const landed = animate ? dropRows(0, changed === "units") : rows.map(() => 0);
      animateHeight(heightBefore);
      const done = fill((row) => landed[row.k]);
      later(landed[focusOf()] ?? 0, markFocus);
      showSum(animate, done + 650);
      describe(focusOf());
    };

    if (wantRows && !rows.length) {
      if (wait) {
        sum.innerHTML = "";
        later(wait, build);
      } else build();
      return;
    }
    if (!wantRows) {
      light([]);
      showSum(false, 0);
      readout.textContent = "";
      animateHeight(heightBefore);
      return;
    }
    // the rows stay: their answers change with the step, the space, or the unit picked
    rowsBox.dataset.kind = `${state.units}-${state.space}`;
    markFocus();
    const before = rowsBox.getBoundingClientRect().height;
    const stagger = changed === "step" ? (from === 3 && state.step === 4 ? 75 : 40) : 30;
    const done = fill((row, k) => k * stagger);
    animateHeight(before);
    showSum(animate, done + 650);
    describe(focusOf());
  }

  // the rows' room grows and shrinks smoothly, so that the page below moves, and does not jump
  function animateHeight(before) {
    const after = rowsBox.getBoundingClientRect().height;
    if (!seen || !moving() || Math.abs(after - before) < 2) return;
    rowsBox.animate([{ height: `${before}px` }, { height: `${after}px` }], {
      duration: Math.min(650, 260 + Math.abs(after - before)),
      easing: "cubic-bezier(0.2, 0.7, 0.2, 1)",
    });
  }

  function go(step) {
    if (step < 1 || step > 4 || step === state.step) return;
    const from = state.step;
    state.step = step;
    sync({ changed: "step", from });
  }

  function pick(k) {
    if (state.step < 2 || !rows[k]) return;
    const before = focusOf();
    state.focus[state.units] = k;
    if (state.step === 3 && before !== k) sync({ changed: "focus" });
    else {
      markFocus();
      describe(k);
    }
  }

  // -- start: the passenger appears when the illustration is first seen ---------------------------------

  unitsChoice.control.set(state.units);
  spaceChoice.control.set(state.space);
  sync({ changed: "resize" });
  gaugeFill.style.width = "0%";
  root.classList.add("is-waiting");

  let lastWidth = root.clientWidth;
  new ResizeObserver(() => {
    if (Math.abs(root.clientWidth - lastWidth) < 2) return;
    lastWidth = root.clientWidth;
    if (rows.length) {
      measure();
      for (const row of rows) if (row.mode === "drawn") setAnswer(row, "drawn");
    }
  }).observe(root);

  const intro = () => {
    root.classList.remove("is-waiting");
    if (!moving()) {
      gaugeFill.style.width = `${data.prediction}%`;
      return;
    }
    featureCells.forEach((cell, k) => {
      cell.animate([{ opacity: 0, transform: "translateY(8px)" }, { opacity: 1, transform: "none" }], {
        duration: 420,
        delay: 80 * k,
        easing: "cubic-bezier(0.2, 0.7, 0.2, 1)",
        fill: "backwards",
      });
    });
    out.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 400, delay: 80 * n + 100, fill: "backwards" });
    const number = out.querySelector(".lst-out-number");
    tween(
      900,
      (p) => {
        gaugeFill.style.width = `${data.prediction * p}%`;
        number.textContent = `${Math.round(data.prediction * p)}%`;
      },
      { delay: 80 * n + 250 },
    );
  };
  let stopWatching = null;
  stopWatching = watchVisibility(stage, (visible) => {
    if (visible && !seen) {
      seen = true;
      intro();
      stopWatching?.();
    }
  });
});
