// The GraphSHAP-IQ orders demo (demos/graphshapiq-orders/index.rst): one prediction explained at
// every order, from the Shapley values to the full decomposition, as the orders demo does for its
// examples (../../orders). Every order is FSII, the Faithful Shapley Interactions, computed exactly
// from GraphSHAP-IQ's Möbius interactions; its faithfulness is the orders demo's R^2 beyond size
// (data.js, by ../../graphshapiq-common/compute/pack.py). The reader steps through the orders and
// watches the explanation take shape on the molecule while the curve of R^2 climbs to 1.
(() => {
  const G = window.ShapiqGraph;
  if (!G) return console.error("The GraphSHAP-IQ orders demo needs ../graphshapiq-common/demo.js first.");
  const { h, svg, bits, signed, value, legend, formula } = G;
  const FLOOR = -0.5; // the curve's lowest value: an R^2 below it is drawn at its bottom

  G.mount("graphshapiq-orders-demo", "SHAPIQ_GRAPHSHAPIQ_ORDERS", (root, data) => {
    const molecules = data.molecules;
    const state = { mol: molecules[0], at: 1 }; // `at`: the position of the order shown (0: order 1)
    const orders = () => state.mol.orders;
    const label = (o) => (o.k === "all" ? ["Full decomposition", "all orders"] : o.k === 1 ? ["Shapley values", "order 1"] : o.k === 2 ? ["Pairs", "order 2"] : [`Groups of ${o.k}`, `order ${o.k}`]);

    // -- the frame --------------------------------------------------------------------------------
    const bar = h("div", "bar"), mols = h("div", "mols");
    mols.setAttribute("role", "group"); mols.setAttribute("aria-label", "Molecule");
    bar.append(mols);
    const body = h("div", "body"), stage = h("div", "stage"), frame = h("div", "frame"), panel = h("div", "panel");
    const plot = svg("svg", { class: "plot", viewBox: "0 0 100 100", role: "img", "aria-label": "The molecule, explained at the order shown" }, frame);
    const caption = h("p", "caption");
    stage.append(frame, caption);
    const live = h("p", "sr"); live.setAttribute("aria-live", "polite");
    body.append(stage, panel);
    root.append(bar, body, live);
    const scene = G.createScene(plot, { duration: 480 });

    const molButtons = molecules.map((m) => {
      const b = h("button", "mol"); b.type = "button"; b.title = m.name;
      b.append(G.tile(m), h("span", "", m.short || m.name));
      b.addEventListener("click", () => { state.mol = m; state.at = Math.min(1, m.orders.length - 1); scene.setMolecule(m); update(); });
      mols.append(b);
      return b;
    });

    // -- the molecule at the order shown -----------------------------------------------------------
    function draw() {
      const m = state.mol, o = orders()[state.at];
      const own = {}, groups = [];
      for (const [mask, v] of o.groups) { const atoms = bits(mask); if (atoms.length === 1) own[atoms[0]] = v; else groups.push({ atoms, v }); }
      groups.sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
      scene.update({
        rings: own, ringLargest: Math.max(1e-9, ...Object.values(own).map(Math.abs)),
        groups: groups.slice(0, o.k === 2 ? 12 : 8), groupLargest: Math.max(1e-9, Math.abs(groups[0]?.v || 0)),
        masked: new Set(), truth: m.truth, truthKey: m.id, truthShown: true,
      });
    }

    // -- the curve of R^2, order by order ------------------------------------------------------------
    function curve() {
      const os = orders(), W = 100, H = 40, pad = 5;
      const X = (i) => pad + ((W - 2 * pad) * i) / Math.max(1, os.length - 1);
      const Y = (r) => pad + (H - 2 * pad) * (1 - (Math.max(FLOOR, r) - FLOOR) / (1 - FLOOR));
      const node = svg("svg", { class: "curve", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", "aria-hidden": "true" });
      svg("line", { class: "exact", x1: pad, y1: Y(1), x2: W - pad, y2: Y(1) }, node);
      svg("line", { class: "zero", x1: pad, y1: Y(0), x2: W - pad, y2: Y(0) }, node);
      svg("polyline", { class: "line", points: os.map((o, i) => `${X(i)},${Y(o.r2)}`).join(" ") }, node);
      os.forEach((o, i) => svg("circle", { class: `dot${i === state.at ? " on" : ""}${o.r2 < FLOOR ? " below" : ""}`, cx: X(i), cy: Y(o.r2), r: i === state.at ? 2.4 : 1.5 }, node));
      return node;
    }

    // -- the panel -----------------------------------------------------------------------------------
    function renderPanel() {
      const m = state.mol, os = orders(), o = os[state.at], [name, index] = label(o), t = data.tasks[m.task];
      const box = h("div", "panel-in");
      const dial = h("div", "stepper"), less = h("button", "step", "−"), more = h("button", "step", "+");
      less.type = more.type = "button";
      less.setAttribute("aria-label", "Lower order"); more.setAttribute("aria-label", "Higher order");
      less.setAttribute("aria-disabled", String(state.at <= 0)); more.setAttribute("aria-disabled", String(state.at >= os.length - 1));
      const go = (by) => { const to = state.at + by; if (to < 0 || to >= os.length) return; state.at = to; update(); panel.querySelectorAll(".step")[by < 0 ? 0 : 1]?.focus({ preventScroll: true }); };
      less.addEventListener("click", () => go(-1)); more.addEventListener("click", () => go(1));
      dial.append(less, h("span", "step-label", name, h("small", "", index)), more);
      const r2 = o.r2 >= 0.9995 ? "1.00" : o.r2 >= 0 ? o.r2.toFixed(2) : "below 0";
      const faith = h("div", "faith", h("p", "faith-head", h("span", "faith-value", r2), h("span", "faith-label", "faithfulness (R² beyond size)")), curve());
      const detail = h("p", "detail");
      if (o.k === 1) detail.append("One number per atom. On their own, the ", h("strong", "", "Shapley values"), " predict the GNN on maskings worse than counting how many atoms a masking keeps: most of what decides the answer is in how atoms combine.");
      else if (o.k === "all") detail.append("Every interaction, of every order: the ", h("strong", "", "Möbius interactions"), ", which GraphSHAP-IQ computes exactly. They reproduce the GNN on every masking.");
      else detail.append("The best explanation with groups of up to ", h("strong", "", `${o.k} atoms`), o.r2 >= 0.95 ? ". It already reproduces nearly all of what the atoms decide." : ". Each order that is added fits the GNN at least as well as the last.");
      const groups = o.groups.map(([mask, v]) => ({ atoms: bits(mask), v })).filter((g) => g.atoms.length >= (o.k === 1 ? 1 : 2)).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
      const largest = Math.max(1e-9, Math.abs(groups[0]?.v || 0)), list = h("ol", "rows");
      for (const g of groups.slice(0, 4)) list.append(G.row(m, g.atoms, g.v, largest, o.k === 1 ? G.atomName(m, g.atoms[0]) : G.groupName(m, g.atoms)));
      const meta = h("p", "meta");
      meta.append(h("span", null, o.k === "all" ? "Möbius interactions, " : "FSII, the Faithful Shapley Interactions, ", "computed ", h("strong", "", "exactly"), " with GraphSHAP-IQ."),
        h("span", null, `GIN, 3 layers: a score of `, value(m.logOdds), ` for “${t.yes}”.`));
      box.append(dial, faith, detail, list, legend(t, m.truth), meta);
      panel.replaceChildren(box);
      live.textContent = `${name}: faithfulness ${r2}.`;
    }

    function update() {
      const m = state.mol, t = data.tasks[m.task];
      molButtons.forEach((b, k) => b.setAttribute("aria-pressed", String(molecules[k] === m)));
      caption.replaceChildren(h("strong", "", m.name), formula(m.formula), h("span", "blurb", `The GNN asks: ${t.question}`));
      draw();
      renderPanel();
    }
    scene.setMolecule(state.mol);
    update();
  });
})();
