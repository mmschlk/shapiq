// The figure of receptive fields ("Exact, and still feasible" on the explorer's page,
// demos/graphshapiq-explorer/index.rst; it has no page of its own): why exact Shapley interactions
// of a GNN stay feasible. A GNN with L layers mixes an atom only with the atoms within L bonds, its
// receptive field; atoms that never share one cannot interact, so GraphSHAP-IQ evaluates the GNN
// only on the maskings within receptive fields. The reader points at an atom to see its field grow
// layer by layer, and compares GraphSHAP-IQ's maskings with all 2^n. The numbers of maskings are
// GraphSHAP-IQ's own (data.js, by ../../graphshapiq-common/compute/pack.py); no GNN is evaluated.
(() => {
  const G = window.ShapiqGraph;
  if (!G) return console.error("The GraphSHAP-IQ exact demo needs ../graphshapiq-common/demo.js first.");
  const { h, svg, reducedMotion } = G;
  const LAYERS = [1, 2, 3, 4];
  const fmt = (n) => n.toLocaleString("en");
  const sup = (n) => String(n).replace(/[0-9]/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);

  G.mount("graphshapiq-exact-demo", "SHAPIQ_GRAPHSHAPIQ_EXACT", (root, data) => {
    const molecules = data.molecules;
    const state = { mol: molecules.find((m) => m.id === "cholesterol") || molecules[molecules.length - 1], layers: 3, focus: null };
    const distances = (m, from) => { const d = m.atoms.map(() => -1), queue = [from]; d[from] = 0; while (queue.length) { const k = queue.shift(); for (const j of G.neighbors(m)[k]) if (d[j] < 0) { d[j] = d[k] + 1; queue.push(j); } } return d; };
    const fieldOf = (k) => new Set(distances(state.mol, k).flatMap((d, j) => (d >= 0 && d <= state.layers ? [j] : [])));
    const central = (m) => { const { pos } = G.geom(m, 78, 13); let best = 0, bd = Infinity; pos.forEach(([x, y], k) => { const d = (x - 50) ** 2 + (y - 50) ** 2; if (d < bd) { bd = d; best = k; } }); return best; };

    // -- the frame --------------------------------------------------------------------------------
    const bar = h("div", "gsq-bar");
    const mols = G.picker("Molecule", molecules, (k) => { state.mol = molecules[k]; state.focus = central(state.mol); build(); });
    const depth = G.switcher("Layers of the GNN", LAYERS.map(String), (k) => { state.layers = LAYERS[k]; pulse(); renderPanel(); });
    bar.append(mols.node, h("div", "gsq-switches", h("span", "gsq-barlab", "GNN layers"), depth.node));
    const body = h("div", "gsq-body"), stage = h("div", "gsq-stage"), frame = h("div", "gsq-frame"), panel = h("div", "gsq-panel");
    const plot = svg("svg", { class: "gsq-plot", viewBox: "0 0 100 100", role: "group", "aria-label": "The molecule; point at an atom to see its receptive field" }, frame);
    const hint = G.hinter(frame);
    const caption = h("p", "gsq-caption");
    stage.append(frame, caption);
    body.append(stage, panel);
    root.append(bar, body);

    // -- the molecule, and the receptive field as a glow that spreads bond by bond ------------------
    let aura = null, discs = [], timers = [];
    function build() {
      const m = state.mol, g = G.geom(m, 78, 13), { s, pos } = g;
      plot.replaceChildren();
      aura = svg("g", { class: "gsq-aura" }, plot);
      const bonds = svg("g", {}, plot), atoms = svg("g", {}, plot), hits = svg("g", {}, plot);
      G.skeleton(bonds, m, 0.14, "gsq-bond", g);
      discs = m.atoms.map((a, k) => {
        const [x, y] = pos[k], disc = svg("circle", { class: "gsq-atom", cx: x, cy: y, r: 0.26 * s }, atoms);
        if (a[0] !== "C") svg("text", { class: "gsq-el", x, y, "font-size": (a[0].length > 1 ? 0.24 : 0.3) * s }, atoms).textContent = a[0];
        svg("circle", { class: "gsq-hit", cx: x, cy: y, r: 0.5 * s, "data-k": k, tabindex: "0", role: "button", "aria-label": `${G.atomName(m, k)} ${k + 1}: show its receptive field` }, hits);
        return disc;
      });
      pulse();
      renderPanel();
    }
    function pulse() {
      timers.forEach(clearTimeout); timers = [];
      aura.replaceChildren();
      discs.forEach((d) => d.classList.remove("gsq-in", "gsq-src"));
      if (state.focus == null) return;
      const m = state.mol, { s, pos } = G.geom(m, 78, 13), d = distances(m, state.focus), still = reducedMotion();
      for (let hop = 0; hop <= state.layers; hop++) {
        const light = () => d.forEach((dk, k) => {
          if (dk !== hop) return;
          svg("circle", { class: still ? "" : "gsq-pop", cx: pos[k][0], cy: pos[k][1], r: 0.56 * s }, aura);
          discs[k].classList.add(k === state.focus ? "gsq-src" : "gsq-in");
        });
        if (still) light(); else timers.push(setTimeout(light, hop * 170));
      }
    }
    function setFocus(k) { if (k < 0 || k === state.focus) return; state.focus = k; hint.hide(); pulse(); renderPanel(); }
    plot.addEventListener("mousedown", (e) => e.preventDefault());
    plot.addEventListener("pointermove", (e) => { if (e.pointerType === "touch") return; const k = atomAt(e); plot.style.cursor = k >= 0 ? "pointer" : ""; setFocus(k); });
    plot.addEventListener("click", (e) => setFocus(atomAt(e)));
    plot.addEventListener("focusin", (e) => { const k = Number(e.target.dataset?.k); if (!Number.isNaN(k)) setFocus(k); });
    function atomAt(e) {
      const point = plot.createSVGPoint(); point.x = e.clientX; point.y = e.clientY;
      const ctm = plot.getScreenCTM(); if (!ctm) return -1;
      const p = point.matrixTransform(ctm.inverse()), { s, pos } = G.geom(state.mol, 78, 13);
      let best = -1, bd = (0.5 * s) ** 2;
      pos.forEach(([x, y], k) => { const dd = (x - p.x) ** 2 + (y - p.y) ** 2; if (dd < bd) { bd = dd; best = k; } });
      return best;
    }

    // -- the panel: all maskings against GraphSHAP-IQ's, counted up ---------------------------------
    let frames = [];
    function countUp(node, to) {
      const from = Number(node.dataset.v || 0), t0 = performance.now(), dur = reducedMotion() ? 0 : 520;
      const step = (now) => { const p = dur ? Math.min(1, (now - t0) / dur) : 1, v = Math.round(from + (to - from) * (1 - (1 - p) ** 3)); node.textContent = fmt(v); node.dataset.v = v; if (p < 1) frames.push(requestAnimationFrame(step)); };
      frames.push(requestAnimationFrame(step));
    }
    const shown = { all: null, ours: null };
    function renderPanel() {
      frames.forEach(cancelAnimationFrame); frames = [];
      const m = state.mol, L = state.layers, n = m.n, all = 2 ** n, ours = m.budgets[L], factor = all / ours;
      const box = h("div", "gsq-panel-in");
      box.append(h("p", "gsq-ctitle", "With ", h("strong", "", `${L} ${L === 1 ? "layer" : "layers"}`), `, an atom reaches ${L} ${L === 1 ? "bond" : "bonds"} far.`));
      const allValue = h("span", "gsq-bignum gsq-muted"), oursValue = h("span", "gsq-bignum gsq-accent");
      allValue.dataset.v = shown.all ?? 0; oursValue.dataset.v = shown.ours ?? 0;
      box.append(h("div", "gsq-stat", h("div", "gsq-statrow", h("span", "gsq-statlab", `All maskings, 2${sup(n)}`), allValue), h("div", "gsq-statrow", h("span", "gsq-statlab", "GraphSHAP-IQ evaluates"), oursValue)));
      const bars = h("div", "gsq-logbars");
      bars.setAttribute("aria-hidden", "true");
      const barRow = (label, count, cls) => { const r = h("div", `gsq-logrow${cls.includes("gsq-soft") ? "" : " gsq-on"}`), track = h("div", "gsq-logbar"), fill = h("i", cls); fill.style.width = `${Math.max(2, (Math.log10(Math.max(2, count)) / Math.log10(all)) * 100)}%`; track.append(fill); r.append(h("span", "gsq-loglab", label), track, h("span", "gsq-logn", fmt(count))); return r; };
      bars.append(barRow("All maskings", all, "gsq-all"));
      for (const LL of LAYERS) bars.append(barRow(`${LL} ${LL === 1 ? "layer" : "layers"}`, m.budgets[LL], LL === L ? "gsq-accent" : "gsq-accent gsq-soft"));
      box.append(bars, h("p", "gsq-barnote", "Logarithmic scale"));
      if (factor > 1.05) {
        const fewer = factor >= 10 ? `${fmt(Math.round(factor))}× fewer` : `${factor.toFixed(1)}× fewer`;
        box.append(h("p", "gsq-punch", h("strong", "gsq-accent", fewer), " evaluations, and still ", h("strong", "", "exact"), "."));
      } else box.append(h("p", "gsq-punch", "Small molecule, little to skip. The larger the molecule, the more GraphSHAP-IQ saves."));
      const line = h("p", "gsq-fieldline");
      if (state.focus != null) { const f = fieldOf(state.focus); line.append("This atom can interact with ", h("strong", "", `${f.size} of ${n} atoms`), "."); }
      box.append(line);
      panel.replaceChildren(box);
      countUp(allValue, all);
      countUp(oursValue, ours);
      shown.all = all; shown.ours = ours;
      depth.set(LAYERS.indexOf(L));
      mols.set(molecules.indexOf(m));
      caption.replaceChildren(h("strong", "", m.name));
    }

    state.focus = central(state.mol);
    build();
    hint.show(`${G.hovers() ? "Hover" : "Tap"} an atom`);
  });
})();
