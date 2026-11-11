// The GraphSHAP-IQ lab (demos/graphshapiq-lab/index.rst): the reader builds a benzene derivative,
// and the page computes its exact Shapley interactions itself, with GraphSHAP-IQ, in a Web Worker.
//
// The GNNs (data.js, written by ../../graphshapiq-common/compute/pack.py) run in the worker as
// chains of affine maps and ReLUs. For the small molecules of the lab, the receptive fields of a
// three-layer GNN cover the whole molecule, so GraphSHAP-IQ evaluates all 2^n maskings: the worker
// computes the Möbius interactions from them, then the Shapley values and the 2-SII (k-SII of order
// 2), exactly as shapiq's MoebiusConverter does (checked against it to 1e-5 by
// ../../graphshapiq-common/compute/check_lab.py). The lab caps a
// molecule at MAX_ATOMS heavy atoms, so that the computation takes at most a few seconds.
(() => {
  const G = window.ShapiqGraph;
  if (!G) return console.error("The GraphSHAP-IQ lab needs ../graphshapiq-common/demo.js first.");
  const { h, svg, sig, bits, signed, percent, value, legend, formula } = G;
  const MAX_ATOMS = 15;
  const sup = (n) => String(n).replace(/[0-9]/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);

  // the groups a position of the ring can carry: heavy atoms with their place in the frame of the
  // position (x outward, y across, in bond lengths), hydrogens, charges, and bonds
  const GROUPS = {
    none: { label: "—", atoms: [], bonds: [], heavy: 0 },
    OH: { label: "OH", name: "hydroxy", atoms: [["O", 1, 0, 1, 0]], bonds: [], heavy: 1 },
    NH2: { label: "NH₂", name: "amino", atoms: [["N", 1, 0, 2, 0]], bonds: [], heavy: 1 },
    NO2: { label: "NO₂", name: "nitro", atoms: [["N", 1, 0, 0, 1], ["O", 1.75, 0.72, 0, -1], ["O", 1.75, -0.72, 0, 0]], bonds: [[0, 1, 1], [0, 2, 2]], heavy: 3 },
    CH3: { label: "CH₃", name: "methyl", atoms: [["C", 1, 0, 3, 0]], bonds: [], heavy: 1 },
    Cl: { label: "Cl", name: "chloro", atoms: [["Cl", 1, 0, 0, 0]], bonds: [], heavy: 1 },
    OMe: { label: "OCH₃", name: "methoxy", atoms: [["O", 1, 0, 0, 0], ["C", 1.9, 0.5, 3, 0]], bonds: [[0, 1, 1]], heavy: 2 },
    COOH: { label: "COOH", name: "carboxy", atoms: [["C", 1, 0, 0, 0], ["O", 1.6, 0.8, 0, 0], ["O", 1.65, -0.75, 1, 0]], bonds: [[0, 1, 2], [0, 2, 1]], heavy: 3 },
  };
  const PALETTE = ["OH", "NH2", "NO2", "CH3", "Cl", "OMe", "COOH"];
  const none = ["none", "none", "none", "none", "none"];
  const PRESETS = [
    { name: "Catechol", slots: ["OH", "OH", ...none.slice(1)], task: "PAINS" },
    { name: "Resorcinol", slots: ["OH", "none", "OH", ...none.slice(2)], task: "PAINS" },
    { name: "Hydroquinone", slots: ["OH", "none", "none", "OH", "none", "none"], task: "PAINS" },
    { name: "Nitrobenzene", slots: ["NO2", ...none], task: "mutagenic" },
    { name: "1,3-Dinitrobenzene", slots: ["NO2", "none", "NO2", ...none.slice(2)], task: "mutagenic" },
    { name: "Benzene", slots: ["none", ...none], task: "PAINS" },
  ];
  const KNOWN = Object.fromEntries([
    ...PRESETS.map((p) => [p.slots.join(), p.name]),
    [["OH", ...none].join(), "Phenol"], [["NH2", ...none].join(), "Aniline"], [["CH3", ...none].join(), "Toluene"],
    [["Cl", ...none].join(), "Chlorobenzene"], [["COOH", ...none].join(), "Benzoic acid"], [["OMe", ...none].join(), "Anisole"],
    [["OH", "OH", "OH", "none", "none", "none"].join(), "Pyrogallol"], [["NH2", "none", "none", "NH2", "none", "none"].join(), "1,4-Diaminobenzene"],
  ]);

  function assemble(slots) {
    const atoms = [], bonds = [];
    for (let i = 0; i < 6; i++) { const a = Math.PI / 2 - (i * Math.PI) / 3; atoms.push(["C", Math.cos(a), Math.sin(a), slots[i] === "none" ? 1 : 0, 0]); }
    for (let i = 0; i < 6; i++) bonds.push([i, (i + 1) % 6, i % 2 === 0 ? 2 : 1]);
    for (let i = 0; i < 6; i++) {
      const group = GROUPS[slots[i]];
      if (!group.atoms.length) continue;
      const a = Math.PI / 2 - (i * Math.PI) / 3, ux = Math.cos(a), uy = Math.sin(a), wx = -uy, wy = ux, start = atoms.length;
      for (const [el, lx, ly, hs, ch] of group.atoms) atoms.push([el, ux + lx * ux + ly * wx, uy + lx * uy + ly * wy, hs, ch]);
      bonds.push([i, start, 1]);
      for (const [p, q, o] of group.bonds) bonds.push([start + p, start + q, o]);
    }
    const counts = {};
    for (const a of atoms) { counts[a[0]] = (counts[a[0]] || 0) + 1; counts.H = (counts.H || 0) + a[3]; }
    const formulaText = ["C", "H", "N", "O", "Cl"].filter((e) => counts[e]).map((e) => e + (counts[e] > 1 ? counts[e] : "")).join("");
    return { atoms, bonds, rings: [[0, 1, 2, 3, 4, 5]], formula: formulaText };
  }

  // the worker: GraphSHAP-IQ for the molecule it is sent (see the header)
  const WORKER = `
  const floats = (s) => { const bin = atob(s), buf = new ArrayBuffer(bin.length), u = new Uint8Array(buf); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new Float32Array(buf); };
  function compile(symbols, m, t) {
    const n = m.atoms.length, S = symbols.length;
    const steps = t.gnn.steps.map((s) => ({ o: s.shape[0], ip: s.shape[1], w: floats(s.w), b: floats(s.b), relu: s.relu, gather: s.gather, sum: s.sum }));
    const deg = new Int32Array(n), edges = [];
    for (const [i, j] of m.bonds) { edges.push(i, j); deg[i]++; deg[j]++; }
    const off = new Int32Array(n + 1); for (let k = 0; k < n; k++) off[k + 1] = off[k] + deg[k];
    const nbr = new Int32Array(off[n]), at = off.slice(0, n);
    for (let e = 0; e < edges.length; e += 2) { const i = edges[e], j = edges[e + 1]; nbr[at[i]++] = j; nbr[at[j]++] = i; }
    const elem = new Int32Array(n); for (let k = 0; k < n; k++) { const x = symbols.indexOf(m.atoms[k][0]); elem[k] = x < 0 ? S - 1 : x; }
    const base = Float32Array.from(t.baseline), width = Math.max(S, ...steps.map((s) => Math.max(s.o, s.ip)));
    const A = new Float32Array(n * width), B = new Float32Array(n * width);
    return (known) => {
      A.fill(0, 0, n * S);
      for (let k = 0; k < n; k++) { const z = k * S; if ((known >> k) & 1) A[z + elem[k]] = 1; else for (let q = 0; q < S; q++) A[z + q] = base[q]; }
      let dim = S, rows = n, src = A, dst = B;
      for (const s of steps) {
        if (s.gather) { for (let k = 0; k < rows; k++) { const z = k * dim; for (let q = 0; q < dim; q++) dst[z + q] = src[z + q]; for (let p = off[k]; p < off[k + 1]; p++) { const y = nbr[p] * dim; for (let q = 0; q < dim; q++) dst[z + q] += src[y + q]; } } [src, dst] = [dst, src]; }
        if (s.sum) { for (let q = 0; q < dim; q++) { let acc = 0; for (let k = 0; k < rows; k++) acc += src[k * dim + q]; dst[q] = acc; } [src, dst] = [dst, src]; rows = 1; }
        for (let k = 0; k < rows; k++) { const i0 = k * dim, o0 = k * s.o; for (let r = 0; r < s.o; r++) { let acc = s.b[r]; const w0 = r * s.ip; for (let q = 0; q < s.ip; q++) acc += s.w[w0 + q] * src[i0 + q]; dst[o0 + r] = s.relu ? (acc > 0 ? acc : 0) : acc; } }
        [src, dst] = [dst, src]; dim = s.o;
      }
      return src[0];
    };
  }
  const count = (x) => { let c = 0; while (x) { c += x & 1; x >>= 1; } return c; };
  onmessage = (event) => {
    const { id, symbols, m, t } = event.data, game = compile(symbols, m, t), n = m.atoms.length, N = 1 << n;
    const v = new Float64Array(N), every = Math.max(1, N >> 4);
    for (let T = 0; T < N; T++) { v[T] = game(T); if (T % every === 0) postMessage({ id, progress: T / N }); }
    const mo = Float64Array.from(v); // the Möbius transform, in place
    for (let i = 0; i < n; i++) { const bit = 1 << i; for (let S = 0; S < N; S++) if (S & bit) mo[S] -= mo[S ^ bit]; }
    // SII of order 2 (each Möbius value shared out to its subsets of 1 and 2 atoms, 1 / (|W| - |T| + 1)),
    // then k-SII (Bernoulli numbers B0 = 1, B1 = -1/2): as shapiq's MoebiusConverter
    const sii = new Map();
    for (let W = 1; W < N; W++) {
      const mw = mo[W]; if (mw === 0) continue;
      const size = count(W), members = []; for (let i = 0; i < n; i++) if ((W >> i) & 1) members.push(i);
      for (let x = 0; x < members.length; x++) {
        const a = 1 << members[x]; sii.set(a, (sii.get(a) || 0) + mw / size);
        for (let y = x + 1; y < members.length; y++) { const ab = a | (1 << members[y]); sii.set(ab, (sii.get(ab) || 0) + mw / (size - 1)); }
      }
    }
    const ksii = new Map(sii);
    for (const [S, value] of sii) if (count(S) === 2) for (let i = 0; i < n; i++) if ((S >> i) & 1) ksii.set(1 << i, (ksii.get(1 << i) || 0) - 0.5 * value);
    const shapley = new Float64Array(n);
    for (let S = 1; S < N; S++) { if (mo[S] === 0) continue; const s = count(S); for (let j = 0; j < n; j++) if ((S >> j) & 1) shapley[j] += mo[S] / s; }
    const pairs = [...ksii].filter(([S]) => count(S) === 2).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const own = Array.from({ length: n }, (_, i) => ksii.get(1 << i) || 0);
    postMessage({ id, done: true, n, budget: N, logOdds: v[N - 1], empty: v[0], shapley: Array.from(shapley), own, pairs: pairs.slice(0, 40) });
  };`;

  G.mount("graphshapiq-lab-demo", "SHAPIQ_GRAPHSHAPIQ_LAB", (root, data) => {
    const tasks = data.tasks;
    const state = { slots: PRESETS[0].slots.slice(), task: tasks.find((t) => t.id === PRESETS[0].task) || tasks[0], sel: null, view: "together", result: null, focus: null, row: null };
    let mol = null, runId = 0, timer = 0;
    const worker = new Worker(URL.createObjectURL(new Blob([WORKER], { type: "text/javascript" })));
    worker.onmessage = (event) => {
      const d = event.data;
      if (d.id !== runId) return; // an earlier molecule's
      if (d.progress != null) progressFill.style.width = `${Math.round(d.progress * 100)}%`;
      else { progress.hidden = true; state.result = d; draw(); renderPanel(); }
    };

    // -- the frame --------------------------------------------------------------------------------
    const bar = h("div", "bar"), presets = h("div", "presets"), taskSwitch = h("div", "switch"), viewSwitch = h("div", "switch");
    presets.setAttribute("role", "group"); presets.setAttribute("aria-label", "Example molecules");
    taskSwitch.setAttribute("role", "group"); taskSwitch.setAttribute("aria-label", "Question");
    viewSwitch.setAttribute("role", "group"); viewSwitch.setAttribute("aria-label", "Explanation");
    const switches = h("div", "switches", viewSwitch, taskSwitch);
    bar.append(presets, switches);
    const body = h("div", "body"), stage = h("div", "stage"), frame = h("div", "frame"), panel = h("div", "panel");
    const plot = svg("svg", { class: "plot", viewBox: "0 0 100 100", role: "group", "aria-label": "The molecule you build" }, frame);
    const slotLayer = svg("g", { class: "slots" });
    const progress = h("div", "progress"), progressFill = h("i");
    progress.append(progressFill); progress.hidden = true; frame.append(progress);
    const caption = h("p", "caption"), palette = h("div", "palette");
    palette.setAttribute("role", "group"); palette.setAttribute("aria-label", "Groups to place");
    stage.append(frame, caption, palette);
    const live = h("p", "sr"); live.setAttribute("aria-live", "polite");
    body.append(stage, panel);
    root.append(bar, body, live);
    const scene = G.createScene(plot);

    const presetButtons = PRESETS.map((p) => {
      const b = h("button", "preset", p.name); b.type = "button";
      b.addEventListener("click", () => { state.slots = p.slots.slice(); state.task = tasks.find((t) => t.id === p.task) || state.task; state.sel = null; rebuild(); });
      presets.append(b);
      return b;
    });
    const taskButtons = tasks.map((t) => { const b = h("button", "opt", t.label); b.type = "button"; b.addEventListener("click", () => { state.task = t; compute(); chrome(); }); taskSwitch.append(b); return b; });
    const VIEWS = [["alone", "Atoms alone"], ["together", "Atoms together"]];
    const viewButtons = VIEWS.map(([id, label]) => { const b = h("button", "opt", label); b.type = "button"; b.addEventListener("click", () => { state.view = id; state.focus = null; state.row = null; chrome(); draw(); renderPanel(); }); viewSwitch.append(b); return b; });
    for (const id of [...PALETTE, "none"]) {
      const b = h("button", id === "none" ? "chip clear" : "chip", id === "none" ? "Remove" : GROUPS[id].label); b.type = "button";
      b.title = id === "none" ? "Remove the group" : `Place ${GROUPS[id].label}`;
      b.addEventListener("click", () => place(id));
      palette.append(b);
    }

    const heavy = (slots) => 6 + slots.reduce((n, s) => n + GROUPS[s].heavy, 0);
    let note = "";
    function place(id) {
      if (state.sel == null) { note = "First pick a position on the ring: a + or a group."; caption_(); return; }
      const trial = state.slots.slice(); trial[state.sel] = id;
      if (heavy(trial) > MAX_ATOMS) { note = `That would make ${heavy(trial)} atoms; the lab computes exactly up to ${MAX_ATOMS}.`; caption_(); return; }
      state.slots = trial; rebuild();
    }
    function rebuild() { note = ""; mol = assemble(state.slots); scene.setMolecule(mol); plot.appendChild(slotLayer); slots(); compute(); chrome(); }
    function slots() {
      slotLayer.replaceChildren();
      const { s, pos } = scene.geom;
      for (let i = 0; i < 6; i++) {
        const a = Math.PI / 2 - (i * Math.PI) / 3, [x, y] = pos[i], empty = state.slots[i] === "none";
        const cx = empty ? x + Math.cos(a) * 0.62 * s : x, cy = empty ? y + Math.sin(a) * 0.62 * s : y;
        const g = svg("g", { class: `slot${empty ? "" : " filled"}${state.sel === i ? " sel" : ""}`, tabindex: "0", role: "button", "aria-pressed": String(state.sel === i), "aria-label": empty ? `Position ${i + 1}: add a group` : `Position ${i + 1}: ${GROUPS[state.slots[i]].name}` }, slotLayer);
        if (empty) {
          svg("circle", { cx, cy, r: 0.26 * s }, g);
          svg("line", { class: "plus", x1: cx - 0.12 * s, y1: cy, x2: cx + 0.12 * s, y2: cy }, g);
          svg("line", { class: "plus", x1: cx, y1: cy - 0.12 * s, x2: cx, y2: cy + 0.12 * s }, g);
        } else svg("circle", { class: "ringpick", cx, cy, r: 0.42 * s }, g);
        const pick = () => { state.sel = state.sel === i ? null : i; note = ""; slots(); caption_(); };
        g.addEventListener("click", (e) => { e.stopPropagation(); pick(); });
        g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
      }
    }
    function caption_() {
      const name = KNOWN[state.slots.join()] || "A substituted benzene";
      const help = note || (state.sel != null ? `Position ${state.sel + 1}: pick a group below, or remove it.` : "Pick a + on the ring, then a group below; or start from an example.");
      const line = h("span", `blurb${note ? " note" : ""}`, help);
      caption.replaceChildren(h("strong", "", name), formula(mol.formula), line);
    }
    function chrome() {
      taskButtons.forEach((b, k) => b.setAttribute("aria-pressed", String(tasks[k] === state.task)));
      viewButtons.forEach((b, k) => b.setAttribute("aria-pressed", String(VIEWS[k][0] === state.view)));
      presetButtons.forEach((b, k) => b.setAttribute("aria-pressed", String(PRESETS[k].slots.join() === state.slots.join())));
      caption_();
    }

    // -- computing: debounced, and only the latest molecule counts ---------------------------------
    function compute() {
      state.result = null; state.focus = null; state.row = null;
      draw(); renderPanel();
      clearTimeout(timer);
      const id = ++runId, m = { atoms: mol.atoms, bonds: mol.bonds }, t = { gnn: state.task.gnn, baseline: state.task.baseline };
      progressFill.style.width = "0%";
      timer = setTimeout(() => { progress.hidden = false; worker.postMessage({ id, symbols: data.symbols, m, t }); }, 120);
    }

    // -- the molecule ------------------------------------------------------------------------------
    function draw() {
      const d = state.result;
      if (!d) { scene.update({ rings: {}, groups: [], masked: new Set(), focus: null }); return; }
      const ownValues = state.view === "alone" ? d.shapley : d.own, own = Object.fromEntries(ownValues.map((v, k) => [k, v]));
      const pairs = d.pairs.map(([c, v]) => ({ atoms: bits(c), v }));
      let groups = [];
      if (state.view === "together") groups = state.row ? [state.row] : state.focus != null ? pairs.filter((g) => g.atoms.includes(state.focus)).slice(0, 8) : pairs.slice(0, 10);
      scene.update({
        rings: own, ringLargest: Math.max(1e-9, ...ownValues.map(Math.abs)), groups, groupLargest: Math.max(1e-9, Math.abs(pairs[0]?.v || 0)),
        masked: new Set(), focus: state.view === "together" ? state.focus : null, dim: state.row ? new Set(state.row.atoms) : null,
      });
    }
    plot.addEventListener("pointermove", (e) => {
      if (e.pointerType === "touch" || !state.result || state.view !== "together") return;
      const k = scene.atomAt(e), f = k < 0 ? null : k;
      if (state.focus !== f) { state.focus = f; state.row = null; draw(); renderPanel(); }
    });
    plot.addEventListener("pointerleave", () => { if (state.focus != null) { state.focus = null; draw(); renderPanel(); } });

    // -- the panel ---------------------------------------------------------------------------------
    function renderPanel() {
      const t = state.task, d = state.result, box = h("div", "panel-in");
      const question = `${t.question[0].toUpperCase()}${t.question.slice(1)}`;
      if (!d) {
        box.append(h("p", "pred-model", question), h("p", "computing", h("span", "spin"), "Computing the exact interactions…"));
        panel.replaceChildren(box);
        return;
      }
      const p = sig(d.logOdds);
      const scale = h("div", "scale"), fill = h("span", "scale-fill"), mark = h("span", "scale-mark");
      fill.style.width = `${p * 100}%`; mark.style.left = `${p * 100}%`; scale.setAttribute("aria-hidden", "true");
      scale.append(fill, h("span", "scale-half"), mark, h("span", "scale-label", t.no), h("span", "scale-label r", t.yes));
      box.append(h("p", "pred-model", `${question} The GNN predicts`), h("p", "pred-number", h("span", "pred-value", percent(p)), h("span", "verdict", p >= 0.5 ? t.yes : t.no)), scale);
      const title = h("p", "title"), list = h("ol", "rows");
      if (state.view === "alone") {
        title.append(h("strong", "", "Atoms alone"), "their Shapley values");
        const rows = d.shapley.map((v, k) => ({ atoms: [k], v })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
        const largest = Math.max(1e-9, Math.abs(rows[0].v));
        for (const g of rows.slice(0, 5)) list.append(G.row(mol, g.atoms, g.v, largest, G.atomName(mol, g.atoms[0])));
      } else {
        title.append(h("strong", "", "Atoms together"), state.focus != null ? `the pairs of ${G.atomName(mol, state.focus).toLowerCase()} ${state.focus + 1}` : "the strongest pairs");
        const pairs = d.pairs.map(([c, v]) => ({ atoms: bits(c), v })), largest = Math.max(1e-9, Math.abs(pairs[0]?.v || 0));
        const rows = state.focus != null ? pairs.filter((g) => g.atoms.includes(state.focus)) : pairs;
        for (const g of rows.slice(0, 5)) {
          const li = G.row(mol, g.atoms, g.v, largest, G.groupName(mol, g.atoms));
          li.tabIndex = 0; li.setAttribute("aria-label", `${G.groupName(mol, g.atoms)}: ${signed(g.v)}`);
          const on = () => { state.row = g; draw(); for (const o of list.children) o.classList.toggle("dim", o !== li); };
          const off = () => { state.row = null; draw(); for (const o of list.children) o.classList.remove("dim"); };
          li.addEventListener("pointerenter", (ev) => { if (ev.pointerType !== "touch") on(); });
          li.addEventListener("pointerleave", (ev) => { if (ev.pointerType !== "touch") off(); });
          li.addEventListener("focus", on); li.addEventListener("blur", off);
          list.append(li);
        }
        if (!rows.length) list.append(h("li", "none", "No pairs of this atom."));
      }
      const meta = h("p", "meta");
      meta.append(
        h("span", null, "Computed ", h("strong", "", "exactly"), ", in your browser, just now: GraphSHAP-IQ evaluated the GNN on ", h("strong", "", d.budget.toLocaleString("en")), ` maskings (all 2${sup(d.n)}).`),
        h("span", null, `GIN, 3 layers: a score of `, value(d.logOdds), ", the masked molecule’s ", value(d.empty), "."),
      );
      box.append(title, list, legend(t), meta);
      panel.replaceChildren(box);
      live.textContent = `The GNN predicts ${percent(p)}, ${p >= 0.5 ? t.yes : t.no}.`;
    }

    rebuild();
  });
})();
