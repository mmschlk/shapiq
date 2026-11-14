// The GraphSHAP-IQ explorer (demos/graphshapiq-explorer/index.rst): eight molecules, three GNNs
// (one per question), and every molecule explained for every question, exactly, with GraphSHAP-IQ.
//
// Over the molecule, the question in plain words (the library's `question`): what it asks, the
// pattern that makes a yes, whether the molecule has it, and what the GNN answers. Under it, three
// views: hide atoms (click atoms to hide them; the GNN runs in the page and answers again), the
// atoms alone (their Shapley values, a ring around each atom), and the atoms together (k-SII of
// orders 2 to 4 and the Möbius interactions: bands between two atoms, shapes around more). The
// explanations are precomputed in data.js (../../graphshapiq-common/compute); the shared library
// (../../graphshapiq-common/assets/demo.js, `ShapiqGraph`) draws and animates the molecule.
//
// The ninth molecule is the reader's own ("Build your own"): groups on the six positions of a
// benzene ring, placed in a fourth view, "Build": a click on a position opens a wheel of the groups
// that can go there. Nothing of it is precomputed. The page runs the GNN for its answer at once, and a Web Worker (`WORKER`) explains it with GraphSHAP-IQ: it
// evaluates the GNN on the maskings inside the receptive fields, and computes the Möbius
// interactions and from them the k-SII of orders 1 to 4, as shapiq does
// (../../graphshapiq-common/compute/check_worker.py compares them with shapiq's). A molecule has at
// most MAX_ATOMS heavy atoms, so that this takes a few seconds at most.
(() => {
  const G = window.ShapiqGraph;
  if (!G) return console.error("The GraphSHAP-IQ explorer needs ../graphshapiq-common/demo.js first.");
  const { h, svg, sig, bits, signed, percent, legend } = G;
  const VIEWS = [["build", "Build"], ["predict", "Hide atoms"], ["alone", "Atoms alone"], ["together", "Atoms together"]];
  const ORDERS = { 2: ["Pairs", "2-SII"], 3: ["Up to 3 atoms", "3-SII"], 4: ["Up to 4 atoms", "4-SII"], M: ["Any size", "Möbius"] };

  const MAX_ATOMS = 15; // of a molecule the reader builds: at most 2^15 maskings
  const OVERALL = 24, PER_ATOM = 8; // the interactions of an order the worker sends: as compute/pack.py keeps

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
  // the molecules to start from: the empty ring (for the question that is asked) and catechol, the
  // example of a PAINS. Move one of its OH groups, and it is none
  const STARTS = [
    { name: "Empty ring", slots: ["none", ...none], task: null },
    { name: "Catechol", slots: ["OH", "OH", ...none.slice(1)], task: "PAINS" },
  ];
  const KNOWN = Object.fromEntries([
    [["OH", "OH", ...none.slice(1)].join(), "Catechol"], [["OH", "none", "OH", ...none.slice(2)].join(), "Resorcinol"],
    [["OH", "none", "none", "OH", "none", "none"].join(), "Hydroquinone"], [["NO2", ...none].join(), "Nitrobenzene"],
    [["none", ...none].join(), "Benzene"], [["NO2", "none", "NO2", ...none.slice(2)].join(), "1,3-Dinitrobenzene"],
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
    const formulaText = ["C", "H", "Cl", "N", "O"].filter((e) => counts[e]).map((e) => e + (counts[e] > 1 ? counts[e] : "")).join("");
    return { atoms, bonds, rings: [[0, 1, 2, 3, 4, 5]], formula: formulaText };
  }
  // the atoms of a question's pattern in the molecule of `slots` (the ground truth the demos
  // outline, as the patterns of ../../graphshapiq-common/compute/config.toml find them): a catechol
  // is the ring with two neighbouring OH groups, a nitro group a nitrogen with its two oxygens
  function patternOf(task, slots) {
    const first = [], atoms = new Set();
    let next = 6;
    for (const slot of slots) { first.push(next); next += GROUPS[slot].atoms.length; }
    if (task.id === "PAINS") {
      for (let i = 0; i < 6; i++) if (slots[i] === "OH" && slots[(i + 1) % 6] === "OH") [0, 1, 2, 3, 4, 5, first[i], first[(i + 1) % 6]].forEach((k) => atoms.add(k));
    } else if (task.id === "mutagenic") {
      slots.forEach((slot, i) => { if (slot === "NO2") [0, 1, 2].forEach((k) => atoms.add(first[i] + k)); });
    }
    return [...atoms].sort((a, b) => a - b);
  }

  const TRASH = "M3.2 4.6h9.6M6.4 4.6V3.1h3.2v1.5M4.6 4.6l.6 8.3h5.6l.6-8.3M6.9 7v3.6M9.1 7v3.6"; // the wheel's "remove"

  // the worker: GraphSHAP-IQ for the molecule it is sent (see the header)
  const WORKER = `
  const floats = (s) => { const bin = atob(s), buf = new ArrayBuffer(bin.length), u = new Uint8Array(buf); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new Float32Array(buf); };
  // the GNN as a game: the log-odds for the molecule whose atoms in the bitmask are known
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
    const game = (known) => {
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
    // the receptive fields: every atom with the atoms within as many bonds as the GNN has layers
    const hops = steps.filter((s) => s.gather).length, fields = new Set();
    for (let k = 0; k < n; k++) {
      let field = 1 << k, edge = [k];
      for (let hop = 0; hop < hops; hop++) { const next = []; for (const i of edge) for (let p = off[i]; p < off[i + 1]; p++) if (!((field >> nbr[p]) & 1)) { field |= 1 << nbr[p]; next.push(nbr[p]); } edge = next; }
      fields.add(field);
    }
    return { game, fields: [...fields] };
  }
  const count = (x) => { let c = 0; while (x) { c += x & 1; x >>= 1; } return c; };
  // the interactions a list keeps: every atom alone, the strongest groups, and every atom's strongest
  function strongest(values, n, overall, perAtom) {
    values.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const groups = values.filter(([S]) => S & (S - 1)), keep = new Set(groups.slice(0, overall).map(([S]) => S));
    for (let k = 0; k < n; k++) { let left = perAtom; for (const [S] of groups) { if (!left) break; if ((S >> k) & 1) { keep.add(S); left--; } } }
    return values.filter(([S]) => !(S & (S - 1)) || keep.has(S)).map(([S, v]) => [S, Math.round(v * 1e4) / 1e4]);
  }
  onmessage = (event) => {
    const { id, symbols, m, t, overall, perAtom } = event.data, { game, fields } = compile(symbols, m, t), n = m.atoms.length, N = 1 << n;
    // GraphSHAP-IQ: the GNN is evaluated on the maskings inside a receptive field only. The Möbius
    // interaction of any other group of atoms is zero, as its atoms never meet in the GNN
    const inside = new Uint8Array(N);
    let budget = 0;
    for (let T = 0; T < N; T++) for (const field of fields) if (!(T & ~field)) { inside[T] = 1; budget++; break; }
    const mo = new Float64Array(N), every = Math.max(1, budget >> 4);
    let done = 0;
    for (let T = 0; T < N; T++) { if (!inside[T]) continue; mo[T] = game(T); if (done++ % every === 0) postMessage({ id, progress: done / budget }); }
    const empty = mo[0];
    // the Möbius transform, in place: a group inside a field has all its subsets inside it too
    for (let i = 0; i < n; i++) { const bit = 1 << i; for (let S = 0; S < N; S++) if (S & bit) mo[S] -= mo[S ^ bit]; }
    for (let S = 0; S < N; S++) if (!inside[S]) mo[S] = 0;
    // SII up to order 4: each Möbius value shared out to its subsets S, by 1 / (|W| - |S| + 1)
    const K = Math.min(4, n), sii = new Float64Array(N), small = [], moebius = [];
    for (let W = 1; W < N; W++) {
      const mw = mo[W]; if (mw === 0) continue;
      moebius.push([W, mw]);
      const at = []; for (let i = 0; i < n; i++) if ((W >> i) & 1) at.push(1 << i);
      const size = at.length;
      for (let a = 0; a < size; a++) {
        sii[at[a]] += mw / size;
        for (let b = a + 1; b < size && K > 1; b++) {
          const ab = at[a] | at[b]; sii[ab] += mw / (size - 1);
          for (let c = b + 1; c < size && K > 2; c++) {
            const abc = ab | at[c]; sii[abc] += mw / (size - 2);
            for (let d = c + 1; d < size && K > 3; d++) sii[abc | at[d]] += mw / (size - 3);
          }
        }
      }
    }
    for (let S = 1; S < N; S++) { const size = count(S); if (size <= K) small.push([S, size]); }
    // k-SII of orders 1 (the Shapley values) to 4: the SII of a group and, weighted by the
    // Bernoulli numbers, of the groups up to the order it is in, as shapiq aggregates them
    const bernoulli = [1, -0.5, 1 / 6, 0], kSII = {};
    for (let k = 1; k <= K; k++) {
      const values = new Map();
      for (const [T, size] of small) {
        if (size > k || sii[T] === 0) continue;
        for (let S = T; S; S = (S - 1) & T) values.set(S, (values.get(S) || 0) + bernoulli[size - count(S)] * sii[T]);
      }
      kSII[k] = strongest([...values], n, overall, perAtom);
    }
    postMessage({ id, done: true, nAtoms: n, budget, logOdds: game(N - 1), empty, kSII, moebius: strongest(moebius, n, overall, perAtom), moebiusCount: moebius.length });
  };`;

  G.mount("graphshapiq-explorer-demo", "SHAPIQ_GRAPHSHAPIQ_EXPLORER", (root, data) => {
    const taskById = (id) => data.tasks.find((t) => t.id === id);
    const first = data.molecules[0];
    const state = { mol: first, task: taskById(first.task), view: "together", order: 2, masked: new Set(), focus: null, pinned: false, row: null, hinted: false };
    // the reader's own molecule: the groups on the ring's positions, the position whose wheel is
    // open (`sel`) and where the positions are on the stage (`at`), the molecule they make, and its explanation (`result`: at first the GNN's answer only, `pending`)
    const build = { slots: STARTS[1].slots.slice(), task: taskById(STARTS[1].task), sel: null, at: [], note: "", mol: null, result: null, run: 0, timer: 0, worker: null, busy: false, picked: false };
    const built = () => state.mol.built === true;
    const ex = () => (built() ? build.result : data.explanations[`${state.mol.id}/${state.task.id}`]);
    const rowsOf = (e, order) => (order === "M" ? e.moebius : e.kSII[String(order)] || []);
    const singles = (e, order) => { const o = {}; for (const [mask, v] of rowsOf(e, order)) { const a = bits(mask); if (a.length === 1) o[a[0]] = v; } return o; };
    const groupsOf = (e, order) => rowsOf(e, order).map(([mask, v]) => ({ atoms: bits(mask), v })).filter((g) => g.atoms.length > 1).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
    const orderSeq = () => { const s = []; for (let k = 2; k <= (built() ? 4 : Math.min(4, Object.keys(ex().kSII).length)); k++) s.push(k); s.push("M"); return s; };

    // -- the frame --------------------------------------------------------------------------------
    const bar = h("div", "gsq-bar"), switches = h("div", "gsq-switches");
    // the tile of the reader's own molecule: a ring with a + where a group can go
    const own = { ...assemble(["CH3", ...none]), name: "Build your own" };
    const mols = G.picker("Molecule", [...data.molecules, own], (k) => {
      if (k === data.molecules.length) { // the reader's own molecule, as they left it
        state.task = build.task; state.view = "build";
        if (build.mol) { state.mol = build.mol; reset(true); } else rebuild();
        return;
      }
      state.mol = data.molecules[k]; state.task = taskById(state.mol.task); state.order = 2;
      if (state.view === "build") state.view = "together";
      reset(true);
    });
    {
      const tile = mols.buttons[data.molecules.length], icon = tile.querySelector("svg"), [x, y] = G.geom(own, 78, 21).pos[6];
      tile.classList.add("gsq-mol-own");
      svg("circle", { class: "gsq-plus-ping", cx: x, cy: y, r: 14 }, icon);
      svg("circle", { class: "gsq-plus-o", cx: x, cy: y, r: 14 }, icon);
      svg("line", { x1: x - 6.5, y1: y, x2: x + 6.5, y2: y }, icon); svg("line", { x1: x, y1: y - 6.5, x2: x, y2: y + 6.5 }, icon);
    }
    const views = G.switcher("View", VIEWS.map((v) => v[1]), (k) => setView(VIEWS[k][0]));
    const ask = G.question(data.tasks, (t) => { state.task = t; if (built()) compute(); reset(false); });
    const info = h("button", "gsq-info", "i");
    info.type = "button"; info.setAttribute("aria-label", "What does the demo show?"); info.setAttribute("aria-expanded", "false");
    const about = h("div", "gsq-about"); about.hidden = true;
    about.innerHTML =
      '<button class="gsq-about-close" type="button" aria-label="Close">×</button>' +
      "<p><strong>What does the demo show?</strong></p>" +
      "<p>A <strong>graph neural network</strong> (GNN) reads a molecule as a graph: atoms, linked by bonds. In each of its three layers, every atom gathers what its neighbours know. At the end, the GNN sums its atoms and answers one yes/no question, each question with a GNN of its own.</p>" +
      "<p>To explain an answer, <code>shapiq</code> hides atoms in many combinations and watches the answer change. The atoms are the <strong>players</strong>: what each adds alone, and what groups add only together, their <strong>interactions</strong>.</p>" +
      "<p>For a GNN these are computed <strong>exactly</strong>: atoms more than three bonds apart cannot interact, so a fraction of all combinations suffices.</p>";
    switches.append(views.node, info);
    bar.append(mols.node, ask.node, switches, about);
    const setAbout = (open) => { about.hidden = !open; info.setAttribute("aria-expanded", String(open)); };
    info.addEventListener("click", () => setAbout(about.hidden));
    about.querySelector(".gsq-about-close").addEventListener("click", () => { setAbout(false); info.focus(); });
    about.addEventListener("focusout", (e) => { if (!about.contains(e.relatedTarget) && e.relatedTarget !== info) setAbout(false); });

    const body = h("div", "gsq-body"), stage = h("div", "gsq-stage"), frame = h("div", "gsq-frame"), panel = h("div", "gsq-panel");
    const plot = svg("svg", { class: "gsq-plot", viewBox: "0 0 100 100", role: "group", "aria-label": "The molecule; its atoms are the players" }, frame);
    const hint = G.hinter(frame);
    const slotLayer = svg("g", { class: "gsq-slots" });
    const progress = h("div", "gsq-progress"), progressFill = h("i");
    progress.append(progressFill); progress.hidden = true; frame.append(progress);
    // the molecules to start from, as drawings along the bottom of the frame (in the view "Build" only)
    const starts = h("div", "gsq-starts");
    starts.setAttribute("role", "group"); starts.setAttribute("aria-label", "Start from"); starts.hidden = true;
    const startButtons = STARTS.map((s, k) => {
      const b = h("button", "gsq-start", G.tile(assemble(s.slots))); b.type = "button";
      b.title = `Start from ${s.name.toLowerCase().replace("empty", "an empty")}`; b.setAttribute("aria-label", b.title);
      b.addEventListener("click", () => start(k));
      starts.append(b);
      return b;
    });
    frame.append(starts);
    const scrim = h("div", "gsq-scrim"), wheel = h("div", "gsq-wheel");
    scrim.hidden = wheel.hidden = true; wheel.tabIndex = -1; wheel.setAttribute("role", "group");
    frame.append(scrim, wheel);
    stage.append(frame);
    const live = h("p", "gsq-sr"); live.setAttribute("aria-live", "polite");
    body.append(stage, panel);
    root.append(bar, body, live);
    const scene = G.createScene(plot);

    function setView(id) { hideWheel(); state.view = id; state.focus = null; state.pinned = false; state.row = null; update(); views.buttons[VIEWS.findIndex((v) => v[0] === id)].focus({ preventScroll: true }); }
    function reset(newMolecule) {
      hideWheel();
      state.masked = new Set(); state.focus = null; state.pinned = false; state.row = null;
      if (!orderSeq().includes(state.order)) state.order = 2;
      if (newMolecule) scene.setMolecule(state.mol);
      update();
    }

    // -- the reader's own molecule: its groups, and its explanation by the worker -----------------
    const heavy = (slots) => 6 + slots.reduce((n, s) => n + GROUPS[s].heavy, 0);
    // a known molecule's name, wherever on the ring its groups are: every rotation, in both directions
    function nameOf(slots) {
      for (const turn of [slots, [slots[0], ...slots.slice(1).reverse()]]) for (let r = 0; r < 6; r++) { const name = KNOWN[[...turn.slice(r), ...turn.slice(0, r)].join()]; if (name) return name; }
      return "A substituted benzene";
    }
    function rebuild() {
      build.note = "";
      build.mol = { ...assemble(build.slots), id: "own", built: true, name: nameOf(build.slots) };
      state.mol = build.mol;
      compute(); // the worker starts later, in a timer: a failure there cannot leave the scene behind
      reset(true);
    }
    // a group for the position whose wheel is open ("none" removes the one there)
    function place(id) {
      const i = build.sel, trial = build.slots.slice();
      if (i == null) return;
      trial[i] = id;
      if (heavy(trial) > MAX_ATOMS) return; // the wheel's hub says so
      if (trial.join() === build.slots.join()) { closeWheel(true); return; }
      build.slots = trial;
      rebuild();
      slotLayer.children[i]?.focus({ preventScroll: true });
    }
    function start(k) { const s = STARTS[k]; build.slots = s.slots.slice(); build.picked = true; if (s.task) state.task = taskById(s.task); rebuild(); }
    // the GNN's answer is one evaluation, at once; the worker explains the latest molecule only. A
    // job still running is stopped (the worker is replaced), so that jobs do not queue up
    function stopWorker() { if (build.worker) build.worker.terminate(); build.worker = null; build.busy = false; }
    function failed() {
      stopWorker(); progress.hidden = true;
      build.result = { ...build.result, pending: false, failed: true };
      if (built()) { draw(); renderPanel(); }
    }
    function startWorker() {
      try { build.worker = new Worker(URL.createObjectURL(new Blob([WORKER], { type: "text/javascript" }))); } catch (err) { console.error(err); return false; }
      build.worker.onerror = (err) => { console.error(err.message || err); failed(); };
      build.worker.onmessage = ({ data: d }) => {
        if (d.id !== build.run) return; // an earlier molecule's
        if (d.progress != null) { progressFill.style.width = `${Math.round(d.progress * 100)}%`; return; }
        build.busy = false; progress.hidden = true;
        build.result = { ...build.result, ...d, p: sig(d.logOdds), pending: false };
        if (built()) { draw(); renderPanel(); }
      };
      return true;
    }
    function compute() {
      const m = build.mol, t = build.task = state.task, logOdds = G.predict(data.symbols, m, t, new Set());
      build.result = { pending: true, logOdds, p: sig(logOdds), nAtoms: m.atoms.length, truth: patternOf(t, build.slots), kSII: {}, moebius: [], moebiusCount: 0 };
      clearTimeout(build.timer);
      const id = ++build.run;
      progressFill.style.width = "0%";
      build.timer = setTimeout(() => {
        if (build.busy) stopWorker();
        if (!build.worker && !startWorker()) return failed();
        build.busy = true; progress.hidden = !built();
        build.worker.postMessage({ id, symbols: data.symbols, m: { atoms: m.atoms, bonds: m.bonds }, t: { gnn: t.gnn, baseline: t.baseline }, overall: OVERALL, perAtom: PER_ATOM });
      }, 120);
    }
    // the positions of the ring, over the molecule: a + where a group can go, the group where one is
    function slots() {
      const focused = slotLayer.contains(document.activeElement) ? Number(document.activeElement.dataset.slot) : null;
      slotLayer.replaceChildren();
      const { s, pos } = scene.geom;
      let next = 6; // the atoms of a position's group follow the ring's, as `assemble` puts them
      for (let i = 0; i < 6; i++) {
        const a = Math.PI / 2 - (i * Math.PI) / 3, [x, y] = pos[i], group = GROUPS[build.slots[i]], empty = !group.heavy;
        const g = svg("g", { class: `gsq-slot${empty ? "" : " gsq-filled"}${build.sel === i ? " gsq-sel" : ""}`, tabindex: "0", role: "button", "data-slot": i, "aria-expanded": String(build.sel === i), "aria-label": empty ? `Position ${i + 1}: add a group` : `Position ${i + 1}: ${group.name}. Change or remove it` }, slotLayer);
        if (empty) {
          const cx = x + Math.cos(a) * 0.62 * s, cy = y + Math.sin(a) * 0.62 * s;
          build.at[i] = [cx, cy];
          svg("circle", { cx, cy, r: 0.26 * s }, g);
          svg("line", { class: "gsq-plus", x1: cx - 0.12 * s, y1: cy, x2: cx + 0.12 * s, y2: cy }, g);
          svg("line", { class: "gsq-plus", x1: cx, y1: cy - 0.12 * s, x2: cx, y2: cy + 0.12 * s }, g);
        } else {
          build.at[i] = pos[next];
          for (const [p, q] of group.bonds) svg("line", { class: "gsq-reach", x1: pos[next + p][0], y1: pos[next + p][1], x2: pos[next + q][0], y2: pos[next + q][1], "stroke-width": 0.5 * s }, g);
          for (let k = 0; k < group.heavy; k++) svg("circle", { cx: pos[next + k][0], cy: pos[next + k][1], r: 0.42 * s }, g);
          next += group.heavy;
        }
        const pick = (byKey) => { build.note = ""; build.picked = true; if (build.sel === i) closeWheel(byKey); else openWheel(i, byKey); };
        g.addEventListener("click", (e) => { e.stopPropagation(); pick(false); });
        g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.stopPropagation(); pick(true); } });
      }
      if (slotLayer.parentNode !== plot || plot.lastChild !== slotLayer) plot.appendChild(slotLayer);
      if (focused != null && !Number.isNaN(focused)) slotLayer.children[focused]?.focus({ preventScroll: true });
    }
    // the wheel: the groups that can go to a position, around the position (and "remove", where a
    // group is). The position shows through its hub, which names the group pointed at and closes it
    function openWheel(i, byKey) {
      build.sel = i;
      draw(); renderPanel();
      const there = build.slots[i], ids = there === "none" ? PALETTE : [...PALETTE, "none"], rest = there === "none" ? "Add a group" : "Change the group";
      const name = h("span", "gsq-wheel-name"), hub = h("button", "gsq-wheel-hub", name); hub.type = "button"; hub.setAttribute("aria-label", "Close");
      hub.addEventListener("click", () => closeWheel(true));
      const say = (text = "", large = false) => { name.textContent = text; name.classList.toggle("gsq-off", large); }; // the group pointed at
      wheel.replaceChildren(hub);
      ids.forEach((id, k) => {
        const group = GROUPS[id], trial = build.slots.slice(); trial[i] = id;
        const large = heavy(trial) > MAX_ATOMS, angle = -Math.PI / 2 + (k * 2 * Math.PI) / ids.length;
        const b = h("button", id === "none" ? "gsq-wheel-item gsq-clear" : "gsq-wheel-item"); b.type = "button";
        if (id === "none") svg("path", { d: TRASH }, svg("svg", { viewBox: "0 0 16 16", "aria-hidden": "true" }, b)); else b.textContent = group.label;
        b.setAttribute("aria-label", id === "none" ? `Remove the ${GROUPS[there].name} group` : `${group.label} (${group.name})${large ? ", too large here" : ""}`);
        if (large) b.setAttribute("aria-disabled", "true"); else if (id === there) b.setAttribute("aria-pressed", "true");
        b.style.setProperty("--dx", Math.cos(angle).toFixed(4)); b.style.setProperty("--dy", Math.sin(angle).toFixed(4)); b.style.setProperty("--i", k);
        const on = () => say(large ? `At most ${MAX_ATOMS} atoms` : id === "none" ? "remove" : group.name, large), off = () => say();
        b.addEventListener("pointerenter", on); b.addEventListener("pointerleave", off); b.addEventListener("focus", on); b.addEventListener("blur", off);
        b.addEventListener("click", () => place(id));
        wheel.append(b);
      });
      wheel.setAttribute("aria-label", `${rest} at position ${i + 1}`);
      // where the position is in the frame (the plot leaves room for the molecules to start from)
      const point = plot.createSVGPoint(), ctm = plot.getScreenCTM(), box = frame.getBoundingClientRect();
      [point.x, point.y] = build.at[i];
      const at = ctm ? point.matrixTransform(ctm) : { x: box.left + (build.at[i][0] / 100) * box.width, y: box.top + (build.at[i][1] / 100) * box.height };
      wheel.style.setProperty("--x", `${at.x - box.left}px`); wheel.style.setProperty("--y", `${at.y - box.top}px`);
      scrim.hidden = wheel.hidden = false;
      hint.hide();
      (byKey ? wheel.querySelector('[aria-pressed="true"]') || wheel.querySelector(".gsq-wheel-item") : wheel).focus({ preventScroll: true });
    }
    function hideWheel() { build.sel = null; scrim.hidden = wheel.hidden = true; }
    function closeWheel(refocus) {
      const i = build.sel;
      if (i == null) return;
      hideWheel(); draw(); renderPanel();
      if (refocus) slotLayer.children[i]?.focus({ preventScroll: true });
    }
    scrim.addEventListener("click", () => closeWheel(false));
    window.addEventListener("resize", () => { if (build.sel != null) closeWheel(false); }); // the wheel would be off its position
    wheel.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { e.preventDefault(); closeWheel(true); return; }
      const by = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
      if (!by) return;
      e.preventDefault();
      const items = [...wheel.querySelectorAll(".gsq-wheel-item")], k = items.indexOf(document.activeElement);
      items[(Math.max(k, by > 0 ? -1 : 0) + by + items.length) % items.length].focus({ preventScroll: true });
    });
    wheel.addEventListener("focusout", (e) => { if (e.relatedTarget && !wheel.contains(e.relatedTarget)) closeWheel(false); }); // the keyboard has left it

    // -- pointing at atoms, and the keyboard ------------------------------------------------------
    function toggle(k) { if (state.masked.has(k)) state.masked.delete(k); else state.masked.add(k); state.hinted = true; draw(); renderPanel(); }
    function pin(k) { state.hinted = true; if (k < 0 || (state.pinned && state.focus === k)) { state.pinned = false; state.focus = null; } else { state.pinned = true; state.focus = k; state.row = null; } draw(); renderPanel(); }
    plot.addEventListener("mousedown", (e) => e.preventDefault()); // a click points; it does not move the keyboard's focus
    plot.addEventListener("pointermove", (e) => {
      if (e.pointerType === "touch") return;
      const k = scene.atomAt(e);
      plot.style.cursor = k >= 0 && state.view !== "alone" && state.view !== "build" ? "pointer" : "";
      if (state.view === "together" && !state.pinned && state.focus !== (k < 0 ? null : k)) { state.focus = k < 0 ? null : k; state.row = null; if (k >= 0) state.hinted = true; draw(); renderPanel(); }
    });
    plot.addEventListener("pointerleave", () => { if (state.view === "together" && !state.pinned && state.focus != null) { state.focus = null; draw(); renderPanel(); } });
    plot.addEventListener("click", (e) => { const k = scene.atomAt(e); if (state.view === "predict") { if (k >= 0) toggle(k); } else if (state.view === "together") pin(k); });
    plot.addEventListener("keydown", (e) => {
      const k = Number(e.target.dataset?.k);
      if (Number.isNaN(k) || (e.key !== "Enter" && e.key !== " ")) return;
      e.preventDefault();
      if (state.view === "predict") toggle(k); else if (state.view === "together") pin(k);
    });
    plot.addEventListener("focusin", (e) => { const k = Number(e.target.dataset?.k); if (!Number.isNaN(k)) scene.keyFocus(k); });
    plot.addEventListener("focusout", (e) => { if (!plot.contains(e.relatedTarget)) scene.keyFocus(null); });

    // -- the molecule: the state as the scene's rings, bands and shapes ---------------------------
    function draw() {
      const m = state.mol, e = ex();
      const building = state.view === "build";
      starts.hidden = !building; frame.classList.toggle("gsq-building", building);
      progress.hidden = !built() || !build.busy;
      startButtons.forEach((b, k) => b.setAttribute("aria-pressed", String(STARTS[k].slots.join() === build.slots.join())));
      if (building) { slots(); if (build.picked) hint.hide(); else hint.show(`${G.hovers() ? "Click" : "Tap"} a + to add a group`); }
      else {
        slotLayer.remove();
        if (state.view === "alone" || state.hinted || state.focus != null) hint.hide();
        else hint.show(state.view === "predict" ? `${G.hovers() ? "Click" : "Tap"} an atom to hide it` : `${G.hovers() ? "Hover" : "Tap"} an atom`);
      }
      for (const hit of scene.hits()) {
        const k = Number(hit.dataset.k), name = G.atomName(m, k);
        if (state.view === "alone" || building) { hit.removeAttribute("tabindex"); hit.removeAttribute("role"); hit.removeAttribute("aria-label"); hit.removeAttribute("aria-pressed"); continue; }
        hit.setAttribute("tabindex", "0"); hit.setAttribute("role", "button");
        if (state.view === "predict") { hit.setAttribute("aria-label", `${name} ${k + 1}, ${state.masked.has(k) ? "hidden" : "shown"}`); hit.setAttribute("aria-pressed", String(state.masked.has(k))); }
        else { hit.setAttribute("aria-label", `Interactions of ${name.toLowerCase()} ${k + 1}`); hit.removeAttribute("aria-pressed"); }
      }
      const explained = state.view !== "predict";
      const own = state.view === "alone" ? singles(e, 1) : state.view === "together" ? singles(e, state.order) : building ? singles(e, 2) : {};
      let groups = [], groupLargest = 1;
      if (building) { // as the atoms together, in pairs: what the reader's groups do, while they place them
        const pairs = groupsOf(e, 2);
        groupLargest = Math.max(1e-9, Math.abs(pairs[0]?.v || 0));
        groups = pairs.slice(0, 10);
      } else if (state.view === "together") {
        const all = groupsOf(e, state.order);
        groupLargest = Math.max(1e-9, Math.abs(all[0]?.v || 0));
        const limit = state.order === "M" ? 4 : state.order === 2 ? 12 : 7;
        groups = state.row ? [state.row] : state.focus != null ? all.filter((g) => g.atoms.includes(state.focus)).slice(0, 8) : all.slice(0, limit);
      }
      scene.update({
        rings: own, ringLargest: Math.max(1e-9, ...Object.values(own).map(Math.abs)), groups, groupLargest,
        masked: explained ? new Set() : state.masked, focus: state.view === "together" ? state.focus : null,
        truth: e.truth, truthKey: `${m.id}/${state.task.id}/${built() ? build.slots.join() : ""}`, truthShown: true, dim: state.row ? new Set(state.row.atoms) : null,
      });
    }

    // -- the panel --------------------------------------------------------------------------------
    function meta(m, t, e) {
      return h("p", "gsq-meta", "Exact with ", h("strong", "", "GraphSHAP-IQ"), `, from ${e.budget.toLocaleString("en")} of ${(2 ** e.nAtoms).toLocaleString("en")} maskings`);
    }
    // the GNN's answer, on the question's card: for the whole molecule, or without the atoms hidden
    function answer(m, t, e) {
      const n = state.view === "predict" ? state.masked.size : 0, p = sig(n ? G.predict(data.symbols, m, t, state.masked) : e.logOdds);
      ask.set({ task: t, subject: m.name, has: e.truth.length > 0, p });
      const said = `${t.question} The GNN answers ${percent(p)}, ${p >= 0.5 ? t.yes : t.no}${n ? `, with ${n} ${n === 1 ? "atom" : "atoms"} hidden` : ""}.`;
      if (live.textContent !== said) live.textContent = said;
      return p;
    }
    function renderPanel() {
      const m = state.mol, t = state.task, e = ex(), content = h("div", "gsq-panel-in"), p = answer(m, t, e);
      if (state.view === "build") buildPanel(content, m, t, e); else if (state.view === "predict") maskingPanel(content, m, t, e, p); else explanationPanel(content, m, t, e);
      panel.replaceChildren(content);
    }
    const computing = () => (ex().failed ? h("p", "gsq-detail gsq-note", "Your browser could not compute the interactions here.") : h("p", "gsq-computing", h("span", "gsq-spin"), "Computing the exact interactions…"));
    function buildPanel(box, m, t, e) {
      box.append(legend(t, e.truth), e.pending || e.failed ? computing() : meta(m, t, e));
    }
    function maskingPanel(box, m, t, e, p) {
      const n = state.masked.size, compare = h("p", "gsq-pred-compare");
      if (n) {
        const d = Math.round(p * 100) - Math.round(e.p * 100);
        compare.append(h("strong", d >= 0 ? "gsq-up" : "gsq-down", `${Math.abs(d)} points ${d >= 0 ? "more" : "less"}`), ` than with all atoms (${percent(e.p)})`);
      } else compare.append(`${G.hovers() ? "Click" : "Tap"} atoms to hide them from the GNN.`);
      const actions = h("div", "gsq-actions");
      const act = (label, fn, hidden) => { const b = h("button", "gsq-act", label); b.type = "button"; b.hidden = !!hidden; b.addEventListener("click", () => { fn(); state.hinted = true; draw(); renderPanel(); }); actions.append(b); };
      if (e.truth.length) act(`Hide the ${t.truthLabel}`, () => { state.masked = new Set(e.truth); });
      act("Hide all", () => { state.masked = new Set(m.atoms.map((_, k) => k)); });
      act("Show all", () => { state.masked = new Set(); }, !n);
      box.append(compare, actions);
    }
    function explanationPanel(box, m, t, e) {
      const title = h("p", "gsq-title"), detail = h("p", "gsq-detail"), slot = h("div", "gsq-slot"), list = h("ol", "gsq-rows");
      if (e.pending || e.failed) { box.append(computing()); return; }
      if (state.view === "alone") {
        detail.append("What each atom adds on its own (its Shapley value).");
        const own = singles(e, 1), rows = Object.keys(own).map((k) => ({ atoms: [Number(k)], v: own[k] })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
        const largest = Math.max(1e-9, Math.abs(rows[0]?.v || 0));
        for (const g of rows.slice(0, 5)) list.append(G.row(m, g.atoms, g.v, largest, G.atomName(m, g.atoms[0])));
      } else {
        const groups = groupsOf(e, state.order), seq = orderSeq(), at = seq.indexOf(state.order), [label, index] = ORDERS[state.order];
        if (state.focus != null) {
          title.append(h("strong", "", G.atomName(m, state.focus)), " and its partners");
        } else {
          detail.append(state.order === "M" ? "What every group adds that no smaller group explains." : `A ring is what an atom adds alone. A ${state.order === 2 ? "band" : "band or shape"} is what atoms add only together.`);
        }
        const stepper = h("div", "gsq-stepper"), less = h("button", "gsq-step", "−"), more = h("button", "gsq-step", "+");
        less.type = more.type = "button";
        less.setAttribute("aria-label", "Smaller groups"); more.setAttribute("aria-label", "Larger groups");
        less.setAttribute("aria-disabled", String(at <= 0)); more.setAttribute("aria-disabled", String(at >= seq.length - 1));
        const go = (by) => { const to = seq[at + by]; if (to === undefined) return; state.order = to; state.row = null; update(); panel.querySelectorAll(".gsq-step")[by < 0 ? 0 : 1]?.focus({ preventScroll: true }); };
        less.addEventListener("click", () => go(-1)); more.addEventListener("click", () => go(1));
        stepper.append(less, h("span", "gsq-step-label", label, h("small", "", index)), more);
        slot.append(stepper);
        const rows = state.focus != null ? groups.filter((g) => g.atoms.includes(state.focus)) : groups;
        const largest = Math.max(1e-9, Math.abs(groups[0]?.v || 0));
        for (const g of rows.slice(0, 5)) {
          const li = G.row(m, g.atoms, g.v, largest, G.groupName(m, g.atoms));
          li.tabIndex = 0; li.setAttribute("aria-label", `${G.groupName(m, g.atoms)}: ${signed(g.v)}`);
          const on = () => { state.row = g; draw(); for (const o of list.children) o.classList.toggle("gsq-dim", o !== li); };
          const off = () => { state.row = null; draw(); for (const o of list.children) o.classList.remove("gsq-dim"); };
          li.addEventListener("pointerenter", (ev) => { if (ev.pointerType !== "touch") on(); });
          li.addEventListener("pointerleave", (ev) => { if (ev.pointerType !== "touch") off(); });
          li.addEventListener("focus", on); li.addEventListener("blur", off);
          list.append(li);
        }
        if (!rows.length) list.append(h("li", "gsq-none", "No interactions of this atom at this order."));
      }
      slot.append(list);
      box.append(...(title.childNodes.length ? [title] : [detail]), slot, legend(t, e.truth));
      box.append(meta(m, t, e));
    }

    function update() {
      const m = state.mol, t = state.task;
      mols.set(m.built ? data.molecules.length : data.molecules.indexOf(m));
      views.node.classList.toggle("gsq-no-build", !m.built); // only the reader's own molecule is built
      views.set(VIEWS.findIndex((v) => v[0] === state.view));
      draw();
      renderPanel();
    }
    scene.setMolecule(state.mol);
    update();
  });
})();
