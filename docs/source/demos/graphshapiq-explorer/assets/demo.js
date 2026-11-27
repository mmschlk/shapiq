// The GraphSHAP-IQ explorer (demos/graphshapiq-explorer/index.rst): seven molecules and one of the reader's own, three GNNs
// (one per question), and every molecule explained for every question, exactly, with GraphSHAP-IQ.
//
// Two views of the molecule. Input: beside it the question's card (the library's `question`), what
// it asks, the pattern that makes a yes, whether the molecule has it, and what the GNN answers.
// Explanation: the Shapley values (a halo around each atom), the k-SII of orders 2 to 4 and the
// Möbius interactions (hyperedges, as shapiq's si_graph_plot draws them). The
// explanations are precomputed in data.js (../../graphshapiq-common/compute); the shared library
// (../../graphshapiq-common/assets/demo.js, `ShapiqGraph`) draws and animates the molecule.
//
// The ninth molecule is the reader's own ("Build your own"): groups on the six positions of a
// benzene ring, in every view. A + on a free position opens a wheel of the groups that can go
// there, and a × on a group removes it. Nothing of it is precomputed. The page runs the GNN for its
// answer at once, and a Web Worker (`WORKER`) explains it with GraphSHAP-IQ: it
// evaluates the GNN on the maskings inside the receptive fields, and computes the Möbius
// interactions and from them the k-SII of orders 1 to 4, as shapiq does
// (../../graphshapiq-common/compute/check_worker.py compares them with shapiq's). A molecule has at
// most MAX_ATOMS heavy atoms, so that this takes a few seconds at most.
(() => {
  const G = window.ShapiqGraph;
  if (!G) return console.error("The GraphSHAP-IQ explorer needs ../graphshapiq-common/demo.js first.");
  const { h, svg, sig, bits, signed, percent, legend } = G;
  const VIEWS = [["predict", "Input"], ["explain", "Explanation"]];
  // the explanations: the Shapley values, the k-SII of orders 2 to 4, and the Möbius interactions
  const KINDS = [[1, "Shapley"], [2, "2-SII"], [3, "3-SII"], [4, "4-SII"], ["M", "Möbius"]];
  // the molecules in the row, grouped by their question (in the questions' order), those with the
  // question's pattern first
  const TASK_ORDER = ["mutagenic", "indole", "PAINS"];

  const ROWS = 7; // the most rows a panel of explanations lists
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
  // the only PAINS besides the catechol that one can build (RDKit's anil_no_alk, checked against all
  // 154,245 molecules of the builder): an NH2 across the ring from one other of these groups
  function anilNoAlk(slots) {
    const on = slots.flatMap((slot, i) => (slot === "none" ? [] : [i])), partner = ["OH", "NH2", "NO2", "CH3", "OMe"];
    if (on.length !== 2 || on[1] - on[0] !== 3) return false;
    const [a, b] = on.map((i) => slots[i]);
    return (a === "NH2" && partner.includes(b)) || (b === "NH2" && partner.includes(a));
  }
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
    data.tasks.sort((a, b) => TASK_ORDER.indexOf(a.id) - TASK_ORDER.indexOf(b.id));
    const taskById = (id) => data.tasks.find((t) => t.id === id);
    const hasPattern = (m) => (data.explanations[`${m.id}/${m.task}`].truth.length ? 0 : 1);
    const MOLS = data.molecules.slice().sort((a, b) => TASK_ORDER.indexOf(a.task) - TASK_ORDER.indexOf(b.task) || hasPattern(a) - hasPattern(b));
    const first = MOLS[0];
    const state = { mol: first, task: taskById(first.task), view: "predict", order: 2, focus: null, pinned: false, row: null, hinted: false };
    // the reader's own molecule: the groups on the ring's positions, the position whose wheel is
    // open (`sel`) and where the positions are on the stage (`at`), the molecule they make, and its explanation (`result`: at first the GNN's answer only, `pending`)
    const build = { slots: STARTS[1].slots.slice(), task: taskById(STARTS[1].task), sel: null, at: [], note: "", mol: null, result: null, run: 0, timer: 0, t0: 0, worker: null, busy: false, picked: false };
    const built = () => state.mol.built === true;
    const ex = () => (built() ? build.result : data.explanations[`${state.mol.id}/${state.task.id}`]);
    const rowsOf = (e, order) => (order === "M" ? e.moebius : e.kSII[String(order)] || []);
    const singles = (e, order) => { const o = {}; for (const [mask, v] of rowsOf(e, order)) { const a = bits(mask); if (a.length === 1) o[a[0]] = v; } return o; };
    const groupsOf = (e, order) => rowsOf(e, order).map(([mask, v]) => ({ atoms: bits(mask), v })).filter((g) => g.atoms.length > 1).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
    const orderSeq = () => KINDS.map((k) => k[0]).filter((k) => k === "M" || built() || ex().kSII[String(k)]);
    const pointing = () => state.view === "explain" && state.order !== 1; // atoms to point at: their interactions
    // the first n of a list sorted by size, and any more of the same size (so that a tie is not split)
    const upTo = (list, n) => { let k = Math.min(n, list.length); while (k > 0 && k < list.length && Math.abs(Math.abs(list[k].v) - Math.abs(list[k - 1].v)) < 1e-4) k++; return list.slice(0, k); };

    // -- the frame --------------------------------------------------------------------------------
    const bar = h("div", "gsq-bar");
    // the tile of the reader's own molecule: a ring with a + where a group can go
    const own = { ...assemble(["CH3", ...none]), name: "Build your own" };
    const mols = G.picker("Molecule", [...MOLS, own], (k) => {
      if (k === MOLS.length) { // the reader's own molecule, as they left it
        state.task = build.task;
        if (build.mol) { state.mol = build.mol; reset(true); } else rebuild();
        return;
      }
      state.mol = MOLS[k]; state.task = taskById(state.mol.task); // the explanation picked stays
      reset(true);
    });
    {
      MOLS.forEach((m, k) => { if (k && m.task !== MOLS[k - 1].task) mols.buttons[k].classList.add("gsq-mol-next"); });
      const tile = mols.buttons[MOLS.length], icon = tile.querySelector("svg"), [x, y] = G.geom(own, 78, 21).pos[6];
      tile.classList.add("gsq-mol-own"); mols.dots[MOLS.length].classList.add("gsq-dot-own");
      svg("circle", { class: "gsq-plus-ping", cx: x, cy: y, r: 14 }, icon);
      svg("circle", { class: "gsq-plus-o", cx: x, cy: y, r: 14 }, icon);
      svg("line", { x1: x - 6.5, y1: y, x2: x + 6.5, y2: y }, icon); svg("line", { x1: x, y1: y - 6.5, x2: x, y2: y + 6.5 }, icon);
    }
    const views = G.switcher("View", VIEWS.map((v) => v[1]), (k) => setView(VIEWS[k][0]));
    const kinds = G.switcher("Explanation", KINDS.map((k) => k[1]), (k) => {
      state.order = KINDS[k][0]; state.row = null; if (state.order === 1) { state.focus = null; state.pinned = false; }
      update(); kinds.buttons[k].focus({ preventScroll: true });
    });
    kinds.node.classList.add("gsq-kinds");
    const ask = G.question(data.tasks, (t) => { if (t.id === state.task.id) return; state.task = t; if (built()) compute(); reset(false); });
    const info = h("button", "gsq-info", "i");
    info.type = "button"; info.setAttribute("aria-label", "What does the demo show?"); info.setAttribute("aria-expanded", "false");
    // the note opens under the bar, in the flow, as on the FIxLIP page: inert while closed
    const about = h("div", "gsq-about"); about.id = `gsq-about-${Math.random().toString(36).slice(2, 8)}`;
    about.inert = true; about.setAttribute("aria-hidden", "true");
    info.setAttribute("aria-controls", about.id);
    about.innerHTML = '<div class="gsq-about-clip"><div class="gsq-about-card">' +
      '<button class="gsq-about-close" type="button" aria-label="Close">×</button>' +
      "<p><strong>What does the demo show?</strong></p>" +
      "<p>A <strong>graph neural network</strong> (GNN) reads a molecule as a graph: atoms, linked by bonds. In each of its three layers, every atom gathers what its neighbours know. At the end, the GNN sums its atoms and answers one yes/no question, each question with a GNN of its own.</p>" +
      "<p>To explain an answer, <code>shapiq</code> hides atoms in many combinations and watches the answer change. The atoms are the <strong>players</strong>: what each adds alone, and what groups add only together, their <strong>interactions</strong>.</p>" +
      "<p>For a GNN these are computed <strong>exactly</strong>. With three layers, atoms more than six bonds apart never meet in the GNN. So for a larger molecule a fraction of all combinations suffices.</p></div></div>";
    const viewbar = h("div", "gsq-viewbar", views.node);
    ask.node.querySelector(".gsq-ask-pick").append(info);
    ask.card.remove(); // the question's card is the panel of the view Input
    bar.append(mols.node, ask.node);
    // it closes with the "i", its ×, or Escape while the focus is in it or on the "i"
    const isOpen = () => about.classList.contains("gsq-about-open");
    const setAbout = (open) => {
      about.classList.toggle("gsq-about-open", open); about.inert = !open; about.setAttribute("aria-hidden", String(!open));
      info.setAttribute("aria-expanded", String(open));
    };
    info.addEventListener("click", () => setAbout(!isOpen()));
    about.querySelector(".gsq-about-close").addEventListener("click", () => { setAbout(false); info.focus({ preventScroll: true }); });
    root.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && isOpen() && (about.contains(e.target) || e.target === info)) { setAbout(false); info.focus({ preventScroll: true }); }
    });

    const body = h("div", "gsq-body"), stage = h("div", "gsq-stage"), frame = h("div", "gsq-frame"), panel = h("div", "gsq-panel");
    const plot = svg("svg", { class: "gsq-plot", viewBox: "0 0 100 100", role: "group", tabindex: "-1", "aria-label": "The molecule; its atoms are the players" }, frame);
    const hint = G.hinter(frame);
    const slotLayer = svg("g", { class: "gsq-slots" });
    const progress = h("div", "gsq-progress"), progressFill = h("i");
    progress.append(progressFill); progress.hidden = true; frame.prepend(viewbar); frame.append(progress);
    // the molecules to start from, as drawings beside the views (for the own molecule only)
    const starts = h("div", "gsq-starts");
    starts.setAttribute("role", "group"); starts.setAttribute("aria-label", "Start from"); starts.hidden = true;
    const startButtons = STARTS.map((s, k) => {
      const b = h("button", "gsq-start", G.tile(assemble(s.slots))); b.type = "button";
      b.title = `Start from ${s.name.toLowerCase().replace("empty", "an empty")}`; b.setAttribute("aria-label", b.title);
      b.addEventListener("click", () => start(k));
      starts.append(b);
      return b;
    });
    viewbar.append(starts);
    // the key of the ground truth, in the frame's bottom left: a shaded patch, as the outline is
    const truthKey = h("p", "gsq-truthkey", h("i"), h("span"));
    truthKey.hidden = true; frame.append(truthKey);
    // the GNN's answer, in the frame's bottom right: in the view Explanation, and in Input too when the
    // card is under the molecule (a narrow demo; the CSS decides)
    const predKey = h("p", "gsq-predkey", h("span", "gsq-predkey-lab", "GNN"), h("strong"), h("span"), h("span", "gsq-predkey-mark"));
    predKey.setAttribute("aria-hidden", "true"); frame.append(predKey);
    const scrim = h("div", "gsq-scrim"), wheel = h("div", "gsq-wheel");
    scrim.hidden = wheel.hidden = true; wheel.tabIndex = -1; wheel.setAttribute("role", "group");
    frame.append(scrim, wheel);
    stage.append(frame);
    const live = h("p", "gsq-sr"); live.setAttribute("aria-live", "polite");
    body.append(stage, panel);
    root.append(bar, about, body, mols.dotsNode, live);
    mols.swipe(root, body, ".gsq-wheel, .gsq-scrim, .gsq-about");
    const scene = G.createScene(plot);

    function setView(id) { hideWheel(); state.view = id; state.focus = null; state.pinned = false; state.row = null; update(); views.buttons[VIEWS.findIndex((v) => v[0] === id)].focus({ preventScroll: true }); }
    function reset(newMolecule) {
      hideWheel();
      state.focus = null; state.pinned = false; state.row = null;
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
      build.slots = trial; build.sel = null;
      rebuild();
      slotLayer.children[i]?.focus({ preventScroll: true });
    }
    function start(k) {
      const s = STARTS[k];
      if (built() && s.slots.join() === build.slots.join() && (!s.task || s.task === state.task.id)) return; // the molecule shown already
      build.slots = s.slots.slice(); build.picked = true; if (s.task) state.task = taskById(s.task); rebuild(); }
    // the GNN's answer is one evaluation, at once; the worker explains the latest molecule only. A
    // job still running is stopped (the worker is replaced), so that jobs do not queue up
    function stopWorker() { if (build.worker) build.worker.terminate(); build.worker = null; build.busy = false; }
    function failed() {
      stopWorker(); progress.hidden = true;
      build.result = { ...build.result, pending: false, failed: true };
      if (built()) { draw(); renderPanel(); }
    }
    let workerUrl = null; // one script for every worker
    function startWorker() {
      try { workerUrl = workerUrl || URL.createObjectURL(new Blob([WORKER], { type: "text/javascript" })); build.worker = new Worker(workerUrl); } catch (err) { console.error(err); return false; }
      const worker = build.worker; // a replaced worker's messages and errors are not heard
      worker.onerror = (err) => { if (worker !== build.worker) return; console.error(err.message || err); failed(); };
      worker.onmessage = ({ data: d }) => {
        if (worker !== build.worker || d.id !== build.run) return; // an earlier molecule's
        if (d.progress != null) { progressFill.style.width = `${Math.round(d.progress * 100)}%`; return; }
        build.busy = false; progress.hidden = true;
        build.result = { ...build.result, ...d, p: sig(d.logOdds), pending: false, failed: false };
        if (built()) { draw(); renderPanel(); if (state.view === "explain") live.textContent += " Its interactions are ready."; }
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
        build.busy = true; build.t0 = performance.now();
        setTimeout(() => { if (build.busy && build.run === id && built()) progress.hidden = false; }, 350);
        build.worker.postMessage({ id, symbols: data.symbols, m: { atoms: m.atoms, bonds: m.bonds }, t: { gnn: t.gnn, baseline: t.baseline }, overall: OVERALL, perAtom: PER_ATOM });
      }, 120);
    }
    // the positions of the ring, over the molecule, in every view: a + where a group can go (it opens
    // the wheel), and a × on the bond of a group that is there (it removes the group). Both stay off
    // the atoms, so that the atoms can still be hidden and pointed at
    let slotKey = "";
    function slots() {
      const key = `${build.slots.join()}|${build.sel}`;
      if (key === slotKey && slotLayer.parentNode === plot && plot.lastChild === slotLayer) return; // nothing changed
      slotKey = key;
      const focused = slotLayer.contains(document.activeElement) ? Number(document.activeElement.dataset.slot) : null;
      slotLayer.replaceChildren();
      const { s, pos } = scene.geom;
      let next = 6; // the atoms of a position's group follow the ring's, as `assemble` puts them
      for (let i = 0; i < 6; i++) {
        const a = Math.PI / 2 - (i * Math.PI) / 3, [x, y] = pos[i], group = GROUPS[build.slots[i]], empty = !group.heavy;
        const g = svg("g", { class: `gsq-slot${empty ? "" : " gsq-filled"}${build.sel === i ? " gsq-sel" : ""}`, tabindex: "0", role: "button", "data-slot": i, "aria-label": empty ? `Position ${i + 1}: add a group` : `Remove the ${group.name} group at position ${i + 1}` }, slotLayer);
        if (empty) {
          const cx = x + Math.cos(a) * 0.62 * s, cy = y + Math.sin(a) * 0.62 * s;
          build.at[i] = [cx, cy];
          g.setAttribute("aria-expanded", String(build.sel === i));
          svg("circle", { class: "gsq-slot-hit", cx, cy, r: 0.34 * s }, g); // room for a finger
          svg("circle", { cx, cy, r: 0.26 * s }, g);
          svg("line", { class: "gsq-plus", x1: cx - 0.12 * s, y1: cy, x2: cx + 0.12 * s, y2: cy }, g);
          svg("line", { class: "gsq-plus", x1: cx, y1: cy - 0.12 * s, x2: cx, y2: cy + 0.12 * s }, g);
        } else {
          const cx = (x + pos[next][0]) / 2, cy = (y + pos[next][1]) / 2, d = 0.07 * s; // the middle of the bond to the group
          svg("circle", { class: "gsq-slot-hit", cx, cy, r: 0.3 * s }, g);
          svg("circle", { cx, cy, r: 0.17 * s }, g);
          svg("line", { class: "gsq-plus", x1: cx - d, y1: cy - d, x2: cx + d, y2: cy + d }, g);
          svg("line", { class: "gsq-plus", x1: cx - d, y1: cy + d, x2: cx + d, y2: cy - d }, g);
          next += group.heavy;
        }
        const pick = (byKey) => {
          build.note = ""; build.picked = true; state.hinted = true;
          if (!empty) { build.sel = i; place("none"); return; }
          if (build.sel === i) closeWheel(byKey); else openWheel(i, byKey);
        };
        g.addEventListener("click", (e) => { e.stopPropagation(); pick(false); });
        g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.stopPropagation(); pick(true); } });
      }
      if (slotLayer.parentNode !== plot || plot.lastChild !== slotLayer) plot.appendChild(slotLayer);
      if (focused != null && !Number.isNaN(focused)) slotLayer.children[focused]?.focus({ preventScroll: true });
    }
    // the wheel: the groups that can go to a position, around the position. The position shows
    // through its hub, which names the group pointed at and closes it
    function openWheel(i, byKey) {
      build.sel = i;
      draw(); renderPanel();
      const there = build.slots[i], ids = PALETTE, rest = "Add a group";
      const name = h("span", "gsq-wheel-name"), hub = h("button", "gsq-wheel-hub", name); hub.type = "button"; hub.setAttribute("aria-label", "Close");
      hub.addEventListener("click", () => closeWheel(true));
      const say = (text = "", large = false) => { name.textContent = text; name.classList.toggle("gsq-off", large); }; // the group pointed at
      wheel.replaceChildren(hub);
      ids.forEach((id, k) => {
        const group = GROUPS[id], trial = build.slots.slice(); trial[i] = id;
        const large = heavy(trial) > MAX_ATOMS, angle = -Math.PI / 2 + (k * 2 * Math.PI) / ids.length;
        const b = h("button", "gsq-wheel-item", group.label); b.type = "button";
        b.setAttribute("aria-label", `${group.label} (${group.name})${large ? ", too large here" : ""}`);
        if (large) b.setAttribute("aria-disabled", "true"); else if (id === there) b.setAttribute("aria-pressed", "true");
        b.style.setProperty("--dx", Math.cos(angle).toFixed(4)); b.style.setProperty("--dy", Math.sin(angle).toFixed(4)); b.style.setProperty("--i", k);
        const on = () => say(large ? `At most ${MAX_ATOMS} atoms` : group.name, large), off = () => say();
        b.addEventListener("pointerenter", on); b.addEventListener("pointerleave", off); b.addEventListener("focus", on); b.addEventListener("blur", off);
        b.addEventListener("click", () => place(id));
        wheel.append(b);
      });
      wheel.setAttribute("aria-label", `${rest} at position ${i + 1}`);
      // where the position is in the frame (the plot sits under the menu)
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
    let frameWidth = 0; // the wheel would be off its position when the frame changes width (not when a phone's address bar hides)
    new ResizeObserver(() => { const w = Math.round(frame.clientWidth); if (w !== frameWidth) { frameWidth = w; if (build.sel != null) closeWheel(false); } }).observe(frame);
    wheel.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { e.preventDefault(); closeWheel(true); return; }
      const by = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
      if (!by) return;
      e.preventDefault();
      const items = [...wheel.querySelectorAll(".gsq-wheel-item")], k = items.indexOf(document.activeElement);
      items[(Math.max(k, by > 0 ? -1 : 0) + by + items.length) % items.length].focus({ preventScroll: true });
    });
    // the keyboard has left it: redraw once the focus has landed, so that it is not on a node replaced
    wheel.addEventListener("focusout", (e) => { if (e.relatedTarget && !wheel.contains(e.relatedTarget) && build.sel != null) { hideWheel(); requestAnimationFrame(() => { draw(); renderPanel(); }); } });

    // -- pointing at atoms, and the keyboard ------------------------------------------------------
    function pin(k) { state.hinted = true; if (k < 0 || (state.pinned && state.focus === k)) { state.pinned = false; state.focus = null; } else { state.pinned = true; state.focus = k; state.row = null; } draw(); renderPanel(); }
    plot.addEventListener("mousedown", (e) => e.preventDefault()); // a click points; it does not move the keyboard's focus
    plot.addEventListener("pointermove", (e) => {
      if (e.pointerType === "touch") return;
      const k = scene.atomAt(e);
      plot.style.cursor = k >= 0 && pointing() ? "pointer" : "";
      if (pointing() && !state.pinned && state.focus !== (k < 0 ? null : k)) { state.focus = k < 0 ? null : k; state.row = null; if (k >= 0) state.hinted = true; draw(); renderPanel(); }
    });
    plot.addEventListener("pointerleave", () => { if (pointing() && !state.pinned && state.focus != null) { state.focus = null; draw(); renderPanel(); } });
    plot.addEventListener("click", (e) => { const k = scene.atomAt(e); if (pointing()) pin(k); });
    plot.addEventListener("keydown", (e) => {
      const k = Number(e.target.dataset?.k);
      if (Number.isNaN(k) || (e.key !== "Enter" && e.key !== " ")) return;
      e.preventDefault();
      if (pointing()) pin(k);
    });
    plot.addEventListener("focusin", (e) => { const k = Number(e.target.dataset?.k); if (!Number.isNaN(k)) scene.keyFocus(k); });
    const rove = G.roving(plot, () => (pointing() ? scene.hits() : []));
    plot.addEventListener("focusout", (e) => { if (!plot.contains(e.relatedTarget)) scene.keyFocus(null); });

    // -- the molecule: the state as the scene's rings, bands and shapes ---------------------------
    function draw() {
      const m = state.mol, e = ex();
      const building = built(); // the reader's own molecule: its positions and starts, in every view
      starts.hidden = !building; frame.classList.toggle("gsq-building", building);
      progress.hidden = !(built() && build.busy && performance.now() - build.t0 > 350); // also back on the molecule while its job runs
      startButtons.forEach((b, k) => b.setAttribute("aria-pressed", String(STARTS[k].slots.join() === build.slots.join())));
      if (building) slots(); else slotLayer.remove();
      if (building && !build.picked) hint.show(`${G.hovers() ? "Click" : "Tap"} a + to add a group`);
      else if (!pointing() || state.hinted || state.focus != null) hint.hide();
      else hint.show(`${G.hovers() ? "Hover" : "Tap"} an atom`);
      for (const hit of scene.hits()) {
        const k = Number(hit.dataset.k), name = G.atomName(m, k);
        if (!pointing()) { hit.removeAttribute("tabindex"); hit.removeAttribute("role"); hit.removeAttribute("aria-label"); hit.removeAttribute("aria-pressed"); continue; }
        hit.setAttribute("role", "button"); hit.setAttribute("aria-pressed", String(state.pinned && state.focus === k));
        hit.setAttribute("aria-label", `Interactions of ${name.toLowerCase()} ${k + 1}`);
      }
      if (pointing()) rove.mark();
      const explained = state.view !== "predict";
      // the SI graph: a halo per atom, and the strongest interactions as hyperedges, all on one
      // scale (as si_graph_plot scales them by the largest value shown)
      const ready = explained && !e.pending && !e.failed, own = ready ? singles(e, state.order) : {};
      let groups = [];
      if (ready && state.order !== 1) {
        // every group the list shows (its first ROWS), and more strong ones up to a limit
        const all = groupsOf(e, state.order), limit = state.order === 2 ? 10 : ROWS, top = Math.abs(all[0]?.v || 0);
        groups = state.row ? [state.row] : state.focus != null ? all.filter((g) => g.atoms.includes(state.focus)).slice(0, ROWS) : upTo(all.filter((g, i) => i < ROWS || Math.abs(g.v) >= 0.08 * top), limit);
      }
      // one scale for the order, whatever is shown (as si_graph_plot scales by every value plotted)
      const largest = ready ? Math.max(1e-9, ...rowsOf(e, state.order).map(([, v]) => Math.abs(v))) : 1;
      const truthLabel = e.truth.length ? `the ${state.task.truthLabel}` : "";
      truthKey.hidden = !truthLabel; truthKey.lastChild.textContent = truthLabel; frame.classList.toggle("gsq-has-key", !!truthLabel);
      scene.update({
        rings: own, ringLargest: largest, groups, groupLargest: largest,
        masked: new Set(), focus: pointing() ? state.focus : null,
        truth: e.truth, truthKey: `${m.id}/${state.task.id}/${built() ? build.slots.join() : ""}`, truthShown: true, dim: state.row ? new Set(state.row.atoms) : null,
      });
    }

    // -- the panel --------------------------------------------------------------------------------
    // the molecule's true answer, where it is known: from the data, and for the reader's own molecule
    // the indole's pattern (PAINS needs all of RDKit's PAINS filters, mutagenic an Ames test)
    function truthOf(m, t, e) {
      if (!m.built) return m.labels?.[t.id] ?? null;
      if (t.id === "PAINS") return e.truth.length > 0 || anilNoAlk(build.slots);
      return t.id === "indole" ? e.truth.length > 0 : null;
    }
    // the GNN's answer, on the question's card, and whether it is right
    function answer(m, t, e) {
      const p = sig(e.logOdds), truth = truthOf(m, t, e);
      ask.set({ task: t, subject: m.name, has: e.truth.length > 0, p, truth });
      frame.dataset.view = state.view;
      const [, num, word, mark] = predKey.children;
      num.textContent = percent(p); word.textContent = p >= 0.5 ? t.yes : t.no;
      mark.textContent = truth === null ? "" : (p >= 0.5) === truth ? "✓" : "✗";
      mark.title = truth === null ? "" : (p >= 0.5) === truth ? "Correct" : "Wrong";
      const said = `${m.name}. ${t.question} The GNN answers ${percent(p)}, ${p >= 0.5 ? t.yes : t.no}.`;
      if (live.textContent !== said) live.textContent = said;
      return p;
    }
    function renderPanel() {
      const m = state.mol, t = state.task, e = ex(), content = h("div", "gsq-panel-in");
      answer(m, t, e);
      if (state.view === "predict") {
        if (panel.contains(ask.card)) content.style.animation = "none"; // no fade on a redraw of the same card
        content.append(ask.card);
      } else explanationPanel(content, m, t, e);
      // the explanations' switch stays in the panel, so that its pill slides from one to the next
      if (kinds.node.parentNode !== panel) panel.prepend(kinds.node);
      kinds.node.hidden = state.view !== "explain";
      const at = document.activeElement, had = panel.contains(at) && !kinds.node.contains(at) ? at.dataset.key ?? "" : undefined; // a row with the keyboard's focus keeps it
      for (const c of [...panel.children]) if (c !== kinds.node) c.remove();
      panel.append(content);
      if (had !== undefined) { const row = content.querySelector(`[data-key="${had}"]`); if (row) row.focus({ preventScroll: true }); else { content.tabIndex = -1; content.focus({ preventScroll: true }); } } // or the panel, if the row is gone
      if (state.view === "explain") kinds.set(KINDS.findIndex((k) => k[0] === state.order));
    }
    const computing = () => (ex().failed ? h("p", "gsq-detail gsq-note", "Your browser could not compute the interactions here.") : h("p", "gsq-computing", h("span", "gsq-spin"), "Computing the exact interactions…"));
    function explanationPanel(box, m, t, e) {
      const title = h("p", "gsq-title"), list = h("ol", "gsq-rows");
      if (e.pending || e.failed) { box.append(computing()); return; }
      if (state.order === 1) {
        const own = singles(e, 1), rows = Object.keys(own).map((k) => ({ atoms: [Number(k)], v: own[k] })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
        const largest = Math.max(1e-9, Math.abs(rows[0]?.v || 0));
        for (const g of rows.slice(0, ROWS)) list.append(G.row(m, g.atoms, g.v, largest, G.atomName(m, g.atoms[0])));
      } else {
        const groups = groupsOf(e, state.order);
        if (state.focus != null) title.append(h("strong", "", G.atomName(m, state.focus)), " and its partners");
        const rows = state.focus != null ? groups.filter((g) => g.atoms.includes(state.focus)) : groups;
        const largest = Math.max(1e-9, Math.abs(groups[0]?.v || 0));
        for (const g of rows.slice(0, ROWS)) {
          const li = G.row(m, g.atoms, g.v, largest, G.groupName(m, g.atoms)), who = g.atoms.map((k) => `${G.atomName(m, k).toLowerCase()} ${k + 1}`).join(", ");
          li.tabIndex = 0; li.dataset.key = g.atoms.join(); li.setAttribute("aria-label", `${who[0].toUpperCase()}${who.slice(1)}: ${signed(g.v)}`);
          const on = () => { state.row = g; draw(); for (const o of list.children) o.classList.toggle("gsq-dim", o !== li); };
          const off = () => { state.row = null; draw(); for (const o of list.children) o.classList.remove("gsq-dim"); };
          li.addEventListener("pointerenter", (ev) => { if (ev.pointerType !== "touch") on(); });
          li.addEventListener("pointerleave", (ev) => { if (ev.pointerType !== "touch") off(); });
          // only the keyboard's focus picks a row (a click focuses it too, and the focus is given back
          // after a redraw: then a row stuck on while the pointer was elsewhere)
          li.addEventListener("focus", () => { if (li.matches(":focus-visible")) on(); }); li.addEventListener("blur", off);
          list.append(li);
        }
        if (!rows.length) list.append(h("li", "gsq-none", "No interactions of this atom here."));
      }
      box.append(...(title.childNodes.length ? [title] : []), list, legend(t, null)); // the ground truth's key is in the frame
    }

    function update() {
      const m = state.mol, t = state.task;
      mols.set(m.built ? MOLS.length : MOLS.indexOf(m));
      views.set(VIEWS.findIndex((v) => v[0] === state.view));
      draw();
      renderPanel();
      tell();
    }

    // -- the write-up's figures: the molecule and the question shown, both ways -------------------
    // As the FIxLIP page's demo and figures do (fixlip-demo-image): the explorer tells the page
    // which molecule and question it shows, by the id in data.js ("build" for the reader's own) and
    // the task's id, in the event graphshapiq-demo-molecule with source null, and keeps it in
    // ShapiqGraph.current for figures that start later. A figure sends the event with itself as
    // the source, and the explorer then shows that molecule (and question) without telling again.
    const EVENT = "graphshapiq-demo-molecule";
    let told = "", hearing = false;
    function tell() {
      const molecule = built() ? "build" : state.mol.id, task = state.task.id, key = `${molecule}/${task}`;
      G.current = { molecule, task };
      if (key === told || hearing) { told = key; return; }
      told = key;
      document.dispatchEvent(new CustomEvent(EVENT, { detail: { molecule, task, source: null } }));
    }
    document.addEventListener(EVENT, (event) => {
      const { molecule, task, source } = event.detail ?? {};
      if (source == null) return; // the explorer's own, or another explorer's
      const m = MOLS.find((x) => x.id === molecule), t = task != null ? taskById(task) : null;
      if (!m || (task != null && !t)) return;
      hearing = true;
      try {
        if (m !== state.mol) { state.mol = m; state.task = t || taskById(m.task); reset(true); } else if (t && t !== state.task) { state.task = t; reset(false); }
      } finally { hearing = false; }
    });

    scene.setMolecule(state.mol);
    update();
  });
})();
