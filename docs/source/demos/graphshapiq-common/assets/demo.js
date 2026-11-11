// What the four GraphSHAP-IQ demos share (see ../README.md), as `window.ShapiqGraph`: small DOM
// helpers, a molecule's drawing, the animated SI-graph scene, the GNN that runs in the page, and
// `mount`, which fills every container of a demo with its data. The page loads this script before
// the demos' own (assets/requires.txt of each demo, see _ext/shapiq_demos.py).
(() => {
  const SVG = "http://www.w3.org/2000/svg";
  const RED = "#ff0d57"; // shapiq's interaction colors, as in the FIxLIP and orders demos
  const BLUE = "#1e88e5";
  const VIOLET = "#6a57cd";
  const NAMES = { C: "Carbon", N: "Nitrogen", O: "Oxygen", P: "Phosphorus", S: "Sulfur", F: "Fluorine", Cl: "Chlorine", Br: "Bromine", I: "Iodine", B: "Boron" };
  const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const hovers = () => window.matchMedia("(hover: hover) and (pointer: fine)").matches; // a mouse, not a finger

  // -- small helpers ------------------------------------------------------------------------------
  function h(tag, className, ...children) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    for (const child of children) if (child != null) node.append(child);
    return node;
  }
  function svg(tag, attributes, parent) {
    const node = document.createElementNS(SVG, tag);
    for (const key in attributes) node.setAttribute(key, attributes[key]);
    if (parent) parent.appendChild(node);
    return node;
  }
  const sig = (x) => 1 / (1 + Math.exp(-x));
  const bits = (mask) => { const atoms = []; let i = 0; while (mask) { if (mask & 1) atoms.push(i); mask >>>= 1; i += 1; } return atoms; };
  const signed = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;
  const percent = (p) => (p > 0.995 ? ">99%" : p < 0.005 ? "<1%" : `${Math.round(p * 100)}%`);
  const colorOf = (v) => (v >= 0 ? RED : BLUE);
  const value = (v) => h("strong", v >= 0 ? "gsq-up" : "gsq-down", signed(v));
  const sameSet = (a, b) => a.length === b.length && a.every((x) => b.includes(x));
  const neighbors = (m) => m._nb || (m._nb = (() => { const nb = m.atoms.map(() => []); for (const [i, j] of m.bonds) { nb[i].push(j); nb[j].push(i); } return nb; })());

  // -- a molecule's drawing: atoms placed in a square of 100 x 100 --------------------------------
  // `room` is the share of the square the molecule may fill, `cap` the longest bond
  function geom(m, room = 72, cap = 14) {
    const key = `_g${room}_${cap}`;
    if (m[key]) return m[key];
    const xs = m.atoms.map((a) => a[1]), ys = m.atoms.map((a) => a[2]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const s = Math.min(room / Math.max(x1 - x0, 1e-6), room / Math.max(y1 - y0, 1e-6), cap);
    return (m[key] = { s, pos: m.atoms.map((a) => [50 + (a[1] - (x0 + x1) / 2) * s, 50 + (a[2] - (y0 + y1) / 2) * s]) });
  }
  // the bonds as lines (double and triple bonds as parallel lines), into an <svg> or <g>
  function skeleton(parent, m, gap = 0.16, cls = "", g = geom(m)) {
    const { s, pos } = g;
    for (const [i, j, order] of m.bonds) {
      const [x1, y1] = pos[i], [x2, y2] = pos[j], L = Math.hypot(x2 - x1, y2 - y1) || 1, px = -(y2 - y1) / L, py = (x2 - x1) / L;
      for (const o of order >= 3 ? [-gap, 0, gap] : order === 2 ? [-gap / 2, gap / 2] : [0]) {
        const attributes = { x1: x1 + px * o * s, y1: y1 + py * o * s, x2: x2 + px * o * s, y2: y2 + py * o * s };
        if (cls) attributes.class = cls;
        svg("line", attributes, parent);
      }
    }
  }
  // a molecule as a tile of a picker: its bonds, and a dot for every atom that is not carbon
  function tile(m) {
    const node = svg("svg", { viewBox: "0 0 100 100", "aria-hidden": "true" });
    const g = geom(m, 78, 21); // a small molecule is drawn larger than on the stage: a tile is small
    skeleton(node, m, 0.16, "", g);
    g.pos.forEach(([x, y], k) => { if (m.atoms[k][0] !== "C") svg("circle", { cx: x, cy: y, r: 3.2 }, node); });
    return node;
  }
  // a row of a list: the molecule with a group's atoms marked, its name, a meter, and its value
  function row(m, atoms, v, largest, name) {
    const li = h("li", "gsq-row");
    const thumb = svg("svg", { viewBox: "0 0 100 100", "aria-hidden": "true" }, li);
    skeleton(thumb, m, 0.16);
    for (const k of atoms) { const [x, y] = geom(m).pos[k]; svg("circle", { cx: x, cy: y, r: atoms.length > 4 ? 6 : 7.5, fill: colorOf(v) }, thumb); }
    const meter = h("span", "gsq-meter"), fill = h("i");
    fill.style.width = `${Math.max(4, (Math.abs(v) / largest) * 100)}%`;
    fill.style.background = colorOf(v);
    meter.append(fill);
    li.append(h("span", "gsq-who", name), meter, h("span", `gsq-num ${v >= 0 ? "gsq-up" : "gsq-down"}`, signed(v)));
    return li;
  }
  const groupName = (m, atoms) => { const els = atoms.map((k) => m.atoms[k][0]); return els.length > 4 ? `${els.slice(0, 3).join(" · ")} +${els.length - 3}` : els.join(" · "); };
  const atomName = (m, k) => NAMES[m.atoms[k][0]] || m.atoms[k][0];
  // the legend: red, blue, and the outline of the ground truth (if the molecule has one)
  function legend(task, truth) {
    const p = h("p", "gsq-legend");
    for (const [color, text] of [[RED, `raises the odds of “${task.yes}”`], [BLUE, "lowers them"]]) {
      const dot = h("i"); dot.style.background = color; p.append(h("span", "gsq-key", dot, text));
    }
    if (truth && truth.length) p.append(h("span", "gsq-key", h("i", "gsq-outline"), `the ${task.truthLabel}`));
    return p;
  }
  // the molecule's formula with subscripts, e.g. C₆H₄N₂O₄
  function formula(text) { const node = h("span", "gsq-formula"); for (const part of text.split(/(\d+)/)) { if (!part) continue; node.append(/^\d+$/.test(part) ? h("sub", null, part) : part); } return node; }

  // -- the GNN, as a chain of affine maps and ReLUs (see compute/pack.py) -------------------------
  function floats(base64) { const bin = atob(base64), buf = new ArrayBuffer(bin.length), u = new Uint8Array(buf); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new Float32Array(buf); }
  const stepsOf = (task) => task._steps || (task._steps = task.gnn.steps.map((s) => ({ shape: s.shape, w: floats(s.w), b: floats(s.b), relu: s.relu, gather: s.gather, sum: s.sum })));
  // the GNN's log-odds for a molecule whose `masked` atoms take the task's baseline features
  function predict(symbols, m, task, masked) {
    const nb = neighbors(m);
    let x = m.atoms.map((a, k) => {
      if (masked.has(k)) return task.baseline.slice();
      const v = new Array(symbols.length).fill(0); let e = symbols.indexOf(a[0]); if (e < 0) e = symbols.length - 1; v[e] = 1; return v;
    });
    let dim = symbols.length;
    for (const s of stepsOf(task)) {
      const [out, inp] = s.shape;
      if (s.gather) x = x.map((row_, k) => { const r = row_.slice(); for (const j of nb[k]) { const xj = x[j]; for (let d = 0; d < dim; d++) r[d] += xj[d]; } return r; });
      if (s.sum) { const z = new Array(dim).fill(0); for (const row_ of x) for (let d = 0; d < dim; d++) z[d] += row_[d]; x = [z]; }
      x = x.map((row_) => { const o = new Array(out); for (let r = 0; r < out; r++) { let acc = s.b[r]; const at = r * inp; for (let d = 0; d < inp; d++) acc += s.w[at + d] * row_[d]; o[r] = s.relu ? Math.max(0, acc) : acc; } return o; });
      dim = out;
    }
    return x[0][0];
  }

  // -- the animated SI-graph scene ----------------------------------------------------------------
  // Atoms are stable nodes; rings (what an atom adds alone), bands (pairs) and shapes (groups of
  // three or more) tween in and out; a masked atom dissolves; the focus glides. Geometry and
  // opacity are tweened with requestAnimationFrame, so that it is smooth in every browser.
  function createScene(plot, options = {}) {
    const ease = (t) => 1 - Math.pow(1 - t, 3);
    const duration = () => (reducedMotion() ? 0 : options.duration || 430);
    const tweens = new Map();
    let frame = 0;
    function tick(now) {
      let alive = false;
      for (const [key, tw] of tweens) {
        const p = tw.dur ? Math.min(1, (now - tw.t0) / tw.dur) : 1, k = ease(p);
        for (const a of tw.anim) a.set(a.from + (a.to - a.from) * k);
        if (p >= 1) { tweens.delete(key); if (tw.done) tw.done(); } else alive = true;
      }
      frame = alive ? requestAnimationFrame(tick) : 0;
    }
    function animate(key, anim, done) {
      tweens.set(key, { t0: performance.now(), dur: duration(), anim, done });
      if (!frame) frame = requestAnimationFrame(tick);
    }
    let layers = null, atoms = [], rings = [], bands = new Map(), shapes = new Map(), focusRing = null, keyRing = null, mol = null, g = null, truthKey = null;
    const hull = (points) => {
      const p = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      if (p.length < 3) return p;
      const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]), lo = [], up = [];
      for (const q of p) { while (lo.length > 1 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
      for (const q of p.slice().reverse()) { while (up.length > 1 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
      return lo.slice(0, -1).concat(up.slice(0, -1));
    };
    function components(m, set) {
      const inSet = new Set(set), seen = new Set(), nb = neighbors(m), out = [];
      for (const a of set) {
        if (seen.has(a)) continue;
        const comp = [], todo = [a]; seen.add(a);
        while (todo.length) { const k = todo.pop(); comp.push(k); for (const j of nb[k]) if (inSet.has(j) && !seen.has(j)) { seen.add(j); todo.push(j); } }
        out.push(comp);
      }
      return out;
    }
    function setMolecule(m) {
      mol = m; g = geom(m);
      tweens.clear();
      plot.replaceChildren();
      const layer = () => svg("g", {}, plot);
      layers = { truth: layer(), shape: layer(), ring: layer(), band: layer(), bond: layer(), atom: layer(), over: layer(), hit: layer() };
      layers.truth.style.opacity = 0;
      bands = new Map(); shapes = new Map(); rings = []; atoms = []; truthKey = null;
      const { s, pos } = g, r = 0.27 * s;
      skeleton(layers.bond, m, 0.15, "gsq-bond", g);
      m.atoms.forEach((a, k) => {
        const [x, y] = pos[k];
        rings.push({ el: svg("circle", { class: "gsq-ring", cx: x, cy: y, r, fill: RED, opacity: 0 }, layers.ring), r, op: 0 });
        const disc = svg("circle", { class: "gsq-atom", cx: x, cy: y, r }, layers.atom);
        let label = null;
        if (a[0] !== "C" || options.carbonLabels) {
          label = svg("text", { class: "gsq-el", x, y, "font-size": (a[0].length > 1 ? 0.26 : 0.32) * s }, layers.atom);
          label.textContent = a[0];
        }
        atoms.push({ disc, label, k, masked: false });
        svg("circle", { class: "gsq-hit", cx: x, cy: y, r: 0.48 * s, "data-k": k }, layers.hit);
      });
      focusRing = svg("circle", { class: "gsq-focus", cx: pos[0][0], cy: pos[0][1], r: r + 0.16 * s, opacity: 0 }, layers.over);
      keyRing = svg("circle", { class: "gsq-keyfocus", cx: pos[0][0], cy: pos[0][1], r: r + 0.22 * s, opacity: 0 }, layers.over);
    }
    function setRing(k, target, color) {
      const ring = rings[k];
      if (color) ring.el.setAttribute("fill", color);
      animate(`ring${k}`, [
        { from: ring.r, to: target.r, set: (v) => { ring.r = v; ring.el.setAttribute("r", Math.max(0, v)); } },
        { from: ring.op, to: target.op, set: (v) => { ring.op = v; ring.el.setAttribute("opacity", Math.max(0, v)); } },
      ]);
    }
    // state: rings (atom -> value), ringLargest, groups ([{atoms, v, emph}]), groupLargest, masked
    // (Set), focus (atom or null), truth (atoms), truthShown, dim (Set of atoms to keep bright)
    function update(state) {
      const { s, pos } = g, r = 0.27 * s;
      const own = state.rings || {}, rl = state.ringLargest || 1;
      for (let k = 0; k < rings.length; k++) {
        const v = own[k];
        if (v != null && Math.abs(v) / rl > 0.02) setRing(k, { r: r + (0.035 + 0.165 * (Math.abs(v) / rl)) * s, op: 1 }, colorOf(v));
        else setRing(k, { r, op: 0 });
      }
      const want = new Map(), gl = state.groupLargest || 1;
      for (const group of state.groups || []) {
        const strength = Math.abs(group.v) / gl, color = colorOf(group.v), em = group.emph == null ? 1 : group.emph;
        if (group.atoms.length === 2) {
          const [a, b] = group.atoms;
          want.set(`b${Math.min(a, b)}-${Math.max(a, b)}`, { kind: "band", a, b, color, w: (0.07 + 0.3 * strength) * s, op: (0.35 + 0.65 * strength) * em });
        } else {
          const points = hull(group.atoms.map((k) => pos[k]));
          want.set(`s${group.atoms.slice().sort((x, y) => x - y).join(",")}`, { kind: "shape", points: points.map((p) => p.join(",")).join(" "), poly: points.length > 2, color, fop: (0.07 + 0.13 * strength) * em, sop: (0.35 + 0.65 * strength) * em });
        }
      }
      for (const [key, band] of bands) {
        const w = want.get(key);
        if (w) {
          band.el.setAttribute("stroke", w.color);
          animate(key, [
            { from: band.w, to: w.w, set: (v) => { band.w = v; band.el.setAttribute("stroke-width", Math.max(0, v)); } },
            { from: band.op, to: w.op, set: (v) => { band.op = v; band.el.setAttribute("opacity", Math.max(0, v)); } },
          ]);
          want.delete(key);
        } else animate(key, [{ from: band.op, to: 0, set: (v) => { band.op = v; band.el.setAttribute("opacity", Math.max(0, v)); } }], () => { band.el.remove(); bands.delete(key); });
      }
      for (const [key, shape] of shapes) {
        const w = want.get(key);
        if (w) {
          shape.el.setAttribute("stroke", w.color); shape.el.setAttribute("fill", w.poly ? w.color : "none");
          animate(key, [
            { from: shape.fop, to: w.fop, set: (v) => { shape.fop = v; shape.el.setAttribute("fill-opacity", Math.max(0, v)); } },
            { from: shape.sop, to: w.sop, set: (v) => { shape.sop = v; shape.el.setAttribute("stroke-opacity", Math.max(0, v)); } },
          ]);
          want.delete(key);
        } else animate(key, [
          { from: shape.fop, to: 0, set: (v) => { shape.fop = v; shape.el.setAttribute("fill-opacity", Math.max(0, v)); } },
          { from: shape.sop, to: 0, set: (v) => { shape.sop = v; shape.el.setAttribute("stroke-opacity", Math.max(0, v)); } },
        ], () => { shape.el.remove(); shapes.delete(key); });
      }
      for (const [key, w] of want) {
        if (w.kind === "band") {
          const [x1, y1] = pos[w.a], [x2, y2] = pos[w.b];
          const band = { el: svg("line", { class: "gsq-band", x1, y1, x2, y2, stroke: w.color, "stroke-width": 0, opacity: 0 }, layers.band), w: 0, op: 0 };
          bands.set(key, band);
          animate(key, [
            { from: 0, to: w.w, set: (v) => { band.w = v; band.el.setAttribute("stroke-width", Math.max(0, v)); } },
            { from: 0, to: w.op, set: (v) => { band.op = v; band.el.setAttribute("opacity", Math.max(0, v)); } },
          ]);
        } else {
          const shape = { el: svg(w.poly ? "polygon" : "polyline", { class: "gsq-shape", points: w.points, fill: w.poly ? w.color : "none", stroke: w.color, "fill-opacity": 0, "stroke-opacity": 0 }, layers.shape), fop: 0, sop: 0 };
          shapes.set(key, shape);
          animate(key, [
            { from: 0, to: w.fop, set: (v) => { shape.fop = v; shape.el.setAttribute("fill-opacity", Math.max(0, v)); } },
            { from: 0, to: w.sop, set: (v) => { shape.sop = v; shape.el.setAttribute("stroke-opacity", Math.max(0, v)); } },
          ]);
        }
      }
      const masked = state.masked || new Set(), keep = state.dim;
      for (const atom of atoms) {
        const m = masked.has(atom.k), faded = keep && !keep.has(atom.k);
        if (m !== atom.masked) { atom.masked = m; atom.disc.classList.toggle("gsq-masked", m); }
        atom.disc.style.opacity = faded ? 0.3 : 1;
        if (atom.label) atom.label.style.opacity = m ? 0 : faded ? 0.3 : 1;
        rings[atom.k].el.style.opacity = faded ? 0.25 : 1;
      }
      moveRing(focusRing, "focus", state.focus);
      // the ground truth: one thin outline around each match of the task's pattern
      const truth = state.truth || [], key = state.truthKey ?? truth.join(",");
      if (key !== truthKey) {
        truthKey = key;
        layers.truth.replaceChildren();
        const R = 0.6 * s, w = 0.42 * s;
        for (const [cls, rad] of [["gsq-lasso-out", R], ["gsq-lasso-in", R - w]]) {
          for (const comp of components(mol, truth)) {
            const inComp = new Set(comp);
            for (const k of comp) svg("circle", { class: cls, cx: pos[k][0], cy: pos[k][1], r: rad, "stroke-width": 0 }, layers.truth);
            for (const [i, j] of mol.bonds) if (inComp.has(i) && inComp.has(j)) svg("line", { class: cls, x1: pos[i][0], y1: pos[i][1], x2: pos[j][0], y2: pos[j][1], "stroke-width": 2 * rad }, layers.truth);
            for (const ringAtoms of mol.rings || []) if (ringAtoms.every((k) => inComp.has(k))) svg("polygon", { class: cls, points: ringAtoms.map((k) => pos[k].join(",")).join(" "), "stroke-width": 0 }, layers.truth);
          }
        }
      }
      const shown = state.truthShown && truth.length ? 0.5 : 0, from = Number(layers.truth.style.opacity || 0);
      if (from !== shown) animate("truth", [{ from, to: shown, set: (v) => { layers.truth.style.opacity = v; } }]);
    }
    function moveRing(node, key, k) {
      const op = Number(node.getAttribute("opacity"));
      if (k == null) { if (op) animate(key, [{ from: op, to: 0, set: (v) => node.setAttribute("opacity", v) }]); return; }
      const [x, y] = g.pos[k], cx = op ? Number(node.getAttribute("cx")) : x, cy = op ? Number(node.getAttribute("cy")) : y;
      animate(key, [
        { from: cx, to: x, set: (v) => node.setAttribute("cx", v) },
        { from: cy, to: y, set: (v) => node.setAttribute("cy", v) },
        { from: op, to: 1, set: (v) => node.setAttribute("opacity", v) },
      ]);
    }
    // the atom under a pointer event (or -1)
    function atomAt(event) {
      const point = plot.createSVGPoint(); point.x = event.clientX; point.y = event.clientY;
      const ctm = plot.getScreenCTM(); if (!ctm || !g) return -1;
      const p = point.matrixTransform(ctm.inverse());
      let best = -1, bd = (0.5 * g.s) ** 2;
      g.pos.forEach(([x, y], k) => { const d = (x - p.x) ** 2 + (y - p.y) ** 2; if (d < bd) { bd = d; best = k; } });
      return best;
    }
    return { setMolecule, update, atomAt, keyFocus: (k) => moveRing(keyRing, "key", k), hits: () => [...layers.hit.children], get geom() { return g; } };
  }

  // -- the controls, as the FIxLIP and orders demos' --------------------------------------------
  // a switch: a few options, the one chosen under a pill that slides to the next one chosen (a copy
  // of the options over them, in the colors of the one chosen, cut to the pill, so that every label
  // turns as the pill passes over it). `set(k)` chooses an option; `pick(k)` hears a click on one
  function switcher(label, options, pick) {
    const node = h("div", "gsq-switch");
    node.setAttribute("role", "group"); node.setAttribute("aria-label", label);
    const buttons = options.map((text, k) => { const b = h("button", "gsq-opt", text); b.type = "button"; b.addEventListener("click", () => pick(k)); node.append(b); return b; });
    const pill = h("div", "gsq-pill", ...options.map((text) => h("span", "gsq-opt", text)));
    pill.setAttribute("aria-hidden", "true"); node.append(pill);
    let at = -1;
    const move = (instant) => {
      const b = buttons[at];
      if (!b || !pill.clientWidth) return;
      if (instant) pill.style.transition = "none";
      pill.style.clipPath = `inset(${b.offsetTop}px ${pill.clientWidth - b.offsetLeft - b.offsetWidth}px ${pill.clientHeight - b.offsetTop - b.offsetHeight}px ${b.offsetLeft}px round 999px)`;
      if (instant) { pill.getBoundingClientRect(); pill.style.transition = ""; }
    };
    let placed = false;
    new ResizeObserver(() => { move(true); placed = pill.clientWidth > 0; }).observe(node);
    return { node, buttons, set(k) { at = k; buttons.forEach((b, i) => b.setAttribute("aria-pressed", String(i === k))); move(!placed || reducedMotion()); } };
  }
  // the molecules to choose from: a tile each, with the molecule's name (`short`, if it has one)
  function picker(label, molecules, pick) {
    const node = h("div", "gsq-mols");
    node.setAttribute("role", "group"); node.setAttribute("aria-label", label);
    const buttons = molecules.map((m, k) => {
      const b = h("button", "gsq-mol", tile(m), h("span", "", m.short || m.name)); b.type = "button"; b.title = m.name;
      b.addEventListener("click", () => pick(k)); node.append(b);
      return b;
    });
    return { node, buttons, set(k) { buttons.forEach((b, i) => b.setAttribute("aria-pressed", String(i === k))); } };
  }
  // the hint over the molecule, until the reader has done what it says: `show(text)` and `hide()`
  function hinter(frame) {
    const node = h("span", "gsq-hint"); node.setAttribute("aria-hidden", "true"); node.hidden = true; frame.append(node);
    return { show(text) { node.textContent = text; node.hidden = false; }, hide() { node.hidden = true; } };
  }

  // -- mount: fill every container of a demo -----------------------------------------------------
  // The page loads the demo's data.js before its demo.js; where it has not (a page written by
  // hand), the data is loaded from the container's data-root, past caches. Without data, or when
  // the demo fails, the container shows its static figure (the directive's :fallback:).
  function mount(className, global, start) {
    const roots = [...document.querySelectorAll(`.${className}`)];
    if (!roots.length) return;
    const fallbacks = roots.map((root) => root.querySelector("noscript")?.textContent ?? "");
    const fail = (root, k, error) => { if (error) console.error(`The demo ${className} could not start:`, error); root.classList.remove("gsq"); root.classList.add("gsq-failed"); root.innerHTML = fallbacks[k]; };
    const begin = () => roots.forEach((root, k) => {
      try {
        root.classList.add("gsq");
        root.replaceChildren();
        start(root, window[global]);
      } catch (error) { fail(root, k, error); }
    });
    if (window[global]) { queueMicrotask(begin); return; }
    const loader = document.createElement("script");
    loader.src = `${roots[0].dataset.root ?? ""}data.js?v=${roots[0].dataset.version ?? ""}`;
    loader.onload = begin;
    loader.onerror = () => roots.forEach((root, k) => fail(root, k, new Error("data.js did not load")));
    document.head.appendChild(loader);
  }

  window.ShapiqGraph = { RED, BLUE, VIOLET, NAMES, reducedMotion, hovers, switcher, picker, hinter, h, svg, sig, bits, signed, percent, colorOf, value, sameSet, neighbors, geom, skeleton, tile, row, groupName, atomName, legend, formula, predict, createScene, mount };
})();
