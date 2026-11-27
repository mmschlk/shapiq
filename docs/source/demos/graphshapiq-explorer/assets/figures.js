// The figures of the GraphSHAP-IQ explorer's page (demos/graphshapiq-explorer/index.rst, the
// shapiq-demo-figure directive), in the system of the FIxLIP page's figures (fixlip/assets).
//
// Every figure explains one step of the write-up on the molecule that the explorer shows, for the
// question it asks. Choosing another molecule in the explorer, or in the selector of any figure,
// changes it in all of them: the event EVENT ({molecule, task, source}), which the explorer sends
// with source null and follows when a figure sends it with itself as the source. A figure that
// starts after the explorer reads ShapiqGraph.current. On a narrow page a figure has dots in place
// of the thumbnails, and on a touch screen a swipe sideways goes to the next molecule. The fixed
// figures (`fixed: true`) have no selector and do not follow the molecule.
//
// The figures read the explorer's data (data.js, window.SHAPIQ_GRAPHSHAPIQ_EXPLORER: the folded
// GNNs, the molecules, the explanations) and what ../compute/ wrote beyond it (figures-data.js,
// window.SHAPIQ_GRAPHSHAPIQ_EXPLORER_FIGURES, one part per script). They draw molecules with the
// shared library of the GraphSHAP-IQ demos (../../graphshapiq-common/assets/demo.js, ShapiqGraph)
// and can run the GNNs in the page (ShapiqGraph.predict). A figure is drawn at the width it has on
// the page, so its text keeps its size on a phone, and its colors are those of figures.css, so it
// follows the theme. Without JavaScript, a figure shows its description.
//
// A figure is an entry of FIGURES, in a marked section of its own below (one agent edits one
// section): `FIGURES.<name> = { fixed, reserve, draw(layer, c, state) }`. See the note above
// FIGURES for what `c` holds, and /mnt/project-files/graphshapiq-writeup/FRAME.md.
(() => {
  const figures = [...document.querySelectorAll("figure.graphshapiq-explorer-figure")];
  if (!figures.length) return;

  const G = window.ShapiqGraph; // the shared library, loaded before the demo's scripts
  const SVG = "http://www.w3.org/2000/svg";
  const EVENT = "graphshapiq-demo-molecule"; // as in demo.js
  const RED = [255, 13, 87]; // #FF0D57, raises the log-odds (as the demo and shapiq's plots)
  const BLUE = [30, 136, 229]; // #1E88E5, lowers them
  const RED_HEX = "#ff0d57";
  const BLUE_HEX = "#1e88e5";
  // the molecules in the explorer's order: by question, those with the question's pattern first
  // (MOLS in demo.js)
  const TASK_ORDER = ["mutagenic", "indole", "PAINS"];
  const root = figures[0].dataset.root;

  // -- helpers -------------------------------------------------------------------------------------

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const add = (parent, ...children) => {
    parent.append(...children.filter((child) => child != null));
    return parent;
  };
  const svgEl = (tag, attributes = {}, text) => {
    const node = document.createElementNS(SVG, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const range = (n) => [...Array(n).keys()];
  const sum = (values) => values.reduce((total, value) => total + value, 0);
  const mean = (values) => sum(values) / Math.max(1, values.length);
  const fmt = (value, digits = 2) => value.toFixed(digits).replace("-", "−");
  const signed = (value, digits = 2) => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
  const percent = (p) => (p > 0.995 ? ">99%" : p < 0.005 ? "<1%" : `${Math.round(100 * p)}%`);
  const sig = (x) => 1 / (1 + Math.exp(-x));
  const thousands = (value) => Math.round(value).toLocaleString("en-US");
  // a value as a color: red raises, blue lowers, the strength is the opacity
  const rgba = (value, top, strength = 0.85) => {
    const [r, g, b] = value >= 0 ? RED : BLUE;
    const alpha = top > 0 ? Math.min(Math.abs(value) / top, 1) * strength : 0;
    return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`;
  };
  const colorOf = (value) => (value >= 0 ? RED_HEX : BLUE_HEX);
  const largest = (values) => Math.max(1e-12, ...[...values].map(Math.abs));
  // a text with *emphasis* as nodes
  const rich = (text) =>
    text.split("*").map((part, k) => (k % 2 ? el("em", "", part) : document.createTextNode(part)));
  const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  // a small generator of numbers that look random, the same for the same seed
  const noise = (seed) => {
    let state = [...String(seed)].reduce((hash, char) => (hash * 31 + char.charCodeAt(0)) >>> 0, 7);
    return () => {
      state = (state * 1664525 + 1013904223) >>> 0;
      return state / 2 ** 32;
    };
  };
  // a coalition as a bitmask of atoms (data.js) and back
  const bits = (mask) => {
    const atoms = [];
    for (let k = 0; mask; k++, mask = Math.floor(mask / 2)) if (mask % 2) atoms.push(k);
    return atoms;
  };
  const maskOf = (atoms) => atoms.reduce((mask, k) => mask + 2 ** k, 0);

  // A slider with its label and its value, in the page's style. `format` writes the value (e.g.
  // "n" for the last order), `values` lists the values the steps stand for (then the input runs
  // over their indices). A drag on it is not a swipe of the figure. `node.set(value)` moves it.
  const slider = (text, [min, max], value, onInput, { format = String, values = null } = {}) => {
    const node = el("label", "gsqf-slider");
    const input = el("input");
    const of = (v) => (values ? values.indexOf(v) : v);
    const at = (i) => (values ? values[i] : i);
    const [low, high] = values ? [0, values.length - 1] : [min, max];
    Object.assign(input, { type: "range", min: low, max: high, step: 1, value: of(value) });
    const output = el("output", "", format(value));
    // the part of the track left of the thumb, which the style sheet fills
    const fill = () => input.style.setProperty("--gsqf-part", (input.value - low) / (high - low || 1));
    node.set = (now) => {
      input.value = of(now);
      output.textContent = format(now);
      fill();
    };
    node.input = input;
    input.addEventListener("input", () => {
      const now = at(Number(input.value));
      output.textContent = format(now);
      fill();
      onInput(now);
    });
    fill();
    for (const type of ["touchstart", "touchmove"])
      node.addEventListener(type, (event) => event.stopPropagation(), { passive: true });
    return add(node, el("span", "", text), input, output);
  };

  // -- the data ------------------------------------------------------------------------------------

  let data = null; // the explorer's (data.js)
  let extra = null; // the figures' (figures-data.js): {<part>: ...}
  let MOLS = []; // the molecules, in the explorer's order
  let current = null; // {molecule, task}: the ids of the molecule and the question shown
  const explanations = new Map();

  const taskOf = (id) => data.tasks.find((t) => t.id === id);
  const moleculeOf = (id) => data.molecules.find((m) => m.id === id);
  // An explanation of the explorer (data.js, explanations["<molecule>/<task>"]: logOdds, p, empty,
  // budget, nAtoms, truth, kSII {"1".."4": [[mask, value]]}, moebius [[mask, value]],
  // moebiusCount), with its terms as lists. Only the strongest terms are in data.js (all singletons,
  // the 24 strongest groups and the 8 strongest of every atom, per order): a figure that needs all
  // of them computes them in ../compute/ (gnn.py).
  const explanationOf = (molecule, task) => {
    const key = `${molecule}/${task}`;
    if (!explanations.has(key)) {
      const e = data.explanations[key];
      if (!e) return null;
      // order: 1 to 4 (k-SII) or "M" (Möbius)
      const rows = (order) => (order === "M" || order === "n" ? e.moebius : e.kSII[String(order)] ?? []);
      const terms = (order) => rows(order).map(([mask, v]) => ({ atoms: bits(mask), v }));
      explanations.set(key, {
        ...e,
        key,
        rows,
        terms,
        // the values of single atoms: {atom: value}
        singles: (order) => Object.fromEntries(terms(order).filter((t) => t.atoms.length === 1).map((t) => [t.atoms[0], t.v])),
        // the groups of two atoms or more, the strongest first
        groups: (order) => terms(order).filter((t) => t.atoms.length > 1).sort((a, b) => Math.abs(b.v) - Math.abs(a.v)),
      });
    }
    return explanations.get(key);
  };
  // The GNN of a question on a molecule whose `masked` atoms (a Set or a list) take the question's
  // baseline features: its log-odds, in the page (ShapiqGraph.predict, float64 on the float32 weights).
  const predict = (m, task, masked = []) => G.predict(data.symbols, m, task, masked instanceof Set ? masked : new Set(masked));

  // -- the motif: a molecule as shapiq's si_graph_plot draws it -------------------------------------

  // the hub of a hyperedge: the centre of its atoms, moved off any atom that sits there (as
  // createScene's hubOf in graphshapiq-common/assets/demo.js, which is not exported)
  function hubOf(g, group) {
    const { s, pos } = g;
    const P = group.map((k) => pos[k]);
    const x = mean(P.map((p) => p[0]));
    const y = mean(P.map((p) => p[1]));
    const clear = (cx, cy) => pos.every((p) => Math.hypot(p[0] - cx, p[1] - cy) >= 0.42 * s);
    if (clear(x, y)) return [x, y];
    const mx = mean(pos.map((p) => p[0]));
    const my = mean(pos.map((p) => p[1]));
    let [a, b] = [P[0], P[1]];
    let far = 0;
    for (const p of P)
      for (const q of P) {
        const d = Math.hypot(p[0] - q[0], p[1] - q[1]);
        if (d > far) [far, a, b] = [d, p, q];
      }
    let nx = -(b[1] - a[1]) / (far || 1);
    let ny = (b[0] - a[0]) / (far || 1);
    if (nx * (x - mx) + ny * (y - my) < 0) [nx, ny] = [-nx, -ny];
    for (const step of [0.5, 0.75, 1])
      for (const side of [1, -1]) {
        const cx = x + side * nx * step * s;
        const cy = y + side * ny * step * s;
        if (clear(cx, cy)) return [cx, cy];
      }
    return [x + nx * 0.5 * s, y + ny * 0.5 * s];
  }
  // a hyperedge of width w as two paths: the round ends, and the bands that narrow halfway (as
  // shapiq's _draw_fancy_hyper_edges, and the demo's edgePaths)
  function edgePaths(g, group, w) {
    const { pos } = g;
    const r = Math.max(0, w) / 2;
    const f = (v) => v.toFixed(2);
    const hub = group.length === 2 ? pos[group[1]] : hubOf(g, group);
    const from = group.length === 2 ? [group[0]] : group;
    const disc = ([x, y]) => `M${f(x - r)} ${f(y)}a${f(r)} ${f(r)} 0 1 1 ${f(2 * r)} 0a${f(r)} ${f(r)} 0 1 1 ${f(-2 * r)} 0Z`;
    let ends = disc(hub);
    let bands = "";
    for (const k of from) {
      const [px, py] = pos[k];
      const dx = hub[0] - px;
      const dy = hub[1] - py;
      const L = Math.hypot(dx, dy);
      ends += disc(pos[k]);
      if (L < 1e-6) continue;
      const nx = (-dy / L) * r;
      const ny = (dx / L) * r;
      const mx = px + dx / 2;
      const my = py + dy / 2;
      bands += `M${f(px - nx)} ${f(py - ny)}Q${f(mx)} ${f(my)} ${f(hub[0] - nx)} ${f(hub[1] - ny)}L${f(hub[0] + nx)} ${f(hub[1] + ny)}Q${f(mx)} ${f(my)} ${f(px + nx)} ${f(py + ny)}Z`;
    }
    return [ends, bands];
  }
  // the connected parts of a set of atoms
  function components(m, atoms) {
    const inside = new Set(atoms);
    const seen = new Set();
    const nb = G.neighbors(m);
    const parts = [];
    for (const a of atoms) {
      if (seen.has(a)) continue;
      const part = [];
      const todo = [a];
      seen.add(a);
      while (todo.length) {
        const k = todo.pop();
        part.push(k);
        for (const j of nb[k])
          if (inside.has(j) && !seen.has(j)) {
            seen.add(j);
            todo.push(j);
          }
      }
      parts.push(part);
    }
    return parts;
  }
  const asMap = (values) => (values instanceof Map ? Object.fromEntries(values) : Array.isArray(values) ? Object.fromEntries(values.map((v, k) => [k, v])) : values ?? {});

  // A molecule as the explorer draws it: bonds, an atom as a disc with the letter of its element
  // (carbons without), in a square of 100 x 100 that the SVG fills (its width is the CSS width, or
  // `size` pixels). Options, all optional:
  //   halos   {atom: value}       a halo around the atom, as the demo's Shapley values and
  //                               si_graph_plot's nodes: its size is the strength, red or blue
  //   tints   {atom: value}       the disc filled red or blue, the strength as the opacity
  //   groups  [{atoms, v, emph}]  hyperedges as si_graph_plot draws them (a band between two atoms,
  //                               bands to a hub for more); `emph` (0 to 1) fades one
  //   largest number              the scale of halos, tints and groups (default: the largest of
  //                               them, one scale for all as si_graph_plot)
  //   masked  Set                 masked atoms: gray, with a faint letter
  //   framed  Set                 atoms in an amber frame (what the figure points at)
  //   dim     Set                 the atoms that stay bright; the others step back
  //   truth   [atoms]             the outline of the question's pattern (light amber, as the demo)
  //   letters "hetero" | "all" | false   which atoms carry their letter (default "hetero")
  //   numbers boolean             the atom's index (from 1) beside it
  //   room, cap                   the share of the square the molecule fills, the longest bond
  //                               (ShapiqGraph.geom; defaults 72 and 14, as the demo)
  //   box     [x, y, w, h]        the part of the square shown (viewBox)
  //   label   string              the description for screen readers (default: the name)
  // It returns the <svg>, with `.geom` ({s, pos}: the bond length and the atoms' places in the
  // square), `.at(k)` (an atom's place) and `.layers` (truth, edges, halos, bonds, atoms, over: <g>
  // elements to draw into, e.g. svgEl("circle", ...) into `.layers.over`).
  function molecule(m, options = {}) {
    const { masked, framed, dim, truth, numbers = false, letters = "hetero", room = 72, cap = 14, box, size } = options;
    const g = G.geom(m, room, cap);
    const { s, pos } = g;
    const r = 0.27 * s;
    const halos = asMap(options.halos);
    const tints = asMap(options.tints);
    const groups = options.groups ?? [];
    const top = options.largest ?? largest([...Object.values(halos), ...Object.values(tints), ...groups.map((group) => group.v), 0]);
    const node = svgEl("svg", { class: "gsqf-molecule", viewBox: (box ?? [0, 0, 100, 100]).join(" "), role: "img", "aria-label": options.label ?? m.name });
    if (size) node.style.width = `${size}px`;
    const layers = {};
    for (const name of ["truth", "edges", "halos", "bonds", "atoms", "over"]) layers[name] = node.appendChild(svgEl("g", { class: `gsqf-${name}` }));
    // the pattern of the question: one outline around each match
    if (truth?.length) {
      const R = 0.54 * s;
      const w = 0.3 * s;
      for (const [cls, rad] of [["gsqf-lasso-out", R], ["gsqf-lasso-in", R - w]])
        for (const part of components(m, truth)) {
          const inside = new Set(part);
          for (const k of part) layers.truth.append(svgEl("circle", { class: cls, cx: pos[k][0], cy: pos[k][1], r: rad }));
          for (const [i, j] of m.bonds)
            if (inside.has(i) && inside.has(j)) layers.truth.append(svgEl("line", { class: cls, x1: pos[i][0], y1: pos[i][1], x2: pos[j][0], y2: pos[j][1], "stroke-width": 2 * rad }));
          for (const ring of m.rings ?? [])
            if (ring.every((k) => inside.has(k))) layers.truth.append(svgEl("polygon", { class: cls, points: ring.map((k) => pos[k].join(",")).join(" ") }));
        }
    }
    // the hyperedges, the weakest first so that the strongest lie on top
    for (const group of [...groups].sort((a, b) => Math.abs(a.v) - Math.abs(b.v))) {
      const strength = Math.min(1, Math.abs(group.v) / top);
      const emph = group.emph ?? 1;
      const [ends, bands] = edgePaths(g, group.atoms, (0.06 + 0.3 * strength) * s);
      const edge = svgEl("g", { class: "gsqf-edge", fill: colorOf(group.v), opacity: ((0.3 + 0.55 * strength) * emph).toFixed(3) });
      edge.append(svgEl("path", { d: ends }), svgEl("path", { d: bands }));
      layers.edges.append(edge);
    }
    for (const [k, v] of Object.entries(halos)) {
      const strength = Math.abs(v) / top;
      if (!(strength > 0.02)) continue;
      const [x, y] = pos[k];
      layers.halos.append(svgEl("circle", { class: "gsqf-halo", cx: x, cy: y, r: (r + (0.035 + 0.165 * strength) * s).toFixed(2), fill: colorOf(v) }));
    }
    G.skeleton(layers.bonds, m, 0.15, "gsqf-bond", g);
    m.atoms.forEach((atom, k) => {
      const [x, y] = pos[k];
      const faded = dim && !dim.has(k);
      const hidden = masked?.has(k);
      const cls = `gsqf-atom${hidden ? " gsqf-masked" : ""}${faded ? " gsqf-faded" : ""}`;
      const group = svgEl("g", { class: cls });
      group.append(svgEl("circle", { class: "gsqf-disc", cx: x, cy: y, r: r.toFixed(2) }));
      if (tints[k] != null && !hidden) group.append(svgEl("circle", { class: "gsqf-tint", cx: x, cy: y, r: r.toFixed(2), fill: rgba(tints[k], top) }));
      if (letters === "all" || (letters && atom[0] !== "C"))
        group.append(svgEl("text", { class: "gsqf-letter", x, y, "font-size": ((atom[0].length > 1 ? 0.26 : 0.32) * s).toFixed(2) }, atom[0]));
      if (numbers) group.append(svgEl("text", { class: "gsqf-number", x: x + 0.3 * s, y: y - 0.3 * s, "font-size": (0.22 * s).toFixed(2) }, String(k + 1)));
      if (framed?.has(k)) group.append(svgEl("circle", { class: "gsqf-frame", cx: x, cy: y, r: (r + 0.11 * s).toFixed(2) }));
      layers.atoms.append(group);
    });
    node.geom = g;
    node.at = (k) => pos[k];
    node.layers = layers;
    return node;
  }
  // a molecule's tile in a selector: its bonds, and a dot for every atom that is not carbon
  const tile = (m) => G.tile(m);

  // -- parts of a figure ---------------------------------------------------------------------------

  // a panel with a title over it
  const panel = (title, ...content) => {
    const node = el("div", "gsqf-panel");
    if (title != null) {
      const head = el("p", "gsqf-title");
      head.append(...(typeof title === "string" ? rich(title) : [title]));
      node.append(head);
    }
    return add(node, ...content);
  };
  // a labelled value, e.g. reading("log-odds", signed(3.98)); `className` "gsqf-up" or "gsqf-down"
  // colors the value
  const reading = (label, value, className = "") => {
    const node = el("p", "gsqf-reading");
    const strong = el("strong", className, value);
    return add(node, el("span", "", label), strong);
  };
  // panels side by side, which stack on a narrow figure: className "gsqf-two" ... "gsqf-five"
  const columns = (className, ...panels) => add(el("div", `gsqf-columns ${className}`.trim()), ...panels);
  // a legend: [[swatch class, text], ...], swatch classes gsqf-amber, gsqf-gray, gsqf-ink, gsqf-red,
  // gsqf-blue, gsqf-round, gsqf-solid, gsqf-dashed (combine them)
  const legend = (items) =>
    add(el("p", "gsqf-legend"), ...items.map(([swatch, text]) => add(el("span"), el("i", `gsqf-swatch ${swatch}`), text)));

  // A part that keeps the height of its highest form, so that nothing moves when the molecule or a
  // slider changes it. `forms` fill a probe of the part (a function each, given the empty probe);
  // the part gets their highest height as its min-height. Call it once the part is on the page
  // (the layer is, while a figure draws).
  const reserve = (part, forms) => {
    if (!part.isConnected) return;
    const width = part.getBoundingClientRect().width;
    if (!width) return;
    const probe = part.cloneNode(false);
    Object.assign(probe.style, { position: "absolute", visibility: "hidden", left: "0", top: "0", width: `${width}px`, minHeight: "0" });
    part.after(probe);
    let high = 0;
    for (const form of forms) {
      probe.replaceChildren();
      form(probe);
      high = Math.max(high, probe.getBoundingClientRect().height);
    }
    probe.remove();
    part.style.minHeight = `${Math.ceil(high)}px`;
  };

  // -- charts --------------------------------------------------------------------------------------

  // A chart as wide as its place on the page, in pixels, with its plot inside the margins. Its
  // height is a number, or a function of its margins and its width. Scales: `linear(domain,
  // pixels)` and `log(domain, pixels)` (base 10, domain > 0). Axes: `xAxis(x, ticks, label,
  // format)`, `yAxis(y, ticks, label, format)`, with `logTicks(low, high)` for a log axis and
  // `power` as its format (10², 10³, ...). `path(points, attributes)`, `draw(tag, attributes,
  // text)`, and `inset(...nodes)` centres a legend or title over the plot, not over the drawing.
  function chart(parent, height, margins = {}) {
    // where the texts of the figure are larger (figures.css, on a narrow page), so is their room
    const type = parseFloat(getComputedStyle(parent).getPropertyValue("--gsqf-type")) || 1;
    const m = { left: 44, right: 12, top: 12, bottom: 30, ...margins };
    for (const side of Object.keys(m)) m[side] = Math.round(m[side] * type);
    const width = Math.max(120, Math.floor(parent.getBoundingClientRect().width));
    if (typeof height === "function") height = height(m, width); // a height that follows the plot
    const svg = svgEl("svg", { class: "gsqf-chart", width, height, viewBox: `0 0 ${width} ${height}` });
    parent.append(svg);
    const box = { x0: m.left, x1: width - m.right, y0: m.top, y1: height - m.bottom };
    const draw = (tag, attributes, text) => svg.appendChild(svgEl(tag, attributes, text));
    const linear = (domain, pixels) => (value) =>
      pixels[0] + ((value - domain[0]) / (domain[1] - domain[0])) * (pixels[1] - pixels[0]);
    const log = (domain, pixels) => {
      const scale = linear(domain.map(Math.log10), pixels);
      return (value) => scale(Math.log10(value));
    };
    // a tick label: a string, or a list of [text, superscript] parts (power)
    const label = (attributes, text) => {
      if (typeof text === "string") return draw("text", attributes, text);
      const node = draw("text", attributes, text[0]);
      node.append(svgEl("tspan", { dy: -5 * type, "font-size": 9 * type }, text[1]));
      return node;
    };
    const xAxis = (x, ticks, text, format = (t) => fmt(t, 1)) => {
      draw("line", { class: "gsqf-axis", x1: box.x0, x2: box.x1, y1: box.y1, y2: box.y1 });
      for (const tick of ticks) label({ x: x(tick), y: box.y1 + 15 * type, "text-anchor": "middle" }, format(tick));
      if (text) draw("text", { x: (box.x0 + box.x1) / 2, y: height - 2, "text-anchor": "middle" }, text);
    };
    const yAxis = (y, ticks, text, format = (t) => fmt(t, 1)) => {
      for (const tick of ticks) {
        draw("line", { class: "gsqf-grid", x1: box.x0, x2: box.x1, y1: y(tick), y2: y(tick) });
        label({ x: box.x0 - 6, y: y(tick) + 4 * type, "text-anchor": "end" }, format(tick));
      }
      if (text)
        draw(
          "text",
          { x: 11 * type, y: (box.y0 + box.y1) / 2, "text-anchor": "middle", transform: `rotate(-90 ${11 * type} ${(box.y0 + box.y1) / 2})` },
          text,
        );
    };
    const path = (points, attributes) =>
      draw("path", { d: points.map(([x, y], k) => `${k ? "L" : "M"} ${x.toFixed(1)} ${y.toFixed(1)}`).join(" "), fill: "none", ...attributes });
    // A legend, a title or a reading of the chart lies over the middle of its plot, and not of the
    // drawing with its margins. Where this would break its line, it stays where it is.
    const inset = (...nodes) => {
      for (const node of nodes) {
        node.style.paddingInline = "";
        const high = node.getBoundingClientRect().height;
        node.style.paddingInline = `${m.left}px ${m.right}px`;
        if (node.getBoundingClientRect().height > high + 1) node.style.paddingInline = "";
      }
    };
    return { svg, width, height, box, margins: m, draw, linear, log, xAxis, yAxis, path, inset, type };
  }
  // whether a part lies at the right of another one, and not under it as on a narrow page
  const beside = (left, right) => right.getBoundingClientRect().left >= left.getBoundingClientRect().right;
  // a step that gives a handful of ticks for a range
  const stepOf = (low, high) => [0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100].find((step) => (high - low) / step <= 6) ?? 10 ** Math.ceil(Math.log10((high - low) / 6));
  const ticksOf = (low, high, step = stepOf(low, high)) => {
    const ticks = [];
    for (let t = Math.ceil(low / step - 1e-9) * step; t <= high + 1e-9; t += step) ticks.push(Math.round(t * 1e6) / 1e6);
    return ticks;
  };
  // the powers of ten from low to high (every `every`-th), for a log axis
  const logTicks = (low, high, every = 1) => {
    const ticks = [];
    for (let p = Math.ceil(Math.log10(low) - 1e-9); p <= Math.log10(high) + 1e-9; p += every) ticks.push(10 ** p);
    return ticks;
  };
  // a power of ten as a tick label: 1, 10, then 10² ... (pass as the format of an axis)
  const power = (value) => {
    const p = Math.round(Math.log10(value));
    return p === 0 ? "1" : p === 1 ? "10" : ["10", String(p).replace("-", "−")];
  };

  // A placeholder: the molecule shown with its Shapley values and the figure's name, until the
  // figure's agent draws the figure. (Remove once every figure is drawn.)
  const placeholder = (layer, c, name) => {
    const ex = c.ex;
    const art = c.molecule ? molecule(c.molecule, { halos: ex?.singles(1), truth: ex?.truth }) : null;
    if (art) art.classList.add("gsqf-placeholder-art");
    layer.append(
      panel(
        name,
        art,
        c.molecule ? reading(`log-odds of “${c.task.yes}”`, signed(ex.logOdds), ex.logOdds >= 0 ? "gsqf-up" : "gsqf-down") : null,
      ),
    );
  };

  // -- the figures ----------------------------------------------------------------------------------
  //
  // FIGURES.<name> = { fixed, reserve, draw(layer, c, state) } (or only the draw function):
  //   fixed    true for a figure that does not follow the molecule: no selector (arrangement,
  //            paper-approximation); its `c` has molecule, task and ex null
  //   reserve  false to keep the frame from reserving the stage's height (see below)
  //   draw     draws the figure into `layer`, an empty <div> that is already on the page, so that
  //            its width can be measured (charts do). It must add listeners only to what it draws.
  //   c        molecule  the molecule shown (data.js molecules[]: id, name, short, task, atoms
  //                      [[element, x, y, hydrogens, charge]], bonds [[i, j, order]], rings, ...)
  //            task      the question shown (data.js tasks[]: id, label, yes, no, gnn, baseline, ...):
  //                      the one the explorer asks, which is the molecule's own (`own`) unless the
  //                      reader asked another one in the explorer
  //            own       the molecule's own question (molecule.task as a task)
  //            ex        explanationOf(molecule.id, task.id)
  //            data      the explorer's data, extra: figures-data.js, part: extra[<name>]
  //            molecules the molecules in the explorer's order, tasks: data.tasks
  //            name      the figure's name, figure: its <figure>, width: the stage's width
  //            redraw()  draws the figure again (as a change, part by part: morph)
  //   state    an object of the figure that lasts while the page is open (a slider's place)
  // For a figure that follows the molecule, the frame draws it once more for every molecule into
  // a hidden probe, at every new width, and gives the stage the highest height (reserve), so that
  // the page does not move when the molecule changes. A probe gets a copy of `state`.
  const FIGURES = {};

  // ===== figure: pipeline =====
  // From a molecule to the log-odds: the receptive field of one atom (amber) grows by one bond per
  // layer, with the atom's node features and its embedding after each layer as strips of their real
  // values; then sum pooling and the linear output layer, which make the log-odds the bias plus one
  // term per atom (tints). Data: compute/pipeline.py (per molecule and question).
  FIGURES.pipeline = {
    draw(layer, c) {
      const { molecule: m, task } = c;
      const d = c.part?.[`${m.id}/${task.id}`];
      if (!d) return placeholder(layer, c, "pipeline");
      const n = m.atoms.length;
      const focus = d.focus;
      // bonds from the focus atom to every atom
      const nb = G.neighbors(m);
      const hops = Array(n).fill(Infinity);
      hops[focus] = 0;
      for (let todo = [focus]; todo.length; ) {
        const k = todo.shift();
        for (const j of nb[k])
          if (hops[j] === Infinity) {
            hops[j] = hops[k] + 1;
            todo.push(j);
          }
      }
      const name = G.atomName(m, focus).toLowerCase();
      // the molecule with the receptive field of `l` layers: an amber glow over its atoms (as the
      // receptive-field figure of the page), the focus atom filled
      const field = (l) => {
        const art = molecule(m, { room: 84, cap: 16, dim: new Set(range(n).filter((k) => hops[k] <= l)), label: `${m.name}: the receptive field of the ${name} atom after ${l} layers` });
        const { s, pos } = art.geom;
        const glow = svgEl("g", { class: "gsqf-pl-glow" });
        for (let k = 0; k < n; k++) if (hops[k] <= l) glow.append(svgEl("circle", { cx: pos[k][0], cy: pos[k][1], r: (0.56 * s).toFixed(2) }));
        for (const [i, j] of m.bonds)
          if (hops[i] <= l && hops[j] <= l) glow.append(svgEl("line", { x1: pos[i][0], y1: pos[i][1], x2: pos[j][0], y2: pos[j][1], "stroke-width": (0.7 * s).toFixed(2) }));
        art.layers.truth.append(glow);
        // the bonds outside the field step back with their atoms (skeleton draws one line per bond
        // order, in the order of the bonds)
        const lines = [...art.layers.bonds.children];
        let at = 0;
        for (const [i, j, order] of m.bonds) {
          const count = order >= 3 ? 3 : order === 2 ? 2 : 1;
          if (!(hops[i] <= l && hops[j] <= l)) for (let t = 0; t < count; t++) lines[at + t]?.classList.add("gsqf-faded");
          at += count;
        }
        [...art.layers.atoms.children].forEach((atom, k) => {
          if (k === focus) atom.classList.add("gsqf-pl-source");
          else if (hops[k] <= l) atom.classList.add("gsqf-pl-in");
        });
        return art;
      };
      // a vector as a strip of cells, the darker the larger the value's magnitude
      const strip = (values) => {
        const top = largest(values);
        const node = svgEl("svg", { class: "gsqf-pl-strip", viewBox: `0 0 ${values.length} 1`, preserveAspectRatio: "none", "aria-hidden": "true" });
        values.forEach((v, k) => node.append(svgEl("rect", { x: k + 0.06, y: 0, width: 0.88, height: 1, "fill-opacity": (0.06 + 0.94 * (Math.abs(v) / top)).toFixed(3) })));
        return node;
      };
      // h_i^(l) in the notation of the text
      const symbol = (letter, l, count) => {
        const node = el("p", "gsqf-pl-symbol");
        add(node, el("i", "", letter), el("sub", "", "i"));
        if (l) node.append(el("sup", "", `(${l})`));
        node.append(document.createTextNode(` · ${count} values`));
        return node;
      };
      const steps = [
        panel("Input", field(0), strip(d.x), symbol("x", 0, d.x.length)),
        ...d.h.map((h, l) => panel(`Layer ${l + 1}`, field(l + 1), strip(h), symbol("h", l + 1, h.length))),
      ];
      // the readout: each atom's term of the log-odds as a tint, the focus atom framed
      const terms = Object.fromEntries(d.terms.map((v, k) => [k, v]));
      const art = molecule(m, { tints: terms, framed: new Set([focus]), label: `${m.name}: the term of each atom in the log-odds` });
      const total = sum(d.terms);
      const line = (label, value, className = "") => add(el("p", className), add(el("span"), ...(typeof label === "string" ? [label] : label)), el("strong", "", value));
      const pct = (p) => (p > 0.999 ? ">99.9%" : p < 0.001 ? "<0.1%" : `${(100 * p).toFixed(1)}%`);
      const sums = add(
        el("div", "gsqf-pl-sum"),
        line([el("i", "gsqf-pl-key"), `term of the ${name} atom`], signed(d.terms[focus])),
        line(`sum of all ${n} atom terms`, signed(total)),
        line("bias of the output layer", signed(d.bias)),
        line(`log-odds of “${task.yes}”`, signed(d.logOdds), `gsqf-pl-total ${d.logOdds >= 0 ? "gsqf-up" : "gsqf-down"}`),
        line("probability", pct(d.p)),
      );
      layer.append(
        add(el("div", "gsqf-pl-steps"), ...steps),
        add(
          el("div", "gsqf-pl-readout"),
          el("p", "gsqf-title", "Sum pooling and linear output layer"),
          add(
            el("div", "gsqf-pl-out"),
            add(el("div", "gsqf-pl-art"), art),
            add(el("div", "gsqf-pl-right"), sums, legend([["gsqf-red gsqf-round", "term raises the log-odds"], ["gsqf-blue gsqf-round", "lowers them"]])),
          ),
        ),
      );
    },
  };
  // ===== end: pipeline =====

  // ===== figure: nitro =====
  // An interaction measured on the model: the pair of atoms with the strongest 2-SII, added alone and
  // together, first to the molecule with all other atoms masked (the discrete derivative there is
  // the pair's Möbius interaction), then to random maskings of the other atoms (its average estimates
  // the pair's 2-SII). Bars as the FIxLIP page's `interaction`. Data: compute/nitro.py.
  FIGURES.nitro = {
    draw(layer, c) {
      const { molecule: m, task } = c;
      const d = c.part?.[`${m.id}/${task.id}`];
      if (!d) return placeholder(layer, c, "nitro");
      const [i, j] = d.pair;
      // the molecule with the two atoms framed, and i and j beside them, away from their bonds
      const art = molecule(m, { framed: new Set([i, j]), label: `${m.name}: the two atoms of the pair` });
      const { s, pos } = art.geom;
      // the place around the atom farthest from every atom, bond and the other name
      const taken = [...pos, ...m.bonds.map(([a, b]) => [(pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2])];
      for (const [k, letter] of [[i, "i"], [j, "j"]]) {
        const [x, y] = pos[k];
        let best = null;
        for (let step = 0; step < 16; step++) {
          const angle = (step / 16) * 2 * Math.PI - Math.PI / 2; // from above, clockwise
          const p = [x + 0.68 * s * Math.cos(angle), y + 0.68 * s * Math.sin(angle)];
          const room = Math.min(...taken.map((q) => Math.hypot(p[0] - q[0], p[1] - q[1])));
          if (!best || room > best.room + 1e-6) best = { p, room };
        }
        taken.push(best.p);
        art.layers.over.append(svgEl("text", { class: "gsqf-nt-name", x: best.p[0].toFixed(2), y: best.p[1].toFixed(2), "font-size": (0.46 * s).toFixed(2) }, letter));
      }
      const { alone, random } = d;
      const groups = [
        ["All other atoms masked", alone, ["MI of the pair", d.moebius]],
        [`Random maskings of the other atoms (mean of ${d.samples})`, random, ["2-SII of the pair", d.sii]],
      ];
      const all = groups.flatMap(([, g, [, v]]) => [g.i, g.j, g.both, g.interaction, v]);
      const step = stepOf(Math.min(0, ...all), Math.max(0, ...all));
      const low = Math.floor(Math.min(0, ...all) / step - 1e-9) * step;
      const high = Math.ceil(Math.max(0, ...all) / step - 1e-9) * step;
      const ticks = ticksOf(low, high, step);
      const at = (v) => ((v - low) / (high - low)) * 100;
      const bars = el("div", "gsqf-nt-bars");
      const bar = (label, value, kind) => {
        const track = el("span", "gsqf-nt-track");
        const tone = kind === "sign" ? (value >= 0 ? "up" : "down") : kind;
        const fill = el("span", `gsqf-nt-fill gsqf-nt-${tone}`);
        const [a, b] = [at(Math.min(0, value)), at(Math.max(0, value))];
        fill.style.left = `${a}%`;
        fill.style.width = `${Math.max(b - a, 0.6)}%`;
        const origin = el("span", "gsqf-nt-origin");
        origin.style.left = `${at(0)}%`;
        const number = el("span", `gsqf-nt-number${kind === "sign" ? (value >= 0 ? " gsqf-up" : " gsqf-down") : ""}`, signed(value));
        add(bars, add(el("span", "gsqf-nt-label"), ...rich(label)), add(track, fill, origin), number);
      };
      for (const [title, g, [name, value]] of groups) {
        bars.append(el("p", "gsqf-nt-group", title));
        bar("Atom *i* added", g.i, "gray");
        bar("Atom *j* added", g.j, "gray");
        bar("Both added", g.both, "ink");
        bar("Interaction", g.interaction, "sign");
        bar(name, value, "amber");
      }
      // the axis under the bars: ticks of the log-odds
      const axis = el("span", "gsqf-nt-axis");
      // the shortest form of a tick (0.5, not 0.50)
      const short = (t) => String(Math.round(t * 100) / 100).replace("-", "−");
      const labels = ticks.map((t) => {
        const tick = el("span", "", short(t));
        tick.style.left = `${at(t)}%`;
        axis.append(tick);
        return tick;
      });
      add(bars, el("span"), axis, el("span"), el("span"), el("span", "gsqf-nt-unit", "change of the log-odds"), el("span"));
      layer.append(
        add(
          el("div", "gsqf-nt"),
          panel(`Atoms *i* (${m.atoms[i][0]}) and *j* (${m.atoms[j][0]})`, add(el("div", "gsqf-nt-art"), art)),
          bars,
        ),
      );
      // on a narrow track, every other tick keeps its label (zero always)
      const room = axis.getBoundingClientRect().width / Math.max(1, ticks.length - 1);
      const widest = Math.max(...labels.map((label) => label.getBoundingClientRect().width));
      if (room && room < widest + 10) {
        const zero = ticks.findIndex((t) => Math.abs(t) < 1e-9);
        labels.forEach((label, k) => (k - zero) % 2 && (label.style.visibility = "hidden"));
      }
    },
  };
  // ===== end: nitro =====

  // ===== figure: maskings =====
  // Four maskings of the graph game: all atoms, no atoms, a substructure masked (amber frames), a
  // random masking, each with the GNN's probability and log-odds. Masked atoms are gray: they take
  // the baseline. Data: compute/maskings.py (the substructure of every molecule is chosen there).
  FIGURES.maskings = {
    draw(layer, c) {
      const { molecule: m, task } = c;
      const rows = c.part?.[`${m.id}/${task.id}`];
      if (!rows) return placeholder(layer, c, "maskings");
      const pct = (p) => (p > 0.999 ? ">99.9%" : p < 0.001 ? "<0.1%" : `${(100 * p).toFixed(1)}%`);
      const panels = rows.map((row, k) => {
        const masked = new Set(row.masked);
        const art = molecule(m, { room: 84, cap: 16, masked, framed: k === 2 ? masked : null, label: `${m.name}: ${row.title.toLowerCase()}` });
        return panel(
          row.title,
          add(el("div", "gsqf-ms-art"), art),
          add(
            el("div", "gsqf-ms-readings"),
            reading("probability", pct(row.p)),
            reading("log-odds", signed(row.logOdds), row.logOdds >= 0 ? "gsqf-up" : "gsqf-down"),
          ),
        );
      });
      layer.append(columns("gsqf-four", ...panels));
    },
  };
  // ===== end: maskings =====

  // ===== figure: order =====
  // The SI-Graph of the molecule at the explanation order the slider sets (k = 1 to 4, and n for the
  // Möbius interactions), drawn as the explorer draws it: every atom's own value as a halo, the
  // strongest groups as hyperedges (data.js keeps only those), one scale per order. Beside it, a
  // table of all five orders from the complete explanations (../compute/order.py): how many
  // non-zero contributions each has, and their sum, which is the same for every order (efficiency).
  // The slider changes the drawing and the column in place (no redraw: the drag goes on).
  {
    const ORDERS = [1, 2, 3, 4, "n"];
    const NAMES = { 1: "SV", 2: "2-SII", 3: "3-SII", 4: "4-SII", n: "MI" };
    const TITLES = { 1: "Shapley values, *k* = 1", 2: "2-SII, *k* = 2", 3: "3-SII, *k* = 3", 4: "4-SII, *k* = 4", n: "Möbius interactions, *k* = *n*" };
    const rowsKey = (k) => (k === "n" ? "M" : k);
    // the SI-Graph of an order, as the explorer chooses what it shows (demo.js, the scene's update):
    // the strongest groups (7, or 10 at order 2, and every other one of 8% of the strongest), and one
    // scale for the order, the largest value of all
    const siGraph = (m, ex, k) => {
      const all = ex.groups(rowsKey(k));
      const top = Math.abs(all[0]?.v ?? 0);
      const limit = k === 2 ? 10 : 7;
      let shown = k === 1 ? [] : all.filter((g, i) => i < 7 || Math.abs(g.v) >= 0.08 * top);
      let count = Math.min(limit, shown.length);
      while (count > 0 && count < shown.length && Math.abs(Math.abs(shown[count].v) - Math.abs(shown[count - 1].v)) < 1e-4) count++;
      shown = shown.slice(0, count);
      const scale = largest(ex.rows(rowsKey(k)).map(([, v]) => v));
      return molecule(m, { halos: ex.singles(rowsKey(k)), groups: shown, largest: scale, label: `${m.name}: SI-Graph of the ${NAMES[k]}` });
    };
    // a part replaced by the next one in the same place: the old fades out, the new one in
    const swap = (holder, next, fade) => {
      for (const old of [...holder.children]) {
        if (!fade || reducedMotion() || old.classList.contains("gsqf-out")) {
          old.remove();
          continue;
        }
        old.classList.add("gsqf-out");
        old.setAttribute("aria-hidden", "true");
        setTimeout(() => old.remove(), 140);
      }
      if (fade && !reducedMotion()) next.classList.add("gsqf-in");
      holder.append(next);
    };
    FIGURES.order = {
      draw(layer, c, state) {
        const { molecule: m, ex } = c;
        const part = c.part?.[`${m.id}/${c.task.id}`];
        if (!part) return placeholder(layer, c, "order");
        const order = () => (ORDERS.includes(state.order) ? state.order : 2);
        const title = el("div", "gsqf-order-title");
        const art = el("div", "gsqf-order-art");
        const table = el("div", "gsqf-order-table");
        table.setAttribute("role", "group");
        table.setAttribute("aria-label", "Contributions of the SI-Graph by explanation order");
        const cells = {}; // the cells of an order's column
        const cell = (k, className, text) => {
          const node = el("span", `gsqf-order-cell ${className}`.trim(), text);
          (cells[k] ??= []).push(node);
          return node;
        };
        const control = slider("explanation order", [0, 4], order(), (k) => {
          state.order = k;
          show(true);
        }, { values: ORDERS });
        control.classList.add("gsqf-order-slider");
        // the rows: the slider, the order (a click on a column sets it), the index, the
        // contributions, their sum
        add(table, el("span", "gsqf-order-label"), control);
        add(table, add(el("span", "gsqf-order-label"), "order ", el("em", "", "k")));
        for (const k of ORDERS) {
          const head = cell(k, "gsqf-order-k", String(k));
          if (k === "n") head.replaceChildren(el("em", "", "n"));
          head.addEventListener("click", () => {
            state.order = k;
            control.set(k);
            show(true);
          });
          table.append(head);
        }
        add(table, el("span", "gsqf-order-label", "index"), ...ORDERS.map((k) => cell(k, "gsqf-order-name", NAMES[k])));
        add(table, el("span", "gsqf-order-label", "contributions"), ...ORDERS.map((k) => cell(k, "", thousands(part.orders[k].count))));
        add(
          table,
          el("span", "gsqf-order-label", "their sum"),
          ...ORDERS.map((k) => cell(k, part.orders[k].sum >= 0 ? "gsqf-up" : "gsqf-down", signed(part.orders[k].sum))),
        );
        const check = el("p", "gsqf-order-check");
        add(
          check,
          el("span", "", "log-odds "),
          // four decimals (as the text: 3.9826 − 0.1774 = 3.8052), the difference rounded from the
          // unrounded values (../compute/order.py)
          el("strong", "", signed(part.logOdds, 4)),
          el("span", "", " − no atoms "),
          el("strong", "", signed(part.empty, 4)),
          el("span", "", " = "),
          el("strong", part.difference >= 0 ? "gsqf-up" : "gsqf-down", signed(part.difference, 4)),
        );
        layer.append(add(el("div", "gsqf-order"), add(el("div", "gsqf-order-left"), title, art), add(el("div", "gsqf-order-right"), table, check)));
        const show = (fade) => {
          const k = order();
          swap(title, add(el("p", "gsqf-title"), ...rich(TITLES[k])), fade);
          swap(art, siGraph(m, ex, k), fade);
          for (const [key, nodes] of Object.entries(cells)) for (const node of nodes) node.classList.toggle("gsqf-order-on", key === String(k));
        };
        show(false);
      },
    };
  }
  // ===== end: order =====

  // ===== figure: arrangement =====
  // Catechol, resorcinol and hydroquinone (../compute/arrangement.py: from their SMILES, turned so
  // that the ring lies in the same place, the first OH at the upper left): the PAINS GNN's
  // probability for each, and its 2-SII as an SI-Graph, the three on one scale so that their
  // strengths compare. The groups shown are those the explorer would show, and always the pair of
  // the two oxygen atoms, whose value the panel reads.
  FIGURES.arrangement = {
    fixed: true,
    draw(layer, c) {
      const part = c.part;
      if (!part) return placeholder(layer, { ...c, molecule: moleculeOf("dopamine"), task: taskOf("PAINS"), ex: explanationOf("dopamine", "PAINS") }, "arrangement");
      const task = taskOf(part.task);
      const all = part.molecules.flatMap((m) => m.kSII2.map(([, v]) => v));
      const scale = largest(all);
      const panels = part.molecules.map((m) => {
        const terms = m.kSII2.map(([atoms, v]) => ({ atoms, v }));
        const halos = Object.fromEntries(terms.filter((t) => t.atoms.length === 1).map((t) => [t.atoms[0], t.v]));
        const pairs = terms.filter((t) => t.atoms.length === 2).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
        const oo = pairs.find((t) => t.atoms.join() === m.oxygens.join());
        // as the explorer at order 2: the 7 strongest, and others of 8% of the strongest, up to 10
        const top = Math.abs(pairs[0].v);
        const shown = pairs.filter((t, i) => i < 7 || Math.abs(t.v) >= 0.08 * top).slice(0, 10);
        if (!shown.includes(oo)) shown.push(oo);
        // the pair the panel reads stands out, the others step back
        const groups = shown.map((t) => ({ ...t, emph: t === oo ? 1 : 0.4 }));
        const art = molecule(m, { halos, groups, largest: scale, framed: new Set(m.oxygens), label: `${m.name}: SI-Graph of the 2-SII for ${task.label}` });
        // the same square around the ring in all three (the bonds have one length, ShapiqGraph.geom's cap)
        const { s, pos } = art.geom;
        const ring = m.rings[0];
        const cx = mean(ring.map((k) => pos[k][0]));
        const cy = mean(ring.map((k) => pos[k][1]));
        art.setAttribute("viewBox", [cx - 2.3 * s, cy - 2.55 * s, 4.6 * s, 4.1 * s].map((v) => v.toFixed(2)).join(" "));
        const head = add(el("p", "gsqf-title"), m.name, el("span", "gsqf-arr-iupac", m.iupac));
        return add(
          el("div", "gsqf-panel gsqf-arr-panel"),
          head,
          art,
          add(
            el("div", "gsqf-arr-readings"),
            reading(`${task.label}?`, `${(100 * m.p).toFixed(1)}%`),
            reading("O–O", signed(oo.v), oo.v >= 0 ? "gsqf-up" : "gsqf-down"),
          ),
        );
      });
      layer.append(add(el("div", "gsqf-arr"), ...panels));
    },
  };
  // ===== end: arrangement =====

  // ===== figure: sparsity =====
  // All 2^n Möbius interactions of the molecule for the question shown, by brute force over all
  // 2^n maskings (../compute/sparsity.py, the GNN in float64), by the size of their set against
  // their absolute value. With up to 2^20 sets, each size is a column of bins of a quarter of a
  // decade; a bin's width grows with the logarithm of its number of sets, amber for the sets
  // inside I (Proposition 3.6), gray for those outside. The dashed line is the rounding of float32:
  // the largest difference between an MI in float32 (as the demo computes the GNN) and in float64.
  {
    const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";
    const sup = (k) => String(k).replace(/\d/g, (d) => SUP[d]);
    // a small positive number as "6 × 10⁻¹³", a larger one with two decimals
    const tiny = (v) => {
      if (v >= 0.01) return fmt(v, 2);
      const p = Math.floor(Math.log10(v));
      return `${Math.round(v / 10 ** p)} × 10⁻${sup(-p)}`;
    };
    const LOW = -16; // values below 10^LOW (and exact zeros) lie in the lowest bin
    FIGURES.sparsity = {
      draw(layer, c) {
        const d = c.part?.cases?.[`${c.molecule.id}/${c.task.id}`];
        if (!d) return placeholder(layer, c, "sparsity");
        const { bins } = c.part;
        const f = d.f64;
        const total = d.inside + d.outside;
        const legendNode = legend([
          ["gsqf-amber", "inside 𝓘"],
          ["gsqf-gray", "outside 𝓘"],
          ["gsqf-dashed", "rounding of float32"],
        ]);
        const holder = el("div", "gsqf-plot");
        const readings = add(
          el("div", "gsqf-sp-readings"),
          reading(`sets outside 𝓘, of ${thousands(total)}`, d.outside ? thousands(d.outside) : "none"),
          reading("largest |MI| outside 𝓘", d.outside ? tiny(f.max_out) : "–"),
          reading("largest |MI| inside 𝓘", fmt(f.max_in, 2)),
        );
        layer.append(add(el("div", "gsqf-sp"), legendNode, holder, readings));
        const plot = chart(holder, 250, { left: 64, right: 14, top: 10, bottom: 36 });
        const { draw, box } = plot;
        plot.inset(legendNode);
        plot.svg.setAttribute("role", "img");
        plot.svg.setAttribute(
          "aria-label",
          `Absolute values of all ${thousands(total)} Möbius interactions of ${c.molecule.name} by the size of their set: ${thousands(d.inside)} inside I, ${thousands(d.outside)} outside`,
        );
        const x = plot.linear([-0.6, d.n + 0.6], [box.x0, box.x1]);
        const y = plot.log([10 ** (LOW - 0.5), 10 ** 1.5], [box.y1, box.y0]);
        const ticks = [1e-16, 1e-12, 1e-8, 1e-4, 1];
        plot.yAxis(y, ticks, "|Möbius interaction|", (t) => (t === 1e-16 ? ["≤10", "−16"] : power(t)));
        const step = d.n > 14 ? 4 : 2;
        plot.xAxis(x, range(d.n + 1).filter((k) => k % step === 0), "size of the set |S|", String);
        // the columns: a bin's half-width grows with log(1 + its sets), on one scale for the figure
        const all = [...f.in, ...f.out];
        const top = Math.log10(1 + Math.max(1, ...all.map(([, , count]) => count)));
        const half = Math.min(14, (0.44 * (box.x1 - box.x0)) / (d.n + 1.2));
        const bin = (b) => Math.max(LOW * bins, b); // the zeros and the smallest values: the lowest bin
        const pixels = y(10 ** (LOW + 1 / bins)) - y(10 ** LOW); // the height of a bin (negative)
        for (const [side, kind] of [
          ["out", "gsqf-gray"],
          ["in", "gsqf-amber"],
        ]) {
          // the counts of every size and bin (the exact zeros are in the lowest bin)
          const counts = new Map();
          for (const [size, b, count] of f[side]) {
            const at = `${size} ${bin(b)}`;
            counts.set(at, (counts.get(at) ?? 0) + count);
          }
          // one path for all bins of a side, so that no seam shows between two bins
          const bars = [];
          for (const [at, count] of counts) {
            const [size, b] = at.split(" ").map(Number);
            const w = Math.max(1, (half * Math.log10(1 + count)) / top);
            const y0 = y(10 ** (b / bins));
            bars.push(`M${(x(size) - w).toFixed(2)} ${y0.toFixed(2)}h${(2 * w).toFixed(2)}V${(y0 + pixels).toFixed(2)}h${(-2 * w).toFixed(2)}Z`);
          }
          if (bars.length) draw("path", { class: `${kind} gsqf-sp-bins`, d: bars.join("") });
        }
        // a molecule whose receptive fields cover it: nothing lies outside I
        if (!d.outside)
          draw("text", { class: "gsqf-sp-all", x: (box.x0 + box.x1) / 2, y: y(1e-12), "text-anchor": "middle" }, `All ${thousands(total)} sets lie inside 𝓘.`);
        // the rounding of float32
        const noise = y(d.noise);
        draw("line", { class: "gsqf-mark", x1: box.x0, x2: box.x1, y1: noise, y2: noise });
        draw("text", { class: "gsqf-sp-note", x: box.x1 - 2, y: noise - 5, "text-anchor": "end" }, `float32: ${tiny(d.noise)}`);
      },
    };
  }
  // ===== end: sparsity =====

  // ===== figure: complexity =====
  // The model calls of exact GraphSHAP-IQ, |I|, against the number of atoms, for the explorer's
  // seven molecules (filled) and the four of the receptive-field figure (rings), for a GNN with the
  // layers of the slider (../compute/complexity.py); the dashed line is all 2^n maskings. The
  // molecule shown is amber and named; the dots slide when the layers change.
  {
    const share = (part) => {
      if (part >= 0.9995) return "100%";
      const digits = part >= 0.1 ? 0 : part >= 0.01 ? 1 : Math.max(1, 1 - Math.floor(Math.log10(100 * part)));
      return `${(100 * part).toFixed(digits)}%`;
    };
    FIGURES.complexity = {
      draw(layer, c, state) {
        const part = c.part;
        if (!part) return placeholder(layer, c, "complexity");
        state.layers ??= 3; // the demo's GNNs
        const k = state.layers - 1;
        const rows = part.molecules;
        const shown = rows.find((row) => row.id === c.molecule.id);
        const legendNode = legend([
          ["gsqf-round gsqf-cx-filled", "demo molecules"],
          ["gsqf-round gsqf-cx-ring", "receptive-field figure"],
          ["gsqf-dashed", "all 2ⁿ maskings"],
        ]);
        const holder = el("div", "gsqf-plot");
        const control = slider("layers", [1, 4], state.layers, (value) => {
          state.layers = value;
          c.redraw();
        });
        const calls = shown.budgets[k];
        const readings = add(
          el("div", "gsqf-cx-readings"),
          reading("model calls of GraphSHAP-IQ", thousands(calls)),
          reading("all maskings, 2ⁿ", thousands(2 ** shown.n)),
          reading("share", share(calls / 2 ** shown.n)),
        );
        layer.append(add(el("div", "gsqf-cx"), legendNode, holder, control, readings));
        const plot = chart(holder, 250, { left: 52, right: 14, top: 10, bottom: 36 });
        const { draw, box } = plot;
        plot.inset(legendNode, control);
        plot.svg.setAttribute("role", "img");
        plot.svg.setAttribute("aria-label", `Model calls of GraphSHAP-IQ for ${state.layers} layers against the number of atoms, for eleven molecules, and all 2 to the n maskings`);
        const x = plot.linear([4, 30], [box.x0, box.x1]);
        const y = plot.log([10, 1e9], [box.y1, box.y0]);
        plot.yAxis(y, [10, 1e3, 1e5, 1e7, 1e9], "model calls", power);
        plot.xAxis(x, [5, 10, 15, 20, 25, 30], "atoms n", String);
        const full = range(27).map((j) => [x(4 + j), y(2 ** (4 + j))]);
        plot.path(full, { class: "gsqf-mark" });
        draw("text", { class: "gsqf-cx-note", x: x(29.4) - 4, y: y(2 ** 29.4) + 14, "text-anchor": "end" }, "2ⁿ");
        // the molecule shown: a line from its dot to all 2^n maskings
        draw("line", { class: "gsqf-cx-gap", x1: x(shown.n), x2: x(shown.n), y1: y(2 ** shown.n), y2: y(calls) });
        // in one order for every molecule and every layer, so that each dot slides to its new place
        for (const row of rows) {
          const on = row === shown;
          const kind = on ? "gsqf-cx-shown" : row.explorer ? "gsqf-cx-filled" : "gsqf-cx-ring";
          const dot = draw("circle", { class: `gsqf-dot gsqf-cx-dot ${kind}`, cx: x(row.n).toFixed(1), cy: y(row.budgets[k]).toFixed(1), r: on ? 5.5 : 4 });
          dot.append(svgEl("title", {}, `${row.name}: ${thousands(row.budgets[k])} of ${thousands(2 ** row.n)} maskings (${row.n} atoms)`));
        }
        // names: the molecule shown, and cholesterol (the largest, which the text names). A name
        // goes right of its dot, left, above or below: where it covers the fewest other dots
        const points = rows.map((row) => [x(row.n), y(row.budgets[k])]);
        const taken = [];
        for (const row of [shown, rows.find((r) => r.id === "cholesterol")]) {
          if (!row || (row !== shown && row.id === shown.id)) continue;
          const name = moleculeOf(row.id)?.short ?? row.name;
          const [px, py] = [x(row.n), y(row.budgets[k])];
          const wide = name.length * 6.9 * plot.type;
          const high = 13 * plot.type;
          const places = [
            [px + 10, py + 4, "start", px + 8, px + 12 + wide],
            [px - 10, py + 4, "end", px - 12 - wide, px - 8],
            [px, py - 13, "middle", px - wide / 2, px + wide / 2],
            [px, py + 20 * plot.type, "middle", px - wide / 2, px + wide / 2],
          ];
          const cost = ([, ty, , x0, x1]) => {
            const [y0, y1] = [ty - high + 2, ty + 3];
            const near = (qx, qy, r) => qx > x0 - r && qx < x1 + r && qy > y0 - r && qy < y1 + r;
            const covered = points.filter(([qx, qy]) => !(qx === px && qy === py) && near(qx, qy, 4)).length;
            const clash = taken.some(([a0, a1, b0, b1]) => x0 < a1 && x1 > a0 && y0 < b1 && y1 > b0);
            const out = x0 < box.x0 || x1 > box.x1 || y0 < box.y0 || y1 > box.y1;
            return covered + 10 * clash + 10 * out;
          };
          const best = places.reduce((a, b) => (cost(b) < cost(a) ? b : a));
          taken.push([best[3], best[4], best[1] - high + 2, best[1] + 3]);
          draw("text", { class: row === shown ? "gsqf-strong" : "", x: best[0], y: best[1], "text-anchor": best[2] }, name);
        }
      },
    };
  }
  // ===== end: complexity =====

  // ===== figure: approximation =====
  // The MSE of 2-SII estimates against GraphSHAP-IQ's exact values, against the model calls, on the
  // molecule's graph game (../compute/approximation.py): GraphSHAP-IQ's approximation variant with
  // lambda = 1, 2, ... (one budget each, amber, exact at the last) and four approximators of shapiq
  // at the same budgets (cool colors; the mean of three seeds, with their range as a band). An MSE
  // of zero, or below 10^-10, lies on the lane "exact" under the axis. The pointer (or a finger)
  // shows the values at a budget.
  {
    const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹"; // for 10⁻⁵
    const sup = (k) => String(k).replace(/\d/g, (d) => SUP[d]);
    const METHODS = ["GraphSHAP-IQ", "KernelSHAP-IQ", "SVARM-IQ", "SHAP-IQ", "Permutation Sampling"];
    const COLORS = { "GraphSHAP-IQ": "gsqf-ap-m0", "KernelSHAP-IQ": "gsqf-ap-m1", "SVARM-IQ": "gsqf-ap-m3", "SHAP-IQ": "gsqf-ap-m4", "Permutation Sampling": "gsqf-ap-m5" };
    const EXACT = 1e-10; // an MSE below this is exact up to rounding
    // an MSE as the text writes it: 113, 0.42, 0.0022, else 3.2 × 10⁻⁵
    const mse = (v) => {
      if (v < EXACT) return "exact";
      if (v >= 10) return thousands(v);
      if (v >= 1e-4) return v.toPrecision(2).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
      const [m, e] = v.toExponential(1).split("e");
      return `${m} × 10⁻${sup(-Number(e))}`;
    };
    const share = (part) => `${(100 * part).toFixed(part >= 0.1 ? 0 : 1)}%`;
    // the case of the molecule and the question shown, else the molecule's own question
    const caseOf = (c) => c.part?.cases?.[`${c.molecule.id}/${c.task.id}`] ?? c.part?.cases?.[`${c.molecule.id}/${c.own.id}`];
    FIGURES.approximation = {
      draw(layer, c) {
        const d = caseOf(c);
        if (!d) return placeholder(layer, c, "approximation");
        const series = METHODS.map((method) => {
          if (method === "GraphSHAP-IQ") return { method, mean: d.graphshapiq, low: d.graphshapiq, high: d.graphshapiq };
          const runs = d[method];
          return { method, mean: runs.map(mean), low: runs.map((r) => Math.min(...r)), high: runs.map((r) => Math.max(...r)) };
        });
        const last = d.budgets.length - 1;
        const legendNode = legend(METHODS.map((method) => [`gsqf-ap-swatch ${COLORS[method]}`, method]));
        const holder = el("div", "gsqf-plot gsqf-ap-holder");
        const tip = el("div", "gsqf-ap-tip");
        tip.hidden = true;
        // the best approximator at the exact budget
        const [best] = series.slice(1).sort((a, b) => a.mean[last] - b.mean[last]);
        const trivial = d.exact >= d.full;
        const readings = add(
          el("div", "gsqf-ap-readings"),
          reading("GraphSHAP-IQ exact at", `${thousands(d.exact)} calls`),
          reading("share of all 2ⁿ maskings", trivial ? "100%" : share(d.exact / d.full)),
          trivial
            ? reading("approximators exact there too", `${series.slice(1).filter((s) => s.mean[last] < EXACT).length} of ${series.length - 1}`)
            : reading("best approximator there", `${best.method}, ${mse(best.mean[last])}`),
        );
        layer.append(add(el("div", "gsqf-ap"), legendNode, holder, readings));
        holder.append(tip);
        const plot = chart(holder, 260, { left: 52, right: 14, top: 10, bottom: 36 });
        const { draw, box } = plot;
        plot.inset(legendNode);
        plot.svg.setAttribute("role", "img");
        plot.svg.setAttribute("aria-label", `MSE of 2-SII estimates against model calls for ${c.molecule.name}: GraphSHAP-IQ is exact with ${thousands(d.exact)} model calls`);
        const lane = box.y1 - 7; // the lane "exact"
        const x = plot.log([d.budgets[0] / 1.35, d.budgets[last] * 1.35], [box.x0, box.x1]);
        const yLog = plot.log([1e-6, 1e3], [lane - 22, box.y0]);
        const y = (v) => (v < EXACT ? lane : yLog(Math.max(v, 1e-6)));
        plot.yAxis(yLog, [1e-6, 1e-4, 1e-2, 1, 1e2], "MSE of 2-SII", power);
        draw("text", { x: box.x0 - 6, y: lane + 4, "text-anchor": "end" }, "exact");
        // ticks of the model calls: the powers of ten, with 2 and 5 times them where there are few
        const lowest = d.budgets[0] / 1.35;
        const highest = d.budgets[last] * 1.35;
        let ticks = logTicks(lowest, highest);
        if (ticks.length < 4) ticks = ticks.flatMap((t) => [t, 3 * t]).concat([ticks[0] / 10 * 3]).filter((t) => t >= lowest && t <= highest).sort((a, b) => a - b);
        plot.xAxis(x, ticks, "model calls", thousands);
        // the bands, then the lines of the approximators, then GraphSHAP-IQ with a dot per lambda
        // the curves stay inside the plot (KernelSHAP-IQ starts far above it with fewer calls than terms)
        const clip = `gsqf-ap-clip-${plot.width}`;
        draw("defs", {}).append(add(svgEl("clipPath", { id: clip }), svgEl("rect", { x: box.x0, y: box.y0 - 2, width: box.x1 - box.x0, height: box.y1 - box.y0 + 2 })));
        const curves = draw("g", { "clip-path": `url(#${clip})` });
        const pathOf = (points) => points.map(([px, py], j) => `${j ? "L" : "M"} ${px.toFixed(1)} ${py.toFixed(1)}`).join(" ");
        for (const s of series.slice(1)) {
          const edge = [...d.budgets.map((b, j) => [x(b), y(s.low[j])]), ...d.budgets.map((b, j) => [x(b), y(s.high[j])]).reverse()];
          curves.append(svgEl("path", { class: `gsqf-ap-band ${COLORS[s.method]}`, d: `${pathOf(edge)} Z` }));
        }
        for (const s of [...series.slice(1), series[0]]) {
          const line = svgEl("path", { class: `gsqf-line gsqf-ap-line ${COLORS[s.method]}`, d: pathOf(d.budgets.map((b, j) => [x(b), y(s.mean[j])])) });
          line.append(svgEl("title", {}, s.method));
          curves.append(line);
        }
        for (const [j, b] of d.budgets.entries()) draw("circle", { class: `gsqf-ap-point ${COLORS["GraphSHAP-IQ"]}`, cx: x(b).toFixed(1), cy: y(d.graphshapiq[j]).toFixed(1), r: j === last ? 4.5 : 3 });
        // the values at a budget, where the pointer is
        const guide = draw("line", { class: "gsqf-guide", y1: box.y0, y2: box.y1, visibility: "hidden" });
        const point = (event) => {
          const rect = plot.svg.getBoundingClientRect();
          const px = ((event.clientX - rect.left) / rect.width) * plot.width;
          const j = d.budgets.reduce((at, b, k) => (Math.abs(x(b) - px) < Math.abs(x(d.budgets[at]) - px) ? k : at), 0);
          const bx = x(d.budgets[j]);
          guide.setAttribute("x1", bx);
          guide.setAttribute("x2", bx);
          guide.setAttribute("visibility", "visible");
          const head = add(el("tr"), el("th", "", `λ = ${d.lambdas[j]}, ${thousands(d.budgets[j])} calls`), el("td", "", "MSE"));
          const rows = series.map((s) => add(el("tr"), add(el("th"), el("i", `gsqf-swatch gsqf-ap-swatch ${COLORS[s.method]}`), s.method), el("td", "", mse(s.mean[j]))));
          tip.replaceChildren(add(el("table"), add(el("thead"), head), add(el("tbody"), ...rows)));
          tip.hidden = false;
          const wide = tip.offsetWidth;
          const at = bx < (box.x0 + box.x1) / 2 ? bx + 10 : bx - 10 - wide;
          tip.style.left = `${Math.max(0, Math.min(plot.width - wide, at))}px`;
        };
        plot.svg.addEventListener("pointermove", point);
        plot.svg.addEventListener("pointerdown", point);
        plot.svg.addEventListener("pointerleave", (event) => {
          if (event.pointerType === "touch") return; // a finger that lifts leaves the values
          guide.setAttribute("visibility", "hidden");
          tip.hidden = true;
        });
      },
    };
  }
  // ===== end: approximation =====

  // ===== figure: paper-approximation =====
  // Figure 4 (left) of the paper: the MSE of 2-SII estimates for a 2-layer GIN on 10 molecules of
  // Mutagenicity against the model evaluations, for GraphSHAP-IQ (amber) and five model-agnostic
  // baselines (cool colors), each as a baseline (solid) and as its interaction-informed variant
  // (dashed), with the paper's bands (the standard error). The coordinates are those of the
  // paper's vector graphic (../compute/paper_approximation.py). A method of the legend that the
  // pointer is over (or that has the focus) is drawn alone.
  {
    const METHODS = ["GraphSHAP-IQ", "KernelSHAP-IQ", "Inc. KernelSHAP-IQ", "SVARM-IQ", "SHAP-IQ", "Permutation Sampling"];
    const colorOf = (method) => `gsqf-pa-m${METHODS.indexOf(method)}`;
    FIGURES["paper-approximation"] = {
      fixed: true,
      draw(layer, c) {
        const part = c.extra?.["paper-approximation"];
        if (!part) return;
        const keys = add(
          el("p", "gsqf-legend gsqf-pa-methods"),
          ...METHODS.map((method) => {
            const item = add(el("span", "gsqf-pa-key"), el("i", `gsqf-swatch gsqf-pa-swatch ${colorOf(method)}`), method);
            item.tabIndex = 0;
            item.dataset.method = method;
            return item;
          }),
        );
        const variants = legend([
          ["gsqf-solid", "baseline"],
          ["gsqf-dashed", "interaction-informed"],
        ]);
        variants.classList.add("gsqf-pa-variants");
        const holder = el("div", "gsqf-plot");
        layer.append(add(el("div", "gsqf-pa"), keys, holder, variants));
        const plot = chart(holder, 270, { left: 52, right: 14, top: 10, bottom: 36 });
        const { draw, box } = plot;
        plot.inset(keys, variants);
        plot.svg.setAttribute("role", "img");
        plot.svg.setAttribute("aria-label", "The paper's MSE of 2-SII estimates against model evaluations for GraphSHAP-IQ and five baselines, with and without their interaction-informed variants");
        const x = plot.linear([0, 8200], [box.x0, box.x1]);
        const y = plot.log([3e-11, 0.3], [box.y1, box.y0]);
        plot.yAxis(y, [1e-10, 1e-8, 1e-6, 1e-4, 1e-2], "MSE", power);
        plot.xAxis(x, [0, 2000, 4000, 6000, 8000], "model evaluations", thousands);
        // the curves stay inside the plot, as in the paper (KernelSHAP-IQ starts far above it)
        const clip = `gsqf-pa-clip-${plot.width}`;
        const defs = draw("defs", {});
        defs.append(add(svgEl("clipPath", { id: clip }), svgEl("rect", { x: box.x0, y: box.y0 - 2, width: box.x1 - box.x0, height: box.y1 - box.y0 + 2 })));
        const groups = new Map();
        const order = [...part.series].sort((a, b) => (a.method === "GraphSHAP-IQ") - (b.method === "GraphSHAP-IQ")); // GraphSHAP-IQ on top
        for (const s of order) {
          if (!groups.has(s.method)) groups.set(s.method, draw("g", { class: `gsqf-pa-series ${colorOf(s.method)}`, "clip-path": `url(#${clip})` }));
          const group = groups.get(s.method);
          if (s.band) {
            const edge = [...s.band.map(([bx, low]) => [x(bx), y(low)]), ...s.band.map(([bx, , high]) => [x(bx), y(high)]).reverse()];
            group.append(svgEl("path", { class: "gsqf-pa-band", d: `${edge.map(([px, py], j) => `${j ? "L" : "M"} ${px.toFixed(1)} ${py.toFixed(1)}`).join(" ")} Z` }));
          }
          const line = svgEl("path", {
            class: `gsqf-line gsqf-pa-line${s.variant === "interaction-informed" ? " gsqf-pa-dashed" : ""}`,
            d: s.points.map(([px, py], j) => `${j ? "L" : "M"} ${x(px).toFixed(1)} ${y(py).toFixed(1)}`).join(" "),
          });
          line.append(svgEl("title", {}, `${s.method}${s.variant === "interaction-informed" ? " (interaction-informed)" : ""}: MSE ${s.points.at(-1)[1].toExponential(1)} at ${thousands(s.points.at(-1)[0])} evaluations`));
          group.append(line);
          for (const [px, py] of s.points) group.append(svgEl("circle", { class: "gsqf-pa-point", cx: x(px).toFixed(1), cy: y(py).toFixed(1), r: 2.4 }));
        }
        // GraphSHAP-IQ named at its curve, under the knee at 10^-7, where no curve runs
        const [sx, sy] = part.series.find((s) => s.method === "GraphSHAP-IQ").points[2];
        draw("text", { class: "gsqf-strong", x: x(sx) - 6, y: y(sy) + 18, "text-anchor": "end" }, "GraphSHAP-IQ");
        // the legend points at a method
        const focus = (method) => {
          for (const [key, group] of groups) group.classList.toggle("gsqf-pa-dimmed", method !== null && key !== method);
          for (const item of keys.children) item.classList.toggle("gsqf-pa-off", method !== null && item.dataset.method !== method);
        };
        for (const item of keys.children) {
          for (const [type, method] of [
            ["pointerenter", item.dataset.method],
            ["pointerleave", null],
            ["focus", item.dataset.method],
            ["blur", null],
          ])
            item.addEventListener(type, () => focus(method));
        }
      },
    };
  }
  // ===== end: paper-approximation =====

  // ===== figure: orders =====
  // For each of the seven molecules (its own question), the sums of its positive and of its negative
  // Möbius interactions of 1, 2, 3 and 4 or more atoms (../compute/orders.py, from the complete
  // MIs), as bars to the right and to the left of zero, on a symmetric log scale (sign · log10(1 +
  // |x|)), so that sums of 3 and of 5,000 can be read on one axis. An ink line marks what all of a
  // molecule's MIs sum to: its log-odds minus the prediction with no atoms. The molecule shown is
  // in a band of amber.
  {
    const SIZES = ["1", "2", "3", "4"];
    const SIZE_LABELS = { 1: "1", 2: "2", 3: "3", 4: "≥4" };
    const symlog = (v) => Math.sign(v) * Math.log10(1 + Math.abs(v));
    // a tick of the symmetric log axis: 0, ±10, ±10², ...
    const tickLabel = (v) => {
      if (v === 0) return "0";
      const p = Math.round(Math.log10(Math.abs(v)));
      const sign = v < 0 ? "−" : "";
      return p === 1 ? `${sign}10` : [`${sign}10`, String(p)];
    };
    FIGURES.orders = {
      draw(layer, c) {
        const part = c.part;
        if (!part) return placeholder(layer, c, "orders");
        const mols = c.molecules.filter((m) => part[m.id]);
        const legendNode = legend([
          ["gsqf-red", "positive MIs"],
          ["gsqf-blue", "negative MIs"],
          ["gsqf-ords-net", "all MIs: log-odds − no atoms"],
          ["gsqf-dashed", "±100"],
        ]);
        const holder = el("div", "gsqf-plot");
        layer.append(add(el("div", "gsqf-ords"), legendNode, holder));
        // a size's row, and the room between two molecules (larger where the texts are, on a phone)
        const type = parseFloat(getComputedStyle(holder).getPropertyValue("--gsqf-type")) || 1;
        const ROW = Math.round(10 * type);
        const GAP = Math.round(9 * type);
        const group = SIZES.length * ROW;
        const plot = chart(holder, (mg) => mg.top + mols.length * group + (mols.length - 1) * GAP + mg.bottom, { left: 116, right: 52, top: 16, bottom: 36 });
        const { draw, box, width } = plot;
        plot.inset(legendNode);
        plot.svg.setAttribute("role", "img");
        plot.svg.setAttribute("aria-label", "Sums of the positive and of the negative Möbius interactions of 1, 2, 3 and 4 or more atoms, for the seven molecules");
        const narrow = box.x1 - box.x0 < 300;
        const x = plot.linear([-4.05, 4.05], [box.x0, box.x1]);
        const X = (v) => x(symlog(v));
        const ticks = [-1e4, -1e3, -1e2, -10, 0, 10, 1e2, 1e3, 1e4];
        const y0 = (k) => box.y0 + k * (group + GAP); // the top of a molecule's rows
        // the band of the molecule shown, under everything
        const shown = mols.findIndex((m) => m.id === c.molecule.id);
        if (shown >= 0) draw("rect", { class: "gsqf-ords-band", x: 0, y: y0(shown) - GAP / 2 + 1, width, height: group + GAP - 2, rx: 6 });
        for (const t of ticks) draw("line", { class: t === 0 ? "gsqf-axis" : "gsqf-grid", x1: X(t), x2: X(t), y1: box.y0 - 4, y2: box.y1 });
        for (const t of [-100, 100]) draw("line", { class: "gsqf-mark", x1: X(t), x2: X(t), y1: box.y0 - 4, y2: box.y1 });
        const labelled = narrow ? ticks.filter((t) => t === 0 || Math.abs(t) === 1e2 || Math.abs(t) === 1e4) : ticks;
        plot.xAxis(X, labelled, narrow ? "sum of the MIs (symmetric log scale)" : "sum of the Möbius interactions of each size (symmetric log scale)", tickLabel);
        // the heads of the columns at the left and the right
        draw("text", { x: box.x0 - 8, y: box.y0 - 6, "text-anchor": "end", class: "gsqf-ords-head" }, "atoms");
        draw("text", { x: width - 2, y: box.y0 - 6, "text-anchor": "end", class: "gsqf-ords-head" }, "all MIs");
        mols.forEach((m, k) => {
          const d = part[m.id];
          const at = y0(k);
          const on = k === shown;
          draw("text", { x: 4, y: at + group / 2, "dominant-baseline": "central", class: on ? "gsqf-strong" : "" }, m.short ?? m.name);
          SIZES.forEach((size, j) => {
            const { positive, negative } = d.moebius[size];
            const yy = at + j * ROW + 1.5;
            const h = ROW - 3;
            draw("text", { x: box.x0 - 8, y: yy + h / 2, "dominant-baseline": "central", "text-anchor": "end", class: "gsqf-ords-size" }, SIZE_LABELS[size]);
            draw("rect", { class: "gsqf-red gsqf-ords-bar", x: X(0), y: yy, width: Math.max(0, X(positive) - X(0)), height: h });
            draw("rect", { class: "gsqf-blue gsqf-ords-bar", x: X(negative), y: yy, width: Math.max(0, X(0) - X(negative)), height: h });
          });
          draw("line", { class: "gsqf-ords-net", x1: X(d.difference), x2: X(d.difference), y1: at - 1, y2: at + group + 1 });
          draw("text", { x: width - 2, y: at + group / 2, "dominant-baseline": "central", "text-anchor": "end", class: on ? "gsqf-strong" : "" }, signed(d.difference));
        });
      },
    };
  }
  // ===== end: orders =====

  // -- the frame of a figure: its selector, its stage, and its dots ----------------------------------

  const specOf = (name) => {
    const spec = FIGURES[name];
    return typeof spec === "function" ? { draw: spec } : spec;
  };
  const frames = figures.map((figure) => {
    const name = figure.dataset.figure;
    const stage = figure.querySelector(".graphshapiq-explorer-figure-stage");
    const spec = specOf(name);
    return { figure, name, stage, spec, follows: !!spec && !spec.fixed, width: -1, buttons: [], dots: [], state: {} };
  });

  const hints = []; // the hints at the swipe that are on the page

  // a molecule chosen in a figure: the figure tells the page, with its molecule's own question
  function choose(id, frame) {
    if (id === current?.molecule) return;
    const task = moleculeOf(id).task;
    document.dispatchEvent(new CustomEvent(EVENT, { detail: { molecule: id, task, source: frame.figure } }));
  }

  function buildSelector(frame) {
    const bar = el("div", "gsqf-bar");
    const label = el("span", "gsqf-shown");
    // the molecule, where the thumbnails are not shown (a narrow page, and in print)
    const shown = el("span", "gsqf-shown-tile");
    const tiles = el("div", "gsqf-tiles");
    tiles.setAttribute("role", "group");
    tiles.setAttribute("aria-label", "Molecule of the figures");
    const dots = el("div", "gsqf-dots");
    dots.setAttribute("role", "group");
    dots.setAttribute("aria-label", "Molecule of the figures");
    for (const m of MOLS) {
      const button = el("button", "gsqf-tile-button");
      button.type = "button";
      button.title = m.name;
      button.setAttribute("aria-label", m.name);
      button.append(tile(m));
      button.addEventListener("click", () => choose(m.id, frame));
      tiles.append(button);
      frame.buttons.push(button);
      const dot = el("button", "gsqf-dot-button");
      dot.type = "button";
      dot.setAttribute("aria-label", m.name);
      dot.addEventListener("click", () => choose(m.id, frame));
      dots.append(dot);
      frame.dots.push(dot);
    }
    add(bar, shown, label, tiles);
    frame.stage.before(bar);
    frame.stage.after(dots);
    if (!hints.length) {
      // only the first figure says that it can be swiped, and only until it was
      const hint = el("p", "gsqf-hint", "Swipe for another molecule");
      dots.after(hint);
      hints.push(hint);
    }
    frame.label = label;
    frame.shown = shown;
    swipeable(frame);
  }

  // on a touch screen, a swipe sideways across a figure goes to the next molecule or the previous
  // one (as the FIxLIP page's figures): the figure follows the finger, and past a fifth of its
  // width, or flicked, the next molecule comes in from the other side. A swipe up or down scrolls.
  function swipeable(frame) {
    const { stage } = frame;
    let swipe = null;
    const shift = (dx) => {
      stage.style.transform = dx ? `translateX(${dx}px)` : "";
      stage.style.opacity = dx ? String(1 - Math.min(0.6, Math.abs(dx) / stage.clientWidth)) : "";
    };
    const at = () => MOLS.findIndex((m) => m.id === current.molecule);
    stage.addEventListener(
      "touchstart",
      (event) => {
        const zoomed = (window.visualViewport?.scale ?? 1) > 1.01; // then a drag pans the page
        const [touch] = event.touches;
        swipe = event.touches.length === 1 && !zoomed ? { x: touch.clientX, y: touch.clientY, dx: 0, sideways: null, moves: [[event.timeStamp, 0]] } : null;
      },
      { passive: true },
    );
    stage.addEventListener(
      "touchmove",
      (event) => {
        if (!swipe) return;
        const [touch] = event.touches;
        const [dx, dy] = [touch.clientX - swipe.x, touch.clientY - swipe.y];
        if (swipe.sideways === null) {
          if (Math.abs(dx) < 4 && Math.abs(dy) < 4) return;
          swipe.sideways = event.touches.length === 1 && Math.abs(dx) > Math.abs(dy);
          if (!swipe.sideways) {
            swipe = null; // the page scrolls
            return;
          }
          stage.classList.add("gsqf-swiping");
        }
        event.preventDefault(); // the drag is the figure's
        const end = (dx > 0 && at() === 0) || (dx < 0 && at() === MOLS.length - 1);
        swipe.dx = end ? dx / 4 : dx; // nothing beyond the first and the last molecule
        swipe.end = end;
        swipe.moves.push([event.timeStamp, dx]);
        shift(swipe.dx);
      },
      { passive: false },
    );
    const endSwipe = (event, cancelled) => {
      if (!swipe) return;
      const { dx, sideways, moves, end } = swipe;
      swipe = null;
      if (!sideways) return;
      stage.classList.remove("gsqf-swiping");
      const [time, from] = moves.find(([t]) => event.timeStamp - t <= 100) ?? moves.at(-1);
      const speed = (moves.at(-1)[1] - from) / Math.max(16, event.timeStamp - time);
      const flick = Math.abs(dx) > 20 && Math.abs(speed) > 0.3 && Math.sign(speed) === Math.sign(dx);
      shift(0);
      if (cancelled || end || !(Math.abs(dx) > stage.clientWidth / 5 || flick)) return;
      frame.entering = dx < 0 ? 1 : -1; // the side the next molecule comes in from
      for (const hint of hints.splice(0)) hint.style.visibility = "hidden"; // its room stays: nothing moves
      choose(MOLS[at() + (dx < 0 ? 1 : -1)].id, frame);
    };
    stage.addEventListener("touchend", (event) => endSwipe(event, false));
    stage.addEventListener("touchcancel", (event) => endSwipe(event, true));
  }

  // From one drawing of a figure to the next, part by part. A part that is the same in both stays as
  // it is, so nothing that is still flickers. The dot of a chart (circle.gsqf-dot) moves to its new
  // place. Every other part fades out before the part in its place fades in, so two texts are never
  // seen in one place.
  const OUT = 140; // milliseconds for the parts that leave, then IN for the parts that come
  const IN = 220;
  const MANY = 48; // more parts than this change as one
  function morph(last, layer) {
    const ownText = (node) => [...node.childNodes].some((child) => child.nodeType === 3 && child.data.trim());
    const isDot = (node) => node.localName === "circle" && node.classList.contains("gsqf-dot");
    // the marks of a part of the old layer (a) and of the new one (b)
    const walk = (a, b) => {
      if (a.isEqualNode(b)) return [[a, "gone"]];
      if (isDot(a) && isDot(b)) {
        const by = ["cx", "cy"].map((at) => Number(a.getAttribute(at)) - Number(b.getAttribute(at)));
        return [[a, "gone"], [b, "slide", by]];
      }
      // the same element with other content: its parts one by one. Otherwise it changes as a whole,
      // and so does an element with many parts that change, such as the points of a scatter plot
      const alike = a.cloneNode(false).isEqualNode(b.cloneNode(false));
      if (alike && b.children.length && a.children.length === b.children.length && !ownText(a) && !ownText(b)) {
        const marks = [...b.children].flatMap((child, k) => walk(a.children[k], child));
        if (marks.filter(([, name]) => name === "in").length <= MANY) return marks;
      }
      return [[a, "out"], [b, "in"]];
    };
    const same = last.children.length === layer.children.length;
    const marks = same ? [...layer.children].flatMap((child, k) => walk(last.children[k], child)) : walk(last, layer);
    for (const [node, name, by] of marks) {
      if (name === "slide") {
        const steps = [{ transform: `translate(${by[0]}px, ${by[1]}px)` }, { transform: "none" }];
        node.animate(steps, { duration: OUT + IN, easing: "cubic-bezier(0.2, 0.7, 0.2, 1)" });
      } else node.classList.add(`gsqf-${name}`);
    }
    return () => {
      for (const [node, name] of marks) node.classList.remove(`gsqf-${name}`);
    };
  }

  // what a figure reads for the molecule and the question shown
  function contextOf(frame, shown = current) {
    const base = { data, extra, part: extra?.[frame.name], molecules: MOLS, tasks: data.tasks, name: frame.name, figure: frame.figure, width: frame.width, redraw: () => render(frame, true) };
    if (!frame.follows || !shown) return { ...base, molecule: null, task: null, own: null, ex: null };
    const m = moleculeOf(shown.molecule);
    const task = taskOf(shown.task) ?? taskOf(m.task);
    return { ...base, molecule: m, task, own: taskOf(m.task), ex: explanationOf(m.id, task.id) };
  }

  // The highest drawing of a figure that follows the molecule, over all molecules (with their own
  // questions, and the question shown for the molecule shown), at the stage's width: its height is
  // the stage's min-height.
  function reserveStage(frame) {
    if (!frame.follows || frame.spec.reserve === false) return;
    const probe = el("div", "gsqf-layer gsqf-probe");
    probe.setAttribute("aria-hidden", "true");
    probe.inert = true;
    frame.stage.append(probe);
    let high = 0;
    for (const m of MOLS) {
      const shown = m.id === current.molecule ? current : { molecule: m.id, task: m.task };
      probe.replaceChildren();
      try {
        frame.spec.draw(probe, contextOf(frame, shown), { ...frame.state });
      } catch {
        continue; // the figure itself reports it when it is drawn
      }
      high = Math.max(high, probe.getBoundingClientRect().height);
    }
    probe.remove();
    frame.stage.style.minHeight = high ? `${Math.ceil(high)}px` : "";
  }

  // A figure is drawn into a layer of its stage. With `fade`, the layer before stays under the new
  // one until the new one has taken its place (morph). Both lie in one cell of the stage, and the
  // stage keeps the height of the highest molecule, so nothing around the figure moves.
  function render(frame, fade = false) {
    const { stage, name, spec } = frame;
    if (!spec) return;
    frame.finish?.(); // a change that is still under way ends here
    const before = [...stage.querySelectorAll(":scope > .gsqf-layer")];
    const layer = el("div", "gsqf-layer");
    stage.append(layer);
    try {
      spec.draw(layer, contextOf(frame), frame.state);
    } catch (error) {
      console.error(`The figure "${name}" could not be drawn:`, error);
      layer.replaceChildren(el("p", "graphshapiq-explorer-figure-missing", stage.getAttribute("aria-label") ?? ""));
    }
    if (frame.follows) {
      const m = moleculeOf(current.molecule);
      const task = taskOf(current.task) ?? taskOf(m.task);
      frame.label.replaceChildren(el("strong", "", m.name), el("span", "gsqf-question", `${task.label}?`));
      if (frame.shown.dataset.molecule !== m.id) {
        frame.shown.replaceChildren(tile(m));
        frame.shown.dataset.molecule = m.id;
      }
      MOLS.forEach((x, k) => {
        frame.buttons[k].setAttribute("aria-pressed", String(x.id === m.id));
        frame.dots[k].setAttribute("aria-pressed", String(x.id === m.id));
      });
    }
    frame.figure.classList.add("gsqf-ready");
    const entering = frame.entering;
    frame.entering = 0;
    const last = before.pop();
    for (const node of before) node.remove();
    if (!fade || !last || reducedMotion()) {
      last?.remove();
      return;
    }
    // after a swipe, the new molecule comes in from the side that the finger came from
    if (entering) {
      last.remove();
      layer.animate([{ opacity: 0, transform: `translateX(${entering * 32}px)` }, { opacity: 1, transform: "none" }], { duration: OUT + IN, easing: "cubic-bezier(0.2, 0.7, 0.2, 1)" });
      return;
    }
    last.inert = true;
    let timer = 0;
    const clean = morph(last, layer);
    frame.finish = () => {
      frame.finish = null;
      clearTimeout(timer);
      last.remove();
      clean();
      for (const animation of layer.getAnimations({ subtree: true })) animation.cancel();
    };
    timer = setTimeout(frame.finish, OUT + IN + 40);
  }

  // a figure at a new width: its room for every molecule, then the figure
  function layout(frame, width) {
    frame.width = width;
    reserveStage(frame);
    render(frame);
  }

  function start() {
    data = window.SHAPIQ_GRAPHSHAPIQ_EXPLORER;
    extra = window.SHAPIQ_GRAPHSHAPIQ_EXPLORER_FIGURES ?? {};
    const hasPattern = (m) => (data.explanations[`${m.id}/${m.task}`]?.truth.length ? 0 : 1);
    MOLS = data.molecules.slice().sort((a, b) => TASK_ORDER.indexOf(a.task) - TASK_ORDER.indexOf(b.task) || hasPattern(a) - hasPattern(b));
    // the molecule the explorer shows (not the reader's own, "build"), or the explorer's first
    const known = (shown) => (shown && moleculeOf(shown.molecule) ? { molecule: shown.molecule, task: taskOf(shown.task) ? shown.task : moleculeOf(shown.molecule).task } : null);
    current = known(G.current) ?? { molecule: MOLS[0].id, task: MOLS[0].task };
    for (const frame of frames) {
      if (!frame.spec) {
        console.error(`The figure "${frame.name}" is not in figures.js.`);
        frame.stage.replaceChildren(el("p", "graphshapiq-explorer-figure-missing", frame.stage.getAttribute("aria-label") ?? ""));
        continue;
      }
      if (frame.follows) buildSelector(frame);
      // drawn at once, before the page is painted; a new width draws the figure again (in the next
      // frame, outside the observer's callback)
      layout(frame, Math.round(frame.stage.getBoundingClientRect().width));
      new ResizeObserver(([entry]) => {
        const now = Math.round(entry.contentRect.width);
        if (now === frame.width) return;
        requestAnimationFrame(() => layout(frame, now));
      }).observe(frame.stage);
    }
    // texts measured in a fallback font: again once the page's fonts are there
    if (document.fonts && document.fonts.status !== "loaded")
      document.fonts.ready.then(() => {
        for (const frame of frames) if (frame.spec) layout(frame, frame.width);
      });
    // another molecule or question, chosen in the explorer or in a figure: all figures show it, and
    // the figure it was chosen in keeps its place on the screen, whatever the figures above it do
    document.addEventListener(EVENT, (event) => {
      const shown = known(event.detail);
      if (!shown || (shown.molecule === current.molecule && shown.task === current.task)) return;
      current = shown;
      const source = event.detail.source instanceof Element ? event.detail.source : null;
      const before = source?.getBoundingClientRect().top;
      for (const frame of frames) if (frame.follows) render(frame, true);
      if (source) window.scrollBy({ top: source.getBoundingClientRect().top - before, behavior: "instant" });
    });
  }

  // the helpers, for trying a drawing in the browser's console (not for the figures, which have
  // them in scope): e.g. document.body.append(ShapiqGraphFigures.molecule(...))
  window.ShapiqGraphFigures = { FIGURES, molecule, chart, slider, panel, reading, columns, legend, reserve, explanationOf, predict, taskOf, moleculeOf, frames, get data() { return data; }, get extra() { return extra; }, get current() { return current; } };

  const missing = (error) => {
    console.error("The figures could not start:", error);
    // the description of every figure, as without JavaScript
    for (const { stage } of frames) stage.replaceChildren(el("p", "graphshapiq-explorer-figure-missing", stage.getAttribute("aria-label") ?? ""));
  };
  if (!G) {
    missing(new Error("ShapiqGraph (graphshapiq-common/assets/demo.js) is not on the page"));
    return;
  }
  const load = (name, file) =>
    window[name]
      ? Promise.resolve()
      : new Promise((resolve, reject) => {
          const loader = document.createElement("script");
          loader.src = `${root}${file}`;
          loader.onload = resolve;
          loader.onerror = reject;
          document.head.appendChild(loader);
        });
  // both are on the page (the directive adds them before this script): then at once
  if (window.SHAPIQ_GRAPHSHAPIQ_EXPLORER && window.SHAPIQ_GRAPHSHAPIQ_EXPLORER_FIGURES) {
    try {
      start();
    } catch (error) {
      missing(error);
    }
  } else
    Promise.all([load("SHAPIQ_GRAPHSHAPIQ_EXPLORER", "data.js"), load("SHAPIQ_GRAPHSHAPIQ_EXPLORER_FIGURES", "figures-data.js")])
      .then(start)
      .catch(missing);
})();
