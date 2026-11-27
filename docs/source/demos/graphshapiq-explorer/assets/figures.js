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
  FIGURES.pipeline = {
    draw(layer, c) {
      placeholder(layer, c, "pipeline");
    },
  };
  // ===== end: pipeline =====

  // ===== figure: nitro =====
  FIGURES.nitro = {
    draw(layer, c) {
      placeholder(layer, c, "nitro");
    },
  };
  // ===== end: nitro =====

  // ===== figure: maskings =====
  FIGURES.maskings = {
    draw(layer, c) {
      placeholder(layer, c, "maskings");
    },
  };
  // ===== end: maskings =====

  // ===== figure: order =====
  FIGURES.order = {
    draw(layer, c) {
      placeholder(layer, c, "order");
    },
  };
  // ===== end: order =====

  // ===== figure: arrangement =====
  FIGURES.arrangement = {
    fixed: true,
    draw(layer, c) {
      placeholder(layer, { ...c, molecule: moleculeOf("dopamine"), task: taskOf("PAINS"), ex: explanationOf("dopamine", "PAINS") }, "arrangement");
    },
  };
  // ===== end: arrangement =====

  // ===== figure: sparsity =====
  FIGURES.sparsity = {
    draw(layer, c) {
      placeholder(layer, c, "sparsity");
    },
  };
  // ===== end: sparsity =====

  // ===== figure: complexity =====
  FIGURES.complexity = {
    draw(layer, c) {
      placeholder(layer, c, "complexity");
    },
  };
  // ===== end: complexity =====

  // ===== figure: approximation =====
  FIGURES.approximation = {
    draw(layer, c) {
      placeholder(layer, c, "approximation");
    },
  };
  // ===== end: approximation =====

  // ===== figure: paper-approximation =====
  FIGURES["paper-approximation"] = {
    fixed: true,
    draw(layer) {
      const holder = el("div", "gsqf-plot");
      layer.append(panel("paper-approximation", holder));
      const plot = chart(holder, 220, { left: 52, bottom: 36 });
      const x = plot.log([1e2, 1e4], [plot.box.x0, plot.box.x1]);
      const y = plot.log([1e-4, 1], [plot.box.y1, plot.box.y0]);
      plot.yAxis(y, logTicks(1e-4, 1), "MSE", power);
      plot.xAxis(x, logTicks(1e2, 1e4), "model calls", power);
    },
  };
  // ===== end: paper-approximation =====

  // ===== figure: orders =====
  FIGURES.orders = {
    draw(layer, c) {
      placeholder(layer, c, "orders");
    },
  };
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
