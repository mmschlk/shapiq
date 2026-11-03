// The orders demo (demos/orders): one prediction, explained at every order of interaction, as a
// stack of sheets. The top sheet holds the Shapley values (order 1); every sheet below adds the
// interactions of one more order (FSII, the Faithful Shapley Interactions, or k-SII, as the data
// says), up to order 4. Below them, blank sheets stand for the orders that follow, up to the bottom
// sheet: the full decomposition, the interactions of all orders, which add up to the model itself.
// Beside the stack, upright, a plot of every order's R^2, its faithfulness: how much of the model's
// output on random coalitions (drawn from the distribution of the Shapley values) the explanation
// reproduces beyond what their sizes alone predict. Every order's point lies at the height of its
// sheet, a line leading to it from the sheet's corner, and the line through them runs down from the
// Shapley values to the full decomposition, which reproduces the model exactly, unless it is
// estimated: what it falls short of 1 is then shaded, the estimation gap.
//
// The examples: the molecule of GraphSHAP-IQ's Figure 1, explained exactly by GraphSHAP-IQ for a
// graph convolutional network; one house and 150 houses of California housing, explained exactly by
// shapiq's TreeExplainer for an XGBoost model of depth 5 (the 150 houses as a heatmap, a column
// each); the same house valued by TabPFN, a black box, explained exactly over all subsets of its
// features with shapiq's TabPFNImputer; and how SigLIP 2, a black box, matches the photo and
// caption of the FIxLIP demo, explained by ProxySHAP through an XGBoost proxy, without adjustment.
// The full decomposition is the interactions of order n, the number of players (or of the trees'
// depth, which already are); the orders above the highest one with an interaction (the trees'
// depth, the network's receptive field, the proxy's depth) all equal it. The explanations are
// precomputed (../compute) into data.js. Pointing at a sheet (hover, keyboard focus on its row, a
// tap that pins it, or the slider left of the stack) slides it out to the side of the stack as it
// lies, under the sheets above it, and the others stay where they are. A ring marks its order on the
// plot; pointing along the plot brings out the sheet of the order pointed at, or the blank sheet of
// an order between. The slider is an arrow down the side of the stack: the explanations grow more
// complex, and more faithful, from order to order. Where the examples do not fit in a row, a stepper
// goes from one to the next, and so does a swipe across the stack. Above the stack, the plan shows
// the problem, as machine learning draws it: the input x wired through the model f, a chip, to its
// prediction f(x). Out of the chip falls the light of the explanation: the example seen from above,
// flat, its input with the explanation pointed at on it (at rest, the Shapley values), and in a few
// words what that explanation is, with a bar by its R^2, and how well it fits. The chip clears as far
// as the explanation fits: the black box opens, all the way where it is exact. The demo starts with
// the problem alone (unless data-reveal="open"): a click on the chip explains it, and the
// explanation is built under it from the top down. The problem stays as the examples change: what
// its parts show flips, as on a board of departures.
//
// Two more views (data-view on the demo; the docs show the stack) float the explanations as layers
// of paper, their marks only, above the example's input on a card, the full decomposition the last
// of them, and below the card the mechanism they explain, the model itself, however complex: the
// explanations reveal more and more of it, order by order. Unfolded ("unfolded"), the layers float
// apart, and the one pointed at slides out, as the stack's sheets do, showing its input under its
// marks. Projected ("projected"), they lie on the card at rest (the top one showing, on the input),
// the mechanism hidden under it; once explained and in view (or after a tap), they unfold, for good,
// and the layer pointed at casts its light onto the card, which wears its marks (the mechanism's
// rises to it, a box of orange light). In both, the others fade as the stack's do; the slider and
// the plot go through the explanations only, the input and the mechanism named under the plot; the
// stepper and the swipe work as in the stack.
//
// The sheets are drawn in an axonometric view: a square face, turned in its plane and squashed
// vertically, so that every corner of every sheet is known here exactly. The script uses them to
// find the sheet under the pointer and to lead every row of the ladder to its sheet.
//
// Without JavaScript, the static figure in the <noscript> shows; when the data does not load, or the
// demo cannot start, the script shows that figure itself.
(() => {
  const roots = [...document.querySelectorAll(".orders-demo")];
  if (!roots.length) return;

  const SVG = "http://www.w3.org/2000/svg";
  const RED = [255, 13, 87];
  const BLUE = [30, 136, 229];
  const TURN = (-45 * Math.PI) / 180; // the sheets' rotation in their plane
  const TILT = 0.5; // how much of a sheet's depth shows (the cosine of the view's elevation)
  // in units of a sheet's side: what shows of a sheet below the one above it, and of a blank sheet
  // of the orders between
  const STRIP = 0.24;
  const BLANK_STRIP = 0.04;
  // how far the blank sheet of the order pointed at slides out of the stack, towards the reader:
  // less than its strip, so that the blank sheet below it still shows
  const PEEK = 0.03;
  // the blank sheets of an example: as many for every example (where it skips as many orders), so
  // that every stack looks as deep
  const BLANKS = 6;
  // the sheet pointed at slides out of the stack to the side, as it lies and keeping its place in
  // the stack (the sheets above it still lie over it), and the others and the rows of the ladder
  // stay where they are: how far it slides, in units of its width
  const SLIDE = 0.32;
  // pixels: the column of the slider, left of the stack; its line in it; how far above the top
  // sheet its handle rests, when no sheet is out
  const RAIL = 42;
  const RAIL_LINE = 28;
  const RAIL_REST = 24;
  const THICKNESS = 4; // pixels: the sheets' edge, below their face
  // the floating layers': the air between the lowest and the input below, in units of a sheet's
  // side; and the depth below the input at which the mechanism lies
  const AIR = 0.3;
  const DEPTH = 0.34;

  const roundTo = (value, digits) => {
    const scale = 10 ** digits;
    return (Math.sign(value) * Math.round(Math.abs(value) * scale)) / scale;
  };
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  // a symbol of the problem, as mathematics sets it: x, f, f(x)
  const math = (text) => el("var", "orders-demo-math", text);
  const add = (parent, tag, attributes = {}, text) => {
    const node = document.createElementNS(SVG, tag);
    for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
    if (text !== undefined) node.textContent = text;
    parent.appendChild(node);
    return node;
  };
  const solid = (value) => `rgb(${(value >= 0 ? RED : BLUE).join(", ")})`;

  // the models' icons, on a 32 x 32 grid: a decision tree for tree ensembles, a graph for graph
  // neural networks, and a black box (an isometric cube, as the sheets) for any other model
  const ICONS = {
    tree:
      '<rect x="11" y="3" width="10" height="6" rx="1.5"/><path d="M16 9v3.5M7 16v-3.5h17V16"/>' +
      '<circle class="orders-demo-icon-leaf" cx="7" cy="19" r="3"/><rect x="19" y="16" width="10" height="6" rx="1.5"/>' +
      '<path d="M24 22v3M19.5 27v-2h9v2"/><circle class="orders-demo-icon-leaf" cx="19.5" cy="29" r="2"/>' +
      '<circle class="orders-demo-icon-leaf" cx="28.5" cy="29" r="2"/>',
    graph:
      '<path d="M9 7l14 2 5 9-6 9-13-1-5-9zM9 7l7 9 12 2M16 16l-7 10"/>' +
      [[9, 7], [23, 9], [28, 18], [22, 27], [9, 26], [4, 17], [16, 16]]
        .map(([x, y]) => `<circle class="orders-demo-icon-node" cx="${x}" cy="${y}" r="2.6"/>`)
        .join(""),
    box:
      '<path class="orders-demo-icon-side" d="M4 10l12 6v13L4 23z"/>' +
      '<path class="orders-demo-icon-side orders-demo-icon-right" d="M28 10l-12 6v13l12-6z"/>' +
      '<path class="orders-demo-icon-side orders-demo-icon-top" d="M16 4l12 6-12 6-12-6z"/>',
  };
  const icon = (name, className) => {
    const node = document.createElementNS(SVG, "svg");
    node.setAttribute("viewBox", "0 0 32 32");
    node.setAttribute("aria-hidden", "true");
    node.setAttribute("class", `orders-demo-icon orders-demo-icon-${name} ${className}`);
    node.innerHTML = ICONS[name];
    return node;
  };
  const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // -- the faces of the sheets: the explanation of one order, in a 100 x 100 square ---------------

  // the largest value on a sheet: what its faces scale to (a face of what a sheet adds to the one
  // above it keeps the sheet's own, so that it is faint where it adds little)
  const largestValue = (sheet) =>
    sheet.largest ?? Math.max(...sheet.first.map(Math.abs), ...sheet.groups.map((group) => Math.abs(group[1])));

  // the connection of a player at p with the center c of its group, pinched in the middle, as in
  // shapiq's SI graph plot
  const bowtie = (p, c, width) => {
    const [dx, dy] = [c[0] - p[0], c[1] - p[1]];
    const length = Math.hypot(dx, dy);
    if (length < 1e-6) return "";
    const [nx, ny] = [(-dy / length) * (width / 2), (dx / length) * (width / 2)];
    const [mx, my] = [p[0] + dx / 2, p[1] + dy / 2];
    const f = (x, y) => `${roundTo(x, 2)} ${roundTo(y, 2)}`;
    return (
      `M${f(p[0] - nx, p[1] - ny)}Q${f(mx, my)} ${f(c[0] - nx, c[1] - ny)}` +
      `L${f(c[0] + nx, c[1] + ny)}Q${f(mx, my)} ${f(p[0] + nx, p[1] + ny)}Z`
    );
  };

  // a disc as a path, clockwise as every bowtie: in one path with them, it is filled once where
  // they overlap (nonzero)
  const discPath = (x, y, r) =>
    `M${roundTo(x - r, 2)} ${y}A${r} ${r} 0 0 1 ${roundTo(x + r, 2)} ${y}A${r} ${r} 0 0 1 ${roundTo(x - r, 2)} ${y}Z`;

  // the smallest share of the largest value on a sheet that is drawn at all
  const FAINTEST = 0.01;

  // an interaction of two or more players: every player joined to the group's center (for a pair,
  // to the other player; else, where given, or the players' centroid), in red or blue; as shapiq's
  // SI graph plot draws it, as wide (widest for the largest value) and as opaque as its share of
  // the largest value on its sheet. Its shapes are one path, so that it shows as one shape, as
  // opaque where they overlap as elsewhere, without a group's opacity (a layer of its own to paint)
  const drawGroup = (parent, points, value, largest, widest, at) => {
    const ratio = Math.abs(value) / largest;
    if (ratio < FAINTEST) return;
    const width = widest * ratio;
    const center =
      points.length === 2
        ? points[1]
        : (at ?? [0, 1].map((axis) => points.reduce((sum, p) => sum + p[axis], 0) / points.length));
    let d = "";
    for (const p of points.length === 2 ? [points[0]] : points) d += bowtie(p, center, width);
    for (const [x, y] of points.length === 2 ? points : [...points, center]) {
      d += discPath(roundTo(x, 2), roundTo(y, 2), roundTo(width / 2, 2));
    }
    add(parent, "path", { d, fill: solid(value), "fill-opacity": roundTo(ratio, 3) });
  };

  // the groups of a sheet; a group's center, where the data gives one, is placed by place
  const drawGroups = (parent, sheet, anchor, largest, widest, place) => {
    const layer = add(parent, "g", { class: "orders-demo-groups" });
    // the weakest first, so that the strongest lie on top
    const groups = [...sheet.groups].sort((a, b) => Math.abs(a[1]) - Math.abs(b[1]));
    for (const [players, value, center] of groups) {
      drawGroup(layer, players.map(anchor), value, largest, widest, center && place(center));
    }
  };

  // the map of points into a box that scales them, keeping their aspect, to fit it, centered
  const fit = (points, [left, top, right, bottom]) => {
    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    const [x0, y0] = [Math.min(...xs), Math.min(...ys)];
    const [w, h] = [Math.max(...xs) - x0, Math.max(...ys) - y0];
    const scale = Math.min((right - left) / w, (bottom - top) / h);
    const [ox, oy] = [left + (right - left - w * scale) / 2, top + (bottom - top - h * scale) / 2];
    return ([x, y]) => [ox + (x - x0) * scale, oy + (y - y0) * scale];
  };

  // the explanations as shapiq's SI graph plot draws them with node_size_scaling=0.5 and
  // size_factor=2, in units of the players' extent: an atom's diameter, and the width of the
  // largest value
  const ATOM = 0.0625;
  const WIDEST = 0.05;
  const extentOf = (points) =>
    Math.max(...[0, 1].map((axis) => Math.max(...points.map((p) => p[axis])) - Math.min(...points.map((p) => p[axis]))));

  // a face's layers: "all" for a sheet of the stack, or, for the floating layers, the "input" (what
  // is explained, on the card) or the "marks" (the explanation, on a layer above it)
  const shows = (layer, part) => layer === "all" || layer === part;

  // the molecule: its bonds and atoms over the explanation, as in GraphSHAP-IQ's figures
  function moleculeFace(svg, example, sheet, layer = "all") {
    const place = fit(example.coords, [9, 5, 93, 87]);
    const position = example.coords.map(place);
    const extent = extentOf(position);
    const [radius, widest] = [(ATOM / 2) * extent, WIDEST * extent];
    const largest = largestValue(sheet);
    if (shows(layer, "marks")) {
      // the groups' centers where shapiq's SI graph plot puts them (../compute/pack.py)
      drawGroups(svg, sheet, (atom) => position[atom], largest, widest, place);
      // a main effect: a disc around its atom, as much wider as a group of its value, over a disc
      // of the sheet's color that hides the groups below it (on a sheet; a layer has no color)
      const halos = add(svg, "g", { class: "orders-demo-halos" });
      sheet.first.forEach((value, atom) => {
        const ratio = Math.abs(value) / largest;
        if (ratio < FAINTEST) return;
        const disc = {
          cx: roundTo(position[atom][0], 2),
          cy: roundTo(position[atom][1], 2),
          r: roundTo(radius + (widest * ratio) / 2, 2),
        };
        if (layer === "all") add(halos, "circle", { ...disc, class: "orders-demo-halo-ground" });
        add(halos, "circle", { ...disc, fill: solid(value), "fill-opacity": roundTo(ratio, 3) });
      });
    }
    if (!shows(layer, "input")) return;
    const bonds = add(svg, "g", { class: "orders-demo-bonds" });
    for (const [i, j] of example.bonds) {
      const [[x1, y1], [x2, y2]] = [position[i], position[j]];
      add(bonds, "line", { x1, y1, x2, y2 });
    }
    const atoms = add(svg, "g", { class: "orders-demo-atoms" });
    example.atoms.forEach((atom, i) => {
      const [x, y] = position[i];
      add(atoms, "circle", { cx: x, cy: y, r: roundTo(radius, 2) });
      add(atoms, "text", { x, y: roundTo(y + 0.05, 2) }, atom);
    });
  }

  // the photo and its caption: the photo's patches on the left, the caption's tokens on the right
  function imageTextFace(svg, example, sheet, url, id, layer = "all") {
    const photo = { x: 4, y: 11, size: 62 };
    const grid = example.grid;
    const cell = photo.size / grid;
    const tokens = { x: 72, width: 24, top: 5, bottom: 95 };
    const row = (tokens.bottom - tokens.top) / example.tokens.length;
    const patches = grid * grid;
    const anchor = (player) =>
      player < patches
        ? [photo.x + ((player % grid) + 0.5) * cell, photo.y + (Math.floor(player / grid) + 0.5) * cell]
        : [tokens.x + 0.6, tokens.top + (player - patches + 0.5) * row];
    const largest = largestValue(sheet);

    const clip = `${id}-photo`;
    add(add(svg, "clipPath", { id: clip }), "rect", {
      x: photo.x,
      y: photo.y,
      width: photo.size,
      height: photo.size,
      rx: 1.6,
    });
    const image = add(svg, "g", { "clip-path": `url(#${clip})` });
    if (shows(layer, "input")) {
      add(image, "image", {
        href: url,
        x: photo.x,
        y: photo.y,
        width: photo.size,
        height: photo.size,
        preserveAspectRatio: "xMidYMid slice",
      });
      add(image, "rect", {
        class: "orders-demo-wash",
        x: photo.x,
        y: photo.y,
        width: photo.size,
        height: photo.size,
      });
    }
    // a main effect tints its patch, or its word, as opaque as its share of the largest value
    if (shows(layer, "marks")) {
      sheet.first.slice(0, patches).forEach((value, patch) => {
        const ratio = Math.abs(value) / largest;
        if (ratio < FAINTEST) return;
        const [cx, cy] = anchor(patch);
        add(image, "rect", {
          x: roundTo(cx - cell / 2, 2),
          y: roundTo(cy - cell / 2, 2),
          width: roundTo(cell, 2),
          height: roundTo(cell, 2),
          fill: solid(value),
          "fill-opacity": roundTo(ratio, 3),
        });
      });
    }
    if (shows(layer, "input")) {
      const lines = add(image, "g", { class: "orders-demo-grid" });
      for (let k = 1; k < grid; k++) {
        const at = roundTo(photo.x + k * cell, 2);
        const across = roundTo(photo.y + k * cell, 2);
        add(lines, "line", { x1: at, y1: photo.y, x2: at, y2: photo.y + photo.size });
        add(lines, "line", { x1: photo.x, y1: across, x2: photo.x + photo.size, y2: across });
      }
    }

    const players = Array.from({ length: patches + example.tokens.length }, (_, player) => anchor(player));
    if (shows(layer, "marks")) drawGroups(svg, sheet, anchor, largest, WIDEST * extentOf(players));

    const words = add(svg, "g", { class: "orders-demo-tokens" });
    example.tokens.forEach((token, k) => {
      const value = sheet.first[patches + k];
      const ratio = Math.abs(value) / largest;
      const y = tokens.top + k * row + row * 0.12;
      const height = row * 0.76;
      if (shows(layer, "input")) {
        add(words, "rect", { x: tokens.x, y: roundTo(y, 2), width: tokens.width, height: roundTo(height, 2), rx: 1.4 });
      }
      if (shows(layer, "marks") && ratio >= FAINTEST) {
        add(words, "rect", {
          class: "orders-demo-tint",
          x: tokens.x,
          y: roundTo(y, 2),
          width: tokens.width,
          height: roundTo(height, 2),
          rx: 1.4,
          fill: solid(value),
          "fill-opacity": roundTo(ratio, 3),
        });
      }
      if (shows(layer, "input")) {
        add(words, "text", { x: tokens.x + tokens.width / 2, y: roundTo(y + height / 2 + 0.1, 2) }, token);
      }
    });
  }

  // one house: its explanation as a tree that grows one level per order. The features are on top,
  // every group of k features hangs at level k, below the middle of its features, and joined to
  // each of them; red raises the value of the house, blue lowers it. The full decomposition of a
  // tree ensemble is as deep as its trees.
  const TREE_GROUPS = [0, 0, 6, 5, 4, 3, 3, 2]; // the strongest groups shown, by level
  function treeFace(svg, example, sheet, layer = "all") {
    const levels = example.largest;
    const [top, bottom] = [24, 88];
    const step = (bottom - top) / (levels - 1);
    const slot = new Map(example.layout.map((feature, k) => [feature, 9 + (k * 82) / (example.layout.length - 1)]));
    const largest = largestValue(sheet);
    const ratioOf = (value) => Math.abs(value) / largest;

    const marks = shows(layer, "marks");
    const guides = add(svg, "g", { class: "orders-demo-tree-levels" });
    for (let level = 1; level <= (marks ? levels : 0); level++) {
      const y = roundTo(top + (level - 1) * step, 2);
      add(guides, "line", { x1: 6, x2: 94, y1: y, y2: y });
      add(guides, "text", { x: 2.5, y }, String(level));
    }

    // the strongest groups of every level, below the middle of their features, apart
    const nodes = [];
    for (let level = 2; level <= levels; level++) {
      // the strongest of the level, however small: the tree reaches as deep as the explanation
      const groups = sheet.groups
        .filter(([players, value]) => players.length === level && value !== 0)
        .slice(0, TREE_GROUPS[level] ?? 2)
        .map(([players, value]) => ({
          players,
          value,
          x: players.reduce((sum, p) => sum + slot.get(p), 0) / players.length,
          y: top + (level - 1) * step,
        }))
        .sort((a, b) => a.x - b.x);
      // apart, then shifted back to their middle, and kept on the sheet
      const middle = groups.reduce((sum, group) => sum + group.x, 0) / (groups.length || 1);
      for (let k = 1; k < groups.length; k++) groups[k].x = Math.max(groups[k].x, groups[k - 1].x + 6.5);
      const moved = groups.reduce((sum, group) => sum + group.x, 0) / (groups.length || 1);
      const shift = Math.min(Math.max(middle - moved, 9 - (groups[0]?.x ?? 9)), 91 - (groups.at(-1)?.x ?? 91));
      for (const group of groups) group.x += shift;
      nodes.push(...groups);
    }

    const links = add(svg, "g", { class: "orders-demo-tree-links" });
    for (const node of marks ? [...nodes].sort((a, b) => Math.abs(a.value) - Math.abs(b.value)) : []) {
      const ratio = ratioOf(node.value);
      for (const player of node.players) {
        const [x1, y1, x2] = [node.x, node.y, slot.get(player)];
        const bend = (y1 - top) * 0.55;
        add(links, "path", {
          d: `M${roundTo(x1, 2)} ${roundTo(y1, 2)}C${roundTo(x1, 2)} ${roundTo(y1 - bend, 2)} ${roundTo(x2, 2)} ${roundTo(top + bend, 2)} ${roundTo(x2, 2)} ${top}`,
          stroke: solid(node.value),
          "stroke-width": roundTo(0.2 + 1.3 * ratio, 2),
          "stroke-opacity": roundTo(0.18 + 0.6 * ratio, 3),
        });
      }
    }
    const groups = add(svg, "g", { class: "orders-demo-tree-groups" });
    for (const node of marks ? nodes : []) {
      const ratio = ratioOf(node.value);
      const r = 1 + 4.6 * ratio ** 0.8;
      add(groups, "circle", {
        cx: roundTo(node.x, 2),
        cy: roundTo(node.y, 2),
        r: roundTo(r, 2),
        fill: solid(node.value),
        "fill-opacity": roundTo(0.4 + 0.6 * ratio, 3),
      });
      if (ratio >= 0.25) {
        add(groups, "text", { x: roundTo(node.x + r + 1, 2), y: roundTo(node.y, 2) }, dollars(node.value));
      }
    }

    // the features: their main effects as halos, their names and values above them, every other
    // one higher, so that they do not touch
    const halos = add(svg, "g", { class: "orders-demo-halos" });
    const features = add(svg, "g", { class: "orders-demo-atoms" });
    const labels = add(svg, "g", { class: "orders-demo-features" });
    example.layout.forEach((feature, k) => {
      const x = roundTo(slot.get(feature), 2);
      const ratio = ratioOf(sheet.first[feature]);
      if (marks && ratio >= 0.02) {
        add(halos, "circle", {
          cx: x,
          cy: top,
          r: roundTo(2.2 + 4.4 * ratio, 2),
          fill: solid(sheet.first[feature]),
          "fill-opacity": roundTo(0.15 + 0.85 * ratio, 3),
        });
      }
      if (!shows(layer, "input")) return;
      add(features, "circle", { cx: x, cy: top, r: 2 });
      const y = k % 2 === 0 ? 5.6 : 12;
      add(labels, "text", { x, y, "text-anchor": "middle", class: "orders-demo-feature-name" }, example.features[feature]);
      add(labels, "text", { x, y: y + 3.3, "text-anchor": "middle", class: "orders-demo-feature-value" }, example.values[feature]);
    });
  }

  // many houses: a heatmap, a column per house sorted by its prediction (the line on top), a row
  // per feature or group of features; the cells are drawn on a canvas, one pixel each
  function heatmapFace(svg, example, sheet, layer = "all") {
    const box = { left: 30, right: 97, top: 23, row: 4.7, band: [6, 18] };
    const columns = example.predictions.length;
    const rows = sheet.rows;
    const width = (box.right - box.left) / columns;

    const [low, high] = [Math.min(...example.predictions), Math.max(...example.predictions)];
    const line = example.predictions.map((value, k) => {
      const x = box.left + (k + 0.5) * width;
      const y = box.band[1] - ((value - low) / (high - low)) * (box.band[1] - box.band[0]);
      return `${roundTo(x, 2)},${roundTo(y, 2)}`;
    });
    if (shows(layer, "input")) {
      add(svg, "path", {
        class: "orders-demo-heat-area",
        d: `M${box.left},${box.band[1]}L${line.join("L")}L${box.right},${box.band[1]}Z`,
      });
      add(svg, "path", { class: "orders-demo-heat-line", d: `M${line.join("L")}` });
      add(svg, "text", { class: "orders-demo-heat-label", x: box.left - 1.5, y: box.band[1] - 2 }, "prediction");
    }
    if (!shows(layer, "marks")) return;

    const canvas = document.createElement("canvas");
    canvas.width = columns;
    canvas.height = rows.length;
    const context = canvas.getContext("2d");
    const pixels = context.createImageData(columns, rows.length);
    rows.forEach(([, values], r) => {
      values.forEach((value, k) => {
        const [red, green, blue] = value >= 0 ? RED : BLUE;
        const offset = 4 * (r * columns + k);
        pixels.data.set([red, green, blue, Math.round(255 * (Math.abs(value) / 99) ** 0.85)], offset);
      });
    });
    context.putImageData(pixels, 0, 0);
    add(svg, "image", {
      href: canvas.toDataURL(),
      x: box.left,
      y: box.top,
      width: box.right - box.left,
      height: roundTo(rows.length * box.row, 2),
      preserveAspectRatio: "none",
      class: "orders-demo-heat-cells",
    });
    const lines = add(svg, "g", { class: "orders-demo-heat-rows" });
    rows.forEach(([players], r) => {
      const y = box.top + r * box.row;
      add(lines, "line", { x1: box.left, x2: box.right, y1: roundTo(y + box.row, 2), y2: roundTo(y + box.row, 2) });
      const name = players.map((feature) => example.features[feature]).join(" × ");
      add(svg, "text", { class: "orders-demo-heat-label", x: box.left - 1.5, y: roundTo(y + box.row / 2, 2) }, name);
    });
  }

  // -- the sheets' names and fits -----------------------------------------------------------------------

  // the name of a sheet: on its face, as its row's tag, and in words
  const sheetName = (sheet) => {
    if (sheet.full) {
      return { face: `Full decomposition (${sheet.order}-${INDEX})`, tag: `${sheet.order}-${INDEX}`, name: "Full decomposition" };
    }
    if (sheet.order === 1) return { face: "Shapley values (SV)", tag: "SV", name: "Shapley values" };
    const tag = `${sheet.order}-${INDEX}`;
    return { face: tag, tag, name: "Shapley interactions" };
  };
  // R^2 as text: two decimals, three close to 1, and 1.00 only where the fit is exact
  // an explanation that reproduces the model exactly (the data rounds R^2 to nine decimals): only
  // that reads 1.00 and exact, and anything short of it at most > 0.999
  const EXACT = 1 - 5e-10;
  const fit2 = (r2) => {
    if (r2 >= EXACT) return "1.00";
    if (r2 < 0.995) return r2.toFixed(2);
    return r2 >= 0.9995 ? "> 0.999" : r2.toFixed(3);
  };
  // how well an explanation fits: exact, very good from R^2 = 0.95, good from 0.8, workable from
  // 0.6, else poor
  const verdict = (r2) => {
    if (r2 >= EXACT) return "exact";
    if (r2 >= 0.95) return "very good";
    if (r2 >= 0.8) return "good";
    if (r2 >= 0.6) return "workable";
    return "poor";
  };
  // the players of an example, in words
  const PLAYERS = {
    graph: { one: "atom", players: "atoms" },
    tree: { one: "feature", players: "features" },
    heatmap: { one: "feature", players: "features" },
    "image-text": { one: "patch or word", players: "patches and words" },
  };
  // an example whose full decomposition is estimated (from a proxy of the model): it falls short of
  // R^2 = 1, the estimation gap
  const estimated = (example) => example.fits.at(-1)[1] < EXACT;
  // the number of players: the features of a house, or what every sheet has a main effect of
  const playersOf = (example) => example.features?.length ?? example.sheets[0].first.length;
  // a house's value in $100,000, as a change in thousands of dollars
  const dollars = (value) => `${value < 0 ? "−" : "+"}$${Math.round(Math.abs(value) * 100)}k`;

  // the fit of an order (order, R^2, number of values): the plot's, or, above the highest order
  // with an interaction, the full decomposition's, which every such order equals
  const fitOf = (example, order) => {
    const [, r2, values] = example.fits.find(([k]) => k === order) ?? example.fits.at(-1);
    return [order, r2, values];
  };

  // the orders between an example's last sheet of interactions and the full decomposition with a
  // blank sheet each: those of its plot, and more up to BLANKS, which equal the full decomposition
  const between = (example) => {
    const [first, last] = [example.sheets.at(-2).order + 1, example.sheets.at(-1).order - 1];
    const plotted = example.fits.filter(([order]) => order >= first && order <= last).length;
    const count = Math.max(0, Math.min(last - first + 1, Math.max(BLANKS, plotted)));
    return Array.from({ length: count }, (_, k) => fitOf(example, first + k));
  };

  // -- the demo -------------------------------------------------------------------------------------

  // the explanations' index, as the sheets' tags name it: FSII (2-FSII), or k-SII (2-SII)
  let INDEX = "SII";
  let plots = 0; // the demos' plots on the page, for ids of their own

  function start(root, data) {
    INDEX = data.index === "k-SII" || !data.index ? "SII" : data.index;
    // what is pointed at: a sheet slid out (its index), or the blank sheet of an order between
    const state = { example: data.examples[0].id, active: null, blank: null, pinned: false };
    let geometry = null;
    const plotId = ++plots;
    // the views (data-view): the stack ("stack", the default, as the docs show it), its sheets
    // stacked, the one pointed at sliding out of it; and two of floating layers, the input on a card,
    // the explanations on layers above it, and below the card, apart from the full decomposition,
    // the mechanism they explain. Unfolded ("unfolded"), the layers float apart, and the one pointed
    // at slides out, as the stack's sheets do, its input showing under its marks. Projected
    // ("projected"), they lie on the card at rest, the mechanism hidden under it, and unfold, float
    // off it, once explained (or after a tap on the card), or while a layer is pointed at, which
    // casts its marks onto the card
    const layered = () => root.dataset.view === "unfolded" || root.dataset.view === "projected";
    const projecting = () => root.dataset.view === "projected";
    // the demo starts with the problem alone (unless data-reveal="open"): a click on the model's chip
    // explains it, and the explanation is built under it, from the top down; once explained, every
    // example is
    let revealed = root.dataset.reveal === "open";
    // the floating layers unfold for good, once: after the model was clicked to explain it, once they
    // have come in and are in view (or tapped at rest); before, only while one is pointed at
    let unfoldedForGood = false;
    const unfolded = () => !projecting() || unfoldedForGood || state.active !== null || state.blank !== null; // the layers, apart

    root.replaceChildren();
    const bar = el("div", "orders-demo-bar");
    const switcher = el("div", "orders-demo-switch");
    switcher.setAttribute("role", "group");
    switcher.setAttribute("aria-label", "Example");
    // the pill around the example chosen, which slides to the next one chosen: a copy of the
    // examples over them, in the colors of the one chosen, cut to the pill, so that every label
    // turns as the pill passes over it
    const pill = el("div", "orders-demo-pill");
    pill.setAttribute("aria-hidden", "true");
    const label = (example, node) => {
      node.append(icon(example.model.icon, "orders-demo-option-icon"), el("span", "orders-demo-option-label", example.label));
      return node;
    };
    const options = data.examples.map((example) => {
      const option = label(example, el("button", "orders-demo-option"));
      option.type = "button";
      option.addEventListener("click", () => choose(example));
      switcher.append(option);
      pill.append(label(example, el("span", "orders-demo-option")));
      return [example.id, option];
    });
    switcher.append(pill);
    // where the examples would not fit in a row (demo.css): a stepper through them instead, the
    // example shown between a button to the previous and one to the next (a swipe across the stack
    // steps too)
    const stepper = el("div", "orders-demo-stepper");
    stepper.setAttribute("role", "group");
    stepper.setAttribute("aria-label", "Example");
    const stepButton = (delta, name, d) => {
      const button = el("button", "orders-demo-step");
      button.type = "button";
      button.setAttribute("aria-label", name);
      add(add(button, "svg", { viewBox: "0 0 16 16", "aria-hidden": "true" }), "path", { d });
      button.addEventListener("click", () => step(delta));
      return button;
    };
    const shown = el("span", "orders-demo-step-shown");
    shown.setAttribute("aria-live", "polite");
    stepper.append(stepButton(-1, "Previous example", "M10 3.5 5.5 8l4.5 4.5"), shown, stepButton(1, "Next example", "M6 3.5l4.5 4.5L6 12.5"));
    bar.append(switcher, stepper);

    // above the stack, the plan: first the problem, as machine learning draws it, the input x wired
    // through the model f, a chip, to its prediction f(x); then, cast out of the chip, why f(x): the
    // input seen from above, flat, with the explanation pointed at on it (at rest, the top one), as
    // the sheets at their slant cannot show it, the orders to explain it at, and in a few words what
    // that explanation is and how well it fits. The chip clears as far as the explanation pointed at
    // fits the model: it opens the black box, all the way where it is exact
    const plan = el("figure", "orders-demo-plan");
    const flow = el("div", "orders-demo-flow");
    flow.setAttribute("aria-live", "polite");
    const cast = document.createElementNS(SVG, "svg"); // the light out of the chip
    cast.setAttribute("class", "orders-demo-cast");
    cast.setAttribute("aria-hidden", "true");
    const castLight = add(add(cast, "defs"), "linearGradient", { id: `orders-cast-${plotId}`, x1: 0, y1: 0, x2: 0, y2: 1 });
    add(castLight, "stop", { offset: 0, class: "orders-demo-cast-top" });
    add(castLight, "stop", { offset: 1, class: "orders-demo-cast-bottom" });
    const castHull = add(cast, "polygon", { fill: `url(#orders-cast-${plotId})` });
    const castEdges = add(cast, "path", { class: "orders-demo-cast-edges" });
    const planView = el("div", "orders-demo-plan-view");
    planView.setAttribute("aria-hidden", "true");
    const planText = el("figcaption", "orders-demo-plan-text");
    const why = el("div", "orders-demo-why");
    why.append(planView, planText);
    plan.append(flow, cast, why);
    // the problem's parts, made once: as the examples change, they stay where they are, and only what
    // they show changes, flipping in as on a board of departures (the input, the model's chip, the
    // prediction), their words running through letters before they settle
    const tagged = (className, symbol, ...parts) => {
      const node = el("div", className);
      const tag = el("span", "orders-demo-wire-tag");
      tag.append(math(symbol));
      node.append(tag, ...parts);
      return node;
    };
    const shows = () => el("span", "orders-demo-wire-shows");
    const [xFace, xName, xShows] = [el("span", "orders-demo-wire-face"), el("span", "orders-demo-wire-name"), shows()];
    xShows.append(xFace, xName);
    const [chipIcon, chipName, chipNote, chipShows] = [el("span", "orders-demo-wire-icon-slot"), el("strong"), el("span", "orders-demo-chip-note"), shows()];
    chipShows.append(chipIcon, chipName, chipNote);
    const [outValue, outNote, outShows] = [el("strong"), el("span", "orders-demo-wire-note"), shows()];
    outShows.append(outValue, outNote);
    const chip = tagged("orders-demo-wire-chip", "f", chipShows);
    const explainTab = el("span", "orders-demo-chip-explain", "Explain");
    const into = el("div", "orders-demo-wire-side orders-demo-wire-into");
    into.append(tagged("orders-demo-wire-x", "x", xShows), el("span", "orders-demo-wire-line"));
    const onto = el("div", "orders-demo-wire-side orders-demo-wire-onto");
    onto.append(el("span", "orders-demo-wire-line"), tagged("orders-demo-wire-out", "f(x)", outShows));
    flow.append(into, chip, onto);
    // unexplained, the chip is a button that explains the example
    chip.addEventListener("click", () => {
      if (!revealed) reveal();
    });
    chip.addEventListener("keydown", (event) => {
      if (revealed || (event.key !== "Enter" && event.key !== " ")) return;
      event.preventDefault();
      reveal();
    });
    // the caption: what the explanation shown is, and how well it fits
    const whyName = el("span", "orders-demo-plan-eyebrow");
    whyName.append("Why ", math("f(x)"), "?");
    const planWords = el("span", "orders-demo-plan-words");
    planText.append(whyName, planWords);
    // its window: swiped aside, it goes out of view, not out of the page
    const planWindow = el("div", "orders-demo-plan-window");
    planWindow.append(plan);

    const stage = el("div", "orders-demo-stage");
    stage.setAttribute("role", "group");
    const stack = el("div", "orders-demo-stack");
    stack.setAttribute("aria-hidden", "true");
    const ladder = el("div", "orders-demo-ladder");
    // beside the stack, in a column of its own, the slider of the orders: an arrow down (every order
    // below is more complex, and fits the model more faithfully), a tick at every sheet, and a handle
    // that brings out the sheet at its tick; it rests above the top sheet. It serves touch screens,
    // where nothing hovers.
    const rail = el("div", "orders-demo-rail");
    rail.style.width = `${RAIL}px`;
    stage.style.setProperty("--orders-rail", `${RAIL}px`); // the stack is cut at the slider's column
    rail.style.setProperty("--orders-rail-line", `${RAIL_LINE}px`);
    const railLine = el("span", "orders-demo-rail-line");
    const railLabel = el("span", "orders-demo-rail-label");
    railLabel.append("more ", el("em", "", "complex"), " and more ", el("em", "", "faithful"));
    railLabel.setAttribute("aria-hidden", "true");
    const ticks = el("span", "orders-demo-rail-ticks");
    ticks.setAttribute("aria-hidden", "true");
    const thumb = el("span", "orders-demo-thumb");
    thumb.tabIndex = 0;
    thumb.setAttribute("role", "slider");
    thumb.setAttribute("aria-orientation", "vertical");
    thumb.setAttribute("aria-label", "Order of the explanation");
    rail.append(railLine, railLabel, ticks, thumb);
    // the stack's window: it starts at the slider's column, so that a stack swiped or slid aside goes
    // out of view there, not under the slider
    const view = el("div", "orders-demo-view");
    view.append(stack);
    // beside the sheets, the plot of the fit, upright
    const side = document.createElementNS(SVG, "svg");
    side.setAttribute("class", "orders-demo-side");
    side.setAttribute("role", "img");
    stage.append(rail, view, ladder, side);
    // its frame stays as the examples change (its grid, its ticks and the name of its axis), and what
    // it plots fades
    const sideFrame = add(side, "g", { class: "orders-demo-side-frame" });
    const sideData = add(side, "g", { class: "orders-demo-side-data" });

    // under the stack, where the stepper shows (demo.css): a dot for every example, the one shown dark
    const dots = el("div", "orders-demo-dots");
    dots.setAttribute("aria-hidden", "true");
    dots.append(...data.examples.map(() => el("span")));
    root.append(bar, planWindow, stage, dots);
    root.classList.add("orders-demo-ready");

    // the stack, top to bottom: the sheets of order 1 to 4, a blank sheet for every order up to
    // the full decomposition, and the sheet of the full decomposition
    let items = [];
    let rows = [];
    let stops = []; // the slider's: its rest, then every sheet and blank sheet, top to bottom
    let cardRow = null; // the floating layers': the input's row, beside its card

    const current = () => data.examples.find((example) => example.id === state.example);
    const sheetItems = () => items.filter((item) => item.sheet);
    const blankItems = () => items.filter((item) => !item.sheet && !item.base);
    // the slider's: no stop at the card or the mechanism; it goes through the explanations only, as
    // the plot does
    const stopItems = () => items.filter((item) => !item.base && !item.mechanism);

    // -- geometry: the size of the sheets, the stage, and every sheet's place ------------------------

    function measure() {
      const width = stage.clientWidth;
      const narrow = width < 560;
      // the plot beside the sheets: a narrow column, where the rows' lines lead
      const ladderWidth = Math.min(narrow ? 71 : 120, width * 0.17);
      const gap = narrow ? 10 : 28;
      const spread = Math.abs(Math.cos(TURN)) + Math.abs(Math.sin(TURN));
      const room = width - ladderWidth - gap - RAIL;
      // the room holds the stack and the sheet slid out of it
      const side = Math.max(110, Math.min(360, room / ((1 + SLIDE) * spread)));
      const sheetHeight = side * TILT * spread;
      // the same height for every example: the most sheets and blank sheets of any
      const n = Math.max(...data.examples.map((example) => example.sheets.length));
      const blanks = Math.max(...data.examples.map((example) => between(example).length));
      // the floating layers' card, below them, and the mechanism below it
      const air = layered() ? (AIR + DEPTH) * side : 0;
      const stack = sheetHeight + (n - 1) * STRIP * side + blanks * BLANK_STRIP * side + air;
      // a sheet slides out to the left: the stack lies right of the middle by half of that (of the
      // floating layers, none does)
      const slideBy = projecting() ? 0 : SLIDE * side * spread;
      return {
        width,
        narrow,
        side,
        slideBy,
        ladderLeft: width - ladderWidth,
        center: RAIL + room / 2 + slideBy / 2,
        height: Math.ceil(stack + 2 * THICKNESS + 16),
        sheetHeight,
      };
    }

    // the place (center) of every item: stacked so that a strip of every sheet shows below the one
    // above it (or, unfolded, as the floating layers lie), and the one pointed at slid out to the
    // side, a blank sheet pointed at peeking out below its strip
    function layout(active) {
      if (projecting()) return layeredLayout(unfolded());
      const g = geometry;
      const pointed = active === null ? null : sheetItems()[active];
      const rest = layered() ? layeredLayout(true) : stackLayout();
      return rest.map(({ x, y }, position) => {
        const item = items[position];
        return { x: x - (item === pointed ? g.slideBy : 0), y: y + (!item.sheet && !item.base && item.order === state.blank ? PEEK * g.side : 0) };
      });
    }
    // the stack's places at rest
    function stackLayout() {
      const g = geometry;
      const h = g.sheetHeight;
      const centers = [];
      let bottom = 0;
      items.forEach((item, position) => {
        if (position === 0) bottom = h;
        else bottom += (item.sheet ? STRIP : BLANK_STRIP) * g.side;
        centers.push(bottom - h / 2);
      });
      const offset = (g.height - bottom) / 2;
      return items.map((_, position) => ({ x: g.center, y: centers[position] + offset }));
    }

    // the floating layers' places: open, as the stack's sheets, the card a breath of air below the
    // lowest, and the mechanism deeper below it; at rest, every layer on the card, in the middle of
    // the stage
    function layeredLayout(open) {
      const g = geometry;
      const h = g.sheetHeight;
      const centers = [];
      let bottom = 0;
      items.forEach((item, position) => {
        if (position === 0) bottom = h;
        else bottom += (item.base ? AIR : item.deep ? DEPTH : item.sheet ? STRIP : BLANK_STRIP) * g.side;
        centers.push(bottom - h / 2);
      });
      const offset = (g.height - bottom) / 2;
      // at rest, the mechanism lies just under the card, its edge showing below the card's
      return items.map((item, position) => ({ x: g.center, y: open ? centers[position] + offset : g.height / 2 + (item.deep ? 2 * THICKNESS : 0) }));
    }

    // the corners of a sheet on the stage (left, top, right, bottom for the turn of -45 degrees)
    const corners = ({ x, y }) => {
      const a = geometry.side / 2;
      return [
        [-a, -a],
        [a, -a],
        [a, a],
        [-a, a],
      ].map(([u, v]) => [
        x + (u * Math.cos(TURN) - v * Math.sin(TURN)),
        y + TILT * (u * Math.sin(TURN) + v * Math.cos(TURN)),
      ]);
    };
    const rightCorner = (place) => corners(place).reduce((a, b) => (b[0] > a[0] ? b : a));

    const inside = ([px, py], polygon) => {
      let sign = 0;
      for (let k = 0; k < polygon.length; k++) {
        const [x1, y1] = polygon[k];
        const [x2, y2] = polygon[(k + 1) % polygon.length];
        const cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1);
        if (cross !== 0) {
          if (sign && Math.sign(cross) !== sign) return false;
          sign = Math.sign(cross);
        }
      }
      return true;
    };

    // the order of the sheets on top of each other, as stacked: a sheet slid out keeps its place
    const zIndex = (item) => items.length - items.indexOf(item);
    // the floating layers': the card under its layers, the mechanism under the card, and the light
    // between them
    const layerZ = (item) => (item.base ? 2 : item.deep ? 0 : zIndex(item) + 3);

    // the item under a point of the stage, where the sheets are going (not where they are while
    // they move), the one on top first. While a sheet slides out, the sheets count where they lie
    // at rest, and the one slid out also where it shows: else the sheet would slide away from under
    // the pointer, hand it to the sheet below, and come back for it, over and over.
    function itemAt(point) {
      const z = layered() ? layerZ : zIndex;
      const order = items.map((_, position) => position).sort((a, b) => z(items[b]) - z(items[a]));
      if (projecting()) {
        const places = layeredLayout(true);
        const position = order.find((p) => inside(point, corners(places[p])));
        return position === undefined ? null : items[position];
      }
      const sliding = state.active !== null;
      const places = layout(null);
      if (sliding) {
        const out = items.indexOf(sheetItems()[state.active]);
        const above = order.slice(0, order.indexOf(out));
        if (
          inside(point, corners(layout(state.active)[out])) &&
          !above.some((p) => inside(point, corners(places[p])))
        ) {
          return items[out];
        }
      }
      const position = order.find((p) => inside(point, corners(places[p])));
      return position === undefined ? null : items[position];
    }

    // -- the examples ---------------------------------------------------------------------------

    // the pill around the example chosen; it slides there, unless instant (as when the buttons
    // change their size)
    function movePill(instant) {
      const option = new Map(options).get(state.example);
      const [top, left] = [option.offsetTop, option.offsetLeft];
      const right = pill.clientWidth - left - option.offsetWidth;
      const bottom = pill.clientHeight - top - option.offsetHeight;
      if (instant) pill.style.transition = "none";
      pill.style.clipPath = `inset(${top}px ${right}px ${bottom}px ${left}px round 999px)`;
      if (instant) {
        pill.getBoundingClientRect();
        pill.style.transition = "";
      }
    }

    // the stepper's example: its icon, its name, and where it is among the examples, coming in from
    // the side it was stepped to from
    function showStep(example, enter = true) {
      const position = data.examples.indexOf(example) + 1;
      shown.replaceChildren(
        icon(example.model.icon, "orders-demo-option-icon"),
        el("strong", "", example.label),
        el("span", "orders-demo-step-count", `${position} of ${data.examples.length}`),
      );
      [...dots.children].forEach((dot, k) => dot.classList.toggle("orders-demo-dot-on", k === position - 1));
      shown.classList.remove("orders-demo-step-enter");
      if (!enter) return;
      shown.getBoundingClientRect(); // restart the entrance
      shown.classList.add("orders-demo-step-enter");
    }
    // step to the next example (1) or the previous one (-1)
    function step(delta) {
      const count = data.examples.length;
      const at = data.examples.findIndex((example) => example.id === state.example);
      stepper.style.setProperty("--orders-step-from", `${delta * 12}px`);
      choose(data.examples[(at + delta + count) % count], delta);
    }

    // choose an example: the pill slides to it while the stack, its fits and the plot fade out, and
    // the example fades in, at rest. Stepped to, the stack and the plot are swiped instead: they
    // leave to one side, out of view (from where a finger let go of them), and the example comes in
    // from the other. Without motion, it shows at once
    let chosen = 0; // the examples chosen so far: only the last one chosen is drawn
    const shift = (pixels) => root.style.setProperty("--orders-shift", `${roundTo(pixels, 1)}px`);
    function choose(example, direction = 0) {
      if (state.example === example.id) return;
      Object.assign(state, { example: example.id, active: null, blank: null, pinned: false });
      for (const [id, option] of options) option.setAttribute("aria-pressed", String(id === example.id));
      movePill(false);
      showStep(example);
      // the plan stays: what it shows flips to the example at once, the card after the problem
      showProblem(example, true);
      showPlan();
      [planView, planText].forEach((node) => flap(node, 300));
      const choice = ++chosen;
      if (reducedMotion()) {
        shift(0);
        root.style.removeProperty("--orders-swipe-opacity");
        render();
        return;
      }
      root.classList.add("orders-demo-fading");
      root.classList.toggle("orders-demo-leaving", direction !== 0);
      shift(-direction * stage.clientWidth);
      // the fade out starts now, and the example is drawn when it ends (at once, if all is faded)
      const fades = [stack, ladder, sideData].flatMap((node) => {
        getComputedStyle(node).opacity;
        return node.getAnimations();
      });
      Promise.allSettled(fades.map((fade) => fade.finished)).then(() => {
        if (choice !== chosen) return;
        render();
        // stepped to, the new stack and plot come in from the side: placed there at once, then
        // sliding home
        root.classList.add("orders-demo-jump");
        shift(direction * stage.clientWidth);
        if (direction) root.style.setProperty("--orders-swipe-opacity", "0.6");
        root.getBoundingClientRect();
        root.classList.remove("orders-demo-jump", "orders-demo-leaving", "orders-demo-fading");
        shift(0);
        root.style.removeProperty("--orders-swipe-opacity");
      });
    }

    // -- drawing --------------------------------------------------------------------------------

    // a sheet of an example: its face, and its name along the bottom edge, from the bottom corner
    // (the part of a sheet that shows)
    // (all of it, or, for a floating layer, its marks, or for their card, the input)
    function drawSheet(example, i, layer = "all") {
      const node = el("div", layer === "marks" ? "orders-demo-sheet orders-demo-layer" : "orders-demo-sheet");
      const svg = add(node, "svg", { viewBox: "0 0 100 100", class: "orders-demo-face" });
      const sheet = example.sheets[i];
      drawFace(svg, example, sheet, layer, `orders-${plotId}-${example.id}-${layer}-${i}`);
      add(svg, "text", { class: "orders-demo-tag", x: 6, y: 95.5 }, layer === "input" ? "Input" : sheetName(sheet).face);
      return node;
    }
    // a face of an example's kind, into svg (id names what it defines, as the photo's clip)
    function drawFace(svg, example, sheet, layer, id) {
      if (example.kind === "graph") moleculeFace(svg, example, sheet, layer);
      else if (example.kind === "tree") treeFace(svg, example, sheet, layer);
      else if (example.kind === "heatmap") heatmapFace(svg, example, sheet, layer);
      else imageTextFace(svg, example, sheet, `${root.dataset.root}${example.image}`, id, layer);
    }
    // every example's sheets, drawn once: an example chosen again shows them as they were drawn (the
    // floating layers apart from the stack's sheets)
    const drawn = new Map();
    function sheetsOf(example) {
      const layer = layered() ? "marks" : "all";
      const key = `${example.id}${layered() ? " layers" : ""}`;
      const nodes = drawn.get(key) ?? [];
      for (let i = nodes.length; i < example.sheets.length; i++) nodes.push(drawSheet(example, i, layer));
      drawn.set(key, nodes);
      return nodes;
    }
    // unfolded, a layer slid out shows its input under its marks: drawn the first time it is needed
    function underOf(example, node, i) {
      if (node.querySelector(".orders-demo-under")) return;
      const svg = document.createElementNS(SVG, "svg");
      svg.setAttribute("viewBox", "0 0 100 100");
      svg.setAttribute("class", "orders-demo-under");
      drawFace(svg, example, example.sheets[i], "input", `orders-${plotId}-${example.id}-under-${i}`);
      node.prepend(svg);
    }
    // the floating layers' card: the input, and over it, the marks of the layer pointed at, projected
    const cards = new Map();
    function cardOf(example) {
      if (!cards.has(example.id)) {
        const card = drawSheet(example, 0, "input");
        card.classList.add("orders-demo-card");
        card.append(el("div", "orders-demo-projection"));
        cards.set(example.id, card);
      }
      return cards.get(example.id);
    }
    // the mechanism, below the card: the full decomposition, all of it (never what it adds), named as
    // the mechanism; where it is estimated, blurred, as only an estimate shows it
    const mechanisms = new Map();
    function mechanismOf(example) {
      if (!mechanisms.has(example.id)) {
        const node = drawSheet(example, example.sheets.length - 1, "marks");
        node.querySelector(".orders-demo-tag").textContent = estimated(example)
          ? "The mechanism, estimated"
          : `The mechanism (${sheetName(example.sheets.at(-1)).tag})`;
        node.classList.toggle("orders-demo-estimated", estimated(example));
        mechanisms.set(example.id, node);
      }
      return mechanisms.get(example.id);
    }
    // the plan's faces, drawn once: a sheet's, all of it, on the input; or the input alone (null)
    const plans = new Map();
    function planFace(example, index) {
      const key = `${example.id} ${index}`;
      if (!plans.has(key)) {
        const svg = document.createElementNS(SVG, "svg");
        svg.setAttribute("viewBox", "0 0 100 100");
        svg.setAttribute("class", "orders-demo-plan-face");
        drawFace(svg, example, example.sheets[index ?? 0], index === null ? "input" : "all", `orders-${plotId}-${example.id}-plan-${index}`);
        plans.set(key, svg);
      }
      return plans.get(key);
    }
    // the problem: the input, a wire into the model's chip (dark for a black box, gray for a model
    // whose structure the explanation uses), and a wire out of it to the prediction, in the red of
    // what raises it; the wires carry the input through, as dashes flowing. The chip lies in the
    // middle, the input and the prediction at either side
    let problemOf = null; // the example the problem shows
    // a word, at once, or (flipping) running through letters before it settles, from the left
    const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    function say(node, text, flip) {
      const run = (node.run ?? 0) + 1;
      node.run = run;
      if (!flip || reducedMotion()) {
        node.textContent = text;
        return;
      }
      const start = performance.now();
      const duration = 380 + 18 * text.length;
      const settle = [...text].map((_, k) => 120 + (k / text.length) * (duration - 160) + 60 * Math.random());
      const frame = (now) => {
        if (node.run !== run) return;
        const t = now - start;
        node.textContent = [...text]
          .map((letter, k) => (t >= settle[k] || letter === " " ? letter : LETTERS[Math.floor(Math.random() * LETTERS.length)]))
          .join("");
        if (t < duration) requestAnimationFrame(frame);
        else if (node.closest(".orders-demo-wire-shows")) fitPart(node.closest(".orders-demo-wire-shows"));
      };
      requestAnimationFrame(frame);
    }
    // a part's new face, flipping down into its place as a flap does (from edge on, so that the old
    // one is gone at once), after those to its left
    const flap = (node, delay) =>
      reducedMotion() ||
      node.animate([{ transform: "rotateX(-90deg)" }, { transform: "none" }], {
        duration: 320,
        delay,
        easing: "cubic-bezier(0.22, 1, 0.36, 1)",
        fill: "backwards",
      });
    // the problem: the input, the model's chip (black for a black box, gray for a model whose
    // structure the explanation uses), and the prediction, in the red of what raises it
    function showProblem(example, flip = false) {
      problemOf = example.id;
      const { problem, model } = example;
      xFace.className = example.kind === "tree" ? "orders-demo-wire-face orders-demo-wire-table" : "orders-demo-wire-face";
      xFace.replaceChildren(...inputFitting(example));
      say(xName, problem.input, flip);
      chip.classList.toggle("orders-demo-chip-black", model.icon === "box");
      chip.classList.toggle("orders-demo-chip-gray", model.icon !== "box");
      chip.title = `${model.title}: ${model.detail}`;
      chipIcon.replaceChildren(icon(model.icon, "orders-demo-wire-icon"));
      say(chipName, model.short, flip);
      chipNote.textContent = model.title;
      say(outValue, problem.output, flip);
      outNote.textContent = problem.value;
      for (const node of [xShows, chipShows, outShows]) fitPart(node);
      if (flip) [xShows, chipShows, outShows].forEach((node, k) => flap(node, k * 90));
    }
    // a part's face, scaled down about its middle where it would not fit its box (a house's table,
    // a long prediction), as all three boxes have one size
    function fitPart(node) {
      node.style.scale = "";
      const box = node.parentElement;
      const style = getComputedStyle(box);
      const width = box.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
      const height = box.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
      if (!width || !node.scrollWidth) return; // not shown yet
      const factor = Math.min(1, width / node.scrollWidth, height / node.scrollHeight);
      if (factor < 0.99) node.style.scale = roundTo(factor, 3);
    }
    // unexplained, the chip calls to be clicked
    function showCall() {
      const model = current().model;
      chip.classList.toggle("orders-demo-chip-call", !revealed);
      if (revealed) {
        for (const name of ["role", "tabindex", "aria-label"]) chip.removeAttribute(name);
        explainTab.remove();
      } else {
        chip.setAttribute("role", "button");
        chip.setAttribute("aria-label", `${model.title}, ${model.detail}: explain its prediction`);
        chip.tabIndex = 0;
        chip.append(explainTab);
      }
    }
    // the input, without marks, for the problem's designs, drawn once per example and size: a graph,
    // or a photo and its caption, as its face; a house as a table of its features, and many houses
    // as a table of houses by features (small, either as the glyph of a table)
    const inputs = new Map();
    function inputOf(example, small = false) {
      const key = `${example.id} ${small}`;
      if (!inputs.has(key)) inputs.set(key, drawInput(example, small));
      return inputs.get(key);
    }
    // a house's table, and for a phone, where it would not fit, the glyph of a table (demo.css)
    const inputFitting = (example) => {
      if (example.kind !== "tree") return [inputOf(example)];
      const [large, small] = [el("span", "orders-demo-input-large"), el("span", "orders-demo-input-small")];
      large.append(inputOf(example));
      small.append(inputOf(example, true));
      return [large, small];
    };
    function drawInput(example, small) {
      const svg = document.createElementNS(SVG, "svg");
      svg.setAttribute("viewBox", "0 0 100 100");
      svg.setAttribute("class", "orders-demo-input-face");
      const tabular = example.kind === "tree" || example.kind === "heatmap";
      if (tabular && small) {
        svg.classList.add("orders-demo-input-glyph");
        add(svg, "rect", { x: 14, y: 20, width: 72, height: 60, rx: 7 });
        add(svg, "path", { d: "M14 36H86M14 51H86M14 66H86M44 20V80" });
      } else if (example.kind === "tree") {
        const table = el("span", "orders-demo-input-table");
        example.features.forEach((name, k) => {
          const cell = el("span", "orders-demo-input-cell");
          cell.append(el("span", "", name), el("strong", "", example.values[k]));
          table.append(cell);
        });
        return table;
      } else if (example.kind === "heatmap") {
        // a head of the features, rows of houses, and how many
        svg.classList.add("orders-demo-input-matrix");
        const [left, top, cell] = [10, 10, 10];
        for (let row = 0; row < 6; row++) {
          for (let column = 0; column < 8; column++) {
            add(svg, "rect", { x: left + column * cell, y: top + row * cell, width: cell - 1.5, height: cell - 1.5, rx: 1.2, class: row ? "" : "orders-demo-input-head" });
          }
        }
        add(svg, "text", { x: 50, y: 86, "text-anchor": "middle" }, `${example.predictions.length} × ${example.features.length}`);
      } else {
        drawFace(svg, example, example.sheets[0], "input", `orders-${plotId}-${example.id}-input`);
      }
      return svg;
    }
    // 2 to the power of n, as text
    const power = (n) => {
      const node = el("span", "orders-demo-power", "2");
      node.append(el("sup", "", String(n)));
      return node;
    };
    // the plan shows the sheet pointed at (all of it, where the sheets show what they add), the
    // mechanism's as the full decomposition, and an order between as the full decomposition where
    // it equals it, else as the input alone; at rest, the top sheet
    // what an explanation is, in a few words, how well it fits, and how it was computed: a sheet's
    // (its index), an order's between the sheets (blank), or the mechanism's
    function wordsOf(example, { index = 0, blank = null, mechanism = false }) {
      const words = PLAYERS[example.kind];
      // a line each, whatever it shows, so that none moves: its name, what it holds, its bar and fit,
      // what it was explained with, and what red and blue mean
      const title = el("span", "orders-demo-plan-title");
      const what = el("span", "orders-demo-plan-what");
      const fit = el("span", "orders-demo-plan-fit");
      const full = example.sheets.at(-1);
      let r2 = full.r2;
      if (mechanism) {
        const n = playersOf(example);
        title.append(el("strong", "", "Mechanism"), " the model itself");
        if (estimated(example)) what.append("up to ", power(n), " terms");
        else what.append(`${full.count.toLocaleString("en")} of `, power(n), " terms");
      } else if (blank !== null) {
        r2 = fitOf(example, blank)[1];
        title.append(el("strong", "", `Order ${blank}`), " between the sheets");
        what.append(blank >= example.largest ? "as the full decomposition" : `groups of up to ${blank} ${words.players}`);
      } else {
        const sheet = example.sheets[index];
        r2 = sheet.r2;
        title.append(el("strong", "", sheetName(sheet).tag), ` ${sheetName(sheet).name}`);
        what.append(
          sheet.order === 1
            ? `each ${words.one} alone`
            : !sheet.full
              ? `groups of up to ${sheet.order} ${words.players}`
              : `groups of any size, ${estimated(example) ? "estimated" : "exact"}`,
        );
      }
      // and a bar by its R^2
      const meter = el("span", "orders-demo-meter");
      const fill = el("span", "orders-demo-meter-fill");
      fill.style.width = `${Math.max(0, Math.min(1, r2)) * 100}%`;
      meter.append(fill);
      fit.append(meter);
      fit.append(el("span", "orders-demo-r2", fit2(r2)), el("span", "orders-demo-verdict", verdict(r2)));
      const method = el("span", "orders-demo-plan-method");
      method.append("Explained with ", el("strong", "", example.model.method), el("span", "orders-demo-plan-how", `, ${example.model.how}`));
      const legend = el("span", "orders-demo-plan-legend");
      for (const [color, meaning] of [["up", "raises"], ["down", "lowers"]]) {
        const key = el("span", "orders-demo-plan-key");
        key.append(el("i", `orders-demo-plan-${color}`), meaning);
        legend.append(key);
      }
      legend.append(math("f(x)"));
      return { nodes: [title, what, fit, method, legend], r2 };
    }

    // the card keeps its place and its size whatever it shows, in every example: its words take a
    // column of their own width, as tall as the most words it can show (measured out of view, for
    // the column's width and the view)
    let settled = "";
    function settleCard() {
      const width = planText.clientWidth;
      const key = `${width} ${layered()}`;
      if (!width || key === settled) return;
      settled = key;
      const probe = el("span", "orders-demo-plan-text orders-demo-plan-probe");
      const words = el("span", "orders-demo-plan-words");
      probe.style.width = `${width}px`;
      probe.append(whyName.cloneNode(true), words);
      why.append(probe);
      // every line as it says the most, at full size: where one is wider than the column (on a
      // narrow screen), all the words shrink alike, the same whatever the card shows; but what an
      // explanation holds, the longest line, takes two lines instead, where it would not fit one
      words.style.setProperty("--orders-words-fit", "1");
      let [widest, widestWhat] = [0, 0];
      for (const example of data.examples) {
        const picks = [
          ...example.sheets.map((_, index) => ({ index })),
          ...between(example).map(([blank]) => ({ blank })),
          ...(layered() ? [{ mechanism: true }] : []),
        ];
        for (const pick of picks) {
          words.replaceChildren(...wordsOf(example, pick).nodes);
          for (const line of words.children) {
            if (line.classList.contains("orders-demo-plan-what")) widestWhat = Math.max(widestWhat, line.scrollWidth);
            else widest = Math.max(widest, line.scrollWidth);
          }
        }
      }
      const fit = Math.min(1, (width - 1) / widest);
      const two = widestWhat * fit > width - 1;
      for (const node of [planWords, words]) {
        node.style.setProperty("--orders-words-fit", roundTo(fit, 3));
        node.classList.toggle("orders-demo-plan-two", two);
      }
      // and as tall as its words, shrunk so, are (every explanation's have as many lines)
      planText.style.minHeight = `${Math.ceil(probe.offsetHeight)}px`;
      probe.remove();
    }

    function showPlan() {
      const example = current();
      const last = example.sheets.length - 1;
      const pointed = state.active !== null ? sheetItems()[state.active] : null;
      let index = pointed ? Math.min(pointed.index, last) : 0;
      if (state.blank !== null) index = state.blank >= example.largest ? last : null;
      const face = planFace(example, index);
      if (planView.firstChild !== face) planView.replaceChildren(face);
      // the mechanism, where only an estimate shows it, blurred
      planView.classList.toggle("orders-demo-estimated", Boolean(pointed?.mechanism) && estimated(example));

      const { nodes, r2 } = wordsOf(example, pointed?.mechanism ? { mechanism: true } : state.blank !== null ? { blank: state.blank } : { index });
      planWords.replaceChildren(...nodes);

      // the chip clears as far as the explanation pointed at fits (at rest, none is): opaque, it is
      // the black box; clear, the explanation has opened it
      const clear = pointed || state.blank !== null ? Math.max(0, Math.min(1, r2)) : 0;
      flow.style.setProperty("--orders-clear", roundTo(clear, 3));
      plan.classList.toggle("orders-demo-plan-lit", pointed !== null || state.blank !== null);
      chip.classList.toggle("orders-demo-chip-open", clear >= 0.6);
      const note = pointed || state.blank !== null ? `R² ${fit2(r2)} explained` : example.model.title;
      if (chipNote.textContent !== note) {
        chipNote.textContent = note;
        fitPart(chipShows);
      }
    }

    // explain: the stage and the plot show, drawn anew for their room, and come in (demo.css), the
    // light out of the chip first, then the card, the sheets one after the other, the rows and the
    // plot
    let revealing = 0;
    function reveal() {
      if (revealed) return;
      revealed = true;
      render();
      if (!reducedMotion()) buildUp();
      // the floating layers unfold once they have come in, and are in view
      unfoldAfter = performance.now() + (reducedMotion() ? 300 : building + 500);
      unfold();
    }
    // the explanation built from the top down: a line runs down from the chip, and every part comes
    // as it passes, the card, the sheets one after the other with their rows, and the plot beside
    // them, drawn down with it (demo.css): each part's moment is how far below the chip it lies
    let building = 0;
    function buildUp() {
      const box = root.getBoundingClientRect();
      const from = flow.getBoundingClientRect().bottom - box.top;
      const length = Math.max(1, box.height - from);
      building = Math.round(Math.max(900, Math.min(2000, length * 2)));
      const at = (node) => {
        const part = node.getBoundingClientRect();
        const y = part.top - box.top + Math.min(part.height * 0.25, 40);
        return `${Math.round(Math.max(0, (y - from) / length) * building)}ms`;
      };
      for (const node of [why, rail, dots, side, ...items.map((item) => item.node), ...ladder.children]) {
        node.style.setProperty("--orders-at", at(node));
      }
      // the plot beside the sheets is drawn down as the line passes it
      side.style.setProperty("--orders-side-time", `${Math.round((stage.clientHeight / length) * building)}ms`);
      root.style.setProperty("--orders-scan-from", `${roundTo(from, 1)}px`);
      root.style.setProperty("--orders-scan-length", `${roundTo(length, 1)}px`);
      root.style.setProperty("--orders-scan-time", `${building}ms`);
      root.classList.remove("orders-demo-revealing");
      root.getBoundingClientRect(); // anew
      root.classList.add("orders-demo-revealing");
      clearTimeout(revealing);
      revealing = setTimeout(() => root.classList.remove("orders-demo-revealing"), building + 1200);
    }
    // the floating layers unfold, for good, once explained, once they have come in, and once the
    // stage is in view (on a phone, scrolled to)
    let unfoldAfter = Infinity;
    let unfolding = 0;
    let stageInView = false;
    new IntersectionObserver(
      ([entry]) => {
        stageInView = entry.isIntersecting;
        unfold();
      },
      { threshold: 0.45 },
    ).observe(stage);
    function unfold() {
      clearTimeout(unfolding);
      if (!projecting() || unfoldedForGood || !revealed || !stageInView) return;
      const wait = unfoldAfter - performance.now();
      if (wait > 0) {
        unfolding = setTimeout(unfold, wait);
        return;
      }
      unfoldedForGood = true;
      place(state.active);
    }

    // the light out of the chip, onto the card of the explanation: from the chip's bottom corners to
    // the card's top ones
    function castLightOut() {
      const box = plan.getBoundingClientRect();
      const [from, to] = [chip.getBoundingClientRect(), why.getBoundingClientRect()];
      const f = (x, y) => `${roundTo(x - box.left, 1)},${roundTo(y - box.top, 1)}`;
      // halfway round the rounded corners of the chip's bottom and of the card's top
      const round = (node) => parseFloat(getComputedStyle(node).borderTopLeftRadius) * (1 - Math.SQRT1_2);
      const [a, b] = [round(chip), round(why)];
      const corners = [f(from.left + a, from.bottom - a), f(from.right - a, from.bottom - a), f(to.right - b, to.top + b), f(to.left + b, to.top + b)];
      cast.setAttribute("width", roundTo(box.width, 1));
      cast.setAttribute("height", roundTo(box.height, 1));
      castHull.setAttribute("points", corners.join(" "));
      castEdges.setAttribute("d", `M${corners[0]}L${corners[3]}M${corners[1]}L${corners[2]}`);
    }

    // the light between the layer pointed at and the card
    const beam = document.createElementNS(SVG, "svg");
    beam.setAttribute("class", "orders-demo-beam");
    beam.setAttribute("aria-hidden", "true");
    const glow = add(add(beam, "defs"), "linearGradient", { id: `orders-beam-${plotId}`, x1: 0, y1: 0, x2: 0, y2: 1 });
    add(glow, "stop", { offset: 0, class: "orders-demo-beam-top" });
    add(glow, "stop", { offset: 1, class: "orders-demo-beam-bottom" });
    const beamHull = add(beam, "polygon", { fill: `url(#orders-beam-${plotId})` });
    const beamEdges = add(beam, "path", { class: "orders-demo-beam-edges" });
    // the examples not shown yet, drawn while the page is idle, a sheet at a time, so that choosing
    // one only has to show it
    const idle = window.requestIdleCallback ?? ((callback) => setTimeout(callback, 100));
    function drawAhead() {
      const example = data.examples.find((other) => (drawn.get(other.id)?.length ?? 0) < other.sheets.length);
      if (!example) return;
      const nodes = drawn.get(example.id) ?? [];
      nodes.push(drawSheet(example, nodes.length));
      drawn.set(example.id, nodes);
      idle(drawAhead);
    }

    function render() {
      const example = current();
      root.classList.toggle("orders-demo-veiled", !revealed);
      Object.assign(state, { active: null, blank: null, pinned: false }); // a new example shows at rest
      for (const [id, option] of options) option.setAttribute("aria-pressed", String(id === example.id));
      stage.setAttribute("aria-label", `${example.label}: its explanations of order 1 to n, stacked, with their fit`);
      stack.replaceChildren();
      for (const row of [...rows, cardRow]) row?.remove();
      if (problemOf !== example.id) showProblem(example);
      showCall();
      geometry = measure();

      const last = example.sheets.at(-1);
      items = [];
      const nodes = sheetsOf(example);
      example.sheets.forEach((sheet, i) => {
        if (sheet === last) {
          for (const fit of between(example)) items.push({ order: fit[0], fit });
        }
        items.push({ order: sheet.order, sheet, index: i, node: nodes[i] });
      });
      for (const item of items) {
        if (!item.sheet) item.node = el("div", layered() ? "orders-demo-sheet orders-demo-blank orders-demo-layer" : "orders-demo-sheet orders-demo-blank");
      }
      if (layered()) {
        // below the card, the mechanism: a sheet of its own, after the example's sheets
        const full = example.sheets.at(-1);
        const card = { order: 0, base: true, node: cardOf(example) };
        items.push(card, { order: full.order, sheet: full, index: example.sheets.length, mechanism: true, deep: true, node: mechanismOf(example) });
      }
      // unfolded, every layer can slide out, the mechanism too: each with its input under its marks
      if (layered() && !projecting()) {
        for (const item of items) if (item.sheet) underOf(example, item.node, Math.min(item.index, example.sheets.length - 1));
      }
      for (const item of items) item.node.classList.toggle("orders-demo-deep", Boolean(item.deep));

      // a row points at its sheet: hovered, focused or tapped, and the arrow keys go to the next row
      const wire = (row, i) => {
        row.addEventListener("pointerenter", (event) => {
          if (event.pointerType === "mouse" && !state.pinned) point(i, null);
        });
        row.addEventListener("focus", () => {
          if (!state.pinned) point(i, null);
        });
        row.addEventListener("click", () => toggle(i, null));
        row.addEventListener("keydown", (event) => {
          const step = { ArrowDown: 1, ArrowUp: -1 }[event.key];
          if (step) {
            event.preventDefault();
            rows[(i + step + rows.length) % rows.length].focus();
          } else if (event.key === "Escape") {
            state.pinned = false;
            point(null, null);
          }
        });
      };
      // a sheet's row: its line on to its point in the plot, and a button for the keyboard (the card
      // above says what the row would)
      rows = example.sheets.map((sheet, i) => {
        const names = sheetName(sheet);
        const row = el("button", "orders-demo-row");
        row.type = "button";
        row.setAttribute("aria-label", `${names.name}, order ${sheet.order}: R² ${fit2(sheet.r2)}, ${verdict(sheet.r2)}`);
        wire(row, i);
        ladder.append(row);
        return row;
      });
      if (layered()) {
        // the mechanism's row, below the plot, which ends at the last explanation: its name (and, for
        // the keyboard, how complex it is, of how complex it could be, a term for every subset of the
        // players, which only an estimate shows)
        const i = rows.length;
        const full = example.sheets.at(-1);
        const n = playersOf(example);
        const row = el("button", "orders-demo-row orders-demo-mechanism-row");
        row.type = "button";
        row.setAttribute(
          "aria-label",
          `The mechanism, the model itself: ${estimated(example) ? `up to 2^${n} terms, estimated` : `${full.count.toLocaleString("en")} of 2^${n} possible terms`}`,
        );
        row.append(el("span", "orders-demo-row-label", "Mechanism"));
        wire(row, i);
        ladder.append(row);
        rows.push(row);
      }
      ticks.replaceChildren(...stopItems().map((item) => el("span", item.sheet ? "orders-demo-tick" : "orders-demo-tick orders-demo-tick-blank")));

      // the floating layers' row of their card: the input, what the layers explain, named
      cardRow = null;
      if (layered()) {
        cardRow = el("div", "orders-demo-row orders-demo-card-row");
        cardRow.setAttribute("aria-hidden", "true");
        cardRow.append(el("span", "orders-demo-row-label", "Input"));
        ladder.append(cardRow);
      }

      drawSide();
      showPlan();
      settleCard();
      castLightOut();
      if (revealed && unfoldAfter === Infinity) unfoldAfter = 0;
      unfold();
      // a new example shows at rest at once: its sheets and rows take their places without moving.
      // Its sheets join the stack only now, placed: measuring the plot and the rows would else style
      // and lay out every element of their faces, and placing them (their size, and the custom
      // properties of their edges, which every element inherits) would do so again
      root.classList.add("orders-demo-instant");
      place(null);
      stack.append(...items.map((item) => item.node));
      if (projecting()) stack.append(beam);
      root.getBoundingClientRect();
      root.classList.remove("orders-demo-instant");
    }

    // move every sheet and row to its place, for a sheet slid out (or null at rest)
    function place(active) {
      const g = geometry;
      stage.style.height = `${g.height}px`;
      root.classList.toggle("orders-demo-narrow", g.narrow);
      const out = typeof active === "number";
      const places = layout(active);
      // the projected layers' rows, slider and beam keep to where they float, open or at rest; the
      // others' follow their sheets
      const floating = layered();
      const anchors = projecting() ? layeredLayout(true) : places;
      const open = floating && unfolded();
      const opening = floating && open !== root.classList.contains("orders-demo-open");
      root.classList.toggle("orders-demo-floating", floating);
      root.classList.toggle("orders-demo-open", open);
      const pointedAt = out ? items.indexOf(sheetItems()[active]) : -1;
      items.forEach((item, position) => {
        const { x, y } = places[position];
        const node = item.node;
        node.style.width = node.style.height = `${g.side}px`;
        node.style.marginLeft = node.style.marginTop = `${-g.side / 2}px`;
        node.style.zIndex = floating ? layerZ(item) : zIndex(item);
        if (floating) {
          // the layers peel off the card one after the other, the top one first, and settle back
          // onto it the lowest first
          const delay = opening ? (open ? position : items.length - position) * 45 : 0;
          node.style.transitionDelay = `${delay}ms`;
          node.classList.toggle("orders-demo-top", position === 0);
        }
        node.style.transform = `translate(${roundTo(x, 2)}px, ${roundTo(y, 2)}px) scaleY(${TILT}) rotate(${(TURN * 180) / Math.PI}deg)`;
        // the edge below the face, in the face's own coordinates: straight down on the stage
        const depth = (item.sheet ? THICKNESS : THICKNESS / 2) / TILT;
        node.style.setProperty("--orders-edge-x", `${roundTo(Math.sin(TURN) * depth, 2)}px`);
        node.style.setProperty("--orders-edge-y", `${roundTo(Math.cos(TURN) * depth, 2)}px`);
        node.classList.toggle("orders-demo-receded", out && !item.base && item.index !== active);
        // (projected, the layers above the one pointed at, through which it shows, unless it is the
        // mechanism, under the card; unfolded, the one slid out, which shows its input under its
        // marks)
        node.classList.toggle("orders-demo-above", out && position < pointedAt && !items[pointedAt].deep);
        node.classList.toggle("orders-demo-out", out && !projecting() && item.index === active);
        node.classList.toggle("orders-demo-peek", !item.sheet && item.order === state.blank);

        if (item.base) {
          const right = rightCorner(anchors[position]);
          cardRow.style.left = `${g.ladderLeft}px`;
          cardRow.style.transform = `translateY(${roundTo(right[1], 2)}px) translateY(-50%)`;
          cardRow.style.setProperty("--orders-lead", `${Math.max(0, roundTo(g.ladderLeft - right[0] - 4, 2))}px`);
        }
        if (!item.sheet) return;
        const row = rows[item.index];
        const right = rightCorner(anchors[position]);
        row.style.left = `${g.ladderLeft}px`;
        row.style.transform = `translateY(${roundTo(right[1], 2)}px) translateY(-50%)`;
        row.style.setProperty("--orders-lead", `${Math.max(0, roundTo(g.ladderLeft - right[0] - 4, 2))}px`);
        row.classList.toggle("orders-demo-active", out && item.index === active);
        row.setAttribute("aria-pressed", String(state.pinned && out && item.index === active));
      });

      // the slider: its stops at the middles of the sheets at rest (without a blank sheet's peek), its
      // line from its rest down to the last explanation, and its handle at the sheet pointed at
      const rest = floating ? anchors : layout(null);
      stops = [
        { y: Math.max(10, rest[0].y - RAIL_REST), item: null },
        ...stopItems().map((item) => ({ y: rest[items.indexOf(item)].y - (!floating && item.order === state.blank ? PEEK * g.side : 0), item })),
      ];
      const [top, bottom] = [stops[0].y, stops.at(-1).y];
      railLine.style.transform = `translateY(${roundTo(top, 2)}px)`;
      railLine.style.height = `${roundTo(bottom - top, 2)}px`;
      // the arrow's label beside its middle
      railLabel.style.top = `${roundTo((stops[1].y + bottom) / 2, 2)}px`;
      [...ticks.children].forEach((tick, p) => {
        tick.style.transform = `translateY(${roundTo(stops[p + 1].y, 2)}px)`;
      });
      const at = Math.max(
        0,
        stops.findIndex(({ item }) =>
          item === null ? !out && state.blank === null : out ? item.index === active : !item.sheet && item.order === state.blank,
        ),
      );
      // placed by translate, which the handle's growing on hover does not scale, unlike a transform
      thumb.style.translate = `0 ${roundTo(stops[at].y, 2)}px`;
      thumb.classList.toggle("orders-demo-thumb-rest", at === 0);
      thumb.setAttribute("aria-valuemin", "0");
      thumb.setAttribute("aria-valuemax", String(stops.length - 1));
      thumb.setAttribute("aria-valuenow", String(at));
      thumb.setAttribute("aria-valuetext", at === 0 ? "none" : stopName(stops[at].item));

      if (floating) project(projecting() && out ? sheetItems()[active] : null, anchors);
    }

    // the light from the layer pointed at down to the card (or from the mechanism up to it), whose
    // input then wears the layer's marks (or none)
    const projected = new WeakMap(); // every card's projection: the node of the layer it wears
    function project(item, anchors) {
      const card = items.find((other) => other.base);
      beam.classList.toggle("orders-demo-beam-on", Boolean(item));
      // the light is over the card, under the layers; the mechanism's own rises under the card
      beam.style.zIndex = item?.deep ? 1 : 3;
      // brightest at the layer it comes from: above the card, or the mechanism's, amber, below it
      beam.classList.toggle("orders-demo-beam-deep", Boolean(item?.deep));
      glow.setAttribute("y1", item?.deep ? 1 : 0);
      glow.setAttribute("y2", item?.deep ? 0 : 1);
      const source = item?.node;
      if ((projected.get(card.node) ?? null) !== (source ?? null)) {
        const marks = source?.querySelector(".orders-demo-face").cloneNode(true);
        marks?.querySelector(".orders-demo-tag")?.remove();
        const projection = card.node.querySelector(".orders-demo-projection");
        projection.replaceChildren(...(marks ? [marks] : []));
        projection.classList.toggle("orders-demo-estimated", Boolean(item?.mechanism) && estimated(current()));
        projected.set(card.node, source ?? null);
      }
      if (!item) return;
      beam.setAttribute("width", geometry.width);
      beam.setAttribute("height", geometry.height);
      // the hull of the layer and the card, one above the other: the layer's left, top and right
      // corners, the card's right, bottom and left ones; and the edges between their corners
      const [upper, lower] = [anchors[items.indexOf(item)], anchors[items.indexOf(card)]].sort((a, b) => a.y - b.y).map(corners);
      const by = (points, pick) => points.reduce(pick);
      const left = (a, b) => (b[0] < a[0] ? b : a);
      const right = (a, b) => (b[0] > a[0] ? b : a);
      const hull = [by(upper, left), by(upper, (a, b) => (b[1] < a[1] ? b : a)), by(upper, right), by(lower, right), by(lower, (a, b) => (b[1] > a[1] ? b : a)), by(lower, left)];
      const f = ([x, y]) => `${roundTo(x, 1)},${roundTo(y, 1)}`;
      beamHull.setAttribute("points", hull.map(f).join(" "));
      beamEdges.setAttribute("d", upper.map((p, k) => `M${f(p)}L${f(lower[k])}`).join(""));
    }

    // -- the plot of the fit, upright beside the sheets: every order at the height of its sheet, from
    // order 1 at the top down to the full decomposition, and the fit growing to the right. A sheet's
    // row keeps its line from the sheet's corner, on to its point. Unfolded, it ends at the last
    // explanation too: the input and the mechanism below them are no explanations ----------------

    let sideMarks = null;
    function drawSide() {
      sideData.replaceChildren();
      sideMarks = null;
      const example = current();
      const g = geometry;
      side.setAttribute("viewBox", `0 0 ${g.width} ${g.height}`);
      side.setAttribute("width", g.width);
      side.setAttribute("height", g.height);
      // every layer's point at the height of its right corner, where the stack (or the unfolded
      // layers) lies at rest
      const rest = layered() ? layeredLayout(true) : layout(null);
      const points = items
        .filter((item) => !item.base && !item.mechanism)
        .map((item) => ({
          item,
          order: item.order,
          r2: item.sheet ? item.sheet.r2 : item.fit[1],
          y: rightCorner(rest[items.indexOf(item)])[1],
        }));
      side.setAttribute(
        "aria-label",
        `Faithfulness (R²) of the explanations, top to bottom: ${points.map((p) => `order ${p.order}: ${fit2(p.r2)}`).join(", ")}` +
          (estimated(example) ? "; what the full decomposition falls short of 1 is the estimation gap" : ""),
      );
      // R^2 from 0, or from 0.5 when every order fits better, so that the rise shows
      const low = Math.min(...points.map((p) => p.r2)) >= 0.5 ? 0.5 : 0;
      const [x0, x1] = [g.ladderLeft + (g.narrow ? 6 : 10), g.width - (g.narrow ? 8 : 12)];
      const x = (r2) => x0 + ((Math.max(low, Math.min(1, r2)) - low) / (1 - low)) * (x1 - x0);
      const f = (value) => roundTo(value, 1);
      const [top, bottom] = [points[0].y - 12, points.at(-1).y + 12];

      // the frame, made for the stage's width: the grid, and at the top, the axis's ticks and over
      // them its name (over the plot's middle, or, where the plot is narrower than its name, ending
      // at its right). It spans the stack: for the next example's, it stretches or shrinks (demo.css)
      if (sideFrame.dataset.width !== String(g.width)) {
        sideFrame.dataset.width = g.width;
        sideFrame.replaceChildren();
        const grid = add(sideFrame, "g", { class: "orders-demo-plot-grid" });
        for (let k = 0; k <= 4; k++) {
          add(grid, "line", { class: "orders-demo-side-rule", "data-x": f(x0 + (k / 4) * (x1 - x0)), x1: 0, x2: 0, y1: 0, y2: 1 });
          if (k % 2 === 0) add(sideFrame, "text", { class: "orders-demo-plot-tick orders-demo-side-level", "data-x": f(x0 + (k / 4) * (x1 - x0)), "text-anchor": "middle" });
        }
        const roomy = x1 - x0 >= 104;
        add(sideFrame, "text", { class: "orders-demo-plot-label orders-demo-side-name", "data-x": f(roomy ? (x0 + x1) / 2 : x1), "text-anchor": roomy ? "middle" : "end" }, "faithfulness (R²)");
      }
      const to = (node, y, scale = "") => {
        node.style.transform = `translate(${node.dataset.x}px, ${f(y)}px)${scale}`;
      };
      for (const rule of sideFrame.querySelectorAll(".orders-demo-side-rule")) to(rule, top, ` scale(1, ${f(bottom - top)})`);
      [...sideFrame.querySelectorAll(".orders-demo-side-level")].forEach((tick, k) => {
        to(tick, top - 9);
        const level = low + (k / 2) * (1 - low);
        const text = level === 0.5 && g.narrow ? ".5" : String(level);
        if (tick.textContent !== text) say(tick, text, tick.textContent !== "");
      });
      to(sideFrame.querySelector(".orders-demo-side-name"), top - 25);

      // the estimation gap: what an estimated full decomposition falls short of 1, a hatched band
      // down the plot, its name along it
      const full = points.at(-1);
      if (full.r2 < EXACT) {
        const id = `orders-side-gap-${plotId}`;
        const hatch = add(add(sideData, "defs"), "pattern", { id, width: 5, height: 5, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
        add(hatch, "line", { class: "orders-demo-plot-gap-hatch", x1: 0, y1: 0, x2: 0, y2: 5 });
        const gap = add(sideData, "g", { class: "orders-demo-plot-gap" });
        const band = { x: f(x(full.r2)), y: top, width: f(x(1) - x(full.r2)), height: f(bottom - top) };
        add(gap, "rect", { ...band, class: "orders-demo-plot-gap-fill" });
        add(gap, "rect", { ...band, fill: `url(#${id})` });
        const [cx, cy] = [f(band.x + band.width / 2), f((points[0].y + full.y) / 2)];
        add(gap, "text", { class: "orders-demo-plot-axis orders-demo-plot-gap-label", x: cx, y: cy, "text-anchor": "middle", transform: `rotate(-90 ${cx} ${cy})` }, "estimation gap");
      }

      // every sheet's line from its row on to its point (the row draws it from the sheet's corner)
      const leads = new Map();
      for (const p of points) {
        if (!p.item.sheet) continue;
        leads.set(p.item, add(sideData, "line", { class: "orders-demo-side-lead", x1: g.ladderLeft, x2: f(x(p.r2) - 4), y1: f(p.y), y2: f(p.y) }));
      }

      // the area left of the line, darkest where it runs furthest right; the line, dashed where it
      // skips orders (they all equal the full decomposition)
      const at = (p) => `${f(x(p.r2))},${f(p.y)}`;
      const fade = `orders-side-fade-${plotId}`;
      const gradient = add(add(sideData, "defs"), "linearGradient", { id: fade, x1: 1, y1: 0, x2: 0, y2: 0 });
      add(gradient, "stop", { class: "orders-demo-plot-fade-top", offset: 0 });
      add(gradient, "stop", { class: "orders-demo-plot-fade-bottom", offset: 1 });
      add(sideData, "path", { class: "orders-demo-plot-area", d: `M${f(x0)},${f(points[0].y)}L${points.map(at).join("L")}L${f(x0)},${f(full.y)}Z`, fill: `url(#${fade})` });
      const skips = points.length > 1 && full.order - points.at(-2).order > 1;
      const solid = skips ? points.slice(0, -1) : points;
      add(sideData, "path", { class: "orders-demo-plot-line", d: `M${solid.map(at).join("L")}` });
      if (skips) add(sideData, "path", { class: "orders-demo-plot-line orders-demo-plot-skipped", d: `M${points.slice(-2).map(at).join("L")}` });
      for (const p of points) {
        const blank = !p.item.sheet;
        const className = blank ? "orders-demo-plot-dot orders-demo-plot-between" : "orders-demo-plot-dot";
        add(sideData, "circle", { class: className, cx: f(x(p.r2)), cy: f(p.y), r: blank ? (g.narrow ? 1.8 : 2.2) : 3 });
      }

      // the ring at the order pointed at, with its R^2
      const ring = add(sideData, "g", { class: "orders-demo-plot-marker" });
      add(ring, "circle", { r: 7 });
      add(ring, "circle", { class: "orders-demo-plot-core", r: 3 });
      const value = add(ring, "text", { class: "orders-demo-plot-value" });

      // a cover over the plot: the order nearest to the pointer, by height
      const cover = add(sideData, "rect", { class: "orders-demo-plot-cover", x: g.ladderLeft, y: f(top - 34), width: f(g.width - g.ladderLeft), height: f(bottom - top + 46) });
      const nearest = (event) => {
        const y = event.clientY - side.getBoundingClientRect().top;
        return points.reduce((best, p) => (Math.abs(p.y - y) < Math.abs(best.y - y) ? p : best), points[0]);
      };
      const go = (p, pin) => {
        const [active, blank] = p.item.sheet ? [p.item.index, null] : [null, p.order];
        if (pin) toggle(active, blank);
        else point(active, blank);
      };
      let touched = false;
      cover.addEventListener("pointermove", (event) => {
        if (event.pointerType === "mouse" && !state.pinned) go(nearest(event), false);
      });
      cover.addEventListener("pointerdown", (event) => {
        touched = event.pointerType !== "mouse";
        if (!touched) return;
        state.pinned = true;
        go(nearest(event), false);
      });
      cover.addEventListener("click", (event) => {
        if (touched) {
          touched = false;
          return;
        }
        go(nearest(event), true);
      });
      sideMarks = { points, leads, ring, value, x, left: x0 };
      mark();
    }

    // the ring on the plot: at the layer pointed at (a sheet's slid out, or a blank sheet's, hidden at
    // rest), its R^2 left of it (where the line has come from above), or right of it, near the axis;
    // and the line of its row, solid
    function mark() {
      if (!sideMarks) return;
      const pointed = state.active !== null ? sheetItems()[state.active] : null;
      const at = sideMarks.points.find((p) => (pointed ? p.item === pointed : state.blank !== null && !p.item.sheet && p.order === state.blank));
      side.classList.toggle("orders-demo-plot-pointing", Boolean(at));
      for (const [item, lead] of sideMarks.leads) lead.classList.toggle("orders-demo-side-lead-on", item === pointed);
      if (!at) return;
      const [cx, cy] = [roundTo(sideMarks.x(at.r2), 1), roundTo(at.y, 1)];
      sideMarks.ring.style.transform = `translate(${cx}px, ${cy}px)`;
      // (right of it only where it fits on the stage)
      const right = cx - 58 < sideMarks.left && cx + 60 <= geometry.width;
      sideMarks.value.textContent = `R² ${fit2(at.r2)}`;
      sideMarks.value.setAttribute("x", right ? 11 : -11);
      sideMarks.value.setAttribute("text-anchor", right ? "start" : "end");
    }

    // point at a sheet (its index), or at the blank sheet of an order between, or at nothing
    function point(active, blank) {
      // during a switch, the sheets, rows and plot on screen are still the old example's
      if (root.classList.contains("orders-demo-fading")) return;
      if (state.active === active && state.blank === blank) return;
      Object.assign(state, { active, blank });
      place(active);
      mark();
      showPlan();
    }

    function toggle(active, blank) {
      const same = state.active === active && state.blank === blank;
      if (state.pinned && same) {
        state.pinned = false;
        point(null, null);
      } else {
        state.pinned = true;
        point(active, blank);
        // pointed at already (hovered, then tapped), it is pinned now: its row and button say so
        if (same && !root.classList.contains("orders-demo-fading")) {
          place(active);
          showPlan();
        }
      }
    }

    // the slider: a stop's name, and bringing out its sheet (the rest puts every sheet back); a sheet
    // brought out by the slider stays out, as one tapped
    const stopName = (item) => (item.mechanism ? "The mechanism, the model itself" : item.sheet ? sheetName(item.sheet).name + (item.sheet.order > 1 && !item.sheet.full ? ` of order ${item.order}` : "") : `order ${item.order}`);
    function slideTo(stop) {
      state.pinned = stop.item !== null;
      if (stop.item === null) point(null, null);
      else if (stop.item.sheet) point(stop.item.index, null);
      else point(null, stop.item.order);
    }
    const stopAt = (clientY) => {
      const y = clientY - stage.getBoundingClientRect().top;
      return stops.reduce((best, stop) => (Math.abs(stop.y - y) < Math.abs(best.y - y) ? stop : best), stops[0]);
    };
    rail.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      rail.setPointerCapture(event.pointerId);
      rail.classList.add("orders-demo-dragging");
      // the arrow keys go on from there, without the ring that shows keyboard focus
      thumb.focus({ preventScroll: true, focusVisible: false });
      slideTo(stopAt(event.clientY));
    });
    rail.addEventListener("pointermove", (event) => {
      if (rail.hasPointerCapture(event.pointerId)) slideTo(stopAt(event.clientY));
    });
    for (const type of ["pointerup", "pointercancel"]) {
      rail.addEventListener(type, () => rail.classList.remove("orders-demo-dragging"));
    }
    thumb.addEventListener("keydown", (event) => {
      const at = Number(thumb.getAttribute("aria-valuenow"));
      const last = stops.length - 1;
      const next = { ArrowDown: at + 1, ArrowRight: at + 1, ArrowUp: at - 1, ArrowLeft: at - 1, Home: 0, End: last }[event.key];
      if (next === undefined) return;
      event.preventDefault();
      slideTo(stops[Math.max(0, Math.min(last, next))]);
    });

    // the plot points at an order: the sheet of that order, or its blank sheet
    function pointAtOrder(order) {
      const index = current().sheets.findIndex((sheet) => sheet.order === order);
      if (index >= 0) point(index, null);
      else point(null, order);
    }

    const pointer = (event) => {
      const box = stage.getBoundingClientRect();
      return [event.clientX - box.left, event.clientY - box.top];
    };
    // the pointer over what points at the sheets (the plan's card, the stage, the plot) keeps what it
    // points at: going from one to another, nothing goes back and comes out again. Only once the
    // pointer has left them all, for a moment, does it go back
    let resting = 0;
    function hold() {
      clearTimeout(resting);
    }
    function rest() {
      clearTimeout(resting);
      resting = setTimeout(() => {
        if (!state.pinned && (state.active !== null || state.blank !== null)) point(null, null);
      }, 280);
    }
    for (const zone of [why, stage]) {
      zone.addEventListener("pointerenter", (event) => {
        if (event.pointerType === "mouse") hold();
      });
      zone.addEventListener("pointerleave", (event) => {
        if (event.pointerType === "mouse") rest();
      });
    }
    stage.addEventListener("pointermove", (event) => {
      if (event.pointerType !== "mouse" || state.pinned || event.target.closest(".orders-demo-row, .orders-demo-rail, .orders-demo-side")) return;
      hold();
      const item = itemAt(pointer(event));
      if (item?.base) return;
      if (item?.sheet) point(item.index, null);
      // a blank sheet comes out as a sheet does, and the sheet slid out goes back: the blank sheet's
      // strip, where the stack at rest has it, stays under the pointer (over the sheet slid out,
      // itemAt gives that sheet)
      else if (item) point(null, item.order);
    });
    // on a touch screen, a swipe sideways across the stack goes to the next example or the previous
    // one, as the stepper does: the stack follows the finger, fading, and past a fifth of its width,
    // or flicked, it leaves to that side; else it springs back. A swipe up or down scrolls the page.
    // Touch events, not pointer events: only a touchmove can be kept from the browser, which on iOS
    // takes a sideways drag over after a few pixels and cancels it for the page
    let swipe = null;
    let swiped = -Infinity; // when the last swipe ended: the click that may follow it is no tap
    stage.addEventListener(
      "touchstart",
      (event) => {
        const free = event.touches.length === 1 && !event.target.closest(".orders-demo-rail") && !root.classList.contains("orders-demo-fading");
        const [touch] = event.touches;
        swipe = free ? { x: touch.clientX, y: touch.clientY, dx: 0, sideways: null, moves: [[event.timeStamp, 0]] } : null;
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
            swipe = null; // the page scrolls, or zooms
            return;
          }
          root.classList.add("orders-demo-swiping");
        }
        event.preventDefault(); // the drag is the stack's: the page neither scrolls nor takes it over
        swipe.dx = dx;
        swipe.moves.push([event.timeStamp, dx]);
        shift(dx);
        root.style.setProperty("--orders-swipe-opacity", String(roundTo(1 - Math.min(0.6, Math.abs(dx) / stage.clientWidth), 3)));
      },
      { passive: false },
    );
    const endSwipe = (event, cancelled) => {
      if (!swipe) return;
      const { dx, sideways, moves } = swipe;
      swipe = null;
      if (!sideways) return;
      swiped = performance.now();
      root.classList.remove("orders-demo-swiping");
      // a flick: fast over the last tenth of a second, in the direction swiped
      const [at, from] = moves.find(([time]) => event.timeStamp - time <= 100) ?? moves.at(-1);
      const speed = (dx - from) / Math.max(16, event.timeStamp - at);
      const flick = Math.abs(dx) > 20 && Math.abs(speed) > 0.3 && Math.sign(speed) === Math.sign(dx);
      if (!cancelled && (Math.abs(dx) > stage.clientWidth / 5 || flick)) {
        step(dx < 0 ? 1 : -1);
      } else {
        shift(0);
        root.style.removeProperty("--orders-swipe-opacity");
      }
    };
    stage.addEventListener("touchend", (event) => endSwipe(event, false));
    stage.addEventListener("touchcancel", (event) => endSwipe(event, true));
    stack.addEventListener("click", (event) => {
      if (performance.now() - swiped < 400) return; // the end of a swipe, not a tap
      if (layered() && !unfolded()) {
        // a tap on the layers at rest unfolds them, for good, none pointed at
        unfoldedForGood = true;
        place(null);
        return;
      }
      const item = itemAt(pointer(event));
      if (layered() && (item === null || item.base)) {
        // unfolded, a tap on the card, or beside the layers, lets go of the layer pointed at
        state.pinned = false;
        point(null, null);
        return;
      }
      if (item === null) {
        state.pinned = false;
        point(null, null);
      } else if (item.sheet) {
        toggle(item.index, null);
      } else {
        toggle(null, item.order);
      }
    });
    // when the keyboard leaves the rows, the stack closes again
    ladder.addEventListener("focusout", (event) => {
      if (!state.pinned && !ladder.contains(event.relatedTarget)) point(null, null);
    });

    let width = 0;
    new ResizeObserver(() => {
      if (stage.clientWidth !== width) {
        width = stage.clientWidth;
        geometry = measure();
        place(state.active);
        drawSide();
      }
    }).observe(root);
    new ResizeObserver(() => movePill(true)).observe(switcher);
    // the light out of the chip follows the plan's layout (its width, its fonts)
    new ResizeObserver(() => {
      settleCard();
      castLightOut();
      for (const node of [xShows, chipShows, outShows]) fitPart(node);
    }).observe(plan);
    document.fonts?.ready.then(() => {
      [xShows, chipShows, outShows].forEach(fitPart);
      settled = ""; // measured anew in the fonts loaded
      settleCard();
      castLightOut();
    });
    // the view (a stack, or the unfolded view) follows the demo's data-view attribute
    new MutationObserver(() => render()).observe(root, { attributes: true, attributeFilter: ["data-view"] });

    // the first example fades in, as every next one
    root.classList.add("orders-demo-fading");
    showStep(current(), false);
    render();
    movePill(true);
    requestAnimationFrame(() => requestAnimationFrame(() => root.classList.remove("orders-demo-fading")));
    idle(drawAhead);
  }

  // -- loading ----------------------------------------------------------------------------------------

  // the static figure of every demo, kept before the demo replaces it (with scripts on, a
  // <noscript>'s content is its HTML as text)
  const fallbacks = new Map(roots.map((root) => [root, root.querySelector("noscript")?.textContent ?? ""]));
  const showFallback = (root) => {
    root.classList.remove("orders-demo-ready", "orders-demo-fading");
    root.classList.add("orders-demo-failed");
    root.innerHTML = fallbacks.get(root);
  };
  const begin = () => {
    for (const root of roots) {
      try {
        start(root, window.SHAPIQ_ORDERS);
      } catch (error) {
        console.error("The explanation of every order could not start:", error);
        showFallback(root);
      }
    }
  };
  if (window.SHAPIQ_ORDERS) {
    begin();
  } else {
    const loader = document.createElement("script");
    loader.src = `${roots[0].dataset.root}data.js?v=${roots[0].dataset.version ?? ""}`; // past caches
    loader.onload = begin;
    loader.onerror = () => roots.forEach(showFallback);
    document.head.appendChild(loader);
  }
})();
