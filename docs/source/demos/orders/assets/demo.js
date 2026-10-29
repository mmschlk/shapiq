// The orders demo (demos/orders): one prediction, explained at every order of interaction, as a
// stack of sheets. The top sheet holds the Shapley values (order 1); every sheet below adds the
// interactions of one more order (FSII, the Faithful Shapley Interactions, or k-SII, as the data
// says), up to order 4. Below them, blank sheets stand for the orders that follow, up to the bottom
// sheet: the full decomposition, the interactions of all orders, which add up to the model itself.
// Beside every sheet is its R^2, its faithfulness: how much of the model's output on random
// coalitions (drawn from the distribution of the Shapley values) the explanation reproduces beyond
// what their sizes alone predict; below the stack a plot of the R^2 of every order: it rises from
// the Shapley values to the full decomposition, which reproduces the model exactly, unless it is
// estimated: what it falls short of 1 is then shaded, the estimation tax.
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
// tap that pins it, or the slider beside the stack) slides it out to the side of the
// stack as it lies, under the sheets above it, and the others stay where they are. A ring marks its
// order on the plot; pointing along the plot brings out the sheet of the order pointed at, or the
// blank sheet of an order between. The slider is an arrow down the side of the stack, and one along
// the plot's axis matches it: the explanations grow more complex, and more faithful, from order to
// order. Where the examples do not fit in a row, a stepper goes from one to the next, and so does a
// swipe across the stack.
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
  const PLOT = { height: 153, left: 58, right: 14, top: 24, bottom: 54 }; // pixels: an axis of 75

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

  const largestValue = (sheet) =>
    Math.max(...sheet.first.map(Math.abs), ...sheet.groups.map((group) => Math.abs(group[1])));

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

  // the molecule: its bonds and atoms over the explanation, as in GraphSHAP-IQ's figures
  function moleculeFace(svg, example, sheet) {
    const place = fit(example.coords, [9, 5, 93, 87]);
    const position = example.coords.map(place);
    const extent = extentOf(position);
    const [radius, widest] = [(ATOM / 2) * extent, WIDEST * extent];
    const largest = largestValue(sheet);
    // the groups' centers where shapiq's SI graph plot puts them (../compute/pack.py)
    drawGroups(svg, sheet, (atom) => position[atom], largest, widest, place);
    // a main effect: a disc around its atom, as much wider as a group of its value, over a disc of
    // the sheet's color that hides the groups below it
    const halos = add(svg, "g", { class: "orders-demo-halos" });
    sheet.first.forEach((value, atom) => {
      const ratio = Math.abs(value) / largest;
      if (ratio < FAINTEST) return;
      const disc = {
        cx: roundTo(position[atom][0], 2),
        cy: roundTo(position[atom][1], 2),
        r: roundTo(radius + (widest * ratio) / 2, 2),
      };
      add(halos, "circle", { ...disc, class: "orders-demo-halo-ground" });
      add(halos, "circle", { ...disc, fill: solid(value), "fill-opacity": roundTo(ratio, 3) });
    });
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
  function imageTextFace(svg, example, sheet, url, id) {
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
    // a main effect tints its patch, or its word, as opaque as its share of the largest value
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
    const lines = add(image, "g", { class: "orders-demo-grid" });
    for (let k = 1; k < grid; k++) {
      const at = roundTo(photo.x + k * cell, 2);
      const across = roundTo(photo.y + k * cell, 2);
      add(lines, "line", { x1: at, y1: photo.y, x2: at, y2: photo.y + photo.size });
      add(lines, "line", { x1: photo.x, y1: across, x2: photo.x + photo.size, y2: across });
    }

    const players = Array.from({ length: patches + example.tokens.length }, (_, player) => anchor(player));
    drawGroups(svg, sheet, anchor, largest, WIDEST * extentOf(players));

    const words = add(svg, "g", { class: "orders-demo-tokens" });
    example.tokens.forEach((token, k) => {
      const value = sheet.first[patches + k];
      const ratio = Math.abs(value) / largest;
      const y = tokens.top + k * row + row * 0.12;
      const height = row * 0.76;
      add(words, "rect", { x: tokens.x, y: roundTo(y, 2), width: tokens.width, height: roundTo(height, 2), rx: 1.4 });
      if (ratio >= FAINTEST) {
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
      add(words, "text", { x: tokens.x + tokens.width / 2, y: roundTo(y + height / 2 + 0.1, 2) }, token);
    });
  }

  // one house: its explanation as a tree that grows one level per order. The features are on top,
  // every group of k features hangs at level k, below the middle of its features, and joined to
  // each of them; red raises the value of the house, blue lowers it. The full decomposition of a
  // tree ensemble is as deep as its trees.
  const TREE_GROUPS = [0, 0, 6, 5, 4, 3, 3, 2]; // the strongest groups shown, by level
  function treeFace(svg, example, sheet) {
    const levels = example.largest;
    const [top, bottom] = [24, 88];
    const step = (bottom - top) / (levels - 1);
    const slot = new Map(example.layout.map((feature, k) => [feature, 9 + (k * 82) / (example.layout.length - 1)]));
    const largest = largestValue(sheet);
    const ratioOf = (value) => Math.abs(value) / largest;

    const guides = add(svg, "g", { class: "orders-demo-tree-levels" });
    for (let level = 1; level <= levels; level++) {
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
    for (const node of [...nodes].sort((a, b) => Math.abs(a.value) - Math.abs(b.value))) {
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
    for (const node of nodes) {
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
      if (ratio >= 0.02) {
        add(halos, "circle", {
          cx: x,
          cy: top,
          r: roundTo(2.2 + 4.4 * ratio, 2),
          fill: solid(sheet.first[feature]),
          "fill-opacity": roundTo(0.15 + 0.85 * ratio, 3),
        });
      }
      add(features, "circle", { cx: x, cy: top, r: 2 });
      const y = k % 2 === 0 ? 5.6 : 12;
      add(labels, "text", { x, y, "text-anchor": "middle", class: "orders-demo-feature-name" }, example.features[feature]);
      add(labels, "text", { x, y: y + 3.3, "text-anchor": "middle", class: "orders-demo-feature-value" }, example.values[feature]);
    });
  }

  // many houses: a heatmap, a column per house sorted by its prediction (the line on top), a row
  // per feature or group of features; the cells are drawn on a canvas, one pixel each
  function heatmapFace(svg, example, sheet) {
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
    add(svg, "path", {
      class: "orders-demo-heat-area",
      d: `M${box.left},${box.band[1]}L${line.join("L")}L${box.right},${box.band[1]}Z`,
    });
    add(svg, "path", { class: "orders-demo-heat-line", d: `M${line.join("L")}` });
    add(svg, "text", { class: "orders-demo-heat-label", x: box.left - 1.5, y: box.band[1] - 2 }, "prediction");

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
    // the model of the example: its icon, what it is, and how it was explained
    const badge = el("div", "orders-demo-model");
    badge.setAttribute("aria-live", "polite");
    const about = el("div", "orders-demo-about");
    about.append(badge);
    bar.append(switcher, stepper, about);

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
    stage.append(rail, view, ladder);

    const footer = el("div", "orders-demo-footer");
    const plotBox = el("figure", "orders-demo-plot");
    const plot = document.createElementNS(SVG, "svg");
    plot.setAttribute("class", "orders-demo-plot-svg");
    plot.setAttribute("role", "img");
    plotBox.append(plot);
    footer.append(plotBox);
    // under the stack, where the stepper shows (demo.css): a dot for every example, the one shown dark
    const dots = el("div", "orders-demo-dots");
    dots.setAttribute("aria-hidden", "true");
    dots.append(...data.examples.map(() => el("span")));
    root.append(bar, stage, dots, footer);
    root.classList.add("orders-demo-ready");

    // the stack, top to bottom: the sheets of order 1 to 4, a blank sheet for every order up to
    // the full decomposition, and the sheet of the full decomposition
    let items = [];
    let rows = [];
    let stops = []; // the slider's: its rest, then every sheet and blank sheet, top to bottom
    let blankRow = null;
    let marker = null;
    let plotWidth = 0;

    const current = () => data.examples.find((example) => example.id === state.example);
    const sheetItems = () => items.filter((item) => item.sheet);
    const blankItems = () => items.filter((item) => !item.sheet);

    // -- geometry: the size of the sheets, the stage, and every sheet's place ------------------------

    function measure() {
      const width = stage.clientWidth;
      const narrow = width < 560;
      // the rows of the ladder: room for a bar, the R^2 and the verdict, all as wide; on narrow
      // screens no ladder, and its room goes to the sheets (the plot below has their fits)
      const ladderWidth = narrow ? 0 : Math.min(256, width * 0.32);
      const gap = narrow ? 0 : 28;
      const spread = Math.abs(Math.cos(TURN)) + Math.abs(Math.sin(TURN));
      const room = width - ladderWidth - gap - RAIL;
      // the room holds the stack and the sheet slid out of it
      const side = Math.max(110, Math.min(360, room / ((1 + SLIDE) * spread)));
      const sheetHeight = side * TILT * spread;
      // the same height for every example: the most sheets and blank sheets of any
      const n = Math.max(...data.examples.map((example) => example.sheets.length));
      const blanks = Math.max(...data.examples.map((example) => between(example).length));
      const stack = sheetHeight + (n - 1) * STRIP * side + blanks * BLANK_STRIP * side;
      // a sheet slides out to the left: the stack lies right of the middle by half of that
      const slideBy = SLIDE * side * spread;
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
    // above it, and the one pointed at slid out to the side of the stack
    function layout(active) {
      const g = geometry;
      const pointed = active === null ? null : sheetItems()[active];
      const h = g.sheetHeight;
      const centers = [];
      let bottom = 0;
      items.forEach((item, position) => {
        if (position === 0) bottom = h;
        else bottom += (item.sheet ? STRIP : BLANK_STRIP) * g.side;
        centers.push(bottom - h / 2);
      });
      const offset = (g.height - bottom) / 2;
      return items.map((item, position) => ({
        x: g.center - (item === pointed ? g.slideBy : 0),
        y: centers[position] + offset + (item.order === state.blank ? PEEK * g.side : 0),
      }));
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

    // the item under a point of the stage, where the sheets are going (not where they are while
    // they move), the one on top first. While a sheet slides out, the sheets count where they lie
    // at rest, and the one slid out also where it shows: else the sheet would slide away from under
    // the pointer, hand it to the sheet below, and come back for it, over and over.
    function itemAt(point) {
      const sliding = state.active !== null;
      const places = layout(null);
      const order = items.map((_, position) => position);
      order.sort((a, b) => zIndex(items[b]) - zIndex(items[a]));
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
      const fades = [stack, ladder, plot].flatMap((node) => {
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
    function drawSheet(example, i) {
      const sheet = example.sheets[i];
      const node = el("div", "orders-demo-sheet");
      const svg = add(node, "svg", { viewBox: "0 0 100 100", class: "orders-demo-face" });
      if (example.kind === "graph") moleculeFace(svg, example, sheet);
      else if (example.kind === "tree") treeFace(svg, example, sheet);
      else if (example.kind === "heatmap") heatmapFace(svg, example, sheet);
      else imageTextFace(svg, example, sheet, `${root.dataset.root}${example.image}`, `orders-${plotId}-${example.id}-${i}`);
      add(svg, "text", { class: "orders-demo-tag", x: 6, y: 95.5 }, sheetName(sheet).face);
      return node;
    }
    // every example's sheets, drawn once: an example chosen again shows them as they were drawn
    const drawn = new Map();
    function sheetsOf(example) {
      const nodes = drawn.get(example.id) ?? [];
      for (let i = nodes.length; i < example.sheets.length; i++) nodes.push(drawSheet(example, i));
      drawn.set(example.id, nodes);
      return nodes;
    }
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
      Object.assign(state, { active: null, blank: null, pinned: false }); // a new example shows at rest
      for (const [id, option] of options) option.setAttribute("aria-pressed", String(id === example.id));
      stage.setAttribute("aria-label", `${example.label}: its explanations of order 1 to n, stacked, with their fit`);
      stack.replaceChildren();
      for (const row of [...rows, blankRow]) row?.remove();
      const text = el("span", "orders-demo-model-text");
      text.append(
        el("strong", "", example.model.title),
        el("span", "", example.model.detail),
        el("span", "orders-demo-model-method", example.model.method),
      );
      badge.replaceChildren(icon(example.model.icon, "orders-demo-model-icon"), text);
      // the badge's entrance, unless it is swiped in with the stack
      badge.classList.remove("orders-demo-model-enter");
      if (!root.classList.contains("orders-demo-leaving")) {
        badge.getBoundingClientRect(); // restart the entrance
        badge.classList.add("orders-demo-model-enter");
      }
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
        if (!item.sheet) item.node = el("div", "orders-demo-sheet orders-demo-blank");
      }

      rows = example.sheets.map((sheet, i) => {
        const names = sheetName(sheet);
        const row = el("button", "orders-demo-row");
        row.type = "button";
        row.setAttribute("aria-label", `${names.name}, order ${sheet.order}: R² ${fit2(sheet.r2)}, ${verdict(sheet.r2)}`);
        const title = el("span", "orders-demo-row-title");
        title.append(el("strong", "", names.tag), el("span", "orders-demo-row-name", names.name));
        const fitBox = el("span", "orders-demo-row-fit");
        const meter = el("span", "orders-demo-meter");
        const fill = el("span", "orders-demo-meter-fill");
        fill.style.width = `${Math.max(0, Math.min(1, sheet.r2)) * 100}%`;
        meter.append(fill);
        // the R^2 and its verdict in one phrase: "R² 0.76, workable"
        const score = el("span", "orders-demo-score");
        score.append(
          el("span", "orders-demo-r2", fit2(sheet.r2)),
          el("span", "orders-demo-verdict", verdict(sheet.r2)),
        );
        fitBox.append(meter, score);
        row.append(title, fitBox);
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
        ladder.append(row);
        return row;
      });
      ticks.replaceChildren(...items.map((item) => el("span", item.sheet ? "orders-demo-tick" : "orders-demo-tick orders-demo-tick-blank")));

      // one row for all blank sheets: the orders between, which the plot shows one by one
      blankRow = null;
      if (between(example).length) {
        blankRow = el("div", "orders-demo-row orders-demo-blank-row");
        blankRow.setAttribute("aria-hidden", "true");
        const title = el("span", "orders-demo-row-title");
        title.append(el("strong", "", "⋮"), el("span", "orders-demo-row-name"));
        blankRow.append(title);
        ladder.append(blankRow);
      }

      drawPlot();
      // a new example shows at rest at once: its sheets and rows take their places without moving.
      // Its sheets join the stack only now, placed: measuring the plot and the rows would else style
      // and lay out every element of their faces, and placing them (their size, and the custom
      // properties of their edges, which every element inherits) would do so again
      root.classList.add("orders-demo-instant");
      place(null);
      stack.append(...items.map((item) => item.node));
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
      const rowY = []; // the height of every sheet's row, for the row of the blank sheets
      items.forEach((item, position) => {
        const { x, y } = places[position];
        const node = item.node;
        node.style.width = node.style.height = `${g.side}px`;
        node.style.marginLeft = node.style.marginTop = `${-g.side / 2}px`;
        node.style.zIndex = zIndex(item);
        node.style.transform = `translate(${roundTo(x, 2)}px, ${roundTo(y, 2)}px) scaleY(${TILT}) rotate(${(TURN * 180) / Math.PI}deg)`;
        // the edge below the face, in the face's own coordinates: straight down on the stage
        const depth = (item.sheet ? THICKNESS : THICKNESS / 2) / TILT;
        node.style.setProperty("--orders-edge-x", `${roundTo(Math.sin(TURN) * depth, 2)}px`);
        node.style.setProperty("--orders-edge-y", `${roundTo(Math.cos(TURN) * depth, 2)}px`);
        node.classList.toggle("orders-demo-receded", out && item.index !== active);
        node.classList.toggle("orders-demo-peek", !item.sheet && item.order === state.blank);

        if (!item.sheet) return;
        const row = rows[item.index];
        const right = rightCorner(places[position]);
        rowY[item.index] = right[1];
        row.style.left = `${g.ladderLeft}px`;
        row.style.transform = `translateY(${roundTo(right[1], 2)}px) translateY(-50%)`;
        row.style.setProperty("--orders-lead", `${Math.max(0, roundTo(g.ladderLeft - right[0] - 4, 2))}px`);
        row.classList.toggle("orders-demo-active", out && item.index === active);
        row.setAttribute("aria-pressed", String(state.pinned && out && item.index === active));
      });

      // the slider: its stops at the middles of the sheets at rest (without a blank sheet's peek), its
      // line from its rest down to the bottom sheet, and its handle at the sheet pointed at
      const rest = layout(null);
      stops = [
        { y: Math.max(10, rest[0].y - RAIL_REST), item: null },
        ...items.map((item, p) => ({ y: rest[p].y - (item.order === state.blank ? PEEK * g.side : 0), item })),
      ];
      const [top, bottom] = [stops[0].y, stops.at(-1).y];
      railLine.style.transform = `translateY(${roundTo(top, 2)}px)`;
      railLine.style.height = `${roundTo(bottom - top, 2)}px`;
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

      // the row of the blank sheets: beside the middle one, where it lies at rest
      if (blankRow) {
        const blanks = blankItems();
        const target = blanks[Math.floor((blanks.length - 1) / 2)];
        const right = rightCorner(places[items.indexOf(target)]);
        if (target.order === state.blank) right[1] -= PEEK * g.side;
        blankRow.classList.toggle("orders-demo-active", state.blank !== null);
        // beside the blank sheet, but clear of the rows above and below it
        const [above, below] = [rows.length - 2, rows.length - 1];
        const half = (row) => row.offsetHeight / 2;
        const y = Math.min(
          Math.max(right[1], rowY[above] + half(rows[above]) + half(blankRow) + 2),
          rowY[below] - half(rows[below]) - half(blankRow) - 2,
        );
        blankRow.style.left = `${g.ladderLeft}px`;
        blankRow.style.transform = `translateY(${roundTo(y, 2)}px) translateY(-50%)`;
        blankRow.style.setProperty("--orders-lead", `${Math.max(0, roundTo(g.ladderLeft - right[0] - 4, 2))}px`);
        // the orders the blank sheets stand for: all between order 4 and n
        const [first, last] = [blanks[0].order, current().sheets.at(-1).order - 1];
        blankRow.querySelector(".orders-demo-row-name").textContent =
          state.blank !== null
            ? `Order ${state.blank}: R² ${fit2(blanks.find((item) => item.order === state.blank).fit[1])}`
            : `Orders ${first}–${last}`;
      }
    }

    // -- the plot of the fit by order: a ring on the line marks the order pointed at -------------

    function drawPlot() {
      const example = current();
      plotWidth = plotBox.clientWidth;
      const [w, h] = [Math.max(200, plotWidth), PLOT.height];
      plot.setAttribute("viewBox", `0 0 ${w} ${h}`);
      plot.setAttribute("width", w);
      plot.setAttribute("height", h);
      plot.setAttribute(
        "aria-label",
        `Faithfulness (R²) by order: ${example.fits.map(([order, r2]) => `${order}: ${fit2(r2)}`).join(", ")}; the highest order is the full decomposition` +
          (example.fits.at(-1)[1] < EXACT ? ", estimated: what its R² falls short of 1 is the estimation tax" : ""),
      );
      plot.replaceChildren();
      const fits = example.fits;
      const n = fits.at(-1)[0];
      const sheetOrders = new Set(example.sheets.map((sheet) => sheet.order));
      const [x0, x1, y0, y1] = [PLOT.left, w - PLOT.right, h - PLOT.bottom, PLOT.top];
      // R^2 from 0, or from 0.5 when every order fits better, so that the rise shows
      const low = Math.min(...fits.map((fit) => fit[1])) >= 0.5 ? 0.5 : 0;
      // the orders evenly spaced, and a break before n where the orders between are left out:
      // they all equal the highest order with an interaction
      const broken = n - fits.at(-2)[0] > 1;
      const slots = fits.map((_, k) => k + (broken && k === fits.length - 1 ? 0.8 : 0));
      const places = new Map(fits.map(([order], k) => [order, x0 + (slots[k] / slots.at(-1)) * (x1 - x0)]));
      // an order left out at the break lies on the dashed line, as far along it as between its ends
      const x = (order) => {
        if (places.has(order)) return places.get(order);
        const from = fits.at(-2)[0];
        return places.get(from) + ((order - from) / (n - from)) * (places.get(n) - places.get(from));
      };
      const y = (r2) => y0 - ((Math.max(low, Math.min(1, r2)) - low) / (1 - low)) * (y0 - y1);

      const grid = add(plot, "g", { class: "orders-demo-plot-grid" });
      for (let k = 0; k <= 4; k++) {
        const level = low + (k / 4) * (1 - low);
        add(grid, "line", { x1: x0, x2: x1, y1: roundTo(y(level), 1), y2: roundTo(y(level), 1) });
        if (k % 2 === 0) {
          add(plot, "text", { class: "orders-demo-plot-tick", x: x0 - 13, y: roundTo(y(level), 1), "text-anchor": "end" }, String(level));
        }
      }
      for (const [order] of fits) {
        const last = order === n;
        add(
          plot,
          "text",
          {
            class: last ? "orders-demo-plot-tick orders-demo-plot-strong" : "orders-demo-plot-tick",
            x: roundTo(x(order), 1),
            y: y0 + 14,
            "text-anchor": "middle",
          },
          String(order),
        );
      }
      // the axes' labels: the order under the axis, the faithfulness along it, turned
      const f = (value) => roundTo(value, 1);
      add(plot, "text", { class: "orders-demo-plot-label", x: f((x0 + x1) / 2), y: y0 + 31, "text-anchor": "middle" }, "order of interaction");
      const middle = f((y0 + y1) / 2);
      add(plot, "text", { class: "orders-demo-plot-label", x: 12, y: middle, "text-anchor": "middle", transform: `rotate(-90 12 ${middle})` }, "faithfulness (R²)");
      // under it, an arrow along the axis: every order to the right is more complex
      const arrowY = h - 6;
      const arrow = add(plot, "g", { class: "orders-demo-plot-complexity" });
      add(arrow, "path", { d: `M${f(x0)},${arrowY}H${f(x1)}M${f(x1 - 5)},${arrowY - 3.5}L${f(x1)},${arrowY}L${f(x1 - 5)},${arrowY + 3.5}` });
      add(arrow, "text", { class: "orders-demo-plot-axis", x: f((x0 + x1) / 2), y: arrowY, "text-anchor": "middle" }, "more complex");

      // the estimation tax: where even the full decomposition is estimated (from a proxy of the
      // model, not from all 2^n coalitions), the R^2 it falls short of 1, hatched across the plot
      const full = fits.at(-1)[1];
      // the note on the full decomposition: above its point, or, over the band, on the line of the
      // band's label, at the other end
      let note = { x: f(x(n) + 4), y: f(y(full) - 15) };
      if (full < EXACT) {
        const id = `orders-tax-${plotId}`;
        const hatch = add(add(plot, "defs", {}), "pattern", {
          id,
          width: 5,
          height: 5,
          patternUnits: "userSpaceOnUse",
          patternTransform: "rotate(45)",
        });
        add(hatch, "line", { class: "orders-demo-plot-tax-hatch", x1: 0, y1: 0, x2: 0, y2: 5 });
        const tax = add(plot, "g", { class: "orders-demo-plot-tax" });
        const band = { x: f(x0), y: f(y(1)), width: f(x1 - x0), height: f(y(full) - y(1)) };
        add(tax, "rect", { ...band, class: "orders-demo-plot-tax-fill" });
        add(tax, "rect", { ...band, fill: `url(#${id})` });
        // its label inside the band, or above the plot where the band is too thin for it
        const inside = band.height >= 15;
        const line = inside ? f((y(1) + y(full)) / 2) : f(y(1) - 8);
        add(tax, "text", { class: "orders-demo-plot-axis orders-demo-plot-tax-label", x: f(x0 + (inside ? 6 : 0)), y: line }, "estimation tax");
        note = { x: f(x1 - (inside ? 6 : 0)), y: line };
      }

      const points = fits.map(([order, r2]) => `${roundTo(x(order), 1)},${roundTo(y(r2), 1)}`);
      // the area under the line, a wash that fades from the line's highest point down to the axis
      const fade = `orders-fade-${plotId}`;
      const gradient = add(add(plot, "defs"), "linearGradient", { id: fade, x1: 0, y1: 0, x2: 0, y2: 1 });
      add(gradient, "stop", { class: "orders-demo-plot-fade-top", offset: 0 });
      add(gradient, "stop", { class: "orders-demo-plot-fade-bottom", offset: 1 });
      add(plot, "path", {
        class: "orders-demo-plot-area",
        d: `M${roundTo(x0, 1)},${y0}L${points.join("L")}L${roundTo(x1, 1)},${y0}Z`,
        fill: `url(#${fade})`,
      });
      // across the break, the line is dashed: the orders it skips have the same fit
      const solid = broken ? points.slice(0, -1) : points;
      add(plot, "path", { class: "orders-demo-plot-line", d: `M${solid.join("L")}` });
      if (broken) {
        add(plot, "path", { class: "orders-demo-plot-line orders-demo-plot-skipped", d: `M${points.slice(-2).join("L")}` });
        // the break: two slashes across the axis, between the highest order with an interaction and n
        const at = (x(fits.at(-2)[0]) + x(n)) / 2;
        const mark = add(plot, "g", { class: "orders-demo-plot-break" });
        add(mark, "rect", { x: roundTo(at - 4.5, 1), y: y0 - 4, width: 9, height: 8 });
        add(mark, "path", { d: `M${roundTo(at - 5, 1)},${y0 + 4}L${roundTo(at - 1, 1)},${y0 - 4}M${roundTo(at + 1, 1)},${y0 + 4}L${roundTo(at + 5, 1)},${y0 - 4}` });
      }
      for (const [order, r2] of fits) {
        add(plot, "circle", {
          class: sheetOrders.has(order) ? "orders-demo-plot-dot" : "orders-demo-plot-dot orders-demo-plot-between",
          cx: roundTo(x(order), 1),
          cy: roundTo(y(r2), 1),
          r: sheetOrders.has(order) ? 3 : 2.4,
        });
      }

      // the last order is the full decomposition: a note by its point
      add(plot, "text", { class: "orders-demo-plot-axis orders-demo-plot-note", ...note, "text-anchor": "end" }, "full decomposition");

      // the marker: a guide down to the axis and a ring on the line, with the order's R^2
      const guide = add(plot, "g", { class: "orders-demo-plot-guide" });
      add(guide, "line", { x1: 0, x2: 0, y1: y1, y2: y0 });
      const ring = add(plot, "g", { class: "orders-demo-plot-marker" });
      add(ring, "circle", { r: 7 });
      add(ring, "circle", { class: "orders-demo-plot-core", r: 3 });
      const label = add(ring, "text", { class: "orders-demo-plot-value", "text-anchor": "middle" });
      const score = add(label, "tspan");
      const rating = add(label, "tspan", { class: "orders-demo-plot-verdict" });
      marker = { guide, ring, label, score, rating, x, y, last: n, middle: (y0 + y1) / 2, bottom: y0 };

      // a cover over the plot: the order nearest to the pointer
      const cover = add(plot, "rect", { class: "orders-demo-plot-cover", x: x0 - 12, y: 0, width: x1 - x0 + 24, height: h });
      const nearest = (event) => {
        const box = plot.getBoundingClientRect();
        const at = ((event.clientX - box.left) * w) / box.width;
        const distance = (order) => Math.abs(x(order) - at);
        return fits.reduce((best, [order]) => (distance(order) < distance(best) ? order : best), fits[0][0]);
      };
      let touched = false;
      cover.addEventListener("pointermove", (event) => {
        if (event.pointerType === "mouse" ? !state.pinned : cover.hasPointerCapture(event.pointerId)) {
          pointAtOrder(nearest(event));
        }
      });
      cover.addEventListener("pointerdown", (event) => {
        touched = event.pointerType !== "mouse";
        if (!touched) return;
        cover.setPointerCapture(event.pointerId);
        state.pinned = true;
        pointAtOrder(nearest(event));
      });
      cover.addEventListener("pointerleave", (event) => {
        if (event.pointerType === "mouse" && !state.pinned) point(null, null);
      });
      cover.addEventListener("click", (event) => {
        if (touched) {
          touched = false;
          return;
        }
        const order = nearest(event);
        const index = current().sheets.findIndex((sheet) => sheet.order === order);
        if (index >= 0) toggle(index, null);
        else toggle(null, order);
      });
      mark();
    }

    // the ring at the order pointed at (a sheet's slid out, or a blank sheet's), hidden at rest
    function mark() {
      if (!marker) return;
      const example = current();
      const order = state.active !== null ? example.sheets[state.active].order : state.blank;
      plot.classList.toggle("orders-demo-plot-pointing", order !== null);
      if (order === null) return;
      const r2 = fitOf(example, order)[1];
      const [cx, cy] = [roundTo(marker.x(order), 1), roundTo(marker.y(r2), 1)];
      marker.ring.style.transform = `translate(${cx}px, ${cy}px)`;
      marker.guide.style.transform = `translateX(${cx}px)`;
      // the value right of the ring (where the line rises away from it), below the ring where the
      // line runs high; left of it at the last order. On narrow screens, which have no ladder, the
      // verdict in a line under it
      marker.score.textContent = `R² ${fit2(r2)}`;
      marker.rating.textContent = geometry.narrow ? verdict(r2) : "";
      const below = marker.y(r2) < marker.middle;
      const last = order === marker.last;
      const lines = geometry.narrow ? 2 : 1;
      const x = last ? 6 : below ? -4 : 12;
      let y = below ? 22 : 0;
      y -= Math.max(0, cy + y + (lines - 1) * 13 + 7 - marker.bottom); // low on the plot: above the axis
      marker.label.setAttribute("text-anchor", last ? "end" : "start");
      for (const [line, span] of [marker.score, marker.rating].entries()) {
        span.setAttribute("x", x);
        span.setAttribute("y", y + line * 13);
      }
    }

    // point at a sheet (its index), or at the blank sheet of an order between, or at nothing
    function point(active, blank) {
      // during a switch, the sheets, rows and plot on screen are still the old example's
      if (root.classList.contains("orders-demo-fading")) return;
      if (state.active === active && state.blank === blank) return;
      Object.assign(state, { active, blank });
      place(active);
      mark();
    }

    function toggle(active, blank) {
      if (state.pinned && state.active === active && state.blank === blank) {
        state.pinned = false;
        point(null, null);
      } else {
        state.pinned = true;
        point(active, blank);
      }
    }

    // the slider: a stop's name, and bringing out its sheet (the rest puts every sheet back); a sheet
    // brought out by the slider stays out, as one tapped
    const stopName = (item) => (item.sheet ? sheetName(item.sheet).name + (item.sheet.order > 1 && !item.sheet.full ? ` of order ${item.order}` : "") : `order ${item.order}`);
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
    stage.addEventListener("pointermove", (event) => {
      if (event.pointerType !== "mouse" || state.pinned || event.target.closest(".orders-demo-row, .orders-demo-rail")) return;
      const item = itemAt(pointer(event));
      if (item?.sheet) point(item.index, null);
      // a blank sheet comes out only from the stack at rest, which it does not change
      else if (item && state.active === null) point(null, item.order);
    });
    stage.addEventListener("pointerleave", (event) => {
      if (event.pointerType === "mouse" && !state.pinned) point(null, null);
    });
    // on a touch screen, a swipe sideways across the stack goes to the next example or the previous
    // one, as the stepper does: the stack follows the finger, fading, and past a fifth of its width,
    // or flicked, it leaves to that side; else it springs back. A swipe up or down scrolls the page.
    // Touch events, not pointer events: only a touchmove can be kept from the browser, which on iOS
    // takes a sideways drag over after a few pixels and cancels it for the page
    let swipe = null;
    let touching = false; // the last press on the stage was a touch: a tap brings out no sheet
    stage.addEventListener("pointerdown", (event) => {
      touching = event.pointerType === "touch";
    });
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
      if (touching) return; // on a touch screen, the stack is for swiping; the slider brings out sheets
      const item = itemAt(pointer(event));
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
        mark();
      }
      if (plotBox.clientWidth !== plotWidth) drawPlot();
    }).observe(root);
    new ResizeObserver(() => movePill(true)).observe(switcher);

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
