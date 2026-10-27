// The orders demo (demos/orders): one prediction, explained at every order of interaction, as a
// stack of sheets. The top sheet holds the Shapley values (order 1); every sheet below adds the
// interactions of one more order (k-SII), up to order 4. Below them, blank sheets stand for the
// orders that follow, up to the bottom sheet: the full decomposition, the interactions of all
// orders, which add up to the model itself. Beside every sheet is its R^2, how much of the model's
// output on random coalitions the explanation reproduces, and below the stack a plot of the R^2 of
// every order: it rises from the Shapley values to the full decomposition, which reproduces the
// model exactly.
//
// Two examples: the molecule of GraphSHAP-IQ's Figure 1, explained exactly by GraphSHAP-IQ for a
// graph convolutional network, and the photo and caption of the FIxLIP demo, explained by an
// illustrative toy game. The explanations are precomputed (../compute) into data.js. Pointing at a
// sheet (hover, keyboard focus on its row, or a tap that pins it) lifts it out of the stack: the
// sheets above it move up, those below it move down, and the sheet tilts towards the reader and
// grows. A ring marks its order on the plot; pointing along the plot lifts the sheet of the order
// pointed at, or brings out the blank sheet of an order between.
//
// The sheets are drawn in an axonometric view: a square face, turned in its plane and squashed
// vertically, so that every corner of every sheet is known here exactly. The script uses them to
// find the sheet under the pointer and to lead every row of the ladder to its sheet.
//
// Without JavaScript, the static figure in the <noscript> shows; when the data does not load, the
// script shows that figure itself.
(() => {
  const roots = [...document.querySelectorAll(".orders-demo")];
  if (!roots.length) return;

  const SVG = "http://www.w3.org/2000/svg";
  const RED = [255, 13, 87];
  const BLUE = [30, 136, 229];
  const TURN = (-45 * Math.PI) / 180; // the sheets' rotation in their plane
  // the poses of a sheet: at rest, lifted out of the stack, and receded while another is lifted;
  // scale is its size, tilt how much of its depth shows (the cosine of the view's elevation)
  const REST = { scale: 1, tilt: 0.5 };
  const LIFTED = { scale: 1.06, tilt: 0.66 };
  const RECEDED = { scale: 0.92, tilt: 0.4 };
  // in units of a sheet's side: what shows of a sheet below the one above it, at rest and while
  // one is lifted, the same for the blank sheets of the orders between, and how far the lifted
  // sheet may cover the bottom corner of the one above it
  const REST_STRIP = 0.24;
  const OPEN_STRIP = 0.115;
  const BLANK_REST_STRIP = 0.04;
  const BLANK_OPEN_STRIP = 0.028;
  const OVERLAP = 0.06;
  // how far the blank sheet of the order pointed at slides out of the stack, towards the reader:
  // less than its strip, so that the blank sheet below it still shows
  const PEEK = 0.03;
  const THICKNESS = 4; // pixels: the sheets' edge, below their face
  const PLOT = { height: 176, left: 42, right: 14, top: 14, bottom: 40 }; // pixels

  const roundTo = (value, digits) => Number(value.toFixed(digits));
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

  // an interaction of two or more players: every player joined to the group's center (for a pair,
  // to the other player), in red or blue, as wide and as opaque as the value is large
  const drawGroup = (parent, points, value, largest, base) => {
    const ratio = Math.abs(value) / largest;
    if (ratio < 0.015) return;
    const width = Math.max(0.3, base * ratio ** 0.75);
    const group = add(parent, "g", {
      fill: solid(value),
      opacity: roundTo(0.08 + 0.8 * ratio ** 1.1, 3),
    });
    const center =
      points.length === 2
        ? points[1]
        : [0, 1].map((axis) => points.reduce((sum, p) => sum + p[axis], 0) / points.length);
    for (const p of points.length === 2 ? [points[0]] : points) {
      const d = bowtie(p, center, width);
      if (d) add(group, "path", { d });
    }
    for (const [x, y] of points.length === 2 ? points : [...points, center]) {
      add(group, "circle", { cx: roundTo(x, 2), cy: roundTo(y, 2), r: roundTo(width / 2, 2) });
    }
  };

  const drawGroups = (parent, sheet, anchor, largest, base) => {
    const layer = add(parent, "g", { class: "orders-demo-groups" });
    // the weakest first, so that the strongest lie on top
    const groups = [...sheet.groups].sort((a, b) => Math.abs(a[1]) - Math.abs(b[1]));
    for (const [players, value] of groups) {
      drawGroup(layer, players.map(anchor), value, largest, base);
    }
  };

  // points scaled into a box, keeping their aspect, centered
  const fit = (points, [left, top, right, bottom]) => {
    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    const [x0, y0] = [Math.min(...xs), Math.min(...ys)];
    const [w, h] = [Math.max(...xs) - x0, Math.max(...ys) - y0];
    const scale = Math.min((right - left) / w, (bottom - top) / h);
    const [ox, oy] = [left + (right - left - w * scale) / 2, top + (bottom - top - h * scale) / 2];
    return points.map(([x, y]) => [ox + (x - x0) * scale, oy + (y - y0) * scale]);
  };

  // the molecule: its bonds and atoms over the explanation, as in GraphSHAP-IQ's figures
  function moleculeFace(svg, example, sheet) {
    const position = fit(example.coords, [9, 5, 93, 87]);
    const largest = largestValue(sheet);
    drawGroups(svg, sheet, (atom) => position[atom], largest, 4.6);
    const halos = add(svg, "g", { class: "orders-demo-halos" });
    sheet.first.forEach((value, atom) => {
      const ratio = Math.abs(value) / largest;
      if (ratio < 0.02) return;
      add(halos, "circle", {
        cx: roundTo(position[atom][0], 2),
        cy: roundTo(position[atom][1], 2),
        r: roundTo(2.3 + 4.4 * ratio, 2),
        fill: solid(value),
        opacity: roundTo(0.15 + 0.85 * ratio, 3),
      });
    });
    const bonds = add(svg, "g", { class: "orders-demo-bonds" });
    for (const [i, j] of example.bonds) {
      const [[x1, y1], [x2, y2]] = [position[i], position[j]];
      add(bonds, "line", { x1, y1, x2, y2 });
    }
    const atoms = add(svg, "g", { class: "orders-demo-atoms" });
    example.atoms.forEach((atom, i) => {
      const [x, y] = position[i];
      add(atoms, "circle", { cx: x, cy: y, r: 2.3 });
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
    sheet.first.slice(0, patches).forEach((value, patch) => {
      const ratio = Math.abs(value) / largest;
      if (ratio < 0.02) return;
      const [cx, cy] = anchor(patch);
      add(image, "rect", {
        x: roundTo(cx - cell / 2, 2),
        y: roundTo(cy - cell / 2, 2),
        width: roundTo(cell, 2),
        height: roundTo(cell, 2),
        fill: solid(value),
        opacity: roundTo(0.12 + 0.6 * ratio, 3),
      });
    });
    const lines = add(image, "g", { class: "orders-demo-grid" });
    for (let k = 1; k < grid; k++) {
      const at = roundTo(photo.x + k * cell, 2);
      const across = roundTo(photo.y + k * cell, 2);
      add(lines, "line", { x1: at, y1: photo.y, x2: at, y2: photo.y + photo.size });
      add(lines, "line", { x1: photo.x, y1: across, x2: photo.x + photo.size, y2: across });
    }

    drawGroups(svg, sheet, anchor, largest, 2.3);

    const words = add(svg, "g", { class: "orders-demo-tokens" });
    example.tokens.forEach((token, k) => {
      const value = sheet.first[patches + k];
      const ratio = Math.abs(value) / largest;
      const y = tokens.top + k * row + row * 0.12;
      const height = row * 0.76;
      add(words, "rect", { x: tokens.x, y: roundTo(y, 2), width: tokens.width, height: roundTo(height, 2), rx: 1.4 });
      if (ratio >= 0.02) {
        add(words, "rect", {
          class: "orders-demo-tint",
          x: tokens.x,
          y: roundTo(y, 2),
          width: tokens.width,
          height: roundTo(height, 2),
          rx: 1.4,
          fill: solid(value),
          opacity: roundTo(0.12 + 0.6 * ratio, 3),
        });
      }
      add(words, "text", { x: tokens.x + tokens.width / 2, y: roundTo(y + height / 2 + 0.1, 2) }, token);
    });
  }

  // -- the sheets' names and the captions of the examples ---------------------------------------------

  // the name of a sheet: on its face, as its row's tag, and in words
  const sheetName = (sheet) => {
    if (sheet.full) return { face: "Full decomposition", tag: "n-SII", name: "Full decomposition" };
    if (sheet.order === 1) return { face: "SV", tag: "SV", name: "Shapley values" };
    const tag = `${sheet.order}-SII`;
    return { face: tag, tag, name: "Shapley interactions" };
  };
  const fit2 = (r2) => r2.toFixed(2);
  const verdict = (r2) => {
    if (r2 >= 0.9995) return "exact";
    if (r2 >= 0.95) return "good";
    if (r2 >= 0.8) return "workable";
    return "poor";
  };
  const count = (n) => n.toLocaleString("en-US");

  // what the caption says, at rest (nothing pointed at), for every sheet, and for every order between
  // the sheets (a blank sheet: its order, R^2 and number of values); the numbers are the data's
  const CAPTIONS = {
    molecule: (example, { sheet, blank }) => {
      if (blank) {
        const [order, r2, values] = blank;
        return [
          [`Shapley interactions of order ${order}: ${count(values)} values of up to ${order} atoms. The orders between 4 and the full decomposition add thousands of values but change the fit little: `, ["R² = ", fit2(r2)], "."],
        ];
      }
      if (!sheet) {
        const p = `${Math.round(example.probability * 1000) / 10}%`;
        return [
          ["A graph neural network predicts that this molecule is mutagenic (", ["probability ", p], "), and every sheet explains this prediction. "],
          ["The Shapley values on top credit every atom alone; each order below adds the interactions of larger groups of atoms, down to the full decomposition, which adds up to the model exactly."],
        ];
      }
      const fit = ["R² = ", fit2(sheet.r2)];
      if (sheet.order === 1) {
        return [
          ["Shapley values: one number per atom. The nitrogen atoms of the three nitro groups get the most credit, but the 30 numbers reproduce the model's output on random parts of the molecule only with ", fit, "."],
        ];
      }
      if (sheet.full) {
        return [
          [`The full decomposition: every group of atoms that the network's two layers see together, ${count(sheet.count)} values of up to ${sheet.order} atoms (larger groups have none), computed exactly by GraphSHAP-IQ with ${count(example.budget)} calls of the model instead of 2³⁰. It is the model itself, written as a sum: `, fit, "."],
        ];
      }
      const texts = {
        2: "The credit moves from the nitrogen atoms to the bonds of their nitro groups, but pairs of atoms are not enough yet: ",
        3: "Groups of three atoms add little: ",
        4: "Groups of four atoms: each nitro group with its carbon, and pieces of the carbon rings. With them, the explanation reproduces the model almost exactly: ",
      };
      return [
        [`Shapley interactions of order ${sheet.order}: ${count(sheet.count)} values of up to ${sheet.order} atoms. ${texts[sheet.order] ?? ""}`, fit, "."],
      ];
    },
    fixlip: (example, { sheet, blank }) => {
      const illustrative = " Illustrative values, after the FIxLIP demo.";
      if (blank) {
        const [order, r2, values] = blank;
        return [
          [`Groups of up to ${order} patches and words: ${count(values)} values. Every order between 4 and the full decomposition adds a little more: `, ["R² = ", fit2(r2)], ".", illustrative],
        ];
      }
      if (!sheet) {
        return [
          ["How well does SigLIP 2 match the photo and its caption? Every sheet explains it, from the patches and words alone (top) to the full decomposition (bottom).", illustrative],
        ];
      }
      const fit = ["R² = ", fit2(sheet.r2)];
      const texts = {
        1: ["Shapley values credit every patch and word alone: “elephant”, and the elephant's head. They miss that “white” and “bird” matter only on the egret: ", fit, ", a poor fit."],
        2: ["Pairs link the words to the regions they name: “bird” and “white” with the egret, “elephant” with the elephant. ", fit, ", a workable explanation."],
        3: ["Groups of three: “white bird” on the egret, “small elephant” on its head. ", fit, "."],
        4: ["“next to” needs both animals: groups of four patches and words. ", fit, "."],
      };
      if (sheet.full) {
        return [
          ["The full decomposition: all orders, up to “a small elephant … next to a white bird” as one group of eight patches and words. ", fit, ".", illustrative],
        ];
      }
      return [texts[sheet.order]];
    },
  };

  // -- the demo -------------------------------------------------------------------------------------

  function start(root, data) {
    // what is pointed at: a lifted sheet (its index), or the blank sheet of an order between
    const state = { example: data.examples[0].id, active: null, blank: null, pinned: false };
    let geometry = null;
    let instance = 0;

    root.replaceChildren();
    const bar = el("div", "orders-demo-bar");
    const switcher = el("div", "orders-demo-switch");
    switcher.setAttribute("role", "group");
    switcher.setAttribute("aria-label", "Example");
    const options = data.examples.map((example) => {
      const option = el("button", "orders-demo-option");
      option.type = "button";
      option.append(el("span", "orders-demo-option-label", example.label), el("span", "orders-demo-option-detail", example.detail));
      option.addEventListener("click", () => {
        if (state.example === example.id) return;
        Object.assign(state, { example: example.id, active: null, blank: null, pinned: false });
        render();
      });
      switcher.append(option);
      return [example.id, option];
    });
    const hint = el("p", "orders-demo-hint", "Point at a sheet, or along the plot.");
    bar.append(switcher, hint);

    const stage = el("div", "orders-demo-stage");
    stage.setAttribute("role", "group");
    const stack = el("div", "orders-demo-stack");
    stack.setAttribute("aria-hidden", "true");
    const ladder = el("div", "orders-demo-ladder");
    stage.append(stack, ladder);

    const footer = el("div", "orders-demo-footer");
    const plotBox = el("figure", "orders-demo-plot");
    const plotTitle = el("figcaption", "orders-demo-plot-title", "Fit to the model (R²) by order");
    const plot = document.createElementNS(SVG, "svg");
    plot.setAttribute("class", "orders-demo-plot-svg");
    plot.setAttribute("role", "img");
    plotBox.append(plotTitle, plot);
    const caption = el("p", "orders-demo-caption");
    caption.setAttribute("aria-live", "polite");
    footer.append(plotBox, caption);
    root.append(bar, stage, footer);
    root.classList.add("orders-demo-ready");

    // the stack, top to bottom: the sheets of order 1 to 4, a blank sheet for every order up to
    // the full decomposition, and the sheet of the full decomposition
    let items = [];
    let rows = [];
    let blankRow = null;
    let marker = null;
    let plotWidth = 0;

    const current = () => data.examples.find((example) => example.id === state.example);
    const sheetItems = () => items.filter((item) => item.sheet);
    const blankItems = () => items.filter((item) => !item.sheet);

    // -- geometry: the size of the sheets, the stage, and every sheet's place and pose ---------------

    function measure() {
      const width = stage.clientWidth;
      const narrow = width < 560;
      const ladderWidth = narrow ? Math.max(92, width * 0.3) : Math.min(240, width * 0.3);
      const gap = narrow ? 8 : 28;
      const spread = Math.abs(Math.cos(TURN)) + Math.abs(Math.sin(TURN));
      const room = width - ladderWidth - gap;
      const side = Math.max(110, Math.min(360, room / (LIFTED.scale * spread)));
      const height = (pose) => side * pose.scale * pose.tilt * spread;
      // the same height for every example: the most sheets and blank sheets of any
      const n = Math.max(...data.examples.map((example) => example.sheets.length));
      const blanks = Math.max(
        ...data.examples.map((example) => {
          const orders = example.sheets.map((sheet) => sheet.order);
          return orders.at(-1) - orders.at(-2) - 1;
        }),
      );
      // the tallest stack: the second sheet lifted, the first above it, the others below
      const open =
        height(RECEDED) + height(LIFTED) - OVERLAP * side + (n - 2) * OPEN_STRIP * side + blanks * BLANK_OPEN_STRIP * side;
      const rest = height(REST) + (n - 1) * REST_STRIP * side + blanks * BLANK_REST_STRIP * side;
      return {
        width,
        narrow,
        side,
        ladderLeft: width - ladderWidth,
        center: room / 2,
        height: Math.ceil(Math.max(open, rest) + 2 * THICKNESS + 16),
        poseHeight: height,
      };
    }

    // the place (center) and pose of every item: stacked so that a strip of every sheet shows below
    // the one above it, and the lifted sheet, which lies over all others, shows whole
    function layout(active) {
      const g = geometry;
      const lifted = active === null ? null : sheetItems()[active];
      const centers = [];
      const poses = [];
      let bottom = 0;
      items.forEach((item, position) => {
        const pose = lifted === null ? REST : item === lifted ? LIFTED : RECEDED;
        const h = g.poseHeight(pose);
        if (position === 0) bottom = h;
        else if (item === lifted) bottom += h - OVERLAP * g.side;
        else if (!item.sheet) bottom += (lifted === null ? BLANK_REST_STRIP : BLANK_OPEN_STRIP) * g.side;
        else bottom += (lifted === null ? REST_STRIP : OPEN_STRIP) * g.side;
        centers.push(bottom - h / 2);
        poses.push(pose);
      });
      const offset = (g.height - bottom) / 2;
      return items.map((item, position) => ({
        pose: poses[position],
        x: g.center,
        y: centers[position] + offset + (item.order === state.blank ? PEEK * g.side : 0),
      }));
    }

    // the corners of a sheet on the stage (left, top, right, bottom for the turn of -45 degrees)
    const corners = ({ pose, x, y }) => {
      const a = geometry.side / 2;
      return [
        [-a, -a],
        [a, -a],
        [a, a],
        [-a, a],
      ].map(([u, v]) => [
        x + pose.scale * (u * Math.cos(TURN) - v * Math.sin(TURN)),
        y + pose.scale * pose.tilt * (u * Math.sin(TURN) + v * Math.cos(TURN)),
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

    const zIndex = (item, active) =>
      active !== null && item === sheetItems()[active] ? items.length + 1 : items.length - items.indexOf(item);

    // the item under a point of the stage, where the sheets are going (not where they are while
    // they move), the one on top first
    function itemAt(point) {
      const places = layout(state.active);
      const order = items.map((_, position) => position);
      order.sort((a, b) => zIndex(items[b], state.active) - zIndex(items[a], state.active));
      const position = order.find((p) => inside(point, corners(places[p])));
      return position === undefined ? null : items[position];
    }

    // -- drawing --------------------------------------------------------------------------------

    function render() {
      const example = current();
      instance += 1;
      for (const [id, option] of options) option.setAttribute("aria-pressed", String(id === example.id));
      stage.setAttribute("aria-label", `${example.label}: its explanations of order 1 to n, stacked, with their fit`);
      stack.replaceChildren();
      for (const row of [...rows, blankRow]) row?.remove();
      geometry = measure();

      const last = example.sheets.at(-1);
      const between = example.fits.filter(([order]) => order > example.sheets.at(-2).order && order < last.order);
      items = [];
      example.sheets.forEach((sheet, i) => {
        if (sheet === last) {
          for (const fitted of between) items.push({ order: fitted[0], fitted });
        }
        const node = el("div", "orders-demo-sheet");
        const svg = add(node, "svg", { viewBox: "0 0 100 100", class: "orders-demo-face" });
        if (example.kind === "graph") moleculeFace(svg, example, sheet);
        else imageTextFace(svg, example, sheet, `${root.dataset.root}${example.image}`, `orders-${instance}-${i}`);
        // the name along the bottom edge, from the bottom corner: the part of a sheet that shows
        add(svg, "text", { class: "orders-demo-tag", x: 6, y: 95.5 }, sheetName(sheet).face);
        items.push({ order: sheet.order, sheet, index: i, node });
      });
      for (const item of items) {
        if (!item.sheet) item.node = el("div", "orders-demo-sheet orders-demo-blank");
      }
      stack.append(...items.map((item) => item.node));

      rows = example.sheets.map((sheet, i) => {
        const names = sheetName(sheet);
        const row = el("button", "orders-demo-row");
        row.type = "button";
        row.setAttribute("aria-label", `${names.name}, order ${sheet.full ? "n" : sheet.order}: R² ${fit2(sheet.r2)}, ${verdict(sheet.r2)}`);
        const title = el("span", "orders-demo-row-title");
        title.append(el("strong", "", names.tag), el("span", "orders-demo-row-name", names.name));
        const fitted = el("span", "orders-demo-row-fit");
        const meter = el("span", "orders-demo-meter");
        const fill = el("span", "orders-demo-meter-fill");
        fill.style.width = `${Math.max(0, Math.min(1, sheet.r2)) * 100}%`;
        meter.append(fill);
        fitted.append(
          meter,
          el("span", "orders-demo-r2", fit2(sheet.r2)),
          el("span", `orders-demo-verdict orders-demo-verdict-${verdict(sheet.r2)}`, verdict(sheet.r2)),
        );
        row.append(title, fitted);
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

      // one row for all blank sheets: the orders between, which the plot shows one by one
      blankRow = null;
      if (between.length) {
        blankRow = el("div", "orders-demo-row orders-demo-blank-row");
        blankRow.setAttribute("aria-hidden", "true");
        const title = el("span", "orders-demo-row-title");
        title.append(el("strong", "", "⋮"), el("span", "orders-demo-row-name"));
        blankRow.append(title);
        ladder.append(blankRow);
      }

      drawPlot();
      // the stack opens out of a pile: first all sheets on top of each other, then their strips
      // show; without motion, the stack is shown at rest at once
      place("pile");
      if (reducedMotion()) {
        place(state.active);
      } else {
        stack.getBoundingClientRect(); // the pile is drawn before the stack opens
        requestAnimationFrame(() => place(state.active));
      }
      describe();
    }

    // move every sheet and row to its place, for a lifted sheet (or null at rest, "pile" for the
    // sheets on top of each other before they open)
    function place(active) {
      const g = geometry;
      stage.style.height = `${g.height}px`;
      root.classList.toggle("orders-demo-narrow", g.narrow);
      const lifted = typeof active === "number";
      const places = active === "pile" ? layout(null).map((p) => ({ ...p, y: g.height / 2 })) : layout(active);
      items.forEach((item, position) => {
        const { pose, x, y } = places[position];
        const node = item.node;
        node.style.width = node.style.height = `${g.side}px`;
        node.style.marginLeft = node.style.marginTop = `${-g.side / 2}px`;
        node.style.zIndex = lifted && item.index === active ? items.length + 1 : items.length - position;
        node.style.transform =
          `translate(${roundTo(x, 2)}px, ${roundTo(y, 2)}px) scale(${pose.scale}) ` +
          `scaleY(${pose.tilt}) rotate(${(TURN * 180) / Math.PI}deg)`;
        // the edge below the face, in the face's own coordinates: straight down on the stage
        const depth = (item.sheet ? THICKNESS : THICKNESS / 2) / (pose.scale * pose.tilt);
        node.style.setProperty("--orders-edge-x", `${roundTo(Math.sin(TURN) * depth, 2)}px`);
        node.style.setProperty("--orders-edge-y", `${roundTo(Math.cos(TURN) * depth, 2)}px`);
        node.classList.toggle("orders-demo-lifted", lifted && item.index === active);
        node.classList.toggle("orders-demo-receded", lifted && item.index !== active);
        node.classList.toggle("orders-demo-peek", !item.sheet && item.order === state.blank);

        if (!item.sheet) return;
        const row = rows[item.index];
        const right = rightCorner(places[position]);
        row.style.left = `${g.ladderLeft}px`;
        row.style.transform = `translateY(${roundTo(right[1], 2)}px) translateY(-50%)`;
        row.style.setProperty("--orders-lead", `${Math.max(0, roundTo(g.ladderLeft - right[0] - 4, 2))}px`);
        row.classList.toggle("orders-demo-active", lifted && item.index === active);
        row.classList.toggle("orders-demo-compact", lifted && item.index !== active);
        row.setAttribute("aria-pressed", String(lifted && item.index === active));
      });

      // the row of the blank sheets: beside the one pointed at, or the middle one
      if (blankRow) {
        const blanks = blankItems();
        const target = blanks.find((item) => item.order === state.blank) ?? blanks[Math.floor((blanks.length - 1) / 2)];
        const right = rightCorner(places[items.indexOf(target)]);
        blankRow.style.left = `${g.ladderLeft}px`;
        blankRow.style.transform = `translateY(${roundTo(right[1], 2)}px) translateY(-50%)`;
        blankRow.style.setProperty("--orders-lead", `${Math.max(0, roundTo(g.ladderLeft - right[0] - 4, 2))}px`);
        blankRow.classList.toggle("orders-demo-active", state.blank !== null);
        blankRow.classList.toggle("orders-demo-compact", lifted);
        const [first, last] = [blanks[0].order, blanks.at(-1).order];
        blankRow.querySelector(".orders-demo-row-name").textContent =
          state.blank !== null
            ? `Order ${state.blank}: R² ${fit2(blanks.find((item) => item.order === state.blank).fitted[1])}`
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
        `R² by order: ${example.fits.map(([order, r2]) => `${order}: ${fit2(r2)}`).join(", ")}; the highest order is the full decomposition`,
      );
      plot.replaceChildren();
      const fits = example.fits;
      const largest = fits.at(-1)[0];
      const sheetOrders = new Set(example.sheets.map((sheet) => sheet.order));
      const [x0, x1, y0, y1] = [PLOT.left, w - PLOT.right, h - PLOT.bottom, PLOT.top];
      // R^2 from 0, or from 0.5 when every order fits better, so that the rise shows
      const low = Math.min(...fits.map((fitted) => fitted[1])) >= 0.5 ? 0.5 : 0;
      const x = (order) => x0 + ((order - 1) / (largest - 1)) * (x1 - x0);
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
        const label = add(plot, "text", { class: "orders-demo-plot-tick", x: roundTo(x(order), 1), y: y0 + 14, "text-anchor": "middle" }, String(order));
        if (order === largest) label.classList.add("orders-demo-plot-strong");
      }
      add(plot, "text", { class: "orders-demo-plot-axis", x: x0, y: h - 6 }, "order");
      add(plot, "text", { class: "orders-demo-plot-axis", x: x1, y: h - 6, "text-anchor": "end" }, "full decomposition");

      const points = fits.map(([order, r2]) => `${roundTo(x(order), 1)},${roundTo(y(r2), 1)}`);
      add(plot, "path", { class: "orders-demo-plot-area", d: `M${roundTo(x0, 1)},${y0}L${points.join("L")}L${roundTo(x1, 1)},${y0}Z` });
      add(plot, "path", { class: "orders-demo-plot-line", d: `M${points.join("L")}` });
      for (const [order, r2] of fits) {
        add(plot, "circle", {
          class: sheetOrders.has(order) ? "orders-demo-plot-dot" : "orders-demo-plot-dot orders-demo-plot-between",
          cx: roundTo(x(order), 1),
          cy: roundTo(y(r2), 1),
          r: sheetOrders.has(order) ? 3 : 2.4,
          "data-order": order,
        });
      }

      // the marker: a guide down to the axis and a ring on the line, with the order's R^2
      const guide = add(plot, "g", { class: "orders-demo-plot-guide" });
      add(guide, "line", { x1: 0, x2: 0, y1: y1, y2: y0 });
      const ring = add(plot, "g", { class: "orders-demo-plot-marker" });
      add(ring, "circle", { r: 7 });
      add(ring, "circle", { class: "orders-demo-plot-core", r: 3 });
      const label = add(ring, "text", { class: "orders-demo-plot-label", "text-anchor": "middle" });
      marker = { guide, ring, label, x, y, fits, largest, middle: (y0 + y1) / 2 };

      // a cover over the plot: the order nearest to the pointer
      const cover = add(plot, "rect", { class: "orders-demo-plot-cover", x: x0 - 12, y: 0, width: x1 - x0 + 24, height: h });
      const nearest = (event) => {
        const box = plot.getBoundingClientRect();
        const at = ((event.clientX - box.left) * w) / box.width;
        const order = Math.round(1 + ((at - x0) / (x1 - x0)) * (largest - 1));
        return Math.max(1, Math.min(largest, order));
      };
      let touched = false;
      cover.addEventListener("pointermove", (event) => {
        if (event.pointerType === "mouse" ? !state.pinned : cover.hasPointerCapture(event.pointerId)) {
          pointAtOrder(nearest(event));
        }
      });
      cover.addEventListener("pointerdown", (event) => {
        if (event.pointerType === "mouse") return;
        touched = true;
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

    // the ring at the order pointed at (a lifted sheet's, or a blank sheet's), hidden at rest
    function mark() {
      if (!marker) return;
      const example = current();
      const order = state.active !== null ? example.sheets[state.active].order : state.blank;
      plot.classList.toggle("orders-demo-plot-pointing", order !== null);
      for (const dot of plot.querySelectorAll(".orders-demo-plot-dot")) {
        dot.classList.toggle("orders-demo-plot-dot-active", Number(dot.dataset.order) === order);
      }
      if (order === null) return;
      const r2 = marker.fits.find(([k]) => k === order)[1];
      const [cx, cy] = [roundTo(marker.x(order), 1), roundTo(marker.y(r2), 1)];
      marker.ring.style.transform = `translate(${cx}px, ${cy}px)`;
      marker.guide.style.transform = `translateX(${cx}px)`;
      // the value below the ring, under the line where it runs high, else right of the ring, where
      // the line rises away from it
      marker.label.textContent = `R² ${fit2(r2)}`;
      const below = marker.y(r2) < marker.middle;
      const anchor = !below ? "start" : order === 1 ? "start" : order === marker.largest ? "end" : "middle";
      marker.label.setAttribute("text-anchor", anchor);
      marker.label.setAttribute("x", !below ? 12 : anchor === "start" ? -6 : anchor === "end" ? 6 : 0);
      marker.label.setAttribute("y", below ? 22 : 0);
    }

    function describe() {
      const example = current();
      const sheet = state.active === null ? null : example.sheets[state.active];
      const blank = state.blank === null ? null : example.fits.find(([order]) => order === state.blank);
      caption.replaceChildren();
      for (const sentence of CAPTIONS[example.id](example, { sheet, blank })) {
        for (const part of sentence) {
          if (Array.isArray(part)) {
            const strong = el("strong", "");
            strong.append(part[0], el("span", "orders-demo-number", part[1]));
            caption.append(strong);
          } else {
            caption.append(part);
          }
        }
      }
    }

    // point at a sheet (its index), or at the blank sheet of an order between, or at nothing
    function point(active, blank) {
      if (state.active === active && state.blank === blank) return;
      Object.assign(state, { active, blank });
      place(active);
      describe();
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
      if (event.pointerType !== "mouse" || state.pinned || event.target.closest(".orders-demo-row")) return;
      const item = itemAt(pointer(event));
      if (item?.sheet) point(item.index, null);
      // a blank sheet comes out only from the stack at rest, which it does not change
      else if (item && state.active === null) point(null, item.order);
    });
    stage.addEventListener("pointerleave", (event) => {
      if (event.pointerType === "mouse" && !state.pinned) point(null, null);
    });
    stack.addEventListener("click", (event) => {
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
      }
      if (plotBox.clientWidth !== plotWidth) drawPlot();
    }).observe(root);

    render();
  }

  // -- loading ----------------------------------------------------------------------------------------

  const showFallback = (root) => {
    // with scripts on, a <noscript>'s content is its HTML as text
    const fallback = root.querySelector("noscript")?.textContent ?? "";
    root.classList.remove("orders-demo-ready");
    root.classList.add("orders-demo-failed");
    root.innerHTML = fallback;
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
    loader.src = `${roots[0].dataset.root}data.js`;
    loader.onload = begin;
    loader.onerror = () => roots.forEach(showFallback);
    document.head.appendChild(loader);
  }
})();
