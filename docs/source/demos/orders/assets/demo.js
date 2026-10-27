// The orders demo (demos/orders): one prediction, explained at every order of interaction, as a
// stack of sheets. The top sheet holds the Shapley values (order 1); every sheet below adds the
// interactions of one more order (k-SII), and the bottom one holds the Möbius transform, the
// interactions of all orders, which add up to the model itself. Next to every sheet is its R^2,
// how much of the model's output on random coalitions the explanation reproduces: it rises from the
// top to the bottom, from the Shapley values to the Möbius transform, which reproduces it exactly.
//
// Two examples: the molecule of GraphSHAP-IQ's Figure 1, explained exactly by GraphSHAP-IQ for a
// graph convolutional network, and the photo and caption of the FIxLIP demo, explained by an
// illustrative toy game. The explanations are precomputed (../compute) into data.js. Pointing at a
// sheet (hover, keyboard focus on its row, or a tap that pins it) lifts it out of the stack: the
// sheets above it move up, those below it move down, and the sheet tilts towards the reader and
// grows.
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
  // one is lifted, and how far the lifted sheet may cover the bottom corner of the one above it
  const REST_STRIP = 0.27;
  const OPEN_STRIP = 0.115;
  const OVERLAP = 0.06;
  const THICKNESS = 4; // pixels: the sheets' edge, below their face

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
    const position = fit(example.coords, [9, 15, 93, 93]);
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
    const photo = { x: 4, y: 20, size: 62 };
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

  const sheetName = (sheet) => {
    if (sheet.moebius) return { tag: "Möbius", order: "Order n", name: "Möbius interactions" };
    if (sheet.order === 1) return { tag: "SV", order: "Order 1", name: "Shapley values" };
    return { tag: `${sheet.order}-SII`, order: `Order ${sheet.order}`, name: "Shapley interactions" };
  };
  const fit2 = (r2) => r2.toFixed(2);
  const verdict = (r2) => {
    if (r2 >= 0.9995) return "exact";
    if (r2 >= 0.95) return "good";
    if (r2 >= 0.8) return "workable";
    return "poor";
  };
  const count = (n) => n.toLocaleString("en-US");

  // what the caption says, at rest (sheet null) and for every sheet; the numbers are the data's
  const CAPTIONS = {
    molecule: (example, sheet) => {
      const p = `${Math.round(example.probability * 1000) / 10}%`;
      if (!sheet) {
        return [
          ["A graph neural network predicts that this molecule is mutagenic (", ["probability ", p], "), and every sheet explains this prediction. "],
          ["The Shapley values on top credit every atom alone; each order below adds the interactions of larger groups of atoms, down to the Möbius transform, which adds up to the model exactly."],
        ];
      }
      const fit = ["R² = ", fit2(sheet.r2)];
      if (sheet.order === 1) {
        return [
          ["Shapley values: one number per atom. The nitrogen atoms of the three nitro groups get the most credit, but the 30 numbers reproduce the model's output on random parts of the molecule only with ", fit, "."],
        ];
      }
      if (sheet.moebius) {
        return [
          [`The Möbius transform: every group of atoms that the network's two layers see together, ${count(sheet.count)} values of up to ${sheet.order} atoms, computed exactly by GraphSHAP-IQ with ${count(example.budget)} calls of the model instead of 2³⁰. It is the model itself, written as a sum: `, fit, "."],
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
    fixlip: (example, sheet) => {
      if (!sheet) {
        return [
          ["How well does SigLIP 2 match the photo and its caption? Every sheet explains it, from the patches and words alone (top) to the interactions of all orders (bottom). "],
          ["Illustrative values, after the FIxLIP demo."],
        ];
      }
      const fit = ["R² = ", fit2(sheet.r2)];
      const texts = {
        1: ["Shapley values credit every patch and word alone: “elephant”, and the elephant's head. They miss that “white” and “bird” matter only on the egret: ", fit, ", a poor fit."],
        2: ["Pairs link the words to the regions they name: “bird” and “white” with the egret, “elephant” with the elephant. ", fit, ", a workable explanation."],
        3: ["Groups of three: “white bird” on the egret, “small elephant” on its head. ", fit, "."],
        4: ["“next to” needs both animals: groups of four patches and words. ", fit, "."],
      };
      if (sheet.moebius) {
        return [
          ["All orders, up to “a small elephant … next to a white bird” as one group of eight patches and words: the whole decomposition, ", fit, "."],
          [" Illustrative values, after the FIxLIP demo."],
        ];
      }
      return [texts[sheet.order]];
    },
  };

  // -- the demo -------------------------------------------------------------------------------------

  function start(root, data) {
    const state = { example: data.examples[0].id, active: null, pinned: false };
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
        state.example = example.id;
        state.active = null;
        state.pinned = false;
        render();
      });
      switcher.append(option);
      return [example.id, option];
    });
    const hint = el("p", "orders-demo-hint", "Point at a sheet to lift it out of the stack.");
    bar.append(switcher, hint);

    const stage = el("div", "orders-demo-stage");
    stage.setAttribute("role", "group");
    const stack = el("div", "orders-demo-stack");
    stack.setAttribute("aria-hidden", "true");
    const ladder = el("div", "orders-demo-ladder");
    stage.append(stack, ladder);
    const caption = el("p", "orders-demo-caption");
    caption.setAttribute("aria-live", "polite");
    root.append(bar, stage, caption);
    root.classList.add("orders-demo-ready");

    let sheets = [];
    let rows = [];

    const current = () => data.examples.find((example) => example.id === state.example);

    // -- geometry: the size of the sheets, the stage, and every sheet's place and pose ---------------

    function measure() {
      const width = stage.clientWidth;
      const narrow = width < 560;
      const ladderWidth = narrow ? Math.max(92, width * 0.3) : Math.min(240, width * 0.3);
      const gap = narrow ? 8 : 28;
      const spread = Math.abs(Math.cos(TURN)) + Math.abs(Math.sin(TURN));
      const room = width - ladderWidth - gap;
      const side = Math.max(110, Math.min(360, room / (LIFTED.scale * spread)));
      const n = current().sheets.length;
      const height = (pose) => side * pose.scale * pose.tilt * spread;
      // the tallest stack: the second sheet lifted, the first above it, the others below
      const open = height(RECEDED) + height(LIFTED) - OVERLAP * side + (n - 2) * OPEN_STRIP * side;
      const rest = height(REST) + (n - 1) * REST_STRIP * side;
      return {
        width,
        narrow,
        side,
        spread,
        ladderLeft: width - ladderWidth,
        center: room / 2,
        height: Math.ceil(Math.max(open, rest) + 2 * THICKNESS + 16),
        poseHeight: height,
      };
    }

    // the place (center) and pose of every sheet: stacked so that a strip of every sheet shows below
    // the one above it, and the lifted sheet, which lies over all others, shows whole
    function layout(active) {
      const g = geometry;
      const n = sheets.length;
      const poses = sheets.map((_, i) => (active === null ? REST : i === active ? LIFTED : RECEDED));
      const strip = (active === null ? REST_STRIP : OPEN_STRIP) * g.side;
      const centers = [];
      let bottom = 0;
      for (let i = 0; i < n; i++) {
        const h = g.poseHeight(poses[i]);
        if (i === 0) bottom = h;
        else if (i === active) bottom += h - OVERLAP * g.side;
        else bottom += strip;
        centers.push(bottom - h / 2);
      }
      const offset = (g.height - bottom) / 2;
      return poses.map((pose, i) => ({ pose, x: g.center, y: centers[i] + offset }));
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

    // the sheet under a point of the stage, where the sheets are going (not where they are while
    // they move), the one on top first
    function sheetAt(point) {
      const places = layout(state.active);
      const order = places.map((_, i) => i).sort((a, b) => zIndex(b) - zIndex(a));
      return order.find((i) => inside(point, corners(places[i]))) ?? null;
    }

    const zIndex = (i) => (i === state.active ? sheets.length + 1 : sheets.length - i);

    // -- drawing --------------------------------------------------------------------------------

    function render() {
      const example = current();
      instance += 1;
      for (const [id, option] of options) option.setAttribute("aria-pressed", String(id === example.id));
      stage.setAttribute(
        "aria-label",
        `${example.label}: its explanations of order 1 to n, stacked, with their fit`,
      );
      stack.replaceChildren();
      for (const row of rows) row.remove();
      geometry = measure();

      sheets = example.sheets.map((sheet, i) => {
        const node = el("div", "orders-demo-sheet");
        const svg = add(node, "svg", { viewBox: "0 0 100 100", class: "orders-demo-face" });
        if (example.kind === "graph") moleculeFace(svg, example, sheet);
        else imageTextFace(svg, example, sheet, `${root.dataset.root}${example.image}`, `orders-${instance}-${i}`);
        add(svg, "text", { class: "orders-demo-tag", x: 5.5, y: 9.5 }, sheetName(sheet).tag);
        stack.append(node);
        return node;
      });

      rows = example.sheets.map((sheet, i) => {
        const names = sheetName(sheet);
        const row = el("button", "orders-demo-row");
        row.type = "button";
        row.setAttribute(
          "aria-label",
          `${names.order}, ${names.name}: R² ${fit2(sheet.r2)}, ${verdict(sheet.r2)}`,
        );
        const title = el("span", "orders-demo-row-title");
        title.append(el("strong", "", names.tag), el("span", "orders-demo-row-name", names.name));
        const fit = el("span", "orders-demo-row-fit");
        const meter = el("span", "orders-demo-meter");
        const fill = el("span", "orders-demo-meter-fill");
        fill.style.width = `${Math.max(0, Math.min(1, sheet.r2)) * 100}%`;
        meter.append(fill);
        fit.append(
          meter,
          el("span", "orders-demo-r2", fit2(sheet.r2)),
          el("span", `orders-demo-verdict orders-demo-verdict-${verdict(sheet.r2)}`, verdict(sheet.r2)),
        );
        row.append(title, fit);
        row.addEventListener("pointerenter", (event) => {
          if (event.pointerType === "mouse" && !state.pinned) activate(i);
        });
        row.addEventListener("focus", () => {
          if (!state.pinned) activate(i);
        });
        row.addEventListener("click", () => toggle(i));
        row.addEventListener("keydown", (event) => {
          const step = { ArrowDown: 1, ArrowUp: -1 }[event.key];
          if (step) {
            event.preventDefault();
            rows[(i + step + rows.length) % rows.length].focus();
          } else if (event.key === "Escape") {
            state.pinned = false;
            activate(null);
          }
        });
        ladder.append(row);
        return row;
      });

      // the stack opens out of a pile: first all sheets on top of each other, then their strips
      // show; without motion, the stack is shown at rest at once
      place(reducedMotion() ? state.active : "pile");
      if (!reducedMotion()) {
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
      root.classList.toggle("orders-demo-open", active !== null && active !== "pile");
      const places =
        active === "pile"
          ? layout(null).map((p) => ({ ...p, y: g.height / 2 }))
          : layout(active);
      places.forEach(({ pose, x, y }, i) => {
        const sheet = sheets[i];
        sheet.style.width = sheet.style.height = `${g.side}px`;
        sheet.style.marginLeft = sheet.style.marginTop = `${-g.side / 2}px`;
        sheet.style.zIndex = active === i ? sheets.length + 1 : sheets.length - i;
        sheet.style.transform =
          `translate(${roundTo(x, 2)}px, ${roundTo(y, 2)}px) scale(${pose.scale}) ` +
          `scaleY(${pose.tilt}) rotate(${(TURN * 180) / Math.PI}deg)`;
        // the edge below the face, in the face's own coordinates: straight down on the stage
        const depth = THICKNESS / (pose.scale * pose.tilt);
        const [dx, dy] = [Math.sin(TURN) * depth, Math.cos(TURN) * depth];
        sheet.style.setProperty("--orders-edge-x", `${roundTo(dx, 2)}px`);
        sheet.style.setProperty("--orders-edge-y", `${roundTo(dy, 2)}px`);
        sheet.classList.toggle("orders-demo-lifted", active === i);
        sheet.classList.toggle("orders-demo-receded", typeof active === "number" && active !== i);

        const row = rows[i];
        const right = corners({ pose, x, y }).reduce((a, b) => (b[0] > a[0] ? b : a));
        row.style.left = `${g.ladderLeft}px`;
        row.style.transform = `translateY(${roundTo(right[1], 2)}px) translateY(-50%)`;
        row.style.setProperty("--orders-lead", `${Math.max(0, roundTo(g.ladderLeft - right[0] - 4, 2))}px`);
        row.classList.toggle("orders-demo-active", active === i);
        row.classList.toggle("orders-demo-compact", typeof active === "number" && active !== i);
        row.setAttribute("aria-pressed", String(active === i));
      });
    }

    function describe() {
      const example = current();
      const sheet = state.active === null ? null : example.sheets[state.active];
      caption.replaceChildren();
      for (const sentence of CAPTIONS[example.id](example, sheet)) {
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

    function activate(i) {
      if (state.active === i) return;
      state.active = i;
      place(i);
      describe();
    }

    function toggle(i) {
      if (state.pinned && state.active === i) {
        state.pinned = false;
        activate(null);
      } else {
        state.pinned = true;
        activate(i);
      }
    }

    const point = (event) => {
      const box = stage.getBoundingClientRect();
      return [event.clientX - box.left, event.clientY - box.top];
    };
    stage.addEventListener("pointermove", (event) => {
      if (event.pointerType !== "mouse" || state.pinned || event.target.closest(".orders-demo-row")) return;
      const i = sheetAt(point(event));
      if (i !== null) activate(i);
    });
    stage.addEventListener("pointerleave", (event) => {
      if (event.pointerType === "mouse" && !state.pinned) activate(null);
    });
    stack.addEventListener("click", (event) => {
      const i = sheetAt(point(event));
      if (i === null) {
        state.pinned = false;
        activate(null);
      } else {
        toggle(i);
      }
    });
    // when the keyboard leaves the rows, the stack closes again
    ladder.addEventListener("focusout", (event) => {
      if (!state.pinned && !ladder.contains(event.relatedTarget)) activate(null);
    });

    let width = 0;
    new ResizeObserver(() => {
      if (stage.clientWidth === width) return;
      width = stage.clientWidth;
      geometry = measure();
      place(state.active);
    }).observe(stage);

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
