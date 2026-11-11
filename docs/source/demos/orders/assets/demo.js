// The orders demo (demos/orders): one prediction, explained at every order of interaction, as
// layers of paper floating above the example's input, on a card. The top layer holds the Shapley
// values (order 1); every layer below adds the interactions of one more order (FSII, the Faithful
// Shapley Interactions, or k-SII, as the data says), up to order 4. Below them, blank layers stand
// for the orders that follow, up to the last one: the full decomposition, the interactions of all
// orders, which add up to the model itself. Below the card lies the mechanism they explain, the
// model itself, however complex: the explanations reveal more and more of it, order by order. Every
// layer holds an explanation's marks only; the card, the input.
//
// Beside them, upright, a plot of every order's R^2, its faithfulness: how much of the model's
// output on random coalitions (drawn from the distribution of the Shapley values) the explanation
// reproduces beyond what their sizes alone predict. Every order's point lies at the height of its
// layer, a line leading to it from the layer's corner, and the line through them runs down from the
// Shapley values to the full decomposition, which reproduces the model exactly, unless it is
// estimated: what it falls short of 1 is then hatched, the estimation gap, named below it. The input
// and the mechanism are named below the plot.
//
// The examples: the molecule of GraphSHAP-IQ's Figure 1, explained exactly by GraphSHAP-IQ for a
// graph convolutional network; one house and 2,000 houses of California housing, explained exactly
// by shapiq's TreeExplainer for an XGBoost model of depth 5 (the 2,000 houses on the map of
// California, a dot each); the same house valued by TabPFN, a black box, explained exactly over all subsets of its
// features with shapiq's TabPFNImputer; and how SigLIP 2, a black box, matches a photo and its
// caption of the FIxLIP demo, explained by ProxySHAP through an XGBoost proxy, without adjustment
// (three photos behind one tab: a picker under the input chooses among them).
// The full decomposition is the interactions of order n, the number of players (or of the trees'
// depth, which already are); the orders above the highest one with an interaction (the trees'
// depth, the network's receptive field, the proxy's depth) all equal it. The explanations are
// precomputed (../compute) into data.js.
//
// The layers lie on the card at rest (the top one showing, on the input), the mechanism hidden under
// it; once explained and in view (or after a tap), they unfold, for good, and the layer pointed at
// (hover, keyboard focus on its row, a tap that pins it, or the slider left of the layers) stays
// where it floats and casts its light onto the card, which wears its marks, in green (the
// mechanism's rises to it, a box of orange light); its name darkens, and the slider's handle and the
// card of the explanation above take its color, and so does the layer itself: its edge, a glow,
// its paper. The stack stays as it is, the others' marks fading. A ring marks the order pointed at on
// the plot, and pointing along the plot brings out the layer of an order, or the blank layer of an
// order between, which peeks out, showing that order's explanation. The
// slider is an arrow down the side of the layers, a circle at every one, through the explanations:
// they grow more detailed, from order to order; below the input, the mechanism's
// orange circle on its own arrow up to the input. Until a layer is first pointed at, an invitation
// lies over the top one: tap a layer (with a mouse, hover one) to explain. Where the examples do not fit in a row, a stepper goes
// from one to the next, and so does a swipe across the layers.
//
// Above them, the plan shows the problem, as machine learning draws it: the input x wired through
// the model f, a chip, to its prediction f(x). Out of the chip falls the light of the explanation:
// the example seen from above, flat, its input with the explanation pointed at on it (at rest, the
// Shapley values), and in a few words what that explanation is, with a bar by its R^2, how well it
// fits, and what it was explained with (the input alone, the card pointed at; the mechanism's, in
// its orange), and its strongest player or group. The card's face can be inspected: the pointer
// over a player on it (an atom, a feature, a house, a patch or a word; on a touch screen, a tap)
// brings out the strongest groups the player is in, and a tip says its own value and theirs; over
// an interaction, the shape of a group between its players, the tip says what its players add
// alone and what they add together, and, where the data has a note on it, the note. The
// chip clears as far as the explanation fits: the black box opens, all the way
// where it is exact. The demo starts with the problem alone (unless data-reveal="open"): a click on
// the chip explains it, and the explanation is built under it from the top down. The problem stays
// as the examples change: what its parts show flips, as on a board of departures.
//
// The layers are drawn in an axonometric view: a square face, turned in its plane and squashed
// vertically, so that every corner of every layer is known here exactly. The script uses them to
// find the layer under the pointer and to lead every row of the ladder to its layer.
//
// A post that explains the figure builds it up step by step, a demo for every step (data-step): the
// problem alone ("problem"); the input, and below it the mechanism ("mechanism"); the Shapley values
// above them, with the plot of their fit ("shapley"); the explanations of the orders up to 4
// ("interactions"); and all of it ("full"). A step points at what it adds, until a reader points
// elsewhere, and again once they let go: the mechanism, the Shapley values, or the pairs (order 2);
// the last step, as the demo without data-step (all of it, as the docs show it), at nothing. What a
// demo points at at rest can be chosen too (data-select): "none", "mechanism", "full" (the full
// decomposition), or an order ("1", "2", ...). The demos of a page show the same example:
// choosing one in any of them chooses it in all. The page hears every choice, and can make one, as
// the event "orders-demo-example" on the document, its detail { example: <id> }.
//
// On a wide screen, the full stack (all of it, or the last step) would not fit a screen below the
// card: the card of the explanation stands left of the layers then, one light out of the chip falling
// onto both, the card and the stack, both explanations, and fading into the page; the layers are as large as the window's
// height lets them be, and the card as large as they are, growing and shrinking with them
// (data-plan="above" keeps the card above). The problem stays on top, and the steps before keep
// their card above the layers: they fit.
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
  // the air beside the layers, in units of their width: the stage's room holds them, and that much
  // more
  const ROOMY = 0.22;
  // (the card beside the layers, little: the room is narrow, and nothing slides out any more)
  const ROOMY_BESIDE = 0.05;
  // beside the layers, the card as large as they are: its face this share of their width, and its
  // column as wide, or, where they are small, as wide as its words need at the least (rem)
  const FACE_BESIDE = 0.8;
  const WORDS_BESIDE = 14;
  // pixels: below the full stack beside its card, the room left in the window; and with the card
  // above the stack, the room the stack leaves of the window's height
  const WINDOW_MARGIN = 24;
  const WINDOW_ROOM = 96;
  // the window's height, without what a phone's toolbars take as they come and go (they would
  // resize the stack as the page scrolls); on a page without a doctype (quirks mode), the root's
  // height is the whole page's, and the window's own is what there is
  const viewportHeight = () =>
    (document.compatMode === "CSS1Compat" && document.documentElement.clientHeight) || window.innerHeight;
  // milliseconds: how long a layer takes to move to its place (demo.css: --orders-move)
  const MOVE = 640;
  // beside the stack, the edges of the light out of the chip: their angle from the vertical
  const SPREAD = (58 * Math.PI) / 180;
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
  const darkScheme = window.matchMedia("(prefers-color-scheme: dark)");

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
    if (ratio < FAINTEST) return null;
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
    return add(parent, "path", { d, fill: solid(value), "fill-opacity": roundTo(ratio, 3) });
  };

  // the groups of a sheet; a group's center, where the data gives one, is placed by place (and
  // every group's shape noted as its mark, where a face is inspected: marks)
  const drawGroups = (parent, sheet, anchor, largest, widest, place, marks) => {
    const layer = add(parent, "g", { class: "orders-demo-groups" });
    // the weakest first, so that the strongest lie on top
    const groups = [...sheet.groups].sort((a, b) => Math.abs(a[1]) - Math.abs(b[1]));
    for (const [players, value, center] of groups) {
      const node = drawGroup(layer, players.map(anchor), value, largest, widest, center && place(center));
      if (node) marks?.push({ node, key: players.join(","), ratio: Math.abs(value) / largest });
    }
  };

  // -- inspecting a face: on the card, a reader points at a player (an atom, a feature, a patch or
  // a word), at a group (its shape between its players, its circle on a house's tree), or at a
  // house, and the face brings out what the explanation says of it, which the card's tip says in
  // numbers, and, where the data has a note on it, in a sentence. A face that can be inspected
  // gives its targets, each a box around its middle in the face's square (x, y, rx, ry; round for
  // a circle), and, for a target, what to bring out and what to say (inspectorOf) -----------------

  // the least share of the largest value on a sheet at which a group's shape can be pointed at:
  // the haze of the weakest is not
  const POINTABLE = 0.08;

  // the strongest groups of a sheet that hold a player, the strongest first (as the data has them)
  const groupsWith = (sheet, player, most = 4) => sheet.groups.filter(([players]) => players.includes(player)).slice(0, most);

  // the marks of a face's groups, each with its group's key ("3,7") and how strong it is drawn: an
  // inspection brings out those of its keys, as strong as reads at the least, and the others
  // step back (demo.css)
  const bringOut = (svg, marks, keys) => {
    svg.classList.toggle("orders-demo-inspecting", Boolean(keys));
    for (const { node, key, ratio, stroke, plain } of marks) {
      const on = Boolean(keys) && keys.has(key);
      node.classList.toggle("orders-demo-on", on);
      if (!plain) node.style[stroke ? "strokeOpacity" : "fillOpacity"] = on ? roundTo(Math.max(0.62, ratio), 3) : "";
    }
  };

  // a face's inspector: its targets, the one that stands for a group of players (find), and for a
  // target, a ring around it, its marks brought out, and what the explanation says of it
  // (describe: the keys of the marks to bring out, a title, and rows of a name and a value); reset
  // puts back what an inspection changed. The shapes of its groups can be pointed at too
  // (groupAt): the strongest under the pointer, as the strongest lie on top; where the players
  // cover the face (a photo's patches), a group's shape under the pointer comes first (over)
  const inspectorOf = (svg, marks, targets, describe, find = () => null, reset = () => {}, over = false) => {
    const ring = add(svg, "rect", { class: "orders-demo-inspect-ring", visibility: "hidden" });
    const shapes = new WeakMap(marks.map((mark) => [mark.node, mark]));
    const groups = new Map(); // a group's target, the same one while the pointer stays on its shape
    return {
      targets,
      find,
      over,
      groupAt(x, y) {
        for (const node of document.elementsFromPoint(x, y)) {
          if (node === svg) break;
          const mark = shapes.get(node);
          if (!mark || mark.plain || mark.ratio < POINTABLE) continue;
          if (!groups.has(mark.key)) groups.set(mark.key, find(mark.key.split(",").map(Number)));
          return groups.get(mark.key);
        }
        return null;
      },
      show(target) {
        const said = describe(target);
        bringOut(svg, marks, said?.keys ?? null);
        const ringed = target.rx > 0 && !target.bare;
        ring.setAttribute("visibility", ringed ? "visible" : "hidden");
        if (ringed) {
          const [w, h] = [target.rx + 0.8, target.ry + 0.8];
          const box = { x: target.x - w, y: target.y - h, width: 2 * w, height: 2 * h, rx: target.round ? w : 1.3 };
          for (const [name, value] of Object.entries(box)) ring.setAttribute(name, roundTo(value, 2));
        }
        return said;
      },
      clear() {
        bringOut(svg, marks, null);
        ring.setAttribute("visibility", "hidden");
        reset();
      },
    };
  };

  // what an explanation says of a player: its own value (of order 1, its Shapley value), and the
  // strongest groups it is in; or of a group, an interaction: its players' own values, what they
  // add together, on top of their own, and for a pair, the sum of the three, what the pair adds
  // in all. With it, the data's note on the player or the group, where it has one (most have
  // none: not every value has a story). title names a player, in two parts; nameOf, in a word
  const sayOf = (example, sheet, target, title, nameOf) => {
    const words = PLAYERS[example.kind];
    if (target.group) {
      const players = target.group;
      const key = players.join(",");
      const value = sheet.groups.find(([others]) => others.join(",") === key)?.[1] ?? 0;
      const own = players.map((player) => sheet.first[player]);
      return {
        keys: new Set([key]),
        title: [groupName(example, players, nameOf, 34), players.length === 2 ? "a pair" : `a group of ${players.length}`],
        // (its own note, or the note of every group as large: "many")
        story: example.notes?.[key] ?? (players.length >= (example.notes?.many?.[0] ?? Infinity) ? example.notes.many[1] : undefined),
        rows: [
          ...(players.length <= 4 ? players.map((player, k) => [`${nameOf(player)} alone`, own[k]]) : []),
          ["together, on top", value],
          ...(players.length === 2 ? [["in sum", own[0] + own[1] + value, undefined, true]] : []),
        ],
        when: `when ${players.length === 2 ? "both" : `all ${players.length}`} are there`,
      };
    }
    const groups = groupsWith(sheet, target.player);
    const others = (players) => players.filter((player) => player !== target.player);
    return {
      keys: new Set(groups.map(([players]) => players.join(","))),
      title: title(target.player),
      story: example.notes?.[String(target.player)],
      rows: [
        [sheet.order === 1 ? "Shapley value" : "alone", sheet.first[target.player]],
        ...groups.map(([players, value]) => [
          others(players).length <= 3 ? `with ${others(players).map(nameOf).join(", ")}` : `with ${others(players).length} others`,
          value,
        ]),
      ],
    };
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

  // a face's layers: "all" for the plan's face, or, for the floating layers, the "input" (what is
  // explained, on the card) or the "marks" (the explanation, on a layer above it)
  const shows = (layer, part) => layer === "all" || layer === part;

  // the elements of a molecule's atoms, in words
  const ELEMENTS = { C: "carbon", N: "nitrogen", O: "oxygen", H: "hydrogen", S: "sulfur", P: "phosphorus", F: "fluorine", Cl: "chlorine", Br: "bromine", I: "iodine" };

  // the molecule: its bonds and atoms over the explanation, as in GraphSHAP-IQ's figures (the
  // card's face can be inspected, atom by atom: it returns its inspector)
  function moleculeFace(svg, example, sheet, layer = "all") {
    const place = fit(example.coords, [9, 5, 93, 87]);
    const position = example.coords.map(place);
    const extent = extentOf(position);
    const [radius, widest] = [(ATOM / 2) * extent, WIDEST * extent];
    const largest = largestValue(sheet);
    const marks = [];
    if (shows(layer, "marks")) {
      // the groups' centers where shapiq's SI graph plot puts them (../compute/pack.py)
      drawGroups(svg, sheet, (atom) => position[atom], largest, widest, place, marks);
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
    if (!shows(layer, "input")) return null;
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
    if (layer !== "all") return null;
    // an atom, named by its element and the atoms it is bonded to (so a nitro group's N reads as
    // one: bonded to C, O, O)
    const bonded = example.atoms.map((_, atom) =>
      example.bonds.filter((bond) => bond.includes(atom)).map(([a, b]) => example.atoms[a === atom ? b : a]),
    );
    const title = (atom) => [example.atoms[atom], `${ELEMENTS[example.atoms[atom]] ?? "atom"}, bonded to ${bonded[atom].join(", ")}`];
    // (an atom's box hardly wider than its disc: between two bonded atoms, their pair's shape shows)
    const targets = position.map(([x, y], atom) => ({ x, y, rx: radius * 1.1, ry: radius * 1.1, round: true, player: atom }));
    // (a group: where its atoms' middle is, with no ring: its shape is brought out)
    const find = (players) =>
      players.length === 1
        ? targets[players[0]]
        : { x: players.reduce((sum, atom) => sum + position[atom][0], 0) / players.length, y: Math.min(...players.map((atom) => position[atom][1])), rx: 0, ry: radius, group: players };
    return inspectorOf(svg, marks, targets, (target) => sayOf(example, sheet, target, title, (atom) => example.atoms[atom]), find);
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
    const marks = [];
    if (shows(layer, "marks")) drawGroups(svg, sheet, anchor, largest, WIDEST * extentOf(players), undefined, marks);

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
        // (a long caption's words smaller, as their rows are)
        const word = add(words, "text", { x: tokens.x + tokens.width / 2, y: roundTo(y + height / 2 + 0.1, 2) }, token);
        if (row < 6) word.setAttribute("font-size", roundTo(row * 0.58, 2));
      }
    });
    if (layer !== "all") return null;
    // (the card's face can be inspected, patch by patch and word by word: its inspector) a patch,
    // named by its row and column in the photo; a word, as the caption has it
    const where = (patch) => [Math.floor(patch / grid) + 1, (patch % grid) + 1];
    const nameOf = (player) => (player < patches ? `patch ${where(player).join("·")}` : `“${example.tokens[player - patches]}”`);
    const title = (player) =>
      player < patches
        ? [`Patch ${where(player).join("·")}`, `row ${where(player)[0]}, column ${where(player)[1]} of the photo`]
        : [`“${example.tokens[player - patches]}”`, `word ${player - patches + 1} of the caption`];
    const targets = players.map(([x, y], player) =>
      player < patches
        ? { x, y, rx: cell / 2, ry: cell / 2, player }
        : { x: tokens.x + tokens.width / 2, y: tokens.top + (player - patches + 0.5) * row, rx: tokens.width / 2, ry: row * 0.38, player },
    );
    const find = (group) =>
      group.length === 1
        ? targets[group[0]]
        : { x: group.reduce((sum, player) => sum + targets[player].x, 0) / group.length, y: Math.min(...group.map((player) => targets[player].y)), rx: 0, ry: cell / 2, group };
    return inspectorOf(svg, marks, targets, (target) => sayOf(example, sheet, target, title, nameOf), find, undefined, true);
  }

  // one house: its explanation as a tree that grows one level per order. The features are on top,
  // every group of k features hangs at level k, below the middle of its features, and joined to
  // each of them; red raises the value of the house, blue lowers it. The full decomposition of a
  // tree ensemble is as deep as its trees. (The card's face can be inspected, feature by feature
  // and group by group: it returns its inspector)
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
    // (the levels are the orders: so the first one says)
    if (marks) add(guides, "text", { class: "orders-demo-tree-order", x: 2.5, y: roundTo(top - 3.6, 2) }, "order");
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
          r: 1 + 4.6 * ratioOf(value) ** 0.8,
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

    // (the groups' marks, for an inspection to bring out: their links, their circles, their values)
    const noted = [];
    const links = add(svg, "g", { class: "orders-demo-tree-links" });
    for (const node of marks ? [...nodes].sort((a, b) => Math.abs(a.value) - Math.abs(b.value)) : []) {
      const ratio = ratioOf(node.value);
      for (const player of node.players) {
        const [x1, y1, x2] = [node.x, node.y, slot.get(player)];
        const bend = (y1 - top) * 0.55;
        const link = add(links, "path", {
          d: `M${roundTo(x1, 2)} ${roundTo(y1, 2)}C${roundTo(x1, 2)} ${roundTo(y1 - bend, 2)} ${roundTo(x2, 2)} ${roundTo(top + bend, 2)} ${roundTo(x2, 2)} ${top}`,
          stroke: solid(node.value),
          "stroke-width": roundTo(0.2 + 1.3 * ratio, 2),
          "stroke-opacity": roundTo(0.18 + 0.6 * ratio, 3),
        });
        noted.push({ node: link, key: node.players.join(","), ratio: 0.18 + 0.6 * ratio, stroke: true });
      }
    }
    const groups = add(svg, "g", { class: "orders-demo-tree-groups" });
    // a group's value beside its circle, the strongest groups' first: right of it, or left of it,
    // where it touches no other circle and no other value; else none (the card's tip says it)
    const taken = nodes.map((node) => [node.x - node.r, node.x + node.r, node.y]);
    const free = (from, to, y) => from > 1 && to < 99 && !taken.some(([a, b, level]) => level === y && from < b + 0.6 && to > a - 0.6);
    const labels = new Map();
    for (const node of marks ? [...nodes].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)) : []) {
      if (ratioOf(node.value) < 0.25) continue;
      const text = dollars(node.value);
      const width = text.length * 1.5; // (of its font, demo.css)
      const sides = [
        [node.x + node.r + 1, node.x + node.r + 1 + width, "start"],
        [node.x - node.r - 1 - width, node.x - node.r - 1, "end"],
      ];
      const side = sides.find(([from, to]) => free(from, to, node.y));
      if (!side) continue;
      taken.push([side[0], side[1], node.y]);
      labels.set(node, { x: roundTo(side[2] === "start" ? side[0] : side[1], 2), anchor: side[2], text });
    }
    for (const node of marks ? nodes : []) {
      const ratio = ratioOf(node.value);
      const key = node.players.join(",");
      const circle = add(groups, "circle", {
        cx: roundTo(node.x, 2),
        cy: roundTo(node.y, 2),
        r: roundTo(node.r, 2),
        fill: solid(node.value),
        "fill-opacity": roundTo(0.4 + 0.6 * ratio, 3),
      });
      noted.push({ node: circle, key, ratio: 0.4 + 0.6 * ratio });
      const label = labels.get(node);
      if (label) {
        const text = add(groups, "text", { x: label.x, y: roundTo(node.y, 2), "text-anchor": label.anchor }, label.text);
        noted.push({ node: text, key, plain: true });
      }
    }

    // the features: their main effects as halos, their names and values above them, every other
    // one higher, so that they do not touch
    const halos = add(svg, "g", { class: "orders-demo-halos" });
    const features = add(svg, "g", { class: "orders-demo-atoms" });
    const names = add(svg, "g", { class: "orders-demo-features" });
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
      add(names, "text", { x, y, "text-anchor": "middle", class: "orders-demo-feature-name" }, example.features[feature]);
      add(names, "text", { x, y: y + 3.3, "text-anchor": "middle", class: "orders-demo-feature-value" }, example.values[feature]);
    });
    if (layer !== "all") return null;
    const targets = [
      ...example.layout.map((feature) => ({ x: slot.get(feature), y: top, rx: 4, ry: 4, round: true, player: feature })),
      ...nodes.map((node) => ({ x: node.x, y: node.y, rx: Math.max(node.r, 2.2), ry: Math.max(node.r, 2.2), round: true, group: node.players })),
    ];
    // (a group the tree does not draw has no place on it: nothing to point at)
    const find = (players) => targets.find((target) => (target.group ?? [target.player]).join(",") === players.join(",")) ?? null;
    const title = (feature) => [example.features[feature], example.values[feature]];
    return inspectorOf(svg, noted, targets, (target) => sayOf(example, sheet, target, title, (feature) => example.features[feature]), find);
  }

  // many houses: the map of California, a dot where every house is, in the color of what one
  // feature, or one group of features, adds to its value (red) or takes off it (blue); beside it,
  // the explanation's features and groups, the strongest first, by how much each moves a house's
  // value on average, the one the dots show marked. The dots show the strongest feature of the
  // Shapley values, and from order 2 on, the strongest group: what the order has that the
  // Shapley values lack. The houses are many: their dots are painted on a canvas, an image of the
  // face, not drawn as elements. (The card's face can be inspected: a row shows itself on the map,
  // and a house says its own values; it returns its inspector)
  const MAP = {
    left: 6, // the state's west, and its north (the face's units)
    top: 15,
    scale: 8, // of a degree of latitude; of longitude, less by the cosine of the state's middle
    shrink: Math.cos((37.2 * Math.PI) / 180),
    dot: 0.62, // a house's radius
    pixels: 720, // the side of the canvas the dots are painted on
    rows: { top: 8.8, step: 3.9, name: 78.6, bar: 79.8, width: 9.2, value: 97.6 },
  };
  // the houses' numbers, written as text (../compute/pack.py): a character of this alphabet for a
  // whole number from 0 on (a row's value, from -45 to 45 of its sheet's scale), or two for a
  // larger one (a coordinate, in hundredths of a degree)
  const ALPHABET = Array.from({ length: 92 }, (_, k) => String.fromCharCode(35 + k)).filter((letter) => letter !== "\\").join("");
  const LEVELS = (ALPHABET.length - 1) / 2;
  const read = (text, width = 1) =>
    Array.from({ length: text.length / width }, (_, k) => {
      let number = 0;
      for (let place = 0; place < width; place++) number = number * ALPHABET.length + ALPHABET.indexOf(text[k * width + place]);
      return number;
    });
  // an example's houses, read once: where each is (longitude, latitude), and of every sheet, its
  // rows' values (whole numbers from -45 to 45)
  const mapsRead = new WeakMap();
  const readMap = (example) => {
    if (!mapsRead.has(example)) {
      const { west, south, lon, lat } = example.coords;
      const [lons, lats] = [read(lon, 2), read(lat, 2)];
      mapsRead.set(example, {
        coords: lons.map((value, house) => [west + value / 100, south + lats[house] / 100]),
        values: new Map(example.sheets.map((sheet) => [sheet, sheet.rows.map((row) => read(row[1]).map((value) => value - LEVELS))])),
      });
    }
    return mapsRead.get(example);
  };
  // dots painted on a canvas, as an image of the face (its 100 x 100): each a place, a color and
  // how opaque it is
  const painted = (dots) => {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = MAP.pixels;
    const context = canvas.getContext("2d");
    context.scale(MAP.pixels / 100, MAP.pixels / 100);
    for (const [x, y, color, opacity] of dots) {
      context.globalAlpha = opacity;
      context.fillStyle = color;
      context.beginPath();
      context.arc(x, y, MAP.dot, 0, 2 * Math.PI);
      context.fill();
    }
    return canvas.toDataURL();
  };
  // the row the map shows at rest
  const mapRow = (sheet) => (sheet.order === 1 ? 0 : Math.max(0, sheet.rows.findIndex(([players]) => players.length > 1)));
  // (dollars without a sign: a size)
  const sized = (value) => `$${Math.round(Math.abs(value) * 100)}k`;
  function mapFace(svg, example, sheet, layer = "all") {
    const west = Math.min(...example.outline.map(([lon]) => lon));
    const north = Math.max(...example.outline.map(([, lat]) => lat));
    const at = ([lon, lat]) => [roundTo(MAP.left + (lon - west) * MAP.shrink * MAP.scale, 2), roundTo(MAP.top + (north - lat) * MAP.scale, 2)];
    const { coords, values } = readMap(example);
    const spots = coords.map(at);
    const picture = (dots) => add(svg, "image", { class: "orders-demo-map-dots", x: 0, y: 0, width: 100, height: 100, href: dots });
    if (shows(layer, "input")) {
      add(svg, "path", { class: "orders-demo-map-land", d: `M${example.outline.map((point) => at(point).join(" ")).join("L")}Z` });
      // the houses, gray (a gray that shows on a light page and on a dark one)
      picture(painted(spots.map(([x, y]) => [x, y, "rgb(122, 130, 152)", 0.42])));
      // the cities, named off the coast
      const places = add(svg, "g", { class: "orders-demo-map-places" });
      for (const [name, lon, lat] of example.places) {
        const [x, y] = at([lon, lat]);
        add(places, "path", { d: `M${x - 0.9} ${y}H${x + 0.9}M${x} ${y - 0.9}V${y + 0.9}` });
        add(places, "text", { x: roundTo(x - 2, 2), y: roundTo(y + 1.6, 2) }, name);
      }
    }
    if (!shows(layer, "marks")) return null;

    const nameOf = (players) => players.map((feature) => example.features[feature]).join(" × ");
    const valuesOf = values.get(sheet);
    const valueOf = (row, house) => (valuesOf[row][house] / LEVELS) * sheet.scale;
    // the dots in the colors of a row, the strongest on top (a row's picture is painted once)
    const pictures = new Map();
    const pictureOf = (r) => {
      if (!pictures.has(r)) {
        const row = valuesOf[r];
        const order = [...row.keys()].sort((a, b) => Math.abs(row[a]) - Math.abs(row[b]));
        pictures.set(r, painted(order.map((house) => [...spots[house], solid(row[house]), (Math.abs(row[house]) / LEVELS) ** 0.85])));
      }
      return pictures.get(r);
    };
    const dots = picture("");
    // what the dots show, named over the map, with how many houses it raises and lowers
    const caption = add(svg, "g", { class: "orders-demo-map-caption" });
    const shown = add(caption, "text", { class: "orders-demo-map-shown", x: MAP.left, y: 4.6 });
    const counts = add(caption, "text", { x: MAP.left, y: 8.3 });
    // the rows: a name, a bar by the mean of its absolute values, and that mean
    const list = add(svg, "g", { class: "orders-demo-map-rows" });
    add(list, "text", { class: "orders-demo-map-head", x: MAP.rows.value, y: 4.6 }, "moves a house's value by, on average");
    const most = Math.max(...sheet.rows.map((row) => row[2]));
    const rows = sheet.rows.map(([players, , mean], r) => {
      const y = roundTo(MAP.rows.top + r * MAP.rows.step, 2);
      const row = add(list, "g", { class: "orders-demo-map-row" });
      add(row, "text", { class: "orders-demo-map-name", x: MAP.rows.name, y }, nameOf(players));
      add(row, "rect", { x: MAP.rows.bar, y: roundTo(y - 1.1, 2), width: roundTo(Math.max(0.5, (mean / most) * MAP.rows.width), 2), height: 2.2, rx: 0.6 });
      add(row, "text", { class: "orders-demo-map-mean", x: MAP.rows.value, y }, sized(mean));
      return row;
    });
    // the colors' scale, under the rows, east of the state: from what lowers a house's value the
    // most (the sheet's scale, or more) to what raises it the most
    const scale = add(svg, "g", { class: "orders-demo-map-scale" });
    const [steps, cell, middle] = [5, 2.2, 84.6];
    for (let k = -steps; k <= steps; k++) {
      add(scale, "rect", { x: roundTo(middle + (k - 0.5) * cell, 2), y: 64.6, width: cell + 0.05, height: 2.2, fill: solid(k), "fill-opacity": roundTo((Math.abs(k) / steps) ** 0.85, 3) });
    }
    add(scale, "text", { x: middle, y: 62.3, "text-anchor": "middle" }, "added to a house's value");
    add(scale, "text", { x: roundTo(middle - (steps + 0.5) * cell, 2), y: 69.4, "text-anchor": "start" }, `−${sized(sheet.scale)}`);
    add(scale, "text", { x: roundTo(middle + (steps + 0.5) * cell, 2), y: 69.4, "text-anchor": "end" }, `+${sized(sheet.scale)}`);
    let showing = -1;
    const paint = (r) => {
      if (r === showing) return;
      showing = r;
      dots.setAttribute("href", pictureOf(r));
      rows.forEach((row, k) => row.classList.toggle("orders-demo-map-row-on", k === r));
      shown.textContent = nameOf(sheet.rows[r][0]);
      const [up, down] = [valuesOf[r].filter((value) => value > 0).length, valuesOf[r].filter((value) => value < 0).length];
      counts.textContent = `raises ${up.toLocaleString("en")} houses' values, lowers ${down.toLocaleString("en")}`;
    };
    const rest = mapRow(sheet);
    paint(rest);
    if (layer !== "all") return null;

    const targets = [
      ...sheet.rows.map((_, r) => {
        const [from, to] = [MAP.rows.name - 24, MAP.rows.value + 1];
        return { x: (from + to) / 2, y: MAP.rows.top + r * MAP.rows.step, rx: (to - from) / 2, ry: MAP.rows.step / 2, bare: true, row: r };
      }),
      ...spots.map(([x, y], house) => ({ x, y, rx: 1.1, ry: 1.1, round: true, house })),
    ];
    const describe = (target) => {
      // a row shows itself on the map, and the map's caption says the rest
      if (target.row !== undefined) {
        paint(target.row);
        return null;
      }
      // a house: where it is, what the model predicts for it, and the strongest of its values
      // (in steps of a 45th of the scale, as the data has them)
      paint(rest);
      const [lon, lat] = coords[target.house];
      const strongest = valuesOf
        .map((row, r) => [sheet.rows[r][0], row[target.house], r])
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 4);
      return {
        title: ["A house", `at ${lat.toFixed(2)}°N, ${(-lon).toFixed(2)}°W`],
        rows: [
          ["predicted value", null, sized(example.predictions[target.house])],
          // (a value at the end of the scale is that large at the least)
          ...strongest.map(([players, value, r]) => [
            nameOf(players),
            valueOf(r, target.house),
            Math.abs(value) === LEVELS ? `${dollars(valueOf(r, target.house))} or more` : undefined,
          ]),
        ],
      };
    };
    const find = (players) => targets.find((target) => target.row !== undefined && sheet.rows[target.row][0].join(",") === players.join(",")) ?? null;
    return inspectorOf(svg, [], targets, describe, find, () => paint(rest));
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
    graph: { one: "atom", players: "atoms", point: "an atom", group: "a link" },
    tree: { one: "feature", players: "features", point: "a feature", group: "a group" },
    map: { one: "feature", players: "features", point: "a house or a row" },
    "image-text": { one: "patch or word", players: "patches and words", point: "a patch or word", group: "a link" },
  };
  // a group's name, its players' joined, or, where that is long, how many they are (long: more
  // than the card's line holds; the tip holds more)
  const groupName = (example, players, nameOf, most = 18) => {
    const name = players.map(nameOf).join(" × ");
    return name.length <= most ? name : `${players.length} ${PLAYERS[example.kind].players}`;
  };
  // a player's name in a word, by the example's kind: an atom's element, a feature's name, a word
  // of the caption, or a patch
  const playerName = (example, player) =>
    example.kind === "graph"
      ? example.atoms[player]
      : example.kind === "image-text"
        ? player < example.grid ** 2
          ? "patch"
          : `“${example.tokens[player - example.grid ** 2]}”`
        : example.features[player];
  // a value as its example's kind writes it, with its sign: a logit to two decimals, a house's
  // value in thousands of dollars, a similarity to three decimals; and what the values are of
  // (a value that rounds to nothing, without a sign)
  const signed = (value, digits) => (Number(Math.abs(value).toFixed(digits)) === 0 ? (0).toFixed(digits) : `${value < 0 ? "−" : "+"}${Math.abs(value).toFixed(digits)}`);
  const VALUE = { graph: (value) => signed(value, 2), tree: (value) => dollars(value), map: (value) => dollars(value), "image-text": (value) => signed(value, 3) };
  const OF = { graph: "the logit of the prediction", tree: "the house's predicted value", map: "the house's predicted value", "image-text": "the similarity" };
  // the strongest of an explanation, for the card to say and to bring out: of the Shapley values,
  // a player; of an order with groups, its strongest group; of many houses, the row their map
  // shows (its players, and the mean of its absolute values)
  const strongestOf = (example, sheet) => {
    if (example.kind === "map") {
      const [players, , mean] = sheet.rows[mapRow(sheet)];
      return { players, text: `±${sized(mean)}`, sign: 0 };
    }
    const [players, value] = sheet.groups.length
      ? sheet.groups[0]
      : sheet.first.map((own, player) => [[player], own]).reduce((best, next) => (Math.abs(next[1]) > Math.abs(best[1]) ? next : best));
    return { players, text: VALUE[example.kind](value), sign: Math.sign(value) };
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

  // the steps of the figure, as a post builds it up (without one, a demo shows all of it)
  const PARTS = ["problem", "mechanism", "shapley", "interactions", "full"];
  // the example chosen, on the document: every demo of the page hears it, and follows
  const EXAMPLE_EVENT = "orders-demo-example";
  // the explanations' index, as the sheets' tags name it: FSII (2-FSII), or k-SII (2-SII)
  let INDEX = "SII";
  let plots = 0; // the demos' plots on the page, for ids of their own

  // the examples a demo shows, one per tab: those data-examples names (their ids, apart by commas
  // or spaces), in its order; without it, or where it names none the data knows, all of them. An
  // example that is another one of a tab (its `of` names the tab's first: another photo and
  // caption) has no tab of its own
  function examplesOf(root, all) {
    const tabs = all.examples.filter((example) => !example.of);
    const ids = (root.dataset.examples ?? "").split(/[\s,]+/).filter(Boolean);
    const known = ids.filter((id) => tabs.some((example) => example.id === id));
    if (known.length < ids.length) {
      const ours = tabs.map((example) => example.id).join(", ");
      console.warn(`The explanation of every order has no example ${ids.filter((id) => !known.includes(id)).join(", ")} (it has ${ours}).`);
    }
    return known.length ? [...new Set(known)].map((id) => tabs.find((example) => example.id === id)) : tabs;
  }

  function start(root, all, life) {
    // (the data, as far as this demo shows it: all it does, it does with its own examples: one per
    // tab, and with each, the other examples of its tab, which the tab's picker shows)
    const tabs = examplesOf(root, all);
    const family = (tab) => [tab, ...all.examples.filter((example) => example.of === tab.id)];
    const data = { ...all, examples: tabs, every: tabs.flatMap(family) };
    const tabOf = (example) => (example.of ? tabs.find((tab) => tab.id === example.of) : example);
    const shownOf = new Map(); // of a tab with several examples, the one it showed last
    // (one example alone: no tabs, no stepper, no swipe to another, demo.css)
    root.classList.toggle("orders-demo-single", data.examples.length === 1);
    INDEX = data.index === "k-SII" || !data.index ? "SII" : data.index;
    // the demo's observers, for as long as it lives (its life ends where it cannot start)
    const watch = (observer) => {
      life?.addEventListener("abort", () => observer.disconnect(), { once: true });
      return observer;
    };
    // what is pointed at: a layer (its index), or the blank layer of an order between
    const state = { example: data.examples[0].id, active: null, blank: null, pinned: false };
    // the step of the figure it shows (data-step), or all of it ("all", as the docs show it, or
    // "full", as the last step of a post)
    const part = PARTS.includes(root.dataset.step) ? root.dataset.step : "all";
    const whole = part === "all" || part === "full";
    // of an example, the layers a step shows: how many of its sheets, from the top (none, the
    // Shapley values, all but the full decomposition, or all), and the blank layers between them
    const shownSheets = (example) =>
      ({ problem: 0, mechanism: 0, shapley: 1, interactions: example.sheets.filter((sheet) => !sheet.full).length })[part] ??
      example.sheets.length;
    const shownBlanks = (example) => (whole ? between(example) : []);
    // what a demo points at, at rest (data-select): nothing ("none"), the mechanism ("mechanism"), the
    // full decomposition ("full"), or the layer of an order ("1", "2", ...), where the step shows it;
    // without it, what the step adds (the mechanism, the Shapley values, the pairs), or nothing (the
    // problem alone, and all of it). As the plan picks it
    const select = root.dataset.select ?? { mechanism: "mechanism", shapley: "1", interactions: "2" }[part] ?? "none";
    const restPick = (example) => {
      // (unexplained, nothing: the chip stays dark, the problem alone)
      if (!revealed) return null;
      if (select === "mechanism") return { mechanism: true };
      const index =
        select === "full"
          ? example.sheets.length - 1
          : example.sheets.findIndex((sheet) => String(sheet.order) === select);
      return index >= 0 && index < shownSheets(example) ? { index } : null;
    };
    // and as the layers do: its index (the sheets shown are the first ones, and the mechanism after
    // them)
    const restActive = () => {
      const pick = restPick(current());
      return !pick ? null : pick.mechanism ? sheetItems().findIndex((item) => item.mechanism) : pick.index;
    };
    let geometry = null;
    const plotId = ++plots;
    // the floating layers: the input on a card, the explanations on layers above it, and below the
    // card, apart from the full decomposition, the mechanism they explain. They lie on the card at
    // rest, the mechanism hidden under it, and unfold, float off it, once explained (or after a tap
    // on the card), or while a layer is pointed at, which casts its marks onto the card
    // the demo starts with the problem alone (unless data-reveal="open"): a click on the model's chip
    // explains it, and the explanation is built under it, from the top down; once explained, every
    // example is. A step shows its part at once (the problem's, the problem alone, for good)
    let revealed = part !== "problem" && (root.dataset.reveal === "open" || part !== "all");
    // the floating layers unfold for good, once: after the model was clicked to explain it, once they
    // have come in and are in view (or tapped at rest); before, only while one is pointed at
    let unfoldedForGood = false;
    const unfolded = () => unfoldedForGood || state.active !== null || state.blank !== null; // the layers, apart

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
      option.addEventListener("click", () => choose(shownOf.get(example.id) ?? example));
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
    // (the example chosen is said by the demo's status line, below, in every layout)
    const shown = el("span", "orders-demo-step-shown");
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
    // (no live region: its parts flip through random letters to the next example's; a line out of
    // view says the example chosen instead, once: the status, below)
    const status = el("span", "orders-demo-status");
    status.setAttribute("aria-live", "polite");
    const cast = document.createElementNS(SVG, "svg"); // the light out of the chip
    cast.setAttribute("class", "orders-demo-cast");
    cast.setAttribute("aria-hidden", "true");
    const castLight = add(add(cast, "defs"), "linearGradient", { id: `orders-cast-${plotId}`, x1: 0, y1: 0, x2: 0, y2: 1 });
    add(castLight, "stop", { offset: 0, class: "orders-demo-cast-top" });
    add(castLight, "stop", { offset: 1, class: "orders-demo-cast-bottom" });
    // beside the layers, the light is round instead: a cone out of the chip's bottom, as wide to
    // either side, its edges dashed, filled with a glow from the chip's middle that fades out a short
    // way from it, well before the card and the page's edges; and its edges fade with it (a mask, as
    // round)
    const castGlowLight = add(cast.querySelector("defs"), "radialGradient", { id: `orders-cast-glow-${plotId}`, gradientUnits: "userSpaceOnUse" });
    for (const [offset, opacity] of [[0, 1], [0.25, 0.8], [0.5, 0.45], [0.75, 0.15], [1, 0]])
      add(castGlowLight, "stop", { offset, class: "orders-demo-cast-glow", "stop-opacity": opacity });
    const castFade = add(cast.querySelector("defs"), "radialGradient", { id: `orders-cast-fade-${plotId}`, gradientUnits: "userSpaceOnUse" });
    for (const [offset, white] of [[0, 1], [0.35, 1], [0.7, 0.35], [1, 0]])
      add(castFade, "stop", { offset, "stop-color": "#fff", "stop-opacity": white });
    const castMask = add(cast.querySelector("defs"), "mask", { id: `orders-cast-mask-${plotId}`, maskUnits: "userSpaceOnUse" });
    const castMaskArea = add(castMask, "rect", { x: 0, y: 0, fill: `url(#orders-cast-fade-${plotId})` });
    const castGroup = add(cast, "g");
    const castHull = add(castGroup, "polygon", { fill: `url(#orders-cast-${plotId})` });
    const castEdges = add(castGroup, "path", { class: "orders-demo-cast-edges" });
    const planView = el("div", "orders-demo-plan-view");
    planView.setAttribute("aria-hidden", "true");
    // the face's name, in its bottom left corner, as the layer wears it (beside the stack, where the
    // face is as large as a layer)
    const planTag = document.createElementNS(SVG, "svg");
    planTag.setAttribute("viewBox", "0 0 100 100");
    planTag.setAttribute("class", "orders-demo-plan-tag");
    const planTagText = add(planTag, "text", { class: "orders-demo-tag", x: 6, y: 95.5 });
    // after a short name, until the face is first inspected, an invitation to (as the layers' own:
    // "tap" on a touch screen, "hover" with a mouse)
    const planTagHint = document.createElementNS(SVG, "tspan");
    planTagHint.setAttribute("class", "orders-demo-tag-hint");
    planTagHint.setAttribute("dx", "2.4");
    const planTagWords = [["orders-demo-hint-touch", "tap"], ["orders-demo-hint-hover", "hover"]].map(([className, word]) => [add(planTagHint, "tspan", { class: className }), word]);
    // (what can be pointed at: a player, and where the face shows groups, their shapes; where the
    // name leaves no room for it, no invitation)
    const tagFace = (name, grouped) => {
      planTagText.textContent = name;
      const words = PLAYERS[current().kind];
      const what = grouped && words.group ? `${words.point} or ${words.group}` : words.point;
      for (const [node, word] of planTagWords) node.textContent = `· ${word} ${what}`;
      if (name.length * 2.55 + (what.length + 8) * 1.5 <= 88) planTagText.append(planTagHint);
    };
    // the card's tip: what the explanation shown says of what is pointed at on its face
    const cardTip = el("div", "orders-demo-tip");
    cardTip.setAttribute("aria-hidden", "true");
    const planText = el("figcaption", "orders-demo-plan-text");
    const why = el("div", "orders-demo-why");
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
    // under the input, where its tab has several examples (the photos of "Describing images"), a
    // picker: a small picture of each, the one shown marked
    const picker = el("div", "orders-demo-picks");
    picker.setAttribute("role", "group");
    picker.setAttribute("aria-label", "Input");
    picker.hidden = true;
    let picks = []; // its buttons, with their examples
    let pickerOf = null; // the tab it is of
    function showPicker(example) {
      const tab = tabOf(example);
      const examples = family(tab);
      picker.hidden = examples.length < 2;
      if (pickerOf !== tab.id) {
        pickerOf = tab.id;
        picks = examples.map((other, k) => {
          const pick = el("button", "orders-demo-pick");
          pick.type = "button";
          const name = other.photo?.title ?? `${tab.label} ${k + 1}`;
          pick.setAttribute("aria-label", name);
          pick.title = name;
          if (other.image) pick.style.backgroundImage = `url("${root.dataset.root}${other.image}")`;
          else pick.textContent = String(k + 1);
          pick.addEventListener("click", () => choose(other));
          return [other.id, pick];
        });
        picker.replaceChildren(...picks.map(([, pick]) => pick));
      }
      for (const [id, pick] of picks) pick.setAttribute("aria-pressed", String(id === example.id));
    }
    const [chipIcon, chipName, chipNote, chipShows] = [el("span", "orders-demo-wire-icon-slot"), el("strong"), el("span", "orders-demo-chip-note"), shows()];
    chipShows.append(chipIcon, chipName, chipNote);
    const [outValue, outNote, outShows] = [el("strong"), el("span", "orders-demo-wire-note"), shows()];
    outShows.append(outValue, outNote);
    const chip = tagged("orders-demo-wire-chip", "f", chipShows);
    const explainTab = el("span", "orders-demo-chip-explain", "Explain");
    // (a wire: its dashes flow in a box of their own, demo.css)
    const wireLine = () => {
      const line = el("span", "orders-demo-wire-line");
      line.append(el("span", "orders-demo-wire-dashes"));
      return line;
    };
    const into = el("div", "orders-demo-wire-side orders-demo-wire-into");
    into.append(tagged("orders-demo-wire-x", "x", xShows, picker), wireLine());
    const onto = el("div", "orders-demo-wire-side orders-demo-wire-onto");
    onto.append(wireLine(), tagged("orders-demo-wire-out", "f(x)", outShows));
    flow.append(into, chip, onto);
    // unexplained, the chip is a button that explains the example
    chip.addEventListener("click", () => {
      if (!revealed && part === "all") reveal();
    });
    chip.addEventListener("keydown", (event) => {
      if (revealed || part !== "all" || (event.key !== "Enter" && event.key !== " ")) return;
      event.preventDefault();
      reveal();
    });
    // the card's question, a pill on its top edge, in the middle, where the light out of the chip
    // meets it; in the card, the face, and the caption: what the explanation shown is, and how well
    // it fits
    const whyName = el("span", "orders-demo-plan-eyebrow");
    whyName.append("Why ", math("f(x)"), "?");
    const planWords = el("span", "orders-demo-plan-words");
    planText.append(planWords);
    why.append(whyName, planView, planText);
    // its window: swiped aside, it goes out of view, not out of the page
    const planWindow = el("div", "orders-demo-plan-window");
    planWindow.append(plan);

    const stage = el("div", "orders-demo-stage");
    stage.setAttribute("role", "group");
    const stack = el("div", "orders-demo-stack");
    stack.setAttribute("aria-hidden", "true");
    const ladder = el("div", "orders-demo-ladder");
    // beside the stack, in a column of its own, the slider of the orders: an arrow down (every order
    // below explains in more detail, and fits the model more faithfully), a tick at every sheet, and a handle
    // that brings out the sheet at its tick; it rests above the top sheet. It serves touch screens,
    // where nothing hovers.
    const rail = el("div", "orders-demo-rail");
    rail.style.width = `${RAIL}px`;
    stage.style.setProperty("--orders-rail", `${RAIL}px`); // the stack is cut at the slider's column
    rail.style.setProperty("--orders-rail-line", `${RAIL_LINE}px`);
    const railLine = el("span", "orders-demo-rail-line");
    const railLabel = el("span", "orders-demo-rail-label");
    railLabel.append("more detailed explanations");
    railLabel.setAttribute("aria-hidden", "true");
    const ticks = el("span", "orders-demo-rail-ticks");
    ticks.setAttribute("aria-hidden", "true");
    const thumb = el("span", "orders-demo-thumb");
    thumb.tabIndex = 0;
    thumb.setAttribute("role", "slider");
    thumb.setAttribute("aria-orientation", "vertical");
    thumb.setAttribute("aria-label", "Order of the explanation");
    // apart from the arrow, at the mechanism's height, its own handle, on a thinner, orange arrow up
    // to the input: the mechanism, which gives the input its prediction
    const railMechLine = el("span", "orders-demo-rail-mech-line");
    // and its label beside it, as the arrow's
    const railMechLabel = el("span", "orders-demo-rail-label orders-demo-rail-mech-label");
    railMechLabel.append("model behavior");
    railMechLabel.setAttribute("aria-hidden", "true");
    const railMech = el("span", "orders-demo-rail-mech");
    railMech.setAttribute("aria-hidden", "true"); // (the mechanism's row serves the keyboard)
    // on a narrow screen, the invitation is a smaller one, at the slider: its first circle calls,
    // "Tap" beside it
    const railNudge = el("span", "orders-demo-rail-nudge");
    railNudge.setAttribute("aria-hidden", "true");
    // ("Tap" on a touch screen, "Click" with a mouse, as the invitation over the layers has it)
    const nudgeWord = el("span", "orders-demo-rail-nudge-word");
    nudgeWord.append(el("span", "orders-demo-hint-touch", "Tap"), el("span", "orders-demo-hint-hover", "Click"));
    railNudge.append(nudgeWord);
    rail.append(railLine, railLabel, ticks, railMechLine, railMechLabel, railMech, railNudge, thumb);
    // the line of the layer pointed at, on from its left corner to the slider's handle, as its row's
    // goes on from its right corner to its point in the plot
    const railLead = el("span", "orders-demo-rail-lead");
    // the stack's window: it starts at the slider's column, so that a stack swiped or slid aside goes
    // out of view there, not under the slider
    const view = el("div", "orders-demo-view");
    view.append(stack);
    // beside the sheets, the plot of the fit, upright
    const side = document.createElementNS(SVG, "svg");
    side.setAttribute("class", "orders-demo-side");
    side.setAttribute("role", "img");
    // an invitation, over the top layer, until a layer is first pointed at: tap one (or, with a
    // mouse, hover one) to explain the prediction by it. On a narrow screen, the slider's speaks
    // instead
    const hint = el("span", "orders-demo-hint");
    hint.setAttribute("aria-hidden", "true");
    hint.append(el("span", "orders-demo-hint-touch", "Tap a layer to explain"), el("span", "orders-demo-hint-hover", "Hover a layer to explain"));
    stage.append(rail, railLead, view, ladder, side, hint);
    // its frame stays as the examples change (its grid, its ticks and the name of its axis), and what
    // it plots fades
    const sideFrame = add(side, "g", { class: "orders-demo-side-frame" });
    const sideData = add(side, "g", { class: "orders-demo-side-data" });

    // under the stack, where the stepper shows (demo.css): a dot for every example, the one shown dark
    const dots = el("div", "orders-demo-dots");
    dots.setAttribute("aria-hidden", "true");
    dots.append(...data.examples.map(() => el("span")));
    // the blur of an estimated mechanism: an SVG filter, which every browser draws on SVG's shapes (a
    // CSS blur() it is not, in Safari), named for this demo (demo.css takes it through
    // --orders-blur)
    const blurs = document.createElementNS(SVG, "svg");
    blurs.setAttribute("class", "orders-demo-defs");
    blurs.setAttribute("aria-hidden", "true");
    const blur = add(add(blurs, "defs"), "filter", { id: `orders-blur-${plotId}`, x: "-20%", y: "-20%", width: "140%", height: "140%" });
    add(blur, "feGaussianBlur", { stdDeviation: 1.6 });
    root.style.setProperty("--orders-blur", `url(#orders-blur-${plotId})`);
    root.append(bar, planWindow, stage, dots, blurs, status, cardTip);
    root.classList.add("orders-demo-ready");

    // the stack, top to bottom: the sheets of order 1 to 4, a blank sheet for every order up to
    // the full decomposition, and the sheet of the full decomposition
    let items = [];
    let rows = [];
    let stops = []; // the slider's: its rest, then every sheet and blank sheet, top to bottom
    let cardRow = null; // the floating layers': the input's row, beside its card

    const current = () => data.every.find((example) => example.id === state.example);
    const sheetItems = () => items.filter((item) => item.sheet);
    const blankItems = () => items.filter((item) => !item.sheet && !item.base);
    // the slider's: no stop at the card or the mechanism; it goes through the explanations only, as
    // the plot does
    const stopItems = () => items.filter((item) => !item.base && !item.mechanism);

    // -- geometry: the size of the sheets, the stage, and every sheet's place ------------------------

    function measure() {
      // beside the card, the layers as large as the window's height lets them be, and the card as
      // large as they are (fitColumns)
      const besides = root.classList.contains("orders-demo-beside");
      const tallSide = besides ? fitColumns() : Infinity;
      const width = stage.clientWidth;
      const narrow = width < 560;
      // the plot beside the sheets: a narrow column, where the rows' lines lead
      const ladderWidth = Math.min(narrow ? 71 : 120, width * 0.17);
      const gap = narrow ? 10 : 28;
      const spread = Math.abs(Math.cos(TURN)) + Math.abs(Math.sin(TURN));
      // between the slider and the layers, room for the line from the handle to the one pointed at
      const lead = narrow ? 14 : 24;
      const room = width - ladderWidth - gap - RAIL - lead;
      // the same height for every example: the most sheets and blank sheets of any, of those the
      // step shows
      const n = Math.max(...data.every.map(shownSheets));
      const blanks = Math.max(...data.every.map((example) => shownBlanks(example).length));
      // the floating layers' card, below them (without them, on top), and the mechanism below it
      const layers = n ? (n - 1) * STRIP + blanks * BLANK_STRIP + AIR : 0;
      // the room holds the layers, and air beside them; beside the card, the window holds them too,
      // and with the card above them, the window holds them but for a little room (on a phone on
      // its side, the stack is not taller than the screen)
      const windowSide = besides ? tallSide : (viewportHeight() - WINDOW_ROOM) / (TILT * spread + layers + DEPTH);
      const side = Math.max(110, Math.min(360, room / ((1 + (besides ? ROOMY_BESIDE : ROOMY)) * spread), windowSide));
      const sheetHeight = side * TILT * spread;
      const stack = sheetHeight + (layers + DEPTH) * side;
      return {
        width,
        narrow,
        side,
        ladderLeft: width - ladderWidth,
        center: RAIL + lead + room / 2,
        height: Math.ceil(stack + 2 * THICKNESS + 16),
        sheetHeight,
      };
    }

    // beside the card: the layers' side as the window's height, below what is above them in the demo,
    // lets it be, and the card as large as they are, growing and shrinking with them: its face
    // (--orders-why-face) a share of their width, its column (--orders-why-column) as wide, or as
    // wide as its words need at the least. The stage's column (--orders-stage-column) as wide as
    // the layers, the slider and the plot need, as narrow or wide a stage lays them out (the plot's
    // column a sixth of it, up to its widest). Where the demo is too narrow for both, the two give
    // way together: the largest layers for which they fit; where it is wider, what is left of it is
    // around them (demo.css)
    function fitColumns() {
      const spread = Math.abs(Math.cos(TURN)) + Math.abs(Math.sin(TURN));
      const n = Math.max(...data.every.map(shownSheets));
      const blanks = Math.max(...data.every.map((example) => shownBlanks(example).length));
      const layers = n ? (n - 1) * STRIP + blanks * BLANK_STRIP + AIR : 0;
      const above = stage.getBoundingClientRect().top - root.getBoundingClientRect().top;
      const tall = viewportHeight() - above - WINDOW_MARGIN - 2 * THICKNESS - 16;
      const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
      const free = root.clientWidth - 1.25 * rem;
      const need = (layersWidth, lead, gap, plot) => {
        const base = RAIL + lead + layersWidth + gap;
        return Math.max(base / 0.83, base + Math.min(plot, (base / 0.83) * 0.17)) + 4;
      };
      const stageFor = (side) => {
        const layersWidth = side * (1 + ROOMY_BESIDE) * spread;
        const wide = need(layersWidth, 24, 28, 120);
        return wide >= 560 ? wide : need(layersWidth, 14, 10, 71);
      };
      const faceFor = (side) => FACE_BESIDE * side * spread;
      const columnFor = (side) => Math.max(WORDS_BESIDE * rem, faceFor(side));
      const fits = (side) => stageFor(side) + columnFor(side) <= free;
      let side = Math.max(110, Math.min(360, tall / (TILT * spread + layers + DEPTH)));
      // (the stage and the card both grow with the layers: the largest that fit, found by halves)
      if (!fits(side)) {
        let [low, high] = [110, side];
        for (let k = 0; k < 20; k++) {
          const middle = (low + high) / 2;
          if (fits(middle)) low = middle;
          else high = middle;
        }
        side = low;
      }
      const column = columnFor(side);
      // (the card's column holds the air before the stage too: demo.css)
      root.style.setProperty("--orders-why-column", `${Math.floor(column + 1.25 * rem)}px`);
      root.style.setProperty("--orders-why-face", `${Math.floor(faceFor(side))}px`);
      root.style.setProperty("--orders-stage-column", `${Math.floor(Math.max(0, Math.min(free - column, stageFor(side))))}px`);
      return side;
    }

    // the place (center) of every item, as the floating layers lie, open or at rest; a blank sheet
    // pointed at peeking out below its strip
    function layout() {
      const g = geometry;
      const peek = (item) => (!item.sheet && !item.base && item.order === state.blank ? PEEK * g.side : 0);
      return layeredLayout(unfolded()).map(({ x, y }, position) => ({ x, y: y + peek(items[position]) }));
    }
    // the floating layers' places: open, a strip of every layer showing below the one above it, the
    // card a breath of air below the lowest, and the mechanism deeper below it; at rest (projected),
    // every layer on the card, in the middle of the stage
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

    // the order of the layers on top of each other: the card under its layers, the mechanism under
    // the card, and the light between them
    const zIndex = (item) => items.length - items.indexOf(item);
    const layerZ = (item) => (item.base ? 2 : item.deep ? 0 : zIndex(item) + 3);

    // the item under a point of the stage, where the layers float open (where they are going, not
    // where they are while they move), the one on top first
    function itemAt(point) {
      const order = items.map((_, position) => position).sort((a, b) => layerZ(items[b]) - layerZ(items[a]));
      const places = layeredLayout(true);
      const position = order.find((p) => inside(point, corners(places[p])));
      return position === undefined ? null : items[position];
    }

    // -- the examples ---------------------------------------------------------------------------

    // the pill around the example chosen; it slides there, unless instant (as when the buttons
    // change their size)
    function movePill(instant) {
      const option = new Map(options).get(tabOf(current()).id);
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
      const position = data.examples.indexOf(tabOf(example)) + 1;
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
      const at = data.examples.indexOf(tabOf(current()));
      stepper.style.setProperty("--orders-step-from", `${delta * 12}px`);
      const next = data.examples[(at + delta + count) % count];
      choose(shownOf.get(next.id) ?? next, delta);
    }

    // choose an example: the pill slides to it while the stack, its fits and the plot fade out, and
    // the example fades in, at rest. Stepped to, the stack and the plot are swiped instead: they
    // leave to one side, out of view (from where a finger let go of them), and the example comes in
    // from the other. Without motion, it shows at once
    let chosen = 0; // the examples chosen so far: only the last one chosen is drawn
    const shift = (pixels) => root.style.setProperty("--orders-shift", `${roundTo(pixels, 1)}px`);
    function choose(example, direction = 0, tell = true) {
      if (state.example === example.id) return;
      Object.assign(state, { example: example.id, active: null, blank: null, pinned: false });
      shownOf.set(tabOf(example).id, example);
      for (const [id, option] of options) option.setAttribute("aria-pressed", String(id === tabOf(example).id));
      movePill(false);
      showStep(example);
      // the plan stays: what it shows flips to the example at once, the card after the problem
      showProblem(example, true);
      showPlan();
      [planView, planText].forEach((node) => flap(node, 300));
      const choice = ++chosen;
      // every demo of the page follows (out of view, at once); the one chosen in says so, once
      if (tell) {
        const { label, model, problem, photo } = example;
        status.textContent = `${label}${photo ? `, ${photo.title}` : ""}: ${model.title}, ${problem.output} (${problem.value})`;
        document.dispatchEvent(new CustomEvent(EXAMPLE_EVENT, { detail: { example: example.id } }));
      }
      if (reducedMotion() || (!tell && !inView)) {
        shift(0);
        root.style.removeProperty("--orders-swipe-opacity");
        // (out of view, its layers are drawn anew only once it comes near the view)
        if (!tell && !inView) stale = true;
        else render();
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
    // (a floating layer's: its marks; or their card's: the input)
    function drawSheet(example, i, layer = "marks") {
      const node = el("div", layer === "marks" ? "orders-demo-sheet orders-demo-layer" : "orders-demo-sheet");
      const svg = add(node, "svg", { viewBox: "0 0 100 100", class: "orders-demo-face" });
      const sheet = example.sheets[i];
      drawFace(svg, example, sheet, layer, `orders-${plotId}-${example.id}-${layer}-${i}`);
      add(svg, "text", { class: "orders-demo-tag", x: 6, y: 95.5 }, layer === "input" ? "Input" : sheetName(sheet).face);
      return node;
    }
    // a face of an example's kind, into svg (id names what it defines, as the photo's clip); a
    // face with all of it, the card's, can be inspected: its inspector is kept by it
    const inspectors = new WeakMap();
    function drawFace(svg, example, sheet, layer, id) {
      const inspector =
        example.kind === "graph"
          ? moleculeFace(svg, example, sheet, layer)
          : example.kind === "tree"
            ? treeFace(svg, example, sheet, layer)
            : example.kind === "map"
              ? mapFace(svg, example, sheet, layer)
              : imageTextFace(svg, example, sheet, `${root.dataset.root}${example.image}`, id, layer);
      if (inspector) inspectors.set(svg, inspector);
    }
    // every example's layers, drawn once: an example chosen again shows them as they were drawn
    const drawn = new Map();
    // (only those the step shows: the problem and the mechanism none, the Shapley values one)
    function sheetsOf(example) {
      const nodes = drawn.get(example.id) ?? [];
      for (let i = nodes.length; i < shownSheets(example); i++) nodes.push(drawSheet(example, i));
      drawn.set(example.id, nodes);
      return nodes;
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
        node.querySelector(".orders-demo-tag").textContent = mechanismTag(example);
        node.classList.toggle("orders-demo-estimated", estimated(example));
        mechanisms.set(example.id, node);
      }
      return mechanisms.get(example.id);
    }
    // the mechanism's name on its layer, and on the plan's face of it
    const mechanismTag = (example) =>
      estimated(example) ? "The mechanism, estimated" : `The mechanism (${sheetName(example.sheets.at(-1)).tag})`;
    // the chip's words as it clears: in its light ink or its dark one, whichever reads better on the
    // color it clears to (its own, mixed with the paper's as the CSS mixes them), and its small print
    // as faint as it can be and still read (4.5:1, or as close as the ink gets)
    const parser = document.createElement("canvas").getContext("2d");
    const rgbOf = (color) => {
      parser.fillStyle = "#000";
      parser.fillStyle = color.trim() || "#000";
      const value = parser.fillStyle; // #rrggbb, or rgba(r, g, b, a)
      return value.startsWith("#") ? [1, 3, 5].map((k) => parseInt(value.slice(k, k + 2), 16)) : value.match(/[\d.]+/g).slice(0, 3).map(Number);
    };
    const luminance = (rgb) => {
      const [r, g, b] = rgb.map((c) => (c / 255 <= 0.04045 ? c / 255 / 12.92 : ((c / 255 + 0.055) / 1.055) ** 2.4));
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    const contrast = (p, q) => (Math.max(luminance(p), luminance(q)) + 0.05) / (Math.min(luminance(p), luminance(q)) + 0.05);
    const blend = (p, q, t) => p.map((c, k) => c * t + q[k] * (1 - t)); // t of p over q
    let [inked, chipClear] = ["", 0];
    // (the chip's colors, read again only when its kind or the page's theme changes: a read of the
    // computed style lays the demo out, and the chip is inked anew at every explanation pointed at)
    let [chipColors, chipColorsKey] = [[], ""];
    function inkChip(clear = chipClear) {
      chipClear = clear;
      const theme = `${chip.classList.contains("orders-demo-chip-gray")} ${document.body.dataset.theme ?? ""} ${darkScheme.matches}`;
      if (theme !== chipColorsKey) {
        const style = getComputedStyle(chip);
        chipColors = ["--orders-chip", "--orders-glass", "--orders-glass-ink"].map((name) => style.getPropertyValue(name));
        chipColorsKey = theme;
      }
      const [chipColor, glass, glassInk] = chipColors;
      const key = `${clear} ${chipColors.join(" ")}`;
      if (key === inked) return;
      inked = key;
      const back = blend(rgbOf(chipColor), rgbOf(glass), 1 - clear);
      const [light, dark] = [[242, 244, 248], rgbOf(glassInk)];
      // (the light one as long as it reads, as the chip is dark; the dark one where it reads better)
      const ink = contrast(light, back) >= 4.5 || contrast(light, back) >= contrast(dark, back) ? light : dark;
      let soft = 1;
      for (let a = 0.62; a < 1; a += 0.04) {
        if (contrast(blend(ink, back, a), back) >= 4.5) {
          soft = a;
          break;
        }
      }
      chip.style.setProperty("--orders-chip-ink", `rgb(${ink.join(" ")})`);
      chip.style.setProperty("--orders-chip-soft", `rgb(${blend(ink, back, soft).map(Math.round).join(" ")})`);
      chip.classList.toggle("orders-demo-chip-open", ink === dark);
    }
    // the plan's faces, drawn once: a sheet's, all of it, on the input; or the input alone (null)
    const plans = new Map();
    function planFace(example, sheet) {
      const name = sheet ? `${sheet.order}${sheet.full ? "-full" : ""}` : "input";
      const key = `${example.id} ${name}`;
      if (!plans.has(key)) {
        const svg = document.createElementNS(SVG, "svg");
        svg.setAttribute("viewBox", "0 0 100 100");
        svg.setAttribute("class", "orders-demo-plan-face");
        drawFace(svg, example, sheet ?? example.sheets[0], sheet ? "all" : "input", `orders-${plotId}-${example.id}-plan-${name}`);
        plans.set(key, svg);
      }
      return plans.get(key);
    }
    // the explanation of an order between, as a blank layer stands for it: of the orders from the
    // highest with an interaction on, the full decomposition; else its own, where the data has it
    const betweenOf = (example, order) =>
      order >= example.largest ? example.sheets.at(-1) : (example.between?.find((sheet) => sheet.order === order) ?? null);
    // and its marks, for the card to wear, projected
    const betweenMarks = new Map();
    function marksOf(example, sheet) {
      const key = `${example.id} ${sheet.order}${sheet.full ? "-full" : ""}`;
      if (!betweenMarks.has(key)) {
        const svg = document.createElementNS(SVG, "svg");
        svg.setAttribute("viewBox", "0 0 100 100");
        svg.setAttribute("class", "orders-demo-face");
        // (its ids without spaces, which would end a url(#...) reference)
        drawFace(svg, example, sheet, "marks", `orders-${plotId}-${example.id}-between-${sheet.order}${sheet.full ? "-full" : ""}`);
        betweenMarks.set(key, svg);
      }
      return betweenMarks.get(key);
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
      showPicker(example);
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
      // (the problem alone, as a step, does not call)
      const calls = !revealed && part === "all";
      chip.classList.toggle("orders-demo-chip-call", calls);
      // (and, once explained, no animation left on it, whatever a browser keeps running)
      chip.style.animation = calls ? "" : "none";
      if (!calls) {
        // (explained from the keyboard, its focus goes on to the slider, not back to the page)
        const focused = document.activeElement === chip;
        for (const name of ["role", "tabindex", "aria-label"]) chip.removeAttribute(name);
        explainTab.remove();
        if (focused) requestAnimationFrame(() => thumb.focus({ preventScroll: true }));
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
      const tabular = example.kind === "tree" || example.kind === "map";
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
      } else if (example.kind === "map") {
        // a head of the features, rows of houses, and how many
        svg.classList.add("orders-demo-input-matrix");
        const [left, top, cell] = [10, 10, 10];
        for (let row = 0; row < 6; row++) {
          for (let column = 0; column < 8; column++) {
            add(svg, "rect", { x: left + column * cell, y: top + row * cell, width: cell - 1.5, height: cell - 1.5, rx: 1.2, class: row ? "" : "orders-demo-input-head" });
          }
        }
        add(svg, "text", { x: 50, y: 86, "text-anchor": "middle" }, `${example.predictions.length.toLocaleString("en")} × ${example.features.length}`);
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
    // -- inspecting the card's face: the pointer over a player (or a group, a house, a row) of the
    // explanation shown brings out on the face what the explanation says of it, and the card's tip
    // says it in numbers, over the face. With a mouse, by hovering; on a touch screen, by a tap (a
    // tap elsewhere lets go). The face's inspector (demo.js: inspectorOf) knows its targets ------
    let inspected = null; // the inspector and its target, while one is inspected
    const inspectorNow = () => inspectors.get(planView.firstChild) ?? null;
    function letGo() {
      if (!inspected) return;
      inspected.inspector.clear();
      inspected = null;
      cardTip.classList.remove("orders-demo-tip-on");
    }
    function inspect(target) {
      const inspector = inspectorNow();
      if (!inspector || !target) {
        letGo();
        return;
      }
      if (inspected?.target === target) return;
      inspected?.inspector.clear();
      inspected = { inspector, target };
      const said = inspector.show(target);
      // the face inspected once, its invitation goes, for good
      root.classList.add("orders-demo-inspected");
      cardTip.classList.toggle("orders-demo-tip-on", Boolean(said));
      if (!said) return;
      const kind = current().kind;
      const title = el("div", "orders-demo-tip-title");
      title.append(el("strong", "", said.title[0]), ` ${said.title[1]}`);
      // a row: a name, and a value in the red of what raises f(x) or the blue of what lowers it (a
      // value without a sign, as a house's prediction, in neither)
      // (a sum, under a rule)
      const rows = said.rows.map(([name, value, text, sum]) => {
        const row = el("div", sum ? "orders-demo-tip-row orders-demo-tip-sum" : "orders-demo-tip-row");
        const tone = value === null ? "" : value >= 0 ? "orders-demo-plan-raises" : "orders-demo-plan-lowers";
        row.append(el("span", "", name), el("b", tone, text ?? VALUE[kind](value)));
        return row;
      });
      // the data's note on it, where it has one, under its name; and under the rows, what the
      // values are of (a group's: when it counts)
      const story = said.story ? [el("div", "orders-demo-tip-story", said.story)] : [];
      cardTip.replaceChildren(title, ...story, ...rows, el("div", "orders-demo-tip-note", `added to ${OF[kind]}${said.when ? ` ${said.when}` : ""}`));
      // over its target, in the middle, or under it where there is no room above; kept in the demo
      const [face, box] = [planView.firstChild.getBoundingClientRect(), root.getBoundingClientRect()];
      const unit = face.width / 100;
      const [x, above, below] = [face.left - box.left + target.x * unit, face.top - box.top + (target.y - target.ry) * unit, face.top - box.top + (target.y + target.ry) * unit];
      const [width, height] = [cardTip.offsetWidth, cardTip.offsetHeight];
      cardTip.style.left = `${roundTo(Math.max(2, Math.min(box.width - width - 2, x - width / 2)), 1)}px`;
      cardTip.style.top = `${roundTo(above - height - 9 >= 0 ? above - height - 9 : below + 9, 1)}px`;
    }
    // the target under the pointer: the nearest of those whose box holds it (a small one's box at
    // least a few pixels wide, more for a finger)
    function targetAt(event) {
      const inspector = inspectorNow();
      if (!inspector) return null;
      const face = planView.firstChild.getBoundingClientRect();
      const [x, y] = [((event.clientX - face.left) / face.width) * 100, ((event.clientY - face.top) / face.height) * 100];
      const least = ((event.pointerType === "mouse" ? 5 : 11) / face.width) * 100;
      const group = inspector.groupAt(event.clientX, event.clientY);
      if (group && inspector.over) return group;
      let [best, nearest] = [null, Infinity];
      for (const target of inspector.targets) {
        if (Math.abs(x - target.x) > Math.max(target.rx, least) || Math.abs(y - target.y) > Math.max(target.ry, least)) continue;
        const distance = Math.hypot(x - target.x, y - target.y);
        if (distance < nearest) [best, nearest] = [target, distance];
      }
      return best ?? group;
    }
    planView.addEventListener("pointermove", (event) => {
      if (event.pointerType === "mouse") inspect(targetAt(event));
    });
    planView.addEventListener("pointerleave", (event) => {
      if (event.pointerType === "mouse") letGo();
    });
    // (a tap on a target inspects it; on the face beside one, or anywhere else in the demo, lets go)
    planView.addEventListener("click", (event) => inspect(targetAt(event)));
    root.addEventListener("click", (event) => {
      if (!event.target.closest(".orders-demo-plan-view, .orders-demo-plan-top")) letGo();
    });

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
      // (its name in the card; beside the stack, where the face wears its name, its order instead)
      const named = el("span", "orders-demo-plan-named");
      const ordered = el("span", "orders-demo-plan-ordered");
      title.append(named, ordered);
      const what = el("span", "orders-demo-plan-what");
      const fit = el("span", "orders-demo-plan-fit");
      const full = example.sheets.at(-1);
      let r2 = full.r2;
      // (the explanation shown: a sheet's, the mechanism's as the full decomposition, an order's
      // between where the data has one)
      const sheet = mechanism ? full : blank !== null ? betweenOf(example, blank) : example.sheets[index];
      if (mechanism) {
        const n = playersOf(example);
        named.append(el("strong", "", "Mechanism"), " the model itself");
        ordered.append(el("strong", "", "The model"), " itself");
        if (estimated(example)) what.append("up to ", power(n), " terms, estimated");
        else what.append(`${full.count.toLocaleString("en")} of `, power(n), " terms");
      } else if (blank !== null) {
        r2 = fitOf(example, blank)[1];
        named.append(el("strong", "", `Order ${blank}`), " between the sheets");
        ordered.append(el("strong", "", `Order ${blank}`), " explanation");
        what.append(blank >= example.largest ? "as the full decomposition" : `groups of up to ${blank} ${words.players}`);
      } else {
        const sheet = example.sheets[index];
        r2 = sheet.r2;
        named.append(el("strong", "", sheetName(sheet).tag), ` ${sheetName(sheet).name}`);
        ordered.append(el("strong", "", `Order ${sheet.order}`), " explanation");
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
      // the mechanism, where only an estimate shows it: what the estimate falls short of 1, the
      // estimation gap, hatched on the bar, and named in place of the verdict
      const gap = mechanism && estimated(example) ? 1 - r2 : 0;
      if (gap) {
        const hatch = el("span", "orders-demo-meter-gap");
        Object.assign(hatch.style, { left: `${r2 * 100}%`, width: `${gap * 100}%` });
        meter.append(hatch);
      }
      fit.append(meter);
      const judged = gap ? el("span", "orders-demo-verdict orders-demo-plan-gap") : el("span", "orders-demo-verdict", verdict(r2));
      // (on a phone, "gap" alone)
      if (gap) judged.append(el("span", "orders-demo-plan-long", "estimation "), `gap ${fit2(gap)}`);
      fit.append(el("span", "orders-demo-r2", fit2(r2)), judged);
      const method = el("span", "orders-demo-plan-method");
      method.append("Explained with ", el("strong", "", example.model.method), el("span", "orders-demo-plan-how", `, ${example.model.how}`));
      const legend = el("span", "orders-demo-plan-legend");
      for (const [color, meaning] of [["up", "raises"], ["down", "lowers"]]) {
        const key = el("span", "orders-demo-plan-key");
        key.append(el("i", `orders-demo-plan-${color}`), meaning, math("f(x)"));
        legend.append(key);
      }
      // its strongest, in a line (short: the card's lines are as wide as its widest): a player of
      // the Shapley values, a group of the orders above (of
      // many houses, what their map shows); pointed at, the face brings it out. An order between
      // without an explanation of its own says nothing, on a line as tall
      const top = el("span", "orders-demo-plan-top");
      if (sheet) {
        const { players, text, sign } = strongestOf(example, sheet);
        const name = el("strong", "", groupName(example, players, (player) => playerName(example, player)));
        const value = el("span", sign > 0 ? "orders-demo-plan-raises" : sign < 0 ? "orders-demo-plan-lowers" : "", text);
        // (named by what it is: a group, a pair, or a player, an atom, a feature, a patch or a word)
        const one = example.kind === "image-text" ? (players[0] < example.grid ** 2 ? "patch" : "word") : PLAYERS[example.kind].one;
        top.append(`Top ${players.length > 2 ? "group" : players.length > 1 ? "pair" : one}: `, name, " ", value);
        top.addEventListener("pointerenter", (event) => {
          if (event.pointerType === "mouse") inspect(inspectorNow()?.find(players) ?? null);
        });
        top.addEventListener("pointerleave", (event) => {
          if (event.pointerType === "mouse") letGo();
        });
        top.addEventListener("click", () => inspect(inspectorNow()?.find(players) ?? null));
      } else top.append("\u00a0");
      return { nodes: [title, what, fit, top, method, legend], r2 };
    }

    // the card keeps its place and its size whatever it shows, in every example: its words take a
    // column of their own width, as tall as the most words it can show (measured out of view, for
    // the column's width and the view)
    let settled = "";
    function settleCard() {
      // (beside the layers, the card's column is the words' room: they take as much of it as their
      // widest line needs, in its middle)
      const besides = root.classList.contains("orders-demo-beside");
      const style = getComputedStyle(why);
      const width = besides ? why.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight) : planText.clientWidth;
      const key = `${width} ${besides}`;
      if (!width || key === settled) return;
      settled = key;
      const probe = el("span", "orders-demo-plan-text orders-demo-plan-probe");
      const words = el("span", "orders-demo-plan-words");
      probe.style.width = `${width}px`;
      probe.append(words);
      why.append(probe);
      // every line as it says the most, at full size: where one is wider than the column (on a
      // narrow screen), all the words shrink alike, the same whatever the card shows; but what an
      // explanation holds, the longest line, takes two lines instead, where it would not fit one
      words.style.setProperty("--orders-words-fit", "1");
      let [widest, widestWhat] = [0, 0];
      // (and how wide their text is, the lines being as wide as the column: beside, their block)
      let [natural, naturalWhat] = [0, 0];
      const range = document.createRange();
      const textWidth = (line) => {
        range.selectNodeContents(line);
        return range.getBoundingClientRect().width;
      };
      // (every explanation's words in a block of their own, all in the probe at once: laid out once,
      // then read, not laid out anew for every one)
      const blocks = data.every.flatMap((example) =>
        [...example.sheets.map((_, index) => ({ index })), ...between(example).map(([blank]) => ({ blank })), { mechanism: true }].map(
          (pick) => {
            const block = el("span", "orders-demo-plan-words");
            block.style.setProperty("--orders-words-fit", "1");
            block.append(...wordsOf(example, pick).nodes);
            return block;
          },
        ),
      );
      probe.replaceChildren(...blocks);
      for (const block of blocks) {
        for (const line of block.children) {
          if (line.classList.contains("orders-demo-plan-what")) {
            widestWhat = Math.max(widestWhat, line.scrollWidth);
            if (besides) naturalWhat = Math.max(naturalWhat, textWidth(line));
          } else {
            widest = Math.max(widest, line.scrollWidth);
            if (besides) natural = Math.max(natural, textWidth(line));
          }
        }
      }
      // (and the last one's words alone again, to measure the height by)
      words.replaceChildren(...blocks.at(-1).childNodes);
      probe.replaceChildren(words);
      // (beside the layers, the words grow into the column, as their widest text lets them, up to a
      // third larger, on lines of one)
      const fit = besides ? Math.min(1.3, (width - 2) / Math.max(natural, naturalWhat)) : Math.min(1, (width - 1) / widest);
      const two = !besides && widestWhat * fit > width - 1;
      for (const node of [planWords, words]) {
        node.style.setProperty("--orders-words-fit", roundTo(fit, 3));
        node.classList.toggle("orders-demo-plan-two", two);
      }
      planText.style.width = besides ? `${Math.ceil(Math.min(width, Math.max(natural, two ? 0 : naturalWhat) * fit)) + 2}px` : "";
      // and as tall as its words, shrunk so, are (every explanation's have as many lines)
      planText.style.minHeight = `${Math.ceil(probe.offsetHeight)}px`;
      probe.remove();
    }

    function showPlan() {
      const example = current();
      const last = example.sheets.length - 1;
      const pointed = state.active !== null ? sheetItems()[state.active] : null;
      // at rest, what the step adds, as pointed at (as soon as it is chosen, before its layers are
      // drawn); without a step, the top layer's explanation, unlit
      const rest = pointed || state.blank !== null ? null : restPick(example);
      const lit = pointed !== null || state.blank !== null || rest !== null;
      // (where the step shows no explanation, and points at nothing, the mechanism, unlit)
      const mechanism =
        state.blank === null && (pointed ? Boolean(pointed.mechanism) : rest ? Boolean(rest.mechanism) : shownSheets(example) === 0);
      const index = pointed && !mechanism ? Math.min(pointed.index, last) : (rest?.index ?? 0);
      // (an order between: its own explanation, or, where the data has none, the input alone)
      const face = planFace(
        example,
        state.blank !== null ? betweenOf(example, state.blank) : mechanism ? example.sheets[last] : example.sheets[index],
      );
      if (planView.firstChild !== face) {
        letGo(); // (of what was inspected on the face before)
        planView.replaceChildren(face, planTag);
      }
      // named as the layer it shows: a sheet, the mechanism, or an order between
      const faced = state.blank !== null ? betweenOf(example, state.blank) : mechanism ? example.sheets[last] : example.sheets[index];
      tagFace(
        state.blank !== null ? `${state.blank}-${INDEX}` : mechanism ? mechanismTag(example) : sheetName(example.sheets[index]).face,
        Boolean(faced?.groups?.length),
      );
      // the mechanism, where only an estimate shows it, blurred
      planView.classList.toggle("orders-demo-estimated", Boolean(mechanism) && state.blank === null && estimated(example));

      const pick = state.blank !== null ? { blank: state.blank } : mechanism ? { mechanism: true } : { index };
      const { nodes, r2 } = wordsOf(example, pick);
      planWords.replaceChildren(...nodes);
      // the mechanism's, in its orange
      plan.classList.toggle("orders-demo-plan-mechanism", mechanism && lit);
      // an explanation's, a little green, as its light on the input (an order between without one
      // of its own, not)
      plan.classList.toggle("orders-demo-plan-xai", lit && !mechanism && (state.blank === null || Boolean(betweenOf(example, state.blank))));

      // the chip clears as far as the explanation pointed at fits (at rest, none is): opaque, it is
      // the black box; clear, the explanation has opened it
      const clear = lit ? Math.max(0, Math.min(1, r2)) : 0;
      flow.style.setProperty("--orders-clear", roundTo(clear, 3));
      plan.classList.toggle("orders-demo-plan-lit", lit);
      inkChip(clear);
      // its R^2 in two decimals, or >.99 short of 1: as wide whatever it is, so that the chip, whose
      // words shrink to fit it, does not change its size from one to the next
      const short = r2 >= EXACT ? "1.00" : r2 >= 0.995 ? ">.99" : r2.toFixed(2);
      const note = lit ? `R² ${short} explained` : example.model.title;
      if (chipNote.textContent !== note) {
        // (refit only where the note's shape changes: its digits are as wide as each other)
        const shape = (text) => text.replace(/\d/g, "0");
        const refit = shape(chipNote.textContent) !== shape(note);
        chipNote.textContent = note;
        if (refit) fitPart(chipShows);
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
    // the explanation built from the top down: every part comes by how far below the chip it lies,
    // the card, the sheets one after the other with their rows, and the plot beside them, drawn
    // down (demo.css)
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
      // the plot beside the sheets is drawn down in the time the build takes to pass it
      side.style.setProperty("--orders-side-time", `${Math.round((stage.clientHeight / length) * building)}ms`);
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
    watch(
      new IntersectionObserver(
        ([entry]) => {
          stageInView = entry.isIntersecting;
          unfold();
        },
        { threshold: 0.45 },
      ),
    ).observe(stage);
    function unfold() {
      clearTimeout(unfolding);
      if (unfoldedForGood || !revealed || !stageInView) return;
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
      // (beside, the plan is no box of its own: the light is drawn in the demo's)
      const besides = root.classList.contains("orders-demo-beside");
      const box = (besides ? root : plan).getBoundingClientRect();
      const [from, to] = [chip.getBoundingClientRect(), why.getBoundingClientRect()];
      const f = (x, y) => `${roundTo(x - box.left, 1)},${roundTo(y - box.top, 1)}`;
      // halfway round the rounded corners of the chip's bottom and of the card's top
      const round = (node) => parseFloat(getComputedStyle(node).borderTopLeftRadius) * (1 - Math.SQRT1_2);
      const [a, b] = [round(chip), round(why)];
      const onCard = [f(from.left + a, from.bottom - a), f(from.right - a, from.bottom - a), f(to.right - b, to.top + b), f(to.left + b, to.top + b)];
      let [hull, edges] = [onCard, `M${onCard[0]}L${onCard[3]}M${onCard[1]}L${onCard[2]}`];
      const glows = besides && geometry !== null && items.length > 0;
      if (glows) {
        // beside: one light, a cone out of the chip's bottom, as wide to either side (its edges at
        // SPREAD from the vertical), filled with a glow that fades out a few centimeters from the chip
        // (a share of the demo's width, at most 240 px: well inside its edges), as its edges do
        const [cx, cy] = [(from.left + from.right) / 2, from.bottom - a];
        const reach = Math.max(150, Math.min(240, 0.26 * box.width));
        const [dx, dy] = [2 * reach * Math.sin(SPREAD), 2 * reach * Math.cos(SPREAD)];
        const [left, right] = [from.left + a, from.right - a];
        hull = [f(left, cy), f(right, cy), f(right + dx, cy + dy), f(left - dx, cy + dy)];
        edges = `M${hull[0]}L${hull[3]}M${hull[1]}L${hull[2]}`;
        for (const node of [castGlowLight, castFade]) {
          node.setAttribute("cx", roundTo(cx - box.left, 1));
          node.setAttribute("cy", roundTo(cy - box.top, 1));
          node.setAttribute("r", roundTo(reach, 1));
        }
        castMaskArea.setAttribute("width", roundTo(box.width, 1));
        castMaskArea.setAttribute("height", roundTo(box.height, 1));
        castHull.setAttribute("fill", `url(#orders-cast-glow-${plotId})`);
        castEdges.setAttribute("mask", `url(#orders-cast-mask-${plotId})`);
      } else {
        castHull.setAttribute("fill", `url(#orders-cast-${plotId})`);
        castEdges.removeAttribute("mask");
      }
      cast.setAttribute("width", roundTo(box.width, 1));
      cast.setAttribute("height", roundTo(box.height, 1));
      castHull.setAttribute("points", hull.join(" "));
      castEdges.setAttribute("d", edges);
    }

    // the light between the layer pointed at and the card
    const beam = document.createElementNS(SVG, "svg");
    beam.setAttribute("class", "orders-demo-beam");
    beam.setAttribute("aria-hidden", "true");
    const glow = add(add(beam, "defs"), "linearGradient", { id: `orders-beam-${plotId}`, x1: 0, y1: 0, x2: 0, y2: 1 });
    add(glow, "stop", { offset: 0, class: "orders-demo-beam-top" });
    add(glow, "stop", { offset: 1, class: "orders-demo-beam-bottom" });
    // its edges, each with a small arrowhead where it reaches the input (green, or the mechanism's
    // orange)
    const tip = add(beam.querySelector("defs"), "marker", { id: `orders-tip-${plotId}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" });
    add(tip, "path", { class: "orders-demo-beam-tip", d: "M0,1L9,5L0,9Z" });
    const beamHull = add(beam, "polygon", { fill: `url(#orders-beam-${plotId})` });
    const beamEdges = [0, 1, 2, 3].map(() => add(beam, "path", { class: "orders-demo-beam-edges", "marker-end": `url(#orders-tip-${plotId})` }));
    // the examples not shown yet, drawn while the page is idle, a sheet at a time, so that choosing
    // one only has to show it
    const idle = window.requestIdleCallback ?? ((callback) => setTimeout(callback, 100));
    function drawAhead() {
      const example = data.every.find((other) => (drawn.get(other.id)?.length ?? 0) < shownSheets(other));
      if (!example) return;
      const nodes = drawn.get(example.id) ?? [];
      nodes.push(drawSheet(example, nodes.length));
      drawn.set(example.id, nodes);
      idle(drawAhead);
    }

    function render() {
      const example = current();
      root.classList.toggle("orders-demo-veiled", !revealed);
      wideEnough();
      Object.assign(state, { active: null, blank: null, pinned: false }); // a new example shows at rest
      for (const [id, option] of options) option.setAttribute("aria-pressed", String(id === tabOf(example).id));
      stage.setAttribute("aria-label", `${example.label}: its explanations of order 1 to n, as layers above its input, with their fit`);
      stack.replaceChildren();
      for (const row of [...rows, cardRow]) row?.remove();
      if (problemOf !== example.id) showProblem(example);
      showCall();
      geometry = measure();

      const last = example.sheets.at(-1);
      const shown = shownSheets(example);
      items = [];
      const nodes = sheetsOf(example);
      example.sheets.slice(0, shown).forEach((sheet, i) => {
        if (sheet === last) {
          for (const fit of shownBlanks(example)) items.push({ order: fit[0], fit });
        }
        items.push({ order: sheet.order, sheet, index: i, node: nodes[i] });
      });
      for (const item of items) {
        if (!item.sheet) item.node = el("div", "orders-demo-sheet orders-demo-blank orders-demo-layer");
      }
      // below the layers, the input's card, and below it the mechanism: a sheet of its own, after the
      // sheets shown (its index, as its row's, theirs and one)
      const card = { order: 0, base: true, node: cardOf(example) };
      items.push(card, { order: last.order, sheet: last, index: shown, mechanism: true, deep: true, node: mechanismOf(example) });
      for (const item of items) item.node.classList.toggle("orders-demo-deep", Boolean(item.deep));
      state.active = restActive();

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
      rows = example.sheets.slice(0, shown).map((sheet, i) => {
        const names = sheetName(sheet);
        const row = el("button", "orders-demo-row");
        row.type = "button";
        row.setAttribute("aria-label", `${names.name}, order ${sheet.order}: R² ${fit2(sheet.r2)}, ${verdict(sheet.r2)}`);
        wire(row, i);
        ladder.append(row);
        return row;
      });
      {
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
        // (where only an estimate shows it, so it says, on a line of its own)
        const name = el("span", "orders-demo-row-label");
        name.append(el("span", "", estimated(example) ? "Mechanism," : "Mechanism"));
        if (estimated(example)) name.append(el("span", "orders-demo-row-note", "estimated"));
        row.append(name);
        wire(row, i);
        ladder.append(row);
        rows.push(row);
      }
      ticks.replaceChildren(...stopItems().map((item) => el("span", item.sheet ? "orders-demo-tick" : "orders-demo-tick orders-demo-tick-blank")));

      // the floating layers' row of their card: the input, what the layers explain, named
      cardRow = el("div", "orders-demo-row orders-demo-card-row");
      cardRow.setAttribute("aria-hidden", "true");
      cardRow.append(el("span", "orders-demo-row-label", "Input"));
      ladder.append(cardRow);

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
      place(state.active);
      stack.append(...items.map((item) => item.node));
      stack.append(beam);
      root.getBoundingClientRect();
      root.classList.remove("orders-demo-instant");
      if (root.classList.contains("orders-demo-beside")) castLightOut();
    }

    // the slider's labels' lengths (written down the slider: their heights), measured once shown,
    // and anew in the fonts loaded
    let labelLengths = [0, 0];
    function labelLength(label, k) {
      if (!labelLengths[k]) labelLengths[k] = label.offsetHeight;
      return labelLengths[k];
    }

    // move every sheet and row to its place, for the sheet pointed at (or null)
    function place(active) {
      const g = geometry;
      stage.style.height = `${g.height}px`;
      root.classList.toggle("orders-demo-narrow", g.narrow);
      const out = typeof active === "number";
      const places = layout();
      // the rows and the beam keep to where the layers float, open or at rest
      const anchors = layeredLayout(true);
      const open = unfolded();
      const opening = open !== root.classList.contains("orders-demo-open");
      root.classList.add("orders-demo-floating");
      root.classList.toggle("orders-demo-open", open);
      // the layer pointed at, or the blank layer: the others fade
      const target = out ? sheetItems()[active] : state.blank !== null ? items.find((item) => !item.sheet && !item.base && item.order === state.blank) : null;
      items.forEach((item, position) => {
        const { x, y } = places[position];
        const node = item.node;
        node.style.width = node.style.height = `${g.side}px`;
        node.style.marginLeft = node.style.marginTop = `${-g.side / 2}px`;
        node.style.zIndex = layerZ(item);
        // the layers peel off the card one after the other, the top one first, and settle back onto
        // it the lowest first
        const delay = opening ? (open ? position : items.length - position) * 45 : 0;
        node.style.transitionDelay = `${delay}ms`;
        node.classList.toggle("orders-demo-top", position === 0);
        node.style.transform = `translate(${roundTo(x, 2)}px, ${roundTo(y, 2)}px) scaleY(${TILT}) rotate(${(TURN * 180) / Math.PI}deg)`;
        // the edge below the face, in the face's own coordinates: straight down on the stage
        const depth = (item.sheet ? THICKNESS : THICKNESS / 2) / TILT;
        node.style.setProperty("--orders-edge-x", `${roundTo(Math.sin(TURN) * depth, 2)}px`);
        node.style.setProperty("--orders-edge-y", `${roundTo(Math.cos(TURN) * depth, 2)}px`);
        node.classList.toggle("orders-demo-receded", Boolean(target) && !item.base && item !== target);
        node.classList.toggle("orders-demo-pointed", item === target);
        // a blank layer pointed at shows its order's explanation
        if (!item.sheet && !item.base) showBlank(item, item === target);
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

      // the slider: its stops at the middles of the layers at rest (without a blank sheet's peek), and
      // its invitation at the first layer's (placed below, once the stops are)
      // past the last explanation, the mechanism's, on its own handle; its line from its rest down to
      // the last explanation, and its handle at the layer pointed at
      const rest = layeredLayout(true);
      const mechanism = items.find((item) => item.deep);
      stops = [
        { y: Math.max(10, rest[0].y - RAIL_REST), item: null },
        ...stopItems().map((item) => ({ y: rest[items.indexOf(item)].y, item })),
        { y: rest[items.indexOf(mechanism)].y, item: mechanism },
      ];
      const [top, bottom] = [stops[0].y, stops.at(-2).y];
      railNudge.style.translate = `0 ${roundTo(stops[1].y, 2)}px`;
      // the invitation over the first layer to point at (the top one; where the step shows no
      // explanation, the mechanism), a little above its middle
      const first = places[items.indexOf(stops[1].item)];
      hint.style.left = `${roundTo(first.x, 2)}px`;
      hint.style.top = `${roundTo(first.y - g.sheetHeight * 0.12, 2)}px`;
      // the arrow goes through the explanations (where the step shows none, there is none), and its
      // label along two of them at least
      railLine.hidden = stops.length < 3;
      railLabel.hidden = stops.length < 4;
      // (where the one explanation, or the mechanism alone, is pointed at already, at rest, no
      // invitation)
      root.classList.toggle("orders-demo-uninvited", restActive() !== null && stops.length < 4);
      // the arrow goes on past the explanations, down to the input's height, which it points at from
      // above, as the mechanism's does from below (each tip a few pixels short of it)
      const [mechY, cardY] = [rest[items.indexOf(mechanism)].y, rest[items.findIndex((item) => item.base)].y];
      const end = cardY - 13; // (its tip 8px below)
      railLine.style.transform = `translateY(${roundTo(top, 2)}px)`;
      railLine.style.height = `${roundTo(end - top, 2)}px`;
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
      // placed by translate, which the handle's growing on hover does not scale, unlike a transform; at
      // the mechanism, on its handle, in its orange
      thumb.style.translate = `0 ${roundTo(stops[at].y, 2)}px`;
      thumb.classList.toggle("orders-demo-thumb-rest", at === 0);
      thumb.classList.toggle("orders-demo-thumb-mechanism", Boolean(stops[at].item?.mechanism));
      thumb.setAttribute("aria-valuemin", String(restActive() === null ? 0 : 1));
      thumb.setAttribute("aria-valuemax", String(stops.length - 1));
      thumb.setAttribute("aria-valuenow", String(at));
      thumb.setAttribute("aria-valuetext", at === 0 ? "none" : stopName(stops[at].item));
      // from the layer pointed at down to the input, the arrow turns green, as the light it casts onto
      // the input, and so does its handle
      const lit = target && !target.deep && (target.sheet || betweenOf(current(), target.order));
      railLine.classList.toggle("orders-demo-rail-line-lit", Boolean(lit));
      thumb.classList.toggle("orders-demo-thumb-lit", Boolean(lit));
      railLine.style.setProperty("--orders-split", lit ? `${roundTo(stops[at].y - top, 2)}px` : "100%");
      // the mechanism's handle at its height, and its arrow up to the input's (its tip 5px below it)
      railMech.style.translate = `0 ${roundTo(mechY, 2)}px`;
      railMech.classList.toggle("orders-demo-rail-mech-on", Boolean(target?.deep));
      railMechLine.style.translate = `0 ${roundTo(cardY + 6, 2)}px`;
      railMechLine.style.height = `${roundTo(Math.max(0, mechY - 9 - cardY - 6), 2)}px`;
      // its label from its tip down, along it and its handle (longer than it, it does not reach up
      // to the arrow's label); short of room below, it rises to end at the slider's bottom, and short
      // of room between the two, it is left out (as the arrow's label is where the arrow is short)
      const labelEnd = railLabel.hidden ? -Infinity : (stops[1].y + bottom) / 2 + labelLength(railLabel, 0) / 2;
      const mechLabelTop = Math.min(cardY + 5, g.height - labelLength(railMechLabel, 1));
      railMechLabel.classList.toggle("orders-demo-rail-mech-label-off", mechLabelTop < Math.max(labelEnd + 10, cardY - 2));
      railMechLabel.style.top = `${roundTo(mechLabelTop, 2)}px`;
      // the handle's line on to the layer it brought out (or the mechanism's handle's on to the
      // mechanism), at the height of the layer's left corner
      const led = target?.deep ? target : at === 0 ? null : stops[at].item;
      railLead.classList.toggle("orders-demo-rail-lead-on", Boolean(led));
      railLead.classList.toggle("orders-demo-rail-lead-deep", Boolean(led?.deep));
      railLead.classList.toggle("orders-demo-rail-lead-lit", Boolean(lit) && led === target);
      if (led) {
        const [lx, ly] = corners(places[items.indexOf(led)]).reduce((a, b) => (b[0] < a[0] ? b : a));
        const from = RAIL_LINE + 10;
        railLead.style.translate = `${roundTo(from, 2)}px ${roundTo(ly, 2)}px`;
        railLead.style.width = `${roundTo(Math.max(0, lx - from - 4), 2)}px`;
      }

      // the layer pointed at casts its marks onto the card: a blank one, its order's
      const cast = target && (target.sheet || betweenOf(current(), target.order)) ? target : null;
      project(cast, anchors);
    }

    // a blank layer pointed at shows its order's explanation, and its name (where the data has none
    // of its own, nothing); let go of, it is blank again
    function showBlank(item, shown) {
      const sheet = shown ? betweenOf(current(), item.order) : null;
      const order = sheet ? item.order : null;
      if ((item.shows ?? null) === order) return;
      item.shows = order;
      item.node.replaceChildren();
      if (!sheet) return;
      const face = marksOf(current(), sheet).cloneNode(true);
      add(face, "text", { class: "orders-demo-tag", x: 6, y: 95.5 }, `${item.order}-${INDEX}`);
      item.node.append(face);
    }

    // the light from the layer pointed at down to the card (or from the mechanism up to it), whose
    // input then wears the layer's marks (or none)
    const projected = new WeakMap(); // every card's projection: the node of the layer it wears
    function project(item, anchors) {
      const card = items.find((other) => other.base);
      beam.classList.toggle("orders-demo-beam-on", Boolean(item));
      // the light is over the card, under the layers; the mechanism's own rises under the card
      beam.style.zIndex = item?.deep ? 1 : 3;
      // brightest at the layer it comes from: above the card, green, or the mechanism's, orange, below
      // it
      beam.classList.toggle("orders-demo-beam-deep", Boolean(item?.deep));
      card.node.classList.toggle("orders-demo-lit", Boolean(item) && !item.deep);
      glow.setAttribute("y1", item?.deep ? 1 : 0);
      glow.setAttribute("y2", item?.deep ? 0 : 1);
      // (a layer's marks, or a blank layer's order's)
      const source = !item ? null : item.sheet ? item.node.querySelector(".orders-demo-face") : marksOf(current(), betweenOf(current(), item.order));
      if ((projected.get(card.node) ?? null) !== (source ?? null)) {
        const marks = source?.cloneNode(true);
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
      // (from the layer, or the mechanism, to the card)
      const [from, to] = [anchors[items.indexOf(item)], anchors[items.indexOf(card)]].map(corners);
      beamEdges.forEach((edge, k) => edge.setAttribute("d", `M${f(from[k])}L${f(to[k])}`));
    }

    // -- the plot of the fit, upright beside the sheets: every order at the height of its sheet, from
    // order 1 at the top down to the full decomposition, and the fit growing to the right. A sheet's
    // row keeps its line from the sheet's corner, on to its point. It ends at the last explanation:
    // the input and the mechanism below them are no explanations ---------------------------------

    // an SVG element moved to new values over a while, frame by frame, settling as --orders-settle
    // does (its places are attributes, which CSS does not transition); at once where there is no
    // way yet, or with reduced motion. A new move takes over from where the last one is
    const glides = new WeakMap();
    function glide(node, target, ms, apply) {
      const last = glides.get(node);
      cancelAnimationFrame(last?.frame ?? 0);
      if (!last || ms === 0 || reducedMotion()) {
        apply(target);
        glides.set(node, { at: target, frame: 0 });
        return;
      }
      const [from, start, now] = [last.at, performance.now(), { at: last.at, frame: 0 }];
      glides.set(node, now);
      const step = (time) => {
        const k = Math.min(1, (time - start) / ms);
        const eased = 1 - (1 - k) ** 4;
        now.at = from.map((value, i) => value + (target[i] - value) * eased);
        apply(now.at);
        if (k < 1) now.frame = requestAnimationFrame(step);
      };
      now.frame = requestAnimationFrame(step);
    }

    let sideMarks = null;
    function drawSide() {
      sideData.replaceChildren();
      sideMarks = null;
      const example = current();
      const g = geometry;
      side.setAttribute("viewBox", `0 0 ${g.width} ${g.height}`);
      side.setAttribute("width", g.width);
      side.setAttribute("height", g.height);
      // every layer's point at the height of its right corner, where the layers float open
      const rest = layeredLayout(true);
      const points = items
        .filter((item) => !item.base && !item.mechanism)
        .map((item) => ({
          item,
          order: item.order,
          r2: item.sheet ? item.sheet.r2 : item.fit[1],
          y: rightCorner(rest[items.indexOf(item)])[1],
        }));
      // (where the step shows no explanation, no plot)
      side.classList.toggle("orders-demo-side-none", !points.length);
      if (!points.length) {
        side.removeAttribute("aria-label");
        sideFrame.replaceChildren();
        delete sideFrame.dataset.width;
        return;
      }
      side.setAttribute(
        "aria-label",
        `Faithfulness (R²) of the explanations, top to bottom: ${points.map((p) => `order ${p.order}: ${fit2(p.r2)}`).join(", ")}` +
          (estimated(example) ? "; what the full decomposition falls short of 1 is the estimation gap" : ""),
      );
      // R^2 from 0, or from 0.5 when every order fits better, so that the rise shows (every order's,
      // those the step shows or not: every step's axis is the same)
      const fits = [...example.sheets.map((sheet) => sheet.r2), ...between(example).map((fit) => fit[1])];
      const low = Math.min(...fits) >= 0.5 ? 0.5 : 0;
      const [x0, x1] = [g.ladderLeft + (g.narrow ? 6 : 10), g.width - (g.narrow ? 8 : 12)];
      const x = (r2) => x0 + ((Math.max(low, Math.min(1, r2)) - low) / (1 - low)) * (x1 - x0);
      const f = (value) => roundTo(value, 1);
      const [top, bottom] = [points[0].y - 12, points.at(-1).y + 12];

      // the frame, made for the stage's width: three rules, and at the top, a tick at each, the
      // axis's lowest R^2, its middle and 1, and over them its name (over the plot's middle, or, where
      // the plot is narrower than its name, ending at its right). It spans the stack: for the next
      // example's, it stretches or shrinks, gliding there. Its places are the elements' attributes,
      // in the plot's own coordinates, as the data's are: CSS transforms on SVG elements are not drawn
      // alike in every browser (in Safari, the frame drifted off the data)
      const rebuilt = sideFrame.dataset.width !== String(g.width);
      if (rebuilt) {
        sideFrame.dataset.width = g.width;
        sideFrame.replaceChildren();
        const grid = add(sideFrame, "g", { class: "orders-demo-plot-grid" });
        for (let k = 0; k <= 2; k++) {
          const at = f(x0 + (k / 2) * (x1 - x0));
          add(grid, "line", { class: "orders-demo-side-rule", x1: at, x2: at, y1: 0, y2: 0 });
          add(sideFrame, "text", { class: "orders-demo-plot-tick orders-demo-side-level", x: at, y: 0, "text-anchor": "middle" });
        }
        const roomy = x1 - x0 >= 104;
        add(sideFrame, "text", { class: "orders-demo-plot-label orders-demo-side-name", x: f(roomy ? (x0 + x1) / 2 : x1), y: 0, "text-anchor": roomy ? "middle" : "end" }, "faithfulness (R²)");
      }
      const [rules, ticks, name] = [sideFrame.querySelectorAll(".orders-demo-side-rule"), sideFrame.querySelectorAll(".orders-demo-side-level"), sideFrame.querySelector(".orders-demo-side-name")];
      glide(sideFrame, [top, bottom], rebuilt ? 0 : 520, ([from, to]) => {
        for (const rule of rules) {
          rule.setAttribute("y1", f(from));
          rule.setAttribute("y2", f(to));
        }
        for (const tick of ticks) tick.setAttribute("y", f(from - 9));
        name.setAttribute("y", f(from - 25));
      });
      [...ticks].forEach((tick, k) => {
        // (on a narrow screen, without the leading zero: .5, .75)
        const level = low + (k / 2) * (1 - low);
        const text = g.narrow && level > 0 && level < 1 ? String(level).slice(1) : String(level);
        if (tick.textContent !== text) say(tick, text, tick.textContent !== "");
      });

      // a name's width below the plot: measured, or, while the plot is hidden, as its letters make it
      const widthOf = (text) => {
        const probe = add(sideData, "text", { class: "orders-demo-plot-axis orders-demo-plot-gap-label" }, text);
        const width = probe.getComputedTextLength() || text.length * 6.2;
        probe.remove();
        return width;
      };

      // the estimation gap: what an estimated full decomposition falls short of 1, a hatched band
      // down the plot (in gray: no explanation's color, nor the mechanism's), its name below it
      const full = points.at(-1);
      if (full.item.sheet?.full && full.r2 < EXACT) {
        const id = `orders-side-gap-${plotId}`;
        const hatch = add(add(sideData, "defs"), "pattern", { id, width: 5, height: 5, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
        add(hatch, "line", { class: "orders-demo-plot-gap-hatch", x1: 0, y1: 0, x2: 0, y2: 5 });
        const gap = add(sideData, "g", { class: "orders-demo-plot-gap" });
        const band = { x: f(x(full.r2)), y: top, width: f(x(1) - x(full.r2)), height: f(bottom - top) };
        add(gap, "rect", { ...band, class: "orders-demo-plot-gap-fill" });
        add(gap, "rect", { ...band, fill: `url(#${id})` });
        // (under the gap, ending at the plot's right where it would run off it; on a narrow screen,
        // "gap", as the card has it)
        const name = g.narrow ? "gap" : "estimation gap";
        const end = Math.min(band.x + band.width / 2 + widthOf(name) / 2, g.width - 2);
        add(gap, "text", { class: "orders-demo-plot-axis orders-demo-plot-gap-label", x: f(end), y: f(bottom + 11), "text-anchor": "end" }, name);
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
      cover.addEventListener("pointermove", (event) => {
        if (event.pointerType === "mouse" && !state.pinned) go(nearest(event), false);
      });
      // a click, or a tap, pins the order there, and lets it go again (a touch that scrolls the page
      // instead is no tap: it leaves what is pointed at as it was)
      cover.addEventListener("click", (event) => go(nearest(event), true));
      sideMarks = { points, leads, ring, value, x, left: x0 };
      mark();
    }

    // the ring on the plot: at the layer pointed at (a sheet's, or a blank sheet's, hidden at rest),
    // its R^2 left of it (where the line has come from above), or right of it, near the axis;
    // and the line of its row, solid
    function mark() {
      if (!sideMarks) return;
      const pointed = state.active !== null ? sheetItems()[state.active] : null;
      const at = sideMarks.points.find((p) => (pointed ? p.item === pointed : state.blank !== null && !p.item.sheet && p.order === state.blank));
      side.classList.toggle("orders-demo-plot-pointing", Boolean(at));
      for (const [item, lead] of sideMarks.leads) lead.classList.toggle("orders-demo-side-lead-on", item === pointed);
      if (!at) return;
      const [cx, cy] = [roundTo(sideMarks.x(at.r2), 1), roundTo(at.y, 1)];
      // (gliding there, as the frame does: its place is its attribute)
      const ring = sideMarks.ring;
      glide(ring, [cx, cy], 260, ([x, y]) => ring.setAttribute("transform", `translate(${roundTo(x, 1)} ${roundTo(y, 1)})`));
      // (right of it only where it fits on the stage)
      const right = cx - 58 < sideMarks.left && cx + 60 <= geometry.width;
      sideMarks.value.textContent = `R² ${fit2(at.r2)}`;
      sideMarks.value.setAttribute("x", right ? 11 : -11);
      sideMarks.value.setAttribute("text-anchor", right ? "start" : "end");
    }

    // point at a sheet (its index), or at the blank sheet of an order between, or at nothing: at
    // what the step adds, if it is a step
    function point(active, blank) {
      // during a switch, the sheets, rows and plot on screen are still the old example's
      if (root.classList.contains("orders-demo-fading")) return;
      if (active === null && blank === null) active = restActive();
      if (state.active === active && state.blank === blank) return;
      Object.assign(state, { active, blank });
      // a layer pointed at, the invitation goes, for good
      if (active !== null || blank !== null) root.classList.add("orders-demo-hinted");
      place(active);
      mark();
      showPlan();
    }

    function toggle(active, blank) {
      const same = state.active === active && state.blank === blank;
      if (state.pinned && same) {
        state.pinned = false;
        point(null, null);
        // (a step's own, let go of, stays out, no longer pinned: its row says so)
        if (state.active === active && !root.classList.contains("orders-demo-fading")) place(active);
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
    // (a step, pointing at what it adds at rest, has no stop at rest)
    const firstStop = () => (restActive() === null ? 0 : 1);
    const stopAt = (clientY) => {
      const y = clientY - stage.getBoundingClientRect().top;
      const reach = stops.slice(firstStop());
      return reach.reduce((best, stop) => (Math.abs(stop.y - y) < Math.abs(best.y - y) ? stop : best), reach[0]);
    };
    // with a mouse, a press anywhere on the slider goes to the stop there, and drags on; on a touch
    // screen, a drag starts at the handle (demo.css: a finger wide), and anywhere else a touch may
    // scroll the page, so only a tap (a click) goes to the stop there
    let railTap = false;
    // (the pointer that drags the handle, if one does: a touch is captured by what it lands on
    // anyway, so capture alone does not tell a drag from a scroll)
    let dragging = null;
    rail.addEventListener("pointerdown", (event) => {
      railTap = event.pointerType !== "mouse" && event.target !== thumb;
      if (event.button !== 0 || event.target === railMech || railTap) return;
      event.preventDefault();
      rail.setPointerCapture(event.pointerId);
      dragging = event.pointerId;
      rail.classList.add("orders-demo-dragging");
      // the arrow keys go on from there, without the ring that shows keyboard focus
      thumb.focus({ preventScroll: true, focusVisible: false });
      slideTo(stopAt(event.clientY));
    });
    // the mechanism's handle points at the mechanism, as its row does
    const mechanismAt = () => sheetItems().findIndex((item) => item.mechanism);
    railMech.addEventListener("pointerenter", (event) => {
      if (event.pointerType === "mouse" && !state.pinned) point(mechanismAt(), null);
    });
    railMech.addEventListener("click", () => toggle(mechanismAt(), null));
    rail.addEventListener("pointermove", (event) => {
      if (dragging === event.pointerId) slideTo(stopAt(event.clientY));
    });
    for (const type of ["pointerup", "pointercancel"]) {
      rail.addEventListener(type, () => {
        dragging = null;
        rail.classList.remove("orders-demo-dragging");
      });
    }
    rail.addEventListener("click", (event) => {
      if (!railTap || event.target === railMech) return;
      railTap = false;
      slideTo(stopAt(event.clientY));
    });
    thumb.addEventListener("keydown", (event) => {
      const at = Number(thumb.getAttribute("aria-valuenow"));
      const last = stops.length - 1;
      const next = { ArrowDown: at + 1, ArrowRight: at + 1, ArrowUp: at - 1, ArrowLeft: at - 1, Home: 0, End: last }[event.key];
      if (next === undefined) return;
      event.preventDefault();
      slideTo(stops[Math.max(firstStop(), Math.min(last, next))]);
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
      // (the input's card stays as it is)
      if (item?.base) return;
      if (item?.sheet) point(item.index, null);
      // a blank sheet peeks out, its strip staying under the pointer
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
        // (on a page zoomed in, a drag sideways pans the page)
        const zoomed = (window.visualViewport?.scale ?? 1) > 1.01;
        const free =
          data.examples.length > 1 &&
          event.touches.length === 1 &&
          !zoomed &&
          !event.target.closest(".orders-demo-rail") &&
          !root.classList.contains("orders-demo-fading");
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
      if (!unfolded()) {
        // a tap on the layers at rest unfolds them, for good, none pointed at
        unfoldedForGood = true;
        place(state.active);
        return;
      }
      const item = itemAt(pointer(event));
      if (item === null || item.base) {
        // a tap on the input's card, or beside the layers, lets go of the one pointed at
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

    // the card beside the layers (demo.css): wide, the full stack, explained, unless data-plan="above"
    function wideEnough() {
      const wide = root.clientWidth >= 640;
      root.classList.toggle("orders-demo-wide", wide);
      root.classList.toggle("orders-demo-beside", wide && whole && revealed && root.dataset.plan !== "above");
    }
    // the stage's width, and the window's height (both hold the layers), and beside the card, the
    // demo's width, whose columns are fit to it, laid out anew
    let width = 0;
    let tall = 0;
    let demoWidth = 0;
    function relayout() {
      // (a demo still drawn for the example shown before is drawn anew first)
      if (stale) {
        stale = false;
        render();
      }
      wideEnough();
      const besides = root.classList.contains("orders-demo-beside");
      if (stage.clientWidth !== width || viewportHeight() !== tall || (besides && root.clientWidth !== demoWidth)) {
        [width, tall, demoWidth] = [stage.clientWidth, viewportHeight(), root.clientWidth];
        geometry = measure();
        place(state.active);
        drawSide();
      }
      settleCard();
      castLightOut();
    }
    watch(new ResizeObserver(relayout)).observe(root);
    window.addEventListener("resize", relayout, { signal: life });
    // (and anew when the page sets data-plan)
    watch(new MutationObserver(relayout)).observe(root, { attributes: true, attributeFilter: ["data-plan"] });
    // (and the chip's ink anew when the page changes its theme)
    watch(new MutationObserver(() => inkChip())).observe(document.body, { attributes: true, attributeFilter: ["data-theme"] });
    darkScheme.addEventListener?.("change", () => inkChip(), { signal: life });
    watch(new ResizeObserver(() => movePill(true))).observe(switcher);
    // the light out of the chip follows the plan's layout (its width, its fonts)
    watch(
      new ResizeObserver(() => {
        settleCard();
        castLightOut();
        for (const node of [xShows, chipShows, outShows]) fitPart(node);
      }),
    ).observe(plan);
    document.fonts?.ready.then(() => {
      if (life?.aborted) return;
      [xShows, chipShows, outShows].forEach(fitPart);
      settled = ""; // measured anew in the fonts loaded
      labelLengths = [0, 0];
      if (geometry !== null) place(state.active);
      settleCard();
      castLightOut();
    });
    // the example chosen in another demo of the page, or by the page
    // (out of view, its endless animations rest, demo.css: orders-demo-away; and an example chosen
    // elsewhere meanwhile is drawn once it comes within a screen of the view)
    let inView = false;
    let stale = false;
    watch(
      new IntersectionObserver(([entry]) => {
        inView = entry.isIntersecting;
        root.classList.toggle("orders-demo-away", !inView);
      }),
    ).observe(root);
    watch(
      new IntersectionObserver(
        ([entry]) => {
          if (!entry.isIntersecting || !stale) return;
          stale = false;
          render();
        },
        { rootMargin: "100% 0px" },
      ),
    ).observe(root);
    document.addEventListener(
      EXAMPLE_EVENT,
      (event) => {
        const example = data.every.find((other) => other.id === event.detail?.example);
        if (example) choose(example, 0, false);
      },
      { signal: life },
    );

    // the first example fades in, as every next one
    wideEnough();
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
  // (none of the layout's classes and sizes stays: the figure shows as a figure)
  const showFallback = (root) => {
    root.className = "orders-demo orders-demo-failed";
    root.removeAttribute("style");
    root.innerHTML = fallbacks.get(root);
  };
  // a demo that cannot start stops listening to the page, the window and its observers (its life
  // ends: start() ties them to it)
  // (every demo in a task of its own: the page responds, and paints, between them)
  const begin = () => {
    for (const root of roots) {
      setTimeout(() => {
        const life = new AbortController();
        try {
          start(root, window.SHAPIQ_ORDERS, life.signal);
        } catch (error) {
          console.error("The explanation of every order could not start:", error);
          life.abort();
          showFallback(root);
        }
      });
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
