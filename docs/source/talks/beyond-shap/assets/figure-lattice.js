/* The lattices of the talk (talks/beyond-shap/index.rst): lattice-sv and lattice-sii.

   Both draw the lattice of all 16 coalitions of four players of the giraffe of the FIxLIP demo
   (SigLIP 2 matching the photo demos/fixlip/assets/giraffe-tree.jpg with its caption): two patches
   of the photo, the giraffe's head and the dead tree, and two words of the caption, "giraffe" and
   "water". The empty coalition lies at the bottom, the full one at the top, one row per size. A
   node shows its value ν(T) and which players are present, a patch and a word in each row.

   lattice-sv   Step 0 the lattice. Step 1 the marginal contributions on the edges: an edge adds
                one player, and pointing at it writes out Δ_i(T) = ν(T ∪ {i}) − ν(T). Step 2 the
                Shapley value of a player (the giraffe's head first, or the player pointed at in the
                legend): its 8 edges, weighted, and its value, with bars of all four.
   lattice-sii  Step 0 the lattice with the pair S of the giraffe's head and the word "giraffe".
                Step 1 the square the pair spans at each coalition T of the other two players, and
                its discrete derivative Δ_S(T). Step 2 the Shapley interaction, the weighted mean of
                the four squares, with the Shapley values of all four in the legend. A click on a player in the
                legend makes another pair.

   The nodes stand where the squares of the pair are faces of their own: a coalition is T ∪ L, with
   T of the other two players and L within the pair, and lies at x(T) + x(L). The squares of
   |T| = 0 and |T| = 2 lie in the middle, below and above each other, those of |T| = 1 left and
   right. Both figures use the layout of the pair {head, "giraffe"}, so that the lattice of the two
   slides is the same. Everything is computed here from the game's 16 whole numbers. */
(function () {
  "use strict";
  const figures = (window.SHAPIQ_TALK_FIGURES = window.SHAPIQ_TALK_FIGURES || {});

  // The game ν(T) of the four players, by bit mask (player i is bit i): the orders demo's full
  // decomposition of SigLIP 2 for the giraffe (the Möbius transform of its ProxySHAP proxy, order
  // 10 of demos/orders/compute/results/fixlip_giraffe-tree.json) restricted to the four players,
  // the other 79 players (patches and tokens) absent. ν(T) is the sum of the Möbius values of all
  // groups within T: the proxy's cosine similarity, relative to the empty coalition, of the photo
  // with only T's patches and the caption with only T's words. Three weak groups that the results
  // do not keep count as 0. The values are rescaled to thousandths of the cosine similarity and
  // rounded to whole numbers, which keeps their signs. compute/lattice.py writes this block.
  // LATTICE (compute/lattice.py)
  const LATTICE = {"players":[{"kind":"patch","patch":44,"name":"giraffe’s head"},{"kind":"patch","patch":17,"name":"dead tree"},{"kind":"word","word":"giraffe"},{"kind":"word","word":"water"}],"grid":8,"values":[0,-3,0,-2,14,23,21,26,17,19,17,20,25,52,37,55]};
  // /LATTICE

  const N = 4;
  const FULL = (1 << N) - 1;
  const PLAYERS = [0, 1, 2, 3];
  const PAIR = [0, 2]; // the giraffe's head and the word "giraffe"
  const nu = (mask) => LATTICE.values[mask];
  const size = (mask) => PLAYERS.filter((p) => (mask >> p) & 1).length;
  const binom = (n, k) => {
    let r = 1;
    for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
    return Math.round(r);
  };
  const subsets = (mask) => {
    const out = [];
    for (let t = 0; t <= FULL; t++) if ((t & mask) === t) out.push(t);
    return out;
  };
  const MINUS = "−";
  const number = (value, { signed = false, digits = 0 } = {}) => {
    const text = Math.abs(value).toFixed(digits);
    if (Number(text) === 0) return text;
    return (value < 0 ? MINUS : signed ? "+" : "") + text;
  };

  // the marginal contributions, the Shapley values, the discrete derivatives and the Shapley
  // interactions, exactly from the whole numbers above
  const delta = (T, i) => nu(T | (1 << i)) - nu(T);
  const svWeight = (t) => [1, N * binom(N - 1, t)]; // as a fraction: 1 / (d C(d-1, |T|))
  const shapley = PLAYERS.map((i) =>
    subsets(FULL & ~(1 << i)).reduce((sum, T) => sum + delta(T, i) / svWeight(size(T))[1], 0),
  );
  const derivative = (T, [a, b]) => nu(T | (1 << a) | (1 << b)) - nu(T | (1 << a)) - nu(T | (1 << b)) + nu(T);
  const siiWeight = (t) => [1, (N - 2 + 1) * binom(N - 2, t)]; // 1 / ((d-s+1) C(d-s, |T|))
  const interaction = (pair) => {
    const rest = FULL & ~(1 << pair[0]) & ~(1 << pair[1]);
    return subsets(rest).reduce((sum, T) => sum + derivative(T, pair) / siiWeight(size(T))[1], 0);
  };
  const EDGES = [];
  for (let T = 0; T <= FULL; T++) for (const i of PLAYERS) if (!((T >> i) & 1)) EDGES.push({ T, i, key: `${T}:${i}` });
  const MAX_DELTA = Math.max(...EDGES.map(({ T, i }) => Math.abs(delta(T, i))));
  const MAX_VALUE = Math.max(...LATTICE.values.map(Math.abs));

  // ---- geometry (SVG units; the SVG scales with the figure's width) ---------------------------
  const VW = 600, UX = 100, CX = 300, NW = 80, NH = 56, UY = 96, PAD = 6;
  const VH = PAD * 2 + NH + 4 * UY;
  const X = (x) => CX + x * UX;
  const Y = (level) => PAD + NH / 2 + (N - level) * UY;

  // where each coalition lies, for a pair (a, b) and the other two players (c, d)
  function layout(pair) {
    const [a, b] = [...pair].sort();
    const [c, d] = PLAYERS.filter((p) => !pair.includes(p));
    const A = 1 << a, B = 1 << b, C = 1 << c, Dm = 1 << d;
    const xT = [[0, 0], [C, -2], [Dm, 2], [C | Dm, 0]];
    const xL = [[0, 0], [A, -0.5], [B, 0.5], [A | B, 0]];
    const at = [];
    for (const [T, xt] of xT) {
      for (const [L, xl] of xL) {
        let x = xt + xl;
        if (T === 0 && L === (A | B)) x = -0.5; // the top of the lower middle square
        if (T === (C | Dm) && L === 0) x = 0.5; // the bottom of the upper middle square
        at[T | L] = { x: X(x), y: Y(size(T | L)) };
      }
    }
    // where each square's discrete derivative is written: beside its node T, outwards
    const badge = new Map([
      [0, { x: X(0) + NW / 2 + 6, y: Y(0), anchor: "start" }],
      [C, { x: X(-2) - NW / 2 - 6, y: Y(1), anchor: "end" }],
      [Dm, { x: X(2) + NW / 2 + 6, y: Y(1), anchor: "start" }],
      [C | Dm, { x: X(0) + NW / 2 + 6, y: Y(4), anchor: "start" }],
    ]);
    return { at, badge, pair: [a, b], rest: C | Dm };
  }

  function lattice(figure, { make }, kind) {
    const sii = kind === "sii";
    const photo = `${figure.dataset.root || ""}../../demos/fixlip/giraffe-tree.jpg`;
    const grid = LATTICE.grid;
    const root = make("div", { class: `tl tl-${kind}` }, figure);
    const players = LATTICE.players;
    const label = (p) => (players[p].kind === "word" ? `“${players[p].word}”` : `the patch of the ${players[p].name}`);

    // ---- glyphs of the players, in HTML ----
    function chip(p, parent, extra = "") {
      const player = players[p];
      if (player.kind === "patch") {
        const row = Math.floor(player.patch / grid), col = player.patch % grid;
        const el = make("span", { class: `tl-chip tl-chip-patch ${extra}`, role: "img", "aria-label": player.name }, parent);
        el.style.backgroundImage = `url("${photo}")`;
        el.style.backgroundSize = `${grid * 100}% ${grid * 100}%`;
        el.style.backgroundPosition = `${(col / (grid - 1)) * 100}% ${(row / (grid - 1)) * 100}%`;
        return el;
      }
      return make("span", { class: `tl-chip tl-chip-word ${extra}` }, parent, player.word);
    }

    // ---- the legend: where the patches lie, and the four players (with their Shapley values) ----
    const legend = make("div", { class: "tl-legend" }, root);
    const thumb = make("div", { class: "tl-photo", "aria-hidden": "true" }, legend);
    thumb.style.backgroundImage = `url("${photo}")`;
    const marks = [];
    players.forEach((player, p) => {
      if (player.kind !== "patch") return;
      const mark = make("span", { class: "tl-photo-mark" }, thumb);
      mark.style.left = `${((player.patch % grid) / grid) * 100}%`;
      mark.style.top = `${(Math.floor(player.patch / grid) / grid) * 100}%`;
      mark.style.width = mark.style.height = `${100 / grid}%`;
      marks[p] = mark;
    });
    const maxShapley = Math.max(...shapley.map(Math.abs));
    const items = PLAYERS.map((p) => {
      const item = make("button", { type: "button", class: "tl-player", "aria-label": label(p) }, legend);
      chip(p, item);
      // the Shapley values of all four, from step 2 (on both slides, so that their legends match)
      const score = make("span", { class: "tl-score talk-from-2" }, item);
      const track = make("span", { class: "tl-bar-track" }, score);
      const bar = make("span", { class: `tl-bar ${shapley[p] >= 0 ? "tl-bar-pos" : "tl-bar-neg"}` }, track);
      bar.style.width = `${(Math.abs(shapley[p]) / maxShapley) * 100}%`;
      make("span", { class: "tl-bar-value" }, score, number(shapley[p], { digits: 2 }));
      item.setAttribute("aria-label", `${label(p)}, Shapley value ${number(shapley[p], { digits: 2 })}`);
      return item;
    });

    // ---- the lattice ----
    const svg = make("svg:svg", { class: "tl-svg", viewBox: `0 0 ${VW} ${VH}`, role: "group" }, root);
    const squareLayer = make("svg:g", { class: "tl-squares" }, svg);
    const edgeLayer = make("svg:g", { class: "tl-edges" }, svg);
    const labelLayer = make("svg:g", { class: "tl-labels", "aria-hidden": "true" }, svg);
    const nodeLayer = make("svg:g", { class: "tl-nodes" }, svg);
    const badgeLayer = make("svg:g", { class: "tl-badges", "aria-hidden": "true" }, svg);

    const describe = (mask) => {
      const names = PLAYERS.filter((p) => (mask >> p) & 1).map((p) => (players[p].kind === "word" ? players[p].word : players[p].name));
      return names.length ? names.join(", ") : "no player";
    };

    // a node: its value above, then the players, a patch and a word in each row
    const nodes = [];
    for (let mask = 0; mask <= FULL; mask++) {
      const value = nu(mask);
      const g = make("svg:g", { class: "tl-node", "data-mask": mask, "aria-label": `ν of ${describe(mask)} is ${number(value)}` }, nodeLayer);
      const box = make("svg:rect", { class: `tl-node-box ${value < 0 ? "tl-neg" : "tl-pos"}`, x: -NW / 2, y: -NH / 2, width: NW, height: NH, rx: 9 }, g);
      box.style.setProperty("--level", Math.abs(value) / MAX_VALUE);
      make("svg:text", { class: "tl-node-value", x: 0, y: -12, "text-anchor": "middle" }, g, number(value));
      PLAYERS.forEach((p) => {
        const on = (mask >> p) & 1;
        const player = players[p];
        const slot = make("svg:g", { class: `tl-slot ${on ? "tl-on" : "tl-off"}`, "data-player": p }, g);
        // two rows: a patch and a word in each (the giraffe's head and "giraffe" above)
        const row = PLAYERS.filter((q) => players[q].kind === player.kind).indexOf(p);
        const y = row ? 10 : -7;
        if (player.kind === "patch") {
          const pic = make("svg:svg", { x: -35, y, width: 15, height: 15, viewBox: `${player.patch % grid} ${Math.floor(player.patch / grid)} 1 1`, preserveAspectRatio: "none", class: "tl-slot-pic" }, slot);
          make("svg:image", { href: photo, x: 0, y: 0, width: grid, height: grid, preserveAspectRatio: "none" }, pic);
          make("svg:rect", { class: "tl-slot-frame", x: -35.5, y: y - 0.5, width: 16, height: 16, rx: 3 }, slot);
        } else {
          make("svg:rect", { class: "tl-slot-frame tl-slot-pill", x: -16, y, width: 52, height: 15, rx: 7.5 }, slot);
          make("svg:text", { class: "tl-slot-word", x: 10, y: y + 11, "text-anchor": "middle" }, slot, player.word);
        }
      });
      nodes[mask] = g;
    }

    // an edge adds player i to T; its line, a wide target to point at, and its label
    const edges = new Map();
    for (const edge of EDGES) {
      const value = delta(edge.T, edge.i);
      const g = make("svg:g", { class: `tl-edge ${value < 0 ? "tl-edge-neg" : "tl-edge-pos"}`, "data-key": edge.key }, edgeLayer);
      const glow = make("svg:line", { class: "tl-edge-glow" }, g);
      const line = make("svg:line", { class: "tl-edge-line" }, g);
      line.style.setProperty("--weight", Math.abs(value) / MAX_DELTA);
      const hit = make("svg:line", { class: "tl-edge-hit", tabindex: -1, role: "button", "aria-label": `Adding ${label(edge.i)} to ${describe(edge.T)}: marginal contribution ${number(value, { signed: true })}` }, g);
      const tag = make("svg:g", { class: "tl-edge-label" }, labelLayer);
      const text = number(value, { signed: true });
      const w = 9 + text.length * 7;
      make("svg:rect", { x: -w / 2, y: -9, width: w, height: 18, rx: 9 }, tag);
      make("svg:text", { x: 0, y: 4.2, "text-anchor": "middle" }, tag, text);
      edges.set(edge.key, { ...edge, value, g, glow, line, hit, tag });
    }

    // tags on nodes: T and T ∪ i on an edge, the signs of a square's corners
    const tags = nodes.map((node) => {
      const pair = {};
      for (const side of ["left", "right"]) {
        const tag = make("svg:g", { class: "tl-tag" }, node);
        make("svg:rect", { y: -9, height: 18, rx: 9 }, tag);
        make("svg:text", { x: 0, y: 4.5, "text-anchor": "middle" }, tag);
        tag.setAttribute("transform", `translate(${side === "left" ? -NW / 2 + 4 : NW / 2 - 4}, ${-NH / 2})`);
        pair[side] = tag;
      }
      return pair;
    });
    function setTag(mask, text, side = "left") {
      const tag = tags[mask][side];
      const w = 10 + [...text].length * 6.4;
      tag.querySelector("rect").setAttribute("x", -w / 2);
      tag.querySelector("rect").setAttribute("width", w);
      tag.querySelector("text").textContent = text;
      tag.classList.add("tl-tag-on");
    }

    // the squares of the pair (lattice-sii): one face per coalition T of the other two players
    const squares = new Map();
    if (sii) {
      for (let T = 0; T <= FULL; T++) {
        const g = make("svg:g", { class: "tl-square", "data-t": T }, squareLayer);
        const face = make("svg:polygon", { class: "tl-square-face" }, g);
        const hit = make("svg:polygon", { class: "tl-square-hit", tabindex: -1, role: "button" }, g);
        const badge = make("svg:g", { class: "tl-square-badge" }, badgeLayer);
        make("svg:rect", { y: -10, height: 20, rx: 10 }, badge);
        make("svg:text", { x: 0, y: 4.5 }, badge);
        squares.set(T, { T, g, face, hit, badge });
      }
    }

    // ---- placing everything, for a pair (animated when the pair changes) ----
    let geometry = layout(PAIR);
    let shown = geometry.at.map((p) => ({ ...p }));
    function place(at) {
      shown = at;
      at.forEach((p, mask) => nodes[mask].setAttribute("transform", `translate(${p.x.toFixed(2)}, ${p.y.toFixed(2)})`));
      for (const edge of edges.values()) {
        const p = at[edge.T], q = at[edge.T | (1 << edge.i)];
        for (const line of [edge.glow, edge.line, edge.hit]) {
          line.setAttribute("x1", p.x.toFixed(2));
          line.setAttribute("y1", p.y.toFixed(2));
          line.setAttribute("x2", q.x.toFixed(2));
          line.setAttribute("y2", q.y.toFixed(2));
        }
        edge.tag.setAttribute("transform", `translate(${((p.x + q.x) / 2).toFixed(2)}, ${((p.y + q.y) / 2).toFixed(2)})`);
      }
    }
    function placeSquares() {
      if (!sii) return;
      const [a, b] = geometry.pair;
      const A = 1 << a, B = 1 << b;
      for (const slot of svg.querySelectorAll(".tl-slot")) slot.classList.toggle("tl-pair", geometry.pair.includes(Number(slot.dataset.player)));
      for (const square of squares.values()) {
        const inRest = (square.T & ~geometry.rest) === 0;
        square.g.classList.toggle("tl-square-none", !inRest);
        square.badge.classList.toggle("tl-square-none", !inRest);
        square.hit.setAttribute("tabindex", -1);
        if (!inRest) continue;
        const T = square.T;
        const corners = [T, T | A, T | A | B, T | B].map((m) => geometry.at[m]);
        const points = corners.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
        square.face.setAttribute("points", points);
        square.hit.setAttribute("points", points);
        const value = derivative(T, geometry.pair);
        square.value = value;
        square.face.classList.toggle("tl-neg", value < 0);
        square.face.style.setProperty("--weight", Math.min(1, Math.abs(value) / 25));
        const spot = geometry.badge.get(T);
        const text = number(value, { signed: true });
        const w = 12 + text.length * 7.5;
        square.badge.setAttribute("transform", `translate(${spot.x}, ${spot.y})`);
        square.badge.querySelector("rect").setAttribute("x", spot.anchor === "start" ? 0 : -w);
        square.badge.querySelector("rect").setAttribute("width", w);
        const label = square.badge.querySelector("text");
        label.setAttribute("x", spot.anchor === "start" ? w / 2 : -w / 2);
        label.setAttribute("text-anchor", "middle");
        label.textContent = text;
        square.hit.setAttribute("aria-label", `The square at ${describe(T)}: discrete derivative ${text}`);
      }
    }
    let animation = 0;
    function relayout(pair) {
      const from = shown.map((p) => ({ ...p }));
      geometry = layout(pair);
      const to = geometry.at;
      placeSquares();
      cancelAnimationFrame(animation);
      const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const start = performance.now();
      const duration = reduce ? 0 : 520;
      const frame = (now) => {
        const t = duration ? Math.min(1, (now - start) / duration) : 1;
        const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
        place(from.map((p, m) => ({ x: p.x + (to[m].x - p.x) * e, y: p.y + (to[m].y - p.y) * e })));
        if (t < 1) animation = requestAnimationFrame(frame);
      };
      animation = requestAnimationFrame(frame);
    }
    place(geometry.at);
    placeSquares();

    // ---- the formula below the lattice, in the style of the deck's definitions ----
    const box = make("div", { class: "shapiq-talk-definition tl-formula", "aria-live": "polite" }, root);
    const rubric = make("p", { class: "rubric tl-rubric" }, box);
    const title = make("span", { class: "tl-title" }, rubric);
    const aside = make("span", { class: "tl-aside" }, rubric);
    const eq = make("div", { class: "tl-eq" }, box);
    const lines = [make("div", { class: "tl-line" }, eq), make("div", { class: "tl-line" }, eq)];

    // a little language for the formulas: strings, variables, subscripts, numbers, glyphs
    const V = (t) => ({ v: t });
    const SUB = (...parts) => ({ sub: parts });
    const NUM = (value, options, key) => ({ num: number(value, options), key });
    const CHIP = (p) => ({ chip: p });
    const FRAC = (a, b) => ({ frac: [a, b] });
    const SET = (mask) => ({ set: mask });
    const GAP = { gap: true };
    let previous = [];
    function write(line, parts) {
      line.replaceChildren();
      const render = (part, parent) => {
        if (part == null) return;
        if (typeof part === "string") parent.append(part);
        else if (Array.isArray(part)) part.forEach((p) => render(p, parent));
        else if (part.v) make("i", { class: "tl-var" }, parent, part.v);
        else if (part.gap) make("span", { class: "tl-gap" }, parent);
        else if (part.sub) render(part.sub, make("sub", { class: "tl-sub" }, parent));
        else if (part.num != null) {
          const el = make("span", { class: "tl-num" }, parent, part.num);
          if (part.key != null) el.dataset.key = part.key;
        } else if (part.chip != null) chip(part.chip, parent, "tl-chip-inline");
        else if (part.frac) {
          const f = make("span", { class: "tl-frac" }, parent);
          make("span", {}, f, String(part.frac[0]));
          make("span", {}, f, String(part.frac[1]));
        } else if (part.set != null) {
          const members = PLAYERS.filter((p) => (part.set >> p) & 1);
          if (!members.length) parent.append("∅");
          else {
            parent.append("{");
            members.forEach((p, k) => {
              if (k) parent.append(", ");
              chip(p, parent, "tl-chip-inline");
            });
            parent.append("}");
          }
        } else if (part.cls) render(part.parts, make("span", { class: part.cls }, parent));
      };
      render(parts, line);
    }
    function formula(name, side, first, second) {
      title.textContent = name;
      aside.replaceChildren();
      if (side) write(aside, side);
      write(lines[0], first);
      write(lines[1], second);
      // numbers that changed slide in
      const nums = [...eq.querySelectorAll(".tl-num")].concat([...aside.querySelectorAll(".tl-num")]);
      const now = nums.map((el) => el.textContent);
      nums.forEach((el, k) => {
        if (previous[k] !== now[k]) el.classList.add("tl-fresh");
      });
      previous = now;
    }
    const nuOf = (mask) => [V("ν"), "(", SET(mask), ")"];
    const signedTerm = (value, first, key) => (first ? NUM(value, {}, key) : [value < 0 ? ` ${MINUS} ` : " + ", NUM(Math.abs(value), {}, key)]);
    const paren = (value) => (value < 0 ? ["(", NUM(value), ")"] : NUM(value));

    // ---- states ----
    let step = 0;
    let rest = null; // what the figure shows while nothing is pointed at
    let selected = 0; // lattice-sv: the player of step 2
    let order = [...PAIR]; // lattice-sii: the pair, the newest last

    function clear() {
      for (const node of nodes) node.classList.remove("tl-hot", "tl-dim");
      for (const tag of tags) for (const side of ["left", "right"]) tag[side].classList.remove("tl-tag-on");
      for (const edge of edges.values()) edge.g.classList.remove("tl-hot", "tl-mine", "tl-dim");
      for (const edge of edges.values()) edge.tag.classList.remove("tl-on", "tl-hot");
      for (const square of squares.values()) {
        square.g.classList.remove("tl-hot", "tl-dim");
        square.badge.classList.remove("tl-hot");
      }
      items.forEach((item) => item.classList.remove("tl-hot"));
      marks.forEach((mark) => mark && mark.classList.remove("tl-hot"));
    }
    const hotPlayer = (p) => {
      items[p].classList.add("tl-hot");
      if (marks[p]) marks[p].classList.add("tl-hot");
    };

    function showNode(mask) {
      clear();
      nodes[mask].classList.add("tl-hot");
      PLAYERS.forEach((p) => (mask >> p) & 1 && hotPlayer(p));
      formula("Value of a Coalition", null, [nuOf(mask), " = ", NUM(nu(mask), {}, "v")], [
        V("ν"), "(∅) = ", NUM(nu(0)), GAP, V("ν"), "(", V("D"), ") = ", NUM(nu(FULL)),
      ]);
    }

    function showEdge({ T, i }) {
      clear();
      const edge = edges.get(`${T}:${i}`);
      edge.g.classList.add("tl-hot");
      edge.tag.classList.add("tl-on", "tl-hot");
      const up = T | (1 << i);
      nodes[T].classList.add("tl-hot");
      nodes[up].classList.add("tl-hot");
      setTag(T, "T");
      setTag(up, "T ∪ i");
      hotPlayer(i);
      formula("Marginal Contribution", null, [
        "Δ", SUB(CHIP(i)), "(", V("T"), ") = ", V("ν"), "(", V("T"), " ∪ {", CHIP(i), "}) ", MINUS, " ", V("ν"), "(", V("T"), ")",
      ], ["= ", NUM(nu(up)), ` ${MINUS} `, paren(nu(T)), " = ", NUM(edge.value, { signed: true })]);
    }

    function showShapley(i, focus = null) {
      clear();
      hotPlayer(i);
      for (const edge of edges.values()) {
        const mine = edge.i === i;
        edge.g.classList.add(mine ? "tl-mine" : "tl-dim");
        edge.tag.classList.toggle("tl-on", mine);
        edge.hit.setAttribute("tabindex", mine && step === 2 ? 0 : -1);
      }
      if (focus) {
        const edge = edges.get(focus);
        edge.g.classList.add("tl-hot");
        edge.tag.classList.add("tl-hot");
        nodes[edge.T].classList.add("tl-hot");
        nodes[edge.T | (1 << i)].classList.add("tl-hot");
      }
      // the 8 edges, grouped by their weights: 1/4 for |T| = 0 and 3, 1/12 for |T| = 1 and 2
      const mine = [...edges.values()].filter((e) => e.i === i).sort((a, b) => size(a.T) - size(b.T) || shown[a.T].x - shown[b.T].x);
      const groups = new Map();
      for (const edge of mine) {
        const w = svWeight(size(edge.T))[1];
        if (!groups.has(w)) groups.set(w, []);
        groups.get(w).push(edge);
      }
      const sum = [];
      [...groups.entries()].sort((a, b) => a[0] - b[0]).forEach(([w, list], k) => {
        if (k) sum.push(" + ");
        sum.push(FRAC(1, w), " · (");
        list.forEach((edge, j) => {
          const term = signedTerm(edge.value, j === 0, edge.key);
          sum.push(focus === edge.key ? { cls: "tl-term-hot", parts: term } : term);
        });
        sum.push(")");
      });
      const total = shapley.reduce((a, b) => a + b, 0);
      formula("Shapley Value", [
        "Σ ", V("φ"), " = ", NUM(total, { digits: 0 }), " = ", V("ν"), "(", V("D"), ") ", MINUS, " ", V("ν"), "(∅)",
      ], [V("φ"), SUB(CHIP(i)), " = ", sum], ["= ", NUM(shapley[i], { digits: 2 }, "phi")]);
    }

    function showSquare(T) {
      clear();
      const pair = geometry.pair;
      const [a, b] = pair;
      const A = 1 << a, B = 1 << b;
      for (const square of squares.values()) square.g.classList.add(square.T === T ? "tl-hot" : "tl-dim");
      squares.get(T).badge.classList.add("tl-hot");
      [[T, "+"], [T | A, MINUS], [T | B, MINUS], [T | A | B, "+"]].forEach(([m, sign]) => {
        nodes[m].classList.add("tl-hot");
        setTag(m, sign, "right");
      });
      setTag(T, "T");
      pair.forEach(hotPlayer);
      const value = derivative(T, pair);
      formula("Discrete Derivative", null, [
        "Δ", SUB(V("S")), "(", V("T"), ") = ", V("ν"), "(", V("T"), " ∪ ", V("S"), ") ", MINUS, " ", V("ν"), "(", V("T"), " ∪ ", CHIP(a), ") ",
        MINUS, " ", V("ν"), "(", V("T"), " ∪ ", CHIP(b), ") + ", V("ν"), "(", V("T"), ")",
      ], [
        "= ", NUM(nu(T | A | B)), ` ${MINUS} `, paren(nu(T | A)), ` ${MINUS} `, paren(nu(T | B)), " + ", paren(nu(T)), " = ",
        NUM(value, { signed: true }),
      ]);
    }

    function showInteraction(focus = null) {
      clear();
      const pair = geometry.pair;
      pair.forEach(hotPlayer);
      for (const square of squares.values()) {
        if (focus != null) square.g.classList.add(square.T === focus ? "tl-hot" : "tl-dim");
      }
      if (focus != null) squares.get(focus).badge.classList.add("tl-hot");
      const list = [...squares.values()].filter((s) => (s.T & ~geometry.rest) === 0);
      const groups = new Map();
      for (const square of list.sort((x, y) => size(x.T) - size(y.T) || x.T - y.T)) {
        const w = siiWeight(size(square.T))[1];
        if (!groups.has(w)) groups.set(w, []);
        groups.get(w).push(square);
      }
      const sum = [];
      [...groups.entries()].sort((x, y) => x[0] - y[0]).forEach(([w, group], k) => {
        if (k) sum.push(" + ");
        sum.push(FRAC(1, w), " · (");
        group.forEach((square, j) => {
          const term = signedTerm(square.value, j === 0, square.T);
          sum.push(focus === square.T ? { cls: "tl-term-hot", parts: term } : term);
        });
        sum.push(")");
      });
      const value = interaction(pair);
      formula("Shapley Interaction", null,
        [V("φ"), SUB(V("S")), " = ", sum], ["= ", NUM(value, { digits: 2 }, "sii")]);
    }

    function showPair() {
      clear();
      const [a, b] = geometry.pair;
      const S = (1 << a) | (1 << b);
      nodes[S].classList.add("tl-hot");
      setTag(S, "S");
      geometry.pair.forEach(hotPlayer);
      formula("A Pair of Players", null, [V("S"), " = ", SET(S)], [V("ν"), "(", V("S"), ") = ", NUM(nu(S), {}, "v")]);
    }

    function render(state) {
      root.classList.toggle("tl-pointing", state !== rest);
      if (!state) return;
      if (state.node != null) showNode(state.node);
      else if (state.edge) showEdge(state.edge);
      else if (state.player != null) showShapley(state.player, state.focus);
      else if (state.pair) showPair();
      else if (state.square != null) showSquare(state.square);
      else if (state.interaction) showInteraction(state.focus);
    }
    const point = (state) => render(state);
    const leave = () => render(rest);

    // ---- what can be pointed at, in each step ----
    // a click does not take the focus, so that the deck's arrow keys go on working after it (the
    // keyboard still reaches everything with Tab)
    const keepFocus = (el) => el.addEventListener("mousedown", (event) => event.preventDefault());
    nodes.forEach((node, mask) => {
      const on = () => step === 0 && point({ node: mask });
      node.addEventListener("pointerenter", on);
      node.addEventListener("pointerleave", () => step === 0 && leave());
    });
    for (const edge of edges.values()) {
      const on = () => {
        if (sii) return;
        if (step === 1) point({ edge });
        else if (step === 2 && edge.i === selected) point({ player: selected, focus: edge.key });
      };
      const pin = () => {
        if (sii) return;
        if (step === 1) rest = { edge };
        else if (step === 2 && edge.i === selected) rest = { player: selected, focus: edge.key };
        render(rest);
      };
      keepFocus(edge.hit);
      edge.hit.addEventListener("pointerenter", on);
      edge.hit.addEventListener("focus", on);
      edge.hit.addEventListener("pointerleave", leave);
      edge.hit.addEventListener("blur", leave);
      edge.hit.addEventListener("click", pin);
      edge.hit.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          pin();
        }
      });
    }
    for (const square of squares.values()) {
      const on = () => {
        if (step === 1) point({ square: square.T });
        else if (step === 2) point({ interaction: true, focus: square.T });
      };
      const pin = () => {
        if (step === 1) rest = { square: square.T };
        else if (step === 2) rest = { interaction: true, focus: square.T };
        render(rest);
      };
      keepFocus(square.hit);
      square.hit.addEventListener("pointerenter", on);
      square.hit.addEventListener("focus", on);
      square.hit.addEventListener("pointerleave", leave);
      square.hit.addEventListener("blur", leave);
      square.hit.addEventListener("click", pin);
      square.hit.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          pin();
        }
      });
    }
    items.forEach((item, p) => {
      keepFocus(item);
      if (!sii) {
        const on = () => step === 2 && point({ player: p });
        item.addEventListener("pointerenter", on);
        item.addEventListener("focus", on);
        item.addEventListener("pointerleave", () => step === 2 && leave());
        item.addEventListener("blur", () => step === 2 && leave());
        item.addEventListener("click", () => {
          if (step !== 2) return;
          selected = p;
          rest = { player: p };
          render(rest);
          updateTabs();
        });
      } else {
        item.addEventListener("click", () => {
          if (order.includes(p)) return;
          order = [order[1], p];
          relayout(order);
          rest = step === 0 ? { pair: true } : step === 1 ? { square: 0 } : { interaction: true };
          render(rest);
          updateTabs();
        });
      }
    });

    // only what the current step lets the reader point at is in the tab order
    function updateTabs() {
      for (const edge of edges.values()) edge.hit.setAttribute("tabindex", !sii && (step === 1 || (step === 2 && edge.i === selected)) ? 0 : -1);
      for (const square of squares.values()) {
        const live = (square.T & ~geometry.rest) === 0;
        square.hit.setAttribute("tabindex", sii && step >= 1 && live ? 0 : -1);
      }
      items.forEach((item) => {
        const live = sii || step === 2;
        item.tabIndex = live ? 0 : -1;
        item.setAttribute("aria-disabled", live ? "false" : "true");
      });
      items.forEach((item, p) => item.setAttribute("aria-pressed", String(sii ? order.includes(p) : step === 2 && p === selected)));
    }

    function show(next) {
      step = next;
      root.dataset.step = String(step);
      if (!sii) {
        rest = step === 0 ? { node: (1 << PAIR[0]) | (1 << PAIR[1]) } : step === 1 ? { edge: { T: 1 << PAIR[0], i: PAIR[1] } } : { player: selected };
      } else {
        rest = step === 0 ? { pair: true } : step === 1 ? { square: 0 } : { interaction: true };
      }
      updateTabs();
      render(rest);
    }
    show(0);
    return { show, steps: 3 };
  }

  figures["lattice-sv"] = (figure, helpers) => lattice(figure, helpers, "sv");
  figures["lattice-sii"] = (figure, helpers) => lattice(figure, helpers, "sii");
})();
