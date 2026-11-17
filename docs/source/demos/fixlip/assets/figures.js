// The figures of the FIxLIP demo's page (demos/fixlip/index.rst, the shapiq-demo-figure directive).
//
// Every figure explains one step of the demo on the image–text pair that the demo shows. Choosing
// another image in the demo, or in the selector of any figure, changes it in all of them (the
// event EVENT, which demo.js sends and follows too). On a narrow page, a figure has dots in place
// of the thumbnails, and on a touch screen a swipe sideways goes to the next image. Two figures have
// a slider of their own, which keeps its place when the figure is drawn again.
//
// The figures read the demo's explanations (data.js) and the predictions of SigLIP 2 for the masks
// they show (figures-data.js, by ../compute/evaluate.py). Four figures do not follow the image:
// the sketch of the pointing game and the three plots of the paper, which are drawn again from the
// data of the paper's figures (../compute/paper.py). A figure is drawn at the width it has on the
// page, so its text keeps its size on a phone, and its colors are those of figures.css, so it
// follows the theme. Without JavaScript, a figure shows its description.
(() => {
  const figures = [...document.querySelectorAll("figure.fixlip-figure")];
  if (!figures.length) return;

  const SVG = "http://www.w3.org/2000/svg";
  const EVENT = "fixlip-demo-image"; // as in demo.js
  const RED = [255, 13, 87];
  const BLUE = [30, 136, 229];
  const MODEL = "siglip2"; // the model of the predictions in figures-data.js
  const STRONGEST = 12; // the interactions drawn in an explanation
  const STATIC = new Set(["pointing", "paper-faithfulness", "paper-curves", "paper-runtime"]);
  // the methods of the paper, in the order of their colors (figures.css)
  const SERIES = ["FIxLIP (p = 0.3)", "FIxLIP (p = 0.5)", "FIxLIP (p = 0.7)", "Shapley values", "Grad-ECLIP", "GAME", "exCLIP"];
  const root = figures[0].dataset.root;

  // -- helpers -------------------------------------------------------------------------------------

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const add = (parent, ...children) => {
    parent.append(...children);
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
  const fmt = (value, digits = 3) => value.toFixed(digits).replace("-", "−");
  const signed = (value, digits = 3) => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
  const percent = (value) => `${Math.round(100 * value)}%`;
  const rgba = (value, largest, strength = 0.85) => {
    const [r, g, b] = value >= 0 ? RED : BLUE;
    const alpha = largest > 0 ? Math.min(Math.abs(value) / largest, 1) * strength : 0;
    return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`;
  };
  const largest = (values) => Math.max(1e-12, ...values.map(Math.abs));
  // a text with *emphasis* as nodes
  const rich = (text) =>
    text.split("*").map((part, k) => (k % 2 ? el("em", "", part) : document.createTextNode(part)));
  const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  // a small generator of numbers that look random, the same for the same seed
  const noise = (seed) => {
    let state = [...seed].reduce((hash, char) => (hash * 31 + char.charCodeAt(0)) >>> 0, 7);
    return () => {
      state = (state * 1664525 + 1013904223) >>> 0;
      return state / 2 ** 32;
    };
  };

  // a slider with its label and its value. A drag on it is not a swipe of the figure
  const slider = (text, [min, max], value, onInput) => {
    const node = el("label", "fixlip-figure-slider");
    const input = el("input");
    Object.assign(input, { type: "range", min, max, step: 1, value });
    const output = el("output", "", String(value));
    // the part of the track left of the thumb, which the style sheet fills
    const fill = () => input.style.setProperty("--fixlip-part", (input.value - min) / (max - min || 1));
    node.set = (now) => {
      input.value = now;
      output.textContent = String(now);
      fill();
    };
    input.addEventListener("input", () => {
      output.textContent = input.value;
      fill();
      onInput(Number(input.value));
    });
    fill();
    for (const type of ["touchstart", "touchmove"])
      node.addEventListener(type, (event) => event.stopPropagation(), { passive: true });
    return add(node, el("span", "", text), input, output);
  };

  // -- the data ------------------------------------------------------------------------------------

  let data = null; // the demo's (data.js)
  let extra = null; // the figures' (figures-data.js)
  let current = null; // the id of the image shown
  const explanations = new Map();

  // an explanation of the demo, with its interactions as a function of two players
  const explanationOf = (image, model) => {
    const key = `${image}/${model}`;
    if (!explanations.has(key)) {
      const e = data.explanations[key];
      const n = e.first.length;
      const at = (i, j) => {
        if (i === j) return 0;
        const [a, b] = i < j ? [i, j] : [j, i];
        return e.pairs[a * n - (a * (a + 1)) / 2 + (b - a - 1)];
      };
      const patches = e.grid * e.grid;
      explanations.set(key, {
        ...e,
        image,
        model,
        n,
        patches,
        at,
        words: e.tokens.map(([text]) => text),
        // the first-order conversion: a main effect and half of the player's interactions
        order1: e.first.map((value, i) => value + 0.5 * sum(range(n).map((j) => at(i, j)))),
      });
    }
    return explanations.get(key);
  };
  // The subsets of players with the highest value, one for every size: the demo's greedy search
  // for cliques (greedy_cliques in ../compute/explain.py), step for step, so that it finds the
  // subsets that ../compute/evaluate.py evaluated the model on. From every player, the search
  // adds the player that raises the value the most, and keeps the best subset of every size.
  const greedy = (first, matrix, n) => {
    const best = Array(n + 1).fill(null);
    const gains = new Float64Array(n);
    const free = new Uint8Array(n);
    for (let start = 0; start < n; start++) {
      const players = [start];
      let value = first[start];
      for (let j = 0; j < n; j++) gains[j] = first[j] + matrix[j * n + start];
      free.fill(1);
      free[start] = 0;
      for (let size = 1; size <= n; size++) {
        if (!best[size] || value > best[size].value) best[size] = { players: [...players], value };
        if (size === n) break;
        let next = -1;
        for (let j = 0; j < n; j++) if (free[j] && (next < 0 || gains[j] > gains[next])) next = j;
        value += gains[next];
        players.push(next);
        free[next] = 0;
        for (let j = 0; j < n; j++) gains[j] += matrix[j * n + next];
      }
    }
    return best.map((found) => found?.players ?? []);
  };
  // the subsets that an explanation values the most and the least, by their size
  const subsetsOf = (ex) => {
    if (!ex.subsets) {
      const { n } = ex;
      const matrix = new Float64Array(n * n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) matrix[i * n + j] = ex.at(i, j);
      ex.subsets = {
        most: greedy(Float64Array.from(ex.first), matrix, n),
        least: greedy(Float64Array.from(ex.first, (value) => -value), matrix.map((value) => -value), n),
      };
    }
    return ex.subsets;
  };
  // what a figure reads for the image shown
  const contextOf = (id) => {
    const image = data.images.find((x) => x.id === id);
    const ex = explanationOf(id, MODEL);
    return { id, image, url: `${root}${id}.jpg`, ex, fig: extra.images[id], model: data.models.find((m) => m.id === MODEL) };
  };
  // a token that is a word, not only punctuation
  const isWord = (ex, k) => /[\p{L}\p{N}]/u.test(ex.words[k - ex.patches]);
  // a player in words: a token as it is (a punctuation mark in quotes), a patch by its place
  const nameOf = (ex, k) => {
    if (k < ex.patches) return `the patch in row ${Math.floor(k / ex.grid) + 1}, column ${(k % ex.grid) + 1}`;
    return isWord(ex, k) ? `*${ex.words[k - ex.patches]}*` : `the token “${ex.words[k - ex.patches]}”`;
  };
  // a small cut of the photo that shows one patch
  const thumbOf = (c, ex, k) => {
    const thumb = el("span", "fixlip-figure-thumb");
    thumb.style.backgroundImage = `url("${c.url}")`;
    thumb.style.backgroundSize = `${ex.grid * 100}%`;
    thumb.style.backgroundPosition = `${((k % ex.grid) / (ex.grid - 1)) * 100}% ${(Math.floor(k / ex.grid) / (ex.grid - 1)) * 100}%`;
    thumb.title = `patch in row ${Math.floor(k / ex.grid) + 1}, column ${(k % ex.grid) + 1}`;
    return thumb;
  };
  // all pairs of players by the strength of their interaction, the strongest first
  const strongest = (ex, keep = () => true) => {
    const pairs = [];
    for (let i = 0; i < ex.n; i++)
      for (let j = i + 1; j < ex.n; j++) if (keep(i, j)) pairs.push({ i, j, value: ex.at(i, j) });
    return pairs.sort((x, y) => Math.abs(y.value) - Math.abs(x.value));
  };
  const crossModal = (ex) => (i, j) => i < ex.patches && j >= ex.patches;

  // -- the tokens of a text ---------------------------------------------------------------------------

  // a token: it takes the room of its bold form (figures.css), so that a frame moves no other token
  const chip = (word) => {
    const node = el("span", "fixlip-figure-token", word);
    node.dataset.word = word;
    return node;
  };
  // the tokens of a text into their block: a punctuation mark stays on the line of the token before it
  const fill = (block, ex, dress = (node) => node) => {
    const groups = [];
    ex.words.forEach((word, k) => {
      const node = dress(chip(word), ex.patches + k);
      if (groups.length && !isWord(ex, ex.patches + k)) groups.at(-1).push(node);
      else groups.push([node]);
    });
    for (const group of groups) block.append(group.length > 1 ? add(el("span", "fixlip-figure-glued"), ...group) : group[0]);
    return block;
  };
  // The block of a text is as high as the block of the longest text of the demo at this width, so
  // that a figure keeps its height when the pair changes. Once the block is on the page.
  const heights = new Map();
  const reserve = (block, ex) => {
    if (!block.isConnected || !ex.model) return;
    const width = Math.round(block.getBoundingClientRect().width);
    const style = getComputedStyle(block);
    const key = [ex.model, width, style.fontSize, style.rowGap].join("|");
    if (width && !heights.has(key)) {
      const probe = block.cloneNode(false);
      Object.assign(probe.style, { position: "absolute", visibility: "hidden", left: "0", top: "0", width: `${width}px`, minHeight: "0" });
      block.after(probe);
      let high = 0;
      for (const image of data.images) {
        if (!data.explanations[`${image.id}/${ex.model}`]) continue;
        probe.replaceChildren();
        high = Math.max(high, fill(probe, explanationOf(image.id, ex.model)).getBoundingClientRect().height);
      }
      probe.remove();
      heights.set(key, Math.ceil(high));
    }
    if (heights.has(key)) block.style.minHeight = `${heights.get(key)}px`;
  };
  const textOf = (ex, dress) => {
    const block = fill(el("div", "fixlip-figure-tokens"), ex, dress);
    block.reserve = () => reserve(block, ex);
    return block;
  };

  // -- a photo with its patches and the tokens of its text ------------------------------------------

  // The photo, cut into its grid of patches, over the tokens of the text. What a player looks
  // like is given per player: `tint` (a color), `kept` (the players that are not masked), `framed`
  // and `members` (sets of players), and `links` (interactions drawn between players).
  function pair(c, options = {}) {
    const ex = options.ex ?? c.ex;
    const { tint, kept, framed, members, links, lined = false, text = true } = options;
    const node = el("div", "fixlip-figure-pair");
    const photo = el("div", "fixlip-figure-photo");
    const picture = el("img");
    picture.src = c.url;
    picture.alt = "";
    const cells = el("div", "fixlip-figure-cells");
    cells.style.setProperty("--fixlip-grid", ex.grid);
    if (lined) cells.classList.add("fixlip-figure-lined");
    const players = [];
    const dress = (player, k) => {
      const color = tint?.(k);
      // a patch is tinted over the photo, a token over its own ground, which hides the lines under it
      if (color && k < ex.patches) player.style.backgroundColor = color;
      else if (color) player.style.backgroundImage = `linear-gradient(${color}, ${color})`;
      if (kept && !kept.has(k)) player.classList.add("fixlip-figure-masked");
      if (framed?.has(k)) player.classList.add("fixlip-figure-framed");
      if (members) player.classList.add(members.has(k) ? "fixlip-figure-member" : "fixlip-figure-rest");
      players.push(player);
      return player;
    };
    for (const k of range(ex.patches)) cells.append(dress(el("span", "fixlip-figure-cell"), k));
    add(node, add(photo, picture, cells));
    if (text) node.append(textOf(ex, dress));
    // room for the arcs between two tokens, also where a pair has none, so that the height stays
    if (links) node.classList.add("fixlip-figure-arcs");
    if (links?.length) {
      const edges = svgEl("svg", { class: "fixlip-figure-edges", "aria-hidden": "true" });
      node.append(edges);
      // once the figure is on the page: between the centers of patches and the tops of tokens, and
      // as an arc under two tokens
      node.drawLinks = () => {
        const box = node.getBoundingClientRect();
        edges.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
        const anchor = (k, under) => {
          const r = players[k].getBoundingClientRect();
          const y = k < ex.patches ? r.top + r.height / 2 : under ? r.bottom - 1 : r.top + 1;
          return [r.left + r.width / 2 - box.left, y - box.top];
        };
        const top = largest(links.map((link) => link.value));
        const scale = Math.min(1, box.width / 300); // thinner lines in a small panel
        for (const { i, j, value } of [...links].sort((x, y) => Math.abs(x.value) - Math.abs(y.value))) {
          const strength = Math.abs(value) / top;
          if (strength < 0.08) continue;
          const both = i >= ex.patches && j >= ex.patches;
          // two tokens of one row: an arc in the gap under the row. Two tokens of different
          // rows: a line from under the upper one to the top of the lower one
          const rows = both ? [i, j].map((k) => players[k].getBoundingClientRect().top) : [0, 0];
          const sameRow = both && Math.abs(rows[0] - rows[1]) < 4;
          const [upper, lower] = rows[0] <= rows[1] ? [i, j] : [j, i];
          const [x1, y1] = both && !sameRow ? anchor(upper, true) : anchor(i, both);
          const [x2, y2] = both && !sameRow ? anchor(lower, false) : anchor(j, both);
          const peak = y1 + Math.min(4 + 0.06 * Math.abs(x2 - x1), 11);
          edges.append(
            svgEl("path", {
              d: sameRow
                ? `M ${x1} ${y1} Q ${(x1 + x2) / 2} ${2 * peak - y1} ${x2} ${y2}`
                : `M ${x1} ${y1} L ${x2} ${y2}`,
              fill: "none",
              stroke: `rgb(${(value >= 0 ? RED : BLUE).join(", ")})`,
              "stroke-linecap": "round",
              "stroke-width": ((1.2 + 5 * strength) * scale).toFixed(1),
              "stroke-opacity": (0.35 + 0.55 * strength).toFixed(2),
            }),
          );
        }
      };
    }
    return node;
  }
  // once a figure, or a part that was drawn again, is on the page: the room of its texts, then the
  // lines of its interactions
  const settle = (root) => {
    for (const block of root.querySelectorAll(".fixlip-figure-tokens")) block.reserve?.();
    for (const node of root.querySelectorAll(".fixlip-figure-pair")) node.drawLinks?.();
  };
  // tints of main effects or attributions: the patches and the tokens on scales of their own
  const tintOf = (ex, values, strength = 0.8) => {
    const tops = [largest(values.slice(0, ex.patches)), largest(values.slice(ex.patches))];
    return (k) => rgba(values[k], tops[k < ex.patches ? 0 : 1], k < ex.patches ? strength * 0.85 : strength);
  };
  // a panel with a title over it, and a card with a number under it
  const panel = (title, ...content) => {
    const node = el("div", "fixlip-figure-panel");
    const head = el("p", "fixlip-figure-title");
    head.append(...(typeof title === "string" ? rich(title) : [title]));
    return add(node, head, ...content);
  };
  const reading = (label, value, className = "") => {
    const node = el("p", `fixlip-figure-reading ${className}`.trim());
    return add(node, el("span", "", label), el("strong", "", value));
  };
  const columns = (className, ...panels) => add(el("div", `fixlip-figure-columns ${className}`.trim()), ...panels);

  // -- charts --------------------------------------------------------------------------------------

  // A chart as wide as its place on the page, in pixels, with its plot inside the margins. Its
  // height is a number, or a function of its margins and its width.
  function chart(parent, height, margins = {}) {
    // where the texts of the figure are larger (figures.css, on a narrow page), so is their room
    const type = parseFloat(getComputedStyle(parent).getPropertyValue("--fixlip-type")) || 1;
    const m = { left: 44, right: 12, top: 12, bottom: 30, ...margins };
    for (const side of Object.keys(m)) m[side] = Math.round(m[side] * type);
    const width = Math.max(120, Math.floor(parent.getBoundingClientRect().width));
    if (typeof height === "function") height = height(m, width); // a height that follows the plot
    const svg = svgEl("svg", { class: "fixlip-figure-chart", width, height, viewBox: `0 0 ${width} ${height}` });
    parent.append(svg);
    const box = { x0: m.left, x1: width - m.right, y0: m.top, y1: height - m.bottom };
    const draw = (tag, attributes, text) => svg.appendChild(svgEl(tag, attributes, text));
    const linear = (domain, pixels) => (value) =>
      pixels[0] + ((value - domain[0]) / (domain[1] - domain[0])) * (pixels[1] - pixels[0]);
    const xAxis = (x, ticks, label, format = (t) => fmt(t, 1)) => {
      draw("line", { class: "fixlip-figure-axis", x1: box.x0, x2: box.x1, y1: box.y1, y2: box.y1 });
      for (const tick of ticks)
        draw("text", { x: x(tick), y: box.y1 + 14 * type, "text-anchor": "middle" }, format(tick));
      if (label) draw("text", { x: (box.x0 + box.x1) / 2, y: height - 2, "text-anchor": "middle" }, label);
    };
    const yAxis = (y, ticks, label, format = (t) => fmt(t, 1)) => {
      for (const tick of ticks) {
        draw("line", { class: "fixlip-figure-grid", x1: box.x0, x2: box.x1, y1: y(tick), y2: y(tick) });
        draw("text", { x: box.x0 - 6, y: y(tick) + 4 * type, "text-anchor": "end" }, format(tick));
      }
      if (label)
        draw(
          "text",
          { x: 11 * type, y: (box.y0 + box.y1) / 2, "text-anchor": "middle", transform: `rotate(-90 ${11 * type} ${(box.y0 + box.y1) / 2})` },
          label,
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
    return { svg, width, height, box, draw, linear, xAxis, yAxis, path, inset, type };
  }
  // whether a part lies at the right of another one, and not under it as on a narrow page
  const beside = (left, right) => right.getBoundingClientRect().left >= left.getBoundingClientRect().right;
  // a step that gives a handful of ticks for a range
  const stepOf = (low, high) => [0.02, 0.05, 0.1, 0.2, 0.5].find((step) => (high - low) / step <= 6) ?? 1;
  const ticksOf = (low, high, step) => {
    const ticks = [];
    for (let t = Math.ceil(low / step - 1e-9) * step; t <= high + 1e-9; t += step) ticks.push(Math.round(t * 1e6) / 1e6);
    return ticks;
  };

  // -- the figures that follow the image -------------------------------------------------------------

  const FIGURES = {
    // from an image–text pair to a similarity: both encoders, their embeddings, the cosine
    encoders(stage, c) {
      const { ex, fig } = c;
      // the embedding as the model computed it (of unit length): its first values as a column
      // vector, each on a ground that is the deeper the larger the value is
      const SHOWN = 4;
      const embedding = (values) => {
        const node = el("div", "fixlip-figure-embedding");
        const shown = values.slice(0, SHOWN);
        const top = largest(shown);
        for (const value of shown) {
          const cell = el("span", "", fmt(value));
          cell.style.setProperty("--fixlip-share", `${Math.round(8 + 42 * (Math.abs(value) / top))}%`);
          node.append(cell);
        }
        return add(node, el("span", "fixlip-figure-more", "\u22EE"));
      };
      // an input with its label under it. The label is no part of the height that the arrows and the
      // embeddings are centered on (figures.css)
      const row = (input, name, symbol, count, values) => [
        add(el("div", "fixlip-figure-input"), add(el("div", "fixlip-figure-held"), input, el("small", "", count))),
        add(el("div", "fixlip-figure-encoder"), el("span", "", name), add(el("small", "fixlip-figure-symbol"), el("i", "", "f"), el("sub", "", symbol))),
        add(el("div", "fixlip-figure-vector"), embedding(values), el("small", "", `embedding of ${fig.embeddings.dimension} values`)),
      ];
      const photo = pair(c, { lined: true, text: false });
      const text = textOf(ex);
      // the room of the longest text is kept around the text and its label, with the text in its middle
      const room = text.reserve;
      text.reserve = () => {
        room();
        // on a narrow page the label lies under the block of the text, which keeps the room itself
        if (getComputedStyle(text.nextElementSibling).position !== "absolute") return;
        text.closest(".fixlip-figure-input").style.minHeight = text.style.minHeight;
        text.style.minHeight = "";
      };
      const result = el("div", "fixlip-figure-similarity");
      add(result, el("span", "", "cosine similarity"), el("strong", "", fmt(ex.similarity)));
      add(
        stage,
        add(
          el("div", "fixlip-figure-encoders"),
          ...row(photo, "Vision encoder", "\u2110", `${ex.patches} patches`, fig.embeddings.image),
          ...row(text, "Language encoder", "\u{1D4AF}", `${ex.words.length} tokens`, fig.embeddings.text),
          el("div", "fixlip-figure-join"),
          result,
        ),
      );
    },

    // every model has its own scale: the similarities of matching and of unrelated pairs
    scale(stage, c) {
      const models = data.models.filter((m) => data.explanations[`${c.id}/${m.id}`]);
      const lane = 74;
      const plot = chart(stage, 14 + lane * models.length + 34, { left: 68, top: 14, bottom: 34 });
      const { draw, box } = plot;
      // one axis for all pairs of the demo, so that it stays in place when the pair changes
      const pairs = Object.values(data.explanations).map((explanation) => explanation.similarity);
      const all = [...models.flatMap((m) => [...m.norms.matching, ...m.norms.unrelated]), ...pairs];
      const ends = [Math.floor(Math.min(...all) * 10) / 10, Math.ceil((Math.max(...all) + 0.02) * 10) / 10];
      const ticks = ticksOf(ends[0], ends[1], 0.1);
      // the axis reaches a little beyond its outer ticks, so that no gridline lies on its ends
      const x = plot.linear([ends[0] - 0.03, ends[1] + 0.03], [box.x0, box.x1]);
      for (const tick of ticks)
        draw("line", { class: "fixlip-figure-grid", x1: x(tick), x2: x(tick), y1: box.y0 - 8, y2: box.y1 });
      plot.xAxis(x, ticks, "cosine similarity");
      models.forEach((m, row) => {
        const y = box.y0 + lane * row + lane / 2;
        draw("text", { class: "fixlip-figure-strong", x: 0, y: y - 5 }, m.label);
        const band = (quantiles, at, kind) => {
          const q = (k) => x(quantiles[k]);
          draw("line", { class: `fixlip-figure-${kind}`, x1: q(1), x2: q(19), y1: at, y2: at, "stroke-width": 2 });
          // the middle half, with a gap at the median
          draw("rect", { class: `fixlip-figure-${kind}`, x: q(5), y: at - 5, width: q(10) - q(5) - 1.5, height: 10, rx: 2 });
          draw("rect", { class: `fixlip-figure-${kind}`, x: q(10) + 1.5, y: at - 5, width: q(15) - q(10) - 1.5, height: 10, rx: 2 });
        };
        band(m.norms.unrelated, y - 18, "gray");
        band(m.norms.matching, y, "amber");
        const value = data.explanations[`${c.id}/${m.id}`].similarity;
        const dot = draw("circle", { class: "fixlip-figure-dot", cx: x(value), cy: y, r: 5.5 });
        dot.append(svgEl("title", {}, `${m.label}, this pair ${fmt(value)}`));
        draw("text", { class: "fixlip-figure-strong", x: x(value), y: y + 21, "text-anchor": "middle" }, fmt(value));
      });
      const legend = add(
        el("p", "fixlip-figure-legend"),
        add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-amber"), "matching pairs"),
        add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-gray"), "unrelated pairs"),
        add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-round"), "this pair"),
      );
      stage.append(legend);
      plot.inset(legend);
    },

    // the model on masked inputs: one modality alone, the image without an object, random masks
    masks(stage, c) {
      const { ex, fig } = c;
      const all = range(ex.n);
      const image = all.slice(0, ex.patches);
      const text = all.slice(ex.patches);
      const [one, two] = fig.objects;
      const without = (thing) => all.filter((k) => !thing.patches.includes(k));
      const card = (label, players, value) =>
        panel(label, reading("similarity", fmt(value)), pair(c, { kept: new Set(players) }));
      add(
        stage,
        columns(
          "fixlip-figure-four",
          card("All tokens", all, fig.presets.full),
          card("No tokens", [], fig.presets.empty),
          card("Image only", image, fig.presets.image),
          card("Text only", text, fig.presets.text),
          card(`No ${one.name}`, without(one), fig.presets["no-one"]),
          card(`No ${two.name}`, without(two), fig.presets["no-two"]),
          card("A random mask", fig.examples["0.5"].players, fig.examples["0.5"].value),
          card("A sparser mask", fig.examples["0.3"].players, fig.examples["0.3"].value),
        ),
      );
    },

    // an interaction measured on the model: a patch and a word, alone and together
    interaction(stage, c) {
      const { ex, fig } = c;
      const measured = fig.together.map((t) => {
        const [patch, word, both] = [t.patch - t.neither, t.word - t.neither, t.both - t.neither];
        return { ...t, patch, word, both, measured: both - patch - word };
      });
      const all = measured.flatMap((t) => [t.patch, t.word, t.both, t.measured, t.interaction]);
      const top = largest(all);
      const low = Math.min(0, ...all);
      const zero = (-low / (top - low)) * 100;
      const panels = measured.map((t) => {
        const [i, j] = t.players;
        const thumb = thumbOf(c, ex, i);
        const title = add(el("span", "fixlip-figure-players"), thumb, el("span", "", "and"), chip(ex.words[j - ex.patches]));
        const bars = el("div", "fixlip-figure-bars");
        const bar = (label, value, kind) => {
          const track = el("span", "fixlip-figure-track");
          const fill = el("span", `fixlip-figure-fill fixlip-figure-fill-${kind === "sign" ? (value >= 0 ? "up" : "down") : kind}`);
          const width = (Math.abs(value) / (top - low)) * 100;
          fill.style.left = `${value >= 0 ? zero : zero - width}%`;
          fill.style.width = `${Math.max(width, 0.6)}%`;
          const origin = el("span", "fixlip-figure-origin");
          origin.style.left = `${zero}%`;
          add(bars, el("span", "fixlip-figure-label", label), add(track, fill, origin), el("span", `fixlip-figure-number ${kind === "sign" ? (value >= 0 ? "fixlip-figure-up" : "fixlip-figure-down") : ""}`.trim(), signed(value)));
        };
        bar("Patch added", t.patch, "gray");
        bar("Word added", t.word, "gray");
        bar("Both added", t.both, "ink");
        bar("Interaction", t.measured, "sign");
        bar("FIxLIP estimate", t.interaction, "sign");
        return panel(title, bars);
      });
      stage.append(columns("fixlip-figure-two", ...panels));
    },

    // the explanation: main effects as tints, the strongest interactions as lines, and a list
    explanation(stage, c) {
      const { ex } = c;
      const picture = pair(c, { tint: tintOf(ex, ex.first), links: strongest(ex).slice(0, STRONGEST) });
      const list = el("div", "fixlip-figure-list");
      // a player in the list: a word in italics, a punctuation mark as a token, a patch as a cut
      const shown = (k) => {
        if (k < ex.patches) return thumbOf(c, ex, k);
        return isWord(ex, k) ? el("em", "", ex.words[k - ex.patches]) : chip(ex.words[k - ex.patches]);
      };
      const line = (names, value) =>
        add(el("p"), names, el("strong", value >= 0 ? "fixlip-figure-up" : "fixlip-figure-down", signed(value)));
      const row = (players, value) => {
        const names = el("span", "fixlip-figure-names");
        players.forEach((k, at) => names.append(...(at ? [" and ", shown(k)] : [shown(k)])));
        return line(names, value);
      };
      const mains = range(ex.n).sort((a, b) => Math.abs(ex.first[b]) - Math.abs(ex.first[a]));
      add(list, el("p", "fixlip-figure-title", "Strongest main effects"), ...mains.slice(0, 4).map((k) => row([k], ex.first[k])));
      add(
        list,
        el("p", "fixlip-figure-title", "Strongest interactions"),
        ...strongest(ex)
          .slice(0, 6)
          .map(({ i, j, value }) => row([i, j], value)),
      );
      // what the page says about every pair: the patches matter through their interactions
      add(
        list,
        el("p", "fixlip-figure-title", "Sums"),
        line(el("span", "fixlip-figure-names", `main effects of the ${ex.patches} patches`), sum(ex.first.slice(0, ex.patches))),
        line(el("span", "fixlip-figure-names", "cross-modal interactions"), sum(strongest(ex, crossModal(ex)).map((p) => p.value))),
      );
      stage.append(columns("fixlip-figure-explained", picture, list));
    },

    // conditioning on a token: every player tinted by its interaction with the framed one
    conditioning(stage, c) {
      const { ex, fig } = c;
      const [patch, word] = fig.together[0].players;
      const other = fig.together[1].players[1];
      const panels = [other, word, patch].map((k) => {
        const top = largest(range(ex.n).map((j) => ex.at(k, j)));
        const title = k >= ex.patches ? `Conditioned on *${ex.words[k - ex.patches]}*` : "Conditioned on a patch";
        return panel(title, pair(c, { tint: (j) => (j === k ? null : rgba(ex.at(k, j), top)), framed: new Set([k]) }));
      });
      stage.append(columns("fixlip-figure-three", ...panels));
    },

    // a clique of the size that the slider sets, and for every size its value in the explanation
    // against what the model gains for the clique alone
    cliques(stage, c, state) {
      const { ex, fig } = c;
      const sizes = Object.keys(ex.cliques).map(Number).sort((a, b) => a - b);
      const chosen = () => (sizes.includes(state.size) ? state.size : sizes.includes(6) ? 6 : sizes[0]);
      const value = (size) => ex.cliques[size][1];
      const gain = (size) => fig.cliques[size] - fig.empty;
      const title = el("p", "fixlip-figure-title");
      const shown = el("div", "fixlip-figure-clique");
      const readings = el("div", "fixlip-figure-readings");
      const holder = el("div", "fixlip-figure-plot");
      const control = slider("tokens in the clique", [sizes[0], sizes.at(-1)], chosen(), (size) => {
        state.size = size;
        show(true);
      });
      // the legend names both lines as the readings under the plot do
      const legend = add(
        el("p", "fixlip-figure-legend"),
        add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-amber"), "value"),
        add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-ink"), "gain on the model"),
      );
      stage.append(
        columns(
          "fixlip-figure-sized",
          add(el("div", "fixlip-figure-panel"), title, shown),
          add(el("div", "fixlip-figure-panel"), legend, add(el("div"), holder, control), readings),
        ),
      );
      const all = sizes.flatMap((size) => [value(size), gain(size)]);
      const step = stepOf(Math.min(0, ...all), Math.max(...all));
      const domain = [Math.floor(Math.min(0, ...all) / step) * step, Math.ceil(Math.max(...all) / step) * step];
      // next to the photo, the plot is as high as the photo
      const high = beside(shown, holder) ? Math.round(shown.getBoundingClientRect().width) : 220;
      const plot = chart(holder, high + 20, { left: 58, right: 12, top: 0, bottom: 20 });
      const { draw, box } = plot;
      plot.inset(legend, readings);
      const x = plot.linear([sizes[0], sizes.at(-1)], [box.x0 + 10, box.x1 - 10]);
      const y = plot.linear(domain, [box.y1, box.y0]);
      // the slider lies under the axis, so that its thumb (16 pixels wide) is under its size
      control.classList.add("fixlip-figure-under");
      control.style.setProperty("--fixlip-left", `${x(sizes[0]) - 8}px`);
      control.style.setProperty("--fixlip-right", `${plot.width - x(sizes.at(-1)) - 8}px`);
      plot.yAxis(y, ticksOf(domain[0], domain[1], step), "change in similarity", (t) => fmt(t, 2));
      plot.xAxis(x, sizes, "", (t) => String(t));
      if (domain[0] < 0) draw("line", { class: "fixlip-figure-axis", x1: box.x0, x2: box.x1, y1: y(0), y2: y(0) });
      const guide = draw("line", { class: "fixlip-figure-guide", y1: box.y0, y2: box.y1 });
      plot.path(sizes.map((size) => [x(size), y(gain(size))]), { class: "fixlip-figure-line fixlip-figure-ink" });
      plot.path(sizes.map((size) => [x(size), y(value(size))]), { class: "fixlip-figure-line fixlip-figure-amber" });
      const dots = [gain, value].map(() => draw("circle", { class: "fixlip-figure-dot", r: 4.5 }));
      // the clique of the size chosen: after the first time, it is settled here
      const show = (again) => {
        const size = chosen();
        const inside = new Set(ex.cliques[size][0]);
        const picture = pair(c, { members: inside, links: strongest(ex, (i, j) => inside.has(i) && inside.has(j)) });
        title.textContent = `Clique of ${size}`;
        shown.replaceChildren(picture);
        if (again) settle(shown);
        readings.replaceChildren(
          reading("value", signed(value(size))),
          reading("gain on the model", signed(gain(size))),
        );
        guide.setAttribute("x1", x(size));
        guide.setAttribute("x2", x(size));
        [gain, value].forEach((of, k) => {
          dots[k].setAttribute("cx", x(size));
          dots[k].setAttribute("cy", y(of(size)));
        });
      };
      show(false);
    },

    // the parameter p decides which masks matter: a random mask and the predictions for 512 of them
    masking(stage, c) {
      const { ex, fig } = c;
      const ps = ["0.3", "0.5", "0.7"];
      const values = ps.flatMap((p) => fig.holdout[p].value);
      const domain = [Math.min(...values, fig.empty) - 0.005, Math.max(...values, fig.presets.full) + 0.005];
      const bins = 22;
      const counts = ps.map((p) => {
        const { value, kept } = fig.holdout[p];
        const count = range(bins).map(() => [0, 0]);
        value.forEach((v, k) => {
          const bin = Math.min(bins - 1, Math.floor(((v - domain[0]) / (domain[1] - domain[0])) * bins));
          count[bin][Number(kept[k])] += 1;
        });
        return count;
      });
      const tallest = Math.max(...counts.flat().map(([a, b]) => a + b));
      const panels = ps.map((p, at) => {
        const card = panel(`*p* = ${p}`, pair(c, { kept: new Set(fig.examples[p].players) }));
        const holder = el("div", "fixlip-figure-plot");
        card.append(holder);
        card.plot = () => {
          const plot = chart(holder, 164, { left: 46, right: 8, top: 8, bottom: 34 });
          const { draw, box } = plot;
          const x = plot.linear(domain, [box.x0, box.x1]);
          const y = plot.linear([0, tallest], [box.y1, box.y0]);
          const count = [20, 50, 100, 200].find((size) => tallest / size <= 4) ?? 500;
          plot.yAxis(y, ticksOf(0, tallest, count), "number of masks", (t) => String(t));
          const width = (box.x1 - box.x0) / bins;
          counts[at].forEach(([without, withWord], bin) => {
            const left = box.x0 + bin * width;
            if (without) draw("rect", { class: "fixlip-figure-gray", x: left + 0.75, width: width - 1.5, y: y(without), height: box.y1 - y(without) });
            if (withWord)
              draw("rect", { class: "fixlip-figure-amber", x: left + 0.75, width: width - 1.5, y: y(without + withWord), height: y(without) - y(without + withWord) - (without ? 1 : 0) });
          });
          for (const mark of [fig.empty, fig.presets.full])
            draw("line", { class: "fixlip-figure-mark", x1: x(mark), x2: x(mark), y1: box.y0, y2: box.y1 });
          const step = Math.max(0.05, stepOf(domain[0], domain[1]));
          plot.xAxis(x, ticksOf(domain[0], domain[1], step), "cosine similarity", (t) => fmt(t, 2));
        };
        return card;
      });
      stage.append(columns("fixlip-figure-three", ...panels));
      for (const card of panels) card.plot();
      const word = nameOf(ex, fig.word);
      stage.append(
        add(
          el("p", "fixlip-figure-legend"),
          add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-amber"), ...rich(`masks that keep ${word}`)),
          add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-gray"), "all other masks"),
          add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-dashed"), "no tokens and all tokens"),
        ),
      );
    },

    // predicting the model on held-out masks: the first-order conversion and the explanation
    faithfulness(stage, c) {
      const { fig } = c;
      const half = fig.holdout["0.5"];
      const all = [...half.value, ...half.order1, ...half.order2];
      const domain = [Math.min(...all) - 0.005, Math.max(...all) + 0.005];
      const panels = [
        ["order1", "First order", "gray"],
        ["order2", "Second order", "amber"],
      ].map(([key, title, kind]) => {
        const card = panel(title);
        const holder = el("div", "fixlip-figure-plot");
        card.append(holder);
        card.plot = () => {
          // a square plot, since both of its axes are predictions on one scale
          const square = (m, width) => width - m.left - m.right + m.top + m.bottom;
          const plot = chart(holder, square, { left: 46, right: 10, top: 10, bottom: 34 });
          const { draw, box } = plot;
          plot.inset(card.firstElementChild);
          const x = plot.linear(domain, [box.x0, box.x1]);
          const y = plot.linear(domain, [box.y1, box.y0]);
          const ticks = ticksOf(domain[0], domain[1], Math.max(0.05, stepOf(domain[0], domain[1])));
          plot.yAxis(y, ticks, `prediction of ${c.model.label}`, (t) => fmt(t, 2));
          plot.xAxis(x, ticks, "prediction of the explanation", (t) => fmt(t, 2));
          draw("line", { class: "fixlip-figure-mark", x1: x(domain[0]), y1: y(domain[0]), x2: x(domain[1]), y2: y(domain[1]) });
          half.value.forEach((value, k) => draw("circle", { class: `fixlip-figure-${kind} fixlip-figure-point`, cx: x(half[key][k]).toFixed(1), cy: y(value).toFixed(1), r: 2.2 }));
          draw("text", { class: "fixlip-figure-strong", x: box.x0 + 8, y: box.y0 + 14 }, `R² = ${fmt(half.r2[key], 2)}`);
          draw("text", { x: box.x0 + 8, y: box.y0 + 30 }, `rank correlation ${fmt(half.rho[key], 2)}`);
        };
        return card;
      });
      stage.append(columns("fixlip-figure-two", ...panels));
      for (const card of panels) card.plot();
    },

    // insertion and deletion: the model's prediction for the subsets valued the most and the least
    curves(stage, c, state) {
      const { ex, fig } = c;
      const n = ex.n;
      const subsets = subsetsOf(ex);
      const chosen = () => Math.min(n, state.size ?? fig.subsets.inserted.length);
      const curve = (key) => [fig.empty, ...fig.curves[key]];
      const lines = {
        "order2-max": curve("order2-max"),
        "order2-min": curve("order2-min"),
        "order1-max": curve("order1-max"),
        "order1-min": curve("order1-min"),
      };
      // the area between the curves as in the paper: on the scale from the empty input (0) to the
      // full input (1), on average over all sizes
      const aid = (order) =>
        mean(lines[`${order}-max`].map((value, k) => value - lines[`${order}-min`][k])) / (fig.presets.full - fig.empty);
      const values = Object.values(lines).flat();
      const domain = [Math.floor(Math.min(...values) * 20) / 20, Math.ceil(Math.max(...values) * 20) / 20];
      const holder = el("div", "fixlip-figure-plot");
      const cards = el("div", "fixlip-figure-subsets");
      const count = (k) => `${k} ${k === 1 ? "token" : "tokens"}`;
      const control = slider("tokens kept", [0, n], chosen(), (size) => {
        state.size = size;
        show();
      });
      const legend = add(
        el("p", "fixlip-figure-legend"),
        add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-amber"), `second order, AID ${fmt(aid("order2"), 2)}`),
        add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-gray"), `first order, AID ${fmt(aid("order1"), 2)}`),
      );
      stage.append(add(el("div", "fixlip-figure-curves"), legend, holder, control, cards));
      const plot = chart(holder, 290, { left: 46, right: 12, top: 10, bottom: 20 });
      const { draw, box } = plot;
      plot.inset(legend);
      // the slider lies under the axis, so that its thumb (16 pixels wide) is under its size
      control.classList.add("fixlip-figure-under");
      control.style.setProperty("--fixlip-left", `${box.x0 - 8}px`);
      control.style.setProperty("--fixlip-right", `${plot.width - box.x1 - 8}px`);
      const x = plot.linear([0, n], [box.x0, box.x1]);
      const y = plot.linear(domain, [box.y1, box.y0]);
      plot.yAxis(y, ticksOf(domain[0], domain[1], 0.05), `prediction of ${c.model.label}`, (t) => fmt(t, 2));
      plot.xAxis(x, ticksOf(0, n, 10), "", (t) => String(t));
      const points = (key) => lines[key].map((value, k) => [x(k), y(value)]);
      const area = [...points("order2-max"), ...points("order2-min").reverse()];
      draw("path", { class: "fixlip-figure-area", d: `${area.map(([px, py], k) => `${k ? "L" : "M"} ${px.toFixed(1)} ${py.toFixed(1)}`).join(" ")} Z` });
      plot.path(points("order1-max"), { class: "fixlip-figure-line fixlip-figure-gray" });
      plot.path(points("order1-min"), { class: "fixlip-figure-line fixlip-figure-gray fixlip-figure-dashed" });
      plot.path(points("order2-max"), { class: "fixlip-figure-line fixlip-figure-amber" });
      plot.path(points("order2-min"), { class: "fixlip-figure-line fixlip-figure-amber fixlip-figure-dashed" });
      // the two families of curves, named in the middle of the plot, inside the area between them
      const middle = Math.round(n / 2);
      const around = range(n + 1).filter((k) => Math.abs(k - middle) <= 10);
      const upper = Math.max(...around.map((k) => y(Math.min(lines["order1-max"][k], lines["order2-max"][k]))));
      const lower = Math.min(...around.map((k) => y(Math.max(lines["order1-min"][k], lines["order2-min"][k]))));
      // the insertion curves are read from the left, where no token is kept, and the deletion
      // curves from the right, where all tokens are kept
      draw("text", { x: x(middle), y: upper + 17, "text-anchor": "middle" }, "insertion \u2192");
      draw("text", { x: x(middle), y: lower - 9, "text-anchor": "middle" }, "\u2190 deletion");
      // the size that the slider sets: a line, a dot on both curves of the second-order
      // explanation, and under the chart the two subsets of this size. The insertion keeps the
      // tokens valued the most, and the deletion leaves the tokens valued the least
      const guide = draw("line", { class: "fixlip-figure-guide", y1: box.y0, y2: box.y1 });
      const kinds = [
        ["order2-max", "most", "Insertion", (size) => `${count(size)} inserted`],
        ["order2-min", "least", "Deletion", (size) => `${count(n - size)} deleted`],
      ];
      const dots = kinds.map(() => draw("circle", { class: "fixlip-figure-dot", r: 4.5 }));
      const show = () => {
        const size = chosen();
        guide.setAttribute("x1", x(size));
        guide.setAttribute("x2", x(size));
        cards.replaceChildren(
          ...kinds.map(([key, kind, title, done], k) => {
            dots[k].setAttribute("cx", x(size));
            dots[k].setAttribute("cy", y(lines[key][size]));
            const card = panel(title, reading("similarity", fmt(lines[key][size])), el("p", "fixlip-figure-count", done(size)));
            card.classList.add("fixlip-figure-subset");
            return add(card, pair(c, { kept: new Set(subsets[kind][size]) }));
          }),
        );
        settle(cards);
      };
      show();
      // with a mouse, a press or a drag on the chart sets the size too
      const pick = (event) => {
        const left = plot.svg.getBoundingClientRect().left;
        const size = Math.round(((event.clientX - left - box.x0) / (box.x1 - box.x0)) * n);
        state.size = Math.max(0, Math.min(n, size));
        control.set(state.size);
        show();
      };
      plot.svg.classList.add("fixlip-figure-scrub");
      plot.svg.addEventListener("pointerdown", (event) => {
        if (event.pointerType === "touch") return;
        plot.svg.setPointerCapture(event.pointerId);
        pick(event);
      });
      plot.svg.addEventListener("pointermove", (event) => {
        if (event.pointerType !== "touch" && event.buttons & 1) pick(event);
      });
    },

    // the same pair explained for two models: their cliques of six tokens
    models(stage, c) {
      const size = "6";
      const models = data.models.filter((m) => data.explanations[`${c.id}/${m.id}`]?.cliques[size]);
      const panels = [...models].reverse().map((m) => {
        const ex = explanationOf(c.id, m.id);
        const inside = new Set(ex.cliques[size][0]);
        return panel(m.label, pair(c, { ex, members: inside, links: strongest(ex, (i, j) => inside.has(i) && inside.has(j)) }));
      });
      stage.append(columns("fixlip-figure-two fixlip-figure-narrow", ...panels));
    },

    // -- the figures that do not follow the image ----------------------------------------------------

    // the pointing game: which interactions count as correct, and the results of the paper
    pointing(stage, _, state) {
      // a pointing game that was played for the page (../compute/pointing.py): four photos of the
      // demo in one image, and a text that names what each of them shows
      const game = extra.pointing;
      const patches = game.grid * game.grid;
      const c = { url: `${root}pointing.jpg`, ex: { grid: game.grid, patches, words: game.names } };
      const top = largest(game.interactions.flat());
      // the photo that a patch lies in, as the place of its name in the text
      const photoOf = (k) => 2 * Math.floor(Math.floor(k / game.grid) / (game.grid / 2)) + Math.floor((k % game.grid) / (game.grid / 2));
      const chosen = () => (state.name ?? 0) % game.names.length;
      const title = el("p", "fixlip-figure-title");
      const mosaic = el("div", "fixlip-figure-pointing");
      const names = el("div", "fixlip-figure-tokens");
      names.setAttribute("role", "group");
      names.setAttribute("aria-label", "Name whose interactions are shown");
      const buttons = game.names.map((name, k) => {
        const button = el("button", "fixlip-figure-token fixlip-figure-name", name);
        button.type = "button";
        button.dataset.word = name; // the room of its bold form (figures.css)
        button.addEventListener("click", () => {
          state.name = k;
          show();
        });
        names.append(button);
        return button;
      });
      const readings = el("div", "fixlip-figure-readings");
      const show = () => {
        const name = chosen();
        const values = game.interactions[name];
        const picture = pair(c, { tint: (k) => rgba(values[k], top), text: false });
        picture.classList.add("fixlip-figure-quarters");
        title.replaceChildren(...rich(`Interactions of the name *${game.names[name]}*`));
        mosaic.replaceChildren(picture);
        buttons.forEach((button, k) => {
          button.classList.toggle("fixlip-figure-framed", k === name);
          button.setAttribute("aria-pressed", String(k === name));
        });
        const inside = sum(values.filter((_, k) => photoOf(k) === name));
        readings.replaceChildren(
          reading("with its photo", signed(inside)),
          reading("with the other photos", signed(sum(values) - inside)),
          reading("recognition over all names", fmt(game.recognition, 2)),
        );
      };
      show();
      const holder = el("div", "fixlip-figure-plot");
      stage.append(
        columns(
          "fixlip-figure-pointed",
          add(el("div", "fixlip-figure-panel"), title, mosaic, names, readings),
          panel("Pointing game recognition for CLIP", holder),
        ),
      );
      // next to the photo, the plot is as high as the photo
      const high = beside(mosaic, holder) ? Math.round(mosaic.getBoundingClientRect().width) : 240;
      const plot = chart(holder, high + 34, { left: 52, right: 138, top: 0, bottom: 34 });
      const { draw, box: area } = plot;
      const x = plot.linear([1, 4], [area.x0 + 8, area.x1 - 4]);
      const y = plot.linear([0, 1], [area.y1, area.y0]);
      plot.yAxis(y, [0, 0.25, 0.5, 0.75, 1], "pointing game recognition", (t) => fmt(t, 2));
      plot.xAxis(x, [1, 2, 3, 4], "class names in the text", (t) => String(t));
      draw("line", { class: "fixlip-figure-mark", x1: area.x0, x2: area.x1, y1: y(0.25), y2: y(0.25) });
      draw("text", { x: area.x0 + 6, y: y(0.25) + 14 }, "random");
      const table = extra.paper.pointing;
      const shown = [
        ["GAME", "gray", "Grad-ECLIP, GAME", 0],
        ["Grad-ECLIP", "gray", "", 0],
        ["Shapley values", "gray", "Shapley values", 0],
        ["exCLIP", "dark", "exCLIP", 0],
        ["FIxLIP (p = 0.5)", "amber", "FIxLIP", 0],
      ];
      for (const [name, kind, label] of shown) {
        const scores = table[name];
        plot.path(scores.map((score, k) => [x(k + 1), y(score)]), { class: `fixlip-figure-line fixlip-figure-${kind}` });
        scores.forEach((score, k) => {
          const dot = draw("circle", { class: `fixlip-figure-${kind} fixlip-figure-marker`, cx: x(k + 1), cy: y(score), r: 4 });
          dot.append(svgEl("title", {}, `${name}, ${k + 1} ${k ? "objects" : "object"}: ${fmt(score, 2)}`));
        });
        if (label) draw("text", { class: kind === "amber" ? "fixlip-figure-strong" : "", x: area.x1 + 4, y: y(scores[3]) + 4 }, `${label} ${fmt(scores[3], 2)}`);
      }
    },

    // the paper's faithfulness of every method, as box plots
    "paper-faithfulness"(stage) {
      const draw = (key, title, label, matching) => {
        const { methods } = extra.paper[key];
        const card = panel(title);
        const holder = el("div", "fixlip-figure-plot");
        card.append(holder);
        stage.append(card);
        const lane = 24;
        const plot = chart(holder, 10 + lane * methods.length + 36, { left: 112, right: 14, top: 10, bottom: 36 });
        const { box } = plot;
        const x = plot.linear([0, 1], [box.x0, box.x1]);
        for (const tick of [0, 0.2, 0.4, 0.6, 0.8, 1])
          plot.draw("line", { class: "fixlip-figure-grid", x1: x(tick), x2: x(tick), y1: box.y0, y2: box.y1 });
        plot.xAxis(x, [0, 0.2, 0.4, 0.6, 0.8, 1], label);
        methods.forEach((method, row) => {
          const y = box.y0 + lane * row + lane / 2;
          const ours = method.name.startsWith("FIxLIP");
          const kind = method.name === matching ? "amber" : ours ? "soft" : "gray";
          plot.draw("text", { class: method.name === matching ? "fixlip-figure-strong" : "", x: box.x0 - 8, y: y + 4, "text-anchor": "end" }, method.name);
          const at = (value) => x(Math.max(0, value ?? -1)); // what lies below 0 is cut at the axis
          const group = plot.draw("g", {});
          const text = ["low", "q1", "median", "q3", "high"].map((k) => (method[k] === null ? "below 0" : fmt(method[k], 2)));
          group.append(svgEl("title", {}, `${method.name}: median ${text[2]}, middle half ${text[1]} to ${text[3]}, whiskers ${text[0]} to ${text[4]}`));
          group.append(svgEl("line", { class: `fixlip-figure-${kind}`, x1: at(method.low), x2: at(method.high), y1: y, y2: y, "stroke-width": 1.5 }));
          if (method.q3 !== null && method.q3 > 0)
            group.append(svgEl("rect", { class: `fixlip-figure-${kind}`, x: at(method.q1), y: y - 7, width: Math.max(2, at(method.q3) - at(method.q1)), height: 14, rx: 2 }));
          if (method.median !== null && method.median > 0)
            group.append(svgEl("line", { class: "fixlip-figure-median", x1: at(method.median), x2: at(method.median), y1: y - 7, y2: y + 7 }));
          if (method.low === null || method.low < 0)
            group.append(svgEl("polygon", { class: `fixlip-figure-${kind}`, points: `${box.x0 - 6},${y} ${box.x0 - 1},${y - 4} ${box.x0 - 1},${y + 4}` }));
          if (method.median === null) plot.draw("text", { x: box.x0 + 6, y: y - 5 }, "median below 0");
        });
      };
      draw("correlation", "*p*-faithfulness correlation for *p* = 0.7", "rank correlation", "FIxLIP (p = 0.7)");
      draw("r2", "R² for *p* = 0.5", "R²", "FIxLIP (p = 0.5)");
    },

    // the paper's insertion and deletion curves: one plot that overlays the ticked methods, and one
    // small plot for every method, which ticks it
    "paper-curves"(stage, _, state) {
      // FIxLIP by its p, as in the box plots, then the other methods as in the paper
      const order = (method) => (method.name.startsWith("FIxLIP") ? Number(method.name.match(/[\d.]+/)?.[0] ?? 0) : 9);
      const methods = [...extra.paper.curves].sort((a, b) => order(a) - order(b));
      // a method keeps its color (figures.css), whichever methods are ticked next to it
      const colorOf = (method) => `fixlip-figure-method-${1 + SERIES.indexOf(method.name)}`;
      state.ticked ??= ["FIxLIP (p = 0.5)", "Shapley values", "exCLIP"];
      const ticked = () => methods.filter((method) => state.ticked.includes(method.name));
      const steps = methods[0].insertion.length - 1;
      const span = [-0.6, 1.1];
      // the paper's curves run from all tokens to none: here from none to all, as for the pair
      const curve = (plot, x, y, values, attributes) => plot.path(values.map((value, k) => [x(k), y(value)]), attributes);

      const holder = el("div", "fixlip-figure-plot fixlip-figure-compared");
      const tip = el("div", "fixlip-figure-tip");
      tip.hidden = true;
      let focused = null;
      let groups = new Map();
      const focus = (name) => {
        focused = name;
        const on = name !== null && groups.has(name);
        for (const [key, group] of groups) group.classList.toggle("fixlip-figure-dimmed", on && key !== name);
      };
      const compare = () => {
        holder.replaceChildren(tip);
        tip.hidden = true;
        const plot = chart(holder, 250, { left: 52, right: 10, top: 10, bottom: 36 });
        const { box, draw } = plot;
        if (holder.isConnected) plot.inset(holder.previousElementSibling, holder.parentNode.nextElementSibling);
        const x = plot.linear([steps, 0], [box.x0, box.x1]);
        const y = plot.linear(span, [box.y1, box.y0]);
        plot.yAxis(y, [-0.5, 0, 0.5, 1], "normalized prediction", (t) => fmt(t, 1));
        draw("line", { class: "fixlip-figure-axis", x1: box.x0, x2: box.x1, y1: y(0), y2: y(0) });
        for (const share of [0, 0.25, 0.5, 0.75, 1])
          draw("text", { x: x(steps * (1 - share)), y: box.y1 + 14, "text-anchor": share === 0 ? "start" : share === 1 ? "end" : "middle" }, percent(share));
        draw("text", { x: (box.x0 + box.x1) / 2, y: plot.height - 2, "text-anchor": "middle" }, "tokens kept");
        const shown = ticked();
        groups = new Map();
        for (const method of shown) {
          const group = draw("g", { class: `fixlip-figure-series ${colorOf(method)}` });
          for (const [key, dashed, what] of [
            ["insertion", "", "insertion"],
            ["deletion", " fixlip-figure-dashed", "deletion"],
          ]) {
            const line = curve(plot, x, y, method[key], { class: `fixlip-figure-line${dashed}` });
            line.append(svgEl("title", {}, `${method.name}, ${what}`));
            group.append(line);
          }
          groups.set(method.name, group);
        }
        if (!shown.length) draw("text", { x: (box.x0 + box.x1) / 2, y: y(0.75) + 4, "text-anchor": "middle" }, "No method is ticked.");
        focus(focused);
        // what the pointer is over: a line at this share of tokens, and the values of the ticked methods
        const guide = draw("line", { class: "fixlip-figure-guide", y1: box.y0, y2: box.y1, visibility: "hidden" });
        const dots = shown.flatMap((method) =>
          ["insertion", "deletion"].map((key) => [method, key, draw("circle", { class: `fixlip-figure-marker ${colorOf(method)}`, r: 3.5, visibility: "hidden" })]),
        );
        const point = (event) => {
          const rect = plot.svg.getBoundingClientRect();
          const px = ((event.clientX - rect.left) / rect.width) * plot.width;
          const k = Math.max(0, Math.min(steps, Math.round(steps * (1 - (px - box.x0) / (box.x1 - box.x0)))));
          guide.setAttribute("x1", x(k));
          guide.setAttribute("x2", x(k));
          guide.setAttribute("visibility", "visible");
          for (const [method, key, dot] of dots) {
            dot.setAttribute("cx", x(k));
            dot.setAttribute("cy", y(method[key][k]));
            dot.setAttribute("visibility", "visible");
          }
          const rows = shown.map((method) =>
            add(
              el("tr"),
              add(el("th"), el("i", `fixlip-figure-swatch ${colorOf(method)}`), method.name),
              el("td", "", fmt(method.insertion[k], 2)),
              el("td", "", fmt(method.deletion[k], 2)),
            ),
          );
          const head = add(el("tr"), el("th", "", `${percent(1 - k / steps)} of the tokens kept`), el("td", "", "insertion"), el("td", "", "deletion"));
          tip.replaceChildren(add(el("table"), add(el("thead"), head), add(el("tbody"), ...rows)));
          tip.hidden = !shown.length;
          // next to the line, on the side with more room, and inside the plot
          const wide = tip.offsetWidth;
          const at = x(k) < (box.x0 + box.x1) / 2 ? x(k) + 10 : x(k) - 10 - wide;
          tip.style.left = `${Math.max(0, Math.min(plot.width - wide, at))}px`;
        };
        const leave = () => {
          guide.setAttribute("visibility", "hidden");
          for (const [, , dot] of dots) dot.setAttribute("visibility", "hidden");
          tip.hidden = true;
        };
        plot.svg.addEventListener("pointermove", point);
        plot.svg.addEventListener("pointerdown", point);
        // a finger that lifts leaves the values in place, a mouse that leaves takes them away
        plot.svg.addEventListener("pointerleave", (event) => {
          if (event.pointerType !== "touch") leave();
        });
      };

      const grid = el("div", "fixlip-figure-multiples");
      grid.setAttribute("role", "group");
      grid.setAttribute("aria-label", "Methods in the plot above");
      const cards = methods.map((method) => {
        const card = el("div", `fixlip-figure-panel fixlip-figure-tick ${colorOf(method)}`);
        if (method.name.startsWith("FIxLIP")) card.classList.add("fixlip-figure-ours");
        const box = el("input");
        box.type = "checkbox";
        box.checked = state.ticked.includes(method.name);
        const head = add(el("label", "fixlip-figure-title"), box, el("span", "", method.name));
        const plotHolder = el("div", "fixlip-figure-plot");
        add(card, head, plotHolder, reading("AID", `${fmt(method.aid, 2)} ± ${fmt(method.sd, 2)}`));
        const set = () => card.classList.toggle("fixlip-figure-ticked", box.checked);
        set();
        box.addEventListener("change", () => {
          state.ticked = methods.map((m) => m.name).filter((name) => (name === method.name ? box.checked : state.ticked.includes(name)));
          set();
          compare();
        });
        // a press anywhere on the card ticks it
        card.addEventListener("click", (event) => {
          if (!head.contains(event.target)) box.click();
        });
        for (const [type, name] of [
          ["pointerenter", method.name],
          ["pointerleave", null],
          ["focusin", method.name],
          ["focusout", null],
        ])
          card.addEventListener(type, () => focus(name));
        grid.append(card);
        return [plotHolder, method];
      });

      stage.append(
        panel("Insertion and deletion curves of the ticked methods", holder),
        add(
          el("p", "fixlip-figure-legend"),
          add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-solid"), "insertion"),
          add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-dashed"), "deletion"),
        ),
        grid,
      );
      compare();
      for (const [plotHolder, method] of cards) {
        const plot = chart(plotHolder, 136, { left: 16, right: 4, top: 6, bottom: 18 });
        const { box } = plot;
        const x = plot.linear([steps, 0], [box.x0, box.x1]);
        const y = plot.linear(span, [box.y1, box.y0]);
        const points = (values) => values.map((value, k) => [x(k), y(value)]);
        const area = [...points(method.insertion), ...points(method.deletion).reverse()];
        plot.draw("path", { class: "fixlip-figure-area", d: `${area.map(([px, py], k) => `${k ? "L" : "M"} ${px.toFixed(1)} ${py.toFixed(1)}`).join(" ")} Z` });
        plot.draw("line", { class: "fixlip-figure-axis", x1: box.x0, x2: box.x1, y1: y(0), y2: y(0) });
        plot.draw("line", { class: "fixlip-figure-mark", x1: box.x0, x2: box.x1, y1: y(1), y2: y(1) });
        for (const tick of [0, 1]) plot.draw("text", { x: box.x0 - 5, y: y(tick) + 4, "text-anchor": "end" }, String(tick));
        plot.draw("text", { x: box.x0, y: plot.height - 3 }, "0%");
        plot.draw("text", { x: box.x1, y: plot.height - 3, "text-anchor": "end" }, "100%");
        curve(plot, x, y, method.insertion, { class: "fixlip-figure-line" });
        curve(plot, x, y, method.deletion, { class: "fixlip-figure-line fixlip-figure-dashed" });
      }
    },

    // the paper's runtime of the two estimators against the number of sampled masks
    "paper-runtime"(stage) {
      const { budgets } = extra.paper.runtime;
      const series = [
        ["model-agnostic", "gray", "Model-agnostic"],
        ["cross-modal", "amber", "Cross-modal"],
      ];
      const holder = el("div", "fixlip-figure-plot");
      stage.append(holder);
      const plot = chart(holder, 280, { left: 52, right: 104, top: 12, bottom: 36 });
      const { draw, box } = plot;
      const x = plot.linear([budgets[0], budgets.at(-1)], [box.x0, box.x1]);
      const y = (seconds) => plot.linear([0, 4], [box.y1, box.y0])(Math.log10(seconds));
      for (const power of [0, 1, 2, 3, 4]) {
        draw("line", { class: "fixlip-figure-grid", x1: box.x0, x2: box.x1, y1: y(10 ** power), y2: y(10 ** power) });
        draw("text", { x: box.x0 - 6, y: y(10 ** power) + 4, "text-anchor": "end" }, (10 ** power).toLocaleString("en-US"));
      }
      draw("text", { x: 11, y: (box.y0 + box.y1) / 2, "text-anchor": "middle", transform: `rotate(-90 11 ${(box.y0 + box.y1) / 2})` }, "seconds");
      draw("line", { class: "fixlip-figure-axis", x1: box.x0, x2: box.x1, y1: box.y1, y2: box.y1 });
      for (const budget of budgets) {
        const label = draw("text", { x: x(budget), y: box.y1 + 16, "text-anchor": "middle" }, "2");
        label.append(svgEl("tspan", { dy: -5 * plot.type, "font-size": 9 * plot.type }, String(budget)));
      }
      draw("text", { x: (box.x0 + box.x1) / 2, y: plot.height - 2, "text-anchor": "middle" }, "sampled masks");
      for (const [key, kind, label] of series) {
        for (const [part, dashed] of [
          ["total", ""],
          ["inference", " fixlip-figure-dashed"],
        ]) {
          const seconds = extra.paper.runtime[key][part];
          plot.path(seconds.map((value, k) => [x(budgets[k]), y(value)]), { class: `fixlip-figure-line fixlip-figure-${kind}${dashed}` });
          if (part === "total") draw("text", { class: kind === "amber" ? "fixlip-figure-strong" : "", x: box.x1 + 8, y: y(seconds.at(-1)) + 4 }, label);
        }
      }
      // the speedups that the paper notes in its figure, between the lines of the two estimators
      for (const [budget, total, inference] of extra.paper.runtime.speedups ?? []) {
        const k = budgets.indexOf(budget);
        for (const [part, times, shift, anchor] of [
          ["total", total, -5, "end"],
          ["inference", inference, 5, "start"],
        ]) {
          const [from, to] = series.map(([key]) => y(extra.paper.runtime[key][part][k]));
          const at = x(budget) + shift;
          draw("line", { class: "fixlip-figure-bracket", x1: at, x2: at, y1: from + 3, y2: to - 3 });
          draw("text", { class: "fixlip-figure-strong", x: at + (shift < 0 ? -4 : 4), y: (from + to) / 2 + 4, "text-anchor": anchor }, `${times}×`);
        }
      }
      const legend = add(
        el("p", "fixlip-figure-legend"),
        add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-solid"), "the whole explanation"),
        add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-dashed"), "model inference only"),
      );
      stage.append(legend);
      plot.inset(legend);
    },
  };

  // -- the frame of a figure: its selector, its stage, and its dots ----------------------------------

  const frames = figures.map((figure) => {
    const name = figure.dataset.figure;
    const stage = figure.querySelector(".fixlip-figure-stage");
    return { figure, name, stage, follows: !STATIC.has(name), width: -1, buttons: [], dots: [], state: {} };
  });

  const hints = []; // the hints at the swipe that are on the page

  function choose(id, frame) {
    if (id === current) return;
    document.dispatchEvent(new CustomEvent(EVENT, { detail: { image: id, source: frame?.figure ?? null } }));
  }

  function buildSelector(frame) {
    const bar = el("div", "fixlip-figure-bar");
    const label = el("span", "fixlip-figure-shown");
    // the photo of the pair, where the thumbnails are not shown (a narrow page, and in print)
    const shown = el("img", "fixlip-figure-shown-photo");
    shown.alt = "";
    const photos = el("div", "fixlip-figure-photos");
    photos.setAttribute("role", "group");
    photos.setAttribute("aria-label", "Image–text pair of the figures");
    const dots = el("div", "fixlip-figure-dots");
    dots.setAttribute("role", "group");
    dots.setAttribute("aria-label", "Image–text pair of the figures");
    for (const image of data.images) {
      const button = el("button", "fixlip-figure-photo-button");
      button.type = "button";
      button.title = image.title ?? "";
      button.setAttribute("aria-label", image.title ?? image.id);
      const thumb = el("img");
      thumb.src = `${root}${image.id}.jpg`;
      thumb.alt = "";
      button.append(thumb);
      button.addEventListener("click", () => choose(image.id, frame));
      photos.append(button);
      frame.buttons.push(button);
      const dot = el("button", "fixlip-figure-dot-button");
      dot.type = "button";
      dot.setAttribute("aria-label", image.title ?? image.id);
      dot.addEventListener("click", () => choose(image.id, frame));
      dots.append(dot);
      frame.dots.push(dot);
    }
    add(bar, shown, label, photos);
    frame.stage.before(bar);
    frame.stage.after(dots);
    if (!hints.length) {
      // only the first figure says that it can be swiped, and only until it was
      const hint = el("p", "fixlip-figure-hint", "Swipe for another pair");
      dots.after(hint);
      hints.push(hint);
    }
    frame.label = label;
    frame.shown = shown;
    swipeable(frame);
  }

  // on a touch screen, a swipe sideways across a figure goes to the next image or the previous one
  // (as the orders demo's examples): the figure follows the finger, and past a fifth of its width,
  // or flicked, the next image comes in from the other side. A swipe up or down scrolls the page.
  function swipeable(frame) {
    const { stage } = frame;
    let swipe = null;
    const shift = (dx) => {
      stage.style.transform = dx ? `translateX(${dx}px)` : "";
      stage.style.opacity = dx ? String(1 - Math.min(0.6, Math.abs(dx) / stage.clientWidth)) : "";
    };
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
          stage.classList.add("fixlip-figure-swiping");
        }
        event.preventDefault(); // the drag is the figure's
        const at = data.images.findIndex((image) => image.id === current);
        const end = (dx > 0 && at === 0) || (dx < 0 && at === data.images.length - 1);
        swipe.dx = end ? dx / 4 : dx; // nothing beyond the first and the last image
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
      stage.classList.remove("fixlip-figure-swiping");
      const [time, from] = moves.find(([t]) => event.timeStamp - t <= 100) ?? moves.at(-1);
      const speed = (moves.at(-1)[1] - from) / Math.max(16, event.timeStamp - time);
      const flick = Math.abs(dx) > 20 && Math.abs(speed) > 0.3 && Math.sign(speed) === Math.sign(dx);
      shift(0);
      if (cancelled || end || !(Math.abs(dx) > stage.clientWidth / 5 || flick)) return;
      const at = data.images.findIndex((image) => image.id === current);
      frame.entering = dx < 0 ? 1 : -1; // the side the next image comes in from
      for (const hint of hints.splice(0)) hint.remove();
      choose(data.images[at + (dx < 0 ? 1 : -1)].id, frame);
    };
    stage.addEventListener("touchend", (event) => endSwipe(event, false));
    stage.addEventListener("touchcancel", (event) => endSwipe(event, true));
  }

  // From one drawing of a figure to the next, part by part. A part that is the same in both stays as
  // it is, so nothing that is still flickers. A photo fades in over the one before. The dot of a chart
  // moves to its new place. Every other part fades out before the part in its place fades in, so
  // two texts are never seen in one place.
  const OUT = 140; // milliseconds for the parts that leave, then IN for the parts that come
  const IN = 220;
  const MANY = 48; // more parts than this change as one
  function morph(last, layer) {
    const ownText = (node) => [...node.childNodes].some((child) => child.nodeType === 3 && child.data.trim());
    const isPhoto = (node) => node.classList.contains("fixlip-figure-photo");
    const isDot = (node) => node.localName === "circle" && node.classList.contains("fixlip-figure-dot");
    // the marks of a part of the old layer (a) and of the new one (b)
    const walk = (a, b) => {
      if (a.isEqualNode(b)) return [[a, "gone"]];
      if (isPhoto(a) && isPhoto(b)) return [[b, "cross"]];
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
        const frames = [{ transform: `translate(${by[0]}px, ${by[1]}px)` }, { transform: "none" }];
        node.animate(frames, { duration: OUT + IN, easing: "cubic-bezier(0.2, 0.7, 0.2, 1)" });
      } else node.classList.add(`fixlip-figure-${name}`);
    }
    return () => {
      for (const [node, name] of marks) node.classList.remove(`fixlip-figure-${name}`);
    };
  }

  // A figure is drawn into a layer of its stage. With `fade`, the layer of the pair before stays
  // under the new one until the new one has taken its place (morph). Both lie in one cell of the
  // stage, and the texts keep their height, so nothing around the figure moves.
  function render(frame, fade = false) {
    const { stage, name } = frame;
    const draw = FIGURES[name];
    if (!draw) return;
    frame.finish?.(); // a change that is still under way ends here
    const before = [...stage.children];
    const layer = el("div", "fixlip-figure-layer");
    stage.append(layer);
    const c = frame.follows ? contextOf(current) : null;
    try {
      draw(layer, c, frame.state);
      settle(layer);
    } catch (error) {
      console.error(`The figure "${name}" could not be drawn:`, error);
      layer.replaceChildren(el("p", "fixlip-figure-missing", stage.getAttribute("aria-label") ?? ""));
    }
    if (frame.follows) {
      frame.label.textContent = c.image.title ?? "";
      if (!frame.shown.src.endsWith(c.url)) frame.shown.src = c.url;
      data.images.forEach((image, k) => {
        frame.buttons[k].setAttribute("aria-pressed", String(image.id === current));
        frame.dots[k].setAttribute("aria-pressed", String(image.id === current));
      });
    }
    frame.figure.classList.add("fixlip-figure-ready");
    const entering = frame.entering;
    frame.entering = 0;
    const last = before.pop();
    for (const node of before) node.remove();
    if (!fade || !last || reducedMotion()) {
      last?.remove();
      return;
    }
    // after a swipe, the new pair comes in from the side that the finger came from
    if (entering) {
      last.remove();
      layer.animate([{ opacity: 0, transform: `translateX(${entering * 32}px)` }, { opacity: 1, transform: "none" }], { duration: OUT + IN, easing: "cubic-bezier(0.2, 0.7, 0.2, 1)" });
      return;
    }
    // the new layer waits, unseen, until its photos can be painted
    last.inert = true;
    layer.style.visibility = "hidden";
    let clean = null;
    let timer = 0;
    frame.finish = () => {
      frame.finish = null;
      clearTimeout(timer);
      last.remove();
      layer.style.visibility = "";
      clean?.();
      for (const animation of layer.getAnimations({ subtree: true })) animation.cancel();
    };
    const finish = frame.finish;
    const photos = [...layer.querySelectorAll("img")].map((photo) => photo.decode().catch(() => {}));
    const patience = new Promise((resolve) => setTimeout(resolve, 300));
    Promise.race([Promise.all(photos), patience]).then(() => {
      if (frame.finish !== finish) return; // ended in the meantime
      last.inert = false; // not a difference between the two drawings
      clean = morph(last, layer);
      last.inert = true;
      layer.style.visibility = "";
      timer = setTimeout(finish, OUT + IN + 40);
    });
  }

  function start() {
    data = window.SHAPIQ_FIXLIP;
    extra = window.SHAPIQ_FIXLIP_FIGURES;
    current = data.images.find((image) => extra.images[image.id])?.id ?? null;
    for (const frame of frames) {
      if (frame.follows) {
        if (!current) continue;
        buildSelector(frame);
      }
      // a new width draws the figure again (in the next frame, outside the observer's callback)
      new ResizeObserver(([entry]) => {
        const now = Math.round(entry.contentRect.width);
        if (now === frame.width) return;
        frame.width = now;
        requestAnimationFrame(() => render(frame));
      }).observe(frame.stage);
    }
    // another image, chosen in the demo or in a figure: all figures show it, and the figure it was
    // chosen in keeps its place on the screen, whatever the figures above it do
    document.addEventListener(EVENT, (event) => {
      const { image, source } = event.detail ?? {};
      if (!image || image === current || !extra.images[image]) return;
      current = image;
      const before = source?.getBoundingClientRect().top;
      for (const frame of frames) if (frame.follows) render(frame, true);
      if (source) window.scrollBy(0, source.getBoundingClientRect().top - before);
    });
  }

  // the data: on the page (the docs load both files before this script), or loaded now
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
  Promise.all([load("SHAPIQ_FIXLIP", "data.js"), load("SHAPIQ_FIXLIP_FIGURES", "figures-data.js")])
    .then(start)
    .catch((error) => {
      console.error("The figures could not start:", error);
      // the description of every figure, as without JavaScript
      for (const { stage } of frames) stage.replaceChildren(el("p", "fixlip-figure-missing", stage.getAttribute("aria-label") ?? ""));
    });
})();
