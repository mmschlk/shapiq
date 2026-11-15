// The figures of the FIxLIP demo's page (demos/fixlip/index.rst, the shapiq-demo-figure directive).
//
// Every figure explains one step of the demo on the image–text pair that the demo shows. Choosing
// another image in the demo, or in the selector of any figure, changes it in all of them (the
// event EVENT, which demo.js sends and follows too). On a narrow page, a figure has dots in place
// of the thumbnails, and on a touch screen a swipe sideways goes to the next image.
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
  const STRONGEST = 20; // the interactions drawn in an explanation
  const STATIC = new Set(["pointing", "paper-faithfulness", "paper-curves", "paper-runtime"]);
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
  // the share of a model's matching pairs that score below a similarity (as demo.js)
  const share = (value, quantiles) => {
    const k = quantiles.findIndex((q) => q > value);
    if (k === 0) return 0;
    if (k < 0) return 1;
    const [low, high] = [quantiles[k - 1], quantiles[k]];
    return (k - 1 + (value - low) / (high - low)) / (quantiles.length - 1);
  };
  // a small generator of numbers that look random, the same for the same seed
  const noise = (seed) => {
    let state = [...seed].reduce((hash, char) => (hash * 31 + char.charCodeAt(0)) >>> 0, 7);
    return () => {
      state = (state * 1664525 + 1013904223) >>> 0;
      return state / 2 ** 32;
    };
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
  // what a figure reads for the image shown
  const contextOf = (id) => {
    const image = data.images.find((x) => x.id === id);
    const ex = explanationOf(id, MODEL);
    return { id, image, url: `${root}${id}.jpg`, ex, fig: extra.images[id], model: data.models.find((m) => m.id === MODEL) };
  };
  // a player in words: a token as it is, a patch by its place
  const nameOf = (ex, k) =>
    k >= ex.patches
      ? `*${ex.words[k - ex.patches]}*`
      : `the patch in row ${Math.floor(k / ex.grid) + 1}, column ${(k % ex.grid) + 1}`;
  // all pairs of players by the strength of their interaction, the strongest first
  const strongest = (ex, keep = () => true) => {
    const pairs = [];
    for (let i = 0; i < ex.n; i++)
      for (let j = i + 1; j < ex.n; j++) if (keep(i, j)) pairs.push({ i, j, value: ex.at(i, j) });
    return pairs.sort((x, y) => Math.abs(y.value) - Math.abs(x.value));
  };
  const crossModal = (ex) => (i, j) => i < ex.patches && j >= ex.patches;

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
    if (text) {
      const tokens = el("div", "fixlip-figure-tokens");
      ex.words.forEach((word, k) => tokens.append(dress(el("span", "fixlip-figure-token", word), ex.patches + k)));
      node.append(tokens);
    }
    if (links?.length) {
      const edges = svgEl("svg", { class: "fixlip-figure-edges", "aria-hidden": "true" });
      node.append(edges);
      if (links.some(({ i, j }) => i >= ex.patches && j >= ex.patches)) node.classList.add("fixlip-figure-arcs");
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
          const [x1, y1] = anchor(i, both);
          const [x2, y2] = anchor(j, both);
          const bulge = both ? 6 + 0.1 * Math.abs(x2 - x1) : 0;
          const peak = Math.max(y1, y2) + Math.min(bulge, 18);
          edges.append(
            svgEl("path", {
              d: both
                ? `M ${x1} ${y1} Q ${(x1 + x2) / 2} ${2 * peak - (y1 + y2) / 2} ${x2} ${y2}`
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
  const drawLinks = (stage) => {
    for (const node of stage.querySelectorAll(".fixlip-figure-pair")) node.drawLinks?.();
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
  // a cell filled with amber by its value between the lowest and the highest: on the strong fills,
  // the text is dark in both themes
  const leveled = (cell, value, low, high) => {
    const level = (value - low) / Math.max(1e-9, high - low);
    cell.style.setProperty("--fixlip-level", level.toFixed(3));
    if (level > 0.45) cell.classList.add("fixlip-figure-bright");
    return cell;
  };
  const columns = (className, ...panels) => add(el("div", `fixlip-figure-columns ${className}`.trim()), ...panels);

  // -- charts --------------------------------------------------------------------------------------

  // A chart as wide as its place on the page, in pixels, with its plot inside the margins.
  function chart(parent, height, margins = {}) {
    const m = { left: 44, right: 12, top: 12, bottom: 30, ...margins };
    const width = Math.max(200, Math.floor(parent.getBoundingClientRect().width));
    const svg = svgEl("svg", { class: "fixlip-figure-chart", width, height, viewBox: `0 0 ${width} ${height}` });
    parent.append(svg);
    const box = { x0: m.left, x1: width - m.right, y0: m.top, y1: height - m.bottom };
    const draw = (tag, attributes, text) => svg.appendChild(svgEl(tag, attributes, text));
    const linear = (domain, pixels) => (value) =>
      pixels[0] + ((value - domain[0]) / (domain[1] - domain[0])) * (pixels[1] - pixels[0]);
    const xAxis = (x, ticks, label, format = (t) => fmt(t, 1)) => {
      draw("line", { class: "fixlip-figure-axis", x1: box.x0, x2: box.x1, y1: box.y1, y2: box.y1 });
      for (const tick of ticks)
        draw("text", { x: x(tick), y: box.y1 + 14, "text-anchor": "middle" }, format(tick));
      if (label) draw("text", { x: (box.x0 + box.x1) / 2, y: height - 2, "text-anchor": "middle" }, label);
    };
    const yAxis = (y, ticks, label, format = (t) => fmt(t, 1)) => {
      for (const tick of ticks) {
        draw("line", { class: "fixlip-figure-grid", x1: box.x0, x2: box.x1, y1: y(tick), y2: y(tick) });
        draw("text", { x: box.x0 - 6, y: y(tick) + 4, "text-anchor": "end" }, format(tick));
      }
      if (label)
        draw(
          "text",
          { x: 11, y: (box.y0 + box.y1) / 2, "text-anchor": "middle", transform: `rotate(-90 11 ${(box.y0 + box.y1) / 2})` },
          label,
        );
    };
    const path = (points, attributes) =>
      draw("path", { d: points.map(([x, y], k) => `${k ? "L" : "M"} ${x.toFixed(1)} ${y.toFixed(1)}`).join(" "), fill: "none", ...attributes });
    return { svg, width, height, box, draw, linear, xAxis, yAxis, path };
  }
  const ticksOf = (low, high, step) => {
    const ticks = [];
    for (let t = Math.ceil(low / step - 1e-9) * step; t <= high + 1e-9; t += step) ticks.push(Math.round(t * 1e6) / 1e6);
    return ticks;
  };

  // -- the figures that follow the image -------------------------------------------------------------

  const FIGURES = {
    // from an image–text pair to a similarity: both encoders, their embeddings, the cosine
    encoders(stage, c) {
      const { ex } = c;
      const embedding = (seed) => {
        const next = noise(`${c.id} ${seed}`);
        const node = el("div", "fixlip-figure-embedding");
        for (const _ of range(8)) {
          const value = el("span");
          value.style.opacity = (0.15 + 0.85 * next()).toFixed(2);
          node.append(value);
        }
        return node;
      };
      const row = (input, name, symbol, count, seed) => [
        add(el("div", "fixlip-figure-input"), input, el("small", "", count)),
        add(el("div", "fixlip-figure-encoder"), el("span", "", name), add(el("small"), el("i", "", "f"), el("sub", "", symbol))),
        add(el("div", "fixlip-figure-vector"), embedding(seed), el("small", "", "embedding")),
      ];
      const photo = pair(c, { lined: true, text: false });
      const text = el("div", "fixlip-figure-tokens");
      for (const word of ex.words) text.append(el("span", "fixlip-figure-token", word));
      const result = el("div", "fixlip-figure-similarity");
      add(result, el("span", "", "cosine similarity"), el("strong", "", fmt(ex.similarity)));
      add(
        stage,
        add(
          el("div", "fixlip-figure-encoders"),
          ...row(photo, "Vision encoder", "I", `${ex.patches} patches`, "image"),
          ...row(text, "Language encoder", "T", `${ex.words.length} tokens`, "text"),
          el("div", "fixlip-figure-join"),
          result,
        ),
      );
      return `${c.model.label} encodes the ${ex.patches} patches of the image and the ${ex.words.length} tokens of the text separately. For this pair, it predicts a similarity of ${fmt(ex.similarity)}.`;
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
      const domain = [Math.floor(Math.min(...all) * 10) / 10, Math.ceil((Math.max(...all) + 0.02) * 10) / 10];
      const x = plot.linear(domain, [box.x0, box.x1]);
      for (const tick of ticksOf(domain[0], domain[1], 0.1))
        draw("line", { class: "fixlip-figure-grid", x1: x(tick), x2: x(tick), y1: box.y0 - 8, y2: box.y1 });
      plot.xAxis(x, ticksOf(domain[0], domain[1], 0.1), "cosine similarity");
      const shares = [];
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
        shares.push([m.label, share(value, m.norms.matching)]);
      });
      stage.append(
        add(
          el("p", "fixlip-figure-legend"),
          add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-amber"), "matching pairs"),
          add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-gray"), "unrelated pairs"),
          add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-round"), "this pair"),
        ),
      );
      return shares
        .map(([label, below], k) =>
          k === 0
            ? `For ${label}, this pair scores higher than ${below < 0.995 ? percent(below) : "all"} of the matching pairs.`
            : `For ${label}, it scores higher than ${below < 0.995 ? percent(below) : "all of them"}.`,
        )
        .join(" ");
    },

    // similarity needs both modalities: the model's prediction with a modality masked
    modalities(stage, c) {
      const { ex, fig } = c;
      const image = range(ex.patches);
      const text = range(ex.words.length).map((k) => ex.patches + k);
      const card = (label, players, value) =>
        panel(label, pair(c, { kept: new Set(players) }), reading("similarity", fmt(value)));
      add(
        stage,
        columns(
          "fixlip-figure-four",
          card("No tokens", [], fig.presets.empty),
          card("Image only", image, fig.presets.image),
          card("Text only", text, fig.presets.text),
          card("All tokens", [...image, ...text], fig.presets.full),
        ),
      );
      return `With the image alone, ${c.model.label} predicts ${fmt(fig.presets.image)}, and with the text alone ${fmt(fig.presets.text)}. An empty input scores ${fmt(fig.presets.empty)}. Only both modalities together reach ${fmt(fig.presets.full)}.`;
    },

    // which words go with which regions: the prediction for one region and one part of the text
    regions(stage, c) {
      const { ex, fig } = c;
      const [one, two] = fig.objects;
      const values = [
        [fig.presets["one-one"], fig.presets["one-two"]],
        [fig.presets["two-one"], fig.presets["two-two"]],
      ];
      const [low, high] = [Math.min(...values.flat()), Math.max(...values.flat())];
      const table = el("div", "fixlip-figure-matrix");
      const wordsOf = (thing) => {
        const node = el("div", "fixlip-figure-tokens");
        for (const k of thing.words) node.append(el("span", "fixlip-figure-token", ex.words[k - ex.patches]));
        return node;
      };
      add(table, el("span"), wordsOf(one), wordsOf(two));
      [one, two].forEach((thing, row) => {
        const head = add(el("div", "fixlip-figure-region"), pair(c, { kept: new Set(thing.patches), text: false }));
        head.append(el("small", "", thing.name));
        table.append(head);
        values[row].forEach((value) => {
          table.append(leveled(el("div", "fixlip-figure-value", fmt(value)), value, low, high));
        });
      });
      stage.append(table);
      const words = (thing) => `*${thing.words.map((k) => ex.words[k - ex.patches]).join(" ")}*`;
      const matched = values[0][0] > values[0][1] && values[1][1] > values[1][0];
      return matched
        ? `The region of the ${one.name} goes with ${words(one)}, and the region of the ${two.name} goes with ${words(two)}. Swapping the words lowers the similarity in both rows.`
        : `The region of the ${one.name} scores ${fmt(values[0][0])} with ${words(one)} and ${fmt(values[0][1])} with ${words(two)}. The region of the ${two.name} scores ${fmt(values[1][0])} and ${fmt(values[1][1])}.`;
    },

    // three views of one explanation: first order, cross-modal interactions only, and all of it
    related(stage, c) {
      const { ex } = c;
      add(
        stage,
        columns(
          "fixlip-figure-three",
          panel("One score per token", pair(c, { tint: tintOf(ex, ex.order1) })),
          panel("Cross-modal interactions only", pair(c, { links: strongest(ex, crossModal(ex)).slice(0, STRONGEST) })),
          panel("FIxLIP", pair(c, { tint: tintOf(ex, ex.first), links: strongest(ex).slice(0, STRONGEST) })),
        ),
      );
      return "";
    },

    // the game assigns a similarity to every mask
    game(stage, c) {
      const { ex, fig } = c;
      const all = range(ex.n);
      const [one, two] = fig.objects;
      const without = (thing) => all.filter((k) => !thing.patches.includes(k));
      const card = (label, players, value) =>
        panel(label, pair(c, { kept: new Set(players) }), reading("similarity", fmt(value)));
      add(
        stage,
        columns(
          "fixlip-figure-five",
          card("All tokens", all, fig.presets.full),
          card("No tokens", [], fig.presets.empty),
          card(`No ${one.name}`, without(one), fig.presets["no-one"]),
          card(`No ${two.name}`, without(two), fig.presets["no-two"]),
          card("A random mask", fig.examples["0.5"].players, fig.examples["0.5"].value),
        ),
      );
      const full = fig.presets.full;
      const moves = (value) => `${value < full ? "lowers" : "raises"} it to ${fmt(value)}`;
      return `Removing the ${one.name} from the image ${moves(fig.presets["no-one"]).replace("it", `the similarity from ${fmt(full)}`)}. Removing the ${two.name} ${moves(fig.presets["no-two"])}.`;
    },

    // an interaction measured on the model: a patch and a word, alone and together
    interaction(stage, c) {
      const { ex, fig } = c;
      const measured = fig.together.map((t) => {
        const [patch, word, both] = [t.patch - t.neither, t.word - t.neither, t.both - t.neither];
        return { ...t, patch, word, both, measured: both - patch - word };
      });
      const top = largest(measured.flatMap((t) => [t.patch, t.word, t.both, t.measured]));
      const low = Math.min(0, ...measured.flatMap((t) => [t.patch, t.word, t.both, t.measured]));
      const zero = (-low / (top - low)) * 100;
      const panels = measured.map((t) => {
        const [i, j] = t.players;
        const thumb = el("span", "fixlip-figure-thumb");
        thumb.style.backgroundImage = `url("${c.url}")`;
        thumb.style.backgroundSize = `${ex.grid * 100}%`;
        thumb.style.backgroundPosition = `${((i % ex.grid) / (ex.grid - 1)) * 100}% ${(Math.floor(i / ex.grid) / (ex.grid - 1)) * 100}%`;
        const title = add(el("span", "fixlip-figure-players"), thumb, el("span", "", "and"), el("span", "fixlip-figure-token", ex.words[j - ex.patches]));
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
        bar("Patch alone", t.patch, "gray");
        bar("Word alone", t.word, "gray");
        bar("Both together", t.both, "ink");
        bar("Interaction", t.measured, "sign");
        return panel(title, bars);
      });
      stage.append(columns("fixlip-figure-two", ...panels));
      const [first, second] = measured;
      const word = (t) => `*${ex.words[t.players[1] - ex.patches]}*`;
      return `On average, the patch alone changes the similarity by ${signed(first.patch)} and the word ${word(first)} alone by ${signed(first.word)}. Together, they change it by ${signed(first.both)}. The difference of ${signed(first.measured)} is their interaction, which the explanation estimates as ${signed(first.interaction)}. With the word ${word(second)}, the same patch has an interaction of ${signed(second.measured)}.`;
    },

    // the explanation: main effects as tints, the strongest interactions as lines, and a list
    explanation(stage, c) {
      const { ex, fig } = c;
      const picture = pair(c, { tint: tintOf(ex, ex.first), links: strongest(ex).slice(0, STRONGEST) });
      const list = el("div", "fixlip-figure-list");
      const row = (text, value) => {
        const line = el("p");
        add(line, add(el("span"), ...rich(text)), el("strong", value >= 0 ? "fixlip-figure-up" : "fixlip-figure-down", signed(value)));
        return line;
      };
      const mains = range(ex.n).sort((a, b) => Math.abs(ex.first[b]) - Math.abs(ex.first[a]));
      const short = (k) => (k >= ex.patches ? `*${ex.words[k - ex.patches]}*` : `patch (${Math.floor(k / ex.grid) + 1}, ${(k % ex.grid) + 1})`);
      add(list, el("p", "fixlip-figure-title", "Strongest main effects"), ...mains.slice(0, 4).map((k) => row(short(k), ex.first[k])));
      add(
        list,
        el("p", "fixlip-figure-title", "Strongest interactions"),
        ...strongest(ex)
          .slice(0, 6)
          .map(({ i, j, value }) => row(`${short(i)} and ${short(j)}`, value)),
      );
      stage.append(columns("fixlip-figure-explained", picture, list));
      const word = fig.word;
      const [link] = strongest(ex, crossModal(ex));
      const patchSum = sum(ex.first.slice(0, ex.patches));
      const crossSum = sum(strongest(ex, crossModal(ex)).map((p) => p.value));
      const alone = fig.presets.word - fig.presets.empty;
      return `The word ${nameOf(ex, word)} has the largest main effect with ${signed(ex.first[word])}. Alone, this word changes the prediction of the model by ${signed(alone)}. The strongest cross-modal interaction links ${nameOf(ex, link.i)} with ${nameOf(ex, link.j)} (${signed(link.value)}). The main effects of all ${ex.patches} patches sum to ${signed(patchSum)}, while the cross-modal interactions sum to ${signed(crossSum)}.`;
    },

    // conditioning on a token: every player tinted by its interaction with the framed one
    conditioning(stage, c) {
      const { ex, fig } = c;
      const [patch, word] = fig.together[0].players;
      const other = fig.together[1].players[1];
      const best = (k) => strongest(ex, (i, j) => i === k || j === k)[0];
      const panels = [other, word, patch].map((k) => {
        const top = largest(range(ex.n).map((j) => ex.at(k, j)));
        const title = k >= ex.patches ? `Conditioned on *${ex.words[k - ex.patches]}*` : "Conditioned on the framed patch";
        return panel(title, pair(c, { tint: (j) => (j === k ? null : rgba(ex.at(k, j), top)), framed: new Set([k]) }));
      });
      stage.append(columns("fixlip-figure-three", ...panels));
      const partner = (k) => {
        const { i, j, value } = best(k);
        return `${nameOf(ex, i === k ? j : i)} (${signed(value)})`;
      };
      return `The word ${nameOf(ex, other)} interacts the most with ${partner(other)}. The word ${nameOf(ex, word)} interacts the most with ${partner(word)}, and this patch with ${partner(patch)}.`;
    },

    // cliques of 2, 6 and 10 tokens, with their value and what the model gains for them alone
    cliques(stage, c) {
      const { ex, fig } = c;
      const sizes = ["2", "6", "10"].filter((size) => ex.cliques[size]);
      const panels = sizes.map((size) => {
        const [members, value] = ex.cliques[size];
        const inside = new Set(members);
        const links = strongest(ex, (i, j) => inside.has(i) && inside.has(j));
        return panel(
          `Clique of ${size}`,
          pair(c, { members: inside, links }),
          reading("value", signed(value)),
          reading("gain on the model", signed(fig.cliques[size] - fig.empty), "fixlip-figure-second"),
        );
      });
      stage.append(columns("fixlip-figure-three", ...panels));
      const gain = (size) => signed(fig.cliques[size] - fig.empty);
      const [small, big] = [sizes[0], sizes.at(-1)];
      return `For the clique of ${small}, ${c.model.label} gains ${gain(small)} over an empty input for a value of ${signed(ex.cliques[small][1])}. For the clique of ${big}, it gains ${gain(big)} for a value of ${signed(ex.cliques[big][1])}.`;
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
        const card = panel(`p = ${p}`, pair(c, { kept: new Set(fig.examples[p].players) }));
        const holder = el("div", "fixlip-figure-plot");
        card.append(holder);
        card.plot = () => {
          const plot = chart(holder, 150, { left: 6, right: 6, top: 8, bottom: 30 });
          const { draw, box } = plot;
          const x = plot.linear(domain, [box.x0, box.x1]);
          const y = plot.linear([0, tallest], [box.y1, box.y0]);
          const width = (box.x1 - box.x0) / bins;
          counts[at].forEach(([without, withWord], bin) => {
            const left = box.x0 + bin * width;
            if (without) draw("rect", { class: "fixlip-figure-gray", x: left + 0.75, width: width - 1.5, y: y(without), height: box.y1 - y(without) });
            if (withWord)
              draw("rect", { class: "fixlip-figure-amber", x: left + 0.75, width: width - 1.5, y: y(without + withWord), height: y(without) - y(without + withWord) - (without ? 1 : 0) });
          });
          for (const mark of [fig.empty, fig.presets.full])
            draw("line", { class: "fixlip-figure-mark", x1: x(mark), x2: x(mark), y1: box.y0, y2: box.y1 });
          plot.xAxis(x, ticksOf(domain[0], domain[1], 0.1), at === 1 ? "similarity of a random mask" : "");
        };
        return card;
      });
      stage.append(columns("fixlip-figure-three", ...panels));
      for (const card of panels) card.plot();
      const word = nameOf(ex, fig.word);
      const half = fig.holdout["0.5"];
      const split = (flag) => mean(half.value.filter((_, k) => half.kept[k] === flag));
      stage.append(
        add(
          el("p", "fixlip-figure-legend"),
          add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-amber"), ...rich(`masks that keep ${word}`)),
          add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-gray"), "all other masks"),
          add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-dashed"), "no tokens and all tokens"),
        ),
      );
      return `The average prediction rises from ${fmt(mean(fig.holdout["0.3"].value))} for p = 0.3 to ${fmt(mean(fig.holdout["0.7"].value))} for p = 0.7. For p = 0.5, the masks that keep the word ${word} score ${fmt(split("1"))} on average, and all others ${fmt(split("0"))}.`;
    },

    // cross-modal sampling: every combination of six masked images and six masked texts
    cross(stage, c) {
      const { ex, fig } = c;
      const { photos, captions, values } = fig.cross;
      const [low, high] = [Math.min(...values.flat()), Math.max(...values.flat())];
      const table = el("div", "fixlip-figure-cross");
      table.append(el("span"));
      const word = fig.word - ex.patches;
      for (const caption of captions) {
        const kept = new Set(caption);
        const text = el("div", "fixlip-figure-marks");
        ex.words.forEach((_, k) => {
          const mark = el("i", kept.has(k) ? "fixlip-figure-on" : "");
          if (k === word) mark.classList.add("fixlip-figure-framed");
          text.append(mark);
        });
        table.append(text);
      }
      photos.forEach((photo, row) => {
        table.append(pair(c, { kept: new Set(photo), text: false }));
        values[row].forEach((value) => {
          const cell = leveled(el("div", "fixlip-figure-value", fmt(value, 2)), value, low, high);
          cell.title = fmt(value);
          table.append(cell);
        });
      });
      stage.append(table);
      const withWord = captions.map((caption) => caption.includes(word));
      const average = (flag) => mean(values.flatMap((row) => row.filter((_, k) => withWord[k] === flag)));
      const some = withWord.some(Boolean) && !withWord.every(Boolean);
      return `Twelve passes of an encoder give all 36 predictions.${some ? ` The texts that keep the word ${nameOf(ex, fig.word)} score ${fmt(average(true))} on average, and the others ${fmt(average(false))}.` : ""}`;
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
          const side = Math.min(300, Math.floor(holder.getBoundingClientRect().width));
          const plot = chart(holder, side, { left: 46, right: 10, top: 10, bottom: 34 });
          const { draw, box } = plot;
          const x = plot.linear(domain, [box.x0, box.x1]);
          const y = plot.linear(domain, [box.y1, box.y0]);
          const ticks = ticksOf(domain[0], domain[1], 0.1);
          plot.yAxis(y, ticks, `prediction of ${c.model.label}`);
          plot.xAxis(x, ticks, "prediction of the explanation");
          draw("line", { class: "fixlip-figure-mark", x1: x(domain[0]), y1: y(domain[0]), x2: x(domain[1]), y2: y(domain[1]) });
          half.value.forEach((value, k) => draw("circle", { class: `fixlip-figure-${kind} fixlip-figure-point`, cx: x(half[key][k]).toFixed(1), cy: y(value).toFixed(1), r: 2.2 }));
          draw("text", { class: "fixlip-figure-strong", x: box.x0 + 8, y: box.y0 + 14 }, `R² = ${fmt(half.r2[key], 2)}`);
          draw("text", { x: box.x0 + 8, y: box.y0 + 30 }, `rank correlation ${fmt(half.rho[key], 2)}`);
        };
        return card;
      });
      stage.append(columns("fixlip-figure-two", ...panels));
      for (const card of panels) card.plot();
      return `For this pair, the second-order explanation reaches an R² of ${fmt(half.r2.order2, 2)}, and its conversion to first order reaches ${fmt(half.r2.order1, 2)}. On sparser masks sampled with p = 0.3, the R² of the explanation is ${fmt(fig.holdout["0.3"].r2.order2, 2)}.`;
    },

    // insertion and deletion: the model's prediction for the subsets valued the most and the least
    curves(stage, c) {
      const { ex, fig } = c;
      const n = ex.n;
      const curve = (key) => [fig.empty, ...fig.curves[key]];
      const lines = {
        "order2-max": curve("order2-max"),
        "order2-min": curve("order2-min"),
        "order1-max": curve("order1-max"),
        "order1-min": curve("order1-min"),
      };
      const aid = (order) => sum(lines[`${order}-max`].map((value, k) => value - lines[`${order}-min`][k]));
      const inserted = fig.subsets.inserted.length;
      const deleted = n - fig.subsets.deleted.length;
      const values = Object.values(lines).flat();
      const domain = [Math.floor(Math.min(...values) * 20) / 20, Math.ceil(Math.max(...values) * 20) / 20];
      const holder = el("div", "fixlip-figure-plot");
      const cards = el("div", "fixlip-figure-subsets");
      add(
        cards,
        panel(`${inserted} tokens kept`, pair(c, { kept: new Set(fig.subsets.inserted) }), reading("similarity", fmt(lines["order2-max"][inserted]))),
        panel(`${deleted} tokens removed`, pair(c, { kept: new Set(fig.subsets.deleted) }), reading("similarity", fmt(lines["order2-min"][n - deleted]))),
      );
      const legend = add(
        el("p", "fixlip-figure-legend"),
        add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-amber"), `second order, AID ${fmt(aid("order2"), 1)}`),
        add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-gray"), `first order, AID ${fmt(aid("order1"), 1)}`),
      );
      stage.append(columns("fixlip-figure-curves", add(el("div"), legend, holder), cards));
      const plot = chart(holder, 330, { left: 46, right: 12, top: 10, bottom: 34 });
      const { draw, box } = plot;
      const x = plot.linear([0, n], [box.x0, box.x1]);
      const y = plot.linear(domain, [box.y1, box.y0]);
      plot.yAxis(y, ticksOf(domain[0], domain[1], 0.05), `prediction of ${c.model.label}`, (t) => fmt(t, 2));
      plot.xAxis(x, ticksOf(0, n, 10), "tokens kept", (t) => String(t));
      const points = (key) => lines[key].map((value, k) => [x(k), y(value)]);
      const area = [...points("order2-max"), ...points("order2-min").reverse()];
      draw("path", { class: "fixlip-figure-area", d: `${area.map(([px, py], k) => `${k ? "L" : "M"} ${px.toFixed(1)} ${py.toFixed(1)}`).join(" ")} Z` });
      plot.path(points("order1-max"), { class: "fixlip-figure-line fixlip-figure-gray" });
      plot.path(points("order1-min"), { class: "fixlip-figure-line fixlip-figure-gray fixlip-figure-dashed" });
      plot.path(points("order2-max"), { class: "fixlip-figure-line fixlip-figure-amber" });
      plot.path(points("order2-min"), { class: "fixlip-figure-line fixlip-figure-amber fixlip-figure-dashed" });
      draw("circle", { class: "fixlip-figure-dot", cx: x(inserted), cy: y(lines["order2-max"][inserted]), r: 4.5 });
      draw("circle", { class: "fixlip-figure-dot", cx: x(n - deleted), cy: y(lines["order2-min"][n - deleted]), r: 4.5 });
      // the two families of curves, named in the middle of the plot, inside the area between them
      const middle = Math.round(n / 2);
      const around = range(n + 1).filter((k) => Math.abs(k - middle) <= 10);
      const upper = Math.max(...around.map((k) => y(Math.min(lines["order1-max"][k], lines["order2-max"][k]))));
      const lower = Math.min(...around.map((k) => y(Math.max(lines["order1-min"][k], lines["order2-min"][k]))));
      draw("text", { x: x(middle), y: upper + 17, "text-anchor": "middle" }, "subsets valued the most");
      draw("text", { x: x(middle), y: lower - 9, "text-anchor": "middle" }, "subsets valued the least");
      return `The AID is ${fmt(aid("order2"), 1)} for the second-order explanation and ${fmt(aid("order1"), 1)} for its conversion to first order. The best subset of ${inserted} tokens reaches a similarity of ${fmt(lines["order2-max"][inserted])} with the second-order explanation and ${fmt(lines["order1-max"][inserted])} with the first-order ranking. The second-order explanation also finds ${deleted} tokens whose removal changes the similarity to ${fmt(lines["order2-min"][n - deleted])}.`;
    },

    // the same pair explained for two models: their cliques of six tokens
    models(stage, c) {
      const size = "6";
      const models = data.models.filter((m) => data.explanations[`${c.id}/${m.id}`]?.cliques[size]);
      const parts = [];
      const panels = [...models].reverse().map((m) => {
        const ex = explanationOf(c.id, m.id);
        const inside = new Set(ex.cliques[size][0]);
        const patches = [...inside].filter((k) => k < ex.patches).length;
        const words = [...inside].filter((k) => k >= ex.patches).map((k) => ex.words[k - ex.patches]);
        parts.push(
          `For ${m.label}, the clique of six contains ${patches} image ${patches === 1 ? "patch" : "patches"}${words.length ? ` and the ${words.length === 1 ? "word" : "words"} *${words.join(" ")}*` : ""}.`,
        );
        return panel(m.label, pair(c, { ex, members: inside, links: strongest(ex, (i, j) => inside.has(i) && inside.has(j)) }), reading("value", signed(ex.cliques[size][1])));
      });
      stage.append(columns("fixlip-figure-two fixlip-figure-narrow", ...panels));
      return parts.join(" ");
    },

    // -- the figures that do not follow the image ----------------------------------------------------

    // the pointing game: which interactions count as correct, and the results of the paper
    pointing(stage) {
      const names = { elephant: "elephant", "giraffe-tree": "giraffe", pineapple: "pineapple", "clock-tower": "clock" };
      const ids = Object.keys(names).filter((id) => data.images.some((image) => image.id === id));
      const sketch = el("div", "fixlip-figure-pointing");
      const mosaic = el("div", "fixlip-figure-mosaic");
      for (const id of ids) {
        const picture = el("img");
        picture.src = `${root}${id}.jpg`;
        picture.alt = "";
        mosaic.append(picture);
      }
      const words = el("div", "fixlip-figure-tokens");
      for (const id of ids) words.append(el("span", "fixlip-figure-token", names[id]));
      const edges = svgEl("svg", { class: "fixlip-figure-edges", "aria-hidden": "true" });
      add(sketch, mosaic, words, edges);
      const holder = el("div", "fixlip-figure-plot");
      stage.append(columns("fixlip-figure-pointed", panel(`Correct interactions of the word *${names[ids[1]]}*`, sketch), panel("Pointing game recognition", holder)));
      // from the second word: a red line to its image, blue lines to the other three
      const box = sketch.getBoundingClientRect();
      edges.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
      const source = words.children[1];
      source.classList.add("fixlip-figure-framed");
      const from = source.getBoundingClientRect();
      [...mosaic.children].forEach((picture, k) => {
        const to = picture.getBoundingClientRect();
        edges.append(
          svgEl("line", {
            x1: from.left + from.width / 2 - box.left,
            y1: from.top - box.top,
            x2: to.left + to.width / 2 - box.left,
            y2: to.top + to.height / 2 - box.top,
            stroke: `rgb(${(k === 1 ? RED : BLUE).join(", ")})`,
            "stroke-width": k === 1 ? 4 : 2.5,
            "stroke-opacity": 0.85,
            "stroke-linecap": "round",
          }),
        );
      });
      const plot = chart(holder, 260, { left: 40, right: 138, top: 10, bottom: 34 });
      const { draw, box: area } = plot;
      const x = plot.linear([1, 4], [area.x0 + 8, area.x1 - 4]);
      const y = plot.linear([0, 1], [area.y1, area.y0]);
      plot.yAxis(y, [0, 0.25, 0.5, 0.75, 1], "", (t) => fmt(t, 2));
      plot.xAxis(x, [1, 2, 3, 4], "class names in the text", (t) => String(t));
      draw("line", { class: "fixlip-figure-mark", x1: area.x0, x2: area.x1, y1: y(0.25), y2: y(0.25) });
      draw("text", { x: area.x0 + 6, y: y(0.25) + 14 }, "random");
      const table = extra.paper.pointing;
      const shown = [
        ["GAME", "gray", "Grad-ECLIP, GAME", 0],
        ["Grad-ECLIP", "gray", "", 0],
        ["Shapley values", "gray", "Shapley values", 0],
        ["exCLIP", "ink", "exCLIP", 0],
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
      return "";
    },

    // the paper's faithfulness of every method, as box plots
    "paper-faithfulness"(stage) {
      const draw = (key, title) => {
        const { methods } = extra.paper[key];
        const card = panel(title);
        const holder = el("div", "fixlip-figure-plot");
        card.append(holder);
        stage.append(card);
        const lane = 24;
        const plot = chart(holder, 10 + lane * methods.length + 32, { left: 112, right: 14, top: 10, bottom: 32 });
        const { box } = plot;
        const x = plot.linear([0, 1], [box.x0, box.x1]);
        for (const tick of [0, 0.2, 0.4, 0.6, 0.8, 1])
          plot.draw("line", { class: "fixlip-figure-grid", x1: x(tick), x2: x(tick), y1: box.y0, y2: box.y1 });
        plot.xAxis(x, [0, 0.2, 0.4, 0.6, 0.8, 1]);
        methods.forEach((method, row) => {
          const y = box.y0 + lane * row + lane / 2;
          const ours = method.name.startsWith("FIxLIP");
          const kind = ours ? "amber" : "gray";
          plot.draw("text", { class: ours ? "fixlip-figure-strong" : "", x: box.x0 - 8, y: y + 4, "text-anchor": "end" }, method.name);
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
        });
      };
      draw("correlation", "*p*-faithfulness correlation for *p* = 0.7");
      draw("r2", "R² for *p* = 0.5");
      return "";
    },

    // the paper's insertion and deletion curves, one small plot for every method
    "paper-curves"(stage) {
      const grid = el("div", "fixlip-figure-multiples");
      stage.append(grid);
      const cards = extra.paper.curves.map((method) => {
        const ours = method.name.startsWith("FIxLIP");
        const card = panel(method.name);
        if (ours) card.classList.add("fixlip-figure-ours");
        const holder = el("div", "fixlip-figure-plot");
        add(card, holder, reading("AID", `${fmt(method.aid, 2)} ± ${fmt(method.sd, 2)}`));
        grid.append(card);
        return [card, holder, method, ours ? "amber" : "gray"];
      });
      for (const [, holder, method, kind] of cards) {
        const plot = chart(holder, 120, { left: 4, right: 4, top: 6, bottom: 6 });
        const { box } = plot;
        const steps = method.insertion.length - 1;
        const x = plot.linear([0, steps], [box.x0, box.x1]);
        const y = plot.linear([-0.6, 1.1], [box.y1, box.y0]);
        const points = (values) => values.map((value, k) => [x(k), y(value)]);
        const area = [...points(method.insertion), ...points(method.deletion).reverse()];
        plot.draw("path", { class: `fixlip-figure-area fixlip-figure-area-${kind}`, d: `${area.map(([px, py], k) => `${k ? "L" : "M"} ${px.toFixed(1)} ${py.toFixed(1)}`).join(" ")} Z` });
        plot.draw("line", { class: "fixlip-figure-axis", x1: box.x0, x2: box.x1, y1: y(0), y2: y(0) });
        plot.path(points(method.insertion), { class: `fixlip-figure-line fixlip-figure-${kind}` });
        plot.path(points(method.deletion), { class: `fixlip-figure-line fixlip-figure-${kind} fixlip-figure-dashed` });
      }
      stage.append(
        add(
          el("p", "fixlip-figure-legend"),
          add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-solid"), "insert the most important tokens"),
          add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-dashed"), "delete the most important tokens"),
        ),
      );
      return "";
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
        label.append(svgEl("tspan", { dy: -5, "font-size": 9 }, String(budget)));
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
      stage.append(
        add(
          el("p", "fixlip-figure-legend"),
          add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-solid"), "the whole explanation"),
          add(el("span"), el("i", "fixlip-figure-swatch fixlip-figure-dashed"), "model inference only"),
        ),
      );
      const runtime = extra.paper.runtime;
      const at = budgets.indexOf(18);
      const seconds = (key, part) => Math.round(runtime[key][part][at]);
      return `With 2¹⁸ sampled masks, the whole explanation takes about ${seconds("cross-modal", "total")} seconds with the cross-modal estimator and about ${seconds("model-agnostic", "total")} seconds with the model-agnostic one. Model inference takes about ${seconds("cross-modal", "inference")} and ${seconds("model-agnostic", "inference")} seconds of that.`;
    },
  };

  // -- the frame of a figure: its selector, its stage, its note, and its dots ------------------------

  const frames = figures.map((figure) => {
    const name = figure.dataset.figure;
    const stage = figure.querySelector(".fixlip-figure-stage");
    const note = el("p", "fixlip-figure-note");
    note.setAttribute("aria-live", "polite");
    stage.after(note);
    return { figure, name, stage, note, follows: !STATIC.has(name), width: -1, buttons: [], dots: [] };
  });

  function choose(id, frame) {
    if (id === current) return;
    document.dispatchEvent(new CustomEvent(EVENT, { detail: { image: id, source: frame?.figure ?? null } }));
  }

  function buildSelector(frame) {
    const bar = el("div", "fixlip-figure-bar");
    const label = el("span", "fixlip-figure-shown");
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
    add(bar, label, photos);
    frame.stage.before(bar);
    frame.note.after(dots);
    frame.label = label;
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
      choose(data.images[at + (dx < 0 ? 1 : -1)].id, frame);
    };
    stage.addEventListener("touchend", (event) => endSwipe(event, false));
    stage.addEventListener("touchcancel", (event) => endSwipe(event, true));
  }

  function render(frame) {
    const { stage, note, name } = frame;
    const draw = FIGURES[name];
    if (!draw) return;
    stage.replaceChildren();
    const c = frame.follows ? contextOf(current) : null;
    let text = "";
    try {
      text = draw(stage, c) ?? "";
      drawLinks(stage);
    } catch (error) {
      console.error(`The figure "${name}" could not be drawn:`, error);
      stage.replaceChildren(el("p", "fixlip-figure-missing", stage.getAttribute("aria-label") ?? ""));
    }
    note.replaceChildren(...rich(text));
    note.hidden = !text;
    if (frame.follows) {
      frame.label.textContent = c.image.title ?? "";
      data.images.forEach((image, k) => {
        frame.buttons[k].setAttribute("aria-pressed", String(image.id === current));
        frame.dots[k].setAttribute("aria-pressed", String(image.id === current));
      });
    }
    frame.figure.classList.add("fixlip-figure-ready");
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
      for (const frame of frames) {
        if (!frame.follows) continue;
        render(frame);
        if (reducedMotion()) continue;
        const from = frame.entering ? `translateX(${frame.entering * 32}px)` : "none";
        frame.entering = 0;
        frame.stage.animate([{ opacity: 0.2, transform: from }, { opacity: 1, transform: "none" }], { duration: 260, easing: "cubic-bezier(0.2, 0.7, 0.2, 1)" });
      }
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
