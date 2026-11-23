/* feature-xai: feature-based explainable AI on the slide of Definition 1 (talks/beyond-shap/index.rst).

   Two rows of small pictures, made of the examples that come back later in the talk.
   Features (step 0): one California house among the rows of its table (the house of the orders
   demo), the photo of the giraffe in its 8 x 8 patches and its caption in its 19 tokens (the
   giraffe of the FIxLIP and orders demos), a week's electricity load in windows, and the molecule
   of the orders demo (GraphSHAP-IQ) with its atoms as nodes. One feature of each is marked.
   Explanations (shown together with the features): the Shapley values of the house (order 1 of the
   orders demo), the Shapley interactions of the giraffe's patches and words (order 2), the
   partial dependence of MedInc with a rug of the data, and the permutation importance of the
   California features. The last two are plausible, not computed. MedInc, marked in the table,
   is marked in all of them.

   The photo is the FIxLIP demo's (_static/demos/fixlip/giraffe-tree.jpg, from data-root). The
   numbers below are copied from demos/orders/assets/data.js (examples "house", "giraffe",
   "molecule"), so that this file loads nothing. */
(function () {
  "use strict";
  const figures = (window.SHAPIQ_TALK_FIGURES = window.SHAPIQ_TALK_FIGURES || {});

  // ---- the data ------------------------------------------------------------------------------
  const COLUMNS = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"];
  const ROWS = [
    // the first two rows of the California housing data
    ["8.33", "41", "7.0", "1.0", "322", "2.6", "37.88", "−122.23"],
    ["8.30", "21", "6.2", "1.0", "2401", "2.1", "37.86", "−122.22"],
  ];
  const HOUSE = ["4.64", "38", "6.3", "1.0", "499", "3.0", "33.94", "−118.32"]; // the orders demo's house
  // its Shapley values (XGBoost, TreeExplainer), in $100k: 2.04 on average, 2.21 for this house
  const HOUSE_SHAP = [0.19853, -0.0043, 0.1717, -0.0109, -0.05373, -0.12727, 0.31483, -0.32149];
  // permutation importance of the California features (increase of the squared error), plausible
  const PFI = [0.61, 0.06, 0.05, 0.01, 0.01, 0.19, 0.88, 0.77];

  const TOKENS = ["a", "giraffe", "has", "its", "front", "legs", "spread", "to", "bend", "down", "for", "a", "drink", "of", "water", "at", "a", "pond", "."];
  const GRID = 8; // players 0..63 are the patches (row by row), 64.. the tokens
  const MARK_PATCH = 20; // the patch of the largest Shapley value
  // the strongest Shapley interactions of order 2 between a patch and a word, and between two words
  const GIRAFFE_PAIRS = [
    [28, 65, 0.01113], [20, 65, 0.00772], [9, 65, 0.00616], [44, 65, 0.00604], [44, 78, 0.00155], [52, 81, 0.00138],
    [78, 81, -0.01541], [65, 69, -0.01077], [65, 81, -0.00923], [76, 78, -0.0068],
  ];
  const PAIR_WORDS = [65, 69, 76, 78, 81]; // giraffe, legs, drink, water, pond

  const ATOMS = "CCCCCCCCCNCCCCOONCNOOOOHHHHHHH".split("");
  const BONDS = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 6], [2, 7], [3, 8], [3, 9], [4, 10], [4, 11], [5, 12], [5, 23], [6, 10], [6, 24], [7, 13], [7, 25], [8, 13], [8, 26], [9, 14], [9, 15], [10, 16], [11, 17], [11, 27], [12, 17], [12, 18], [13, 28], [16, 19], [16, 20], [17, 29], [18, 21], [18, 22]];
  const COORDS = [[0.5835, 0.3688], [0.4447, 0.3851], [0.667, 0.4809], [0.6388, 0.2405], [0.3893, 0.5135], [0.3612, 0.273], [0.6117, 0.6093], [0.8058, 0.4647], [0.7777, 0.2242], [0.5553, 0.1284], [0.4728, 0.6256], [0.2505, 0.5297], [0.2223, 0.2893], [0.8612, 0.3363], [0.6107, 0], [0.4636, 0.1279], [0.4175, 0.7539], [0.167, 0.4176], [0.1388, 0.1772], [0.2787, 0.7702], [0.501, 0.866], [0, 0.1934], [0.1942, 0.0488], [0.3869, 0.185], [0.6952, 0.7214], [0.8893, 0.5768], [0.833, 0.0958], [0.1952, 0.6581], [1, 0.32], [0.0282, 0.4339]];
  const MARK_ATOM = 18; // a nitrogen of a nitro group, of the largest Shapley value

  // a fixed draw, so that the figure is the same at every load
  const random = (seed) => () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

  figures["feature-xai"] = (figure, { make }) => {
    figure.classList.add("talk-fx");
    const photo = `${figure.dataset.root || ""}../../demos/fixlip/giraffe-tree.jpg`;
    const font = getComputedStyle(figure).fontFamily || "sans-serif";
    const measure = document.createElement("canvas").getContext("2d");
    const width = (text, size, weight = 400) => {
      measure.font = `${weight} ${size}px ${font}`;
      return measure.measureText(text).width;
    };

    const row = (title, cls, from) => {
      const block = make("div", { class: `talk-fx-row ${cls}${from ? ` talk-from-${from}` : ""}` }, figure);
      make("div", { class: "talk-fx-title" }, block, title);
      return make("div", { class: "talk-fx-cards" }, block);
    };
    const card = (parent, cls, name, note, tag) => {
      const element = make("div", { class: `talk-fx-card ${cls}` }, parent);
      const visual = make("div", { class: "talk-fx-visual" }, element);
      const caption = make("div", { class: "talk-fx-caption" }, element);
      make("strong", {}, caption, name);
      make("span", {}, caption, note);
      if (tag) make("em", { class: `talk-fx-tag talk-fx-tag-${tag}` }, caption, tag);
      return visual;
    };
    const svg = (parent, w, h) => make("svg:svg", { viewBox: `0 0 ${w} ${h}`, class: "talk-fx-svg", "aria-hidden": "true" }, parent);
    const text = (parent, x, y, value, cls = "talk-fx-label", anchor = "start") =>
      make("svg:text", { x, y, class: cls, "text-anchor": anchor }, parent, value);
    const line = (parent, x1, y1, x2, y2, cls) => make("svg:line", { x1, y1, x2, y2, class: cls }, parent);

    // ---- features ------------------------------------------------------------------------------
    const data = row("Features of Five Kinds of Data", "talk-fx-data");

    // a table: columns are the features, the house of the orders demo is its last row
    const table = make("table", { class: "talk-fx-table" }, card(data, "talk-fx-tabular", "Tabular", "8 columns"));
    const head = make("tr", {}, make("thead", {}, table));
    COLUMNS.forEach((name, c) => make("th", { class: c === 0 ? "talk-fx-col-mark" : "" }, head, name));
    const body = make("tbody", {}, table);
    const tableRow = (values, cls) => {
      const tr = make("tr", { class: cls }, body);
      values.forEach((value, c) => make("td", { class: c === 0 ? "talk-fx-col-mark" : "" }, tr, value));
    };
    ROWS.forEach((values) => tableRow(values, "talk-fx-other"));
    tableRow(COLUMNS.map(() => "⋮"), "talk-fx-gap");
    tableRow(HOUSE, "talk-fx-house");

    // the photo in its patches
    {
      const s = svg(card(data, "talk-fx-image", "Image", "64 patches"), 104, 104);
      make("svg:image", { href: photo, x: 2, y: 2, width: 100, height: 100, preserveAspectRatio: "xMidYMid slice" }, s);
      const cell = 100 / GRID;
      for (let k = 1; k < GRID; k++) {
        line(s, 2 + k * cell, 2, 2 + k * cell, 102, "talk-fx-cut");
        line(s, 2, 2 + k * cell, 102, 2 + k * cell, "talk-fx-cut");
      }
      make("svg:rect", { x: 2 + (MARK_PATCH % GRID) * cell, y: 2 + Math.floor(MARK_PATCH / GRID) * cell, width: cell, height: cell, class: "talk-fx-patch-mark" }, s);
    }

    // the caption in its tokens
    {
      const chips = make("div", { class: "talk-fx-tokens" }, card(data, "talk-fx-text", "Text", "19 tokens"));
      TOKENS.forEach((token, k) => make("span", { class: k === 1 ? "talk-fx-token talk-fx-token-mark" : "talk-fx-token" }, chips, token));
    }

    // four days of electricity load, hour by hour, in windows of 12 hours
    {
      const W = 176, H = 104, top = 6, bottom = 86;
      const s = svg(card(data, "talk-fx-series", "Time series", "8 windows"), W, H);
      const next = random(11);
      const hours = 96;
      const load = Array.from({ length: hours }, (_, t) => {
        const h = t % 24;
        const day = Math.floor(t / 24);
        return 0.42 + 0.26 * Math.exp(-(((h - 8.5) / 2.4) ** 2)) + 0.4 * Math.exp(-(((h - 19) / 2.6) ** 2)) + 0.12 * Math.exp(-(((h - 13) / 3) ** 2)) +
          [0, 0.04, -0.03, 0.06][day] * Math.exp(-(((h - 19) / 3) ** 2)) + 0.035 * (next() - 0.5) * 2;
      });
      const x = (t) => 2 + (t / (hours - 1)) * (W - 4);
      const y = (v) => bottom - ((v - 0.3) / 0.9) * (bottom - top);
      for (let k = 0; k < 8; k++) {
        make("svg:rect", { x: x(k * 12) + (k ? 0.75 : 0), y: top - 4, width: x(12) - x(0) - 1.5, height: bottom - top + 6, class: k === 5 ? "talk-fx-window talk-fx-window-mark" : "talk-fx-window" }, s);
      }
      make("svg:polyline", { points: load.map((v, t) => `${x(t).toFixed(1)},${y(v).toFixed(1)}`).join(" "), class: "talk-fx-series-line" }, s);
      ["Mon", "Tue", "Wed", "Thu"].forEach((day, d) => text(s, x(d * 24 + 12), 100, day, "talk-fx-tick talk-fx-tick-large", "middle"));
    }

    // the molecule, its atoms as nodes
    {
      const W = 122, H = 104, m = 7;
      const scale = Math.min((W - 2 * m) / 1, (H - 2 * m) / 0.866);
      const at = ([cx, cy]) => [m + cx * scale + (W - 2 * m - scale) / 2, m + cy * scale + (H - 2 * m - 0.866 * scale) / 2];
      const s = svg(card(data, "talk-fx-graph", "Graph", "30 atoms"), W, H);
      BONDS.forEach(([a, b]) => {
        const [x1, y1] = at(COORDS[a]);
        const [x2, y2] = at(COORDS[b]);
        line(s, x1, y1, x2, y2, "talk-fx-bond");
      });
      ATOMS.forEach((atom, k) => {
        const [cx, cy] = at(COORDS[k]);
        const r = atom === "H" ? 2.6 : atom === "C" ? 4 : 5.2;
        make("svg:circle", { cx, cy, r, class: `talk-fx-atom talk-fx-atom-${atom}${k === MARK_ATOM ? " talk-fx-atom-mark" : ""}` }, s);
        if (atom === "N" || atom === "O") text(s, cx, cy + 2.4, atom, "talk-fx-atom-name", "middle");
      });
    }

    // ---- explanations --------------------------------------------------------------------------
    const shapes = row("Explanations Take Many Shapes", "talk-fx-shapes");
    const W = 210, H = 84;

    // horizontal bars with names, sorted by size, and an axis below with its name under the names
    const bars = (s, values, { left, zero, scale, signed, ticks, name, unit }) => {
      const order = values.map((v, i) => i).sort((a, b) => Math.abs(values[b]) - Math.abs(values[a]));
      const step = 8.5;
      order.forEach((i, k) => {
        const value = values[i];
        const y = 1 + k * step;
        text(s, left - 4, y + 6.2, COLUMNS[i], i === 0 ? "talk-fx-label talk-fx-label-mark" : "talk-fx-label", "end");
        const w = Math.max(Math.abs(value) * scale, 0.8);
        const cls = signed ? (value > 0 ? "talk-pos" : "talk-neg") : i === 0 ? "talk-fx-bar-mark" : "talk-fx-bar";
        make("svg:rect", { x: value >= 0 ? zero : zero - w, y, width: w, height: step - 2, rx: 1, class: cls }, s);
      });
      const base = 1 + values.length * step + 1;
      line(s, zero, 0, zero, base, "talk-fx-zero");
      line(s, left, base, W - 2, base, "talk-fx-axis");
      for (const tick of ticks) {
        const x = zero + tick * scale;
        line(s, x, base, x, base + 2.5, "talk-fx-axis");
        if (x > left + 1) text(s, x, base + 10.5, unit(tick), "talk-fx-tick", "middle");
      }
      text(s, left - 4, base + 10.5, name, "talk-fx-axis-name", "end");
    };

    // the house's Shapley values, in $k
    bars(svg(card(shapes, "talk-fx-shap", "SHAP", "a score per feature", "local"), W, H), HOUSE_SHAP, {
      left: 54, zero: 54 + (W - 56) / 2, scale: (W - 60) / 2 / 0.34, signed: true, ticks: [-0.3, 0, 0.3],
      name: "Shapley value", unit: (v) => (v > 0 ? "+$" : v < 0 ? "−$" : "") + Math.round(Math.abs(v) * 100) + (v ? "k" : ""),
    });

    // the giraffe's interactions: patches of the photo and words of the caption, linked by their
    // pair's value, the words' pairs as arcs right of the words
    {
      const s = svg(card(shapes, "talk-fx-pairs", "Interactions", "a score per group", "local"), W, H);
      const side = 82, cell = side / GRID, top = 1;
      make("svg:image", { href: photo, x: 0, y: top, width: side, height: side, preserveAspectRatio: "xMidYMid slice" }, s);
      for (let k = 1; k < GRID; k++) {
        line(s, k * cell, top, k * cell, top + side, "talk-fx-cut");
        line(s, 0, top + k * cell, side, top + k * cell, "talk-fx-cut");
      }
      const size = 8.5, pad = 4, chipX = 128, step = 16.5;
      const words = PAIR_WORDS.map((player, k) => ({ player, token: TOKENS[player - 64], y: top + 8 + k * step }));
      words.forEach((word) => (word.w = width(word.token, size) + 2 * pad));
      const chipRight = chipX + Math.max(...words.map((word) => word.w));
      const find = (player) => words.find((word) => word.player === player);
      const strength = (value) => 0.7 + (Math.abs(value) / 0.0155) * 2.8;
      const links = make("svg:g", {}, s);
      const patches = new Set();
      GIRAFFE_PAIRS.forEach(([a, b, value]) => {
        const cls = `talk-fx-link ${value > 0 ? "talk-pos-line" : "talk-neg-line"}`;
        if (a < GRID * GRID) {
          patches.add(a);
          const [x1, y1] = [((a % GRID) + 0.5) * cell, top + (Math.floor(a / GRID) + 0.5) * cell];
          const word = find(b);
          make("svg:path", { d: `M${x1},${y1} C${x1 + 30},${y1} ${chipX - 26},${word.y} ${chipX},${word.y}`, class: cls, "stroke-width": strength(value) }, links);
        } else {
          const [p, q] = [find(a), find(b)];
          const reach = 6 + Math.abs(q.y - p.y) * 0.32;
          make("svg:path", { d: `M${chipX + p.w},${p.y} C${chipRight + reach},${p.y} ${chipRight + reach},${q.y} ${chipX + q.w},${q.y}`, class: cls, "stroke-width": strength(value) }, links);
        }
      });
      patches.forEach((p) => make("svg:rect", { x: (p % GRID) * cell, y: top + Math.floor(p / GRID) * cell, width: cell, height: cell, class: "talk-fx-patch-on" }, s));
      words.forEach((word) => {
        make("svg:rect", { x: chipX, y: word.y - 6.5, width: word.w, height: 13, rx: 6.5, class: word.player === 65 ? "talk-fx-chip talk-fx-chip-mark" : "talk-fx-chip" }, s);
        text(s, chipX + word.w / 2, word.y + 3, word.token, "talk-fx-word", "middle");
      });
    }

    // partial dependence of MedInc, with a rug of the data, and the house on the curve
    {
      const s = svg(card(shapes, "talk-fx-pdp", "PDP", "a curve per feature", "global"), W, H);
      const left = 26, right = W - 4, top = 3, bottom = 62;
      const pdp = (v) => 0.9 + 3.2 / (1 + Math.exp(-(v - 4.6) / 1.9));
      const x = (v) => left + (v / 15) * (right - left);
      const y = (v) => bottom - ((v - 0.5) / 4) * (bottom - top);
      for (const tick of [1, 2, 3, 4]) {
        line(s, left, y(tick), right, y(tick), "talk-fx-grid");
        text(s, left - 3, y(tick) + 2.7, String(tick), "talk-fx-tick", "end");
      }
      const next = random(5);
      for (let k = 0; k < 140; k++) {
        // incomes of the data: about log-normal, median near 3.5
        const g = Math.sqrt(-2 * Math.log(next() + 1e-9)) * Math.cos(2 * Math.PI * next());
        const v = Math.min(15, Math.max(0.5, Math.exp(Math.log(3.5) + 0.48 * g)));
        line(s, x(v), bottom - 4.5, x(v), bottom, "talk-fx-rug");
      }
      make("svg:polyline", { points: Array.from({ length: 61 }, (_, k) => `${x(0.5 + (k / 60) * 14.5).toFixed(1)},${y(pdp(0.5 + (k / 60) * 14.5)).toFixed(1)}`).join(" "), class: "talk-fx-curve" }, s);
      make("svg:circle", { cx: x(4.64), cy: y(pdp(4.64)), r: 2.6, class: "talk-fx-house-dot" }, s);
      line(s, left, bottom, right, bottom, "talk-fx-axis");
      for (const tick of [0, 5, 10, 15]) {
        line(s, x(tick), bottom, x(tick), bottom + 2.5, "talk-fx-axis");
        text(s, x(tick), bottom + 10.5, String(tick), "talk-fx-tick", "middle");
      }
      text(s, (left + right) / 2, H - 0.5, "MedInc ($10k)", "talk-fx-axis-name", "middle");
      const yName = text(s, 0, 0, "Value ($100k)", "talk-fx-axis-name", "middle");
      yName.setAttribute("transform", `translate(6 ${(top + bottom) / 2}) rotate(-90)`);
    }

    // permutation importance of the California features
    bars(svg(card(shapes, "talk-fx-pfi", "PFI", "importance for the model", "global"), W, H), PFI, {
      left: 54, zero: 54, scale: (W - 62) / 0.9, signed: false, ticks: [0, 0.4, 0.8],
      name: "Error increase", unit: (v) => String(v),
    });

    return { steps: 1 };
  };
})();
