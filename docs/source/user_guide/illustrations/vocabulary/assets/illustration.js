// The illustration "vocabulary" (chapter 2.1 of the user guide): features are the universal
// language of machine learning models. Four inputs, a row of a table, a photo, a sentence, and a
// molecule, split into their features: the columns, the patches, the tokens, and the atoms. Each
// feature flies out of its place in the input as a chip, its number on the place it left, and pulses
// carry the features into one model f, which predicts. Whatever the input, the model reads features, and an
// explanation speaks about them. When the illustration is first seen, it cycles through the four
// inputs until the reader picks one; pointing at a chip or at a part of the input lights up the
// pair. With reduced motion, nothing flies and nothing cycles.
//
// The house is the first of the California housing data, and its price is the prediction of the
// model of the illustration units-and-forms (../units-and-forms/compute/compute.py). The other
// predictions only name what such a model predicts.
window.shapiqIllustrations?.define("vocabulary", (stage, kit) => {
  const { el, svg, tween, ease, reducedMotion, watchVisibility, segmented } = kit;
  const uid = `voc-${Math.random().toString(36).slice(2, 8)}`;
  const PRICE = 4.469; // the house's predicted price, in $100,000

  // -- the four inputs ---------------------------------------------------------------------------

  const HOUSES = [
    // MedInc, HouseAge, AveRooms, Population, Latitude, Longitude of the first three houses
    [8.3252, 41, 6.9841, 322, 37.88, -122.23],
    [8.3014, 21, 6.2381, 2401, 37.86, -122.22],
    [7.2574, 52, 8.2881, 496, 37.85, -122.24],
  ];
  const COLUMNS = [
    // head, name, format, unit (for the chip's title and label), as in the California housing data
    ["Income", "Median income", (v) => v.toFixed(2), "in $10,000, of the block"],
    ["Age", "House age", (v) => v.toFixed(0), "in years, the median of the block"],
    ["Rooms", "Rooms", (v) => v.toFixed(2), "per household, on average"],
    ["Pop.", "Population", (v) => v.toFixed(0), "of the block"],
    ["Lat.", "Latitude", (v) => v.toFixed(2), "in degrees north"],
    ["Lon.", "Longitude", (v) => v.toFixed(2).replace("-", "−"), "in degrees east"],
  ];
  const TOKENS = ["the", "movie", "was", "not", "bad", "at", "all"];
  const ATOMS = [
    // symbol, name, x, y, and where its number sits: a nitrobenzene, as in the mutagenic molecules
    // of MUTAG, numbered as chemists do, the ring from the atom that holds the nitro group
    ...[0, 1, 2, 3, 4, 5].map((k) => {
      const angle = (Math.PI / 3) * k;
      const [x, y] = [112 + 54 * Math.cos(angle), 120 - 54 * Math.sin(angle)];
      // outside the ring, away from its center; the first below its bond to the nitrogen
      const out = k === 0 ? [180, 143] : [x + 24 * Math.cos(angle), y - 24 * Math.sin(angle)];
      return ["C", "carbon", x, y, ...out];
    }),
    ["N", "nitrogen", 216, 120, 216, 146],
    ["O", "oxygen", 256, 84, 284, 84],
    ["O", "oxygen", 256, 156, 284, 156],
  ];
  const BONDS = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [6, 7], [6, 8]];

  const MODALITIES = [
    {
      id: "table",
      label: "Table",
      what: "the columns of a table",
      model: { short: "Boosting", title: "Tree ensemble", icon: "tree" },
      output: ["predicted price", (p) => `$${Math.round(PRICE * 100 * p)}k`],
      build: buildTable,
    },
    {
      id: "image",
      label: "Image",
      what: "the patches of an image",
      model: { short: "ViT", title: "Black box", icon: "box" },
      output: ["predicted class", () => "house"],
      build: buildImage,
    },
    {
      id: "text",
      label: "Text",
      what: "the tokens of a text",
      model: { short: "LM", title: "Black box", icon: "box" },
      output: ["predicted sentiment", () => "positive"],
      build: buildText,
    },
    {
      id: "graph",
      label: "Graph",
      what: "the nodes of a graph",
      model: { short: "GNN", title: "Graph neural network", icon: "graph" },
      output: ["predicted property", () => "mutagenic"],
      build: buildGraph,
    },
  ];

  // The parts of an input carry the numbers of their features, as their chips do: whatever the
  // input, its features are numbered 1, …, d. An HTML number, or an SVG one at (x, y).
  const badge = (k) => {
    const node = el("span", "voc-badge", String(k + 1));
    node.setAttribute("aria-hidden", "true");
    return node;
  };
  const badgeIn = (parent, k, x, y, r) => {
    const group = svg("g", { class: "voc-badge-svg", transform: `translate(${x} ${y})`, "aria-hidden": "true" }, parent);
    svg("circle", { r }, group);
    svg("text", {}, group).textContent = String(k + 1);
    return group;
  };

  // each build function fills the input panel (its one element) and returns the features: per
  // feature, the chip's label and value, its preview, its number on the input (badge), and the
  // parts of the input it lights up (the first is where its chip flies from)
  function buildTable(panel) {
    const table = el("table", "voc-table");
    const numbers = table.createTHead().insertRow();
    numbers.className = "voc-table-index";
    const head = table.tHead.insertRow();
    const parts = COLUMNS.map(() => []);
    const badges = COLUMNS.map((column, k) => {
      const cell = el("th");
      const node = badge(k);
      cell.appendChild(node);
      numbers.appendChild(cell);
      return node;
    });
    COLUMNS.forEach(([short], k) => {
      const th = el("th", "", short);
      head.appendChild(th);
      parts[k].push(th, badges[k]);
    });
    const body = table.createTBody();
    HOUSES.forEach((house, row) => {
      const tr = body.insertRow();
      tr.className = row === 0 ? "voc-table-house" : "voc-table-other";
      house.forEach((value, k) => {
        const td = tr.insertCell();
        td.textContent = COLUMNS[k][2](value);
        if (row === 0) parts[k].unshift(td);
        else parts[k].push(td);
      });
    });
    const more = body.insertRow();
    more.className = "voc-table-more";
    const cell = more.insertCell();
    cell.colSpan = COLUMNS.length;
    cell.textContent = "⋮";
    panel.appendChild(table);
    // the number heads the column: the column is the feature
    return COLUMNS.map(([, name, format, unit], k) => ({
      text: name,
      value: format(HOUSES[0][k]),
      unit,
      badge: badges[k],
      parts: parts[k],
    }));
  }

  function buildImage(panel) {
    const picture = svg("svg", { viewBox: "0 0 300 300", class: "voc-image", role: "img" });
    picture.setAttribute("aria-label", "A drawing of a house, cut into 3 × 3 patches");
    svg("use", { href: `#${uid}-scene`, width: 300, height: 300 }, picture);
    const patches = [];
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        patches.push(
          svg("rect", { x: col * 100, y: row * 100, width: 100, height: 100, class: "voc-patch" }, picture),
        );
      }
    }
    // each patch numbered in its corner
    const badges = patches.map((patch, k) => badgeIn(picture, k, (k % 3) * 100 + 17, Math.floor(k / 3) * 100 + 17, 12));
    panel.appendChild(picture);
    return patches.map((patch, k) => {
      const preview = svg("svg", {
        viewBox: `${(k % 3) * 100} ${Math.floor(k / 3) * 100} 100 100`,
        class: "voc-chip-preview",
        "aria-hidden": "true",
      });
      svg("use", { href: `#${uid}-scene`, width: 300, height: 300 }, preview);
      return { text: `patch ${k + 1}`, value: "", preview, badge: badges[k], parts: [patch, badges[k]] };
    });
  }

  function buildText(panel) {
    const sentence = el("p", "voc-sentence");
    // each token numbered under it
    const tokens = TOKENS.map((token, k) => {
      const span = el("span", "voc-token", token);
      span.appendChild(badge(k));
      sentence.appendChild(span);
      return span;
    });
    panel.appendChild(sentence);
    return tokens.map((span, k) => ({
      text: `“${TOKENS[k]}”`,
      value: "",
      badge: span.lastChild,
      parts: [span],
    }));
  }

  function buildGraph(panel) {
    const graph = svg("svg", { viewBox: "0 0 300 240", class: "voc-graph", role: "img" });
    graph.setAttribute("aria-label", "A molecule as a graph: a ring of six carbon atoms and a nitro group");
    svg("circle", { cx: 112, cy: 120, r: 30, class: "voc-ring" }, graph);
    for (const [a, b] of BONDS) {
      const [, , x1, y1] = ATOMS[a];
      const [, , x2, y2] = ATOMS[b];
      svg("line", { x1, y1, x2, y2, class: "voc-bond" }, graph);
    }
    // the double bond of N=O, a second line beside the first
    svg("line", { x1: 222, y1: 108, x2: 250, y2: 79, class: "voc-bond voc-bond-second" }, graph);
    const atoms = ATOMS.map(([symbol, , x, y]) => {
      const group = svg("g", { class: "voc-atom", transform: `translate(${x} ${y})` }, graph);
      svg("circle", { r: 15 }, group);
      svg("text", {}, group).textContent = symbol;
      return group;
    });
    // each atom numbered beside it, as in a drawing of chemistry
    const badges = ATOMS.map(([, , , , bx, by], k) => badgeIn(graph, k, bx, by, 10));
    panel.appendChild(graph);
    return atoms.map((group, k) => ({
      text: ATOMS[k][0],
      value: ATOMS[k][1],
      badge: badges[k],
      parts: [group, badges[k]],
    }));
  }

  // the scene of the image, once per illustration, so that the chips can show their patches
  const defs = svg("svg", { class: "voc-defs", "aria-hidden": "true", width: 0, height: 0 });
  defs.innerHTML = `
    <defs>
      <linearGradient id="${uid}-sky" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#a9d8f5"/><stop offset="1" stop-color="#eaf6fd"/>
      </linearGradient>
    </defs>
    <symbol id="${uid}-scene" viewBox="0 0 300 300">
      <rect width="300" height="300" fill="url(#${uid}-sky)"/>
      <g fill="#ffffff" opacity="0.92">
        <ellipse cx="70" cy="70" rx="34" ry="14"/><ellipse cx="95" cy="62" rx="26" ry="16"/>
        <ellipse cx="48" cy="66" rx="20" ry="12"/><ellipse cx="236" cy="48" rx="22" ry="9"/>
      </g>
      <path d="M0 190 Q 60 148 120 178 T 240 168 T 300 182 V 300 H 0 Z" fill="#a3bccb"/>
      <path d="M0 214 Q 150 198 300 214 V 300 H 0 Z" fill="#8fbf8a"/>
      <path d="M150 236 Q 142 270 118 300 H 178 Q 166 270 162 236 Z" fill="#ddd2b9"/>
      <rect x="183" y="112" width="13" height="30" fill="#5a6078"/>
      <rect x="105" y="150" width="100" height="86" fill="#f6f3ec" stroke="#4b5068" stroke-width="2"/>
      <polygon points="94,156 155,104 216,156" fill="#5a6078"/>
      <rect x="145" y="196" width="22" height="40" rx="2" fill="#8b6b55"/>
      <rect x="117" y="170" width="22" height="20" fill="#cfe3f7" stroke="#4b5068" stroke-width="1.5"/>
      <rect x="173" y="170" width="22" height="20" fill="#cfe3f7" stroke="#4b5068" stroke-width="1.5"/>
      <rect x="246" y="186" width="9" height="46" fill="#7a5b45"/>
      <circle cx="250" cy="172" r="26" fill="#5f9e69"/><circle cx="236" cy="184" r="16" fill="#6aab74"/>
    </symbol>`;

  // the models' icons, as the orders demo draws them (demos/orders/assets/demo.js): a decision tree
  // for tree ensembles, a graph for graph neural networks, and a black box for any other model
  const ICONS = {
    tree:
      '<rect x="11" y="3" width="10" height="6" rx="1.5"/><path d="M16 9v3.5M7 16v-3.5h17V16"/>' +
      '<circle class="voc-icon-leaf" cx="7" cy="19" r="3"/><rect x="19" y="16" width="10" height="6" rx="1.5"/>' +
      '<path d="M24 22v3M19.5 27v-2h9v2"/><circle class="voc-icon-leaf" cx="19.5" cy="29" r="2"/>' +
      '<circle class="voc-icon-leaf" cx="28.5" cy="29" r="2"/>',
    graph:
      '<path d="M9 7l14 2 5 9-6 9-13-1-5-9zM9 7l7 9 12 2M16 16l-7 10"/>' +
      [[9, 7], [23, 9], [28, 18], [22, 27], [9, 26], [4, 17], [16, 16]]
        .map(([x, y]) => `<circle class="voc-icon-node" cx="${x}" cy="${y}" r="2.6"/>`)
        .join(""),
    box:
      '<path class="voc-icon-side" d="M4 10l12 6v13L4 23z"/>' +
      '<path class="voc-icon-side voc-icon-right" d="M28 10l-12 6v13l12-6z"/>' +
      '<path class="voc-icon-side voc-icon-top" d="M16 4l12 6-12 6-12-6z"/>',
  };
  const icon = (name) => {
    const node = svg("svg", { viewBox: "0 0 32 32", "aria-hidden": "true", class: `voc-icon voc-icon-${name}` });
    node.innerHTML = ICONS[name];
    return node;
  };
  const label = (html) => {
    const node = el("p", "voc-column-label");
    node.innerHTML = html;
    return node;
  };

  // -- the layout ----------------------------------------------------------------------------------

  const root = el("div", "voc");
  const head = el("div", "voc-head");
  const tagline = el("p", "voc-tagline");
  const switcher = segmented(
    "Input",
    MODALITIES.map((m) => [m.id, m.label]),
    (id, byUser) => {
      if (!byUser) return;
      stopAutoplay();
      show(MODALITIES.findIndex((m) => m.id === id));
    },
  );
  head.append(switcher.node, tagline);

  const body = el("div", "voc-body");
  const inputColumn = el("div", "voc-input-column");
  const input = el("div", "voc-input");
  const INPUT_LABEL = 'input <span class="shapiq-ill-math">x</span>';
  inputColumn.append(label(INPUT_LABEL), input);
  const features = el("div", "voc-features");
  const chips = el("div", "voc-chips");
  features.append(label("features"), chips);
  const model = el("div", "voc-model");
  const box = el("div", "voc-model-box");
  const boxIcon = el("span", "voc-model-icon");
  box.append(
    el("span", "voc-tag shapiq-ill-math", "f"),
    boxIcon,
    el("strong", "voc-model-name"),
    el("span", "voc-model-title"),
  );
  const flow = el("div", "voc-flow");
  flow.setAttribute("aria-hidden", "true");
  flow.appendChild(el("span", "voc-flow-dashes"));
  const output = el("div", "voc-output");
  output.append(
    el("span", "voc-tag shapiq-ill-math", "f(x)"),
    el("span", "voc-output-label"),
    el("span", "voc-output-value"),
  );
  model.append(label('model <span class="shapiq-ill-math">f</span>'), box, flow, output);
  // the prediction, and the flowing dashes to it, come and go together
  const showPrediction = (shown) => {
    output.classList.toggle("is-shown", shown);
    flow.classList.toggle("is-shown", shown);
  };
  const wires = svg("svg", { class: "voc-wires", "aria-hidden": "true" });
  body.append(inputColumn, features, model, wires);

  const foot = el("p", "voc-foot");
  foot.innerHTML =
    'Different inputs, one vocabulary: the features <span class="voc-set"><i>D</i>&#8202;=&#8202;{1,&#8202;…,&#8202;<span class="voc-count"></span>}</span>.';
  root.append(head, body, foot, defs);
  stage.appendChild(root);

  // -- pointing at a feature -----------------------------------------------------------------------

  let current = { features: [], chips: [], wires: [] };
  let pinned = null;
  let active = null; // the feature pointed at or pinned, which the wires keep when redrawn
  const setActive = (k) => {
    active = k;
    current.features.forEach((feature, j) => {
      const on = j === k;
      for (const part of feature.parts) part.classList.toggle("is-active", on);
      current.chips[j]?.classList.toggle("is-active", on);
      for (const wire of current.wires[j] ?? []) wire.classList.toggle("is-active", on);
    });
    root.classList.toggle("voc-pointing", k !== null);
  };
  const point = (k) => setActive(k ?? pinned);

  // -- wires ---------------------------------------------------------------------------------------

  // The wires from the chips into the model, which the pulses run along. Wide, they run to the
  // right, narrow, down. (The input is not wired to its chips: its parts carry their numbers.)
  const curve = ([x1, y1], [x2, y2], horizontal) => {
    if (horizontal) {
      const mid = (x1 + x2) / 2;
      return `M${x1} ${y1} C${mid} ${y1} ${mid} ${y2} ${x2} ${y2}`;
    }
    const mid = (y1 + y2) / 2;
    return `M${x1} ${y1} C${x1} ${mid} ${x2} ${mid} ${x2} ${y2}`;
  };

  // draws the wires; returns them, for the pulses
  const drawWires = (animate) => {
    const base = body.getBoundingClientRect();
    wires.setAttribute("viewBox", `0 0 ${base.width} ${base.height}`);
    wires.replaceChildren();
    const boxRect = box.getBoundingClientRect();
    const horizontal = features.getBoundingClientRect().left > input.getBoundingClientRect().right - 4;
    const modelIn = horizontal
      ? [boxRect.left - base.left, boxRect.top + boxRect.height / 2 - base.top]
      : [boxRect.left + boxRect.width / 2 - base.left, boxRect.top - base.top];
    const into = current.chips.map((node, k) => {
      const chip = node.getBoundingClientRect();
      const chipOut = horizontal
        ? [chip.right - base.left, chip.top + chip.height / 2 - base.top]
        : [chip.left + chip.width / 2 - base.left, chip.bottom - base.top];
      const wire = svg("path", { d: curve(chipOut, modelIn, horizontal), class: "voc-wire" }, wires);
      if (animate && !reducedMotion()) {
        const length = wire.getTotalLength();
        wire.style.strokeDasharray = `${length}`;
        wire.animate([{ strokeDashoffset: length }, { strokeDashoffset: 0 }], {
          duration: 520,
          delay: 28 * k,
          easing: "cubic-bezier(0.4, 0, 0.2, 1)",
          fill: "backwards",
        });
      }
      return wire;
    });
    current.wires = into.map((wire) => [wire]);
    setActive(active);
    return into;
  };

  // -- showing an input ------------------------------------------------------------------------------

  let run = 0;
  let index = -1;
  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, reducedMotion() ? 0 : ms));
  const frame = () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));

  const chipOf = (feature, k) => {
    const chip = el("button", "shapiq-ill-chip voc-chip");
    chip.type = "button";
    chip.appendChild(el("span", "shapiq-ill-chip-index", String(k + 1)));
    if (feature.preview) chip.appendChild(feature.preview);
    const text = el("span", "shapiq-ill-chip-text", feature.text);
    if (feature.value) text.append(" ", el("span", "shapiq-ill-chip-value", feature.value));
    chip.appendChild(text);
    return chip;
  };

  const makeChip = (feature, k) => {
    const chip = chipOf(feature, k);
    const described = `${feature.text} ${feature.value} ${feature.unit ?? ""}`.trim();
    chip.setAttribute("aria-label", `Feature ${k + 1}: ${described}`);
    if (feature.unit) chip.title = described;
    chip.setAttribute("aria-pressed", "false");
    chip.addEventListener("pointerenter", () => point(k));
    chip.addEventListener("pointerleave", () => point(null));
    chip.addEventListener("focus", () => point(k));
    chip.addEventListener("blur", () => point(null));
    // a click pins the feature (or unpins it): the chip stays pressed, and its pair lit
    chip.addEventListener("click", () => {
      stopAutoplay();
      pinned = pinned === k ? null : k;
      current.chips.forEach((other, j) => {
        other.classList.toggle("is-selected", j === pinned);
        other.setAttribute("aria-pressed", String(j === pinned));
      });
      point(k);
    });
    for (const part of feature.parts) {
      part.addEventListener("pointerenter", () => point(k));
      part.addEventListener("pointerleave", () => point(null));
    }
    return chip;
  };

  const show = async (next, { intro = false } = {}) => {
    const token = ++run;
    const alive = () => token === run;
    clearTimeout(timer);
    const modality = MODALITIES[next];
    const moving = !reducedMotion();
    waiting = false;
    // the parts are not numbered yet: their numbers come with the chips
    input.classList.toggle("voc-unnamed", moving);
    pinned = null;
    setActive(null);
    box.classList.remove("is-reading");

    // the old input and its chips leave
    if (index >= 0 && moving && !intro) {
      [...chips.children].forEach((chip, k) =>
        chip.animate([{ opacity: 1, transform: "none" }, { opacity: 0, transform: "scale(0.6)" }], {
          duration: 200,
          delay: 16 * k,
          easing: "ease-in",
          fill: "forwards",
        }),
      );
      for (const node of [input, wires, output, flow]) {
        node.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 200, fill: "forwards" });
      }
      await wait(260);
      if (!alive()) return;
    }
    index = next;
    switcher.set(modality.id);
    tagline.innerHTML = `The model reads <strong>${modality.what}</strong>.`;
    boxIcon.replaceChildren(icon(modality.model.icon));
    box.querySelector(".voc-model-name").textContent = modality.model.short;
    box.querySelector(".voc-model-title").textContent = modality.model.title;
    box.classList.toggle("voc-model-black", modality.model.icon === "box");
    output.querySelector(".voc-output-label").textContent = modality.output[0];
    output.querySelector(".voc-output-value").textContent = modality.output[1](1);
    showPrediction(false);
    input.replaceChildren();
    chips.replaceChildren();
    wires.replaceChildren();
    for (const node of [input, wires, output, flow]) node.getAnimations().forEach((a) => a.cancel());

    const built = modality.build(input);
    const made = built.map(makeChip);
    chips.append(...made);
    current = { features: built, chips: made, wires: [] };
    chips.style.visibility = "";
    const count = root.querySelector(".voc-count");
    count.textContent = String(built.length);

    if (!moving) {
      input.classList.remove("voc-unnamed");
      drawWires(false);
      showPrediction(true);
      return;
    }

    // the input appears, and its features fly out of it into their chips
    input.animate([{ opacity: 0, transform: "scale(0.97)" }, { opacity: 1, transform: "none" }], {
      duration: 380,
      easing: "cubic-bezier(0.2, 0.7, 0.2, 1)",
    });
    for (const chip of made) chip.style.opacity = "0";
    await frame();
    if (!alive()) return;
    const flight = 640;
    const gap = 70;
    made.forEach((chip, k) => {
      const from = built[k].parts[0].getBoundingClientRect();
      const to = chip.getBoundingClientRect();
      const dx = from.left + from.width / 2 - (to.left + to.width / 2);
      const dy = from.top + from.height / 2 - (to.top + to.height / 2);
      chip.style.opacity = "";
      chip.animate(
        [
          { transform: `translate(${dx}px, ${dy}px) scale(0.3)`, opacity: 0 },
          { transform: `translate(${dx * 0.15}px, ${dy * 0.15}px) scale(1.06)`, opacity: 1, offset: 0.7 },
          { transform: "none", opacity: 1 },
        ],
        { duration: flight, delay: 140 + gap * k, easing: "cubic-bezier(0.2, 0.8, 0.3, 1)", fill: "backwards" },
      );
      for (const part of built[k].parts) {
        part.animate([{ filter: "none" }, { filter: "brightness(1.25) saturate(1.3)" }, { filter: "none" }], {
          duration: 420,
          delay: 100 + gap * k,
        });
      }
      built[k].badge.animate(
        [
          { opacity: 0, transform: "scale(0.2)" },
          { opacity: 1, transform: "scale(1.3)", offset: 0.6 },
          { opacity: 1, transform: "none" },
        ],
        { duration: 420, delay: 90 + gap * k, easing: "cubic-bezier(0.3, 1.4, 0.5, 1)", fill: "backwards" },
      );
    });
    input.classList.remove("voc-unnamed");
    await wait(140 + gap * made.length + flight - 200);
    if (!alive()) return;

    // the wires connect them, and pulses carry the features into the model
    const into = drawWires(true);
    await wait(520);
    if (!alive()) return;
    const finish = () => {
      box.classList.remove("is-reading");
      box.animate([{ transform: "scale(1)" }, { transform: "scale(1.07)" }, { transform: "scale(1)" }], {
        duration: 420,
        easing: "cubic-bezier(0.3, 1.4, 0.5, 1)",
      });
      showPrediction(true);
      const value = output.querySelector(".voc-output-value");
      output.animate([{ opacity: 0, transform: "translateY(6px) scale(0.9)" }, { opacity: 1, transform: "none" }], {
        duration: 460,
        easing: "cubic-bezier(0.3, 1.4, 0.5, 1)",
      });
      tween(700, (p) => {
        value.textContent = modality.output[1](p);
      });
      scheduleNext();
    };
    box.classList.add("is-reading");
    let arrived = 0;
    into.forEach((path, k) => {
      const pulse = svg("circle", { r: 3.4, class: "voc-pulse" }, wires);
      const length = path.getTotalLength();
      tween(
        560,
        (p) => {
          const point = path.getPointAtLength(p * length);
          pulse.setAttribute("cx", point.x);
          pulse.setAttribute("cy", point.y);
          pulse.setAttribute("opacity", p < 0.92 ? 1 : (1 - p) / 0.08);
        },
        {
          easing: ease.inOut,
          delay: 45 * k,
          done: () => {
            pulse.remove();
            arrived += 1;
            if (arrived === into.length && alive()) finish();
          },
        },
      );
    });
  };

  // -- the room of the tallest input -----------------------------------------------------------------

  // every input gets the room of the tallest, measured on a hidden copy of the three columns at the
  // width of the figure, so that the page does not jump while the inputs cycle
  const probe = el("div", "voc-body voc-probe");
  probe.setAttribute("aria-hidden", "true");
  probe.inert = true;
  root.appendChild(probe);
  let measured = -1;
  const reserve = () => {
    const width = root.clientWidth;
    if (!width || width === measured) return;
    measured = width;
    let tallest = 0;
    for (const modality of MODALITIES) {
      const panel = el("div", "voc-input");
      const column = el("div", "voc-input-column");
      column.append(label(INPUT_LABEL), panel);
      const list = el("div", "voc-chips");
      const listColumn = el("div", "voc-features");
      listColumn.append(label("features"), list);
      list.append(...modality.build(panel).map(chipOf));
      const copy = model.cloneNode(true);
      copy.querySelector(".voc-model-name").textContent = modality.model.short;
      copy.querySelector(".voc-model-title").textContent = modality.model.title;
      copy.querySelector(".voc-model-box").classList.toggle("voc-model-black", modality.model.icon === "box");
      copy.querySelector(".voc-output-label").textContent = modality.output[0];
      copy.querySelector(".voc-output-value").textContent = modality.output[1](1);
      probe.replaceChildren(column, listColumn, copy);
      tallest = Math.max(tallest, probe.getBoundingClientRect().height);
    }
    probe.replaceChildren();
    body.style.minHeight = `${Math.ceil(tallest)}px`;
  };
  reserve();
  document.fonts?.ready.then(() => {
    measured = -1;
    reserve();
  });

  // -- cycling through the inputs ------------------------------------------------------------------

  // An input stays for DWELL, then the next comes. While the reader points at the illustration, it
  // is off-screen, or the tab is hidden, the cycle rests, and when the reader is back, the input
  // stays for a full DWELL again.
  let autoplay = !reducedMotion();
  let visible = false;
  let hovering = false;
  let resting = false;
  let timer = 0;
  const DWELL = 3800;
  const stopAutoplay = () => {
    autoplay = false;
    clearTimeout(timer);
  };
  const scheduleNext = () => {
    clearTimeout(timer);
    resting = false;
    if (!autoplay) return;
    timer = setTimeout(() => {
      if (!autoplay) return;
      if (!visible || hovering || document.hidden) resting = true;
      else show((index + 1) % MODALITIES.length);
    }, DWELL);
  };
  const resume = () => {
    if (resting && visible && !hovering && !document.hidden) scheduleNext();
  };
  stage.addEventListener("pointerenter", () => {
    hovering = true;
  });
  stage.addEventListener("pointerleave", () => {
    hovering = false;
    resume();
  });
  document.addEventListener("visibilitychange", resume);
  stage.addEventListener("focusin", stopAutoplay);

  new ResizeObserver(() => {
    reserve();
    if (current.chips.length && current.wires.length) drawWires(false);
  }).observe(body);

  // The first input waits, without its chips (hidden, so not focusable either), until the
  // illustration is seen; then it plays, unless the reader has picked an input before.
  let waiting = false;
  if (reducedMotion()) {
    show(0);
  } else {
    show(0, { intro: true });
    run += 1; // stop that first run after its setup: it plays when seen
    chips.style.visibility = "hidden";
    input.classList.add("voc-unnamed");
    showPrediction(false);
    waiting = true;
  }
  watchVisibility(stage, (seen) => {
    visible = seen;
    if (seen && waiting) show(index, { intro: true });
    resume();
  });
});
