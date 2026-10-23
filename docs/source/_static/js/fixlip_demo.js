// The interactive explanation of the landing page: which words of a caption go with which regions
// of a photo, for SigLIP 2 and CLIP, explained as in FIxLIP (Baniecki et al., NeurIPS 2025).
//
// The players are the image patches of the photo (7 x 7 for CLIP, 8 x 8 for SigLIP 2, both
// ViT-B/32 as in FIxLIP's figures) and the tokens of the caption. Their main
// effects and pairwise interactions (FBII, FIxLIP with p = 0.5) are precomputed per photo and
// model in _static/fixlip/data.js, so the page only looks values up. At rest, the demo shows the
// clique of FIxLIP's greedy search, the players that together raise the similarity the most, on
// the photo and as a graph of their interactions, as in FIxLIP's figures; or the full explanation,
// every player tinted by its main effect and the strongest interactions drawn between them; or no
// explanation at all, only the photo, the caption, and the model's similarity, for contrast.
// Pointing at a player (hover, keyboard focus, or a tap that pins it) lights up all its
// interactions, red for those that raise the similarity and blue for those that lower it.
//
// Without JavaScript, or when the data does not load, the static figure stays in place.
(() => {
  const root = document.querySelector(".fixlip-demo");
  if (!root) return;

  const SVG = "http://www.w3.org/2000/svg";
  const RED = [255, 13, 87];
  const BLUE = [30, 136, 229];
  const TOP = 5; // the strongest interactions listed for a player
  const STRONGEST = 12; // the interactions drawn in the full explanation
  const VIEWS = [
    ["none", "None", ""],
    ["clique", "Clique", ""],
    ["full", "Full", " explanation"], // the second part goes on small screens
  ];

  const loader = document.createElement("script");
  loader.src = `${root.dataset.root}data.js`;
  loader.onload = () => window.SHAPIQ_FIXLIP && start(window.SHAPIQ_FIXLIP);
  document.head.appendChild(loader);

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const signed = (value) => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(3)}`;
  const solid = (value) => `rgb(${(value >= 0 ? RED : BLUE).join(", ")})`;
  const valued = (value) => Object.assign(el("strong", "", signed(value)), { style: `color: ${solid(value)}` });
  const tint = (value, largest, strength = 0.85) => {
    const [r, g, b] = value >= 0 ? RED : BLUE;
    const alpha = largest > 0 ? Math.min(Math.abs(value) / largest, 1) * strength : 0;
    return { color: `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`, alpha };
  };
  // a region of the photo as a background: the photo at 7 times its size, moved to the region
  const region = (node, url, k, grid) => {
    node.style.backgroundImage = `url("${url}")`;
    node.style.backgroundSize = `${grid * 100}%`;
    node.style.backgroundPosition = `${((k % grid) / (grid - 1)) * 100}% ${(Math.floor(k / grid) / (grid - 1)) * 100}%`;
  };

  function start(data) {
    const has = (image, model) => `${image}/${model}` in data.explanations;
    const first = data.images.find((image) => data.models.some((model) => has(image.id, model.id)));
    if (!first) return;
    const state = {
      image: first.id,
      model: data.models.find((model) => has(first.id, model.id)).id,
      size: 6,
      view: "clique",
      focus: null,
      pinned: false,
    };

    // -- the frame: photos, models, the stage with the photo and the caption, and the panel ------
    root.replaceChildren();
    const bar = el("div", "fixlip-demo-bar");
    const photos = el("div", "fixlip-demo-photos");
    photos.setAttribute("role", "group");
    photos.setAttribute("aria-label", "Photo");
    const views = el("div", "fixlip-demo-switch");
    views.setAttribute("role", "group");
    views.setAttribute("aria-label", "Explanation");
    const models = el("div", "fixlip-demo-switch");
    models.setAttribute("role", "group");
    models.setAttribute("aria-label", "Model");
    const switches = el("div", "fixlip-demo-switches");
    switches.append(views, models);
    bar.append(photos, switches);

    const body = el("div", "fixlip-demo-body");
    const stage = el("div", "fixlip-demo-stage");
    const frame = el("div", "fixlip-demo-image");
    const photo = el("img");
    photo.decoding = "async";
    const cells = el("div", "fixlip-demo-cells");
    frame.append(photo, cells);
    const caption = el("p", "fixlip-demo-caption");
    const edges = document.createElementNS(SVG, "svg");
    edges.setAttribute("class", "fixlip-demo-edges");
    edges.setAttribute("aria-hidden", "true");
    stage.append(frame, caption, edges);

    const panel = el("div", "fixlip-demo-panel");
    const title = el("p", "fixlip-demo-title");
    const detail = el("p", "fixlip-demo-detail");
    // one slot of a fixed height for both views, so that the panel keeps its size: the clique's
    // size and graph at rest, the strongest interactions of a player when pointed at
    const slot = el("div", "fixlip-demo-slot");
    const list = el("ol", "fixlip-demo-list");
    const graph = el("div", "fixlip-demo-graph");
    graph.setAttribute("aria-hidden", "true"); // the panel's text and the photo say the same
    const links = document.createElementNS(SVG, "svg");
    links.setAttribute("class", "fixlip-demo-links");
    const sizer = el("div", "fixlip-demo-size");
    const smaller = el("button", "fixlip-demo-step", "−");
    const larger = el("button", "fixlip-demo-step", "+");
    const sizeLabel = el("span", "fixlip-demo-size-label");
    smaller.type = larger.type = "button";
    smaller.setAttribute("aria-label", "Fewer players in the clique");
    larger.setAttribute("aria-label", "More players in the clique");
    sizer.append(smaller, sizeLabel, larger);
    const legend = el("p", "fixlip-demo-legend");
    legend.innerHTML =
      '<span class="fixlip-demo-key fixlip-demo-key-red"></span>raises the similarity' +
      '<span class="fixlip-demo-key fixlip-demo-key-blue"></span>lowers it';
    const meta = el("p", "fixlip-demo-meta");
    const live = el("p", "fixlip-demo-live");
    live.setAttribute("aria-live", "polite");
    // without an explanation: only the model's prediction, and the photo's credit
    const prediction = el("div", "fixlip-demo-prediction");
    const predictionMain = el("div", "fixlip-demo-prediction-main");
    const predictionModel = el("p", "fixlip-demo-prediction-model");
    const predictionValue = el("p", "fixlip-demo-prediction-value");
    const predictionWhy = el("p", "fixlip-demo-prediction-why");
    const predictionCredit = el("p", "fixlip-demo-meta");
    const toView = (view, label) => {
      const button = el("button", "fixlip-demo-link", label);
      button.type = "button";
      button.addEventListener("click", () => setView(view));
      return button;
    };
    predictionMain.append(
      predictionModel,
      predictionValue,
      el("p", "fixlip-demo-prediction-label", "similarity of the photo and its caption"),
      predictionWhy,
    );
    predictionWhy.append("But why? See the ", toView("clique", "clique"), " or the ", toView("full", "full explanation"), ".");
    prediction.append(predictionMain, predictionCredit);
    slot.append(sizer, graph, list);
    panel.append(title, detail, slot, legend, meta, prediction, live);
    body.append(stage, panel);
    root.append(bar, body);

    const photoButtons = data.images.map((image) => {
      const button = el("button", "fixlip-demo-photo");
      button.type = "button";
      button.setAttribute("aria-label", image.title);
      button.title = image.title;
      const thumb = el("img");
      thumb.src = `${root.dataset.root}${image.id}.jpg`;
      thumb.alt = "";
      thumb.loading = "lazy";
      button.append(thumb);
      button.addEventListener("click", () => {
        state.image = image.id;
        reset();
      });
      photos.append(button);
      return button;
    });
    const viewButtons = VIEWS.map(([view, label, more]) => {
      const button = el("button", "fixlip-demo-option", label);
      if (more) button.append(el("span", "fixlip-demo-more", more));
      button.type = "button";
      button.addEventListener("click", () => setView(view));
      views.append(button);
      return button;
    });
    const modelButtons = data.models.map((model) => {
      const button = el("button", "fixlip-demo-option", model.label);
      button.type = "button";
      button.addEventListener("click", () => {
        state.model = model.id;
        reset();
      });
      models.append(button);
      return button;
    });
    smaller.addEventListener("click", () => setSize(state.size - 1));
    larger.addEventListener("click", () => setSize(state.size + 1));

    // -- the players of the current photo and model --------------------------------------------
    let ex = null; // the explanation: tokens, main effects, interactions, cliques
    let players = []; // the button of every player: the regions, then the tokens
    let floor = 0; // the 99th percentile of the interactions, the least that tints scale to
    let strongest = []; // the strongest interactions, the strongest first, for the full explanation
    let crossModal = []; // the strongest interactions between a region and a token, listed there
    let grid = 7; // the photo's regions are grid x grid patches, by the model
    let regionCount = grid * grid;

    const interaction = (i, j) => {
      if (i === j) return 0;
      const [a, b] = i < j ? [i, j] : [j, i];
      const n = ex.first.length;
      return ex.pairs[a * n - (a * (a + 1)) / 2 + (b - a - 1)]; // the upper triangle, row by row
    };
    const name = (player) =>
      player < regionCount
        ? `the region in row ${Math.floor(player / grid) + 1}, column ${(player % grid) + 1}`
        : `“${ex.tokens[player - regionCount][0]}”`;

    function reset() {
      // a photo without an explanation for the model switches to a model that has one
      if (!has(state.image, state.model)) state.model = data.models.find((m) => has(state.image, m.id)).id;
      ex = data.explanations[`${state.image}/${state.model}`];
      grid = ex.grid ?? 7;
      regionCount = grid * grid;
      cells.style.setProperty("--fixlip-grid", grid);
      const magnitudes = ex.pairs.map(Math.abs).sort((a, b) => a - b);
      floor = magnitudes[Math.floor(0.99 * (magnitudes.length - 1))];
      const n = ex.first.length;
      strongest = [];
      for (let i = 0; i < n; i += 1) for (let j = i + 1; j < n; j += 1) strongest.push({ i, j, value: interaction(i, j) });
      strongest = strongest.sort((x, y) => Math.abs(y.value) - Math.abs(x.value));
      crossModal = strongest.filter(({ i, j }) => i < regionCount && j >= regionCount).slice(0, TOP);
      strongest = strongest.slice(0, STRONGEST);
      for (const pairs of [strongest, crossModal])
        for (const pair of pairs) pair.strength = Math.abs(pair.value) / Math.abs(pairs[0].value);
      const image = data.images.find((x) => x.id === state.image);
      const model = data.models.find((x) => x.id === state.model);
      state.focus = null;
      state.pinned = false;
      photoButtons.forEach((b, k) => b.setAttribute("aria-pressed", String(data.images[k].id === state.image)));
      modelButtons.forEach((b, k) => {
        b.setAttribute("aria-pressed", String(data.models[k].id === state.model));
        b.disabled = !has(state.image, data.models[k].id);
      });
      photoButtons.forEach((b, k) => {
        b.disabled = !data.models.some((m) => has(data.images[k].id, m.id));
      });
      photo.src = `${root.dataset.root}${image.id}.jpg`;
      photo.alt = image.alt;

      cells.replaceChildren();
      caption.replaceChildren();
      players = [];
      for (let k = 0; k < regionCount; k += 1) {
        const cell = el("button", "fixlip-demo-cell");
        cell.type = "button";
        cell.tabIndex = k === 0 ? 0 : -1; // one stop for the grid; the arrow keys move within it
        cell.setAttribute("aria-label", `Interactions of ${name(k)}`);
        cells.append(cell);
        players.push(cell);
      }
      ex.tokens.forEach(([text, space], k) => {
        if (space && k > 0) caption.append(" ");
        const token = el("button", "fixlip-demo-token", text);
        token.type = "button";
        token.setAttribute("aria-label", `Interactions of “${text}”`);
        caption.append(token);
        players.push(token);
      });
      players.forEach((button, player) => {
        button.addEventListener("pointerenter", (event) => {
          if (event.pointerType !== "touch" && !state.pinned && state.view !== "none") show(player);
        });
        button.addEventListener("focus", () => {
          if (!state.pinned && state.view !== "none") show(player);
        });
        button.addEventListener("click", () => {
          if (state.view === "none") return;
          const again = state.pinned && state.focus === player;
          state.pinned = !again;
          show(again ? null : player);
        });
      });
      const credit = image.credit;
      const photoCredit = () => [
        Object.assign(el("a", "", credit.title), { href: credit.url }),
        credit.author ? ` by ${credit.author}, ` : ", ",
        Object.assign(el("a", "", credit.license), { href: credit.licenseUrl }),
        ", cropped.",
      ];
      meta.replaceChildren(
        `${model.label} (${model.detail}): similarity ${ex.similarity.toFixed(2)}. Explained by ${data.method}; ` +
          `the explanation fits held-out coalitions with R² = ${ex.r2.toFixed(2)}. Photo: `,
        ...photoCredit(),
      );
      predictionModel.textContent = `${model.label} (${model.detail}) predicts`;
      predictionValue.textContent = ex.similarity.toFixed(2);
      predictionCredit.replaceChildren("Photo: ", ...photoCredit());
      setInteractive();
      show(null);
    }

    // the players can be pointed at, unless there is no explanation to show
    function setInteractive() {
      players.forEach((button, k) => {
        button.tabIndex = state.view === "none" || (k < regionCount && k > 0) ? -1 : 0;
      });
    }

    // how far the photo moves towards the middle without an explanation: by 30% of the panel's
    // width, into the panel, whose prediction moves aside (none when the panel is under the photo)
    function setShift() {
      const gap = panel.offsetLeft - (stage.offsetLeft + stage.offsetWidth);
      const beside = panel.offsetTop < stage.offsetTop + stage.offsetHeight / 2;
      const shift = beside ? Math.round(0.3 * panel.offsetWidth) : 0;
      root.style.setProperty("--fixlip-shift", `${shift}px`);
      root.style.setProperty("--fixlip-overlap", `${beside ? Math.max(0, shift - gap) : 0}px`);
    }

    function setSize(size) {
      state.size = Math.max(2, Math.min(10, size));
      if (state.focus === null) show(null);
    }

    function setView(view) {
      state.view = view;
      state.pinned = false;
      viewButtons.forEach((b, k) => b.setAttribute("aria-pressed", String(VIEWS[k][0] === view)));
      setShift();
      root.classList.toggle("fixlip-demo-full", view === "full");
      root.classList.toggle("fixlip-demo-none", view === "none");
      setInteractive();
      show(null);
    }

    // -- the views: the clique or the full explanation at rest, the interactions of a player when
    // pointed at
    function show(player) {
      state.focus = player;
      root.classList.toggle("fixlip-demo-focused", player !== null);
      list.replaceChildren();
      if (player !== null) showPlayer(player);
      else if (state.view === "full") showFull();
      else if (state.view === "none") showNone();
      else showClique();
    }

    // no explanation: the photo and the caption as they are, and the model's similarity
    function showNone() {
      players.forEach((button) => {
        button.classList.remove("fixlip-demo-in", "fixlip-demo-focus");
        button.style.setProperty("--fixlip-tint", "transparent");
        button.style.removeProperty("color");
      });
      live.textContent = `No explanation: similarity ${ex.similarity.toFixed(2)}`;
      drawEdges([]);
    }

    function showClique() {
      const [clique, value] = ex.cliques[state.size];
      const inside = new Set(clique);
      players.forEach((button, k) => {
        button.classList.toggle("fixlip-demo-in", inside.has(k));
        button.classList.remove("fixlip-demo-focus");
        button.style.setProperty("--fixlip-tint", "transparent");
        button.style.removeProperty("color");
      });
      const regions = clique.filter((k) => k < regionCount).length;
      title.textContent = "The players that raise the similarity the most together";
      const tokens = clique.length - regions;
      detail.replaceChildren(
        `${regions} ${regions === 1 ? "region" : "regions"} and ${tokens} ${tokens === 1 ? "token" : "tokens"} ` +
          "(FIxLIP's greedy search) that add ",
        valued(value),
        " together, linked by their interactions.",
      );
      sizer.hidden = false;
      graph.hidden = false;
      list.hidden = true;
      sizeLabel.textContent = `${state.size} players`;
      smaller.disabled = state.size <= 2;
      larger.disabled = state.size >= 10;
      live.textContent = `Clique of ${clique.length} players`;
      drawEdges(cliquePairs(clique));
      drawGraph(clique);
    }

    // the full explanation: every player tinted by what it adds on its own, the strongest
    // interactions drawn between them, the five strongest between a region and a token listed
    function showFull() {
      const largest = Math.max(...ex.first.map(Math.abs));
      players.forEach((button, k) => {
        button.classList.remove("fixlip-demo-in", "fixlip-demo-focus");
        // on a square-root scale, so that the regions stay visible next to the words
        const { color, alpha } = tint(Math.sign(ex.first[k]) * Math.sqrt(Math.abs(ex.first[k])), Math.sqrt(largest), 0.75);
        button.style.setProperty("--fixlip-tint", color);
        button.style.color = k >= regionCount && alpha > 0.55 ? "#ffffff" : "";
      });
      title.textContent = "The full explanation";
      detail.textContent =
        "Each region and word is colored by what it adds alone; " +
        `lines link the ${STRONGEST} strongest interactions. Listed: the strongest image–text pairs.`;
      sizer.hidden = true;
      graph.hidden = true;
      list.hidden = false;
      list.classList.add("fixlip-demo-list-pairs");
      for (const { i, j, value, strength } of crossModal) {
        const item = el("li");
        const pair = el("span", "fixlip-demo-pair");
        pair.append(swatchOf(i), el("span", "fixlip-demo-times", "×"), swatchOf(j));
        const bar = el("span", "fixlip-demo-bar-value");
        bar.style.setProperty("--fixlip-width", `${strength * 100}%`);
        bar.style.setProperty("--fixlip-color", solid(value));
        item.append(pair, bar, el("span", "fixlip-demo-number", signed(value)));
        list.append(item);
      }
      live.textContent = "The full explanation";
      drawEdges(strongest);
    }

    function showPlayer(player) {
      const n = ex.first.length;
      const values = players.map((_, k) => interaction(player, k));
      // a player's strongest interaction is in full color, unless it is weak: then all stay faint
      const largest = Math.max(floor, ...values.map(Math.abs));
      players.forEach((button, k) => {
        button.classList.remove("fixlip-demo-in");
        button.classList.toggle("fixlip-demo-focus", k === player);
        const { color, alpha } = k === player ? { color: "transparent", alpha: 0 } : tint(values[k], largest);
        button.style.setProperty("--fixlip-tint", color);
        if (k >= regionCount) button.style.color = alpha > 0.55 ? "#ffffff" : "";
      });
      if (player < regionCount) {
        const swatch = el("span", "fixlip-demo-swatch fixlip-demo-swatch-region fixlip-demo-swatch-title");
        region(swatch, photo.src, player, grid);
        title.replaceChildren("Interactions of ", swatch, ` row ${Math.floor(player / grid) + 1}, column ${(player % grid) + 1}`);
      } else {
        title.textContent = `Interactions of ${name(player)}`;
      }
      detail.replaceChildren(
        "On its own, it adds ",
        valued(ex.first[player]),
        " to the similarity. With the players below, it adds more (red) or less (blue) than both alone.",
      );
      sizer.hidden = true;
      graph.hidden = true;
      list.hidden = false;
      list.classList.remove("fixlip-demo-list-pairs");
      const order = [...Array(n).keys()]
        .filter((k) => k !== player)
        .sort((a, b) => Math.abs(values[b]) - Math.abs(values[a]))
        .slice(0, TOP);
      for (const k of order) {
        const item = el("li");
        const swatch = swatchOf(k);
        const bar = el("span", "fixlip-demo-bar-value");
        bar.style.setProperty("--fixlip-width", `${(Math.abs(values[k]) / largest) * 100}%`);
        bar.style.setProperty("--fixlip-color", solid(values[k]));
        item.append(swatch, bar, el("span", "fixlip-demo-number", signed(values[k])));
        list.append(item);
      }
      live.textContent = `Interactions of ${name(player)}`;
      drawEdges([]);
    }

    // a player in the panel: a thumbnail of a region, or the text of a token
    function swatchOf(k) {
      const swatch = el("span", "fixlip-demo-swatch");
      if (k < regionCount) {
        swatch.classList.add("fixlip-demo-swatch-region");
        swatch.title = `Row ${Math.floor(k / grid) + 1}, column ${(k % grid) + 1}`;
        region(swatch, photo.src, k, grid);
      } else {
        swatch.textContent = ex.tokens[k - regionCount][0];
      }
      return swatch;
    }
    // the interactions within a clique, the weakest first (to be drawn under the strongest), each
    // with its strength relative to the strongest
    function cliquePairs(clique) {
      const pairs = [];
      for (let a = 0; a < clique.length; a += 1)
        for (let b = a + 1; b < clique.length; b += 1) pairs.push([a, b, interaction(clique[a], clique[b])]);
      const largest = Math.max(...pairs.map(([, , value]) => Math.abs(value)));
      return pairs
        .map(([a, b, value]) => ({
          a,
          b,
          i: clique[a],
          j: clique[b],
          value,
          strength: largest > 0 ? Math.abs(value) / largest : 0,
        }))
        .sort((x, y) => x.strength - y.strength);
    }
    // an edge, straight or (with a bulge) arched upwards, as wide and as opaque as it is strong
    const line = (x1, y1, x2, y2, value, strength, width, opacity, bulge = 0) => {
      const node = document.createElementNS(SVG, "path");
      const top = Math.min(y1, y2) - bulge;
      const attributes = {
        d: bulge ? `M ${x1} ${y1} Q ${(x1 + x2) / 2} ${2 * top - (y1 + y2) / 2} ${x2} ${y2}` : `M ${x1} ${y1} L ${x2} ${y2}`,
        fill: "none",
        stroke: solid(value),
        "stroke-linecap": "round",
        "stroke-width": (width[0] + width[1] * strength).toFixed(1),
        "stroke-opacity": (opacity[0] + opacity[1] * strength).toFixed(2),
      };
      for (const [key, v] of Object.entries(attributes)) node.setAttribute(key, v);
      return node;
    };

    // interactions on the photo, the weakest under the strongest: between the centers of regions
    // and the tops of tokens, arched between two tokens
    function drawEdges(pairs) {
      edges.replaceChildren();
      if (!pairs.length) return;
      const box = stage.getBoundingClientRect();
      edges.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
      const anchor = (k) => {
        const r = players[k].getBoundingClientRect();
        const y = k < regionCount ? r.top + r.height / 2 : r.top;
        return [r.left + r.width / 2 - box.left, y - box.top];
      };
      for (const { i, j, value, strength } of [...pairs].sort((x, y) => x.strength - y.strength)) {
        if (strength < 0.08) continue;
        const [x1, y1] = anchor(i);
        const [x2, y2] = anchor(j);
        const bulge = i >= regionCount && j >= regionCount ? 6 + 0.1 * Math.abs(x2 - x1) : 0;
        edges.append(line(x1, y1, x2, y2, value, strength, [1.5, 7], [0.3, 0.6], bulge));
      }
    }

    // the clique as a graph in the panel: its players on an ellipse, the regions first, linked by
    // their interactions as thick as they are strong, the strongest ones labeled
    function drawGraph(clique) {
      const nodes = clique.map((k) => {
        const node = swatchOf(k);
        node.classList.add("fixlip-demo-node");
        node.addEventListener("pointerenter", () => players[k].classList.add("fixlip-demo-focus"));
        node.addEventListener("pointerleave", () => players[k].classList.remove("fixlip-demo-focus"));
        return node;
      });
      links.replaceChildren();
      graph.replaceChildren(links, ...nodes);
      const { width, height } = graph.getBoundingClientRect();
      if (!width) return;
      const halfWidth = Math.max(...nodes.map((node) => node.offsetWidth / 2));
      const halfHeight = Math.max(...nodes.map((node) => node.offsetHeight / 2));
      const ry = height / 2 - halfHeight - 2;
      const rx = Math.min(width / 2 - halfWidth - 2, 1.9 * ry);
      const at = clique.map((_, k) => {
        const angle = Math.PI + (2 * Math.PI * k) / clique.length; // from the left, clockwise
        return [width / 2 + rx * Math.cos(angle), height / 2 + ry * Math.sin(angle)];
      });
      nodes.forEach((node, k) => {
        node.style.left = `${at[k][0]}px`;
        node.style.top = `${at[k][1]}px`;
      });
      links.setAttribute("viewBox", `0 0 ${width} ${height}`);
      const pairs = cliquePairs(clique);
      for (const { a, b, value, strength } of pairs) {
        links.append(line(...at[a], ...at[b], value, strength, [0.75, 9], [0.1, 0.8]));
      }
      // the labels of the strongest interactions, along their edges where they do not overlap
      const placed = [];
      for (const { a, b, value, strength } of pairs.slice(-4).reverse()) {
        if (strength < 0.3) break;
        const spot = [0.5, 0.35, 0.65]
          .map((t) => [at[a][0] + t * (at[b][0] - at[a][0]), at[a][1] + t * (at[b][1] - at[a][1])])
          .find(([x, y]) => placed.every(([u, v]) => Math.abs(x - u) > 46 || Math.abs(y - v) > 16));
        if (!spot) continue;
        placed.push(spot);
        const label = document.createElementNS(SVG, "text");
        label.setAttribute("x", spot[0]);
        label.setAttribute("y", spot[1]);
        label.setAttribute("fill", solid(value));
        label.textContent = signed(value);
        links.append(label);
        const box = label.getBBox();
        const backdrop = document.createElementNS(SVG, "rect");
        for (const [key, v] of Object.entries({ x: box.x - 3, y: box.y - 1, width: box.width + 6, height: box.height + 2, rx: 4 }))
          backdrop.setAttribute(key, v);
        links.insertBefore(backdrop, label);
      }
    }

    cells.addEventListener("keydown", (event) => {
      const moves = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -grid, ArrowDown: grid };
      const k = players.indexOf(event.target);
      if (!(event.key in moves) || k < 0 || k >= regionCount) return;
      const next = k + moves[event.key];
      const sameRow = Math.floor(next / grid) === Math.floor(k / grid);
      if (next < 0 || next >= regionCount || (Math.abs(moves[event.key]) === 1 && !sameRow)) return;
      event.preventDefault();
      players[k].tabIndex = -1;
      players[next].tabIndex = 0;
      players[next].focus();
    });

    stage.addEventListener("pointerleave", () => {
      if (!state.pinned && state.focus !== null) show(null);
    });
    stage.addEventListener("focusout", (event) => {
      if (!state.pinned && !stage.contains(event.relatedTarget)) show(null);
    });
    document.addEventListener("click", (event) => {
      if (state.pinned && !stage.contains(event.target)) {
        state.pinned = false;
        show(null);
      }
    });
    root.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && state.focus !== null) {
        state.pinned = false;
        show(null);
      }
    });
    const redraw = new ResizeObserver(() => {
      setShift();
      if (state.focus === null && ex) show(null);
    });
    redraw.observe(stage);
    redraw.observe(graph);

    root.classList.add("fixlip-demo-ready");
    viewButtons.forEach((b, k) => b.setAttribute("aria-pressed", String(VIEWS[k][0] === state.view)));
    reset();
  }
})();
