// The FIxLIP demo (demos/fixlip/index.rst, also on the landing page): which words of a caption go
// with which regions of a photo, for SigLIP 2 and CLIP, explained as in FIxLIP (Baniecki et al.,
// NeurIPS 2025).
//
// The players are the image patches of the photo (7 x 7 for CLIP, 8 x 8 for SigLIP 2, both
// ViT-B/32 as in FIxLIP's figures) and the tokens of the caption. Their main effects and pairwise
// interactions (FBII, FIxLIP with p = 0.5) are precomputed per photo and model in data.js (by the
// scripts in ../compute), so the page only looks values up. At rest, the demo shows the
// clique of FIxLIP's greedy search, the players that together raise the similarity the most, on
// the photo and as a graph of their interactions, as in FIxLIP's figures; or the full explanation,
// every player tinted by its main effect and the strongest interactions drawn between them; or no
// explanation at all, only the photo, the caption, and the model's similarity, for contrast. There,
// the caption is an excerpt of a longer (invented) text; with an explanation, the text around it
// fades and its tokens glide into a line of their own, as the model sees only them.
// Pointing at a player (hover, keyboard focus, or a tap that pins it) lights up all its
// interactions, red for those that raise the similarity and blue for those that lower it.
//
// Without JavaScript, the static figure in the <noscript> shows; when the data does not load or
// does not fit, the script shows that figure itself.
(() => {
  const root = document.querySelector(".fixlip-demo");
  if (!root) return;
  // with scripts on, a <noscript>'s content is its HTML as text
  const fallback = root.querySelector("noscript")?.textContent ?? "";
  const showFallback = () => {
    root.classList.remove("fixlip-demo-ready");
    root.classList.add("fixlip-demo-failed");
    root.innerHTML = fallback;
  };

  const SVG = "http://www.w3.org/2000/svg";
  const RED = [255, 13, 87];
  const BLUE = [30, 136, 229];
  const TOP = 5; // the strongest interactions listed for a player
  const STRONGEST = 12; // the interactions drawn in the full explanation
  const VIEWS = [
    // the view, its label, and the label on small screens
    ["none", "No explanation", "None"],
    ["clique", "Clique", "Clique"],
    ["full", "Full explanation", "Full"],
  ];

  const loader = document.createElement("script");
  loader.src = `${root.dataset.root}data.js?v=${root.dataset.version ?? ""}`; // past caches
  loader.onload = () => {
    try {
      start(window.SHAPIQ_FIXLIP);
    } catch (error) {
      console.error("The interactive explanation could not start:", error);
      showFallback();
    }
  };
  loader.onerror = showFallback;
  document.head.appendChild(loader);

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const signed = (value) => {
    const size = Math.abs(value);
    if (size < 5e-6) return "0.000";
    const decimals = Math.max(3, Math.min(5, 1 - Math.floor(Math.log10(size))));
    return `${value >= 0 ? "+" : "−"}${size.toFixed(decimals)}`;
  };
  const solid = (value) => `rgb(${(value >= 0 ? RED : BLUE).join(", ")})`;
  // a value in a text: red or blue by its sign, in shades that read well as text in either theme
  const sign = (value) => (value >= 0 ? "fixlip-demo-up" : "fixlip-demo-down");
  const valued = (value) => el("strong", sign(value), signed(value));
  const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const tint = (value, largest, strength = 0.85) => {
    const [r, g, b] = value >= 0 ? RED : BLUE;
    const alpha = largest > 0 ? Math.min(Math.abs(value) / largest, 1) * strength : 0;
    return { color: `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`, alpha };
  };
  // how a similarity compares with the model's on COCO photos with their own captions and with
  // unrelated ones (model.norms: the 0th, 5th, ..., 100th percentiles of both): the share of photos
  // with their own caption below it, and a word for it
  const share = (value, quantiles) => {
    const k = quantiles.findIndex((q) => q > value);
    if (k === 0) return 0;
    if (k < 0) return 1;
    const [low, high] = [quantiles[k - 1], quantiles[k]];
    return (k - 1 + (value - low) / (high - low)) / (quantiles.length - 1);
  };
  const verdict = (value, norms) => {
    const below = share(value, norms.matching);
    if (below >= 0.95) return "very high";
    if (below >= 0.75) return "high";
    if (below >= 0.25) return "typical";
    return value > norms.unrelated[19] ? "low" : "like unrelated";
  };
  // a region of the photo as a background: the photo at grid times its size, moved to the region
  const region = (node, url, k, grid) => {
    node.style.backgroundImage = `url("${url}")`;
    node.style.backgroundSize = `${grid * 100}%`;
    node.style.backgroundPosition = `${((k % grid) / (grid - 1)) * 100}% ${(Math.floor(k / grid) / (grid - 1)) * 100}%`;
  };

  function start(data) {
    const has = (image, model) => `${image}/${model}` in data.explanations;
    const first = data.images.find((image) => data.models.some((model) => has(image.id, model.id)));
    if (!first) throw new Error("no photo has an explanation");
    const state = {
      image: first.id,
      model: data.models.find((model) => has(first.id, model.id)).id,
      size: 6, // the clique's, kept within the sizes of the explanation's cliques
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
    // what the models are, for readers who have not met them, behind a small "i"
    function normsSentence() {
      const known = data.models.filter((m) => m.norms);
      if (!known.length) return "";
      const median = (m, kind) => m.norms[kind][10].toFixed(2).replace("-", "−");
      const own = known.map((m) => `${median(m, "matching")} (${m.label})`).join(" and ");
      const unrelated = known.map((m) => median(m, "unrelated")).join(" and ");
      return (
        "<p>Similarities are small numbers: COCO photos score about " +
        `${own} with their own captions, and ${unrelated} with unrelated ones.</p>`
      );
    }
    const info = el("button", "fixlip-demo-info", "i");
    info.type = "button";
    info.setAttribute("aria-label", "What are CLIP and SigLIP 2?");
    info.setAttribute("aria-expanded", "false");
    info.setAttribute("aria-controls", "fixlip-demo-about");
    const about = el("div", "fixlip-demo-about");
    about.id = "fixlip-demo-about";
    about.hidden = true;
    about.innerHTML =
      '<p class="fixlip-demo-about-title">What are CLIP and SigLIP 2?</p>' +
      "<p>Vision–language models: they turn a photo and a sentence each into a list of numbers. " +
      "When both say the same thing, the two lists point the same way, and their " +
      "<strong>similarity</strong> is high.</p>" +
      normsSentence() +
      "<p>They see a photo as a grid of square <strong>patches</strong> (here 32 × 32 pixels each: " +
      "7 × 7 for CLIP, 8 × 8 for SigLIP 2) and a sentence as <strong>tokens</strong>, words or " +
      "pieces of them, and look at all of them at once.</p>" +
      "<p>They learned from millions of photos with their captions from the web which go together, " +
      "but were never told where in a photo a giraffe is. That is why, in the photo of the giraffe " +
      "at the pond, they also see one in the dead tree behind it.</p>" +
      "<p>To explain a prediction, <code>shapiq</code> hides patches and words in many combinations " +
      "and watches how the similarity changes. The patches and words are the <strong>players</strong>: " +
      "what each adds on its own, and what pairs of them add together (their " +
      "<strong>interactions</strong>). A <strong>clique</strong> is a group of players that raise " +
      "the similarity the most together.</p>";
    const close = el("button", "fixlip-demo-about-close", "×");
    close.type = "button";
    close.setAttribute("aria-label", "Close");
    about.prepend(close);
    const setAbout = (open) => {
      about.hidden = !open;
      info.setAttribute("aria-expanded", String(open));
    };
    info.addEventListener("click", () => setAbout(about.hidden));
    about.addEventListener("focusout", (event) => {
      if (!about.contains(event.relatedTarget) && event.relatedTarget !== info) setAbout(false);
    });
    close.addEventListener("click", () => {
      setAbout(false);
      info.focus();
    });
    switches.append(views, models, info);
    bar.append(photos, switches, about);

    const body = el("div", "fixlip-demo-body");
    const stage = el("div", "fixlip-demo-stage");
    const frame = el("div", "fixlip-demo-image");
    const photo = el("img");
    photo.decoding = "async";
    const cells = el("div", "fixlip-demo-cells");
    frame.append(photo, cells);
    const caption = el("p", "fixlip-demo-caption");
    const spoken = el("p", "fixlip-demo-sr"); // the caption as one sentence, for screen readers
    const edges = document.createElementNS(SVG, "svg");
    edges.setAttribute("class", "fixlip-demo-edges");
    edges.setAttribute("aria-hidden", "true");
    // the edges before the caption: over its tokens' chips (see the CSS), under their words
    stage.append(frame, spoken, edges, caption);

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
    const predictionNumber = el("p", "fixlip-demo-prediction-number");
    const predictionValue = el("span", "fixlip-demo-prediction-value");
    const predictionVerdict = el("span", "fixlip-demo-verdict");
    predictionNumber.append(predictionValue, predictionVerdict);
    // where the similarity is among those of unrelated pairs and of photos with their own caption
    const scale = el("div", "fixlip-demo-scale");
    scale.setAttribute("aria-hidden", "true"); // the sentence under it says the same
    const scaleUnrelated = el("span", "fixlip-demo-scale-band fixlip-demo-scale-unrelated");
    const scaleMatching = el("span", "fixlip-demo-scale-band fixlip-demo-scale-matching");
    const scaleMark = el("span", "fixlip-demo-scale-mark");
    const scaleLabels = [el("span", "fixlip-demo-scale-label", "unrelated captions"), el("span", "fixlip-demo-scale-label", "own captions")];
    scale.append(scaleUnrelated, scaleMatching, scaleMark, ...scaleLabels);
    const predictionCompare = el("p", "fixlip-demo-prediction-compare");
    const predictionWhy = el("p", "fixlip-demo-prediction-why");
    const predictionCredit = el("p", "fixlip-demo-meta");
    const toView = (view, label) => {
      const button = el("button", "fixlip-demo-link", label);
      button.type = "button";
      button.addEventListener("click", () => {
        setView(view);
        viewButtons[VIEWS.findIndex(([v]) => v === view)].focus(); // the link itself goes away
      });
      return button;
    };
    predictionMain.append(
      predictionModel,
      predictionNumber,
      el("p", "fixlip-demo-prediction-label", "similarity of the photo and its caption"),
      scale,
      predictionCompare,
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
      button.setAttribute("aria-label", image.title ?? image.id);
      button.title = image.title ?? "";
      const thumb = el("img");
      thumb.src = `${root.dataset.root}${image.id}.jpg`;
      thumb.alt = "";
      thumb.loading = "lazy";
      button.append(thumb);
      button.addEventListener("click", () =>
        crossFade(() => {
          state.image = image.id;
          reset();
        }, true),
      );
      photos.append(button);
      return button;
    });
    const viewButtons = VIEWS.map(([view, label, short]) => {
      const button = el("button", "fixlip-demo-option");
      button.append(el("span", "fixlip-demo-long", label), el("span", "fixlip-demo-short", short));
      button.setAttribute("aria-label", label);
      button.type = "button";
      button.addEventListener("click", () => setView(view));
      views.append(button);
      return button;
    });
    const modelButtons = data.models.map((model) => {
      const button = el("button", "fixlip-demo-option", model.label);
      button.type = "button";
      button.addEventListener("click", () =>
        crossFade(() => {
          state.model = model.id;
          reset();
        }),
      );
      models.append(button);
      return button;
    });
    const step = (button, by) =>
      button.addEventListener("click", () => {
        if (button.getAttribute("aria-disabled") !== "true") setSize(state.size + by);
      });
    step(smaller, -1);
    step(larger, 1);

    // -- the players of the current photo and model --------------------------------------------
    let ex = null; // the explanation: tokens, main effects, interactions, cliques
    let players = []; // the button of every player: the regions, then the tokens
    let floor = 0; // the 99th percentile of the interactions, the least that tints scale to
    let strongest = []; // the strongest interactions, the strongest first, for the full explanation
    let crossModal = []; // the strongest interactions between a region and a token, listed there
    let grid = 7; // the photo's regions are grid x grid patches, by the model
    let regionCount = grid * grid;
    let sizes = [2, 10]; // the smallest and the largest clique of the explanation
    let roving = 0; // the region in the tab order; the arrow keys move within the grid
    let rovingToken = 0; // the same for the caption's tokens
    let glideTimer = 0;
    let arriveTimer = 0;
    let started = false; // nothing is announced while the demo builds itself

    const interaction = (i, j) => {
      if (i === j) return 0;
      const [a, b] = i < j ? [i, j] : [j, i];
      const n = ex.first.length;
      return ex.pairs[a * n - (a * (a + 1)) / 2 + (b - a - 1)]; // the upper triangle, row by row
    };
    const place = (k) => `row ${Math.floor(k / grid) + 1}, column ${(k % grid) + 1}`;
    const name = (player) =>
      player < regionCount ? `the region in ${place(player)}` : `“${ex.tokens[player - regionCount][0]}”`;
    const describe = (player, note = "") =>
      players[player].setAttribute("aria-label", `Interactions of ${name(player)}${note}`);

    function reset() {
      // a photo without an explanation for the model switches to a model that has one
      if (!has(state.image, state.model)) state.model = data.models.find((m) => has(state.image, m.id)).id;
      ex = data.explanations[`${state.image}/${state.model}`];
      grid = ex.grid ?? 7;
      regionCount = grid * grid;
      cells.style.setProperty("--fixlip-grid", grid);
      const known = Object.keys(ex.cliques).map(Number);
      sizes = [Math.min(...known), Math.max(...known)];
      state.size = Math.max(sizes[0], Math.min(sizes[1], state.size));
      roving = 0;
      rovingToken = 0;
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
      photoButtons.forEach((b, k) => {
        b.setAttribute("aria-pressed", String(data.images[k].id === state.image));
        b.disabled = !data.models.some((m) => has(data.images[k].id, m.id));
      });
      modelButtons.forEach((b, k) => {
        b.setAttribute("aria-pressed", String(data.models[k].id === state.model));
        b.disabled = !has(state.image, data.models[k].id);
      });
      photo.src = `${root.dataset.root}${image.id}.jpg`;
      photo.alt = image.alt ?? "";

      cells.replaceChildren();
      caption.replaceChildren();
      caption.classList.remove("fixlip-demo-arriving", "fixlip-demo-renewed"); // see crossFade
      players = [];
      for (let k = 0; k < regionCount; k += 1) {
        const cell = el("button", "fixlip-demo-cell");
        cell.type = "button";
        cells.append(cell);
        players.push(cell);
      }
      const { inner, tokens } = captionContent(image, ex);
      caption.append(inner);
      players.push(...tokens);
      spoken.textContent = `Caption: ${captionString(ex)}`;
      players.forEach((button, player) => {
        describe(player);
        button.addEventListener("pointerenter", (event) => {
          if (event.pointerType !== "touch" && !state.pinned && state.view !== "none") show(player);
        });
        button.addEventListener("focus", () => {
          setRoving(player);
          if (!state.pinned && state.view !== "none") show(player, true);
        });
        button.addEventListener("click", () => {
          if (state.view === "none") return;
          const again = state.pinned && state.focus === player;
          state.pinned = !again;
          show(again ? null : player, true);
        });
      });
      meta.replaceChildren(...metaText(image, model, ex));
      predictionModel.textContent = `${model.label} (${model.detail}) predicts`;
      predictionValue.textContent = ex.similarity.toFixed(2);
      setScale(model.norms, ex.similarity);
      predictionCredit.replaceChildren(...creditText(image));
      setInteractive();
      placeExcerpt();
      show(null, true);
    }

    const creditText = (image) => [
      "Photo: ",
      Object.assign(el("a", "", image.credit?.title ?? "source"), { href: image.credit?.url ?? "" }),
      image.credit?.author ? ` by ${image.credit.author}, ` : ", ",
      Object.assign(el("a", "", image.credit?.license ?? ""), { href: image.credit?.licenseUrl ?? "" }),
      ", cropped.",
    ];
    const metaText = (image, model, e) => [
      `${model.label} (${model.detail}): similarity ${e.similarity.toFixed(2)}` +
        (model.norms ? `, ${verdict(e.similarity, model.norms)}` : "") +
        `. Explained by ${data.method}` +
        (e.r2 === undefined ? ". " : `; the explanation fits held-out coalitions with R² = ${e.r2.toFixed(2)}. `),
      ...creditText(image),
    ];
    // the word for the similarity, its place on the scale (the middle 90% of unrelated pairs and of
    // photos with their own caption), and the share of photos with their own caption below it
    function setScale(norms, value) {
      for (const node of [predictionVerdict, scale, predictionCompare]) node.hidden = !norms;
      if (!norms) return;
      const word = verdict(value, norms);
      predictionVerdict.textContent = word;
      predictionVerdict.dataset.verdict = word.replace(" ", "-");
      const low = Math.min(norms.unrelated[1], value) - 0.03;
      const high = Math.max(norms.matching[19], value) + 0.03;
      const at = (x) => `${((100 * (x - low)) / (high - low)).toFixed(1)}%`;
      const band = (node, quantiles) =>
        Object.assign(node.style, { left: at(quantiles[1]), width: `calc(${at(quantiles[19])} - ${at(quantiles[1])})` });
      band(scaleUnrelated, norms.unrelated);
      band(scaleMatching, norms.matching);
      scaleMark.style.left = at(value);
      scaleLabels[0].style.left = at(norms.unrelated[10]);
      scaleLabels[1].style.left = at(norms.matching[10]);
      const below = Math.round(100 * share(value, norms.matching));
      predictionCompare.textContent =
        below >= 50
          ? `Higher than for ${below}% of COCO photos with their own caption.`
          : `Lower than for ${100 - below}% of COCO photos with their own caption.`;
    }

    // the caption's tokens with spaces where the text has them: not before punctuation, nor inside
    // quotes, which CLIP's tokenizer marks as words of their own
    const spaced = (e) => {
      let quoted = false;
      let opened = false; // after an opening quote
      return e.tokens.map(([text, space], k) => {
        let gap = k > 0 && space && !opened && !/^[.,;:!?]/.test(text);
        opened = false;
        if (text === '"') {
          gap &&= !quoted;
          opened = !quoted;
          quoted = !quoted;
        }
        return [gap ? " " : "", text];
      });
    };
    const captionString = (e) => spaced(e).flat().join("");

    // the caption: its tokens within the text around it, whose words fade with their distance to
    // the caption (the text shows only without an explanation, and is not read out)
    function captionContent(image, e) {
      const inner = el("span", "fixlip-demo-caption-inner");
      const [before, after] = image.excerpt ?? ["", ""];
      const around = (text, side) => {
        const span = el("span", "fixlip-demo-context");
        span.setAttribute("aria-hidden", "true");
        const words = text ? text.split(" ") : [];
        words.forEach((word, k) => {
          const distance = side === "before" ? words.length - k : k + 1;
          const fading = el("span", "", word);
          fading.style.opacity = (0.8 * Math.max(0, 1 - (distance - 1) / 18)).toFixed(2);
          span.append(...(side === "before" ? [fading, " "] : [" ", fading]));
        });
        return span;
      };
      const tokens = e.tokens.map(([text]) => {
        const token = el("button", "fixlip-demo-token", text);
        token.type = "button";
        return token;
      });
      inner.append(around(before, "before"));
      let word = null; // a word with the punctuation and quotes around it, kept on one line
      spaced(e).forEach(([space], k) => {
        if (word && !space) {
          word.append(tokens[k]);
          return;
        }
        word = el("span", "fixlip-demo-word");
        word.append(tokens[k]);
        inner.append(...(space ? [space] : []), word);
      });
      inner.append(around(after, "after"));
      return { inner, tokens };
    }

    // without an explanation, the caption's box shows its lines in the middle of the excerpt
    function placeExcerpt() {
      const inner = caption.firstElementChild;
      if (!inner) return;
      inner.style.transform = "";
      if (state.view !== "none") return;
      // offsets (from the caption, the tokens' positioned parent) ignore the tokens' transforms
      // while they glide
      const tokens = players.slice(regionCount);
      const top = Math.min(...tokens.map((t) => t.offsetTop));
      const bottom = Math.max(...tokens.map((t) => t.offsetTop + t.offsetHeight));
      const middle = caption.clientHeight / 2;
      inner.style.transform = `translateY(${Math.round(middle - (top + bottom) / 2)}px)`;
    }

    // the caption, the model's line, and the credit keep the room of the longest of all photos and
    // models, so that switching between them moves nothing
    function reserveRoom() {
      const all = Object.entries(data.explanations).map(([key, e]) => {
        const [image, model] = key.split("/");
        return [data.images.find((x) => x.id === image), data.models.find((x) => x.id === model), e];
      });
      reserveCaption(all);
      reserve(meta, all.map(([image, model, e]) => metaText(image, model, e)));
      reserve(predictionCredit, data.images.map(creditText));
    }

    // the caption's room: that of its tallest line of tokens, and of its tallest excerpt (the
    // caption's lines and half a line of the text around it above and below), which next to the
    // panel keeps the card's height in all views (under the photo, the views differ anyway)
    function reserveCaption(all) {
      const width = caption.getBoundingClientRect().width;
      if (!width) return;
      const probe = caption.cloneNode(false);
      Object.assign(probe.style, { position: "absolute", visibility: "hidden", width: `${width}px`, minHeight: "0", height: "auto" });
      caption.after(probe);
      let tokens = 0;
      let excerpt = 0;
      for (const [image, , e] of all) {
        for (const asExcerpt of [false, true]) {
          probe.classList.toggle("fixlip-demo-excerpt", asExcerpt);
          const content = captionContent(image, e);
          probe.replaceChildren(content.inner);
          if (!asExcerpt) {
            tokens = Math.max(tokens, probe.getBoundingClientRect().height);
            continue;
          }
          const boxes = content.tokens.map((t) => t.getBoundingClientRect());
          const lines = Math.max(...boxes.map((b) => b.bottom)) - Math.min(...boxes.map((b) => b.top));
          excerpt = Math.max(excerpt, lines + parseFloat(getComputedStyle(probe).lineHeight));
        }
      }
      probe.remove();
      const room = beside() ? Math.max(tokens, excerpt) : tokens;
      caption.style.minHeight = `${Math.ceil(room)}px`;
      caption.style.setProperty("--fixlip-caption-height", `${Math.ceil(Math.max(tokens, excerpt))}px`);
    }

    // the least height of a node: that of the tallest of the contents, in a hidden copy of it
    function reserve(node, contents) {
      const width = node.getBoundingClientRect().width;
      if (!width) return; // not laid out (hidden on small screens): keep what it has
      const probe = node.cloneNode(false);
      Object.assign(probe.style, { position: "absolute", visibility: "hidden", width: `${width}px`, minHeight: "0" });
      node.after(probe);
      let tallest = 0;
      for (const content of contents) {
        probe.replaceChildren(...content);
        tallest = Math.max(tallest, probe.getBoundingClientRect().height);
      }
      probe.remove();
      node.style.minHeight = `${Math.ceil(tallest)}px`;
    }

    // another model has another grid of patches (CLIP 7 x 7, SigLIP 2 8 x 8, as in FIxLIP), another
    // photo another photo too: the old ones fade out over the new ones, which fade in (a new caption
    // fades in)
    function crossFade(change, newPhoto = false) {
      for (const ghost of stage.querySelectorAll(".fixlip-demo-ghost")) ghost.remove();
      if (reducedMotion()) {
        change();
        return;
      }
      const old = newPhoto ? [frame, edges] : [cells, edges];
      const ghosts = old.map((node) => {
        const ghost = node.cloneNode(true);
        ghost.classList.add("fixlip-demo-ghost");
        ghost.setAttribute("aria-hidden", "true");
        ghost.setAttribute("inert", "");
        if (node === frame)
          Object.assign(ghost.style, { position: "absolute", top: "0", left: "0", width: `${frame.offsetWidth}px` });
        node.after(ghost);
        return ghost;
      });
      change();
      // a new caption fades in: as a whole as an excerpt, and else by its tokens' words and chips
      // (a fading caption would lift its chips over the edges, see the CSS)
      const fresh = newPhoto && state.view === "none" ? [cells, edges, caption] : [cells, edges];
      caption.classList.toggle("fixlip-demo-renewed", newPhoto && state.view !== "none");
      for (const node of fresh) node.classList.add("fixlip-demo-fresh");
      requestAnimationFrame(() =>
        requestAnimationFrame(() => {
          for (const node of fresh) node.classList.remove("fixlip-demo-fresh");
          for (const ghost of ghosts) {
            ghost.classList.add("fixlip-demo-gone");
            setTimeout(() => ghost.remove(), 600);
          }
        }),
      );
    }

    // the players can be pointed at, unless there is no explanation to show (then they are inert);
    // the grid has one stop in the tab order, the region last pointed at
    function setInteractive() {
      const none = state.view === "none";
      cells.inert = none;
      players.forEach((button, k) => {
        if (k >= regionCount) button.inert = none;
        button.tabIndex = k === roving || k === regionCount + rovingToken ? 0 : -1;
      });
    }

    function setRoving(k) {
      if (k < regionCount) {
        players[roving].tabIndex = -1;
        roving = k;
      } else {
        players[regionCount + rovingToken].tabIndex = -1;
        rovingToken = k - regionCount;
      }
      players[k].tabIndex = 0;
    }

    // how far the photo moves towards the middle without an explanation: so far that the photo and
    // the prediction next to it are centered together (not at all when the panel is under the photo)
    const PREDICTION_GAP = 24; // px between the photo and the prediction
    const beside = () => panel.offsetTop < stage.offsetTop + stage.offsetHeight / 2;
    function setShift() {
      const gap = panel.offsetLeft - (stage.offsetLeft + stage.offsetWidth);
      let shift = 0;
      if (beside()) {
        predictionMain.style.width = "max-content"; // its widest line, unwrapped
        const text = Math.min(predictionMain.offsetWidth, panel.offsetWidth - PREDICTION_GAP);
        predictionMain.style.width = "";
        const group = stage.offsetWidth + PREDICTION_GAP + text;
        shift = Math.max(0, Math.round((body.clientWidth - group) / 2 - (stage.offsetLeft - body.offsetLeft)));
      }
      root.style.setProperty("--fixlip-shift", `${shift}px`);
      root.style.setProperty("--fixlip-overlap", `${Math.max(0, shift - gap)}px`);
    }

    function setSize(size) {
      state.size = Math.max(sizes[0], Math.min(sizes[1], size));
      if (state.focus === null) show(null, true);
    }

    function setView(view) {
      // into or out of the excerpt: its text fades, and the tokens glide from where they are to
      // where they go (first, last, invert, play)
      const tokens = players.slice(regionCount);
      const glide = (view === "none") !== (state.view === "none") && !reducedMotion();
      const first = glide ? tokens.map((t) => t.getBoundingClientRect()) : [];
      for (const ghost of stage.querySelectorAll(".fixlip-demo-caption.fixlip-demo-ghost")) ghost.remove();
      clearTimeout(arriveTimer);
      caption.classList.remove("fixlip-demo-arriving");
      if (glide && state.view === "none") fadeExcerpt();

      state.view = view;
      state.pinned = false;
      viewButtons.forEach((b, k) => b.setAttribute("aria-pressed", String(VIEWS[k][0] === view)));
      setShift();
      root.classList.toggle("fixlip-demo-full", view === "full");
      root.classList.toggle("fixlip-demo-none", view === "none");
      caption.classList.toggle("fixlip-demo-excerpt", view === "none");
      reserve(predictionCredit, data.images.map(creditText)); // laid out only now on small screens
      setInteractive();
      for (const t of tokens) {
        t.style.transition = "none";
        t.style.transform = "";
      }
      placeExcerpt();
      show(null, true);
      if (!glide) {
        for (const t of tokens) t.style.transition = "";
        return;
      }
      caption.classList.add("fixlip-demo-gliding");
      // the text around the caption fades in only when the excerpt arrives (not with each photo)
      if (view === "none") {
        caption.classList.add("fixlip-demo-arriving");
        arriveTimer = setTimeout(() => caption.classList.remove("fixlip-demo-arriving"), 1000);
      }
      tokens.forEach((t, k) => {
        const last = t.getBoundingClientRect();
        t.style.transform = `translate(${first[k].left - last.left}px, ${first[k].top - last.top}px)`;
      });
      caption.getBoundingClientRect(); // the tokens start where they were
      if (view !== "none") edges.classList.add("fixlip-demo-fresh"); // the edges follow the tokens
      clearTimeout(glideTimer);
      requestAnimationFrame(() => {
        for (const t of tokens) {
          t.style.transition = "";
          t.style.transform = "";
        }
        // the edges once the tokens are in place (their glide takes 650 ms): a gliding token is a
        // layer of its own, with its chip over the edges
        clearTimeout(glideTimer);
        glideTimer = setTimeout(() => {
          caption.classList.remove("fixlip-demo-gliding");
          edges.classList.remove("fixlip-demo-fresh");
        }, 700);
      });
    }

    // the excerpt's text fades out in a copy over the caption (its tokens hidden: they glide away)
    function fadeExcerpt() {
      const ghost = caption.cloneNode(true);
      ghost.classList.add("fixlip-demo-ghost");
      ghost.setAttribute("aria-hidden", "true");
      ghost.setAttribute("inert", "");
      Object.assign(ghost.style, {
        top: `${caption.offsetTop}px`,
        left: `${caption.offsetLeft}px`,
        width: `${caption.offsetWidth}px`,
      });
      caption.after(ghost);
      requestAnimationFrame(() => requestAnimationFrame(() => ghost.classList.add("fixlip-demo-gone")));
      setTimeout(() => ghost.remove(), 700);
    }

    // -- the views: the clique or the full explanation at rest, the interactions of a player when
    // pointed at; screen readers hear what changed when it was asked for (a click, the keyboard, a
    // switch), not on every hover
    function show(player, announce = false) {
      state.focus = player;
      root.classList.toggle("fixlip-demo-focused", player !== null);
      if (player !== null || state.view !== "none") list.replaceChildren(); // (it fades out otherwise)
      let message;
      if (player !== null) message = showPlayer(player);
      else if (state.view === "full") message = showFull();
      else if (state.view === "none") message = showNone();
      else message = showClique();
      if (announce && started) live.textContent = message;
    }

    // no explanation: the photo and the caption as they are, and the model's similarity
    function showNone() {
      players.forEach((button) => {
        button.classList.remove("fixlip-demo-in", "fixlip-demo-focus");
        button.style.setProperty("--fixlip-tint", "transparent");
        button.style.removeProperty("color");
      });
      const note = predictionCompare.hidden ? "" : ` ${predictionVerdict.textContent}. ${predictionCompare.textContent}`;
      return `No explanation. ${predictionModel.textContent} a similarity of ${ex.similarity.toFixed(2)},${note}`;
    }

    function showClique() {
      const [clique, value] = ex.cliques[state.size];
      const inside = new Set(clique);
      players.forEach((button, k) => {
        describe(k, inside.has(k) ? ", in the clique" : "");
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
          "(FIxLIP's greedy search) that together raise the similarity by ",
        valued(value),
        ", linked by their interactions.",
      );
      sizer.hidden = false;
      graph.hidden = false;
      list.hidden = true;
      sizeLabel.textContent = `${state.size} players`;
      smaller.setAttribute("aria-disabled", String(state.size <= sizes[0]));
      larger.setAttribute("aria-disabled", String(state.size >= sizes[1]));
      drawEdges(cliquePairs(clique));
      drawGraph(clique);
      return `Clique of ${clique.length} players, ${clique.map(name).join(", ")}, that add ${signed(value)} together.`;
    }

    // the full explanation: every player tinted by what it adds on its own, the strongest
    // interactions drawn between them, the five strongest between a region and a token listed
    function showFull() {
      const largest = Math.max(...ex.first.map(Math.abs));
      players.forEach((button, k) => {
        describe(k);
        button.classList.remove("fixlip-demo-in", "fixlip-demo-focus");
        // on a square-root scale, so that the regions stay visible next to the words
        const { color, alpha } = tint(Math.sign(ex.first[k]) * Math.sqrt(Math.abs(ex.first[k])), Math.sqrt(largest), 0.75);
        button.style.setProperty("--fixlip-tint", color);
        button.classList.toggle("fixlip-demo-strong", alpha > 0.55);
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
      drawEdges(strongest);
      const [top] = crossModal;
      return top
        ? `The full explanation. The strongest image–text pair: ${name(top.i)} and ${name(top.j)}, ${signed(top.value)}.`
        : "The full explanation.";
    }

    function showPlayer(player) {
      const n = ex.first.length;
      const values = players.map((_, k) => interaction(player, k));
      // a player's strongest interaction is in full color, unless it is weak: then all stay faint
      const largest = Math.max(floor, ...values.map(Math.abs));
      players.forEach((button, k) => {
        describe(k);
        button.classList.remove("fixlip-demo-in");
        button.classList.toggle("fixlip-demo-focus", k === player);
        const { color, alpha } = k === player ? { color: "transparent", alpha: 0 } : tint(values[k], largest);
        button.style.setProperty("--fixlip-tint", color);
        button.classList.toggle("fixlip-demo-strong", alpha > 0.55);
      });
      if (player < regionCount) {
        const swatch = el("span", "fixlip-demo-swatch fixlip-demo-swatch-region fixlip-demo-swatch-title");
        swatch.setAttribute("aria-hidden", "true");
        region(swatch, photo.src, player, grid);
        title.replaceChildren("Interactions of ", swatch, ` ${place(player)}`);
      } else {
        title.textContent = `Interactions of ${name(player)}`;
      }
      detail.replaceChildren(
        "On its own, it adds ",
        valued(ex.first[player]),
        " to the similarity. Together with each player below, it adds more (red) or less (blue) than the two do alone.",
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
      drawEdges([]);
      const [best] = order;
      return (
        `Interactions of ${name(player)}: on its own, it adds ${signed(ex.first[player])}; ` +
        `most with ${name(best)}, ${signed(values[best])}.`
      );
    }

    // a player in the panel: a thumbnail of a region, or the text of a token
    function swatchOf(k) {
      const swatch = el("span", "fixlip-demo-swatch");
      if (k < regionCount) {
        swatch.classList.add("fixlip-demo-swatch-region");
        swatch.title = `Row ${Math.floor(k / grid) + 1}, column ${(k % grid) + 1}`;
        swatch.setAttribute("role", "img");
        swatch.setAttribute("aria-label", place(k));
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
    function drawGraph(clique, crowded = false) {
      graph.classList.toggle("fixlip-demo-crowded", crowded); // smaller nodes when they would overlap
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
      const rx = Math.min(width / 2 - halfWidth - 2, 2.6 * ry);
      // the players on the ellipse from the left, clockwise, turned so that the fewest overlap
      const around = (turn) =>
        clique.map((_, k) => {
          const angle = Math.PI + (2 * Math.PI * (k + turn)) / clique.length;
          return [width / 2 + rx * Math.cos(angle), height / 2 + ry * Math.sin(angle)];
        });
      const overlaps = (points) => {
        let count = 0;
        for (let a = 0; a < points.length; a += 1)
          for (let b = a + 1; b < points.length; b += 1)
            count +=
              Math.abs(points[a][0] - points[b][0]) < (nodes[a].offsetWidth + nodes[b].offsetWidth) / 2 + 4 &&
              Math.abs(points[a][1] - points[b][1]) < (nodes[a].offsetHeight + nodes[b].offsetHeight) / 2 + 4;
        return count;
      };
      const at = [0, 0.5, 0.25, 0.75]
        .map(around)
        .reduce((best, points) => (overlaps(points) < overlaps(best) ? points : best));
      if (!crowded && overlaps(at) > 0) {
        drawGraph(clique, true);
        return;
      }
      nodes.forEach((node, k) => {
        node.style.left = `${at[k][0]}px`;
        node.style.top = `${at[k][1]}px`;
      });
      links.setAttribute("viewBox", `0 0 ${width} ${height}`);
      const pairs = cliquePairs(clique);
      for (const { a, b, value, strength } of pairs) {
        links.append(line(...at[a], ...at[b], value, strength, [0.75, 9], [0.1, 0.8]));
      }
      // the labels of the strongest interactions, along their edges where they overlap neither each
      // other nor a node
      const placed = [];
      const free = ([x, y]) =>
        placed.every(([u, v]) => Math.abs(x - u) > 46 || Math.abs(y - v) > 16) &&
        nodes.every((node, k) =>
          Math.abs(x - at[k][0]) > node.offsetWidth / 2 + 24 || Math.abs(y - at[k][1]) > node.offsetHeight / 2 + 9,
        );
      for (const { a, b, value, strength } of pairs.slice(-4).reverse()) {
        if (strength < 0.3) break;
        const spot = [0.5, 0.4, 0.6, 0.3, 0.7]
          .map((t) => [at[a][0] + t * (at[b][0] - at[a][0]), at[a][1] + t * (at[b][1] - at[a][1])])
          .find(free);
        if (!spot) continue;
        placed.push(spot);
        const label = document.createElementNS(SVG, "text");
        label.setAttribute("x", spot[0]);
        label.setAttribute("y", spot[1]);
        label.setAttribute("class", sign(value));
        label.textContent = signed(value);
        links.append(label);
        const box = label.getBBox();
        const backdrop = document.createElementNS(SVG, "rect");
        for (const [key, v] of Object.entries({ x: box.x - 3, y: box.y - 1, width: box.width + 6, height: box.height + 2, rx: 4 }))
          backdrop.setAttribute(key, v);
        links.insertBefore(backdrop, label);
      }
    }

    caption.addEventListener("keydown", (event) => {
      const moves = { ArrowLeft: -1, ArrowRight: 1, Home: -Infinity, End: Infinity };
      const k = players.indexOf(event.target);
      if (!(event.key in moves) || k < regionCount) return;
      const next = Math.max(regionCount, Math.min(players.length - 1, k + moves[event.key]));
      event.preventDefault();
      players[next].focus();
    });
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

    stage.addEventListener("keydown", (event) => {
      // back to the view at rest before the browser picks the next stop, so that the size control
      // (hidden while a player is shown) is in the tab order
      if (event.key === "Tab" && state.focus !== null) {
        state.pinned = false;
        show(null);
      }
    });
    stage.addEventListener("pointerleave", () => {
      if (!state.pinned && state.focus !== null) show(null);
    });
    stage.addEventListener("focusout", (event) => {
      if (!state.pinned && !stage.contains(event.relatedTarget)) show(null);
    });
    document.addEventListener("pointerdown", (event) => {
      if (!about.hidden && !about.contains(event.target) && !info.contains(event.target)) setAbout(false);
      if (state.pinned && !stage.contains(event.target)) {
        state.pinned = false;
        show(null);
      }
    });
    root.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !about.hidden) {
        setAbout(false);
        info.focus();
      } else if (event.key === "Escape" && state.focus !== null) {
        state.pinned = false;
        show(null);
      }
    });
    // a new width lays the demo out again (in the next frame, outside the observer's callback)
    let width = -1;
    new ResizeObserver(([entry]) => {
      const now = Math.round(entry.contentRect.width);
      if (now === width) return;
      width = now;
      requestAnimationFrame(() => {
        setShift();
        reserveRoom();
        placeExcerpt();
        if (state.focus === null) show(null);
      });
    }).observe(root);

    root.classList.add("fixlip-demo-ready");
    reset();
    setView(state.view);
    started = true;
  }
})();
