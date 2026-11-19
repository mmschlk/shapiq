/* The figures of the talk "One Explanation Does Not Fit All" (talks/beyond-shap/index.rst).

   Every <div class="shapiq-talk-figure" data-figure="<name>"> of the page is drawn by FIGURES[name].
   A figure on a slide that builds up in steps follows the slide's step (the event
   "shapiq-talk-step" of _static/js/talk.js, and data-talk-step on the slide). A figure keeps its
   height in every step, and its parts fade in, so that nothing on the slide moves.

   same-shap   Three models of two features, with the same Shapley values. Step 0 shows the
               Shapley values only, step 1 what the models do, step 2 their Shapley interactions.
   effects     The same three models, with the pure, partial and full effect of each feature
               [Fumagalli et al., 2025]. Step 0 shows the partial effects (the Shapley values),
               step 1 the pure and the full effects around them.
   framework   The unified framework of feature-based explanations [Fumagalli et al., 2025]: how
               much the feature distribution and the higher-order interactions influence an
               explanation, with the methods that fall into each cell. Steps 0 to 2 are the three
               types of explanations (individual, joint, interaction); the reader can pick one too. */
(function () {
  "use strict";

  const SVG = "http://www.w3.org/2000/svg";
  const make = (tag, attributes = {}, parent = null, text = null) => {
    const element = tag.startsWith("svg:") ? document.createElementNS(SVG, tag.slice(4)) : document.createElement(tag);
    for (const [key, value] of Object.entries(attributes)) {
      if (key === "class") element.setAttribute("class", value);
      else element.setAttribute(key, value);
    }
    if (text != null) element.textContent = text;
    if (parent) parent.append(element);
    return element;
  };
  const format = (value) => (value > 0 ? "+" : value < 0 ? "−" : "") + Math.abs(value).toLocaleString("en");

  // ---- three models, one explanation ---------------------------------------------------------
  // Two features explained at x = (1, 1) against the baseline (0, 0). v(S) is the prediction when
  // only the features in S keep their value. All three models get the Shapley values (1, 1). Their
  // Shapley interactions of order 2 (here equal to the Möbius transform) differ.
  const MODELS = [
    {
      name: "Additive",
      formula: "f(x) = x₁ + x₂",
      says: "Each feature adds on its own",
      v: { "": 0, "1": 1, "2": 1, "12": 2 },
    },
    {
      name: "Synergy",
      formula: "f(x) = 2 · x₁ · x₂",
      says: "Only both together count",
      v: { "": 0, "1": 0, "2": 0, "12": 2 },
    },
    {
      name: "Redundancy",
      formula: "f(x) = 2 · max(x₁, x₂)",
      says: "Either feature is enough",
      v: { "": 0, "1": 2, "2": 2, "12": 2 },
    },
  ];
  for (const model of MODELS) {
    const v = model.v;
    model.shapley = [
      0.5 * (v["1"] - v[""]) + 0.5 * (v["12"] - v["2"]),
      0.5 * (v["2"] - v[""]) + 0.5 * (v["12"] - v["1"]),
    ];
    model.main = [v["1"] - v[""], v["2"] - v[""]];
    model.full = [v["12"] - v["2"], v["12"] - v["1"]];
    model.pair = v["12"] - v["1"] - v["2"] + v[""];
  }

  function sameShap(figure) {
    const row = make("div", { class: "talk-same" }, figure);
    const cards = MODELS.map((model, index) => {
      const card = make("div", { class: "talk-same-card" }, row);
      const head = make("div", { class: "talk-same-head" }, card);
      make("div", { class: "talk-same-model" }, head, `Model ${"ABC"[index]}`);
      const named = make("div", { class: "talk-same-name talk-from-1" }, head);
      make("strong", {}, named, model.name);
      make("span", {}, named, model.formula);

      // the Shapley values, as bars from 0 to 2
      const shap = make("div", { class: "talk-same-part" }, card);
      make("div", { class: "talk-same-label" }, shap, "Shapley values");
      const bars = make("svg:svg", { viewBox: "0 0 240 74", class: "talk-same-bars", "aria-hidden": "true" }, shap);
      const x0 = 34;
      const scale = 90;
      make("svg:line", { x1: x0, x2: x0, y1: 4, y2: 62, class: "talk-axis" }, bars);
      model.shapley.forEach((value, feature) => {
        const y = 10 + feature * 26;
        make("svg:text", { x: x0 - 8, y: y + 13, class: "talk-feature", "text-anchor": "end" }, bars, `x${"₁₂"[feature]}`);
        make("svg:rect", { x: x0, y, width: value * scale, height: 18, rx: 3, class: value >= 0 ? "talk-pos" : "talk-neg" }, bars);
        make("svg:text", { x: x0 + value * scale + 6, y: y + 13, class: "talk-value" }, bars, format(value));
      });
      [0, 1, 2].forEach((tick) => {
        make("svg:text", { x: x0 + tick * scale, y: 73, class: "talk-tick", "text-anchor": "middle" }, bars, tick);
      });

      // what the model does: the prediction for each set of present features
      const game = make("div", { class: "talk-same-part talk-from-1" }, card);
      make("div", { class: "talk-same-label" }, game, "Prediction with features present");
      const grid = make("div", { class: "talk-same-game" }, game);
      [["", "none"], ["1", "x₁"], ["2", "x₂"], ["12", "x₁, x₂"]].forEach(([set, label]) => {
        const cell = make("div", { class: "talk-same-cell" }, grid);
        cell.style.setProperty("--level", model.v[set] / 2);
        make("span", {}, cell, label);
        make("strong", {}, cell, model.v[set]);
      });

      // the Shapley interactions: main effects on the features, the pair as the link between them
      const si = make("div", { class: "talk-same-part talk-from-2" }, card);
      make("div", { class: "talk-same-label" }, si, "Shapley interactions");
      const graph = make("svg:svg", { viewBox: "0 0 240 84", class: "talk-same-graph", "aria-hidden": "true" }, si);
      const nodes = [[62, 42], [178, 42]];
      if (model.pair) {
        make("svg:line", {
          x1: nodes[0][0], y1: 42, x2: nodes[1][0], y2: 42,
          class: model.pair > 0 ? "talk-pos-line" : "talk-neg-line",
          "stroke-width": 4 + 4 * Math.abs(model.pair),
        }, graph);
        make("svg:text", { x: 120, y: 24, class: "talk-value", "text-anchor": "middle" }, graph, format(model.pair));
      } else {
        make("svg:line", { x1: nodes[0][0], y1: 42, x2: nodes[1][0], y2: 42, class: "talk-none-line" }, graph);
        make("svg:text", { x: 120, y: 24, class: "talk-value talk-muted", "text-anchor": "middle" }, graph, "0");
      }
      nodes.forEach(([x, y], feature) => {
        const main = model.main[feature];
        const r = 13 + 6 * Math.abs(main);
        make("svg:circle", { cx: x, cy: y, r, class: main > 0 ? "talk-pos" : main < 0 ? "talk-neg" : "talk-zero" }, graph);
        make("svg:text", { x, y: y + 4.5, class: "talk-node", "text-anchor": "middle" }, graph, `x${"₁₂"[feature]}`);
        make("svg:text", { x: feature ? x + r + 6 : x - r - 6, y: y + 4.5, class: "talk-value", "text-anchor": feature ? "start" : "end" }, graph, format(main));
      });
      make("div", { class: "talk-same-says talk-from-2" }, card, model.says);
      return card;
    });

    swipe(figure, row, cards);
    return { cards };
  }

  // on a phone, the cards are a row to swipe through, with a dot for each
  function swipe(figure, row, cards) {
    const dots = make("div", { class: "talk-same-dots", role: "tablist", "aria-label": "Models" }, figure);
    const marks = cards.map((card, index) => {
      const dot = make("button", { type: "button", role: "tab", "aria-label": `Model ${"ABC"[index]}` }, dots);
      dot.addEventListener("click", () => row.scrollTo({ left: card.offsetLeft - row.offsetLeft - (row.clientWidth - card.offsetWidth) / 2, behavior: "smooth" }));
      return dot;
    });
    const mark = () => {
      const middle = row.scrollLeft + row.clientWidth / 2;
      let best = 0;
      cards.forEach((card, index) => {
        const centre = card.offsetLeft - row.offsetLeft + card.offsetWidth / 2;
        const bestCentre = cards[best].offsetLeft - row.offsetLeft + cards[best].offsetWidth / 2;
        if (Math.abs(centre - middle) < Math.abs(bestCentre - middle)) best = index;
      });
      marks.forEach((dot, index) => dot.setAttribute("aria-selected", String(index === best)));
    };
    row.addEventListener("scroll", mark, { passive: true });
    mark();
  }

  // ---- pure, partial and full effects ------------------------------------------------------------
  // Fumagalli et al. (2025): an individual explanation takes in none of a feature's interactions
  // (pure, its main effect), a share of them (partial, the Shapley value) or all of them (full,
  // what the prediction loses when the feature alone is removed).
  const EFFECTS = [
    { key: "main", name: "Pure", says: "on its own", from: 1 },
    { key: "shapley", name: "Partial", says: "shared", from: 0 },
    { key: "full", name: "Full", says: "when removed", from: 1 },
  ];
  const VERDICTS = ["All three agree", "Alone nothing, removed everything", "Alone everything, removed nothing"];

  function effects(figure) {
    const row = make("div", { class: "talk-same talk-effects" }, figure);
    const cards = MODELS.map((model, index) => {
      const card = make("div", { class: "talk-same-card" }, row);
      const head = make("div", { class: "talk-same-head" }, card);
      make("div", { class: "talk-same-model" }, head, `Model ${"ABC"[index]}`);
      const named = make("div", { class: "talk-same-name" }, head);
      make("strong", {}, named, model.name);
      make("span", {}, named, model.formula);

      const part = make("div", { class: "talk-same-part" }, card);
      const bars = make("svg:svg", { viewBox: "0 0 260 196", class: "talk-same-bars", "aria-hidden": "true" }, part);
      const x0 = 110;
      const scale = 60;
      make("svg:line", { x1: x0, x2: x0, y1: 2, y2: 182, class: "talk-axis" }, bars);
      EFFECTS.forEach((effect, rank) => {
        const top = 6 + rank * 60;
        const group = make("svg:g", { class: `talk-from-${effect.from}` }, bars);
        make("svg:text", { x: 0, y: top + 17, class: "talk-effect-name" }, group, effect.name);
        make("svg:text", { x: 0, y: top + 36, class: "talk-tick" }, group, effect.says);
        model[effect.key].forEach((value, feature) => {
          const y = top + feature * 24;
          make("svg:text", { x: x0 - 6, y: y + 14, class: "talk-feature", "text-anchor": "end" }, group, `x${"₁₂"[feature]}`);
          if (value) make("svg:rect", { x: x0, y, width: value * scale, height: 19, rx: 3, class: value > 0 ? "talk-pos" : "talk-neg" }, group);
          make("svg:text", { x: x0 + value * scale + 6, y: y + 14, class: value ? "talk-value" : "talk-value talk-muted" }, group, format(value));
        });
      });
      [0, 1, 2].forEach((tick) => {
        make("svg:text", { x: x0 + tick * scale, y: 195, class: "talk-tick", "text-anchor": "middle" }, bars, tick);
      });
      make("div", { class: "talk-same-says talk-from-1" }, card, VERDICTS[index]);
      return card;
    });
    swipe(figure, row, cards);
    return { cards };
  }

  // ---- the unified framework -----------------------------------------------------------------
  // Fumagalli et al. (2025), Section 4.1 and Table 2: local feature-based explanations by the
  // fANOVA decomposition they rest on (b, m or c, the columns) and how they summarize the Möbius
  // transform (pure, partial or full effects, the rows), for each type of explanation.
  const DISTRIBUTIONS = [
    { key: "b", name: "Baseline", says: "Absent features take a fixed value" },
    { key: "m", name: "Marginal", says: "Averaged over their distribution" },
    { key: "c", name: "Conditional", says: "Averaged given the present features" },
  ];
  // "*" is a method defined on any game: it has a variant for each of the three distributions, so it
  // spans the row [Fumagalli et al., 2025, Appendix E; Herbinger et al., 2026]. Global feature
  // effects (PDP, M plot, ALE) stand where the paper places them. "≈" marks path-dependent
  // TreeSHAP, which approximates the conditional distribution by the trees' own splits.
  const TYPES = [
    {
      name: "Individual",
      rows: { pure: "Main effect", partial: "Shapley value", full: "All effects with the feature" },
      methods: [
        ["b", "pure", "ArchAttribute"],
        ["m", "pure", "PDP"],
        ["c", "pure", "M plot"],
        ["c", "pure", "ALE"],
        ["b", "partial", "Baseline SHAP"],
        ["m", "partial", "Interventional SHAP"],
        ["c", "partial", "Observational SHAP"],
        ["c", "partial", "≈ Path-dependent TreeSHAP"],
        ["b", "full", "Occlusion-1"],
        ["b", "full", "ICE"],
        ["c", "full", "PredDiff-1"],
      ],
    },
    {
      name: "Joint",
      rows: { pure: "Effects within the group", partial: "Generalized value", full: "All effects that touch the group" },
      methods: [
        ["b", "pure", "ArchAttribute"],
        ["m", "pure", "PDP"],
        ["c", "pure", "M plot"],
        ["*", "partial", "Joint Shapley values"],
        ["b", "full", "Occlusion-patch"],
        ["b", "full", "ICE"],
        ["c", "full", "PredDiff-patch"],
      ],
    },
    {
      name: "Interaction",
      rows: { pure: "Möbius transform", partial: "Shapley interaction", full: "Co-Möbius transform" },
      methods: [
        ["*", "pure", "Shapley-GAM"],
        ["*", "partial", "k-SVs (k-SII)"],
        ["*", "partial", "Faith-SHAP"],
        ["*", "partial", "Shapley-Taylor"],
        ["c", "full", "PredDiff-interaction"],
      ],
    },
  ];
  const ROWS = [
    { key: "pure", name: "Pure" },
    { key: "partial", name: "Partial" },
    { key: "full", name: "Full" },
  ];

  function framework(figure) {
    figure.classList.add("talk-frame");
    const switcher = make("div", { class: "talk-frame-types", role: "radiogroup", "aria-label": "Type of explanation" }, figure);
    const buttons = TYPES.map((type, index) => {
      const button = make("button", { type: "button", role: "radio", "aria-checked": "false" }, switcher, type.name);
      button.addEventListener("click", () => show(index));
      return button;
    });

    const table = make("div", { class: "talk-frame-grid" }, figure);
    const corner = make("div", { class: "talk-frame-corner" }, table);
    make("span", {}, corner, "Feature distribution →");
    make("span", {}, corner, "Interactions ↓");
    for (const distribution of DISTRIBUTIONS) {
      const head = make("div", { class: "talk-frame-col" }, table);
      make("strong", {}, head, distribution.name);
      make("span", {}, head, distribution.says);
    }
    const cells = {};
    const rowNames = {};
    for (const row of ROWS) {
      const head = make("div", { class: "talk-frame-row" }, table);
      make("strong", {}, head, row.name);
      rowNames[row.key] = make("span", { class: "talk-swap" }, head);
      for (const distribution of DISTRIBUTIONS) {
        cells[distribution.key + row.key] = make("div", { class: "talk-frame-cell" }, table);
      }
    }

    let shown = -1;
    function show(index) {
      if (index === shown) return;
      shown = index;
      const type = TYPES[index];
      buttons.forEach((button, other) => button.setAttribute("aria-checked", String(other === index)));
      for (const row of ROWS) swap(rowNames[row.key], type.rows[row.key]);
      for (const row of ROWS) {
        // a method of any game spans the row, with a word that it takes each of the three
        const any = type.methods.filter(([d, r]) => d === "*" && r === row.key).map(([, , name]) => name);
        DISTRIBUTIONS.forEach((distribution, column) => {
          const cell = cells[distribution.key + row.key];
          cell.classList.toggle("talk-frame-span", any.length > 0 && column === 0);
          cell.classList.toggle("talk-frame-gone", any.length > 0 && column > 0);
          const names = any.length
            ? column === 0 ? [...any, "§ with any of the three distributions"] : []
            : type.methods.filter(([d, r]) => d === distribution.key && r === row.key).map(([, , name]) => name);
          swap(cell, names, true);
        });
      }
    }
    return { show };
  }

  // a text or a list of chips that fades out before the next fades in, in a place that keeps its size
  function swap(element, content, chips = false) {
    const key = JSON.stringify(content);
    if (element.dataset.content === key) return;
    const first = element.dataset.content == null;
    element.dataset.content = key;
    const fill = () => {
      element.textContent = "";
      if (chips) {
        // a chip that starts with "≈" approximates its cell (path-dependent TreeSHAP, whose absent
        // features follow the splits of the trees, near the conditional distribution)
        for (const name of content) {
          if (name.startsWith("§ ")) {
            make("span", { class: "talk-frame-any" }, element, name.slice(2));
            continue;
          }
          const approx = name.startsWith("≈ ");
          const chip = make("span", { class: approx ? "talk-chip talk-chip-approx" : "talk-chip" }, element, name);
          if (approx) chip.title = "An approximation: absent features follow the trees' own splits";
        }
      }
      else element.textContent = content;
      element.classList.remove("talk-out");
    };
    if (first || window.matchMedia("(prefers-reduced-motion: reduce)").matches) fill();
    else {
      element.classList.add("talk-out");
      clearTimeout(element.shapiqSwap);
      element.shapiqSwap = setTimeout(fill, 160);
    }
  }

  // ---- steps ---------------------------------------------------------------------------------

  const FIGURES = { "same-shap": sameShap, effects, framework };

  for (const figure of document.querySelectorAll(".shapiq-talk-figure")) {
    const draw = FIGURES[figure.dataset.figure];
    if (!draw) continue;
    figure.textContent = "";
    const api = draw(figure);
    const slide = figure.closest(".shapiq-slide");
    const follow = (step) => {
      figure.dataset.step = String(step);
      for (const part of figure.querySelectorAll('[class*="talk-from-"]')) {
        const from = Number(/talk-from-(\d+)/.exec(part.getAttribute("class"))[1]);
        part.classList.toggle("talk-later", from > step);
      }
      if (api.show) api.show(Math.min(step, TYPES.length - 1));
    };
    // without a slide that says otherwise, all of it
    follow(slide && slide.dataset.talkStep != null ? Number(slide.dataset.talkStep) : 99);
    if (slide) slide.addEventListener("shapiq-talk-step", (event) => follow(event.detail.step));
  }
})();
