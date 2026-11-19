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
   faithfulness  How well an explanation predicts the model: for the credit applicant of the orders
               demo, what the model predicts for each coalition of features against what the
               explanation of an order predicts, with its R². Step 0 shows the Shapley values,
               step 1 the explanation of order 2. The reader can pick any order.
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

  // ---- faithfulness ------------------------------------------------------------------------------
  // The credit applicant of the orders demo. For each of the 254 coalitions of its 8 features (but
  // the empty and the full one): the model's prediction with only these features present, and an
  // explanation's, the sum of its values over the coalition's subsets. A point's size is the
  // coalition's probability under the Shapley distribution, which weighs the R² (beyond size).
  // FAITHFULNESS (compute/faithfulness.py)
  const FAITHFULNESS = {"features":["Credit limit","Age","Status Sep","Status Aug","Status Jul","Bill Sep","Paid Sep","Paid Aug"],"probability":[1.0,1.0,0.167,1.0,0.167,0.167,0.067,1.0,0.167,0.167,0.067,0.167,0.067,0.067,0.05,1.0,0.167,0.167,0.067,0.167,0.067,0.067,0.05,0.167,0.067,0.067,0.05,0.067,0.05,0.05,0.067,1.0,0.167,0.167,0.067,0.167,0.067,0.067,0.05,0.167,0.067,0.067,0.05,0.067,0.05,0.05,0.067,0.167,0.067,0.067,0.05,0.067,0.05,0.05,0.067,0.067,0.05,0.05,0.067,0.05,0.067,0.067,0.167,1.0,0.167,0.167,0.067,0.167,0.067,0.067,0.05,0.167,0.067,0.067,0.05,0.067,0.05,0.05,0.067,0.167,0.067,0.067,0.05,0.067,0.05,0.05,0.067,0.067,0.05,0.05,0.067,0.05,0.067,0.067,0.167,0.167,0.067,0.067,0.05,0.067,0.05,0.05,0.067,0.067,0.05,0.05,0.067,0.05,0.067,0.067,0.167,0.067,0.05,0.05,0.067,0.05,0.067,0.067,0.167,0.05,0.067,0.067,0.167,0.067,0.167,0.167,1.0,1.0,0.167,0.167,0.067,0.167,0.067,0.067,0.05,0.167,0.067,0.067,0.05,0.067,0.05,0.05,0.067,0.167,0.067,0.067,0.05,0.067,0.05,0.05,0.067,0.067,0.05,0.05,0.067,0.05,0.067,0.067,0.167,0.167,0.067,0.067,0.05,0.067,0.05,0.05,0.067,0.067,0.05,0.05,0.067,0.05,0.067,0.067,0.167,0.067,0.05,0.05,0.067,0.05,0.067,0.067,0.167,0.05,0.067,0.067,0.167,0.067,0.167,0.167,1.0,0.167,0.067,0.067,0.05,0.067,0.05,0.05,0.067,0.067,0.05,0.05,0.067,0.05,0.067,0.067,0.167,0.067,0.05,0.05,0.067,0.05,0.067,0.067,0.167,0.05,0.067,0.067,0.167,0.067,0.167,0.167,1.0,0.067,0.05,0.05,0.067,0.05,0.067,0.067,0.167,0.05,0.067,0.067,0.167,0.067,0.167,0.167,1.0,0.05,0.067,0.067,0.167,0.067,0.167,0.167,1.0,0.067,0.167,0.167,1.0,0.167,1.0,1.0],"model":[-1.464,-1.77,-1.618,-1.313,-0.546,-1.028,-0.226,-0.897,-0.812,-1.004,-0.925,-0.48,0.308,-0.341,0.424,-1.142,-0.899,-1.391,-1.107,-0.952,-0.036,-0.712,0.229,-0.492,-0.407,-0.609,-0.536,-0.079,0.707,0.028,0.786,-1.618,-1.521,-1.789,-1.635,-1.452,-0.719,-1.257,-0.485,-1.253,-1.186,-1.3,-1.235,-0.971,-0.231,-0.925,-0.207,-0.944,-0.839,-1.1,-0.94,-0.813,-0.06,-0.624,0.166,-0.32,-0.346,-0.339,-0.369,-0.056,0.594,-0.005,0.622,-1.641,-1.564,-1.69,-1.543,-1.52,-0.761,-1.07,-0.267,-0.932,-0.849,-0.91,-0.818,-0.604,0.198,-0.265,0.518,-1.188,-0.984,-1.295,-1.032,-1.146,-0.26,-0.757,0.162,-0.509,-0.441,-0.509,-0.439,-0.193,0.585,0.101,0.854,-1.757,-1.671,-1.773,-1.611,-1.658,-0.926,-1.283,-0.504,-1.374,-1.301,-1.293,-1.208,-1.165,-0.411,-0.926,-0.189,-1.046,-0.964,-1.059,-0.904,-0.992,-0.26,-0.636,0.14,-0.414,-0.448,-0.318,-0.343,-0.227,0.418,0.005,0.63,-2.015,-1.839,-2.192,-1.961,-1.1,-0.069,-0.909,0.136,-1.628,-1.356,-1.503,-1.217,-0.23,0.924,-0.087,1.039,-1.86,-1.677,-2.068,-1.838,-0.992,0.056,-0.817,0.239,-1.372,-1.218,-1.249,-1.084,0.046,1.084,0.182,1.192,-2.05,-1.873,-2.13,-1.899,-1.284,-0.281,-1.129,-0.112,-1.82,-1.623,-1.732,-1.543,-0.588,0.464,-0.657,0.345,-1.617,-1.52,-1.692,-1.544,-0.87,0.069,-0.714,0.237,-0.986,-0.959,-0.872,-0.856,0.254,1.143,0.205,1.044,-2.13,-1.965,-2.09,-1.852,-1.365,-0.329,-0.985,0.083,-1.722,-1.451,-1.411,-1.109,-0.421,0.749,-0.049,1.108,-1.952,-1.805,-1.961,-1.75,-1.222,-0.2,-0.877,0.171,-1.458,-1.33,-1.165,-1.011,-0.115,0.909,0.234,1.243,-2.207,-2.028,-2.077,-1.826,-1.522,-0.508,-1.17,-0.124,-1.979,-1.769,-1.717,-1.501,-0.827,0.247,-0.685,0.354,-1.737,-1.653,-1.618,-1.464,-1.052,-0.121,-0.715,0.245,-1.123,-1.096,-0.852,-0.823,0.066,0.959,0.211],"orders":{"1":{"r2":0.2381,"y":[-1.061,-1.497,-0.994,-0.469,0.034,-0.402,0.101,-0.868,-0.365,-0.8,-0.298,0.227,0.73,0.294,0.797,-1.049,-0.546,-0.982,-0.479,0.046,0.548,0.113,0.616,-0.353,0.15,-0.286,0.217,0.742,1.245,0.809,1.312,-1.659,-1.156,-1.592,-1.089,-0.564,-0.062,-0.497,0.006,-0.963,-0.46,-0.896,-0.393,0.132,0.635,0.199,0.702,-1.145,-0.642,-1.077,-0.575,-0.05,0.453,0.017,0.52,-0.449,0.054,-0.381,0.122,0.646,1.149,0.714,1.216,-1.612,-1.109,-1.545,-1.042,-0.517,-0.014,-0.45,0.053,-0.916,-0.413,-0.849,-0.346,0.179,0.682,0.246,0.749,-1.098,-0.595,-1.03,-0.527,-0.003,0.5,0.065,0.568,-0.401,0.102,-0.334,0.169,0.694,1.196,0.761,1.264,-1.708,-1.205,-1.64,-1.137,-0.613,-0.11,-0.545,-0.043,-1.011,-0.509,-0.944,-0.441,0.084,0.586,0.151,0.654,-1.193,-0.69,-1.126,-0.623,-0.098,0.405,-0.031,0.472,-0.497,0.006,-0.43,0.073,0.598,1.101,0.665,1.168,-1.665,-1.162,-1.598,-1.095,-0.57,-0.067,-0.503,-0.0,-0.969,-0.466,-0.902,-0.399,0.126,0.629,0.193,0.696,-1.151,-0.648,-1.083,-0.58,-0.056,0.447,0.012,0.514,-0.454,0.048,-0.387,0.116,0.641,1.143,0.708,1.211,-1.761,-1.258,-1.693,-1.19,-0.666,-0.163,-0.598,-0.096,-1.064,-0.562,-0.997,-0.494,0.031,0.533,0.098,0.601,-1.246,-0.743,-1.179,-0.676,-0.151,0.352,-0.084,0.419,-0.55,-0.047,-0.483,0.02,0.545,1.048,0.612,1.115,-1.713,-1.211,-1.646,-1.143,-0.618,-0.116,-0.551,-0.048,-1.017,-0.514,-0.95,-0.447,0.078,0.581,0.145,0.648,-1.199,-0.696,-1.132,-0.629,-0.104,0.399,-0.037,0.466,-0.503,0.0,-0.435,0.068,0.592,1.095,0.66,1.162,-1.809,-1.306,-1.742,-1.239,-0.714,-0.211,-0.647,-0.144,-1.113,-0.61,-1.045,-0.543,-0.018,0.485,0.05,0.552,-1.294,-0.791,-1.227,-0.724,-0.199,0.304,-0.132,0.371,-0.598,-0.095,-0.531,-0.028,0.497,1.0,0.564]},"2":{"r2":0.9803,"y":[-1.44,-1.673,-1.534,-1.296,-0.419,-1.212,-0.32,-0.963,-0.888,-1.092,-1.001,-0.492,0.336,-0.428,0.415,-1.188,-1.105,-1.314,-1.216,-0.937,-0.102,-0.871,-0.02,-0.432,-0.398,-0.577,-0.528,0.021,0.808,0.069,0.871,-1.613,-1.567,-1.757,-1.696,-1.548,-0.749,-1.499,-0.685,-1.223,-1.226,-1.386,-1.374,-0.955,-0.205,-0.925,-0.16,-0.838,-0.833,-0.998,-0.978,-0.79,-0.033,-0.758,0.015,-0.292,-0.337,-0.472,-0.501,-0.042,0.666,-0.029,0.695,-1.662,-1.539,-1.589,-1.451,-1.469,-0.593,-1.204,-0.313,-1.03,-0.956,-0.976,-0.887,-0.635,0.193,-0.388,0.454,-1.273,-1.191,-1.217,-1.12,-1.098,-0.263,-0.849,0.0,-0.486,-0.453,-0.449,-0.401,-0.108,0.678,0.122,0.923,-1.745,-1.7,-1.707,-1.647,-1.756,-0.958,-1.525,-0.712,-1.324,-1.328,-1.305,-1.294,-1.132,-0.382,-0.92,-0.156,-0.957,-0.954,-0.936,-0.917,-0.985,-0.229,-0.771,0.001,-0.38,-0.426,-0.378,-0.408,-0.206,0.502,-0.011,0.712,-2.121,-1.838,-2.197,-1.899,-1.052,-0.016,-0.935,0.116,-1.44,-1.207,-1.535,-1.286,-0.168,0.819,-0.07,0.932,-1.96,-1.719,-2.052,-1.796,-0.908,0.086,-0.808,0.202,-1.124,-0.931,-1.235,-1.027,0.131,1.076,0.212,1.173,-2.102,-1.897,-2.212,-1.992,-1.235,-0.278,-1.153,-0.18,-1.632,-1.476,-1.761,-1.59,-0.562,0.346,-0.499,0.425,-1.541,-1.378,-1.668,-1.489,-0.692,0.224,-0.626,0.305,-0.915,-0.801,-1.061,-0.932,0.136,1.003,0.183,1.065,-2.235,-1.953,-2.129,-1.832,-1.241,-0.206,-0.942,0.108,-1.523,-1.29,-1.436,-1.187,-0.326,0.66,-0.046,0.955,-2.061,-1.82,-1.971,-1.716,-1.084,-0.091,-0.802,0.206,-1.193,-1.002,-1.123,-0.916,-0.014,0.93,0.249,1.209,-2.25,-2.046,-2.178,-1.959,-1.459,-0.502,-1.195,-0.223,-1.748,-1.593,-1.696,-1.525,-0.755,0.153,-0.509,0.414,-1.676,-1.514,-1.621,-1.444,-0.903,0.012,-0.655,0.275,-1.019,-0.906,-0.983,-0.855,-0.043,0.823,0.185]},"3":{"r2":0.9992,"y":[-1.465,-1.789,-1.631,-1.32,-0.547,-1.024,-0.224,-0.915,-0.821,-0.982,-0.888,-0.476,0.319,-0.316,0.45,-1.135,-0.915,-1.398,-1.124,-0.956,-0.057,-0.705,0.217,-0.51,-0.413,-0.594,-0.5,-0.098,0.706,0.039,0.809,-1.632,-1.538,-1.764,-1.614,-1.453,-0.714,-1.214,-0.45,-1.222,-1.168,-1.287,-1.234,-0.946,-0.22,-0.932,-0.239,-0.966,-0.833,-1.097,-0.914,-0.842,-0.061,-0.609,0.192,-0.296,-0.323,-0.339,-0.371,-0.038,0.613,-0.009,0.604,-1.641,-1.561,-1.71,-1.557,-1.533,-0.771,-1.058,-0.255,-0.948,-0.861,-0.864,-0.761,-0.59,0.206,-0.256,0.525,-1.191,-1.009,-1.313,-1.064,-1.145,-0.277,-0.73,0.174,-0.53,-0.459,-0.479,-0.397,-0.197,0.588,0.099,0.864,-1.762,-1.674,-1.742,-1.584,-1.676,-0.936,-1.261,-0.483,-1.366,-1.306,-1.284,-1.211,-1.128,-0.388,-0.944,-0.223,-1.056,-0.95,-1.051,-0.88,-1.023,-0.261,-0.63,0.165,-0.409,-0.448,-0.32,-0.351,-0.186,0.458,-0.003,0.618,-2.027,-1.846,-2.153,-1.919,-1.108,-0.074,-0.912,0.145,-1.605,-1.347,-1.503,-1.249,-0.205,0.933,-0.074,1.03,-1.892,-1.695,-2.046,-1.799,-0.991,0.065,-0.83,0.244,-1.352,-1.195,-1.257,-1.108,0.068,1.111,0.186,1.189,-2.043,-1.874,-2.122,-1.903,-1.239,-0.248,-1.146,-0.136,-1.805,-1.595,-1.747,-1.543,-0.619,0.442,-0.681,0.343,-1.612,-1.511,-1.681,-1.534,-0.818,0.112,-0.721,0.223,-0.973,-0.947,-0.883,-0.868,0.242,1.124,0.205,1.044,-2.145,-1.973,-2.075,-1.836,-1.37,-0.338,-0.956,0.112,-1.7,-1.438,-1.407,-1.135,-0.391,0.759,-0.046,1.083,-1.982,-1.814,-1.956,-1.725,-1.224,-0.19,-0.86,0.206,-1.428,-1.287,-1.158,-1.011,-0.096,0.938,0.219,1.228,-2.188,-2.016,-2.076,-1.84,-1.487,-0.485,-1.18,-0.145,-1.986,-1.759,-1.741,-1.507,-0.847,0.238,-0.7,0.361,-1.712,-1.626,-1.605,-1.46,-1.017,-0.097,-0.722,0.227,-1.116,-1.094,-0.855,-0.83,0.054,0.939,0.21]},"4":{"r2":1.0,"y":[-1.464,-1.769,-1.618,-1.313,-0.545,-1.028,-0.228,-0.897,-0.813,-1.006,-0.924,-0.48,0.307,-0.341,0.424,-1.142,-0.9,-1.389,-1.105,-0.951,-0.035,-0.713,0.227,-0.491,-0.406,-0.613,-0.536,-0.081,0.706,0.028,0.788,-1.617,-1.522,-1.791,-1.634,-1.453,-0.721,-1.257,-0.483,-1.255,-1.186,-1.299,-1.234,-0.971,-0.228,-0.924,-0.207,-0.942,-0.838,-1.103,-0.942,-0.814,-0.062,-0.623,0.167,-0.323,-0.349,-0.334,-0.367,-0.055,0.595,-0.003,0.621,-1.641,-1.563,-1.69,-1.544,-1.52,-0.76,-1.07,-0.268,-0.933,-0.851,-0.909,-0.816,-0.602,0.197,-0.265,0.519,-1.188,-0.985,-1.293,-1.032,-1.145,-0.261,-0.757,0.161,-0.508,-0.441,-0.51,-0.44,-0.193,0.583,0.101,0.855,-1.758,-1.674,-1.771,-1.609,-1.655,-0.927,-1.284,-0.504,-1.372,-1.297,-1.293,-1.21,-1.166,-0.411,-0.926,-0.19,-1.047,-0.964,-1.06,-0.904,-0.991,-0.261,-0.638,0.141,-0.416,-0.447,-0.317,-0.343,-0.228,0.419,0.005,0.63,-2.015,-1.838,-2.192,-1.96,-1.1,-0.071,-0.91,0.137,-1.629,-1.355,-1.503,-1.217,-0.23,0.924,-0.088,1.039,-1.859,-1.676,-2.068,-1.838,-0.99,0.056,-0.821,0.238,-1.372,-1.217,-1.248,-1.087,0.044,1.082,0.186,1.193,-2.051,-1.873,-2.128,-1.899,-1.283,-0.28,-1.129,-0.113,-1.819,-1.622,-1.731,-1.544,-0.587,0.462,-0.657,0.345,-1.619,-1.52,-1.691,-1.546,-0.872,0.068,-0.712,0.238,-0.986,-0.959,-0.872,-0.855,0.255,1.143,0.203,1.044,-2.128,-1.967,-2.092,-1.854,-1.367,-0.334,-0.983,0.086,-1.724,-1.452,-1.408,-1.107,-0.419,0.752,-0.05,1.107,-1.951,-1.806,-1.964,-1.75,-1.223,-0.201,-0.877,0.173,-1.461,-1.33,-1.164,-1.01,-0.116,0.911,0.235,1.243,-2.209,-2.028,-2.076,-1.824,-1.522,-0.504,-1.172,-0.125,-1.977,-1.768,-1.719,-1.502,-0.829,0.246,-0.685,0.354,-1.738,-1.652,-1.617,-1.464,-1.053,-0.12,-0.714,0.245,-1.121,-1.097,-0.853,-0.823,0.066,0.959,0.211]},"5":{"r2":1.0,"y":[-1.464,-1.77,-1.618,-1.313,-0.546,-1.028,-0.226,-0.897,-0.812,-1.004,-0.925,-0.48,0.308,-0.341,0.424,-1.142,-0.899,-1.391,-1.107,-0.952,-0.036,-0.712,0.229,-0.492,-0.407,-0.609,-0.536,-0.079,0.707,0.028,0.786,-1.618,-1.521,-1.789,-1.635,-1.452,-0.719,-1.257,-0.485,-1.253,-1.186,-1.3,-1.235,-0.971,-0.231,-0.925,-0.207,-0.944,-0.839,-1.1,-0.94,-0.813,-0.06,-0.624,0.166,-0.32,-0.346,-0.339,-0.369,-0.056,0.594,-0.005,0.622,-1.641,-1.564,-1.69,-1.543,-1.52,-0.761,-1.07,-0.267,-0.932,-0.849,-0.91,-0.818,-0.604,0.198,-0.265,0.518,-1.188,-0.984,-1.295,-1.032,-1.146,-0.26,-0.757,0.162,-0.509,-0.441,-0.509,-0.439,-0.193,0.585,0.101,0.854,-1.757,-1.671,-1.773,-1.611,-1.658,-0.926,-1.283,-0.504,-1.374,-1.301,-1.293,-1.208,-1.165,-0.411,-0.926,-0.189,-1.046,-0.964,-1.059,-0.904,-0.992,-0.26,-0.636,0.14,-0.414,-0.448,-0.318,-0.343,-0.227,0.418,0.005,0.63,-2.015,-1.839,-2.192,-1.961,-1.1,-0.069,-0.909,0.136,-1.628,-1.356,-1.503,-1.217,-0.23,0.924,-0.087,1.039,-1.86,-1.677,-2.068,-1.838,-0.992,0.056,-0.817,0.239,-1.372,-1.218,-1.249,-1.084,0.046,1.084,0.182,1.192,-2.05,-1.873,-2.13,-1.899,-1.284,-0.281,-1.129,-0.112,-1.82,-1.623,-1.732,-1.543,-0.588,0.464,-0.657,0.345,-1.617,-1.52,-1.692,-1.544,-0.87,0.069,-0.714,0.237,-0.986,-0.959,-0.872,-0.856,0.254,1.143,0.205,1.044,-2.13,-1.965,-2.09,-1.852,-1.365,-0.329,-0.985,0.083,-1.722,-1.451,-1.411,-1.109,-0.421,0.749,-0.049,1.108,-1.952,-1.805,-1.961,-1.75,-1.222,-0.2,-0.877,0.171,-1.458,-1.33,-1.165,-1.011,-0.115,0.909,0.234,1.243,-2.207,-2.028,-2.077,-1.826,-1.522,-0.508,-1.17,-0.124,-1.979,-1.769,-1.717,-1.501,-0.827,0.247,-0.685,0.354,-1.737,-1.653,-1.618,-1.464,-1.052,-0.121,-0.715,0.245,-1.123,-1.096,-0.852,-0.823,0.066,0.959,0.211]}}};
  // /FAITHFULNESS

  function faithfulness(figure) {
    const data = FAITHFULNESS;
    const orders = Object.keys(data.orders);
    figure.classList.add("talk-faith");
    const switcher = make("div", { class: "talk-frame-types", role: "radiogroup", "aria-label": "Order of the explanation" }, figure);
    const buttons = orders.map((order) => {
      const button = make("button", { type: "button", role: "radio", "aria-checked": "false" }, switcher, order === "1" ? "Shapley values" : `Order ${order}`);
      button.addEventListener("click", () => show(orders.indexOf(order)));
      return button;
    });

    const W = 520, H = 330, left = 58, right = 16, top = 14, bottom = 46;
    const all = data.model.concat(...orders.map((order) => data.orders[order].y));
    const lo = Math.floor(Math.min(...all)), hi = Math.ceil(Math.max(...all));
    const x = (v) => left + ((v - lo) / (hi - lo)) * (W - left - right);
    const y = (v) => H - bottom - ((v - lo) / (hi - lo)) * (H - top - bottom);
    const plot = make("div", { class: "talk-faith-plot" }, figure);
    const svg = make("svg:svg", { viewBox: `0 0 ${W} ${H}`, class: "talk-faith-svg", role: "img", "aria-label": "Predictions of the model against those of the explanation" }, plot);
    for (let tick = lo; tick <= hi; tick += 1) {
      make("svg:line", { x1: x(tick), x2: x(tick), y1: top, y2: H - bottom, class: "talk-grid" }, svg);
      make("svg:line", { x1: left, x2: W - right, y1: y(tick), y2: y(tick), class: "talk-grid" }, svg);
      make("svg:text", { x: x(tick), y: H - bottom + 16, class: "talk-tick", "text-anchor": "middle" }, svg, tick);
      make("svg:text", { x: left - 8, y: y(tick) + 4, class: "talk-tick", "text-anchor": "end" }, svg, tick);
    }
    make("svg:line", { x1: x(lo), y1: y(lo), x2: x(hi), y2: y(hi), class: "talk-faith-diagonal" }, svg);
    // (the label along the diagonal, above it, where no point lies)
    const at = hi - 0.45, angle = (Math.atan2(y(lo) - y(hi), x(hi) - x(lo)) * 180) / Math.PI;
    make("svg:text", { x: x(at), y: y(at) - 6, class: "talk-faith-note", "text-anchor": "middle", transform: `rotate(${-angle} ${x(at)} ${y(at) - 6})` }, svg, "faithful");
    make("svg:text", { x: (left + W - right) / 2, y: H - 6, class: "talk-axis-name", "text-anchor": "middle" }, svg, "the model, with only some features (log-odds)");
    make("svg:text", { x: 20, y: (top + H - bottom) / 2, class: "talk-axis-name", "text-anchor": "middle", transform: `rotate(-90 20 ${(top + H - bottom) / 2})` }, svg, "the explanation");
    const points = data.model.map((value, index) => make("svg:circle", {
      cx: x(value), cy: y(value), r: 1.8 + 4.2 * Math.sqrt(data.probability[index]), class: "talk-faith-point",
    }, svg));

    const readout = make("div", { class: "talk-faith-readout" }, figure);
    make("span", { class: "talk-faith-label" }, readout, "Faithfulness (R²)");
    const meter = make("span", { class: "talk-faith-meter" }, readout);
    const fill = make("span", {}, meter);
    const score = make("strong", {}, readout);

    let shown = -1;
    function show(index) {
      if (index === shown) return;
      shown = index;
      const order = orders[index];
      buttons.forEach((button, other) => button.setAttribute("aria-checked", String(other === index)));
      data.orders[order].y.forEach((value, k) => points[k].setAttribute("cy", y(value)));
      const r2 = data.orders[order].r2;
      fill.style.transform = `scaleX(${Math.max(0, r2)})`;
      score.textContent = r2.toFixed(2);
    }
    // (the slide's steps show the Shapley values and the pairs, its last step the pairs too)
    return { show, steps: 2 };
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
  // TreeSHAP, which approximates the conditional distribution by the trees' own splits. The last
  // entry of a method is its key in references.bib: the chip links to the bibliography of the
  // docs, as every citation does (the slide holds a hidden citation of each, .shapiq-cite-<key>).
  const TYPES = [
    {
      name: "Individual",
      rows: { pure: "Main effect", partial: "Shapley value", full: "All effects with the feature" },
      methods: [
        ["b", "pure", "ArchAttribute", "Tsang.2020"],
        ["m", "pure", "PDP", "Friedman.2001"],
        ["c", "pure", "M plot", "Apley.2020"],
        ["c", "pure", "ALE", "Apley.2020"],
        ["b", "partial", "Baseline SHAP", "Sundararajan.2020a"],
        ["m", "partial", "Interventional SHAP", "Janzing.2020"],
        ["c", "partial", "Observational SHAP", "Lundberg.2017"],
        ["c", "partial", "≈ Path-dependent TreeSHAP", "Lundberg.2020"],
        ["b", "full", "Occlusion-1", "Zeiler.2014"],
        ["b", "full", "ICE", "Goldstein.2015"],
        ["c", "full", "PredDiff-1", "Zintgraf.2017"],
      ],
    },
    {
      name: "Joint",
      rows: { pure: "Effects within the group", partial: "Generalized value", full: "All effects that touch the group" },
      methods: [
        ["b", "pure", "ArchAttribute", "Tsang.2020"],
        ["m", "pure", "PDP", "Friedman.2001"],
        ["c", "pure", "M plot", "Apley.2020"],
        ["*", "partial", "Joint Shapley values", "Harris.2022"],
        ["b", "full", "Occlusion-patch", "Zeiler.2014"],
        ["b", "full", "ICE", "Goldstein.2015"],
        ["c", "full", "PredDiff-patch", "Zintgraf.2017"],
      ],
    },
    {
      name: "Interaction",
      rows: { pure: "Möbius transform", partial: "Shapley interaction", full: "Co-Möbius transform" },
      methods: [
        ["*", "pure", "Shapley-GAM", "Bordt.2023"],
        ["*", "partial", "k-SVs (k-SII)", "Bordt.2023"],
        ["*", "partial", "Faith-SHAP", "Tsai.2022"],
        ["*", "partial", "Shapley-Taylor", "Sundararajan.2020"],
        ["c", "full", "PredDiff-interaction", "Blucher.2022"],
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
        const any = type.methods.filter(([d, r]) => d === "*" && r === row.key).map(([, , name, key]) => [name, key]);
        DISTRIBUTIONS.forEach((distribution, column) => {
          const cell = cells[distribution.key + row.key];
          cell.classList.toggle("talk-frame-span", any.length > 0 && column === 0);
          cell.classList.toggle("talk-frame-gone", any.length > 0 && column > 0);
          const names = any.length
            ? column === 0 ? [...any, ["§ with any of the three distributions"]] : []
            : type.methods.filter(([d, r]) => d === distribution.key && r === row.key).map(([, , name, key]) => [name, key]);
          swap(cell, names, true);
        });
      }
    }
    return { show, steps: TYPES.length };
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
        for (const [name, key] of content) {
          if (name.startsWith("§ ")) {
            make("span", { class: "talk-frame-any" }, element, name.slice(2));
            continue;
          }
          const approx = name.startsWith("≈ ");
          // the citation of the docs for the key, as the page writes it (to the bibliography)
          const slug = key && key.toLowerCase().replace(/[^a-z0-9]+/g, "-");
          const cite = slug && document.querySelector(`.shapiq-cite-${slug} a`);
          const chip = make(cite ? "a" : "span", { class: approx ? "talk-chip talk-chip-approx" : "talk-chip" }, element, name);
          // (in a new tab, so that a click during the talk does not leave the deck)
          if (cite) Object.assign(chip, { href: cite.href, target: "_blank", rel: "noopener" });
          chip.title = [approx ? "An approximation: absent features follow the trees' own splits." : "", cite ? cite.title : ""].filter(Boolean).join(" ");
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

  const FIGURES = { "same-shap": sameShap, effects, faithfulness, framework };

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
      if (api.show) api.show(Math.min(step, api.steps - 1));
    };
    // without a slide that says otherwise, all of it
    follow(slide && slide.dataset.talkStep != null ? Number(slide.dataset.talkStep) : 99);
    if (slide) slide.addEventListener("shapiq-talk-step", (event) => follow(event.detail.step));
  }
})();
