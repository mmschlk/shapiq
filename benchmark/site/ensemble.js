"use strict";

// Precomputed report-wide analysis; no fitting or filter-dependent statistics here.
(() => {
  const section = document.getElementById("ensemble");
  if (!section) return;
  const targetSelect = document.getElementById("ensembleTarget");
  const budgetSelect = document.getElementById("ensembleBudget");
  const weights = document.getElementById("ensembleWeights");
  const scores = document.getElementById("ensembleScores");
  const note = document.getElementById("ensembleNote");
  const plot = document.getElementById("ensemblePca");
  let artifact = null;
  const reportId = () => (typeof data === "undefined" ? null : data?.snapshot_id);
  const matches = () => artifact && artifact.report_id === reportId();
  const number = (value) => Number.isFinite(value) ? Number(value.toPrecision(4)).toString() : "—";
  const metric = (value) => value === null || Number.isFinite(value);
  const metrics = (value) => value && metric(value.r2) && metric(value.nmse);

  function validate(value) {
    if (value?.version !== 1 || typeof value.report_id !== "string" || !value.report_id ||
        !Array.isArray(value.panels) || !value.panels.length || value.panels.length > 1000) return false;
    const keys = new Set();
    return value.panels.every((panel) => {
      const key = JSON.stringify([panel.target, panel.relative_budget]);
      if (keys.has(key)) return false;
      keys.add(key);
      return typeof panel.target === "string" && panel.target.length > 0 &&
        Number.isFinite(panel.relative_budget) && panel.relative_budget > 0 &&
        Array.isArray(panel.methods) && panel.methods.length > 0 && panel.methods.length <= 100 &&
        panel.methods.every((method) => typeof method === "string") &&
        new Set(panel.methods).size === panel.methods.length &&
        Array.isArray(panel.coefficients) && panel.coefficients.length === panel.methods.length &&
        panel.coefficients.every(Number.isFinite) && metrics(panel.train) &&
        metrics(panel.held_out) && metrics(panel.equal_weights) &&
        Number.isInteger(panel.n_panels) && panel.n_panels > 0 &&
        Number.isInteger(panel.n_datasets) && panel.n_datasets > 1 &&
        Array.isArray(panel.folds) && panel.folds.length > 1 &&
        panel.folds.every((fold) => Array.isArray(fold.held_out_datasets) &&
          fold.held_out_datasets.length && fold.held_out_datasets.every((name) => typeof name === "string")) &&
        Array.isArray(panel.pca?.points) && panel.pca.points.length === panel.methods.length &&
        new Set(panel.pca.points.map((point) => point.method)).size === panel.methods.length &&
        panel.pca.points.every((point) => panel.methods.includes(point.method) &&
          Number.isFinite(point.x) && Number.isFinite(point.y)) &&
        Array.isArray(panel.pca.explained_variance) && panel.pca.explained_variance.length === 2 &&
        panel.pca.explained_variance.every((v) => Number.isFinite(v) && v >= 0 && v <= 1);
    });
  }

  function row(body, values) {
    const tr = document.createElement("tr");
    values.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value;
      tr.append(td);
    });
    body.append(tr);
  }

  function svgNode(name, attributes, text) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function scatter(pca) {
    plot.replaceChildren();
    const svg = svgNode("svg", { viewBox: "0 0 540 310", role: "img", "aria-label": "PCA of normalized estimator errors" });
    const bounds = (key) => {
      const values = pca.points.map((point) => point[key]);
      const min = Math.min(0, ...values), max = Math.max(0, ...values);
      const pad = (max - min || 1) * 0.12;
      return [min - pad, max + pad];
    };
    const [xmin, xmax] = bounds("x"), [ymin, ymax] = bounds("y");
    const x = (value) => 50 + 425 * (value - xmin) / (xmax - xmin);
    const y = (value) => 255 - 225 * (value - ymin) / (ymax - ymin);
    svg.append(svgNode("line", { x1: 50, x2: 475, y1: y(0), y2: y(0), class: "ensembleAxis" }),
      svgNode("line", { x1: x(0), x2: x(0), y1: 30, y2: 255, class: "ensembleAxis" }),
      svgNode("text", { x: 270, y: 292, "text-anchor": "middle" }, `PC1 · ${number(100 * pca.explained_variance[0])}% variance`),
      svgNode("text", { x: 16, y: 150, transform: "rotate(-90 16 150)", "text-anchor": "middle" }, `PC2 · ${number(100 * pca.explained_variance[1])}% variance`));
    pca.points.forEach((point) => {
      const dot = svgNode("circle", { cx: x(point.x), cy: y(point.y), r: 4, tabindex: 0 });
      dot.append(svgNode("title", {}, `${point.method}: ${number(point.x)}, ${number(point.y)}`));
      svg.append(dot, svgNode("text", { x: x(point.x) + 7, y: y(point.y) - 7, class: "ensemblePointLabel" }, point.method));
    });
    plot.append(svg);
  }

  function render() {
    section.hidden = !matches();
    if (section.hidden) return;
    const panel = artifact.panels.find((p) => p.target === targetSelect.value && p.relative_budget === Number(budgetSelect.value));
    if (!panel) { section.hidden = true; return; }
    weights.replaceChildren(); scores.replaceChildren();
    panel.methods.map((method, i) => ({ method, alpha: panel.coefficients[i] }))
      .sort((a, b) => Math.abs(b.alpha) - Math.abs(a.alpha))
      .forEach(({ method, alpha }) => row(weights, [method, `${alpha > 0 ? "+" : ""}${number(alpha)}`]));
    row(scores, ["Fitted combination · held out", number(panel.held_out.r2), number(panel.held_out.nmse)]);
    row(scores, ["Equal weights · held out", number(panel.equal_weights.r2), number(panel.equal_weights.nmse)]);
    row(scores, ["Fitted combination · training", number(panel.train.r2), number(panel.train.nmse)]);
    note.textContent = `${panel.n_panels.toLocaleString()} matched game panels · ${panel.n_datasets} datasets · ${panel.folds.length} dataset-grouped folds. Higher R² and lower nMSE are better; — means undefined.`;
    scatter(panel.pca);
  }

  function budgets() {
    const previous = budgetSelect.value;
    budgetSelect.replaceChildren();
    artifact.panels.filter((p) => p.target === targetSelect.value)
      .sort((a, b) => a.relative_budget - b.relative_budget)
      .forEach((panel) => budgetSelect.add(new Option(`${panel.relative_budget} × players`, String(panel.relative_budget))));
    if ([...budgetSelect.options].some((option) => option.value === previous)) budgetSelect.value = previous;
    render();
  }

  function sync() {
    section.hidden = !matches();
    if (!section.hidden) render();
  }
  targetSelect.addEventListener("change", budgets);
  budgetSelect.addEventListener("change", render);
  // app.js updates this provenance element whenever it installs a new report.
  new MutationObserver(sync).observe(document.getElementById("snapshot"), { childList: true, subtree: true, characterData: true });
  fetch("ensemble.json").then(async (response) => {
    if (!response.ok) return;
    const text = await response.text();
    if (text.length > 2 * 1024 * 1024) return;
    const value = JSON.parse(text);
    if (!validate(value)) return;
    artifact = value;
    [...new Set(artifact.panels.map((panel) => panel.target))].forEach((target) => targetSelect.add(new Option(target, target)));
    budgets();
  }).catch(() => { section.hidden = true; });
})();
