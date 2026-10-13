"use strict";

// Precomputed report-wide analysis; no fitting or filter-dependent statistics here.
(() => {
  const section = document.getElementById("ensemble");
  if (!section) return;
  const targetSelect = document.getElementById("ensembleTarget");
  const budgetSelect = document.getElementById("ensembleBudget");
  const methods = document.getElementById("ensembleMethods");
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
        panel.regularization?.kind === "ridge" &&
        Number.isFinite(panel.regularization.penalty) && panel.regularization.penalty > 0 &&
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
    const svg = svgNode("svg", { viewBox: "0 0 640 320", role: "img", "aria-label": "PCA of normalized estimator errors" });
    const bounds = (key) => {
      const values = pca.points.map((point) => point[key]);
      const min = Math.min(0, ...values), max = Math.max(0, ...values);
      const pad = (max - min || 1) * 0.12;
      return [min - pad, max + pad];
    };
    const [xmin, xmax] = bounds("x"), [ymin, ymax] = bounds("y");
    const x = (value) => 62 + 553 * (value - xmin) / (xmax - xmin);
    const y = (value) => 273 - 250 * (value - ymin) / (ymax - ymin);
    svg.append(svgNode("rect", { x: 62, y: 23, width: 553, height: 250, fill: "#f8faff", rx: 6 }));
    for (let i = 0; i <= 4; i++) {
      const xx = xmin + (xmax - xmin) * i / 4;
      const yy = ymin + (ymax - ymin) * i / 4;
      svg.append(
        svgNode("line", { x1: 62, x2: 615, y1: y(yy), y2: y(yy), stroke: "#dfe6f0", "stroke-dasharray": "3 4" }),
        svgNode("text", { x: 55, y: y(yy) + 4, "text-anchor": "end" }, Number(yy.toPrecision(2)).toString()),
        svgNode("text", { x: x(xx), y: 292, "text-anchor": "middle" }, Number(xx.toPrecision(2)).toString()),
      );
    }
    svg.append(
      svgNode("text", { x: 338, y: 317, "text-anchor": "middle" }, `PC1 · ${Math.round(100 * pca.explained_variance[0])}% variance`),
      svgNode("text", { x: 62, y: 12 }, `PC2 · ${Math.round(100 * pca.explained_variance[1])}% variance`),
    );
    const legend = document.createElement("div");
    legend.className = "legend";
    pca.points.forEach((point) => {
      const color = colorFor(point.method);
      const label = `${point.method}: PC1 ${number(point.x)}, PC2 ${number(point.y)}`;
      const dot = svgNode("circle", { cx: x(point.x), cy: y(point.y), r: 4, fill: color,
        stroke: "white", "stroke-width": 1, tabindex: 0, role: "img", "aria-label": label });
      dot.append(svgNode("title", {}, label));
      bindHighlight(dot, point.method, label);
      svg.append(dot);
      const button = document.createElement("button");
      button.type = "button";
      const swatch = document.createElement("span");
      swatch.className = "ensembleSwatch";
      swatch.style.backgroundColor = color;
      button.append(swatch, document.createTextNode(methodLabel(point.method)));
      bindHighlight(button, point.method, label);
      // Bring even coincident points to the front when their legend is selected.
      button.addEventListener("pointerenter", () => svg.append(dot));
      button.addEventListener("focus", () => svg.append(dot));
      legend.append(button);
    });
    plot.append(svg, legend);
  }

  function render() {
    section.hidden = !matches();
    if (section.hidden) return;
    const panel = artifact.panels.find((p) => p.target === targetSelect.value && p.relative_budget === Number(budgetSelect.value));
    if (!panel) { section.hidden = true; return; }
    weights.replaceChildren(); methods.replaceChildren();
    const ordered = panel.methods.map((method, i) => ({ method, alpha: panel.coefficients[i] }))
      .sort((a, b) => Math.abs(b.alpha) - Math.abs(a.alpha));
    const header = document.createElement("tr");
    ["Estimator", ...ordered.map(({ method }) => methodLabel(method))].forEach((name) => {
      const th = document.createElement("th");
      th.scope = "col";
      th.textContent = name;
      header.append(th);
    });
    methods.append(header);
    row(weights, ["Coefficient α", ...ordered.map(({ alpha }) => `${alpha > 0 ? "+" : ""}${number(alpha)}`)]);
    scores.textContent = `Held-out normalized R²: ${number(panel.held_out.r2)} · nMSE: ${number(panel.held_out.nmse)} (equal weights: ${number(panel.equal_weights.r2)} R², ${number(panel.equal_weights.nmse)} nMSE).`;
    note.textContent = `${panel.n_panels.toLocaleString()} matched panels · ${panel.n_datasets} datasets · ${panel.folds.length} dataset-grouped folds · ridge λ = ${number(panel.regularization.penalty)}.`;
    scatter(panel.pca);
  }

  function budgets() {
    const previous = budgetSelect.value;
    budgetSelect.replaceChildren();
    artifact.panels.filter((p) => p.target === targetSelect.value)
      .sort((a, b) => a.relative_budget - b.relative_budget)
      .forEach((panel) => budgetSelect.add(new Option(`${panel.relative_budget} × players`, String(panel.relative_budget))));
    budgetSelect.value = [...budgetSelect.options].some((option) => option.value === previous)
      ? previous : budgetSelect.options[budgetSelect.options.length - 1].value;
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
    [...new Set(artifact.panels.map((panel) => panel.target))].forEach((target) => targetSelect.add(new Option(targetLabel(target), target)));
    if ([...targetSelect.options].some((option) => option.value === "SV · order 1")) targetSelect.value = "SV · order 1";
    budgets();
  }).catch(() => { section.hidden = true; });
})();
