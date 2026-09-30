"use strict";
const $ = (id) => document.getElementById(id);
const palette = [
  "#426BE0",
  "#BF2355",
  "#007C59",
  "#A34B87",
  "#947000",
  "#287FA8",
];
let data,
  selectedRows = [],
  chartNames = [],
  methodTargets = new Map(),
  unsupportedTargets = new Map(),
  tableSort = { key: "default", descending: false },
  chartFamilyExplicit = false;
const methodIndex = (method) =>
  Math.max(
    0,
    Object.keys(data?.methods || {})
      .sort()
      .indexOf(method),
  );
const colorFor = (method) => palette[methodIndex(method) % palette.length];
const dashFor = (method) =>
  ["", "7 3", "2 3", "9 3 2 3"][
    Math.floor(methodIndex(method) / palette.length) % 4
  ];
const isSynthetic = (game) => Boolean(game.metadata?.synthetic);
const replicationLabel = () =>
  data.suite.game_seeds?.length
    ? `${data.suite.game_seeds.length} game instances per setting · ${data.suite.seeds.length} estimator run${data.suite.seeds.length === 1 ? "" : "s"} per instance and budget`
    : `${data.suite.seeds.length} estimator seeds per game`;
const familyNames = {
  local_explanation: "Model explanations",
  data_valuation: "Training data valuation",
  ensemble_selection: "Ensemble selection",
  feature_selection: "Feature selection",
  global_fidelity: "Global model fidelity",
  image_explanation: "Image explanations",
  text_explanation: "Text explanations",
  uncertainty: "Predictive uncertainty",
  cluster: "Clustering",
  unsupervised: "Unsupervised learning",
  synthetic: "Synthetic diagnostics",
};
const familyLabel = (family) =>
  Object.hasOwn(familyNames, family)
    ? familyNames[family]
    : family.replaceAll("_", " ");
const targetLabel = (value) =>
  ({
    SV: "Shapley Values",
    "k-SII": "Shapley Interaction Indices (k-SII)",
    SII: "Shapley Interaction Indices (SII)",
    STII: "Shapley–Taylor Interactions (STII)",
    FSII: "Faithful Shapley Interactions (FSII)",
    FBII: "Faithful Banzhaf Interactions (FBII)",
  })[value.split(" · ")[0]] || value;
function numericOrder(a, b, descending = false) {
  const missing = Number(!Number.isFinite(a)) - Number(!Number.isFinite(b));
  return (
    missing ||
    (Number.isFinite(a) && Number.isFinite(b)
      ? descending
        ? b - a
        : a - b
      : 0)
  );
}
function rankingOrder(a, b) {
  const { key, descending } = tableSort;
  const value = (row) =>
    key === "method"
      ? methodLabel(row.method)
      : key === "capability"
        ? capability(row.method)
        : key === "coverage"
          ? row.planned
            ? row.valid / row.planned
            : null
          : row[key];
  const order =
    key === "default"
      ? Number(b.complete) - Number(a.complete) ||
        numericOrder(a.median, b.median)
      : ["method", "capability"].includes(key)
        ? value(a).localeCompare(value(b)) * (descending ? -1 : 1)
        : numericOrder(value(a), value(b), descending);
  return order || a.method.localeCompare(b.method);
}

const format = (n) => (Number.isFinite(n) ? n.toPrecision(4) : "—");
const target = (game) => `${game.index} · order ${game.order}`;
const methodNames = {
  PermutationSamplingSV: "Permutation · values",
  PermutationSamplingSII: "Permutation · SII",
  PermutationSamplingSTII: "Permutation · STII",
  RegressionFBII: "Regression · FBII",
  RegressionFSII: "Regression · FSII",
};
const methodLabel = (method) =>
  Object.hasOwn(methodNames, method) ? methodNames[method] : method;
function showMethod(method) {
  if ($("showVariants").value === "all") return true;
  const targets = methodTargets.get(method);
  return (
    method !== "InconsistentKernelSHAPIQ" &&
    (targets?.has($("target").value) ||
      !unsupportedTargets.get(method)?.has($("target").value))
  );
}
function capability(method) {
  const targets = [...(methodTargets.get(method) || [])];
  const values = targets.some(
    (name) => name.startsWith("SV ·") || name.startsWith("BV ·"),
  );
  const interactions = targets.some(
    (name) => !name.startsWith("SV ·") && !name.startsWith("BV ·"),
  );
  return values && interactions
    ? "Values + Interactions"
    : values
      ? "Values"
      : interactions
        ? "Interactions"
        : "Unverified";
}
function methodDetails(method) {
  const details = Object.hasOwn(window.METHOD_DETAILS || {}, method)
    ? window.METHOD_DETAILS[method]
    : null;
  const content = document.createElement("div"),
    description = document.createElement("p"),
    links = document.createElement("div");
  content.className = "methodDetails";
  description.textContent =
    details?.description ||
    "Local estimator. See the implementation supplied with this report.";
  links.className = "methodSources";
  [details?.paper, details?.implementation]
    .filter(Boolean)
    .forEach((source) => {
      if (!source.url?.startsWith("https://")) return;
      const link = document.createElement("a");
      link.href = source.url;
      link.textContent = source.title;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      links.append(link);
    });
  content.append(description, links);
  return content;
}
let nextMethodDetailsId = 0;
function bindMethodDetails(button, panel) {
  panel.id = `method-details-${++nextMethodDetailsId}`;
  panel.hidden = true;
  button.classList.add("detailToggle");
  button.setAttribute("aria-expanded", "false");
  button.setAttribute("aria-controls", panel.id);
  button.addEventListener("click", () => {
    panel.hidden = !panel.hidden;
    button.setAttribute("aria-expanded", String(!panel.hidden));
    $("chartTooltip").hidden = true;
  });
}
function options(id, values, all) {
  $(id).replaceChildren();
  if (all) $(id).add(new Option(all, ""));
  values.forEach((value) => $(id).add(new Option(value, value)));
}
function load(value) {
  if (
    value.schema_version !== 1 ||
    !value.runs ||
    !value.games?.length ||
    !value.suite?.budgets?.length ||
    !value.suite?.seeds?.length ||
    !value.records ||
    !value.methods
  )
    throw Error(
      "Select an exported benchmark data.json (generated by shapiq_benchmark.report).",
    );
  data = {
    ...value,
    records: value.records.map((record) => {
      if (record.worker || !record.worker_id) return record;
      if (!Object.hasOwn(value.workers || {}, record.worker_id))
        throw Error("The report is missing referenced worker metadata.");
      return { ...record, worker: value.workers[record.worker_id] };
    }),
  };
  chartFamilyExplicit = false;
  const gamesById = new Map(data.games.map((game) => [game.id, game]));
  methodTargets = new Map(
    Object.keys(data.methods).map((method) => [method, new Set()]),
  );
  unsupportedTargets = new Map(
    Object.keys(data.methods).map((method) => [method, new Set()]),
  );
  data.records.forEach((record) => {
    const game = gamesById.get(record.game_id);
    if (game) {
      const registry =
        record.status === "unsupported" ? unsupportedTargets : methodTargets;
      registry.get(record.method)?.add(target(game));
    }
  });
  options("target", [...new Set(data.games.map(target))]);
  [...$("target").options].forEach(
    (option) => (option.textContent = targetLabel(option.value)),
  );
  options(
    "family",
    [...new Set(data.games.map((g) => g.family))],
    "All families",
  );
  [...$("family").options].forEach((option) => {
    if (option.value) option.textContent = familyLabel(option.value);
  });
  $("budget").replaceChildren(new Option("All relative budgets", ""));
  const relative = document.createElement("optgroup");
  relative.label = "Queries per player · B/d";
  const measuredRatios =
    data.suite.relative_budgets ||
    [
      ...new Set(
        data.games.flatMap((g) => gameBudgets(g).map((b) => b / g.n_players)),
      ),
    ].sort((a, b) => a - b);
  const ratios = [
    ...new Set([
      ...measuredRatios,
      ...(data.suite.name?.startsWith("all-families")
        ? [0.5, 1, 2, 4, 8, 16, 32, 64, 128]
        : []),
    ]),
  ].sort((a, b) => a - b);
  ratios.forEach((r) =>
    relative.append(
      new Option(
        `${Number(r.toPrecision(4))} × players${data.records.some((row) => row.status !== "unsupported" && row.budget === Math.ceil(r * gamesById.get(row.game_id)?.n_players)) ? "" : " · pending"}`,
        `r:${r}`,
      ),
    ),
  );
  $("budget").append(relative);
  options("methods", Object.keys(data.methods));
  [...$("methods").options].forEach((option) => (option.selected = true));
  $("snapshot").textContent =
    `Snapshot ${data.snapshot_id.slice(0, 12)} · ${replicationLabel()}. Full provenance in the download.`;
  $("notice").textContent = "";
  $("methodCount").textContent = Object.keys(data.methods).length;
  $("gameCount").textContent = new Set(
    data.games
      .filter((g) => !isSynthetic(g))
      .map((g) => g.metadata?.case_id || g.id),
  ).size;
  $("runCount").textContent = new Intl.NumberFormat().format(
    data.records.filter((r) => r.status !== "unsupported").length,
  );
  $("methodSearch").value = "";
  buildMethodPicker();
  renderCoverage();
  render();
}
function buildMethodPicker() {
  $("methodOptions").replaceChildren();
  [...$("methods").options].forEach((option) => {
    const label = document.createElement("label"),
      input = document.createElement("input");
    label.className = "methodOption";
    label.hidden =
      !showMethod(option.value) ||
      !`${option.value} ${methodLabel(option.value)}`
        .toLowerCase()
        .includes($("methodSearch").value.toLowerCase());
    input.type = "checkbox";
    input.checked = option.selected;
    input.value = option.value;
    input.addEventListener("change", () => {
      option.selected = input.checked;
      render();
    });
    label.append(input, document.createTextNode(methodLabel(option.value)));
    label.title = option.value;
    $("methodOptions").append(label);
  });
}
function renderCoverage() {
  $("libraryCoverage").replaceChildren();
  const coverage = data.coverage || [];
  if (!coverage.length) {
    const p = document.createElement("p");
    p.textContent = `${data.games.length} measured game definitions. See the methodology for scope.`;
    $("libraryCoverage").append(p);
    return;
  }
  const families = [...new Set(data.games.map((g) => g.family))];
  families.forEach((family) => {
    const cases = new Set(
      data.games
        .filter((g) => g.family === family)
        .map((g) => g.metadata?.case_id || g.id),
    );
    const row = document.createElement("div");
    row.className = "coverageItem";
    const label = document.createElement("strong"),
      count = document.createElement("span");
    const instances = new Set(
      data.games
        .filter((game) => game.family === family)
        .map((game) =>
          JSON.stringify([
            game.metadata?.case_id || game.id,
            game.metadata?.instance_seed ?? 0,
          ]),
        ),
    );
    label.textContent = familyLabel(family);
    count.textContent = `${cases.size} setups · ${instances.size} instances`;
    row.append(label, count);
    $("libraryCoverage").append(row);
  });
}
const gameBudgets = (game) =>
  data.suite.budgets_by_game?.[game.id] || data.suite.budgets;
function selection(family = $("family").value, allBudgets = false) {
  const games = data.games.filter(
    (g) =>
      isSynthetic(g) === ($("panel").value === "diagnostic") &&
      target(g) === $("target").value &&
      (!family || g.family === family) &&
      g.n_players >= Number($("minPlayers").value) &&
      g.n_players <= Number($("maxPlayers").value),
  );
  const methods = [...$("methods").selectedOptions].map((o) => o.value),
    game_budgets = {};
  const cells = games.flatMap((g) => {
    const chosen = allBudgets ? "" : $("budget").value;
    let budgets = chosen.startsWith("r:")
      ? [Math.ceil(Number(chosen.slice(2)) * g.n_players)]
      : gameBudgets(g);
    const cap =
      allBudgets || $("cap").value === ""
        ? null
        : Number($("cap").value) * g.n_players;
    if (cap !== null)
      budgets = [Math.max(-1, ...budgets.filter((b) => b <= cap))];
    game_budgets[g.id] = budgets;
    return budgets.flatMap((budget) =>
      data.suite.seeds.map((seed) => ({ game_id: g.id, budget, seed })),
    );
  });
  const keys = new Set(cells.map(cellKey));
  const rows = data.records.filter(
    (r) => methods.includes(r.method) && keys.has(cellKey(r)),
  );
  const zero = new Set([
    ...data.records.filter((r) => r.zero_truth_energy).map((r) => r.game_id),
    ...data.games.filter((g) => g.metadata?.zero_truth_energy).map((g) => g.id),
  ]);
  return {
    panel_ids: games.map((g) => g.id),
    games: games.filter((g) => !zero.has(g.id)),
    methods,
    game_budgets,
    budgets: [...new Set(Object.values(game_budgets).flat())].sort(
      (a, b) => a - b,
    ),
    rows: rows.filter((r) => !zero.has(r.game_id)),
    cells: cells.filter((r) => !zero.has(r.game_id)),
    excluded: games.filter((g) => zero.has(g.id)).length,
  };
}
const cellKey = (r) => JSON.stringify([r.game_id, r.budget, r.seed]);
function weightedCells(s) {
  const families = [...new Set(s.games.map((g) => g.family))];
  const weights = new Map();
  s.games.forEach((g) => {
    const family = s.games.filter((x) => x.family === g.family),
      strata = [...new Set(family.map((x) => x.stratum))];
    const count = family.filter((x) => x.stratum === g.stratum).length,
      cells = s.cells.filter((c) => c.game_id === g.id);
    cells.forEach((c) =>
      weights.set(
        cellKey(c),
        1 / families.length / strata.length / count / cells.length,
      ),
    );
  });
  return weights;
}
function summary(rows, s) {
  const good = rows.filter((r) => r.status === "ok" && Number.isFinite(r.nmse)),
    weights = weightedCells(s),
    successfulWeight = good.reduce(
      (sum, r) => sum + weights.get(cellKey(r)),
      0,
    );
  let cumulative = 0,
    median = null;
  const ordered = [...good]
    .filter((r) => weights.get(cellKey(r)) > 0)
    .sort((a, b) => a.nmse - b.nmse);
  for (let i = 0; i < ordered.length; i++) {
    cumulative += weights.get(cellKey(ordered[i]));
    const fraction = cumulative / successfulWeight;
    if (fraction + 1e-14 >= 0.5) {
      median =
        Math.abs(fraction - 0.5) <= 1e-14 && i + 1 < ordered.length
          ? ordered[i].nmse / 2 + ordered[i + 1].nmse / 2
          : ordered[i].nmse;
      break;
    }
  }
  return {
    average:
      successfulWeight > 0
        ? good.reduce((sum, r) => sum + r.nmse * weights.get(cellKey(r)), 0) /
          successfulWeight
        : null,
    median,
    valid: good.length,
    planned: s.cells.length,
    failed: rows.filter((r) => r.status === "failed").length,
    unsupported: rows.filter((r) => r.status === "unsupported").length,
    missing: s.cells.length - rows.length,
    complete: s.cells.length > 0 && good.length === s.cells.length,
  };
}
const same = (a, b) =>
  JSON.stringify([...a].sort()) === JSON.stringify([...b].sort());
function render() {
  const s = selection();
  const visibleMethods = s.methods.filter(showMethod);
  buildMethodPicker();
  selectedRows = s.rows;
  $("chartTooltip").hidden = true;
  const preset = (data.presets || []).find(
    (p) =>
      same(p.game_ids, s.panel_ids) &&
      same(p.methods, s.methods) &&
      (!p.panel || p.panel === $("panel").value) &&
      s.panel_ids.every((id) =>
        same(p.game_budgets?.[id] || p.budgets, s.game_budgets[id]),
      ),
  );
  const allChartGames = selection("", true);
  const chartFamilies = [
    ...new Set(allChartGames.games.map((game) => game.family)),
  ];
  const previousFamily = chartFamilyExplicit
    ? $("chartFamily").value
    : $("family").value;
  options("chartFamily", chartFamilies, "All families");
  [...$("chartFamily").options].forEach((option) => {
    if (option.value) option.textContent = familyLabel(option.value);
  });
  $("chartFamily").value = chartFamilies.includes(previousFamily)
    ? previousFamily
    : "";
  const chartPanel = selection($("chartFamily").value, true);
  const allCurves = $("chartLimit").value === "all";
  chartNames = visibleMethods
    .map((method) => ({
      method,
      ...summary(
        chartPanel.rows.filter((r) => r.method === method),
        chartPanel,
      ),
    }))
    .filter((item) => Number.isFinite(item.median))
    .sort((a, b) =>
      allCurves
        ? a.method.localeCompare(b.method)
        : Number(b.complete) - Number(a.complete) ||
          a.median - b.median ||
          a.method.localeCompare(b.method),
    )
    .slice(0, allCurves ? Infinity : 5)
    .map((item) => item.method);
  const chartPending =
    chartPanel.cells.length > 0 &&
    !chartPanel.rows.some((r) => r.status !== "unsupported") &&
    chartPanel.rows.length <
      chartPanel.cells.length * chartPanel.methods.length;
  const tablePending =
    s.cells.length > 0 &&
    !s.rows.some((row) => row.status !== "unsupported") &&
    s.rows.length < s.cells.length * s.methods.length;
  $("notice").textContent = tablePending
    ? "Table results pending for this budget. Charts show all measured budgets."
    : chartPending
      ? "Chart results pending for this selection."
      : "";
  $("budgetChartNote").textContent = allCurves
    ? "Family-balanced median · Coverage on hover"
    : "Family-balanced median · Complete coverage preferred";
  const gameUnit = data.suite.game_seeds?.length ? "game instances" : "games";
  $("chartPanelMeta").textContent =
    `${chartPanel.games.length} ${gameUnit} · All measured budgets`;
  $("gameDetails").textContent =
    `${replicationLabel()}. Frozen payoffs and exact ground truth are included in the reproduction bundle.`;
  $("panelSummary").textContent =
    `${s.games.length} ${gameUnit} · ${s.cells.length} cells / estimator${s.excluded ? ` · ${s.excluded} zero-energy excluded` : ""}`;
  $("methodLabel").textContent = `${visibleMethods.length} shown`;
  renderLeaderboard(s, preset, visibleMethods);
  renderRunIssues(s);
  renderPerformanceCharts(chartPanel, chartPending);
  renderHistory(s, preset);
  renderHardware(s);
}

function renderPerformanceCharts(chartPanel, chartPending) {
  const gamesById = new Map(chartPanel.games.map((g) => [g.id, g]));
  const chartRatios =
    data.suite.relative_budgets ||
    [
      ...new Set(
        chartPanel.cells.map(
          (cell) => cell.budget / gamesById.get(cell.game_id).n_players,
        ),
      ),
    ].sort((a, b) => a - b);
  const panelAt = (ratio, games = chartPanel.games) => {
    const cells = games.flatMap((g) =>
      data.suite.seeds.map((seed) => ({
        game_id: g.id,
        budget: Math.ceil(ratio * g.n_players),
        seed,
      })),
    );
    return { ...chartPanel, games, cells };
  };
  const rowsAt = (rows, panel) => {
    const keys = new Set(panel.cells.map(cellKey));
    return rows.filter((r) => keys.has(cellKey(r)));
  };
  const point = (rows, panel, ratio) => {
    const stats = summary(rows, panel);
    if (!Number.isFinite(stats.median)) return null;
    const good = rows.filter(
      (r) => r.status === "ok" && Number.isFinite(r.nmse),
    );
    const used = good.every((r) => Number.isFinite(r.queries))
      ? good.map((r) => r.queries / gamesById.get(r.game_id).n_players)
      : [];
    const low = Math.min(...used),
      high = Math.max(...used);
    return {
      x: ratio,
      y: stats.median,
      relativeBudget: ratio,
      coverage: `${stats.valid}/${stats.planned} successful runs · ${panel.games.length} games`,
      queryUsage: used.length
        ? `${format(low)}${high > low ? `–${format(high)}` : ""} × players`
        : null,
    };
  };
  const budgetSeries = chartNames.map((method) => ({
    name: methodLabel(method),
    method,
    color: colorFor(method),
    points: chartRatios
      .map((ratio) => {
        const panel = panelAt(ratio);
        return point(
          rowsAt(
            chartPanel.rows.filter((r) => r.method === method),
            panel,
          ),
          panel,
          ratio,
        );
      })
      .filter(Boolean),
  }));
  chart(
    "budgetChart",
    budgetSeries,
    "Query budget per player · B / d",
    false,
    "median",
  );
  const profiles = new Map();
  chartPanel.rows.forEach((row) => {
    const worker = row.worker;
    const verified =
      worker?.cpu_model && worker?.thread_pools?.length && row.timing_profile;
    const key = JSON.stringify([
      row.timing_profile,
      worker?.cpu_model,
      worker?.machine,
      worker?.thread_pools,
      worker?.thread_environment,
      ...(verified ? [] : [row.run_id, row.game_id]),
    ]);
    if (!profiles.has(key))
      profiles.set(key, {
        rows: [],
        games: new Set(),
        label: `${worker?.cpu_model || "Unverified hardware"} · ${row.timing_profile || "diagnostic"}`,
      });
    const profile = profiles.get(key);
    profile.rows.push(row);
    profile.games.add(row.game_id);
  });
  const timeSeries = [];
  chartNames.forEach((method) =>
    profiles.forEach((profile) => {
      const games = chartPanel.games.filter((g) => profile.games.has(g.id));
      const points = chartRatios
        .map((ratio) => {
          const panel = panelAt(ratio, games);
          const rows = rowsAt(
            profile.rows.filter(
              (r) => r.method === method && Number.isFinite(r.seconds),
            ),
            panel,
          );
          const result = point(rows, panel, ratio);
          if (result)
            result.x = summary(
              rows.map((r) => ({
                ...r,
                nmse: Number.isFinite(r.nmse) ? r.seconds : null,
              })),
              panel,
            ).average;
          return result;
        })
        .filter(Boolean)
        .sort((a, b) => a.x - b.x);
      if (points.length)
        timeSeries.push({
          name: methodLabel(method),
          method,
          color: colorFor(method),
          profile: profile.label,
          points,
        });
    }),
  );
  chart("timeChart", timeSeries, "Mean seconds · diagnostic", false, "median");
  if (chartPending)
    ["budgetChart", "timeChart"].forEach(
      (id) => ($(id).textContent = "Results pending for this selection."),
    );
}

function renderLeaderboard(s, preset, visibleMethods) {
  document.querySelectorAll("[data-sort]").forEach((button) => {
    const active = tableSort.key === button.dataset.sort;
    button
      .closest("th")
      .setAttribute(
        "aria-sort",
        active
          ? tableSort.key === "default"
            ? "other"
            : tableSort.descending
              ? "descending"
              : "ascending"
          : "none",
      );
    button.querySelector(".sortArrow").textContent = active
      ? tableSort.key === "default"
        ? "↺"
        : tableSort.descending
          ? "↓"
          : "↑"
      : "↕";
  });
  const sortByElo = tableSort.key === "elo";
  const summaries = visibleMethods
    .map((method) => ({
      method,
      ...summary(
        s.rows.filter((r) => r.method === method),
        s,
      ),
      elo: preset?.rows.find((r) => r.method === method)?.elo,
    }))
    .sort(rankingOrder);
  $("ranking").replaceChildren();
  let rank = 0;
  summaries.forEach((item) => {
    const stats = preset?.rows.find((r) => r.method === item.method),
      tr = document.createElement("tr"),
      detailRow = document.createElement("tr"),
      detailCell = document.createElement("td");
    detailRow.className = "methodDetailRow";
    detailCell.colSpan = 7;
    detailCell.append(methodDetails(item.method));
    detailRow.append(detailCell);
    const state = item.complete
      ? "Complete"
      : [
          item.failed ? `${item.failed} failed` : "",
          item.unsupported ? `${item.unsupported} unsupported` : "",
          item.missing ? `${item.missing} missing` : "",
          item.valid < item.planned &&
          !item.failed &&
          !item.unsupported &&
          !item.missing
            ? "Undefined nMSE"
            : "",
        ]
          .filter(Boolean)
          .join(" · ");
    [
      (sortByElo ? Number.isFinite(item.elo) : Number.isFinite(item.average))
        ? ++rank
        : "—",
      methodLabel(item.method),
      capability(item.method),
      format(item.average),
      format(item.median),
      format(stats?.elo),
      `${item.valid} / ${item.planned}`,
    ].forEach((value, i) => {
      const td = document.createElement("td");
      if (i === 1) {
        const dot = document.createElement("span");
        dot.className = "methodDot";
        dot.style.background = chartNames.includes(item.method)
          ? colorFor(item.method)
          : "#c9c0d4";
        const button = document.createElement("button");
        button.type = "button";
        button.className = "methodLink";
        button.textContent = value;
        button.title = `About ${item.method}`;
        bindMethodDetails(button, detailRow);
        bindHighlight(button, item.method, `About ${item.method}`);
        td.append(dot, button);
      } else if (i === 6) {
        const pill = document.createElement("span");
        pill.className = "coveragePill" + (item.complete ? " complete" : "");
        td.title = state || "No games";
        pill.textContent = value;
        td.append(pill);
      } else {
        td.textContent = value;
        if (i === 4 && Number.isFinite(item.median))
          td.className = "scoreStrong";
        if (i === 3 || i === 4)
          td.title =
            "Available successful runs; weights renormalized over observed results. Coverage differs between estimators.";
      }
      tr.append(td);
    });
    tr.tabIndex = 0;
    bindHighlight(
      tr,
      item.method,
      `${item.method} · ${Number.isFinite(item.average) ? `median nMSE ${format(item.median)} over available successful runs` : "No successful runs"} · ${item.valid}/${item.planned} coverage${state ? ` · ${state}` : ""}${chartNames.includes(item.method) ? "" : " · not shown in the current chart"}`,
    );
    $("ranking").append(tr, detailRow);
  });
  if (!summaries.length)
    appendRow("ranking", ["—", "Select an estimator", "—", "—", "—", "—", "—"]);
  $("statisticsNote").textContent = preset
    ? "Elo. Paired comparisons of shared successful runs. Ratings depend on the selected panel and competitor set; hiding variants does not change them."
    : "Elo. Available for preset panels with the full competitor set. Custom filters still show error and coverage.";
}

function renderRunIssues(s) {
  $("failures").replaceChildren();
  const failures = new Map();
  s.rows
    .filter((r) => r.status !== "ok")
    .forEach((r) => {
      const explanation =
        r.status === "unsupported"
          ? "not supported for this explanation"
          : {
              TimeoutError: "time limit reached",
              MemoryError: "memory allocation failed",
              ValueError:
                "input, configuration, or numerical validation failed",
              ModuleNotFoundError: "optional dependency unavailable",
              ImportError: "dependency could not load",
              LinAlgError: "linear algebra failed",
              BudgetExceededError: "query budget exceeded",
            }[r.error_type] || "estimator error";
      const key = `${methodLabel(r.method)}: ${explanation}`;
      failures.set(key, (failures.get(key) || 0) + 1);
    });
  for (const [description, count] of [...failures]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)) {
    const item = document.createElement("li");
    item.textContent = `${description} — ${count} run(s)`;
    $("failures").append(item);
  }
}

function renderHistory(s, preset) {
  const metric = $("historyMetric").value,
    history = preset?.history,
    end = new Date().getUTCFullYear() + new Date().getUTCMonth() / 12;
  const year = (date) => {
    const d = new Date(date + "T00:00:00Z");
    return (
      d.getUTCFullYear() +
      (d - new Date(Date.UTC(d.getUTCFullYear(), 0, 1))) / 31557600000
    );
  };
  const historyMethods = (history?.methods || [])
    .map((method) => ({
      ...method,
      ...summary(
        s.rows.filter((row) => row.method === method.method),
        s,
      ),
    }))
    .filter((method) => method.complete);
  const historySeries = historyMethods
    .filter((method) => showMethod(method.method))
    .map((method) => ({
      name: methodLabel(method.method),
      method: method.method,
      releaseYear: method.date.slice(0, 4),
      color: colorFor(method.method),
      points: [
        {
          x: year(method.date),
          y: metric === "median" ? method.median : method.average,
        },
        { x: end, y: metric === "median" ? method.median : method.average },
      ],
    }));
  const frontier = [],
    steps = [];
  historyMethods
    .map((method) => ({
      date: method.date,
      value: metric === "median" ? method.median : method.average,
    }))
    .sort((a, b) => a.date.localeCompare(b.date) || a.value - b.value)
    .forEach((entry) => {
      if (!frontier.length || entry.value < frontier.at(-1).value)
        frontier.push(entry);
    });
  frontier.forEach((p, i) => {
    if (i) steps.push({ x: year(p.date), y: frontier[i - 1].value });
    steps.push({ x: year(p.date), y: p.value });
  });
  if (steps.length) {
    steps.push({ x: end, y: steps.at(-1).y });
    historySeries.push({
      name: "Best dated result",
      color: "#28213e",
      points: steps,
    });
  }
  chart("historyChart", historySeries, "Publication year", true, metric);
  $("historySources").replaceChildren();
  (history?.methods || []).forEach((m) => {
    const a = document.createElement("a");
    if (!/^https:\/\/arxiv\.org\//.test(m.url)) return;
    a.href = m.url;
    a.textContent = `${m.method} (${m.date})`;
    a.rel = "noopener";
    $("historySources").append(a, document.createTextNode(" · "));
  });
  if (history?.unknown_dates?.length)
    $("historySources").append(
      document.createTextNode(
        `Dates unverified: ${history.unknown_dates.join(", ")}.`,
      ),
    );
}

function renderHardware(s) {
  const workers = s.rows.map((r) => r.worker).filter(Boolean);
  $("hardware").textContent = workers.length
    ? `Measured workers: ${[...new Set(workers.map((w) => w.cpu_model))].join("; ")}. Profiles: ${[...new Set(s.rows.map((r) => r.timing_profile).filter(Boolean))].join(", ")}. Per-run placement and thread details are included in JSON downloads.`
    : "Worker hardware was not recorded in this older result.";
}

function appendRow(id, values) {
  const tr = document.createElement("tr");
  values.forEach((value) => {
    const td = document.createElement("td");
    td.textContent = value;
    tr.append(td);
  });
  $(id).append(tr);
  return tr;
}
function highlight(method) {
  document.querySelectorAll("[data-method]").forEach((node) => {
    node.classList.toggle(
      "dimmed",
      Boolean(method) && node.dataset.method !== method,
    );
    node.classList.toggle(
      "emphasis",
      Boolean(method) && node.dataset.method === method,
    );
  });
}
function bindHighlight(node, method, message) {
  if (!method) return;
  node.dataset.method = method;
  node.setAttribute("aria-describedby", "chartTooltip");
  const show = (event) => {
    highlight(method);
    const tip = $("chartTooltip"),
      rect = node.getBoundingClientRect();
    tip.textContent = message;
    tip.hidden = false;
    const left = Number.isFinite(event.clientX)
        ? event.clientX
        : rect.left + rect.width / 2,
      top = Number.isFinite(event.clientY) ? event.clientY : rect.bottom;
    tip.style.left = `${Math.max(10, Math.min(left + 13, window.innerWidth - tip.offsetWidth - 10))}px`;
    tip.style.top = `${Math.max(10, Math.min(top + 13, window.innerHeight - tip.offsetHeight - 10))}px`;
  };
  const clear = () => {
    highlight(null);
    $("chartTooltip").hidden = true;
  };
  node.addEventListener("pointerenter", show);
  node.addEventListener("pointermove", show);
  node.addEventListener("pointerleave", clear);
  node.addEventListener("focus", show);
  node.addEventListener("blur", clear);
}
function balanceHistoryLegend(legend) {
  if (!legend?.children.length) return;
  const columns =
    window.innerWidth <= 720 ? 2 : window.innerWidth <= 1050 ? 3 : 4;
  const count = legend.children.length,
    rows = Math.ceil(count / columns);
  const perRow = Math.floor(count / rows),
    extra = count % rows;
  let position = 0;
  for (let row = 0; row < rows; row++) {
    const items = perRow + (row < extra ? 1 : 0);
    for (let item = 0; item < items; item++)
      legend.children[position++].style.gridColumn = `span ${12 / items}`;
  }
}
let historyChartState;
function chart(id, series, xlabel, dates = false, metric = "mean") {
  if (dates) historyChartState = { series, xlabel, metric };
  const box = $(id);
  box.replaceChildren();
  const points = series.flatMap((s) => s.points);
  if (!points.length) {
    const p = document.createElement("p");
    p.className = "emptyNote";
    p.textContent = "No successful results in this view.";
    box.append(p);
    return;
  }
  const ns = "http://www.w3.org/2000/svg",
    svg = document.createElementNS(ns, "svg");
  const width = dates ? Math.max(240, box.clientWidth) : 480,
    height = dates ? Math.min(360, Math.max(275, width * 0.32)) : 275,
    plotWidth = width - 87,
    plotBottom = height - 47,
    plotHeight = plotBottom - 23;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  svg.setAttribute(
    "aria-label",
    `${metric === "median" ? "Median" : "Mean"} normalized error versus ${xlabel}`,
  );
  const queryAxis = id === "budgetChart";
  const errorFloor = 1e-12,
    timeFloor = 1e-9;
  const logX = (value) =>
    queryAxis ? Math.log2(value) : Math.log10(Math.max(value, timeFloor));
  const xLow = dates
    ? Math.floor(Math.min(...points.map((p) => p.x)))
    : Math.floor(Math.min(...points.map((p) => logX(p.x))));
  const xHigh = Math.max(
    xLow + 1,
    dates
      ? Math.ceil(Math.max(...points.map((p) => p.x)))
      : Math.ceil(Math.max(...points.map((p) => logX(p.x)))),
  );
  const yLow = Math.floor(
    Math.log10(Math.max(Math.min(...points.map((p) => p.y)), errorFloor)),
  );
  const yHigh = Math.max(
    yLow + 1,
    Math.ceil(Math.log10(Math.max(...points.map((p) => p.y), errorFloor))),
  );
  const xPosition = (value) =>
    62 + ((value - xLow) / (xHigh - xLow)) * plotWidth;
  const yPosition = (value) =>
    plotBottom - ((value - yLow) / (yHigh - yLow)) * plotHeight;
  const x = (value) => xPosition(dates ? value : logX(value));
  const y = (value) => yPosition(Math.log10(Math.max(value, errorFloor)));
  const hasErrorFloor = points.some((p) => p.y <= errorFloor);
  const hasTimeFloor =
    !dates && !queryAxis && points.some((p) => p.x <= timeFloor);
  function element(tag, attributes, text, parent = svg) {
    const node = document.createElementNS(ns, tag);
    Object.entries(attributes).forEach(([key, value]) =>
      node.setAttribute(key, value),
    );
    if (text !== undefined) node.textContent = text;
    parent.append(node);
    return node;
  }
  const tick = (value) =>
    value === 0
      ? "0"
      : Math.abs(value) >= 1000
        ? `${Number((value / 1000).toPrecision(2))}k`
        : Math.abs(value) < 0.001
          ? value.toExponential(1)
          : String(Number(value.toPrecision(2)));
  const ticks = (low, high, intervals = 4) => {
    const step = Math.max(1, Math.ceil((high - low) / intervals));
    const values = [];
    for (let value = low; value <= high; value += step) values.push(value);
    if (values.at(-1) !== high) values.push(high);
    return values;
  };
  element("rect", {
    x: 62,
    y: 23,
    width: plotWidth,
    height: plotHeight,
    fill: "#f8faff",
    rx: 6,
  });
  if (hasErrorFloor)
    element("rect", {
      x: 62,
      y: plotBottom - 10,
      width: plotWidth,
      height: 12,
      fill: "#edf2ff",
    });
  ticks(yLow, yHigh).forEach((exponent) => {
    const yy = yPosition(exponent);
    element("line", {
      x1: 62,
      x2: width - 25,
      y1: yy,
      y2: yy,
      stroke: "#dfe6f0",
      "stroke-dasharray": "3 4",
    });
    element(
      "text",
      {
        x: 55,
        y: yy + 4,
        "text-anchor": "end",
        "font-size": 10,
        fill: "#596980",
      },
      hasErrorFloor && exponent === -12
        ? "≤1e-12"
        : Math.abs(exponent) > 3
          ? `1e${exponent}`
          : tick(10 ** exponent),
    );
  });
  const xTicks = ticks(
    xLow,
    xHigh,
    dates
      ? Math.max(2, Math.min(7, Math.floor(plotWidth / 75)))
      : queryAxis
        ? 8
        : 4,
  );
  // Keep both end labels; omit a nearby interior year on narrow screens.
  if (
    dates &&
    xTicks.length > 2 &&
    xPosition(xTicks.at(-1)) - xPosition(xTicks.at(-2)) < 40
  )
    xTicks.splice(-2, 1);
  xTicks.forEach((value) => {
    const label = dates
      ? String(value)
      : hasTimeFloor && value === -9
        ? "≤1e-9"
        : tick((queryAxis ? 2 : 10) ** value);
    element(
      "text",
      {
        x: xPosition(value),
        y: height - 29,
        "text-anchor": "middle",
        "font-size": 10,
        fill: "#596980",
      },
      label,
    );
  });
  element(
    "text",
    {
      x: 62 + plotWidth / 2,
      y: height - 3,
      "text-anchor": "middle",
      "font-size": 11,
      fill: "#596980",
    },
    `${xlabel}${dates ? "" : " · log"}`,
  );
  element(
    "text",
    { x: 62, y: 12, "font-size": 10, fill: "#596980" },
    `${metric === "median" ? "Median" : "Mean"} nMSE · log ↓`,
  );
  const legend = document.createElement("div");
  legend.className = "legend";
  series.forEach((s) => {
    const method = s.method || s.name,
      color = s.color || colorFor(method),
      dash = s.name === "Best dated result" ? "" : dashFor(method),
      marker = Math.floor(methodIndex(method) / palette.length) % 4;
    const group = element("g", {
      class: s.name === "Best dated result" ? "series frontier" : "series",
      "data-method": method,
    });
    const line = element(
      "polyline",
      {
        points: s.points.map((p) => `${x(p.x)},${y(p.y)}`).join(" "),
        fill: "none",
        stroke: color,
        "stroke-width": s.name === "Best dated result" ? 2.8 : 2.2,
        "stroke-dasharray": dash,
        "stroke-linejoin": "round",
        "stroke-linecap": "round",
        tabindex: 0,
        role: "img",
        "aria-label": `${s.name} error curve`,
      },
      undefined,
      group,
    );
    bindHighlight(
      line,
      method,
      `${s.name}${s.profile ? ` · ${s.profile}` : ""}`,
    );
    s.points.forEach((p) => {
      const cx = x(p.x),
        cy = y(p.y),
        base = {
          fill: "#fff",
          stroke: color,
          "stroke-width": 1.7,
          tabindex: 0,
          role: "img",
        };
      let dot;
      if (marker === 1)
        dot = element(
          "rect",
          { ...base, x: cx - 3, y: cy - 3, width: 6, height: 6 },
          undefined,
          group,
        );
      else if (marker === 2)
        dot = element(
          "polygon",
          {
            ...base,
            points: `${cx},${cy - 4} ${cx + 4},${cy + 3} ${cx - 4},${cy + 3}`,
          },
          undefined,
          group,
        );
      else if (marker === 3)
        dot = element(
          "polygon",
          {
            ...base,
            points: `${cx},${cy - 4} ${cx + 4},${cy} ${cx},${cy + 4} ${cx - 4},${cy}`,
          },
          undefined,
          group,
        );
      else
        dot = element("circle", { ...base, cx, cy, r: 3.5 }, undefined, group);
      const label = `${s.name}\n${metric === "median" ? "Median" : "Mean"} nMSE: ${format(p.y)}\n${xlabel}: ${format(p.x)}${s.profile ? `\nProfile: ${s.profile}` : ""}${p.coverage ? `\nCoverage: ${p.coverage}` : ""}${p.relativeBudget !== undefined ? `\nBudget: ${format(p.relativeBudget)} × players${p.queryUsage ? ` · used ${p.queryUsage}` : ""}` : ""}`;
      dot.setAttribute("aria-label", label);
      element("title", {}, label, dot);
      bindHighlight(dot, method, label);
    });
    const label = document.createElement("button"),
      swatch = document.createElement("span");
    label.type = "button";
    swatch.className = "legendSwatch";
    swatch.style.borderTop = `2px ${dash ? "dashed" : "solid"} ${color}`;
    label.append(
      swatch,
      document.createTextNode(
        `${s.name}${s.releaseYear ? ` (${s.releaseYear})` : ""}`,
      ),
    );
    const item = document.createElement("div");
    item.className = "legendItem";
    item.append(label);
    if (Object.hasOwn(data.methods, method)) {
      const details = methodDetails(method);
      bindMethodDetails(label, details);
      item.append(details);
    }
    bindHighlight(
      label,
      method,
      `${s.name}${s.profile ? ` · ${s.profile}` : ""}`,
    );
    legend.append(item);
  });
  box.append(svg);
  if (hasErrorFloor || hasTimeFloor) {
    const note = document.createElement("p");
    note.className = "scaleNote";
    note.textContent = [
      hasErrorFloor
        ? "nMSE ≤1e-12, including zero, shares the bottom band. Exact values on hover."
        : "",
      hasTimeFloor
        ? "Zero and sub-nanosecond times share the ≤1e-9 s position."
        : "",
    ]
      .filter(Boolean)
      .join(" ");
    box.append(note);
  }
  box.append(legend);
  if (dates) balanceHistoryLegend(legend);
}

function download(kind) {
  const s = selection();
  let content, type;
  if (kind === "json") {
    content = JSON.stringify(
      {
        snapshot_id: data.snapshot_id,
        selection: {
          games: s.games.map((g) => g.id),
          methods: s.methods,
          budgets: s.budgets,
          cells: s.cells,
          seeds: data.suite.seeds,
          game_seeds: data.suite.game_seeds,
        },
        runs: data.runs,
        records: selectedRows,
      },
      null,
      2,
    );
    type = "application/json";
  } else {
    const fields = [
      "game_id",
      "method",
      "budget",
      "seed",
      "status",
      "nmse",
      "mse",
      "queries",
      "seconds",
      "timing_profile",
      "run_id",
    ];
    const quote = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
    content = [
      fields.join(","),
      ...selectedRows.map((row) =>
        fields.map((field) => quote(row[field])).join(","),
      ),
    ].join("\n");
    type = "text/csv";
  }
  const url = URL.createObjectURL(new Blob([content], { type })),
    link = document.createElement("a");
  link.href = url;
  link.download = `shapiq-selection.${kind}`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
[
  "panel",
  "target",
  "family",
  "minPlayers",
  "maxPlayers",
  "budget",
  "methods",
  "cap",
  "historyMetric",
  "chartLimit",
  "chartFamily",
  "showVariants",
].forEach((id) =>
  $(id).addEventListener("change", () => {
    if (id === "panel") $("family").value = "";
    if (id === "chartFamily") chartFamilyExplicit = true;
    if (data) render();
  }),
);
document.querySelectorAll("[data-sort]").forEach((button) => {
  button.addEventListener("click", () => {
    const key = button.dataset.sort;
    tableSort = {
      key,
      descending:
        key === "default"
          ? false
          : tableSort.key === key
            ? !tableSort.descending
            : ["elo", "coverage"].includes(key),
    };
    if (data) render();
  });
});
$("downloadJson").addEventListener("click", () => {
  if (data) download("json");
});
$("downloadCsv").addEventListener("click", () => {
  if (data) download("csv");
});
$("upload").addEventListener("change", async (event) => {
  try {
    const file = event.target.files[0];
    if (file) load(JSON.parse(await file.text()));
  } catch (error) {
    $("notice").textContent = error.message;
  }
});
fetch("data.json")
  .then((response) => {
    if (!response.ok) throw Error("No bundled data");
    return response.json();
  })
  .then(load)
  .catch(() => {
    $("notice").textContent = "Open a local report to get started.";
  });

$("methodSearch").addEventListener("input", buildMethodPicker);
["selectAll", "selectNone"].forEach((id) =>
  $(id).addEventListener("click", () => {
    [...$("methods").options].forEach(
      (option) => (option.selected = id === "selectAll"),
    );
    buildMethodPicker();
    if (data) render();
  }),
);
document.addEventListener("click", (event) =>
  document.querySelectorAll(".toolbar details[open]").forEach((details) => {
    if (!details.contains(event.target)) details.open = false;
  }),
);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    document
      .querySelectorAll(".toolbar details[open]")
      .forEach((details) => (details.open = false));
    highlight(null);
    $("chartTooltip").hidden = true;
  }
});

window.addEventListener("resize", () => {
  if (historyChartState) {
    const expanded = new Set(
      [
        ...$("historyChart").querySelectorAll('button[aria-expanded="true"]'),
      ].map((button) => button.dataset.method),
    );
    const { series, xlabel, metric } = historyChartState;
    chart("historyChart", series, xlabel, true, metric);
    $("historyChart")
      .querySelectorAll(".detailToggle")
      .forEach((button) => {
        if (expanded.has(button.dataset.method)) button.click();
      });
  }
  highlight(null);
  $("chartTooltip").hidden = true;
});
