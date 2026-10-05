"use strict";
const palette = [
  "#426BE0",
  "#BF2355",
  "#007C59",
  "#A34B87",
  "#947000",
  "#287FA8",
];
let data,
  pendingRuns = 0,
  selectedRows = [],
  chartNames = [],
  methodTargets = new Map(),
  unsupportedTargets = new Map(),
  tableSort = { key: "default", descending: false },
  chartFamilyExplicit = false;
let partitionClient = null,
  partitionView = null,
  partitionKey = null,
  partitionRenderVersion = 0,
  partitionDownloadController = null;
let uploadVersion = 0;
let sourceVersion = 0,
  targetVersion = 0,
  localReportFiles = null,
  localSelected = false;
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
const isUnderBudget = (row) =>
  row.status === "failed" && row.failure_reason === "insufficient_budget";
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

const methodNames = {
  PermutationSamplingSV: "Permutation · values",
  PermutationSamplingSII: "Permutation · SII",
  PermutationSamplingSTII: "Permutation · STII",
  RegressionFBII: "Regression · FBII",
  RegressionFSII: "Regression · FSII",
};
const methodLabel = (method) =>
  Object.hasOwn(methodNames, method) ? methodNames[method] : method;
// These pairs use identical base configurations for SV under the benchmark defaults.
const svRepresentatives = {
  KernelSHAPIQ: "KernelSHAP",
  SVARMIQ: "SVARM",
};
function showMethod(method) {
  if ($("showVariants").value === "all") return true;
  if (
    $("target").value === "SV · order 1" &&
    Object.hasOwn(svRepresentatives, method) &&
    Object.hasOwn(data.methods, svRepresentatives[method])
  )
    return false;
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
function methodDetails(method, showHeading = false) {
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
  const parameters = data?.methods?.[method]?.parameters;
  if (parameters && Object.keys(parameters).length) {
    description.textContent += ` Run settings: ${Object.entries(parameters)
      .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
      .join(", ")}.`;
    if (method === "OddSHAP" && parameters.ridge > 0)
      description.textContent +=
        " Ridge applies at budgets up to 3d, except full enumeration.";
  }
  if ($("target").value === "SV · order 1") {
    const aliases = Object.keys(svRepresentatives).filter(
      (alias) =>
        svRepresentatives[alias] === method &&
        Object.hasOwn(data.methods, alias),
    );
    if (aliases.length)
      description.textContent += ` For Shapley values, ${aliases.join(", ")} uses the same estimator configuration. Its separate results are available under All variants.`;
  }
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
  if (showHeading) {
    const heading = document.createElement("h3");
    heading.textContent = methodLabel(method);
    content.append(heading);
  }
  content.append(description, links);
  return content;
}
let nextMethodDetailsId = 0;
function bindMethodDetails(button, panel, exclusiveRoot = null) {
  panel.id = `method-details-${++nextMethodDetailsId}`;
  panel.hidden = true;
  button.classList.add("detailToggle");
  button.setAttribute("aria-expanded", "false");
  button.setAttribute("aria-controls", panel.id);
  button.addEventListener("click", () => {
    if (panel.hidden && exclusiveRoot) {
      exclusiveRoot
        .querySelectorAll('button[aria-expanded="true"]')
        .forEach((other) => {
          other.setAttribute("aria-expanded", "false");
          $(other.getAttribute("aria-controls")).hidden = true;
        });
    }
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
function restoreRecords(value, rows) {
  return rows.map((record) => {
    if (record.worker || !record.worker_id) return record;
    if (!Object.hasOwn(value.workers || {}, record.worker_id))
      throw Error("The report is missing referenced worker metadata.");
    const restored = {
      ...record,
      worker: { ...value.workers[record.worker_id] },
    };
    for (const field of ["peak_rss_bytes", "process_cpu_seconds"]) {
      const key = `worker_${field}`;
      if (Object.hasOwn(restored, key)) {
        restored.worker[field] = restored[key];
        delete restored[key];
      }
    }
    return restored;
  });
}
async function load(value, files = null) {
  const partitioned = value.layout === "partitioned-v1";
  if (
    partitioned &&
    (!value.catalog?.targets?.length ||
      !value.assets ||
      !value.suite?.seeds?.length ||
      !value.methods)
  )
    throw Error("Select a complete partitioned benchmark manifest.");
  if (
    !partitioned &&
    (value.schema_version !== 1 ||
      !value.runs ||
      !value.games?.length ||
      !value.suite?.budgets?.length ||
      !value.suite?.seeds?.length ||
      !value.records ||
      !value.methods)
  )
    throw Error(
      "Select an exported benchmark data.json (generated by shapiq_benchmark.report).",
    );
  const targets = partitioned
    ? value.catalog.targets.map((entry) => entry.target)
    : [...new Set(value.games.map(target))];
  const initialTarget =
    targets.find((name) => name === "SV · order 1") || targets[0];
  const version = ++sourceVersion;
  partitionDownloadController?.abort();
  ++targetVersion;
  if (value.record_shards) {
    $("notice").textContent = "Loading Shapley benchmark results…";
    const descriptor = value.record_shards.find(
      (shard) => shard.target === initialTarget,
    );
    if (!descriptor)
      throw Error("The report is missing its first explanation target.");
    const decoded = await BenchmarkRecords.read(value, descriptor, files);
    if (version !== sourceVersion) return;
    value = { ...value, ...decoded };
  }
  partitionClient?.close();
  partitionClient = partitioned
    ? new BenchmarkPartitionClient(value, files)
    : null;
  partitionView = null;
  partitionKey = null;
  localReportFiles = files;
  data = partitioned
    ? { ...value, games: [], records: [] }
    : { ...value, records: restoreRecords(value, value.records) };
  pendingRuns = countPendingRuns();
  chartFamilyExplicit = false;
  if (!partitioned) updateRecordControls(true);
  $("signalRule").textContent = data.suite.min_signal_ratio
    ? `Enumerated games also require RMS ground truth ≥ ${data.suite.min_signal_ratio} × payoff standard deviation. The same exclusions apply to every method.`
    : "";
  const gamesById = new Map(data.games.map((game) => [game.id, game]));
  methodTargets = new Map(
    Object.keys(data.methods).map((method) => [method, new Set()]),
  );
  unsupportedTargets = new Map(
    Object.keys(data.methods).map((method) => [method, new Set()]),
  );
  if (data.method_targets) {
    for (const [method, statuses] of Object.entries(data.method_targets)) {
      methodTargets.set(method, new Set(statuses.supported || []));
      unsupportedTargets.set(method, new Set(statuses.unsupported || []));
    }
  } else {
    data.records.forEach((record) => {
      const game = gamesById.get(record.game_id);
      if (game) {
        const registry =
          record.status === "unsupported" ? unsupportedTargets : methodTargets;
        registry.get(record.method)?.add(target(game));
      }
    });
  }
  options("target", targets);
  $("target").value = initialTarget;
  if (partitioned) updateRecordControls(true);
  [...$("target").options].forEach(
    (option) => (option.textContent = targetLabel(option.value)),
  );
  options(
    "family",
    partitioned
      ? data.catalog.families
      : [...new Set(data.games.map((g) => g.family))],
    "All families",
  );
  [...$("family").options].forEach((option) => {
    if (option.value) option.textContent = familyLabel(option.value);
  });
  options(
    "dataset",
    partitioned
      ? data.catalog.datasets
      : [
          ...new Set(
            data.games.map((g) => g.metadata?.dataset || "Unrecorded dataset"),
          ),
        ].sort(),
    "All datasets",
  );
  options(
    "model",
    partitioned
      ? data.catalog.models
      : [...new Set(data.games.map(modelProfile))].sort(),
    "All prediction models",
  );
  $("budget").replaceChildren(new Option("All relative budgets", ""));
  const relative = document.createElement("optgroup");
  relative.label = "Queries per player · B/d";
  const measuredRatios =
    data.suite.relative_budgets ||
    data.catalog?.relative_budgets ||
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
        `${Number(r.toPrecision(4))} × players${(partitioned ? targetCatalog().observed_relative_budgets?.includes(r) : data.records.some((row) => row.status !== "unsupported" && row.budget === Math.ceil(r * gamesById.get(row.game_id)?.n_players))) ? "" : " · pending"}`,
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
  $("gameCount").textContent = partitioned
    ? data.catalog.real_case_count
    : new Set(
        data.games
          .filter((g) => !isSynthetic(g))
          .map((g) => g.metadata?.case_id || g.id),
      ).size;
  $("runCount").textContent = new Intl.NumberFormat().format(
    data.evaluated_count ??
      data.records.filter((r) => r.status !== "unsupported").length,
  );
  $("methodSearch").value = "";
  buildMethodPicker();
  renderReportSummary();
  return render();
}
function targetCatalog() {
  return (
    data.catalog?.targets.find((entry) => entry.target === $("target").value) ||
    data.catalog ||
    {}
  );
}
function updateRecordControls(reset = false) {
  const hasCosts = partitionClient
    ? targetCatalog().has_estimated_costs
    : data.records.some((row) =>
        Number.isFinite(row.estimated_uncached_seconds),
      );
  $("timeMetric").options[1].disabled = !hasCosts;
  if (
    reset ||
    (!hasCosts && $("timeMetric").value === "estimated_uncached_seconds")
  )
    $("timeMetric").value = hasCosts ? "estimated_uncached_seconds" : "seconds";
  const games = new Map(data.games.map((game) => [game.id, game]));
  for (const option of $("budget").options) {
    if (!option.value.startsWith("r:")) continue;
    const ratio = Number(option.value.slice(2));
    const measured = partitionClient
      ? targetCatalog().observed_relative_budgets?.includes(ratio)
      : data.records.some(
          (row) =>
            row.status !== "unsupported" &&
            row.budget === Math.ceil(ratio * games.get(row.game_id)?.n_players),
        );
    option.textContent = `${Number(ratio.toPrecision(4))} × players${measured ? "" : " · pending"}`;
  }
}

async function loadTarget() {
  const report = data;
  const descriptor = report.record_shards.find(
    (shard) => shard.target === $("target").value,
  );
  const version = ++targetVersion,
    source = sourceVersion;
  data.records = [];
  data.presets = [];
  render();
  $("notice").textContent = `Loading ${targetLabel($("target").value)}…`;
  try {
    if (!descriptor)
      throw Error("The report is missing this explanation target.");
    const decoded = await BenchmarkRecords.read(
      data,
      descriptor,
      localReportFiles,
    );
    if (
      version !== targetVersion ||
      source !== sourceVersion ||
      data !== report
    )
      return;
    data.records = restoreRecords(data, decoded.records);
    data.presets = decoded.presets;
    updateRecordControls();
    render();
  } catch (error) {
    if (
      version === targetVersion &&
      source === sourceVersion &&
      data === report
    )
      $("notice").textContent = error.message;
  }
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
const gameBudgets = (game) =>
  data.suite.budgets_by_game?.[game.id] || data.suite.budgets;
function countPendingRuns() {
  if (partitionClient)
    return Math.max(0, data.catalog.planned_cells - data.record_count);
  const planned =
    data.games.reduce((sum, game) => sum + gameBudgets(game).length, 0) *
    Object.keys(data.methods).length *
    data.suite.seeds.length;
  return Math.max(0, planned - (data.record_count ?? data.records.length));
}

function selection(family = $("family").value, allBudgets = false) {
  const games = data.games.filter(
    (g) =>
      isSynthetic(g) === ($("panel").value === "diagnostic") &&
      ($("includeControls").checked ||
        g.metadata?.game_quality?.role !== "control") &&
      target(g) === $("target").value &&
      (!family || g.family === family) &&
      (!$("dataset").value ||
        (g.metadata?.dataset || "Unrecorded dataset") === $("dataset").value) &&
      (!$("model").value || modelProfile(g) === $("model").value) &&
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
  const scoreOrder = $("scoreOrder").value;
  const rows = data.records
    .filter((r) => methods.includes(r.method) && keys.has(cellKey(r)))
    .map((row) =>
      scoreOrder
        ? {
            ...row,
            nmse: row.order_scores?.[scoreOrder]?.nmse ?? null,
            mse: row.order_scores?.[scoreOrder]?.mse ?? null,
            score_order: Number(scoreOrder),
          }
        : row,
    );
  const zero = new Set([
    ...data.records.filter((r) => r.zero_truth_energy).map((r) => r.game_id),
    ...data.games
      .filter(
        (g) =>
          g.metadata?.zero_truth_energy ||
          g.metadata?.score_eligible === false ||
          (scoreOrder &&
            g.metadata?.order_scores?.[scoreOrder]?.score_eligible !== true),
      )
      .map((g) => g.id),
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
  const families = new Map(),
    cellsByGame = new Map();
  for (const game of s.games) {
    if (!families.has(game.family)) families.set(game.family, new Map());
    const strata = families.get(game.family);
    strata.set(game.stratum, (strata.get(game.stratum) || 0) + 1);
  }
  for (const cell of s.cells) {
    if (!cellsByGame.has(cell.game_id)) cellsByGame.set(cell.game_id, []);
    cellsByGame.get(cell.game_id).push(cell);
  }
  const weights = new Map();
  s.games.forEach((g) => {
    const strata = families.get(g.family),
      count = strata.get(g.stratum),
      cells = cellsByGame.get(g.id) || [];
    cells.forEach((c) =>
      weights.set(
        cellKey(c),
        1 / families.size / strata.size / count / cells.length,
      ),
    );
  });
  return weights;
}
function compensatedSum(values) {
  let sum = 0,
    correction = 0;
  for (const value of values) {
    const adjusted = value - correction,
      next = sum + adjusted;
    correction = next - sum - adjusted;
    sum = next;
  }
  return sum;
}
function summary(rows, s) {
  const good = rows.filter((r) => r.status === "ok" && Number.isFinite(r.nmse)),
    weights = weightedCells(s),
    // Match Python's accurately summed mass at weighted-median boundaries.
    successfulWeight = compensatedSum(good.map((r) => weights.get(cellKey(r))));
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
    underBudget: rows.filter(isUnderBudget).length,
    unsupported: rows.filter((r) => r.status === "unsupported").length,
    missing: s.cells.length - rows.length,
    complete: s.cells.length > 0 && good.length === s.cells.length,
  };
}
const same = (a, b) =>
  JSON.stringify([...a].sort()) === JSON.stringify([...b].sort());
function partitionRequest() {
  const selected = {
    target: $("target").value,
    panel: $("panel").value,
    include_controls: $("includeControls").checked,
    family: $("family").value,
    dataset: $("dataset").value,
    model: $("model").value,
    min_players: Number($("minPlayers").value),
    max_players: Number($("maxPlayers").value),
    relative_budget: $("budget").value.startsWith("r:")
      ? Number($("budget").value.slice(2))
      : null,
    cap: $("cap").value === "" ? null : Number($("cap").value),
  };
  return {
    selection: selected,
    chart_selection: {
      ...selected,
      family: chartFamilyExplicit ? $("chartFamily").value : selected.family,
    },
    methods: [...$("methods").selectedOptions].map((option) => option.value),
    score_order: $("scoreOrder").value || null,
    timing_metric: $("timeMetric").value,
  };
}

async function renderPartitioned() {
  const client = partitionClient,
    version = ++partitionRenderVersion;
  $("includeControls").disabled = !data.catalog.has_controls;
  if ($("includeControls").disabled) $("includeControls").checked = false;
  for (const option of $("scoreOrder").options)
    option.disabled =
      Boolean(option.value) &&
      !targetCatalog().score_orders.includes(Number(option.value));
  if ($("scoreOrder").selectedOptions[0]?.disabled) $("scoreOrder").value = "";
  buildMethodPicker();
  const request = partitionRequest(),
    key = JSON.stringify(request);
  try {
    if (key !== partitionKey)
      $("notice").textContent = "Loading benchmark selection…";
    if (key === partitionKey) client.cancel();
    let result =
      key === partitionKey ? partitionView : await client.query(request);
    if (client !== partitionClient || version !== partitionRenderVersion)
      return;
    // Match the legacy fallback without turning an automatic family into an explicit choice.
    let desired = request.chart_selection.family;
    if (desired && !result.chart_families.includes(desired)) {
      desired = "";
      if (key !== partitionKey) {
        request.chart_selection.family = "";
        result = await client.query(request);
        if (client !== partitionClient || version !== partitionRenderVersion)
          return;
      }
    }
    partitionView = result;
    partitionKey = key;
    options("chartFamily", result.chart_families, "All families");
    [...$("chartFamily").options].forEach((option) => {
      if (option.value) option.textContent = familyLabel(option.value);
    });
    $("chartFamily").value = desired;
    const visible = request.methods.filter(showMethod),
      all = $("chartLimit").value === "all";
    chartNames = result.chart_ranking
      .filter(
        (row) => visible.includes(row.method) && Number.isFinite(row.median),
      )
      .sort((a, b) =>
        all
          ? a.method.localeCompare(b.method)
          : Number(b.complete) - Number(a.complete) ||
            a.median - b.median ||
            a.method.localeCompare(b.method),
      )
      .slice(0, all ? Infinity : 5)
      .map((row) => row.method);
    const commonOption = [...$("eloPanel").options].find(
      (option) => option.value === "common",
    );
    commonOption.disabled = !targetCatalog().has_common_panel;
    if (commonOption.disabled) $("eloPanel").value = "available";
    const s = { methods: request.methods },
      unit = data.suite.game_seeds?.length ? "game instances" : "games";
    $("chartTooltip").hidden = true;
    $("notice").textContent = [
      pendingRuns
        ? `Provisional results · ${pendingRuns.toLocaleString()} cells pending. Coverage is incomplete; missing results do not count as zero error.`
        : "",
      result.table_pending
        ? "Table results pending for this budget. Charts show all measured budgets."
        : result.chart_pending
          ? "Chart results pending for this selection."
          : "",
    ]
      .filter(Boolean)
      .join(" ");
    $("budgetChartNote").textContent = all
      ? "Family-balanced median · Coverage on hover"
      : "Family-balanced median · Complete coverage preferred";
    $("chartPanelMeta").textContent =
      `${result.chart_selection.game_count} ${unit} · All measured budgets`;
    $("gameDetails").textContent =
      `${replicationLabel()}. Frozen payoffs and exact ground truth are included in the reproduction bundle.`;
    $("panelSummary").textContent =
      `${result.selection.game_count} ${unit} · ${result.selection.planned} cells / estimator${result.selection.excluded ? ` · ${result.selection.excluded} negligible-truth cases excluded` : ""}`;
    if (request.score_order)
      $("panelSummary").textContent +=
        ` · ${$("scoreOrder").selectedOptions[0].textContent}${result.selection.median_energy_share === null ? "" : ` · median truth-energy share ${(100 * result.selection.median_energy_share).toFixed(1)}%`}`;
    $("methodLabel").textContent = `${visible.length} shown`;
    renderLeaderboard(s, result.preset, visible, result.table);
    renderHistory(s, result.preset, result.table);
    renderPerformanceCharts(null, result.chart_pending, result);
    for (const [id, rows] of Object.entries(result.issues)) {
      $(id).replaceChildren();
      const texts = rows.length
        ? rows.map((row) => `${row.description} — ${row.count} run(s)`)
        : [
            id === "underBudget"
              ? "No under-budget runs in this selection."
              : "No other failed runs in this selection.",
          ];
      texts.forEach((text) => {
        const item = document.createElement("li");
        item.textContent = text;
        $(id).append(item);
      });
    }
    const hardware = result.hardware;
    $("hardware").textContent = hardware.cpu_models.length
      ? `Measured workers: ${hardware.cpu_models.join("; ")}. Profiles: ${hardware.timing_profiles.join(", ")}. Per-run placement and thread details are included in JSON downloads.`
      : "Worker hardware was not recorded in this older result.";
  } catch (error) {
    if (
      client === partitionClient &&
      version === partitionRenderVersion &&
      error.name !== "AbortError"
    )
      $("notice").textContent = error.message;
  }
}

async function downloadPartitioned(kind) {
  partitionDownloadController?.abort();
  const controller = new AbortController();
  partitionDownloadController = controller;
  const report = data,
    request = { ...partitionRequest(), elo_panel: $("eloPanel").value };
  try {
    await BenchmarkDownloads.save(report, request, kind, {
      files: localReportFiles,
      signal: controller.signal,
    });
  } catch (error) {
    if (
      data === report &&
      controller === partitionDownloadController &&
      error.name !== "AbortError"
    )
      $("notice").textContent = error.message;
  }
}

function render() {
  if (partitionClient) return renderPartitioned();
  $("includeControls").disabled = !data.games.some(
    (game) => game.metadata?.game_quality?.role === "control",
  );
  if ($("includeControls").disabled) $("includeControls").checked = false;
  const commonOption = [...$("eloPanel").options].find(
    (option) => option.value === "common",
  );
  commonOption.disabled = !(data.presets || []).some(
    (preset) => preset.common_panel,
  );
  if (commonOption.disabled) $("eloPanel").value = "available";
  const targetGames = data.games.filter(
    (game) => target(game) === $("target").value,
  );
  for (const option of $("scoreOrder").options)
    option.disabled =
      Boolean(option.value) &&
      !targetGames.some(
        (game) => game.order > 1 && game.metadata?.order_scores?.[option.value],
      );
  if ($("scoreOrder").selectedOptions[0]?.disabled) $("scoreOrder").value = "";
  const s = selection();
  const visibleMethods = s.methods.filter(showMethod);
  buildMethodPicker();
  selectedRows = s.rows;
  $("chartTooltip").hidden = true;
  const preset = (data.presets || []).find(
    (p) =>
      (p.score_order == null ? "" : String(p.score_order)) ===
        $("scoreOrder").value &&
      Boolean(p.include_controls) === $("includeControls").checked &&
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
  const selectionNotice = tablePending
    ? "Table results pending for this budget. Charts show all measured budgets."
    : chartPending
      ? "Chart results pending for this selection."
      : "";
  $("notice").textContent = [
    pendingRuns
      ? `Provisional results · ${pendingRuns.toLocaleString()} cells pending. Coverage is incomplete; missing results do not count as zero error.`
      : "",
    selectionNotice,
  ]
    .filter(Boolean)
    .join(" ");
  $("budgetChartNote").textContent = allCurves
    ? "Family-balanced median · Coverage on hover"
    : "Family-balanced median · Complete coverage preferred";
  const gameUnit = data.suite.game_seeds?.length ? "game instances" : "games";
  $("chartPanelMeta").textContent =
    `${chartPanel.games.length} ${gameUnit} · All measured budgets`;
  $("gameDetails").textContent =
    `${replicationLabel()}. Frozen payoffs and exact ground truth are included in the reproduction bundle.`;
  $("panelSummary").textContent =
    `${s.games.length} ${gameUnit} · ${s.cells.length} cells / estimator${s.excluded ? ` · ${s.excluded} negligible-truth cases excluded` : ""}`;
  if ($("scoreOrder").value) {
    const shares = s.games
      .map(
        (game) =>
          game.metadata.order_scores[$("scoreOrder").value].energy_share,
      )
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    const middle = Math.floor(shares.length / 2);
    const medianShare = shares.length
      ? (shares[middle] + shares[Math.floor((shares.length - 1) / 2)]) / 2
      : null;
    $("panelSummary").textContent +=
      ` · ${$("scoreOrder").selectedOptions[0].textContent}${medianShare === null ? "" : ` · median truth-energy share ${(100 * medianShare).toFixed(1)}%`}`;
  }
  $("methodLabel").textContent = `${visibleMethods.length} shown`;
  renderLeaderboard(s, preset, visibleMethods);
  renderRunIssues(s);
  renderPerformanceCharts(chartPanel, chartPending);
  renderHistory(s, preset);
  renderHardware(s);
}

function renderLeaderboard(s, preset, visibleMethods, computed = null) {
  const common = $("eloPanel").value === "common";
  const rating = (method) =>
    common
      ? preset?.common_panel?.ratings?.[method]
      : preset?.rows.find((row) => row.method === method)?.elo;
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
  });
  const sortByElo = tableSort.key === "elo";
  const summaries = visibleMethods
    .map((method) => ({
      method,
      ...(computed
        ? computed.find((row) => row.method === method)
        : summary(
            s.rows.filter((r) => r.method === method),
            s,
          )),
      elo: rating(method),
    }))
    .sort(rankingOrder);
  $("ranking").replaceChildren();
  let rank = 0;
  summaries.forEach((item) => {
    const tr = document.createElement("tr"),
      detailRow = document.createElement("tr"),
      detailCell = document.createElement("td");
    detailRow.className = "methodDetailRow";
    detailCell.colSpan = 7;
    detailCell.append(methodDetails(item.method));
    detailRow.append(detailCell);
    const state = item.complete
      ? "Complete"
      : [
          item.underBudget ? `${item.underBudget} under budget` : "",
          item.failed > item.underBudget
            ? `${item.failed - item.underBudget} failed`
            : "",
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
      format(item.elo),
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
    ? common
      ? `Elo uses the same ${preset.common_panel?.cells ?? 0} successful cells for all ${preset.common_panel?.methods?.length ?? 0} methods with results (${((preset.common_panel?.coverage_weight ?? 0) * 100).toFixed(1)}% of panel weight). Error columns still use each method's available cells. No shared cells means no common-panel rating.`
      : "Elo pairs available successful runs; missing outcomes can change rankings. Compare “same cells for all” to check this sensitivity. Hiding variants does not change the competitor set."
    : "Available for preset panels with the full competitor set. Custom filters still show error and coverage.";
}

function renderRunIssues(s) {
  const groups = { underBudget: new Map(), failures: new Map() },
    players = new Map(s.games.map((game) => [game.id, game.n_players]));
  for (const row of s.rows.filter((r) => r.status === "failed")) {
    const underBudget = isUnderBudget(row);
    const explanation = underBudget
      ? Number.isFinite(row.minimum_budget)
        ? `requires at least ${Number(format(row.minimum_budget / players.get(row.game_id)))} × players`
        : row.method === "ProxySPEX"
          ? "too few samples for five-fold proxy fitting"
          : "budget below the sparse transform's minimum"
      : {
          TimeoutError: "time limit reached",
          MemoryError: "memory allocation failed",
          ValueError: "input, configuration, or numerical validation failed",
          ModuleNotFoundError: "optional dependency unavailable",
          ImportError: "dependency could not load",
          LinAlgError: "linear algebra failed",
          BudgetExceededError: "query budget exceeded",
        }[row.error_type] || "estimator error";
    const group = groups[underBudget ? "underBudget" : "failures"];
    const key = `${methodLabel(row.method)}: ${explanation}`;
    group.set(key, (group.get(key) || 0) + 1);
  }
  for (const [id, counts] of Object.entries(groups)) {
    $(id).replaceChildren();
    for (const [description, count] of [...counts]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)) {
      const item = document.createElement("li");
      item.textContent = `${description} — ${count} run(s)`;
      $(id).append(item);
    }
    if (!counts.size) {
      const item = document.createElement("li");
      item.textContent =
        id === "underBudget"
          ? "No under-budget runs in this selection."
          : "No other failed runs in this selection.";
      $(id).append(item);
    }
  }
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
function download(kind) {
  if (partitionClient) return downloadPartitioned(kind);
  const s = selection();
  let content, type;
  if (kind === "json") {
    content = JSON.stringify(
      {
        snapshot_id: data.snapshot_id,
        composition: data.composition || null,
        selection: {
          games: s.games.map((g) => g.id),
          methods: s.methods,
          budgets: s.budgets,
          cells: s.cells,
          model: $("model").value || null,
          dataset: $("dataset").value || null,
          score_order: $("scoreOrder").value
            ? Number($("scoreOrder").value)
            : null,
          include_controls: $("includeControls").checked,
          elo_panel: $("eloPanel").value,
          seeds: data.suite.seeds,
          game_seeds: data.suite.game_seeds,
        },
        games: s.games,
        protocol: data.suite.protocol || null,
        snapshot_provenance: data.snapshot_provenance,
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
      "score_order",
      "mse",
      "queries",
      "seconds",
      "estimated_oracle_seconds",
      "estimated_uncached_seconds",
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
  "dataset",
  "model",
  "minPlayers",
  "maxPlayers",
  "budget",
  "methods",
  "cap",
  "historyMetric",
  "chartLimit",
  "chartFamily",
  "timeMetric",
  "showVariants",
  "scoreOrder",
  "eloPanel",
  "includeControls",
].forEach((id) =>
  $(id).addEventListener("change", () => {
    if (id === "target" && partitionClient) {
      updateRecordControls();
      $("eloPanel").value = "available";
    }
    if (id === "target" && data?.record_shards) {
      loadTarget();
      return;
    }
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
  const version = ++uploadVersion;
  try {
    const files = new Map(
      [...event.target.files].map((file) => [file.name, file]),
    );
    const file = files.get("data.json") || event.target.files[0];
    if (file) {
      localSelected = true;
      ++sourceVersion;
      ++targetVersion;
      ++partitionRenderVersion;
      partitionClient?.cancel();
      partitionDownloadController?.abort();
      $("notice").textContent = "Loading local report…";
      const content = await file.text();
      if (version !== uploadVersion) return;
      await load(JSON.parse(content), files);
    }
  } catch (error) {
    if (version === uploadVersion) $("notice").textContent = error.message;
  }
});
fetch("data.json")
  .then((response) => {
    if (!response.ok) throw Error("No bundled data");
    return response.json();
  })
  .then((value) => {
    if (!localSelected) return load(value);
  })
  .catch((error) => {
    if (!localSelected)
      $("notice").textContent =
        error.message === "No bundled data"
          ? "Open a local report to get started."
          : error.message;
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
