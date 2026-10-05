const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const root = path.resolve(__dirname, "../..");
const app = fs.readFileSync(path.join(root, "benchmark/site/app.js"), "utf8");
const charts = fs.readFileSync(
  path.join(root, "benchmark/site/charts.js"),
  "utf8",
);
const code = fs.readFileSync(
  path.join(root, "benchmark/site/query.js"),
  "utf8",
);
const context = vm.createContext({
  crypto: require("node:crypto").webcrypto,
  TextEncoder,
});
vm.runInContext(code, context);
const api = context.BenchmarkQuery;
const plain = (x) => JSON.parse(JSON.stringify(x));
function fixture() {
  const timing = {
    protocol: "batch-amortized-wall-seconds-v1",
    profiles: [{ cpu_model: "CPU", rate: 2 }],
  };
  const games = Array.from({ length: 7 }, (_, i) => ({
    id: `g${i}`,
    family: i < 4 ? "f" : "other",
    stratum: `s${i % 2}`,
    n_players: 4 + i,
    index: "SV",
    order: 1,
    sequence: i,
    budgets: [0.5, 1, 2].map((x) => Math.ceil(x * (4 + i))),
    row_zero_truth_energy: i === 5,
    metadata: {
      dataset: i % 2 ? "B" : "A",
      model_profile: "forest",
      synthetic: i === 6,
      game_quality: { role: i === 4 ? "control" : "core" },
      order_scores: { 1: { score_eligible: i !== 3, energy_share: 0.25 } },
      ...(i !== 2 ? { evaluation_timing: structuredClone(timing) } : {}),
    },
  }));
  games[1].metadata.evaluation_timing.profiles[0].preparation_hardware = {
    device: "cpu",
    cpu_model: "CPU",
  };
  const worker = {
    cpu_model: "CPU",
    machine: "x86",
    thread_pools: [{ num_threads: 1 }],
    thread_environment: { OMP_NUM_THREADS: "1" },
  };
  const rows = [];
  for (const game of games)
    for (const method of ["A", "B", "Hidden"])
      for (const budget of game.budgets)
        for (const seed of [0, 1]) {
          if (method === "B" && seed === 1 && budget === game.budgets[1])
            continue;
          const sequence = rows.length,
            status =
              sequence % 11 === 0
                ? "failed"
                : sequence % 13 === 0
                  ? "unsupported"
                  : "ok";
          rows.push({
            sequence,
            game_id: game.id,
            method,
            budget,
            seed,
            status,
            nmse: (sequence % 9) / 7,
            queries: budget - 1,
            seconds: sequence / 100,
            estimated_uncached_seconds: sequence / 50,
            run_id: "r",
            worker: sequence % 7 === 0 ? null : worker,
            timing_profile: "diagnostic",
            zero_truth_energy: game.id === "g5" && method === "Hidden",
            order_scores: { 1: { nmse: sequence % 5 } },
            error_type: "ValueError",
            failure_reason: sequence % 22 === 0 ? "insufficient_budget" : null,
            minimum_budget: 10,
          });
        }
  const data = {
    games,
    records: rows,
    methods: { A: {}, B: {}, Hidden: {} },
    suite: {
      seeds: [0, 1],
      relative_budgets: [0.5, 1, 2],
      budgets_by_game: Object.fromEntries(games.map((g) => [g.id, g.budgets])),
    },
  };
  return data;
}
function input(data) {
  const assets = { games: [], profiles: [], metrics: [] },
    blocks = new Map();
  const add = (kind, rows, fields = {}) => {
    const d = { file: `${kind}-${blocks.size}`, kind, ...fields };
    assets[kind].push(d);
    blocks.set(d.file, rows);
  };
  add("games", data.games);
  add(
    "profiles",
    data.records.some((r) => r.worker)
      ? [
          {
            id: "w",
            type: "worker",
            value: data.records.find((r) => r.worker).worker,
          },
        ]
      : [],
  );
  for (const method of Object.keys(data.methods))
    for (const family of [...new Set(data.games.map((g) => g.family))]) {
      const ids = new Set(
        data.games.filter((g) => g.family === family).map((g) => g.id),
      );
      const rows = data.records
        .filter((r) => r.method === method && ids.has(r.game_id))
        .map(({ worker, ...r }) => ({
          ...r,
          ...(worker ? { worker_id: "w" } : {}),
        }));
      // Reverse descriptors/rows deliberately; encounter sequence controls numerical order.
      add("metrics", rows.reverse(), {
        method,
        family,
        target: "SV · order 1",
      });
    }
  const calls = [];
  return {
    manifest: {
      layout: "partitioned-v1",
      methods: data.methods,
      suite: data.suite,
      assets,
    },
    calls,
    read: async (d) => {
      calls.push(d);
      const rows = blocks.get(d.file);
      return {
        count: rows.length,
        row: (i) => rows[i],
        get: (f, i) => rows[i][f],
      };
    },
  };
}
function request(overrides = {}) {
  return {
    selection: { target: "SV · order 1", panel: "real" },
    chart_selection: { target: "SV · order 1", panel: "real" },
    methods: ["A", "B"],
    score_order: "",
    timing_metric: "seconds",
    ...overrides,
  };
}
function legacy(data, req) {
  let active = req.selection;
  const controls = {};
  const node = () => ({
    children: [],
    replaceChildren() {
      this.children = [];
    },
    append(x) {
      this.children.push(x);
    },
  });
  const $ = (id) => controls[id] || (controls[id] = node());
  function set(s) {
    active = s;
    for (const [id, value] of Object.entries({
      panel: s.panel || "real",
      target: s.target,
      family: s.family || "",
      dataset: s.dataset || "",
      model: s.model || "",
      minPlayers: s.min_players ?? 0,
      maxPlayers: s.max_players ?? 9999,
      budget: s.relative_budget != null ? `r:${s.relative_budget}` : "",
      cap: s.cap ?? "",
      scoreOrder: req.score_order,
      timeMetric: req.timing_metric,
    }))
      $(id).value = String(value);
    $("includeControls").checked = Boolean(s.include_controls);
    $("methods").selectedOptions = req.methods.map((value) => ({ value }));
  }
  const output = {};
  const c = vm.createContext({
    data,
    $,
    document: { createElement: node },
    isSynthetic: (g) => Boolean(g.metadata?.synthetic),
    target: (g) => `${g.index} · order ${g.order}`,
    modelProfile: (g) =>
      g.metadata?.model_profile || g.metadata?.model || "No model recorded",
    gameBudgets: (g) => g.budgets,
    isUnderBudget: (r) =>
      r.status === "failed" && r.failure_reason === "insufficient_budget",
    chartNames: req.methods,
    methodLabel: (m) => m,
    colorFor: () => "",
    format: (n) => (Number.isFinite(n) ? n.toPrecision(4) : "—"),
    chart: (id, series) => (output[id] = series),
  });
  vm.runInContext(
    app.slice(
      app.indexOf("function selection("),
      app.indexOf("function render()"),
    ),
    c,
  );
  vm.runInContext(
    charts.slice(
      charts.indexOf("function canonicalOracleTiming"),
      charts.indexOf("function renderHistory"),
    ),
    c,
  );
  vm.runInContext(
    app.slice(
      app.indexOf("function renderRunIssues"),
      app.indexOf("function appendRow"),
    ),
    c,
  );
  set(req.selection);
  const tablePanel = c.selection();
  c.renderRunIssues(tablePanel);
  c.renderHardware(tablePanel);
  const table = req.methods.map((method) => ({
    method,
    ...c.summary(
      tablePanel.rows.filter((r) => r.method === method),
      tablePanel,
    ),
  }));
  set(req.chart_selection);
  const cp = c.selection(req.chart_selection.family || "", true);
  c.renderPerformanceCharts(cp, false);
  const strip = (series) => series.map(({ name, color, ...rest }) => rest);
  return plain({
    issues: Object.fromEntries(
      ["underBudget", "failures"].map((k) => [
        k,
        $(k)
          .children.filter((n) => n.textContent.includes(" — "))
          .map((n) => {
            const [description, count] = n.textContent.split(" — ");
            return { description, count: parseInt(count) };
          }),
      ]),
    ),
    hardware_text: $("hardware").textContent,
    table,
    chart_ranking: req.methods.map((method) => ({
      method,
      ...c.summary(
        cp.rows.filter((r) => r.method === method),
        cp,
      ),
    })),
    budget_series: strip(output.budgetChart),
    time_series: strip(output.timeChart),
  });
}
for (const timing_metric of ["seconds", "estimated_uncached_seconds"])
  for (const variant of [
    "default",
    "filtered",
    "order",
    "controls",
    "diagnostic",
  ])
    test(`legacy exact ${timing_metric}/${variant}`, async () => {
      const data = fixture(),
        options = input(data);
      let r = request({ timing_metric });
      if (variant === "filtered") {
        r.selection = {
          ...r.selection,
          family: "f",
          dataset: "A",
          relative_budget: 1,
          cap: 2,
        };
        r.chart_selection = { ...r.chart_selection, family: "other" };
      }
      if (variant === "order") r.score_order = "1";
      if (variant === "controls") {
        r.selection.include_controls = true;
        r.chart_selection.include_controls = true;
      }
      if (variant === "diagnostic") {
        r.selection.panel = "diagnostic";
        r.chart_selection.panel = "diagnostic";
      }
      const got = await api.query(options.manifest, r, options),
        expected = legacy(data, r);
      const h = got.hardware;
      assert.equal(
        h.cpu_models.length
          ? `Measured workers: ${h.cpu_models.join("; ")}. Profiles: ${h.timing_profiles.join(", ")}. Per-run placement and thread details are included in JSON downloads.`
          : "Worker hardware was not recorded in this older result.",
        expected.hardware_text,
      );
      delete expected.hardware_text;
      for (const key of Object.keys(expected))
        assert.deepEqual(plain(got[key]), expected[key], key);
      assert.equal(got.preset, null);
      assert(!("game_ids" in got.selection));
    });
test("hidden-method zero exclusion with indexed pruning", async () => {
  const data = fixture(),
    o = input(data);
  const got = await api.query(o.manifest, request({ methods: ["A"] }), o);
  assert.equal(got.selection.excluded, 1);
  assert(!o.calls.some((d) => d.method === "Hidden" || d.method === "B"));
});
test("bounds and canceled queries reject rather than truncate", async () => {
  const o = input(fixture());
  await assert.rejects(
    api.query(o.manifest, request(), { ...o, limits: { method_rows: 1 } }),
    /memory limit/,
  );
  const signal = AbortSignal.abort();
  await assert.rejects(
    api.query(o.manifest, request(), { ...o, signal }),
    /abort/i,
  );
  let reads = 0;
  const controller = new AbortController();
  await assert.rejects(
    api.query(o.manifest, request(), {
      ...o,
      signal: controller.signal,
      read: async (d) => {
        const b = await o.read(d);
        if (++reads === 2) controller.abort();
        return b;
      },
    }),
    /abort/i,
  );
});
test("preset lookup authenticates semantic match; custom selection has no invented Elo", async () => {
  const data = fixture(),
    o = input(data),
    r = request();
  const ids = data.games
    .filter(
      (g) =>
        !g.metadata.synthetic && g.metadata.game_quality.role !== "control",
    )
    .map((g) => g.id);
  const p = {
    game_ids: ids,
    methods: r.methods,
    game_budgets: Object.fromEntries(data.games.map((g) => [g.id, g.budgets])),
    score_order: null,
    include_controls: false,
    panel: "real",
    rows: [{ method: "A", elo: 1234 }],
    history: { unchanged: true },
  };
  let selector;
  const got = await api.query(o.manifest, r, {
    ...o,
    lookupPreset: async (x) => {
      selector = x;
      return p;
    },
  });
  assert.deepEqual(plain(got.preset), { rows: p.rows, history: p.history });
  assert(
    !Object.hasOwn(got.preset, "game_ids") &&
      !Object.hasOwn(got.preset, "game_budgets"),
  );
  assert.match(selector.sha256, /^[a-f0-9]{64}$/);
  const bad = await api.query(o.manifest, r, {
    ...o,
    preset: { ...p, methods: ["A"] },
  });
  assert.equal(bad.preset, null);
});
test("198 equal weights hit the exact compensated half-mass midpoint", async () => {
  const data = fixture(),
    g = data.games[0],
    r = data.records.find((r) => r.status === "ok");
  data.games = Array.from({ length: 198 }, (_, i) => ({
    ...g,
    id: `g${i}`,
    stratum: "one",
    sequence: i,
    budgets: [4],
    row_zero_truth_energy: false,
  }));
  data.records = data.games.map((g, i) => ({
    ...r,
    game_id: g.id,
    sequence: i,
    method: "A",
    budget: 4,
    seed: 0,
    status: "ok",
    nmse: i < 99 ? 0 : 1,
    zero_truth_energy: false,
  }));
  data.suite.seeds = [0];
  data.suite.relative_budgets = [1];
  const o = input(data),
    q = request({ methods: ["A"] });
  const got = await api.query(o.manifest, q, o);
  assert.equal(got.table[0].median, 0.5);
  assert.deepEqual(plain(got.table), legacy(data, q).table);
});
test("zero-game profile encounter cannot reorder surviving curves", async () => {
  const data = fixture(),
    worker = data.records.find((r) => r.worker).worker;
  data.games = data.games.slice(0, 3).map((g, i) => ({
    ...g,
    row_zero_truth_energy: i === 0,
    budgets: [g.n_players],
  }));
  data.records = data.games.map((g, i) => ({
    sequence: i,
    game_id: g.id,
    method: "A",
    budget: g.n_players,
    seed: 0,
    status: "ok",
    nmse: i,
    seconds: i,
    queries: g.n_players,
    worker,
    run_id: "r",
    timing_profile: i === 1 ? "B" : "A",
    zero_truth_energy: i === 0,
  }));
  data.suite.seeds = [0];
  data.suite.relative_budgets = [1];
  const q = request({ methods: ["A"] }),
    o = input(data);
  const got = await api.query(o.manifest, q, o);
  assert.deepEqual(plain(got.time_series), legacy(data, q).time_series);
  assert.deepEqual(plain(got.time_series.map((s) => s.profile)), [
    "CPU · B",
    "CPU · A",
  ]);
});
test("empty and all-pending panels retain planned coverage and no finite scores", async () => {
  const data = fixture(),
    o = input({ ...data, records: [] });
  const result = await api.query(o.manifest, request(), o);
  assert(result.table_pending);
  assert(result.chart_pending);
  assert(
    result.table.every(
      (s) => s.median === null && s.valid === 0 && s.missing === s.planned,
    ),
  );
  const empty = await api.query(
    o.manifest,
    request({
      selection: { target: "SV · order 1", min_players: 999 },
      chart_selection: { target: "SV · order 1", min_players: 999 },
    }),
    o,
  );
  assert(!empty.table_pending);
  assert.equal(empty.selection.planned, 0);
});
test("selector hash uses codepoint sorting and is order invariant", async () => {
  const p = {
      panel_ids: ["\u{10000}", "\ue000"],
      game_budgets: { "\u{10000}": [8, 2], "\ue000": [3, 1] },
    },
    q = request({ methods: ["\u{10000}", "\ue000"], score_order: "2" });
  const hash = await api.selectorHash(p, q);
  const expected = require("node:crypto")
    .createHash("sha256")
    .update(
      JSON.stringify([
        2,
        false,
        "real",
        ["\ue000", "\u{10000}"],
        [
          ["\ue000", [1, 3]],
          ["\u{10000}", [2, 8]],
        ],
      ]),
    )
    .digest("hex");
  assert.equal(hash, expected);
  assert.equal(
    hash,
    await api.selectorHash(
      { ...p, panel_ids: [...p.panel_ids].reverse() },
      { ...q, methods: [...q.methods].reverse() },
    ),
  );
});
test("narrow family requests do not fetch unrelated metrics", async () => {
  const data = fixture(),
    o = input(data),
    q = request({ methods: ["A"] });
  q.selection.family = "f";
  q.chart_selection.family = "f";
  await api.query(o.manifest, q, o);
  assert(
    o.calls
      .filter((d) => d.kind === "metrics")
      .every((d) => d.family === "f" && d.method === "A"),
  );
});
test("many successful unverified profiles use linear row filtering work", async () => {
  const measure = async (n) => {
    const data = fixture(),
      game = data.games[0],
      row = data.records[1];
    data.games = Array.from({ length: n }, (_, i) => ({
      ...game,
      id: `p${i}`,
      sequence: i,
      budgets: [4],
      row_zero_truth_energy: false,
    }));
    data.records = data.games.map((g, i) => ({
      ...row,
      game_id: g.id,
      sequence: i,
      method: "A",
      budget: 4,
      seed: 0,
      status: "ok",
      nmse: i,
      seconds: i + 1,
      worker: null,
      zero_truth_energy: false,
    }));
    data.suite.seeds = [0];
    data.suite.relative_budgets = [1];
    const c = vm.createContext({
      crypto: require("node:crypto").webcrypto,
      TextEncoder,
    });
    vm.runInContext(
      `globalThis.predicateCalls=0; const originalFilter=Array.prototype.filter; Array.prototype.filter=function(fn,...args){return originalFilter.call(this,(...xs)=>{predicateCalls++;return fn(...xs);},...args);};`,
      c,
    );
    vm.runInContext(code, c);
    const o = input(data),
      result = await c.BenchmarkQuery.query(
        o.manifest,
        request({ methods: ["A"] }),
        o,
      );
    assert.equal(result.time_series.length, n);
    return c.predicateCalls;
  };
  const small = await measure(40),
    large = await measure(80);
  assert(large < small * 2.5, `filter work grew ${small} -> ${large}`);
});
test("pathological timing output fails clearly instead of truncating", async () => {
  const o = input(fixture());
  await assert.rejects(
    api.query(o.manifest, request(), { ...o, limits: { series: 1 } }),
    /timing output exceeds memory limit/,
  );
});
