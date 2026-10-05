const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const vm = require("node:vm");
const { spawnSync } = require("node:child_process");
const root = path.resolve(__dirname, "../..");
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "shapiq-download-test-"));
const python =
  process.env.PYTHON ||
  (fs.existsSync(path.join(root, ".venv/bin/python"))
    ? path.join(root, ".venv/bin/python")
    : "python");
const result = spawnSync(
  python,
  [
    "-c",
    `
import copy,json,sys
from pathlib import Path
from tests.shapiq_benchmark.test_partitioned import fixture_data
from shapiq_benchmark.partitioned import write_partitioned_report
from shapiq_benchmark.record_store import RecordStore
root=Path(sys.argv[1]); data=fixture_data()
data['suite'].update(game_seeds=[0,1],protocol={'name':'fixture'}, relative_budgets=[1,2])
data['composition']={'panels': {str(i): ['α😀',i] for i in range(500)}}
base=copy.deepcopy(data['games'][0])
for name,family in [('other','second'),('zero','local')]:
    game=copy.deepcopy(base); game.update(id=name,family=family); data['games'].append(game)
    data['records'].extend([{**copy.deepcopy(r),'game_id':name} for r in data['records'][:2]])
data['methods']['Hidden']=copy.deepcopy(data['methods']['KernelSHAP'])
data['records'].append({**copy.deepcopy(data['records'][0]),'method':'Hidden','game_id':'zero','zero_truth_energy':True})
data['records']=data['records'][::2]+data['records'][1::2]
(root/'expected.json').write_text(json.dumps(data,allow_nan=False))
with RecordStore(root/'rows.sqlite') as store:
    store.extend(data['records'])
    write_partitioned_report({**data,'records':store},root/'site',block_rows=2,max_bytes=8192)
`,
    tmp,
  ],
  {
    cwd: root,
    env: {
      ...process.env,
      PYTHONPATH: path.join(root, "src"),
      PYTHONDONTWRITEBYTECODE: "1",
    },
    encoding: "utf8",
    timeout: 60000,
  },
);
assert.equal(result.status, 0, result.stdout + result.stderr);
test.after(() => fs.rmSync(tmp, { recursive: true, force: true }));
globalThis.crypto = require("node:crypto").webcrypto;
for (const name of [
  "partitions",
  "partition-details",
  "query",
  "partition-download",
])
  require(path.join(root, "benchmark/site", `${name}.js`));
const data = JSON.parse(fs.readFileSync(path.join(tmp, "expected.json")));
const manifest = JSON.parse(fs.readFileSync(path.join(tmp, "site/data.json")));
const files = new Map(
  Object.values(manifest.assets)
    .flat()
    .map((d) => [
      d.file,
      new Blob([fs.readFileSync(path.join(tmp, "site", d.file))]),
    ]),
);
const app = fs.readFileSync(path.join(root, "benchmark/site/app.js"), "utf8");
const plain = (x) => JSON.parse(JSON.stringify(x));
const request = (extra = {}) => ({
  selection: {
    target: "SV · order 1",
    panel: "real",
    include_controls: true,
    min_players: 0,
    max_players: 1000,
  },
  methods: ["KernelSHAP", "SVARM"],
  score_order: null,
  elo_panel: "common",
  ...extra,
});
function legacy(req) {
  const s = req.selection;
  const inputs = {
    family: s.family || "",
    panel: s.panel,
    target: s.target,
    dataset: s.dataset || "",
    model: s.model || "",
    minPlayers: s.min_players ?? 0,
    maxPlayers: s.max_players ?? Infinity,
    budget: s.relative_budget == null ? "" : `r:${s.relative_budget}`,
    cap: s.cap ?? "",
    scoreOrder: req.score_order ?? "",
  };
  const c = vm.createContext({
    data,
    $: (id) =>
      id === "methods"
        ? { selectedOptions: req.methods.map((value) => ({ value })) }
        : id === "includeControls"
          ? { checked: s.include_controls }
          : { value: String(inputs[id]) },
    isSynthetic: (g) => Boolean(g.metadata?.synthetic),
    target: (g) => `${g.index} · order ${g.order}`,
    modelProfile: (g) =>
      g.metadata?.model_profile || g.metadata?.model || "No model recorded",
    gameBudgets: (g) =>
      data.suite.budgets_by_game?.[g.id] || data.suite.budgets,
  });
  vm.runInContext(
    app.slice(
      app.indexOf("function selection("),
      app.indexOf("function weightedCells("),
    ),
    c,
  );
  const p = plain(c.selection());
  return {
    snapshot_id: data.snapshot_id,
    composition: data.composition || null,
    selection: {
      games: p.games.map((g) => g.id),
      methods: p.methods,
      budgets: p.budgets,
      cells: p.cells,
      model: s.model || null,
      dataset: s.dataset || null,
      score_order: req.score_order ? Number(req.score_order) : null,
      include_controls: Boolean(s.include_controls),
      elo_panel: req.elo_panel,
      seeds: data.suite.seeds,
      game_seeds: data.suite.game_seeds,
    },
    games: p.games,
    protocol: data.suite.protocol || null,
    snapshot_provenance: data.snapshot_provenance,
    runs: data.runs,
    records: p.rows,
  };
}
function spoolFixture() {
  const state = { closed: 0, batches: [], rows: new Map(), cells: new Set() };
  return {
    state,
    createSpool: async () => ({
      async add(rows) {
        state.batches.push(rows.length);
        for (const r of rows) {
          assert(
            !state.rows.has(r.sequence) && !state.cells.has(r.cell),
            "duplicate cell or sequence",
          );
          state.rows.set(r.sequence, r);
          state.cells.add(r.cell);
        }
      },
      async *rows() {
        for (const r of [...state.rows.values()].sort(
          (a, b) => a.sequence - b.sequence,
        ))
          yield r.value;
      },
      async close() {
        state.closed++;
      },
    }),
  };
}
function sinkFixture() {
  const state = { text: "", closed: 0, aborted: 0, active: 0, max: 0 };
  return {
    state,
    sink: {
      async write(t) {
        state.active++;
        state.max = Math.max(state.active, state.max);
        await new Promise((resolve) => setImmediate(resolve));
        state.text += t;
        state.active--;
      },
      async close() {
        state.closed++;
      },
      async abort() {
        state.aborted++;
        state.text = "";
      },
    },
  };
}
async function run(req = request(), kind = "json", options = {}) {
  const spool = spoolFixture(),
    sink = sinkFixture();
  const result = await BenchmarkDownloads.stream(manifest, req, kind, {
    files,
    ...spool,
    ...sink,
    ...options,
  });
  assert.equal(spool.state.closed, 1);
  assert.equal(sink.state.closed, 1);
  assert.equal(sink.state.aborted, 0);
  assert.equal(sink.state.max, 1);
  return { result, text: sink.state.text, spool: spool.state };
}
for (const [name, req] of [
  ["full selected panel", request()],
  ["score-order projection", request({ score_order: 1 })],
  [
    "interaction target with null/absent workers",
    request({
      selection: {
        target: "SII · order 2",
        panel: "real",
        include_controls: true,
      },
    }),
  ],
  [
    "family and budget cap",
    request({
      selection: {
        target: "SV · order 1",
        panel: "real",
        family: "local",
        cap: 1,
      },
    }),
  ],
  [
    "missing requested budget",
    request({
      selection: {
        target: "SV · order 1",
        panel: "real",
        relative_budget: 0.5,
      },
    }),
  ],
  [
    "empty capped panel",
    request({ selection: { target: "SV · order 1", panel: "real", cap: 0.5 } }),
  ],
])
  test(`actual Python writer → JSON matches legacy: ${name}`, async () => {
    const actual = await run(req);
    assert.deepEqual(JSON.parse(actual.text), legacy(req));
    assert.equal(actual.result.records, legacy(req).records.length);
  });
test("CSV matches legacy quoting, order projection and sequence exactly", async () => {
  const req = request({ score_order: 1 }),
    expected = legacy(req);
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
  const csv = [
    fields.join(","),
    ...expected.records.map((r) =>
      fields
        .map((k) => '"' + String(r[k] ?? "").replaceAll('"', '""') + '"')
        .join(","),
    ),
  ].join("\n");
  assert.equal((await run(req, "csv")).text, csv);
});
test("hidden-method zero rows exclude a game; methods and other targets are never read", async () => {
  const seen = [];
  const read = async (d) => {
    seen.push(d);
    return BenchmarkPartitions.read(manifest, d, { files });
  };
  const value = JSON.parse((await run(request(), "json", { read })).text);
  assert(!value.selection.games.includes("zero"));
  assert(!value.records.some((r) => r.game_id === "zero"));
  assert(
    seen
      .filter((d) => d.kind === "raw")
      .every((d) => d.target === "SV · order 1" && d.method !== "Hidden"),
  );
});
for (const [name, limits] of [
  ["row count", { rows: 1 }],
  ["row bytes", { row_bytes: 10 }],
  ["spool bytes", { spool_bytes: 10 }],
  ["games", { games: 1 }],
  ["profiles", { profiles: 1 }],
])
  test(`${name} limit fails without a successful partial output`, async () => {
    const spool = spoolFixture(),
      sink = sinkFixture();
    await assert.rejects(
      BenchmarkDownloads.stream(manifest, request(), "json", {
        files,
        ...spool,
        ...sink,
        limits,
      }),
      /limit/,
    );
    assert.equal(sink.state.closed, 0);
    assert.equal(sink.state.aborted, 1);
    assert.equal(sink.state.text, "");
    assert(spool.state.closed <= 1);
  });
test("corrupt late block aborts sink and removes spool", async () => {
  const spool = spoolFixture(),
    sink = sinkFixture();
  let raw = 0;
  const read = (d) => {
    if (d.kind === "raw" && ++raw === 2) throw Error("corrupt checksum");
    return BenchmarkPartitions.read(manifest, d, { files });
  };
  await assert.rejects(
    BenchmarkDownloads.stream(manifest, request(), "json", {
      ...spool,
      ...sink,
      read,
    }),
    /checksum/,
  );
  assert.equal(spool.state.closed, 1);
  assert.equal(sink.state.aborted, 1);
  assert.equal(sink.state.closed, 0);
});
test("cancellation during row spill aborts sink and removes spool", async () => {
  const spool = spoolFixture(),
    sink = sinkFixture(),
    controller = new AbortController();
  const make = spool.createSpool;
  spool.createSpool = async () => {
    const s = await make();
    return {
      ...s,
      async add(rows) {
        await s.add(rows);
        controller.abort(Error("cancelled"));
      },
    };
  };
  await assert.rejects(
    BenchmarkDownloads.stream(manifest, request(), "json", {
      files,
      ...spool,
      ...sink,
      signal: controller.signal,
    }),
    /cancelled/,
  );
  assert.equal(spool.state.closed, 1);
  assert.equal(sink.state.aborted, 1);
});
test("sink failure aborts output and deletes temporary rows", async () => {
  const spool = spoolFixture(),
    sink = sinkFixture();
  sink.sink.write = async () => {
    throw Error("disk quota");
  };
  await assert.rejects(
    BenchmarkDownloads.stream(manifest, request(), "json", {
      files,
      ...spool,
      ...sink,
    }),
    /disk quota/,
  );
  assert.equal(spool.state.closed, 1);
  assert.equal(sink.state.closed, 0);
  assert.equal(sink.state.aborted, 1);
});
test("Blob fallback is bounded before append, never offers truncated output", async () => {
  const spool = spoolFixture();
  let downloaded = 0;
  await assert.rejects(
    BenchmarkDownloads.save(manifest, request(), "json", {
      files,
      ...spool,
      maxBlobBytes: 10,
      download: () => downloaded++,
    }),
    /at most/,
  );
  assert.equal(downloaded, 0);
  assert.equal(spool.state.closed, 1);
});
test("Blob fallback offers exact complete JSON only after spool cleanup", async () => {
  const spool = spoolFixture();
  let blob;
  await BenchmarkDownloads.save(manifest, request(), "json", {
    files,
    ...spool,
    download: (b) => {
      assert.equal(spool.state.closed, 1);
      blob = b;
    },
  });
  assert.deepEqual(JSON.parse(await blob.text()), legacy(request()));
});
test("request is captured before asynchronous reads", async () => {
  const req = request(),
    expected = legacy(req);
  let once = false;
  const read = (d) => {
    if (!once) {
      once = true;
      req.methods.length = 0;
      req.selection.target = "SII · order 2";
    }
    return BenchmarkPartitions.read(manifest, d, { files });
  };
  assert.deepEqual(
    JSON.parse((await run(req, "json", { read })).text),
    expected,
  );
});
test("invalid format or limits abort already-open sinks", async () => {
  for (const options of [
    { kind: "bad" },
    { kind: "json", limits: { rows: 0 } },
  ]) {
    const sink = sinkFixture();
    await assert.rejects(
      BenchmarkDownloads.stream(manifest, request(), options.kind, {
        ...sink,
        limits: options.limits,
      }),
      /format|limits/,
    );
    assert.equal(sink.state.aborted, 1);
  }
});
test("unavailable IndexedDB produces a clear failure", async () => {
  await assert.rejects(
    BenchmarkDownloads.indexedSpool(),
    /storage is unavailable/,
  );
});

test("spool write failure, duplicate cell and missing provenance all abort cleanly", async () => {
  for (const mode of ["quota", "duplicate", "provenance", "worker"]) {
    const spool = spoolFixture(),
      sink = sinkFixture();
    const original = spool.createSpool;
    if (mode === "quota")
      spool.createSpool = async () => ({
        ...(await original()),
        async add() {
          throw Error("quota exceeded");
        },
      });
    let first;
    const read = async (d) => {
      const block = await BenchmarkPartitions.read(manifest, d, { files });
      if (d.kind !== "raw") return block;
      return {
        ...block,
        row(i) {
          const row = block.row(i);
          if (mode === "provenance") row.run_id = "missing";
          if (mode === "worker") row.worker_id = "missing";
          if (mode === "duplicate") {
            first ??= { ...row };
            return { ...first, sequence: row.sequence };
          }
          return row;
        },
      };
    };
    await assert.rejects(
      BenchmarkDownloads.stream(manifest, request(), "json", {
        ...spool,
        ...sink,
        read,
      }),
      /quota|duplicate|provenance|worker/,
    );
    assert.equal(sink.state.closed, 0);
    assert.equal(sink.state.aborted, 1);
    assert.equal(spool.state.closed, 1);
  }
});
test("file picker stream closes on success and aborts on invalid selection", async () => {
  try {
    for (const valid of [true, false]) {
      const sink = sinkFixture(),
        spool = spoolFixture();
      globalThis.showSaveFilePicker = async () => ({
        createWritable: async () => sink.sink,
      });
      const req = request();
      if (!valid) req.methods = ["unknown"];
      const task = BenchmarkDownloads.save(manifest, req, "csv", {
        files,
        ...spool,
      });
      if (valid) {
        await task;
        assert.equal(sink.state.closed, 1);
      } else {
        await assert.rejects(task, /unknown/);
        assert.equal(sink.state.aborted, 1);
      }
    }
  } finally {
    delete globalThis.showSaveFilePicker;
  }
});
