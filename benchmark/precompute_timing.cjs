#!/usr/bin/env node
// Reuse the browser reducer offline; no separate timing or weighting implementation.
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { Worker, isMainThread, parentPort, workerData } = require("node:worker_threads");

async function compute({ report, request }) {
  globalThis.crypto ||= crypto.webcrypto;
  require("./site/partitions.js");
  require("./site/query.js");
  const manifest = JSON.parse(fs.readFileSync(path.join(report, "data.json")));
  const files = { get(name) {
    const bytes = fs.readFileSync(path.join(report, name));
    return { size: bytes.length, arrayBuffer: async () =>
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) };
  } };
  const read = (descriptor) => BenchmarkPartitions.read(manifest, descriptor, { files });
  const games = [];
  for (const descriptor of manifest.assets.games) {
    if (descriptor.target !== request.selection.target) continue;
    const block = await read(descriptor);
    for (let i = 0; i < block.count; i++) games.push(block.row(i));
  }
  const panel = BenchmarkQuery.selectionPanel(
    games, request.selection, manifest.suite, request.score_order,
  );
  const selector = await BenchmarkQuery.selectorHash(panel, request);
  const result = await BenchmarkQuery.query(manifest, request, { read });
  return { selector, metric: request.timing_metric, series: result.time_series };
}

async function main() {
  const [report, output, count = "4"] = process.argv.slice(2);
  const workers = Number(count);
  if (!report || !output || !Number.isInteger(workers) || workers < 1 || workers > 32)
    throw Error("Usage: node benchmark/precompute_timing.cjs REPORT OUTPUT [1–32 workers]");
  const bytes = fs.readFileSync(path.join(report, "data.json"));
  const manifest = JSON.parse(bytes);
  const tasks = [];
  for (const target of manifest.catalog.targets) {
    for (const family of ["", ...target.families]) {
      for (const order of [null, ...target.score_orders]) {
        for (const metric of ["seconds", "estimated_uncached_seconds"]) {
          const selection = { target: target.target, panel: "real", include_controls: false,
            family, dataset: "", model: "", min_players: 1, max_players: 10000,
            relative_budget: null, cap: null };
          tasks.push({ report, request: { selection, chart_selection: selection,
            methods: Object.keys(manifest.methods), score_order: order,
            timing_metric: metric } });
        }
      }
    }
  }
  const cache = { schema_version: 1, snapshot_id: manifest.snapshot_id,
    manifest_sha256: crypto.createHash("sha256").update(bytes).digest("hex"),
    query_sha256: crypto.createHash("sha256")
      .update(fs.readFileSync(path.join(__dirname, "site/query.js"))).digest("hex"),
    entries: {} };
  let next = 0, done = 0;
  await Promise.all(Array.from({ length: workers }, async () => {
    while (next < tasks.length) {
      const task = tasks[next++];
      const value = await new Promise((resolve, reject) => {
        const worker = new Worker(__filename, { workerData: task });
        let result;
        worker.once("message", (message) => { result = message; });
        worker.once("error", reject);
        worker.once("exit", (code) => code === 0 && result
          ? resolve(result) : reject(Error(`Timing worker exited ${code}`)));
      });
      const entry = cache.entries[value.selector] ||= {};
      if (entry[value.metric] && JSON.stringify(entry[value.metric]) !== JSON.stringify(value.series))
        throw Error("Conflicting timing selector");
      entry[value.metric] = value.series;
      console.log(`${++done}/${tasks.length} ${task.request.selection.target} ${task.request.selection.family || "all"} ${value.metric}`);
    }
  }));
  const serialized = JSON.stringify(cache) + "\n";
  fs.writeFileSync(output, serialized, { flag: "wx" });
  const metadata = { snapshot_id: cache.snapshot_id, bytes: Buffer.byteLength(serialized),
    sha256: crypto.createHash("sha256").update(serialized).digest("hex"),
    query_sha256: cache.query_sha256,
    inputs_sha256: crypto.createHash("sha256").update(JSON.stringify({
      assets: manifest.assets, suite: manifest.suite, methods: manifest.methods,
    })).digest("hex") };
  fs.writeFileSync(output.replace(/\.json$/, "-metadata.js"),
    "globalThis.BenchmarkTimingSource = Object.freeze(" + JSON.stringify(metadata) + ");\n",
    { flag: "wx" });
}

if (isMainThread) main().catch((error) => { console.error(error); process.exitCode = 1; });
else compute(workerData).then((result) => parentPort.postMessage(result));
