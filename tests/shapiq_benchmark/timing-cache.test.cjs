const test = require("node:test");
const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.resolve(__dirname, "../../benchmark/site/timing-cache.js"), "utf8");
const hash = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const plain = (value) => JSON.parse(JSON.stringify(value));
const buffer = (bytes) => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);

function fixture({ changeCache, changeMetadata, fetcher } = {}) {
  const manifest = {
    snapshot_id: "a".repeat(64),
    assets: { games: [{ sha256: "b".repeat(64) }] },
    suite: { seeds: [0, 1], relative_budgets: [0.5, 1] },
    methods: { A: { parameters: {} } },
  };
  const query = Buffer.from("// fixed reducer bytes\n");
  const series = [{ method: "A", profile: "CPU", points: [{ x: 2, y: 0.25 }] }];
  const cache = { schema_version: 1, snapshot_id: manifest.snapshot_id,
    query_sha256: hash(query), entries: { selector: { seconds: series } } };
  changeCache?.(cache);
  const bytes = Buffer.from(JSON.stringify(cache));
  const metadata = { snapshot_id: manifest.snapshot_id, bytes: bytes.length,
    sha256: hash(bytes), query_sha256: hash(query),
    inputs_sha256: hash(JSON.stringify({ assets: manifest.assets, suite: manifest.suite, methods: manifest.methods })) };
  changeMetadata?.(metadata);
  const calls = [], deadlines = [];
  const response = (body) => ({ ok: true, arrayBuffer: async () => buffer(body) });
  const context = vm.createContext({ crypto: crypto.webcrypto, TextEncoder, TextDecoder,
    URL, AbortController, BenchmarkTimingSource: metadata,
    setTimeout(callback, delay) { deadlines.push(delay); return setTimeout(callback, 15); },
    clearTimeout,
    fetch: async (url, options) => {
      calls.push(String(url));
      if (fetcher) return fetcher(String(url), options, { bytes, query, response });
      return response(String(url).endsWith("query.js") ? query : bytes);
    },
  });
  vm.runInContext(source, context);
  const lookup = (selected = manifest, options = {}) => context.BenchmarkTiming.lookup(
    selected, "selector", "seconds", { baseURL: new URL("https://example.test/report/"), ...options },
  );
  return { manifest, bytes, series, context, calls, deadlines, lookup };
}

test("authenticated timing is reused without repeated downloads", async () => {
  const f = fixture();
  const results = await Promise.all([f.lookup(), f.lookup()]);
  results.forEach((value) => assert.deepEqual(plain(value), f.series));
  assert.deepEqual(plain(await f.lookup()), f.series);
  assert.deepEqual(f.calls, ["https://example.test/report/timing-cache.json", "https://example.test/report/query.js"]);
});

for (const field of ["snapshot_id", "assets", "suite", "methods"])
  test(`changed report ${field} rejects even an already cached result`, async () => {
    const f = fixture();
    await f.lookup();
    const changed = structuredClone(f.manifest);
    changed[field] = field === "snapshot_id" ? "c".repeat(64) : { changed: true };
    assert.equal(await f.lookup(changed), null);
    assert.equal(f.calls.length, 2);
  });

for (const changed of ["cache", "query"])
  test(`changed ${changed} bytes fail authentication`, async () => {
    const f = fixture({ fetcher(url, _options, { bytes, query, response }) {
      const body = Buffer.from(url.endsWith("query.js") ? query : bytes);
      if (url.endsWith("query.js") === (changed === "query")) body[0] ^= 1;
      return response(body);
    } });
    assert.equal(await f.lookup(), null);
  });

for (const field of ["schema_version", "snapshot_id", "query_sha256"])
  test(`authenticated cache with wrong ${field} is rejected`, async () => {
    const f = fixture({ changeCache(cache) { cache[field] = field === "schema_version" ? 2 : "d".repeat(64); } });
    assert.equal(await f.lookup(), null);
  });

test("missing local cache cannot borrow the hosted report cache", async () => {
  const f = fixture();
  await f.lookup();
  assert.equal(await f.lookup(f.manifest, { files: new Map() }), null);
  assert.equal(f.calls.length, 2);
});

test("local cache authenticates its own bytes and the reducer", async () => {
  const f = fixture();
  const files = new Map([["timing-cache.json", { arrayBuffer: async () => buffer(f.bytes) }]]);
  assert.deepEqual(plain(await f.lookup(f.manifest, { files })), f.series);
  assert.deepEqual(f.calls, ["https://example.test/report/query.js"]);
});

test("optional fetch timeout aborts and returns fallback instead of hanging", async () => {
  let aborted = false;
  const f = fixture({ fetcher(_url, { signal }) {
    return new Promise((_resolve, reject) => signal.addEventListener("abort", () => {
      aborted = true;
      reject(signal.reason);
    }, { once: true }));
  } });
  assert.equal(await f.lookup(), null);
  assert(aborted);
  assert.deepEqual(f.deadlines, [5000]);
});

test("unknown selector or metric falls back without fabricated curves", async () => {
  const f = fixture();
  const lookup = f.context.BenchmarkTiming.lookup;
  const options = { baseURL: new URL("https://example.test/report/") };
  assert.equal(await lookup(f.manifest, "custom", "seconds", options), null);
  assert.equal(await lookup(f.manifest, "selector", "estimated_uncached_seconds", options), null);
});
