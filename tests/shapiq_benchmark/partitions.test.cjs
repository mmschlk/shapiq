// node --test tests/shapiq_benchmark/partitions.test.cjs
const assert = require("node:assert/strict");
const { test } = require("node:test");
const { createHash, webcrypto } = require("node:crypto");
globalThis.crypto = webcrypto;
require("../../benchmark/site/partitions.js");

const snapshot = "a".repeat(64);
function fixture(change = () => {}, text) {
  const payload = {
    snapshot_id: snapshot, kind: "metrics", target: "SV · order 1",
    method: "KernelSHAP", family: "local", codec: "columns-v2", count: 3,
    columns: {
      score: { values: [null, null, -0], missing: [0] },
      method: { values: [0, 0, 0], dictionary: ["KernelSHAP"] },
      nested: { values: [0, 1, 0], dictionary: [{ a: 1 }, [null, "x"]] },
    },
  };
  change(payload);
  // JSON.stringify erases signed zero; Python's writer emits -0.0 faithfully.
  const bytes = Buffer.from(text ?? JSON.stringify(payload).replace('[null,null,0]', '[null,null,-0.0]'));
  const descriptor = {
    file: "partition-metrics-0.json", sha256: createHash("sha256").update(bytes).digest("hex"),
    bytes: bytes.length, count: payload.count, kind: payload.kind, snapshot_id: snapshot,
    target: payload.target, method: payload.method, family: payload.family,
  };
  return {
    payload, descriptor, bytes,
    manifest: { schema_version: 1, layout: "partitioned-v1", snapshot_id: snapshot },
    options: { files: new Map([[descriptor.file, new Blob([bytes])]]) },
  };
}
const read = (f) => BenchmarkPartitions.read(f.manifest, f.descriptor, f.options);

test("column access preserves absent/null/signed zero/nested values without expanding rows", async () => {
  const f = fixture(), block = await read(f);
  assert.equal(block.count, 3);
  assert.equal(block.has("score", 0), false);
  assert.equal(block.get("score", 0), undefined);
  assert.equal(block.has("score", 1), true);
  assert.equal(block.get("score", 1), null);
  assert.ok(Object.is(block.get("score", 2), -0));
  assert.deepEqual(block.get("nested", 1), [null, "x"]);
  assert.equal(block.get("method", 2), "KernelSHAP");
  assert.equal(Object.hasOwn(block.row(0), "score"), false);
  assert.equal(block.records, undefined);
  assert.throws(() => block.row(3), /row position/);
  assert.throws(() => block.get("score", -1), /row position/);
});

for (const [name, mutate] of [
  ["length", (p) => p.columns.score.values.pop()],
  ["dictionary index", (p) => p.columns.method.values[0] = 1],
  ["dictionary type", (p) => p.columns.method.dictionary = {}],
  ["missing outside range", (p) => p.columns.score.missing = [3]],
  ["repeated missing", (p) => p.columns.score.missing = [0, 0]],
  ["unknown codec", (p) => p.codec = "columns-v9"],
  ["noninteger count", (p) => p.count = 3.5],
  ["missing columns", (p) => delete p.columns],
  ["unbounded empty rows", (p) => { p.count = Number.MAX_SAFE_INTEGER; p.columns = {}; }],
  ["other snapshot", (p) => p.snapshot_id = "b".repeat(64)],
]) test(`reject authenticated malformed ${name}`, async () => {
  await assert.rejects(read(fixture(mutate)), /Invalid benchmark block/);
});

for (const [name, mutate] of [
  ["path traversal", (f) => f.descriptor.file = "../partition-metrics-0.json"],
  ["wrong kind filename", (f) => f.descriptor.file = "partition-raw-0.json"],
  ["checksum", (f) => f.descriptor.sha256 = "f".repeat(64)],
  ["byte size", (f) => f.descriptor.bytes++],
  ["row count", (f) => f.descriptor.count++],
  ["target", (f) => f.descriptor.target = "SII · order 2"],
  ["method", (f) => f.descriptor.method = "OddSHAP"],
  ["family", (f) => f.descriptor.family = "other"],
  ["unsupported layout", (f) => f.manifest.layout = "v0"],
  ["oversize block", (f) => f.options.maxBytes = 10],
]) test(`reject ${name}`, async () => {
  const f = fixture(); mutate(f);
  await assert.rejects(read(f), /Invalid benchmark block/);
});

test("numeric overflow rejects even when checksum is valid", async () => {
  const f = fixture();
  await assert.rejects(read(fixture(() => {}, f.bytes.toString().replace("-0.0", "1e999"))), /nonfinite/);
});

test("aborted local reads do not deliver a stale result", async () => {
  const f = fixture(), controller = new AbortController();
  controller.abort(); f.options.signal = controller.signal;
  await assert.rejects(read(f), { name: "AbortError" });
});

test("network reads enforce size as bytes arrive and bind contents", async () => {
  const previous = globalThis.fetch, f = fixture();
  f.options = { baseURL: "https://example.test/benchmark/" };
  let received;
  try {
    globalThis.fetch = async (url) => { received = String(url); return new Response(f.bytes); };
    assert.equal((await read(f)).count, 3);
    assert.equal(received, "https://example.test/benchmark/partition-metrics-0.json");
    globalThis.fetch = async () => new Response(Buffer.concat([f.bytes, Buffer.from("x")]));
    await assert.rejects(read(f), /exceeds declared size/);
    globalThis.fetch = async () => new Response(f.bytes.subarray(1));
    await assert.rejects(read(f), /truncated/);
    globalThis.fetch = async () => new Response("error", { status: 404 });
    await assert.rejects(read(f), /download failed/);
  } finally { globalThis.fetch = previous; }
});
