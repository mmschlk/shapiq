const assert = require("node:assert/strict");
const { test } = require("node:test");
require("../../benchmark/site/partition-details.js");
const { Assembly } = BenchmarkDetails;
const row = (path, fields) => ({
  type: "report",
  id: "suite",
  fragment: { path, ...fields },
});
const leaf = (path, value) => ({ ...row(path, {}), value });
function complete(rows) {
  const a = new Assembly();
  rows.forEach((r) => a.push(r));
  return a.finish();
}
const fixture = () => [
  row([], { kind: "object", length: 3 }),
  row(["budgets"], { kind: "array", length: 3 }),
  leaf(["budgets", 0], null),
  leaf(["budgets", 1], -0),
  leaf(["budgets", 2], 11),
  row(["__proto__"], { kind: "object", length: 1 }),
  leaf(["__proto__", "safe"], "own property"),
  leaf(["nested"], { unicode: "🦋 é", missing: null }),
];

test("assembly rejects aggregate limits without returning a partial object", () => {
  const fragments = new Assembly({ maxFragments: 1 });
  fragments.push(fixture()[0]);
  assert.throws(() => fragments.push(fixture()[1]), /assembly limits/);
  assert.throws(() => fragments.finish(), /missing children/);
  const characters = new Assembly({ maxCharacters: 4 });
  assert.throws(
    () => characters.push({ id: "x", value: "long" }),
    /assembly limits/,
  );
  assert.throws(
    () => new Assembly({ maxFragments: Infinity }),
    /assembly limits/,
  );
});

test("fragments preserve nested values, signed zero and own prototype-like keys", () => {
  const value = complete(fixture());
  assert.deepEqual(value.budgets, [null, -0, 11]);
  assert.equal(Object.getPrototypeOf(value), Object.prototype);
  assert.ok(Object.hasOwn(value, "__proto__"));
  assert.equal(value.__proto__.safe, "own property");
  assert.equal({}.safe, undefined);
  assert.deepEqual(value.nested, { unicode: "🦋 é", missing: null });
});
test("small whole values retain null and empty containers", () => {
  for (const value of [null, {}, [], 1, "x"])
    assert.deepEqual(complete([{ id: "x", value }]), value);
  assert.deepEqual(complete([row([], { kind: "array", length: 0 })]), []);
});
for (const [name, mutate] of [
  ["missing child", (rows) => rows.pop()],
  ["duplicate child", (rows) => rows.push(rows[2])],
  ["child before root", (rows) => rows.shift()],
  ["child before parent", (rows) => rows.splice(1, 1)],
  ["outside array", (rows) => (rows[2].fragment.path[1] = 4)],
  ["string array index", (rows) => (rows[2].fragment.path[1] = "0")],
  ["numeric object key", (rows) => (rows[1].fragment.path[0] = 1)],
  ["invalid container", (rows) => (rows[0].fragment.kind = "map")],
  ["invalid length", (rows) => (rows[0].fragment.length = -1)],
  ["mixed identities", (rows) => (rows[2].id = "other")],
  ["repeated root", (rows) => rows.push(rows[0])],
  ["container and leaf", (rows) => (rows[0].value = 3)],
  [
    "whole and fragmented",
    (rows) => rows.unshift({ type: "report", id: "suite", value: {} }),
  ],
])
  test(`reject ${name}`, () => {
    const rows = fixture();
    mutate(rows);
    assert.throws(() => complete(rows), /Invalid benchmark details/);
  });

function block(rows) {
  return {
    count: rows.length,
    get: (key, i) => rows[i][key],
    row: (i) => rows[i],
  };
}
test("object lookup reads only matching blocks and restores fragments across boundaries", async () => {
  const rows = fixture(),
    called = [];
  const descriptors = [
    { id: 0, objects: [["game", "other"]] },
    ...[1, 2].map((id) => ({ id, objects: [["report", "suite"]] })),
  ];
  const value = await BenchmarkDetails.object(
    { assets: { details: descriptors } },
    { type: "report", id: "suite" },
    {
      read: async (descriptor) => {
        called.push(descriptor.id);
        return block(descriptor.id === 1 ? rows.slice(0, 4) : rows.slice(4));
      },
    },
  );
  assert.deepEqual(called, [1, 2]);
  assert.deepEqual(value, complete(rows));
});
test("preset lookup retains first equivalent match and skips other selectors", async () => {
  const hash = "a".repeat(64),
    called = [];
  const manifest = {
    assets: {
      summaries: [
        { id: 0, selectors: ["b".repeat(64)] },
        { id: 1, selectors: [hash] },
      ],
    },
  };
  const options = {
    read: async (d) => {
      called.push(d.id);
      return block([
        {
          id: "first",
          selector_sha256: hash,
          value: { id: "first", score: -0 },
        },
        {
          id: "second",
          selector_sha256: hash,
          value: { id: "second", score: 1 },
        },
      ]);
    },
  };
  assert.deepEqual(await BenchmarkDetails.preset(manifest, hash, options), {
    id: "first",
    score: -0,
  });
  assert.deepEqual(called, [1]);
  assert.equal(
    await BenchmarkDetails.preset(manifest, "c".repeat(64), options),
    null,
  );
});
test("cancellation discards a completed local read", async () => {
  const controller = new AbortController();
  const manifest = {
    assets: { details: [{ objects: [["report", "suite"]] }] },
  };
  await assert.rejects(
    BenchmarkDetails.object(
      manifest,
      { type: "report", id: "suite" },
      {
        signal: controller.signal,
        read: async () => {
          controller.abort();
          return block(fixture());
        },
      },
    ),
    { name: "AbortError" },
  );
});
