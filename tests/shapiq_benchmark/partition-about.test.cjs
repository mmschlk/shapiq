const { test } = require("node:test");
const assert = require("node:assert/strict");
require("../../benchmark/site/partition-details.js");
require("../../benchmark/site/partition-about.js");
const game = {
  id: "g",
  index: "SV",
  order: 1,
  family: "local",
  n_players: 11,
  metadata: {
    dataset: "test",
    model: "forest",
    class: "shapiq.game.Foo",
    train_indices: [1, 2, 4],
    training_rows: null,
    validation_rows: 4,
    validation_indices: [2],
    fourier_spectrum: {
      degree_mass: [0, 0.1, 0.2, 0.3, 0.4],
      large: [1, 2, 3],
    },
    model_parameters: { max_depth: null },
    huge_private_unused_array: Array(100).fill(1),
  },
};
const rows = [
  { type: "game", id: "g", value: game },
  {
    type: "report",
    id: "suite",
    value: { seeds: [0], game_seeds: [0, 1, 2, 3] },
  },
  {
    type: "report",
    id: "snapshot_provenance",
    value: { git_commit: "a".repeat(40) },
  },
];
function fixture(rowsValue = rows) {
  const descriptor = {
    objects: [
      ["game", "g"],
      ["report", "suite"],
      ["report", "snapshot_provenance"],
    ],
  };
  return {
    manifest: {
      schema_version: 1,
      snapshot_id: "x",
      game_count: 1,
      assets: { details: [descriptor] },
    },
    read: async () => ({
      count: rowsValue.length,
      row: (i) => rowsValue[i],
      get: (key, i) => rowsValue[i][key],
    }),
  };
}
test("project keeps rendered fields, preserves Fourier masses and normalizes recorded row counts", () => {
  const projected = BenchmarkAbout.project(game);
  assert.equal(projected.metadata.training_rows, 3);
  assert.equal(projected.metadata.validation_rows, 4);
  assert.deepEqual(projected.metadata.fourier_spectrum, {
    degree_mass: [0, 0.1, 0.2, 0.3, 0.4],
  });
  assert.deepEqual(projected.metadata.model_parameters, { max_depth: null });
  assert.equal(projected.metadata.huge_private_unused_array, undefined);
  assert.equal(projected.metadata.train_indices, undefined);
});
test("read details once and retain only the projected games plus report settings", async () => {
  const f = fixture();
  let reads = 0;
  const result = await BenchmarkAbout.load(f.manifest, {
    read: async (d) => {
      reads++;
      return f.read(d);
    },
  });
  assert.equal(reads, 1);
  assert.deepEqual(result.games, [BenchmarkAbout.project(game)]);
  assert.deepEqual(result.suite, rows[1].value);
  assert.deepEqual(result.snapshot_provenance, rows[2].value);
});
test("metadata cap and incomplete or repeated games fail without a partial report", async () => {
  let f = fixture();
  await assert.rejects(
    BenchmarkAbout.load(f.manifest, { read: f.read, maxCharacters: 2 }),
    /size limit/,
  );
  f = fixture(rows.slice(1));
  await assert.rejects(
    BenchmarkAbout.load(f.manifest, { read: f.read }),
    /incomplete/,
  );
  f = fixture([...rows, rows[0]]);
  await assert.rejects(
    BenchmarkAbout.load(f.manifest, { read: f.read }),
    /repeated/,
  );
});
test("canceled metadata does not complete or read unrelated detail blocks", async () => {
  const f = fixture(),
    controller = new AbortController();
  controller.abort();
  await assert.rejects(
    BenchmarkAbout.load(f.manifest, {
      read: f.read,
      signal: controller.signal,
    }),
    { name: "AbortError" },
  );
  f.manifest.assets.details.unshift({ objects: [["report", "composition"]] });
  let count = 0;
  await BenchmarkAbout.load(f.manifest, {
    read: async (d) => {
      assert.notDeepEqual(d.objects, [["report", "composition"]]);
      count++;
      return f.read(d);
    },
  });
  assert.equal(count, 1);
});
