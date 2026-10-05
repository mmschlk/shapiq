"use strict";
// Keep the fields displayed by protocol.js, releasing each full game's metadata immediately.
globalThis.BenchmarkAbout = (() => {
  const fields = [
    "game_kind",
    "recipe",
    "truth_method",
    "case_id",
    "class",
    "model_profile",
    "model",
    "semantics",
    "player_unit",
    "instance_seed",
    "dataset",
    "dataset_source",
    "dataset_source_url",
    "dataset_target_note",
    "dataset_preprocessing",
    "model_parameters",
    "output_scale",
    "preparation_hardware",
  ];
  function project(game) {
    const m = game.metadata || {},
      metadata = {};
    for (const key of fields) if (Object.hasOwn(m, key)) metadata[key] = m[key];
    if (m.fourier_spectrum)
      metadata.fourier_spectrum = {
        degree_mass: m.fourier_spectrum.degree_mass,
      };
    for (const [count, indices] of [
      ["training_rows", "train_indices"],
      ["validation_rows", "validation_indices"],
      ["test_rows", "test_indices"],
      ["background_size", "background_indices"],
    ]) {
      const value = m[count] ?? m[indices]?.length;
      if (value !== undefined) metadata[count] = value;
    }
    return {
      id: game.id,
      index: game.index,
      order: game.order,
      family: game.family,
      n_players: game.n_players,
      metadata,
    };
  }
  async function load(
    manifest,
    { read, signal, maxCharacters = 32 * 1024 * 1024, maxGames = 250000 } = {},
  ) {
    const require = (condition, message) => {
      if (!condition)
        throw Error(`Invalid benchmark About report: ${message}.`);
    };
    require(Number.isSafeInteger(maxCharacters) &&
      maxCharacters > 0 &&
      Number.isSafeInteger(maxGames) &&
      maxGames > 0, "metadata limits");
    const result = {
      schema_version: manifest.schema_version,
      snapshot_id: manifest.snapshot_id,
      games: [],
    };
    const wanted = (type, id) =>
      type === "game" ||
      (type === "report" && ["suite", "snapshot_provenance"].includes(id));
    const seen = new Set();
    let active,
      identity,
      remaining = maxCharacters;
    const finish = () => {
      if (!active) return;
      const value = active.finish(),
        [type, id] = identity;
      require(!seen.has(JSON.stringify(identity)), "repeated metadata object");
      seen.add(JSON.stringify(identity));
      const kept = type === "game" ? project(value) : value;
      remaining -= JSON.stringify(kept).length;
      require(remaining >= 0, "displayed metadata exceeds its size limit");
      if (type === "game") {
        require(value.id === id &&
          result.games.length < maxGames, "game identity or count");
        result.games.push(kept);
      } else
        Object.defineProperty(result, id, { value: kept, enumerable: true });
      active = null;
    };
    for (const descriptor of manifest.assets.details) {
      if (!descriptor.objects.some(([type, id]) => wanted(type, id))) continue;
      signal?.throwIfAborted();
      const block = await read(descriptor);
      for (let i = 0; i < block.count; i++) {
        signal?.throwIfAborted();
        const type = block.get("type", i),
          id = block.get("id", i);
        if (!wanted(type, id)) continue;
        if (!identity || identity[0] !== type || identity[1] !== id) {
          finish();
          identity = [type, id];
          active = new BenchmarkDetails.Assembly();
        }
        active.push(block.row(i));
      }
    }
    finish();
    require(result.games.length === manifest.game_count &&
      result.suite?.seeds?.length, "incomplete metadata");
    signal?.throwIfAborted();
    return result;
  }
  return { project, load };
})();
