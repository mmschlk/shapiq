"use strict";
// Columns preserve every float and distinguish an absent field from explicit null.
window.BenchmarkRecords = {
  decode(payload) {
    if (
      payload.codec !== "columns-v1" ||
      !Number.isSafeInteger(payload.count) ||
      payload.count < 0
    )
      throw Error("Unknown benchmark record encoding.");
    const rows = Array.from({ length: payload.count }, () => ({}));
    for (const [field, column] of Object.entries(payload.columns)) {
      if (!Array.isArray(column.values) || column.values.length !== rows.length)
        throw Error("Invalid benchmark column length.");
      const missing = new Set(column.missing || []);
      if (
        [...missing].some(
          (i) => !Number.isSafeInteger(i) || i < 0 || i >= rows.length,
        )
      )
        throw Error("Invalid missing-value position.");
      if (
        column.dictionary &&
        !column.dictionary.every((value) => typeof value === "string")
      )
        throw Error("Invalid string dictionary.");
      column.values.forEach((value, i) => {
        if (missing.has(i)) return;
        if (column.dictionary && value !== null) {
          if (
            !Number.isSafeInteger(value) ||
            value < 0 ||
            value >= column.dictionary.length
          )
            throw Error("Invalid string dictionary reference.");
          value = column.dictionary[value];
        }
        Object.defineProperty(rows[i], field, {
          value,
          enumerable: true,
          writable: true,
        });
      });
    }
    return rows;
  },
  async read(manifest, descriptor, files) {
    if (!/^records-[a-z-]+-\d+\.json$/.test(descriptor.file))
      throw Error("Invalid benchmark shard filename.");
    let bytes;
    if (files) {
      const file = files.get(descriptor.file);
      if (!file)
        throw Error(
          `Select data.json and its companion ${descriptor.file} file together.`,
        );
      bytes = await file.arrayBuffer();
    } else {
      const response = await fetch(descriptor.file);
      if (!response.ok)
        throw Error("The selected explanation data could not be loaded.");
      bytes = await response.arrayBuffer();
    }
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    const hash = [...new Uint8Array(digest)]
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
    if (hash !== descriptor.sha256)
      throw Error("Benchmark shard checksum mismatch.");
    const payload = JSON.parse(new TextDecoder().decode(bytes));
    if (
      payload.snapshot_id !== manifest.snapshot_id ||
      payload.target !== descriptor.target ||
      payload.count !== descriptor.count
    )
      throw Error(
        "Benchmark shard belongs to a different report or explanation.",
      );
    return { records: this.decode(payload), presets: payload.presets };
  },
};
