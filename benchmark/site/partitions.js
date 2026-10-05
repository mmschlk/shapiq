"use strict";
// Shared by the page and its worker. A block stays columnar until a row is requested.
globalThis.BenchmarkPartitions = (() => {
  const kinds = new Set(["raw", "metrics", "games", "details", "runs", "profiles", "summaries"]);
  const maximumBytes = 64 * 1024 * 1024;
  const require = (valid, message) => {
    if (!valid) throw Error(`Invalid benchmark block: ${message}.`);
  };

  function columns(payload) {
    require(payload.codec === "columns-v2", "codec");
    require(Number.isSafeInteger(payload.count) && payload.count >= 0, "row count");
    require(payload.columns && typeof payload.columns === "object" && !Array.isArray(payload.columns), "columns");
    const fields = Object.keys(payload.columns), missing = new Map();
    require(payload.count === 0 || fields.length > 0, "positive row count without columns");
    for (const field of fields) {
      const column = payload.columns[field];
      require(column && Array.isArray(column.values) && column.values.length === payload.count, "column length");
      const absent = column.missing ?? [];
      require(Array.isArray(absent) && absent.every((i) => Number.isSafeInteger(i) && i >= 0 && i < payload.count), "missing positions");
      const positions = new Set(absent);
      require(positions.size === absent.length, "repeated missing position");
      missing.set(field, positions);
      if (Object.hasOwn(column, "dictionary")) {
        require(Array.isArray(column.dictionary), "dictionary");
        column.values.forEach((value, i) => {
          if (positions.has(i) || value === null) return;
          require(Number.isSafeInteger(value) && value >= 0 && value < column.dictionary.length, "dictionary reference");
        });
      }
    }
    const index = (i) => require(Number.isSafeInteger(i) && i >= 0 && i < payload.count, "row position");
    const has = (field, i) => {
      index(i);
      return missing.has(field) && !missing.get(field).has(i);
    };
    const get = (field, i) => {
      if (!has(field, i)) return undefined;
      const column = payload.columns[field], value = column.values[i];
      return Object.hasOwn(column, "dictionary") && value !== null ? column.dictionary[value] : value;
    };
    return {
      count: payload.count, fields: Object.freeze(fields), has, get,
      row(i) {
        index(i);
        return Object.fromEntries(fields.filter((field) => has(field, i)).map((field) => [field, get(field, i)]));
      },
    };
  }

  async function bytesFor(descriptor, { files, signal, baseURL }) {
    signal?.throwIfAborted();
    if (files) {
      const file = files.get(descriptor.file);
      require(file && file.size === descriptor.bytes, "missing file or byte size");
      return new Uint8Array(await file.arrayBuffer());
    }
    const response = await fetch(baseURL ? new URL(descriptor.file, baseURL) : descriptor.file, { signal });
    require(response.ok && response.body, "download failed");
    const reader = response.body.getReader();
    const bytes = new Uint8Array(descriptor.bytes);
    let offset = 0;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        require(offset + value.byteLength <= bytes.byteLength, "download exceeds declared size");
        bytes.set(value, offset);
        offset += value.byteLength;
      }
      require(offset === bytes.byteLength, "truncated download");
      return bytes;
    } finally {
      await reader.cancel();
      reader.releaseLock();
    }
  }

  async function read(manifest, descriptor, options = {}) {
    const cap = options.maxBytes ?? maximumBytes;
    require(Number.isSafeInteger(cap) && cap > 0 && cap <= maximumBytes, "byte limit");
    require(manifest.layout === "partitioned-v1" && manifest.schema_version === 1, "manifest layout");
    require(typeof manifest.snapshot_id === "string" && /^[a-f0-9]{64}$/.test(manifest.snapshot_id), "snapshot identity");
    require(kinds.has(descriptor.kind), "kind");
    require(new RegExp(`^partition-${descriptor.kind}-[0-9]+\\.json$`).test(descriptor.file), "filename");
    require(descriptor.snapshot_id === manifest.snapshot_id, "descriptor snapshot");
    require(typeof descriptor.sha256 === "string" && /^[a-f0-9]{64}$/.test(descriptor.sha256), "checksum");
    require(Number.isSafeInteger(descriptor.bytes) && descriptor.bytes > 0 && descriptor.bytes <= cap, "byte size");
    require(Number.isSafeInteger(descriptor.count) && descriptor.count >= 0, "descriptor count");
    const bytes = await bytesFor(descriptor, options);
    options.signal?.throwIfAborted();
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    const hash = Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
    require(hash === descriptor.sha256, "checksum mismatch");
    const payload = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes), (_key, value) => {
      require(typeof value !== "number" || Number.isFinite(value), "nonfinite number");
      return value;
    });
    require(payload && typeof payload === "object", "payload");
    for (const key of ["snapshot_id", "kind", "count", "target", "method", "family"])
      require(payload[key] === descriptor[key], `mismatched ${key}`);
    for (const key of ["objects", "selectors"]) {
      require(JSON.stringify(payload[key]) === JSON.stringify(descriptor[key]), `mismatched ${key}`);
      if (Object.hasOwn(payload, key)) require(Array.isArray(payload[key]), `invalid ${key}`);
    }
    options.signal?.throwIfAborted();
    return columns(payload);
  }

  return Object.freeze({ read });
})();
