"use strict";
// Optional, authenticated curves produced by the same full query reducer.
globalThis.BenchmarkTiming = (() => {
  let cachedSource, pending;
  const digest = async (bytes) => Array.from(new Uint8Array(
    await crypto.subtle.digest("SHA-256", bytes),
  ), (x) => x.toString(16).padStart(2, "0")).join("");
  async function lookup(manifest, selector, metric, { files, baseURL } = {}) {
    const metadata = globalThis.BenchmarkTimingSource;
    if (!metadata || metadata.snapshot_id !== manifest.snapshot_id ||
        !(metadata.bytes > 0 && metadata.bytes <= 16 * 1024 * 1024)) return null;
    const inputs = JSON.stringify({ assets: manifest.assets, suite: manifest.suite, methods: manifest.methods });
    if (await digest(new TextEncoder().encode(inputs)) !== metadata.inputs_sha256) return null;
    const source = files || String(baseURL || "./");
    if (source !== cachedSource) {
      cachedSource = source;
      pending = (async () => {
        const file = files?.get("timing-cache.json");
        if (files && !file) return null;
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 5000);
        try {
          const options = { signal: controller.signal };
          const response = file || await fetch(new URL("timing-cache.json", baseURL), options);
          if (!file && !response.ok) return null;
          const bytes = await response.arrayBuffer();
          if (bytes.byteLength !== metadata.bytes || await digest(bytes) !== metadata.sha256) return null;
          const query = await fetch(new URL("query.js", baseURL), options);
          if (!query.ok || await digest(await query.arrayBuffer()) !== metadata.query_sha256) return null;
          const cache = JSON.parse(new TextDecoder().decode(bytes));
          return cache.schema_version === 1 && cache.snapshot_id === metadata.snapshot_id &&
            cache.query_sha256 === metadata.query_sha256 ? cache : null;
        } finally {
          clearTimeout(timeout);
        }
      })().catch(() => null);
    }
    const cache = await pending;
    const series = cache?.entries?.[selector]?.[metric];
    return Array.isArray(series) ? series : null;
  }
  return Object.freeze({ lookup });
})();
