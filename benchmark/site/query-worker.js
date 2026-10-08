"use strict";
importScripts("partitions.js", "partition-details.js", "query.js", "timing-cache-metadata.js", "timing-cache.js");

let controller;
const metadataCache = new Map();
let metadataBytes = 0;
const metadataLimit = 32 * 1024 * 1024;
self.onmessage = async ({ data: message }) => {
  controller?.abort();
  controller = new AbortController();
  const active = controller,
    signal = active.signal;
  try {
    if (!message || typeof message !== "object")
      throw Error("Invalid benchmark worker request.");
    if (message.type === "cancel") return;
    if (message.type !== "query")
      throw Error("Unknown benchmark worker request.");
    const { manifest, request, files } = message;
    const pendingReads = new Map();
    const read = async (descriptor) => {
      signal.throwIfAborted();
      const cacheable = ["games", "summaries"].includes(descriptor.kind) &&
        descriptor.bytes <= metadataLimit;
      const key = `${manifest.snapshot_id}:${JSON.stringify(descriptor)}`;
      if (cacheable && metadataCache.has(key)) return metadataCache.get(key).block;
      if (cacheable && pendingReads.has(key)) return pendingReads.get(key);
      const promise = BenchmarkPartitions.read(manifest, descriptor, {
        files, signal, baseURL: new URL("./", self.location.href),
      }).then((block) => {
        signal.throwIfAborted();
        if (cacheable) {
          while (metadataBytes + descriptor.bytes > metadataLimit) {
            const oldest = metadataCache.keys().next().value;
            metadataBytes -= metadataCache.get(oldest).bytes;
            metadataCache.delete(oldest);
          }
          metadataCache.set(key, { block, bytes: descriptor.bytes });
          metadataBytes += descriptor.bytes;
        }
        return block;
      }).finally(() => pendingReads.delete(key));
      if (cacheable) pendingReads.set(key, promise);
      return promise;
    };
    const result = await BenchmarkQuery.query(manifest, request, {
      read,
      signal,
      preferPresets: !request.load_details,
      lookupTiming: (selector, metric) => BenchmarkTiming.lookup(
        manifest, selector, metric, { files, baseURL: new URL("./", self.location.href) },
      ),
      lookupPreset: ({ sha256 }) =>
        BenchmarkDetails.preset(manifest, sha256, { read, signal }),
    });
    if (!signal.aborted) self.postMessage({ id: message.id, result });
  } catch (error) {
    if (!signal.aborted)
      self.postMessage({ id: message?.id, error: error.message });
  }
};
