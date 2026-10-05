"use strict";
importScripts("partitions.js", "partition-details.js", "query.js");

let controller;
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
    const read = (descriptor) =>
      BenchmarkPartitions.read(manifest, descriptor, {
        files,
        signal,
        baseURL: new URL("./", self.location.href),
      });
    const result = await BenchmarkQuery.query(manifest, request, {
      read,
      signal,
      lookupPreset: ({ sha256 }) =>
        BenchmarkDetails.preset(manifest, sha256, { read, signal }),
    });
    if (!signal.aborted) self.postMessage({ id: message.id, result });
  } catch (error) {
    if (!signal.aborted)
      self.postMessage({ id: message?.id, error: error.message });
  }
};
