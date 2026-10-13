"use strict";
// One worker per open report. Scientific rows stay off the page thread.
globalThis.BenchmarkPartitionClient = class {
  constructor(manifest, files = null) {
    this.manifest = manifest;
    this.files = files;
    this.serial = 0;
    this.pending = null;
    this.failure = null;
    this.closed = false;
    this.lastKey = null;
    this.lastPromise = null;
    this.worker = new Worker("query-worker.js");
    this.worker.onmessage = ({ data }) => {
      if (!this.pending || data.id !== this.pending.id) return;
      const pending = this.pending;
      this.pending = null;
      if (data.error) {
        this.lastKey = null;
        pending.reject(Error(data.error));
      } else pending.resolve(data.result);
    };
    const failed = (error) => {
      this.failure = error;
      this.pending?.reject(error);
      this.pending = null;
      this.worker.terminate();
    };
    this.worker.onerror = (event) =>
      failed(Error(event.message || "Benchmark worker failed."));
    this.worker.onmessageerror = () =>
      failed(Error("Benchmark worker response could not be decoded."));
  }
  query(request) {
    if (this.failure) return Promise.reject(this.failure);
    if (this.closed)
      return Promise.reject(
        new DOMException("The report was closed.", "AbortError"),
      );
    const key = JSON.stringify(request);
    if (key === this.lastKey) return this.lastPromise;
    this.lastKey = key;
    this.pending?.reject(
      new DOMException("A newer selection replaced this query.", "AbortError"),
    );
    const id = ++this.serial;
    this.lastPromise = new Promise((resolve, reject) => {
      this.pending = { id, resolve, reject };
      this.worker.postMessage({
        type: "query",
        id,
        manifest: this.manifest,
        files: this.files,
        request,
      });
    });
    return this.lastPromise;
  }
  cancel() {
    if (!this.pending) return;
    this.pending.reject(
      new DOMException("The selection was canceled.", "AbortError"),
    );
    this.pending = null;
    this.lastKey = null;
    this.lastPromise = null;
    this.worker.postMessage({ type: "cancel" });
  }
  close() {
    this.closed = true;
    this.pending?.reject(
      new DOMException("The report was closed.", "AbortError"),
    );
    this.pending = null;
    this.worker.terminate();
  }
};
