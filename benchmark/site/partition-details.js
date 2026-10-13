"use strict";
// Reconstruct only the requested metadata object or preset, in authenticated block order.
globalThis.BenchmarkDetails = (() => {
  const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
  const require = (valid, message) => {
    if (!valid) throw Error(`Invalid benchmark details: ${message}.`);
  };

  class Assembly {
    constructor({
      maxFragments = 1000000,
      maxCharacters = 32 * 1024 * 1024,
    } = {}) {
      require(Number.isSafeInteger(maxFragments) &&
        maxFragments > 0 &&
        Number.isSafeInteger(maxCharacters) &&
        maxCharacters > 0, "assembly limits");
      this.remainingFragments = maxFragments;
      this.remainingCharacters = maxCharacters;
      this.started = false;
      this.containers = new Map();
    }
    push(row) {
      require(row && typeof row.id === "string", "object identity");
      this.remainingFragments--;
      this.remainingCharacters -= JSON.stringify(row).length;
      require(this.remainingFragments >= 0 &&
        this.remainingCharacters >=
          0, "object exceeds configured assembly limits");
      if (!this.started)
        this.identity = [row.type, row.id, row.selector_sha256];
      require(this.identity.every(
        (v, i) => v === [row.type, row.id, row.selector_sha256][i],
      ), "mixed objects");
      if (!own(row, "fragment")) {
        require(!this.started &&
          own(row, "value"), "repeated or absent object");
        this.started = this.whole = true;
        this.value = row.value;
        return;
      }
      require(!this.whole, "mixed whole and fragmented object");
      const fragment = row.fragment;
      require(fragment && Array.isArray(fragment.path), "fragment path");
      const path = fragment.path;
      let value;
      if (own(fragment, "kind")) {
        require(["object", "array"].includes(fragment.kind), "container kind");
        require(Number.isSafeInteger(fragment.length) &&
          fragment.length >= 0, "container length");
        require(!own(row, "value"), "container with a leaf value");
        value = fragment.kind === "array" ? [] : {};
        this.containers.set(value, { length: fragment.length, count: 0 });
      } else {
        require(own(row, "value") && !own(fragment, "length"), "leaf value");
        value = row.value;
      }
      if (!path.length) {
        require(!this.started, "repeated root");
        this.value = value;
        this.started = true;
        return;
      }
      require(this.started, "child before root");
      let parent = this.value;
      for (const key of path.slice(0, -1)) {
        this.key(parent, key);
        require(own(parent, key), "child before parent");
        parent = parent[key];
      }
      const key = path[path.length - 1];
      this.key(parent, key);
      require(!own(parent, key), "repeated child");
      Object.defineProperty(parent, key, {
        value,
        enumerable: true,
        writable: true,
        configurable: true,
      });
      const container = this.containers.get(parent);
      container.count++;
      require(container.count <= container.length, "too many children");
    }
    key(parent, key) {
      require(this.containers.has(parent), "undeclared parent container");
      require(Array.isArray(parent)
        ? Number.isSafeInteger(key) &&
            key >= 0 &&
            key < this.containers.get(parent).length &&
            key < 4294967295
        : typeof key === "string", "child key");
    }
    finish() {
      require(this.started, "object not found");
      for (const { length, count } of this.containers.values())
        require(length === count, "missing children");
      return this.value;
    }
  }

  async function object(manifest, { type, id }, { read, signal, limits }) {
    const assembly = new Assembly(limits);
    for (const descriptor of manifest.assets.details) {
      if (!descriptor.objects?.some(([t, key]) => t === type && key === id))
        continue;
      signal?.throwIfAborted();
      const block = await read(descriptor);
      for (let i = 0; i < block.count; i++)
        if (block.get("type", i) === type && block.get("id", i) === id)
          assembly.push(block.row(i));
    }
    signal?.throwIfAborted();
    return assembly.finish();
  }

  async function preset(manifest, sha256, { read, signal, limits }) {
    require(/^[a-f0-9]{64}$/.test(sha256), "selector hash");
    let assembly = null,
      id;
    for (const descriptor of manifest.assets.summaries) {
      if (!descriptor.selectors?.includes(sha256)) continue;
      signal?.throwIfAborted();
      const block = await read(descriptor);
      for (let i = 0; i < block.count; i++) {
        if (block.get("selector_sha256", i) !== sha256) continue;
        if (!assembly) {
          assembly = new Assembly(limits);
          id = block.get("id", i);
        }
        // Multiple equivalent presets retain the legacy first-match rule.
        if (block.get("id", i) === id) assembly.push(block.row(i));
      }
    }
    signal?.throwIfAborted();
    return assembly ? assembly.finish() : null;
  }

  return Object.freeze({ Assembly, object, preset });
})();
