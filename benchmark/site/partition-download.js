"use strict";
// Selected rows spill to a disposable browser database; sink writes provide backpressure.
globalThis.BenchmarkDownloads = (() => {
  const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
  const require = (ok, message) => {
    if (!ok) throw Error(`Benchmark download: ${message}.`);
  };
  const encoder = new TextEncoder();
  const fields = [
    "game_id",
    "method",
    "budget",
    "seed",
    "status",
    "nmse",
    "score_order",
    "mse",
    "queries",
    "seconds",
    "estimated_oracle_seconds",
    "estimated_uncached_seconds",
    "timing_profile",
    "run_id",
  ];
  const csv = (row) =>
    fields
      .map((k) => `"${String(row[k] ?? "").replaceAll('"', '""')}"`)
      .join(",");

  async function indexedSpool({ signal } = {}) {
    require(globalThis.indexedDB, "temporary browser storage is unavailable; enable local storage for this site");
    const name = `shapiq-download-${crypto.randomUUID()}`;
    signal?.throwIfAborted();
    let db,
      closed = false;
    const remove = () =>
      new Promise((resolve, reject) => {
        const request = indexedDB.deleteDatabase(name);
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
        request.onblocked = () =>
          reject(Error("Temporary download storage is blocked during cleanup"));
      });
    try {
      db = await new Promise((resolve, reject) => {
        const request = indexedDB.open(name, 1);
        let blocked = false;
        const abort = () => {
          blocked = true;
          reject(signal.reason);
        };
        signal?.addEventListener("abort", abort, { once: true });
        request.onupgradeneeded = () => {
          const store = request.result.createObjectStore("rows", {
            keyPath: "sequence",
          });
          store.createIndex("cell", "cell", { unique: true });
        };
        request.onsuccess = () => {
          signal?.removeEventListener("abort", abort);
          if (blocked) request.result.close();
          else resolve(request.result);
        };
        request.onerror = () => {
          signal?.removeEventListener("abort", abort);
          reject(request.error);
        };
        request.onblocked = () => {
          blocked = true;
          reject(Error("Temporary download storage is blocked"));
        };
      });
      signal?.throwIfAborted();
    } catch (error) {
      db?.close();
      await remove();
      throw error;
    }
    const transaction = (mode, operation) =>
      new Promise((resolve, reject) => {
        signal?.throwIfAborted();
        const tx = db.transaction("rows", mode);
        const abort = () => tx.abort();
        signal?.addEventListener("abort", abort, { once: true });
        tx.oncomplete = () => {
          signal?.removeEventListener("abort", abort);
          resolve();
        };
        tx.onabort = tx.onerror = () => {
          signal?.removeEventListener("abort", abort);
          reject(
            signal?.aborted
              ? signal.reason
              : tx.error || Error("Temporary download storage failed"),
          );
        };
        try {
          operation(tx.objectStore("rows"));
        } catch (error) {
          tx.abort();
          reject(error);
        }
      });
    return {
      add(rows) {
        return transaction("readwrite", (store) => {
          for (const row of rows) store.add(row);
        });
      },
      async *rows() {
        let after;
        while (true) {
          const page = [];
          let bytes = 0;
          await transaction("readonly", (store) => {
            const request = store.openCursor(
              after === undefined
                ? undefined
                : IDBKeyRange.lowerBound(after, true),
            );
            request.onsuccess = () => {
              const cursor = request.result;
              if (!cursor) return;
              page.push(cursor.value);
              bytes += cursor.value.bytes;
              if (page.length < 128 && bytes < 1024 * 1024) cursor.continue();
            };
          });
          if (!page.length) return;
          for (const row of page) {
            signal?.throwIfAborted();
            yield row.value;
            after = row.sequence;
          }
        }
      },
      async close() {
        if (!closed) {
          db.close();
          await remove();
          closed = true;
        }
      },
    };
  }

  async function stream(manifest, originalRequest, kind, options) {
    const signal = options.signal;
    let request;
    const sink = options.sink,
      read =
        options.read || ((d) => BenchmarkPartitions.read(manifest, d, options));
    const limits = {
      games: 250000,
      game_bytes: 64 * 1024 * 1024,
      profiles: 50000,
      profile_bytes: 32 * 1024 * 1024,
      runs: 250000,
      rows: 1000000,
      row_bytes: 16 * 1024 * 1024,
      spool_bytes: 1024 * 1024 * 1024,
      ...options.limits,
    };
    const check = () => signal?.throwIfAborted();
    let spool,
      buffer = "",
      bufferBytes = 0,
      totalBytes = 0,
      count = 0;
    const flush = async () => {
      if (!buffer) return;
      check();
      await sink.write(buffer);
      check();
      totalBytes += bufferBytes;
      buffer = "";
      bufferBytes = 0;
    };
    const write = async (text) => {
      check();
      const bytes = encoder.encode(text).byteLength;
      if (bufferBytes + bytes > 256 * 1024) await flush();
      if (bytes > 256 * 1024) {
        await sink.write(text);
        check();
        totalBytes += bytes;
      } else {
        buffer += text;
        bufferBytes += bytes;
      }
    };
    // Metadata can span many blocks; serialize its reconstructed object without a second huge string.
    const json = async (value) => {
      if (value === null || typeof value !== "object") {
        await write(JSON.stringify(value));
        return;
      }
      const array = Array.isArray(value);
      await write(array ? "[" : "{");
      let first = true;
      for (const key of Object.keys(value)) {
        if (!array && value[key] === undefined) continue;
        if (!first) await write(",");
        first = false;
        if (!array) await write(`${JSON.stringify(key)}:`);
        await json(value[key] === undefined ? null : value[key]);
      }
      await write(array ? "]" : "}");
    };
    try {
      require(["json", "csv"].includes(kind), "unknown format");
      require(Object.values(limits).every(
        (n) => Number.isSafeInteger(n) && n > 0,
      ), "invalid limits");
      request = structuredClone(originalRequest);
      check();
      require(request.methods.length === new Set(request.methods).size &&
        request.methods.every((m) =>
          own(manifest.methods, m),
        ), "unknown or repeated methods");
      const games = [],
        profiles = new Map();
      let gameBytes = 0;
      for (const descriptor of manifest.assets.games) {
        if (descriptor.target !== request.selection.target) continue;
        check();
        const block = await read(descriptor);
        check();
        for (let i = 0; i < block.count; i++) {
          const game = block.row(i);
          require(own(
            game,
            "row_zero_truth_energy",
          ), "game index lacks global truth flags");
          require(Number.isSafeInteger(game.sequence) &&
            game.sequence >= 0, "invalid game sequence");
          gameBytes += encoder.encode(JSON.stringify(game)).byteLength;
          games.push(game);
          require(games.length <= limits.games &&
            gameBytes <=
              limits.game_bytes, "game index exceeds the memory limit");
        }
      }
      games.sort((a, b) => a.sequence - b.sequence);
      require(new Set(games.map((g) => g.id)).size ===
        games.length, "repeated game");
      const panel = BenchmarkQuery.selectionPanel(
        games,
        request.selection,
        manifest.suite,
        request.score_order,
      );
      const families = new Set(panel.games.map((g) => g.family)),
        methods = new Set(request.methods);
      let profileBytes = 0;
      for (const descriptor of manifest.assets.profiles) {
        check();
        const block = await read(descriptor);
        check();
        for (let i = 0; i < block.count; i++) {
          const row = block.row(i);
          require(!profiles.has(row.id), "repeated profile");
          profiles.set(row.id, row);
          profileBytes += encoder.encode(JSON.stringify(row)).byteLength;
          require(profiles.size <= limits.profiles &&
            profileBytes <=
              limits.profile_bytes, "profiles exceed the memory limit");
        }
      }
      const runIds = new Set();
      for (const descriptor of manifest.assets.runs) {
        check();
        const block = await read(descriptor);
        check();
        for (let i = 0; i < block.count; i++) {
          const row = block.row(i);
          require(typeof row.id === "string" &&
            !runIds.has(row.id), "invalid or repeated run");
          require(profiles.get(row.source_id)?.type ===
            "source", "missing run source");
          runIds.add(row.id);
          require(runIds.size <=
            limits.runs, "run index exceeds the memory limit");
        }
      }
      spool = await (options.createSpool || indexedSpool)({ signal });
      let batch = [],
        batchBytes = 0,
        spoolBytes = 0;
      for (const descriptor of manifest.assets.raw) {
        if (
          descriptor.target !== request.selection.target ||
          !methods.has(descriptor.method) ||
          !families.has(descriptor.family)
        )
          continue;
        check();
        const block = await read(descriptor);
        check();
        for (let i = 0; i < block.count; i++) {
          const game = block.get("game_id", i),
            budget = block.get("budget", i),
            seed = block.get("seed", i);
          if (!panel.grids.get(game)?.has(budget) || !panel.seeds.has(seed))
            continue;
          const row = block.row(i),
            sequence = row.sequence;
          require(Number.isSafeInteger(sequence) &&
            sequence >= 0 &&
            row.method === descriptor.method &&
            methods.has(row.method), "invalid row identity");
          require(runIds.has(row.run_id), "missing row provenance");
          delete row.sequence;
          if (own(row, "worker_id")) {
            const profile = profiles.get(row.worker_id);
            require(profile?.type === "worker", "missing worker profile");
            row.worker = { ...profile.value };
            delete row.worker_id;
            for (const field of ["peak_rss_bytes", "process_cpu_seconds"])
              if (own(row, `worker_${field}`)) {
                row.worker[field] = row[`worker_${field}`];
                delete row[`worker_${field}`];
              }
          }
          if (request.score_order) {
            row.nmse = row.order_scores?.[request.score_order]?.nmse ?? null;
            row.mse = row.order_scores?.[request.score_order]?.mse ?? null;
            row.score_order = Number(request.score_order);
          }
          const value = kind === "json" ? JSON.stringify(row) : csv(row),
            bytes = encoder.encode(value).byteLength;
          require(bytes <=
            limits.row_bytes, "a selected row exceeds the byte limit");
          count++;
          spoolBytes += bytes;
          require(count <= limits.rows &&
            spoolBytes <=
              limits.spool_bytes, "selection exceeds temporary storage limits; select fewer games or methods");
          if (batch.length && batchBytes + bytes > 1024 * 1024) {
            await spool.add(batch);
            batch = [];
            batchBytes = 0;
            check();
          }
          batch.push({
            sequence,
            cell: JSON.stringify([
              row.game_id,
              row.method,
              row.budget,
              row.seed,
            ]),
            value,
            bytes,
          });
          batchBytes += bytes;
        }
        if (batch.length) {
          await spool.add(batch);
          batch = [];
          batchBytes = 0;
          check();
        }
      }
      if (kind === "json") {
        const directory = new Map(),
          wanted = new Set(
            panel.games.map((g) => JSON.stringify(["game", g.id])),
          );
        for (const id of ["composition", "snapshot_provenance"])
          wanted.add(JSON.stringify(["report", id]));
        for (const d of manifest.assets.details)
          for (const pair of d.objects || []) {
            const key = JSON.stringify(pair);
            if (!wanted.has(key)) continue;
            if (!directory.has(key)) directory.set(key, []);
            directory.get(key).push(d);
          }
        const detail = async (type, id, fallback) => {
          const descriptors = directory.get(JSON.stringify([type, id]));
          if (!descriptors) return fallback;
          return BenchmarkDetails.object(
            {
              ...manifest,
              assets: { ...manifest.assets, details: descriptors },
            },
            { type, id },
            { read, signal },
          );
        };
        await write('{"snapshot_id":');
        await json(manifest.snapshot_id);
        await write(',"composition":');
        await json((await detail("report", "composition", null)) || null);
        await write(',"selection":{"games":');
        await json(panel.games.map((g) => g.id));
        await write(',"methods":');
        await json(request.methods);
        await write(',"budgets":');
        await json(
          [...new Set(Object.values(panel.game_budgets).flat())].sort(
            (a, b) => a - b,
          ),
        );
        await write(',"cells":[');
        let first = true;
        for (const g of panel.games)
          for (const budget of panel.game_budgets[g.id])
            for (const seed of manifest.suite.seeds) {
              if (!first) await write(",");
              first = false;
              await json({ game_id: g.id, budget, seed });
            }
        const selection = {
          model: request.selection.model || null,
          dataset: request.selection.dataset || null,
          score_order: request.score_order ? Number(request.score_order) : null,
          include_controls: Boolean(request.selection.include_controls),
          elo_panel: request.elo_panel || "available",
          seeds: manifest.suite.seeds,
          game_seeds: manifest.suite.game_seeds,
        };
        await write("]");
        for (const [key, value] of Object.entries(selection))
          if (value !== undefined) {
            await write(`,${JSON.stringify(key)}:`);
            await json(value);
          }
        await write('},"games":[');
        first = true;
        for (const game of panel.games) {
          const full = await detail("game", game.id, undefined);
          require(full !== undefined, "missing game details");
          if (!first) await write(",");
          first = false;
          await json(full);
        }
        await write('],"protocol":');
        await json(manifest.suite.protocol || null);
        const provenance = await detail(
          "report",
          "snapshot_provenance",
          undefined,
        );
        if (provenance !== undefined) {
          await write(',"snapshot_provenance":');
          await json(provenance);
        }
        await write(',"runs":{');
        first = true;
        for (const descriptor of manifest.assets.runs) {
          check();
          const block = await read(descriptor);
          check();
          for (let i = 0; i < block.count; i++) {
            const row = block.row(i),
              source = profiles.get(row.source_id);
            require(source?.type === "source", "missing run source");
            const run = { ...source.value };
            if (own(row, "execution")) run.execution = row.execution;
            if (!first) await write(",");
            first = false;
            await write(`${JSON.stringify(row.id)}:`);
            await json(run);
          }
        }
        await write('},"records":[');
        first = true;
        for await (const value of spool.rows()) {
          if (!first) await write(",");
          first = false;
          await write(value);
        }
        await write("]}");
      } else {
        await write(fields.join(","));
        for await (const value of spool.rows()) {
          await write("\n");
          await write(value);
        }
      }
      await flush();
      check();
      await spool.close();
      spool = null;
      await sink.close();
      return { records: count, bytes: totalBytes };
    } catch (error) {
      try {
        await sink.abort?.(error);
      } finally {
        if (spool) await spool.close();
      }
      throw error;
    }
  }

  async function save(manifest, request, kind, options = {}) {
    require(["json", "csv"].includes(kind), "unknown format");
    const type = kind === "json" ? "application/json" : "text/csv";
    if (options.sink) return stream(manifest, request, kind, options);
    let sink,
      parts = [],
      bytes = 0;
    if (globalThis.showSaveFilePicker) {
      const file = await showSaveFilePicker({
        suggestedName: `shapiq-selection.${kind}`,
        types: [
          {
            description: "Benchmark selection",
            accept: { [type]: [`.${kind}`] },
          },
        ],
      });
      sink = await file.createWritable();
    } else {
      const maximum = options.maxBlobBytes ?? 64 * 1024 * 1024;
      require(Number.isSafeInteger(maximum) &&
        maximum > 0 &&
        maximum <= 64 * 1024 * 1024, "invalid fallback limit");
      sink = {
        async write(text) {
          const chunk = encoder.encode(text);
          require(bytes + chunk.byteLength <=
            maximum, "this browser can save at most 64 MiB without file streaming; select fewer games or methods");
          parts.push(chunk);
          bytes += chunk.byteLength;
        },
        async close() {},
        async abort() {
          parts = [];
        },
      };
    }
    const result = await stream(manifest, request, kind, { ...options, sink });
    if (parts.length) {
      const blob = new Blob(parts, { type });
      parts = [];
      if (options.download) await options.download(blob);
      else {
        const url = URL.createObjectURL(blob),
          link = document.createElement("a");
        link.href = url;
        link.download = `shapiq-selection.${kind}`;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }
    }
    return result;
  }
  return Object.freeze({ save, stream, indexedSpool });
})();
