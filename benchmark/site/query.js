"use strict";
// Pure, worker-compatible reductions. No DOM, retained block cache, or scientific approximations.
globalThis.BenchmarkQuery = (() => {
  const target = (g) => `${g.index} · order ${g.order}`;
  const format = (n) => (Number.isFinite(n) ? n.toPrecision(4) : "—");
  const same = (a, b) =>
    JSON.stringify([...a].sort()) === JSON.stringify([...b].sort());
  const require = (condition, message) => {
    if (!condition) throw Error(`Invalid benchmark query: ${message}.`);
  };
  const canonicalOracleTiming = (timing) => {
    if (timing?.protocol !== "batch-amortized-wall-seconds-v1") return timing;
    return {
      ...timing,
      profiles: timing.profiles.map((profile) => {
        const h = profile.preparation_hardware;
        if (
          profile.cpu_model &&
          h?.device === "cpu" &&
          h.cpu_model === profile.cpu_model &&
          Object.keys(h).every((key) => ["device", "cpu_model"].includes(key))
        ) {
          const { preparation_hardware, ...rest } = profile;
          return rest;
        }
        return profile;
      }),
    };
  };
  const profileKey = (row, worker, game, estimated) =>
    JSON.stringify([
      row.timing_profile,
      worker?.cpu_model,
      worker?.machine,
      worker?.thread_pools,
      worker?.thread_environment,
      ...(estimated
        ? [canonicalOracleTiming(game.metadata?.evaluation_timing)]
        : []),
      ...(worker?.cpu_model &&
      worker?.thread_pools?.length &&
      row.timing_profile
        ? []
        : [row.run_id, row.game_id]),
    ]);

  function panel(games, selection, suite, zero, allBudgets = false) {
    const selected = games.filter(
      (g) =>
        Boolean(g.metadata?.synthetic) === (selection.panel === "diagnostic") &&
        (selection.include_controls ||
          g.metadata?.game_quality?.role !== "control") &&
        target(g) === selection.target &&
        (!selection.family || g.family === selection.family) &&
        (!selection.dataset ||
          (g.metadata?.dataset || "Unrecorded dataset") ===
            selection.dataset) &&
        (!selection.model ||
          (g.metadata?.model_profile ||
            g.metadata?.model ||
            "No model recorded") === selection.model) &&
        g.n_players >= (selection.min_players ?? 0) &&
        g.n_players <= (selection.max_players ?? Infinity),
    );
    const budgets = {};
    for (const game of selected) {
      let grid =
        !allBudgets && selection.relative_budget != null
          ? [Math.ceil(selection.relative_budget * game.n_players)]
          : game.budgets;
      if (!allBudgets && selection.cap != null)
        grid = [
          Math.max(
            -1,
            ...grid.filter((b) => b <= selection.cap * game.n_players),
          ),
        ];
      budgets[game.id] = grid;
    }
    return makePanel(
      selected.filter((g) => !zero.has(g.id)),
      budgets,
      suite.seeds,
      selected.map((g) => g.id),
      selected.filter((g) => zero.has(g.id)).length,
    );
  }

  function makePanel(
    games,
    budgets,
    seeds,
    ids = games.map((g) => g.id),
    excluded = 0,
  ) {
    const families = new Map(),
      weights = new Map(),
      grids = new Map();
    for (const g of games) {
      if (!families.has(g.family)) families.set(g.family, new Map());
      const strata = families.get(g.family);
      strata.set(g.stratum, (strata.get(g.stratum) || 0) + 1);
    }
    let planned = 0;
    for (const g of games) {
      const strata = families.get(g.family),
        size = budgets[g.id].length * seeds.length;
      weights.set(
        g.id,
        1 / families.size / strata.size / strata.get(g.stratum) / size,
      );
      grids.set(g.id, new Set(budgets[g.id]));
      planned += size;
    }
    return {
      games,
      game_budgets: budgets,
      panel_ids: ids,
      excluded,
      planned,
      weights,
      grids,
      seeds: new Set(seeds),
    };
  }
  const contains = (p, c, i) =>
    p.grids.get(c.game[i])?.has(c.budget[i]) && p.seeds.has(c.seed[i]);
  const publicPanel = (p, scoreOrder) => {
    const shares = p.games
      .map((g) => g.metadata?.order_scores?.[scoreOrder]?.energy_share)
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    return {
      game_count: p.games.length,
      excluded: p.excluded,
      planned: p.planned,
      median_energy_share:
        scoreOrder && shares.length
          ? (shares[Math.floor(shares.length / 2)] +
              shares[Math.floor((shares.length - 1) / 2)]) /
            2
          : null,
    };
  };
  const codepointOrder = (a, b) => {
    const x = Array.from(a, (c) => c.codePointAt(0)),
      y = Array.from(b, (c) => c.codePointAt(0));
    for (let i = 0; i < Math.min(x.length, y.length); i++)
      if (x[i] !== y[i]) return x[i] - y[i];
    return x.length - y.length;
  };
  async function selectorHash(p, request) {
    const value = [
      request.score_order ? Number(request.score_order) : null,
      Boolean(request.selection.include_controls),
      request.selection.panel || "real",
      [...request.methods].sort(codepointOrder),
      [...p.panel_ids]
        .sort(codepointOrder)
        .map((id) => [id, [...p.game_budgets[id]].sort((a, b) => a - b)]),
    ];
    const digest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(JSON.stringify(value)),
    );
    return Array.from(new Uint8Array(digest), (b) =>
      b.toString(16).padStart(2, "0"),
    ).join("");
  }

  function summary(c, indexes, p, time = false) {
    const good = indexes.filter(
      (i) =>
        c.status[i] === "ok" &&
        Number.isFinite(c.score[i]) &&
        (!time || Number.isFinite(c.time[i])),
    );
    let mass = 0,
      correction = 0,
      numerator = 0;
    for (const i of good) {
      const weight = p.weights.get(c.game[i]);
      const adjusted = weight - correction,
        next = mass + adjusted;
      correction = next - mass - adjusted;
      mass = next;
      numerator += (time ? c.time[i] : c.score[i]) * weight;
    }
    const value = (i) => (time ? c.time[i] : c.score[i]);
    const ordered = good
      .filter((i) => p.weights.get(c.game[i]) > 0)
      .sort((a, b) => value(a) - value(b));
    let cumulative = 0,
      median = null;
    for (let n = 0; n < ordered.length; n++) {
      cumulative += p.weights.get(c.game[ordered[n]]);
      const fraction = cumulative / mass;
      if (fraction + 1e-14 >= 0.5) {
        median =
          Math.abs(fraction - 0.5) <= 1e-14 && n + 1 < ordered.length
            ? value(ordered[n]) / 2 + value(ordered[n + 1]) / 2
            : value(ordered[n]);
        break;
      }
    }
    return {
      average: mass > 0 ? numerator / mass : null,
      median,
      valid: good.length,
      planned: p.planned,
      failed: indexes.filter((i) => c.status[i] === "failed").length,
      underBudget: indexes.filter((i) => c.under[i]).length,
      unsupported: indexes.filter((i) => c.status[i] === "unsupported").length,
      missing: p.planned - indexes.length,
      complete: p.planned > 0 && good.length === p.planned,
    };
  }

  function matchedPreset(preset, p, request) {
    if (!preset) return null;
    return (preset.score_order == null ? "" : String(preset.score_order)) ===
      String(request.score_order || "") &&
      Boolean(preset.include_controls) ===
        Boolean(request.selection.include_controls) &&
      same(preset.game_ids, p.panel_ids) &&
      same(preset.methods, request.methods) &&
      (!preset.panel || preset.panel === (request.selection.panel || "real")) &&
      p.panel_ids.every((id) =>
        same(preset.game_budgets?.[id] || preset.budgets, p.game_budgets[id]),
      )
      ? preset
      : null;
  }

  async function query(manifest, request, options = {}) {
    const signal = options.signal,
      read =
        options.read || ((d) => BenchmarkPartitions.read(manifest, d, options));
    const limits = {
      games: 250000,
      method_rows: 1000000,
      profiles: 50000,
      memberships: 1000000,
      series: 20000,
      points: 200000,
      ...options.limits,
    };
    require(Object.values(limits).every(
      (value) => Number.isSafeInteger(value) && value > 0,
    ), "memory limits");
    const check = () => signal?.throwIfAborted();
    require(request.methods.length === new Set(request.methods).size &&
      request.methods.every((m) =>
        Object.hasOwn(manifest.methods, m),
      ), "methods");
    require(request.selection.target ===
      request.chart_selection.target, "mixed targets");
    require(["seconds", "estimated_uncached_seconds"].includes(
      request.timing_metric,
    ), "timing metric");
    const estimated = request.timing_metric === "estimated_uncached_seconds",
      games = [],
      workers = new Map();
    for (const d of manifest.assets.games) {
      if (d.target && d.target !== request.selection.target) continue;
      check();
      const block = await read(d);
      check();
      for (let i = 0; i < block.count; i++) {
        const game = block.row(i);
        if (target(game) === request.selection.target) games.push(game);
        require(games.length <=
          limits.games, "game index exceeds memory limit");
      }
    }
    games.sort((a, b) => a.sequence - b.sequence);
    const byId = new Map(games.map((g) => [g.id, g]));
    require(byId.size === games.length, "repeated game");
    for (const d of manifest.assets.profiles) {
      check();
      const block = await read(d);
      check();
      for (let i = 0; i < block.count; i++)
        if (block.get("type", i) === "worker") {
          const id = block.get("id", i),
            value = block.get("value", i);
          if (workers.has(id))
            require(JSON.stringify(workers.get(id)) ===
              JSON.stringify(value), "conflicting worker");
          workers.set(id, value);
          require(workers.size <=
            limits.profiles, "worker index exceeds memory limit");
        }
    }
    const zero = new Set(
      games
        .filter(
          (g) =>
            g.row_zero_truth_energy ||
            g.metadata?.zero_truth_energy ||
            g.metadata?.score_eligible === false ||
            (request.score_order &&
              g.metadata?.order_scores?.[request.score_order]
                ?.score_eligible !== true),
        )
        .map((g) => g.id),
    );
    // Membership spans all selected methods, including failures and unsupported rows.
    const provisional = panel(
      games,
      request.chart_selection,
      manifest.suite,
      new Set(),
      true,
    );
    const profiles = new Map(),
      selectedMethods = new Set(request.methods);
    let memberships = 0;
    const tableProvisional = panel(
      games,
      request.selection,
      manifest.suite,
      new Set(),
    );
    const selectedFamilies = new Set(
      [...provisional.games, ...tableProvisional.games].map((g) => g.family),
    );
    const rowZeroIndexed = games.every((g) =>
      Object.hasOwn(g, "row_zero_truth_energy"),
    );
    const descriptors = manifest.assets.metrics.filter(
      (d) =>
        (!d.target || d.target === request.selection.target) &&
        (!rowZeroIndexed ||
          ((!d.method || selectedMethods.has(d.method)) &&
            (!d.family || selectedFamilies.has(d.family)))),
    );
    const rowProfile = (row) => {
      const game = byId.get(row.game_id);
      if (estimated && !game.metadata?.evaluation_timing) return null;
      require(row.worker_id == null ||
        workers.has(row.worker_id), "unknown worker profile");
      const worker = workers.get(row.worker_id),
        key = profileKey(row, worker, game, estimated);
      return {
        key,
        label: `${worker?.cpu_model || "Unverified hardware"} · ${row.timing_profile || "diagnostic"}`,
      };
    };
    for (const d of descriptors) {
      check();
      const block = await read(d);
      check();
      for (let i = 0; i < block.count; i++) {
        const id = block.get("game_id", i);
        require(byId.has(id), "unknown game");
        if (block.get("zero_truth_energy", i)) zero.add(id);
        if (
          !selectedMethods.has(block.get("method", i)) ||
          !provisional.grids.get(id)?.has(block.get("budget", i)) ||
          !provisional.seeds.has(block.get("seed", i))
        )
          continue;
        const row = block.row(i),
          p = rowProfile(row);
        if (!p) continue;
        if (!profiles.has(p.key))
          profiles.set(p.key, {
            ...p,
            games: new Set(),
            sequences: new Map(),
            sequence: row.sequence,
          });
        const profile = profiles.get(p.key);
        profile.sequences.set(
          id,
          Math.min(profile.sequences.get(id) ?? Infinity, row.sequence),
        );
        if (!profile.games.has(id)) {
          profile.games.add(id);
          memberships++;
        }
        require(profiles.size <= limits.profiles &&
          memberships <=
            limits.memberships, "timing index exceeds memory limit");
      }
    }
    const tablePanel = panel(games, request.selection, manifest.suite, zero);
    const chartPanel = panel(
      games,
      request.chart_selection,
      manifest.suite,
      zero,
      true,
    );
    const chartIds = new Set(chartPanel.games.map((g) => g.id));
    for (const [key, p] of profiles) {
      p.games = new Set([...p.games].filter((id) => chartIds.has(id)));
      if (!p.games.size) profiles.delete(key);
      else
        p.sequence = [...p.games].reduce(
          (minimum, id) => Math.min(minimum, p.sequences.get(id)),
          Infinity,
        );
      delete p.sequences;
    }
    const orderedProfiles = [...profiles.values()].sort(
      (a, b) => a.sequence - b.sequence,
    );
    orderedProfiles.forEach((profile, index) => {
      profile.index = index;
    });
    const ratios =
      manifest.suite.relative_budgets ||
      [
        ...new Set(
          chartPanel.games.flatMap((g) =>
            g.budgets.map((b) => b / g.n_players),
          ),
        ),
      ].sort((a, b) => a - b);
    const at = (ratio, selected = chartPanel.games) =>
      makePanel(
        selected,
        Object.fromEntries(
          selected.map((g) => [g.id, [Math.ceil(ratio * g.n_players)]]),
        ),
        manifest.suite.seeds,
      );
    const table = [],
      chartRanking = [],
      budgetSeries = [],
      timeSeries = [];
    const issues = { underBudget: new Map(), failures: new Map() },
      hardware = new Map(),
      timingProfiles = new Map();
    const methodLabel =
      options.methodLabel ||
      ((method) =>
        ({
          PermutationSamplingSV: "Permutation · values",
          PermutationSamplingSII: "Permutation · SII",
          PermutationSamplingSTII: "Permutation · STII",
          RegressionFBII: "Regression · FBII",
          RegressionFSII: "Regression · FSII",
        })[method] || method);
    const remember = (map, key, sequence) =>
      map.set(key, Math.min(map.get(key) ?? Infinity, sequence));
    const addIssue = (row) => {
      const under = row.failure_reason === "insufficient_budget";
      const explanation = under
        ? Number.isFinite(row.minimum_budget)
          ? `requires at least ${Number(format(row.minimum_budget / byId.get(row.game_id).n_players))} × players`
          : row.method === "ProxySPEX"
            ? "too few samples for five-fold proxy fitting"
            : "budget below the sparse transform's minimum"
        : {
            TimeoutError: "time limit reached",
            MemoryError: "memory allocation failed",
            ValueError: "input, configuration, or numerical validation failed",
            ModuleNotFoundError: "optional dependency unavailable",
            ImportError: "dependency could not load",
            LinAlgError: "linear algebra failed",
            BudgetExceededError: "query budget exceeded",
          }[row.error_type] || "estimator error";
      const description = `${methodLabel(row.method)}: ${explanation}`,
        map = issues[under ? "underBudget" : "failures"];
      const value = map.get(description) || {
        description,
        count: 0,
        sequence: row.sequence,
      };
      value.count++;
      value.sequence = Math.min(value.sequence, row.sequence);
      map.set(description, value);
    };
    let outputPoints = 0;
    let tableRows = 0,
      tableNonUnsupported = false,
      chartRows = 0,
      chartNonUnsupported = false;
    const point = (c, indexes, p, ratio) => {
      const stats = summary(c, indexes, p);
      if (!Number.isFinite(stats.median)) return null;
      const good = indexes.filter(
        (i) => c.status[i] === "ok" && Number.isFinite(c.score[i]),
      );
      let low = Infinity,
        high = -Infinity,
        valid = good.length > 0;
      for (const i of good) {
        const q = c.queries[i];
        if (!Number.isFinite(q)) {
          valid = false;
          break;
        }
        const ratio = q / byId.get(c.game[i]).n_players;
        low = Math.min(low, ratio);
        high = Math.max(high, ratio);
      }
      return {
        x: ratio,
        y: stats.median,
        relativeBudget: ratio,
        coverage: `${stats.valid}/${stats.planned} successful runs · ${p.games.length} games`,
        queryUsage: valid
          ? `${format(low)}${high > low ? `–${format(high)}` : ""} × players`
          : null,
      };
    };
    for (const method of request.methods) {
      check();
      // Only scalar columns for one estimator survive a block read.
      const c = Object.fromEntries(
        [
          "sequence",
          "game",
          "budget",
          "seed",
          "status",
          "score",
          "queries",
          "time",
          "profile",
          "under",
        ].map((k) => [k, []]),
      );
      for (const d of descriptors) {
        if (d.method && d.method !== method) continue;
        check();
        const block = await read(d);
        check();
        for (let i = 0; i < block.count; i++) {
          if (block.get("method", i) !== method) continue;
          const id = block.get("game_id", i),
            budget = block.get("budget", i),
            seed = block.get("seed", i);
          if (
            zero.has(id) ||
            !(
              tablePanel.grids.get(id)?.has(budget) ||
              chartPanel.grids.get(id)?.has(budget)
            ) ||
            !tablePanel.seeds.has(seed)
          )
            continue;
          const row = block.row(i),
            p = rowProfile(row);
          if (tablePanel.grids.get(id)?.has(budget)) {
            const worker = workers.get(row.worker_id);
            if (worker) remember(hardware, worker.cpu_model, row.sequence);
            if (row.timing_profile)
              remember(timingProfiles, row.timing_profile, row.sequence);
            if (row.status === "failed") addIssue(row);
          }
          c.sequence.push(row.sequence);
          c.game.push(id);
          c.budget.push(budget);
          c.seed.push(seed);
          c.status.push(row.status);
          c.score.push(
            request.score_order
              ? (row.order_scores?.[request.score_order]?.nmse ?? null)
              : row.nmse,
          );
          c.queries.push(row.queries);
          c.time.push(row[request.timing_metric]);
          c.profile.push(p ? profiles.get(p.key)?.index : undefined);
          c.under.push(
            row.status === "failed" &&
              row.failure_reason === "insufficient_budget",
          );
          require(c.sequence.length <=
            limits.method_rows, "estimator selection exceeds memory limit");
        }
      }
      const order = c.sequence
        .map((_v, i) => i)
        .sort((a, b) => c.sequence[a] - c.sequence[b]);
      const tableIndexes = order.filter((i) => contains(tablePanel, c, i)),
        chartIndexes = order.filter((i) => contains(chartPanel, c, i));
      table.push({ method, ...summary(c, tableIndexes, tablePanel) });
      chartRanking.push({ method, ...summary(c, chartIndexes, chartPanel) });
      tableRows += tableIndexes.length;
      chartRows += chartIndexes.length;
      tableNonUnsupported ||= tableIndexes.some(
        (i) => c.status[i] !== "unsupported",
      );
      chartNonUnsupported ||= chartIndexes.some(
        (i) => c.status[i] !== "unsupported",
      );
      budgetSeries.push({
        method,
        points: ratios
          .map((ratio) => {
            const p = at(ratio);
            return point(
              c,
              chartIndexes.filter((i) => contains(p, c, i)),
              p,
              ratio,
            );
          })
          .filter(Boolean),
      });
      const byProfile = new Map(),
        successfulProfiles = new Set();
      for (const i of chartIndexes) {
        const id = c.profile[i];
        if (id == null || !Number.isFinite(c.time[i])) continue;
        if (!byProfile.has(id)) byProfile.set(id, []);
        byProfile.get(id).push(i);
        if (c.status[i] === "ok" && Number.isFinite(c.score[i]))
          successfulProfiles.add(id);
      }
      // Failed/unverified profiles can number in the thousands but have no curve.
      // Group once; never rescan every method row for each timing profile.
      for (const id of [...successfulProfiles].sort((a, b) => a - b)) {
        const profile = orderedProfiles[id];
        const selected = [...profile.games]
          .map((id) => byId.get(id))
          .sort((a, b) => a.sequence - b.sequence);
        const points = ratios
          .map((ratio) => {
            const p = at(ratio, selected),
              indexes = byProfile.get(id).filter((i) => contains(p, c, i));
            const result = point(c, indexes, p, ratio);
            if (result) result.x = summary(c, indexes, p, true).average;
            return result;
          })
          .filter(Boolean)
          .sort((a, b) => a.x - b.x);
        if (points.length) {
          outputPoints += points.length;
          timeSeries.push({ method, profile: profile.label, points });
          require(timeSeries.length <= limits.series &&
            outputPoints <=
              limits.points, "timing output exceeds memory limit");
        }
      }
    }
    check();
    const selector = {
      panel_ids: tablePanel.panel_ids,
      game_budgets: tablePanel.game_budgets,
    };
    const candidate = options.lookupPreset
      ? await options.lookupPreset(
          {
            sha256: await selectorHash(tablePanel, request),
            selection: selector,
          },
          signal,
        )
      : options.preset;
    check();
    const compactIssues = Object.fromEntries(
      Object.entries(issues).map(([kind, map]) => [
        kind,
        [...map.values()]
          .sort((a, b) => b.count - a.count || a.sequence - b.sequence)
          .slice(0, 6)
          .map(({ description, count }) => ({ description, count })),
      ]),
    );
    const matched = matchedPreset(candidate, tablePanel, request);
    // Large selector grids stay in the worker; retain the exact display fields.
    const keys = (map) =>
      [...map].sort((a, b) => a[1] - b[1]).map(([key]) => key);
    return {
      table,
      chart_ranking: chartRanking,
      budget_series: budgetSeries,
      time_series: timeSeries,
      selection: publicPanel(tablePanel, request.score_order),
      chart_selection: publicPanel(chartPanel, request.score_order),
      chart_families: [
        ...new Set(
          panel(
            games,
            { ...request.chart_selection, family: "" },
            manifest.suite,
            zero,
            true,
          ).games.map((g) => g.family),
        ),
      ],
      issues: compactIssues,
      hardware: {
        cpu_models: keys(hardware),
        timing_profiles: keys(timingProfiles),
      },
      table_pending:
        tablePanel.planned > 0 &&
        !tableNonUnsupported &&
        tableRows < tablePanel.planned * request.methods.length,
      chart_pending:
        chartPanel.planned > 0 &&
        !chartNonUnsupported &&
        chartRows < chartPanel.planned * request.methods.length,
      preset: matched
        ? Object.fromEntries(
            ["id", "rows", "history", "common_panel", "uncertainty", "elo_l2"]
              .filter((key) => Object.hasOwn(matched, key))
              .map((key) => [key, matched[key]]),
          )
        : null,
    };
  }
  return Object.freeze({ query, canonicalOracleTiming, selectorHash });
})();
