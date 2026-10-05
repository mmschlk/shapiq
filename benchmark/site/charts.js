"use strict";
// Chart series and rendering; app.js owns page state and shared scoring helpers.

function canonicalOracleTiming(timing) {
  if (timing?.protocol !== "batch-amortized-wall-seconds-v1") return timing;
  return {
    ...timing,
    profiles: timing.profiles.map((profile) => {
      const hardware = profile.preparation_hardware;
      if (
        profile.cpu_model &&
        hardware?.device === "cpu" &&
        hardware.cpu_model === profile.cpu_model &&
        Object.keys(hardware).every((key) =>
          ["device", "cpu_model"].includes(key),
        )
      ) {
        // Older CPU records omit this redundant object; their timing is equivalent.
        const { preparation_hardware, ...cpu } = profile;
        return cpu;
      }
      return profile;
    }),
  };
}

function renderPerformanceCharts(chartPanel, chartPending, computed = null) {
  if (computed) {
    const decorate = (series) =>
      chartNames
        .flatMap((method) => series.filter((s) => s.method === method))
        .map((s) => ({
          ...s,
          name: methodLabel(s.method),
          color: colorFor(s.method),
        }));
    const estimated = $("timeMetric").value === "estimated_uncached_seconds";
    $("timeChartNote").textContent = estimated
      ? "Estimated estimator work + batch-amortized oracle costs; not measured end-to-end runtime."
      : "Measured estimator runtime includes cache lookups for cached games; live oracle calls for uncached games. Diagnostic timing.";
    chart(
      "budgetChart",
      decorate(computed.budget_series),
      "Query budget per player · B / d",
      false,
      "median",
    );
    chart(
      "timeChart",
      decorate(computed.time_series),
      estimated
        ? "Mean estimated work + oracle seconds"
        : "Mean measured estimator seconds",
      false,
      "median",
    );
    if (chartPending)
      ["budgetChart", "timeChart"].forEach(
        (id) => ($(id).textContent = "Results pending for this selection."),
      );
    return;
  }
  const gamesById = new Map(chartPanel.games.map((g) => [g.id, g]));
  const chartRatios =
    data.suite.relative_budgets ||
    [
      ...new Set(
        chartPanel.cells.map(
          (cell) => cell.budget / gamesById.get(cell.game_id).n_players,
        ),
      ),
    ].sort((a, b) => a - b);
  const panelAt = (ratio, games = chartPanel.games) => {
    const cells = games.flatMap((g) =>
      data.suite.seeds.map((seed) => ({
        game_id: g.id,
        budget: Math.ceil(ratio * g.n_players),
        seed,
      })),
    );
    return { ...chartPanel, games, cells };
  };
  const rowsAt = (rows, panel) => {
    const keys = new Set(panel.cells.map(cellKey));
    return rows.filter((r) => keys.has(cellKey(r)));
  };
  const point = (rows, panel, ratio) => {
    const stats = summary(rows, panel);
    if (!Number.isFinite(stats.median)) return null;
    const good = rows.filter(
      (r) => r.status === "ok" && Number.isFinite(r.nmse),
    );
    const used = good.every((r) => Number.isFinite(r.queries))
      ? good.map((r) => r.queries / gamesById.get(r.game_id).n_players)
      : [];
    const low = Math.min(...used),
      high = Math.max(...used);
    return {
      x: ratio,
      y: stats.median,
      relativeBudget: ratio,
      coverage: `${stats.valid}/${stats.planned} successful runs · ${panel.games.length} games`,
      queryUsage: used.length
        ? `${format(low)}${high > low ? `–${format(high)}` : ""} × players`
        : null,
    };
  };
  const budgetSeries = chartNames.map((method) => ({
    name: methodLabel(method),
    method,
    color: colorFor(method),
    points: chartRatios
      .map((ratio) => {
        const panel = panelAt(ratio);
        return point(
          rowsAt(
            chartPanel.rows.filter((r) => r.method === method),
            panel,
          ),
          panel,
          ratio,
        );
      })
      .filter(Boolean),
  }));
  chart(
    "budgetChart",
    budgetSeries,
    "Query budget per player · B / d",
    false,
    "median",
  );
  const profiles = new Map();
  const timeMetric = $("timeMetric").value;
  const estimatedTime = timeMetric === "estimated_uncached_seconds";
  $("timeChartNote").textContent = estimatedTime
    ? "Estimated estimator work + batch-amortized oracle costs; not measured end-to-end runtime."
    : "Measured estimator runtime includes cache lookups for cached games; live oracle calls for uncached games. Diagnostic timing.";
  chartPanel.rows.forEach((row) => {
    if (
      estimatedTime &&
      !gamesById.get(row.game_id)?.metadata?.evaluation_timing
    )
      return;
    const worker = row.worker;
    const verified =
      worker?.cpu_model && worker?.thread_pools?.length && row.timing_profile;
    const key = JSON.stringify([
      row.timing_profile,
      worker?.cpu_model,
      worker?.machine,
      worker?.thread_pools,
      worker?.thread_environment,
      ...(estimatedTime
        ? [
            canonicalOracleTiming(
              gamesById.get(row.game_id)?.metadata?.evaluation_timing,
            ),
          ]
        : []),
      ...(verified ? [] : [row.run_id, row.game_id]),
    ]);
    if (!profiles.has(key))
      profiles.set(key, {
        rows: [],
        games: new Set(),
        label: `${worker?.cpu_model || "Unverified hardware"} · ${row.timing_profile || "diagnostic"}`,
      });
    const profile = profiles.get(key);
    profile.rows.push(row);
    profile.games.add(row.game_id);
  });
  const timeSeries = [];
  chartNames.forEach((method) =>
    profiles.forEach((profile) => {
      const games = chartPanel.games.filter((g) => profile.games.has(g.id));
      const points = chartRatios
        .map((ratio) => {
          const panel = panelAt(ratio, games);
          const rows = rowsAt(
            profile.rows.filter(
              (r) => r.method === method && Number.isFinite(r[timeMetric]),
            ),
            panel,
          );
          const result = point(rows, panel, ratio);
          if (result)
            result.x = summary(
              rows.map((r) => ({
                ...r,
                nmse: Number.isFinite(r.nmse) ? r[timeMetric] : null,
              })),
              panel,
            ).average;
          return result;
        })
        .filter(Boolean)
        .sort((a, b) => a.x - b.x);
      if (points.length)
        timeSeries.push({
          name: methodLabel(method),
          method,
          color: colorFor(method),
          profile: profile.label,
          points,
        });
    }),
  );
  chart(
    "timeChart",
    timeSeries,
    estimatedTime
      ? "Mean estimated work + oracle seconds"
      : "Mean measured estimator seconds",
    false,
    "median",
  );
  if (chartPending)
    ["budgetChart", "timeChart"].forEach(
      (id) => ($(id).textContent = "Results pending for this selection."),
    );
}

function renderHistory(s, preset, computed = null) {
  const metric = $("historyMetric").value,
    history = preset?.history,
    end = new Date().getUTCFullYear() + new Date().getUTCMonth() / 12;
  const year = (date) => {
    const d = new Date(date + "T00:00:00Z");
    return (
      d.getUTCFullYear() +
      (d - new Date(Date.UTC(d.getUTCFullYear(), 0, 1))) / 31557600000
    );
  };
  const historyMethods = (history?.methods || [])
    .map((method) => ({
      ...method,
      ...(computed
        ? computed.find((row) => row.method === method.method)
        : summary(
            s.rows.filter((row) => row.method === method.method),
            s,
          )),
    }))
    .filter((method) => method.complete);
  const historySeries = historyMethods
    .filter((method) => showMethod(method.method))
    .map((method) => ({
      name: methodLabel(method.method),
      method: method.method,
      releaseYear: method.date.slice(0, 4),
      color: colorFor(method.method),
      points: [
        {
          x: year(method.date),
          y: metric === "median" ? method.median : method.average,
        },
        { x: end, y: metric === "median" ? method.median : method.average },
      ],
    }));
  chart("historyChart", historySeries, "Publication year", true, metric);
  $("historySources").replaceChildren();
  (history?.methods || []).forEach((m) => {
    const a = document.createElement("a");
    if (!/^https:\/\/arxiv\.org\//.test(m.url)) return;
    a.href = m.url;
    a.textContent = `${m.method} (${m.date})`;
    a.rel = "noopener";
    $("historySources").append(a, document.createTextNode(" · "));
  });
  if (history?.unknown_dates?.length)
    $("historySources").append(
      document.createTextNode(
        `Dates unverified: ${history.unknown_dates.join(", ")}.`,
      ),
    );
}

function highlight(method) {
  document.querySelectorAll("[data-method]").forEach((node) => {
    node.classList.toggle(
      "dimmed",
      Boolean(method) && node.dataset.method !== method,
    );
    node.classList.toggle(
      "emphasis",
      Boolean(method) && node.dataset.method === method,
    );
  });
}
function bindHighlight(node, method, message) {
  if (!method) return;
  node.dataset.method = method;
  node.setAttribute("aria-describedby", "chartTooltip");
  const show = (event) => {
    highlight(method);
    const tip = $("chartTooltip"),
      rect = node.getBoundingClientRect();
    tip.textContent = message;
    tip.hidden = false;
    const left = Number.isFinite(event.clientX)
        ? event.clientX
        : rect.left + rect.width / 2,
      top = Number.isFinite(event.clientY) ? event.clientY : rect.bottom;
    tip.style.left = `${Math.max(10, Math.min(left + 13, window.innerWidth - tip.offsetWidth - 10))}px`;
    tip.style.top = `${Math.max(10, Math.min(top + 13, window.innerHeight - tip.offsetHeight - 10))}px`;
  };
  const clear = () => {
    highlight(null);
    $("chartTooltip").hidden = true;
  };
  node.addEventListener("pointerenter", show);
  node.addEventListener("pointermove", show);
  node.addEventListener("pointerleave", clear);
  node.addEventListener("focus", show);
  node.addEventListener("blur", clear);
}
function endpointLabelPositions(series, y, top, bottom) {
  const ordered = [...series].sort(
    (a, b) =>
      b.points.at(-1).y - a.points.at(-1).y || a.name.localeCompare(b.name),
  );
  const positions = [];
  ordered.forEach((item, i) => {
    positions.push(
      Math.max(y(item.points.at(-1).y), i ? positions[i - 1] + 24 : top),
    );
  });
  // Work back from the bottom when a cluster would extend beyond the plot.
  if (positions.at(-1) > bottom) {
    positions[positions.length - 1] = bottom;
    for (let i = positions.length - 2; i >= 0; i--)
      positions[i] = Math.min(positions[i], positions[i + 1] - 24);
  }
  return new Map(ordered.map((item, i) => [item, positions[i]]));
}
let historyChartState;
function chart(id, series, xlabel, dates = false, metric = "mean") {
  if (dates) historyChartState = { series, xlabel, metric };
  const box = $(id);
  box.replaceChildren();
  const points = series.flatMap((s) => s.points);
  if (!points.length) {
    const p = document.createElement("p");
    p.className = "emptyNote";
    p.textContent = dates
      ? "History requires complete coverage and a verified publication date."
      : "No successful results in this view.";
    box.append(p);
    return;
  }
  const ns = "http://www.w3.org/2000/svg",
    svg = document.createElementNS(ns, "svg");
  const rightLabels = dates && window.innerWidth > 720,
    labelWidth = rightLabels ? 236 : 0,
    width = dates ? Math.max(240, box.clientWidth) : 480,
    height = dates
      ? Math.max(
          Math.min(360, Math.max(275, width * 0.32)),
          rightLabels ? series.length * 24 + 70 : 0,
        )
      : 275,
    plotWidth = width - 87 - labelWidth,
    plotBottom = height - 47,
    plotHeight = plotBottom - 23;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  svg.setAttribute(
    "aria-label",
    `${metric === "median" ? "Median" : "Mean"} normalized error versus ${xlabel}`,
  );
  const queryAxis = id === "budgetChart";
  const errorFloor = 1e-12,
    timeFloor = 1e-9;
  const logX = (value) =>
    queryAxis ? Math.log2(value) : Math.log10(Math.max(value, timeFloor));
  const xLow = dates
    ? Math.floor(Math.min(...points.map((p) => p.x)))
    : Math.floor(Math.min(...points.map((p) => logX(p.x))));
  const xHigh = Math.max(
    xLow + 1,
    dates
      ? Math.ceil(Math.max(...points.map((p) => p.x)))
      : Math.ceil(Math.max(...points.map((p) => logX(p.x)))),
  );
  const yLow = Math.floor(
    Math.log10(Math.max(Math.min(...points.map((p) => p.y)), errorFloor)),
  );
  const yHigh = Math.max(
    yLow + 1,
    Math.ceil(Math.log10(Math.max(...points.map((p) => p.y), errorFloor))),
  );
  const xPosition = (value) =>
    62 + ((value - xLow) / (xHigh - xLow)) * plotWidth;
  const yPosition = (value) =>
    plotBottom - ((value - yLow) / (yHigh - yLow)) * plotHeight;
  const x = (value) => xPosition(dates ? value : logX(value));
  const y = (value) => yPosition(Math.log10(Math.max(value, errorFloor)));
  const hasErrorFloor = points.some((p) => p.y <= errorFloor);
  const hasTimeFloor =
    !dates && !queryAxis && points.some((p) => p.x <= timeFloor);
  function element(tag, attributes, text, parent = svg) {
    const node = document.createElementNS(ns, tag);
    Object.entries(attributes).forEach(([key, value]) =>
      node.setAttribute(key, value),
    );
    if (text !== undefined) node.textContent = text;
    parent.append(node);
    return node;
  }
  const tick = (value) =>
    value === 0
      ? "0"
      : Math.abs(value) >= 1000
        ? `${Number((value / 1000).toPrecision(2))}k`
        : Math.abs(value) < 0.001
          ? value.toExponential(1)
          : String(Number(value.toPrecision(2)));
  const ticks = (low, high, intervals = 4) => {
    const step = Math.max(1, Math.ceil((high - low) / intervals));
    const values = [];
    for (let value = low; value <= high; value += step) values.push(value);
    if (values.at(-1) !== high) values.push(high);
    return values;
  };
  element("rect", {
    x: 62,
    y: 23,
    width: plotWidth,
    height: plotHeight,
    fill: "#f8faff",
    rx: 6,
  });
  if (hasErrorFloor)
    element("rect", {
      x: 62,
      y: plotBottom - 10,
      width: plotWidth,
      height: 12,
      fill: "#edf2ff",
    });
  ticks(yLow, yHigh).forEach((exponent) => {
    const yy = yPosition(exponent);
    element("line", {
      x1: 62,
      x2: 62 + plotWidth,
      y1: yy,
      y2: yy,
      stroke: "#dfe6f0",
      "stroke-dasharray": "3 4",
    });
    element(
      "text",
      {
        x: 55,
        y: yy + 4,
        "text-anchor": "end",
        "font-size": 10,
        fill: "#596980",
      },
      hasErrorFloor && exponent === -12
        ? "≤1e-12"
        : Math.abs(exponent) > 3
          ? `1e${exponent}`
          : tick(10 ** exponent),
    );
  });
  const xTicks = ticks(
    xLow,
    xHigh,
    dates
      ? Math.max(2, Math.min(7, Math.floor(plotWidth / 75)))
      : queryAxis
        ? 8
        : 4,
  );
  // Keep both end labels; omit a nearby interior year on narrow screens.
  if (
    dates &&
    xTicks.length > 2 &&
    xPosition(xTicks.at(-1)) - xPosition(xTicks.at(-2)) < 40
  )
    xTicks.splice(-2, 1);
  xTicks.forEach((value) => {
    const label = dates
      ? String(value)
      : hasTimeFloor && value === -9
        ? "≤1e-9"
        : tick((queryAxis ? 2 : 10) ** value);
    element(
      "text",
      {
        x: xPosition(value),
        y: height - 29,
        "text-anchor": "middle",
        "font-size": 10,
        fill: "#596980",
      },
      label,
    );
  });
  element(
    "text",
    {
      x: 62 + plotWidth / 2,
      y: height - 3,
      "text-anchor": "middle",
      "font-size": 11,
      fill: "#596980",
    },
    `${xlabel}${dates ? "" : " · log"}`,
  );
  element(
    "text",
    { x: 62, y: 12, "font-size": 10, fill: "#596980" },
    `${metric === "median" ? "Median" : "Mean"} nMSE · log ↓`,
  );
  const legend = document.createElement("div");
  legend.className = dates
    ? `legend historyLegend${rightLabels ? " endpointLegend" : ""}`
    : "legend";
  const labelPositions = dates
      ? endpointLabelPositions(series, y, 23, plotBottom)
      : new Map(),
    legendItems = new Map();
  series.forEach((s) => {
    const method = s.method || s.name,
      color = s.color || colorFor(method),
      dash = dashFor(method),
      marker = Math.floor(methodIndex(method) / palette.length) % 4;
    const group = element("g", {
      class: "series",
      "data-method": method,
    });
    const line = element(
      "polyline",
      {
        points: s.points.map((p) => `${x(p.x)},${y(p.y)}`).join(" "),
        fill: "none",
        stroke: color,
        "stroke-width": 2.2,
        "stroke-dasharray": dash,
        "stroke-linejoin": "round",
        "stroke-linecap": "round",
        tabindex: 0,
        role: "img",
        "aria-label": `${s.name} error curve`,
      },
      undefined,
      group,
    );
    bindHighlight(
      line,
      method,
      `${s.name}${s.profile ? ` · ${s.profile}` : ""}`,
    );
    s.points.forEach((p) => {
      const cx = x(p.x),
        cy = y(p.y),
        base = {
          fill: "#fff",
          stroke: color,
          "stroke-width": 1.7,
          tabindex: 0,
          role: "img",
        };
      let dot;
      if (marker === 1)
        dot = element(
          "rect",
          { ...base, x: cx - 3, y: cy - 3, width: 6, height: 6 },
          undefined,
          group,
        );
      else if (marker === 2)
        dot = element(
          "polygon",
          {
            ...base,
            points: `${cx},${cy - 4} ${cx + 4},${cy + 3} ${cx - 4},${cy + 3}`,
          },
          undefined,
          group,
        );
      else if (marker === 3)
        dot = element(
          "polygon",
          {
            ...base,
            points: `${cx},${cy - 4} ${cx + 4},${cy} ${cx},${cy + 4} ${cx - 4},${cy}`,
          },
          undefined,
          group,
        );
      else
        dot = element("circle", { ...base, cx, cy, r: 3.5 }, undefined, group);
      const label = `${s.name}\n${metric === "median" ? "Median" : "Mean"} nMSE: ${format(p.y)}\n${xlabel}: ${format(p.x)}${s.profile ? `\nProfile: ${s.profile}` : ""}${p.coverage ? `\nCoverage: ${p.coverage}` : ""}${p.relativeBudget !== undefined ? `\nBudget: ${format(p.relativeBudget)} × players${p.queryUsage ? ` · used ${p.queryUsage}` : ""}` : ""}`;
      dot.setAttribute("aria-label", label);
      element("title", {}, label, dot);
      bindHighlight(dot, method, label);
    });
    // Timing profiles keep separate curves but share one estimator legend entry.
    const legendKey = dates ? s : method;
    if (legendItems.has(legendKey)) return;
    const label = document.createElement("button"),
      swatch = document.createElement("span");
    label.type = "button";
    swatch.className = "legendSwatch";
    swatch.style.borderTop = `2px ${dash ? "dashed" : "solid"} ${color}`;
    label.append(
      swatch,
      document.createTextNode(
        `${s.name}${s.releaseYear ? ` (${s.releaseYear})` : ""}`,
      ),
    );
    const item = document.createElement("div");
    item.className = "legendItem";
    item.append(label);
    if (dates) {
      const endpoint = s.points.at(-1),
        value = document.createElement("span");
      value.className = "endpointValue";
      value.textContent = `${format(endpoint.y)} nMSE`;
      item.append(value);
      if (rightLabels) {
        const labelY = labelPositions.get(s);
        label.style.left = `${width - labelWidth}px`;
        label.style.top = `${labelY}px`;
        label.style.width = `${labelWidth - 4}px`;
      }
    }
    if (Object.hasOwn(data.methods, method)) {
      const details = methodDetails(method, dates);
      bindMethodDetails(label, details, dates ? legend : null);
      item.append(details);
    }
    bindHighlight(
      label,
      method,
      `${s.name}${dates ? ` · ${metric === "median" ? "Median" : "Mean"} nMSE ${format(s.points.at(-1).y)}` : ""}`,
    );
    legendItems.set(legendKey, item);
    if (!dates) legend.append(item);
  });
  box.append(svg);
  if (hasErrorFloor || hasTimeFloor) {
    const note = document.createElement("p");
    note.className = "scaleNote";
    note.textContent = [
      hasErrorFloor
        ? "nMSE ≤1e-12, including zero, shares the bottom band. Exact values on hover."
        : "",
      hasTimeFloor
        ? "Zero and sub-nanosecond times share the ≤1e-9 s position."
        : "",
    ]
      .filter(Boolean)
      .join(" ");
    box.append(note);
  }
  if (dates)
    labelPositions.forEach((position, item) =>
      legend.append(legendItems.get(item)),
    );
  box.append(legend);
}
