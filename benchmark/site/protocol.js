"use strict";

// Report metadata is the source of truth. A newer roadmap never relabels old results.
const modelProfile = (game) =>
  game.metadata?.model_profile || game.metadata?.model || "No model recorded";
const sourceRoot = () => {
  const commit = data.snapshot_provenance?.git_commit;
  return `https://github.com/rtealwitter/shapiq/blob/${/^[a-f0-9]{40}$/.test(commit || "") ? commit : "benchmark"}/`;
};
function sourceLink(label, url) {
  const link = document.createElement("a");
  link.textContent = label;
  if (typeof url === "string" && url.startsWith("https://")) {
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  }
  return link;
}
function protocolParagraph(label, text) {
  const paragraph = document.createElement("p"),
    heading = document.createElement("strong");
  heading.textContent = `${label}. `;
  paragraph.append(heading, document.createTextNode(text));
  return paragraph;
}
function renderProtocol() {
  const protocol = data.suite.protocol;
  const models = [...new Set(data.games.map(modelProfile))];
  const datasets = [
    ...new Set(data.games.map((g) => g.metadata?.dataset).filter(Boolean)),
  ];
  const players = data.games.map((game) => game.n_players);
  const summary = $("protocolSummary");
  summary.replaceChildren(
    document.createTextNode(
      `${protocol?.name || "Earlier benchmark preview"} · ${datasets.length} data sources · ${Math.min(...players)}–${Math.max(...players)} players. `,
    ),
  );
  const jump = document.createElement("a");
  jump.href = "#protocol";
  jump.textContent = "Datasets, models & budgets";
  jump.addEventListener("click", () => {
    $("protocol").open = true;
  });
  summary.append(jump);
  const overview = $("protocolOverview");
  overview.replaceChildren();
  overview.append(
    protocolParagraph(
      "This report",
      protocol?.description ||
        "Earlier frozen configurations. The stronger-model rollout is separate; its datasets and models are not yet measured here.",
    ),
  );
  overview.append(
    protocolParagraph(
      "Budgets",
      `The benchmark grid is 0.5, 1, 2, 4, 8, 16, 32, 64 and 128 × d. ${data.suite.relative_budgets?.length ? `This report schedules ${data.suite.relative_budgets.join(", ")} × d.` : "This older report's relative budgets are derived from its recorded query caps."} Caps round up to whole queries; actual usage may be smaller. Cached lookups count as queries.`,
    ),
  );
  overview.append(
    protocolParagraph(
      "Players",
      "d counts what a coalition selects: features for prediction explanations, training examples or groups for valuation, models for ensemble selection, and tokens or image regions for text and images. It is not always the dataset's column count.",
    ),
  );
  overview.append(
    protocolParagraph(
      "Instances",
      `${replicationLabel()}. The construction seed controls the recorded data split, player selection and fitted model. These are separate from estimator randomness.`,
    ),
  );
  overview.append(
    protocolParagraph(
      "Prediction models",
      `${models.join(", ")}. These define the games; the estimators in the leaderboard approximate their Shapley values or interactions.`,
    ),
  );
  const provenance = protocolParagraph(
    "Reproduce",
    `Snapshot ${data.snapshot_id.slice(0, 12)}. `,
  );
  provenance.append(
    sourceLink("Source at preparation", sourceRoot()),
    document.createTextNode(" · "),
    sourceLink(
      "Configuration and adapters",
      sourceRoot() + "src/shapiq_benchmark/",
    ),
  );
  if (protocol?.source_url)
    provenance.append(
      document.createTextNode(" · "),
      sourceLink("Experiment configuration", protocol.source_url),
    );
  overview.append(provenance);

  const settings = new Map();
  data.games.forEach((game) => {
    const m = game.metadata || {};
    const key = JSON.stringify([
      m.dataset,
      m.game_kind || m.recipe || m.case_id || game.family,
      modelProfile(game),
      game.n_players,
    ]);
    if (!settings.has(key)) settings.set(key, []);
    settings.get(key).push(game);
  });
  const container = $("protocolSettings");
  container.replaceChildren();
  for (const games of settings.values()) {
    const game = games[0],
      m = game.metadata || {};
    const detail = document.createElement("details"),
      title = document.createElement("summary");
    detail.className = "protocolSetting";
    title.textContent = `${m.dataset || "No dataset recorded"} · ${(m.game_kind || m.recipe || m.case_id || game.family).replaceAll("_", " ")} · ${modelProfile(game)} · ${game.n_players} ${m.player_unit || "player"}${game.n_players === 1 ? "" : "s"}`;
    detail.append(title);
    detail.append(
      protocolParagraph(
        "Payoff",
        m.semantics ||
          "See the recorded construction in the reproduction download.",
      ),
    );
    if (m.output_scale)
      detail.append(protocolParagraph("Output", m.output_scale));
    if (m.preparation_hardware)
      detail.append(
        protocolParagraph(
          "Preparation hardware",
          [
            m.preparation_hardware.gpu_model ||
              m.preparation_hardware.cpu_model ||
              m.preparation_hardware.device,
            m.preparation_hardware.inference_precision,
            m.preparation_hardware.torch_version
              ? `PyTorch ${m.preparation_hardware.torch_version}`
              : null,
            m.preparation_hardware.cuda_version
              ? `CUDA ${m.preparation_hardware.cuda_version}`
              : null,
          ].filter(Boolean).join(" · ") +
            ". Cached-query time charges refer to this preparation hardware; estimator execution uses its separately recorded CPU.",
        ),
      );
    const rowCounts = [
      ["training_rows", "train_indices", "fitting"],
      ["validation_rows", "validation_indices", "validation"],
      ["test_rows", "test_indices", "held-out"],
      ["background_size", "background_indices", "background"],
    ].flatMap(([count, indices, label]) => {
      const counts = [
        ...new Set(
          games
            .map((g) => g.metadata?.[count] ?? g.metadata?.[indices]?.length)
            .filter(Number.isFinite),
        ),
      ].sort((a, b) => a - b);
      return counts.length ? [`${counts.join("/")} ${label} rows`] : [];
    });
    detail.append(
      protocolParagraph(
        "Data used",
        rowCounts.length
          ? rowCounts.join("; ") +
              ". Counts vary across instances where listed. Full row and column identities are in the reproduction download."
          : "Row counts and model settings were not included in this older website export. Use the reproduction download for the frozen recipe.",
      ),
    );
    if (m.dataset_target_note)
      detail.append(protocolParagraph("Dataset target", m.dataset_target_note));
    if (m.dataset_source)
      detail.append(protocolParagraph("Loader", m.dataset_source));
    const source = document.createElement("p");
    if (m.dataset_source_url)
      source.append(
        sourceLink("Underlying dataset", m.dataset_source_url),
        document.createTextNode(" · "),
      );
    if (m.dataset_source?.startsWith("shapiq"))
      source.append(
        sourceLink(
          "shapiq dataset loader",
          sourceRoot() + "src/shapiq_games/datasets/",
        ),
        document.createTextNode(" · "),
      );
    const module =
      typeof m.class === "string" &&
      /^shapiq(?:_games)?(?:\.[A-Za-z_][A-Za-z_0-9]*)+$/.test(m.class)
        ? `src/${m.class.split(".").slice(0, -1).join("/")}.py`
        : "src/shapiq_benchmark/games.py";
    source.append(sourceLink("Game implementation", sourceRoot() + module));
    detail.append(source);
    if (m.fourier_spectrum?.degree_mass) {
      const mass = m.fourier_spectrum.degree_mass;
      const percent = (value) => `${(100 * value).toFixed(1)}%`;
      detail.append(
        protocolParagraph(
          "Interaction structure",
          m.fourier_spectrum.constant
            ? "Constant payoff: no nonconstant Fourier energy."
            : `Uniform-coalition Fourier energy: ${percent(mass[1] || 0)} at order 1, ${percent((mass[2] || 0) + (mass[3] || 0))} at orders 2–3, ${percent(mass.slice(4).reduce((a, b) => a + b, 0))} at orders 4+. This describes the first listed frozen instance, not Shapley interaction indices.${m.stochastic_frozen ? " Frozen sampling noise can contribute to this spectrum." : ""}`,
        ),
      );
    }
    if (m.model_parameters) {
      const parameters = document.createElement("pre");
      parameters.textContent = JSON.stringify(m.model_parameters, null, 2);
      detail.append(
        protocolParagraph(
          "Model parameters",
          "Recorded parameters for the first listed construction instance; full per-instance metadata is in the download.",
        ),
        parameters,
      );
    }
    container.append(detail);
  }
}
