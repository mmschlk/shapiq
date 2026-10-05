"use strict";
const $ = (id) => document.getElementById(id);
const replicationLabel = () =>
  data.suite.game_seeds?.length
    ? `${data.suite.game_seeds.length} game instances per setting · ${data.suite.seeds.length} estimator run${data.suite.seeds.length === 1 ? "" : "s"} per instance and budget`
    : `${data.suite.seeds.length} estimator seeds per game`;
const targetLabel = (value) =>
  ({
    SV: "Shapley Values",
    "k-SII": "Shapley Interaction Indices (k-SII)",
    SII: "Shapley Interaction Indices (SII)",
    STII: "Shapley–Taylor Interactions (STII)",
    FSII: "Faithful Shapley Interactions (FSII)",
    FBII: "Faithful Banzhaf Interactions (FBII)",
  })[value.split(" · ")[0]] || value;
const target = (game) => `${game.index} · order ${game.order}`;

// Describe the loaded snapshot, never planned games from a newer roadmap.
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
const distinct = (values) => [
  ...new Set(values.filter((v) => v !== undefined && v !== null && v !== "")),
];
const readable = (name) => name.replaceAll("_", " ");
function grouped(games, key) {
  const groups = new Map();
  for (const game of games) {
    const name = key(game);
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(game);
  }
  return groups;
}
function protocolParagraph(label, text) {
  const paragraph = document.createElement("p"),
    heading = document.createElement("strong");
  heading.textContent = `${label}. `;
  paragraph.append(heading, document.createTextNode(text));
  return paragraph;
}
function protocolTable(id, headings, rows) {
  const container = $(id),
    wrapper = document.createElement("div"),
    table = document.createElement("table"),
    head = document.createElement("thead"),
    body = document.createElement("tbody");
  wrapper.className = "provenanceTableWrap";
  wrapper.tabIndex = 0;
  wrapper.setAttribute("role", "region");
  wrapper.setAttribute("aria-label", headings.join(", "));
  table.className = "provenanceTable";
  const header = document.createElement("tr");
  headings.forEach((label) => {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    header.append(cell);
  });
  head.append(header);
  rows.forEach((values) => {
    const row = document.createElement("tr");
    values.forEach((value, index) => {
      const cell = document.createElement(index ? "td" : "th");
      if (!index) cell.scope = "row";
      cell.append(
        value instanceof Node ? value : document.createTextNode(String(value)),
      );
      row.append(cell);
    });
    body.append(row);
  });
  table.append(head, body);
  wrapper.append(table);
  container.replaceChildren(wrapper);
}
function lines(...items) {
  const block = document.createElement("div");
  for (const item of items.filter(Boolean)) {
    const line = document.createElement("div");
    line.append(item instanceof Node ? item : document.createTextNode(item));
    block.append(line);
  }
  return block;
}
const gameKind = (game) => {
  const m = game.metadata || {};
  return (
    m.game_kind ||
    m.recipe ||
    {
      InterventionalTreeSHAPIQ: "interventional_tree",
      KNNExplainer: "knn",
      ProductKernelExplainer: "product_kernel",
    }[m.truth_method] ||
    m.case_id ||
    game.family
  );
};
const gameName = (game) =>
  gameDescriptions[gameKind(game)]?.[0] || readable(gameKind(game));
function adapterPath(game) {
  if (game.metadata?.model_profile) return "src/shapiq_benchmark/models.py";
  if (
    ["text", "image", "tabpfn", "causal_local", "causal_global"].includes(
      gameKind(game),
    )
  )
    return "src/shapiq_benchmark/media.py";
  return game.metadata?.class
    ? "src/shapiq_benchmark/families.py"
    : "src/shapiq_benchmark/games.py";
}
function implementation(game) {
  const cls = game.metadata?.class;
  const module =
    typeof cls === "string" &&
    /^shapiq(?:_games)?(?:\.[A-Za-z_][A-Za-z_0-9]*)+$/.test(cls)
      ? `src/${cls.split(".").slice(0, -1).join("/")}.py`
      : adapterPath(game);
  return sourceLink(
    cls?.split(".").at(-1) || "Benchmark adapter",
    sourceRoot() + module,
  );
}
const gameDescriptions = {
  local_baseline: [
    "Baseline feature removal",
    "Explain a fitted prediction model by replacing absent features with fixed mean values.",
  ],
  local_baseline_forest: [
    "Forest baseline feature removal",
    "Explain a fitted forest using fixed mean values for absent features.",
  ],
  local_marginal: [
    "Marginal feature removal",
    "Average predictions over fixed background rows when features are absent.",
  ],
  local_gaussian: [
    "Gaussian conditional explanations",
    "Fill absent features using a fitted conditional Gaussian distribution.",
  ],
  local_copula: [
    "Gaussian-copula explanations",
    "Use a Gaussian copula to model dependencies when filling absent features.",
  ],
  local_conditional: [
    "Tree-based conditional explanations",
    "Use tree embeddings to sample absent features conditionally.",
  ],
  global_fidelity: [
    "Global prediction fidelity",
    "Measure how well selected features preserve the full model’s predictions across examples.",
  ],
  feature_selection: [
    "Feature selection",
    "Retrain on selected features and measure held-out regression error or classification accuracy.",
  ],
  data_valuation: [
    "Training-example valuation",
    "Retrain on selected training examples and measure held-out regression error or classification accuracy.",
  ],
  dataset_valuation: [
    "Dataset-group valuation",
    "Treat groups of training rows as players and evaluate models trained on selected groups.",
  ],
  ensemble: [
    "Ensemble selection",
    "Combine selected models by averaging regression predictions or voting on classes, then measure held-out performance.",
  ],
  forest_ensemble: [
    "Forest ensemble selection",
    "Treat individual trees as players and evaluate their average regression predictions or class votes.",
  ],
  uncertainty: [
    "Prediction uncertainty",
    "Explain how features affect a forest’s predictive entropy.",
  ],
  cluster: [
    "Clustering",
    "Measure the quality of K-means clusters formed from selected features.",
  ],
  unsupervised: [
    "Feature dependence",
    "Measure total correlation among selected, discretized features.",
  ],
  pathdependent_tree: [
    "Path-dependent tree explanations",
    "Explain tree predictions using training-path frequencies for absent features.",
  ],
  interventional_tree: [
    "Interventional tree explanations",
    "Explain tree predictions by averaging over reference rows; larger cases use exact tree ground truth.",
  ],
  product_kernel: [
    "Product-kernel explanations",
    "Explain RBF support-vector scores by including selected feature factors; larger cases have exact kernel ground truth.",
  ],
  knn: [
    "Nearest-neighbor valuation",
    "Treat training examples as players and score the selected nearest neighbors’ agreement with the test label, including larger games with exact ground truth.",
  ],
  tnn: [
    "Threshold-neighbor valuation",
    "Score label agreement among selected training examples within a fixed distance of a test point.",
  ],
  weighted_knn: [
    "Weighted nearest-neighbor valuation",
    "Evaluate selected training examples using distance-weighted class votes.",
  ],
  binary_weighted_knn: [
    "Binary weighted-neighbor valuation",
    "Compare distance-weighted votes for the explained class against a fixed alternative class.",
  ],
  unanimity: [
    "Unanimity games",
    "Synthetic checks where a coalition scores only when it contains a designated group of players.",
  ],
  soum: [
    "Sums of unanimity games",
    "Synthetic combinations of fixed player groups, used as controlled interaction checks.",
  ],
  dummy: [
    "Dummy-player checks",
    "Synthetic additive contributions plus a fixed interaction test which players affect each term.",
  ],
  random: [
    "Random games",
    "Synthetic random coalition payoffs are frozen into a reproducible table.",
  ],
  text: [
    "Text explanations",
    "Mask tokens in four authored sentences and explain an IMDb-trained DistilBERT sentiment model.",
  ],
  image: [
    "Image explanations",
    "Mask superpixels in four bundled images and explain a pretrained ResNet18 classifier.",
  ],
  tabpfn: [
    "TabPFN explanations",
    "Remove features from a TabPFN classifier’s training context and explain a class probability.",
  ],
  causal_global: [
    "Global confounding explanations",
    "Use simulated treatments and outcomes to attribute confounding across a dataset.",
  ],
  causal_local: [
    "Local confounding explanations",
    "Use the same causal simulation to attribute confounding for one example.",
  ],
};
function legacyDatasetSources() {
  const sklearn =
    "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.";
  return {
    adult_census: [
      "Census demographics and whether annual income exceeds $50,000",
      "https://archive.ics.uci.edu/dataset/2/adult",
    ],
    mushroom: [
      "Mushroom characteristics and edible / poisonous labels",
      "https://archive.ics.uci.edu/dataset/73/mushroom",
    ],
    ionosphere: [
      "Radar measurements and good / bad ionosphere returns",
      "https://archive.ics.uci.edu/dataset/52/ionosphere",
    ],
    wine_quality: [
      "Red and white wine measurements and quality scores",
      "https://archive.ics.uci.edu/dataset/186/wine+quality",
    ],
    communities_and_crime: [
      "Community demographics and violent-crime rates",
      "https://shap.readthedocs.io/en/latest/generated/shap.datasets.communitiesandcrime.html",
    ],
    nhanesi: [
      "NHANES I health survey and survival follow-up",
      "https://shap.readthedocs.io/en/latest/generated/shap.datasets.nhanesi.html",
    ],
    "synthetic diagnostic": [
      "Generated coalition payoffs; no external dataset",
      sourceRoot() + "src/shapiq_games/synthetic/",
    ],
    california_housing: [
      "California Housing census features and house values",
      `${sklearn}fetch_california_housing.html`,
    ],
    iris: ["Iris flower measurements and species", `${sklearn}load_iris.html`],
    diabetes: [
      "Diabetes measurements and disease progression",
      `${sklearn}load_diabetes.html`,
    ],
    bike_sharing: [
      "Bike Sharing weather, calendar features and rental counts",
      "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
    ],
    wine: [
      "Wine chemical measurements and classes",
      `${sklearn}load_wine.html`,
    ],
    breast_cancer: [
      "Wisconsin breast-cancer measurements and labels",
      `${sklearn}load_breast_cancer.html`,
    ],
    digits: ["Handwritten digit images", `${sklearn}load_digits.html`],
    "authored sentiment examples": [
      "Four authored sentiment sentences",
      `${sourceRoot()}src/shapiq_benchmark/media.py`,
    ],
    "ImageNet bundled examples": [
      "Bundled ImageNet images",
      "https://www.image-net.org/",
    ],
    "Curth-VDS synthetic": [
      "Curth-VDS simulated causal data",
      `${sourceRoot()}src/shapiq_games/benchmark/causal_xai/benchmark.py`,
    ],
  };
}
function renderGames() {
  const rows = [...grouped(data.games, gameKind)].map(([kind, games]) => {
    const values = (key) => distinct(games.map((g) => g.metadata?.[key]));
    const spectra = [
      ...grouped(
        games.filter((g) => g.metadata?.fourier_spectrum),
        (g) => g.metadata.case_id + ":" + g.metadata.instance_seed,
      ).values(),
    ].map((group) => group[0].metadata.fourier_spectrum);
    const fourthOrder = spectra.map((s) =>
      s.degree_mass.slice(4).reduce((a, b) => a + b, 0),
    );
    const spectrum = fourthOrder.length
      ? `Order 4+ Fourier energy: ${(100 * Math.min(...fourthOrder)).toFixed(1)}–${(100 * Math.max(...fourthOrder)).toFixed(1)}% across recorded instances.`
      : "";
    return [
      gameName(games[0]),
      lines(
        gameDescriptions[kind]?.[1] ||
          values("semantics").join("; ") ||
          "Frozen coalition payoffs.",
        spectrum,
      ),
      lines(
        `${distinct(games.map((g) => g.n_players))
          .sort((a, b) => a - b)
          .join(", ")} players`,
        values("player_unit").join(" / ") || "Player unit not recorded",
      ),
      lines(
        ...values("truth_method"),
        ...distinct(games.map((g) => g.metadata?.class || adapterPath(g))).map(
          (cls) =>
            implementation(
              games.find((g) => (g.metadata?.class || adapterPath(g)) === cls),
            ),
        ),
      ),
    ];
  });
  protocolTable(
    "protocolGames",
    [
      "Game construction",
      "What the coalition is worth",
      "Players",
      "Exact reference & source",
    ],
    rows,
  );
}
function renderDatasets() {
  const legacy = legacyDatasetSources();
  const rows = [
    ...grouped(data.games, (g) => g.metadata?.dataset || "No dataset recorded"),
  ].map(([name, games]) => {
    const declared = data.suite.protocol?.datasets?.find((d) => d.id === name);
    const values = (key) => distinct(games.map((g) => g.metadata?.[key]));
    const loader = values("dataset_source")[0] || declared?.source;
    let loaderURL;
    if (loader?.startsWith("sklearn."))
      loaderURL = `https://scikit-learn.org/stable/modules/generated/${loader}.html`;
    else if (loader?.startsWith("shapiq"))
      loaderURL =
        sourceRoot() +
        (loader.startsWith("shapiq_games.")
          ? "src/shapiq_games/datasets/_all.py"
          : "src/shapiq/datasets/_all.py");
    const origin =
      values("dataset_source_url")[0] ||
      declared?.source_url ||
      legacy[name]?.[1];
    const counts = [
      ["training_rows", "train_indices", "fitting"],
      ["validation_rows", "validation_indices", "validation"],
      ["test_rows", "test_indices", "held-out"],
      ["background_size", "background_indices", "background"],
    ].flatMap(([count, indices, label]) => {
      const numbers = distinct(
        games.map((g) => g.metadata?.[count] ?? g.metadata?.[indices]?.length),
      ).sort((a, b) => a - b);
      return numbers.length ? [`${numbers.join(" / ")} ${label} rows`] : [];
    });
    return [
      declared?.label || readable(name),
      lines(
        legacy[name]?.[0],
        declared?.task
          ? `${declared.task}${declared.n_features ? ` · ${declared.n_features} source columns` : ""}`
          : "",
        ...values("dataset_target_note"),
        ...values("dataset_preprocessing"),
      ),
      counts.length
        ? counts.join("; ")
        : "Row counts not recorded in this export.",
      lines(
        loaderURL
          ? sourceLink(loader.split(".").at(-1), loaderURL)
          : sourceLink(
              "Recorded data adapter",
              sourceRoot() + adapterPath(games[0]),
            ),
        origin
          ? sourceLink("Original data", origin)
          : "Original source not recorded.",
      ),
    ];
  });
  protocolTable(
    "protocolDatasets",
    [
      "Dataset / input source",
      "Data & target",
      "Rows used across settings",
      "Sources",
    ],
    rows,
  );
}
function renderModels() {
  const rows = [...grouped(data.games, modelProfile)].map(([name, games]) => {
    const values = (key) => distinct(games.map((g) => g.metadata?.[key]));
    const parameters = new Map();
    games.forEach((g) =>
      Object.entries(g.metadata?.model_parameters || {}).forEach(
        ([key, value]) => {
          if (key === "random_state") return;
          if (!parameters.has(key)) parameters.set(key, []);
          parameters
            .get(key)
            .push(value === null ? "unlimited" : String(value));
        },
      ),
    );
    const settings = [...parameters].map(
      ([key, values]) => `${readable(key)}: ${distinct(values).join(" / ")}`,
    );
    const output = values("output_scale");
    return [
      lines(
        readable(name),
        ...values("model").filter((model) => model !== name),
      ),
      lines(
        ...settings,
        !settings.length ? "Parameters not recorded in this export." : "",
      ),
      output.length
        ? output.join("; ")
        : "See the game’s payoff rule; output scale was not separately recorded.",
      lines(
        ...distinct(games.map(adapterPath)).map((path) =>
          sourceLink("Model setup in shapiq", sourceRoot() + path),
        ),
      ),
    ];
  });
  protocolTable(
    "protocolModels",
    [
      "Model used by the game",
      "Recorded parameters",
      "Explained output",
      "Source",
    ],
    rows,
  );
}
function renderRunSettings() {
  const protocol = data.suite.protocol;
  const settings = $("protocolSettings");
  settings.replaceChildren(
    protocolParagraph(
      "Budgets",
      `${data.suite.relative_budgets?.length ? data.suite.relative_budgets.join(", ") + " × d" : "This report’s relative budgets are derived from its recorded query caps"}. Caps round up to whole queries; actual usage may be smaller. Cached lookups count as queries.`,
    ),
    protocolParagraph(
      "Replication",
      `${replicationLabel()}. Game seeds: ${data.suite.game_seeds?.join(", ") || "not recorded"}; estimator seeds: ${data.suite.seeds.join(", ")}. Game seeds determine the recorded data split, player selection and model fit, where applicable.`,
    ),
    protocolParagraph(
      "Explanation targets",
      distinct(data.games.map((g) => target(g)))
        .map(targetLabel)
        .join("; ") +
        ". Shapley values assign scores to players; interaction indices score groups of players.",
    ),
    protocolParagraph(
      "Ground truth",
      protocol?.exact_reference ||
        "Exact references use the methods listed under Games. Enumerated games share a frozen payoff table; larger structured games use their recorded exact solver.",
    ),
  );
  if (protocol?.minimum_players)
    settings.append(
      protocolParagraph(
        "Eligibility",
        `At least ${protocol.minimum_players} players; enumeration up to ${protocol.maximum_enumerated_players}. ${protocol.minimum_signal_ratio ? `RMS ground-truth attribution / payoff standard deviation must be at least ${protocol.minimum_signal_ratio}. ` : ""}Structured games may use a certified lower bound on this ratio; the exact definition is recorded in their metadata. Only compatible dataset, model and construction combinations are scheduled.`,
      ),
    );
  if (Object.keys(data.suite.method_parameters || {}).length)
    settings.append(
      protocolParagraph(
        "Estimator settings",
        Object.entries(data.suite.method_parameters)
          .map(
            ([name, values]) =>
              `${name}: ${Object.entries(values)
                .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
                .join(", ")}`,
          )
          .join("; ") +
          ". Other constructor settings use their recorded source defaults.",
      ),
    );
  const exclusions = data.suite.preparation_exclusions || [];
  const preflight = data.suite.preparation_preflight;
  if (preflight) {
    settings.append(
      protocolParagraph(
        "Preparation checks",
        `${exclusions.length} recipe configurations excluded before estimator evaluation. The projected construction and enumeration limit is ${preflight.maximum_seconds_per_instance / 3600} hours per game instance; exact-solver cost is additional. All four seeds must pass. Projections are estimates, not measured full-run times.`,
      ),
    );
    if (exclusions.length) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "Excluded preparation configurations";
      details.append(summary);
      for (const row of exclusions) {
        details.append(
          protocolParagraph(
            row.spec.id,
            row.reason === "projected_cost_limit"
              ? "Projected preparation cost exceeds the declared limit."
              : "A construction, payoff validation or pilot time limit failed; see the reproduction archive for per-seed outcomes.",
          ),
        );
      }
      settings.append(details);
    }
  }
  if (protocol?.hardware)
    settings.append(protocolParagraph("Hardware", protocol.hardware));
  const prep = distinct(
    data.games.map((g) => {
      const h = g.metadata?.preparation_hardware;
      return h
        ? [
            h.gpu_model || h.cpu_model || h.device,
            h.inference_precision,
            h.torch_version && `PyTorch ${h.torch_version}`,
            h.cuda_version && `CUDA ${h.cuda_version}`,
          ]
            .filter(Boolean)
            .join(" · ")
        : null;
    }),
  );
  if (prep.length)
    settings.append(
      protocolParagraph(
        "Preparation hardware",
        prep.join("; ") +
          ". Cached-query charges retain this hardware; estimator CPU timing is recorded separately.",
      ),
    );
  const reproduce = protocolParagraph(
    "Reproduce",
    `Snapshot ${data.snapshot_id.slice(0, 12)}. `,
  );
  reproduce.append(
    sourceLink("Frozen source", sourceRoot()),
    document.createTextNode(" · "),
    sourceLink("Game adapters", sourceRoot() + "src/shapiq_benchmark/"),
  );
  settings.append(reproduce);
  const pairs = [
    ...grouped(data.games, (g) =>
      JSON.stringify([
        gameKind(g),
        g.metadata?.dataset,
        modelProfile(g),
        g.n_players,
      ]),
    ),
  ].map(([, games]) => [
    gameName(games[0]),
    readable(games[0].metadata?.dataset || "No dataset recorded"),
    readable(modelProfile(games[0])),
    games[0].n_players,
  ]);
  protocolTable(
    "protocolCombinations",
    ["Construction", "Dataset / input", "Model", "d"],
    pairs,
  );
}
function renderReportSummary(linkToAbout = true) {
  const protocol = data.suite.protocol;
  const datasets =
    data.catalog?.datasets ||
    distinct(data.games.map((g) => g.metadata?.dataset));
  const players = data.catalog
    ? [data.catalog.min_players, data.catalog.max_players]
    : data.games.map((g) => g.n_players);
  const summary = $("protocolSummary");
  summary.replaceChildren(
    document.createTextNode(
      `${protocol?.name || "Earlier benchmark preview"} · ${datasets.length} data sources · ${Math.min(...players)}–${Math.max(...players)} players. `,
    ),
  );
  const excluded =
    data.catalog?.preparation_exclusion_count ??
    data.suite.preparation_exclusions?.length ??
    0;
  if (excluded)
    summary.append(
      document.createTextNode(
        `${excluded} preparation configurations excluded; reasons in About. `,
      ),
    );
  if (linkToAbout) {
    const jump = document.createElement("a");
    jump.href = "about.html";
    jump.textContent = "About the benchmark ↗";
    summary.append(jump);
  }
}
