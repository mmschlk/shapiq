"use strict";

// Share the catalog renderers with the results page, without loading its charts.
let data, aboutController;
let aboutVersion = 0;
async function loadAbout(value, files = null) {
  aboutController?.abort();
  aboutController = new AbortController();
  const controller = aboutController,
    version = ++aboutVersion;
  if (value.layout === "partitioned-v1") {
    data = value;
    renderReportSummary(false);
    $("aboutStatus").textContent =
      "Loading the recorded game, dataset and model details…";
    value = await BenchmarkAbout.load(value, {
      signal: controller.signal,
      read: (descriptor) =>
        BenchmarkPartitions.read(value, descriptor, {
          files,
          signal: controller.signal,
        }),
    });
    if (version !== aboutVersion) return;
  }
  if (
    value.schema_version !== 1 ||
    !value.games?.length ||
    !value.suite?.seeds?.length ||
    typeof value.snapshot_id !== "string"
  )
    throw Error("Choose an exported benchmark JSON report.");
  data = value;
  renderReportSummary(false);
  renderGames();
  renderDatasets();
  renderModels();
  renderRunSettings();
  $("aboutStatus").textContent =
    "These tables describe the loaded report. Planned extensions are not measured results.";
}

const players = $("playerCount");
function showCoalitions() {
  $("playerCountLabel").textContent = players.value;
  $("coalitionCount").textContent = (2 ** Number(players.value)).toLocaleString(
    "en-US",
  );
}
players.addEventListener("input", showCoalitions);
showCoalitions();

// An explicit local selection takes precedence over a slower bundled download.
let localSelected = false,
  uploadVersion = 0;
$("aboutUpload").addEventListener("change", async (event) => {
  const files = new Map(
    [...event.target.files].map((file) => [file.name, file]),
  );
  const file =
    files.get("data.json") || files.get("about.json") || event.target.files[0];
  if (!file) return;
  localSelected = true;
  aboutController?.abort();
  const version = ++aboutVersion,
    upload = ++uploadVersion;
  try {
    const value = JSON.parse(await file.text());
    if (version !== aboutVersion) return;
    await loadAbout(value, files);
  } catch (error) {
    if (upload === uploadVersion && error.name !== "AbortError")
      $("aboutStatus").textContent = error.message;
  }
});
fetch("about.json")
  .then((response) => {
    if (!response.ok) throw Error("Report metadata is unavailable.");
    return response.json();
  })
  .then((value) => {
    if (!localSelected) return loadAbout(value);
  })
  .catch(() => {
    if (!localSelected)
      $("aboutStatus").textContent =
        "Report tables could not be loaded. The guide below still applies; open a local exported JSON report to see its settings.";
  });
