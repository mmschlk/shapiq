"use strict";

// Share the catalog renderers with the results page, without loading its charts.
let data;
function loadAbout(value) {
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
let localSelected = false;
$("aboutUpload").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  localSelected = true;
  try {
    loadAbout(JSON.parse(await file.text()));
  } catch (error) {
    $("aboutStatus").textContent = error.message;
  }
});
fetch("about.json")
  .then((response) => {
    if (!response.ok) throw Error("Report metadata is unavailable.");
    return response.json();
  })
  .then((value) => {
    if (!localSelected) loadAbout(value);
  })
  .catch(() => {
    if (!localSelected)
      $("aboutStatus").textContent =
        "Report tables could not be loaded. The guide below still applies; open a local exported JSON report to see its settings.";
  });
