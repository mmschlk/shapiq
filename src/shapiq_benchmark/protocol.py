"""Construction and reference adapters for the compact benchmark manifest."""

from __future__ import annotations

from shapiq_benchmark.families import FAMILY_CATALOG, dataset_compatibility, profile_compatibility

BUDGET_MULTIPLIERS = (0.5, 1, 2, 4, 8, 16, 32)
GAME_SEEDS = (0, 1, 2, 3)
_MODELS = {
    "local_baseline": ("linear", "random_forest"),
    "feature_selection": ("linear", "random_forest"),
    "dataset_valuation": ("linear", "random_forest"),
    "pathdependent_tree": ("random_forest",),
    "interventional_tree": ("random_forest",),
    "knn": ("knn",),
    "tnn": ("tnn",),
}
CONSTRUCTIONS = {
    name: {"models": list(models), "player_unit": FAMILY_CATALOG[name]["player_unit"]}
    for name, models in _MODELS.items()
}
_STRUCTURED = {"pathdependent_tree", "interventional_tree", "knn", "tnn"}


def adapter_reason(family: str, dataset: str, model: str, *, structured: bool) -> str | None:
    """Reject unsupported model/reference combinations before preparation."""
    if reason := dataset_compatibility(family, dataset):
        return reason
    if model not in CONSTRUCTIONS[family]["models"]:
        return "Unsupported construction/model pair."
    if structured != (family in _STRUCTURED):
        return "Construction does not match its exact-reference adapter."
    return None if family in {"knn", "tnn"} else profile_compatibility(family, dataset, model)


def recipe_spec(row: dict, targets: list[dict]) -> list[dict]:
    """Expand a recipe into the targets supported by its exact reference."""
    family, model = row["family"], row["model_profile"]
    common = {key: row[key] for key in ("id", "n_players", "dataset")}
    if row["reference"] != "structured":
        return [{**common, "family": family, "model_profile": model}]
    oracle = "tree" if family == "interventional_tree" else family
    supported = (
        targets
        if family == "interventional_tree"
        else [target for target in targets if target["index"] in ("SV", "k-SII", "SII")]
        if family == "pathdependent_tree"
        else [target for target in targets if (target["index"], target["order"]) == ("SV", 1)]
    )
    return [
        {
            **common,
            "id": f"{row['id']}-{target['index'].lower()}-{target['order']}",
            "oracle": oracle,
            **target,
            **(
                {"row_selection": "stratified"}
                if family in {"knn", "tnn"}
                else {"model_profile": model}
            ),
        }
        for target in supported
    ]
