"""Join authenticated retries of transient dataset outages, before alias removal."""

from __future__ import annotations

from shapiq_benchmark.runner import identity

CONFIG = (
    "protocol",
    "relative_budgets",
    "seeds",
    "game_seeds",
    "methods",
    "targets",
    "min_players",
    "min_signal_ratio",
    "method_parameters",
    "duplicate_registry",
)
TRANSIENT_HTTP = {"Gateway Time-out", "Bad Gateway", "Service Unavailable", "Request Timeout"}


def merge_recovery(data: dict, context: dict, extra: dict, retry: dict) -> None:
    """Accept only unchanged recipes previously excluded by a transient HTTP error.

    Both callers have already authenticated their complete campaign artifacts.
    Original exclusions remain in the authenticated component; the public panel
    replaces retried outage entries with the retry's outcome, including any new
    scientific exclusion. It never treats a failed qualification as a success.
    """

    def require(condition: bool, message: str) -> None:  # noqa: FBT001 -- assertion helper
        if not condition:
            raise ValueError(message)

    require(context["root"] != retry["root"], "Cannot supplement a campaign with itself")
    for key in ("source", "scripts"):
        require(
            context["campaign"].get(key) == retry["campaign"].get(key),
            f"Supplement has incompatible {key}",
        )
    ids = {key[1] for key in retry["requested"]}
    previous = set(context.setdefault("recovered_ids", []))
    require(bool(ids) and not ids & previous, "Empty or repeated recovery recipes")
    for key, requested in retry["requested"].items():
        parent = context["requested"].get(key)
        require(parent is not None, "Supplement contains an unrelated recipe")
        require(parent["spec"] == requested["spec"], "Supplement changed recipe specification")
        for setting in CONFIG:
            require(
                parent["suite"].get(setting) == requested["suite"].get(setting),
                f"Supplement has incompatible {setting}",
            )
        exclusion = context["exclusions"].get(key[1], {})
        instances = exclusion.get("instances", [])
        require(
            exclusion.get("reason") == "preflight_failed"
            and len(instances) == len(parent["suite"]["game_seeds"])
            and {item.get("seed") for item in instances} == set(parent["suite"]["game_seeds"])
            and all(
                item.get("status") == "failed"
                and item.get("error_type") == "HTTPError"
                and item.get("reason") in TRANSIENT_HTTP
                for item in instances
            ),
            "Supplement may only retry complete transient HTTP exclusions",
        )
    require(
        not {game["id"] for game in data["games"]} & {game["id"] for game in extra["games"]},
        "Overlapping actual game IDs in supplement",
    )
    for name, method in extra["methods"].items():
        require(
            name not in data["methods"] or data["methods"][name] == method,
            "Conflicting method sources",
        )
    suite, extra_suite = data["suite"], extra["suite"]
    for key in ("games", "records", "coverage"):
        data[key].extend(extra[key])
    data["methods"].update(extra["methods"])
    for key, run in extra["runs"].items():
        require(key not in data["runs"] or data["runs"][key] == run, "Conflicting run provenance")
        data["runs"][key] = run
    suite["preparation_exclusions"] = [
        row for row in suite["preparation_exclusions"] if row["spec"]["id"] not in ids
    ] + extra_suite["preparation_exclusions"]
    for name in ("preparation_preflight",):
        if name in extra_suite:
            current = suite.setdefault(name, {**extra_suite[name], "families": []})
            require(
                {k: v for k, v in current.items() if k != "families"}
                == {k: v for k, v in extra_suite[name].items() if k != "families"},
                "Supplement changed preparation gate",
            )
            current["families"] = [row for row in current["families"] if row["id"] not in ids]
            current["families"].extend(extra_suite[name].get("families", []))
    suite["budgets_by_game"].update(extra_suite["budgets_by_game"])
    suite["budgets"] = sorted({b for grid in suite["budgets_by_game"].values() for b in grid})
    context["fingerprints"].update(retry["fingerprints"])
    context["recovered_ids"] = sorted(previous | ids)
    data["composition"].setdefault("supplements", []).append(
        {
            "parent_campaign_sha256": identity(context["campaign"]),
            "retried_recipe_ids": sorted(ids),
            "composition": extra["composition"],
        }
    )
