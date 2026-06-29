"""UI components for the leaderboard."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

from leaderboard.metrics import METRICS
from leaderboard.scoring.elo_scorer import EloScorer
from leaderboard.storage.connection import DatabaseClientFactory, DBConnectionError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()
T = TypeVar("T")

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent
RESULTS_PATH = PROJECT_ROOT / "data" / "results_raw.jsonl"

ZERO_THRESHOLD = 1e-7

COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]
DASH_STYLES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
GLOBAL_APPROX_STYLES = {}

LOADING_METHOD = "mongodb"  # "local" or "mongodb" or "huggingface"

# ELO Leaderboard: Budget-Buckets
BUDGET_BUCKETS = [
    {"label": "Low (250)", "budget": 250},
    {"label": "Low-Medium (500)", "budget": 500},
    {"label": "Medium (1000)", "budget": 1000},
    {"label": "Medium-High (5k)", "budget": 5000},
    {"label": "High (10k)", "budget": 10000},
]


# Temporary seed determination
SEED_IDs = ["approx_seed", "seed"]  # List of possible seed identifier columns in the raw data

# Compare-Tab: Globale Variablen
MAX_COLS = 5
DEFAULT_COLS = 2


def update_global_styles(df: pd.DataFrame) -> None:
    """Fill the global approximator-style mapping from the full dataset.

    Assigns a unique, stable color and dash style to every approximator so
    that each retains consistent visual identity across all plots.

    Args:
        df: Aggregated DataFrame containing at least an ``approximator_name`` column.
    """
    all_approxs = sorted(df["approximator_name"].unique().tolist())

    GLOBAL_APPROX_STYLES.clear()

    for idx, approx in enumerate(all_approxs):
        GLOBAL_APPROX_STYLES[approx] = {
            "color": COLORS[idx % len(COLORS)],
            "dash": DASH_STYLES[idx % len(DASH_STYLES)],
        }


def reload_data() -> pd.DataFrame:
    """Reload the raw data from the configured source and re-aggregate it.

    Returns:
        Updated aggregated DataFrame ready for the leaderboard UI.
    """
    return load_and_aggregate(method=LOADING_METHOD, path=RESULTS_PATH)


def load_and_aggregate(method: str = "mongodb", path: str | Path = RESULTS_PATH) -> pd.DataFrame:
    """Load raw run records from the specified backend and return an aggregated DataFrame.

    Args:
        method: Database backend identifier (``"mongodb"`` or ``"local"``).
        path: Path to the local JSONL file; only used when *method* is ``"local"``.

    Returns:
        Aggregated DataFrame with one row per (approximator, game, budget) combination.

    Raises:
        DBConnectionError: If the database connection test fails.
    """
    db_client = DatabaseClientFactory.create_client(
        method, db_args={"LOCAL_DB_PATH": str(path)} if method == "local" else {}
    )
    if not db_client.test_connection():
        raise DBConnectionError from None

    return db_client.load_dataframe()


def format_value[T](col: str, x: T, runtime_cols: list[str]) -> str:
    """Formats a value for display in the leaderboard, handling small values and runtime formatting.

    Args:
        col: The column name (used to determine if it's a runtime column).
        x: The value to format.
        runtime_cols: List of column names that represent runtimes, which should be rounded to 4 decimals.

    Returns:
        A formatted string representation of the value.
    """
    if not isinstance(x, float):
        return str(x)
    if abs(x) < ZERO_THRESHOLD and col not in runtime_cols:
        return "0"
    if col in runtime_cols:
        return f"{x:.4f}"
    return f"{x:.4e}"


def _build_leaderboard(df: pd.DataFrame, selected_metrics: list[str]) -> pd.DataFrame:
    """Build a formatted leaderboard DataFrame with human-readable column names.

    Selects the best budget per approximator (lowest MSE), orders and renames
    columns, and formats all numeric values for display.

    Args:
        df: Aggregated DataFrame containing metric mean/std columns and runtime columns.
        selected_metrics: Metric names to include (e.g. ``["mse", "mae"]``).

    Returns:
        Formatted leaderboard DataFrame sorted by ascending MSE.
    """
    avail_metrics = sorted([m for m in selected_metrics if f"{m}_mean" in df.columns])

    # Dynamic rename-Map
    rename_map = {
        "approximator_name": "Approximator",
        "budget": "Budget at best MSE",
        "ground_truth_method": "GT Method",
        "runtime_mean": "Runtime mean (s)",
        "runtime_min": "Runtime min (s)",
        "runtime_max": "Runtime max (s)",
        "n_seeds": "Seeds",
    }

    for m in avail_metrics:
        rename_map[f"{m}_mean"] = f"{m.upper()} (mean)"
        rename_map[f"{m}_std"] = f"{m.upper()} (std)"

    best = df.loc[df.groupby("approximator_name")["mse_mean"].idxmin()]
    best = best.sort_values("mse_mean").rename(columns=rename_map)

    runtime_cols = ["Runtime mean (s)", "Runtime min (s)", "Runtime max (s)"]
    metric_cols = [c for m in avail_metrics for c in [f"{m.upper()} (mean)", f"{m.upper()} (std)"]]

    col_order = [
        "Approximator",
        "Budget at best MSE",
        *metric_cols,
        "GT Method",
        "Runtime mean (s)",
        "Runtime min (s)",
        "Runtime max (s)",
        "Seeds",
    ]
    col_order = [c for c in col_order if c in best.columns]

    leaderboard_df = best[col_order].copy()
    for col in leaderboard_df.columns:
        leaderboard_df[col] = leaderboard_df[col].apply(
            lambda x, col=col: format_value(col, x, runtime_cols)
        )
    return leaderboard_df


def get_leaderboard_global(
    df_agg: pd.DataFrame, selected_approxs: list[str], selected_metrics: list[str]
) -> pd.DataFrame:
    """Compute the global leaderboard by aggregating metrics across all games.

    Groups by approximator and budget, averages metric means and stds over all
    games, then delegates to ``_build_leaderboard`` for formatting.

    Args:
        df_agg: Aggregated DataFrame produced by ``load_and_aggregate``.
        selected_approxs: Approximator names to include.
        selected_metrics: Metric names to display.

    Returns:
        Formatted leaderboard DataFrame covering all games.
    """
    df_filtered = df_agg[df_agg["approximator_name"].isin(selected_approxs)]

    # Aggregate over all games
    global_agg = (
        df_filtered.groupby(["approximator_name", "budget"])
        .agg(
            **{
                **{
                    f"{m}_mean": (f"{m}_mean", "mean")
                    for m in METRICS
                    if f"{m}_mean" in df_agg.columns
                },
                **{
                    f"{m}_std": (f"{m}_std", "mean")
                    for m in METRICS
                    if f"{m}_std" in df_agg.columns
                },
                "ground_truth_method": ("ground_truth_method", "first"),
                "runtime_mean": ("runtime_mean", "mean"),
                "runtime_min": ("runtime_min", "min"),
                "runtime_max": ("runtime_max", "max"),
                "n_seeds": ("n_seeds", "sum"),
            }
        )
        .reset_index()
    )

    return _build_leaderboard(global_agg, selected_metrics)


def get_leaderboard_game(
    df_agg: pd.DataFrame,
    selected_game: str,
    selected_approxs: list[str],
    selected_metrics: list[str],
) -> pd.DataFrame:
    """Compute the leaderboard filtered to a single game.

    Args:
        df_agg: Aggregated DataFrame produced by ``load_and_aggregate``.
        selected_game: Name of the game to filter by.
        selected_approxs: Approximator names to include.
        selected_metrics: Metric names to display.

    Returns:
        Formatted leaderboard DataFrame for the selected game.
    """
    df_filtered = df_agg[
        (df_agg["game_name"] == selected_game)
        & (df_agg["approximator_name"].isin(selected_approxs))
    ]

    return _build_leaderboard(df_filtered, selected_metrics)


def get_plot(
    df_agg: pd.DataFrame,
    selected_game: str,
    metric: str,
    selected_approximators: list[str],
    yaxis_range: list[float] | None = None,
) -> go.Figure:
    """Generate a log-scale line plot of one metric across budgets.

    Draws one line per approximator with a shaded ±1 std band. Missing values
    are replaced with zero before plotting.

    Args:
        df_agg: Aggregated DataFrame produced by ``load_and_aggregate``.
        selected_game: Name of the game to visualize.
        metric: Metric column prefix (e.g. ``"mse"``); columns ``{metric}_mean``
            and ``{metric}_std`` must exist in *df_agg*.
        selected_approximators: Approximator names to include.
        yaxis_range: Optional ``[log10_min, log10_max]`` range for the y-axis.

    Returns:
        Plotly Figure with one trace per approximator and a shaded std band.
    """
    df_filtered = df_agg[
        (df_agg["game_name"] == selected_game)
        & (df_agg["approximator_name"].isin(selected_approximators))
    ]
    fig = go.Figure()

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    # Force numeric
    df_filtered["budget"] = pd.to_numeric(df_filtered["budget"], errors="coerce")
    df_filtered[mean_col] = pd.to_numeric(df_filtered[mean_col], errors="coerce")
    df_filtered[std_col] = pd.to_numeric(df_filtered[std_col], errors="coerce")

    # Check if the required columns have valid numeric data - if there are NaNs - replace them with 0
    if df_filtered[mean_col].isna().any():
        df_filtered[mean_col] = df_filtered[mean_col].fillna(0)
    if df_filtered[std_col].isna().any():
        df_filtered[std_col] = df_filtered[std_col].fillna(0)

    for approx_name, group in df_filtered.groupby("approximator_name"):
        style = GLOBAL_APPROX_STYLES.get(approx_name, {"color": "#000000", "dash": "solid"})
        color = style["color"]
        dash = style["dash"]

        sorted_group = group.sort_values("budget")
        approx_str = str(approx_name)

        # Linie
        fig.add_trace(
            go.Scatter(
                x=sorted_group["budget"],
                y=sorted_group[mean_col],
                mode="lines+markers",
                name=approx_name,
                line={"dash": dash, "color": color},
                legendgroup=approx_str,
            )
        )

        # Fehlerband
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(
            go.Scatter(
                x=pd.concat([sorted_group["budget"], sorted_group["budget"].iloc[::-1]]),
                y=pd.concat(
                    [
                        sorted_group[mean_col] + sorted_group[std_col].fillna(0),
                        (sorted_group[mean_col] - sorted_group[std_col]).fillna(0).iloc[::-1],
                    ]
                ),
                fill="toself",
                fillcolor=f"rgba({r}, {g}, {b}, 0.15)",
                line={"color": "rgba(255,255,255,0)"},
                showlegend=False,
                hoverinfo="skip",
                name=approx_name,
                legendgroup=approx_str,
            )
        )

    fig.update_layout(
        title=f"{metric.upper()} across Budgets - {selected_game}",
        xaxis_title="Budget (coalition evaluations)",
        yaxis_title=f"{metric.upper()} (mean over seeds)",
        yaxis={
            "type": "log",
            "range": yaxis_range,
        },
        legend_title="Approximator",
        hovermode="x unified",
    )

    return fig


def get_plot_single(
    df_agg: pd.DataFrame,
    selected_game: str,
    metric: str,
    approximator: str,
    yaxis_range: list[float] | None = None,
) -> go.Figure:
    """Generate a metric-across-budgets plot for a single approximator.

    Convenience wrapper around :func:`get_plot` that accepts a single
    approximator name instead of a list.

    Args:
        df_agg: Aggregated DataFrame produced by ``load_and_aggregate``.
        selected_game: Name of the game to visualize.
        metric: Metric column prefix (e.g. ``"mse"``).
        approximator: Name of the approximator to plot.
        yaxis_range: Optional ``[log10_min, log10_max]`` range for the y-axis.

    Returns:
        Plotly Figure for the single approximator.
    """
    return get_plot(df_agg, selected_game, metric, [approximator], yaxis_range)


# --- Daten laden ---
def _with_spinner(message: str, fn):
    import threading
    SPINNER = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    done = threading.Event()

    def spin():
        i = 0
        while not done.is_set():
            print(f"\r{SPINNER[i % len(SPINNER)]}  {message}", end="", flush=True)
            i += 1
            done.wait(0.1)
        print(f"\r✓  {message}")

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    result = fn()
    done.set()
    t.join()
    return result

print("\n🚀 Starting shapiq Leaderboard...\n")


df_agg = _with_spinner(
    "Connecting to database & loading records...",
    lambda: load_and_aggregate(method=LOADING_METHOD, path=RESULTS_PATH),
)

# Raw DB client for details tab
db_client = DatabaseClientFactory.create_client(
    LOADING_METHOD,
    db_args={"LOCAL_DB_PATH": str(RESULTS_PATH)} if LOADING_METHOD == "local" else {},
)

raw_records = _with_spinner("Loading raw records...", db_client.get_all)
_all_games = db_client.get_games()
_all_approxs = db_client.get_approximators()
_unique_configs = db_client.get_unique_configs()
_all_budgets = sorted({c.budget for c in _unique_configs})
_all_indices = sorted({c.index for c in _unique_configs})
_all_n_players = sorted({c.n_players for c in _unique_configs})
_all_max_orders = sorted({c.max_order for c in _unique_configs})
_all_gt_methods = sorted({c.ground_truth_method for c in _unique_configs})
_all_seeds = sorted({
    int(s) for r in raw_records
    if isinstance(s := r.get("approx_seed"), int | float | str)
})

_with_spinner("Computing global styles...", lambda: update_global_styles(df_agg) or True)

available_metrics = [m for m in METRICS if f"{m}_mean" in df_agg.columns]

print("\n✨ Ready!\n")


def compute_yranges(g: str, approximators: list[str]) -> dict[str, list[float] | None]:
    """Compute shared log-scale y-axis ranges for a game and approximator set.

    Args:
        g: Game name to filter by.
        approximators: Approximator names to include in the range calculation.

    Returns:
        Dict mapping each metric name to a ``[log10_min, log10_max]`` range,
        or ``None`` if no valid values exist for that metric.
    """
    df_filtered = df_agg[
        (df_agg["game_name"] == g) & (df_agg["approximator_name"].isin(approximators))
    ]
    yranges = {}
    for m in available_metrics:
        col = f"{m}_mean"

        valid_values = pd.to_numeric(df_filtered[col], errors="coerce")
        valid_values = valid_values[valid_values > ZERO_THRESHOLD]

        if not valid_values.empty:
            y_min = valid_values.min()
            y_max = valid_values.max()
            yranges[m] = [float(np.log10(y_min) - 0.5), float(np.log10(y_max) + 0.5)]
        else:
            yranges[m] = None

    return yranges


def update_compare_plots(*args: Any) -> tuple[go.Figure, ...]:
    """Recompute all side-by-side comparison plots after a dropdown change.

    Expects ``args`` to be laid out as
    ``[approx_0..approx_{MAX_COLS-1}, game_0..game_{MAX_COLS-1}, n_cols]``.

    Args:
        *args: Flattened sequence of approximator names, game names, and the
            active column count as defined by the Compare-tab Gradio inputs.

    Returns:
        Tuple of Plotly Figures — one per (metric x column) combination,
        in row-major order (all columns for metric 0, then metric 1, …).
    """
    # args enthält: [0..4] Approximatoren, [5..9] Games, [10] Anzahl aktiver Spalten (n_cols)
    approx_vals = list(args[:MAX_COLS])
    game_vals = list(args[MAX_COLS : 2 * MAX_COLS])
    n_cols = args[-1]

    # Approximatoren und Spiele herausfiltern, die gerade aktiv sichtbar sind
    active_approxs = approx_vals[:n_cols]
    active_games = game_vals[:n_cols]

    # Y-Achsen-Limits über alle aktiven Kombinationen hinweg
    yranges = {}
    for m in available_metrics:
        col = f"{m}_mean"

        df_filtered = df_agg[
            (df_agg["game_name"].isin(active_games))
            & (df_agg["approximator_name"].isin(active_approxs))
        ]

        if not df_filtered.empty:
            valid_values = pd.to_numeric(df_filtered[col], errors="coerce")
            valid_values = valid_values[valid_values > ZERO_THRESHOLD]

            if not valid_values.empty:
                y_min = valid_values.min()
                y_max = valid_values.max()

                if pd.isna(y_min) or pd.isna(y_max):
                    yranges[m] = None
                else:
                    # Puffer für die Log-Skala berechnen
                    yranges[m] = [float(np.log10(y_min) - 0.5), float(np.log10(y_max) + 0.5)]
            else:
                yranges[m] = None
        else:
            yranges[m] = None

    outputs = []
    for m in available_metrics:
        # KORREKTUR: Wir nutzen extend mit einer List-Comprehension für die Spalten
        outputs.extend(
            [
                get_plot_single(df_agg, game_vals[i], m, approx_vals[i], yranges.get(m))
                for i in range(MAX_COLS)
            ]
        )

    return tuple(outputs)


def compute_elo_for_bucket(
    df_raw_records: list[dict], budget: int, metric: str = "all"
) -> tuple[pd.DataFrame, go.Figure, str]:
    """Run ELO scoring for a specific budget bucket and return table + plot.

    Args:
        df_raw_records: Raw benchmark records as a list of dicts.
        budget: The budget value to filter by (e.g. 250, 500, 1000, 5000, 10000).
        metric: Metric to score by. Use "all" (default) to include all metrics.

    Returns:
        A tuple of (leaderboard DataFrame, Plotly bar chart Figure, info markdown string).
    """
    scorer = EloScorer(
        budgets=[budget],
        metric_names=[str(metric)] if metric != "all" else None,
    )
    result = scorer.score(df_raw_records)

    if not result.rows:
        empty_df = pd.DataFrame(
            [
                {
                    "Rank": "-",
                    "Approximator": f"No data available for budget {budget}",
                    "ELO Score": "-",
                    "Matches": "-",
                    "Wins": "-",
                    "Losses": "-",
                    "Ties": "-",
                }
            ]
        )
        fig = go.Figure()
        fig.update_layout(
            title=f"No data for budget {budget}",
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": f"No approximators found for budget {budget}",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 16, "color": "gray"},
                }
            ],
        )
        return empty_df, fig, "No data"

    rows_data = [
        {
            "Rank": row.rank,
            "Approximator": row.approximator_name,
            "ELO Score": f"{row.score:.1f}",
            "Matches": row.metadata.get("n_matches", 0),
            "Wins": row.metadata.get("wins", 0),
            "Losses": row.metadata.get("losses", 0),
            "Ties": row.metadata.get("ties", 0),
        }
        for row in result.rows
    ]
    leaderboard_df = pd.DataFrame(rows_data)

    # Sort by rank for the bar chart (best = highest ELO, shown left)
    sorted_rows = sorted(result.rows, key=lambda r: r.score, reverse=True)
    approx_names = [r.approximator_name for r in sorted_rows]
    elo_scores = [r.score for r in sorted_rows]

    bar_colors = [
        GLOBAL_APPROX_STYLES.get(name, {"color": "#636EFA"})["color"] for name in approx_names
    ]

    fig = go.Figure(
        go.Bar(
            x=approx_names,
            y=elo_scores,
            marker={"color": bar_colors},
            text=[f"{s:.1f}" for s in elo_scores],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>ELO: %{y:.1f}<extra></extra>",
        )
    )

    y_min = min(elo_scores) - 20 if elo_scores else 950
    y_max = max(elo_scores) + 30 if elo_scores else 1050

    fig.update_layout(
        title=f"ELO Ratings — Budget {budget}",
        xaxis_title="Approximator",
        yaxis_title="ELO Score",
        yaxis={"range": [y_min, y_max]},
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin={"t": 60, "b": 80},
    )

    n_bs = result.metadata.get("n_bootstrap_samples", 0)
    n_perm = result.metadata.get("n_permutations", 1)
    n_bs = n_bs if isinstance(n_bs, int) else 0
    n_perm = n_perm if isinstance(n_perm, int) else 1
    metric_label = metric or "all"
    info_md = (
        f"Metric: **{metric_label}** | Bootstrap samples: **{n_bs}** | Permutations: **{n_perm}**"
    )

    return leaderboard_df, fig, info_md


def _records_to_df(records: list[dict], metric_filter: list[str] | None = None) -> pd.DataFrame:
    """Flatten raw benchmark records into a tabular DataFrame.

    Serializes nested dict/list fields to JSON strings, expands the ``metrics``
    sub-dict into top-level columns, and optionally restricts which metric
    columns are included.

    Args:
        records: Raw benchmark records as returned by the database client.
        metric_filter: If given, only metric keys present in this list are
            added as columns. ``None`` includes all metrics.

    Returns:
        DataFrame with one row per record and individual metric values as columns.
        Returns an empty DataFrame if *records* is empty.
    """
    rows = []
    for r in records:
        row = {
            k: json.dumps(v) if isinstance(v, dict | list) else v
            for k, v in r.items()
            if k != "metrics"
        }
        r_metrics = r.get("metrics") or {}
        row.update(
            {k: v for k, v in r_metrics.items() if metric_filter is None or k in metric_filter}
        )
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# --- Gradio App ---
with gr.Blocks(title="shapiq Leaderboard") as demo:
    gr.Markdown("""
    # shapiq Approximator Leaderboard
    Comparison of Shapley value approximators across games, budgets, and seeds.
    """)

    # Store df in gradio state to allow reloading
    df_state = gr.State(value=df_agg)

    raw_state = gr.State(value=raw_records)

    with gr.Row():
        reload_btn = gr.Button("Reload Data", variant="secondary", scale=0)

    with gr.Tab("ELO Leaderboard"):
        gr.Markdown("""
        ## ELO Leaderboard by Budget Bucket

        Approximators are ranked using pairwise ELO comparisons within each budget tier.
        A higher ELO score means the approximator consistently outperforms others at this budget.
        """)

        elo_metric_filter = gr.Dropdown(
            choices=["all", *available_metrics],
            value="all",
            label="Metric (optional — default: all metrics)",
            multiselect=False,
        )

        # State: current bucket index (0-4), start with Medium (1000) = index 2
        elo_bucket_idx_state = gr.State(value=2)

        with gr.Row(equal_height=True):
            elo_prev_btn = gr.Button("◀ Lower Budget", scale=0, variant="secondary")
            elo_bucket_label = gr.Markdown(
                value=f"### {BUDGET_BUCKETS[2]['label']}",
                elem_id="elo-bucket-label",
            )
            elo_next_btn = gr.Button("Higher Budget ▶", scale=0, variant="secondary")

        with gr.Row():
            elo_approx_filter = gr.CheckboxGroup(
                choices=df_agg["approximator_name"].unique().tolist(),
                value=df_agg["approximator_name"].unique().tolist(),
                label="Approximators",
            )
            with gr.Column(scale=0, min_width=120):
                elo_deselect_btn = gr.Button("Alle abwählen", size="sm")
                elo_reset_btn = gr.Button("Zurücksetzen", size="sm")

        elo_deselect_btn.click(fn=list, outputs=elo_approx_filter)
        elo_reset_btn.click(
            fn=lambda: df_agg["approximator_name"].unique().tolist(), outputs=elo_approx_filter
        )

        # Pre-compute initial ELO display for Medium (1000) bucket
        _elo_init_table, _elo_init_fig, _elo_init_info = compute_elo_for_bucket(
            raw_records, int(BUDGET_BUCKETS[2]["budget"])
        )

        with gr.Row():
            with gr.Column(scale=3):
                elo_info_md = gr.Markdown(value=_elo_init_info)
                elo_table = gr.Dataframe(
                    value=_elo_init_table,
                    interactive=False,
                    label="Rankings",
                    max_height=1000,
                )
            with gr.Column(scale=2):
                elo_plot = gr.Plot(value=_elo_init_fig, label="ELO Scores")

        def update_elo_tab(
            bucket_idx: int,
            raw_records: list[dict],
            selected_approxs: list[str],
            metric: str = "all",
        ) -> tuple[str, pd.DataFrame, go.Figure, str]:
            """Compute the ELO leaderboard for the given budget bucket.

            Args:
                bucket_idx: Index into ``BUDGET_BUCKETS`` for the target budget.
                raw_records: All raw benchmark records from the database.
                selected_approxs: Approximator names to include.
                metric: Metric to score by; ``"all"`` includes every metric.

            Returns:
                Tuple of (bucket label markdown, leaderboard DataFrame,
                bar chart Figure, info markdown string).
            """
            bucket = BUDGET_BUCKETS[bucket_idx]
            filtered = [r for r in raw_records if r.get("approximator_name") in selected_approxs]
            table_df, fig, info_md = compute_elo_for_bucket(filtered, int(bucket["budget"]), metric)
            label_md = f"### {bucket['label']}"
            return label_md, table_df, fig, info_md

        def elo_navigate(
            current_idx: int,
            delta: int,
            raw_records: list[dict],
            selected_approxs: list[str],
            metric: str = "all",
        ) -> Iterator[Any]:
            """Navigate between budget buckets and stream UI updates.

            Hides the leaderboard table first to force a size reset, then yields
            the recomputed content for the new bucket.

            Args:
                current_idx: Current bucket index.
                delta: Direction to move (``-1`` for lower, ``+1`` for higher budget).
                raw_records: All raw benchmark records from the database.
                selected_approxs: Approximator names to include.
                metric: Metric to score by; ``"all"`` includes every metric.

            Yields:
                Two partial Gradio update tuples: first hides the table,
                second populates it with the new bucket's data.
            """
            new_idx = max(0, min(len(BUDGET_BUCKETS) - 1, current_idx + delta))
            yield new_idx, gr.update(), gr.update(visible=False), gr.update(), gr.update(), gr.update(), gr.update()
            label_md, table_df, fig, info_md = update_elo_tab(new_idx, raw_records, selected_approxs, metric)
            yield (
                new_idx,
                label_md,
                gr.update(value=table_df, visible=True, max_height=1000),
                fig,
                info_md,
                gr.update(interactive=(new_idx > 0)),
                gr.update(interactive=(new_idx < len(BUDGET_BUCKETS) - 1)),
            )

        def elo_prev(
            idx: int, raw_records: list[dict], selected_approxs: list[str], metric: str = "all"
        ) -> Iterator[Any]:
            """Navigate to the previous (lower) budget bucket.

            Args:
                idx: Current bucket index.
                raw_records: All raw benchmark records from the database.
                selected_approxs: Approximator names to include.
                metric: Metric to score by; ``"all"`` includes every metric.

            Yields:
                Gradio update tuples forwarded from :func:`elo_navigate`.
            """
            yield from elo_navigate(idx, -1, raw_records, selected_approxs, metric)

        def elo_next(
            idx: int, raw_records: list[dict], selected_approxs: list[str], metric: str = "all"
        ) -> Iterator[Any]:
            """Navigate to the next (higher) budget bucket.

            Args:
                idx: Current bucket index.
                raw_records: All raw benchmark records from the database.
                selected_approxs: Approximator names to include.
                metric: Metric to score by; ``"all"`` includes every metric.

            Yields:
                Gradio update tuples forwarded from :func:`elo_navigate`.
            """
            yield from elo_navigate(idx, +1, raw_records, selected_approxs, metric)

        elo_prev_btn.click(
            fn=elo_prev,
            inputs=[elo_bucket_idx_state, raw_state, elo_approx_filter, elo_metric_filter],
            outputs=[elo_bucket_idx_state, elo_bucket_label, elo_table, elo_plot, elo_info_md, elo_prev_btn,
                     elo_next_btn],
        )
        elo_next_btn.click(
            fn=elo_next,
            inputs=[elo_bucket_idx_state, raw_state, elo_approx_filter, elo_metric_filter],
            outputs=[elo_bucket_idx_state, elo_bucket_label, elo_table, elo_plot, elo_info_md, elo_prev_btn,
                     elo_next_btn],
        )

        def elo_filter_update(
            idx: int, raw_records: list[dict], selected_approxs: list[str], metric: str = "all"
        ) -> Iterator[Any]:
            """Recompute the ELO tab after an approximator or metric filter change.

            Hides the table first to force a re-render, then yields the updated content.

            Args:
                idx: Current bucket index.
                raw_records: All raw benchmark records from the database.
                selected_approxs: Approximator names to include.
                metric: Metric to score by; ``"all"`` includes every metric.

            Yields:
                Two partial Gradio update tuples: first hides the table,
                second populates it with the recomputed data.
            """
            yield gr.update(), gr.update(visible=False), gr.update(), gr.update()
            label_md, table_df, fig, info_md = update_elo_tab(
                idx, raw_records, selected_approxs, metric
            )
            yield label_md, gr.update(value=table_df, visible=True, max_height=1000), fig, info_md

        elo_approx_filter.change(
            fn=elo_filter_update,
            inputs=[elo_bucket_idx_state, raw_state, elo_approx_filter, elo_metric_filter],
            outputs=[elo_bucket_label, elo_table, elo_plot, elo_info_md],
        )

        elo_metric_filter.change(
            fn=elo_filter_update,
            inputs=[elo_bucket_idx_state, raw_state, elo_approx_filter, elo_metric_filter],
            outputs=[elo_bucket_label, elo_table, elo_plot, elo_info_md],
        )

        gr.Markdown("---\n## All Budget Buckets — Side-by-Side Overview")

        all_bucket_tables = []
        all_bucket_plots = []
        all_bucket_infos = []

        with gr.Row():
            for bucket in BUDGET_BUCKETS:
                with gr.Column():
                    gr.Markdown(f"### {bucket['label']}")
                    _t, _f, _info = compute_elo_for_bucket(raw_records, int(bucket["budget"]))
                    all_bucket_infos.append(gr.Markdown(value=_info))
                    all_bucket_plots.append(gr.Plot(value=_f))
                    all_bucket_tables.append(
                        gr.Dataframe(value=_t, interactive=False, max_height=1000)
                    )

        def update_all_buckets(
            raw_records: list[dict], selected_approxs: list[str], metric: str = "all"
        ) -> tuple[Any, ...]:
            """Recompute ELO results for all budget buckets simultaneously.

            Args:
                raw_records: All raw benchmark records from the database.
                selected_approxs: Approximator names to include.
                metric: Metric to score by; ``"all"`` includes every metric.

            Returns:
                Flat tuple of ``(info_md, table_df, figure)`` repeated for each
                bucket in ``BUDGET_BUCKETS`` order.
            """
            filtered = [r for r in raw_records if r.get("approximator_name") in selected_approxs]
            outputs = []
            for bucket in BUDGET_BUCKETS:
                t, f, info = compute_elo_for_bucket(filtered, int(bucket["budget"]), metric)
                outputs.extend([info, t, f])
            return tuple(outputs)

        elo_approx_filter.change(
            fn=update_all_buckets,
            inputs=[raw_state, elo_approx_filter, elo_metric_filter],
            outputs=[
                item
                for i in range(len(BUDGET_BUCKETS))
                for item in [all_bucket_infos[i], all_bucket_tables[i], all_bucket_plots[i]]
            ],
        )

        elo_metric_filter.change(
            fn=update_all_buckets,
            inputs=[raw_state, elo_approx_filter, elo_metric_filter],
            outputs=[
                item
                for i in range(len(BUDGET_BUCKETS))
                for item in [all_bucket_infos[i], all_bucket_tables[i], all_bucket_plots[i]]
            ],
        )

    with gr.Tab("Leaderboard"):
        gr.Markdown("## Global Leaderboard (all games)")

        with gr.Row():
            with gr.Row():
                lb_approx_filter = gr.CheckboxGroup(
                    choices=df_agg["approximator_name"].unique().tolist(),
                    value=df_agg["approximator_name"].unique().tolist(),
                    label="Approximators",
                )
                with gr.Column(scale=0, min_width=120):
                    lb_approx_deselect = gr.Button("Alle abwählen", size="sm")
                    lb_approx_reset = gr.Button("Zurücksetzen", size="sm")
            with gr.Row():
                lb_metric_filter = gr.CheckboxGroup(
                    choices=available_metrics,
                    value=available_metrics,
                    label="Metrics",
                )
                with gr.Column(scale=0, min_width=120):
                    lb_metric_deselect = gr.Button("Alle abwählen", size="sm")
                    lb_metric_reset = gr.Button("Zurücksetzen", size="sm")

        lb_approx_deselect.click(fn=list, outputs=lb_approx_filter)
        lb_approx_reset.click(
            fn=lambda: df_agg["approximator_name"].unique().tolist(), outputs=lb_approx_filter
        )
        lb_metric_deselect.click(fn=list, outputs=lb_metric_filter)
        lb_metric_reset.click(fn=lambda: available_metrics, outputs=lb_metric_filter)

        global_leaderboard = gr.Dataframe(
            value=get_leaderboard_global(
                df_agg, df_agg["approximator_name"].unique().tolist(), available_metrics
            ),
            interactive=False,
        )

        gr.Markdown("## Per-Game Leaderboard")
        game_dropdown_lb = gr.Dropdown(
            choices=df_agg["game_name"].unique().tolist(),
            value=df_agg["game_name"].iloc[0],
            label="Game",
        )
        game_leaderboard = gr.Dataframe(
            value=get_leaderboard_game(
                df_agg,
                df_agg["game_name"].iloc[0],
                df_agg["approximator_name"].unique().tolist(),
                available_metrics,
            ),
            interactive=False,
        )

        def force_global_dataframe_rerender(
            df: pd.DataFrame, approxs: list[str], metrics: list[str]
        ) -> Iterator[Any]:
            """Forces a re-render of the global leaderboard dataframe.

            First hides the component to reset the UI state, then calculates the
            filtered leaderboard and displays it again.

            Args:
                df: Current aggregated DataFrame.
                approxs: Selected approximator names.
                metrics: Selected metric names.

            Yields:
                Two Gradio updates: first hides the component, then shows it
                with the recalculated leaderboard data.
            """
            yield gr.update(visible=False)
            filtered_df = get_leaderboard_global(df, approxs, metrics)
            yield gr.update(value=filtered_df, visible=True)

        def force_game_dataframe_rerender(
            game: str, df: pd.DataFrame, approxs: list[str], metrics: list[str]
        ) -> Iterator[Any]:
            """Forces a re-render of the per-game leaderboard dataframe.

            First hides the component to reset the UI state, then calculates the
            game-specific leaderboard and displays it again.

            Args:
                game: Selected game name.
                df: Current aggregated DataFrame.
                approxs: Selected approximator names.
                metrics: Selected metric names.

            Yields:
                Two Gradio updates: first hides the component, then shows it
                with the recalculated leaderboard data.
            """
            yield gr.update(visible=False)
            filtered_df = get_leaderboard_game(df, game, approxs, metrics)
            yield gr.update(value=filtered_df, visible=True)

        for component in [lb_approx_filter, lb_metric_filter]:
            component.change(
                fn=force_global_dataframe_rerender,
                inputs=[df_state, lb_approx_filter, lb_metric_filter],
                outputs=global_leaderboard,
            )
            component.change(
                fn=force_game_dataframe_rerender,
                inputs=[game_dropdown_lb, df_state, lb_approx_filter, lb_metric_filter],
                outputs=game_leaderboard,
            )

        game_dropdown_lb.change(
            fn=force_game_dataframe_rerender,
            inputs=[game_dropdown_lb, df_state, lb_approx_filter, lb_metric_filter],
            outputs=game_leaderboard,
        )

    game_dropdowns = {}
    approx_checkboxes = {}
    metric_plots = {}

    with gr.Tab("Metrics across Budgets"):
        jump_buttons = {}
        for metric in available_metrics:
            with gr.Tab(f"{metric.upper()} across Budgets"):
                game_dropdowns[metric] = gr.Dropdown(
                    choices=df_agg["game_name"].unique().tolist(),
                    value=df_agg["game_name"].iloc[0],
                    label="Game",
                )
                with gr.Row():
                    approx_checkboxes[metric] = gr.CheckboxGroup(
                        choices=df_agg["approximator_name"].unique().tolist(),
                        value=df_agg["approximator_name"].unique().tolist(),
                        label="Approximatoren",
                    )
                    with gr.Column(scale=0, min_width=120):
                        deselect_btn = gr.Button("Alle abwählen", size="sm")
                        reset_btn = gr.Button("Zurücksetzen", size="sm")
                        jump_buttons[metric] = gr.Button(
                            "🔍 Open in Detailed Data Tab", size="sm", variant="secondary",
                        )

                deselect_btn.click(fn=list, outputs=approx_checkboxes[metric])
                reset_btn.click(
                    fn=lambda: df_agg["approximator_name"].unique().tolist(),
                    outputs=approx_checkboxes[metric],
                )

                metric_plots[metric] = gr.Plot(
                    value=get_plot(
                        df_agg,
                        df_agg["game_name"].iloc[0],
                        metric,
                        df_agg["approximator_name"].unique().tolist(),
                    )
                )

                # metric als default-Argument binden, sonst nimmt die Lambda immer den letzten Wert
                for component in [game_dropdowns[metric], approx_checkboxes[metric]]:
                    component.change(
                        fn=lambda g, df, a, m=metric: get_plot(df, g, m, a),
                        inputs=[game_dropdowns[metric], df_state, approx_checkboxes[metric]],
                        outputs=metric_plots[metric],
                    )

    with gr.Tab("Compare Approximators"):
        gr.Markdown("## Side-by-side Approximator Comparison")

        gr.HTML("<style>.compare-jump-btn { margin-left: auto !important; } .hidden-col { display: none !important; }</style>")
        with gr.Row():
            add_col_btn = gr.Button("+ Add Approximator", scale=0, variant="primary")
            remove_col_btn = gr.Button("- Remove Approximator", scale=0, variant="primary", interactive=False)
            n_cols_state = gr.State(value=DEFAULT_COLS)
            compare_jump_btn = gr.Button(
                "🔍 Open in Detailed Data Tab",
                scale=0,
                variant="secondary",
                elem_classes=["compare-jump-btn"],
            )

        # Pro Spalte: Approximator-Dropdown + Game-Dropdown
        compare_column_containers = []
        compare_approx_dropdowns = []
        compare_game_dropdowns = []

        approxs = df_agg["approximator_name"].unique().tolist()
        games = df_agg["game_name"].unique().tolist()

        with gr.Row():
            for i in range(MAX_COLS):
                with gr.Column(elem_classes=[] if i < DEFAULT_COLS else ["hidden-col"]) as col:
                    compare_column_containers.append(col)
                    compare_approx_dropdowns.append(
                        gr.Dropdown(
                            choices=approxs,
                            value=approxs[i % len(approxs)],
                            label=f"Approximator {i + 1}",
                        )
                    )
                    compare_game_dropdowns.append(
                        gr.Dropdown(
                            choices=games,
                            value=games[0],
                            label="Game",
                        )
                    )

        # Plots pro Metrik pro Spalte
        compare_plot_rows = {}  # compare_plot_rows[metric] = [plot0, plot1, ...]
        _yranges = compute_yranges(
            games[0], [approxs[0], approxs[1 % len(approxs)], approxs[2 % len(approxs)]]
        )

        for m in available_metrics:
            gr.Markdown(f"### {m.upper()} across Budgets")
            with gr.Row():
                plots = [
                    gr.Plot(
                        value=get_plot(df_agg, games[0], m, [approxs[i % len(approxs)]], _yranges.get(m)),
                        elem_classes=[] if i < DEFAULT_COLS else ["hidden-col"],
                    )
                    for i in range(MAX_COLS)
                ]
                compare_plot_rows[m] = plots

        # Spalte hinzufuegen/entfernen
        all_col_components = [
            *compare_column_containers,
            *[p for m in available_metrics for p in compare_plot_rows[m]],
        ]

        def update_col_visibility(n: int, delta: int, *dropdown_values: Any) -> list[Any]:
            """Updates the visibility and plots of side-by-side comparison columns.

            Calculates the new column count, dynamically adjusts the visibility
            of components, and recalculates the shared y-axis ranges.
            """
            new_n = max(2, min(MAX_COLS, n + delta))
            updates = []

            # Spalten-Container (Inklusive der darin liegenden Dropdowns) updaten
            updates = [gr.update(elem_classes=[] if c_idx < new_n else ["hidden-col"]) for c_idx in range(MAX_COLS)]

            # Gemeinsame Y-Range über alle aktiven Spalten berechnen
            active_approxs = [dropdown_values[i] for i in range(new_n)]
            active_games = [dropdown_values[MAX_COLS + i] for i in range(new_n)]

            # Plots berechnen UND sichtbar schalten
            for m in available_metrics:
                yr_shared = compute_yranges(active_games[0], active_approxs).get(m)

                for p_idx in range(MAX_COLS):
                    is_visible = p_idx < new_n

                    if is_visible:
                        approx_val = dropdown_values[p_idx]
                        game_val = dropdown_values[MAX_COLS + p_idx]
                        plot_figure = get_plot_single(df_agg, game_val, m, approx_val, yr_shared)
                        updates.append(gr.update(value=plot_figure, elem_classes=[]))
                    else:
                        updates.append(gr.update(elem_classes=["hidden-col"]))

            return [new_n, *updates, gr.update(interactive=(new_n < MAX_COLS)), gr.update(interactive=(new_n > 2))]

        click_inputs = [n_cols_state, *compare_approx_dropdowns, *compare_game_dropdowns]

        add_col_btn.click(
            fn=lambda n, *args: update_col_visibility(n, +1, *args),
            inputs=click_inputs,
            outputs=[n_cols_state, *all_col_components, add_col_btn, remove_col_btn],
        )
        remove_col_btn.click(
            fn=lambda n, *args: update_col_visibility(n, -1, *args),
            inputs=click_inputs,
            outputs=[n_cols_state, *all_col_components, add_col_btn, remove_col_btn],
        )

        # Plot Update bei Aenderung
        all_inputs = compare_approx_dropdowns + compare_game_dropdowns
        plot_outputs = [p for m in available_metrics for p in compare_plot_rows[m]]

        for component in all_inputs:
            component.change(
                fn=update_compare_plots, inputs=[*all_inputs, n_cols_state], outputs=plot_outputs
            )

    with gr.Tab("Detailed Data"):
        gr.Markdown("## Detailed Data\nDirect MongoDB query of individual runs.")

        with gr.Row():
            det_game = gr.Dropdown(choices=_all_games, label="Game", multiselect=True)
            det_approx = gr.Dropdown(choices=_all_approxs, label="Approximator", multiselect=True)
            det_budget = gr.Dropdown(choices=[str(b) for b in _all_budgets], label="Budget", multiselect=True)
            det_index = gr.Dropdown(choices=_all_indices, label="Index", multiselect=True)
            det_max_order = gr.Dropdown(choices=[str(o) for o in _all_max_orders], label="Max Order", multiselect=True)

        with gr.Row():
            det_n_players = gr.Dropdown(choices=[str(n) for n in _all_n_players], label="N Players", multiselect=True)
            det_gt_method = gr.Dropdown(choices=_all_gt_methods, label="Ground Truth Method", multiselect=True)
            det_seed = gr.Dropdown(choices=[str(s) for s in _all_seeds], label="Approx Seed", multiselect=True)
            det_metric = gr.Dropdown(choices=available_metrics, label="Metrics", multiselect=True)

        with gr.Row():
            det_search_btn = gr.Button("Search", variant="primary")
            det_reset_btn = gr.Button("Filter zurücksetzen", variant="secondary")

        det_count = gr.Markdown(f"**{len(raw_records)} Runs gefunden.**")
        det_table = gr.Dataframe(value=_records_to_df(raw_records), interactive=False)

        async def query_raw(
            games: list[str],
            approxs: list[str],
            budgets: list[str],
            indices: list[str],
            metrics: list[str],
            n_players: list[str],
            max_orders: list[str],
            seeds: list[str],
            gt_methods: list[str],
        ) -> AsyncGenerator[tuple[Any, Any], None]:
            """Query and display raw benchmark records with optional filters.

            Fetches all records from the database, applies the active filter
            selections, and streams progressive Gradio UI updates to display
            results or an empty state.

            Args:
                games: Game names to include; empty list means no filter.
                approxs: Approximator names to include; empty list means no filter.
                budgets: Budget values (as strings) to include; empty list means no filter.
                indices: Interaction index names to include; empty list means no filter.
                metrics: Metric columns to expand; empty list means all metrics.
                n_players: Player counts to include; empty list means no filter.
                max_orders: Max order values to include; empty list means no filter.
                seeds: Approximator seed values to include; empty list means no filter.
                gt_methods: Ground truth method names to include; empty list means no filter.

            Yields:
                Gradio update tuples for the count label and the result table,
                streaming intermediate hide/show states to force a re-render.
            """
            yield "", gr.update(visible=False)

            all_records = db_client.get_all()

            filtered = []
            for r in all_records:
                if games and r.get("game_name") not in games:
                    continue
                if approxs and r.get("approximator_name") not in approxs:
                    continue
                if budgets and str(r.get("budget")) not in budgets:
                    continue
                if indices and r.get("index") not in indices:
                    continue
                if n_players and str(r.get("n_players")) not in n_players:
                    continue
                if max_orders and str(r.get("max_order")) not in max_orders:
                    continue
                if seeds and str(r.get("approx_seed")) not in seeds:
                    continue
                if gt_methods and r.get("ground_truth_method") not in gt_methods:
                    continue
                filtered.append(r)

            if not filtered:
                yield "**0 Runs gefunden.**", gr.update(value=pd.DataFrame(), visible=True)
                return

            result_df = _records_to_df(filtered, metrics or None)

            yield f"**{len(result_df)} Runs gefunden.**", gr.update(value=result_df, visible=True)
            await asyncio.sleep(0.05)
            yield gr.update(), gr.update(visible=False)
            await asyncio.sleep(0.05)
            yield gr.update(), gr.update(value=result_df, visible=True)


        _det_filters = [det_game, det_approx, det_budget, det_index, det_metric,
                        det_n_players, det_max_order, det_seed, det_gt_method]

        det_search_btn.click(
            fn=query_raw,
            inputs=_det_filters,
            outputs=[det_count, det_table],
        )

        def reset_det_filters() -> tuple[list, ...]:
            """Reset all Detailed Data Tab filters to their empty default state.

            Returns:
                Tuple of empty lists, one for each dropdown in ``_det_filters``.
            """
            return tuple([] for _ in _det_filters)


        det_reset_btn.click(fn=reset_det_filters, outputs=_det_filters)

    def on_reload() -> tuple[Any, ...]:
        """Reloads the raw dataset and refreshes all UI components.

        Fetches the fresh data from the source, updates the global visualization
        styles, and rebuilds all tables and plots for the interface.

        Returns:
            Flat tuple consumed by the Gradio ``reload_btn`` outputs: updated
            ``df_state``, ``raw_state``, global leaderboard DataFrame, game
            dropdown update, per-game leaderboard DataFrame, and for each
            available metric a game dropdown update, approximator checkbox
            update, and plot Figure.
        """
        new_df = reload_data()
        new_raw = db_client.get_all()
        update_global_styles(new_df)

        games = new_df["game_name"].unique().tolist()
        approxs = new_df["approximator_name"].unique().tolist()
        first_game = games[0]

        outputs: list[Any] = [
            new_df,
            new_raw,
            get_leaderboard_global(new_df, approxs, available_metrics),
            gr.Dropdown(choices=games, value=first_game),
            get_leaderboard_game(new_df, first_game, approxs, available_metrics),
        ]

        for m in available_metrics:
            outputs.extend(
                [
                    gr.Dropdown(choices=games, value=first_game),
                    gr.CheckboxGroup(choices=approxs, value=approxs),
                    get_plot(new_df, first_game, m, approxs),
                ]
            )

        return tuple(outputs)

    reload_btn.click(
        fn=on_reload,
        inputs=[],
        outputs=[
            df_state,
            raw_state,
            global_leaderboard,
            game_dropdown_lb,
            game_leaderboard,
            *[
                comp
                for m in available_metrics
                for comp in [game_dropdowns[m], approx_checkboxes[m], metric_plots[m]]
            ],
        ],
    )

    for metric in available_metrics:
        jump_buttons[metric].click(
            fn=lambda game, approxs: ([game] if game else [], approxs or [], []),
            inputs=[game_dropdowns[metric], approx_checkboxes[metric]],
            outputs=[det_game, det_approx, det_metric],
        ).then(
            fn=query_raw,
            inputs=_det_filters,
            outputs=[det_count, det_table],
        ).then(
            fn=lambda: gr.Info("Daten geladen — bitte zum 'Detailed Data' Tab wechseln"),
            inputs=[],
            outputs=[],
        )

    compare_jump_btn.click(
        fn=lambda n, *vals: (
            list(set(vals[MAX_COLS : MAX_COLS + n])),
            list(set(vals[:n])),
            [],
        ),
        inputs=[n_cols_state, *compare_approx_dropdowns, *compare_game_dropdowns],
        outputs=[det_game, det_approx, det_metric],
    ).then(
        fn=query_raw,
        inputs=_det_filters,
        outputs=[det_count, det_table],
    ).then(
        fn=lambda: gr.Info("Daten geladen — bitte zum 'Detailed Data' Tab wechseln"),
        inputs=[],
        outputs=[],
    )

demo.launch()
