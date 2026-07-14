"""UI components for the leaderboard."""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Iterator

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

from leaderboard.metrics import METRICS
from leaderboard.scoring.cd_scorer import CriticalDifferenceScorer
from leaderboard.scoring.elo_scorer import EloScorer
from leaderboard.storage.connection import DatabaseClientFactory, DBConnectionError
from leaderboard.storage.connection.utilities import process_raw_runs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

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

# ELO scoring configuration
ELO_N_BOOTSTRAP_SAMPLES = 200
ELO_N_PERMUTATIONS = 10


@dataclass
class InitialData:
    """Data container for leaderboard UI."""

    df_agg: pd.DataFrame
    raw_records: list[dict[str, Any]]
    db_client: Any
    all_games: list[str]
    all_approxs: list[str]
    all_budgets: list[int]
    all_indices: list[str]
    default_index: str
    all_n_players: list[int]
    all_max_orders: list[int]
    all_gt_methods: list[str]
    all_seeds: list[int]
    available_metrics: list[str]


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
def _with_spinner(message: str, fn: Callable[[], T]) -> T:
    """Run a function while printing a small terminal spinner."""
    import threading

    SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    done = threading.Event()

    def spin() -> None:
        i = 0
        while not done.is_set():
            print(f"\r{SPINNER[i % len(SPINNER)]}  {message}", end="", flush=True)  # noqa: T201
            i += 1
            done.wait(0.1)
        print(f"\r✓  {message}")  # noqa: T201

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    result = fn()
    done.set()
    t.join()
    return result


def load_initial_data() -> InitialData:
    """Load and prepare all data required for the initial UI build."""
    db_client = DatabaseClientFactory.create_client(
        LOADING_METHOD,
        db_args={"LOCAL_DB_PATH": str(RESULTS_PATH)} if LOADING_METHOD == "local" else {},
    )

    if not db_client.test_connection():
        raise DBConnectionError from None

    raw_records = _with_spinner("Loading raw records...", db_client.get_all)
    df_agg = _with_spinner(
        "Aggregating raw records...",
        lambda: process_raw_runs(raw_records),
    )

    _all_games = db_client.get_games()
    _all_approxs = db_client.get_approximators()
    _unique_configs = db_client.get_unique_configs()
    _all_budgets = sorted(set({c.budget for c in _unique_configs}))
    _all_indices = sorted(set({c.index for c in _unique_configs}))
    _default_index = "SV" if "SV" in _all_indices else (_all_indices[0] if _all_indices else "all")
    _all_n_players = sorted(set({c.n_players for c in _unique_configs}))
    _all_max_orders = sorted(set({c.max_order for c in _unique_configs}))
    _all_gt_methods = sorted(set({c.ground_truth_method for c in _unique_configs}))
    _all_seeds = sorted(
        set(
            {
                int(s)
                for r in raw_records
                if isinstance(s := r.get("approx_seed"), int | float | str)
            }
        )
    )

    _with_spinner("Computing global styles...", lambda: update_global_styles(df_agg) or True)

    available_metrics = [m for m in METRICS if f"{m}_mean" in df_agg.columns]

    return InitialData(
        df_agg=df_agg,
        raw_records=raw_records,
        db_client=db_client,
        all_games=_all_games,
        all_approxs=_all_approxs,
        all_budgets=_all_budgets,
        all_indices=_all_indices,
        default_index=_default_index,
        all_n_players=_all_n_players,
        all_max_orders=_all_max_orders,
        all_gt_methods=_all_gt_methods,
        all_seeds=_all_seeds,
        available_metrics=available_metrics,
    )


def compute_elo_for_bucket(
    df_raw_records: list[dict],
    budget: int,
    metric: str = "all",
    index: str = "all",
    game: str = "all",
    approx_styles: dict[str, dict[str, str]] | None = None,
) -> tuple[pd.DataFrame, go.Figure, str]:
    """Run ELO scoring for a specific budget bucket and return table + plot.

    Args:
        df_raw_records: Raw benchmark records as a list of dicts.
        budget: The budget value to filter by (e.g. 250, 500, 1000, 5000, 10000).
        metric: Metric to score by. Use "all" (default) to include all metrics.
        index: Interaction index to filter by (e.g. "SV"). Use "all" to include all indices.
        game: Game name to filter by. Use "all" to include all games.
        approx_styles: style definitions for plotting.

    Returns:
        A tuple of (leaderboard DataFrame, Plotly bar chart Figure, info markdown string).
    """
    scorer = EloScorer(
        budgets=[budget],
        metric_names=[str(metric)] if metric != "all" else None,
        indices=[str(index)] if index != "all" else None,
        game_names=[str(game)] if game != "all" else None,
        n_bootstrap_samples=ELO_N_BOOTSTRAP_SAMPLES,
        n_permutations=ELO_N_PERMUTATIONS,
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
                    "text": f"No approximators/data found for budget {budget} under this configuration",
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

    style_map = approx_styles if approx_styles is not None else GLOBAL_APPROX_STYLES

    bar_colors = [style_map.get(name, {"color": "#636EFA"})["color"] for name in approx_names]

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
    y_range = max(elo_scores) - min(elo_scores) if elo_scores else 100
    y_max = max(elo_scores) + max(50, y_range * 0.15) if elo_scores else 1050

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
    index_label = index or "all"
    info_md = (
        f"Metric: **{metric_label}** | Index: **{index_label}** | Game: **{game if game != 'all' else 'all'}** | "
        f"Bootstrap samples: **{n_bs}** | Permutations: **{n_perm}**"
    )

    return leaderboard_df, fig, info_md


def compute_elo_for_bucket_worker(
    args: tuple[list[dict], int, str, str, str, dict[str, dict[str, str]]],
) -> tuple[int, pd.DataFrame, go.Figure, str]:
    """Compute one ELO bucket in a separate process."""
    bucket_records, budget, metric, index, game, approx_styles = args
    table_df, fig, info_md = compute_elo_for_bucket(
        bucket_records,
        budget,
        metric,
        index,
        game,
        approx_styles,
    )
    return budget, table_df, fig, info_md


def compute_all_elo_buckets_parallel(
    raw_records: list[dict],
    selected_approxs: list[str],
    metric: str = "all",
    index: str = "all",
    game: str = "all",
) -> tuple[Any, ...]:
    """Compute all ELO budget buckets in parallel.

    Returns:
        Flat tuple in the order:
        info_md, table_0, fig_0, table_1, fig_1, ...
    """
    filtered = [
        record for record in raw_records if record.get("approximator_name") in selected_approxs
    ]

    approx_styles = GLOBAL_APPROX_STYLES.copy()

    tasks = []

    for bucket in BUDGET_BUCKETS:
        budget_value = int(bucket["budget"])
        bucket_records = [record for record in filtered if record.get("budget") == budget_value]
        tasks.append((bucket_records, budget_value, metric, index, game, approx_styles))

    max_workers = min(len(tasks), 6)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(compute_elo_for_bucket_worker, tasks))

    outputs: list[Any] = []
    first_info: str | None = None

    for _budget, table_df, fig, info_md in results:
        if first_info is None:
            first_info = info_md
        outputs.extend([table_df, fig])

    return first_info or "", *outputs


def compute_cd_for_bucket(
    df_raw_records: list[dict],
    budget: int,
    metric: str = "all",
    index: str = "all",
    game: str = "all",
) -> go.Figure:
    """Run Critical Difference scoring for a specific budget bucket and return a Plotly figure.

    Args:
        df_raw_records: Raw benchmark records as a list of dicts.
        budget: The budget value to filter by.
        metric: Metric to score by. Use "all" to include all metrics.
        index: Interaction index to filter by. Use "all" to include all indices.
        game: Game name to filter by. Use "all" to include all games.


    Returns:
        A Plotly Figure with the CD diagram.
    """
    scorer = CriticalDifferenceScorer(
        budgets=[budget],
        metric_names=[str(metric)] if metric != "all" else None,
        indices=[str(index)] if index != "all" else None,
        game_names=[str(game)] if game != "all" else None,
    )
    result = scorer.score(df_raw_records)
    cd_result = result.metadata.get("cd_result")

    if cd_result is None or len(cd_result.mean_ranks) < 2:
        fig = go.Figure()
        fig.update_layout(
            title=f"CD Diagram — no data for budget {budget}",
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "Not enough data for a CD diagram at this configuration.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 14, "color": "gray"},
                }
            ],
        )
        return fig

    return CriticalDifferenceScorer.plot_cd_diagram_plotly(
        cd_result,
        title=f"Critical Difference Diagram — Budget {budget}",
    )


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


def build_app() -> gr.Blocks:
    """Build and return the Gradio leaderboard app."""
    # --- Gradio App ---

    data = load_initial_data()

    df_agg = data.df_agg
    raw_records = data.raw_records
    db_client = data.db_client
    _all_games = data.all_games
    _all_approxs = data.all_approxs
    _all_budgets = data.all_budgets
    _all_indices = data.all_indices
    _default_index = data.default_index
    _all_n_players = data.all_n_players
    _all_max_orders = data.all_max_orders
    _all_gt_methods = data.all_gt_methods
    _all_seeds = data.all_seeds
    available_metrics = data.available_metrics

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

            elo_index_filter = gr.Dropdown(
                choices=["all", *_all_indices],
                value=_default_index,
                label="Index",
                multiselect=False,
            )

            elo_game_filter = gr.Dropdown(
                choices=["all", *_all_games],
                value="all",
                label="Game (optional — default: all games)",
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
                    elo_jump_btn = gr.Button(
                        "🔍 Open in Detailed Data Tab", size="sm", variant="secondary"
                    )

            elo_deselect_btn.click(fn=list, outputs=elo_approx_filter)
            elo_reset_btn.click(
                fn=lambda: df_agg["approximator_name"].unique().tolist(), outputs=elo_approx_filter
            )

            # Pre-compute initial ELO display for Medium (1000) bucket
            _initial_all_bucket_outputs = _with_spinner(
                "Computing initial ELO ratings for all budget buckets...",
                lambda: compute_all_elo_buckets_parallel(
                    raw_records=raw_records,
                    selected_approxs=df_agg["approximator_name"].unique().tolist(),
                    metric="all",
                    index=_default_index,
                    game="all",
                ),
            )
            _elo_init_info = _initial_all_bucket_outputs[0]
            _elo_init_table = _initial_all_bucket_outputs[1 + 2 * 2]
            _elo_init_fig = _initial_all_bucket_outputs[1 + 2 * 2 + 1]

            _elo_init_cd_fig = _with_spinner(
                f"Computing CD diagram for initial bucket ({BUDGET_BUCKETS[2]['label']})...",
                lambda: compute_cd_for_bucket(
                    raw_records, int(BUDGET_BUCKETS[2]["budget"]), "all", _default_index
                ),
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
                with gr.Column(scale=2), gr.Tabs():
                    with gr.TabItem("ELO Scores"):
                        elo_plot = gr.Plot(value=_elo_init_fig, label="ELO Scores")
                    with gr.TabItem("CD Diagram"):
                        elo_cd_plot = gr.Plot(value=_elo_init_cd_fig, label="CD Diagram")

            def update_elo_tab(
                bucket_idx: int,
                raw_records: list[dict],
                selected_approxs: list[str],
                metric: str = "all",
                index: str = "all",
                game: str = "all",
            ) -> tuple[str, pd.DataFrame, go.Figure, str, go.Figure]:
                """Compute the ELO leaderboard and CD diagram for the given budget bucket.

                Args:
                    bucket_idx: Index into ``BUDGET_BUCKETS`` for the target budget.
                    raw_records: All raw benchmark records from the database.
                    selected_approxs: Approximator names to include.
                    metric: Metric to score by; ``"all"`` includes every metric.
                    index: Interaction index to filter by (e.g. "SV"). Use "all" to include all indices.
                    game: Game name to filter by. Use "all" to include all games.

                Returns:
                    Tuple of (bucket label markdown, leaderboard DataFrame,
                    ELO bar chart Figure, info markdown string, CD diagram Figure).
                """
                bucket = BUDGET_BUCKETS[bucket_idx]
                filtered = [
                    r for r in raw_records if r.get("approximator_name") in selected_approxs
                ]
                budget = int(bucket["budget"])
                table_df, fig, info_md = compute_elo_for_bucket(
                    filtered, budget, metric, index, game, GLOBAL_APPROX_STYLES.copy()
                )
                cd_fig = compute_cd_for_bucket(filtered, budget, metric, index, game)
                label_md = f"### {bucket['label']}"
                return label_md, table_df, fig, info_md, cd_fig

            def elo_navigate(
                current_idx: int,
                delta: int,
                raw_records: list[dict],
                selected_approxs: list[str],
                metric: str = "all",
                index: str = "all",
                game: str = "all",
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
                    index: Interaction index to filter by (e.g. "SV"). Use "all" to include all indices.
                    game: Game name to filter by. Use "all" to include all games.


                Returns:
                    Iterator of two Gradio update tuples.
                """
                new_idx = max(0, min(len(BUDGET_BUCKETS) - 1, current_idx + delta))
                yield (
                    new_idx,
                    gr.update(),
                    gr.update(visible=False),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )
                label_md, table_df, fig, info_md, cd_fig = update_elo_tab(
                    new_idx, raw_records, selected_approxs, metric, index, game
                )
                yield (
                    new_idx,
                    label_md,
                    gr.update(value=table_df, visible=True, max_height=1000),
                    fig,
                    info_md,
                    cd_fig,
                    gr.update(interactive=(new_idx > 0)),
                    gr.update(interactive=(new_idx < len(BUDGET_BUCKETS) - 1)),
                )

            def elo_prev(
                idx: int,
                raw_records: list[dict],
                selected_approxs: list[str],
                metric: str = "all",
                index: str = "all",
                game: str = "all",
            ) -> Iterator[Any]:
                """Navigate to the previous (lower) budget bucket.

                Args:
                    idx: Current bucket index.
                    raw_records: All raw benchmark records from the database.
                    selected_approxs: Approximator names to include.
                    metric: Metric to score by; ``"all"`` includes every metric.
                    index: Interaction index to filter by (e.g. "SV"). Use "all" to include all indices.
                    game: Game name to filter by. Use "all" to include all games.


                Yields:
                    Gradio update tuples forwarded from :func:`elo_navigate`.
                """
                yield from elo_navigate(idx, -1, raw_records, selected_approxs, metric, index, game)

            def elo_next(
                idx: int,
                raw_records: list[dict],
                selected_approxs: list[str],
                metric: str = "all",
                index: str = "all",
                game: str = "all",
            ) -> Iterator[Any]:
                """Navigate to the next (higher) budget bucket.

                Args:
                    idx: Current bucket index.
                    raw_records: All raw benchmark records from the database.
                    selected_approxs: Approximator names to include.
                    metric: Metric to score by; ``"all"`` includes every metric.
                    index: Interaction index to filter by (e.g. "SV"). Use "all" to include all indices.
                    game: Game name to filter by. Use "all" to include all games.


                Yields:
                    Gradio update tuples forwarded from :func:`elo_navigate`.
                """
                yield from elo_navigate(idx, +1, raw_records, selected_approxs, metric, index, game)

            elo_prev_btn.click(
                fn=elo_prev,
                inputs=[
                    elo_bucket_idx_state,
                    raw_state,
                    elo_approx_filter,
                    elo_metric_filter,
                    elo_index_filter,
                    elo_game_filter,
                ],
                outputs=[
                    elo_bucket_idx_state,
                    elo_bucket_label,
                    elo_table,
                    elo_plot,
                    elo_info_md,
                    elo_cd_plot,
                    elo_prev_btn,
                    elo_next_btn,
                ],
            )
            elo_next_btn.click(
                fn=elo_next,
                inputs=[
                    elo_bucket_idx_state,
                    raw_state,
                    elo_approx_filter,
                    elo_metric_filter,
                    elo_index_filter,
                    elo_game_filter,
                ],
                outputs=[
                    elo_bucket_idx_state,
                    elo_bucket_label,
                    elo_table,
                    elo_plot,
                    elo_info_md,
                    elo_cd_plot,
                    elo_prev_btn,
                    elo_next_btn,
                ],
            )

            def elo_filter_update(
                idx: int,
                raw_records: list[dict],
                selected_approxs: list[str],
                metric: str = "all",
                index: str = "all",
                game: str = "all",
            ) -> Iterator[Any]:
                """Recompute the ELO tab after an approximator or metric filter change.

                Hides the table first to force a re-render, then yields the updated content.

                Args:
                    idx: Current bucket index.
                    raw_records: All raw benchmark records from the database.
                    selected_approxs: Approximator names to include.
                    metric: Metric to score by; ``"all"`` includes every metric.
                    index: Interaction index to filter by (e.g. "SV"). Use "all" to include all indices.
                    game: Game name to filter by. Use "all" to include all games.

                Returns:
                    Iterator of two Gradio update tuples.
                """
                yield gr.update(), gr.update(visible=False), gr.update(), gr.update(), gr.update()
                label_md, table_df, fig, info_md, cd_fig = update_elo_tab(
                    idx,
                    raw_records,
                    selected_approxs,
                    metric,
                    index,
                    game,
                )
                yield (
                    label_md,
                    gr.update(value=table_df, visible=True, max_height=1000),
                    fig,
                    info_md,
                    cd_fig,
                )

            elo_approx_filter.change(
                fn=elo_filter_update,
                inputs=[
                    elo_bucket_idx_state,
                    raw_state,
                    elo_approx_filter,
                    elo_metric_filter,
                    elo_index_filter,
                    elo_game_filter,
                ],
                outputs=[elo_bucket_label, elo_table, elo_plot, elo_info_md, elo_cd_plot],
            )

            elo_metric_filter.change(
                fn=elo_filter_update,
                inputs=[
                    elo_bucket_idx_state,
                    raw_state,
                    elo_approx_filter,
                    elo_metric_filter,
                    elo_index_filter,
                    elo_game_filter,
                ],
                outputs=[elo_bucket_label, elo_table, elo_plot, elo_info_md, elo_cd_plot],
            )

            elo_index_filter.change(
                fn=elo_filter_update,
                inputs=[
                    elo_bucket_idx_state,
                    raw_state,
                    elo_approx_filter,
                    elo_metric_filter,
                    elo_index_filter,
                    elo_game_filter,
                ],
                outputs=[elo_bucket_label, elo_table, elo_plot, elo_info_md, elo_cd_plot],
            )

            elo_game_filter.change(
                fn=elo_filter_update,
                inputs=[
                    elo_bucket_idx_state,
                    raw_state,
                    elo_approx_filter,
                    elo_metric_filter,
                    elo_index_filter,
                    elo_game_filter,
                ],
                outputs=[elo_bucket_label, elo_table, elo_plot, elo_info_md, elo_cd_plot],
            )

            gr.Markdown("---\n## All Budget Buckets — Side-by-Side Overview")
            all_buckets_info_md = gr.Markdown(value=_elo_init_info)

            all_bucket_tables = []
            all_bucket_plots = []

            with gr.Row():
                for i, bucket in enumerate(BUDGET_BUCKETS):
                    with gr.Column():
                        gr.Markdown(f"### {bucket['label']}")

                        table = _initial_all_bucket_outputs[1 + 2 * i]
                        fig = _initial_all_bucket_outputs[1 + 2 * i + 1]

                        all_bucket_plots.append(gr.Plot(value=fig))
                        all_bucket_tables.append(
                            gr.Dataframe(value=table, interactive=False, max_height=1000)
                        )

            def update_all_buckets(
                raw_records: list[dict],
                selected_approxs: list[str],
                metric: str = "all",
                index: str = "all",
                game: str = "all",
            ) -> Iterator[Any]:
                """Recompute ELO results for all budget buckets simultaneously.

                Args:
                    raw_records: All raw benchmark records from the database.
                    selected_approxs: Approximator names to include.
                    metric: Metric to score by; ``"all"`` includes every metric.
                    index: Interaction index to filter by (e.g. "SV"). Use "all" to include all indices.
                    game: Game name to filter by. Use "all" to include all games.

                Returns:
                    Iterator of two Gradio update tuples.
                """
                # Erst alle Tabellen verstecken
                hide_outputs = [gr.update()]
                for _ in BUDGET_BUCKETS:
                    hide_outputs.extend([gr.update(visible=False), gr.update()])
                yield tuple(hide_outputs)

                parallel_outputs = compute_all_elo_buckets_parallel(
                    raw_records=raw_records,
                    selected_approxs=selected_approxs,
                    metric=metric,
                    index=index,
                    game=game,
                )

                first_info = parallel_outputs[0]
                outputs = []

                for i in range(len(BUDGET_BUCKETS)):
                    table = parallel_outputs[1 + 2 * i]
                    fig = parallel_outputs[1 + 2 * i + 1]
                    outputs.extend([gr.update(value=table, visible=True, max_height=1000), fig])

                yield first_info, *outputs

            _all_bucket_outputs = [
                all_buckets_info_md,
                *[
                    item
                    for i in range(len(BUDGET_BUCKETS))
                    for item in [all_bucket_tables[i], all_bucket_plots[i]]
                ],
            ]

            elo_approx_filter.change(
                fn=update_all_buckets,
                inputs=[
                    raw_state,
                    elo_approx_filter,
                    elo_metric_filter,
                    elo_index_filter,
                    elo_game_filter,
                ],
                outputs=_all_bucket_outputs,
            )

            elo_metric_filter.change(
                fn=update_all_buckets,
                inputs=[
                    raw_state,
                    elo_approx_filter,
                    elo_metric_filter,
                    elo_index_filter,
                    elo_game_filter,
                ],
                outputs=_all_bucket_outputs,
            )

            elo_index_filter.change(
                fn=update_all_buckets,
                inputs=[
                    raw_state,
                    elo_approx_filter,
                    elo_metric_filter,
                    elo_index_filter,
                    elo_game_filter,
                ],
                outputs=_all_bucket_outputs,
            )

            elo_game_filter.change(
                fn=update_all_buckets,
                inputs=[
                    raw_state,
                    elo_approx_filter,
                    elo_metric_filter,
                    elo_index_filter,
                    elo_game_filter,
                ],
                outputs=_all_bucket_outputs,
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
                                "🔍 Open in Detailed Data Tab",
                                size="sm",
                                variant="secondary",
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

            gr.HTML(
                "<style>.compare-jump-btn { margin-left: auto !important; } .hidden-col { display: none !important; }</style>"
            )
            with gr.Row():
                add_col_btn = gr.Button("+ Add Approximator", scale=0, variant="primary")
                remove_col_btn = gr.Button(
                    "- Remove Approximator", scale=0, variant="primary", interactive=False
                )
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
                            value=get_plot(
                                df_agg, games[0], m, [approxs[i % len(approxs)]], _yranges.get(m)
                            ),
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
                updates = [
                    gr.update(elem_classes=[] if c_idx < new_n else ["hidden-col"])
                    for c_idx in range(MAX_COLS)
                ]

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
                            plot_figure = get_plot_single(
                                df_agg, game_val, m, approx_val, yr_shared
                            )
                            updates.append(gr.update(value=plot_figure, elem_classes=[]))
                        else:
                            updates.append(gr.update(elem_classes=["hidden-col"]))

                return [
                    new_n,
                    *updates,
                    gr.update(interactive=(new_n < MAX_COLS)),
                    gr.update(interactive=(new_n > 2)),
                ]

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
                    fn=update_compare_plots,
                    inputs=[*all_inputs, n_cols_state],
                    outputs=plot_outputs,
                )

        with gr.Tab("Detailed Data"):
            gr.Markdown("## Detailed Data\nDirect MongoDB query of individual runs.")

            with gr.Row():
                det_game = gr.Dropdown(choices=_all_games, label="Game", multiselect=True)
                det_approx = gr.Dropdown(
                    choices=_all_approxs, label="Approximator", multiselect=True
                )
                det_budget = gr.Dropdown(
                    choices=[str(b) for b in _all_budgets], label="Budget", multiselect=True
                )
                det_index = gr.Dropdown(choices=_all_indices, label="Index", multiselect=True)
                det_max_order = gr.Dropdown(
                    choices=[str(o) for o in _all_max_orders], label="Max Order", multiselect=True
                )

            with gr.Row():
                det_n_players = gr.Dropdown(
                    choices=[str(n) for n in _all_n_players], label="N Players", multiselect=True
                )
                det_gt_method = gr.Dropdown(
                    choices=_all_gt_methods, label="Ground Truth Method", multiselect=True
                )
                det_seed = gr.Dropdown(
                    choices=[str(s) for s in _all_seeds], label="Approx Seed", multiselect=True
                )
                det_metric = gr.Dropdown(
                    choices=available_metrics, label="Metrics", multiselect=True
                )

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

                yield (
                    f"**{len(result_df)} Runs gefunden.**",
                    gr.update(value=result_df, visible=True),
                )
                await asyncio.sleep(0.05)
                yield gr.update(), gr.update(visible=False)
                await asyncio.sleep(0.05)
                yield gr.update(), gr.update(value=result_df, visible=True)

            _det_filters = [
                det_game,
                det_approx,
                det_budget,
                det_index,
                det_metric,
                det_n_players,
                det_max_order,
                det_seed,
                det_gt_method,
            ]

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
            new_raw = db_client.get_all()
            new_df = process_raw_runs(new_raw)
            update_global_styles(new_df)

            games = new_df["game_name"].unique().tolist()
            approxs = new_df["approximator_name"].unique().tolist()
            first_game = games[0]

            outputs: list[Any] = [
                new_df,
                new_raw,
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

        elo_jump_btn.click(
            fn=lambda approxs, budget_idx, metric, index, game: (
                [game] if game != "all" else [],
                approxs or [],
                [str(BUDGET_BUCKETS[budget_idx]["budget"])],
                [index] if index != "all" else [],
                [metric] if metric != "all" else [],
            ),
            inputs=[
                elo_approx_filter,
                elo_bucket_idx_state,
                elo_metric_filter,
                elo_index_filter,
                elo_game_filter,
            ],
            outputs=[det_game, det_approx, det_budget, det_index, det_metric],
        ).then(
            fn=query_raw,
            inputs=_det_filters,
            outputs=[det_count, det_table],
        ).then(
            fn=lambda: gr.Info("Daten geladen — bitte zum 'Detailed Data' Tab wechseln"),
            inputs=[],
            outputs=[],
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

    return demo


def main() -> None:
    """Build and launch the Gradio leaderboard app."""
    print("\n🚀 Starting shapiq Leaderboard...\n")  # noqa: T201
    demo = build_app()
    print("\n✨ Ready!\n")  # noqa: T201
    demo.launch()


if __name__ == "__main__":
    main()
