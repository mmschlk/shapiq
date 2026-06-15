"""UI components for the leaderboard."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

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

LOADING_METHOD = "mongodb"  # "local" or "mongodb"

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
    """Fills the global mapping based on ALL approximators in the dataset.

    This ensures that each approximator retains a unique and stable color and
    dash style across all plots.
    """
    all_approxs = sorted(df["approximator_name"].unique().tolist())

    GLOBAL_APPROX_STYLES.clear()

    for idx, approx in enumerate(all_approxs):
        GLOBAL_APPROX_STYLES[approx] = {
            "color": COLORS[idx % len(COLORS)],
            "dash": DASH_STYLES[idx % len(DASH_STYLES)],
        }


def reload_data() -> pd.DataFrame:
    """Reloads the raw data and re-aggregates it, returning the updated aggregated DataFrame."""
    return load_and_aggregate(method=LOADING_METHOD, path=RESULTS_PATH)


def load_and_aggregate(method: str = "mongodb", path: str | Path = RESULTS_PATH) -> pd.DataFrame:
    """Loads raw run data from the specified source, processes it, and returns an aggregated DataFrame."""
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
    """Builds the leaderboard DataFrame with formatted and ordered columns."""
    avail_metrics = sorted(
        [m for m in selected_metrics if f"{m}_mean" in df.columns]
    )

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
    """Computes the global leaderboard across all games and budgets.

    Aggregates performance metrics by calculating the mean over all games
    for each approximator and budget configuration, then structures the
    leaderboard using the best configuration found.
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
    """Computes the leaderboard for a specific game.

    Filters the data for the selected game and approximators, then determines
    the best budget configuration based on the lowest MSE.
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
    """Generates a line plot of the specified metric across budgets.

    Plots separate lines for each approximator and shaded areas representing
    the standard deviation across seeds for the selected game.
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
    """Generates a line plot for a single specified approximator.

    Wraps the generic get_plot function by encapsulating the single
    approximator string into a list.
    """
    return get_plot(df_agg, selected_game, metric, [approximator], yaxis_range)


# --- Daten laden ---
df_agg = load_and_aggregate(method=LOADING_METHOD, path=RESULTS_PATH)
update_global_styles(df_agg)

available_metrics = [m for m in METRICS if f"{m}_mean" in df_agg.columns]


def compute_yranges(g: str, approximators: list[str]) -> dict[str, list[float] | None]:
    """Computes common y-axis ranges for a game and given approximators.

    Filters out values below the defined threshold to ensure clean log-scaling
    across all available metrics.
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
    """Updates all comparison plots dynamically based on selected configurations.

    Calculates common y-axis limits strictly using only the visible columns
    and filters out extreme or zero-bound variations.
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


def compute_elo_for_bucket(df_raw_records: list[dict], budget: int) -> tuple[pd.DataFrame, go.Figure]:
    """Run ELO scoring for a specific budget bucket and return table + plot.

    Args:
        df_raw_records: Raw benchmark records as a list of dicts.
        budget: The budget value to filter by (e.g. 250, 500, 1000, 5000, 10000).

    Returns:
        A tuple of (leaderboard DataFrame, Plotly bar chart Figure).
    """
    scorer = EloScorer(budgets=[budget])
    result = scorer.score(df_raw_records)

    if not result.rows:
        empty_df = pd.DataFrame(columns=["Rank", "Approximator", "ELO Score", "Matches", "Wins", "Losses", "Ties"])
        fig = go.Figure()
        fig.update_layout(title=f"No data for budget {budget}")
        return empty_df, fig

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
        GLOBAL_APPROX_STYLES.get(name, {"color": "#636EFA"})["color"]
        for name in approx_names
    ]

    fig = go.Figure(
        go.Bar(
            x=approx_names,
            y=elo_scores,
            marker_color=bar_colors,
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

    return leaderboard_df, fig


# --- Gradio App ---
with gr.Blocks(title="shapiq Leaderboard") as demo:
    gr.Markdown("""
    # shapiq Approximator Leaderboard
    Comparison of Shapley value approximators across games, budgets, and seeds.
    """)

    # Store df in gradio state to allow reloading
    df_state = gr.State(value=df_agg)

    with gr.Row():
        reload_btn = gr.Button("Reload Data", variant="secondary", scale=0)

    with gr.Tab("Leaderboard"):
        gr.Markdown("## Global Leaderboard (all games)")

        with gr.Row():
            lb_approx_filter = gr.CheckboxGroup(
                choices=df_agg["approximator_name"].unique().tolist(),
                value=df_agg["approximator_name"].unique().tolist(),
                label="Approximators",
            )
            lb_metric_filter = gr.CheckboxGroup(
                choices=available_metrics,
                value=available_metrics,
                label="Metrics",
            )

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

    for metric in available_metrics:
        with gr.Tab(f"{metric.upper()} across Budgets"):
            game_dropdowns[metric] = gr.Dropdown(
                choices=df_agg["game_name"].unique().tolist(),
                value=df_agg["game_name"].iloc[0],
                label="Game",
            )
            approx_checkboxes[metric] = gr.CheckboxGroup(
                choices=df_agg["approximator_name"].unique().tolist(),
                value=df_agg["approximator_name"].unique().tolist(),
                label="Approximatoren",
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

        with gr.Row():
            add_col_btn = gr.Button("+ Add Approximator", scale=0)
            remove_col_btn = gr.Button("- Remove", scale=0)
            n_cols_state = gr.State(value=DEFAULT_COLS)

        # Pro Spalte: Approximator-Dropdown + Game-Dropdown
        compare_column_containers = []
        compare_approx_dropdowns = []
        compare_game_dropdowns = []

        approxs = df_agg["approximator_name"].unique().tolist()
        games = df_agg["game_name"].unique().tolist()

        with gr.Row():
            for i in range(MAX_COLS):
                with gr.Column(visible=(i < DEFAULT_COLS)) as col:
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
                        )
                        if i < DEFAULT_COLS
                        else None,
                        visible=(i < DEFAULT_COLS),
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
            updates = [gr.update(visible=(c_idx < new_n)) for c_idx in range(MAX_COLS)]

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
                        updates.append(gr.update(value=plot_figure, visible=True))
                    else:
                        updates.append(gr.update(visible=False))

            return [new_n, *updates]

        click_inputs = [n_cols_state, *compare_approx_dropdowns, *compare_game_dropdowns]

        add_col_btn.click(
            fn=lambda n, *args: update_col_visibility(n, +1, *args),
            inputs=click_inputs,
            outputs=[n_cols_state, *all_col_components],
        )

        remove_col_btn.click(
            fn=lambda n, *args: update_col_visibility(n, -1, *args),
            inputs=click_inputs,
            outputs=[n_cols_state, *all_col_components],
        )

        # Plot Update bei Aenderung
        all_inputs = compare_approx_dropdowns + compare_game_dropdowns
        plot_outputs = [p for m in available_metrics for p in compare_plot_rows[m]]

        for component in all_inputs:
            component.change(
                fn=update_compare_plots, inputs=[*all_inputs, n_cols_state], outputs=plot_outputs
            )

    with gr.Tab("ELO Leaderboard"):
        gr.Markdown("""
        ## ELO Leaderboard by Budget Bucket

        Approximators are ranked using pairwise ELO comparisons within each budget tier.
        A higher ELO score means the approximator consistently outperforms others at this budget.
        """)

        # State: current bucket index (0–4), start with Medium (1000) = index 2
        elo_bucket_idx_state = gr.State(value=2)

        with gr.Row(equal_height=True):
            elo_prev_btn = gr.Button("◀ Lower Budget", scale=0, variant="secondary")
            elo_bucket_label = gr.Markdown(
                value=f"### {BUDGET_BUCKETS[2]['label']}",
                elem_id="elo-bucket-label",
            )
            elo_next_btn = gr.Button("Higher Budget ▶", scale=0, variant="secondary")

        # Pre-compute initial ELO display for Medium (1000) bucket
        _elo_init_records = []
        for _, _row in df_agg.iterrows():
            _elo_init_records.append({
                "run_id": f"{_row['game_name']}-{_row['approximator_name']}-{_row.get('budget', 0)}",
                "game_id": _row["game_name"],
                "game_name": _row["game_name"],
                "index": _row.get("index", "SV"),
                "max_order": _row.get("max_order", 1),
                "budget": int(_row["budget"]) if not pd.isna(_row["budget"]) else 0,
                "ground_truth_method": _row.get("ground_truth_method", ""),
                "approximator_name": _row["approximator_name"],
                "run_failed": False,
                "metrics": {
                    m: _row.get(f"{m}_mean")
                    for m in available_metrics
                    if f"{m}_mean" in df_agg.columns and not pd.isna(_row.get(f"{m}_mean"))
                },
            })
        _elo_init_table, _elo_init_fig = compute_elo_for_bucket(_elo_init_records, BUDGET_BUCKETS[2]["budget"])

        with gr.Row():
            with gr.Column(scale=1):
                elo_table = gr.Dataframe(
                    value=_elo_init_table,
                    interactive=False,
                    label="Rankings",
                )
            with gr.Column(scale=2):
                elo_plot = gr.Plot(value=_elo_init_fig, label="ELO Scores")

        def update_elo_tab(bucket_idx: int, df: pd.DataFrame) -> tuple[Any, ...]:
            """Compute ELO leaderboard and plot for the given bucket index."""
            bucket = BUDGET_BUCKETS[bucket_idx]
            # Convert aggregated df back to raw-record-like dicts for the ELO scorer.
            # The ELO scorer expects raw records with nested "metrics" dict.
            # We reconstruct minimal records from the aggregated data.
            raw_records = []
            for _, row in df.iterrows():
                for metric in available_metrics:
                    mean_col = f"{metric}_mean"
                    if mean_col not in df.columns:
                        continue
                    mean_val = row.get(mean_col)
                    if pd.isna(mean_val):
                        continue
                record = {
                    "run_id": f"{row['game_name']}-{row['approximator_name']}-{row.get('budget', 0)}-{row.get('seed', 0)}",
                    "game_id": row["game_name"],
                    "game_name": row["game_name"],
                    "index": row.get("index", "SV"),
                    "max_order": row.get("max_order", 1),
                    "budget": int(row["budget"]) if not pd.isna(row["budget"]) else 0,
                    "ground_truth_method": row.get("ground_truth_method", ""),
                    "approximator_name": row["approximator_name"],
                    "run_failed": False,
                    "metrics": {
                        metric: row.get(f"{metric}_mean")
                        for metric in available_metrics
                        if f"{metric}_mean" in df.columns and not pd.isna(row.get(f"{metric}_mean"))
                    },
                }
                raw_records.append(record)

            table_df, fig = compute_elo_for_bucket(raw_records, bucket["budget"])
            label_md = f"### {bucket['label']}"
            return label_md, table_df, fig

        def elo_navigate(current_idx: int, delta: int, df: pd.DataFrame) -> tuple[Any, ...]:
            """Navigate between budget buckets."""
            new_idx = max(0, min(len(BUDGET_BUCKETS) - 1, current_idx + delta))
            label_md, table_df, fig = update_elo_tab(new_idx, df)
            return new_idx, label_md, table_df, fig

        elo_prev_btn.click(
            fn=lambda idx, df: elo_navigate(idx, -1, df),
            inputs=[elo_bucket_idx_state, df_state],
            outputs=[elo_bucket_idx_state, elo_bucket_label, elo_table, elo_plot],
        )
        elo_next_btn.click(
            fn=lambda idx, df: elo_navigate(idx, +1, df),
            inputs=[elo_bucket_idx_state, df_state],
            outputs=[elo_bucket_idx_state, elo_bucket_label, elo_table, elo_plot],
        )

    def on_reload() -> tuple[Any, ...]:
        """Reloads the raw dataset and refreshes all UI components.

        Fetches the fresh data from the source, updates the global visualization
        styles, and rebuilds all tables and plots for the interface.
        """
        new_df = reload_data()
        update_global_styles(new_df)

        games = new_df["game_name"].unique().tolist()
        approxs = new_df["approximator_name"].unique().tolist()
        first_game = games[0]

        outputs: list[Any] = [
            new_df,
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

demo.launch()
