"""UI components for the leaderboard."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeVar

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

from leaderboard.metrics import METRICS
from leaderboard.storage.connection import DatabaseClientFactory, DBConnectionError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()
T = TypeVar("T")

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent
RESULTS_PATH = PROJECT_ROOT / "data" / "results_raw.jsonl"

ZERO_THRESHOLD = 1e-7
DASH_STYLES = ["solid", "dash", "dot", "dashdot", "longdash"]
LOADING_METHOD = "mongodb"  # "local" or "mongodb"

# Temporary seed determination
SEED_IDs = ["approx_seed", "seed"]  # List of possible seed identifier columns in the raw data

# Compare-Tab: Globale Variablen
MAX_COLS = 5
DEFAULT_COLS = 2


def reload_data() -> pd.DataFrame:
    """Reloads the raw data and re-aggregates it, returning the updated aggregated DataFrame."""
    return load_and_aggregate(method=LOADING_METHOD, path=RESULTS_PATH)


def load_and_aggregate(method: str = "mongodb", path: str = RESULTS_PATH) -> pd.DataFrame:
    """Loads raw run data from the specified source, processes it, and returns an aggregated DataFrame."""
    db_client = DatabaseClientFactory.create_client(
        method, db_args={"LOCAL_DB_PATH": path} if method == "local" else {}
    )
    if not db_client.test_connection():
        raise DBConnectionError from None

    return db_client.load_dataframe()


def format_value(col: str, x: T, runtime_cols: list[str]) -> str:
    """Formats a value for display in the leaderboard, handling small values and runtime formatting.

    Args:
        col: The column name (used to determine if it's a runtime column).
        x: The value to format.
        runtime_cols: List of column names that represent runtimes, which should be rounded to 4 decimals.

    Returns:
        A formatted string representation of the value.
    """
    if not isinstance(x, float):
        return x
    if isinstance(x, float) and abs(x) < ZERO_THRESHOLD and col not in runtime_cols:
        return "0"
    if col in runtime_cols:
        return round(x, 4)
    return f"{x:.4e}"


def _build_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    avail_metrics = [m for m in METRICS if f"{m}_mean" in df.columns]

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
    metric_cols = [f"{m.upper()} (mean)" for m in avail_metrics] + [
        f"{m.upper()} (std)" for m in avail_metrics
    ]

    col_order = (
        ["Approximator", "Budget at best MSE"]
        + metric_cols
        + ["GT Method", "Runtime mean (s)", "Runtime min (s)", "Runtime max (s)", "Seeds"]
    )
    col_order = [c for c in col_order if c in best.columns]

    leaderboard_df = best[col_order].copy()
    for col in leaderboard_df.columns:
        leaderboard_df[col] = leaderboard_df[col].apply(
            lambda x, col=col: format_value(col, x, runtime_cols)
        )
    return leaderboard_df


def get_leaderboard_global(df_agg: pd.DataFrame) -> pd.DataFrame:
    """Computes the global leaderboard by finding the best (lowest) MSE for each approximator across all games and budgets.

    Returns: per game aggregated DataFrame.
    """
    # Aggregate over all games
    global_agg = (
        df_agg.groupby(["approximator_name", "budget"])
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

    return _build_leaderboard(global_agg)


def get_leaderboard_game(df_agg: pd.DataFrame, selected_game: str) -> pd.DataFrame:
    """Computes the leaderboard for a specific game by finding the best (lowest) MSE for each approximator across all budgets for that game.

    Returns: per game aggregated DataFrame.
    """
    df_filtered = df_agg[df_agg["game_name"] == selected_game]

    return _build_leaderboard(df_filtered)


def get_plot(
    df_agg: pd.DataFrame, selected_game: str, metric, selected_approximators, yaxis_range=None
) -> go.Figure:
    """Generates a line plot of the specified metric (MSE or MAE) across budgets for the selected game,
    with separate lines for each approximator and shaded areas representing standard deviation across seeds.
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

    for i, (approx_name, group) in enumerate(df_filtered.groupby("approximator_name")):
        dash = DASH_STYLES[i % len(DASH_STYLES)]
        sorted_group = group.sort_values("budget")

        # Linie
        fig.add_trace(
            go.Scatter(
                x=sorted_group["budget"],
                y=sorted_group[mean_col],
                mode="lines+markers",
                name=approx_name,
                line=dict(dash=dash),
            )
        )

        # Fehlerband
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
                fillcolor="rgba(0,100,255,0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip",
                name=approx_name,
            )
        )

    fig.update_layout(
        title=f"{metric.upper()} across Budgets - {selected_game}",
        xaxis_title="Budget (coalition evaluations)",
        yaxis_title=f"{metric.upper()} (mean over seeds)",
        yaxis=dict(
            type="log",
            range=yaxis_range,
        ),
        legend_title="Approximator",
        hovermode="x unified",
    )

    return fig


def get_plot_single(df_agg, selected_game, metric, approximator, yaxis_range=None) -> go.Figure:
    return get_plot(df_agg, selected_game, metric, [approximator], yaxis_range)


# --- Daten laden ---
df_agg = load_and_aggregate(method=LOADING_METHOD, path=RESULTS_PATH)

available_metrics = [m for m in METRICS if f"{m}_mean" in df_agg.columns]


def compute_yranges(g, approximators: list):
    df_filtered = df_agg[
        (df_agg["game_name"] == g) &
        (df_agg["approximator_name"].isin(approximators))
    ]
    yranges = {}
    for m in available_metrics:
        col = f"{m}_mean"
        y_min = max(df_filtered[col].min(), 1e-300)
        y_max = df_filtered[col].max()
        yranges[m] = [np.log10(y_min) - 0.5, np.log10(y_max) + 0.5]
    return yranges


def update_compare_plots(*args):
    # args enthält:
    # [0..4] Approximatoren, [5..9] Games, [10] Anzahl aktiver Spalten (n_cols)
    approx_vals = list(args[:MAX_COLS])
    game_vals = list(args[MAX_COLS:2 * MAX_COLS])
    n_cols = args[-1]

    # Approximatoren und Spiele herausfiltern, die gerade aktiv sichtbar sind
    active_approxs = approx_vals[:n_cols]
    active_games = game_vals[:n_cols]

    # Y-Achsen-Limits über alle aktiven Kombinationen hinweg
    yranges = {}
    for m in available_metrics:
        col = f"{m}_mean"

        df_filtered = df_agg[
            (df_agg["game_name"].isin(active_games)) &
            (df_agg["approximator_name"].isin(active_approxs))
            ]

        if not df_filtered.empty:
            y_min = max(df_filtered[col].min(), 1e-300)
            y_max = df_filtered[col].max()
            if pd.isna(y_min) or pd.isna(y_max):
                yranges[m] = None
            else:
                yranges[m] = [np.log10(y_min) - 0.5, np.log10(y_max) + 0.5]
        else:
            yranges[m] = None

    outputs = []
    for m in available_metrics:
        for i in range(MAX_COLS):
            outputs.append(get_plot_single(df_agg, game_vals[i], m, approx_vals[i], yranges.get(m)))

    return tuple(outputs)


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
        global_leaderboard = gr.Dataframe(value=get_leaderboard_global(df_agg), interactive=False)

        gr.Markdown("## Per-Game Leaderboard")
        game_dropdown_lb = gr.Dropdown(
            choices=df_agg["game_name"].unique().tolist(),
            value=df_agg["game_name"].iloc[0],
            label="Game",
        )
        game_leaderboard = gr.Dataframe(
            value=get_leaderboard_game(df_agg, df_agg["game_name"].iloc[0]), interactive=False
        )
        game_dropdown_lb.change(
            fn=lambda g, df: get_leaderboard_game(df, g),
            inputs=[game_dropdown_lb, df_state],
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
                    compare_approx_dropdowns.append(gr.Dropdown(
                        choices=approxs,
                        value=approxs[i % len(approxs)],
                        label=f"Approximator {i+1}",
                    ))
                    compare_game_dropdowns.append(gr.Dropdown(
                        choices=games,
                        value=games[0],
                        label="Game",
                    ))

        # Plots pro Metrik pro Spalte
        compare_plot_rows = {}  # compare_plot_rows[metric] = [plot0, plot1, ...]
        _yranges = compute_yranges(games[0], [approxs[0], approxs[1 % len(approxs)], approxs[2 % len(approxs)]])

        for m in available_metrics:
            gr.Markdown(f"### {m.upper()} across Budgets")
            with gr.Row():
                plots = []
                for i in range(MAX_COLS):
                    plots.append(gr.Plot(
                        value=get_plot(df_agg, games[0], m, [approxs[i % len(approxs)]], _yranges.get(m)) if i < DEFAULT_COLS else None,
                        visible=(i < DEFAULT_COLS),
                    ))
                compare_plot_rows[m] = plots

        # Spalte hinzufuegen/entfernen
        all_col_components = compare_column_containers + \
                             [p for m in available_metrics for p in compare_plot_rows[m]]

        def update_col_visibility(n, delta):
            new_n = max(2, min(MAX_COLS, n + delta))
            updates = []

            # Spalten-Container (Inklusive der darin liegenden Dropdowns) updaten
            for c_idx in range(MAX_COLS):
                updates.append(gr.update(visible=(c_idx < new_n)))

            # Plots updaten
            for m in available_metrics:
                for p_idx in range(MAX_COLS):
                    updates.append(gr.update(visible=(p_idx < new_n)))
            return [new_n] + updates


        add_col_btn.click(
            fn=lambda n: update_col_visibility(n, +1),
            inputs=[n_cols_state],
            outputs=[n_cols_state] + all_col_components
        )
        remove_col_btn.click(
            fn=lambda n: update_col_visibility(n, -1),
            inputs=[n_cols_state],
            outputs=[n_cols_state] + all_col_components
        )

        # Plot Update bei Aenderung
        all_inputs = compare_approx_dropdowns + compare_game_dropdowns
        plot_outputs = [p for m in available_metrics for p in compare_plot_rows[m]]

        for component in all_inputs:
            component.change(
                fn=update_compare_plots,
                inputs=all_inputs + [n_cols_state],
                outputs=plot_outputs
            )

    def on_reload() -> tuple[Any, ...]:
        """Reloads the raw data, re-aggregates it, and updates all components with the new data."""
        new_df = reload_data()
        games = new_df["game_name"].unique().tolist()
        approxs = new_df["approximator_name"].unique().tolist()
        first_game = games[0]

        outputs: list[Any] = [
            new_df,
            get_leaderboard_global(new_df),
            gr.Dropdown(choices=games, value=first_game),
            get_leaderboard_game(new_df, first_game),
        ]

        for m in available_metrics:
            outputs.append(gr.Dropdown(choices=games, value=first_game))
            outputs.append(gr.CheckboxGroup(choices=approxs, value=approxs))
            outputs.append(get_plot(new_df, first_game, m, approxs))

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
                for comp in [game_dropdowns[m], game_dropdowns[m], metric_plots[m]]
            ],
        ],
    )

demo.launch()
