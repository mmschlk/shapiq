"""UI components for the leaderboard."""

from __future__ import annotations

import logging

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from leaderboard.storage.connection import MongoDBClient, MongoDBConnectionError
from leaderboard.ui.ui_exceptions import UnknownDataLoadingMethodError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

RESULTS_PATH = "results_raw.jsonl"
LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
LOADING_METHOD = "mongodb"  # "local" or "mongodb"

# Temporary seed determination
SEED_IDs = ["approx_seed", "seed"]  # List of possible seed identifier columns in the raw data


def reload_data() -> pd.DataFrame:
    """Reloads the raw data and re-aggregates it, returning the updated aggregated DataFrame."""
    return load_and_aggregate(method=LOADING_METHOD, path=RESULTS_PATH)


def load_and_aggregate(method: str = "mongodb", path: str = RESULTS_PATH) -> pd.DataFrame:
    """Loads raw run data from the specified source, processes it, and returns an aggregated DataFrame."""
    if method == "local":
        runs_df = _local_load(path)
    elif method == "mongodb":
        # Create a client and load data from MongoDB
        mongoDBClient = MongoDBClient.from_env()

        # Check if we can connect to the database
        if not mongoDBClient.test_connection():
            raise MongoDBConnectionError from None

        runs_df = _mongodb_load(mongoDBClient)
    else:
        raise UnknownDataLoadingMethodError(method)

    # If runs_df is empty - populate it with a dummy entry to avoid errors
    if runs_df.empty:
        runs_df = pd.DataFrame(
            [
                {
                    "game_name": "N/A",
                    "approximator_name": "N/A",
                    "budget": 0,
                    "mse": 0,
                    "mae": 0,
                    "ground_truth_method": "N/A",
                    "runtime_seconds": 0,
                    "approx_seed": 0,
                }
            ]
        )

    # Rename "seed" column to "approx_seed" if it exists, for consistency
    if "seed" in runs_df.columns and "approx_seed" not in runs_df.columns:
        runs_df = runs_df.rename(columns={"seed": "approx_seed"})

    # If now there is both a seed column and an approx_seed column, drop the "seed" column
    # Copy seed values to approx_seed if they exist, to avoid losing data
    if "seed" in runs_df.columns and "approx_seed" in runs_df.columns:
        runs_df["approx_seed"] = runs_df["approx_seed"].combine_first(runs_df["seed"])
        runs_df = runs_df.drop(columns=["seed"])

    return _aggregate(runs_df)


def _mongodb_load(mongoDBClient: MongoDBClient) -> pd.DataFrame:
    """Loads all runs from MongoDB and aggregates them into the format used
    by the implementation of the leaderboard ui and logic.

    Returns a DataFrame with columns:
        game_name, approximator_name, budget,
        mse_mean, mse_std, mae_mean, mae_std,
        ground_truth_method,
        runtime_mean, runtime_min, runtime_max,
        n_seeds
    """
    # Fetch all raw runs from the database
    raw_runs = mongoDBClient.get_all()

    if not raw_runs:
        return pd.DataFrame()

    raw_runs_df = pd.DataFrame(raw_runs)

    # Drop failed runs, matching the original filter
    raw_runs_df = raw_runs_df[~raw_runs_df["run_failed"]]

    # Flatten the nested metrics dict (same as pd.json_normalize in original)
    metrics_df = pd.json_normalize(raw_runs_df["metrics"])

    return pd.concat([raw_runs_df.drop(columns=["metrics"]), metrics_df], axis=1)


def _local_load(path: str) -> pd.DataFrame:
    """Loads raw run data from a local JSONL file, filters out failed runs, and flattens the metrics dict."""
    runs_df = pd.read_json(path, lines=True)
    runs_df = runs_df[~runs_df["run_failed"]]

    # metrics-Dict auseinandernehmen
    metrics_df = pd.json_normalize(runs_df["metrics"])

    return pd.concat([runs_df.drop(columns=["metrics"]), metrics_df], axis=1)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates the raw run data by game_name, approximator_name, and budget, computing mean and std for mse and mae."""
    # Aggregieren over seeds
    agg = (
        df.groupby(["game_name", "approximator_name", "budget"])
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            ground_truth_method=("ground_truth_method", "first"),
            runtime_mean=("runtime_seconds", "mean"),
            runtime_min=("runtime_seconds", "min"),
            runtime_max=("runtime_seconds", "max"),
            n_seeds=("approx_seed", "count"),
        )
        .reset_index()
    )

    logging.info("Aggregated data shape: %s", agg.shape)

    return agg


def get_leaderboard_global(df_agg: pd.DataFrame) -> pd.DataFrame:
    """Computes the global leaderboard by finding the best (lowest) MSE for each approximator across all games and budgets.

    Returns: per game aggregated DataFrame.
    """
    # Über alle Games aggregieren
    global_agg = (
        df_agg.groupby(["approximator_name", "budget"])
        .agg(
            mse_mean=("mse_mean", "mean"),
            mse_std=("mse_std", "mean"),
            mae_mean=("mae_mean", "mean"),
            mae_std=("mae_std", "mean"),
            ground_truth_method=("ground_truth_method", "first"),
            runtime_mean=("runtime_mean", "mean"),
            runtime_min=("runtime_min", "min"),
            runtime_max=("runtime_max", "max"),
            n_seeds=("n_seeds", "sum"),
        )
        .reset_index()
    )

    best = global_agg.loc[global_agg.groupby("approximator_name")["mse_mean"].idxmin()]
    best = best.sort_values("mse_mean")
    best = best.rename(
        columns={
            "approximator_name": "Approximator",
            "budget": "Budget at best MSE",
            "mse_mean": "MSE (mean)",
            "mse_std": "MSE (std)",
            "mae_mean": "MAE (mean)",
            "mae_std": "MAE (std)",
            "ground_truth_method": "GT Method",
            "runtime_mean": "Runtime mean (s)",
            "runtime_min": "Runtime min (s)",
            "runtime_max": "Runtime max (s)",
            "n_seeds": "Seeds",
        }
    )

    runtime_cols = ["Runtime mean (s)", "Runtime min (s)", "Runtime max (s)"]

    def format_value(col: str, x: float) -> str:
        if not isinstance(x, float):
            return x
        if col in runtime_cols:
            return round(x, 4)
        return f"{x:.4e}"

    leaderboard_df = best[
        [
            "Approximator",
            "Budget at best MSE",
            "MSE (mean)",
            "MSE (std)",
            "MAE (mean)",
            "MAE (std)",
            "GT Method",
            "Runtime mean (s)",
            "Runtime min (s)",
            "Runtime max (s)",
            "Seeds",
        ]
    ].copy()
    for col in leaderboard_df.columns:
        leaderboard_df[col] = leaderboard_df[col].apply(lambda x, col=col: format_value(col, x))

    return leaderboard_df


def get_leaderboard_game(df_agg: pd.DataFrame, selected_game: str) -> pd.DataFrame:
    """Computes the leaderboard for a specific game by finding the best (lowest) MSE for each approximator across all budgets for that game.
    Returns: per game aggregated DataFrame.
    """
    df_filtered = df_agg[df_agg["game_name"] == selected_game]
    best = df_filtered.loc[df_filtered.groupby("approximator_name")["mse_mean"].idxmin()]
    best = best.sort_values("mse_mean")
    best = best.rename(
        columns={
            "approximator_name": "Approximator",
            "budget": "Budget at best MSE",
            "mse_mean": "MSE (mean)",
            "mse_std": "MSE (std)",
            "mae_mean": "MAE (mean)",
            "mae_std": "MAE (std)",
            "ground_truth_method": "GT Method",
            "runtime_mean": "Runtime mean (s)",
            "runtime_min": "Runtime min (s)",
            "runtime_max": "Runtime max (s)",
            "n_seeds": "Seeds",
        }
    )

    runtime_cols = ["Runtime mean (s)", "Runtime min (s)", "Runtime max (s)"]

    def format_value(col: str, x: float) -> str:
        if not isinstance(x, float):
            return x
        if col in runtime_cols:
            return round(x, 4)
        return f"{x:.4e}"

    leaderboard_df = best[
        [
            "Approximator",
            "Budget at best MSE",
            "MSE (mean)",
            "MSE (std)",
            "MAE (mean)",
            "MAE (std)",
            "GT Method",
            "Runtime mean (s)",
            "Runtime min (s)",
            "Runtime max (s)",
            "Seeds",
        ]
    ].copy()

    for col in leaderboard_df.columns:
        leaderboard_df[col] = leaderboard_df[col].apply(lambda x, col=col: format_value(col, x))

    return leaderboard_df


def get_plot(df_agg: pd.DataFrame, selected_game: str, metric: str = "mse") -> plt.Figure:
    """Generates a line plot of the specified metric (MSE or MAE) vs. budget for the selected game, with separate lines for each approximator and shaded areas representing standard deviation across seeds."""
    df_filtered = df_agg[df_agg["game_name"] == selected_game]
    fig, ax = plt.subplots(figsize=(8, 5))

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
        style = LINE_STYLES[i % len(LINE_STYLES)]
        sorted_group = group.sort_values("budget")

        ax.plot(
            sorted_group["budget"],
            sorted_group[mean_col],
            marker="o",
            linestyle=style,
            label=approx_name,
        )
        ax.fill_between(
            sorted_group["budget"],
            sorted_group[mean_col] - sorted_group[std_col].fillna(0),
            sorted_group[mean_col] + sorted_group[std_col].fillna(0),
            alpha=0.2,
        )

    ax.set_xlabel("Budget (coalition evaluations)")
    ax.set_ylabel(f"{metric.upper()} (mean over seeds)")
    ax.set_title(f"{metric.upper()} vs. Budget")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(visible=True, alpha=0.3)
    plt.tight_layout()
    return fig


# --- Daten laden ---
df_agg = load_and_aggregate(method=LOADING_METHOD, path=RESULTS_PATH)

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

    with gr.Tab("MSE vs. Budget"):
        game_dropdown_mse = gr.Dropdown(
            choices=df_agg["game_name"].unique().tolist(),
            value=df_agg["game_name"].iloc[0],
            label="Game",
        )

        plot_mse = gr.Plot(
            value=get_plot(df_agg, df_agg["game_name"].iloc[0])  # direkt beim Start rendern
        )
        game_dropdown_mse.change(
            fn=lambda g, df: get_plot(df, g, "mse"),
            inputs=[game_dropdown_mse, df_state],
            outputs=plot_mse,
        )

    with gr.Tab("MAE vs. Budget"):
        game_dropdown_mae = gr.Dropdown(
            choices=df_agg["game_name"].unique().tolist(),
            value=df_agg["game_name"].iloc[0],
            label="Game",
        )
        plot_mae = gr.Plot(value=get_plot(df_agg, df_agg["game_name"].iloc[0], "mae"))
        game_dropdown_mae.change(
            fn=lambda g, df: get_plot(df, g, "mae"),
            inputs=[game_dropdown_mae, df_state],
            outputs=plot_mae,
        )

    def on_reload() -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        gr.Dropdown,
        pd.DataFrame,
        gr.Dropdown,
        plt.Figure,
        gr.Dropdown,
        plt.Figure,
    ]:
        """Reloads the raw data, re-aggregates it, and updates all components with the new data."""
        new_df = reload_data()
        games = new_df["game_name"].unique().tolist()
        first_game = games[0]
        return (
            new_df,
            get_leaderboard_global(new_df),
            gr.Dropdown(choices=games, value=first_game),
            get_leaderboard_game(new_df, first_game),
            gr.Dropdown(choices=games, value=first_game),
            get_plot(new_df, first_game, "mse"),
            gr.Dropdown(choices=games, value=first_game),
            get_plot(new_df, first_game, "mae"),
        )

    reload_btn.click(
        fn=on_reload,
        inputs=[],
        outputs=[
            df_state,
            global_leaderboard,  # gr.Dataframe for global leaderboard
            game_dropdown_lb,
            game_leaderboard,
            game_dropdown_mse,
            plot_mse,
            game_dropdown_mae,
            plot_mae,
        ],
    )


demo.launch()
