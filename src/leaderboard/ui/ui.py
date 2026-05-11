import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_PATH = "results_raw.jsonl"
LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


def load_and_aggregate(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df = df[df["run_failed"] == False]

    # metrics-Dict auseinandernehmen
    metrics_df = pd.json_normalize(df["metrics"])
    df = pd.concat([df.drop("metrics", axis=1), metrics_df], axis=1)

    # Aggregieren über Seeds
    agg = df.groupby(["game_name", "approximator_name", "budget"]).agg(
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        ground_truth_method=("ground_truth_method", "first"),
        runtime_mean=("runtime_seconds", "mean"),
        runtime_min=("runtime_seconds", "min"),
        runtime_max=("runtime_seconds", "max"),
        n_seeds=("seed", "count"),
    ).reset_index()

    return agg


def get_leaderboard_global(df_agg: pd.DataFrame) -> pd.DataFrame:
    # Über alle Games aggregieren
    global_agg = df_agg.groupby(["approximator_name", "budget"]).agg(
        mse_mean=("mse_mean", "mean"),
        mse_std=("mse_std", "mean"),
        mae_mean=("mae_mean", "mean"),
        mae_std=("mae_std", "mean"),
        ground_truth_method=("ground_truth_method", "first"),
        runtime_mean=("runtime_mean", "mean"),
        runtime_min=("runtime_min", "min"),
        runtime_max=("runtime_max", "max"),
        n_seeds=("n_seeds", "sum"),
    ).reset_index()

    best = global_agg.loc[global_agg.groupby("approximator_name")["mse_mean"].idxmin()]
    best = best.sort_values("mse_mean")
    best = best.rename(columns={
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
    })

    runtime_cols = ["Runtime mean (s)", "Runtime min (s)", "Runtime max (s)"]

    def format_value(col, x):
        if not isinstance(x, float):
            return x
        if col in runtime_cols:
            return round(x, 4)
        return f"{x:.4e}"

    df = best[["Approximator", "Budget at best MSE", "MSE (mean)", "MSE (std)", "MAE (mean)", "MAE (std)", "GT Method",
               "Runtime mean (s)", "Runtime min (s)", "Runtime max (s)", "Seeds"]].copy()
    for col in df.columns:
        df[col] = df[col].apply(lambda x: format_value(col, x))
    return df


def get_leaderboard_game(df_agg: pd.DataFrame, selected_game: str) -> pd.DataFrame:
    df_filtered = df_agg[df_agg["game_name"] == selected_game]
    best = df_filtered.loc[df_filtered.groupby("approximator_name")["mse_mean"].idxmin()]
    best = best.sort_values("mse_mean")
    best = best.rename(columns={
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
    })

    runtime_cols = ["Runtime mean (s)", "Runtime min (s)", "Runtime max (s)"]

    def format_value(col, x):
        if not isinstance(x, float):
            return x
        if col in runtime_cols:
            return round(x, 4)
        return f"{x:.4e}"

    df = best[["Approximator", "Budget at best MSE", "MSE (mean)", "MSE (std)", "MAE (mean)", "MAE (std)", "GT Method",
               "Runtime mean (s)", "Runtime min (s)", "Runtime max (s)", "Seeds"]].copy()
    for col in df.columns:
        df[col] = df[col].apply(lambda x: format_value(col, x))
    return df


def get_plot(df_agg: pd.DataFrame, selected_game: str, metric: str = "mse"):
    df_filtered = df_agg[df_agg["game_name"] == selected_game]
    fig, ax = plt.subplots(figsize=(8, 5))

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    for i, (approx_name, group) in enumerate(df_filtered.groupby("approximator_name")):
        style = LINE_STYLES[i % len(LINE_STYLES)]
        group = group.sort_values("budget")
        ax.plot(group["budget"], group[mean_col], marker="o", linestyle=style, label=approx_name)
        ax.fill_between(
            group["budget"],
            group[mean_col] - group[std_col].fillna(0),
            group[mean_col] + group[std_col].fillna(0),
            alpha=0.2,
        )

    ax.set_xlabel("Budget (coalition evaluations)")
    ax.set_ylabel(f"{metric.upper()} (mean over seeds)")
    ax.set_title(f"{metric.upper()} vs. Budget")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# --- Daten laden ---
df_agg = load_and_aggregate(RESULTS_PATH)

# --- Gradio App ---
with gr.Blocks(title="shapiq Leaderboard") as demo:
    gr.Markdown("""
    # shapiq Approximator Leaderboard
    Comparison of Shapley value approximators across games, budgets, and seeds.
    """)

    with gr.Tab("Leaderboard"):
        gr.Markdown("## Global Leaderboard (all games)")
        gr.Dataframe(value=get_leaderboard_global(df_agg), interactive=False)

        gr.Markdown("## Per-Game Leaderboard")
        game_dropdown_lb = gr.Dropdown(
            choices=df_agg["game_name"].unique().tolist(),
            value=df_agg["game_name"].iloc[0],
            label="Game"
        )
        game_leaderboard = gr.Dataframe(
            value=get_leaderboard_game(df_agg, df_agg["game_name"].iloc[0]),
            interactive=False
        )
        game_dropdown_lb.change(
            fn=lambda g: get_leaderboard_game(df_agg, g),
            inputs=game_dropdown_lb,
            outputs=game_leaderboard
        )

    with gr.Tab("MSE vs. Budget"):
        game_dropdown_mse = gr.Dropdown(
            choices=df_agg["game_name"].unique().tolist(),
            value=df_agg["game_name"].iloc[0],
            label="Game"
        )

        plot_mse = gr.Plot(
            value=get_plot(df_agg, df_agg["game_name"].iloc[0])  # direkt beim Start rendern
        )
        game_dropdown_mse.change(
            fn=lambda g: get_plot(df_agg, g, "mse"),
            inputs=game_dropdown_mse,
            outputs=plot_mse
        )

    with gr.Tab("MAE vs. Budget"):
        game_dropdown_mae = gr.Dropdown(
            choices=df_agg["game_name"].unique().tolist(),
            value=df_agg["game_name"].iloc[0],
            label="Game"
        )
        plot_mae = gr.Plot(value=get_plot(df_agg, df_agg["game_name"].iloc[0], "mae"))
        game_dropdown_mae.change(
            fn=lambda g: get_plot(df_agg, g, "mae"),
            inputs=game_dropdown_mae,
            outputs=plot_mae
        )

demo.launch()
