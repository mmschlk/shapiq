import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_PATH = "results_raw.jsonl"


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
        n_seeds=("seed", "count"),
    ).reset_index()

    return agg


def get_leaderboard_global(df_agg: pd.DataFrame) -> pd.DataFrame:
    # Über alle Games aggregieren
    global_agg = df_agg.groupby(["approximator_name", "budget"]).agg(
        mse_mean=("mse_mean", "mean"),
        mse_std=("mse_std", "mean"),
        mae_mean=("mae_mean", "mean"),
        n_seeds=("n_seeds", "sum"),
    ).reset_index()

    best = global_agg.loc[global_agg.groupby("approximator_name")["mse_mean"].idxmin()]
    best = best.sort_values("mse_mean")
    best = best.rename(columns={
        "approximator_name": "Approximator",
        "budget": "Best Budget",
        "mse_mean": "MSE (mean)",
        "mse_std": "MSE (std)",
        "mae_mean": "MAE (mean)",
        "n_seeds": "Seeds",
    })
    return best[["Approximator", "Best Budget", "MSE (mean)", "MSE (std)", "MAE (mean)", "Seeds"]].round(20)


def get_leaderboard_game(df_agg: pd.DataFrame, selected_game: str) -> pd.DataFrame:
    df_filtered = df_agg[df_agg["game_name"] == selected_game]
    best = df_filtered.loc[df_filtered.groupby("approximator_name")["mse_mean"].idxmin()]
    best = best.sort_values("mse_mean")
    best = best.rename(columns={
        "approximator_name": "Approximator",
        "budget": "Best Budget",
        "mse_mean": "MSE (mean)",
        "mse_std": "MSE (std)",
        "mae_mean": "MAE (mean)",
        "n_seeds": "Seeds",
    })
    return best[["Approximator", "Best Budget", "MSE (mean)", "MSE (std)", "MAE (mean)", "Seeds"]].round(20)


def get_plot(df_agg: pd.DataFrame, selected_game: str):
    df_filtered = df_agg[df_agg["game_name"] == selected_game]
    fig, ax = plt.subplots(figsize=(8, 5))

    for approx_name, group in df_filtered.groupby("approximator_name"):
        group = group.sort_values("budget")
        ax.plot(group["budget"], group["mse_mean"], marker="o", label=approx_name)
        ax.fill_between(
            group["budget"],
            group["mse_mean"] - group["mse_std"].fillna(0),
            group["mse_mean"] + group["mse_std"].fillna(0),
            alpha=0.2,
        )

    ax.set_xlabel("Budget (coalition evaluations)")
    ax.set_ylabel("MSE (mean over seeds)")
    ax.set_title("MSE vs. Budget")
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
        game_dropdown = gr.Dropdown(
            choices=df_agg["game_name"].unique().tolist(),
            value=df_agg["game_name"].iloc[0],
            label="Game"
        )

        plot = gr.Plot(
            value=get_plot(df_agg, df_agg["game_name"].iloc[0])  # direkt beim Start rendern
        )
        game_dropdown.change(
            fn=lambda g: get_plot(df_agg, g),
            inputs=game_dropdown,
            outputs=plot
        )

demo.launch()
