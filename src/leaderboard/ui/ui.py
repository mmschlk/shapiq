import gradio as gr
import pandas as pd
import plotly.graph_objects as go

RESULTS_PATH = "results_raw.jsonl"
ZERO_THRESHOLD = 1e-7
DASH_STYLES = ["solid", "dash", "dot", "dashdot", "longdash"]


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

def format_value(col, x, runtime_cols):
    if not isinstance(x, (float, int)):
        return x
    if isinstance(x, float) and abs(x) < ZERO_THRESHOLD and col not in runtime_cols:
        return "0"
    if col in runtime_cols:
        return round(x, 4)
    return f"{x:.4e}"


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

    df = best[["Approximator", "Budget at best MSE", "MSE (mean)", "MSE (std)", "MAE (mean)", "MAE (std)", "GT Method",
               "Runtime mean (s)", "Runtime min (s)", "Runtime max (s)", "Seeds"]].copy()

    for col in df.columns:
        df[col] = df[col].apply(lambda x: format_value(col, x, runtime_cols))
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

    df = best[["Approximator", "Budget at best MSE", "MSE (mean)", "MSE (std)", "MAE (mean)", "MAE (std)", "GT Method",
               "Runtime mean (s)", "Runtime min (s)", "Runtime max (s)", "Seeds"]].copy()
    for col in df.columns:
        df[col] = df[col].apply(lambda x: format_value(col, x, runtime_cols))
    return df


def get_plot(df_agg: pd.DataFrame, selected_game: str, metric, selected_approximators):
    df_filtered = df_agg[
        (df_agg["game_name"] == selected_game) &
        (df_agg["approximator_name"].isin(selected_approximators))
    ]
    fig = go.Figure()

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    for i, (approx_name, group) in enumerate(df_filtered.groupby("approximator_name")):
        dash = DASH_STYLES[i % len(DASH_STYLES)]
        group = group.sort_values("budget")

        # Linie
        fig.add_trace(go.Scatter(
            x=group["budget"],
            y=group[mean_col],
            mode="lines+markers",
            name=approx_name,
            line=dict(dash=dash),
        ))

        # Fehlerband
        fig.add_trace(go.Scatter(
            x=pd.concat([group["budget"], group["budget"].iloc[::-1]]),
            y=pd.concat([
                group[mean_col] + group[std_col].fillna(0),
                (group[mean_col] - group[std_col]).fillna(0).iloc[::-1]
            ]),
            fill="toself",
            fillcolor="rgba(0,100,255,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            hoverinfo="skip",
            name=approx_name,
        ))

    fig.update_layout(
        title=f"{metric.upper()} vs. Budget - {selected_game}",
        xaxis_title="Budget (coalition evaluations)",
        yaxis_title=f"{metric.upper()} (mean over seeds)",
        yaxis_type="log",
        legend_title="Approximator",
        hovermode="x unified",
    )

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

        approx_checkboxes = gr.CheckboxGroup(
            choices=df_agg["approximator_name"].unique().tolist(),
            value=df_agg["approximator_name"].unique().tolist(),  # alle standardmäßig an
            label="Approximatoren"
        )

        plot_mse = gr.Plot(
            value=get_plot(df_agg, df_agg["game_name"].iloc[0], "mse", df_agg["approximator_name"].unique().tolist())  # direkt beim Start rendern
        )

        for component in [game_dropdown_mse, approx_checkboxes]:
            component.change(
                fn=lambda g, a: get_plot(df_agg, g, "mse", a),
                inputs=[game_dropdown_mse, approx_checkboxes],
                outputs=plot_mse
        )

    with gr.Tab("MAE vs. Budget"):
        game_dropdown_mae = gr.Dropdown(
            choices=df_agg["game_name"].unique().tolist(),
            value=df_agg["game_name"].iloc[0],
            label="Game"
        )

        approx_checkboxes = gr.CheckboxGroup(
            choices=df_agg["approximator_name"].unique().tolist(),
            value=df_agg["approximator_name"].unique().tolist(),  # alle standardmäßig an
            label="Approximatoren"
        )

        plot_mae = gr.Plot(
            value=get_plot(df_agg, df_agg["game_name"].iloc[0], "mae", df_agg["approximator_name"].unique().tolist())
        )

        for component in [game_dropdown_mae, approx_checkboxes]:
            component.change(
                fn=lambda g, a: get_plot(df_agg, g, "mae", a),
                inputs=[game_dropdown_mae, approx_checkboxes],
                outputs=plot_mae
            )

demo.launch()
