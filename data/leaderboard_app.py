import gradio as gr
import pandas as pd
import os

# --- 1. Hilfsfunktion zum Laden oder Erstellen der Mock-Daten ---
def load_data():
    csv_file = "data/shapiq_benchmark_results.csv"
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    else:
        # Fallback Mock-Daten, falls die CSV mal geloescht wird
        data = [
            {"game_name": "CaliforniaHousing", "approximator_name": "KernelSHAP-IQ", "index": "SV", "budget": 100, "metric": "MSE", "value": 0.045},
            {"game_name": "CaliforniaHousing", "approximator_name": "SVARM-IQ", "index": "SV", "budget": 100, "metric": "MSE", "value": 0.038},
        ]
        return pd.DataFrame(data)

# Wir laden die CSV-Daten EINMAL beim Start der App
df_all = load_data()

# NEU: Wir lesen alle einzigartigen (unique) Spiele, Indizes und Budgets direkt aus der Tabelle!
available_games = df_all["game_name"].unique().tolist()
available_indices = df_all["index"].unique().tolist()
available_budgets = sorted(df_all["budget"].unique().tolist()) # Sortiert die Budgets aufsteigend

# --- 2. Filter-Funktion ---
def filter_data(game, index_type, budget):
    df_filtered = df_all[
        (df_all["game_name"] == game) & 
        (df_all["index"] == index_type) & 
        (df_all["budget"] == budget)
    ]
    df_filtered = df_filtered.sort_values(by="value")
    return df_filtered, df_filtered

# --- 3. Das Gradio UI ---
with gr.Blocks(title="shapiq Leaderboard") as demo:
    
    gr.Markdown("# 🏆 shapiq Benchmark Leaderboard")
    gr.Markdown("Wähle einen Datensatz, einen Index und ein Budget aus, um die Leistung der Approximatoren zu vergleichen.")
    
    with gr.Row():
        # NEU: Wir geben die Listen aus der CSV an die Dropdowns und waehlen das erste Element als Standard-Wert
        game_dropdown = gr.Dropdown(choices=available_games, value=available_games[0], label="Datensatz / Game")
        index_dropdown = gr.Dropdown(choices=available_indices, value=available_indices[0], label="Interaktions-Index")
        budget_dropdown = gr.Dropdown(choices=available_budgets, value=available_budgets[0], label="Budget")
    
    gr.Markdown("### 📊 Metriken (MSE - Niedriger ist besser)")
    with gr.Row():
        error_plot = gr.BarPlot(
            x="value", 
            y="approximator_name", 
            title="Vergleich der Approximatoren (MSE)",
            tooltip=["approximator_name", "value"],
            x_title="Mean Squared Error (MSE)",
            y_title="Approximator",
            height=300
        )
        
    gr.Markdown("### 📄 Rohdaten (CSV)")
    with gr.Row():
        results_table = gr.Dataframe(headers=["game_name", "approximator_name", "index", "budget", "metric", "value"])

    # --- 4. Interaktivität verbinden ---
    inputs = [game_dropdown, index_dropdown, budget_dropdown]
    outputs = [results_table, error_plot]
    
    game_dropdown.change(fn=filter_data, inputs=inputs, outputs=outputs)
    index_dropdown.change(fn=filter_data, inputs=inputs, outputs=outputs)
    budget_dropdown.change(fn=filter_data, inputs=inputs, outputs=outputs)
    
    demo.load(fn=filter_data, inputs=inputs, outputs=outputs)

# --- 5. App starten ---
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base())