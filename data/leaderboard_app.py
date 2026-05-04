import gradio as gr
import pandas as pd
import os

# --- 1. Hilfsfunktion zum Laden oder Erstellen der Mock-Daten ---
def load_mock_data():
    csv_file = "shapiq_benchmark_results.csv"
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    else:
        # Fallback Mock-Daten
        data = [
            {"game_name": "CaliforniaHousing", "approximator_name": "KernelSHAP-IQ", "index": "SV", "budget": 100, "metric": "MSE", "value": 0.045},
            {"game_name": "CaliforniaHousing", "approximator_name": "SVARM-IQ", "index": "SV", "budget": 100, "metric": "MSE", "value": 0.038},
            {"game_name": "CaliforniaHousing", "approximator_name": "Permutation", "index": "SV", "budget": 100, "metric": "MSE", "value": 0.112},
            {"game_name": "CaliforniaHousing", "approximator_name": "KernelSHAP-IQ", "index": "SV", "budget": 500, "metric": "MSE", "value": 0.012},
            {"game_name": "CaliforniaHousing", "approximator_name": "SVARM-IQ", "index": "SV", "budget": 500, "metric": "MSE", "value": 0.015},
            {"game_name": "CaliforniaHousing", "approximator_name": "Permutation", "index": "SV", "budget": 500, "metric": "MSE", "value": 0.085},
            {"game_name": "Mutagenicity", "approximator_name": "KernelSHAP-IQ", "index": "SII", "budget": 1000, "metric": "MSE", "value": 0.008},
            {"game_name": "Mutagenicity", "approximator_name": "ProxySHAP", "index": "SII", "budget": 1000, "metric": "MSE", "value": 0.005},
        ]
        return pd.DataFrame(data)

df_all = load_mock_data()

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
# Das Theme wurde hier entfernt (für Gradio 6)
with gr.Blocks(title="shapiq Leaderboard") as demo:
    
    gr.Markdown("# 🏆 shapiq Benchmark Leaderboard")
    gr.Markdown("Wähle einen Datensatz, einen Index und ein Budget aus, um die Leistung der Approximatoren zu vergleichen.")
    
    with gr.Row():
        game_dropdown = gr.Dropdown(choices=["CaliforniaHousing", "Mutagenicity"], value="CaliforniaHousing", label="Datensatz / Game")
        index_dropdown = gr.Dropdown(choices=["SV", "SII"], value="SV", label="Interaktions-Index")
        # Wichtig: Wir wandeln das Budget in int um, da es in den Dropdowns manchmal als String übergeben wird
        budget_dropdown = gr.Dropdown(choices=[100, 500, 1000, 5000], value=100, label="Budget")
    
    gr.Markdown("### 📊 Metriken (MSE - Niedriger ist besser)")
    with gr.Row():
        # Der Parameter 'horizontal=True' wurde entfernt
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
    # Das Theme wird nun hier im launch() übergeben!
    demo.launch(theme=gr.themes.Base())