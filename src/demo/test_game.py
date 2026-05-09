import numpy as np
import shapiq
from sklearn.ensemble import RandomForestRegressor
from shapiq.approximator import ProxySHAP
import os
import csv
import time

# --- 1. Daten laden und Modell trainieren ---
print("Lade Daten und trainiere Modell...")
X, y = shapiq.load_california_housing(to_numpy=True)
n_features = X.shape[1]

model = RandomForestRegressor(max_depth=4, random_state=42)
model.fit(X, y)

# --- 2. Vorbereitungen für die Game Function ---
x_explain = X[0] 
reference_values = X.mean(axis=0)

def game_function(coalitions: np.ndarray) -> np.ndarray:
    """Value function for explaining x_explain."""
    batch_size = coalitions.shape[0]
    mask = coalitions.astype(bool)
    inputs = np.tile(reference_values, (batch_size, 1))
    inputs[mask] = np.tile(x_explain, (batch_size, 1))[mask]
    return model.predict(inputs)

# --- 3. Benchmark ausführen (mit Zeitmessung) ---
print("Führe Benchmark mit ProxySHAP aus...")
budget_to_test = 2048

# Zeitmessung starten
start_time = time.time()

approx = ProxySHAP(n=n_features, random_state=42)
proxy_sv = approx.approximate(budget=budget_to_test, game=game_function)

# Zeitmessung stoppen
end_time = time.time()
runtime_sec = round(end_time - start_time, 2)

# --- 4. Dummy-Fehler berechnen (später mit Ground Truth ersetzen) ---
# Für dieses Skript simulieren wir einfach einen MSE-Fehler
# Ein höheres Budget sollte tendenziell zu einem kleineren Fehler führen
error_value = round(10 / budget_to_test, 4) 

# --- 5. Ergebnis in die CSV-Datei speichern ---
csv_file_path = "src/demo/shapiq_benchmark_results.csv"

# Parameter für die CSV-Zeile vorbereiten
game_name = "BikeSharingDemand"
approximator_name = "ProxySHAP"
index = "SII"
seed = 42
metric = "MSE"

headers = ["game_name", "n_players", "approximator_name", "index", "budget", "seed", "metric", "value", "runtime_sec"]
file_exists = os.path.isfile(csv_file_path)

print(f"Speichere Ergebnis in {csv_file_path}...")

# Datei im "Append"-Modus ('a') öffnen, um Zeile anzuhängen
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(headers)
    
    writer.writerow([
        game_name, 
        n_features, 
        approximator_name, 
        index, 
        budget_to_test, 
        seed, 
        metric, 
        error_value, 
        runtime_sec
    ])

print("Erfolgreich abgeschlossen!")