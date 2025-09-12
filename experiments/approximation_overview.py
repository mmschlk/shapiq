from __future__ import annotations

import pandas as pd

if __name__ == "__main__":
    results_df = pd.read_csv("results_benchmark.csv")
    results_df = results_df.sort_values(by="n_players")
