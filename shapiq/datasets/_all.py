"""This module contains functions to load datasets."""

import os

import pandas as pd

GITHUB_DATA_URL = "https://github.com/mmschlk/shapiq/raw/main/data/"


def load_bike() -> pd.DataFrame:
    """Load the bike-sharing dataset from a Kaggle competition.

    Original source: https://www.kaggle.com/c/bike-sharing-demand

    Note:
        The function and preprocessing is taken from the `sage` package.

    Returns:
        The bike-sharing dataset as a pandas DataFrame.
    """
    data = pd.read_csv(os.path.join(GITHUB_DATA_URL, "bike.csv"))
    columns = data.columns.tolist()

    # Split and remove datetime column.
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["year"] = data["datetime"].dt.year
    data["month"] = data["datetime"].dt.month
    data["day"] = data["datetime"].dt.day
    data["hour"] = data["datetime"].dt.hour
    data = data.drop("datetime", axis=1)

    # Reorder and rename columns.
    data = data[["year", "month", "day", "hour"] + columns[1:]]
    data.columns = list(map(str.title, data.columns))

    return data
