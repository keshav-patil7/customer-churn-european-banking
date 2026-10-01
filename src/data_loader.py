"""Data loader module for European Banking Customer Churn project.

Handles path resolution and robust loading of raw and processed datasets.
"""
from pathlib import Path
from typing import Optional
import pandas as pd


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_data_path(filename: str = "European_Bank.csv", processed: bool = False) -> Path:
    """Resolve data file path checking standard locations."""
    root = get_project_root()
    if processed:
        candidates = [
            root / "data" / "processed" / filename,
            root / "data" / "processed" / "European_Bank_cleaned.csv",
            root / "data" / filename,
        ]
    else:
        candidates = [
            root / "data" / "raw" / filename,
            root / "data" / filename,
            root / filename,
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not locate dataset '{filename}'. Checked paths: {[str(p) for p in candidates]}"
    )


def load_raw_data(filename: str = "European_Bank.csv") -> pd.DataFrame:
    """Load the raw European Banking dataset into a pandas DataFrame."""
    path = get_data_path(filename, processed=False)
    df = pd.read_csv(path)
    return df


def load_cleaned_data(filename: str = "European_Bank_cleaned.csv") -> pd.DataFrame:
    """Load the processed and feature-engineered European Banking dataset."""
    try:
        path = get_data_path(filename, processed=True)
        return pd.read_csv(path)
    except FileNotFoundError:
        # If processed data is not yet generated, fallback to cleaning and engineering raw data
        from src.data_cleaning import clean_data
        from src.feature_engineering import add_features

        raw_df = load_raw_data()
        cleaned_df = clean_data(raw_df)
        final_df = add_features(cleaned_df)
        return final_df
