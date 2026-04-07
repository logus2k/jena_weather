"""Data ingestion - load, validate, and clean the raw Jena Climate dataset."""

import pandas as pd
from pathlib import Path


def load_dataset(file_path):
    """Load CSV dataset and return raw DataFrame."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    return df


def remove_duplicates(df):
    """Remove duplicated rows and return cleaned DataFrame."""
    n_before = len(df)
    df = df.drop_duplicates().copy()
    n_removed = n_before - len(df)
    return df, n_removed


def parse_datetime(df, time_col="Date Time"):
    """Parse datetime column and sort chronologically."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], dayfirst=True)
    df = df.sort_values(time_col).reset_index(drop=True)
    return df


def validate_dataset(df, time_col="Date Time"):
    """Run quality checks and return a summary dict."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "missing_total": int(df.isna().sum().sum()),
        "missing_per_column": df.isna().sum().to_dict(),
        "date_range": (str(df[time_col].min()), str(df[time_col].max())),
        "column_names": df.columns.tolist(),
    }


def ingest(file_path, time_col="Date Time"):
    """Full ingestion pipeline: load -> validate -> clean -> parse datetime.

    Returns (df, summary) where summary contains quality check results.
    """
    df = load_dataset(file_path)
    pre_summary = validate_dataset(df, time_col)

    df, n_duplicates_removed = remove_duplicates(df)
    df = parse_datetime(df, time_col)

    post_summary = validate_dataset(df, time_col)
    post_summary["duplicates_removed"] = n_duplicates_removed

    return df, post_summary
