"""
Data ingestion module for Jena Climate dataset.

Loads the raw CSV, validates schema, and returns a cleaned DataFrame
with parsed datetime index.
"""

import os
import pandas as pd


EXPECTED_COLUMNS = [
    "Date Time", "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)",
    "rh (%)", "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)",
    "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)",
    "wv (m/s)", "max. wv (m/s)", "wd (deg)",
]


def ingest(cfg, project_root=None):
    """Load and validate the Jena Climate CSV.

    Parameters
    ----------
    cfg : OmegaConf DictConfig or dict
        Must contain ``cfg.data.file`` with the CSV path (relative to project root).
    project_root : str, optional
        Base directory for resolving relative paths.  When *None* the current
        working directory is used.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``Date Time`` parsed as datetime, sorted, deduplicated,
        and with stripped column names.
    """
    csv_path = cfg.data.file if hasattr(cfg, "data") else cfg["data"]["file"]
    if project_root:
        csv_path = os.path.join(project_root, csv_path)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Validate expected columns
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Parse datetime
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")

    # Deduplicate and sort
    df = df.drop_duplicates(keep="first")
    df = df.sort_values("Date Time").reset_index(drop=True)

    date_min = df["Date Time"].min().strftime("%Y-%m-%d")
    date_max = df["Date Time"].max().strftime("%Y-%m-%d")
    print(f"[Ingestion] Loaded {len(df):,} records, {len(df.columns)} columns")
    print(f"[Ingestion] Date range: {date_min} to {date_max}")
    print(f"[Ingestion] Original interval: 10 min")

    return df
