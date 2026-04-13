"""Derive a single-year subset CSV from the full Jena Climate dataset.

This is the derivation script for `data/jena_climate_2012.csv`, which is
a one-year slice of `data/jena_climate_2009_2016.csv`. The slice is used
for fast-iteration experiments where a full 8-year training pass is too
slow to iterate on.

Usage (from the jena_weather project root):

    python -m src.data.filter_year \
        --input data/jena_climate_2009_2016.csv \
        --year 2012 \
        --output data/jena_climate_2012.csv

After running, remember to `dvc add data/jena_climate_2012.csv` and
`dvc push` so the derived file is tracked and available to collaborators.

Note: this script preserves the CSV verbatim (same columns, same dtypes,
same row ordering) - it only filters rows by the `Date Time` column's
year component. No resampling, no feature engineering.
"""

from __future__ import annotations

import argparse
import pandas as pd


def filter_year(input_path: str, output_path: str, year: int) -> int:
    """Read the input CSV, keep only rows whose 'Date Time' belongs to `year`,
    and write the filtered data to `output_path`. Returns the row count
    of the output (excluding the header)."""
    df = pd.read_csv(input_path)
    # The raw CSV stores timestamps as DD.MM.YYYY HH:MM:SS strings.
    parsed = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    mask = parsed.dt.year == year
    filtered = df.loc[mask]
    filtered.to_csv(output_path, index=False, quoting=1)  # quoting=1 = QUOTE_ALL for header stability
    return len(filtered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source CSV path")
    parser.add_argument("--output", required=True, help="Destination CSV path")
    parser.add_argument("--year", type=int, required=True, help="Calendar year to extract")
    args = parser.parse_args()
    n = filter_year(args.input, args.output, args.year)
    print(f"Wrote {n} rows for year={args.year} to {args.output}")


if __name__ == "__main__":
    main()
