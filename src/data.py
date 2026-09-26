"""Load and validate the raw heart-disease dataset, then write the clean copy.

The raw file (data/raw/heart.csv) is never modified. All validation happens
on a copy, which is saved to data/clean/heart_clean.csv.
"""

import pandas as pd

from .config import DATA_CLEAN, DATA_RAW, FEATURES, TARGET

EXPECTED_COLUMNS = FEATURES + [TARGET]

# Plausible value ranges for every column, from the dataset documentation.
# Used as a schema check: a value outside the range means the file changed.
VALUE_RANGES = {
    "age": (20, 90),
    "sex": (0, 1),
    "cp": (0, 3),
    "trestbps": (80, 220),
    "chol": (100, 700),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (60, 220),
    "exang": (0, 1),
    "oldpeak": (0.0, 7.0),
    "slope": (0, 2),
    "ca": (0, 4),
    "thal": (0, 3),
    "target": (0, 1),
}


def load_raw(path=DATA_RAW):
    """Read the raw CSV exactly as downloaded."""
    return pd.read_csv(path)


def validate(df):
    """Check schema, missing values, and value ranges. Raise on any failure."""
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    if df.isna().sum().sum() != 0:
        raise ValueError("Missing values found")
    for col, (lo, hi) in VALUE_RANGES.items():
        if not df[col].between(lo, hi).all():
            raise ValueError(f"Column {col!r} has values outside [{lo}, {hi}]")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError("target must be 0/1")
    return df


def build_clean(raw_path=DATA_RAW, out_path=DATA_CLEAN):
    """Load, validate, and save the analysis-ready dataset (all 1,025 rows)."""
    df = validate(load_raw(raw_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    clean = build_clean()
    print(f"Clean dataset: {clean.shape[0]} rows x {clean.shape[1]} columns")
    print(clean[TARGET].value_counts().to_string())
