"""Load, validate, and de-duplicate the raw heart-disease dataset.

The raw file (data/raw/heart.csv) is never modified. All validation and
cleaning happens on a copy, which is saved to data/clean/heart_clean.csv.

The cleaning step that matters most: the raw file lists 1,025 rows but
only 302 unique patients. In the version circulating on Kaggle, every
patient record was duplicated roughly 3-4 times (one 8 times). Left in,
those copies leak across any random train/test split (measured: 202 of
205 test rows would have an identical twin in training), so evaluation
would reward memorization instead of generalization. Exact duplicates
are removed BEFORE the split, keeping the first occurrence per patient.
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
    """Load, validate, drop exact duplicate patient records, and save.

    1,025 raw rows become 302 unique patients. The removed 723 copies are
    the difference between an honest evaluation and a memorization test.
    """
    df = validate(load_raw(raw_path))
    n_raw = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    print(
        f"Data-quality audit: {n_raw} raw rows, "
        f"{n_raw - len(df)} exact duplicate records removed, "
        f"{len(df)} unique patients kept"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    clean = build_clean()
    print(f"Clean dataset: {clean.shape[0]} rows x {clean.shape[1]} columns")
    print(clean[TARGET].value_counts().to_string())
