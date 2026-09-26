"""Tableau-ready exports.

Two files, written to data/processed/tableau/:

1. tableau_feature_importance.csv
   Long format: one row per feature per group (Overall / Male / Female),
   with the permutation importance and, for the gender groups, the 95%
   bootstrap interval. This powers the side-by-side gender view.

2. tableau_patients_labeled.csv
   The full patient-level dataset with human-readable labels for every
   coded variable, plus an age-group column. Prevalence is computed inside
   Tableau as AVG(target); this file powers the heatmap and grouped-bar
   views and carries the dashboard filters.
"""

import pandas as pd

from .config import TABLEAU_EXPORT

LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-anginal Pain",
        3: "Asymptomatic",
    },
    "fbs": {0: "FBS <= 120 mg/dl", 1: "FBS > 120 mg/dl"},
    "restecg": {
        0: "Normal",
        1: "ST-T Abnormality",
        2: "LV Hypertrophy",
    },
    "exang": {0: "No", 1: "Yes"},
    "slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
    "thal": {
        0: "Unknown",
        1: "Fixed Defect",
        2: "Normal",
        3: "Reversible Defect",
    },
    "target": {0: "No Disease", 1: "Disease"},
}

AGE_BINS = [28, 40, 50, 60, 70, 80]
AGE_LABELS = ["29-40", "41-50", "51-60", "61-70", "71-80"]


def export_patients_labeled(df, out_path=None):
    """Patient-level export with readable labels for every coded field."""
    if out_path is None:
        out_path = TABLEAU_EXPORT / "tableau_patients_labeled.csv"
    labeled = df.copy()
    for col, mapping in LABELS.items():
        labeled[f"{col}_label"] = labeled[col].map(mapping)
    labeled["age_group"] = pd.cut(
        labeled["age"], bins=AGE_BINS, labels=AGE_LABELS
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(out_path, index=False)
    return labeled


def export_importance_tableau(permutation_overall, gender_ci, out_path=None):
    """Long-format importance table: Overall rows plus Male/Female rows."""
    if out_path is None:
        out_path = TABLEAU_EXPORT / "tableau_feature_importance.csv"
    overall = permutation_overall.rename(columns={"importance": "importance"})[
        ["feature", "importance"]
    ].copy()
    overall["group"] = "Overall"
    overall["ci_low"] = pd.NA
    overall["ci_high"] = pd.NA

    gender = gender_ci[["feature", "group", "importance", "ci_low", "ci_high"]].copy()

    table = pd.concat(
        [overall[["feature", "group", "importance", "ci_low", "ci_high"]], gender],
        ignore_index=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    return table
