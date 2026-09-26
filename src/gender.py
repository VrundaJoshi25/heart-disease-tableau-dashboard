"""Gender subgroup analysis: does feature importance differ between
male and female patients?

Protocol per subgroup:
- Point estimate: train a Random Forest on the subgroup's training rows,
  measure permutation importance on the subgroup's held-out test rows.
- Uncertainty: bootstrap the subgroup's training rows N_BOOTSTRAP times,
  refit each time, re-measure permutation importance on the same fixed
  test rows, and take the 2.5th/97.5th percentiles as the interval.

If a feature ranks high for one gender and low for the other with
non-overlapping intervals, the gender difference is supported. Overlapping
intervals mean "suggestive, not conclusive at this sample size".

Bootstrap iterations are independent (each uses seed + iteration), so they
are parallelized with joblib without affecting reproducibility.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score

from .config import (
    BOOTSTRAP_PERM_REPEATS,
    BOOTSTRAP_RF_PARAMS,
    FEATURES,
    N_BOOTSTRAP,
    N_PERM_REPEATS,
    SEED,
    TARGET,
)
from .importance import permutation_importance_table
from .model import train_rf

GENDERS = {1: "Male", 0: "Female"}


def _subset(X, y, sex_value):
    mask = X["sex"] == sex_value
    return X[mask], y[mask]


def _fast_permutation_importance(model, X, y, seed, n_repeats):
    """Lightweight permutation importance (accuracy drop), no sklearn wrapper.

    Used inside the bootstrap loop where sklearn's permutation_importance
    call overhead would dominate the runtime.
    """
    baseline = accuracy_score(y, model.predict(X))
    rng = np.random.default_rng(seed)
    X_shuffled = X.copy()
    drops = np.zeros(len(X.columns))
    for col_idx, col in enumerate(X.columns):
        original = X[col].to_numpy()
        col_drops = []
        for _ in range(n_repeats):
            X_shuffled[col] = rng.permutation(original)
            shuffled_score = accuracy_score(y, model.predict(X_shuffled))
            col_drops.append(baseline - shuffled_score)
        X_shuffled[col] = original
        drops[col_idx] = np.mean(col_drops)
    return drops


def _bootstrap_one(b, Xtr, ytr, Xte, yte, seed):
    """One bootstrap resample: refit on resampled training rows, re-measure
    importance on the fixed test rows. Module-level so joblib can pickle it."""
    rng = np.random.default_rng(seed + b)
    idx = rng.integers(0, len(Xtr), size=len(Xtr))  # resample with replacement
    model = train_rf(
        Xtr.iloc[idx],
        ytr.iloc[idx],
        **{**BOOTSTRAP_RF_PARAMS, "random_state": seed + b},
    )
    return _fast_permutation_importance(
        model, Xte, yte, seed=seed + b, n_repeats=BOOTSTRAP_PERM_REPEATS
    )


def gender_point_estimate(X_train, X_test, y_train, y_test, sex_value, seed=SEED):
    """Permutation importance for one gender subgroup (no intervals)."""
    Xtr, ytr = _subset(X_train, y_train, sex_value)
    Xte, yte = _subset(X_test, y_test, sex_value)
    model = train_rf(Xtr, ytr)
    return permutation_importance_table(
        model, Xte, yte, seed=seed, n_repeats=N_PERM_REPEATS
    )


def bootstrap_gender(X_train, X_test, y_train, y_test, sex_value,
                     n_bootstrap=N_BOOTSTRAP, seed=SEED, n_jobs=-1):
    """Bootstrap confidence intervals for one gender subgroup."""
    Xtr, ytr = _subset(X_train, y_train, sex_value)
    Xte, yte = _subset(X_test, y_test, sex_value)

    collected = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_one)(b, Xtr, ytr, Xte, yte, seed)
        for b in range(n_bootstrap)
    )
    collected = np.asarray(collected)

    point = gender_point_estimate(
        X_train, X_test, y_train, y_test, sex_value, seed=seed
    )
    point = point.set_index("feature").loc[FEATURES].reset_index()
    return pd.DataFrame(
        {
            "feature": FEATURES,
            "group": GENDERS[sex_value],
            "importance": point["importance"].to_numpy(),
            "ci_low": np.percentile(collected, 2.5, axis=0),
            "ci_high": np.percentile(collected, 97.5, axis=0),
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)


def full_gender_analysis(X_train, X_test, y_train, y_test,
                         n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Point estimates + bootstrap intervals for both genders."""
    frames = [
        bootstrap_gender(X_train, X_test, y_train, y_test, sex_value,
                         n_bootstrap=n_bootstrap, seed=seed)
        for sex_value in (1, 0)  # Male first, then Female
    ]
    return pd.concat(frames, ignore_index=True)


def subgroup_sizes(df):
    """Row counts and disease rates per gender, for reporting."""
    counts = df["sex"].map(GENDERS).value_counts()
    target_rate = df.groupby(df["sex"].map(GENDERS))[TARGET].mean()
    return counts, target_rate
