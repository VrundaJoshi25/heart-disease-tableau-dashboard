"""Feature importance, computed three ways.

1. Built-in impurity importance: fast, but known to favor continuous
   features and to split credit between correlated ones.
2. Permutation importance on the held-out test set: shuffle one feature at
   a time and measure how much accuracy drops. This is the honest ranking,
   because it measures what the model actually relies on, on data it never
   saw during training.
3. SHAP values: same "what matters" question, but with direction attached
   (does a high value of this feature push risk up or down?).
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from .config import FEATURES, N_PERM_REPEATS, SEED


def builtin_importance(model):
    """Impurity-based importance from the fitted Random Forest."""
    table = pd.DataFrame(
        {"feature": FEATURES, "importance": model.feature_importances_}
    )
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation_importance_table(model, X, y, seed=SEED, n_repeats=N_PERM_REPEATS):
    """Permutation importance measured on the given (held-out) data."""
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=seed, n_jobs=-1
    )
    table = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": result.importances_mean,
            "std": result.importances_std,
        }
    )
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def shap_importance(model, X):
    """Mean |SHAP| per feature plus direction of effect on disease risk."""
    import shap  # imported here so the rest of the pipeline needs no shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # Binary classifier: either a list [class 0, class 1] or a single
    # array of shape (n_samples, n_features, 2). Keep class 1 = disease.
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1])
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[:, :, 1]

    rows = []
    for i, feature in enumerate(FEATURES):
        corr = float(spearmanr(X[feature].to_numpy(), values[:, i]).statistic)
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": np.abs(values[:, i]).mean(),
                "direction": "higher raises risk" if corr > 0 else "higher lowers risk",
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
