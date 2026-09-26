"""Run the full analysis pipeline end to end.

One command reproduces every number in the README and every table that
feeds the Tableau dashboard:

    python -m src.run_analysis

Outputs:
    data/clean/heart_clean.csv                  validated analysis dataset
    data/processed/evaluation_metrics.csv       held-out test-set metrics
    data/processed/confusion_matrix.csv         held-out confusion matrix
    data/processed/importance_builtin.csv       impurity importance
    data/processed/importance_permutation.csv   permutation importance (test set)
    data/processed/importance_shap.csv          mean |SHAP| + direction
    data/processed/gender_importance_ci.csv     per-gender importance + 95% CI
    data/processed/tableau/*.csv                Tableau-ready exports
    figures/*.png                               charts for README / notebook
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .config import (
    DATA_PROCESSED,
    FIGURES,
    N_BOOTSTRAP,
    SEED,
    TARGET,
)
from .data import build_clean
from .gender import full_gender_analysis, subgroup_sizes
from .importance import (
    builtin_importance,
    permutation_importance_table,
    shap_importance,
)
from .model import evaluate, train_rf
from .split import stratified_split
from .tableau_export import export_importance_tableau, export_patients_labeled


def _save(df, name):
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / name
    df.to_csv(path, index=False)
    return path


def _plot_confusion_matrix(cm, metrics):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], labels=["No Disease", "Disease"])
    ax.set_yticks([0, 1], labels=["No Disease", "Disease"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"Held-out test set (n={metrics['n_test']}): "
        f"accuracy {metrics['accuracy']:.3f}"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_confusion_matrix.png", dpi=150)
    plt.close(fig)


def _plot_roc(model, X_test, y_test, metrics):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve, held-out test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_roc_curve.png", dpi=150)
    plt.close(fig)


def _plot_importance_methods(builtin, permutation, shap_table):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (builtin, "importance", "Built-in impurity importance"),
        (permutation, "importance", "Permutation importance (held-out test set)"),
        (shap_table, "mean_abs_shap", "SHAP mean |value| (test set)"),
    ]
    for ax, (table, col, title) in zip(axes, panels, strict=True):
        ordered = table.sort_values(col)
        ax.barh(ordered["feature"], ordered[col], color="#3b7ddd")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_importance_methods.png", dpi=150)
    plt.close(fig)


def _plot_gender_importance(gender_ci, order):
    male = gender_ci[gender_ci["group"] == "Male"].set_index("feature")
    female = gender_ci[gender_ci["group"] == "Female"].set_index("feature")
    y = np.arange(len(order))
    height = 0.38

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y + height / 2, male.loc[order, "importance"], height,
            xerr=[male.loc[order, "importance"] - male.loc[order, "ci_low"],
                  male.loc[order, "ci_high"] - male.loc[order, "importance"]],
            label="Male", color="#3b7ddd", capsize=3)
    ax.barh(y - height / 2, female.loc[order, "importance"], height,
            xerr=[female.loc[order, "importance"] - female.loc[order, "ci_low"],
                  female.loc[order, "ci_high"] - female.loc[order, "importance"]],
            label="Female", color="#e07b39", capsize=3)
    ax.set_yticks(y, labels=order)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation importance (accuracy drop when shuffled)")
    ax.set_title(
        f"Feature importance by gender, 95% bootstrap intervals "
        f"({N_BOOTSTRAP} resamples)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_gender_importance.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    print(f"Seed: {SEED}\n")

    # --- Data -----------------------------------------------------------
    df = build_clean()
    counts, target_rate = subgroup_sizes(df)
    print(f"Dataset: {len(df)} patients, {df.shape[1]} columns")
    print(f"Disease prevalence: {df[TARGET].mean():.3f}")
    print(f"Subgroups: {counts.to_dict()}, "
          f"disease rate {target_rate.round(3).to_dict()}\n")

    # --- Split BEFORE anything else -------------------------------------
    X_train, X_test, y_train, y_test = stratified_split(df)
    print(f"Split: {len(X_train)} train / {len(X_test)} test (stratified)")
    print(f"Disease ratio train {y_train.mean():.3f} / "
          f"test {y_test.mean():.3f}\n")

    # --- Model -----------------------------------------------------------
    model = train_rf(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    _save(
        pd.DataFrame(
            {
                "metric": ["accuracy", "roc_auc", "n_train", "n_test"],
                "value": [metrics["accuracy"], metrics["roc_auc"],
                          len(X_train), metrics["n_test"]],
            }
        ),
        "evaluation_metrics.csv",
    )
    _save(
        pd.DataFrame(
            metrics["confusion_matrix"],
            index=["actual_no_disease", "actual_disease"],
            columns=["pred_no_disease", "pred_disease"],
        ).reset_index(names="actual"),
        "confusion_matrix.csv",
    )
    print(f"Held-out test set: accuracy {metrics['accuracy']:.4f}, "
          f"ROC-AUC {metrics['roc_auc']:.4f}")
    print(f"Confusion matrix:\n{metrics['confusion_matrix']}\n")

    # --- Importance, three ways ------------------------------------------
    builtin = _save(builtin_importance(model), "importance_builtin.csv")
    permutation = permutation_importance_table(model, X_test, y_test)
    _save(permutation, "importance_permutation.csv")
    shap_table = shap_importance(model, X_test)
    _save(shap_table, "importance_shap.csv")
    print("Permutation importance (top 5):")
    print(permutation.head(5).to_string(index=False), "\n")

    # --- Gender analysis with bootstrap intervals -------------------------
    print(f"Gender analysis ({N_BOOTSTRAP} bootstrap resamples per gender)...")
    gender_ci = full_gender_analysis(X_train, X_test, y_train, y_test)
    _save(gender_ci, "gender_importance_ci.csv")
    print(gender_ci.to_string(index=False), "\n")

    # --- Tableau exports ---------------------------------------------------
    export_importance_tableau(permutation, gender_ci)
    export_patients_labeled(df)
    print("Tableau exports written to data/processed/tableau/\n")

    # --- Figures -------------------------------------------------------------
    _plot_confusion_matrix(metrics["confusion_matrix"], metrics)
    _plot_roc(model, X_test, y_test, metrics)
    _plot_importance_methods(
        pd.read_csv(builtin), permutation, shap_table
    )
    order = permutation["feature"].tolist()
    _plot_gender_importance(gender_ci, order)
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
