"""Random Forest training and held-out test-set evaluation.

The model is trained on the training set only. Evaluation metrics come from
the test set, which the model never saw during training.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from .config import RF_PARAMS


def train_rf(X_train, y_train, **params):
    """Fit a Random Forest on the training data and return the fitted model."""
    model = RandomForestClassifier(**{**RF_PARAMS, **params})
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    """Evaluate on the held-out test set: accuracy, confusion matrix, ROC-AUC."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "n_test": len(y_test),
    }
