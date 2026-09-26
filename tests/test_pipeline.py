"""Pipeline sanity tests. Small checks, automatic, forever.

Run with:  python -m pytest tests/ -q
"""

import pytest

from src.config import FEATURES, TARGET
from src.data import build_clean
from src.importance import builtin_importance, permutation_importance_table
from src.model import train_rf
from src.split import stratified_split


@pytest.fixture(scope="module")
def clean():
    return build_clean()


@pytest.fixture(scope="module")
def split(clean):
    return stratified_split(clean)


def test_clean_data_integrity(clean):
    """The analysis dataset is exactly the validated raw file."""
    assert clean.shape == (1025, 14)
    assert list(clean.columns) == FEATURES + [TARGET]
    assert clean.isna().sum().sum() == 0
    assert set(clean[TARGET].unique()) <= {0, 1}


def test_split_preserves_disease_ratio(split):
    """The 80/20 split keeps the same disease ratio in both halves."""
    X_train, X_test, y_train, y_test = split
    assert len(X_train) == 820
    assert len(X_test) == 205
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_importance_tables_cover_all_features(split):
    """Every importance table ranks all 13 features, with no nulls."""
    X_train, X_test, y_train, y_test = split
    model = train_rf(X_train, y_train, n_estimators=50)  # small = fast test
    tables = [
        builtin_importance(model),
        permutation_importance_table(model, X_test, y_test, n_repeats=5),
    ]
    for table in tables:
        assert len(table) == 13
        assert set(table["feature"]) == set(FEATURES)
        assert table["importance"].notna().all()
