"""Pipeline sanity tests. Small checks, automatic, forever.

Run with:  python -m pytest tests/ -q
"""

import pandas as pd
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
    """Analysis dataset = validated raw file with duplicate records removed."""
    assert clean.shape == (302, 14)
    assert list(clean.columns) == FEATURES + [TARGET]
    assert clean.isna().sum().sum() == 0
    assert set(clean[TARGET].unique()) <= {0, 1}
    assert not clean.duplicated().any()  # the leakage guard


def test_split_preserves_disease_ratio(split):
    """The 80/20 split keeps the same disease ratio in both halves."""
    X_train, X_test, y_train, y_test = split
    assert len(X_train) == 241
    assert len(X_test) == 61
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_no_patient_leaks_across_the_split(split):
    """No patient record may sit in both train and test.

    Regression test for the duplicate-records bug: with copies left in,
    202 of 205 test rows had an identical twin in training.
    """
    X_train, X_test, y_train, y_test = split
    train_rows = set(map(tuple, pd.concat([X_train, y_train], axis=1).to_numpy()))
    test_rows = set(map(tuple, pd.concat([X_test, y_test], axis=1).to_numpy()))
    assert train_rows.isdisjoint(test_rows)


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
