"""Train/test split. This happens BEFORE any modeling or importance work.

The split is stratified on the target so both halves keep the same disease
ratio, and the seed is fixed so the split is identical on every run.
The test set is locked away until evaluation.
"""

from sklearn.model_selection import train_test_split

from .config import FEATURES, SEED, TARGET, TEST_SIZE


def stratified_split(df, test_size=TEST_SIZE, seed=SEED):
    """Return X_train, X_test, y_train, y_test with matching disease ratios."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
