"""Project-wide configuration: paths, constants, and the fixed random seed.

Every module imports from here so that the whole pipeline is reproducible:
one seed, one set of paths, one definition of the feature list.
"""

from pathlib import Path

# Fixed seed for every stochastic step (split, Random Forest, permutation,
# bootstrap). Changing this one number changes every reported result.
SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "heart.csv"
DATA_CLEAN = ROOT / "data" / "clean" / "heart_clean.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
TABLEAU_EXPORT = DATA_PROCESSED / "tableau"
FIGURES = ROOT / "figures"

TARGET = "target"
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

TEST_SIZE = 0.2

# Random Forest hyperparameters. n_jobs does not affect results, only speed.
RF_PARAMS = {
    "n_estimators": 500,
    "random_state": SEED,
    "n_jobs": -1,
}

# Permutation importance repeats for the point estimates.
N_PERM_REPEATS = 30

# Gender subgroup bootstrap: number of resamples, and cheaper settings so
# that 2 x N_BOOTSTRAP model fits finish in minutes, not hours. The
# bootstrap iterations are parallelized across processes (n_jobs inside
# each fit is 1 to avoid oversubscription).
N_BOOTSTRAP = 500
BOOTSTRAP_RF_PARAMS = {
    "n_estimators": 200,
    "random_state": SEED,
    "n_jobs": 1,
}
BOOTSTRAP_PERM_REPEATS = 5
