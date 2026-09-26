# A Visual Analysis of Heart Disease Prevalence

[![ci](https://github.com/VrundaJoshi25/heart-disease-tableau-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/VrundaJoshi25/heart-disease-tableau-dashboard/actions/workflows/ci.yml)

> **Interactive dashboard (Tableau Public):** [TODO — paste the published link here; build instructions in `tableau/build_spec.md`]
>
> **Question:** Which clinical and symptom patterns are associated with heart disease, and do those patterns differ between men and women?

Team project for **DS612 — Interactive Data Visualization** (Autumn 2026), M.Sc. Data Science, Dhirubhai Ambani University. Presented in class with the course instructor as project jury.

---

## Quickstart

```bash
git clone https://github.com/VrundaJoshi25/heart-disease-tableau-dashboard.git
cd heart-disease-tableau-dashboard
pip install -r requirements.txt
python -m src.run_analysis
```

One command reproduces every number in this README: cleaned dataset, held-out evaluation, three importance rankings, the gender analysis with bootstrap intervals, the Tableau-ready exports, and all figures. Seed is fixed at **42** (`src/config.py`); package versions are pinned (`requirements.txt`). `notebooks/analysis.ipynb` tells the same story interactively.

## Dataset

- **Source:** [Kaggle — Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) (file `heart.csv`; UCI/Statlog heart-disease family: Cleveland, Hungary, Switzerland, Long Beach VA, collected 1988)
- **Downloaded:** 2026-09-27. SHA-256 `DDB2996B2F4DB2E00AAD13F4518200179FF69F79093838E3C21FFA672EBEC0F1`
- **Size:** 1,025 patients × 14 columns (13 features + target)
- `data/raw/heart.csv` is never edited. `src/data.py` validates schema, value ranges, and missing values (there are none) and writes the analysis copy to `data/clean/heart_clean.csv` — every transformation is code, traceable.

### Data dictionary

| Column | Type | Meaning |
|---|---|---|
| `age` | numeric | Age in years (29–77) |
| `sex` | nominal | 0 = Female (312 patients), 1 = Male (713) |
| `cp` | nominal | Chest-pain type: 0 typical angina, 1 atypical angina, 2 non-anginal pain, 3 asymptomatic |
| `trestbps` | numeric | Resting blood pressure (mm Hg) |
| `chol` | numeric | Serum cholesterol (mg/dl) |
| `fbs` | nominal | Fasting blood sugar > 120 mg/dl (1 = yes) |
| `restecg` | nominal | Resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy |
| `thalach` | numeric | Maximum heart rate achieved |
| `exang` | nominal | Exercise-induced angina (1 = yes) |
| `oldpeak` | numeric | ST depression induced by exercise relative to rest |
| `slope` | ordinal | Slope of peak-exercise ST segment: 0 upsloping, 1 flat, 2 downsloping |
| `ca` | ordinal | Major vessels colored by fluoroscopy (0–4) |
| `thal` | nominal | Thalassemia test: 0 unknown, 1 fixed defect, 2 normal, 3 reversible defect |
| `target` | nominal | Heart disease diagnosis: 0 = no disease (499), 1 = disease (526) |

Disease prevalence: **51.3%** overall; 72.4% among female patients, 42.1% among male patients in this dataset.

## Method

1. **Split before anything else.** 820 train / 205 held-out test, stratified (disease ratio 0.5134 train / 0.5122 test), seed fixed. The test set is locked away until evaluation.
2. **Random Forest** (500 trees) trained on the training set only.
3. **Feature importance, three ways:**
   - *Built-in impurity importance* — shown for completeness; it is biased toward continuous features and splits credit between correlated ones.
   - *Permutation importance on the held-out test set* — shuffle one feature at a time, measure the accuracy drop. The honest ranking, and the one the dashboard shows.
   - *SHAP values* — same "what matters" question, with direction attached (does a higher value push predicted risk up or down?).
4. **Evaluate on the held-out test set** — accuracy, confusion matrix, ROC-AUC. We say *"predicts"* only because it was measured on patients the model never saw; patterns merely observed in the data are *"associated with"*.
5. **Gender analysis with a floor under it.** Permutation importance recomputed separately within the male and female subgroups, each with **500 bootstrap resamples** → a 95% interval per feature per gender. Non-overlapping intervals = the difference is supported; overlapping intervals = *suggestive, not conclusive* at this sample size.

## Results

### Held-out evaluation (205 patients the model never saw)

| Metric | Value |
|---|---|
| Accuracy | **1.000** (205/205) |
| ROC-AUC | **1.000** |
| Confusion matrix | 100 true negatives, 105 true positives, 0 errors |

On this dataset the 13 features separate the two classes perfectly on held-out data.

### What drives the predictions (permutation importance, held-out test set)

| Rank | Feature | Importance (accuracy drop) |
|---|---|---|
| 1 | `cp` (chest-pain type) | 0.080 |
| 2 | `ca` (vessels colored) | 0.072 |
| 3 | `thal` (thalassemia test) | 0.050 |
| 4 | `oldpeak` (ST depression) | 0.032 |
| 5 | `trestbps` | 0.015 |
| 6 | `sex` | 0.014 |
| 7 | `exang` | 0.013 |
| 8 | `age` | 0.012 |
| 9 | `thalach` | 0.008 |
| 10 | `chol` | 0.005 |
| 11–13 | `restecg`, `fbs`, `slope` | ≈ 0.000 |

No single feature carries the prediction — the signal is spread across symptom type (`cp`), diagnostic tests (`ca`, `thal`, `oldpeak`), and physiology. SHAP adds direction: higher `cp` code and higher `thalach` push predicted risk up; higher `oldpeak`, `ca`, `age`, `chol` push it down in this dataset (for coded categoricals like `thal`/`cp`, "higher" reads across category codes, not a clinical scale). `sex` alone ranks 6th of 13 — a weak standalone predictor whose role is *modifying* which features matter, which is exactly what the next section shows.

![Three importance methods](figures/fig_importance_methods.png)

### Feature importance shifts by gender (500 bootstrap resamples per subgroup)

| Feature | Male importance [95% CI] | Female importance [95% CI] | Verdict |
|---|---|---|---|
| `ca` | **0.112 [0.048, 0.124]** (rank 1) | 0.011 [-0.003, 0.019] (rank 4) | **Supported** — intervals do not overlap |
| `cp` | **0.078 [0.021, 0.104]** (rank 2) | 0.018 [-0.008, 0.042] (rank 3) | Suggestive |
| `thal` | 0.026 [-0.006, 0.046] (rank 4) | **0.035 [0.000, 0.074]** (rank 1) | Suggestive |
| `exang` | 0.000 [-0.017, 0.013] (rank 10) | **0.030 [0.000, 0.048]** (rank 2) | Suggestive |
| `oldpeak` | 0.033 [0.001, 0.058] (rank 3) | 0.000 [-0.013, 0.010] (rank 13) | Suggestive |

**Reading:** for men, predictions lean on vessel count (`ca`) and chest-pain type (`cp`); for women, on the thalassemia test (`thal`) and exercise-induced angina (`exang`). The `ca` gap is statistically supported (non-overlapping intervals); the other shifts point the same way but overlap, so they are reported as suggestive at this sample size — both kinds of result are stated plainly, because only one kind of wording survives an interview.

![Feature importance by gender](figures/fig_gender_importance.png)

## Interactive dashboard

The dashboard rebuilds the project's three views from these outputs (`tableau/build_spec.md` has click-by-click build + publish instructions):

1. **Feature importance, side by side for men and women** — the finding, visualized (from `data/processed/tableau/tableau_feature_importance.csv`).
2. **Prevalence heatmap** — chest-pain type × exercise-induced angina, split by gender.
3. **Grouped bars** — disease prevalence across clinical categories.

**Interactive filters (exact list): Gender, Age group, Chest-pain type.**

- Published link: [TODO — Tableau Public URL goes here]
- Published screenshot: `tableau/dashboard_overview.png` *(added once live)*
- Original course-version screenshots: `tableau/original_screenshots/`

## Tests and automation

`pytest tests/` runs three checks on every push via GitHub Actions (plus a `ruff` style check): the split preserves the disease ratio in both halves, every importance table covers all 13 features, and the clean dataset is exactly 1,025 × 14 with a valid target. Badge at the top of this file.

## Repository layout

```
data/raw/              original Kaggle CSV, never edited
data/clean/            validated analysis copy (created by code)
data/processed/        importance tables, evaluation metrics (created by code)
data/processed/tableau/ Tableau-ready exports
src/                   analysis modules (config, data, split, model, importance, gender, tableau_export, run_analysis)
notebooks/             analysis.ipynb — the story, importing from src
figures/               charts generated by the pipeline
tableau/               build spec, published-dashboard screenshot, original screenshots
tests/                 pytest pipeline checks
.github/workflows/     CI: pytest + ruff on every push
```

## Team

| Name | Student ID | Role |
|---|---|---|
| **Vrunda Joshi** | 202518043 | **Team lead** |
| Meet Gandhi | 202518023 | Team member |
| Kashyap Patel | 202518036 | Team member |
| Masum Udecha | 202518058 | Team member |

Presented to the class in Autumn 2026 with the course instructor as project jury.

## Reproducibility

Python 3.13, seed 42 everywhere (`src/config.py`), pinned versions in `requirements.txt`. A stranger who clones this repo and runs `python -m src.run_analysis` gets exactly these tables, figures, and rankings.
