# Methods in Depth

This document gives a detailed, reproducibility-oriented description of the data processing
and modeling used in *"Global Prediction of Hospital Stay in Traumatic Brain Injury Using
Transfer Learning: A Multinational, Multicenter Study."* It is written to directly address the
five items raised in review:

1. the extent and pattern of missing data ([§3](#3-missing-data-extent-pattern-and-handling)),
2. the hyperparameter tuning process ([§5](#5-models-and-hyperparameters)),
3. the validation strategy ([§4](#4-validation-strategy)),
4. the number of repetitions performed ([§6](#6-repetitions-and-statistical-testing)), and
5. the specific implementation of the weighted transfer-learning approach ([§5.3](#53-weighted-transfer-learning-the-exact-implementation)).

Every step below maps to a specific notebook and cell so it can be located and re-run.
Reproducibility seeds: all data splits and estimators are seeded with `random_state=42`
(the standalone notebook uses `random_state=0` for the model in some cells — see [§9](#9-known-discrepancies-and-notes)).

---

## 1. Cohorts and overall design

Four TBI cohorts were analyzed: **India** (multi-center registry, TITCO), **Jordan**
(pediatric), **California** (adults with measured blood alcohol concentration), and **Florida**
(adults with severe blunt TBI in intensive care). The outcome is hospital **length of stay
(LOS)**, modeled as a right-censored **time-to-event** variable; in-hospital **mortality** is
the censoring indicator where available.

Cohort sizes after cleaning:

| Cohort | N | LOS, mean ± SD (days) | Mortality (%) |
|---|---|---|---|
| India | 7,978 | 8.2 ± 13.9 | 25.0 |
| California | 583 | 8.5 ± 10.9 | 7.0 |
| Florida | available on request | ≈49 ± 42 (reported) | — |
| Jordan | 112 | 12.0 ± 16.1 | 17.9 |

(Values computed from the cleaned CSVs in this repository; Florida figures are as reported in
the manuscript and are not redistributed here.)

The experiment is a **cross-cohort, few-shot transfer-learning protocol**: one cohort is the
**source domain** (used for pretraining) and another is the **target domain** (a small slice is
used for fine-tuning, the rest for testing). All 12 ordered source→target pairs were run
([analysis.ipynb](analysis.ipynb) cell 16; [transfer_learning_experiment.ipynb](transfer_learning_experiment.ipynb) cell 3).

---

## 2. Feature engineering and harmonization

All harmonization happens in [preprocessing.ipynb](preprocessing.ipynb).

**Common schema.** Each raw dataset is renamed/mapped to a common column set:
`age, sex, sbp, hr, rr, gcs, los, MOI, event` (cells 7–8). `sex` is coded 1 = male, 0 = female.
`event` = 1 if the patient died in hospital, else 0.

**LOS construction.** For India, LOS is computed as the difference between discharge and
admission timestamps (`dodd/todd` − `doa/toa`) in days; for the other cohorts, LOS is taken
directly from the reported length-of-stay field (cell 8). In the modeling notebooks LOS is
clipped to a minimum of 1 day (`df["los"].clip(lower=1)`) so that survival times are strictly
positive ([analysis.ipynb](analysis.ipynb) cell 0).

**Mechanism of injury (MOI).** Free-text/categorical MOI fields are mapped with a single
rule-based function (`map_moi`, cell 5) into a harmonized taxonomy:
Motor Vehicle Collision, Motorcycle Crash, Bicycle Crash, Pedestrian Struck, Fall,
Assault / Blunt Force, Gunshot Wound (GSW), Burn / Fire / Blast, Animal / Environmental, and
Other / Unknown. Rare or site-specific causes (e.g., "alligator bite") are folded into
*Animal / Environmental*; unmatched values fall through to *Other / Unknown*.

**Vital-sign categorization.** Continuous SBP, HR, and RR are converted to **age-specific
clinical categories** using pediatric/adult reference thresholds (cell 6):

- **Heart rate (`hr_cat`):** `tachycardia` vs `normal`, with age cutoffs (>160 if <1 y;
  >150 if <3 y; >140 if <6 y; >120 if <12 y; >100 otherwise).
- **Respiratory rate (`rr_cat`):** `tachypnea` vs `normal` (>60 if <1 y; >40 if <3 y;
  >34 if <6 y; >30 if <12 y; >20 otherwise).
- **SBP (`sbp_cat`):** `low` vs `normal` (<70 if <1 y; <70 + 2·age if <10 y; <90 otherwise).

For **Jordan**, the vital signs were already recorded as categories in the source data, so the
categorical columns are taken directly and the (uninformative, zero-filled) raw numeric vitals
are dropped (cell 8).

**Final cleaned columns:** `age, sex, sbp, hr, rr, gcs, los, MOI, event, hr_cat, rr_cat, sbp_cat`
(Jordan omits raw `sbp/hr/rr`). Files written in cell 16.

---

## 3. Missing data: extent, pattern, and handling

**Inspection.** Missingness was inspected with the `missingno` matrix and per-column null
counts ([analysis.ipynb](analysis.ipynb) cell 1, e.g. `msno.matrix(india); india.isnull().sum()`).
This visualizes both the **extent** (how many values are missing per feature) and the
**pattern** (whether missingness co-occurs across rows/columns).

> **Reporting note for the manuscript.** The CSVs shipped in this repo are *post-imputation*
> and therefore contain no missing values. To report per-feature missingness for the paper,
> run the snippet in [§8](#8-snippet-report-pre-imputation-missingness) on the **raw**
> (pre-imputation) datasets; this produces the count and percentage of missing values per
> feature per cohort that should accompany the methods.

**Imputation.** Missing numeric values are imputed per cohort with scikit-learn's
**`IterativeImputer`** (multivariate, model-based) using its default estimator, **Bayesian
Ridge regression**, seeded with `random_state=42` ([preprocessing.ipynb](preprocessing.ipynb) cell 8):

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(random_state=42)
numerical_cols = ['age', 'sex', 'sbp', 'hr', 'rr', 'gcs', 'los']
```

Key implementation details:
- Imputation is **fit and applied separately within each cohort** (no cross-cohort leakage).
- Only columns that exist and contain at least one observed value are imputed; a feature that
  is entirely absent in a given cohort is filled with `0` and excluded from imputation. (This
  affects Jordan's raw numeric vitals, which are absent and were dropped — see [§2](#2-feature-engineering-and-harmonization).)
- Vital-sign **categories** (`hr_cat/rr_cat/sbp_cat`) are derived **after** imputation from the
  (imputed) numeric values, except for Jordan where they come directly from source categories.

**Limitation of note.** Because LOS (the outcome) is included in the imputation feature set,
and imputation is done before splitting, imputed values can in principle carry information
across the train/test boundary. This is a standard simplification; a fully leakage-free
alternative would fit the imputer inside each training fold only.

---

## 4. Validation strategy

Two complementary evaluation schemes are used.

### 4.1 Few-shot transfer evaluation (primary results)

Implemented in `run_fewshot_transfer_analysis` ([analysis.ipynb](analysis.ipynb) cell 15),
run for all 12 ordered pairs (cell 16). For each source→target pair:

1. The **target** cohort is split once into a **held-out 20% test set** and an 80% pool, with
   `train_test_split(test_size=0.2, random_state=42)`. The 20% test set is **fixed** and used
   to evaluate every model and every few-shot ratio for that pair.
2. From the 80% pool, a **few-shot training subset** is drawn at ratios **5%, 10%, 20%** using
   `StratifiedShuffleSplit(train_size=ratio, random_state=42)`, **stratified on the mortality
   event** so that the rare event class is represented at each ratio.
3. The three models (Baseline, Standard TL, Weighted TL — see [§5](#5-models-and-hyperparameters))
   are trained and evaluated on the fixed 20% test set.

**Metrics:**
- **C-index** via `concordance_index_censored` on the held-out test set — the primary
  discrimination metric (ability to correctly rank patients by predicted time-to-discharge).
- **Cumulative/dynamic time-dependent AUC** via `cumulative_dynamic_auc`, evaluated over a grid
  of 50 time points spanning the test-set LOS range (`np.linspace(min_time, max_time-1e-6, 50)`),
  producing the AUC-versus-day curves.

### 4.2 Cross-validated pooled-cohort evaluation (secondary)

A separate analysis ([analysis.ipynb](analysis.ipynb) cells 4–9) evaluates **every combination
of cohorts** (all 1-, 2-, 3-, and 4-cohort unions) with **5-fold cross-validation**
(`KFold(n_splits=5, shuffle=True, random_state=42)`), reporting internal C-index, dynamic AUC,
and sensitivity, plus **external AUC** when the pooled model is applied to each cohort in turn.
This is what selects and trains the pooled "best-combination" model serialized to
`final_survival_model.pkl` ([§7](#7-pooled-cohort-model-final_survival_modelpkl)). This pipeline
uses the **full** feature set (see [§9](#9-known-discrepancies-and-notes)).

> The compact notebook [transfer_learning_experiment.ipynb](transfer_learning_experiment.ipynb)
> contains an earlier `run_transfer_experiment` that evaluates Direct/Standard/Weighted transfer
> with 5-fold CV on the target and an unrationed fine-tune; the **reported few-shot (5/10/20%)
> results come from `run_fewshot_transfer_analysis` in [analysis.ipynb](analysis.ipynb)**.

---

## 5. Models and hyperparameters

### 5.1 Base learner

All models are **Gradient Boosting Survival Analysis** (`sksurv.ensemble.GradientBoostingSurvivalAnalysis`,
GBSA) — an additive ensemble of regression trees that minimizes the negative partial
log-likelihood of the Cox model for right-censored data.

**Features used by the transfer-learning models.** The few-shot TL models use
`['age', 'sex', 'gcs']` (`to_X` in [analysis.ipynb](analysis.ipynb) cell 15 and
[transfer_learning_experiment.ipynb](transfer_learning_experiment.ipynb) cell 1). The
full categorical feature set (with one-hot encoded `sex, hr_cat, rr_cat, sbp_cat, MOI`) is used
by the **pooled-cohort** pipeline in [§4.2](#42-cross-validated-pooled-cohort-evaluation-secondary)/[§7](#7-pooled-cohort-model-final_survival_modelpkl).
See [§9](#9-known-discrepancies-and-notes) for how this relates to the manuscript text.

**Hyperparameters.** The number of boosting stages (`n_estimators`) is the only hyperparameter
varied, and it is **fixed by design** (not grid-searched): models are pretrained with
**100** estimators and fine-tuned up to **150** via warm-starting (below). All other GBSA
hyperparameters use scikit-survival defaults (learning rate 0.1, max tree depth 3, subsample 1.0,
`loss='coxph'`). The 5:1 weighting in the weighted variant ([§5.3](#53-weighted-transfer-learning-the-exact-implementation))
is likewise a fixed design choice. There was therefore **no automated hyperparameter search**;
the "grid search" in [analysis.ipynb](analysis.ipynb) cells 4–9 is a search over **cohort
combinations**, not over hyperparameters.

### 5.2 The three conditions

For each few-shot subset `(X_fewshot, y_fewshot)` and fixed test set `(X_test, y_test)`:

**(a) Baseline (internal target-only model).**
```python
model_b = GradientBoostingSurvivalAnalysis(n_estimators=100, random_state=42)
model_b.fit(X_fewshot, y_fewshot)
```

**(b) Standard transfer learning (warm-start fine-tuning).** The model is first fit on the
**entire source** cohort with 100 trees; `warm_start=True` retains those trees while
`n_estimators` is raised to 150 and the model is fit again on the **target few-shot** data.
This adds 50 boosting stages trained on target data on top of the 100 source-trained stages:
```python
model_s = GradientBoostingSurvivalAnalysis(n_estimators=100, warm_start=True, random_state=42)
model_s.fit(X_source, y_source)        # pretrain on full source (100 stages)
model_s.set_params(n_estimators=150)   # allow 50 more stages
model_s.fit(X_fewshot, y_fewshot)      # fine-tune on target few-shot
```

**(c) Weighted transfer learning** — identical to (b) but the target fine-tuning step passes
class-balanced **sample weights** (see [§5.3](#53-weighted-transfer-learning-the-exact-implementation)).

### 5.3 Weighted transfer learning: the exact implementation

The weighted variant differs from Standard TL **only** in the final fine-tuning call, which
supplies per-sample weights computed from the mortality event label:

```python
from sklearn.utils import compute_sample_weight

model_w = GradientBoostingSurvivalAnalysis(n_estimators=100, warm_start=True, random_state=42)
model_w.fit(X_source, y_source)         # pretrain on full source
model_w.set_params(n_estimators=150)    # allow 50 more stages
sw = compute_sample_weight(class_weight={1: 5, 0: 1},
                           y=y_fewshot["event"].astype(int))   # deaths weighted 5×
model_w.fit(X_fewshot, y_fewshot, sample_weight=sw)            # weighted fine-tuning
```

Precise behavior:
- Weights are derived from the **binary mortality event** (`event` = 1 death, 0 censored), via
  `compute_sample_weight(class_weight={1: 5, 0: 1}, ...)`. This assigns weight **5 to every
  deceased (event = 1) patient and weight 1 to every censored/surviving patient** — a fixed
  5:1 up-weighting of the minority (death) class to counter class imbalance, which varies
  widely across cohorts (mortality 7%–25%; [§1](#1-cohorts-and-overall-design)).
- The weights enter only the **target fine-tuning** step; source pretraining is unweighted.
- Weights scale each sample's contribution to the gradient-boosting loss at every boosting
  stage of the fine-tuning phase.

This is the only mechanistic difference between the Standard and Weighted conditions, so any
performance gap between them is attributable to this class-balanced reweighting.

---

## 6. Repetitions and statistical testing

For each (pair, ratio), the three models are evaluated inside a **5-iteration loop**
(`for _ in range(5)` in [analysis.ipynb](analysis.ipynb) cell 15), and the C-index is reported
as **mean ± SD** over those iterations (`print_cindex_summary`).

**Significance testing.** Within each (pair, ratio), the per-iteration C-index vectors are
compared with the **Wilcoxon signed-rank test** (`scipy.stats.wilcoxon`) for the three
contrasts: Baseline vs Standard TL, Baseline vs Weighted TL, and Standard TL vs Weighted TL
(`compare_models`, cell 15), with significance at p < 0.05.

> **Important caveat on the repetition loop.** As written, the 5 iterations re-fit the models
> with the **same fixed `random_state` and the same few-shot subset**, so GBSA (which is
> deterministic given fixed data and seed) returns **identical predictions each iteration** —
> the reported SD is ≈0 and the Wilcoxon test operates on tied vectors. To obtain genuine
> variability across repetitions (and meaningful SDs / p-values), vary the seed per iteration
> for **both** the few-shot resampling and the estimator, e.g.:
> ```python
> for rep in range(5):
>     seed = 42 + rep
>     sss = StratifiedShuffleSplit(n_splits=1, train_size=ratio, random_state=seed)
>     ...
>     model_b = GradientBoostingSurvivalAnalysis(n_estimators=100, random_state=seed)
> ```
> This is flagged so the manuscript's "number of repetitions" claim can be made accurate; see
> [§9](#9-known-discrepancies-and-notes).

A separate one-way **ANOVA** compares mean LOS across the four cohorts
([analysis.ipynb](analysis.ipynb) cell 11, `scipy.stats.f_oneway`).

---

## 7. Pooled-cohort model (`final_survival_model.pkl`)

[analysis.ipynb](analysis.ipynb) cells 4–9 build a survival pipeline on **unions of cohorts**:

```python
numerical_features  = ['age', 'gcs']
categorical_features = ['sex', 'hr_cat', 'rr_cat', 'sbp_cat', 'MOI']
ColumnTransformer([('num', 'passthrough', numerical_features),
                   ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
GradientBoostingSurvivalAnalysis(n_estimators=150, random_state=0)
```

Each cohort-combination is scored by 5-fold CV (internal C-index, dynamic AUC, sensitivity) and
external AUC on every cohort; the best-scoring combination is refit on all its data and
serialized with `joblib` to `final_survival_model.pkl` (cell 6). This artifact is the
**pooled, full-feature** model — it is distinct from the few-shot transfer-learning models in
[§5](#5-models-and-hyperparameters), which use only `age, sex, gcs`.

---

## 8. Snippet: report pre-imputation missingness

Run this on the **raw** datasets (before [preprocessing.ipynb](preprocessing.ipynb) imputes
them) to produce a per-feature missingness table for the manuscript:

```python
import pandas as pd

raw = {  # map each cohort to its raw file and the columns of interest
    "India":      ("trauma_india_brain_injury.csv", ['age','sex','sbp_1','hr_1','rr_1','gcs_t_1']),
    "Jordan":     ("traumatic_brain_injury.csv",    ['Gender','age of diagnosis','ER-HR','ER-RR','ER-systolic BP','GCS in ER']),
    "Florida":    ("updated_dataset (1).csv",       ['Age','Gender','SBP','HR','RR','GCS']),
    "California":  ("traumatic_brain_injury_usa.csv",['Age','Sex','SBP','HR','RR','Trauma GCS']),
}

rows = []
for name, (path, cols) in raw.items():
    df = pd.read_csv(path)
    for c in cols:
        n_missing = df[c].isna().sum()
        rows.append({"Cohort": name, "Feature": c,
                     "N": len(df), "Missing (n)": int(n_missing),
                     "Missing (%)": round(100 * n_missing / len(df), 1)})
report = pd.DataFrame(rows)
print(report.to_string(index=False))
```

---

## 9. Known discrepancies and notes

These are documented so the code, the manuscript, and the reviewer response stay consistent.
Items 1 and 2 have been **resolved** in
[transfer_learning_experiment_fixed.ipynb](transfer_learning_experiment_fixed.ipynb)
(see [§10](#10-corrected-experiment-and-results)).

1. **Feature set used by the transfer-learning models.** *(Resolved in the fixed notebook.)*
   The original few-shot TL models ([§5](#5-models-and-hyperparameters)) used only
   **`age, sex, gcs`**, whereas the manuscript Methods describe a base feature set that also
   includes physiologic (vital-sign) categories and MOI. The corrected notebook uses the
   **full** feature set (age, GCS numeric + sex, hr_cat, rr_cat, sbp_cat, MOI one-hot), matching
   the manuscript.

2. **Repetition loop produced no variance.** *(Resolved in the fixed notebook.)* In the original
   ([§6](#6-repetitions-and-statistical-testing)), fixed seeds + a fixed few-shot subset made the
   5 iterations identical (SD ≈ 0). The corrected notebook seeds the target resample **and** the
   estimator per iteration (`seed = 42 + rep`, 20 repetitions), giving real variability and valid
   Wilcoxon tests.

3. **Seed inconsistency.** The few-shot analysis uses `random_state=42` throughout; the compact
   `run_transfer_experiment` and the pooled pipeline use `random_state=0` for the estimator.
   Results are reproducible within each notebook but the seed should be stated per analysis.

4. **Imputation before splitting** includes the outcome (LOS) among imputed features
   ([§3](#3-missing-data-extent-pattern-and-handling)); a leakage-free variant would fit the
   imputer within training folds only.

---

## 10. Corrected experiment and results

[transfer_learning_experiment_fixed.ipynb](transfer_learning_experiment_fixed.ipynb) re-runs the
few-shot transfer-learning experiment with the two corrections above applied: the **full
harmonized feature set** (age, GCS numeric + sex, hr_cat, rr_cat, sbp_cat, MOI one-hot, encoded
with a single encoder shared across cohorts → a 20-column feature space) and **20 genuine
repetitions** (target resample and estimator seeded per iteration). Each source cohort is
pretrained once and reused via deep-copy (a speed optimization only; it does not change the
model). The run below includes **all four cohorts (12 ordered pairs)**.

**C-index (mean ± SD over 20 repetitions), 5% target fine-tuning.** Significance vs. baseline
(Wilcoxon signed-rank): \*\*\* p<0.001, \*\* p<0.01, \* p<0.05, ns = not significant.

| Source → Target | Baseline | Standard TL | Weighted TL | TL effect |
|---|---|---|---|---|
| India → Jordan | 0.601 ± 0.21 | 0.806 ± 0.14 | 0.807 ± 0.15 | **↑ \*\*\*** |
| India → California | 0.622 ± 0.17 | 0.821 ± 0.08 | 0.827 ± 0.07 | **↑ \*\*\*** |
| California → Jordan | 0.601 ± 0.21 | 0.844 ± 0.08 | 0.845 ± 0.08 | **↑ \*\*\*** |
| Jordan → California | 0.622 ± 0.17 | 0.670 ± 0.14 | 0.672 ± 0.14 | ↑ ns |
| Florida → Jordan | 0.601 ± 0.21 | 0.629 ± 0.23 | 0.656 ± 0.22 | ↑ ns |
| Florida → California | 0.622 ± 0.17 | 0.598 ± 0.19 | 0.616 ± 0.19 | ns |
| California → India | 0.741 ± 0.02 | 0.736 ± 0.01 | 0.740 ± 0.02 | ns |
| Jordan → India | 0.741 ± 0.02 | 0.719 ± 0.02 | 0.724 ± 0.02 | **↓ \*\*\*** |
| Florida → India | 0.741 ± 0.02 | 0.722 ± 0.02 | 0.737 ± 0.02 | **↓ \*\*\*** |
| California → Florida | 0.565 ± 0.10 | 0.526 ± 0.12 | 0.514 ± 0.13 | ↓ ns |
| Jordan → Florida | 0.565 ± 0.10 | 0.522 ± 0.13 | 0.518 ± 0.14 | ↓ ns |
| India → Florida | 0.565 ± 0.10 | 0.504 ± 0.12 | 0.488 ± 0.12 | **↓ \*** |

(Full results, incl. 10% and 20% ratios, in `fixed_results_summary.csv`; per-repetition values
in `fixed_results_raw.csv`; Wilcoxon p-values in `fixed_results_wilcoxon.csv`; figures in
`fixed_cindex_by_source.png` and `results_auc/`.)

**Interpretation.** The corrected analysis sharpens — and partly revises — the original narrative:

- **TL helps significantly when the target is a small cohort distinct from the source**
  (→ Jordan, → California): India→Jordan 0.60→0.81, California→Jordan 0.60→0.84,
  India→California 0.62→0.82, all p<0.001 at 5% fine-tuning. *This is the paper's core finding
  and it holds strongly.*
- **TL does not help — and can significantly hurt — when the target already has abundant data**
  (→ India: own baseline ≈0.74, TL slightly lower) **or is too distinct** (→ Florida, the
  severe-ICU, long-LOS cohort: TL falls below baseline). Source knowledge either isn't needed or
  actively misleads. This is consistent with the domain-shift literature already cited in the
  Discussion (Bernico; Pfisterer).
- **Weighted ≈ Standard TL.** Only 1 of 12 pairs shows a significant Standard-vs-Weighted
  difference at 5%. The claim that *weighted TL achieved the largest gains* is **not supported**
  by the corrected analysis.

> The original draft's numbers (e.g. India→Jordan 0.16→0.69) reflected the 3-feature models and
> the no-variance repetition loop; the sub-0.5 baselines were degenerate artifacts of training
> on a handful of rows with three features. The corrected baselines behave sensibly (~0.55–0.74).
> **Manuscript figures (Fig 4/5) and all quoted C-index/AUC numbers must be refreshed from these
> outputs, and three secondary claims reworded** — see the change list in the project notes.
