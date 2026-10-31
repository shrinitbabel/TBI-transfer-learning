# TBI Length-of-Stay Estimator (side project)

A browser-based calculator that estimates, for a patient with traumatic brain injury, the risk of
dying in hospital and the likely length of stay if discharged alive. It is **not** part of the
published transfer-learning study and is for research use only.

## Model

- **Data:** pooled India (TITCO), Jordan and California cohorts built by
  `revision2/01_build_cohorts.py` (8,685 patients). Florida was excluded because neither the
  mortality nor the length-of-stay predictions were better than chance for that cohort.
- **Inputs:** age, sex, GCS, ICU admission, mechanism of injury, heart rate, respiratory rate and
  systolic blood pressure. Vital signs become age-specific categories, as in the paper.
- **Death risk:** gradient-boosted classifier.
- **Length of stay (among patients discharged alive):** gradient-boosted quantile trees on log
  length of stay, giving the median and an 80% range. A split-conformal correction keeps the
  range's coverage at its stated 80%.

## How the model was chosen

`benchmark_methods.py` compares five approaches on the same 5-fold cross-validation
(`benchmark_results.csv`):
- survival boosting with competing risks (the approach in the paper)
- boosted classifier plus boosted quantile trees
- the same with conformal calibration
- NGBoost
- a DeepHit neural network, implemented in PyTorch

All five discriminate mortality equally (AUROC ≈ 0.85). The conformal boosted model had the
smallest length-of-stay error (4.9 days, against 5.4 days for predicting the median for everyone).
Its 80% range contained 80% of stays.

## Files

| File | Purpose |
|---|---|
| `train_pooled_model.py` | shared preprocessing and the survival-boosting baseline |
| `compare_methods.py`, `benchmark_methods.py` | method comparisons (5-fold CV) |
| `train_calculator.py` | final model; writes `model.json` and `validation.json`; checks the browser replica against scikit-learn |
| `template.html`, `build_page.py` | build `index.html` (standalone) and `page.html` (body only) |

## Rebuild

```bash
python revision2/01_build_cohorts.py
python los_calculator/benchmark_methods.py      # optional, about 15 min
python los_calculator/train_calculator.py
python los_calculator/build_page.py
```

Predictions run entirely in the browser, and nothing the user enters is sent anywhere.
