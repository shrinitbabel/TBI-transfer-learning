# TBI Length-of-Stay Estimator (side project)

A browser-based calculator that estimates how long a patient with traumatic brain injury will
stay in hospital. It is **not** part of the published transfer-learning study and is for
research use only.

- **Model:** one Gradient Boosting Survival Analysis model (150 trees) trained on the pooled
  cohorts built by `revision2/01_build_cohorts.py`: India (TITCO), Jordan, California and Florida.
- **Outcome:** time to live discharge, with in-hospital death as a competing event
  (Fine–Gray subdistribution coding, as in the paper).
- **Inputs:** age, sex, GCS, ICU admission, mechanism of injury, heart rate, respiratory rate,
  systolic blood pressure. Vital signs are converted to age-specific categories, as in the paper.
- **Outputs:** probability of having been discharged alive by each day, the predicted median
  stay among patients discharged alive, and the probability of not being discharged alive
  (approximately the in-hospital death risk).
- **Validation:** 5-fold cross-validation, reported per cohort in `validation.json` and on the page.

The page is a single static file (`index.html`) with the model embedded. Predictions are
computed in the browser, and nothing the user enters is sent anywhere.

## Rebuild

```bash
python revision2/01_build_cohorts.py                         # cohorts (incl. ICU flag)
python los_calculator/train_pooled_model.py --include-florida # model.json, validation.json
python los_calculator/build_page.py                          # index.html
```

`train_pooled_model.py` checks that its pure-Python replica of the browser computation matches
scikit-survival's predictions before exporting the model.

## Limitations

- **Cohort coverage:** trained on four urban cohorts. The model has no hospital or country input,
  so it predicts an average across these systems.
- **Florida performance:** discrimination for the Florida ICU cohort is near chance. Long
  intensive-care stays are predicted poorly.
- **ICU input:** "Admitted to intensive care" means any ICU stay during the admission, which is not
  always known at presentation.
