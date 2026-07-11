# Global Prediction of Hospital Stay in Traumatic Brain Injury Using Transfer Learning

This repository contains the data-processing and modeling code for the study
**"Global Prediction of Hospital Stay in Traumatic Brain Injury Using Transfer Learning: A Multinational, Multicenter Study."**

We frame hospital **length of stay (LOS)** after traumatic brain injury (TBI) as a
right-censored **time-to-event (survival)** problem and ask whether **few-shot transfer
learning (TL)** lets a survival model trained on one cohort be adapted, with only a small
amount of local data, to an epidemiologically distinct cohort.

> **For the full, reproducibility-oriented description of the methods** — missing-data
> handling, hyperparameters, validation strategy, number of repetitions, and the exact
> implementation of the weighted transfer-learning approach — see **[METHODS.md](METHODS.md)**.

---

## Study at a glance

- **Task:** predict LOS as a survival outcome, with in-hospital mortality as the censoring indicator.
- **Model:** Gradient Boosting Survival Analysis (GBSA, `scikit-survival`).
- **Three training conditions per source→target pair:**
  1. **Baseline** — GBSA trained only on the small target subset (internal model).
  2. **Standard TL** — GBSA pretrained on the full source cohort, then fine-tuned on the target subset.
  3. **Weighted TL** — same as Standard TL, but target fine-tuning uses class-balanced sample weights.
- **Few-shot regimes:** 5%, 10%, and 20% of the target cohort used for fine-tuning.
- **Metrics:** concordance index (C-index) and cumulative/dynamic time-dependent AUC.
- **Design:** all 12 ordered source→target pairs across the four cohorts.

## Cohorts

| Cohort | Population | Source | In repo |
|---|---|---|---|
| **India** | Multi-center trauma registry (TITCO), predominantly young adults | Open access | `india_clean.csv` |
| **Jordan** | Pediatric trauma patients | Open access | `jordan_clean.csv` |
| **California (USA)** | Adults with measured blood alcohol concentration | Open access | `california_clean.csv` |
| **Florida (USA)** | Adults with severe blunt TBI admitted to intensive care | Single-institution | **Available on request** from the corresponding author |

The cleaned CSVs in this repository are the **post-preprocessing** files produced by
[preprocessing.ipynb](preprocessing.ipynb) (see [METHODS.md](METHODS.md) for the full pipeline).
The Florida file is not redistributed here; the modeling notebooks expect a `florida_clean.csv`
in the same format.

### Open-access dataset citations
1. **Jordan** — Raffee L, Al-Mistarehi AH, Alawneh K, et al. *BIG Score in Pediatric Trauma Patients Dataset.* 2024. doi:10.5281/zenodo.10644773
2. **California** — Brigode W. *Data for: Alcohol in Traumatic Brain Injury: Toxic or Therapeutic?* 2021. doi:10.17632/w5mgnjy3cn.1
3. **India** — TITCO collaborators. *The original anonymized TITCO cohort.* 2020. https://zenodo.org/records/7832819

---

## Repository structure

```
TBI-transfer-learning/
├── preprocessing.ipynb                # Raw → cleaned cohorts (harmonization, imputation)
├── analysis.ipynb                     # Main analysis: few-shot TL, stats, figures, tables
├── transfer_learning_experiment.ipynb # Original compact TL experiment (age/sex/GCS only)
├── transfer_learning_experiment_fixed.ipynb  # ★ Corrected TL experiment (full feature set)
├── india_clean.csv                    # Cleaned India cohort (N = 7,978)
├── jordan_clean.csv                   # Cleaned Jordan cohort (N = 112)
├── california_clean.csv               # Cleaned California cohort (N = 583)
│   # florida_clean.csv                # NOT included — request from corresponding author
├── final_survival_model.pkl          # Serialized pooled-cohort GBSA pipeline (see METHODS.md §7)
├── moi_plot.png                       # Mechanism-of-injury figure
├── METHODS.md                         # In-depth, reproducible methods
├── LICENSE
└── README.md
```

### What each notebook does
- **[preprocessing.ipynb](preprocessing.ipynb)** — loads the four raw datasets, harmonizes
  feature names and units, maps free-text mechanism of injury (MOI) to a common taxonomy,
  derives age-specific vital-sign categories, imputes missing values, and writes the
  `*_clean.csv` files.
- **[analysis.ipynb](analysis.ipynb)** — the primary results notebook. Contains the
  **few-shot transfer-learning experiment** (`run_fewshot_transfer_analysis`) that produces
  the C-index comparisons (Baseline vs. Standard TL vs. Weighted TL at 5/10/20%) with
  Wilcoxon significance testing and dynamic-AUC curves, plus the cohort-description
  tables/figures (Kaplan–Meier LOS curves, MOI breakdowns, vital-sign and GCS distributions,
  LOS ANOVA) and the pooled-cohort model exported to `final_survival_model.pkl`.
- **[transfer_learning_experiment.ipynb](transfer_learning_experiment.ipynb)** — a compact,
  standalone version of the transfer experiment (`run_transfer_experiment`,
  `run_fewshot_dynamic_auc`) useful for re-running a single source→target pair.
- **[transfer_learning_experiment_fixed.ipynb](transfer_learning_experiment_fixed.ipynb)** — ★
  the **corrected** few-shot TL experiment. It uses the **full harmonized feature set**
  (age, GCS, sex, vital-sign categories, and MOI — one-hot encoded with a single shared
  encoder so source and target share an identical feature space) and uses **genuine
  per-iteration repetitions** so the reported mean ± SD and Wilcoxon tests are valid. It
  writes `fixed_results_raw.csv`, `fixed_results_summary.csv`, `fixed_results_wilcoxon.csv`,
  the C-index figures (`fig4_cindex_<source>.png`, one 3D panel per source cohort), and the
  dynamic-AUC figures under `results_auc/` (`auc_<src>_to_<tgt>_<pct>.png` — 36 figures, one per
  source→target pair per few-shot level, each comparing Baseline vs Standard vs Weighted TL).
  See [METHODS.md §9](METHODS.md) for what changed and why.

---

## Quickstart

### Requirements
- Python ≥ 3.9 (developed on 3.13)
- `pandas`, `numpy`, `scikit-learn`, `scikit-survival`, `scipy`, `matplotlib`, `seaborn`,
  `missingno`, `plotly`, `joblib`

```bash
pip install pandas numpy scikit-learn scikit-survival scipy matplotlib seaborn missingno plotly joblib
```

### Reproduce the pipeline
1. **(Optional) Re-create the cleaned cohorts.** Place the raw source files alongside the
   notebook and run [preprocessing.ipynb](preprocessing.ipynb). This regenerates the
   `*_clean.csv` files. (The cleaned files are already provided, so this step is only needed
   if you want to rebuild from raw data or add the Florida cohort.)
2. **Run the experiments.** Open [analysis.ipynb](analysis.ipynb) and run all cells to
   reproduce the few-shot TL results, statistics, and figures. To run a single pair only:

   ```python
   # from transfer_learning_experiment.ipynb
   result = run_transfer_experiment("India", "Jordan", plot_auc=True)
   run_fewshot_dynamic_auc(india, jordan, "India", "Jordan")
   ```

All data splits and model seeds are fixed (`random_state=42`); see [METHODS.md §4–§5](METHODS.md)
for the exact validation and repetition scheme.

---

## Data availability

The California, India, and Jordan datasets are open-access and cited above. The Florida
dataset is available on request from the corresponding author. Code is available at
<https://github.com/shrinitbabel/TBI-transfer-learning>.

## License

See [LICENSE](LICENSE).
