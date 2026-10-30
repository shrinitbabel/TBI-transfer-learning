# Revision 2 — corrected analysis

Code for the second round of revisions. The original notebooks in the repository root are
left unchanged; everything here is rebuilt from the raw source files.

## What changed (and why)

| Issue (Reviewer #1) | Previous pipeline | Revision 2 |
|---|---|---|
| Cohort eligibility | Pre-filtered CSVs, criteria undocumented | Rebuilt from raw files: India = TITCO patients with ≥1 ICD-10 S06 code; Jordan = "any brain injuries" = Yes; California = all (TBI dataset); Florida = institutional ICU blunt-TBI extract. Flow counts in `outputs/cohort_flow.csv` |
| Survival endpoint | Model event = in-hospital **death** (discharge censored) | Event = **live discharge**; death = competing event (Fine–Gray subdistribution coding). Cause-specific coding as sensitivity (`--coding cs`) |
| India LOS | TITCO date-shifted years (>2262) overflowed pandas → NaN → **imputed** (887 patients) | LOS computed from date parts; missing/negative LOS excluded, never imputed |
| Leakage | IterativeImputer fit on whole cohort **including LOS** | Imputer fit on predictors only, on training rows only, inside each split |
| Implausible values | Kept (e.g. RR = 0, GCS = 0) | Set to missing; counts in `outputs/missingness.csv` |
| Encoding | Encoder fit on all cohorts | Fixed, pre-specified category lists |
| Evaluation | Dynamic AUC on the whole target cohort | All metrics on untouched 20% test sets, 50 repeated splits, 95% intervals, FDR-corrected Wilcoxon |
| Capacity | Baseline 100 trees vs TL 150 | Baseline 150 trees |
| Metrics | C-index, dynamic AUC | + integrated Brier score (and scaled vs null), calibration, MAE (days) |

## Run

```bash
python 01_build_cohorts.py                    # raw_data/ (+ ../florida_clean.csv) -> outputs/cohorts, flow, missingness
python 02_transfer_experiment.py --reps 50    # primary analysis (competing risks)
python 02_transfer_experiment.py --reps 50 --coding cs   # sensitivity (cause-specific)
python 03_figures_tables.py                   # Figures 2, 4, 5, supplement AUC/calibration, result tables
python 04_descriptives_flow.py                # Supplemental Table 2, flow diagram, GCS figure
```

`raw_data/` holds the open-access source files (TITCO: zenodo.org/records/7832819; Jordan:
doi:10.5281/zenodo.10644773; California: doi:10.17632/w5mgnjy3cn.1). The Florida cohort is
restricted and is not redistributed (`outputs/cohorts/florida_r2.csv` is git-ignored).
Runtime for both codings is about 15 minutes on a laptop.
