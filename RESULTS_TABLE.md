# Results Summary — Transfer Learning Performance Across All Cohort Pairs

Discrimination performance (Harrell's **concordance index**, mean ± SD over **20 repetitions**) for the three training conditions, for every ordered source → target cohort pair and each few-shot fine-tuning ratio (5%, 10%, 20% of target data).

Models: Gradient Boosting Survival Analysis on the harmonized feature set (age, GCS, sex, heart-rate / respiratory-rate / blood-pressure categories, and mechanism of injury). **Baseline** = target-only model; **Standard TL** = source-pretrained model fine-tuned on the target few-shot sample; **Weighted TL** = as Standard, with class-balanced (5:1) sample weighting during fine-tuning.

Asterisks mark a statistically significant difference **versus the baseline** (Wilcoxon signed-rank, paired across repetitions): `*` p<0.05, `**` p<0.01, `***` p<0.001. Values higher than baseline indicate transfer learning improved discrimination; lower values indicate it degraded it.

Generated from `fixed_results_summary.csv` / `fixed_results_wilcoxon.csv` (see [transfer_learning_experiment_fixed.ipynb](transfer_learning_experiment_fixed.ipynb)).

## All pairs × all ratios

| Source | Target | Fine-tune | Baseline | Standard TL | Weighted TL |
|---|---|---|---|---|---|
| **India** | Jordan | 5% | 0.601 ± 0.208 | 0.806 ± 0.139*** | 0.807 ± 0.152*** |
|  |  | 10% | 0.663 ± 0.241 | 0.793 ± 0.153 | 0.793 ± 0.153* |
|  |  | 20% | 0.746 ± 0.186 | 0.806 ± 0.140 | 0.825 ± 0.145 |
| | | | | | |
| **India** | California | 5% | 0.622 ± 0.173 | 0.821 ± 0.081*** | 0.827 ± 0.072*** |
|  |  | 10% | 0.611 ± 0.135 | 0.820 ± 0.056*** | 0.811 ± 0.056*** |
|  |  | 20% | 0.613 ± 0.158 | 0.826 ± 0.056*** | 0.816 ± 0.062*** |
| | | | | | |
| **California** | Jordan | 5% | 0.601 ± 0.208 | 0.844 ± 0.080*** | 0.845 ± 0.081*** |
|  |  | 10% | 0.663 ± 0.241 | 0.849 ± 0.110*** | 0.846 ± 0.082** |
|  |  | 20% | 0.746 ± 0.186 | 0.834 ± 0.110* | 0.832 ± 0.109* |
| | | | | | |
| **Jordan** | California | 5% | 0.622 ± 0.173 | 0.670 ± 0.136 | 0.672 ± 0.144* |
|  |  | 10% | 0.611 ± 0.135 | 0.692 ± 0.120* | 0.691 ± 0.121* |
|  |  | 20% | 0.613 ± 0.158 | 0.665 ± 0.119 | 0.705 ± 0.110** |
| | | | | | |
| **Florida** | Jordan | 5% | 0.601 ± 0.208 | 0.629 ± 0.227 | 0.656 ± 0.215 |
|  |  | 10% | 0.663 ± 0.241 | 0.689 ± 0.219 | 0.670 ± 0.213 |
|  |  | 20% | 0.746 ± 0.186 | 0.722 ± 0.206 | 0.706 ± 0.176 |
| | | | | | |
| **Florida** | California | 5% | 0.622 ± 0.173 | 0.598 ± 0.189 | 0.616 ± 0.185 |
|  |  | 10% | 0.611 ± 0.135 | 0.608 ± 0.151 | 0.629 ± 0.157 |
|  |  | 20% | 0.613 ± 0.158 | 0.584 ± 0.144 | 0.616 ± 0.150 |
| | | | | | |
| **California** | India | 5% | 0.741 ± 0.019 | 0.736 ± 0.014 | 0.740 ± 0.017 |
|  |  | 10% | 0.759 ± 0.013 | 0.750 ± 0.011*** | 0.750 ± 0.013*** |
|  |  | 20% | 0.770 ± 0.014 | 0.756 ± 0.012*** | 0.756 ± 0.014*** |
| | | | | | |
| **Jordan** | India | 5% | 0.741 ± 0.019 | 0.719 ± 0.018*** | 0.724 ± 0.021*** |
|  |  | 10% | 0.759 ± 0.013 | 0.732 ± 0.015*** | 0.737 ± 0.012*** |
|  |  | 20% | 0.770 ± 0.014 | 0.743 ± 0.016*** | 0.743 ± 0.014*** |
| | | | | | |
| **Florida** | India | 5% | 0.741 ± 0.019 | 0.722 ± 0.021*** | 0.737 ± 0.022 |
|  |  | 10% | 0.759 ± 0.013 | 0.738 ± 0.018*** | 0.750 ± 0.015** |
|  |  | 20% | 0.770 ± 0.014 | 0.746 ± 0.017*** | 0.758 ± 0.015*** |
| | | | | | |
| **California** | Florida | 5% | 0.565 ± 0.100 | 0.526 ± 0.124 | 0.514 ± 0.129 |
|  |  | 10% | 0.616 ± 0.094 | 0.549 ± 0.116* | 0.545 ± 0.117* |
|  |  | 20% | 0.554 ± 0.121 | 0.538 ± 0.104 | 0.558 ± 0.118 |
| | | | | | |
| **Jordan** | Florida | 5% | 0.565 ± 0.100 | 0.522 ± 0.134 | 0.518 ± 0.139 |
|  |  | 10% | 0.616 ± 0.094 | 0.562 ± 0.153 | 0.556 ± 0.140 |
|  |  | 20% | 0.554 ± 0.121 | 0.526 ± 0.108 | 0.531 ± 0.113 |
| | | | | | |
| **India** | Florida | 5% | 0.565 ± 0.100 | 0.504 ± 0.121* | 0.488 ± 0.117** |
|  |  | 10% | 0.616 ± 0.094 | 0.533 ± 0.125* | 0.509 ± 0.112** |
|  |  | 20% | 0.554 ± 0.121 | 0.502 ± 0.120 | 0.514 ± 0.136 |

## Headline view — 5% fine-tuning

| Source → Target | Baseline | Standard TL | Weighted TL | TL effect |
|---|---|---|---|---|
| India → Jordan | 0.601 | 0.806 ± 0.139*** | 0.807 ± 0.152*** | **improves** |
| India → California | 0.622 | 0.821 ± 0.081*** | 0.827 ± 0.072*** | **improves** |
| California → Jordan | 0.601 | 0.844 ± 0.080*** | 0.845 ± 0.081*** | **improves** |
| Jordan → California | 0.622 | 0.670 ± 0.136 | 0.672 ± 0.144* | no sig. change |
| Florida → Jordan | 0.601 | 0.629 ± 0.227 | 0.656 ± 0.215 | no sig. change |
| Florida → California | 0.622 | 0.598 ± 0.189 | 0.616 ± 0.185 | no sig. change |
| California → India | 0.741 | 0.736 ± 0.014 | 0.740 ± 0.017 | no sig. change |
| Jordan → India | 0.741 | 0.719 ± 0.018*** | 0.724 ± 0.021*** | **degrades** |
| Florida → India | 0.741 | 0.722 ± 0.021*** | 0.737 ± 0.022 | **degrades** |
| California → Florida | 0.565 | 0.526 ± 0.124 | 0.514 ± 0.129 | no sig. change |
| Jordan → Florida | 0.565 | 0.522 ± 0.134 | 0.518 ± 0.139 | no sig. change |
| India → Florida | 0.565 | 0.504 ± 0.121* | 0.488 ± 0.117** | **degrades** |

**Summary.** Transfer learning produced large, statistically significant improvements when the target was a small cohort distinct from the source (→ Jordan, → California). It conferred no significant benefit — and in some cases significantly degraded performance — when the target already had abundant data (→ India) or was an extreme outlier (→ Florida; severe-ICU, long-stay). Standard and weighted transfer learning performed comparably (a significant Standard-vs-Weighted difference appeared in only 1 of 12 pairs at 5%).
