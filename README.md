# Transfer Learning for Length of Stay Prediction in Traumatic Brain Injury

## Overview
This project investigates the application of few-shot transfer learning to improve length of stay (LOS) prediction models across diverse traumatic brain injury (TBI) cohorts from different international settings.

## Problem Statement
Length of stay prediction for TBI patients is complex due to:
- Variation in clinical severity
- System-level pressures
- Limited generalization of single-dataset models
- Resource constraints in diverse healthcare settings

## Dataset
The study analyzes four distinct TBI cohorts:
1. **India**: Multi-center dataset
2. **Jordan**: Pediatric patient cohort
3. **Florida (USA)**: Adult patients with severe blunt TBI in intensive care
4. **California (USA)**: Adult patients with measured blood alcohol concentration

## Methodology
Our approach involves:
- Gradient Boosting Survival Analysis models
- Three training conditions:
  - Baseline
  - Standard Transfer Learning
  - Weighted Transfer Learning
- Few-shot fine-tuning using:
  - 5% of target cohort data
  - 10% of target cohort data
  - 20% of target cohort data

## Evaluation Metrics
- Concordance index (C-index)
- Time-dependent AUC

## Repository Structure
```
├── analysis.ipynb
├── preprocessing.ipynb
├── transfer_learning_experiment.ipynb
├── california_clean.csv
├── florida_clean.csv `# Request for access from corresponding author`
├── full_clean.csv `# Request for access from corresponding author`
├── india_clean.csv
└── jordan_clean.csv
```

## License
See [LICENSE](LICENSE) file for details.

## Usage Guide

### Prerequisites
- Python 3.7+
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - scikit-survival
  - matplotlib

### Running Transfer Learning Experiments
The main experiment code is in `transfer_learning_experiment.ipynb`. This notebook contains:

1. **Data Loading and Preprocessing**
   ```python
   # Load datasets
   india = pd.read_csv('india_clean.csv')
   jordan = pd.read_csv('jordan_clean.csv')
   florida = pd.read_csv('florida_clean.csv')
   california = pd.read_csv('california_clean.csv')
   ```

2. **Transfer Learning Functions**
   - `run_transfer_experiment()`: Performs transfer learning between source and target datasets
   - `run_fewshot_dynamic_auc()`: Evaluates few-shot learning performance using dynamic AUC

3. **Model Training Options**
   - Direct Transfer
   - Standard Fine-tuning
   - Weighted Fine-tuning

4. **Performance Visualization**
   - Dynamic AUC curves
   - Concordance index comparisons
   - Few-shot learning performance analysis

### Example Usage
```python
# Run transfer learning experiment from India to Jordan
result = run_transfer_experiment("India", "Jordan", plot_auc=True)

# Evaluate few-shot learning performance
run_fewshot_dynamic_auc(india, jordan, "India", "Jordan")
```

## Open Access Datasets
1.	`Jordan`: Raffee L, Al-Mistarehi AH, Alawneh K, et al. BIG Score in Pediatric Trauma Patients Dataset. Published online February 10, 2024. doi:10.5281/zenodo.10644773
2.	`California`: Brigode W. Data for: Alcohol in Traumatic Brain Injury: Toxic or Therapeutic? 2021;1. doi:10.17632/w5mgnjy3cn.1
3.	`India`: collaborators TITCO (TITCO). The original anonymized TITCO cohort. Published online September 10, 2020. Accessed September 1, 2025. https://zenodo.org/records/7832819

