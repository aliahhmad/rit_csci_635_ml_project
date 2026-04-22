# Predicting Default Risk of Lending Club Loans

### CSCI-635 | Machine Learning

---

## Project Description

The objective of this project is to predict whether a loan will default or not using the Lending Club Loan dataset. Loan default prediction is a binary classification problem. We aim to compare the effectiveness of traditional and modern machine learning techniques for estimating borrower risk.

> Replicating (somewhat): [Predicting Default Risk of Lending Club Loans — CS229](https://cs229.stanford.edu/proj2015/199_report.pdf)

---

## Dataset

- **`accepted_2007_to_2018Q4.csv`** — 2,260,701 rows × 151 columns of accepted loans from 2007–2018.

> Source: [Kaggle — Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

---

## Phase 1 — Data Preprocessing

The pipeline prepares the raw data for binary classification. The key steps are:

- **Label construction** — Binary target from `loan_status`: 1 for non-default (Fully Paid, Current), 0 for default (Charged Off, Late, etc.)
- **Leakage removal** — Drop post-origination columns (payment history, hardship flags, settlement details) that would not be available at prediction time
- **Column cleaning** — Remove admin/ID columns, columns with >10% missing values, and high-cardinality text columns
- **Feature engineering** — Parse string fields (`term`, `int_rate`, `emp_length`, dates) into numeric types; apply TF-IDF on loan descriptions if available
- **Feature ablation** — Drop features identified in the paper as hurting model specificity
- **Encoding** — One-hot encode low-cardinality categorical columns
- **Train/test split** — 70/30 time-ordered split (no shuffle) to prevent future-data leakage
- **Scaling** — MinMaxScaler to [-1, 1], fit on training data only, for use with Logistic Regression and the Neural Network

### Phase 1 Validation

Validation in preprocessing focused on preventing leakage. The train/test split was time-ordered rather than shuffled, so later loans were held out as the test period. The scaler was fit only on the training data and then applied to the test data, which prevents information from the test set from influencing feature scaling.

---

## Output Files (Phase 1)

| File                                       | Description                                               |
| ------------------------------------------ | --------------------------------------------------------- |
| `X_train.csv` / `X_test.csv`               | Unscaled features — for tree-based models and Naive Bayes |
| `X_train_scaled.csv` / `X_test_scaled.csv` | [-1, 1] scaled features — for Logistic Regression and Neural Network |
| `y_train.csv` / `y_test.csv`               | Binary labels (1 = non-default, 0 = default)              |

---

## Phase 2 — Model Training and Results

All three models were evaluated on the same preprocessed test split. Reported metrics below use the latest saved run files in `results/` dated `2026-04-22`.

### 1) Logistic Regression

**Setup (short):** trained on scaled features (`X_train_scaled.csv`) with hyperparameter tuning for regularization strength, solver choice, and class weighting.

**Validation:** Logistic Regression uses a manual grid search with stratified holdout validation. The training set is split into a training subset and validation subset, preserving class balance. Candidate settings are ranked by validation AUC, then the best configuration is retrained on the full training set before final test evaluation.

**Latest results at threshold `0.5`:**

- Accuracy: `0.6363`
- Precision: `0.9152`
- Recall: `0.6423`
- F1-score: `0.7548`
- ROC-AUC: `0.6715`
- Average Precision: `0.9306`

**Threshold behavior:**

- At `0.3`, recall rises to `0.8702` but bad approvals increase to `52,625`
- At `0.6`, precision rises to `0.9331` and bad approvals drop to `17,093`, with lower recall (`0.4830`)

### 2) Neural Network (MLP)

**Setup (short):** dense MLP on scaled features with class-aware training, dropout, early stopping, and tuned architecture/training parameters.

**Validation:** The MLP uses a manual hyperparameter search with stratified holdout validation. Several combinations of hidden-layer structure, dropout, learning rate, and batch size are trained on the training subset and compared using validation AUC. This is not k-fold cross-validation; each configuration is evaluated on one stratified validation split. The selected configuration is retrained as the final MLP before test evaluation.

**Latest results at threshold `0.5`:**

- Accuracy: `0.6470`
- Precision: `0.9258`
- Recall: `0.6468`
- F1-score: `0.7616`
- ROC-AUC: `0.7035`
- Average Precision: `0.9374`

**Threshold behavior:**

- At `0.3`, recall is high (`0.9359`) but bad approvals increase to `57,717`
- At `0.6`, precision improves to `0.9432` and bad approvals fall to `13,508`, with recall reduced to `0.4545`

### 3) XGBoost

**Setup (short):** gradient-boosted trees on engineered tabular features with row weighting for the minority default class.

**Validation:** XGBoost uses a manual grid search with early stopping. A smaller class-balanced tuning subset is created to speed up experimentation, candidate tree configurations are compared using AUC, and early stopping selects an appropriate boosting round. The best parameter set is then used to train the final XGBoost model before evaluation on the full test split.

**Latest results at threshold `0.5`:**

- Accuracy: `0.6374`
- Precision: `0.9332`
- Recall: `0.6290`
- F1-score: `0.7515`
- ROC-AUC: `0.7204`
- Average Precision: `0.9422`

**Threshold behavior:**

- At `0.3`, recall increases to `0.9110` but bad approvals rise to `52,303`
- At `0.6`, precision improves to `0.9486` and bad approvals drop to `12,191`, with recall reduced to `0.4552`

### Phase 2 Summary (Consistent Comparison at `0.5`)

| Model               | Accuracy | Precision |   Recall | F1-score |  ROC-AUC | Avg Precision |
| ------------------- | -------: | --------: | -------: | -------: | -------: | ------------: |
| Logistic Regression | `0.6363` |  `0.9152` | `0.6423` | `0.7548` | `0.6715` |      `0.9306` |
| Neural Network      | `0.6470` |  `0.9258` | `0.6468` | `0.7616` | `0.7035` |      `0.9374` |
| XGBoost             | `0.6374` |  `0.9332` | `0.6290` | `0.7515` | `0.7204` |      `0.9422` |

At threshold `0.5`, the Neural Network has the strongest F1-score and recall, while XGBoost has the highest precision and ranking metrics (ROC-AUC and average precision). Logistic Regression remains the simplest and most interpretable baseline, with performance close to the other models after tuning.

---

## Phase 3 — Results and Evaluation

Phase 3 consolidates the saved Phase 2 model outputs into a final comparison workflow. The evaluation notebook loads each model's `2026-04-22` summary and threshold files, combines the metrics, generates visual comparisons, and exports a final model comparison table.

### Phase 3 Validation

Phase 3 validates consistency across models by comparing them on the same held-out test split and the same threshold grid. The summary files compare default-threshold performance at `0.5`, while the threshold files compare how each model behaves as the approval cutoff changes from `0.3` to `0.8`.

### Phase 3 Evaluation Artifacts

| File                                             | Description                                      |
| ------------------------------------------------ | ------------------------------------------------ |
| `Phase_3_Results_And_Evaluation.ipynb`           | Notebook for combined model evaluation           |
| `results/plots_combined.py`                      | Script version of the combined plotting workflow |
| `results/plots/final_model_comparison_table.csv` | Final summary table sorted by ROC-AUC            |
| `results/plots/model_comparison_bar.png`         | Bar chart comparing model metrics at threshold 0.5 |
| `results/plots/precision_recall_curve.png`       | Precision-recall curves across thresholds        |
| `results/plots/f1_threshold_comparison.png`      | F1-score trends across probability thresholds    |
| `results/plots/confusion_matrices_grid.png`      | Confusion matrices for all three models          |
| `results/plots/roc_curve_comparison.png`         | ROC curve comparison for all three models        |

---

## Dependencies

```
pandas
numpy
scikit-learn
nltk
xgboost
matplotlib
seaborn
```

---

> **Note:** AI assistance was used in generating code comments and documentation throughout this project.
