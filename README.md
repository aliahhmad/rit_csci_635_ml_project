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
- **Scaling** — MinMaxScaler to [-1, 1], fit on training data only, for use with SVM and Logistic Regression

---

## Output Files (Phase 1)

| File | Description |
|------|-------------|
| `X_train.csv` / `X_test.csv` | Unscaled features — for tree-based models and Naive Bayes |
| `X_train_scaled.csv` / `X_test_scaled.csv` | [-1, 1] scaled features — for SVM and Logistic Regression |
| `y_train.csv` / `y_test.csv` | Binary labels (1 = non-default, 0 = default) |

---

## Phase 2 — Model Training and Results

All three models were evaluated on the same preprocessed test split. Reported metrics below use the latest saved run files in `results/` dated `2026-04-10`.

### 1) Logistic Regression

**Setup (short):** trained on scaled features (`X_train_scaled.csv`) with class balancing.

**Latest results at threshold `0.5`:**

- Accuracy: `0.6362`
- Precision: `0.9152`
- Recall: `0.6421`
- F1-score: `0.7547`
- ROC-AUC: `0.6713`
- Average Precision: `0.9305`

**Threshold behavior:**

- At `0.3`, recall rises to `0.8702` but bad approvals increase to `52,627`
- At `0.6`, precision rises to `0.9331` and bad approvals drop to `17,095`, with lower recall (`0.4828`)

### 2) Neural Network (MLP)

**Setup (short):** dense MLP on scaled features with class-aware training.

**Latest results at threshold `0.5`:**

- Accuracy: `0.6804`
- Precision: `0.9184`
- Recall: `0.6952`
- F1-score: `0.7913`
- ROC-AUC: `0.6958`
- Average Precision: `0.9358`

**Threshold behavior:**

- At `0.3`, recall is very high (`0.9699`) but bad approvals increase to `64,823`
- At `0.6`, precision improves to `0.9380` and bad approvals fall to `15,905`, with recall reduced to `0.4868`

### 3) XGBoost

**Setup (short):** gradient-boosted trees on engineered tabular features.

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

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Avg Precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | `0.6362` | `0.9152` | `0.6421` | `0.7547` | `0.6713` | `0.9305` |
| Neural Network | `0.6804` | `0.9184` | `0.6952` | `0.7913` | `0.6958` | `0.9358` |
| XGBoost | `0.6374` | `0.9332` | `0.6290` | `0.7515` | `0.7204` | `0.9422` |

At threshold `0.5`, the Neural Network has the strongest overall balance (accuracy, recall, and F1), while XGBoost has the highest precision and ranking metrics (ROC-AUC and average precision).

---

## Dependencies

```
pandas
numpy
scikit-learn
nltk
xgboost
matplotlib
```

---

> **Note:** AI assistance was used in generating code comments and documentation throughout this project.
