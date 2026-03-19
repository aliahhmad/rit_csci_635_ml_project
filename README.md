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

## Output Files

| File | Description |
|------|-------------|
| `X_train.csv` / `X_test.csv` | Unscaled features — for tree-based models and Naive Bayes |
| `X_train_scaled.csv` / `X_test_scaled.csv` | [-1, 1] scaled features — for SVM and Logistic Regression |
| `y_train.csv` / `y_test.csv` | Binary labels (1 = non-default, 0 = default) |

---

## Dependencies

```
pandas
numpy
scikit-learn
nltk
```

---

> **Note:** AI assistance was used in generating code comments and documentation throughout this project.