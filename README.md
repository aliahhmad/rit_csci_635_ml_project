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

## Phase 2 — Neural Network

Phase 2 applies a multilayer perceptron (MLP) to the preprocessed Lending Club features for binary loan default prediction. Because the final dataset is tabular rather than image-based, the neural network uses a dense architecture with one sigmoid output node instead of a flattened image pipeline or multi-class softmax output.

The implemented model uses:

- **Input shape** — 167 engineered features per loan
- **Hidden layers** — `Dense(256, relu) -> Dropout(0.3) -> Dense(128, relu) -> Dropout(0.2) -> Dense(64, relu)`
- **Output layer** — `Dense(1, sigmoid)` for binary classification
- **Loss** — `binary_crossentropy`
- **Optimizer** — Adam with learning rate `1e-3`
- **Metrics** — Accuracy, Precision, Recall, and AUC

To address class imbalance, the network is trained with **class weights**, which penalize mistakes on defaulted loans more heavily than mistakes on non-default loans. This prevents the model from collapsing into a majority-class predictor.

### Phase 2 Results

The latest saved neural-network run still showed useful ranking ability during training, with validation AUC reaching approximately **0.82**. This indicates that the model is learning signal above random chance, but the classification behavior at specific thresholds changed noticeably compared with the earlier run.

At the default decision threshold of `0.5`, the saved evaluation in the notebook reported:

- **Accuracy** — `0.39`
- **Class 0 (default)** — Precision `0.13`, Recall `0.65`, F1 `0.21`
- **Class 1 (non-default)** — Precision `0.87`, Recall `0.35`, F1 `0.50`
- **Macro F1** — `0.35`
- **Weighted F1** — `0.46`

These results show that the model's probability ranking is still stronger than its hard class predictions at the default cutoff. In other words, the network appears to separate examples reasonably well in terms of score ordering, but the `0.5` threshold is no longer producing the strongest classification behavior.

### Threshold Strategy

Because credit-risk prediction is sensitive to the cost of false positives and false negatives, the model was also evaluated across several probability thresholds instead of relying only on the default cutoff of `0.5`.

| Threshold | Precision | Recall | F1-score |
|------|------|------|------|
| `0.3` | `0.8717` | `1.0000` | `0.9315` |
| `0.4` | `0.8718` | `0.9988` | `0.9310` |
| `0.5` | `0.8703` | `0.3476` | `0.4968` |
| `0.6` | `0.7500` | `0.0000` | `0.0000` |
| `0.7` | `0.0000` | `0.0000` | `0.0000` |
| `0.8` | `0.0000` | `0.0000` | `0.0000` |

This threshold sweep showed a clear tradeoff:

- Lower thresholds improve recall and F1-score
- Higher thresholds improve precision but reduce recall

However, the latest run also shows that very low thresholds can become too permissive. At `0.3`, the model approves essentially every loan, which creates perfect or near-perfect recall for the non-default class but is not a realistic bank strategy because it also allows a large number of bad approvals.

This means the threshold sweep is still valuable, but it should now be interpreted as a business tradeoff study rather than proof that the lowest threshold is best. In the current run:

- `0.3` and `0.4` maximize recall and F1, but they are extremely aggressive
- `0.5` is more selective, though recall drops sharply
- `0.6+` becomes too conservative and effectively stops approving loans

The main takeaway is that threshold tuning remains important, but the latest results suggest the model needs additional calibration or tuning before a final deployment threshold can be recommended confidently.

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
