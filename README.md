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
- **Training setup** — `validation_split=0.2`, `batch_size=256`, maximum `20` epochs
- **Regularization / stopping** — Early stopping on `val_loss` with `patience=5` and `restore_best_weights=True`

To address class imbalance, the network is trained with **class weights**, which penalize mistakes on defaulted loans more heavily than mistakes on non-default loans. This prevents the model from collapsing into a majority-class predictor.

### Phase 2 - Neural Network - Results

The latest neural-network run uses **early stopping** to monitor validation loss and restore the best weights instead of keeping the final epoch blindly. In practice, training stopped after **13 / 20** epochs, which helps reduce overfitting while preserving the strongest validation checkpoint.

During training, the model reached:

- **Best validation loss** — `0.3987`
- **Validation accuracy** — `0.7823`
- **Validation AUC** — approximately `0.826`
- **Validation precision** — `0.9664`
- **Validation recall** — `0.7899`

These results show that the model is learning meaningful ranking signal and that early stopping keeps the best-performing validation state.

At the default decision threshold of `0.5`, the saved evaluation in the notebook reported:

- **Accuracy** — `0.65`
- **Class 0 (default)** — Precision `0.22`, Recall `0.67`, F1 `0.33`
- **Class 1 (non-default)** — Precision `0.93`, Recall `0.65`, F1 `0.76`
- **Macro F1** — `0.55`
- **Weighted F1** — `0.71`

Compared with the earlier README version, these results are substantially stronger and more stable. The network now captures a meaningful share of the default class while still maintaining high precision on the non-default class.

### Phase 2 - Neural Network - Threshold Strategy

Because credit-risk prediction is sensitive to the cost of false positives and false negatives, the model was also evaluated across several probability thresholds instead of relying only on the default cutoff of `0.5`.

| Threshold | Precision | Recall | F1-score | Approved Loans (`TP + FP`) | Bad Approvals (`FP`) |
|------|------|------|------|------|------|
| `0.3` | `0.8908` | `0.9340` | `0.9119` | `517,883` | `56,536` |
| `0.4` | `0.9097` | `0.8152` | `0.8599` | `442,631` | `39,955` |
| `0.5` | `0.9298` | `0.6490` | `0.7644` | `344,760` | `24,194` |
| `0.6` | `0.9461` | `0.4888` | `0.6446` | `255,220` | `13,769` |
| `0.7` | `0.9607` | `0.3299` | `0.4912` | `169,637` | `6,670` |
| `0.8` | `0.9736` | `0.1961` | `0.3265` | `99,504` | `2,628` |

This threshold sweep showed a clear tradeoff:

- Lower thresholds approve more loans and improve recall / F1-score
- Higher thresholds improve precision and reduce bad approvals, but they also reject more potentially good loans

The updated results make the tradeoff much easier to interpret from a banking perspective:

- `0.3` delivers the highest recall and F1-score, but it also creates the most bad approvals
- `0.5` is a balanced middle ground with strong precision (`0.9298`) and substantially fewer bad approvals than `0.3`
- `0.7` and `0.8` are much more conservative and sharply reduce the number of approvals, which may be desirable if minimizing risky approvals matters most

The main takeaway is that **threshold tuning remains a business decision**, not just a modeling decision. Early stopping improved training stability, and the updated results now show a more credible range of operating points for different lending strategies.

---

## Phase 2 — Logistic Regression

Phase 2 also trains a Logistic Regression baseline on the preprocessed, scaled Lending Club feature set (`X_train_scaled.csv`, `X_test_scaled.csv`).  
The implementation uses:

- **Solver** — `newton-cg`
- **Class weighting** — `class_weight='balanced'`
- **Convergence tolerance** — `1e-4`

### Phase 2 - Logistic Regression - Optimization

Newton's method converged successfully at iteration **6** with:

- **Final loss** — `0.5866`
- **Max gradient at convergence** — `3.1229e-05`

### Phase 2 - Logistic Regression - Threshold Evaluation

Classification reports were generated at thresholds `0.3`, `0.4`, `0.5`, and `0.6`.  
Below are class-wise F1 and overall accuracy from the notebook output:

| Threshold | Accuracy | F1 (Class 0: Default) | F1 (Class 1: Non-default) |
|------|------|------|------|
| `0.3` | `0.80` | `0.27` | `0.88` |
| `0.4` | `0.73` | `0.30` | `0.83` |
| `0.5` | `0.64` | `0.31` | `0.76` |
| `0.6` | `0.52` | `0.30` | `0.64` |

Observed tradeoff from the run:

- As threshold increases, the model becomes more conservative on predicting non-default (`1`)
- **Precision for class 1 increases** while **recall for class 1 decreases**
- Recall for default class (`0`) increases as the threshold rises

### Phase 2 - Logistic Regression - Final Metrics (0.5 threshold)

- **Accuracy** — `0.6417`
- **Precision** — `0.9223`
- **Recall** — `0.6432`
- **F1-score** — `0.7578`
- **AUC-ROC** — `0.6910`

### Phase 2 - Logistic Regression - Top Positive Coefficients

Top features by positive coefficient magnitude:

| Feature | Coefficient |
|------|------|
| `total_bc_limit` | `3.996128` |
| `num_tl_90g_dpd_24m` | `1.894668` |
| `total_il_high_credit_limit` | `1.741458` |
| `tot_hi_cred_lim` | `1.497507` |
| `mort_acc` | `1.094652` |
| `num_actv_rev_tl` | `1.072439` |
| `issue_d` | `0.877694` |
| `mo_sin_old_rev_tl_op` | `0.678206` |
| `revol_bal` | `0.633575` |
| `earliest_cr_line` | `0.604793` |

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
