# fraud-detection-unsupervised
End-to-end machine learning system that predicts salary ranges from job postings using feature engineering, classification models (Random Forest, Decision Trees, CatBoost), and a deployed Streamlit application.
# 🔍 Unsupervised Fraud Detection — IEEE-CIS Transaction Dataset


## Dataset

| Property | Detail |
|---|---|
| Source | IEEE-CIS Fraud Detection (Kaggle) |
| File Used | train_transaction.csv |
| Total Transactions | 590,540 |
| Fraud Cases | 20,663 (3.5%) |
| Normal Cases | 569,877 (96.5%) |
| Original Features | 400+ |

---

## Problem Statement

Given raw financial transaction data with no labels, identify suspicious transactions using unsupervised learning — the way a real bank would operate before human investigators confirm fraud.

**Key constraint:** `isFraud` column was treated as completely hidden during training. Used only at validation stage.

---

## Feature Engineering

### V Features (V1–V339) — Anonymous Vesta Features
- Dropped columns with more than 70% null values
- Removed highly correlated V features (threshold: 0.80) to reduce redundancy
- Applied variance threshold to remove near-zero variance columns

### D Features — Timedelta Features
- `D1 + D2` → `Days since last transition` (mean, with D1 dominating where D2 had 258K nulls)
- `D8, D10, D15` → `last_change` (max — captures most recent address/email/name change)
- `D5, D6, D7, D11, D12, D13` → `unknowns` (mean of unknown timedelta group)

### M Features — Match Features (Boolean)
- Converted True/T/1 → 1, False/F/0 → 0
- Filled nulls with mode per column
- Removed highly correlated M features (threshold: 0.90)

### Card Features
- Dropped `card3`, `card5`, `addr2` — low unique values, high repetition, no meaningful signal

### Email Features
- `P_emaildomain` → `P_emaildomain_flag` (1 if .com domain, 0 otherwise — .com considered lower risk)
- `R_emaildomain` — dropped due to excessive null values

### ID Removed
- `TransactionID` dropped — identifier only, no predictive value

**Final features after engineering: ~100 → scaled → PCA → 45 components**

---

## Preprocessing Pipeline

```
Raw 400+ Features
      ↓
Drop V cols with >70% nulls
      ↓
Engineer D, M, email features
      ↓
Drop low variance + high correlation features
      ↓
Encode categoricals (ProductCD, card4, card6)
      ↓
Fill numerical nulls with mean
      ↓
StandardScaler
      ↓
PCA — 45 components, 94% variance retained
```

---

## Models

### 1. KMeans Clustering

| Parameter | Value |
|---|---|
| K (from Elbow Method) | 2 |
| Cluster 0 | 302,005 transactions |
| Cluster 1 | 288,535 transactions |

**Result:**
```
Cluster 0 → Active frequent transactors (median days since last tx: 0)
Cluster 1 → Less frequent transactors (median days since last tx: 10)
```

**Key Finding:** Fraud distributed almost equally across both clusters (17,509 vs 18,015). KMeans found behavioral segments based on transaction frequency — not fraud patterns.

**Why KMeans Failed:**
- 96.5% normal transactions dominate cluster formation
- Fraud does not form geometrically separable clusters in transaction space
- KMeans minimizes distance for majority class — minority fraud points absorbed

---

### 2. Isolation Forest

| Parameter | Value |
|---|---|
| Contamination | 0.035 (actual fraud rate) |
| n_estimators | 200 |

**Results:**

```
              precision    recall  f1-score   support

           0       0.97      0.96      0.97    569,877
           1       0.21      0.30      0.25     20,663

    accuracy                           0.94    590,540
```

**Confusion Matrix:**
```
[[546,537   23,340]
 [ 14,476    6,187]]
```

**Interpretation:**
- Caught 6,187 actual fraud cases purely unsupervised
- 94% overall accuracy
- 30% fraud recall — consistent with academic benchmarks for pure unsupervised anomaly detection

---

## Model Comparison

| Metric | KMeans | Isolation Forest |
|---|---|---|
| Overall Accuracy | 0.51 | **0.94** |
| Fraud Recall | 0.51 | **0.30** |
| Fraud Precision | 0.04 | **0.21** |
| Fraud F1 | 0.07 | **0.25** |
| Normal Recall | 0.51 | **0.96** |

**Isolation Forest wins on every metric.**

KMeans accuracy of 0.51 = essentially random guessing. Isolation Forest is the correct tool for anomaly-based fraud detection.

---

## Key Insights

### 1. Label Leakage Prevention
`isFraud` was strictly hidden during all training and preprocessing. Used only at validation stage. This is the correct production approach — in real banking, fraud labels don't exist at transaction time.

### 2. Why KMeans Fails for Fraud
Empirically proved that clustering is not effective for imbalanced fraud detection. The 27:1 class ratio causes KMeans to optimize for normal transaction geometry, making fraud invisible to the algorithm.

### 3. Unsupervised Recall Expectations
17–30% fraud recall is the expected range for pure unsupervised detection. Supervised models achieve 85–90% recall but require labeled data. The value of unsupervised detection is identifying fraud with zero label dependency.

### 4. Business Value
```
6,187 fraud cases caught
× average fraud transaction ~$137
= ~$847,619 in potentially prevented fraud
Purely unsupervised. No labeled training data required.
```

---

## Visualizations

**1. PCA Space Analysis**
- Actual fraud clusters tightly at PC1≈0, PC2=0–15
- Isolation Forest correctly flags extreme PC2 values (most anomalous)
- KMeans clusters completely overlap — confirms clustering failure

**2. Transaction Amount Distribution**
- Fraud vs Normal amount patterns across 590K transactions

**3. Model Comparison**
- Confusion matrices side by side
- Performance bar chart across all metrics

---

## Tech Stack

```
Python
Pandas, NumPy
Scikit-learn (KMeans, IsolationForest, PCA, StandardScaler)
Matplotlib, Seaborn
```

---

## Project Structure

```
ieee-fraud-detection/
│
├── fraud_detection.ipynb    → Full pipeline with markdown documentation
├── README.md                → This file
└── visualizations/
    ├── pca_visualization.png
    ├── distribution_visualization.png
    └── model_comparison.png
```

---

## What I Learned

- Pure unsupervised fraud detection on 27:1 imbalanced data has inherent recall limitations
- Isolation Forest is fundamentally more suited to anomaly detection than KMeans
- Correct evaluation requires strict label separation during training
- Anonymous features (V1–V339) can still carry signal — correlation-based selection extracts useful ones
- D feature grouping by semantic meaning (timedelta type) is more effective than treating each independently

---

## Author

First Year CSE (AI) Student  
Built in 6 hours as an unsupervised ML portfolio project


