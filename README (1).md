# Predicting KSE-100 Index Exclusion Using Machine Learning

A machine learning classification project that investigates whether the future exclusion of non-financial firms from the **KSE-100 Index** can be predicted using financial, market, and macroeconomic indicators.

The project was developed as a Final Year Project for the **MS in Business Analytics at FAST School of Management, National University of Computer and Emerging Sciences (FAST-NUCES), Lahore**.

---

## Project Overview

Index exclusion can have important consequences for companies and investors. Removal from a major benchmark can lead to forced selling by index-tracking funds, reduced visibility and liquidity, and potential pressure on the firm's market value.

Despite extensive research on bankruptcy, financial distress, stock returns, and index rebalancing, the predictive problem of **which firms are likely to be removed from an emerging-market index** remains relatively underexplored.

This project investigates whether machine learning can be used as an **early-warning tool for KSE-100 index exclusion**.

The study focuses specifically on non-financial firms and combines:

- Financial indicators
- Market indicators
- Macroeconomic indicators

The project compares four supervised classification algorithms and evaluates their ability to identify the relatively rare exclusion events.

---

## Research Questions

The analysis addresses three primary questions:

1. **Can machine learning models predict the removal of non-financial firms from the KSE-100?**
2. **Which financial, market, and macroeconomic variables are most important in predicting index exclusion?**
3. **Which machine learning model performs best for this prediction problem?**

---

## Dataset

The final panel dataset contains:

| Attribute | Description |
|---|---|
| Period | 2020–2024 |
| Observations | 337 firm-year observations |
| Companies | 77 non-financial firms |
| Industries | 14 |
| Target | Index Exclusion Status |
| Exclusion observations | 21 |
| Exclusion rate | 6.2% |
| Predictors | 11 |

The dependent variable is binary:

- `1` = firm was excluded from the KSE-100
- `0` = firm remained in the KSE-100

The dataset is structured as panel data, where each observation represents a **firm-year**.

Because index exclusion is a rare event, the dataset contains substantial class imbalance.

---

## Variables

### Financial Variables

| Variable | Description |
|---|---|
| Debt Equity Ratio | Measures financial leverage |
| Current Ratio | Measures short-term liquidity |
| ROA | Return on Assets |
| Interest Coverage | Ability to service interest obligations |
| EBIT Margin | Operating profitability |
| Operating Cash Flow | Cash generated from operations |

### Market Variables

| Variable | Description |
|---|---|
| Stock Return | Annual stock performance |
| Stock Volatility | Measure of stock price volatility |
| Trading Volume | Market liquidity indicator |

### Macroeconomic Variables

| Variable | Description |
|---|---|
| GDP | Gross Domestic Product |
| Inflation | Annual inflation rate |

---

## Data Sources

The dataset was constructed by combining publicly available secondary data from multiple sources:

- **Pakistan Stock Exchange (PSX)** — KSE-100 index composition, stock returns, volatility, trading volume, and company information
- **State Bank of Pakistan (SBP)** — financial statement variables
- **World Bank** — GDP and inflation indicators
- **Securities and Exchange Commission of Pakistan (SECP)** — additional verification of financial information

The data was manually collected and linked at the firm-year level.

---

## Methodology

The analytical workflow consisted of the following stages:

```text
Raw Data
    │
    ▼
Data Cleaning
    │
    ▼
Missing Value Treatment
    │
    ▼
Outlier Treatment
    │
    ▼
Feature Scaling
    │
    ▼
Exploratory Analysis
    │
    ▼
Train-Test Split
    │
    ├───────────────┐
    ▼               ▼
Class Weights     SMOTE
    │               │
    ▼               ▼
Machine Learning Models
    │
    ▼
Model Evaluation
    │
    ├── Accuracy
    ├── Precision
    ├── Recall
    ├── F1 Score
    └── ROC-AUC
    │
    ▼
Feature Importance
    │
    ▼
Robustness Check
```

### Preprocessing

**Missing values**

Missing financial observations were replaced using median imputation.

**Outliers**

Selected financial and market variables were winsorized at the 1st and 99th percentiles.

**Feature scaling**

Independent variables were standardized using z-score normalization, with the scaler fitted only on the training data to avoid data leakage.

**Train-test split**

The dataset was divided into:

- 80% training set — 269 observations
- 20% test set — 68 observations

Stratified sampling was used to preserve the class distribution.

**Class imbalance**

Two approaches were investigated:

1. Balanced class weights
2. Synthetic Minority Over-sampling Technique (SMOTE)

SMOTE was applied only to the training data, while the test set retained its natural class distribution.

---

## Machine Learning Models

Four supervised classification algorithms were developed and compared.

### 1. Logistic Regression

Used as an interpretable baseline model.

Balanced class weights were used to address the rare-event problem.

### 2. Decision Tree

A constrained Decision Tree was developed using:

- Maximum depth = 5
- Minimum samples per leaf = 5
- Balanced class weights

### 3. Random Forest

An ensemble of decision trees was trained on SMOTE-balanced data.

- Number of trees = 300

### 4. Gradient Boosting

A sequential tree-based ensemble was trained using:

- 200 iterations
- Learning rate = 0.05

---

## Model Evaluation

Because index exclusion is a rare event, overall accuracy alone is not sufficient to evaluate the models.

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Confusion matrices and ROC curves were also examined.

Recall was particularly important because failing to identify a genuinely excluded firm represents a false negative in the proposed early-warning application.

---

## Results

### Model Performance on Test Set

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.779 | 0.077 | 0.250 | 0.118 | 0.770 |
| **Decision Tree** | **0.853** | **0.250** | **0.750** | **0.375** | **0.816** |
| Random Forest | 0.882 | 0.000 | 0.000 | 0.000 | 0.742 |
| Gradient Boosting | 0.926 | 0.000 | 0.000 | 0.000 | 0.402 |

The **Decision Tree was the strongest model for the project's primary objective: detecting excluded firms**.

Although Random Forest and Gradient Boosting achieved higher overall accuracy, neither identified any excluded firms in the test set. The Decision Tree correctly identified **3 of the 4 excluded firms** in the test set, producing a recall of **0.75** and ROC-AUC of **0.816**.

This demonstrates why accuracy alone can be misleading in highly imbalanced classification problems.

---

## Key Predictors

Feature importance analysis identified the following variables as the most influential predictors:

| Rank | Variable | Mean Importance |
|---:|---|---:|
| 1 | Operating Cash Flow | 0.1958 |
| 2 | Trading Volume | 0.1354 |
| 3 | Stock Volatility | 0.1221 |
| 4 | ROA | 0.1110 |
| 5 | EBIT Margin | 0.1056 |
| 6 | Stock Return | 0.0813 |
| 7 | Interest Coverage | 0.0742 |
| 8 | Current Ratio | 0.0608 |
| 9 | Debt Equity Ratio | 0.0489 |
| 10 | Inflation | 0.0336 |
| 11 | GDP | 0.0313 |

The three most consistently important predictors were:

**Operating Cash Flow, Trading Volume, and Stock Volatility.**

This suggests that firm-level financial strength and market liquidity were more informative for exclusion prediction than the macroeconomic variables included in this study.

---

## SMOTE and Robustness Check

A key part of the project was examining whether SMOTE actually improved minority-class prediction.

The Random Forest model was evaluated both with and without SMOTE:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| RF + SMOTE | 0.882 | 0.000 | 0.000 | 0.000 | 0.742 |
| RF without SMOTE | 0.926 | 0.000 | 0.000 | 0.000 | **0.844** |

SMOTE did **not** improve minority-class detection in this dataset. In fact, the Random Forest without SMOTE achieved a substantially higher ROC-AUC.

The likely explanation is the extremely small number of minority observations available for generating synthetic examples. Only **17 excluded observations were present in the training set** before SMOTE.

This is an important finding of the project: **a commonly recommended imbalance technique does not necessarily improve out-of-sample performance when the minority class is extremely small.**

---

## Key Findings

### 1. Index exclusion can be predicted to a useful degree

The Decision Tree achieved an ROC-AUC of **0.816** and identified 75% of excluded firms in the held-out test set.

### 2. Firm-level variables were more influential

Operating Cash Flow, Trading Volume, and Stock Volatility emerged as the strongest predictors.

### 3. Accuracy was misleading

The ensemble models achieved high accuracy but failed to identify any excluded firms.

This highlights the importance of evaluating minority-class metrics rather than relying on accuracy alone.

### 4. Simpler models can outperform complex ensembles

The Decision Tree outperformed Random Forest and Gradient Boosting for the project's primary objective despite being considerably simpler.

### 5. SMOTE was not effective in this setting

The small number of actual exclusion events limited the ability of SMOTE to generate representative synthetic minority observations.

---

## Business / Practical Application

The proposed model can be viewed as a potential **early-warning screening tool** rather than a replacement for the formal KSE-100 index review process.

Potential users include:

### Investors and Fund Managers

Identify firms that may have elevated exclusion risk and evaluate potential portfolio implications before index rebalancing.

### Corporate Management

Monitor indicators associated with exclusion risk and identify areas where financial or market performance may require attention.

### Regulators and Market Institutions

Use predictive analytics as an additional screening mechanism alongside the existing rule-based index recomposition process.

---

## Technology Stack

**Programming Language**

- Python 3.11

**Data Analysis**

- pandas
- NumPy

**Visualization**

- Matplotlib
- Seaborn

**Machine Learning**

- scikit-learn
- imbalanced-learn

**Statistical Analysis**

- statsmodels

**Environment**

- Jupyter Notebook

---

## Repository Structure

```text
kse100-index-exclusion-ml/
│
├── data/
│   └── IE Dataset.xlsx
│
├── notebooks/
│   └── IE_Analysis 2.ipynb
│
├── report/
│   └── Final FYP.pdf
│
├── README.md
└── requirements.txt
```

> **Note:** The raw dataset contains manually collected financial and market information. If the dataset cannot be publicly redistributed, the `data/` directory should instead contain a data description and instructions for obtaining or reconstructing the dataset.

---

## Reproducibility

### 1. Clone the repository

```bash
git clone https://github.com/<username>/<repository-name>.git
cd <repository-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open

```text
notebooks/IE_Analysis 2.ipynb
```

Run the notebook sequentially from data preparation through model evaluation and robustness analysis.

---

## Limitations

The results should be interpreted within the limitations of the study:

- The dataset contains only **337 observations and 21 exclusion events**, limiting statistical power and generalizability.
- The study covers only five years, from **2020–2024**, a period characterized by significant economic volatility.
- Sector was not included as a model predictor despite potentially containing useful information.
- SMOTE was not extensively hyperparameter-tuned.
- The models were evaluated using a single stratified hold-out split rather than time-series or panel-aware cross-validation.
- The small number of minority observations makes synthetic oversampling particularly challenging.

Future research could address these limitations by expanding the dataset, adding more predictors, tuning the resampling and model parameters, and using panel-aware or time-based validation.

---

## Academic Project

**Title:** Predicting Index Exclusion of Non-Financial Firms from KSE-100 Index Using Machine Learning

**Author:** Abdullah Anayat  
**Registration:** 24L-7377  
**Program:** MS Business Analytics  
**Institution:** FAST School of Management, National University of Computer and Emerging Sciences  
**Advisor:** Dr. Omer Mehmood  
**Year:** 2026

---

## Disclaimer

This project is an academic research exercise. The predictive model should not be interpreted as financial advice or as an official prediction of Pakistan Stock Exchange index composition.

The model is intended to demonstrate the application of machine learning and financial analytics to an emerging-market index exclusion problem.
