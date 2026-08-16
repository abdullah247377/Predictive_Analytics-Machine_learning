# 📊 Index Exclusion Prediction

### An end-to-end machine learning case study in imbalanced binary classification

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20models-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-data%20wrangling-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-4C9A2A?style=flat-square)](https://imbalanced-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)

*Originally developed as a Final Year Project (MS Business Analytics, NUCES) — organized here as a general-purpose ML portfolio piece.*

## Overview

Can a set of financial, market, and macroeconomic signals flag — a year ahead of time — which firms are about to be dropped from a stock market index? This project builds a complete supervised learning pipeline to answer that on a panel of **337 firm-year observations** (77 firms, 2020–2024), framed as a **binary classification problem under severe class imbalance** (94% / 6%).

Rather than tuning one model in isolation, the project runs as a **comparative benchmark**: four classifiers are trained on identical stratified splits, evaluated with the same metric suite, and interpreted side by side — closer to how a model gets selected in practice than a single-shot notebook.

> The dataset happens to be finance (PSX / KSE-100 constituent firms), but the techniques — imputation, outlier handling, SMOTE vs. class-weighting, multi-model benchmarking, interpretability — transfer directly to churn, fraud, credit risk, or any other rare-event classification problem.

## What This Project Demonstrates

- 🧹 **Data engineering** — median imputation, 1st/99th-percentile winsorization, z-score standardization, scaler fit on train and applied to test only (no leakage)
- ⚖️ **Imbalanced classification** — `class_weight='balanced'` vs. SMOTE oversampling, benchmarked head-to-head instead of assumed
- 🤖 **Model comparison** — Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting trained under identical, reproducible conditions
- 📈 **Evaluation beyond accuracy** — precision, recall, F1, ROC-AUC, and confusion matrices, since accuracy alone is meaningless on a 94/6 split
- 🔍 **Interpretability** — logistic regression coefficients & odds ratios alongside Random Forest / Gradient Boosting feature importances
- 🧪 **Statistical diagnostics** — VIF for multicollinearity, correlation analysis, stratified train/test splitting
- ✅ **Robustness testing** — the best model re-run with and without SMOTE to stress-test a design choice, not just assume it helped

## Pipeline

```mermaid
flowchart LR
    A["Raw Data<br/>337 rows x 16 cols"] --> B["Clean & Impute<br/>median fill"]
    B --> C["Outlier Treatment<br/>winsorize 1-99%"]
    C --> D["Feature Scaling<br/>z-score standardize"]
    D --> E["Diagnostics<br/>VIF + correlation"]
    E --> F["Stratified Split<br/>80 / 20"]
    F --> G["Balance Classes<br/>SMOTE"]
    G --> H["Train 4 Models<br/>LR, DT, RF, GB"]
    H --> I["Evaluate<br/>ROC-AUC, F1, recall"]
    I --> J["Interpret<br/>feature importance"]
```

## Dataset at a Glance

| | |
|---|---|
| Observations | 337 firm-year rows · 77 firms · 14 sectors · 2020–2024 |
| Target | `Index Exclusion Status` — binary, 6.2% positive class |
| Features | 11 predictors — 6 financial, 3 market, 2 macroeconomic |
| Split | 80% train (269) / 20% test (68), stratified |
| Train class balance | 252 vs. 17 → rebalanced to 252 vs. 252 via SMOTE |

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression *(class-weighted)* | 0.779 | 0.077 | 0.250 | 0.118 | 0.770 |
| **Decision Tree** *(class-weighted)* | **0.853** | **0.250** | **0.750** | **0.375** | **0.816** |
| Random Forest *(SMOTE)* | 0.882 | 0.000 | 0.000 | 0.000 | 0.742 |
| Gradient Boosting *(SMOTE)* | 0.927 | 0.000 | 0.000 | 0.000 | 0.402 |

The **Decision Tree** (`max_depth=5`, `min_samples_leaf=5`) was the strongest model on the held-out test set — not the most complex one. It caught 3 of the 4 excluded firms (recall = 0.75) at ROC-AUC = 0.816.

## Visual Highlights

<table>
<tr>
<td width="50%">
<img src="assets/class_distribution.png" alt="Class distribution"/>
<sub><b>Severe class imbalance</b> — the positive class is just 6.2% of the sample, the central challenge the whole pipeline is built around.</sub>
</td>
<td width="50%">
<img src="assets/roc_curves.png" alt="ROC curves comparison"/>
<sub><b>ROC comparison</b> across all four models on the untouched test set — Decision Tree leads at AUC = 0.816.</sub>
</td>
</tr>
<tr>
<td width="50%">
<img src="assets/confusion_matrices.png" alt="Confusion matrices"/>
<sub><b>Confusion matrices</b> show what aggregate accuracy hides — both SMOTE ensembles never once flag a true positive.</sub>
</td>
<td width="50%">
<img src="assets/feature_importance.png" alt="Feature importance"/>
<sub><b>Feature importance</b> (Random Forest + Gradient Boosting, averaged) — cash flow and liquidity dominate.</sub>
</td>
</tr>
</table>

## Key Insights

- **Simple beat complex.** A depth-limited Decision Tree outperformed both ensemble methods — on a small, severely imbalanced dataset (17 minority training examples), model capacity wasn't the bottleneck.
- **SMOTE is not a free lunch.** Both SMOTE-trained ensembles post strong accuracy but **zero recall** — they never once identified a true minority case. A dedicated robustness check (Random Forest with vs. without SMOTE) confirmed it: the non-SMOTE version actually scored a *higher* ROC-AUC (0.844 vs. 0.742). Synthetic points interpolated from only 17 real examples didn't generalize to the holdout set — a useful, honest negative result rather than a cherry-picked win.
- **The most predictive signals were behavioral, not accounting ratios.** Operating cash flow, trading volume, and stock volatility outranked leverage or profitability ratios in both tree-based importances and logistic regression effect sizes.
- **Macro variables added the least.** GDP and inflation ranked lowest in importance — firm-level fundamentals dominated the macro backdrop for this kind of event.

## Tech Stack

**Language:** Python 3.11 · **Data:** pandas, NumPy · **Modeling:** scikit-learn · **Imbalance handling:** imbalanced-learn (SMOTE) · **Diagnostics:** statsmodels (VIF) · **Visualization:** Matplotlib, Seaborn · **Environment:** Jupyter Notebook

## Project Structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── IE_Analysis_2.ipynb
├── data/
│   └── IE_Dataset.xlsx
└── assets/
    ├── class_distribution.png
    ├── roc_curves.png
    ├── confusion_matrices.png
    └── feature_importance.png
```

## Getting Started

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
jupyter notebook notebooks/IE_Analysis_2.ipynb
```

> The notebook's first cell points `FILE_PATH` at a local path — update it to `data/IE_Dataset.xlsx` before running end to end.

## Future Work

- Hyperparameter tuning for SMOTE (`k_neighbors`) and the tree ensembles (grid / random search)
- Panel-aware validation (walk-forward or time-series split) instead of a single holdout
- Compare against Borderline-SMOTE / ADASYN for the small-minority-class regime
- Add SHAP for local, per-prediction interpretability

## Author

**Abdullah Anayat**
MS Business Analytics, FAST School of Management — National University of Computer & Emerging Sciences (NUCES), Lahore
Final Year Project advised by **Dr. Omer Mehmood**

*Add your email / LinkedIn / GitHub links here.*
