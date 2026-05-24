# Real-Time Fraud Detection System with Explainable AI

## Project Overview

This project is an advanced Machine Learning based Fraud Detection System developed using the IEEE-CIS Fraud Detection dataset.

The system detects fraudulent transactions in real time using:

- LightGBM
- XGBoost
- Isolation Forest
- SHAP Explainable AI
- Streamlit Dashboard

The project also includes:

- Risk segmentation
- Threshold optimization
- Fraud analytics
- Interactive visualizations
- Explainable AI dashboard

---

# Problem Statement

Financial fraud causes billions of dollars in losses every year. Traditional rule-based systems often fail to detect modern fraud patterns and cannot explain their predictions.

This project builds an intelligent fraud detection pipeline capable of:

- Detecting fraudulent transactions
- Handling severe class imbalance
- Explaining model predictions using SHAP
- Providing live fraud analytics through Streamlit

---

# Dataset

Dataset Source:

https://www.kaggle.com/c/ieee-fraud-detection/data

Required files:

- train_transaction.csv
- train_identity.csv

Place both files inside the `data/` folder.

---

# Project Structure

```text
FraudDetection_[SuryaPratapSingh]/
│
├── analysis.ipynb
├── README.md
├── requirements.txt
│
├── charts/
│   ├── fraud_distribution.png
│   ├── transaction_amount_distribution.png
│   ├── correlation_heatmap.png
│   ├── LightGBM_roc_curve.png
│   ├── XGBoost_roc_curve.png
│   ├── threshold_optimization.png
│   ├── shap_summary.png
│   ├── shap_waterfall.png
│   ├── shap_dependence.png
│   └── fraud_rate_by_hour.png
│
├── data/
│   ├── train_transaction.csv
│   └── train_identity.csv
│
└── dashboard/
    ├── app.py
    ├── model.pkl
    ├── preprocessor.pkl
    └── feature_columns.pkl