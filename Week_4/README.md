# Week 4 - Customer Churn Prediction Using Machine Learning

## Internship Project

This project demonstrates an end-to-end machine learning workflow for predicting customer churn using Python and Scikit-learn.

## Objective

Predict whether a customer is likely to churn based on customer account and behavioral information.

## Models

1. Logistic Regression
2. Decision Tree Classifier

## Preprocessing

- Missing-value imputation
- Numerical feature scaling
- One-hot encoding of categorical features
- Stratified train/test split
- Scikit-learn Pipeline and ColumnTransformer

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

## Dataset

The dataset is self-generated synthetic data for the internship assignment. It contains 2,200 customer records.

Important: this is not proprietary or real telecom-company data.

## Project Structure

```text
week4-customer-churn-ml/
│
├── customer_churn_prediction.py
├── week4_customer_churn_dataset.csv
├── model_comparison.csv
├── Week_4_Machine_Learning_Model_Development_and_Evaluation.docx
└── visualizations/
    ├── class_distribution.png
    ├── confusion_matrix_logistic_regression.png
    ├── model_metric_comparison.png
    └── roc_curve_model_comparison.png
```

## Installation

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Run

```bash
python customer_churn_prediction.py
```

The script will print the model evaluation results and regenerate the visualizations.

## Main Result

The project evaluates Logistic Regression and Decision Tree models using both threshold-based metrics and ROC-AUC. Special attention is given to recall because false negatives can represent customers who churn without receiving a retention intervention.

## Author

Chirantan Bag
Computer Science Engineering
