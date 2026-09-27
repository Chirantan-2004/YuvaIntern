# WEEK 4 - MACHINE LEARNING MODEL DEVELOPMENT & EVALUATION
# Project: Customer Churn Prediction Using Python

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, classification_report
)

RANDOM_STATE = 42
DATA_FILE = "week4_customer_churn_dataset.csv"
OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load data
df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("CUSTOMER CHURN - MACHINE LEARNING PROJECT")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["churn"].value_counts())

# 2. Target distribution
churn_counts = df["churn"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
plt.bar(["Stayed", "Churned"], churn_counts.values)
plt.title("Customer Churn Distribution")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")

for i, value in enumerate(churn_counts.values):
    plt.text(i, value + 10, str(value), ha="center")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "class_distribution.png"),
    dpi=180
)
plt.close()

# 3. Features and target
X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

numeric_features = [
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "usage_hours_month"
]

categorical_features = [
    "contract_type",
    "payment_method",
    "internet_service",
    "senior_customer",
    "auto_pay"
]

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

# 5. Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 6. Models
logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])

decision_tree_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=15,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])

# 7. Train
print("\nTraining models...")
logistic_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)

# 8. Predictions
lr_predictions = logistic_model.predict(X_test)
lr_probabilities = logistic_model.predict_proba(X_test)[:, 1]

tree_predictions = decision_tree_model.predict(X_test)
tree_probabilities = decision_tree_model.predict_proba(X_test)[:, 1]

# 9. Evaluation function
def evaluate_model(name, y_true, predictions, probabilities):
    results = {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1 Score": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities)
    }

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    for metric, value in results.items():
        print(f"{metric:<12}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        predictions,
        target_names=["Stayed", "Churned"],
        zero_division=0
    ))

    return results

lr_metrics = evaluate_model(
    "LOGISTIC REGRESSION",
    y_test,
    lr_predictions,
    lr_probabilities
)

tree_metrics = evaluate_model(
    "DECISION TREE",
    y_test,
    tree_predictions,
    tree_probabilities
)

# 10. Model comparison
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

comparison = pd.DataFrame({
    "Logistic Regression": [lr_metrics[m] for m in metrics],
    "Decision Tree": [tree_metrics[m] for m in metrics]
}, index=metrics)

print("\nMODEL COMPARISON")
print(comparison.round(4))

comparison.to_csv("model_comparison.csv")

# 11. Confusion matrix
cm = confusion_matrix(y_test, lr_predictions)
tn, fp, fn, tp = cm.ravel()

print("\nLOGISTIC REGRESSION CONFUSION MATRIX")
print(cm)
print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)

fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stayed", "Churned"]
).plot(ax=ax)

ax.set_title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix_logistic_regression.png"
    ),
    dpi=180
)
plt.close()

# 12. ROC curve
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probabilities)
tree_fpr, tree_tpr, _ = roc_curve(y_test, tree_probabilities)

plt.figure(figsize=(8, 6))

plt.plot(
    lr_fpr,
    lr_tpr,
    label=f"Logistic Regression (AUC = {lr_metrics['ROC-AUC']:.3f})"
)

plt.plot(
    tree_fpr,
    tree_tpr,
    label=f"Decision Tree (AUC = {tree_metrics['ROC-AUC']:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    "--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Model Comparison")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "roc_curve_model_comparison.png"
    ),
    dpi=180
)
plt.close()

# 13. Metric comparison
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    [lr_metrics[m] for m in metrics],
    width,
    label="Logistic Regression"
)

plt.bar(
    x + width / 2,
    [tree_metrics[m] for m in metrics],
    width,
    label="Decision Tree"
)

plt.xticks(x, metrics, rotation=20)
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title("Machine Learning Model Performance Comparison")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "model_metric_comparison.png"
    ),
    dpi=180
)
plt.close()

print("\nProject completed successfully.")
print("Visualizations saved in:", OUTPUT_DIR)
