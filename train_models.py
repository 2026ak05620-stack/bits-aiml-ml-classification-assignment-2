
# ============================================================
# BITS PILANI - M.Tech AIML
# Machine Learning Assignment 2
# Dry Bean Classification using Five ML Models
# ============================================================

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

DATA_PATH = Path("data/Dry_Bean_Dataset.xlsx")
MODEL_DIR = Path("model")
RESULT_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("\n========== LOADING DATASET ==========")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}. "
        "Please place Dry_Bean_Dataset.xlsx inside the data folder."
    )

df = pd.read_excel(DATA_PATH)

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())

# Target column
TARGET_COLUMN = "Class"

if TARGET_COLUMN not in df.columns:
    raise ValueError("Target column 'Class' was not found.")

# Separate features and target
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Check dataset
if X.shape[1] < 12 or len(df) < 500:
    raise ValueError("Dataset does not meet assignment requirements.")

if X.isnull().any().any() or y.isnull().any():
    raise ValueError("Dataset contains missing values. Please inspect it.")

if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
    raise ValueError("All input features must be numerical.")

print("\nNumber of input features:", X.shape[1])
print("Number of observations:", len(df))
print("Number of target classes:", y.nunique())
print("\nClass distribution:")
print(y.value_counts())


# ------------------------------------------------------------
# 3. TRAIN-TEST SPLIT
# ------------------------------------------------------------

print("\n========== TRAIN TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Save test data with the actual target column.
# This CSV will be uploaded to our Streamlit application.
test_data = X_test.copy()
test_data[TARGET_COLUMN] = y_test

test_data.to_csv("test_data.csv", index=False)

print("\nTest data saved as test_data.csv")


# ------------------------------------------------------------
# 4. DEFINE FIVE CLASSIFICATION MODELS
# ------------------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ]),

    "Decision Tree": Pipeline([
        ("classifier", DecisionTreeClassifier(
            random_state=42
        ))
    ]),

    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]),

    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ]),

    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ])
}


# ------------------------------------------------------------
# 5. TRAIN MODELS AND CALCULATE METRICS
# ------------------------------------------------------------

print("\n========== MODEL TRAINING AND EVALUATION ==========")

results = []

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl"
}

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    # Train model
    model.fit(X_train, y_train)

    # Predict target classes
    y_pred = model.predict(X_test)

    # Predict class probabilities for multiclass AUC
    y_proba = model.predict_proba(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(
        y_test,
        y_proba,
        labels=model.classes_,
        multi_class="ovr",
        average="weighted"
    )

    precision = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    # Store results
    results.append({
        "ML Model Name": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })

    # Save trained model
    model_path = MODEL_DIR / model_files[model_name]

    joblib.dump(model, model_path)

    print(f"{model_name} completed!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")

    print("Saved model:", model_path)


# ------------------------------------------------------------
# 6. CREATE MODEL COMPARISON TABLE
# ------------------------------------------------------------

print("\n========== FINAL MODEL COMPARISON ==========")

results_df = pd.DataFrame(results)

# Sort models by weighted F1 score
results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

# Save comparison results
results_df.to_csv(
    RESULT_DIR / "model_comparison.csv",
    index=False
)

print(results_df.round(4).to_string(index=False))


# ------------------------------------------------------------
# 7. IDENTIFY OVERALL WINNER
# ------------------------------------------------------------

winner = results_df.iloc[0]

print("\n========== OVERALL WINNER ==========")

print("Best performing model:", winner["ML Model Name"])
print("Weighted F1 Score:", round(winner["F1 Score"], 4))


# ------------------------------------------------------------
# 8. SAVE METRICS IN JSON FORMAT
# ------------------------------------------------------------

metrics_dict = {
    "evaluation_method": "Weighted metrics; multiclass AUC using One-vs-Rest",
    "training_records": len(X_train),
    "testing_records": len(X_test),
    "number_of_features": X.shape[1],
    "number_of_classes": int(y.nunique()),
    "winner_by_weighted_f1": winner["ML Model Name"],
    "models": results_df.to_dict(orient="records")
}

with open(MODEL_DIR / "metrics.json", "w") as file:
    json.dump(metrics_dict, file, indent=4)


# ------------------------------------------------------------
# 9. COMPLETION MESSAGE
# ------------------------------------------------------------

print("\n==========================================")
print("ALL FIVE MODELS TRAINED SUCCESSFULLY!")
print("==========================================")

print("\nGenerated files:")
print("1. test_data.csv")
print("2. model/logistic_regression.pkl")
print("3. model/decision_tree.pkl")
print("4. model/knn.pkl")
print("5. model/naive_bayes.pkl")
print("6. model/svm.pkl")
print("7. results/model_comparison.csv")
print("8. model/metrics.json")

print("\nAssignment model implementation completed!")
