
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("Dry Bean Classification Model Evaluation")
st.write(
    "Upload a test dataset, select a trained classification model, "
    "and evaluate its performance."
)


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

MODEL_DIR = Path("model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl"
}


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# --------------------------------------------------
# DATA UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")
        st.write("Dataset shape:", data.shape)
        st.dataframe(data.head())

        target_column = "Class"

        if target_column not in data.columns:
            st.error(
                "The uploaded CSV must contain the target column 'Class'."
            )
            st.stop()

        # Select trained model
        selected_model_name = st.selectbox(
            "Select a classification model",
            list(MODEL_FILES.keys())
        )

        model_path = MODEL_DIR / MODEL_FILES[selected_model_name]

        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            st.stop()

        model = load_model(model_path)

        # Use the feature columns and their original training order
        if hasattr(model, "feature_names_in_"):
            feature_columns = list(model.feature_names_in_)
        else:
            feature_columns = [
                column for column in data.columns
                if column != target_column
            ]

        missing_columns = [
            column for column in feature_columns
            if column not in data.columns
        ]

        if missing_columns:
            st.error(
                f"Missing required feature columns: {missing_columns}"
            )
            st.stop()

        X_test = data[feature_columns]
        y_test = data[target_column]

        # --------------------------------------------------
        # PREDICTION
        # --------------------------------------------------

        predictions = model.predict(X_test)

        # --------------------------------------------------
        # EVALUATION METRICS
        # --------------------------------------------------

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        )

        mcc = matthews_corrcoef(y_test, predictions)

        auc = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)

            try:
                auc = roc_auc_score(
                    y_test,
                    probabilities,
                    labels=model.classes_,
                    multi_class="ovr",
                    average="weighted"
                )
            except ValueError:
                auc = None

        # --------------------------------------------------
        # DISPLAY RESULTS
        # --------------------------------------------------

        st.subheader("Model Evaluation Results")
        st.write("Selected model:", selected_model_name)

        metric_columns = st.columns(6)

        metric_values = [
            ("Accuracy", accuracy),
            ("AUC", auc),
            ("Precision", precision),
            ("Recall", recall),
            ("F1 Score", f1),
            ("MCC", mcc)
        ]

        for column, (metric_name, metric_value) in zip(
            metric_columns, metric_values
        ):
            with column:
                if metric_value is None:
                    st.metric(metric_name, "N/A")
                else:
                    st.metric(metric_name, f"{metric_value:.4f}")

        # --------------------------------------------------
        # CONFUSION MATRIX
        # --------------------------------------------------

        st.subheader("Confusion Matrix")

        labels = sorted(set(y_test) | set(predictions))

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=labels
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title("Confusion Matrix")

        st.pyplot(fig)
        plt.close(fig)

        # --------------------------------------------------
        # CLASSIFICATION REPORT
        # --------------------------------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            predictions,
            zero_division=0,
            output_dict=True
        )

        st.dataframe(pd.DataFrame(report).transpose())

        # --------------------------------------------------
        # DOWNLOAD PREDICTIONS
        # --------------------------------------------------

        output_data = data.copy()
        output_data["Predicted_Class"] = predictions

        st.download_button(
            label="Download Predictions CSV",
            data=output_data.to_csv(index=False),
            file_name="dry_bean_predictions.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"An error occurred: {error}")

else:
    st.info("Please upload a test CSV file to begin evaluation.")


# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

comparison_path = Path("results/model_comparison.csv")

if comparison_path.exists():
    st.divider()
    st.subheader("Comparison of All Five Models")

    comparison = pd.read_csv(comparison_path)
    st.dataframe(comparison, use_container_width=True)
