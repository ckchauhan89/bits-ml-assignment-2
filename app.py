"""Evaluate five saved classification models on uploaded test data."""

import csv
import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from evaluation import evaluate, validate


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide",
)

st.title("Breast Cancer Classification")
st.write(
    "Compare five machine learning classifiers using the "
    "UCI Breast Cancer Wisconsin (Diagnostic) dataset."
)

# Load saved model information.
manifest_path = ROOT / "model" / "manifest.json"

if not manifest_path.exists():
    st.error("Required saved model information is missing.")
    st.stop()

try:
    manifest = json.loads(manifest_path.read_text())
except (OSError, ValueError) as error:
    st.error(f"Unable to read model information: {error}")
    st.stop()


@st.cache_resource
def load_models(signature):
    """Load fitted models without retraining."""
    return {
        name: joblib.load(ROOT / "model" / filename)
        for name, filename in manifest["models"].items()
    }


# Required: model selection.
st.header("Model Selection")

selected = st.selectbox(
    "Choose a classification model",
    options=list(manifest["models"]),
)

# Required: test CSV upload.
st.header("Test Dataset")

upload = st.file_uploader(
    "Upload a test CSV containing all input features "
    "and the target column",
    type=["csv"],
)

if upload is None:
    st.stop()

# Validate the uploaded test data.
try:
    content = upload.getvalue().decode("utf-8-sig")
    header = next(csv.reader(io.StringIO(content)), [])

    if len(header) != len(set(header)):
        raise ValueError("Duplicate column names are not allowed.")

    frame = pd.read_csv(io.StringIO(content))
    X, y = validate(frame, manifest["features"])

except (
    ValueError,
    TypeError,
    UnicodeDecodeError,
    pd.errors.ParserError,
) as error:
    st.error(f"Invalid test CSV: {error}")
    st.stop()

# Evaluate all five saved models on the same uploaded data.
try:
    signature = tuple(
        (
            filename,
            (ROOT / "model" / filename).stat().st_mtime_ns,
        )
        for filename in manifest["models"].values()
    )

    models = load_models(signature)

    rows = []
    predictions = {}

    for name, model in models.items():
        metrics, prediction = evaluate(model, X, y)

        predictions[name] = prediction

        rows.append({
            "ML Model Name": name,
            **metrics,
        })

    results = pd.DataFrame(rows).set_index("ML Model Name")

except Exception as error:
    st.error(f"Unable to evaluate saved models: {error}")
    st.stop()

# Required: six evaluation metrics.
st.header("Evaluation Metrics")
st.subheader(selected)

columns = st.columns(6)

for column, (metric, value) in zip(
    columns,
    results.loc[selected].items(),
):
    column.metric(
        label=metric,
        value="N/A" if pd.isna(value) else f"{value:.4f}",
    )

if y.nunique() < 2:
    st.warning(
        "AUC is undefined because the uploaded test data "
        "contains only one target class."
    )

# Required: confusion matrix or classification report.
# This application uses a confusion matrix.
st.header("Confusion Matrix")

fig, ax = plt.subplots(figsize=(6, 4))

ConfusionMatrixDisplay.from_predictions(
    y,
    predictions[selected],
    labels=[0, 1],
    display_labels=["Benign (0)", "Malignant (1)"],
    cmap="Blues",
    colorbar=False,
    ax=ax,
)

ax.set_title(f"{selected} — Confusion Matrix")
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# Required: visible results for all five models.
st.header("Model Performance Comparison")

st.dataframe(
    results.style.format("{:.4f}", na_rep="N/A"),
    width="stretch",
)