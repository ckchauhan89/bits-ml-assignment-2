"""Inference-only Streamlit app. Upload labeled holdout data, never refit."""
import csv
import io
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from evaluation import validate, evaluate

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='UCI Classification Explorer', layout='wide')
st.title('UCI Classification Explorer')
st.caption('Compare five saved classifiers on labeled test data. Educational model evaluation.')
st.info('Target: 0 = benign; 1 = malignant (positive class). Upload only held-out test data.')
manifest_path = ROOT / 'model' / 'manifest.json'
if not manifest_path.exists():
    st.error('Saved models are missing. Run python train_models.py in BITS Virtual Lab first.')
    st.stop()
manifest = json.loads(manifest_path.read_text())

@st.cache_resource
def load_models(signature):
    return {name: joblib.load(ROOT / 'model' / filename)
            for name, filename in manifest['models'].items()}

sample = ROOT / 'test_data.csv'
if sample.exists():
    st.download_button('Download held-out test CSV', sample.read_bytes(),
                       file_name='test_data.csv', mime='text/csv')
with st.expander('Required CSV columns and metric definitions'):
    st.write(manifest['features'] + ['target'])
    st.write('Precision, Recall and F1 use malignant=1. AUC uses positive-class probabilities. '
             'Undefined precision/recall/F1 are reported as zero. AUC needs both classes.')
selected = st.selectbox('Classifier', list(manifest['models']))
upload = st.file_uploader('Upload labeled test CSV', type=['csv'])
if upload is None:
    st.write('Upload test_data.csv to calculate results for all five models.')
    st.stop()
try:
    content = upload.getvalue().decode('utf-8-sig')
    header = next(csv.reader(io.StringIO(content)), [])
    if len(header) != len(set(header)):
        raise ValueError('Duplicate column names are not allowed.')
    frame = pd.read_csv(io.StringIO(content))
    X, y = validate(frame, manifest['features'])
except (ValueError, TypeError, UnicodeDecodeError, pd.errors.ParserError) as error:
    st.error(f'Invalid test CSV: {error}')
    st.stop()
try:
    signature = tuple((f, (ROOT / 'model' / f).stat().st_mtime_ns)
                      for f in manifest['models'].values())
    models = load_models(signature)
    rows, predictions = [], {}
    for name, model in models.items():
        metrics, predictions[name] = evaluate(model, X, y)
        rows.append({'ML Model Name': name, **metrics})
except Exception as error:
    st.error(f'Model evaluation failed: {error}. Check saved models and dependency versions.')
    st.stop()
result = pd.DataFrame(rows).set_index('ML Model Name')
st.subheader(f'{selected}: {len(X)} test rows')
for column, (metric, value) in zip(st.columns(6), result.loc[selected].items()):
    column.metric(metric, 'N/A' if pd.isna(value) else f'{value:.4f}')
if y.nunique() < 2:
    st.warning('AUC is undefined because this upload contains only one target class.')
left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, predictions[selected], labels=[0, 1], display_labels=['Benign (0)', 'Malignant (1)'],
        cmap='Blues', colorbar=False, ax=ax)
    ax.set_title(f'{selected}: confusion matrix')
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.write('Classification report')
    report = classification_report(y, predictions[selected], labels=[0, 1],
        target_names=['Benign', 'Malignant'], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T)
st.subheader('All five models on the uploaded test data')
st.dataframe(result.style.format('{:.4f}', na_rep='N/A'), width='stretch')
st.download_button('Download comparison CSV', result.to_csv().encode(),
                   file_name='uploaded_test_metrics.csv', mime='text/csv')
st.caption('Models are loaded from saved artifacts. No training occurs in this application.')
