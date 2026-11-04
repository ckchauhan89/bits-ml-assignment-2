"""Shared validation and binary metrics for training and the deployed app."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

def validate(frame, features):
    if frame.empty:
        raise ValueError('The CSV has no data rows.')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names are not allowed.')
    missing = set(features + ['target']) - set(frame.columns)
    extra = set(frame.columns) - set(features + ['target'])
    if missing or extra:
        raise ValueError(f'Column mismatch. Missing: {sorted(missing)}; unexpected: {sorted(extra)}')
    try:
        X = frame.loc[:, features].apply(pd.to_numeric, errors='raise')
        y = pd.to_numeric(frame['target'], errors='raise')
    except (ValueError, TypeError) as error:
        raise ValueError('All features and target must be numeric.') from error
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('Features must contain finite numbers without missing values.')
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError('Target must be 0 (benign) or 1 (malignant).')
    return X, y.astype(int)

def evaluate(model, X, y):
    prediction = model.predict(X)
    positive_index = list(model.classes_).index(1)
    scores = model.predict_proba(X)[:, positive_index]
    metrics = {
        'Accuracy': accuracy_score(y, prediction),
        'AUC': roc_auc_score(y, scores) if len(np.unique(y)) == 2 else np.nan,
        'Precision': precision_score(y, prediction, pos_label=1, zero_division=0),
        'Recall': recall_score(y, prediction, pos_label=1, zero_division=0),
        'F1': f1_score(y, prediction, pos_label=1, zero_division=0),
        'MCC': matthews_corrcoef(y, prediction),
    }
    return metrics, prediction
