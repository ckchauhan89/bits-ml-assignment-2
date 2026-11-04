"""Run on BITS Virtual Lab; fixed shared holdout, no hyperparameter search."""
import json
from pathlib import Path
import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from prepare_data import prepare
from evaluation import validate, evaluate

ROOT = Path(__file__).resolve().parent

def main():
    frame = pd.read_csv(prepare())
    features = list(frame.columns.drop('target'))
    X, y = validate(frame, features)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    models = {
        'Logistic Regression': ('logistic_regression.joblib', make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))),
        'Decision Tree': ('decision_tree.joblib', DecisionTreeClassifier(random_state=42)),
        'KNN': ('knn.joblib', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
        'Naive Bayes': ('naive_bayes.joblib', GaussianNB()),
        'Support Vector Machine': ('svm.joblib', make_pipeline(
            StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42))),
    }
    (ROOT / 'model').mkdir(exist_ok=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    test = X_test.copy()
    test['target'] = y_test
    test.to_csv(ROOT / 'test_data.csv', index=False)
    rows = []
    files = {}
    for name, (filename, model) in models.items():
        model.fit(X_train, y_train)
        metrics, _ = evaluate(model, X_test, y_test)
        rows.append({'ML Model Name': name, **metrics})
        joblib.dump(model, ROOT / 'model' / filename)
        files[name] = filename
    manifest = {
        'features': features, 'target': 'target', 'positive_class': 'malignant',
        'labels': {'0': 'benign', '1': 'malignant'}, 'models': files,
        'random_state': 42, 'train_rows': len(X_train), 'test_rows': len(X_test),
        'sklearn_version': sklearn.__version__,
        'source': 'https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic',
    }
    (ROOT / 'model' / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / 'results' / 'metrics.csv', index=False)
    print(result.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print('\nReview these results, write your observations, and capture genuine BITS Lab execution evidence.')

if __name__ == '__main__':
    main()
