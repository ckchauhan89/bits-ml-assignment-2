"""Meaningful consistency checks; no network or retraining required."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from evaluation import validate, evaluate

ROOT = Path(__file__).resolve().parent

def main():
    manifest = json.loads((ROOT / 'model' / 'manifest.json').read_text())
    test = pd.read_csv(ROOT / 'test_data.csv')
    X, y = validate(test, manifest['features'])
    expected = pd.read_csv(ROOT / 'results' / 'metrics.csv').set_index('ML Model Name')
    assert len(manifest['models']) == 5 and X.shape == (114, 30)
    for name, filename in manifest['models'].items():
        model = joblib.load(ROOT / 'model' / filename)
        actual, _ = evaluate(model, X, y)
        np.testing.assert_allclose(list(actual.values()), expected.loc[name].to_numpy(dtype=float), atol=1e-12)
        one_class = y == 0
        metrics, _ = evaluate(model, X.loc[one_class], y.loc[one_class])
        assert np.isnan(metrics['AUC'])
    bad_cases = [test.drop(columns=['target']), test.assign(target=3),
                 test.assign(**{manifest['features'][0]: np.inf}),
                 test.assign(extra=0), test.iloc[:0]]
    for bad in bad_cases:
        try:
            validate(bad, manifest['features'])
        except ValueError:
            continue
        raise AssertionError('Invalid upload was accepted')
    # Row order may change; class/feature meanings must not.
    reversed_columns = test[list(reversed(test.columns))]
    reordered, _ = validate(reversed_columns, manifest['features'])
    pd.testing.assert_frame_equal(reordered, X)
    print('PASS: five saved models reproduce metrics; single-class AUC is undefined; invalid CSVs rejected; feature order restored.')

if __name__ == '__main__':
    main()
