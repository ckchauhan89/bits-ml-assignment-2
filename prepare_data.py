"""Export scikit-learn's bundled copy of UCI dataset 17; no network required."""
from pathlib import Path
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parent

def prepare():
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.data.copy()
    # sklearn uses malignant=0; explicitly remap so malignant is positive.
    frame['target'] = (dataset.target == 0).astype(int)
    destination = ROOT / 'data' / 'breast_cancer.csv'
    destination.parent.mkdir(exist_ok=True)
    frame.to_csv(destination, index=False)
    assert frame.shape == (569, 31)
    print(f'Exported {destination}: 569 rows, 30 predictors, target 1=malignant, 0=benign')
    return destination

if __name__ == '__main__':
    prepare()
