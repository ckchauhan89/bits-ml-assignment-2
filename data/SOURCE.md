# Dataset provenance
UCI Breast Cancer Wisconsin (Diagnostic), dataset 17.
Source: https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic
DOI: https://doi.org/10.24432/C5DW2B
Original investigators: W. Nick Street, William H. Wolberg and Olvi L. Mangasarian.

This package exports scikit-learn's bundled copy using load_breast_cancer(as_frame=True).
It does not claim to have downloaded the current UCI archive directly.
Official loader documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
The UCI dataset is distributed under CC BY 4.0; preserve attribution and record transformations.

569 rows; 30 real-valued features. The raw identifier is absent in the bundled feature matrix.
Original sklearn class encoding: 0=malignant, 1=benign.
Our exported target: 1=malignant, 0=benign, using (original_target == 0).
Feature column names are preserved from sklearn; see model/manifest.json.
The full export is data/breast_cancer.csv. test_data.csv is the stratified 20% holdout only.
No synthetic examples or generated target values are used.
