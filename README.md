# Machine Learning Assignment 2: Breast Cancer Classification

## a. Problem statement

Implement and compare five machine learning classification models to classify breast tumor measurements as benign or malignant using the UCI Breast Cancer Wisconsin (Diagnostic) dataset.

Evaluate all five models on the same held-out test dataset using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC). Provide an interactive Streamlit application that loads saved models and evaluates uploaded test data without retraining.

## b. Dataset description

| Attribute | Description |
| --- | --- |
| Dataset | Breast Cancer Wisconsin (Diagnostic) |
| Repository | UCI Machine Learning Repository, dataset 17 |
| Source | https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic |
| DOI | https://doi.org/10.24432/C5DW2B |
| Classification task | Binary classification |
| Number of instances | 569 |
| Number of input features | 30 numerical features |
| Target column | `target` |
| Target encoding | 0 = Benign; 1 = Malignant |
| Class distribution | 357 benign and 212 malignant |
| Missing predictor values | None |
| Training samples | 455 |
| Test samples | 114 |

The dataset satisfies the assignment requirements of at least 500 instances and 12 input features.

### Features

The 30 predictors represent the mean, standard error and worst values of ten characteristics of cell nuclei: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension.

### Dataset preparation

The dataset is loaded using scikit-learn's bundled copy of the UCI dataset:

```python
load_breast_cancer(as_frame=True)
```

The original scikit-learn labels are remapped to `0 = Benign` and `1 = Malignant`, making malignant the positive class. No identifier column is used as an input feature.

### Training and test split

A stratified 80:20 split is performed using `random_state=42`. All five classifiers use the same 455 training samples and 114 held-out test samples. The test samples, including their target labels, are exported to `test_data.csv`.

The target column is excluded from the input features. It is used as the training label and as the actual label for evaluating test predictions.

### Preprocessing

`StandardScaler` is fitted on training data only for Logistic Regression, KNN and SVM. Each fitted scaler is saved together with its classifier in a pipeline. Decision Tree and Gaussian Naive Bayes use unscaled features.

Test data is transformed using the fitted preprocessing where applicable. No preprocessing or classifier is fitted on uploaded test data.

## c. GitHub Repository Link

https://github.com/ckchauhan89/bits-ml-assignment-2

## d. Models used

All five classifiers are trained on the same dataset and evaluated on the same held-out test samples.

| ML Model Name | Configuration |
| --- | --- |
| Logistic Regression | StandardScaler followed by LogisticRegression with `max_iter=2000` and `random_state=42` |
| Decision Tree | DecisionTreeClassifier with default tree parameters and `random_state=42` |
| KNN | StandardScaler followed by KNeighborsClassifier with `n_neighbors=5` |
| Naive Bayes | GaussianNB with default parameters |
| Support Vector Machine | StandardScaler followed by SVC with RBF kernel, `probability=True` and `random_state=42` |

### Evaluation metrics comparison

The following results were obtained on the 114 held-out test samples. Values are rounded to six decimal places.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.964912 | 0.996032 | 0.975000 | 0.928571 | 0.951220 | 0.924518 |
| Decision Tree | 0.929825 | 0.924603 | 0.904762 | 0.904762 | 0.904762 | 0.849206 |
| KNN | 0.956140 | 0.982308 | 0.974359 | 0.904762 | 0.938272 | 0.905824 |
| Naive Bayes | 0.938596 | 0.993386 | 1.000000 | 0.833333 | 0.909091 | 0.871489 |
| Support Vector Machine | 0.973684 | 0.994709 | 1.000000 | 0.928571 | 0.962963 | 0.944155 |

### Metric calculation

- **Accuracy:** Proportion of correctly classified samples.
- **AUC:** ROC-AUC calculated using predicted probabilities for malignant class `1`.
- **Precision:** Proportion of predicted malignant samples that are actually malignant.
- **Recall:** Proportion of actual malignant samples correctly identified.
- **F1:** Harmonic mean of precision and recall.
- **MCC:** Matthews Correlation Coefficient calculated from predicted and actual classes.

Precision, Recall and F1 use malignant (`1`) as the positive class, with `zero_division=0`. When an uploaded test file contains only one target class, AUC is undefined and displayed as `N/A`.

### Observations about model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Achieved 96.49% accuracy, F1 of 0.951220 and MCC of 0.924518. It produced the highest AUC, 0.996032, indicating the strongest ranking performance across thresholds in this experiment. Recall of 0.928571 shows that some malignant samples were missed at the classification threshold. |
| Decision Tree | Achieved 92.98% accuracy, the lowest among the five classifiers. Precision and recall were both 0.904762. Its AUC of 0.924603 and MCC of 0.849206 were also the lowest. This configuration performed less effectively on the held-out test set; these results alone do not establish whether overfitting caused the difference. |
| KNN | Achieved 95.61% accuracy, precision of 0.974359 and F1 of 0.938272. High precision indicates relatively few false-positive malignant predictions, while recall of 0.904762 indicates missed malignant samples. It performed below Logistic Regression and SVM on accuracy, F1 and MCC. |
| Naive Bayes | Achieved 93.86% accuracy and precision of 1.000000, indicating no false-positive malignant predictions on this test set. However, its recall of 0.833333 was the lowest, indicating the largest proportion of missed malignant samples. Its AUC of 0.993386 shows that strong ranking performance did not translate into equally strong recall at the classification threshold. |
| Support Vector Machine | Achieved the highest accuracy, 97.37%, highest F1, 0.962963, and highest MCC, 0.944155. Precision was 1.000000 and recall was 0.928571. It made no false-positive malignant predictions but still missed some malignant samples. Its AUC of 0.994709 was slightly below Logistic Regression. |
| Overall, Winner for your dataset? | Support Vector Machine is the overall winner for this held-out evaluation when accuracy, F1 and MCC are the selection criteria. Logistic Regression achieved the highest AUC and matched SVM's recall. The conclusion applies to this dataset split and these model configurations; a single holdout does not demonstrate universal superiority. |

## Live Streamlit Application

[Open Breast Cancer Classification App](https://bits-ml-assignment-2-ceeaydztsriylgr4zv9pim.streamlit.app/)

## Run the project

Local testing was completed using Python 3.13. From the project directory, install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Prepare the dataset, train all five models and generate evaluation results:

```bash
python3 train_models.py
```

Verify the saved models and recorded results:

```bash
python3 verify_project.py
```

Start the Streamlit application:

```bash
python3 -m streamlit run app.py
```

The training script generates the saved model artifacts, manifest, test CSV and metrics file. To run the application with the included saved models, installation and the Streamlit command are sufficient; retraining is separate from application execution.

## Application usage

1. Open the Streamlit application.
2. Select a classification model.
3. Upload `test_data.csv` from this repository.
4. View the six evaluation metrics for the selected model.
5. Inspect its confusion matrix.
6. Compare all five classifiers in the model performance comparison table.

The uploaded CSV must contain all 30 input features and the `target` column, using `0` for benign and `1` for malignant. Upload only held-out test samples.

The application loads previously trained models. It does not retrain classifiers or refit preprocessing on uploaded test data.

## Streamlit Community Cloud deployment

Deploy the GitHub repository `ckchauhan89/bits-ml-assignment-2`, branch `main`, with `app.py` as the entry point and Python 3.13 as the interpreter. Include `requirements.txt`, all five saved models, `model/manifest.json` and the source files in the repository.

After deployment, upload `test_data.csv` and verify all five models against the recorded results. Use compatible dependency versions when loading saved models. If the scikit-learn version changes, retrain and verify the models before deployment.

## Project files

| File or folder | Purpose |
| --- | --- |
| `app.py` | Runs the Streamlit interface. Accepts a labeled test CSV, loads saved models, displays six evaluation metrics and a confusion matrix for the selected model, and compares all five classifiers. No training occurs here. |
| `prepare_data.py` | Loads scikit-learn's bundled UCI dataset, remaps the target labels and exports `data/breast_cancer.csv`. Called automatically by `train_models.py`. |
| `train_models.py` | Creates the shared stratified split, fits preprocessing and all five classifiers on training data only, evaluates the test set, and saves models, test data, metadata and metrics. |
| `evaluation.py` | Provides shared CSV validation and six-metric calculation functions. Used by training, verification and the application. |
| `verify_project.py` | Checks that saved models reproduce recorded metrics. Also checks invalid-data rejection, feature ordering and undefined AUC for single-class test data. |
| `requirements.txt` | Lists the Python packages and versions needed to install and run the project, including Streamlit Community Cloud deployment. |
| `requirements-tested.txt` | Optional record of direct dependency versions used during reference verification. Installation instructions use `requirements.txt`. |
| `data/breast_cancer.csv` | Full exported dataset: 569 samples, 30 input features and the target. Used in the training workflow, not as the held-out upload file. |
| `data/SOURCE.md` | Documents dataset provenance, attribution, feature information and target-label transformation. |
| `test_data.csv` | Contains 114 held-out samples, all 30 input features and actual target labels. Used to evaluate saved models in the app. |
| `model/logistic_regression.joblib` | Fitted StandardScaler and Logistic Regression classifier saved together as a pipeline. |
| `model/decision_tree.joblib` | Fitted Decision Tree classifier. |
| `model/knn.joblib` | Fitted StandardScaler and KNN classifier saved together as a pipeline. |
| `model/naive_bayes.joblib` | Fitted Gaussian Naive Bayes classifier. |
| `model/svm.joblib` | Fitted StandardScaler and RBF Support Vector Machine classifier saved together as a pipeline. |
| `model/manifest.json` | Records feature names and order, label meanings, model filenames, split information and training scikit-learn version. Used by the app to validate inputs and locate saved models. |
| `results/metrics.csv` | Stores six evaluation metrics for each classifier on the shared holdout. Used for result verification and the README comparison table. |
| `README.md` | Describes the problem, dataset, repository, models, results, observations, execution and file purposes. |