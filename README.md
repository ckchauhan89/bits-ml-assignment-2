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

The predictors represent the mean, standard error and worst values of ten characteristics of cell nuclei:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

### Dataset preparation

The dataset is loaded using scikit-learn's bundled UCI dataset:

```python
load_breast_cancer(as_frame=True)
```

The original scikit-learn labels are remapped so that malignant is the positive class:

- `0`: Benign
- `1`: Malignant

No identifier column is used as an input feature.

### Training and test split

A stratified 80:20 split is performed using `random_state=42`. All five classifiers use the same 455 training samples and 114 test samples.

The test samples, including their target labels, are exported to `test_data.csv`.

### Preprocessing

`StandardScaler` is fitted on training data only for Logistic Regression, KNN and SVM. Each fitted scaler is saved together with its classifier in a pipeline.

Decision Tree and Gaussian Naive Bayes use unscaled features.

Test data is transformed using the fitted preprocessing. No preprocessing or classifier is fitted on uploaded test data.

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

The following results were obtained on the 114 held-out test samples.

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

Precision, Recall and F1 use malignant (`1`) as the positive class, with `zero_division=0`.

When an uploaded test file contains only one target class, AUC is undefined and displayed as `N/A`.

### Observations about model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Achieved 96.49% accuracy, an F1 score of 0.951220 and MCC of 0.924518. It produced the highest AUC, 0.996032, indicating the strongest ranking performance across thresholds in this experiment. Recall of 0.928571 shows that some malignant samples were missed at the model's classification threshold. |
| Decision Tree | Achieved 92.98% accuracy, the lowest among the five classifiers. Precision and recall were both 0.904762. Its AUC of 0.924603 and MCC of 0.849206 were also the lowest. This configuration performed less effectively than the other models on the held-out test set; these results alone do not establish whether overfitting caused the difference. |
| KNN | Achieved 95.61% accuracy, precision of 0.974359 and F1 of 0.938272. Its high precision indicates relatively few false-positive malignant predictions, while recall of 0.904762 indicates missed malignant samples. It performed below Logistic Regression and SVM on accuracy, F1 and MCC. |
| Naive Bayes | Achieved 93.86% accuracy and precision of 1.000000, indicating no false-positive malignant predictions on this test set. However, its recall of 0.833333 was the lowest, indicating the largest proportion of missed malignant samples. Its high AUC of 0.993386 shows that strong ranking performance did not translate into equally strong recall at the classification threshold. |
| Support Vector Machine | Achieved the highest accuracy, 97.37%, highest F1 score, 0.962963, and highest MCC, 0.944155. Precision was 1.000000 and recall was 0.928571. It made no false-positive malignant predictions but still missed some malignant samples. Its AUC of 0.994709 was slightly below Logistic Regression. |
| Overall, Winner for your dataset? | Support Vector Machine is the overall winner for this held-out evaluation when accuracy, F1 and MCC are used as the selection criteria. Logistic Regression achieved the highest AUC and matched SVM's recall. The conclusion applies to this dataset split and these model configurations; a single holdout does not demonstrate universal superiority. |

## Run the project

Use a compatible Python environment. Local testing was completed using Python 3.13.

Install dependencies:

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

The training script generates the saved model artifacts, manifest, test CSV and metrics file.

## Application usage

1. Open the Streamlit application.
2. Select a classification model.
3. Upload `test_data.csv` from this repository.
4. View the six evaluation metrics for the selected model.
5. Inspect its confusion matrix.
6. Compare all five classifiers in the model performance comparison table.

The uploaded CSV must contain all 30 input features and the `target` column, using `0` for benign and `1` for malignant.

The application loads previously trained models. It does not retrain classifiers or refit preprocessing on uploaded test data.

## Streamlit Community Cloud deployment

1. Push the source code, dependencies, test data and saved model artifacts to GitHub.
2. Sign in to Streamlit Community Cloud using GitHub.
3. Select repository `ckchauhan89/bits-ml-assignment-2`.
4. Select branch `main` and entry-point file `app.py`.
5. Choose Python 3.13 in Advanced settings.
6. Deploy and check the installation logs.
7. Upload `test_data.csv` and verify all five models.

Use compatible dependency versions when loading saved models. If the scikit-learn version changes, retrain and verify the models before deployment.

## Project files

| File or folder | Purpose |
| --- | --- |
| `app.py` | Streamlit application for evaluating saved models |
| `prepare_data.py` | Load and export the UCI dataset and remap target labels |
| `train_models.py` | Split data, train classifiers, calculate metrics and save artifacts |
| `evaluation.py` | Validate uploaded data and calculate evaluation metrics |
| `verify_project.py` | Check model-result consistency and input validation |
| `requirements.txt` | Deployment dependencies |
| `requirements-tested.txt` | Dependency versions recorded during reference verification |
| `data/breast_cancer.csv` | Full exported dataset |
| `data/SOURCE.md` | Dataset provenance |
| `test_data.csv` | Labeled held-out test samples |
| `model/` | Five fitted classifiers or pipelines and `manifest.json` |
| `results/metrics.csv` | Evaluation results generated by the training script |
| `README.md` | Problem statement, dataset description, models, results and observations |
| `START_HERE.md` | Initial setup and submission guidance |

## BITS Virtual Lab execution and submission

The assignment requires execution on BITS Virtual Lab and one genuine screenshot as evidence. Local testing and cloud deployment do not replace this requirement.

Submit one PDF containing the following in order:

1. Clickable GitHub repository link.
2. Clickable live Streamlit application link.
3. Screenshot showing assignment execution on BITS Virtual Lab.
4. Required README content, including the metric comparison and model observations tables.

## Acknowledgment

AI assistance was used in preparing the reference implementation and documentation. The author must review and understand the implementation, verify the reported results and follow the assignment's academic integrity requirements.