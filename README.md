# Machine Learning Assignment 2: UCI Classification Explorer

> Learning/reference implementation generated with AI assistance. The assignment permits AI for learning support, not direct copy-paste submissions. Understand, adapt and explain the implementation, run it on BITS Virtual Lab, maintain genuine commits, and replace the placeholders below before submitting. Bundled model results were generated in a reference environment, not BITS Virtual Lab. No execution screenshot or deployment is claimed.

## a. Problem statement
Compare Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and Support Vector Machine for binary classification of breast tumor measurements as benign or malignant. Evaluate held-out data using Accuracy, ROC-AUC, Precision, Recall, F1 and Matthews Correlation Coefficient, and demonstrate inference using Streamlit. This is an educational classification experiment, not a clinical tool.

## b. Dataset description
- **Dataset:** Breast Cancer Wisconsin (Diagnostic), UCI dataset 17.
- **Source:** https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic
- **DOI:** https://doi.org/10.24432/C5DW2B
- **Distribution used:** scikit-learn's bundled `load_breast_cancer(as_frame=True)` copy, exported by `prepare_data.py`. No runtime download is needed. This is a real UCI dataset, not synthetic data. See `data/SOURCE.md`.
- **Dimensions:** 569 records and 30 numerical predictors, meeting the minimum 500 records/12 input features.
- **Features:** mean, standard error and worst values of radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension.
- **Target:** `target`, explicitly remapped to **0=benign, 1=malignant**. scikit-learn's original target encoding is reversed by `prepare_data.py`.
- **Class counts:** 357 benign, 212 malignant. No identifier is used as a predictor. The dataset has no missing predictor values.
- **Split:** stratified 80:20 train/test split, random seed 42; 455 training and 114 held-out test rows. All five models use exactly the same split.
- **Preprocessing:** fit StandardScaler on training rows only for Logistic Regression, KNN and SVM. The fitted scaler and classifier are saved together as a pipeline. Decision Tree and GaussianNB use unscaled inputs.
- The split, seed and hyperparameters are implementation choices, not prescribed assignment settings. No hyperparameter search was performed. Do not tune repeatedly against the held-out test set.

## c. GitHub Repository Link
**TODO:** Create your dedicated repository and replace this line with its real URL.
Suggested URL only (not verified or created): https://github.com/ckchauhan89/bits-ml-assignment-2

## d. Models used
| Model | Reference configuration |
| --- | --- |
| Logistic Regression | StandardScaler; max_iter=2000; random_state=42 |
| Decision Tree | Default tree parameters; random_state=42 |
| KNN | StandardScaler; 5 neighbors |
| Naive Bayes | GaussianNB defaults |
| Support Vector Machine | StandardScaler; RBF kernel; probability=True; random_state=42 |

### Evaluation metrics comparison
**Reference run only.** Rerun `python train_models.py` on BITS Virtual Lab and replace this table with your verified results from `results/metrics.csv`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.964912 | 0.996032 | 0.975000 | 0.928571 | 0.951220 | 0.924518 |
| Decision Tree | 0.929825 | 0.924603 | 0.904762 | 0.904762 | 0.904762 | 0.849206 |
| KNN | 0.956140 | 0.982308 | 0.974359 | 0.904762 | 0.938272 | 0.905824 |
| Naive Bayes | 0.938596 | 0.993386 | 1.000000 | 0.833333 | 0.909091 | 0.871489 |
| Support Vector Machine | 0.973684 | 0.994709 | 1.000000 | 0.928571 | 0.962963 | 0.944155 |

Precision, Recall and F1 use malignant=1, with zero_division=0. AUC uses the probability column corresponding to class 1. MCC is calculated from hard predictions. Undefined AUC on a one-class upload is displayed as N/A. These are binary metrics, not macro/weighted multiclass averages.

### Observations about model performance
Write your own explanation after running and inspecting the models. The questions below are guidance, not completed observations.

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | TODO: interpret its recall and AUC; inspect false negatives and explain the linear baseline. |
| Decision Tree | TODO: inspect errors and discuss the unpruned tree's limitations on this holdout. |
| KNN | TODO: discuss distance-based predictions and the effect of scaling; compare recall. |
| Naive Bayes | TODO: compare precision with recall and discuss its conditional independence assumption. |
| Support Vector Machine | TODO: compare F1, MCC and AUC with other models and inspect the confusion matrix. |
| Overall, Winner for your dataset? | TODO: state a criterion, justify it using actual values and acknowledge conflicting metrics. A single holdout does not prove general superiority. |

## Run on BITS Virtual Lab
Use Python **3.12** for the tested setup. Use VS Code or JupyterLab if available inside the lab; no particular IDE is required by the assignment.

```bash
python -m venv .venv
```
Activate on Linux/macOS: `source .venv/bin/activate`; Windows PowerShell: `.venv\Scripts\Activate.ps1`.
If the lab's managed environment does not permit virtual environments, use its provided interpreter and package installation mechanism.

```bash
python -m pip install -r requirements.txt
python train_models.py
python verify_project.py
python -m streamlit run app.py
```
The training command exports the dataset and overwrites the saved models, manifest, test CSV and metrics. Use the lab's preview/forwarded-port mechanism if needed. Capture **one genuine screenshot** showing BITS Virtual Lab and successful execution. This project does not supply fabricated lab evidence.

## Application usage
1. Start the app and download `test_data.csv`, or use the copy in the repository.
2. Upload only held-out test data. CSV must contain the 30 exact feature names and `target`, encoded as 0/1.
3. Select any of the five models. Inspect all six metrics, confusion matrix, classification report and the all-model comparison.
4. The app loads fitted artifacts only. It never calls fit on uploaded data and never refits preprocessing.
5. Missing/extra/duplicate columns, nonnumeric values, NaN/infinity and invalid labels are rejected. A single-class upload has undefined AUC.

## Deploy on Streamlit Community Cloud
1. Create a dedicated GitHub repository under your existing account; a new account is unnecessary.
2. Commit your adapted source, requirements, README, data provenance, test CSV and all five model artifacts/manifest. Do not commit `.venv`, credentials or caches.
3. Open https://streamlit.io/cloud and sign in with GitHub.
4. Create an app, choose your repository and branch (typically `main`) and set the entry point to `app.py`.
5. In advanced settings choose Python 3.12, matching this tested environment. Deploy and inspect build logs.
6. Open the app in a private browser window, upload `test_data.csv`, and verify all five classifiers against your recorded metrics.
7. Add the actual live URL below. Resolve repository/app permissions so evaluators can open both links.

**Live app URL:** TODO after deployment.

All direct dependencies are pinned to the versions used for reference verification. `requirements-tested.txt` records the exact direct versions from this reference check. Retrain if changing scikit-learn versions; never load untrusted joblib/pickle files.

## Files
- `prepare_data.py`: export UCI dataset from scikit-learn; remap labels.
- `train_models.py`: shared split, training, evaluation and artifact generation.
- `evaluation.py`: common CSV validation and six metrics.
- `app.py`: inference-only Streamlit frontend.
- `verify_project.py`: saved-model/metric consistency and upload-validation checks.
- `requirements.txt`: deployment dependencies.
- `requirements-tested.txt`: direct package versions recorded during reference verification.
- `data/breast_cancer.csv`: full exported dataset for training.
- `data/SOURCE.md`: provenance and column/label details.
- `test_data.csv`: 114 labeled held-out rows; 30 features plus target.
- `model/*.joblib`: five fitted classifiers/pipelines.
- `model/manifest.json`: feature order, classes, filenames and training settings.
- `results/metrics.csv`: unrounded reference metrics, regenerated by training.
- `START_HERE.md`: ordered execution and submission checklist.

## Final submission PDF (required order)
1. Clickable GitHub repository link.
2. Clickable deployed Streamlit app link.
3. One genuine screenshot of assignment execution on BITS Virtual Lab.
4. The required README content, including both filled tables and your observations.

Replace all TODOs, confirm actual lab results, customize the interface, verify links and complete the portal's **Submit** action. Deadline in assignment: **8 November 2026, 23:59**; confirm portal timezone. No draft submissions or deadline extensions are accepted.
