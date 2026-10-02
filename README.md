# Credit Scoring Model

A machine-learning project that explores consumer credit data and estimates a borrower's risk of becoming seriously delinquent within two years. It includes exploratory analysis, model comparison and tuning, and a Streamlit app for interactive and batch predictions.

> **Live app:** [Credit Scoring Model](https://creditscoringmodelpop.streamlit.app/)

This is an educational demonstration, not a credit decision system or financial advice. Its scores must not be used to approve, deny, price, or otherwise make decisions about real applicants.

## What the app does

- Scores a single applicant from a set of financial and repayment-history inputs.
- Shows a risk-score gauge and a configurable high-risk threshold.
- Includes safe and risky example profiles and a reset option.
- Provides what-if analysis by changing one input while holding the others constant.
- Accepts a CSV for batch scoring, displays a preview, and lets you download predictions.
- Displays model comparison and held-out test metrics, confusion matrix, precision-recall curves, and available EDA/model charts.

The displayed score is the model's `predict_proba` output. Because the model was trained with class weighting, this score is intended as a relative ranking, not a calibrated probability. The threshold controls the app's low/high-risk label and can be adjusted in the sidebar.

## Project structure

```text
.
├── app.py                         # Streamlit application
├── requirements.txt               # Pinned Python dependencies
├── data/
│   └── cs-training.csv            # Give Me Some Credit training data
├── models/
│   ├── credit_model.joblib        # Trained model loaded by the app
│   ├── config.json                # Saved decision threshold
│   ├── results.csv                # Model comparison results
│   └── test_scores.csv            # Held-out labels and model scores
├── notebooks/
│   ├── 01_eda.ipynb               # Data exploration and visualizations
│   └── 02_modeling.ipynb          # Feature preparation, training, and evaluation
├── src/
│   └── features.py                # Shared cleaning and feature engineering
└── assets/                        # EDA, model, and evaluation charts
```

The Streamlit app needs `models/credit_model.joblib` and `models/config.json` to start. The training CSV is required to rerun the notebooks, but not to score applicants in the app. Charts and CSV result files are optional; the app shows relevant messages or skips charts when they are absent.

## Run locally

### Requirements

- Python 3.11 or newer
- `pip`
- The model and config files in `models/` (included in this project)

From the repository root, create and activate a virtual environment, then install the pinned dependencies:

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, allow it for the current terminal only and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the app from the repository root:

```bash
streamlit run app.py
```

Streamlit prints a local URL, typically `http://localhost:8501`. Stop the server with `Ctrl+C`.

`requirements.txt` pins the project environment, including the notebook and analysis packages as well as the app's runtime dependencies. Installing the full file may take a few minutes.

## Use the app

### Single-profile scoring

Open the **Predict** tab and enter the applicant's age, income, debt ratio, credit utilization, credit accounts, dependents, and late-payment counts. You can use the example profile buttons or mark income as unknown. The gauge displays the model score; a score at or above the selected threshold is labeled high risk.

The default saved threshold is approximately `0.541`. It was selected from out-of-fold training predictions to maximize precision while requiring recall of at least 70%. Lowering the threshold generally flags more applicants; raising it generally flags fewer. This changes the classification trade-off, not the underlying model score.

### Batch CSV scoring

Download the CSV template from the batch-scoring expander, or upload a CSV containing all 10 input fields below. The app accepts these short names or the original Give Me Some Credit column names. It adds `risk_score` and `decision` to the output and provides a downloadable predictions CSV.

| Short column name | Original dataset column | Meaning |
| --- | --- | --- |
| `utilization` | `RevolvingUtilizationOfUnsecuredLines` | Credit utilization ratio |
| `age` | `age` | Borrower's age in years |
| `late_30_59` | `NumberOfTime30-59DaysPastDueNotWorse` | Times 30-59 days past due |
| `debt_ratio` | `DebtRatio` | Debt ratio |
| `income` | `MonthlyIncome` | Monthly income; blank values are allowed |
| `open_lines` | `NumberOfOpenCreditLinesAndLoans` | Open credit lines and loans |
| `late_90` | `NumberOfTimes90DaysLate` | Times 90 or more days past due |
| `real_estate_loans` | `NumberRealEstateLoansOrLines` | Real-estate loans or lines |
| `late_60_89` | `NumberOfTime60-89DaysPastDueNotWorse` | Times 60-89 days past due |
| `dependents` | `NumberOfDependents` | Number of dependents |

The CSV must include all 10 input columns. If present, the dataset's `Unnamed: 0` index column and `default` target column are ignored. The interface limits the number of rows shown in its preview; the downloadable output contains the scored rows.

## Data and modeling

The notebooks use the **Give Me Some Credit** dataset from the Kaggle competition. The supplied training CSV contains approximately 150,000 borrower records and a binary target, `SeriousDlqin2yrs`, indicating serious delinquency within two years. The positive class is uncommon (about 6.7% in the exploratory analysis), so accuracy alone would be misleading.

The workflow in the notebooks includes:

1. Explore missing values, class balance, distributions, relationships, and anomalous records.
2. Rename the dataset columns to concise internal feature names and remove records with invalid ages below 18.
3. Flag special late-payment values `96` and `98`, then treat them as missing values.
4. Engineer features including total late payments, any 90-day delinquency, monthly debt, income per person, missing-income indicator, over-limit utilization indicator, and log income. Clip extreme utilization and debt-ratio values using caps defined in `src/features.py`.
5. Compare a dummy baseline, logistic regression, a decision tree, random forest variants, and XGBoost, using ROC-AUC and classification metrics. The comparison also explores class weighting and SMOTE for the imbalanced target.
6. Tune the final Random Forest with randomized search and cross-validation, select a decision threshold from cross-validated predictions, and evaluate it on a held-out test set.

In the saved model comparison, XGBoost has the highest listed ROC-AUC (`0.864`), narrowly ahead of the Random Forest (`0.862`). The deployed app uses the tuned Random Forest, selected for the overall modeling workflow and interpretability, with the saved threshold in `models/config.json`. The app's **Model performance** tab calculates test ROC-AUC from saved scores and recalculates precision, recall, F1, and the confusion matrix as you move the threshold.

### Re-run the notebooks

Install the dependencies above, then open the notebooks in Jupyter or VS Code with the project's virtual environment selected as the kernel. Run `notebooks/01_eda.ipynb` for exploration, followed by `notebooks/02_modeling.ipynb` for training and evaluation. The notebooks use paths relative to their own directory and expect the dataset at `data/cs-training.csv`.

The modeling notebook saves or refreshes the model, threshold config, comparison table, and held-out scores under `models/`, and writes plots under `assets/`. Re-running training can overwrite those generated artifacts. The shipped model files let you use the app without training again.

## Deployment

The hosted app is available at **[https://creditscoringmodelpop.streamlit.app/](https://creditscoringmodelpop.streamlit.app/)**.

To deploy a copy with Streamlit Community Cloud, connect the repository, choose the branch containing the app, set the app entry point to `app.py`, and deploy. Ensure `requirements.txt`, `models/credit_model.joblib`, and `models/config.json` are included in the deployed repository. The dataset is not needed by the running app.

## Metrics glossary

- **ROC-AUC:** Measures how well the model ranks positive cases above negative cases across thresholds; `0.5` is chance ranking.
- **Precision:** Of the applicants flagged high risk, the share who were positive cases.
- **Recall:** Of the positive cases, the share flagged high risk.
- **F1-score:** Harmonic mean of precision and recall at the chosen threshold.
- **Threshold:** The score cutoff used to label a case high risk; it trades off precision and recall.

## Limitations and responsible use

- The dataset is historical and comes from a single source; it may not represent current borrowers or other populations.
- The model has not been audited for fairness, calibration, stability, or regulatory compliance.
- A high-risk label is a model output, not a verified statement about an individual.
- The threshold embodies a trade-off and may not be appropriate for another context.
- This project is for learning and demonstration only. Do not use it for real lending or other consequential decisions.

## Troubleshooting

- **Model files not found:** Confirm that `models/credit_model.joblib` and `models/config.json` exist, then run Streamlit from the project root.
- **Missing CSV columns:** Download the template in the app and ensure the uploaded file has all 10 required features.
- **Import or package errors:** Activate the intended virtual environment and install `requirements.txt` into that environment.
- **Notebook cannot find the dataset:** Confirm the file is at `data/cs-training.csv` and run notebook cells with the notebook's normal relative working directory.
