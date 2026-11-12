# Smart Health Advisory System

Heart disease classification with Logistic Regression, Random Forest, and XGBoost, presented through a Streamlit application.

**Educational prototype using synthetic data. Predictions are not medical diagnoses, and model probabilities are not clinically validated cardiovascular risks.**

## Quick start

Use Python 3.13. Run these commands from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If the environment already exists, skip its creation. If PowerShell blocks activation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The application loads the saved pipeline from `models/`. If the model is missing, train it first:

```powershell
python train_model.py
```

With uv installed, `uv sync --locked` provides an alternative environment setup. Open the URL printed by Streamlit and use `Ctrl+C` to stop the server.

## Project structure

```text
.
├── app.py                 # Streamlit interface
├── predict.py             # Input validation, prediction, and advisory
├── heart_guidance.py      # Educational condition topics and matching rules
├── train_model.py         # Model training, selection, and evaluation
├── health_data.py         # Shared data preparation and feature schema
├── audit_data.py          # Reproducible data-quality checks
├── data/                  # Original and cleaned datasets
├── notebooks/             # Existing exploration and model notebook
├── models/                # Saved pipeline and input metadata
├── results/               # Model metrics, classification reports, and plots
├── report/
│   └── college_report.md
├── requirements.txt
├── pyproject.toml
└── uv.lock
```

The datasets retain their original contents. The notebook's data paths work from either the project directory or `notebooks/`. Its older outputs are historical; the training script produces the current application results.

## Training and prediction

```powershell
python audit_data.py       # Inspect dataset quality
python train_model.py      # Train all three models and save the selected pipeline
python predict.py          # Predict using the saved schema's example defaults
```

For a custom JSON object containing all 19 input fields:

```powershell
python predict.py --input path/to/input.json
```

The exact fields, allowed categories, and numerical bounds are in `models/model_metadata.json`. Explicit JSON `null` values are handled by saved preprocessing. Missing keys, extra fields, invalid values, and contradictory blood-pressure pairs are rejected. Numerical bounds describe the synthetic training data, not medical reference ranges.

The Streamlit form uses this same schema and never retrains at startup. Predictions and general lifestyle suggestions are shown separately; the advisory does not prescribe medication.

The result also shows an **illustrative model-score category** based on the unrounded class-1 probability:

| Model-score category | Score interval |
|---|---|
| Low | 0% to below 25% |
| Moderate | 25% to below 50% |
| High | 50% to below 75% |
| Very high | 75% to 100% |

These equal-width bands are a display convention, not clinically validated risk groups or disease-severity levels. For example, 15.5% is a Low model score; it does not rule out disease. Very high does not mean severe disease. The binary prediction, trained model, and general advisory rules are unchanged. The prediction API and CLI include this summary as `model_score_band`; it is `null` if a probability is unavailable.

## Condition-related guidance

After prediction, the app can show several educational topics together, explain the reported factors that matched, and offer two practical steps for each. A separate comparison table explains the connections between the five conditions, heart attacks and cardiac arrest, with medical-source links.

| Topic | Factors used by the educational matching rules |
|---|---|
| Coronary artery disease | Reported smoking, hyperlipidemia, hypertension, diabetes, previous heart attack, sedentary activity or unhealthy diet; confirmed-unit cholesterol, BMI or BP flags |
| Blood-pressure health / hypertensive heart disease | Reported hypertension or a confirmed-unit BP flag |
| Heart failure prevention | Reported hypertension, diabetes or previous heart attack; confirmed-unit BMI or BP flags |
| Heart rhythm / arrhythmias | Reported smoking, High alcohol intake, hypertension, diabetes or previous heart attack; confirmed-unit BMI flag |
| Heart muscle / cardiomyopathy | Reported High alcohol intake, with amount and duration explicitly unknown |

These are topic matches, **not disease predictions, diagnostic criteria, or validated risk groups**. They appear independently of class 0/1 and its score. Obesity and cholesterol alone do not establish cardiomyopathy or hypertension. No match is not an all-clear; unknown inputs stay unknown in these rules rather than using the model's imputed values.

Numerical guidance is opt-in: confirm weight in kg, height in cm, BP in mmHg and total cholesterol in mg/dL. Adult BMI is recalculated from the entered weight and height (never the inconsistent CSV BMI), with a screening flag at ≥30. BP is flagged at systolic ≥130 or diastolic ≥80; a single reading cannot diagnose hypertension. Total cholesterol ≥200 triggers a discussion of a full lipid panel, not a diagnosis. The source dataset's units remain unverified, and these checks do not change the model's inputs or training limits.

Python callers can pass `advisory_units_confirmed=True` to `predict_patient()`; the CLI equivalent is `--confirm-advisory-units`. Guidance is returned in `condition_guidance`, separate from the model result. Source links and the comparison table are maintained in `heart_guidance.py` (reviewed 23 October 2026).

Actual prediction of multiple diseases would require a new dataset with independently confirmed labels for each condition, relevant clinical measurements, and clinical validation of a multilabel model. The current binary synthetic dataset cannot support that claim.

## Dataset and preprocessing

The original dataset contains **50,000 rows and 21 columns**, with `Heart_Disease` as its binary target: 26,827 class-0 rows and 23,173 class-1 rows. The earlier README attributes it to Kaggle, but the exact URL, author, license, generator, units, and label timing are unverified.

- **BMI:** 92.756% of supplied values differ from the weight/height calculation by more than 1. BMI is excluded; both datasets are preserved.
- **Alcohol:** 20,109 literal `None` values are conservatively treated as `Unknown`. Their intended meaning is undocumented.
- **Blood pressure:** 2,235 inconsistent pairs are masked during training and imputed using training medians. No rows are discarded.
- **Outliers:** original continuous fields have no 1.5-IQR outliers; recalculated BMI in the cleaned file has 55. BMI is excluded from training.
- **Preprocessing:** numerical median imputation, binary mode imputation, categorical one-hot encoding, and numerical scaling for Logistic Regression are fitted inside each pipeline.

Training uses an 80/20 stratified split with seed 42. An additional split within the training partition provides 32,000 fitting rows and 8,000 validation rows. Selection prioritizes validation recall, F1, precision, ROC-AUC, and a smaller F1 gap. The fixed tie order selects Random Forest when it ties with XGBoost. Final evaluation uses the 10,000 test rows.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.923900 | 0.918178 | 0.917584 | 0.917881 | 0.982195 |
| **Random Forest (selected)** | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **1.000000** |
| XGBoost | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

All 50,000 labels match a rule involving age, cholesterol, hypertension, diabetes, and prior heart attack. The perfect tree-model scores reflect this synthetic relationship, not clinical accuracy. The original generator is unknown, and the historical notebook had already explored this test split.

Detailed train/validation/test metrics, classification reports, and comparison plots are in `results/`. The [college report](report/college_report.md) explains the algorithms, metric definitions, false negatives, preprocessing decisions, limitations, and future work.

## Notes

Keep the Python modules, saved pipeline, and metadata together. Joblib files should only be loaded from trusted sources. Generated models are Git-ignored; include `models/` in a local submission or run training after cloning. Exclude `.venv/` and caches from submission archives.

Clinical validity, probability calibration, and subgroup fairness have not been established. Future work requires documented data provenance and independent evaluation. General advisory topics follow [WHO cardiovascular disease guidance](https://www.who.int/en/health-topics/hypertension/cardiovascular-diseases).
