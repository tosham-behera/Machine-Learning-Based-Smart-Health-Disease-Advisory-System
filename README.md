# Smart Health Advisory System

An educational **multilabel classifier trained entirely on locally generated fictional data**. It predicts five synthetic labels independently: coronary artery disease, hypertensive heart disease, heart failure, arrhythmia and cardiomyopathy. Several labels, or none, can be positive for the same example.

**This is not a diagnostic application. Synthetic labels are invented, not clinically confirmed. Scores and test metrics do not establish real-world medical accuracy, disease severity, or risk of heart attack or cardiac arrest.**

## Run in VS Code

Open the project folder and run this in its terminal (works in Git Bash or PowerShell):

```text
./.venv/Scripts/python.exe -m streamlit run app.py
```

When the virtual environment is activated, the short command is `streamlit run app.py`. Open the local URL printed in the terminal. Stop with `Ctrl+C`.

For a new installation, use Python 3.13 and install `requirements.txt`, or use `uv sync --locked`. The existing local environment and dependencies are unchanged.

## Use the demonstration

1. Choose **Fictional example A**, **Fictional example B**, or **Enter values**. Presets are made-up examples, not recommended measurements.
2. Select **Predict synthetic labels**. All five condition scores appear, each with a fixed 50% positive-label threshold.
3. Multiple positive labels can appear together. A score below the threshold does not rule out disease. Scores do not sum to 100%.
4. Lifestyle guidance remains separate: it uses reported factors, explains its matches, and offers two practical steps per topic.

Clinical fields include exercise ST depression (mm), ejection fraction (%), left-ventricular wall thickness (mm), and an ECG rhythm category. They must come from test reports for actual measurements, not guesses. Blank numbers and Unknown categories are accepted with explicit warnings; imputation does not replace testing. Enter-values mode starts blank. The synthetic age range is 18–90; range support does not imply clinical validation for those ages.

Illustrative score bands: **Low** below 25%, **Moderate** 25% to below 50%, **High** 50% to below 75%, **Very high** 75–100%. These are arbitrary display bands, not medical-risk or severity categories.

## Reproduce the dataset and model

From an activated environment:

```text
python generate_data.py
python audit_data.py
python train_model.py
python predict.py --example A
python predict.py --example B --confirm-advisory-units
```

The generator defaults to 20,000 rows and seed 42. Its output is `data/synthetic_heart_conditions.csv` with a companion JSON data card documenting units, bounds, seed, label counts and hashes. Run generation before retraining if changing the generator. Never mix an old model with new metadata.

For custom input: `python predict.py --input path/to/example.json`. Include all 24 fields in `models/model_metadata.json`; explicit JSON `null` means unknown. All five predictions are returned under `conditions`. `predicted_conditions` contains only synthetic labels whose scores reach 0.5. The previous single `predicted_class`/`estimated_probability` API is replaced. Numeric lifestyle screening is enabled with `--confirm-advisory-units` or `advisory_units_confirmed=True` in Python.

## Dataset design

- 24 input features plus 5 binary target columns; no target is used as a feature.
- Feature distributions, prevalence and coefficients are arbitrary teaching assumptions, not clinical estimates.
- Features are generated first. Five logistic equations then produce probabilities; Bernoulli sampling produces noisy labels. The generator source contains the complete equations.
- BMI is not a stored input. Lifestyle guidance recalculates it from confirmed kg/cm values. `None` in Alcohol Intake explicitly means no alcohol in this version; `Unknown` means missing information.
- The 20,000 generated examples include **7,967 with multiple labels** and **5,187 with no positive labels**.
- No diagnoses are inferred from the previous binary dataset; it is preserved in the archive.

## Training and results

Preprocessing is fitted inside each pipeline. Logistic Regression, Random Forest and XGBoost each fit five binary estimators with `MultiOutputClassifier`. A seeded random split allocates 12,800 fitting rows, 3,200 validation rows and 4,000 test rows. Validation macro F1, then macro average precision, selects the model; it is refitted on the 16,000 training rows. Splits are disjoint, not stratified, and checked for both classes of every target. Thresholds are fixed at 0.5, not tuned on test data.

**Logistic Regression was selected.** Synthetic held-out results:

| Model | Macro F1 | Macro recall | Macro ROC-AUC | All five labels correct |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6665 | 0.6025 | 0.8547 | 42.93% |
| Random Forest | 0.6568 | 0.5805 | 0.8466 | 42.90% |
| XGBoost | 0.6658 | 0.5980 | 0.8563 | 43.03% |
| Training-prevalence baseline | 0.0000 | 0.0000 | 0.5000 | 26.27% |

When the four clinical test fields are hidden, the selected model's synthetic test macro F1 falls to **0.2221**. This stress test highlights how little the lifestyle-only inputs establish in this simulation. It is not clinical validation or a complete missing-data evaluation. There is no single clinical "accuracy" claim.

## Project layout

```text
app.py                 Streamlit UI
predict.py             Validated multilabel predictions and separate advice
health_data.py         Versioned feature/target schema and preprocessing
generate_data.py       Reproducible fictional-data generator
train_model.py         Model comparison, selection and evaluation
audit_data.py          Data integrity and label distribution checks
heart_guidance.py      Source-linked educational guidance
data/                  Active generated CSV and data card
models/                Selected pipeline and matching metadata
results/               Current metrics, audits and plots
notebooks/             Updated exploration and prediction notebook
report/                Current college report
archive/binary_v1/     Preserved original application, data, model and report
```

The original notebook directory was open during migration, so its contents were copied into the archive before the active notebook was updated. The archived version can be run from inside its folder with the same environment; it is not imported by the new application. Exclude the archive and `.venv` from a minimal submission. Model binaries are Git-ignored: train after cloning or include the `models` folder in a local handover. Only load trusted joblib artifacts.

The notebook reads the active dataset and saved results; it does not silently train or regenerate them. The [college report](report/college_report.md) documents design, metrics and limitations. Actual clinical use would require appropriately sourced patient data with confirmed labels, relevant clinical measurements, independent evaluation, calibration and expert review. Adding disease names to synthetic labels does not supply that evidence.

## Workflow and evaluation pages

The app opens on **How this project works**, covering the workflow, dataset generation, machine-learning concepts and evaluation metrics. The sidebar also provides **Prediction** and **Model performance**. Source code is maintained in project files and is not displayed in the interface.

Run `python evaluate_project.py` to reproduce input checks, local prediction timing and three separate 5,000-row generation experiments (seeds 101, 202, 303). The fixed Logistic Regression model is trained on 80% of each generated dataset and evaluated on the remaining 20%, with complete versus hidden clinical inputs. Mean macro F1 is 0.6550 versus 0.2081; standard deviations are 0.0105 and 0.0151. These are same-generator robustness results, not clinical external validation. Results are saved in `results/pbl_verification.json` and `results/robustness_experiments.csv`.
