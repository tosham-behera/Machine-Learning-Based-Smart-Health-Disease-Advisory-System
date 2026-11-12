# 1. Title

**Smart Health Advisory System — Heart Disease Prediction Using Machine Learning**

College project report, prepared on 20 October 2026.

**Scope:** educational binary classification using synthetic data, with a separate rule-based general health advisory. This prototype is not a clinically validated diagnostic system or cardiovascular risk calculator.

# 2. Abstract

This project extends an existing exploratory notebook into a reproducible machine learning application. The supplied synthetic dataset contains 50,000 records, 21 columns, and the binary target `Heart_Disease`. Auditing identified an ambiguous alcohol token, inconsistent BMI measurements, invalid blood-pressure pairs, and an exact relationship between five features and every target label. The final workflow preserves the original files, excludes BMI, handles invalid measurements in fitted preprocessing, and uses 19 input features.

Logistic Regression, Random Forest, and XGBoost were compared using common stratified partitions. An inner validation split selected Random Forest through a predeclared ranking. On 10,000 test rows, Logistic Regression achieved accuracy 0.923900, recall 0.917584, F1 0.917881, and ROC-AUC 0.982195. Random Forest and XGBoost each achieved 1.0 on those metrics. Perfect results are consistent with recovery of the observed synthetic label relationship and do not establish medical accuracy.

The selected pipeline is saved with joblib and used by a validated prediction function and Streamlit interface. The interface returns class 0 or 1, an estimated model probability, and separate preventive suggestions. Dataset provenance, label timing, clinical relevance, and probability calibration remain unresolved limitations.

# 3. Introduction

A practical machine learning workflow must inspect data, separate training and evaluation, apply consistent transformations, save the trained system, validate new inputs, and communicate output meaning. Healthcare-themed data requires care because a dataset label can be mistaken for a diagnosis.

This project uses synthetic records to demonstrate these engineering steps. Existing notebook exploration was retained while a standalone workflow was added for a college demonstration. The advisory component covers general tobacco avoidance, activity, diet, and blood-pressure follow-up, consistent with [WHO cardiovascular-disease prevention guidance](https://www.who.int/en/health-topics/hypertension/cardiovascular-diseases). It does not infer treatment from the model score.

# 4. Problem Statement

The initial project had an EDA and training notebook but lacked a saved complete pipeline, reusable prediction function, application, and full reproducibility checks. Some saved outputs conflicted with notebook order. Categorical encoding occurred before splitting, and the requested XGBoost model was absent.

The technical problem is to classify `Heart_Disease` from validated inputs while maintaining consistent preprocessing and transparent evaluation. The communication problem is to explain why performance on undocumented synthetic labels cannot establish diagnosis or future cardiovascular risk.

# 5. Objectives

1. Preserve the notebook and both supplied CSVs.
2. Audit missingness, duplicates, types, classes, measurements, and target relationships.
3. Document BMI use and avoid unjustified deletion of records.
4. Use stratified partitions, fixed seeds, and training-fitted preprocessing.
5. Train the three requested models on common partitions.
6. Compare classification metrics and train/test performance.
7. Select using validation data and save the fitted pipeline.
8. Provide validated prediction and a matching Streamlit form.
9. Separate model output from general advisory.
10. Supply actual verification evidence, run commands, and limitations.

# 6. Dataset Description

The original file is `data/synthetic_heart_disease_dataset.csv`: 50,000 rows, 20 potential input columns, and the binary target `Heart_Disease`. Excluding BMI leaves 19 input features.

| Property | Observed value |
| --- | --- |
| Original dimensions | 50,000 × 21 |
| Target classes | 0 and 1 |
| Class 0 | 26,827 rows, 53.654% |
| Class 1 | 23,173 rows, 46.346% |
| Exact duplicates | 0 |
| Blank missing values when literal strings are preserved | 0 |
| Literal `None` in `Alcohol_Intake` | 20,109 rows, 40.218% |
| Final features | 19 |

The original README attributes the data to Kaggle. Its exact URL, creator, version, license, data dictionary, and generator remain unknown. No specific Kaggle citation or real-patient provenance can be established.

| Exact field | Type / current accepted values | Unit or meaning |
| --- | --- | --- |
| `Age` | Numerical, 30–79 | Years |
| `Gender` | Female, Male, Unknown | Supplied categories |
| `Weight` | Numerical, 50–119 | kg, assumed |
| `Height` | Numerical, 150–199 | cm, assumed |
| `Smoking` | Current, Former, Never, Unknown | Reported status |
| `Alcohol_Intake` | High, Low, Moderate, Unknown | Undocumented intake categories |
| `Physical_Activity` | Active, Moderate, Sedentary, Unknown | Undocumented activity categories |
| `Diet` | Average, Healthy, Unhealthy, Unknown | Undocumented diet categories |
| `Stress_Level` | High, Low, Medium, Unknown | Undocumented stress categories |
| `Hypertension` | 0, 1, unknown | No/Yes |
| `Diabetes` | 0, 1, unknown | No/Yes |
| `Hyperlipidemia` | 0, 1, unknown | No/Yes |
| `Family_History` | 0, 1, unknown | No/Yes; detailed definition unknown |
| `Previous_Heart_Attack` | 0, 1, unknown | Assumed history before prediction |
| `Systolic_BP` | Numerical, 100–179 | mmHg, assumed |
| `Diastolic_BP` | Numerical, 60–119 | mmHg, assumed |
| `Heart_Rate` | Numerical, 60–109 | beats/min, assumed |
| `Blood_Sugar_Fasting` | Numerical, 70–179 | mg/dL, assumed |
| `Cholesterol_Total` | Numerical, 150–299 | mg/dL, assumed |

Units have not been verified against a source dictionary. Bounds are derived from training data and are demonstration limits, not medical reference ranges. Numerical and binary fields also allow an explicit unknown value. Neither BMI nor the target appears as a form input.

# 7. Exploratory Data Analysis

The preserved notebook includes missingness summaries, class and categorical distributions, numerical summaries, box plots, correlations, BMI checks, and feature importance. The notebook is stored in `notebooks/`; the reproducible audit is saved in `results/data_audit.json`.

The target imbalance is modest, so stratification preserves proportions without removing or oversampling rows. Gender entries are 25,110 Female and 24,890 Male. Smoking entries are 30,160 Never, 9,965 Current, and 9,875 Former. These synthetic category frequencies are not population prevalence estimates.

The original continuous numerical fields contain no nonpositive values or values outside their standard 1.5-IQR fences. Recalculated BMI in the existing cleaned file has 55 IQR outliers; BMI is omitted from the final model, and these rows are retained. Binary indicators are categories and are not screened with continuous-variable IQR fences. These checks do not prove plausibility: separately bounded fields can still form inconsistent combinations.

Important findings:

- **Alcohol:** 20,109 literal `None` tokens become missing under default pandas parsing. Without documentation, they could mean abstinence or unknown information, so they are not proven missing measurements.
- **BMI:** assuming kg/cm, `Weight / (Height / 100) ** 2` differs from the supplied BMI by mean absolute 8.534774 and maximum 34.888889. For 46,378 rows (92.756%), the difference exceeds 1. Supplied versus calculated BMI correlation is 0.003405.
- **Blood pressure:** 2,235 rows (4.47%) have systolic pressure ≤ diastolic pressure.
- **Target pattern:** every label equals the “at least two true” rule among `Age > 55`, `Cholesterol_Total > 240`, `Hypertension == 1`, `Diabetes == 1`, and `Previous_Heart_Attack == 1`.

The last observation is an exact property of this file, not confirmation of its original generation procedure. It explains why tree models can reproduce all labels.

# 8. Data Preprocessing

`health_data.py` defines features and deterministic cleaning; `train_model.py` builds each fitted pipeline. The same transformation code is used during training and prediction.

**BMI.** The existing cleaned CSV recalculates BMI to two decimals. This creates arithmetic consistency without verifying weight or height. The final models omit BMI and retain weight and height, avoiding a conflicting or redundant input. Both original CSVs remain unchanged.

**Categorical fields.** Loading preserves literal `None`. Cleaning maps this ambiguous token, empty categorical entries, and missing categories to `Unknown`, retaining the team's conservative choice without claiming abstinence. `OneHotEncoder(handle_unknown="ignore")` learns categories from training only. The public interface accepts schema choices and Unknown while rejecting arbitrary category spellings.

**Numerical and binary fields.** Numerical missing values use training medians; binary missing values use training modes. If a historical BP pair is invalid, both measurements are masked and imputed. Rows are retained, and measurements are not swapped. The public prediction function rejects an explicitly invalid BP pair so it can be corrected.

**Scaling and leakage.** Logistic Regression's numerical branch applies `StandardScaler` after imputation. A `ColumnTransformer` joins numerical, binary, and categorical branches inside `Pipeline`. Medians, modes, encodings, and scaling statistics are learned only from the current fitting partition, consistent with [scikit-learn composition guidance](https://scikit-learn.org/stable/modules/compose.html) and [its advice on avoiding preprocessing leakage](https://scikit-learn.org/1.8/common_pitfalls.html).

All 50,000 rows remain available. Form categories, defaults, and bounds are derived from the final 40,000-row training partition.

# 9. Proposed Methodology

```text
Original CSV → schema/target/type/duplicate checks → exclude BMI
    ↓
Stratified 80/20 split: 40,000 training / 10,000 test
    ↓
Training-only split: 32,000 fitting / 8,000 validation
    ↓
Fit three complete pipelines → compare validation predictions
    ↓
Select model by a predeclared ranking
    ↓
Refit each candidate on all 40,000 training rows
    ↓
Evaluate the same 10,000 test rows; save metrics and plots
    ↓
Save selected pipeline + schema → validated input → prediction
    ↓
Display separate general advisory and limitations
```

Selection maximizes validation recall, then F1, precision, and ROC-AUC, then minimizes the absolute inner-training/validation F1 gap. Exact ties prefer Logistic Regression, Random Forest, then XGBoost. This ranking is fixed before test predictions.

No hyperparameter search or threshold tuning is used. The observed target rule is documented as a limitation, not substituted for the trained model.

# 10. Machine Learning Algorithms

**Logistic Regression** learns a weighted linear combination of inputs and maps it to a binary probability. It is a useful simple baseline. Numerical standardization supports comparable feature scales. Settings: `max_iter=2000`, `random_state=42`.

**Random Forest** aggregates randomized decision trees and can capture nonlinear thresholds and interactions. Settings: `n_estimators=200`, `random_state=42`, `n_jobs=2`.

**XGBoost** builds gradient-boosted decision trees sequentially to improve fitted predictions. The project uses `XGBClassifier` with 200 trees, depth 4, learning rate 0.1, binary logistic objective, log-loss evaluation, histogram tree construction, seed 42, and 2 jobs. The [official Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html) documents this interface.

The preserved notebook's scikit-learn `GradientBoostingClassifier` is historical work and is distinct from XGBoost.

# 11. Experimental Setup

The experiment ran with **Python 3.13.11** and these pinned dependencies:

| Package | Version |
| --- | --- |
| pandas | 3.0.5 |
| NumPy | 2.5.3 |
| scikit-learn | 1.9.1 |
| XGBoost | 3.4.1 |
| Matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| Streamlit | 1.66.0 |
| joblib | 1.6.0 |

Both splits use stratification and seed 42:

| Partition | Rows | Class 0 | Class 1 |
| --- | ---: | ---: | ---: |
| Outer training | 40,000 | 21,462 | 18,538 |
| Inner fitting | 32,000 | 17,170 | 14,830 |
| Validation | 8,000 | 4,292 | 3,708 |
| Test | 10,000 | 5,365 | 4,635 |

Inner fitting and validation together form outer training. Final refitting uses only outer training rows. The standard binary decision threshold is not tuned; metadata records 0.5.

The source SHA-256 is `4981fc17fdb431a5375e6ffa83e891972b14c4b18c2f8cbac9c39df7d65fca7a`. Saved metadata includes source/model hashes, versions, seed, schema, split sizes, and selection rationale. Partition membership can be reproduced from the documented stratified splits and random seed. Aggregate metrics and classification reports are stored in `results/`.

The historical notebook already explored the same seed-42 outer split, and the audit inspected the full dataset. Test rows are separate from current model fitting, but this is not a fresh external blind study.

# 12. Evaluation Metrics

For class 1 as the positive dataset label:

| Metric | Definition | Meaning |
| --- | --- | --- |
| Accuracy | `(TP + TN) / all rows` | Overall correct fraction |
| Precision | `TP / (TP + FP)` | Fraction of predicted positives labeled positive |
| Recall | `TP / (TP + FN)` | Fraction of labeled positives found |
| F1 | `2 × precision × recall / (precision + recall)` | Balance of precision and recall |
| ROC-AUC | Area under true-positive-rate/false-positive-rate curve | Ranking quality across thresholds |
| Confusion matrix | `[[TN, FP], [FN, TP]]` | Counts correct/error categories |

TP and TN are correctly predicted positives and negatives; FP and FN are incorrectly predicted positives and negatives. Classification reports also include class-0 scores, macro/weighted averages, and support.

A false negative can give false reassurance if a classifier is misused in screening, motivating recall-first selection. Precision matters because extra positive predictions can create false alarms, while F1 summarizes the tradeoff. Here, all positives and errors refer to synthetic labels. ROC-AUC and predicted probabilities do not establish calibration or clinically valid risk.

# 13. Results and Model Comparison

These numbers come from executed `results/model_comparison.csv`, `results/classification_reports.json`, and `models/model_metadata.json`, not copied historical notebook outputs.

## Validation used for selection

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Absolute F1 gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.923750 | 0.914393 | 0.921791 | 0.918077 | 0.982626 | 0.001831 |
| Random Forest | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| XGBoost | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |

**Random Forest was selected.** It and XGBoost tied on all selection criteria; the predeclared tie order chose Random Forest. This does not imply that Random Forest is generally better than XGBoost. Test scores did not determine the choice.

## Final test performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.923900 | 0.918178 | 0.917584 | 0.917881 | 0.982195 |
| Random Forest | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| XGBoost | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## Training versus testing

| Model | Train accuracy | Test accuracy | Train recall | Test recall | Train F1 | Test F1 | Train AUC | Test AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.925300 | 0.923900 | 0.919247 | 0.917584 | 0.919396 | 0.917881 | 0.983397 | 0.982195 |
| Random Forest | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| XGBoost | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

Logistic Regression's accuracy gap is 0.001400 and F1 gap approximately 0.001515. Both tree models have zero gaps on these metrics. This shows no large internal gap, not generalization to real clinical records.

## Test confusion counts and class reports

| Model | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 4,986 | 379 | 382 | 4,253 |
| Random Forest | 5,365 | 0 | 0 | 4,635 |
| XGBoost | 5,365 | 0 | 0 | 4,635 |

Logistic Regression misses 382 of 4,635 positive test labels. Its class-0 precision/recall/F1 are 0.928838/0.929357/0.929097, with support 5,365. Its class-1 values are 0.918178/0.917584/0.917881, with support 4,635. Both tree models have precision/recall/F1 of 1.0 for both classes with these same supports. Complete reports for every partition are saved in JSON.

Actual generated figures are `results/model_comparison.png`, `results/confusion_matrices.png`, and `results/roc_curves.png`. They use executed predictions, not illustrative numbers.

# 14. Discussion

The strongest result concerns the dataset: five simple indicators exactly reproduce every label. Trees can learn these threshold interactions, explaining perfect scores more directly than any claim that the project solves diagnosis.

Logistic Regression performs well but represents inputs linearly. Its residual errors are consistent with the threshold interactions in the observed label rule. This is an interpretation of the experiment, not a causal medical conclusion.

The BMI findings show why semantic checks matter. Arithmetic recalculation cannot certify the underlying measurements. Exclusion is transparent and preserves source records. Alcohol token meaning similarly cannot be resolved from pandas missingness alone.

Pipelines prevent fitted test statistics from entering preprocessing. They cannot resolve unknown label timing, possible target proxies, undocumented source data, or prior inspection of the historical test partition. High recall is useful for the stated comparison, while zero synthetic false negatives provides no assurance of disease detection in a real person.

# 15. System Implementation

| File | Responsibility |
| --- | --- |
| `health_data.py` | Loading, features, cleaning, and schema |
| `audit_data.py` | Data-quality statistics and hashes |
| `train_model.py` | Pipelines, validation selection, refitting, evaluation, plots, persistence |
| `predict.py` | Artifact checks, validation, prediction, separate advisory |
| `heart_guidance.py` | Source-linked educational condition topics and transparent matching rules |
| `app.py` | Streamlit form, cached loading, output, disclaimers |
| `data/`, `notebooks/` | Supplied datasets and exploratory notebook |

The selected pipeline is saved as `models/best_heart_disease_model.joblib`. `model_metadata.json` contains the feature schema and experiment details. Loading checks that pipeline feature names/order and class labels agree with metadata.

`predict_patient()` accepts a mapping, creates a DataFrame in training order, invokes the fitted pipeline, and returns a class and class-1 probability. It rejects missing/extra keys, unsupported categories, invalid numerical/binary values, out-of-range measurements, wholly unknown input, and invalid BP pairs. Explicit `None`/JSON `null` is handled through saved preprocessing with a warning.

Streamlit renders its 19 fields from the same schema and caches the model without retraining. Result wording is “Model predicts class 0: No heart disease indicated by the model” or “Model predicts class 1: Heart disease indicated by the model.” No extra medical risk categories are created.

The interface additionally summarizes the unrounded class-1 probability using four illustrative model-score bands: Low (0% to below 25%), Moderate (25% to below 50%), High (50% to below 75%), and Very high (75% to 100%). These equal-width cutoffs are an arbitrary display convention, not calibrated clinical thresholds, patient risk groups, or disease-severity labels. Low does not rule out disease, and Very high does not imply severe disease. This display does not change the binary prediction, training, or advisory rules. The prediction function also returns `model_score_band`, or `None` when no probability is available.

Separate advisory rules address tobacco, activity, diet, and reported hypertension. They cite WHO and do not prescribe medication. No patient database is written.

An additional educational guidance section can display multiple topics: coronary artery disease, blood-pressure health / hypertensive heart disease, heart failure prevention, arrhythmias and cardiomyopathy. Each topic lists the matching reported factors, two practical actions and a medical source. These are transparent topic-selection rules, not learned disease predictions or diagnostic criteria. The comparison table describes possible connections with heart attacks and cardiac arrest without asserting that those outcomes are inevitable. Lifestyle factors are not the only causes of these conditions.

Numeric screening requires explicit confirmation of kg, cm, mmHg and mg/dL. It recalculates adult BMI from entered height and weight and flags BMI ≥30, BP systolic ≥130 or diastolic ≥80, and total cholesterol ≥200. These are prompts for discussion, not diagnoses. BMI is a limited screening measure, a single BP reading does not establish hypertension, and total cholesterol does not replace a full lipid assessment. Source-dataset units remain unverified. Missing inputs are not imputed for this guidance. Topics appear independently of the synthetic model score, and no match does not establish absence of disease or risk.

The exact topic mappings are documented in the README and `heart_guidance.py`, with NHLBI, MedlinePlus, CDC and AHA links reviewed on 23 October 2026. A true multilabel disease model would require confirmed condition-specific labels and suitable clinical measurements, followed by independent clinical validation. The current dataset supplies neither those labels nor evidence of clinical validity.

After creating and activating a Python 3.13 environment:

```text
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The uv alternative, without activating PowerShell:

```powershell
uv sync --locked
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Command-line prediction using the model schema defaults:

```text
python predict.py
```

In PowerShell without activation, replace `python` with `.\.venv\Scripts\python.exe`. Training reloads the saved pipeline and asserts equality of all test predictions before and after serialization. The development workflow verified training, saved-model loading, valid and invalid predictions, and Streamlit operation. Temporary verification files are not part of the final project structure.

The datasets are stored in `data/` and the notebook in `notebooks/`. Dataset contents are unchanged; notebook file paths were adjusted to the new layout. Generated model files are Git-ignored, so a ready-to-run submission archive must include them or instruct the recipient to train first.

# 16. Limitations

1. Synthetic examples do not establish performance on a real clinical population.
2. Exact source, creator, license, and generator are unknown.
3. The deterministic label relationship makes excellent scores possible without clinical generalizability.
4. `Previous_Heart_Attack` is assumed to be prior history; its timing and possible target/proxy leakage are unresolved.
5. BMI and BP inconsistencies reduce confidence; units lack authoritative documentation.
6. Alcohol `None` cannot be confidently interpreted as missingness or abstinence.
7. The historical outer test split was already examined; no external dataset was evaluated.
8. Imputation and Unknown handling are engineering accommodations without clinical validation.
9. Probabilities have no demonstrated calibration or validated cardiovascular-risk meaning.
10. Limited ranges/categories and absent subgroup or prospective studies constrain applicability.
11. General advisory is incomplete and cannot replace professional judgment.

# 17. Future Scope

First establish dataset documentation, licensing, units, label definitions, and measurement timing. A documented, ethically usable dataset with a clear prediction-time definition would strengthen evaluation.

Further experiments could use training-only cross-validation, investigate proxy features, assess calibration and subgroup performance, and report confidence intervals. Independent external and prospective evaluations would be required before medical claims.

Application improvements could include accessibility review and user-tested educational explanations. Any clinical use would require appropriate expert review, privacy controls, and clinical validation. No such deployment is claimed.

# 18. Conclusion

The project now connects a synthetic CSV to reproducible training, validated inputs, a saved pipeline, and Streamlit demonstration. It compares all three requested algorithms, selects Random Forest through validation, and preserves existing work.

Random Forest and XGBoost reproduce all synthetic test labels; Logistic Regression achieves 0.923900 accuracy. The exact five-feature target relationship limits what these results mean. The project demonstrates classification and software integration, while transparent discussion of BMI, alcohol ambiguity, BP inconsistency, and provenance prevents unsupported clinical claims.

# 19. References

Primary documentation consulted on 20 October 2026 follows. Stable web documentation may change; actual installed versions are recorded in Section 11 and metadata.

1. scikit-learn developers. [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html). `Pipeline` and `ColumnTransformer`.
2. scikit-learn developers. [Common pitfalls and recommended practices, version 1.8](https://scikit-learn.org/1.8/common_pitfalls.html). Preprocessing and leakage guidance; the experiment itself used version 1.9.1.
3. XGBoost developers. [Python API reference](https://xgboost.readthedocs.io/en/stable/python/python_api.html). Classifier API and parameters.
4. Streamlit. [App testing](https://docs.streamlit.io/develop/api-reference/app-testing). Streamlit application-testing documentation.
5. World Health Organization. [Cardiovascular diseases](https://www.who.int/en/health-topics/hypertension/cardiovascular-diseases). General preventive advisory topics.
6. Supplied materials: `data/synthetic_heart_disease_dataset.csv`, `data/heart_disease_cleaned.csv`, and `notebooks/Smart_Health_Advisory_Heart_Disease_ML_Project.ipynb`. The historical README says Kaggle, but an exact dataset citation cannot be supplied because its URL, author, license, and generation documentation are unknown.

Numerical findings come from executed local data and model artifacts, not the general documentation references.

# 20. Generative AI Usage Disclosure

OpenAI Codex assistance was used to inspect the supplied project, identify inconsistencies, implement preprocessing and application code, create automated checks, interpret executed results, and draft documentation. Existing teammate EDA/cleaning and the supplied notebook were preserved.

Metrics and data counts come from executed Python code and saved artifacts, not invented model outputs. AI-generated code and explanations do not constitute medical validation. The submitting student should review the implementation, understand its assumptions, confirm verification evidence, and disclose this assistance according to the college's academic-integrity requirements.
