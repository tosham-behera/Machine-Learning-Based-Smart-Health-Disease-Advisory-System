# Smart Health Advisory System: Synthetic Multilabel Classification

## Abstract

This project demonstrates classification of five overlapping fictional heart-condition labels. A reproducible generator creates 20,000 records with 24 features and five targets. Logistic Regression, Random Forest and XGBoost are compared using disjoint fitting, validation and test partitions. Streamlit presents each synthetic-label score alongside clearly separate educational lifestyle guidance.

This is a simulation of an ML workflow, not a clinical diagnostic system. All labels are generated from invented equations. Successful classification measures agreement with those equations and their random variation, not the ability to identify diseases in real people.

## Objective and scope

The previous binary dataset only labeled Heart_Disease as 0 or 1. It could not support learned condition-specific predictions. Version 2 replaces the active data and pipeline with five targets: CAD, Hypertensive_Heart_Disease, Heart_Failure, Arrhythmia and Cardiomyopathy. Each target is independently binary; several can be present at once. A row with five zeros is a synthetic no-label example, not a medically healthy person.

The original data, scripts, notebook, results and model are preserved together in archive/binary_v1. Old performance metrics and unknown provenance do not describe the new generated dataset.

## Data provenance and reproducibility

The data are generated locally by generate_data.py, version 2.0.0, using NumPy's seeded generator (seed 42). No real patient records, external dataset, or clinically adjudicated labels are used. The dataset JSON card records the seed, generator hash, CSV hash, schema, ranges and counts. All coefficients, marginal distributions and prevalence are arbitrary educational choices.

The generator creates features before labels. Age spans 18–90. Lifestyle categories and binary history fields are sampled, and blood pressure, glucose and cholesterol have deliberately correlated variation. Latent random variables introduce overlapping variations in cardiac measurements. These latent variables are not exported as features or targets. Five logistic equations are evaluated on the recorded features, and each label is sampled as a Bernoulli draw. Noise means even a correct reconstruction of the generator cannot identify every sampled label.

No target column is used to construct an input feature. This prevents direct target-copy leakage, but it does not make the dataset representative of clinical populations. The simplified generator also omits many possible causes and can create combinations that do not correspond to real clinical cases.

## Feature schema and labels

Eleven numerical features: age, weight, height, systolic and diastolic BP, heart rate, fasting blood sugar, total cholesterol, exercise ST depression, ejection fraction, and LV wall thickness.

Six binary features: hypertension history, diabetes history, hyperlipidemia history, general family history, previous heart attack, and exertional chest pain.

Seven categorical features: gender, smoking, alcohol intake, physical activity, diet, stress level, and ECG rhythm. ECG is deliberately simplified to Regular/Irregular; it is not an arrhythmia diagnosis. Family_History is broad and does not represent genetic confirmation of any condition. Physical activity, diet, stress and alcohol categories are simplified illustrative categories, not quantified clinical measures.

Units are explicitly defined by the generator: years, kg, cm, mmHg, beats/min, mg/dL, mm and %. Input bounds are generator support, not medical reference ranges. BMI is not a model feature; optional lifestyle guidance calculates it from weight and height. Alcohol None means no alcohol, unlike the ambiguous literal in the archived dataset.

All 20,000 records are complete and contain no duplicate feature rows or contradictory BP pairs. The target counts are:

- Coronary artery disease: 6,512 positive fictional labels
- Hypertensive heart disease: 6,460 positive fictional labels
- Heart failure: 4,932 positive fictional labels
- Heart arrhythmia: 4,133 positive fictional labels
- Cardiomyopathy: 5,193 positive fictional labels

There are 7,967 multi-label rows and 5,187 rows without positive labels. These are simulated proportions, not estimates of prevalence.

## Preprocessing and splitting

Numerical fields receive training-median imputation. Binary fields use training-mode imputation. Categorical missingness is explicit Unknown, with one-hot encoding and unseen-category handling. Logistic Regression also standardizes numerical inputs. Every preprocessing step is fitted inside the estimator pipeline, using only its fitting partition.

Random seed 42 produces 16,000 train and 4,000 test records. A second split of the training partition yields 12,800 fit and 3,200 validation records. These random splits are not stratified; all five targets are checked for both classes in each split. Split indices do not overlap, and feature duplicates are rejected before splitting. Split sizes and positive counts are saved in model_metadata.json.

## Algorithms and model selection

MultiOutputClassifier fits one binary estimator for each target. Logistic Regression uses max_iter=2000 and scaled numerical inputs. Random Forest uses 100 trees, maximum depth 12, and at least five samples per leaf. XGBoost uses 100 trees, depth 3, learning rate 0.07, histogram tree construction and binary logistic loss.

The ranking is fixed before testing: validation macro F1, then macro average precision, then stable model order for an exact tie. Logistic Regression was selected. All candidates are refitted on the same 16,000 training records, then evaluated on the 4,000 held-out records. Test results do not select the winner. All five decision thresholds remain 0.5. There is no test-set threshold tuning or clinical calibration.

## Evaluation

Macro precision, recall and F1 give each condition equal weight. Micro F1 aggregates decisions across labels. Subset accuracy requires all five labels to be correct for a record. Hamming loss is the fraction of individual label decisions that are wrong. ROC-AUC and average precision summarize ranking, while Brier scores report squared probability error against the fictional labels. None of these establishes clinical validity.

For the selected model, test macro F1 is 0.6665, macro recall 0.6025, macro ROC-AUC 0.8547, subset accuracy 42.93%, and Hamming loss 0.1599. A constant training-prevalence baseline has zero macro F1 at the fixed threshold and 26.27% subset accuracy. The random label process makes perfect scores neither expected nor required.

| Synthetic condition | Precision | Recall | F1 | ROC-AUC | Test positive support |
|---|---:|---:|---:|---:|---:|
| Coronary artery disease | 0.7233 | 0.5507 | 0.6253 | 0.8304 | 1291 |
| Hypertensive heart disease | 0.8101 | 0.6776 | 0.7379 | 0.9003 | 1284 |
| Heart failure | 0.7316 | 0.5902 | 0.6533 | 0.8759 | 942 |
| Heart arrhythmia | 0.7343 | 0.6075 | 0.6649 | 0.7909 | 828 |
| Cardiomyopathy | 0.7317 | 0.5867 | 0.6512 | 0.8760 | 1004 |

The saved classification reports include a 2x2 confusion matrix for every model, split and target. Rows are synthetic truth; columns are predicted labels in order 0, 1. False negatives remain substantial, making real-world reassurance particularly inappropriate. Full model comparisons and plots are in results/.

## Missing clinical inputs

A separate stress test replaces exercise ST depression, ejection fraction and LV wall thickness with missing values and ECG rhythm with Unknown on the same test records. Selected-model macro F1 falls from 0.6665 to 0.2221, and macro ROC-AUC falls from 0.8547 to 0.6457. This is descriptive evaluation of one missingness pattern, not selection or tuning, and not a substitute for testing other missingness mechanisms.

The form allows unknown measurements with a prominent warning. Enter-values mode starts blank; clinical values must not be guessed. Fictional examples A and B provide complete illustrative profiles for exploring the app. Numerical lifestyle rules use only supplied measurements after unit confirmation, never model-imputed values.

## Application and persistence

The saved pipeline is models/heart_condition_model.joblib. Metadata identifies the versioned schema, ordered features and targets, fixed thresholds, splits, hashes and package versions. Loading rejects incompatible target order, stale schemas and mismatched model checksums. Serialization is checked by comparing all test scores exactly before and after reload.

predict_patient returns five condition results and a list of synthetic labels above threshold. It rejects missing/extra keys, non-finite numbers, invalid categories, values outside generator ranges, wholly unknown records and contradictory BP pairs. Explicit null is allowed for missing fields. Model probabilities are extracted using each estimator's actual class order.

Streamlit caches the model using artifact modification stamps, so retraining changes the cache key. It shows all five scores, their binary threshold decisions and illustrative Low/Moderate/High/Very high bands. Scores are independent outputs and need not sum to 100%. The bands are a display convention, not severity or medical risk.

## Lifestyle guidance and condition background

Lifestyle advice remains separate from model output. Reported smoking, BP history, diabetes, lipid history, activity and other factors can select multiple educational topics, each with two practical actions and a source link. These topic rules are not diagnoses or an explanation of the ML model's internals. A comparison table describes possible connections with heart attacks and cardiac arrest without claiming inevitability.

BMI, BP and cholesterol screening flags follow their linked general references, with stated limitations. The numerical unit checkbox affects this guidance only. No medications are prescribed, no treatment is selected from a model label, and no submitted patient database is written.

## Limitations and future work

The generator contains invented assumptions, simplified categories, broad family history and coarse cardiac test fields. It is not a validated physiological simulator. Data diversity, demographic coverage, unit clarity and probabilistic noise do not create clinical validity. There are no patient-confirmed diagnoses, temporal outcomes, independent hospital data, subgroup validation, clinical calibration or diagnostic performance claims.

This is classification of fictional present-condition labels, not prediction of future heart attacks, cardiac arrest or survival. A positive synthetic label is not a diagnosis; a negative one is not reassurance. Actual clinical work would require suitable real data with confirmed overlapping diagnoses, well-defined measurement timing, ethical data handling, patient-level external validation, and expert oversight.

## Reproduction and references

Run generate_data.py, audit_data.py, train_model.py, then streamlit run app.py. The notebook explores the active data and saved results without automatically retraining. The original version is archived separately and is not imported by the active app.

- [scikit-learn multioutput classification](https://scikit-learn.org/stable/modules/multiclass.html)
- [NHLBI heart failure diagnosis](https://www.nhlbi.nih.gov/health/heart-failure/diagnosis)
- [NHLBI cardiomyopathy diagnosis](https://www.nhlbi.nih.gov/health/cardiomyopathy/diagnosis)
- [NHLBI arrhythmia diagnosis](https://www.nhlbi.nih.gov/health/arrhythmias/diagnosis)
- [NHLBI coronary heart disease diagnosis](https://www.nhlbi.nih.gov/health/coronary-heart-disease/diagnosis)
- Lifestyle and comparison-table references are linked in heart_guidance.py and the app. They support educational background, not the generator's coefficients or predictions.
