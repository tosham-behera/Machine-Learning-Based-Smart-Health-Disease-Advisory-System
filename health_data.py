"""Versioned schema and preprocessing for the synthetic multilabel demonstration."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
DATA_PATH = DATA_DIR / 'synthetic_heart_conditions.csv'
SCHEMA_VERSION = 'synthetic-multilabel-v2'
TARGET_NAMES = {
    'CAD': 'Coronary artery disease',
    'Hypertensive_Heart_Disease': 'Hypertensive heart disease',
    'Heart_Failure': 'Heart failure',
    'Arrhythmia': 'Heart arrhythmia',
    'Cardiomyopathy': 'Cardiomyopathy',
}
TARGET_COLUMNS = list(TARGET_NAMES)
# Generator bounds, not medical reference intervals or clinical coverage.
NUMERIC_SPECS = {
    'Age': (18, 90, 45, 1, 'years'),
    'Weight': (40, 180, 70, 1, 'kg'),
    'Height': (140, 205, 170, 1, 'cm'),
    'Systolic_BP': (90, 179, 115, 1, 'mmHg'),
    'Diastolic_BP': (50, 119, 75, 1, 'mmHg'),
    'Heart_Rate': (45, 140, 75, 1, 'beats/min'),
    'Blood_Sugar_Fasting': (65, 250, 90, 1, 'mg/dL'),
    'Cholesterol_Total': (100, 350, 180, 1, 'mg/dL'),
    'Exercise_ST_Depression': (0, 5, 0, 0.1, 'mm'),
    'Ejection_Fraction': (20, 75, 60, 1, '%'),
    'LV_Wall_Thickness': (6, 20, 9, 0.1, 'mm'),
}
NUMERIC_COLUMNS = list(NUMERIC_SPECS)
BINARY_COLUMNS = ['Hypertension', 'Diabetes', 'Hyperlipidemia', 'Family_History',
                  'Previous_Heart_Attack', 'Exertional_Chest_Pain']
CATEGORY_CHOICES = {
    'Gender': ['Female', 'Male', 'Unknown'],
    'Smoking': ['Never', 'Former', 'Current', 'Unknown'],
    'Alcohol_Intake': ['None', 'Low', 'Moderate', 'High', 'Unknown'],
    'Physical_Activity': ['Active', 'Moderate', 'Sedentary', 'Unknown'],
    'Diet': ['Healthy', 'Average', 'Unhealthy', 'Unknown'],
    'Stress_Level': ['Low', 'Medium', 'High', 'Unknown'],
    'ECG_Rhythm': ['Regular', 'Irregular', 'Unknown'],
}
CATEGORICAL_COLUMNS = list(CATEGORY_CHOICES)
FEATURE_COLUMNS = NUMERIC_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS
CLINICAL_FIELDS = {'Exercise_ST_Depression', 'Ejection_Fraction', 'LV_Wall_Thickness', 'ECG_Rhythm'}


def build_input_schema(training_features=None):
    """Use documented generator support, not test-set extrema."""
    schema = {}
    for name, (lower, upper, default, step, unit) in NUMERIC_SPECS.items():
        schema[name] = {'kind': 'numeric', 'label': f"{name.replace('_', ' ')} ({unit})",
                        'min': lower, 'max': upper, 'default': float(default), 'step': step}
    for name in BINARY_COLUMNS:
        schema[name] = {'kind': 'binary', 'label': name.replace('_', ' '), 'default': 0}
    for name, choices in CATEGORY_CHOICES.items():
        schema[name] = {'kind': 'categorical', 'label': name.replace('_', ' '),
                        'categories': choices, 'default': 'Unknown' if name == 'ECG_Rhythm' else choices[0]}
    return {name: schema[name] for name in FEATURE_COLUMNS}


def clean_features(frame):
    cleaned = frame.copy()
    for name in NUMERIC_COLUMNS + BINARY_COLUMNS:
        cleaned[name] = pd.to_numeric(cleaned[name], errors='raise').astype(float)
    for name in CATEGORICAL_COLUMNS:
        cleaned[name] = cleaned[name].astype('object').fillna('Unknown')
    return cleaned


def example_inputs(profile="A"):
    """Fixed fictional examples, not recommended measurements or patient records."""
    values = {name: spec['default'] for name, spec in build_input_schema().items()}
    values['ECG_Rhythm'] = 'Regular'
    if profile == 'B':
        values.update(Age=65.0, Weight=100.0, Height=170.0, Systolic_BP=155.0,
                      Diastolic_BP=95.0, Cholesterol_Total=280.0, Hypertension=1,
                      Diabetes=1, Hyperlipidemia=1, Previous_Heart_Attack=1,
                      Exercise_ST_Depression=3.0, Ejection_Fraction=32.0,
                      LV_Wall_Thickness=16.0, ECG_Rhythm='Irregular',
                      Exertional_Chest_Pain=1, Smoking='Current', Alcohol_Intake='High',
                      Physical_Activity='Sedentary', Diet='Unhealthy')
    elif profile != 'A':
        raise ValueError('Unknown fictional example.')
    return values


def read_dataset(path=DATA_PATH):
    frame = pd.read_csv(path, keep_default_na=False, na_values=[''])
    if list(frame.columns) != FEATURE_COLUMNS + TARGET_COLUMNS:
        raise ValueError('Dataset columns do not match the version-2 feature/target schema.')
    if frame.isna().any().any():
        raise ValueError('Generated training records must be complete.')
    for name, (lower, upper, *_) in NUMERIC_SPECS.items():
        frame[name] = pd.to_numeric(frame[name], errors='raise')
        if not np.isfinite(frame[name]).all() or not frame[name].between(lower, upper).all():
            raise ValueError(f'Invalid generated range for {name}.')
    for name in BINARY_COLUMNS + TARGET_COLUMNS:
        if not set(frame[name].unique()) <= {0, 1}:
            raise ValueError(f'{name} must be binary.')
    for name in TARGET_COLUMNS:
        if set(frame[name].unique()) != {0, 1}:
            raise ValueError(f'Both classes are needed for {name}.')
    for name, choices in CATEGORY_CHOICES.items():
        if not frame[name].isin(choices).all():
            raise ValueError(f'Unsupported category in {name}.')
    if (frame.Systolic_BP <= frame.Diastolic_BP).any():
        raise ValueError('Systolic BP must exceed diastolic BP.')
    if frame[FEATURE_COLUMNS].duplicated().any():
        raise ValueError('Duplicate feature rows could cross split boundaries.')
    return frame


def positive_probabilities(model, features):
    """Respect each estimator's class order when collecting positive scores."""
    outputs = model.predict_proba(features)
    classes = model.named_steps['classifier'].classes_
    if len(outputs) != len(TARGET_COLUMNS) or len(classes) != len(TARGET_COLUMNS):
        raise ValueError('Model output count does not match the five targets.')
    if any(set(values) != {0, 1} for values in classes):
        raise ValueError('Each target must have binary classes 0 and 1.')
    scores = np.column_stack([output[:, list(values).index(1)]
                              for output, values in zip(outputs, classes)])
    if scores.shape != (len(features), len(TARGET_COLUMNS)) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Model returned invalid condition scores.')
    return scores
