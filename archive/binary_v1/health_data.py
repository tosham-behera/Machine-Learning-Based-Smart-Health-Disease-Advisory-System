"""Shared, deterministic data preparation. No statistics are fitted here."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TARGET = "Heart_Disease"
CATEGORICAL_COLUMNS = [
    "Gender", "Smoking", "Alcohol_Intake", "Physical_Activity", "Diet", "Stress_Level",
]
BINARY_COLUMNS = [
    "Hypertension", "Diabetes", "Hyperlipidemia", "Family_History",
    "Previous_Heart_Attack",
]
NUMERIC_COLUMNS = [
    "Age", "Weight", "Height", "Systolic_BP", "Diastolic_BP", "Heart_Rate",
    "Blood_Sugar_Fasting", "Cholesterol_Total",
]
ORIGINAL_COLUMNS = [
    "Age", "Gender", "Weight", "Height", "BMI", "Smoking", "Alcohol_Intake",
    "Physical_Activity", "Diet", "Stress_Level", "Hypertension", "Diabetes",
    "Hyperlipidemia", "Family_History", "Previous_Heart_Attack", "Systolic_BP",
    "Diastolic_BP", "Heart_Rate", "Blood_Sugar_Fasting", "Cholesterol_Total", TARGET,
]
# The supplied BMI conflicts with weight and height. Keep the CSV unchanged and
# omit this unreliable, redundant feature from both training and the UI.
FEATURE_COLUMNS = [c for c in ORIGINAL_COLUMNS if c not in ("BMI", TARGET)]


def read_dataset(path):
    """Preserve literal 'None'; pandas otherwise silently treats it as NA."""
    frame = pd.read_csv(path, keep_default_na=False, na_values=[""])
    if list(frame.columns) != ORIGINAL_COLUMNS:
        raise ValueError(f"Expected these dataset columns in order: {ORIGINAL_COLUMNS}")
    for column in NUMERIC_COLUMNS + BINARY_COLUMNS + ["BMI", TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if np.isinf(frame[column].to_numpy(dtype=float)).any():
            raise ValueError(f"{column} contains infinite values.")
    if frame[TARGET].isna().any() or set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("Heart_Disease must contain both classes 0 and 1, without missing labels.")
    for column in BINARY_COLUMNS:
        if not set(frame[column].dropna().unique()).issubset({0, 1}):
            raise ValueError(f"{column} must contain 0, 1, or missing values.")
    if frame.duplicated().any():
        raise ValueError("Duplicate rows need review before a leakage-safe split.")
    return frame


def clean_features(frame):
    """Apply fixed rules identically during fitting and new predictions."""
    cleaned = frame.copy()
    for column in CATEGORICAL_COLUMNS:
        # 'None' is ambiguous in this undocumented source: retain the team's
        # conservative Unknown convention without claiming it means abstinence.
        cleaned[column] = (
            cleaned[column].astype("object").replace({"None": np.nan, "": np.nan})
            .fillna("Unknown")
        )
    for column in NUMERIC_COLUMNS + BINARY_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise").astype(float)
    for column in NUMERIC_COLUMNS:
        cleaned.loc[cleaned[column] <= 0, column] = np.nan
    # Preserve rows, but do not learn from impossible paired BP measurements.
    invalid_bp = cleaned["Systolic_BP"] <= cleaned["Diastolic_BP"]
    cleaned.loc[invalid_bp, ["Systolic_BP", "Diastolic_BP"]] = np.nan
    return cleaned


def build_input_schema(training_features):
    """Derive UI fields and demo limits from the training data, never the test."""
    cleaned = clean_features(training_features)
    units = {
        "Age": "years", "Weight": "kg (assumed)", "Height": "cm (assumed)",
        "Systolic_BP": "mmHg (assumed)", "Diastolic_BP": "mmHg (assumed)",
        "Heart_Rate": "beats/min (assumed)",
        "Blood_Sugar_Fasting": "mg/dL (assumed)",
        "Cholesterol_Total": "mg/dL (assumed)",
    }
    schema = {}
    for column in FEATURE_COLUMNS:
        values = cleaned[column]
        label = column.replace("_", " ")
        if column in units:
            label += f" ({units[column]})"
        if column in CATEGORICAL_COLUMNS:
            categories = sorted(set(values.dropna()) | {"Unknown"})
            schema[column] = {
                "kind": "categorical", "label": label, "categories": categories,
                "default": str(values.mode().iloc[0]),
            }
        elif column in BINARY_COLUMNS:
            schema[column] = {
                "kind": "binary", "label": label, "min": 0, "max": 1,
                "default": int(values.mode().iloc[0]),
            }
        else:
            schema[column] = {
                "kind": "numeric", "label": label, "min": float(values.min()),
                "max": float(values.max()), "default": float(values.median()),
                "step": 1.0,
            }
    return schema
