"""Validate one example and make a prediction with the saved fitted pipeline.

Run ``python predict.py`` for fictional example A, or ``--example B`` for B.
Run ``python predict.py --input example.json`` to read your own JSON object.
Only load model artifacts you trust: joblib files contain executable Python data.
"""

import argparse
import json
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from heart_guidance import condition_guidance
from health_data import (
    SCHEMA_VERSION, FEATURE_COLUMNS, TARGET_COLUMNS, TARGET_NAMES, CLINICAL_FIELDS,
    positive_probabilities, example_inputs,
)


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "heart_condition_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
ADVISORY_SOURCE = (
    "https://www.who.int/en/health-topics/hypertension/cardiovascular-diseases"
)
DISCLAIMER = (
    "Trained on synthetic data. Percentages are model estimates, not clinically "
    "validated disease risks or diagnoses; low scores do not rule out disease. "
    "Lifestyle suggestions are general guidance based on reported factors."
)

# Illustrative display bands only; these cutoffs have no clinical validation.
MODEL_SCORE_BANDS = (
    (0.25, "Low", "0% to below 25%"),
    (0.50, "Moderate", "25% to below 50%"),
    (0.75, "High", "50% to below 75%"),
    (1.00, "Very high", "75% to 100%"),
)


def model_score_band(probability):
    """Summarize an unrounded class-1 score, never disease risk or severity."""
    if probability is None:
        return None
    if (
        isinstance(probability, (bool, np.bool_))
        or not isinstance(probability, Real)
        or not np.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("Model probability must be a finite number from 0 to 1.")
    for upper, label, interval in MODEL_SCORE_BANDS:
        if probability < upper or upper == 1:
            return {
                "label": label,
                "range": interval,
                "message": (
                    f"The model's score for the synthetic heart-disease label "
                    f"falls in the {label.lower()} band ({interval})."
                ),
            }


def _validate_artifacts(model, metadata):
    """Prevent a stale or mismatched schema from producing misleading results."""
    if not isinstance(metadata, dict):
        raise ValueError("Model metadata must be a JSON object.")
    if (metadata.get("schema_version") != SCHEMA_VERSION
            or getattr(model, "synthetic_schema_version_", None) != SCHEMA_VERSION):
        raise ValueError("An incompatible model was found. Run python train_model.py.")
    feature_names = metadata.get("feature_names")
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or not all(isinstance(name, str) for name in feature_names)
        or len(set(feature_names)) != len(feature_names)
    ):
        raise ValueError("Model metadata must contain unique feature_names.")
    schema = metadata.get("input_schema")
    if not isinstance(schema, dict) or set(schema) != set(feature_names):
        raise ValueError("The input schema must match the model's feature names.")
    trained_features = getattr(model, "feature_names_in_", None)
    if trained_features is None or list(trained_features) != feature_names:
        raise ValueError(
            "Saved model and input schema have different features or feature order. "
            "Run python train_model.py to recreate matching artifacts."
        )
    if feature_names != FEATURE_COLUMNS:
        raise ValueError("The saved feature order does not match this application.")
    if (metadata.get("target_columns") != TARGET_COLUMNS
            or getattr(model, "target_columns_", None) != TARGET_COLUMNS):
        raise ValueError("Model and metadata must share the same five target columns in order.")
    if metadata.get("target_names") != TARGET_NAMES:
        raise ValueError("Target display names do not match this application.")
    thresholds = metadata.get("thresholds", {})
    if thresholds != {name: 0.5 for name in TARGET_COLUMNS}:
        raise ValueError("Expected the documented 0.5 synthetic-label thresholds.")
    for name in feature_names:
        specification = schema[name]
        if not isinstance(specification, dict):
            raise ValueError(f"Invalid input schema for {name}.")
        kind = specification.get("kind")
        if kind == "numeric":
            lower = specification.get("min")
            upper = specification.get("max")
            if (
                not isinstance(lower, Real)
                or not isinstance(upper, Real)
                or not np.isfinite([lower, upper]).all()
                or lower > upper
            ):
                raise ValueError(f"Invalid numerical bounds for {name}.")
        elif kind == "categorical":
            categories = specification.get("categories")
            if (
                not isinstance(categories, list)
                or not categories
                or not all(isinstance(category, str) for category in categories)
                or "Unknown" not in categories
            ):
                raise ValueError(f"Invalid categorical choices for {name}.")
        elif kind != "binary":
            raise ValueError(f"Unsupported input kind for {name}: {kind!r}.")


def load_artifacts(model_path=None, metadata_path=None):
    """Load the fitted pipeline and its schema using script-relative defaults."""
    model_path = Path(model_path) if model_path is not None else MODEL_PATH
    metadata_path = Path(metadata_path) if metadata_path is not None else METADATA_PATH
    for path in (model_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required artifact is missing: {path.name}. "
                "Run python train_model.py from the project directory first."
            )
    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    model = joblib.load(model_path)
    _validate_artifacts(model, metadata)
    from audit_data import file_sha256
    if metadata.get("model_sha256") != file_sha256(model_path):
        raise ValueError("Model file and metadata checksum do not match. Retrain the model.")
    return model, metadata


def validate_patient(patient, metadata):
    """Require every named field, while allowing explicit None for unknown data."""
    if not isinstance(patient, Mapping):
        raise ValueError("Patient information must be a dictionary or JSON object.")
    feature_names = metadata["feature_names"]
    missing_keys = [name for name in feature_names if name not in patient]
    extra_keys = [str(name) for name in patient if name not in feature_names]
    if missing_keys:
        raise ValueError("Missing required fields: " + ", ".join(missing_keys))
    if extra_keys:
        raise ValueError("Unexpected fields: " + ", ".join(extra_keys))

    values = {}
    missing_values = []
    for name in feature_names:
        specification = metadata["input_schema"][name]
        value = patient[name]
        kind = specification["kind"]
        if value is None:
            # The pipeline imputes numerical data; categorical missingness is explicit.
            values[name] = "Unknown" if kind == "categorical" else np.nan
            missing_values.append(name)
            continue
        if kind in {"numeric", "binary"}:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError(
                    f"{name} must be a finite number, or null for a missing value."
                )
            if kind == "binary":
                if value not in (0, 1):
                    raise ValueError(f"{name} must be 0, 1, or null.")
                values[name] = int(value)
            else:
                lower, upper = specification["min"], specification["max"]
                if not lower <= value <= upper:
                    raise ValueError(
                        f"{name} must be between {lower:g} and {upper:g}. "
                        "These are synthetic training-data limits, "
                        "not medical reference ranges."
                    )
                values[name] = float(value)
        else:
            if not isinstance(value, str) or value not in specification["categories"]:
                choices = ", ".join(specification["categories"])
                raise ValueError(f"{name} must be one of: {choices}.")
            values[name] = value
            if value == "Unknown":
                missing_values.append(name)

    if len(missing_values) == len(feature_names):
        raise ValueError("At least one field must contain a known value.")
    systolic = values.get("Systolic_BP", np.nan)
    diastolic = values.get("Diastolic_BP", np.nan)
    if np.isfinite(systolic) and np.isfinite(diastolic) and systolic <= diastolic:
        raise ValueError("Systolic_BP must be greater than Diastolic_BP.")

    warnings = []
    if missing_values:
        warnings.append(
            "Missing or Unknown values were supplied for: "
            + ", ".join(missing_values)
            + ". The saved preprocessing handles them; the result has less "
            "information and should be interpreted cautiously."
        )
    if any(name in missing_values for name in CLINICAL_FIELDS):
        warnings.append(
            "Clinical measurements are unknown. Imputation is not a substitute for "
            "ECG or imaging tests, and condition scores may change substantially. "
            "Do not guess clinical values to obtain a result."
        )
    # Explicit column order guarantees the same layout used during training.
    return pd.DataFrame([values], columns=feature_names), warnings


def general_advisory(patient):
    """Simple educational rules, separate from the model's predicted class."""
    advice = []
    if patient.get("Smoking") == "Current":
        advice.append(
            "Consider asking a healthcare professional for support to stop smoking."
        )
    else:
        advice.append("Avoid tobacco use and exposure to second-hand smoke.")
    if patient.get("Physical_Activity") == "Sedentary":
        advice.append(
            "Discuss suitable ways to add regular physical activity to your routine "
            "with a healthcare professional."
        )
    else:
        advice.append("Maintain regular physical activity suited to your abilities.")
    if patient.get("Diet") in {"Unhealthy", "Average"}:
        advice.append(
            "Work toward a balanced diet with fruit and vegetables, and reduce salt."
        )
    else:
        advice.append("Continue to choose a balanced diet with fruit and vegetables.")
    if patient.get("Hypertension") == 1:
        advice.append(
            "Discuss your reported hypertension and blood-pressure monitoring "
            "with a qualified healthcare professional."
        )
    else:
        advice.append(
            "Have blood pressure checked and discuss concerning measurements "
            "with a qualified healthcare professional."
        )
    return advice


def predict_patient(patient, model=None, metadata=None, *, advisory_units_confirmed=False):
    """Return five synthetic-label predictions, scores, and separate advice.

    Supplying both ``model`` and ``metadata`` allows Streamlit to reuse its cache.
    Otherwise the artifacts are loaded from the project's models directory.
    """
    if model is None and metadata is None:
        model, metadata = load_artifacts()
    elif model is None or metadata is None:
        raise ValueError("Provide both model and metadata, or neither.")
    _validate_artifacts(model, metadata)
    frame, warnings = validate_patient(patient, metadata)
    probabilities = positive_probabilities(model, frame)[0]
    conditions = []
    for name, probability in zip(TARGET_COLUMNS, probabilities):
        probability = float(probability)
        threshold = metadata["thresholds"][name]
        conditions.append({
            "target": name, "condition": TARGET_NAMES[name],
            "predicted_label": int(probability >= threshold),
            "estimated_probability": probability, "threshold": threshold,
            "model_score_band": model_score_band(probability),
        })
    positives = [item["condition"] for item in conditions if item["predicted_label"] == 1]
    return {
        "task": "synthetic_multilabel_classification",
        "message": ("Conditions above the model threshold: " + ", ".join(positives)) if positives
            else "All five condition scores are below the 50% model threshold.",
        "conditions": conditions,
        "predicted_conditions": positives,
        "warnings": warnings,
        "advisory": general_advisory(patient),
        "condition_guidance": condition_guidance(
            patient, units_confirmed=advisory_units_confirmed
        ),
        "advisory_source": ADVISORY_SOURCE,
        "disclaimer": DISCLAIMER,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Path to a patient JSON object.")
    parser.add_argument("--example", choices=["A", "B"], default="A",
                        help="Fictional preset when --input is not supplied.")
    parser.add_argument(
        "--confirm-advisory-units", action="store_true",
        help="Confirm kg, cm, mmHg and mg/dL for optional numerical health guidance.",
    )
    arguments = parser.parse_args()
    try:
        model, metadata = load_artifacts()
        if arguments.input is None:
            patient = example_inputs(arguments.example)
        else:
            with arguments.input.open(encoding="utf-8") as input_file:
                patient = json.load(input_file)
        result = predict_patient(
            patient, model=model, metadata=metadata,
            advisory_units_confirmed=arguments.confirm_advisory_units,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
