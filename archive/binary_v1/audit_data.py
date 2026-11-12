"""Reproduce the data-quality audit without modifying either supplied CSV."""

import hashlib
import json

import pandas as pd

from health_data import (
    ROOT, DATA_DIR, TARGET, BINARY_COLUMNS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, read_dataset,
)


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_dataset(source_path=None, cleaned_path=None):
    source_path = source_path or DATA_DIR / "synthetic_heart_disease_dataset.csv"
    cleaned_path = cleaned_path or DATA_DIR / "heart_disease_cleaned.csv"
    original = read_dataset(source_path)
    calculated_bmi = original["Weight"] / (original["Height"] / 100) ** 2
    differences = (original["BMI"] - calculated_bmi).abs()
    numeric_summary = {}
    for column in NUMERIC_COLUMNS + ["BMI"]:
        values = original[column]
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        numeric_summary[column] = {
            "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "nonpositive": int((values <= 0).sum()),
            "iqr_outliers": int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()),
        }
    # Audit only: this relationship was discovered in the existing synthetic
    # dataset. It is NEVER used to generate training labels or app predictions.
    flags = (
        (original["Age"] > 55).astype(int)
        + (original["Cholesterol_Total"] > 240).astype(int)
        + original["Hypertension"] + original["Diabetes"]
        + original["Previous_Heart_Attack"]
    )
    report = {
        "source_file": source_path.name, "source_sha256": file_sha256(source_path),
        "rows": len(original), "columns": len(original.columns),
        "dtypes": {c: str(t) for c, t in original.dtypes.items()},
        "blank_missing_counts": original.isna().sum().to_dict(),
        "duplicates": int(original.duplicated().sum()),
        "class_counts": original[TARGET].value_counts().sort_index().to_dict(),
        "alcohol_literal_none_count": int(original["Alcohol_Intake"].eq("None").sum()),
        "alcohol_default_parser_na_count": int(pd.read_csv(source_path)["Alcohol_Intake"].isna().sum()),
        "numeric_summary": numeric_summary,
        "categories": {c: original[c].value_counts(dropna=False).to_dict()
                       for c in CATEGORICAL_COLUMNS + BINARY_COLUMNS},
        "bmi": {
            "mean_absolute_difference": float(differences.mean()),
            "max_absolute_difference": float(differences.max()),
            "difference_over_1_count": int((differences > 1).sum()),
            "original_calculated_correlation": float(original["BMI"].corr(calculated_bmi)),
            "decision": "Omit BMI from final model; preserve both supplied CSVs.",
        },
        "invalid_bp_pair_count": int((original["Systolic_BP"] <= original["Diastolic_BP"]).sum()),
        "observed_label_rule": {
            "expression": "(Age > 55) + (Cholesterol_Total > 240) + Hypertension + Diabetes + Previous_Heart_Attack >= 2",
            "matching_rows": int(((flags >= 2).astype(int) == original[TARGET]).sum()),
            "interpretation": "Observed relationship only; original generation procedure is unknown.",
        },
        "provenance": {
            "source": "Existing README attributes dataset to Kaggle; exact URL and author unknown.",
            "generation_process": "Unknown; no generator code or dataset documentation supplied.",
            "units": "Assumed from names/ranges, not verified by a source data dictionary.",
            "temporal_information": "Unknown. Previous_Heart_Attack is assumed to mean prior history; proxy leakage cannot be ruled out.",
        },
    }
    if cleaned_path.exists():
        cleaned = read_dataset(cleaned_path)
        bmi_q1, bmi_q3 = cleaned["BMI"].quantile([0.25, 0.75])
        bmi_iqr = bmi_q3 - bmi_q1
        bmi_outliers = (
            (cleaned["BMI"] < bmi_q1 - 1.5 * bmi_iqr)
            | (cleaned["BMI"] > bmi_q3 + 1.5 * bmi_iqr)
        )
        expected = original.copy()
        expected["BMI"] = calculated_bmi.round(2)
        expected["Alcohol_Intake"] = expected["Alcohol_Intake"].replace("None", "Unknown").fillna("Unknown")
        report["existing_cleaned"] = {
            "file": cleaned_path.name, "sha256": file_sha256(cleaned_path),
            "rows": len(cleaned), "missing_counts": cleaned.isna().sum().to_dict(),
            "matches_documented_transform": bool(cleaned.equals(expected)),
            "recalculated_bmi_iqr_outliers": int(bmi_outliers.sum()),
        }
    return report


def main():
    report = audit_dataset()
    destination = ROOT / "results"
    destination.mkdir(exist_ok=True)
    (destination / "data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in [
        "rows", "columns", "duplicates", "class_counts", "bmi",
        "alcohol_literal_none_count", "invalid_bp_pair_count", "observed_label_rule",
    ]}, indent=2))


if __name__ == "__main__":
    main()
