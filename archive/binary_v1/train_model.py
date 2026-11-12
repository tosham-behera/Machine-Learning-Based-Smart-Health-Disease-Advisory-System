"""Train, select, evaluate, and save the three requested ML pipelines."""

import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a desktop window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from audit_data import audit_dataset, file_sha256
from health_data import (
    ROOT, DATA_DIR, TARGET, FEATURE_COLUMNS, NUMERIC_COLUMNS, BINARY_COLUMNS,
    CATEGORICAL_COLUMNS, build_input_schema, clean_features, read_dataset,
)

SEED = 42
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def make_pipeline(classifier, scale=False):
    """Each fit learns imputation, encoding, and scaling from that split only."""
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC_COLUMNS),
        ("binary", SimpleImputer(strategy="most_frequent", keep_empty_features=True), BINARY_COLUMNS),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
    ])
    return Pipeline([
        ("clean", FunctionTransformer(clean_features, feature_names_out="one-to-one")),
        ("preprocess", preprocessing),
        ("classifier", classifier),
    ])


def candidate_models():
    return {
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=2000, random_state=SEED), scale=True),
        "Random Forest": make_pipeline(
            RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=2)),
        "XGBoost": make_pipeline(
            XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                          objective="binary:logistic", eval_metric="logloss",
                          tree_method="hist", random_state=SEED, n_jobs=2)),
    }


def evaluate(model, features, labels):
    predicted = model.predict(features)
    positive_column = list(model.classes_).index(1)
    probability = model.predict_proba(features)[:, positive_column]
    metrics = {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probability)),
    }
    details = {
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            labels, predicted, labels=[0, 1], output_dict=True, zero_division=0),
    }
    return metrics, details, predicted, probability


def save_plots(comparison, details, test_labels, probabilities, directory):
    test_scores = comparison[comparison["split"] == "test"].set_index("model")
    axis = test_scores[METRICS].plot.bar(figsize=(11, 5), ylim=(0, 1.08), rot=0)
    axis.set_title("Held-out test metrics — synthetic labels only")
    axis.set_ylabel("Score")
    axis.legend(loc="lower right")
    axis.figure.tight_layout()
    axis.figure.savefig(directory / "model_comparison.png", dpi=160)
    plt.close(axis.figure)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, name in zip(axes, probabilities):
        ConfusionMatrixDisplay(np.array(details[name]["test"]["confusion_matrix"]),
                               display_labels=["Class 0", "Class 1"]).plot(
                                   ax=axis, colorbar=False, cmap="Blues")
        axis.set_title(name)
    figure.suptitle("Test confusion matrices — rows: actual; columns: predicted")
    figure.tight_layout()
    figure.savefig(directory / "confusion_matrices.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 5))
    for name, probability in probabilities.items():
        false_positive, true_positive, _ = roc_curve(test_labels, probability)
        auc = test_scores.loc[name, "roc_auc"]
        axis.plot(false_positive, true_positive, label=f"{name} (AUC {auc:.4f})")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate",
             title="Test ROC curves — synthetic dataset")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(directory / "roc_curves.png", dpi=160)
    plt.close(figure)


def train(data_path, output_dir):
    results_dir = output_dir / "results"
    model_dir = output_dir / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    frame = read_dataset(data_path)
    features, target = frame[FEATURE_COLUMNS], frame[TARGET].astype(int)
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=SEED)
    fit_x, validation_x, fit_y, validation_y = train_test_split(
        train_x, train_y, test_size=0.2, stratify=train_y, random_state=SEED)
    # Specify the ranking BEFORE seeing any test predictions.
    # Recall and F1 are primary; precision, AUC and generalization gap break ties.
    priority = {"Logistic Regression": 0, "Random Forest": 1, "XGBoost": 2}
    models = candidate_models()
    records, details, validation_scores = [], {}, {}
    for name, pipeline in models.items():
        print(f"Fitting {name}: inner training={len(fit_x)}, validation={len(validation_x)}", flush=True)
        pipeline.fit(fit_x, fit_y)
        details[name] = {}
        for split, x_values, y_values in [
            ("inner_train", fit_x, fit_y), ("validation", validation_x, validation_y)
        ]:
            scores, diagnostic, _, _ = evaluate(pipeline, x_values, y_values)
            records.append({"model": name, "split": split, **scores})
            details[name][split] = diagnostic
            if split == "inner_train":
                training_f1 = scores["f1"]
            else:
                validation_scores[name] = {
                    **scores, "f1_gap": abs(training_f1 - scores["f1"])
                }
    selected = max(models, key=lambda name: (
        validation_scores[name]["recall"], validation_scores[name]["f1"],
        validation_scores[name]["precision"], validation_scores[name]["roc_auc"],
        -validation_scores[name]["f1_gap"], -priority[name],
    ))
    print(f"Selected using validation only: {selected}", flush=True)
    probabilities = {}
    selected_pipeline = None
    for name, prototype in models.items():
        # Refit all three once on the same complete 80% training partition.
        pipeline = clone(prototype).fit(train_x, train_y)
        for split, x_values, y_values in [
            ("train", train_x, train_y), ("test", test_x, test_y)
        ]:
            scores, diagnostic, predicted, probability = evaluate(pipeline, x_values, y_values)
            records.append({"model": name, "split": split, **scores})
            details[name][split] = diagnostic
            if split == "test":
                probabilities[name] = probability
        if name == selected:
            selected_pipeline = pipeline
    comparison = pd.DataFrame(records)
    comparison.to_csv(results_dir / "model_comparison.csv", index=False)
    (results_dir / "classification_reports.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8")
    split_sizes = {
        "train": len(train_x), "test": len(test_x),
        "inner_train": len(fit_x), "validation": len(validation_x),
    }
    model_path = model_dir / "best_heart_disease_model.joblib"
    joblib.dump(selected_pipeline, model_path, compress=3)
    # Confirm serialization preserves real predictions.
    restored = joblib.load(model_path)
    np.testing.assert_array_equal(restored.predict(test_x), selected_pipeline.predict(test_x))
    metadata = {
        "model_name": selected, "feature_names": FEATURE_COLUMNS,
        "input_schema": build_input_schema(train_x), "threshold": 0.5,
        "target": TARGET, "classes": [0, 1], "random_state": SEED,
        "selection_rule": "Validation recall, then F1, precision, ROC-AUC, smaller F1 gap; exact ties prefer Logistic Regression, Random Forest, then XGBoost.",
        "selection_reason": f"{selected} ranked first by the predeclared validation rule. Test scores did not determine selection.",
        "validation_scores": validation_scores,
        "split_sizes": split_sizes,
        "source_file": data_path.name, "source_sha256": file_sha256(data_path),
        "model_sha256": file_sha256(model_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "package_versions": {package: importlib.metadata.version(package) for package in [
            "pandas", "numpy", "scikit-learn", "xgboost", "joblib", "matplotlib", "streamlit",
        ]},
        "limitations": [
            "Educational synthetic-data classifier; not a medical diagnosis or validated risk score.",
            "Labels exactly match an observed five-feature rule in the supplied dataset.",
            "Original generator, precise Kaggle source, units and temporal definitions are unverified.",
            "Previous_Heart_Attack assumes prior history; proxy or target-definition leakage cannot be excluded.",
            "Test split has already been explored in the historical notebook; it is not an external blind test.",
            "BMI excluded. Invalid historical BP pairs are imputed using training medians.",
            "Missing numeric values and Unknown categories have not been clinically validated.",
        ],
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_plots(comparison, details, test_y, probabilities, results_dir)
    audit = audit_dataset(data_path, DATA_DIR / "heart_disease_cleaned.csv")
    (results_dir / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(comparison[comparison["split"].isin(["validation", "train", "test"])].to_string(index=False))
    print(f"Saved {model_path.name}; verified reload predictions match.", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "synthetic_heart_disease_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    train(args.data.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
