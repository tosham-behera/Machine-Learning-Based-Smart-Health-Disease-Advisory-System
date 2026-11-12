"""Train and compare three multilabel classifiers on fictional condition labels."""
import argparse
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, hamming_loss, confusion_matrix, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from audit_data import audit_dataset, file_sha256
from health_data import (ROOT, DATA_PATH, FEATURE_COLUMNS, NUMERIC_COLUMNS, BINARY_COLUMNS,
    CATEGORICAL_COLUMNS, TARGET_COLUMNS, TARGET_NAMES, SCHEMA_VERSION,
    build_input_schema, clean_features, read_dataset, positive_probabilities, CLINICAL_FIELDS)

SEED = 42
THRESHOLD = 0.5


def make_pipeline(classifier, scale=False):
    numeric = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if scale:
        numeric.append(('scale', StandardScaler()))
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline(numeric), NUMERIC_COLUMNS),
        ('binary', SimpleImputer(strategy='most_frequent', keep_empty_features=True), BINARY_COLUMNS),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLUMNS),
    ])
    return Pipeline([
        ('clean', FunctionTransformer(clean_features, feature_names_out='one-to-one')),
        ('preprocess', preprocessing), ('classifier', MultiOutputClassifier(classifier, n_jobs=1)),
    ])


def candidate_models():
    return {
        'Logistic Regression': make_pipeline(LogisticRegression(max_iter=2000, random_state=SEED), True),
        'Random Forest': make_pipeline(RandomForestClassifier(n_estimators=100, max_depth=12,
                                      min_samples_leaf=5, random_state=SEED, n_jobs=2)),
        'XGBoost': make_pipeline(XGBClassifier(n_estimators=100, max_depth=3, learning_rate=.07,
            objective='binary:logistic', eval_metric='logloss', tree_method='hist', random_state=SEED, n_jobs=2)),
    }


def score_predictions(labels, probability):
    predicted = (probability >= THRESHOLD).astype(int)
    metrics = {
        'subset_accuracy': float(accuracy_score(labels, predicted)),
        'hamming_loss': float(hamming_loss(labels, predicted)),
        'macro_precision': float(precision_score(labels, predicted, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(labels, predicted, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(labels, predicted, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(labels, predicted, average='micro', zero_division=0)),
        'macro_roc_auc': float(roc_auc_score(labels, probability, average='macro')),
        'macro_average_precision': float(average_precision_score(labels, probability, average='macro')),
    }
    per_condition = {}
    for i, name in enumerate(TARGET_COLUMNS):
        actual, pred, prob = labels[name], predicted[:, i], probability[:, i]
        per_condition[name] = {
            'precision': float(precision_score(actual, pred, zero_division=0)),
            'recall': float(recall_score(actual, pred, zero_division=0)),
            'f1': float(f1_score(actual, pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(actual, prob)),
            'average_precision': float(average_precision_score(actual, prob)),
            'positive_support': int(actual.sum()),
            'brier_score': float(np.mean((actual.to_numpy() - prob) ** 2)),
            'confusion_matrix': confusion_matrix(actual, pred, labels=[0, 1]).tolist(),
        }
    return metrics, per_condition


def train(data_path=DATA_PATH, output_dir=ROOT):
    frame = read_dataset(data_path)
    x, y = frame[FEATURE_COLUMNS], frame[TARGET_COLUMNS]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.2, random_state=SEED)
    fit_x, val_x, fit_y, val_y = train_test_split(train_x, train_y, test_size=.2, random_state=SEED)
    assert not (set(train_x.index) & set(test_x.index))
    assert not (set(fit_x.index) & set(val_x.index))
    for labels in (fit_y, val_y, train_y, test_y):
        if any(set(labels[name]) != {0, 1} for name in TARGET_COLUMNS):
            raise ValueError('A split lacks a target class; generate a larger dataset.')
    models = candidate_models()
    records, reports, validation = [], {}, {}
    # Predeclared ranking: validation macro F1, then average precision; stable ties.
    for name, pipeline in models.items():
        print(f'Fitting {name} on {len(fit_x)} rows...', flush=True)
        pipeline.fit(fit_x, fit_y)
        reports[name] = {}
        for split, features, labels in [('inner_train', fit_x, fit_y), ('validation', val_x, val_y)]:
            metrics, detail = score_predictions(labels, positive_probabilities(pipeline, features))
            records.append({'model': name, 'split': split, **metrics})
            reports[name][split] = detail
            if split == 'validation':
                validation[name] = metrics
    names = list(models)
    selected = max(names, key=lambda name: (validation[name]['macro_f1'],
        validation[name]['macro_average_precision'], -names.index(name)))
    print(f'Validation selected: {selected}', flush=True)
    selected_model = None
    for name, prototype in models.items():
        print(f'Refitting {name} on {len(train_x)} rows...', flush=True)
        model = clone(prototype).fit(train_x, train_y)
        for split, features, labels in [('train', train_x, train_y), ('test', test_x, test_y)]:
            metrics, detail = score_predictions(labels, positive_probabilities(model, features))
            records.append({'model': name, 'split': split, **metrics})
            reports[name][split] = detail
        if name == selected:
            selected_model = model
    baseline_scores = np.tile(train_y.mean().to_numpy(), (len(test_y), 1))
    baseline, baseline_detail = score_predictions(test_y, baseline_scores)
    records.append({'model': 'Training-prevalence baseline', 'split': 'test', **baseline})
    reports['Training-prevalence baseline'] = {'test': baseline_detail}
    masked = test_x.copy()
    for name in CLINICAL_FIELDS:
        masked[name] = 'Unknown' if name in CATEGORICAL_COLUMNS else np.nan
    masked_metrics, masked_detail = score_predictions(test_y, positive_probabilities(selected_model, masked))
    records.append({'model': selected, 'split': 'test_missing_clinical_fields', **masked_metrics})
    reports[selected]['test_missing_clinical_fields'] = masked_detail
    results, models_dir = output_dir / 'results', output_dir / 'models'
    results.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    selected_model.synthetic_schema_version_ = SCHEMA_VERSION
    selected_model.target_columns_ = TARGET_COLUMNS
    model_path = models_dir / 'heart_condition_model.joblib'
    joblib.dump(selected_model, model_path, compress=3)
    restored = joblib.load(model_path)
    np.testing.assert_allclose(positive_probabilities(restored, test_x),
                               positive_probabilities(selected_model, test_x), rtol=0, atol=0)
    comparison = pd.DataFrame(records)
    comparison.to_csv(results / 'model_comparison.csv', index=False)
    (results / 'classification_reports.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    audit = audit_dataset(data_path)
    (results / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    metadata = {
        'schema_version': SCHEMA_VERSION, 'model_name': selected, 'task': 'synthetic_multilabel_classification',
        'feature_names': FEATURE_COLUMNS, 'input_schema': build_input_schema(),
        'target_columns': TARGET_COLUMNS, 'target_names': TARGET_NAMES,
        'thresholds': {name: THRESHOLD for name in TARGET_COLUMNS},
        'selection_rule': 'Validation macro F1, then macro average precision; stable model order breaks exact ties.',
        'split_method': 'Disjoint seeded random split 64% fit / 16% validation / 20% test; not stratified. Every target has both classes in each split.',
        'split_sizes': {'inner_train': len(fit_x), 'validation': len(val_x), 'train': len(train_x), 'test': len(test_x)},
        'split_positive_counts': {split: {name: int(labels[name].sum()) for name in TARGET_COLUMNS}
                                  for split, labels in [('inner_train', fit_y), ('validation', val_y), ('train', train_y), ('test', test_y)]},
        'validation_scores': validation, 'random_state': SEED,
        'source_file': data_path.name, 'source_sha256': file_sha256(data_path),
        'model_sha256': file_sha256(model_path), 'created_utc': datetime.now(timezone.utc).isoformat(),
        'package_versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'joblib']},
        'limitations': ['All labels and feature distributions are fictional generator assumptions.',
            'Evaluation measures agreement with the same synthetic process, not clinical accuracy.',
            'Scores are not personal disease probabilities, diagnoses, severity or future-event risk.',
            'Missing-input imputation can substantially change results; clinical inputs must not be guessed.',
            'Age 18-90 and other ranges are generator limits, not evidence of validation across populations.'],
    }
    (models_dir / 'model_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    test_scores = comparison[comparison['split'] == 'test'].set_index('model')
    ax = test_scores[['macro_f1', 'macro_recall', 'macro_roc_auc']].plot.bar(figsize=(10, 5), ylim=(0, 1), rot=15)
    ax.set_title('Synthetic held-out results - not clinical accuracy')
    ax.figure.tight_layout()
    ax.figure.savefig(results / 'model_comparison.png', dpi=150)
    plt.close(ax.figure)
    figure, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    for axis, name in zip(axes, TARGET_COLUMNS):
        matrix = np.array(reports[selected]['test'][name]['confusion_matrix'])
        axis.imshow(matrix, cmap='Blues')
        for row in range(2):
            for col in range(2):
                axis.text(col, row, str(matrix[row, col]), ha='center', va='center', color='black')
        axis.set(title=name.replace('_', '\n'), xticks=[0, 1], yticks=[0, 1], xlabel='Predicted', ylabel='Synthetic label')
    figure.tight_layout()
    figure.savefig(results / 'confusion_matrices.png', dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 5))
    scores = positive_probabilities(selected_model, test_x)
    for i, name in enumerate(TARGET_COLUMNS):
        fpr, tpr, _ = roc_curve(test_y[name], scores[:, i])
        axis.plot(fpr, tpr, label=name)
    axis.plot([0, 1], [0, 1], '--', color='gray')
    axis.set(xlabel='False positive rate', ylabel='True positive rate', title='Synthetic test ROC curves')
    axis.legend()
    figure.tight_layout()
    figure.savefig(results / 'roc_curves.png', dpi=150)
    plt.close(figure)
    print(test_scores.to_string(), flush=True)
    print('Saved multilabel pipeline and verified exact score equality after reload.', flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA_PATH)
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    train(args.data.resolve(), args.output_dir.resolve())


if __name__ == '__main__':
    main()
