"""Audit the active generated multilabel dataset without changing it."""
import hashlib
import json
import pandas as pd
from health_data import ROOT, DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMNS, read_dataset


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_dataset(source_path=DATA_PATH):
    frame = read_dataset(source_path)
    labels = frame[TARGET_COLUMNS]
    return {
        'source_file': source_path.name, 'source_sha256': file_sha256(source_path),
        'rows': len(frame), 'features': len(FEATURE_COLUMNS), 'targets': TARGET_COLUMNS,
        'missing_counts': frame.isna().sum().to_dict(),
        'duplicate_feature_rows': int(frame[FEATURE_COLUMNS].duplicated().sum()),
        'invalid_bp_pairs': int((frame.Systolic_BP <= frame.Diastolic_BP).sum()),
        'positive_counts': {name: int(labels[name].sum()) for name in TARGET_COLUMNS},
        'label_count_distribution': {str(k): int(v) for k, v in labels.sum(axis=1).value_counts().sort_index().items()},
        'cooccurrence_counts': labels.T.dot(labels).to_dict(),
        'provenance': 'Locally generated fictional data; see generate_data.py and the dataset JSON card.',
    }


def main():
    report = audit_dataset()
    destination = ROOT / 'results'
    destination.mkdir(exist_ok=True)
    (destination / 'data_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
