"""Generate fictional multilabel examples; not a clinical disease simulator.

All distributions and coefficients below are arbitrary teaching choices. Labels
are sampled AFTER features, with random variation, and never fed back as inputs.
No real patients, diagnostic criteria or clinical prevalence estimates are used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from health_data import (DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMNS, NUMERIC_SPECS,
                         build_input_schema, SCHEMA_VERSION, read_dataset)

GENERATOR_VERSION = "2.0.0"


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -30, 30)))


def generate_dataset(rows=20000, seed=42):
    if rows < 1000:
        raise ValueError("Use at least 1000 rows for the five-target experiment.")
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(index=range(rows))
    d["Age"] = rng.integers(18, 91, rows)
    d["Gender"] = rng.choice(["Female", "Male"], rows)
    d["Height"] = np.clip(rng.normal(170, 10, rows), 140, 205)
    body_index = np.clip(rng.normal(28, 6, rows), 18, 48)
    d["Weight"] = np.clip(body_index * (d.Height / 100) ** 2, 40, 180)
    for name, choices, weights in [
        ("Smoking", ["Never", "Former", "Current"], [.55, .20, .25]),
        ("Alcohol_Intake", ["None", "Low", "Moderate", "High"], [.25, .35, .25, .15]),
        ("Physical_Activity", ["Active", "Moderate", "Sedentary"], [.30, .40, .30]),
        ("Diet", ["Healthy", "Average", "Unhealthy"], [.30, .45, .25]),
        ("Stress_Level", ["Low", "Medium", "High"], [.30, .45, .25]),
    ]:
        d[name] = rng.choice(choices, rows, p=weights)
    d["Hypertension"] = rng.binomial(1, sigmoid(-1.3 + .025*(d.Age-45) + .07*(body_index-28)))
    d["Diabetes"] = rng.binomial(1, sigmoid(-1.8 + .02*(d.Age-45) + .08*(body_index-28)))
    d["Hyperlipidemia"] = rng.binomial(1, .30, rows)
    d["Family_History"] = rng.binomial(1, .25, rows)
    d["Previous_Heart_Attack"] = rng.binomial(1, sigmoid(-2.6 + .03*(d.Age-45)))
    d["Systolic_BP"] = np.clip(rng.normal(118 + 23*d.Hypertension + .18*(d.Age-45), 12), 90, 179)
    d["Diastolic_BP"] = np.minimum(np.clip(rng.normal(74 + 12*d.Hypertension, 9), 50, 119), d.Systolic_BP - 15)
    d["Heart_Rate"] = np.clip(rng.normal(78, 16, rows), 45, 140)
    d["Blood_Sugar_Fasting"] = np.clip(rng.normal(92 + 65*d.Diabetes, 20), 65, 250)
    d["Cholesterol_Total"] = np.clip(rng.normal(185 + 55*d.Hyperlipidemia, 30), 100, 350)
    # Latent variation creates overlapping measurements without copying a target.
    muscle_variation = rng.binomial(1, .25, rows)
    ischemia_variation = rng.binomial(1, .28, rows)
    d["Exercise_ST_Depression"] = np.clip(rng.normal(.35 + 1.9*ischemia_variation, .65), 0, 5)
    d["Ejection_Fraction"] = np.clip(rng.normal(61 - 23*muscle_variation - 5*d.Previous_Heart_Attack, 7), 20, 75)
    d["LV_Wall_Thickness"] = np.clip(rng.normal(9 + 2.8*d.Hypertension + 2*muscle_variation, 1.8), 6, 20)
    d["ECG_Rhythm"] = np.where(rng.random(rows) < (.12 + .20*muscle_variation), "Irregular", "Regular")
    d["Exertional_Chest_Pain"] = rng.binomial(1, .12 + .50*ischemia_variation)
    for name in NUMERIC_SPECS:
        d[name] = d[name].round(1)
    # Arbitrary soft relationships; these equations are NOT medical risk models.
    logits = {
        "CAD": -3.0 + 1.2*d.Exercise_ST_Depression + .8*d.Exertional_Chest_Pain
               + .012*(d.Cholesterol_Total-180) + .025*(d.Age-45)
               + .5*(d.Smoking == "Current") + .4*d.Diabetes,
        "Hypertensive_Heart_Disease": -2.4 + .07*(d.Systolic_BP-120)
               + .50*(d.LV_Wall_Thickness-9) + .7*d.Hypertension,
        "Heart_Failure": -2.7 + .14*(60-d.Ejection_Fraction)
               + .6*d.Hypertension + .5*d.Diabetes + .4*d.Previous_Heart_Attack,
        "Arrhythmia": -2.8 + 3.3*(d.ECG_Rhythm == "Irregular")
               + .035*abs(d.Heart_Rate-75) + .5*(d.Alcohol_Intake == "High"),
        "Cardiomyopathy": -3.0 + .11*(60-d.Ejection_Fraction)
               + .40*(d.LV_Wall_Thickness-9) + .8*(d.Alcohol_Intake == "High")
               + .5*d.Family_History,
    }
    for name in TARGET_COLUMNS:
        d[name] = rng.binomial(1, sigmoid(logits[name]))
    return d[FEATURE_COLUMNS + TARGET_COLUMNS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    frame = generate_dataset(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    read_dataset(args.output)
    card = {
        "schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION,
        "generator_file": "generate_data.py", "seed": args.seed, "rows": args.rows,
        "source": "Entirely generated locally; no real patient records or Kaggle source.",
        "label_process": "Five Bernoulli draws from arbitrary logistic equations in generate_dataset; features generated first.",
        "limitations": "Fictional labels, distributions and prevalence. No diagnosis, clinical validity, prognosis or calibrated personal risk.",
        "target_columns": TARGET_COLUMNS, "input_schema": build_input_schema(),
        "positive_counts": {name: int(frame[name].sum()) for name in TARGET_COLUMNS},
        "multiple_label_rows": int((frame[TARGET_COLUMNS].sum(axis=1) > 1).sum()),
        "no_label_rows": int((frame[TARGET_COLUMNS].sum(axis=1) == 0).sum()),
        "data_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(json.dumps({key: card[key] for key in ["rows", "positive_counts", "multiple_label_rows", "no_label_rows"]}, indent=2))


if __name__ == "__main__":
    main()
