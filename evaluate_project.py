"""Reproduce PBL evaluation evidence without replacing the selected model."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from generate_data import generate_dataset
from health_data import FEATURE_COLUMNS,TARGET_COLUMNS,CLINICAL_FIELDS,positive_probabilities,example_inputs
from predict import load_artifacts,predict_patient
from train_model import candidate_models,score_predictions

def main():
    model,meta=load_artifacts()
    checks={}
    checks['example_A_no_labels']=len(predict_patient(example_inputs('A'),model,meta)['predicted_conditions'])==0
    checks['example_B_multiple_labels']=len(predict_patient(example_inputs('B'),model,meta)['predicted_conditions'])>1
    for name,updates in [('invalid_age',{'Age':17}),('contradictory_BP',{'Systolic_BP':90,'Diastolic_BP':110}),('nonfinite',{'Weight':float('nan')})]:
        try:predict_patient(dict(example_inputs(),**updates),model,meta)
        except ValueError:checks[name]=True
        else:checks[name]=False
    unknown=example_inputs()
    for name in CLINICAL_FIELDS:unknown[name]=None
    checks['unknown_clinical_warning']=any('Clinical measurements' in w for w in predict_patient(unknown,model,meta)['warnings'])
    checks['reproducible_generator']=generate_dataset(1000,7).equals(generate_dataset(1000,7))
    if not all(checks.values()):raise AssertionError(checks)
    rows=[]
    for seed in [101,202,303]:
        data=generate_dataset(5000,seed)
        xtrain,xtest,ytrain,ytest=train_test_split(data[FEATURE_COLUMNS],data[TARGET_COLUMNS],test_size=.2,random_state=seed)
        fitted=candidate_models()['Logistic Regression'].fit(xtrain,ytrain)
        for scenario in ['complete','clinical_missing']:
            x=xtest.copy()
            if scenario=='clinical_missing':
                for field in CLINICAL_FIELDS:x[field]='Unknown' if field=='ECG_Rhythm' else np.nan
            metrics,_=score_predictions(ytest,positive_probabilities(fitted,x))
            rows.append({'seed':seed,'scenario':scenario,**metrics})
    start=time.perf_counter()
    for _ in range(20):predict_patient(example_inputs('B'),model,meta)
    evidence={'checks':checks,'all_passed':all(checks.values()),'mean_single_prediction_ms':1000*(time.perf_counter()-start)/20,
      'model_bytes':Path('models/heart_condition_model.joblib').stat().st_size,
      'experiment':'Three newly generated 5000-row datasets; 80/20 split; fixed Logistic Regression, no tuning; same generator family, not external clinical validation.'}
    Path('results/pbl_verification.json').write_text(json.dumps(evidence,indent=2))
    pd.DataFrame(rows).to_csv('results/robustness_experiments.csv',index=False)
    print(json.dumps(evidence,indent=2));print(pd.DataFrame(rows).groupby('scenario').macro_f1.agg(['mean','std']))
if __name__=='__main__':main()
