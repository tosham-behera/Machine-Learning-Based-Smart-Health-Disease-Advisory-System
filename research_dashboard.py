"""Saved model evaluation results and prediction performance."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parent


def render_research():
    st.header('Model performance')
    st.write('Compare model performance with complete and missing clinical inputs in the synthetic dataset.')
    st.subheader('Held-out model comparison')
    scores=pd.read_csv(ROOT/'results/model_comparison.csv')
    st.dataframe(scores[scores.split=='test'],hide_index=True)
    st.caption('Models share the same synthetic test partition. Published clinical scores cannot be directly ranked against these results.')
    path=ROOT/'results/robustness_experiments.csv'
    if path.exists():
        st.subheader('Repeated-generation missing-data experiment')
        data=pd.read_csv(path)
        st.dataframe(data,hide_index=True)
        st.dataframe(data.groupby('scenario').macro_f1.agg(['mean','std']))
        st.caption('Three separate 5,000-row generated datasets; 80/20 splits; fixed Logistic Regression. Standard deviation across runs, not a confidence interval. Same generator family, not external clinical validation.')
    path=ROOT/'results/pbl_verification.json'
    if path.exists():
        evidence=json.loads(path.read_text())
        st.subheader('Prediction performance')
        latency, size = st.columns(2)
        latency.metric('Average prediction time', f"{evidence['mean_single_prediction_ms']:.1f} ms")
        size.metric('Saved model size', f"{evidence['model_bytes'] / 1024:.1f} KB")
