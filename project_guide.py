"""Machine-learning workflow, dataset and evaluation overview."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parent

def render_about():
    st.header('Project overview')
    st.write('A multilabel classification pipeline comparing Logistic Regression, Random Forest and XGBoost. Each condition has a separate output, allowing multiple positive labels for one record.')
    card=json.loads((ROOT/'data/synthetic_heart_conditions.json').read_text())
    metadata=json.loads((ROOT/'models/model_metadata.json').read_text())
    records, features, targets = st.columns(3)
    records.metric('Synthetic records', f"{card['rows']:,}")
    features.metric('Input features', len(metadata['feature_names']))
    targets.metric('Target labels', len(metadata['target_columns']))
    st.caption('Generated data with assumption-based labels; performance has not been validated on clinical records.')
    workflow,data,concepts,metrics=st.tabs(['Workflow','Dataset','Machine learning','Model evaluation'])
    with workflow:
        st.markdown('''1. **Generate:** create 24 input fields and five noisy binary labels using a repeatable random seed.
2. **Validate:** check ranges, category values, duplicate inputs and blood-pressure consistency.
3. **Split:** reserve 4,000 test rows; split the remaining 16,000 into 12,800 fitting and 3,200 validation rows.
4. **Preprocess:** learn numerical medians, binary modes and categorical encoding from fitting data only. Standardize numbers for Logistic Regression.
5. **Compare:** train Logistic Regression, Random Forest and XGBoost, each with five binary estimators.
6. **Select:** rank validation macro F1, then average precision. Refit the selected approach on all 16,000 training rows.
7. **Evaluate:** inspect held-out per-condition metrics, a baseline and missing-input experiments.
8. **Predict:** validate the form, load the saved pipeline and show five scores. The app does not retrain on submitted examples.
9. **Advise:** apply separate, source-linked lifestyle rules to reported inputs. These rules are not the model's explanation or a treatment plan.''')
    with data:
        card=json.loads((ROOT/'data/synthetic_heart_conditions.json').read_text())
        st.metric('Generated examples',card['rows'])
        st.dataframe(pd.DataFrame({'Synthetic target':list(card['positive_counts']),'Positive labels':list(card['positive_counts'].values())}),hide_index=True)
        st.write(f"Multiple-label examples: {card['multiple_label_rows']:,}. Examples with no positive label: {card['no_label_rows']:,}.")
        st.markdown('''**How values are chosen:** age is sampled from integer ages 18–90. Height is sampled around 170 cm, with a standard deviation of 10 cm, then bounded to 140–205 cm. Weight is calculated from a generated body-size value and height. Some measurements are deliberately related—for example, generated hypertension history influences generated BP.

**How labels are chosen:** five invented equations combine selected inputs into a value called a logit. A sigmoid converts it to a number between 0 and 1. A Bernoulli draw then selects 0 or 1. A probability of 0.7 means a 70% chance of label 1 in that draw; it is not a fixed label and is not an estimate of a real person's risk.

**Why seed 42?** It makes the random sequence repeatable. The number 42 has no clinical significance. We also evaluate independently generated datasets with seeds 101, 202 and 303.

**Why randomness?** It prevents perfectly deterministic labels. It does not make the assumptions clinically realistic. No target is copied into the model's inputs.''')
        st.latex(r'p = \frac{1}{1+e^{-z}}, \qquad y \sim \mathrm{Bernoulli}(p)')
        st.caption('24 features = 11 numerical + 6 binary + 7 categorical fields. Five target columns are excluded from all input features. ECG and imaging values should not be guessed.')
    with concepts:
        topics={
          'Supervised learning and multilabel classification': 'Supervised learning fits a relationship between inputs X and known training labels y. Our labels are generated, not patient-confirmed. Multilabel means five independent yes/no outputs. Multiclass would choose one mutually exclusive class; that would prevent overlapping conditions.',
          'Features, targets and data leakage': 'Features are the 24 inputs. Targets are the five synthetic condition labels. Leakage occurs when information unavailable at prediction time enters training or evaluation. We exclude targets, reject duplicate feature rows, keep partitions disjoint, and fit preprocessing only on training data. Same-generator testing still cannot establish real-world validity.',
          'Preprocessing and pipelines': 'Numerical missing values use training medians; binary fields use modes. One-hot encoding creates indicators for category values. Numerical standardization helps Logistic Regression. A Pipeline applies exactly the same fitted transformations at training and prediction time. Imputation does not recover a missing clinical test.',
          'Logistic Regression': 'Each target gets a weighted linear combination of transformed features followed by a sigmoid. Training estimates coefficients from labels. Despite its name, Logistic Regression is used for classification. It was selected by validation macro F1 here. The generator uses logistic equations, so this benchmark partly favors that model family; this is not proof it is best for clinical data.',
          'Random Forest and XGBoost': 'Random Forest combines many randomized decision trees to reduce variance. XGBoost adds trees sequentially to improve a loss function. They can capture nonlinear relationships. Our experiment fixes their hyperparameters rather than claiming an exhaustive tuning search.',
          'Training, validation and testing': 'Training learns parameters; validation chooses the model; the held-out test partition estimates performance after that choice. Hyperparameters control model structure, such as tree depth. Repeatedly choosing changes based on test scores turns the test set into development data. New generator seeds test repeatability within the same simulation, not external clinical validity.',
          'Overfitting and underfitting': 'Overfitting learns accidental training patterns and fails on new examples; underfitting misses useful patterns. Compare training and validation results rather than assuming high training accuracy is sufficient. Our random label draws impose irreducible uncertainty: perfect prediction is neither guaranteed nor a sensible target.',
          'Thresholds, scores and calibration': 'Each target score is compared with 0.5. The five scores do not need to sum to 100%. Low/Moderate/High/Very high are display bands, not severity. Calibration asks whether events occur at the frequency implied by scores. Brier scores measure squared probability error; they alone do not prove calibration or medical validity.',
          'Explainability versus causality': 'A model coefficient or feature importance describes model behavior, not what caused a disease. Our lifestyle topic matches are separate rules, not feature-attribution explanations. Changing an input and seeing a different score cannot establish that an intervention will improve health.',
          'Deployment, cost and privacy': 'The application runs locally with Streamlit and loads a saved machine-learning pipeline. Hardware and hosting still have costs. The app does not create a patient database. Do not enter identifying or sensitive records into a public application. Real deployment would need access controls, HTTPS, monitoring, privacy review and clinical evidence.',
        }
        for title,text in topics.items():
            with st.expander(title):st.write(text)
    with metrics:
        st.table([
          {'Metric':'Precision','Meaning':'Of positive predictions, how many match positive labels?'},
          {'Metric':'Recall','Meaning':'Of actual positive labels, how many are detected?'},
          {'Metric':'F1','Meaning':'Harmonic mean of precision and recall; balances the two.'},
          {'Metric':'Macro average','Meaning':'Average each condition equally, regardless of prevalence.'},
          {'Metric':'Micro F1','Meaning':'Pool label decisions across all conditions before calculating F1.'},
          {'Metric':'Subset accuracy','Meaning':'A row is correct only if all five predictions match.'},
          {'Metric':'Hamming loss','Meaning':'Fraction of individual label decisions that are wrong; lower is better.'},
          {'Metric':'ROC-AUC / average precision','Meaning':'Ranking metrics across thresholds; neither is clinical accuracy.'},
        ])
        st.write('Confusion matrices contain true negatives, false positives, false negatives and true positives. A false negative is a missed synthetic positive label; it must never be described as proof a person is disease-free.')
        scores=pd.read_csv(ROOT/'results/model_comparison.csv')
        st.dataframe(scores[scores.split=='test'],hide_index=True)
        st.write('The training-prevalence baseline uses the same constant probability for each condition across all test examples. Beating that baseline shows learned predictive structure in the generated data, not clinical benefit.')
        st.caption('Use Model performance in the sidebar for repeated-generation experiments and measured prediction latency.')
    def open_prediction():
        st.session_state['project_page']='Prediction'
    st.button('Open prediction',on_click=open_prediction,type='primary')
