"""Streamlit interface for the synthetic five-condition classification application."""
import pickle
import streamlit as st
from health_data import CLINICAL_FIELDS, example_inputs
from heart_guidance import CONDITION_GUIDE, SOURCES
from predict import (ADVISORY_SOURCE, DISCLAIMER, MODEL_SCORE_BANDS, MODEL_PATH,
                     METADATA_PATH, load_artifacts, predict_patient)


@st.cache_resource
def cached_artifacts(model_stamp, metadata_stamp):
    """Invalidate the cache when either saved artifact changes."""
    return load_artifacts()


def main():
    st.set_page_config(page_title='Heart Condition Classification', page_icon='🫀', layout='centered')
    st.title('Smart Health Advisory System')
    st.subheader('Synthetic heart-condition classification')
    st.warning(DISCLAIMER)
    sections = ['Project overview', 'Prediction', 'Model performance']
    previous = st.session_state.get('project_page')
    renamed = {'Prediction demo': 'Prediction', 'Research and evaluation': 'Model performance'}
    if previous in renamed:
        st.session_state['project_page'] = renamed[previous]
    elif previous is not None and previous not in sections:
        st.session_state['project_page'] = sections[0]
    page = st.sidebar.radio('Project section', sections, key='project_page')
    if page == 'Project overview':
        from project_guide import render_about
        render_about()
        return
    if page == 'Model performance':
        from research_dashboard import render_research
        render_research()
        return
    st.write('Enter measurements to compare the five condition scores, or select a fictional example to explore the model.')
    try:
        model, metadata = cached_artifacts(MODEL_PATH.stat().st_mtime_ns, METADATA_PATH.stat().st_mtime_ns)
    except (OSError, ValueError, ImportError, EOFError, pickle.UnpicklingError) as error:
        st.error(f'The saved model could not be loaded: {error}')
        st.info('The saved model needs to be generated or retrained before predictions are available. See the project setup instructions.')
        st.stop()
    st.caption(f"Saved model: {metadata['model_name']} · {len(metadata['feature_names'])} inputs · "
               '5 condition outputs')
    scenario = st.selectbox('Input mode', ['Fictional example A', 'Fictional example B', 'Enter values'], key='scenario')
    if st.session_state.get('_last_scenario') != scenario:
        if scenario == 'Enter values':
            values = {name: 'Unknown' if spec['kind'] == 'categorical' else None
                      for name, spec in metadata['input_schema'].items()}
        else:
            values = example_inputs('B' if scenario.endswith('B') else 'A')
        for name, value in values.items():
            st.session_state[f'input_{name}'] = value
        st.session_state['advisory_units_confirmed'] = scenario != 'Enter values'
        st.session_state['_last_scenario'] = scenario
    st.write('Ranges come from the fictional generator, not medical reference ranges. '
             'Clinical test fields require actual test reports; leave them blank or Unknown rather than guessing.')
    with st.form('prediction_form'):
        columns = st.columns(2)
        patient = {}
        for index, name in enumerate(metadata['feature_names']):
            spec = metadata['input_schema'][name]
            with columns[index % 2]:
                help_text = 'A clinical test measurement, not something to estimate yourself.' if name in CLINICAL_FIELDS else None
                if spec['kind'] == 'numeric':
                    patient[name] = st.number_input(spec['label'], min_value=float(spec['min']),
                        max_value=float(spec['max']), value=None, step=float(spec.get('step', 1)),
                        key=f'input_{name}', help=help_text)
                elif spec['kind'] == 'binary':
                    patient[name] = st.selectbox(spec['label'], [0, 1, None],
                        format_func=lambda value: {0:'No (0)', 1:'Yes (1)', None:'Unknown'}[value],
                        key=f'input_{name}')
                else:
                    patient[name] = st.selectbox(spec['label'], spec['categories'],
                        key=f'input_{name}', help=help_text)
        units_confirmed = st.checkbox('Measurements use kg, cm, mmHg and mg/dL (cholesterol).',
            help='Enables numerical lifestyle guidance. The synthetic examples use these documented units.',
            key='advisory_units_confirmed')
        submitted = st.form_submit_button('Calculate model probabilities', type='primary')
    if submitted:
        try:
            result = predict_patient(patient, model=model, metadata=metadata,
                                     advisory_units_confirmed=units_confirmed)
        except ValueError as error:
            st.error(f'Please check your inputs: {error}')
        else:
            st.divider()
            st.subheader('Condition probabilities')
            st.info(result['message'])
            st.table([{
                'Condition': item['condition'],
                'Model probability': f"{item['estimated_probability']:.1%}",
                'Score level': item['model_score_band']['label'],
                'Prediction': 'Above threshold' if item['predicted_label'] else 'Below threshold',
            } for item in result['conditions']])
            st.caption('Above threshold means a model probability of 50% or higher. Each condition is evaluated separately; percentages do not sum to 100%.')
            with st.expander('How the illustrative score bands work'):
                st.table([{'Band': label, 'Score range': interval} for _, label, interval in MODEL_SCORE_BANDS])
                st.caption('Score levels group the unrounded model probabilities; they do not describe disease severity.')
            for warning in result['warnings']:
                st.warning(warning)
            st.subheader('Lifestyle guidance')
            guidance = result['condition_guidance']
            if not units_confirmed:
                st.caption('Confirm the measurement units above to include weight, blood-pressure and cholesterol guidance.')
            elif guidance['notes']:
                with st.expander('Measurement details'):
                    for note in guidance['notes']:
                        st.write(note)
            if guidance['calculated_bmi'] is not None:
                st.write(f"Calculated adult BMI: **{guidance['calculated_bmi']:.1f} kg/m²** (screening only)")
            if not guidance['topics']:
                missing = guidance['missing_fields']
                if missing:
                    st.write('For more specific suggestions, complete: ' + ', '.join(name.replace('_', ' ') for name in missing) + '.')
                else:
                    st.write('No additional targeted lifestyle suggestions were identified from the supplied information. General preventive suggestions are available below.')
            for topic in guidance['topics']:
                with st.container(border=True):
                    st.markdown(f"#### {topic['title']}")
                    st.write('Reported factors: ' + '; '.join(topic['matched_factors']))
                    st.markdown('**Suggested steps**')
                    for action in topic['actions']:
                        st.markdown(f'- {action}')
                    st.markdown(f"[Medical background]({topic['source']})")
            with st.expander('Condition background: heart attacks and cardiac arrest'):
                st.caption('Background summary based on NHLBI, MedlinePlus and American Heart Association references. Sources are linked below.')
                st.table(list(CONDITION_GUIDE))
                st.markdown('Sources: ' + ' · '.join(f'[{name}]({url})' for name, url in SOURCES.items()))
            with st.expander('General preventive suggestions'):
                for advice in result['advisory']:
                    st.markdown(f'- {advice}')
                st.markdown(f'[WHO cardiovascular health guidance]({ADVISORY_SOURCE})')
            st.caption('For new chest pain or severe shortness of breath, seek emergency care regardless of model output. '
                       'If someone collapses and is not breathing normally, call local emergency services '
                       'and follow dispatcher instructions for CPR/AED use.')
    st.divider()
    st.caption('Educational synthetic-data project · Submitted examples are not written to a patient database.')


if __name__ == '__main__':
    main()
