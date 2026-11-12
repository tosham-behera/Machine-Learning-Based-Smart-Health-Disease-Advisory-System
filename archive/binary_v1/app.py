"""Streamlit demonstration: streamlit run app.py."""

import pickle

import streamlit as st

from heart_guidance import CONDITION_GUIDE, SOURCES

from predict import (
    ADVISORY_SOURCE,
    DISCLAIMER,
    MODEL_SCORE_BANDS,
    load_artifacts,
    predict_patient,
)


@st.cache_resource
def cached_artifacts():
    """Load once per app process; never train a model in the user interface."""
    return load_artifacts()


def main():
    st.set_page_config(
        page_title="Smart Health Advisory System",
        page_icon="🫀",
        layout="centered",
    )
    st.title("Smart Health Advisory System")
    st.subheader("Heart Disease Prediction Using Machine Learning")
    st.write(
        "Explore an educational machine learning project using synthetic examples. "
        "Enter an example below to see the saved model's binary classification and "
        "general preventive health suggestions."
    )
    st.warning(DISCLAIMER)
    try:
        model, metadata = cached_artifacts()
    except (OSError, ValueError, ImportError, EOFError, pickle.UnpicklingError) as error:
        st.error(f"The saved model could not be loaded: {error}")
        st.code("python train_model.py", language="bash")
        st.info("Run this command from the project folder, then restart Streamlit.")
        st.stop()

    st.caption(
        f"Saved model: {metadata['model_name']} · "
        f"{len(metadata['feature_names'])} input features · Synthetic data only"
    )
    st.markdown("### Example information")
    st.write(
        "The fields and allowed values come from the trained model's saved schema. "
        "Numerical limits describe the synthetic training data; they are not medical "
        "reference ranges. Use Unknown or clear a numerical field for missing data."
    )
    with st.form("prediction_form"):
        columns = st.columns(2)
        patient = {}
        for position, name in enumerate(metadata["feature_names"]):
            specification = metadata["input_schema"][name]
            label = specification.get("label", name.replace("_", " "))
            with columns[position % len(columns)]:
                if specification["kind"] == "numeric":
                    patient[name] = st.number_input(
                        label,
                        min_value=float(specification["min"]),
                        max_value=float(specification["max"]),
                        value=float(specification["default"]),
                        step=float(specification.get("step", 1)),
                        key=f"input_{name}",
                    )
                elif specification["kind"] == "binary":
                    choices = [0, 1, None]
                    patient[name] = st.selectbox(
                        label,
                        choices,
                        index=choices.index(specification["default"]),
                        format_func=lambda value: {
                            0: "No (0)",
                            1: "Yes (1)",
                            None: "Unknown",
                        }[value],
                        key=f"input_{name}",
                    )
                else:
                    choices = specification["categories"]
                    patient[name] = st.selectbox(
                        label,
                        choices,
                        index=choices.index(specification["default"]),
                        key=f"input_{name}",
                    )
        units_confirmed = st.checkbox(
            "My entered measurements use kg, cm, mmHg and mg/dL (cholesterol).",
            help="Enable optional BMI, blood-pressure and cholesterol guidance. "
            "The original synthetic dataset's units have not been independently verified.",
            key="advisory_units_confirmed",
        )
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        try:
            result = predict_patient(
                patient, model=model, metadata=metadata,
                advisory_units_confirmed=units_confirmed,
            )
        except ValueError as error:
            st.error(f"Please check your inputs: {error}")
        else:
            st.divider()
            st.subheader("Experimental model result")
            st.info(result["message"])
            probability = result["estimated_probability"]
            if probability is not None:
                score_column, category_column = st.columns(2)
                score_column.metric(
                    "Estimated model probability of class 1",
                    f"{probability:.1%}",
                )
                band = result["model_score_band"]
                category_column.metric("Illustrative model-score category", band["label"])
                st.write(band["message"])
                st.caption(
                    "These four equal score bands are illustrative and have not been "
                    "clinically validated. They do not measure your cardiovascular "
                    "risk or disease severity. Low does not rule out disease; "
                    "Very high does not mean severe disease."
                )
                with st.expander("How the model-score categories work"):
                    st.table([
                        {"Category": label, "Class-1 model score": interval}
                        for _, label, interval in MODEL_SCORE_BANDS
                    ])
                    st.caption(
                        "Categories use the unrounded score. They summarize the "
                        "existing probability; the model still predicts only class 0 or 1."
                    )
            for warning in result["warnings"]:
                st.warning(warning)
            st.subheader("Heart-health topics related to your inputs")
            st.info(
                "Educational guidance based on reported factors — not predicted diseases. "
                "Several topics may apply together. These rules do not estimate your risk "
                "of a specific disease, heart attack or cardiac arrest."
            )
            guidance = result["condition_guidance"]
            for note in guidance["notes"]:
                st.caption(note)
            if guidance["calculated_bmi"] is not None:
                st.write(f"Calculated adult BMI: **{guidance['calculated_bmi']:.1f} kg/m²** (screening only)")
            if not guidance["topics"]:
                st.write(
                    "No topics matched the information supplied under these limited rules. "
                    "Unknown values are not treated as absent risk factors. "
                    "This does not mean that disease or risk is absent."
                )
            for topic in guidance["topics"]:
                with st.container(border=True):
                    st.markdown(f"#### {topic['title']}")
                    st.write("Why this topic appears:")
                    for factor in topic["matched_factors"]:
                        st.markdown(f"- {factor}")
                    st.write(topic["explanation"])
                    st.markdown("**Two practical steps**")
                    for action in topic["actions"]:
                        st.markdown(f"- {action}")
                    st.markdown(f"[Read the medical background]({topic['source']})")
            with st.expander("How these conditions relate to heart attacks and cardiac arrest"):
                st.write(
                    "General background, not a diagnosis. These conditions have lifestyle "
                    "and non-lifestyle causes. A heart attack involves interrupted blood "
                    "supply; cardiac arrest is a sudden loss of effective pumping."
                )
                st.table(list(CONDITION_GUIDE))
                st.caption("The dataset contains no labels for these five conditions. "
                           "Specific diagnoses require clinical assessment and appropriate tests.")
                st.markdown("Sources: " + " · ".join(
                    f"[{name}]({url})" for name, url in SOURCES.items()
                ))
            st.caption(
                "For possible heart-attack symptoms such as new chest pain or severe "
                "shortness of breath, seek emergency care regardless of the model score. "
                "If someone collapses and is not breathing normally, call local emergency "
                "services and follow dispatcher instructions for CPR/AED use."
            )
            st.markdown("### General health advisory")
            st.caption(
                "These simple rules use the information you entered. They are "
                "separate from the machine learning prediction."
            )
            for advice in result["advisory"]:
                st.markdown(f"- {advice}")
            st.markdown(
                f"Source: [World Health Organization — cardiovascular diseases]"
                f"({ADVISORY_SOURCE})"
            )
            st.warning("Experimental result only. " + DISCLAIMER)

    st.divider()
    st.caption(
        "College demonstration · No clinical validation · "
        "The app does not write submitted examples to a patient database."
    )


if __name__ == "__main__":
    main()
