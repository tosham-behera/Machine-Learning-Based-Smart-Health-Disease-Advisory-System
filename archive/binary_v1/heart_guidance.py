"""Educational topic matching from reported inputs, separate from the ML model.

Sources reviewed 2026-10-23. Matches are not diagnoses or risk estimates.
Only confirmed measurement units enable numerical screening rules. Never use
imputed model inputs to infer a person's reported health factors.
"""

SOURCES = {
    "CAD": "https://www.nhlbi.nih.gov/health/coronary-heart-disease/causes",
    "BP": "https://medlineplus.gov/ency/article/000163.htm",
    "Heart failure": "https://www.nhlbi.nih.gov/health/heart-failure/causes",
    "Arrhythmias": "https://www.nhlbi.nih.gov/health/arrhythmias/causes",
    "Cardiomyopathy": "https://www.nhlbi.nih.gov/health/cardiomyopathy/causes",
    "Heart muscle": "https://www.nhlbi.nih.gov/health/cardiomyopathy",
    "Cardiac arrest": "https://www.nhlbi.nih.gov/health/cardiac-arrest",
    "Arrest causes": "https://www.heart.org/en/health-topics/cardiac-arrest/causes-of-cardiac-arrest",
    "BMI": "https://www.cdc.gov/bmi/adult-calculator/bmi-categories.html",
    "BP readings": "https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings",
    "Cholesterol": "https://medlineplus.gov/cholesterollevelswhatyouneedtoknow.html",
    "Healthy living": "https://www.nhlbi.nih.gov/health/heart-healthy-living",
}

# The table explains conditions generally; it does not describe the user's heart.
CONDITION_GUIDE = (
    {
        "Condition": "Coronary artery disease (CAD)",
        "Contributing factors": "Smoking, abnormal cholesterol, high blood pressure, diabetes; diet and inactivity also matter.",
        "Heart attack connection": "Plaque-related clots can block a coronary artery and cause a heart attack.",
        "Cardiac arrest connection": "A heart attack can trigger a dangerous rhythm and cardiac arrest; this is not inevitable.",
    },
    {
        "Condition": "Hypertensive heart disease",
        "Contributing factors": "Long-standing high blood pressure; excess sodium and excess weight can contribute to high BP.",
        "Heart attack connection": "High BP can damage arteries and increase heart attack risk.",
        "Cardiac arrest connection": "Heart damage from long-standing high BP can contribute to heart failure and dangerous rhythms.",
    },
    {
        "Condition": "Heart failure",
        "Contributing factors": "Coronary disease, high blood pressure, diabetes, obesity, valve disease and other causes.",
        "Heart attack connection": "A heart attack can cause heart failure, but heart failure also occurs without a prior attack.",
        "Cardiac arrest connection": "Some people with heart failure develop dangerous rhythms that can lead to arrest.",
    },
    {
        "Condition": "Heart arrhythmias",
        "Contributing factors": "Smoking, heavy alcohol use, other heart conditions, inherited conditions and some medicines.",
        "Heart attack connection": "An arrhythmia is not an artery blockage. A heart attack can cause an arrhythmia.",
        "Cardiac arrest connection": "Certain rhythms, including ventricular fibrillation, can cause arrest. Many arrhythmias do not.",
    },
    {
        "Condition": "Cardiomyopathy",
        "Contributing factors": "Inherited or acquired heart-muscle disease; heavy alcohol use is one possible contributor.",
        "Heart attack connection": "Heart-muscle disease is distinct from an artery-blockage heart attack; the conditions can coexist.",
        "Cardiac arrest connection": "Some forms can cause dangerous rhythms and sudden cardiac arrest.",
    },
)

def condition_guidance(patient, *, units_confirmed=False):
    """Match multiple educational topics using already-validated, original inputs.

    Confirmed units: kg, cm, mmHg, and mg/dL for total cholesterol. The source
    dataset's units remain unverified. The app's adult schema is age 30-79.
    """
    factors = {}
    reported = (
        ("smoking", patient.get("Smoking") == "Current", "Current smoking reported"),
        ("alcohol", patient.get("Alcohol_Intake") == "High", "Alcohol intake reported as High (amount and duration unknown)"),
        ("inactive", patient.get("Physical_Activity") == "Sedentary", "Sedentary activity reported"),
        ("diet", patient.get("Diet") == "Unhealthy", "Diet reported as Unhealthy"),
        ("bp", patient.get("Hypertension") == 1, "History of hypertension reported"),
        ("diabetes", patient.get("Diabetes") == 1, "History of diabetes reported"),
        ("lipids", patient.get("Hyperlipidemia") == 1, "History of hyperlipidemia reported"),
        ("attack", patient.get("Previous_Heart_Attack") == 1, "Previous heart attack reported"),
    )
    for key, present, description in reported:
        if present:
            factors[key] = description

    bmi = None
    notes = []
    if units_confirmed:
        weight, height, age = (patient.get(name) for name in ("Weight", "Height", "Age"))
        if weight is not None and height is not None and age is not None and age >= 20:
            bmi = weight / (height / 100) ** 2
            if bmi >= 30:
                factors["bmi"] = f"Calculated BMI {bmi:.1f} kg/m²: adult obesity screening range"
            notes.append("BMI is a screening measure, not a diagnosis; it does not distinguish fat from muscle.")
        else:
            notes.append("Adult BMI screening needs known adult age, weight and height.")
        systolic, diastolic = patient.get("Systolic_BP"), patient.get("Diastolic_BP")
        if ((systolic is not None and systolic >= 130)
                or (diastolic is not None and diastolic >= 80)):
            factors["bp_reading"] = "Entered BP reaches the AHA high-BP range (systolic ≥130 or diastolic ≥80 mmHg)"
            notes.append("One BP reading cannot confirm hypertension or hypertensive heart disease; arrange repeat measurements and review.")
        cholesterol = patient.get("Cholesterol_Total")
        if cholesterol is not None and cholesterol >= 200:
            factors["cholesterol"] = f"Total cholesterol {cholesterol:g} mg/dL: above the general desirable adult range"
            notes.append("Total cholesterol alone is incomplete; LDL, HDL and a clinical assessment are needed.")
    else:
        notes.append("Numerical screening was skipped. Confirm kg, cm, mmHg and mg/dL in the form to include BMI, BP and cholesterol readings.")

    quit_smoking = "Ask for support to quit smoking and avoid second-hand smoke."
    activity = "Discuss gradual physical activity and a sustainable weight-management plan suited to your health."
    diet = "Choose vegetables, fruit, whole grains and unsaturated fats; reduce foods high in saturated fat."
    alcohol = "Discuss reducing alcohol safely with a clinician; heavy daily drinking may need medical support to avoid withdrawal."
    bp = "Arrange repeat BP checks and discuss management with a healthcare professional."
    definitions = (
        ("cad", "Coronary artery disease (CAD)",
         ("smoking", "lipids", "cholesterol", "bp", "bp_reading", "diabetes", "bmi", "inactive", "diet", "attack"),
         "These factors are relevant to coronary health; they do not show whether an artery is blocked.",
         [quit_smoking if "smoking" in factors else diet,
          "Discuss a full cholesterol panel and your overall cardiovascular risk with a clinician."], "CAD"),
        ("hypertensive", "Blood-pressure health / hypertensive heart disease",
         ("bp", "bp_reading"),
         "Long-standing hypertension can affect the heart. These inputs do not demonstrate heart damage.",
         [bp, "Reduce salt and highly processed foods; choose an activity plan appropriate for your health."], "BP"),
        ("failure", "Heart failure prevention",
         ("bp", "bp_reading", "diabetes", "bmi", "attack"),
         "These factors can contribute to heart failure, but the app has no examination or heart-function tests.",
         [activity if "bmi" in factors else "Discuss management of your reported conditions with your clinician.",
          "Keep follow-up appointments for blood pressure, diabetes or previous heart disease as applicable."], "Heart failure"),
        ("rhythm", "Heart rhythm / arrhythmias",
         ("smoking", "alcohol", "bp", "diabetes", "bmi", "attack"),
         "These factors can be associated with arrhythmias. A pulse value cannot identify a rhythm disorder; an ECG may be needed.",
         [quit_smoking if "smoking" in factors else activity,
          alcohol if "alcohol" in factors else "Discuss recurrent palpitations with a clinician."], "Arrhythmias"),
        ("muscle", "Heart muscle / cardiomyopathy",
         ("alcohol",),
         "Heavy alcohol use can damage heart muscle. The High label has no defined dose or duration; cardiomyopathy cannot be inferred.",
         [alcohol, "Discuss your alcohol history and any heart symptoms with a clinician."], "Cardiomyopathy"),
    )
    topics = []
    for key, title, relevant, explanation, actions, source in definitions:
        matches = [factors[factor] for factor in relevant if factor in factors]
        if matches:
            topics.append({"id": key, "title": title, "matched_factors": matches,
                           "explanation": explanation, "actions": actions,
                           "source": SOURCES[source]})
    return {"topics": topics, "calculated_bmi": bmi, "notes": notes,
            "units_confirmed": units_confirmed}
