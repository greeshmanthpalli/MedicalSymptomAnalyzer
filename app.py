
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="MedInsight | Symptom Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. PROFESSIONAL CSS DESIGN
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F3F7FC;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102C4A 0%, #174D79 100%);
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.2);
}

.hero {
    background: linear-gradient(120deg, #102F50, #176AA5, #1599B8);
    padding: 35px 35px;
    border-radius: 20px;
    margin-bottom: 25px;
    color: white;
    box-shadow: 0 8px 25px rgba(16,65,105,0.15);
}

.hero h1 {
    color: white;
    font-size: 35px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #DCEEFF;
    font-size: 15px;
    margin-bottom: 0;
}

.section-heading {
    color: #173F63;
    font-size: 23px;
    font-weight: 750;
    margin-top: 15px;
    margin-bottom: 8px;
}

.section-subtitle {
    color: #6A8096;
    font-size: 14px;
    margin-bottom: 18px;
}

.info-card {
    background: white;
    padding: 20px;
    border: 1px solid #DFEAF4;
    border-radius: 15px;
    box-shadow: 0 3px 12px rgba(30,75,110,0.05);
    margin-bottom: 12px;
}

.info-card h4 {
    color: #174D79;
    margin: 0 0 8px 0;
}

.info-card p {
    color: #657B90;
    font-size: 14px;
    margin: 0;
}

.result-card {
    background: white;
    border: 1px solid #DCE8F3;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 14px rgba(20,70,110,0.05);
}

.result-title {
    color: #173F63;
    font-size: 18px;
    font-weight: 700;
}

.result-score {
    color: #1378B5;
    font-size: 25px;
    font-weight: 800;
}

.small-label {
    color: #71869A;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stButton > button {
    background: linear-gradient(90deg, #176AA5, #1599B8);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 45px;
}

.stButton > button:hover {
    background: #104F7F;
    color: white;
    border: none;
}

[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 13px;
    border: 1px solid #DFEAF4;
}

[data-testid="stExpander"] {
    background: white;
    border: 1px solid #DCE8F3;
    border-radius: 13px;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #176AA5, #20A7C8);
}

.footer {
    text-align: center;
    padding: 25px;
    color: #8094A8;
    font-size: 13px;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD RANDOM FOREST MODEL
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "disease_model.joblib"


@st.cache_resource
def load_model():

    bundle = joblib.load(MODEL_PATH)

    if isinstance(bundle, dict):
        model = bundle["model"]
        features = bundle["features"]

    else:
        model = bundle

        if not hasattr(model, "feature_names_in_"):
            raise ValueError(
                "Model does not contain feature names."
            )

        features = list(model.feature_names_in_)

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Model must support predict_proba()."
        )

    return model, list(features)


try:
    model, features = load_model()

except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.info(
        "Make sure disease_model.joblib is in the same folder as app.py."
    )
    st.stop()


# =====================================================
# 4. SYMPTOM DISPLAY NAMES
# =====================================================

SPECIAL_LABELS = {
    "itching": "Itching",
    "skin_rash": "Skin Rash",
    "nodal_skin_eruptions": "Nodal Skin Eruptions",
    "continuous_sneezing": "Continuous Sneezing",
    "shivering": "Shivering",
    "chills": "Chills",
    "joint_pain": "Joint Pain",
    "stomach_pain": "Stomach Pain",
    "acidity": "Acidity",
    "ulcers_on_tongue": "Ulcers on Tongue",
    "muscle_wasting": "Muscle Wasting",
    "vomiting": "Vomiting",
    "burning_micturition": "Burning Micturition",
    "spotting_urination": "Spotting Urination",
    "fatigue": "Fatigue",
    "weight_gain": "Weight Gain",
    "anxiety": "Anxiety",
    "cold_hands_and_feets": "Cold Hands and Feet",
    "mood_swings": "Mood Swings",
    "weight_loss": "Weight Loss",
    "restlessness": "Restlessness",
    "lethargy": "Lethargy",
    "patches_in_throat": "Patches in Throat",
    "irregular_sugar_level": "Irregular Sugar Level",
    "cough": "Cough",
    "high_fever": "High Fever",
    "breathlessness": "Breathlessness",
    "sweating": "Sweating",
    "dehydration": "Dehydration",
    "indigestion": "Indigestion",
    "headache": "Headache",
    "chest_pain": "Chest Pain",
    "dizziness": "Dizziness",
    "cramps": "Cramps",
    "bruising": "Bruising",
    "obesity": "Obesity",
    "swollen_legs": "Swollen Legs",
    "swollen_blood_vessels": "Swollen Blood Vessels",
    "puffy_face_and_eyes": "Puffy Face and Eyes",
    "enlarged_thyroid": "Enlarged Thyroid",
    "brittle_nails": "Brittle Nails",
    "swollen_extremeties": "Swollen Extremities",
    "excessive_hunger": "Excessive Hunger",
    "slurred_speech": "Slurred Speech",
    "knee_pain": "Knee Pain",
    "hip_joint_pain": "Hip Joint Pain",
    "muscle_weakness": "Muscle Weakness",
    "stiff_neck": "Stiff Neck",
    "swelling_joints": "Swelling of Joints",
    "movement_stiffness": "Movement Stiffness",
    "spinning_movements": "Spinning Movements",
    "loss_of_balance": "Loss of Balance",
    "unsteadiness": "Unsteadiness",
    "weakness_of_one_body_side": "Weakness of One Body Side",
    "loss_of_smell": "Loss of Smell",
    "bladder_discomfort": "Bladder Discomfort",
    "foul_smell_of_urine": "Foul Smell of Urine",
    "passage_of_gases": "Passage of Gases",
    "internal_itching": "Internal Itching",
    "depression": "Depression",
    "irritability": "Irritability",
    "muscle_pain": "Muscle Pain",
    "altered_sensorium": "Altered Sensorium",
    "red_spots_over_body": "Red Spots Over Body",
    "belly_pain": "Belly Pain",
    "abnormal_menstruation": "Abnormal Menstruation",
    "dischromic_patches": "Dischromic Patches",
    "watering_from_eyes": "Watering from Eyes",
    "increased_appetite": "Increased Appetite",
    "polyuria": "Frequent Urination",
    "family_history": "Family History",
    "mucoid_sputum": "Mucoid Sputum",
    "rusty_sputum": "Rusty Sputum",
    "visual_disturbances": "Visual Disturbances",
    "receiving_blood_transfusion": "Receiving Blood Transfusion",
    "receiving_unsterile_injections": "Receiving Unsterile Injections",
    "coma": "Coma",
    "stomach_bleeding": "Stomach Bleeding",
    "distention_of_abdomen": "Abdominal Distention",
    "history_of_alcohol_consumption": "History of Alcohol Consumption",
    "fluid_overload": "Fluid Overload",
    "blood_in_sputum": "Blood in Sputum",
    "prominent_veins_on_calf": "Prominent Veins on Calf",
    "palpitations": "Palpitations",
    "painful_walking": "Painful Walking",
    "pus_filled_pimples": "Pus-Filled Pimples",
    "blackheads": "Blackheads",
    "scurring": "Scurring",
    "skin_peeling": "Skin Peeling",
    "silver_like_dusting": "Silver-Like Dusting",
    "small_dents_in_nails": "Small Dents in Nails",
    "inflammatory_nails": "Inflammatory Nails",
    "blister": "Blister",
    "red_sore_around_nose": "Red Sore Around Nose",
    "yellow_crust_ooze": "Yellow Crust Ooze"
}


def display_symptom(symptom):
    return SPECIAL_LABELS.get(
        symptom,
        symptom.replace("_", " ").title()
    )


# =====================================================
# 5. DISEASE INFORMATION
# =====================================================

DISEASE_INFO = {
    "Drug Reaction": (
        "Medication-related condition",
        "An unwanted or harmful response that may occur after exposure to a medication."
    ),
    "GERD": (
        "Digestive system",
        "Gastroesophageal reflux disease involves repeated backflow of stomach contents into the esophagus."
    ),
    "Gastroenteritis": (
        "Digestive system",
        "Inflammation of the stomach and intestines, commonly associated with digestive symptoms."
    ),
    "Acne": (
        "Skin condition",
        "A common skin condition involving blocked hair follicles and oil glands."
    ),
    "Allergy": (
        "Immune system",
        "An immune-system response to a substance that is usually harmless to most people."
    ),
    "Arthritis": (
        "Musculoskeletal system",
        "A general term for conditions affecting joints, often involving pain or inflammation."
    ),
    "Bronchial Asthma": (
        "Respiratory system",
        "A long-term airway condition involving inflammation and narrowing of the breathing passages."
    ),
    "Cervical spondylosis": (
        "Musculoskeletal system",
        "Age-related changes affecting the discs and joints in the neck."
    ),
    "Chicken pox": (
        "Infectious disease",
        "A contagious infection caused by the varicella-zoster virus."
    ),
    "Chronic cholestasis": (
        "Liver and biliary system",
        "A condition involving reduced or blocked flow of bile from the liver."
    ),
    "Common Cold": (
        "Respiratory system",
        "A common viral infection affecting the upper respiratory tract."
    ),
    "Dengue": (
        "Infectious disease",
        "A viral infection transmitted through infected mosquitoes."
    ),
    "Diabetes": (
        "Endocrine system",
        "A group of metabolic conditions involving elevated blood glucose levels."
    ),
    "Dimorphic hemmorhoids(piles)": (
        "Digestive system",
        "Hemorrhoids are swollen veins in or around the lower rectum and anus."
    ),
    "Fungal infection": (
        "Infectious disease",
        "An infection caused by fungi, which can affect the skin or other body areas."
    ),
    "Heart attack": (
        "Cardiovascular emergency",
        "A medical emergency caused by blocked blood flow to part of the heart muscle."
    ),
    "Hepatitis B": (
        "Liver condition",
        "A viral infection caused by hepatitis B virus that affects the liver."
    ),
    "Hepatitis C": (
        "Liver condition",
        "A viral infection caused by hepatitis C virus that affects the liver."
    ),
    "Hepatitis D": (
        "Liver condition",
        "A liver infection caused by hepatitis D virus, which occurs only with hepatitis B infection."
    ),
    "Hepatitis E": (
        "Liver condition",
        "A viral infection caused by hepatitis E virus that affects the liver."
    ),
    "hepatitis A": (
        "Liver condition",
        "A contagious viral liver infection caused by hepatitis A virus."
    ),
    "Hypertension": (
        "Cardiovascular system",
        "A condition in which blood pressure remains persistently higher than recommended levels."
    ),
    "Hyperthyroidism": (
        "Endocrine system",
        "A condition in which the thyroid gland produces more thyroid hormones than the body needs."
    ),
    "Hypoglycemia": (
        "Metabolic condition",
        "A condition in which blood glucose falls below the normal range."
    ),
    "Hypothyroidism": (
        "Endocrine system",
        "A condition in which the thyroid gland does not produce enough thyroid hormones."
    ),
    "Impetigo": (
        "Skin condition",
        "A contagious bacterial skin infection affecting the outer layers of the skin."
    ),
    "Jaundice": (
        "Liver and biliary system",
        "Yellowing of the skin and eyes caused by increased bilirubin. Jaundice is a sign, not a disease itself."
    ),
    "Malaria": (
        "Infectious disease",
        "A parasitic infection transmitted through the bites of infected Anopheles mosquitoes."
    ),
    "Migraine": (
        "Neurological condition",
        "A neurological condition that can cause recurring headache episodes and other symptoms."
    ),
    "Osteoarthristis": (
        "Musculoskeletal system",
        "Osteoarthritis involves changes and breakdown of joint tissues over time."
    ),
    "Paralysis (brain hemorrhage)": (
        "Neurological emergency",
        "The dataset label refers to paralysis associated with bleeding in or around the brain."
    ),
    "Peptic ulcer diseae": (
        "Digestive system",
        "Peptic ulcers are open sores in the lining of the stomach or upper small intestine."
    ),
    "Pneumonia": (
        "Respiratory system",
        "An infection that causes inflammation in the air sacs of one or both lungs."
    ),
    "Psoriasis": (
        "Skin condition",
        "A chronic immune-mediated skin condition that causes inflammation and increased skin-cell growth."
    ),
    "Tuberculosis": (
        "Infectious disease",
        "An infectious disease caused by bacteria that most often affects the lungs."
    ),
    "Typhoid": (
        "Infectious disease",
        "A bacterial infection caused by Salmonella Typhi, usually spread through contaminated food or water."
    ),
    "Urinary tract infection": (
        "Urinary system",
        "An infection affecting parts of the urinary system, including the bladder or urethra."
    ),
    "Varicose veins": (
        "Cardiovascular system",
        "Enlarged, twisted veins that commonly occur in the legs."
    ),
    "(vertigo) Paroymsal  Positional Vertigo": (
        "Inner-ear condition",
        "Benign paroxysmal positional vertigo (BPPV) causes brief spinning sensations triggered by certain head movements."
    ),
    "Alcoholic hepatitis": (
        "Liver condition",
        "Liver inflammation associated with prolonged or heavy alcohol use."
    ),
    "AIDS": (
        "Immune system condition",
        "Acquired immunodeficiency syndrome is an advanced stage of HIV infection."
    )
}


# =====================================================
# 6. HEADER
# =====================================================

st.markdown("""
<div class="hero">
    <h1>🩺 MedInsight</h1>
    <p>Medical Symptom Analyzer</p>
    <p>Understand symptom patterns through an educational
    Random Forest machine-learning demonstration.</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Educational demonstration only. This application does not "
    "diagnose diseases or replace a qualified healthcare professional."
)


# =====================================================
# 7. SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("## 🩺 MedInsight")

    st.caption("MEDICAL SYMPTOM ANALYZER")

    st.divider()

    st.markdown("### Technology Stack")

    st.write("🐍 Python")
    st.write("🌲 Random Forest")
    st.write("📊 Scikit-learn")
    st.write("🌐 Streamlit")
    st.write("📁 Pandas and Joblib")

    st.divider()

    with st.expander("ℹ️ About Project"):

        st.write("**Project:** Medical Symptom Analyzer")

        st.write(
            "An educational machine-learning application "
            "that demonstrates symptom-based classification."
        )

        st.markdown("**Objectives**")

        st.write("• Select multiple symptoms")
        st.write("• Record individual symptom duration")
        st.write("• Display three highest-ranked model outputs")
        st.write("• Present general disease information")

        st.write("**Algorithm:** Random Forest")
        st.write("**Language:** Python")
        st.write("**Framework:** Streamlit")

    st.divider()

    st.caption(
        "For educational use only. Not a diagnostic system."
    )


# =====================================================
# 8. RESET BUTTON
# =====================================================

reset_col, spacer_col = st.columns([1, 4])

with reset_col:

    if st.button("↻ Reset All", key="reset_form"):

        keys_to_clear = [
            key for key in list(st.session_state.keys())
            if key == "selected_symptoms"
            or key == "symptom_search"
            or key.startswith("duration_")
        ]

        for key in keys_to_clear:
            del st.session_state[key]

        st.rerun()


# =====================================================
# 9. SYMPTOM SELECTION
# =====================================================

st.markdown(
    '<div class="section-heading">01 | Symptom Selection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Search and select the symptoms you are experiencing.</div>',
    unsafe_allow_html=True
)

search_text = st.text_input(
    "Search symptoms",
    placeholder="Type a symptom, e.g. headache or fever...",
    key="symptom_search"
)

available_symptoms = [
    feature for feature in features
    if search_text.lower() in display_symptom(feature).lower()
]

selected_symptoms = st.multiselect(
    "Select symptoms",
    options=available_symptoms,
    format_func=display_symptom,
    placeholder="Choose symptoms from the list",
    key="selected_symptoms"
)

if search_text and not available_symptoms:
    st.warning("No matching symptom found.")


# =====================================================
# 10. SELECTED SYMPTOMS DISPLAY
# =====================================================

if selected_symptoms:

    st.markdown("### Selected Symptoms")

    cols = st.columns(3)

    for index, symptom in enumerate(selected_symptoms):

        with cols[index % 3]:

            st.markdown(
                f"""
                <div class="info-card">
                    <h4>✓ {display_symptom(symptom)}</h4>
                    <p>Selected symptom</p>
                </div>
                """,
                unsafe_allow_html=True
            )


# =====================================================
# 11. INDIVIDUAL DURATION INPUT
# =====================================================

duration_values = {}

if selected_symptoms:

    st.markdown(
        '<div class="section-heading">02 | Symptom Duration</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Enter the duration separately for each selected symptom.</div>',
        unsafe_allow_html=True
    )

    duration_cols = st.columns(2)

    for index, symptom in enumerate(selected_symptoms):

        with duration_cols[index % 2]:

            duration_values[symptom] = st.number_input(
                f"{display_symptom(symptom)} — Duration (days)",
                min_value=0,
                max_value=3650,
                value=1,
                step=1,
                key=f"duration_{symptom}"
            )

    st.caption(
        "Enter 0 if the symptom started today. "
        "Duration is recorded for the summary only."
    )


# =====================================================
# 12. ANALYZE BUTTON
# =====================================================

st.divider()

analyze = st.button(
    "🔍 Analyze Symptoms",
    type="primary",
    use_container_width=True
)


# =====================================================
# 13. RANDOM FOREST PREDICTION
# =====================================================

if analyze:

    if len(selected_symptoms) < 2:

        st.warning(
            "Please select at least two symptoms to continue."
        )

    elif "chest_pain" in selected_symptoms:

        st.error(
            "Chest pain can have serious causes. Please seek prompt "
            "medical assessment. If severe, persistent, or accompanied "
            "by breathing difficulty, sweating, or fainting, contact "
            "emergency services. No model prediction will be displayed."
        )

    else:

        input_data = pd.DataFrame(
            [
                [
                    1 if feature in selected_symptoms else 0
                    for feature in features
                ]
            ],
            columns=features
        )

        try:

            probabilities = model.predict_proba(input_data)[0]

            classes = model.classes_

            results = sorted(
                zip(classes, probabilities),
                key=lambda item: item[1],
                reverse=True
            )

            top_three = results[:3]

            # -----------------------------------------
            # RESULTS HEADER
            # -----------------------------------------

            st.divider()

            st.markdown(
                '<div class="section-heading">03 | Analysis Results</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-subtitle">Top three outputs from the Random Forest model.</div>',
                unsafe_allow_html=True
            )

            st.success(
                "Analysis completed successfully."
            )

            st.caption(
                "Scores are uncalibrated model classification scores, "
                "not medically validated probabilities."
            )

            # -----------------------------------------
            # TOP THREE CARDS
            # -----------------------------------------

            for rank, (disease, score) in enumerate(
                top_three, start=1
            ):

                percentage = float(score) * 100

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="small-label">
                            RANK {rank}
                        </div>
                        <div class="result-title">
                            {disease}
                        </div>
                        <div class="result-score">
                            {percentage:.2f}%
                        </div>
                        <div class="small-label">
                            MODEL CLASSIFICATION SCORE
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(
                    min(max(float(score), 0.0), 1.0)
                )

            # -----------------------------------------
            # DISEASE INFORMATION
            # -----------------------------------------

            st.divider()

            st.markdown(
                '<div class="section-heading">04 | Disease Information</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-subtitle">General educational information about the highest-ranked dataset labels.</div>',
                unsafe_allow_html=True
            )

            for rank, (disease, score) in enumerate(
                top_three, start=1
            ):

                category, description = DISEASE_INFO.get(
                    disease,
                    (
                        "General condition",
                        "General information for this dataset label is not currently available."
                    )
                )

                with st.expander(
                    f"#{rank} {disease} — {float(score) * 100:.2f}%",
                    expanded=True
                ):

                    st.markdown(f"**Category:** {category}")

                    st.write(description)

                    st.caption(
                        "Educational information only. This does not "
                        "confirm that you have this condition."
                    )

            # -----------------------------------------
            # SYMPTOM SUMMARY
            # -----------------------------------------

            st.divider()

            st.markdown(
                '<div class="section-heading">05 | Symptom Summary</div>',
                unsafe_allow_html=True
            )

            summary_rows = []

            for symptom in selected_symptoms:

                days = duration_values[symptom]

                duration_text = (
                    "Started today"
                    if days == 0
                    else f"{days} day(s)"
                )

                summary_rows.append({
                    "Symptom": display_symptom(symptom),
                    "Duration": duration_text
                })

            summary_df = pd.DataFrame(summary_rows)

            st.dataframe(
                summary_df,
                hide_index=True,
                use_container_width=True
            )

            st.warning(
                "Symptom duration is displayed for information only "
                "and was not used by the Random Forest model. Model "
                "scores do not establish the likelihood of having a "
                "disease. Consult a qualified healthcare professional "
                "for assessment."
            )

        except Exception as error:

            st.error(
                f"An error occurred during analysis: {error}"
            )


# =====================================================
# 14. FOOTER
# =====================================================

st.divider()

st.markdown("""
<div class="footer">
    <h3 style="color:#176AA5;">MedInsight</h3>
    <p>Medical Symptom Analyzer</p>
    <p>Powered by Python, Streamlit and Random Forest</p>
    <p>Educational demonstration only — Not a medical diagnostic tool.</p>
</div>
""", unsafe_allow_html=True)
