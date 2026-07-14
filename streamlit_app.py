import os
import numpy as np
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CVD Risk Assessment in Diabetic Patients",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The model and scaler files are expected to be in the same folder as this app.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RF_PATH = os.path.join(BASE_DIR, "rf.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ---------------------------------------------------------------------------
# Load model and scaler (cached so they are only loaded once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model and scaler...")
def load_model_and_scaler():
    if not os.path.exists(RF_PATH):
        raise FileNotFoundError(
            f"Could not find model file: {RF_PATH}. "
            "Please make sure rf.pkl is placed in the same directory as this app."
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Could not find scaler file: {SCALER_PATH}. "
            "Please make sure scaler.pkl is placed in the same directory as this app."
        )
    model = joblib.load(RF_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


model, scaler = load_model_and_scaler()

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
# The original CSV uses readable display names; the saved model/scaler use
# lower-case / abbreviated names. Keep the display names for the UI and map
# them to the model names internally.
FEATURE_MAP = {
    "Age": "age",
    "Retire": "retire",
    "Self Reported Health Status": "srh",
    "ADL Score": "adlab_c",
    "High Intensity Exercise": "vgact_c",
    "Moderate Intensity Exercise": "mdact_c",
    "Drinking": "drinkl",
    "Hypertension": "hibpe",
    "Dyslipidemia": "dyslipe",
    "Lung disease": "lunge",
    "Liver diseases": "livere",
    "Kidney disease": "kidneye",
    "Gastric disease": "digeste",
    "Memory disorders": "memrye",
    "Hospitalization within one year": "hospital",
    "Chest pain": "da042s6",
}

DISPLAY_FEATURES = list(FEATURE_MAP.keys())
MODEL_FEATURES = list(FEATURE_MAP.values())

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Prediction of Cardiovascular Disease in Middle-Aged and Elderly Patients with Diabetes Mellitus")
st.markdown(
    """
    This app, built on CHARLS 2020 data with a Random Forest model, assesses the
    cardiovascular disease risk probability of middle-aged and elderly diabetic patients.

    Enter the relevant clinical and health information in the left panel, then click **Predict**.
    """
)

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Patient Information")

    # --- Demographics ---
    st.subheader("Demographics")
    age = st.slider("How old are you?", min_value=45, max_value=85, value=60)
    retire = st.selectbox(
        "Have you retired?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        index=0,
    )

    # --- Health status ---
    st.subheader("Health Status")
    srh = st.selectbox(
        "Self-Reported Health Status Score (1=Very Good, 5=Very Poor)",
        options=[1, 2, 3, 4, 5],
        index=2,
        help="1 = Very Good, 2 = Good, 3 = Fair, 4 = Poor, 5 = Very Poor",
    )
    adlab_c = st.selectbox(
        "ADL Score: daily living activities with difficulty (0–6)",
        options=list(range(7)),
        index=0,
        help=(
            "Daily living activities include: using the toilet, feeding yourself, "
            "dressing yourself, controlling bowel and bladder movements, getting in and out of bed, "
            "bathing yourself."
        ),
    )

    # --- Lifestyle ---
    st.subheader("Lifestyle")
    vgact_c = st.selectbox(
        "High Intensity Exercise",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        index=0,
    )
    mdact_c = st.selectbox(
        "Moderate Intensity Exercise",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        index=0,
    )
    drinkl = st.selectbox(
        "Drinking",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        index=0,
    )

    # --- Medical conditions ---
    st.subheader("Medical Conditions")
    hibpe = st.selectbox(
        "Has any doctor ever told you that you have hypertension?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        index=0,
    )
    dyslipe = st.selectbox(
        (
            "Have you been diagnosed with Dyslipidemia (elevation of low-density lipoprotein, "
            "triglycerides, and total cholesterol, or a low high-density lipoprotein level) by a doctor?"
        ),
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        index=0,
    )
    lunge = st.selectbox(
        "Have you been diagnosed with Lung disease by a doctor?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        index=0,
    )
    livere = st.selectbox(
        "Have you been diagnosed with Liver disease by a doctor?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        index=0,
    )
    kidneye = st.selectbox(
        "Have you been diagnosed with Kidney disease (except for tumor or cancer) by a doctor?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        index=0,
    )
    digeste = st.selectbox(
        "Have you been diagnosed with Gastric disease by a doctor?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        index=0,
    )
    memrye = st.selectbox(
        "Have you been diagnosed with Memory disorders by a doctor?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        index=0,
    )

    # --- Healthcare utilization ---
    st.subheader("Healthcare Utilization")
    hospital = st.selectbox(
        "Have you received inpatient care in the past year?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        index=0,
    )
    da042s6 = st.selectbox(
        "Do you often have chest pain?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        index=0,
    )

# ---------------------------------------------------------------------------
# Assemble inputs
# ---------------------------------------------------------------------------
values = [
    age, retire, srh, adlab_c, vgact_c, mdact_c, drinkl,
    hibpe, dyslipe, lunge, livere, kidneye, digeste, memrye,
    hospital, da042s6,
]

input_df = pd.DataFrame([values], columns=DISPLAY_FEATURES)
input_model = input_df.rename(columns=FEATURE_MAP)
original_values = input_df.copy()

# Apply the same scaler used during model training
input_scaled = scaler.transform(input_model)
input_scaled_df = pd.DataFrame(input_scaled, columns=MODEL_FEATURES)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if st.button("Predict", use_container_width=True):
    predicted_class = model.predict(input_scaled_df)[0]
    predicted_proba = model.predict_proba(input_scaled_df)[0]
    disease_proba = float(predicted_proba[1])

    # ---- Result cards ----
    st.subheader("Predicted Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("CVD Probability", f"{disease_proba * 100:.1f}%")
    c2.metric(
        "Risk Level",
        "High" if disease_proba >= 0.5 else "Low",
    )
    c3.metric("Predicted Class", "CVD" if predicted_class == 1 else "No CVD")

    # ---- Progress bar ----
    st.progress(disease_proba, text=f"CVD Probability: {disease_proba * 100:.1f}%")

    # ---- Interpretation ----
    if predicted_class == 1:
        st.error(
            f"Based on the model assessment, you have a **high risk** of cardiovascular disease, "
            f"with a predicted probability of **{disease_proba * 100:.1f}%**.",
        )
    else:
        st.success(
            f"Based on the model assessment, you have a **low risk** of cardiovascular disease, "
            f"with a predicted probability of **{disease_proba * 100:.1f}%**.",
        )

    # ---- Probability table ----
    proba_df = pd.DataFrame(
        {
            "Outcome": ["No CVD", "CVD"],
            "Probability": [predicted_proba[0], predicted_proba[1]],
        }
    )
    st.dataframe(
        proba_df,
        column_config={
            "Probability": st.column_config.ProgressColumn(
                "Probability",
                format="%.3f",
                min_value=0.0,
                max_value=1.0,
            )
        },
        hide_index=True,
        use_container_width=True,
    )

    # ---- SHAP force plot explanation ----
    st.subheader("SHAP Force Plot Explanation")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled_df)

        # SHAP versions differ in output shape: list of two arrays vs. 3D array.
        if isinstance(shap_values, list):
            shap_values_class = shap_values[1][0]
        else:
            shap_values_class = shap_values[0, :, 1]

        # Base value for the positive class
        if isinstance(explainer.expected_value, (list, tuple, np.ndarray)):
            base_value = explainer.expected_value[1]
        else:
            base_value = explainer.expected_value

        plt.figure(figsize=(14, 5))
        shap.force_plot(
            base_value,
            shap_values_class,
            features=original_values.values[0],
            feature_names=DISPLAY_FEATURES,
            matplotlib=True,
            show=False,
        )
        plt.tight_layout()
        st.pyplot(plt.gcf())
    except Exception as e:
        st.warning(f"SHAP explanation could not be generated: {e}")
