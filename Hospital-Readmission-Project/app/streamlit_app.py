"""
Hospital Readmission Prediction – Streamlit Web Application
============================================================
Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f, #2C7BB6);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
    }
    .metric-card {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 4px solid #2C7BB6;
    }
    .risk-high {
        background: #fff0f0;
        border-left: 4px solid #D7191C;
        padding: 15px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        color: #D7191C;
    }
    .risk-low {
        background: #f0fff0;
        border-left: 4px solid #1A9641;
        padding: 15px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        color: #1A9641;
    }
    .section-header {
        color: #1e3a5f;
        font-size: 20px;
        font-weight: bold;
        border-bottom: 2px solid #2C7BB6;
        padding-bottom: 5px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 Hospital Readmission Prediction System</h1>
    <p>AI-powered tool to predict 30-day hospital readmission risk</p>
    <small>B.Tech CSE (AI/ML) | GNA University, Phagwara | By Mahdi Nawaz</small>
</div>
""", unsafe_allow_html=True)

# ── Load model artifacts ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model    = joblib.load(os.path.join(base, "models", "best_model.pkl"))
    scaler   = joblib.load(os.path.join(base, "models", "scaler.pkl"))
    encoders = joblib.load(os.path.join(base, "models", "encoders.pkl"))
    features = joblib.load(os.path.join(base, "models", "feature_names.pkl"))
    return model, scaler, encoders, features

try:
    model, scaler, encoders, feature_names = load_artifacts()
    st.sidebar.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"Could not load model: {e}. Please run ml_pipeline.py first.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – Navigation
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.image("https://img.icons8.com/color/96/hospital.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🔮 Predict Readmission", "📊 Data Analytics", "ℹ️ About the Project"]
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict Readmission":
    st.markdown('<div class="section-header">Patient Information</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Patient Details")
        age    = st.slider("Age", 18, 90, 55)
        gender = st.selectbox("Gender", ["Male", "Female"])
        blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])

    with col2:
        st.subheader("🏨 Admission Details")
        primary_diagnosis = st.selectbox("Primary Diagnosis", [
            "Diabetes", "Hypertension", "Heart Failure", "COPD",
            "Pneumonia", "Sepsis", "Renal Failure", "Stroke"
        ])
        admission_type = st.selectbox("Admission Type", ["Emergency", "Elective", "Urgent"])
        discharge_disposition = st.selectbox("Discharge Disposition", [
            "Home", "Skilled Nursing", "Rehab", "Another Hospital", "Expired"
        ])
        length_of_stay = st.slider("Length of Stay (days)", 1, 30, 5)

    with col3:
        st.subheader("🏥 Clinical & Hospital Info")
        num_procedures          = st.slider("Number of Procedures", 0, 8, 2)
        num_medications         = st.slider("Number of Medications", 1, 20, 5)
        num_previous_admissions = st.slider("Previous Admissions", 0, 10, 1)
        comorbidity_count       = st.slider("Number of Comorbidities", 0, 5, 1)
        hospital_type = st.selectbox("Hospital Type", ["Government", "Private", "Trust"])
        hospital_size = st.selectbox("Hospital Size", ["Small", "Medium", "Large"])
        num_beds      = st.slider("Hospital Beds", 20, 800, 200)
        insurance_type = st.selectbox("Insurance Type", ["Ayushman Bharat", "Private Insurance", "Self Pay"])
        total_charges    = st.number_input("Total Charges (₹)", 5000, 500000, 50000, step=5000)
        insurance_covered = st.number_input("Insurance Covered (₹)", 0, 500000, 20000, step=5000)

    # ── Encode & Predict ──────────────────────────────────────────────────────
    if st.button("🔮 Predict Readmission Risk", type="primary", use_container_width=True):
        raw_input = {
            "age": age,
            "gender": gender,
            "blood_group": blood_group,
            "primary_diagnosis": primary_diagnosis,
            "admission_type": admission_type,
            "length_of_stay": length_of_stay,
            "discharge_disposition": discharge_disposition,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "num_previous_admissions": num_previous_admissions,
            "hospital_type": hospital_type,
            "hospital_size": hospital_size,
            "num_beds": num_beds,
            "insurance_type": insurance_type,
            "total_charges": total_charges,
            "insurance_covered": insurance_covered,
            "comorbidity_count": comorbidity_count,
            "out_of_pocket": max(0, total_charges - insurance_covered),
        }

        input_df = pd.DataFrame([raw_input])

        # Encode categoricals
        cat_cols = ["gender", "blood_group", "primary_diagnosis", "admission_type",
                    "discharge_disposition", "hospital_type", "hospital_size", "insurance_type"]
        for col in cat_cols:
            enc = encoders[col]
            val = input_df[col].values[0]
            if val in enc.classes_:
                input_df[col] = enc.transform([val])
            else:
                input_df[col] = 0  # unseen category fallback

        input_df = input_df[feature_names]
        input_scaled = scaler.transform(input_df)

        prob = model.predict_proba(input_scaled)[0][1]
        pred = int(prob >= 0.40)  # threshold tuned for recall

        st.divider()
        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            if pred == 1:
                st.markdown(f"""
                <div class="risk-high">
                    ⚠️ HIGH READMISSION RISK<br>
                    <span style="font-size:28px">{prob*100:.1f}%</span> probability
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    ✅ LOW READMISSION RISK<br>
                    <span style="font-size:28px">{prob*100:.1f}%</span> probability
                </div>
                """, unsafe_allow_html=True)

        with col_res2:
            # Gauge chart
            fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(polar=True))
            theta = np.linspace(0, np.pi, 200)
            ax.set_theta_offset(np.pi)
            ax.set_theta_direction(-1)
            ax.set_ylim(0, 1)
            ax.plot(theta[:100], [0.9] * 100, color="#1A9641", linewidth=10, alpha=0.4)
            ax.plot(theta[100:], [0.9] * 100, color="#D7191C", linewidth=10, alpha=0.4)
            needle_theta = np.pi * (1 - prob)
            ax.annotate("", xy=(needle_theta, 0.85), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2.5))
            ax.set_axis_off()
            ax.set_title(f"Risk Gauge: {prob*100:.1f}%", fontweight="bold")
            st.pyplot(fig)
            plt.close()

        # Clinical recommendations
        st.subheader("📋 Clinical Recommendations")
        recs = []
        if pred == 1:
            recs += ["📞 Schedule follow-up appointment within 7 days of discharge"]
            recs += ["💊 Ensure complete medication reconciliation before discharge"]
            recs += ["📋 Provide detailed discharge instructions in patient's language"]
            if num_previous_admissions > 2:
                recs += ["🔁 High repeat admission history – consider care coordination program"]
            if primary_diagnosis in ["Heart Failure", "COPD", "Renal Failure"]:
                recs += [f"❗ {primary_diagnosis} patients require enhanced post-discharge monitoring"]
            if comorbidity_count >= 2:
                recs += ["🏥 Multiple comorbidities – refer to disease management program"]
        else:
            recs += ["✅ Standard discharge process recommended"]
            recs += ["📞 Routine follow-up at 30 days"]
            recs += ["💊 Provide standard medication list and instructions"]

        for r in recs:
            st.info(r)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – DATA ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Analytics":
    st.markdown('<div class="section-header">Data Analytics Dashboard</div>', unsafe_allow_html=True)

    plot_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")

    tab1, tab2, tab3, tab4 = st.tabs(["EDA Overview", "Correlation Heatmap", "Model Evaluation", "Feature Importance"])

    with tab1:
        img_path = os.path.join(plot_dir, "01_eda_overview.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Exploratory Data Analysis Overview", use_column_width=True)
        else:
            st.warning("Plot not found. Please run ml_pipeline.py first.")

    with tab2:
        img_path = os.path.join(plot_dir, "02_correlation_heatmap.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Feature Correlation Heatmap", use_column_width=True)

    with tab3:
        img_path = os.path.join(plot_dir, "03_model_evaluation.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Model Evaluation – All Models", use_column_width=True)

    with tab4:
        img_path = os.path.join(plot_dir, "04_feature_importance.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Feature Importance (Random Forest)", use_column_width=True)

    # Live dataset stats
    st.divider()
    st.subheader("📋 Dataset Summary")
    try:
        adm = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "admissions.csv"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Patients", f"{len(adm):,}")
        c2.metric("Readmitted", f"{adm['readmitted_30days'].sum():,}")
        c3.metric("Readmission Rate", f"{adm['readmitted_30days'].mean():.1%}")
        c4.metric("Avg Length of Stay", f"{adm['length_of_stay'].mean():.1f} days")
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About the Project":
    st.markdown('<div class="section-header">About This Project</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🏥 Hospital Readmission Analysis & Prediction System

    **Project by:** Mahdi Nawaz  
    **Degree:** B.Tech (Computer Science Engineering – AI/ML)  
    **University:** GNA University, Phagwara  
    **Year:** 2026

    ---

    ### 🎯 Objective
    To build an intelligent system that predicts whether a patient is likely to be
    readmitted to a hospital within 30 days of discharge, helping healthcare
    professionals take proactive measures.

    ---

    ### 🔧 Technologies Used
    | Layer | Technology |
    |---|---|
    | Language | Python 3.x |
    | Data Processing | Pandas, NumPy |
    | Visualization | Matplotlib, Seaborn |
    | Machine Learning | Scikit-learn |
    | Web Application | Streamlit |
    | Model Persistence | Joblib |

    ---

    ### 🤖 ML Models Implemented
    - **Logistic Regression** – Primary model (best ROC-AUC)
    - **Random Forest** – Ensemble tree-based model
    - **Gradient Boosting** – Sequential boosting ensemble

    ---

    ### 📂 Project Structure
    ```
    hospital_readmission/
    ├── data/
    │   ├── generate_data.py
    │   ├── patients.csv
    │   ├── admissions.csv
    │   ├── hospitals.csv
    │   ├── diagnoses.csv
    │   └── billing.csv
    ├── models/
    │   ├── best_model.pkl
    │   ├── scaler.pkl
    │   └── encoders.pkl
    ├── plots/
    │   ├── 01_eda_overview.png
    │   ├── 02_correlation_heatmap.png
    │   ├── 03_model_evaluation.png
    │   └── 04_feature_importance.png
    ├── app/
    │   └── streamlit_app.py
    ├── ml_pipeline.py
    └── requirements.txt
    ```

    ---

    ### 📚 References
    - [Kaggle India Hospital Readmission Dataset](https://www.kaggle.com/datasets/digutlaranjithkumar/india-hospital-readmission-dataset-20152024)
    - [Scikit-learn Documentation](https://scikit-learn.org)
    - [Streamlit Documentation](https://docs.streamlit.io)
    """)
