import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Hospital Diabetes Dashboard",
    page_icon="🏥",
    layout="centered"
)

st.title(" Diabetes Readmission Predictor")
st.markdown("Predict the risk of a patient being readmitted within 30 days.")
st.markdown("---")

# ===============================
# Load Trained Model
# ===============================
try:
    model = joblib.load("gradient_boosting_model_fe.pkl")  # or your trained .pkl
except FileNotFoundError:
    st.error(" Model file not found. Please ensure your .pkl file is available.")
    st.stop()

# ===============================
# Inputs
# ===============================
st.subheader(" Patient Information")
col1, col2 = st.columns([1, 2])

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"])
    age = st.selectbox(
        "Age Group",
        ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
         "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
    )

with col2:
    time_in_hospital = st.slider("Length of Hospital Stay (days)", 1, 14, 3)
    num_medications = st.slider("Number of Medications", 0, 50, 10)
    num_lab_procedures = st.slider("Number of Lab Procedures", 0, 100, 40)

st.markdown("---")

# ===============================
# Predict Button
# ===============================
if st.button(" Predict Readmission Risk"):

    # Input validation
    if num_medications < 0 or num_lab_procedures < 0:
        st.warning(" Values cannot be negative.")
        st.stop()

    # Build input dataframe
    input_df = pd.DataFrame({
        "gender": [gender],
        "race": [race],
        "age": [age],
        "time_in_hospital": [time_in_hospital],
        "num_medications": [num_medications],
        "num_lab_procedures": [num_lab_procedures]
    })

    # One-hot encode categorical features
    input_encoded = pd.get_dummies(input_df)

    # Align with training features
    input_encoded = input_encoded.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Predict
    try:
        prediction = model.predict(input_encoded)[0]
        proba = model.predict_proba(input_encoded)[0][1]  # probability of readmission
    except Exception:
        st.error(" Prediction failed. Please check your inputs.")
        st.stop()

    # ===============================
    # Display Results
    # ===============================
    st.markdown("---")
    st.subheader(" Prediction Result")

    if prediction == 1:
        st.markdown(f"<h2 style='color:red'> High Risk of Readmission</h2>", unsafe_allow_html=True)
        st.metric("Readmission Probability", f"{proba*100:.1f}%")
        st.write("This patient has a higher likelihood of being readmitted within 30 days and may require closer follow-up after discharge.")
    else:
        st.markdown(f"<h2 style='color:green'> Low Risk of Readmission</h2>", unsafe_allow_html=True)
        st.metric("Readmission Probability", f"{proba*100:.1f}%")
        st.write("This patient is unlikely to be readmitted within 30 days.")

    # ===============================
    # Patient Summary
    # ===============================
    with st.expander(" Patient Summary"):
        st.table({
            "Attribute": ["Gender", "Race", "Age Group", "Hospital Stay (days)", "Medications", "Lab Procedures"],
            "Value": [gender, race, age, time_in_hospital, num_medications, num_lab_procedures]
        })

    # ===============================
    # Feature Importance Chart
    # ===============================
    with st.expander(" Top Features Influencing Prediction"):
        fi = model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": model.feature_names_in_,
            "Importance": fi
        }).sort_values(by="Importance", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(x="Importance", y="Feature", data=fi_df, ax=ax)
        st.pyplot(fig)

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption("MLDP Project | Diabetes Readmission Prediction | 2026")
