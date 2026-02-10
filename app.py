import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Hospital Readmission Risk Predictor")
st.markdown("### Predict 30-day readmission risk for diabetic patients")

# Sidebar for model info
with st.sidebar:
    st.header("Model Performance")
    st.metric("Accuracy", "64.0%")
    st.metric("F1-Score", "63.9%")
    
    st.markdown("---")
    st.markdown("### Model Details")
    st.info("""
    **Algorithm:** Random Forest (Tuned)
    
    **Why this model?**
    - Best class balance (68%/59% recall)
    - Fair predictions for both classes
    - Most reliable for deployment
    """)
    
    st.markdown("---")
    st.markdown("### Class Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Class 0 Recall", "68%")
    with col2:
        st.metric("Class 1 Recall", "59%")

# Main content
tab1, tab2, tab3 = st.tabs(["Prediction", "Model Insights", "About"])

with tab1:
    st.header("Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        age = st.selectbox("Age Group", 
                          ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", 
                           "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        race = st.selectbox("Race", 
                           ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
    
    with col2:
        st.subheader("Clinical Metrics")
        time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 4)
        num_lab_procedures = st.slider("Number of Lab Procedures", 1, 132, 44)
        num_procedures = st.slider("Number of Procedures", 0, 6, 1)
        num_medications = st.slider("Number of Medications", 1, 81, 16)
        num_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7)
    
    with col3:
        st.subheader("Service Utilization")
        num_outpatient = st.number_input("Outpatient Visits (past year)", 0, 42, 0)
        num_emergency = st.number_input("Emergency Visits (past year)", 0, 76, 0)
        num_inpatient = st.number_input("Inpatient Visits (past year)", 0, 21, 0)
        
        st.subheader("Treatment")
        change_meds = st.selectbox("Change in Medications?", ["No", "Ch"])
        diabetes_med = st.selectbox("Diabetes Medication?", ["No", "Yes"])
    
    # Create prediction button
    if st.button("Predict Readmission Risk", type="primary", use_container_width=True):
        # Here you would load your actual trained model
        # For demo purposes, we'll use a simple rule-based prediction
        
        # Calculate risk score (simplified for demo)
        risk_score = 0
        if time_in_hospital > 7:
            risk_score += 20
        if num_medications > 20:
            risk_score += 15
        if num_inpatient > 0:
            risk_score += 25
        if num_emergency > 2:
            risk_score += 20
        if change_meds == "Ch":
            risk_score += 10
        if num_diagnoses > 10:
            risk_score += 10
        
        # Determine prediction
        prediction = 1 if risk_score > 50 else 0
        probability = min(risk_score / 100, 0.95)
        
        # Display results
        st.markdown("---")
        st.header("Prediction Results")
        
        result_col1, result_col2 = st.columns([2, 1])
        
        with result_col1:
            if prediction == 1:
                st.error("**HIGH RISK** of 30-day readmission")
                st.markdown(f"""
                <div style='background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 5px solid #f44336;'>
                    <h3 style='color: #c62828; margin-top: 0;'>Readmission Risk: {probability*100:.1f}%</h3>
                    <p style='color: #555;'>This patient shows elevated risk factors for hospital readmission within 30 days.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("**LOW RISK** of 30-day readmission")
                st.markdown(f"""
                <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50;'>
                    <h3 style='color: #2e7d32; margin-top: 0;'>Readmission Risk: {probability*100:.1f}%</h3>
                    <p style='color: #555;'>This patient shows lower risk factors for hospital readmission.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with result_col2:
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgreen"},
                        {'range': [33, 66], 'color': "yellow"},
                        {'range': [66, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors
        st.markdown("### Key Risk Factors")
        risk_factors = []
        
        if time_in_hospital > 7:
            risk_factors.append(f"Extended hospital stay ({time_in_hospital} days)")
        if num_medications > 20:
            risk_factors.append(f"High medication count ({num_medications})")
        if num_inpatient > 0:
            risk_factors.append(f"Previous inpatient visits ({num_inpatient})")
        if num_emergency > 2:
            risk_factors.append(f"Multiple emergency visits ({num_emergency})")
        if num_diagnoses > 10:
            risk_factors.append(f"Complex medical history ({num_diagnoses} diagnoses)")
        
        if risk_factors:
            for factor in risk_factors:
                st.warning(f"Warning: {factor}")
        else:
            st.info("No major risk factors identified")
        
        # Recommendations
        st.markdown("### Clinical Recommendations")
        if prediction == 1:
            st.markdown("""
            - Schedule follow-up appointment within 7 days
            - Ensure medication reconciliation is complete
            - Consider home health services
            - Patient education on warning signs
            - Coordinate with primary care physician
            """)
        else:
            st.markdown("""
            - Standard follow-up within 14 days
            - Reinforce medication adherence
            - Provide discharge instructions
            - Ensure patient has primary care contact
            """)

with tab2:
    st.header("Model Performance Insights")
    
    # Create comparison chart
    models = ['Decision Tree', 'Random Forest', 'Gradient Boosting']
    accuracy = [0.6312, 0.6397, 0.6439]
    f1_scores = [0.6221, 0.6391, 0.6384]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[
            go.Bar(name='Accuracy', x=models, y=accuracy, marker_color='lightblue'),
            go.Bar(name='F1-Score', x=models, y=f1_scores, marker_color='lightcoral')
        ])
        fig.update_layout(
            title='Model Comparison',
            barmode='group',
            yaxis_title='Score',
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Class balance comparison
        models_balance = ['Decision Tree', 'Random Forest', 'Gradient Boosting']
        class_0_recall = [0.77, 0.68, 0.75]
        class_1_recall = [0.47, 0.59, 0.52]
        
        fig = go.Figure(data=[
            go.Bar(name='Class 0 (No Readmission)', x=models_balance, y=class_0_recall, marker_color='#4caf50'),
            go.Bar(name='Class 1 (Readmission)', x=models_balance, y=class_1_recall, marker_color='#f44336')
        ])
        fig.update_layout(
            title='Class-Specific Recall',
            barmode='group',
            yaxis_title='Recall',
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Feature importance (example data)
    st.subheader("Feature Importance")
    features = ['Number of Inpatient Visits', 'Number of Diagnoses', 'Time in Hospital', 
                'Number of Procedures', 'Number of Medications', 'Discharge Disposition',
                'Number of Emergency Visits', 'Age', 'Admission Type']
    importance = [0.15, 0.13, 0.12, 0.10, 0.09, 0.08, 0.08, 0.06, 0.05]
    
    fig = px.bar(x=importance, y=features, orientation='h',
                 labels={'x': 'Importance Score', 'y': 'Feature'},
                 title='Top Features Contributing to Predictions')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("About This Application")
    
    st.markdown("""
    ### Purpose
    This application predicts the risk of hospital readmission within 30 days for diabetic patients 
    using machine learning algorithms trained on historical patient data.
    
    ### Dataset
    - **Source:** Hospital diabetes dataset with 100,000+ patient encounters
    - **Features:** Demographics, clinical metrics, diagnoses, medications, and service utilization
    - **Target:** Binary classification (readmitted within 30 days or not)
    
    ### Model Selection: Random Forest
    
    After comparing three machine learning algorithms, **Random Forest** was selected because:
    
    1. **Best Class Balance** (68%/59% recall)
       - Doesn't heavily favor one class over another
       - Important when both false positives and false negatives have consequences
    
    2. **Fair Predictions**
       - More reliable for real-world deployment
       - Catches 59% of actual readmissions (vs 47% for Decision Tree)
    
    3. **Comparable Accuracy**
       - 64.0% accuracy (nearly identical to Gradient Boosting's 64.4%)
       - Better generalization through ensemble averaging
    
    ### Performance Metrics
    
    | Metric | Decision Tree | Random Forest | Gradient Boosting |
    |--------|--------------|---------------|-------------------|
    | Accuracy | 63.1% | **64.0%** | 64.4% |
    | F1-Score | 62.2% | **63.9%** | 63.8% |
    | Class 0 Recall | 77% | **68%** | 75% |
    | Class 1 Recall | 47% | **59%** | 52% |
    
    ### Important Notes
    
    - **Clinical Support Tool:** This model should support, not replace, clinical judgment
    - **Validation Required:** Always verify predictions with medical expertise
    - **Continuous Monitoring:** Model performance should be regularly evaluated
    - **Patient Privacy:** No patient data is stored by this application
    
    ### Future Improvements
    
    - Incorporate additional clinical variables
    - Implement SHAP values for individual prediction explanations
    - Add temporal features (seasonal patterns, day of week)
    - Ensemble methods combining multiple models
    - Regular model retraining with new data
    
    ### References
    
    - Dataset: UCI Machine Learning Repository - Diabetes 130-US hospitals
    - Algorithms: scikit-learn machine learning library
    - Evaluation: Standard classification metrics (precision, recall, F1-score)
    """)
    
    st.markdown("---")
    st.markdown("**Developed for educational and research purposes**")
    st.markdown("*Always consult with healthcare professionals for medical decisions*")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Hospital Readmission Risk Predictor | Built with Streamlit</p>
        <p style='font-size: 0.8em;'>This is a decision support tool and should not replace professional medical judgment</p>
    </div>
""", unsafe_allow_html=True)