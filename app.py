import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('models/fetal_health_best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_model()

st.set_page_config(page_title="Fetal Health Predictor", layout="wide")
st.title("🤰 Fetal Health Classification System")
st.markdown("""
This system uses Cardiotocography (CTG) data to predict fetal health status.
**Classes:** 1 = Normal, 2 = Suspect, 3 = Pathological
""")

# Create input form
with st.form("prediction_form"):
    st.subheader("Enter All 21 CTG Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**FHR Signal Features**")
        lb = st.number_input("LB (Baseline FHR)", value=120.0, step=1.0, help="Baseline fetal heart rate")
        ac = st.number_input("AC (Accelerations)", value=0.0, step=0.1, help="Number of accelerations per second")
        fm = st.number_input("FM (Fetal Movements)", value=0.0, step=0.1, help="Number of fetal movements per second")
        uc = st.number_input("UC (Uterine Contractions)", value=0.0, step=0.1, help="Number of uterine contractions per second")
        dl = st.number_input("DL (Decelerations - Light)", value=0.0, step=0.1)
        ds = st.number_input("DS (Decelerations - Severe)", value=0.0, step=0.1)
        dp = st.number_input("DP (Decelerations - Progressive)", value=0.0, step=0.1)
        
    with col2:
        st.markdown("**Variability Features**")
        astv = st.number_input("ASTV (Abnormal Short Term Variability)", value=20.0, step=1.0)
        mstv = st.number_input("MSTV (Mean Short Term Variability)", value=1.0, step=0.1)
        altv = st.number_input("ALTV (Abnormal Long Term Variability)", value=20.0, step=1.0)
        mltv = st.number_input("MLTV (Mean Long Term Variability)", value=5.0, step=0.5)
        
        st.markdown("**Histogram Features**")
        width = st.number_input("Width (Histogram Width)", value=50.0, step=1.0)
        min_val = st.number_input("Min (Histogram Minimum)", value=50.0, step=1.0)
        max_val = st.number_input("Max (Histogram Maximum)", value=150.0, step=1.0)
        nmax = st.number_input("Nmax (Number of histogram peaks)", value=1.0, step=1.0)
        
    with col3:
        st.markdown("**Statistical Features**")
        nzeros = st.number_input("Nzeros (Number of zeros)", value=0.0, step=1.0)
        mode_val = st.number_input("Mode (Histogram mode)", value=120.0, step=1.0)
        mean_val = st.number_input("Mean (Histogram mean)", value=120.0, step=1.0)
        median_val = st.number_input("Median (Histogram median)", value=120.0, step=1.0)
        variance_val = st.number_input("Variance (Histogram variance)", value=20.0, step=1.0)
        tendency = st.number_input("Tendency (Histogram tendency)", value=0.0, step=0.1)
    
    submitted = st.form_submit_button("Predict Fetal Health", type="primary")

if submitted:
    # Create feature array in EXACT order as training
    features = np.array([[
        lb, ac, fm, uc, dl, ds, dp, astv, mstv, altv, mltv,
        width, min_val, max_val, nmax, nzeros, mode_val, mean_val, median_val, variance_val, tendency
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Display results
    st.markdown("---")
    st.subheader("Prediction Results")
    
    # Color-coded result
    if prediction == 1:
        st.success("## ✅ NORMAL - Fetal health appears normal")
        color = "green"
    elif prediction == 2:
        st.warning("## ⚠️ SUSPECT - Further monitoring recommended")
        color = "orange"
    else:
        st.error("## 🚨 PATHOLOGICAL - Immediate medical consultation advised")
        color = "red"
    
    # Show probabilities
    st.markdown("### Probability Distribution")
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Normal (1)', 'Suspect (2)', 'Pathological (3)'],
            y=probabilities * 100,
            marker_color=['green', 'orange', 'red'],
            text=[f"{p*100:.1f}%" for p in probabilities],
            textposition='auto'
        )
    ])
    fig.update_layout(
        title="Prediction Confidence by Class",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Medical guidance
    with st.expander("📋 Medical Interpretation Guide"):
        st.markdown("""
        **What these results mean:**
        
        - **Normal (Class 1)**: Healthy fetal state, no immediate concerns
        - **Suspect (Class 2)**: Borderline results, recommend non-stress test (NST) monitoring
        - **Pathological (Class 3)**: Abnormal findings, immediate medical evaluation needed
        
        **Important note:** This is a decision support tool only. Always consult with healthcare professionals.
        """)
    
    # Show feature importance for this prediction
    if prediction != 1:
        st.info("⚠️ High-risk features detected - clinical correlation recommended")