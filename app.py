
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

model_path = Path(__file__).resolve().parent / "random_forest_model.pkl"
model = joblib.load(model_path)

st.title("❤️ Heart Disease Prediction App")

st.sidebar.title("ℹ️ Feature Value Guide")
st.sidebar.markdown("""
- **sex**: 1 = male, 0 = female
- **cp**: Chest pain type (1-4)
- **fbs**: 1 = yes, 0 = no
- **restecg**: ECG results (0-2)
- **exang**: 1 = yes, 0 = no
- **slope**: Slope (0-2)
- **thal**: 3 = normal, 6 = fixed defect, 7 = reversible defect
""")

with st.form("heart_form"):
    st.subheader("Enter Patient Details")

    age = st.text_input("Age", "50")
    sex = st.text_input("Sex (1=male, 0=female)", "1")
    cp = st.text_input("Chest Pain Type (1-4)", "3")
    trestbps = st.text_input("Resting Blood Pressure", "120")
    chol = st.text_input("Cholesterol", "200")
    fbs = st.text_input("Fasting Blood Sugar > 120", "0")
    restecg = st.text_input("Resting ECG Results", "1")
    thalach = st.text_input("Maximum Heart Rate", "150")
    exang = st.text_input("Exercise Induced Angina", "0")
    oldpeak = st.text_input("ST Depression (Oldpeak)", "1.0")
    slope = st.text_input("Slope", "1")
    ca = st.text_input("Number of Major Vessels", "0")
    thal = st.text_input("Thal", "3")

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        feature_names = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]

        input_df = pd.DataFrame([[
            int(age), int(sex), int(cp), int(trestbps), int(chol),
            int(fbs), int(restecg), int(thalach), int(exang),
            float(oldpeak), int(slope), int(ca), int(thal)
        ]], columns=feature_names)

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High Risk of Heart Disease (Confidence: {prob:.2%})")
        else:
            st.success(f"✅ Low Risk of Heart Disease (Confidence: {prob:.2%})")

    except ValueError:
        st.warning("Please ensure all inputs are valid numbers.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
