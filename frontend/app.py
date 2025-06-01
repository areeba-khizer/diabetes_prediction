import streamlit as st
import requests
import pandas as pd

st.title("Diabetes Classification Predictor")

with st.form("prediction_form"):
    gender = st.selectbox("Gender", options=["Female", "Male"])
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", options=[0, 1])
    heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", options=[0, 1])
    smoking_history = st.selectbox("Smoking History", options=["never", "No Info", "current", "former"])
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    HbA1c_level = st.number_input("HbA1c Level", min_value=0.0, step=0.1)
    blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0.0, step=0.1)
    submit = st.form_submit_button("Predict")

if submit:
    input_data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": HbA1c_level,
        "blood_glucose_level": blood_glucose_level,
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['status']}")
            st.write(f"Explanation: {result['explanation']}")

            # Show chart of input values with typical ranges
            ranges = {
                "Age": (0, 120),
                "BMI": (10, 50),
                "HbA1c Level": (4.0, 15.0),
                "Blood Glucose Level": (70, 200),
            }

            # Prepare data for bar chart
            data = {
                "Input Value": [age, bmi, HbA1c_level, blood_glucose_level],
                "Typical Min": [r[0] for r in ranges.values()],
                "Typical Max": [r[1] for r in ranges.values()],
            }
            df = pd.DataFrame(data, index=ranges.keys())

            st.bar_chart(df[["Input Value"]])

            st.write("**Typical Ranges:**")
            for feature, (min_val, max_val) in ranges.items():
                st.write(f"- {feature}: {min_val} to {max_val}")

        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
