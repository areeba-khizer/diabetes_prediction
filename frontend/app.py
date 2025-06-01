import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Classifier", layout="centered")

st.title("🩺 Diabetes Classification Predictor")
st.markdown("Enter patient details below to classify the diabetes condition.")

# Form Layout in 2 columns
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", options=["Female", "Male"])
        age = st.number_input("Age (years)", min_value=1, max_value=120, step=1)
        hypertension = st.selectbox("Hypertension", options=[0, 1], help="0 = No, 1 = Yes")
        heart_disease = st.selectbox("Heart Disease", options=[0, 1], help="0 = No, 1 = Yes")

    with col2:
        smoking_history = st.selectbox("Smoking History", options=["never", "No Info", "current", "former"])
        bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, step=0.1)
        HbA1c_level = st.number_input("HbA1c Level (%)", min_value=0.0, step=0.1)
        blood_glucose_level = st.number_input("Blood Glucose Level (mg/dL)", min_value=0.0, step=0.1)

    submit = st.form_submit_button("🔍 Predict")

# Reference ranges for visualization
ranges = {
    "Age": (1, 100),
    "BMI": (18.5, 30.0),
    "HbA1c Level": (4.0, 6.4),
    "Blood Glucose Level": (70, 140)
}

def show_comparison_chart(values_dict):
    df = pd.DataFrame({
        "Input": list(values_dict.values()),
        "Min (Normal)": [v[0] for v in ranges.values()],
        "Max (Normal)": [v[1] for v in ranges.values()]
    }, index=list(ranges.keys()))

    st.subheader("📊 Health Indicator Comparison")
    st.bar_chart(df[["Input"]])
    
    st.caption("🧠 Reference Ranges:")
    for k, (low, high) in ranges.items():
        st.write(f"- **{k}**: {low} to {high}")

if submit:
    # Determine age group and typical ranges
    if age <= 39:
        age_group = "Young Adults (18–39)"
        ranges = {
            "BMI": (18.5, 24.9),
            "HbA1c Level": (4.0, 5.6),
            "Blood Glucose Level": (70, 99)
        }
    elif 40 <= age <= 64:
        age_group = "Middle-Aged Adults (40–64)"
        ranges = {
            "BMI": (18.5, 27.0),
            "HbA1c Level": (4.0, 6.4),
            "Blood Glucose Level": (70, 110)
        }
    else:
        age_group = "Older Adults (65+)"
        ranges = {
            "BMI": (22.0, 27.0),
            "HbA1c Level": (5.5, 7.5),
            "Blood Glucose Level": (70, 130)
        }

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
        response = requests.post("https://diabetes-backend.azurewebsites.net/", json=input_data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['status']}")
            st.write(f"Explanation: {result['explanation']}")

            # Display chart comparing input with age-specific ranges
            st.subheader(f"Input vs Normal Ranges ({age_group})")

            chart_data = {
                "Input Value": [bmi, HbA1c_level, blood_glucose_level],
                "Typical Min": [v[0] for v in ranges.values()],
                "Typical Max": [v[1] for v in ranges.values()],
            }
            df = pd.DataFrame(chart_data, index=["BMI", "HbA1c Level", "Blood Glucose Level"])
            st.bar_chart(df[["Input Value"]])

            # Show typical ranges as text
            st.write("Typical Ranges for Your Age Group:")
            for feature, (min_val, max_val) in ranges.items():
                st.write(f"- {feature}: {min_val} to {max_val}")
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
