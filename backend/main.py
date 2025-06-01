import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://diabetes-frontend.azurewebsites.net",  # Your deployed frontend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class InputData(BaseModel):
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float

# Preprocessing
def preprocess_input(data: InputData) -> pd.DataFrame:
    gender_map = {'Female': 0, 'Male': 1}
    smoking_map = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3}

    if data.gender not in gender_map:
        raise HTTPException(status_code=400, detail=f"Invalid gender: {data.gender}")
    if data.smoking_history not in smoking_map:
        raise HTTPException(status_code=400, detail=f"Invalid smoking history: {data.smoking_history}")

    df = pd.DataFrame([{
        'gender': gender_map[data.gender],
        'age': data.age,
        'hypertension': data.hypertension,
        'heart_disease': data.heart_disease,
        'smoking_history': smoking_map[data.smoking_history],
        'bmi': data.bmi,
        'HbA1c_level': data.HbA1c_level,
        'blood_glucose_level': data.blood_glucose_level
    }])
    return df

# Prediction interpretation
def interpret_diabetes_classification(prediction: int):
    if prediction == 0:
        return "No Diabetes", "The model predicts no diabetes for the patient."
    else:
        return "Diabetes", "The model predicts the presence of diabetes. Please consult a doctor."

# Load model at startup
model = None
model_path = "models/random_forest_classifier"  # Relative path (must be deployed with app)

try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    model = mlflow.sklearn.load_model(model_path)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")

# API endpoint
@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    df = preprocess_input(data)
    prediction = model.predict(df)[0]
    status, explanation = interpret_diabetes_classification(prediction)
    return {
        "prediction": int(prediction),
        "status": status,
        "explanation": explanation
    }
