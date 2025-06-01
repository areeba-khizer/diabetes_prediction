import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://diabetes-frontend.azurewebsites.net",  # your frontend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float

def preprocess_input(data: InputData) -> pd.DataFrame:
    gender_map = {'Female': 0, 'Male': 1}
    smoking_map = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3}

    if data.gender not in gender_map:
        raise HTTPException(status_code=400, detail=f"Invalid gender: {data.gender}")
    if data.smoking_history not in smoking_map:
        raise HTTPException(status_code=400, detail=f"Invalid smoking_history: {data.smoking_history}")

    gender = gender_map[data.gender]
    smoking = smoking_map[data.smoking_history]

    df = pd.DataFrame([{
        'gender': gender,
        'age': data.age,
        'hypertension': data.hypertension,
        'heart_disease': data.heart_disease,
        'smoking_history': smoking,
        'bmi': data.bmi,
        'HbA1c_level': data.HbA1c_level,
        'blood_glucose_level': data.blood_glucose_level
    }])
    return df

def interpret_diabetes_classification(prediction: int):
    if prediction == 0:
        return "No Diabetes", "The model predicts no diabetes for the patient."
    else:
        return "Diabetes", "The model predicts the presence of diabetes. Please consult a doctor."

# Load model once at startup from MLflow model registry or local path
try:
    model_path = r"C:\\ml_app_assignment\\ml\\mlruns\\232882814052274270\\18ee43c57fb645ffb53444c0ef6a9c27\\artifacts\\random_forest_classifier"
    model = mlflow.sklearn.load_model(model_path)
    print("MLflow model loaded successfully.")
except Exception as e:
    print(f"Failed to load MLflow model: {e}")
    model = None

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
