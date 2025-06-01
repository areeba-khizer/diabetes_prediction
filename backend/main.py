import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Set MLflow tracking URI (your local server)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Update model URI with your actual run ID and artifact path (make sure this matches your run!)
model_uri = "runs:/18ee43c57fb645ffb53444c0ef6a9c27/random_forest_classifier"
model = mlflow.sklearn.load_model(model_uri)

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema - update fields to match your dataset's features
class InputData(BaseModel):
    gender: str           # categorical, e.g., 'Male' or 'Female'
    age: int
    hypertension: int     # 0 or 1
    heart_disease: int    # 0 or 1
    smoking_history: str  # categorical, e.g., 'never', 'current', 'former', 'No Info'
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float

# Encode categorical features as your model expects
def preprocess_input(data: InputData):
    # Map gender
    gender_map = {'Female': 0, 'Male': 1}
    gender = gender_map.get(data.gender, -1)  # Use -1 or handle unknown

    # Map smoking_history
    smoking_map = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3}
    smoking = smoking_map.get(data.smoking_history, -1)

    # Build dataframe in correct order matching training
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

# Optional: interpret model output if classification
def interpret_diabetes_classification(prediction):
    if prediction == 0:
        return "No Diabetes", "The model predicts no diabetes for the patient."
    else:
        return "Diabetes", "The model predicts the presence of diabetes. Please consult a doctor."

@app.post("/predict")
def predict(data: InputData):
    df = preprocess_input(data)
    prediction = model.predict(df)[0]
    status, explanation = interpret_diabetes_classification(prediction)
    return {
        "prediction": int(prediction),
        "status": status,
        "explanation": explanation
    }
