import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this
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

# Enhanced model loading with debugging
model = None

def load_model():
    global model
    
    # Debug information
    print("=== MODEL LOADING DEBUG INFO ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {os.path.abspath(__file__)}")
    
    # Try multiple possible paths
    possible_paths = [
        "models/random_forest_classifier",
        "./models/random_forest_classifier",
        os.path.join(os.path.dirname(__file__), "models", "random_forest_classifier"),
        os.path.join(os.getcwd(), "models", "random_forest_classifier")
    ]
    
    print("Checking possible model paths:")
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        exists = os.path.exists(path)
        print(f"  {path} -> {abs_path} (exists: {exists})")
        
        if exists:
            try:
                print(f"Attempting to load model from: {path}")
                model = mlflow.sklearn.load_model(path)
                print("✅ Model loaded successfully!")
                return True
            except Exception as e:
                print(f"❌ Failed to load model from {path}: {e}")
                continue
    
    # List directory contents for debugging
    print("\nDirectory contents:")
    try:
        print(f"Current directory contents: {os.listdir('.')}")
        if os.path.exists("models"):
            print(f"Models directory contents: {os.listdir('models')}")
        else:
            print("Models directory does not exist!")
    except Exception as e:
        print(f"Error listing directories: {e}")
    
    print("❌ Could not load model from any path")
    return False

# Load model at startup
load_model()

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "working_directory": os.getcwd()
    }

# API endpoint
@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model is not loaded. Please check the model path and ensure the model file exists."
        )
    
    try:
        df = preprocess_input(data)
        prediction = model.predict(df)[0]
        status, explanation = interpret_diabetes_classification(prediction)
        return {
            "prediction": int(prediction),
            "status": status,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
