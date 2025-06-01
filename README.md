# Diabetes Prediction Web App

This project is a full-stack machine learning web application that predicts diabetes based on user input features. It includes:

- **ML Model Training & Tracking** using `MLflow`
- **FastAPI Backend** for model inference
- **Streamlit Frontend** for user interaction
- **Model Deployment** using MLflow model registry

---

## Dataset
- https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

---

## Features

- Train & track a **Random Forest Classifier** using MLflow
- Predict diabetes status with a REST API
- Interactive frontend using Streamlit
- Logs metrics like accuracy, precision, recall, and F1
- Displays result status and explanation
- Includes visualization for prediction range

---

## Install Dependencies
 - pip install -r requirements.txt

---


## Start MlFlow
- mlflow ui

---

## Running App
- uvicorn main:app --reload (for backend)
- streamlit run app.py (for frontend)

---



