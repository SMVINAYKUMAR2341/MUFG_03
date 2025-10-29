from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Dict
import os

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using machine learning models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model and scaler
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "data/processed/scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
    print(f"Scaler loaded successfully from {SCALER_PATH}")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None


# Input data model
class PatientData(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0=Female, 1=Male)")
    chest_pain_type: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    resting_blood_pressure: int = Field(..., ge=80, le=200, description="Resting blood pressure (mm Hg)")
    cholesterol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fasting_blood_sugar: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0=False, 1=True)")
    resting_ecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    max_heart_rate: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exercise_induced_angina: int = Field(..., ge=0, le=1, description="Exercise induced angina (0=No, 1=Yes)")
    st_depression: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    st_slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    num_major_vessels: int = Field(..., ge=0, le=3, description="Number of major vessels (0-3)")
    thalassemia: int = Field(..., ge=0, le=3, description="Thalassemia test result (0-3)")

    class Config:
        schema_extra = {
            "example": {
                "age": 58,
                "sex": 1,
                "chest_pain_type": 1,
                "resting_blood_pressure": 134,
                "cholesterol": 246,
                "fasting_blood_sugar": 0,
                "resting_ecg": 0,
                "max_heart_rate": 155,
                "exercise_induced_angina": 0,
                "st_depression": 0.4,
                "st_slope": 1,
                "num_major_vessels": 1,
                "thalassemia": 2
            }
        }


# Prediction response model
class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability: float
    risk_level: str
    message: str


@app.get("/")
async def root():
    """Serve the frontend HTML page"""
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    scaler_status = "loaded" if scaler is not None else "not loaded"
    
    return {
        "status": "healthy" if (model and scaler) else "unhealthy",
        "model_status": model_status,
        "scaler_status": scaler_status
    }


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    model_type = type(model).__name__
    
    # Get model parameters
    params = model.get_params()
    
    return {
        "model_type": model_type,
        "model_parameters": params,
        "features": [
            "age", "sex", "chest_pain_type", "resting_blood_pressure",
            "cholesterol", "fasting_blood_sugar", "resting_ecg",
            "max_heart_rate", "exercise_induced_angina", "st_depression",
            "st_slope", "num_major_vessels", "thalassemia"
        ],
        "target": "heart_disease (0=No, 1=Yes)"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient_data: PatientData):
    """
    Make a prediction for a patient
    
    - **Input**: Patient diagnostic data
    - **Output**: Prediction with probability and risk assessment
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([patient_data.dict()])
        
        # Scale the features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Get probability of positive class (heart disease)
        disease_probability = probability[1]
        
        # Determine risk level
        if disease_probability < 0.3:
            risk_level = "Low Risk"
        elif disease_probability < 0.6:
            risk_level = "Moderate Risk"
        elif disease_probability < 0.8:
            risk_level = "High Risk"
        else:
            risk_level = "Very High Risk"
        
        # Create prediction label
        prediction_label = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        
        # Create response message
        if prediction == 1:
            message = f"The model predicts heart disease with {disease_probability*100:.1f}% confidence. {risk_level}. Please consult with a healthcare professional."
        else:
            message = f"The model predicts no heart disease with {(1-disease_probability)*100:.1f}% confidence. {risk_level}. Continue regular health monitoring."
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label=prediction_label,
            probability=float(disease_probability),
            risk_level=risk_level,
            message=message
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(patients: list[PatientData]):
    """
    Make predictions for multiple patients
    
    - **Input**: List of patient diagnostic data
    - **Output**: List of predictions
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")
    
    try:
        results = []
        for patient_data in patients:
            # Convert input to DataFrame
            input_data = pd.DataFrame([patient_data.dict()])
            
            # Scale the features
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            disease_probability = probability[1]
            
            # Determine risk level
            if disease_probability < 0.3:
                risk_level = "Low Risk"
            elif disease_probability < 0.6:
                risk_level = "Moderate Risk"
            elif disease_probability < 0.8:
                risk_level = "High Risk"
            else:
                risk_level = "Very High Risk"
            
            prediction_label = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
            
            results.append({
                "prediction": int(prediction),
                "prediction_label": prediction_label,
                "probability": float(disease_probability),
                "risk_level": risk_level
            })
        
        return {"predictions": results, "count": len(results)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
