import gradio as gr
import joblib
import numpy as np
import pandas as pd
import os

# Load model and scaler
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "data/processed/scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None

# Feature names
FEATURE_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, 
                         thalach, exang, oldpeak, slope, ca, thal):
    """
    Predict heart disease risk based on patient data.
    """
    if model is None or scaler is None:
        return "❌ Model not loaded. Please check configuration.", "N/A", "N/A"
    
    try:
        # Create input dataframe
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                   thalach, exang, oldpeak, slope, ca, thal]], 
                                 columns=FEATURE_NAMES)
        
        # Scale the features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Get probability of disease (class 1)
        disease_prob = probability[1]
        
        # Determine risk level
        if disease_prob < 0.3:
            risk_level = "🟢 Low Risk"
            risk_color = "#2ecc71"
        elif disease_prob < 0.5:
            risk_level = "🟡 Moderate Risk"
            risk_color = "#f39c12"
        elif disease_prob < 0.7:
            risk_level = "🟠 High Risk"
            risk_color = "#e67e22"
        else:
            risk_level = "🔴 Very High Risk"
            risk_color = "#e74c3c"
        
        # Result message
        if prediction == 1:
            result = f"⚠️ **Heart Disease Detected**"
        else:
            result = f"✅ **No Heart Disease Detected**"
        
        probability_text = f"{disease_prob * 100:.1f}%"
        
        return result, probability_text, risk_level
        
    except Exception as e:
        return f"❌ Error during prediction: {str(e)}", "N/A", "N/A"

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Heart Disease Prediction") as demo:
    gr.Markdown(
        """
        # ❤️ Heart Disease Prediction System
        
        ### Predict heart disease risk using machine learning
        
        This system uses a **Random Forest model** trained on 400 patient records 
        with **84% accuracy** and **0.82 ROC-AUC score**.
        
        Enter patient medical information below to get a risk assessment.
        """
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👤 Patient Demographics")
            age = gr.Slider(minimum=20, maximum=100, value=50, step=1, 
                          label="Age (years)")
            sex = gr.Radio(choices=[("Female", 0), ("Male", 1)], 
                         value=1, label="Sex")
            
            gr.Markdown("### 🩺 Vital Signs")
            trestbps = gr.Slider(minimum=80, maximum=200, value=120, step=1,
                               label="Resting Blood Pressure (mm Hg)")
            chol = gr.Slider(minimum=100, maximum=600, value=200, step=1,
                           label="Serum Cholesterol (mg/dl)")
            thalach = gr.Slider(minimum=60, maximum=220, value=150, step=1,
                              label="Maximum Heart Rate Achieved")
            
            gr.Markdown("### 💊 Medical Tests")
            fbs = gr.Radio(choices=[("No", 0), ("Yes", 1)], value=0,
                         label="Fasting Blood Sugar > 120 mg/dl")
            restecg = gr.Radio(
                choices=[("Normal", 0), ("ST-T Abnormality", 1), 
                        ("Left Ventricular Hypertrophy", 2)],
                value=0, label="Resting ECG Results"
            )
            
        with gr.Column():
            gr.Markdown("### 💔 Cardiac Symptoms")
            cp = gr.Radio(
                choices=[("Typical Angina", 0), ("Atypical Angina", 1),
                        ("Non-Anginal Pain", 2), ("Asymptomatic", 3)],
                value=0, label="Chest Pain Type"
            )
            exang = gr.Radio(choices=[("No", 0), ("Yes", 1)], value=0,
                           label="Exercise Induced Angina")
            
            gr.Markdown("### 📊 Advanced Tests")
            oldpeak = gr.Slider(minimum=0, maximum=6.5, value=1.0, step=0.1,
                              label="ST Depression (Exercise vs Rest)")
            slope = gr.Radio(
                choices=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
                value=1, label="Slope of Peak Exercise ST Segment"
            )
            ca = gr.Slider(minimum=0, maximum=4, value=0, step=1,
                         label="Number of Major Vessels (Fluoroscopy)")
            thal = gr.Radio(
                choices=[("Normal", 1), ("Fixed Defect", 2), 
                        ("Reversible Defect", 3)],
                value=2, label="Thalassemia"
            )
    
    predict_btn = gr.Button("🔍 Predict Heart Disease Risk", variant="primary", size="lg")
    
    gr.Markdown("---")
    gr.Markdown("## 📋 Prediction Results")
    
    with gr.Row():
        with gr.Column():
            result_output = gr.Markdown(label="Diagnosis")
        with gr.Column():
            probability_output = gr.Textbox(label="Disease Probability", 
                                          interactive=False)
        with gr.Column():
            risk_output = gr.Textbox(label="Risk Level", interactive=False)
    
    # Connect button to prediction function
    predict_btn.click(
        fn=predict_heart_disease,
        inputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
               exang, oldpeak, slope, ca, thal],
        outputs=[result_output, probability_output, risk_output]
    )
    
    gr.Markdown(
        """
        ---
        ### 📊 Model Information
        
        - **Algorithm**: Random Forest Classifier
        - **Accuracy**: 84%
        - **ROC-AUC**: 0.82
        - **Recall**: 81.8%
        - **Training Data**: 400 patients, 13 features
        
        ### ⚠️ Disclaimer
        
        This tool is for educational purposes only and should not be used as a 
        substitute for professional medical advice, diagnosis, or treatment.
        
        ---
        
        **Developed by**: Amazon ML Challenge Team
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
