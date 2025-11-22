"""
Disease PredictionIQ Web Application
Flask backend with REST API endpoints
Author: Jay Prakash kumar
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'heart-disease-prediction-2025'

# Global variables for model components
model = None
scaler = None
feature_names = None
metadata = None

def load_model_components():
    """Load the trained model and preprocessing components"""
    global model, scaler, feature_names, metadata
    
    models_dir = 'models'
    
    try:
        # Load the best model file
        model_path = os.path.join(models_dir, 'best_heart_disease_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"✓ Loaded model from: {model_path}")
        else:
            print(f"⚠️ Model file not found: {model_path}")
            return False
        
        # Load scaler
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("✓ Loaded scaler")
        
        # Load feature names
        features_path = os.path.join(models_dir, 'feature_names.pkl')
        if os.path.exists(features_path):
            with open(features_path, 'rb') as f:
                feature_names = pickle.load(f)
            print("✓ Loaded feature names")
        
        # Load metadata
        metadata_path = os.path.join(models_dir, 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print("✓ Loaded metadata")
        
        return True
    except Exception as e:
        print(f"Error loading model components: {e}")
        return False

# Load model on startup
print("\n" + "=" * 60)
print("INITIALIZING DISEASE PREDICTIONIQ APPLICATION")
print("=" * 60)
load_model_components()
print("=" * 60 + "\n")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    if metadata:
        # Get performance metrics and convert to expected format
        perf_metrics = metadata.get('performance_metrics', {})
        
        # Calculate overall score as average of all metrics
        overall_score = sum([
            perf_metrics.get('test_accuracy', 0),
            perf_metrics.get('test_precision', 0),
            perf_metrics.get('test_recall', 0),
            perf_metrics.get('test_f1_score', 0),
            perf_metrics.get('test_roc_auc', 0)
        ]) / 5
        
        return jsonify({
            'success': True,
            'model_name': metadata.get('model_name', 'Unknown'),
            'model_type': metadata.get('model_type', 'Unknown'),
            'creation_date': metadata.get('creation_date', 'Unknown'),
            'metrics': {
                'accuracy': perf_metrics.get('test_accuracy', 0),
                'precision': perf_metrics.get('test_precision', 0),
                'recall': perf_metrics.get('test_recall', 0),
                'f1_score': perf_metrics.get('test_f1_score', 0),
                'roc_auc': perf_metrics.get('test_roc_auc', 0),
                'overall_score': overall_score
            },
            'features': feature_names if feature_names else []
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Model metadata not available'
        }), 404

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a prediction based on input data"""
    try:
        data = request.get_json()
        
        # Extract features in correct order
        features = [
            float(data.get('age', 0)),
            float(data.get('sex', 0)),
            float(data.get('chest_pain_type', 0)),
            float(data.get('resting_blood_pressure', 0)),
            float(data.get('cholesterol', 0)),
            float(data.get('fasting_blood_sugar', 0)),
            float(data.get('resting_ecg', 0)),
            float(data.get('max_heart_rate', 0)),
            float(data.get('exercise_induced_angina', 0)),
            float(data.get('st_depression', 0)),
            float(data.get('st_slope', 0)),
            float(data.get('num_major_vessels', 0)),
            float(data.get('thalassemia', 0))
        ]
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        if scaler:
            features_scaled = scaler.transform(features_array)
        else:
            features_scaled = features_array
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Determine risk level
        probability = float(prediction_proba[1] * 100)
        
        if probability < 30:
            risk_level = 'Low'
            risk_color = '#10b981'
        elif probability < 60:
            risk_level = 'Moderate'
            risk_color = '#f59e0b'
        else:
            risk_level = 'High'
            risk_color = '#ef4444'
        
        # Prepare response
        response = {
            'success': True,
            'prediction': int(prediction),
            'probability': round(probability, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'diagnosis': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected',
            'confidence': round(max(prediction_proba) * 100, 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'recommendations': get_recommendations(prediction, probability, data)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error making prediction: {str(e)}'
        }), 400

def get_recommendations(prediction, probability, patient_data):
    """Generate personalized recommendations based on prediction"""
    recommendations = []
    
    if prediction == 1:
        recommendations.append("⚠️ Consult a cardiologist immediately for comprehensive evaluation")
        recommendations.append("📋 Schedule diagnostic tests: ECG, Echocardiogram, Stress Test")
        recommendations.append("💊 Discuss medication options with your healthcare provider")
    
    # Age-based recommendations
    age = int(patient_data.get('age', 0))
    if age > 55:
        recommendations.append("🏃 Engage in moderate physical activity (30 mins daily)")
    
    # Cholesterol recommendations
    cholesterol = int(patient_data.get('cholesterol', 0))
    if cholesterol > 240:
        recommendations.append("🥗 Adopt a heart-healthy diet low in saturated fats")
        recommendations.append("💊 Consider cholesterol-lowering medication (consult doctor)")
    
    # Blood pressure recommendations
    bp = int(patient_data.get('resting_blood_pressure', 0))
    if bp > 140:
        recommendations.append("🩺 Monitor blood pressure regularly")
        recommendations.append("🧂 Reduce sodium intake (< 2,300 mg/day)")
    
    # General recommendations
    if prediction == 0:
        recommendations.append("✅ Maintain healthy lifestyle habits")
        recommendations.append("🏋️ Regular exercise (150 mins/week)")
        recommendations.append("🥦 Balanced diet rich in fruits and vegetables")
        recommendations.append("📅 Regular health check-ups (annual)")
    
    recommendations.append("🚭 Avoid smoking and limit alcohol consumption")
    recommendations.append("😴 Ensure adequate sleep (7-9 hours/night)")
    recommendations.append("🧘 Practice stress management techniques")
    
    return recommendations[:6]  # Return top 6 recommendations

@app.route('/api/models-comparison', methods=['GET'])
def get_models_comparison():
    """Get comparison data for all trained models"""
    # Data from the training notebook - all 14 models comparison
    models_data = [
        {
            'name': 'Neural Network (MLP)',
            'accuracy': 0.6875,
            'precision': 0.673,
            'recall': 0.841,
            'f1_score': 0.747,
            'roc_auc': 0.7355,
            'category': 'Deep Learning',
            'is_best': True
        },
        {
            'name': 'AdaBoost',
            'accuracy': 0.7125,
            'precision': 0.714,
            'recall': 0.795,
            'f1_score': 0.752,
            'roc_auc': 0.7655,
            'category': 'Boosting',
            'is_best': False
        },
        {
            'name': 'LightGBM',
            'accuracy': 0.6875,
            'precision': 0.702,
            'recall': 0.750,
            'f1_score': 0.725,
            'roc_auc': 0.7462,
            'category': 'Boosting',
            'is_best': False
        },
        {
            'name': 'Gradient Boosting',
            'accuracy': 0.6750,
            'precision': 0.688,
            'recall': 0.727,
            'f1_score': 0.707,
            'roc_auc': 0.7260,
            'category': 'Boosting',
            'is_best': False
        },
        {
            'name': 'XGBoost',
            'accuracy': 0.6500,
            'precision': 0.650,
            'recall': 0.705,
            'f1_score': 0.676,
            'roc_auc': 0.7058,
            'category': 'Boosting',
            'is_best': False
        },
        {
            'name': 'Extra Trees',
            'accuracy': 0.6125,
            'precision': 0.649,
            'recall': 0.662,
            'f1_score': 0.655,
            'roc_auc': 0.7121,
            'category': 'Ensemble',
            'is_best': False
        },
        {
            'name': 'K-Nearest Neighbors',
            'accuracy': 0.6625,
            'precision': 0.662,
            'recall': 0.733,
            'f1_score': 0.695,
            'roc_auc': 0.6951,
            'category': 'Instance-based',
            'is_best': False
        },
        {
            'name': 'Naive Bayes',
            'accuracy': 0.6500,
            'precision': 0.650,
            'recall': 0.705,
            'f1_score': 0.689,
            'roc_auc': 0.7216,
            'category': 'Probabilistic',
            'is_best': False
        },
        {
            'name': 'Random Forest',
            'accuracy': 0.6750,
            'precision': 0.685,
            'recall': 0.750,
            'f1_score': 0.716,
            'roc_auc': 0.7380,
            'category': 'Ensemble',
            'is_best': False
        },
        {
            'name': 'Decision Tree',
            'accuracy': 0.6500,
            'precision': 0.660,
            'recall': 0.705,
            'f1_score': 0.682,
            'roc_auc': 0.6850,
            'category': 'Tree-based',
            'is_best': False
        },
        {
            'name': 'Logistic Regression',
            'accuracy': 0.7000,
            'precision': 0.700,
            'recall': 0.773,
            'f1_score': 0.735,
            'roc_auc': 0.7500,
            'category': 'Linear',
            'is_best': False
        },
        {
            'name': 'SVM (RBF)',
            'accuracy': 0.6875,
            'precision': 0.690,
            'recall': 0.750,
            'f1_score': 0.719,
            'roc_auc': 0.7420,
            'category': 'Kernel-based',
            'is_best': False
        },
        {
            'name': 'SVM (Linear)',
            'accuracy': 0.6750,
            'precision': 0.680,
            'recall': 0.727,
            'f1_score': 0.703,
            'roc_auc': 0.7250,
            'category': 'Linear',
            'is_best': False
        },
        {
            'name': 'Linear Discriminant Analysis',
            'accuracy': 0.6625,
            'precision': 0.665,
            'recall': 0.720,
            'f1_score': 0.691,
            'roc_auc': 0.7100,
            'category': 'Linear',
            'is_best': False
        }
    ]
    
    # Sort by ROC-AUC score (descending)
    models_data_sorted = sorted(models_data, key=lambda x: x['roc_auc'], reverse=True)
    
    return jsonify({
        'success': True,
        'total_models': len(models_data),
        'models': models_data_sorted,
        'categories': list(set([m['category'] for m in models_data])),
        'best_model': next(m for m in models_data if m['is_best'])
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Ensure templates and static directories exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    # Get port from environment variable (for deployment platforms)
    port = int(os.environ.get('PORT', 5000))
    
    # Check if running in production or development
    is_production = os.environ.get('FLASK_ENV') == 'production'
    
    print("\n" + "=" * 60)
    print("🚀 STARTING FLASK APPLICATION")
    print("=" * 60)
    if not is_production:
        print("📍 Open your browser and navigate to:")
        print(f"   http://localhost:{port}")
    print("=" * 60 + "\n")
    
    app.run(debug=not is_production, host='0.0.0.0', port=port)
