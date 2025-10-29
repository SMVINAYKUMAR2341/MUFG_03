# Heart Disease Detection - Capstone Project
## Data Science Classification Challenge

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

A comprehensive machine learning project for predicting heart disease using clinical diagnostic data, featuring data exploration, model optimization, and production-ready REST API deployment.

## 🎯 Project Overview

This capstone project implements a complete end-to-end machine learning pipeline for heart disease prediction:

- ✅ **Data Exploration & Preprocessing** - Comprehensive EDA with visualizations
- ✅ **Multiple Classification Algorithms** - Decision Trees, Random Forest, Logistic Regression, SVM
- ✅ **Hyperparameter Optimization** - Grid Search with Stratified K-Fold Cross-Validation
- ✅ **Model Evaluation** - Accuracy, Precision, Recall, F1-Score, ROC-AUC
- ✅ **Feature Analysis** - Feature importance and correlation analysis
- ✅ **REST API** - FastAPI with Pydantic validation
- ✅ **Docker Deployment** - Containerized application ready for production

## 📊 Dataset

**400 patient records** with 14 features:

| Category | Features |
|----------|----------|
| **Demographics** | age, sex |
| **Clinical Measurements** | chest_pain_type, resting_blood_pressure, cholesterol, fasting_blood_sugar |
| **Diagnostic Tests** | resting_ecg, max_heart_rate, exercise_induced_angina, st_depression, st_slope, num_major_vessels, thalassemia |
| **Target** | heart_disease (0=No, 1=Yes) |

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) Docker for containerized deployment

### Installation

1. **Clone or download the project:**
```powershell
cd d:\CapstoneProject2
```

2. **Install dependencies:**
```powershell
pip install -r requirements.txt
```

### Running the Project

#### Phase 1: Data Preprocessing & EDA
```powershell
python analysis_preprocessing.py
```

**Outputs:**
- `data/processed/X_train.csv`, `X_test.csv` - Train/test features
- `data/processed/X_train_scaled.csv`, `X_test_scaled.csv` - Scaled features
- `data/processed/y_train.csv`, `y_test.csv` - Target variables
- `data/processed/scaler.pkl` - Fitted StandardScaler
- `visualizations/*.png` - EDA visualizations

#### Phase 2: Baseline Model Training
```powershell
python model_training.py
```

**Outputs:**
- `results/baseline_model_comparison.csv` - Model performance metrics
- `visualizations/models/*.png` - Model visualizations
- Confusion matrices, ROC curves, feature importance

#### Phase 3: Hyperparameter Tuning
```powershell
python model_tuning.py
```

**Outputs:**
- `models/best_model.pkl` - Best performing model
- `models/*_optimized.pkl` - All optimized models
- `results/optimized_model_comparison.csv` - Tuned model metrics
- `results/best_hyperparameters.csv` - Optimal hyperparameters
- `visualizations/tuning/*.png` - Optimization visualizations

### Running the API

#### Option 1: Local Development
```powershell
# Start the FastAPI server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Option 2: Docker Deployment
```powershell
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

**Access the API:**
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

## 📡 API Endpoints

### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "scaler_status": "loaded"
}
```

### 2. Model Information
```http
GET /model-info
```

### 3. Single Prediction
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Heart Disease Detected",
  "probability": 0.85,
  "risk_level": "Very High Risk",
  "message": "The model predicts heart disease with 85.0% confidence. Very High Risk. Please consult with a healthcare professional."
}
```

### 4. Batch Prediction
```http
POST /batch-predict
Content-Type: application/json
```

**Request:** Array of patient data objects

## 📈 Model Performance

| Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|---------------|-----------|--------|----------|---------|
| **Random Forest (Optimized)** | 0.675 | 0.667 | 0.818 | 0.735 | **0.744** |
| Logistic Regression | 0.650 | 0.643 | 0.818 | 0.720 | 0.739 |
| SVM | 0.550 | 0.550 | 1.000 | 0.710 | 0.741 |
| Decision Tree | 0.588 | 0.600 | 0.750 | 0.667 | 0.623 |

**Best Model:** Random Forest with optimized hyperparameters
- **ROC-AUC:** 0.744
- **Recall:** 0.818 (High sensitivity - good for medical screening)
- **F1-Score:** 0.735

### Optimized Hyperparameters (Random Forest)
- `n_estimators`: 50
- `max_depth`: 7
- `max_features`: 'sqrt'
- `min_samples_split`: 5
- `min_samples_leaf`: 1

## 📁 Project Structure

```
CapstoneProject2/
├── data/
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── X_train_scaled.csv
│       ├── X_test_scaled.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       └── scaler.pkl
├── models/
│   ├── best_model.pkl
│   ├── decision_tree_optimized.pkl
│   ├── random_forest_optimized.pkl
│   ├── logistic_regression_optimized.pkl
│   └── svm_optimized.pkl
├── results/
│   ├── baseline_model_comparison.csv
│   ├── optimized_model_comparison.csv
│   ├── best_hyperparameters.csv
│   └── baseline_vs_optimized.csv
├── visualizations/
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── age_distribution.png
│   ├── numerical_features_boxplots.png
│   ├── pairplot_key_features.png
│   ├── models/
│   │   ├── model_comparison.png
│   │   ├── confusion_matrices.png
│   │   ├── roc_curves.png
│   │   ├── feature_importance.png
│   │   └── decision_tree.png
│   └── tuning/
│       ├── baseline_vs_optimized.png
│       ├── improvement_percentage.png
│       ├── training_time.png
│       └── metrics_heatmap.png
├── api/
│   ├── __init__.py
│   └── main.py
├── analysis_preprocessing.py
├── model_training.py
├── model_tuning.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

## 🔧 Technologies Used

| Category | Technologies |
|----------|-------------|
| **Data Science** | pandas, numpy, scipy |
| **Visualization** | matplotlib, seaborn |
| **Machine Learning** | scikit-learn |
| **API Framework** | FastAPI, Pydantic, Uvicorn |
| **Deployment** | Docker, Docker Compose |
| **Serialization** | joblib |

## 🎓 Learning Outcomes

This project demonstrates:

- ✅ Classification algorithm implementation and comparison
- ✅ Hyperparameter tuning techniques (GridSearchCV, Cross-Validation)
- ✅ Model evaluation in healthcare contexts (emphasis on recall/sensitivity)
- ✅ Feature importance analysis and interpretability
- ✅ RESTful API development with FastAPI
- ✅ Model serialization and deployment
- ✅ Containerization with Docker
- ✅ Production-ready ML pipeline design

## 📊 Success Metrics

### ✅ Minimum Requirements Met
- [x] All 4 algorithms implemented and evaluated
- [x] Grid search optimization completed
- [x] Stratified K-fold cross-validation implemented
- [x] Comprehensive model comparison and recommendation

### 🎯 Excellence Indicators Achieved
- [x] ROC-AUC > 0.74 on test set
- [x] Comprehensive feature importance analysis
- [x] Professional-quality visualizations
- [x] Business recommendations included
- [x] Production-ready REST API deployed
- [x] Docker containerization implemented

## 🚀 Future Enhancements

- [ ] Add more ensemble methods (XGBoost, LightGBM, CatBoost)
- [ ] Implement SHAP values for model interpretability
- [ ] Add real-time monitoring and logging
- [ ] Create web frontend interface
- [ ] Implement A/B testing framework
- [ ] Add model versioning and MLOps pipeline
- [ ] Integrate with cloud deployment (AWS, Azure, GCP)

## 📝 Usage Examples

### Python Script Example
```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('data/processed/scaler.pkl')

# Prepare patient data
patient_data = {
    'age': 58, 'sex': 1, 'chest_pain_type': 1,
    'resting_blood_pressure': 134, 'cholesterol': 246,
    'fasting_blood_sugar': 0, 'resting_ecg': 0,
    'max_heart_rate': 155, 'exercise_induced_angina': 0,
    'st_depression': 0.4, 'st_slope': 1,
    'num_major_vessels': 1, 'thalassemia': 2
}

# Convert to DataFrame and scale
df = pd.DataFrame([patient_data])
df_scaled = scaler.transform(df)

# Make prediction
prediction = model.predict(df_scaled)[0]
probability = model.predict_proba(df_scaled)[0][1]

print(f"Prediction: {'Heart Disease' if prediction == 1 else 'No Disease'}")
print(f"Probability: {probability:.2%}")
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

## ⚠️ Disclaimer

This model is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## 👥 Contributors

**Capstone Project** - Data Science Classification Challenge

## 📄 License

This project is created for educational purposes and is free to use and modify.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Contact

For questions or feedback about this project, please open an issue in the repository.

---

