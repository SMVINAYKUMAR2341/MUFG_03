import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, make_scorer)
import warnings
import os
import time
import joblib
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs('results', exist_ok=True)
os.makedirs('visualizations/tuning', exist_ok=True)

print("="*70)
print("PHASE 3: HYPERPARAMETER OPTIMIZATION")
print("="*70)

# Load preprocessed data
print("\n1. LOADING PREPROCESSED DATA")
print("-"*70)
X_train = pd.read_csv('data/processed/X_train_scaled.csv')
X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# Define hyperparameter grids
print("\n2. DEFINING HYPERPARAMETER GRIDS")
print("-"*70)

param_grids = {
    'Decision Tree': {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    },
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }
}

for model_name, grid in param_grids.items():
    print(f"\n{model_name}:")
    print(f"  Grid size: {np.prod([len(v) for v in grid.values()])} combinations")
    for param, values in grid.items():
        print(f"  {param}: {values}")

# Initialize models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Grid Search with Cross-Validation
print("\n3. PERFORMING GRID SEARCH WITH CROSS-VALIDATION")
print("-"*70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = 'roc_auc'  # Use ROC-AUC as primary metric

tuning_results = {}
best_models = {}

for model_name in models.keys():
    print(f"\n{model_name}:")
    print(f"  Starting grid search...")
    
    start_time = time.time()
    
    grid_search = GridSearchCV(
        estimator=models[model_name],
        param_grid=param_grids[model_name],
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    
    # Predictions
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    tuning_results[model_name] = {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba),
        'training_time': elapsed_time,
        'y_pred': y_test_pred,
        'y_proba': y_test_proba,
        'cv_results': grid_search.cv_results_
    }
    
    print(f"  Completed in {elapsed_time:.2f} seconds")
    print(f"  Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    print(f"  Test ROC-AUC: {tuning_results[model_name]['roc_auc']:.4f}")
    print(f"  Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"    {param}: {value}")

# Compare optimized models
print("\n4. OPTIMIZED MODEL COMPARISON")
print("-"*70)

comparison_df = pd.DataFrame({
    'Model': list(tuning_results.keys()),
    'Best CV ROC-AUC': [tuning_results[m]['best_cv_score'] for m in tuning_results.keys()],
    'Test Accuracy': [tuning_results[m]['test_accuracy'] for m in tuning_results.keys()],
    'Precision': [tuning_results[m]['precision'] for m in tuning_results.keys()],
    'Recall': [tuning_results[m]['recall'] for m in tuning_results.keys()],
    'F1-Score': [tuning_results[m]['f1'] for m in tuning_results.keys()],
    'Test ROC-AUC': [tuning_results[m]['roc_auc'] for m in tuning_results.keys()],
    'Training Time (s)': [tuning_results[m]['training_time'] for m in tuning_results.keys()]
})

comparison_df = comparison_df.sort_values('Test ROC-AUC', ascending=False)
print(comparison_df.to_string(index=False))

# Save results
comparison_df.to_csv('results/optimized_model_comparison.csv', index=False)
print("\nSaved: results/optimized_model_comparison.csv")

# Save best parameters
best_params_df = pd.DataFrame([
    {'Model': model, 'Parameter': param, 'Value': value}
    for model in tuning_results.keys()
    for param, value in tuning_results[model]['best_params'].items()
])
best_params_df.to_csv('results/best_hyperparameters.csv', index=False)
print("Saved: results/best_hyperparameters.csv")

# Detailed classification reports
print("\n5. DETAILED CLASSIFICATION REPORTS (OPTIMIZED MODELS)")
print("-"*70)

for model_name in tuning_results.keys():
    print(f"\n{model_name}:")
    print(classification_report(y_test, tuning_results[model_name]['y_pred'],
                                target_names=['No Disease', 'Disease']))

# Load baseline results for comparison
print("\n6. BASELINE VS OPTIMIZED COMPARISON")
print("-"*70)

baseline_df = pd.read_csv('results/baseline_model_comparison.csv')

improvement_df = pd.DataFrame({
    'Model': comparison_df['Model'],
    'Baseline ROC-AUC': [baseline_df[baseline_df['Model'] == m]['ROC-AUC'].values[0] 
                         for m in comparison_df['Model']],
    'Optimized ROC-AUC': comparison_df['Test ROC-AUC'],
    'Improvement': comparison_df['Test ROC-AUC'].values - 
                   [baseline_df[baseline_df['Model'] == m]['ROC-AUC'].values[0] 
                    for m in comparison_df['Model']],
    'Improvement %': ((comparison_df['Test ROC-AUC'].values - 
                       [baseline_df[baseline_df['Model'] == m]['ROC-AUC'].values[0] 
                        for m in comparison_df['Model']]) /
                      [baseline_df[baseline_df['Model'] == m]['ROC-AUC'].values[0] 
                       for m in comparison_df['Model']] * 100)
})

print(improvement_df.to_string(index=False))
improvement_df.to_csv('results/baseline_vs_optimized.csv', index=False)
print("\nSaved: results/baseline_vs_optimized.csv")

# VISUALIZATIONS
print("\n7. GENERATING VISUALIZATIONS")
print("-"*70)

# 1. Baseline vs Optimized Comparison
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(improvement_df))
width = 0.35

bars1 = ax.bar(x - width/2, improvement_df['Baseline ROC-AUC'], width,
               label='Baseline', color='skyblue')
bars2 = ax.bar(x + width/2, improvement_df['Optimized ROC-AUC'], width,
               label='Optimized', color='darkgreen')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('ROC-AUC Score', fontsize=12)
ax.set_title('Baseline vs Optimized Model Performance', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(improvement_df['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('visualizations/tuning/baseline_vs_optimized.png', dpi=300, bbox_inches='tight')
print("Saved: baseline_vs_optimized.png")
plt.close()

# 2. Improvement Percentage
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['green' if x > 0 else 'red' for x in improvement_df['Improvement %']]
bars = ax.barh(improvement_df['Model'], improvement_df['Improvement %'], color=colors)

ax.set_xlabel('Improvement (%)', fontsize=12)
ax.set_title('Performance Improvement After Optimization', fontsize=16, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    label_x_pos = width + (1 if width > 0 else -1)
    ax.text(label_x_pos, bar.get_y() + bar.get_height()/2,
            f'{width:.2f}%', ha='left' if width > 0 else 'right',
            va='center', fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/tuning/improvement_percentage.png', dpi=300, bbox_inches='tight')
print("Saved: improvement_percentage.png")
plt.close()

# 3. Training Time Comparison
fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(comparison_df['Model'], comparison_df['Training Time (s)'], color='coral')
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Training Time (seconds)', fontsize=12)
ax.set_title('Hyperparameter Tuning Time', fontsize=16, fontweight='bold')
ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for i, (model, time_val) in enumerate(zip(comparison_df['Model'], 
                                            comparison_df['Training Time (s)'])):
    ax.text(i, time_val, f'{time_val:.1f}s', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/tuning/training_time.png', dpi=300, bbox_inches='tight')
print("Saved: training_time.png")
plt.close()

# 4. Metrics Heatmap
metrics_data = comparison_df[['Model', 'Test Accuracy', 'Precision', 'Recall', 
                               'F1-Score', 'Test ROC-AUC']].set_index('Model')

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(metrics_data.T, annot=True, fmt='.3f', cmap='YlGnBu',
            cbar_kws={'label': 'Score'}, ax=ax)
ax.set_title('Optimized Model Metrics Heatmap', fontsize=16, fontweight='bold')
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Metric', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/tuning/metrics_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: metrics_heatmap.png")
plt.close()

# Save best model
print("\n8. SAVING BEST MODEL")
print("-"*70)

best_model_name = comparison_df.iloc[0]['Model']
best_model = best_models[best_model_name]
best_auc = comparison_df.iloc[0]['Test ROC-AUC']

joblib.dump(best_model, 'models/best_model.pkl')
print(f"Saved best model ({best_model_name}) to models/best_model.pkl")

# Save all optimized models
for model_name, model in best_models.items():
    filename = f"models/{model_name.lower().replace(' ', '_')}_optimized.pkl"
    joblib.dump(model, filename)
    print(f"Saved {model_name} to {filename}")

# Summary
print("\n" + "="*70)
print("HYPERPARAMETER OPTIMIZATION COMPLETE!")
print("="*70)

print(f"\nBest Model: {best_model_name}")
print(f"Best Test ROC-AUC: {best_auc:.4f}")
print(f"\nBest Hyperparameters:")
for param, value in tuning_results[best_model_name]['best_params'].items():
    print(f"  {param}: {value}")

print(f"\nPerformance Metrics:")
print(f"  Accuracy: {tuning_results[best_model_name]['test_accuracy']:.4f}")
print(f"  Precision: {tuning_results[best_model_name]['precision']:.4f}")
print(f"  Recall: {tuning_results[best_model_name]['recall']:.4f}")
print(f"  F1-Score: {tuning_results[best_model_name]['f1']:.4f}")

print(f"\nAll results saved in 'results/' directory")
print(f"All models saved in 'models/' directory")
print(f"Visualizations saved in 'visualizations/tuning/' directory")
print("="*70)
