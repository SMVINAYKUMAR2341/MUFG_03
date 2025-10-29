import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve, auc)
import warnings
import os
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs('visualizations/models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("="*70)
print("PHASE 2: BASELINE MODEL TRAINING")
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
print(f"Target distribution - Train: {np.bincount(y_train)}")
print(f"Target distribution - Test: {np.bincount(y_test)}")

# Initialize models
print("\n2. INITIALIZING MODELS")
print("-"*70)

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(probability=True, random_state=42)
}

print(f"Models initialized: {list(models.keys())}")

# Train and evaluate models
print("\n3. TRAINING AND EVALUATING BASELINE MODELS")
print("-"*70)

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba),
        'y_pred': y_test_pred,
        'y_proba': y_test_proba,
        'confusion_matrix': confusion_matrix(y_test, y_test_pred)
    }
    
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Train Accuracy: {results[name]['train_accuracy']:.4f}")
    print(f"  Test Accuracy: {results[name]['test_accuracy']:.4f}")
    print(f"  Test ROC-AUC: {results[name]['roc_auc']:.4f}")

# Display results comparison
print("\n4. MODEL COMPARISON")
print("-"*70)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'CV ROC-AUC': [results[m]['cv_mean'] for m in results.keys()],
    'Test Accuracy': [results[m]['test_accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()],
    'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()]
})

comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
print(comparison_df.to_string(index=False))

# Save results
comparison_df.to_csv('results/baseline_model_comparison.csv', index=False)
print("\nSaved: results/baseline_model_comparison.csv")

# Detailed classification reports
print("\n5. DETAILED CLASSIFICATION REPORTS")
print("-"*70)

for name in results.keys():
    print(f"\n{name}:")
    print(classification_report(y_test, results[name]['y_pred'], 
                                target_names=['No Disease', 'Disease']))

# VISUALIZATIONS
print("\n6. GENERATING VISUALIZATIONS")
print("-"*70)

# 1. Model Comparison Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

metrics = ['Test Accuracy', 'Precision', 'Recall', 'ROC-AUC']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    comparison_df.plot(x='Model', y=metric, kind='bar', ax=ax, 
                       color='skyblue', legend=False)
    ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)

plt.tight_layout()
plt.savefig('visualizations/models/model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: model_comparison.png")
plt.close()

# 2. Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for idx, (name, result) in enumerate(results.items()):
    cm = result['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    axes[idx].set_title(f'{name} - Confusion Matrix', fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('visualizations/models/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrices.png")
plt.close()

# 3. ROC Curves
plt.figure(figsize=(12, 8))

for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_proba'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, 
             label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - All Models', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/models/roc_curves.png', dpi=300, bbox_inches='tight')
print("Saved: roc_curves.png")
plt.close()

# 4. Feature Importance (for tree-based models)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Decision Tree
dt_model = results['Decision Tree']['model']
feature_names = X_train.columns
dt_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

axes[0].barh(dt_importance['feature'], dt_importance['importance'], color='steelblue')
axes[0].set_xlabel('Importance', fontsize=12)
axes[0].set_title('Decision Tree - Top 10 Features', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()

# Random Forest
rf_model = results['Random Forest']['model']
rf_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

axes[1].barh(rf_importance['feature'], rf_importance['importance'], color='darkgreen')
axes[1].set_xlabel('Importance', fontsize=12)
axes[1].set_title('Random Forest - Top 10 Features', fontsize=14, fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('visualizations/models/feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: feature_importance.png")
plt.close()

# 5. Decision Tree Visualization
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=feature_names, 
          class_names=['No Disease', 'Disease'],
          filled=True, rounded=True, fontsize=10, max_depth=3)
plt.title('Decision Tree Structure (Max Depth = 3)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/models/decision_tree.png', dpi=300, bbox_inches='tight')
print("Saved: decision_tree.png")
plt.close()

# Summary
print("\n" + "="*70)
print("BASELINE MODEL TRAINING COMPLETE!")
print("="*70)

best_model = comparison_df.iloc[0]['Model']
best_auc = comparison_df.iloc[0]['ROC-AUC']
print(f"\nBest Model: {best_model}")
print(f"Best ROC-AUC: {best_auc:.4f}")
print(f"\nAll models trained and evaluated successfully!")
print(f"Results saved in 'results/' directory")
print(f"Visualizations saved in 'visualizations/models/' directory")
print("="*70)
