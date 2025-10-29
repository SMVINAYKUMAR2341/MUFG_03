import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import os
import joblib
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs('visualizations', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("HEART DISEASE DATASET - ANALYSIS & PREPROCESSING")
print("="*70)

# Load dataset
print("\n1. LOADING DATA...")
df = pd.read_csv('heart_disease_dataset.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Basic Info
print("\n2. DATASET OVERVIEW")
print("-"*70)
print(df.head(10))
print(f"\nShape: {df.shape}")
print(f"\nData Types:\n{df.dtypes}")

# Statistical Summary
print("\n3. STATISTICAL SUMMARY")
print("-"*70)
print(df.describe())

# Missing Values
print("\n4. DATA QUALITY CHECK")
print("-"*70)
missing = df.isnull().sum()
print(f"Missing Values:\n{missing}")
print(f"Total Missing: {missing.sum()}")

duplicates = df.duplicated().sum()
print(f"Duplicate Rows: {duplicates}")

# Target Distribution
print("\n5. TARGET VARIABLE ANALYSIS")
print("-"*70)
print(df['heart_disease'].value_counts())
print(f"\nPercentage:\n{df['heart_disease'].value_counts(normalize=True) * 100}")

# Feature Analysis
print("\n6. FEATURE ANALYSIS")
print("-"*70)

categorical_features = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 
                        'resting_ecg', 'exercise_induced_angina', 'st_slope',
                        'num_major_vessels', 'thalassemia']

numerical_features = ['age', 'resting_blood_pressure', 'cholesterol', 
                      'max_heart_rate', 'st_depression']

for feature in categorical_features:
    if feature in df.columns:
        print(f"\n{feature}:\n{df[feature].value_counts().sort_index()}")

# Correlation
print("\n7. CORRELATION ANALYSIS")
print("-"*70)
corr_matrix = df.corr()
target_corr = corr_matrix['heart_disease'].sort_values(ascending=False)
print(f"\nCorrelation with target:\n{target_corr}")

# Outliers
print("\n8. OUTLIER DETECTION")
print("-"*70)
for feature in numerical_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[feature] < Q1 - 1.5*IQR) | (df[feature] > Q3 + 1.5*IQR)]
    print(f"{feature}: {len(outliers)} outliers")

# PREPROCESSING
print("\n" + "="*70)
print("PREPROCESSING PIPELINE")
print("="*70)

df_processed = df.copy()

# Handle missing values
print("\n9. HANDLING MISSING VALUES")
if df_processed.isnull().sum().sum() > 0:
    for col in numerical_features:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    for col in categorical_features:
        if col in df_processed.columns and df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    print("Missing values handled")
else:
    print("No missing values")

# Remove duplicates
print("\n10. REMOVING DUPLICATES")
if duplicates > 0:
    df_processed.drop_duplicates(inplace=True)
    print(f"Removed {duplicates} duplicates")
else:
    print("No duplicates")

# Split features and target
print("\n11. SPLITTING FEATURES AND TARGET")
X = df_processed.drop('heart_disease', axis=1)
y = df_processed['heart_disease']
print(f"Features: {X.shape}, Target: {y.shape}")

# Train-test split
print("\n12. TRAIN-TEST SPLIT")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training: {X_train.shape[0]}, Testing: {X_test.shape[0]}")
print(f"Train target:\n{y_train.value_counts()}")
print(f"Test target:\n{y_test.value_counts()}")

# Feature Scaling
print("\n13. FEATURE SCALING")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
print("Features scaled using StandardScaler")

# Save preprocessed data
print("\n14. SAVING PREPROCESSED DATA")
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
X_train_scaled_df.to_csv('data/processed/X_train_scaled.csv', index=False)
X_test_scaled_df.to_csv('data/processed/X_test_scaled.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

# Save scaler
joblib.dump(scaler, 'data/processed/scaler.pkl')
print("All files saved to data/processed/")

# VISUALIZATIONS
print("\n15. GENERATING VISUALIZATIONS")
print("-"*70)

# 1. Target Distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=df, x='heart_disease', palette='Set2')
plt.title('Heart Disease Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Heart Disease (0=No, 1=Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom')
plt.tight_layout()
plt.savefig('visualizations/target_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: target_distribution.png")
plt.close()

# 2. Correlation Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: correlation_heatmap.png")
plt.close()

# 3. Age Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='age', hue='heart_disease', bins=20, 
             kde=True, palette='Set1', alpha=0.6)
plt.title('Age Distribution by Heart Disease', fontsize=16, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Heart Disease', labels=['No', 'Yes'])
plt.tight_layout()
plt.savefig('visualizations/age_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: age_distribution.png")
plt.close()

# 4. Numerical Features Boxplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, feature in enumerate(numerical_features):
    sns.boxplot(data=df, x='heart_disease', y=feature, palette='Set2', ax=axes[idx])
    axes[idx].set_title(f'{feature} by Heart Disease', fontweight='bold')
    axes[idx].set_xlabel('Heart Disease')
axes[-1].axis('off')
plt.tight_layout()
plt.savefig('visualizations/numerical_features_boxplots.png', dpi=300, bbox_inches='tight')
print("Saved: numerical_features_boxplots.png")
plt.close()

# 5. Pairplot for key features
print("Generating pairplot (may take a moment)...")
key_features = ['age', 'cholesterol', 'max_heart_rate', 'st_depression', 'heart_disease']
pairplot_df = df[key_features].sample(min(200, len(df)), random_state=42)  # Sample for speed
pairplot = sns.pairplot(pairplot_df, hue='heart_disease', palette='Set1', diag_kind='kde')
pairplot.fig.suptitle('Pairplot of Key Features', y=1.02, fontsize=16, fontweight='bold')
plt.savefig('visualizations/pairplot_key_features.png', dpi=300, bbox_inches='tight')
print("Saved: pairplot_key_features.png")
plt.close()

# Summary
print("\n" + "="*70)
print("PREPROCESSING COMPLETE!")
print("="*70)
print(f"Processed: {df_processed.shape[0]} rows, {df_processed.shape[1]} columns")
print(f"Training: {X_train.shape[0]} samples (80%)")
print(f"Testing: {X_test.shape[0]} samples (20%)")
print(f"Features: {X.shape[1]}")
print("\nData ready for modeling!")
print("="*70)
