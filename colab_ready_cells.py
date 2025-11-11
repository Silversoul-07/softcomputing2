"""
GOOGLE COLAB READY CELLS
========================
Copy and paste these cells directly into Google Colab

Each section marked with ### CELL X ### should go in a separate Colab cell
"""

### CELL 1: Install Dependencies ###
# Run this first to install all required packages
"""
!pip install --quiet pytorch-tabular scikit-learn imbalanced-learn xgboost lightgbm torch
"""

### CELL 2: Upload the diabetes_model_trainer.py file ###
# Upload the trainer module
"""
from google.colab import files
uploaded = files.upload()  # Select and upload diabetes_model_trainer.py
"""

### CELL 3: Load Your Data ###
# Load your diabetes dataset(s)
"""
import pandas as pd
from sklearn.impute import SimpleImputer

# Option A: Upload CSV files
from google.colab import files
uploaded_data = files.upload()  # Upload your CSV file(s)

# Load single file
df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

# OR load multiple files and combine
# df1 = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
# df2 = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
# df3 = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
# df = pd.concat([df1, df2, df3], ignore_index=True)

# Clean data
if 'Diabetes_012' in df.columns:
    df = df.drop(columns=['Diabetes_012'])

df = df.drop_duplicates()

# Handle missing values
imputer = SimpleImputer(strategy='median')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print(f"✅ Loaded {len(df)} samples")
print(f"📊 Features: {list(df.columns)}")
print(f"🎯 Target distribution:\\n{df['Diabetes_binary'].value_counts()}")
"""

### CELL 4: Import and Run Training ###
# Train all models with one function call
"""
from diabetes_model_trainer import train_and_test

# Train all models (this will take 10-20 minutes)
results = train_and_test(
    df=df,
    target_col='Diabetes_binary',
    test_size=0.2,
    use_smote=True,
    n_features=50,
    train_traditional=True,      # XGBoost, LightGBM
    train_deep_learning=True,    # FT-Transformer, TabNet
    create_ensemble=True         # Ensemble models
)

print("\\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
"""

### CELL 5: View Results ###
# Display comprehensive results
"""
import pandas as pd

# Show all results in a nice table
print("\\n📊 ALL MODEL RESULTS:")
print("="*80)
results_display = results['summary_df'].copy()
print(results_display.to_string(index=False))

# Highlight best model
print("\\n" + "="*80)
print("🏆 BEST PERFORMING MODEL")
print("="*80)
best = results['best_model']
print(f"Model:      {best['Model']}")
print(f"Type:       {best['Type']}")
print(f"Accuracy:   {best['Accuracy']:.4f}")
print(f"Precision:  {best['Precision']:.4f}")
print(f"Recall:     {best['Recall']:.4f}")
print(f"F1 Score:   {best['F1 Score']:.4f}")
print(f"AUC:        {best['AUC']:.4f}")
print("="*80)
"""

### CELL 6: Compare with Baseline ###
# Compare your results with the original notebook baseline
"""
# Original notebook baseline (from the notebook you shared)
baseline_results = {
    'LightGBM (SMOTE)': {'accuracy': 0.863, 'auc': 0.823},
    'FT-Transformer (SMOTE)': {'accuracy': 0.915, 'auc': 0.766},
    'FT-Transformer + LightGBM Ensemble': {'accuracy': 0.923, 'auc': 0.778}
}

print("\\n" + "="*80)
print("📈 COMPARISON WITH ORIGINAL NOTEBOOK BASELINE")
print("="*80)

print("\\n🔵 Original Baseline:")
for model, metrics in baseline_results.items():
    print(f"  {model:40s} → Acc: {metrics['accuracy']:.3f}, AUC: {metrics['auc']:.3f}")

print("\\n🟢 Our Best Model:")
best = results['best_model']
print(f"  {best['Model']:40s} → Acc: {best['Accuracy']:.3f}, AUC: {best['AUC']:.3f}")

# Calculate improvement
baseline_best_acc = 0.923  # Original best accuracy
improvement_acc = (best['Accuracy'] - baseline_best_acc) * 100

print("\\n" + "="*80)
if improvement_acc > 0:
    print(f"🎉 CONGRATULATIONS! You BEAT the baseline by {improvement_acc:.2f}% accuracy!")
else:
    print(f"📊 Result: {improvement_acc:.2f}% vs baseline (close competition!)")
print("="*80)
"""

### CELL 7 (OPTIONAL): Access Individual Model Predictions ###
# Get predictions from specific models
"""
# Access XGBoost model
if 'XGBoost' in results['individual_results']:
    xgb_model = results['individual_results']['XGBoost']['model']
    xgb_predictions = results['individual_results']['XGBoost']['predictions']
    xgb_probabilities = results['individual_results']['XGBoost']['predictions_proba']

    print("XGBoost model available for predictions!")
    print(f"Sample predictions: {xgb_predictions[:10]}")
    print(f"Sample probabilities: {xgb_probabilities[:10]}")

# Access best ensemble predictions
if results['ensemble_results']:
    # Find best ensemble
    best_ensemble_name = max(results['ensemble_results'].items(),
                            key=lambda x: x[1]['accuracy'])[0]
    best_ensemble_preds = results['ensemble_results'][best_ensemble_name]['predictions']
    best_ensemble_proba = results['ensemble_results'][best_ensemble_name]['predictions_proba']

    print(f"\\nBest Ensemble: {best_ensemble_name}")
    print(f"Sample predictions: {best_ensemble_preds[:10]}")
"""

### CELL 8 (OPTIONAL): Quick Training - Skip Deep Learning ###
# If you want faster training, skip deep learning models
"""
# This trains only XGBoost and LightGBM (much faster, ~2-3 minutes)
results_quick = train_and_test(
    df=df,
    target_col='Diabetes_binary',
    test_size=0.2,
    train_traditional=True,      # Keep traditional models
    train_deep_learning=False,   # Skip deep learning (saves time)
    create_ensemble=True
)

print("\\n📊 Quick Training Results:")
print(results_quick['summary_df'].to_string(index=False))
"""

### CELL 9 (OPTIONAL): Feature Importance Analysis ###
# Analyze which features are most important
"""
import matplotlib.pyplot as plt
import seaborn as sns

# Get selected features
selected_features = results['selected_features']
print(f"\\n🎯 Top {len(selected_features)} Selected Features:")
print(selected_features)

# Plot feature importance from XGBoost
if 'XGBoost' in results['individual_results']:
    xgb_model = results['individual_results']['XGBoost']['model']

    # Get feature importance
    importance = xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': importance
    }).sort_values('Importance', ascending=False).head(20)

    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Top 20 Most Important Features (XGBoost)', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()

    print("\\n📊 Top 10 Most Important Features:")
    print(feature_importance_df.head(10).to_string(index=False))
"""

### CELL 10 (OPTIONAL): Save Best Model ###
# Save your best model for later use
"""
import pickle

# Save best traditional model (XGBoost or LightGBM)
best_traditional = max(
    [(name, res) for name, res in results['individual_results'].items()
     if name in ['XGBoost', 'LightGBM']],
    key=lambda x: x[1]['accuracy']
)

model_name, model_results = best_traditional
model = model_results['model']

# Save model
with open(f'best_{model_name.lower()}_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save scaler and feature selector
with open('scaler.pkl', 'wb') as f:
    pickle.dump(results['scaler'], f)

with open('feature_selector.pkl', 'wb') as f:
    pickle.dump(results['feature_selector'], f)

# Save selected features list
with open('selected_features.pkl', 'wb') as f:
    pickle.dump(results['selected_features'], f)

print(f"✅ Saved {model_name} model and preprocessing objects!")
print("Files saved:")
print(f"  - best_{model_name.lower()}_model.pkl")
print("  - scaler.pkl")
print("  - feature_selector.pkl")
print("  - selected_features.pkl")

# Download files
from google.colab import files
files.download(f'best_{model_name.lower()}_model.pkl')
files.download('scaler.pkl')
files.download('feature_selector.pkl')
files.download('selected_features.pkl')
"""

### COMPLETE SINGLE CELL VERSION ###
# If you prefer everything in one cell (paste this in a single Colab cell):
"""
# Install dependencies
!pip install --quiet pytorch-tabular scikit-learn imbalanced-learn xgboost lightgbm torch

# Upload trainer file
from google.colab import files
print("📁 Upload diabetes_model_trainer.py:")
uploaded = files.upload()

# Upload data
print("\\n📁 Upload your diabetes CSV file(s):")
uploaded_data = files.upload()

# Load and prepare data
import pandas as pd
from sklearn.impute import SimpleImputer

# Adjust filename as needed
df = pd.read_csv(list(uploaded_data.keys())[0])

if 'Diabetes_012' in df.columns:
    df = df.drop(columns=['Diabetes_012'])
df = df.drop_duplicates()

imputer = SimpleImputer(strategy='median')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print(f"✅ Loaded {len(df)} samples")

# Train models
from diabetes_model_trainer import train_and_test

results = train_and_test(
    df=df,
    target_col='Diabetes_binary',
    test_size=0.2
)

# Show results
print("\\n" + "="*80)
print("🏆 FINAL RESULTS")
print("="*80)
print(results['summary_df'].to_string(index=False))

print("\\n🥇 BEST MODEL:")
best = results['best_model']
print(f"  {best['Model']} → Accuracy: {best['Accuracy']:.4f}, AUC: {best['AUC']:.4f}")

# Compare with baseline
baseline_best = 0.923
improvement = (best['Accuracy'] - baseline_best) * 100
print(f"\\n📈 Improvement over baseline: {improvement:+.2f}%")
"""
