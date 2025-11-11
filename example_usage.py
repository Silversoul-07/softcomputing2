"""
Example usage of diabetes_model_trainer in Google Colab
========================================================

This script demonstrates how to use the comprehensive diabetes model trainer
to beat baseline performance.
"""

# ============================================================================
# STEP 1: Install dependencies (run this in a Colab cell)
# ============================================================================
"""
!pip install --quiet pytorch-tabular scikit-learn imbalanced-learn xgboost lightgbm pandas numpy
"""

# ============================================================================
# STEP 2: Import the trainer
# ============================================================================
from diabetes_model_trainer import train_and_test
import pandas as pd

# ============================================================================
# STEP 3: Load your data
# ============================================================================
# Option A: Load from multiple CSV files (as in the notebook)
df1 = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
df2 = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
df3 = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
df = pd.concat([df1, df2, df3], ignore_index=True)

# Drop the Diabetes_012 column and keep only Diabetes_binary
if 'Diabetes_012' in df.columns:
    df = df.drop(columns=['Diabetes_012'])

# Handle missing values
from sklearn.impute import SimpleImputer
df = df.drop_duplicates()
imputer = SimpleImputer(strategy='median')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Option B: Load from single file
# df = pd.read_csv("your_diabetes_data.csv")

print(f"Loaded {len(df)} samples")
print(f"Class distribution: {df['Diabetes_binary'].value_counts().to_dict()}")

# ============================================================================
# STEP 4: Train all models
# ============================================================================
print("\n" + "="*80)
print("Starting comprehensive model training...")
print("="*80)

results = train_and_test(
    df=df,
    target_col='Diabetes_binary',
    test_size=0.2,
    use_smote=True,
    n_features=50,
    scaler_type='robust',
    train_traditional=True,      # Train XGBoost and LightGBM
    train_deep_learning=True,    # Train FT-Transformer and TabNet
    create_ensemble=True         # Create ensemble models
)

# ============================================================================
# STEP 5: View results
# ============================================================================
print("\n" + "="*80)
print("📊 ALL RESULTS")
print("="*80)
print(results['summary_df'].to_string(index=False))

print("\n" + "="*80)
print("🏆 BEST MODEL")
print("="*80)
for key, value in results['best_model'].items():
    print(f"  {key}: {value}")

# ============================================================================
# STEP 6: Access specific models
# ============================================================================
# Get XGBoost model and predictions
if 'XGBoost' in results['individual_results']:
    xgb_model = results['individual_results']['XGBoost']['model']
    xgb_accuracy = results['individual_results']['XGBoost']['accuracy']
    print(f"\nXGBoost Accuracy: {xgb_accuracy:.4f}")

# Get best ensemble predictions
if results['ensemble_results']:
    best_ensemble = max(results['ensemble_results'].items(),
                       key=lambda x: x[1]['accuracy'])
    ensemble_name, ensemble_metrics = best_ensemble
    print(f"\nBest Ensemble: {ensemble_name}")
    print(f"Accuracy: {ensemble_metrics['accuracy']:.4f}")
    print(f"AUC: {ensemble_metrics['auc']:.4f}")

# ============================================================================
# STEP 7: Quick training (if you want to skip some models)
# ============================================================================
# Train only traditional models (faster)
"""
results_quick = train_and_test(
    df=df,
    target_col='Diabetes_binary',
    test_size=0.2,
    train_traditional=True,
    train_deep_learning=False,  # Skip deep learning for speed
    create_ensemble=False
)
"""

# Train only deep learning models
"""
results_dl = train_and_test(
    df=df,
    target_col='Diabetes_binary',
    test_size=0.2,
    train_traditional=False,
    train_deep_learning=True,
    create_ensemble=False
)
"""

print("\n✅ Training complete!")
