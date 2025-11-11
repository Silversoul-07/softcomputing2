"""
Comprehensive Diabetes Prediction Model Trainer
================================================
This module provides advanced preprocessing, feature engineering, and training
for multiple state-of-the-art models to beat baseline performance.

Models included:
- XGBoost
- LightGBM
- FT-Transformer
- TabNet (Google's deep learning model for tabular data)
- Model Ensembles

Usage:
    from diabetes_model_trainer import train_and_test

    results = train_and_test(
        train_path='diabetes_data.csv',
        target_col='Diabetes_binary',
        test_size=0.2
    )
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report
)

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import torch
import torch.serialization
from pytorch_tabular import TabularModel
from pytorch_tabular.models import (
    FTTransformerConfig,
    TabNetModelConfig,
    CategoryEmbeddingModelConfig
)
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig

# Monkey patch torch.load to handle weights_only parameter
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load


class AdvancedFeatureEngineering:
    """
    Advanced feature engineering for diabetes prediction
    """

    def __init__(self):
        self.feature_names = []

    def create_features(self, df):
        """
        Create comprehensive feature set with domain knowledge

        Args:
            df: Input dataframe

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        print("🔧 Starting advanced feature engineering...")

        # ===== 1. BMI-based features =====
        print("  → Creating BMI-based features...")
        # BMI categories (CDC standard)
        df['BMI_underweight'] = (df['BMI'] < 18.5).astype(int)
        df['BMI_normal'] = ((df['BMI'] >= 18.5) & (df['BMI'] < 25)).astype(int)
        df['BMI_overweight'] = ((df['BMI'] >= 25) & (df['BMI'] < 30)).astype(int)
        df['BMI_obese'] = ((df['BMI'] >= 30) & (df['BMI'] < 35)).astype(int)
        df['BMI_severely_obese'] = (df['BMI'] >= 35).astype(int)

        # BMI squared (non-linear relationship)
        df['BMI_squared'] = df['BMI'] ** 2
        df['BMI_cubed'] = df['BMI'] ** 3

        # ===== 2. Age-based features =====
        print("  → Creating age-based features...")
        # Age is categorical 1-13, create age groups
        df['age_young'] = (df['Age'] <= 4).astype(int)  # 18-39
        df['age_middle'] = ((df['Age'] >= 5) & (df['Age'] <= 9)).astype(int)  # 40-64
        df['age_senior'] = (df['Age'] >= 10).astype(int)  # 65+

        # Age squared for non-linearity
        df['Age_squared'] = df['Age'] ** 2

        # ===== 3. Interaction features =====
        print("  → Creating interaction features...")
        # BMI x Age (critical interaction for diabetes)
        df['BMI_Age_interaction'] = df['BMI'] * df['Age']
        df['BMI_Age_squared'] = df['BMI'] * (df['Age'] ** 2)

        # High-risk combinations
        df['HighRisk_Obese_Senior'] = ((df['BMI'] >= 30) & (df['Age'] >= 10)).astype(int)
        df['HighRisk_Obese_HighBP'] = ((df['BMI'] >= 30) & (df['HighBP'] == 1)).astype(int)
        df['HighRisk_Triple'] = ((df['BMI'] >= 30) & (df['HighBP'] == 1) & (df['HighChol'] == 1)).astype(int)

        # BMI x Physical Activity
        df['BMI_PhysActivity'] = df['BMI'] * df['PhysActivity']
        df['BMI_NoActivity'] = df['BMI'] * (1 - df['PhysActivity'])

        # ===== 4. Health condition combinations =====
        print("  → Creating health condition features...")
        # Count of health issues
        df['health_issues_count'] = (
            df['HighBP'] + df['HighChol'] + df['Stroke'] +
            df['HeartDiseaseorAttack'] + df['DiffWalk']
        )

        # Cardiovascular risk profile
        df['cardio_risk'] = df['HighBP'] + df['HighChol'] + df['HeartDiseaseorAttack']
        df['high_cardio_risk'] = (df['cardio_risk'] >= 2).astype(int)

        # Lifestyle risk score
        df['lifestyle_risk'] = (
            (1 - df['PhysActivity']) +
            df['Smoker'] +
            df['HvyAlcoholConsump'] +
            (1 - df['Fruits']) +
            (1 - df['Veggies'])
        )

        # ===== 5. Mental & Physical health features =====
        print("  → Creating mental/physical health features...")
        # Poor health days (mental + physical)
        df['total_poor_health_days'] = df['MentHlth'] + df['PhysHlth']
        df['has_poor_health'] = (df['total_poor_health_days'] > 0).astype(int)
        df['chronic_poor_health'] = (df['total_poor_health_days'] >= 15).astype(int)

        # Health ratio
        df['MentHlth_ratio'] = df['MentHlth'] / 30
        df['PhysHlth_ratio'] = df['PhysHlth'] / 30

        # Mental-Physical interaction
        df['MentPhys_interaction'] = df['MentHlth'] * df['PhysHlth']

        # ===== 6. Socioeconomic features =====
        print("  → Creating socioeconomic features...")
        # Healthcare access issues
        df['healthcare_barrier'] = (
            (1 - df['AnyHealthcare']) +
            df['NoDocbcCost']
        )

        # Low socioeconomic status
        df['low_ses'] = ((df['Income'] <= 3) & (df['Education'] <= 3)).astype(int)

        # Income-Education interaction
        df['Income_Education'] = df['Income'] * df['Education']

        # ===== 7. General health indicators =====
        print("  → Creating general health features...")
        # Poor general health
        df['poor_general_health'] = (df['GenHlth'] >= 4).astype(int)
        df['excellent_health'] = (df['GenHlth'] == 1).astype(int)

        # GenHlth interactions
        df['GenHlth_BMI'] = df['GenHlth'] * df['BMI']
        df['GenHlth_Age'] = df['GenHlth'] * df['Age']

        # ===== 8. Behavioral features =====
        print("  → Creating behavioral features...")
        # Healthy lifestyle score
        df['healthy_lifestyle'] = (
            df['PhysActivity'] +
            df['Fruits'] +
            df['Veggies'] +
            (1 - df['Smoker']) +
            (1 - df['HvyAlcoholConsump'])
        )

        # Compliance with preventive care
        df['preventive_care'] = df['CholCheck'] + df['AnyHealthcare']

        # ===== 9. Polynomial features for key variables =====
        print("  → Creating polynomial features...")
        # BMI polynomials
        df['BMI_log'] = np.log1p(df['BMI'])
        df['BMI_sqrt'] = np.sqrt(df['BMI'])

        # Age polynomials
        df['Age_cubed'] = df['Age'] ** 3

        # ===== 10. Binned features =====
        print("  → Creating binned features...")
        # BMI bins (more granular)
        df['BMI_bin'] = pd.cut(df['BMI'], bins=[0, 18.5, 22, 25, 27.5, 30, 35, 40, 100],
                               labels=False, duplicates='drop')

        # Age-BMI risk zones
        df['age_bmi_risk_zone'] = 0
        df.loc[(df['Age'] >= 7) & (df['BMI'] >= 25), 'age_bmi_risk_zone'] = 1
        df.loc[(df['Age'] >= 10) & (df['BMI'] >= 30), 'age_bmi_risk_zone'] = 2
        df.loc[(df['Age'] >= 11) & (df['BMI'] >= 35), 'age_bmi_risk_zone'] = 3

        print(f"✅ Feature engineering complete! Created {len(df.columns)} features")

        return df


def advanced_preprocessing(X_train, X_test, y_train, scaler_type='robust'):
    """
    Advanced preprocessing with multiple scaling options

    Args:
        X_train, X_test: Feature matrices
        y_train: Training target
        scaler_type: 'robust', 'standard', or 'power'

    Returns:
        Preprocessed X_train, X_test, fitted scaler
    """
    print(f"🔄 Applying {scaler_type} scaling...")

    if scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'power':
        scaler = PowerTransformer(method='yeo-johnson')
    else:
        scaler = StandardScaler()

    # Identify columns to scale (exclude binary)
    binary_cols = [col for col in X_train.columns if X_train[col].nunique() <= 2]
    non_binary_cols = [col for col in X_train.columns if col not in binary_cols]

    print(f"  → Scaling {len(non_binary_cols)} non-binary features")
    print(f"  → Preserving {len(binary_cols)} binary features")

    # Scale only non-binary columns
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[non_binary_cols] = scaler.fit_transform(X_train[non_binary_cols])
    X_test_scaled[non_binary_cols] = scaler.transform(X_test[non_binary_cols])

    return X_train_scaled, X_test_scaled, scaler


def select_best_features(X_train, X_test, y_train, k=50, method='f_classif'):
    """
    Advanced feature selection using multiple methods

    Args:
        X_train, X_test: Feature matrices
        y_train: Training target
        k: Number of features to select
        method: 'f_classif' or 'mutual_info'

    Returns:
        Selected X_train, X_test, selected feature names
    """
    print(f"🎯 Selecting top {k} features using {method}...")

    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    else:
        selector = SelectKBest(score_func=f_classif, k=k)

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()].tolist()

    print(f"✅ Selected features: {len(selected_features)}")

    return X_train_selected, X_test_selected, selected_features, selector


def train_traditional_models(X_train, y_train, X_test, y_test, use_smote=True):
    """
    Train traditional ML models (XGBoost, LightGBM) with optimized hyperparameters

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        use_smote: Whether to apply SMOTE

    Returns:
        Dictionary of results
    """
    print("\n" + "="*70)
    print("🚀 TRAINING TRADITIONAL ML MODELS")
    print("="*70)

    results = {}

    # Apply SMOTE if requested
    if use_smote:
        print("\n📊 Applying SMOTE for class balance...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"  → Training samples after SMOTE: {len(X_train_resampled)}")
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    # Define optimized models
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=50,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            force_col_wise=True
        )
    }

    for model_name, model in models.items():
        print(f"\n{'─'*70}")
        print(f"Training {model_name}...")
        print(f"{'─'*70}")

        # Train
        model.fit(X_train_resampled, y_train_resampled)

        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_proba)

        results[model_name] = {
            'model': model,
            'predictions': y_pred,
            'predictions_proba': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

        print(f"\n📊 {model_name} Results:")
        print(f"  ✓ Accuracy:  {accuracy:.4f}")
        print(f"  ✓ Precision: {precision:.4f}")
        print(f"  ✓ Recall:    {recall:.4f}")
        print(f"  ✓ F1 Score:  {f1:.4f}")
        print(f"  ✓ AUC Score: {auc:.4f}")

    return results


def train_deep_learning_models(X_train, y_train, X_test, y_test, feature_names, use_smote=True):
    """
    Train deep learning models (FT-Transformer, TabNet) using pytorch-tabular

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        feature_names: List of feature names
        use_smote: Whether to apply SMOTE

    Returns:
        Dictionary of results
    """
    print("\n" + "="*70)
    print("🧠 TRAINING DEEP LEARNING MODELS")
    print("="*70)

    results = {}
    target_col = 'target'

    # Apply SMOTE if requested
    if use_smote:
        print("\n📊 Applying SMOTE for class balance...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"  → Training samples after SMOTE: {len(X_train_resampled)}")
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    # Create validation split
    X_train_dl, X_val, y_train_dl, y_val = train_test_split(
        X_train_resampled, y_train_resampled,
        test_size=0.15,
        stratify=y_train_resampled,
        random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_dl_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_dl),
        columns=feature_names
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=feature_names
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names
    )

    # Prepare dataframes
    train_data = pd.concat([X_train_dl_scaled, pd.Series(y_train_dl, name=target_col).reset_index(drop=True)], axis=1)
    val_data = pd.concat([X_val_scaled, pd.Series(y_val, name=target_col).reset_index(drop=True)], axis=1)
    test_data = pd.concat([X_test_scaled, pd.Series(y_test, name=target_col).reset_index(drop=True)], axis=1)

    # Data configuration
    data_config = DataConfig(
        target=[target_col],
        continuous_cols=list(feature_names),
        categorical_cols=[]
    )

    # Trainer configuration
    trainer_config = TrainerConfig(
        auto_lr_find=False,
        batch_size=1024,
        max_epochs=50,
        early_stopping_patience=10,
        checkpoints=None,
        load_best=True
    )

    optimizer_config = OptimizerConfig()

    # Define models
    models_config = {
        'FT-Transformer': FTTransformerConfig(
            task="classification",
            learning_rate=1e-3,
            input_embed_dim=64,
            num_heads=8,
            num_attn_blocks=6,
            attn_dropout=0.2,
            ff_dropout=0.2,
            add_norm_dropout=0.2
        ),
        'TabNet': TabNetModelConfig(
            task="classification",
            learning_rate=2e-2,
            n_d=64,
            n_a=64,
            n_steps=5,
            gamma=1.5,
            lambda_sparse=1e-4,
            mask_type='entmax'
        )
    }

    for model_name, model_config in models_config.items():
        print(f"\n{'─'*70}")
        print(f"Training {model_name}...")
        print(f"{'─'*70}")

        try:
            # Build model
            tabular_model = TabularModel(
                data_config=data_config,
                model_config=model_config,
                optimizer_config=optimizer_config,
                trainer_config=trainer_config,
                verbose=False
            )

            # Train
            tabular_model.fit(train=train_data, validation=val_data)

            # Predict
            predictions = tabular_model.predict(test_data)
            y_pred = predictions[f'{target_col}_prediction'].values
            y_pred_proba = predictions[f'{target_col}_1.0_probability'].values

            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            auc = roc_auc_score(y_test, y_pred_proba)

            results[model_name] = {
                'model': tabular_model,
                'predictions': y_pred,
                'predictions_proba': y_pred_proba,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }

            print(f"\n📊 {model_name} Results:")
            print(f"  ✓ Accuracy:  {accuracy:.4f}")
            print(f"  ✓ Precision: {precision:.4f}")
            print(f"  ✓ Recall:    {recall:.4f}")
            print(f"  ✓ F1 Score:  {f1:.4f}")
            print(f"  ✓ AUC Score: {auc:.4f}")

        except Exception as e:
            print(f"  ❌ Error training {model_name}: {e}")
            continue

    return results


def create_ensembles(all_results, y_test):
    """
    Create ensemble models by averaging predictions

    Args:
        all_results: Dictionary of all model results
        y_test: Test target

    Returns:
        Dictionary of ensemble results
    """
    print("\n" + "="*70)
    print("🎭 CREATING ENSEMBLE MODELS")
    print("="*70)

    ensemble_results = {}

    # Get all available prediction probabilities
    available_models = [name for name in all_results.keys()
                       if 'predictions_proba' in all_results[name]]

    print(f"\n📋 Available models for ensembling: {available_models}")

    # Create various ensemble combinations
    ensembles = [
        (['XGBoost', 'LightGBM'], 'XGBoost + LightGBM'),
        (['XGBoost', 'FT-Transformer'], 'XGBoost + FT-Transformer'),
        (['LightGBM', 'FT-Transformer'], 'LightGBM + FT-Transformer'),
        (['XGBoost', 'LightGBM', 'FT-Transformer'], 'XGBoost + LightGBM + FT-Transformer'),
    ]

    # Add TabNet ensembles if available
    if 'TabNet' in available_models:
        ensembles.extend([
            (['XGBoost', 'TabNet'], 'XGBoost + TabNet'),
            (['LightGBM', 'TabNet'], 'LightGBM + TabNet'),
            (['FT-Transformer', 'TabNet'], 'FT-Transformer + TabNet'),
            (['XGBoost', 'LightGBM', 'TabNet'], 'XGBoost + LightGBM + TabNet'),
            (['XGBoost', 'LightGBM', 'FT-Transformer', 'TabNet'], 'All Models Ensemble'),
        ])

    for models, ensemble_name in ensembles:
        # Check if all models in ensemble are available
        if not all(model in available_models for model in models):
            continue

        print(f"\n{'─'*70}")
        print(f"Creating {ensemble_name}...")

        # Average predictions
        proba_sum = np.zeros_like(all_results[models[0]]['predictions_proba'])
        for model in models:
            proba_sum += all_results[model]['predictions_proba']

        y_pred_proba = proba_sum / len(models)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_proba)

        ensemble_results[ensemble_name] = {
            'predictions': y_pred,
            'predictions_proba': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

        print(f"  ✓ Accuracy:  {accuracy:.4f}")
        print(f"  ✓ AUC Score: {auc:.4f}")

    return ensemble_results


def print_final_summary(all_results, ensemble_results):
    """
    Print comprehensive summary of all models

    Args:
        all_results: Dictionary of individual model results
        ensemble_results: Dictionary of ensemble results
    """
    print("\n" + "="*70)
    print("🏆 FINAL RESULTS SUMMARY")
    print("="*70)

    # Combine all results
    summary_data = []

    for model_name, results in all_results.items():
        summary_data.append({
            'Model': model_name,
            'Type': 'Individual',
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1 Score': results['f1'],
            'AUC': results['auc']
        })

    for model_name, results in ensemble_results.items():
        summary_data.append({
            'Model': model_name,
            'Type': 'Ensemble',
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1 Score': results['f1'],
            'AUC': results['auc']
        })

    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Accuracy', ascending=False)

    # Format and print
    print("\n")
    print(summary_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # Highlight best model
    best_model = summary_df.iloc[0]
    print("\n" + "="*70)
    print("🥇 BEST MODEL")
    print("="*70)
    print(f"  Model:     {best_model['Model']}")
    print(f"  Type:      {best_model['Type']}")
    print(f"  Accuracy:  {best_model['Accuracy']:.4f}")
    print(f"  Precision: {best_model['Precision']:.4f}")
    print(f"  Recall:    {best_model['Recall']:.4f}")
    print(f"  F1 Score:  {best_model['F1 Score']:.4f}")
    print(f"  AUC:       {best_model['AUC']:.4f}")
    print("="*70)

    return summary_df


def train_and_test(
    data_path=None,
    df=None,
    target_col='Diabetes_binary',
    test_size=0.2,
    use_smote=True,
    n_features=50,
    scaler_type='robust',
    train_traditional=True,
    train_deep_learning=True,
    create_ensemble=True
):
    """
    Main function to train and test all models with comprehensive preprocessing

    Args:
        data_path: Path to CSV file (optional if df provided)
        df: Pandas DataFrame (optional if data_path provided)
        target_col: Name of target column
        test_size: Test set size (default 0.2)
        use_smote: Apply SMOTE for class balance (default True)
        n_features: Number of features to select (default 50)
        scaler_type: 'robust', 'standard', or 'power' (default 'robust')
        train_traditional: Train XGBoost and LightGBM (default True)
        train_deep_learning: Train FT-Transformer and TabNet (default True)
        create_ensemble: Create ensemble models (default True)

    Returns:
        Dictionary containing:
        - 'individual_results': Results from individual models
        - 'ensemble_results': Results from ensemble models
        - 'summary_df': DataFrame with all results
        - 'best_model': Best performing model info
    """

    print("\n" + "="*70)
    print("🏥 COMPREHENSIVE DIABETES PREDICTION MODEL TRAINER")
    print("="*70)

    # ===== 1. Load data =====
    if df is None:
        if data_path is None:
            raise ValueError("Either data_path or df must be provided")
        print(f"\n📂 Loading data from {data_path}...")
        df = pd.read_csv(data_path)

    print(f"  ✓ Loaded {len(df)} samples with {len(df.columns)} columns")

    # ===== 2. Feature engineering =====
    feature_engineer = AdvancedFeatureEngineering()
    df_engineered = feature_engineer.create_features(df)

    # ===== 3. Train-test split =====
    print(f"\n✂️  Splitting data (test_size={test_size})...")
    X = df_engineered.drop(columns=[target_col])
    y = df_engineered[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    print(f"  ✓ Train: {len(X_train)} samples")
    print(f"  ✓ Test:  {len(X_test)} samples")
    print(f"  ✓ Class distribution: {dict(y_train.value_counts())}")

    # ===== 4. Preprocessing =====
    X_train_scaled, X_test_scaled, scaler = advanced_preprocessing(
        X_train, X_test, y_train, scaler_type=scaler_type
    )

    # ===== 5. Feature selection =====
    X_train_selected, X_test_selected, selected_features, selector = select_best_features(
        X_train_scaled, X_test_scaled, y_train, k=n_features, method='f_classif'
    )

    # Convert to DataFrame for deep learning models
    X_train_df = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_df = pd.DataFrame(X_test_selected, columns=selected_features)

    # ===== 6. Train models =====
    all_results = {}

    if train_traditional:
        traditional_results = train_traditional_models(
            X_train_selected, y_train, X_test_selected, y_test, use_smote=use_smote
        )
        all_results.update(traditional_results)

    if train_deep_learning:
        dl_results = train_deep_learning_models(
            X_train_df, y_train, X_test_df, y_test,
            selected_features, use_smote=use_smote
        )
        all_results.update(dl_results)

    # ===== 7. Create ensembles =====
    ensemble_results = {}
    if create_ensemble and len(all_results) > 1:
        ensemble_results = create_ensembles(all_results, y_test)

    # ===== 8. Print summary =====
    summary_df = print_final_summary(all_results, ensemble_results)

    # Return comprehensive results
    return {
        'individual_results': all_results,
        'ensemble_results': ensemble_results,
        'summary_df': summary_df,
        'best_model': summary_df.iloc[0].to_dict(),
        'selected_features': selected_features,
        'scaler': scaler,
        'feature_selector': selector
    }


if __name__ == "__main__":
    print("This module should be imported. Use: from diabetes_model_trainer import train_and_test")
