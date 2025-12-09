"""Advanced Predictive Modeling for Risk-Based Pricing.

Models:
1. Claim Severity Prediction (for policies with claims > 0)
2. Premium Optimization
3. Claim Probability Prediction (binary classification)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import PROCESSED_DATA_DIR, MODELS_DIR, FINAL_REPORTS_DIR, RANDOM_SEED
from utils.logger import logger

# Try importing optional dependencies
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")


def normalize_data_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to handle different data formats."""
    df = df.copy()
    
    # Map standard insurance dataset columns
    column_mapping = {
        'charges': 'TotalClaims',
        'Charges': 'TotalClaims',
        'region': 'Province',
        'Region': 'Province',
        'sex': 'Gender',
        'Sex': 'Gender'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Estimate premium if not available
    if 'TotalClaims' in df.columns and 'TotalPremium' not in df.columns:
        # Estimate premium as claims * 1.2 (20% margin)
        df['TotalPremium'] = df['TotalClaims'] * 1.2
        logger.info("Estimated TotalPremium from TotalClaims")
    
    return df


def prepare_modeling_data(df: pd.DataFrame) -> tuple:
    """Prepare data for modeling with feature engineering and encoding."""
    logger.info("Preparing data for modeling...")
    df = normalize_data_for_modeling(df)
    
    # Feature engineering
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
        df['TransactionYear'] = df['TransactionMonth'].dt.year
        df['TransactionMonthNum'] = df['TransactionMonth'].dt.month
    
    # Create derived features
    if 'age' in df.columns:
        # Age bins
        df['age_bin'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], 
                               labels=['18-30', '31-40', '41-50', '51-60', '60+'])
    
    if 'bmi' in df.columns:
        # BMI categories
        df['bmi_category'] = pd.cut(df['bmi'], 
                                    bins=[0, 18.5, 25, 30, np.inf],
                                    labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    if 'TotalClaims' in df.columns and 'TotalPremium' in df.columns:
        df['loss_ratio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, np.nan)
        df['margin'] = df['TotalPremium'] - df['TotalClaims']
    
    if 'RegistrationYear' in df.columns:
        current_year = pd.Timestamp.now().year
        df['VehicleAge'] = current_year - df['RegistrationYear']
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            logger.info(f"Filled {df[col].isnull().sum()} missing values in {col} with median")
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        if col not in ['TotalClaims', 'TotalPremium']:  # Don't encode targets
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str).fillna('Unknown'))
            label_encoders[col] = le
    
    # Select features (exclude target columns and original categorical)
    exclude_cols = ['TotalClaims', 'TotalPremium', 'loss_ratio', 'margin', 'PolicyID', 
                    'charges', 'Charges']  # Exclude original charges column to avoid leakage
    exclude_cols += list(categorical_cols)  # Use encoded versions
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remove any columns that are duplicates or derived from targets
    feature_cols = [col for col in feature_cols if not col.startswith('Total')]
    
    logger.info(f"Prepared {len(feature_cols)} features for modeling")
    logger.info(f"Features: {feature_cols[:10]}...")  # Show first 10
    
    return df, feature_cols, label_encoders


def train_claim_severity_model(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Train model to predict TotalClaims for policies with claims > 0.
    Target: TotalClaims (subset where claims > 0)
    Metrics: RMSE, R-squared
    """
    logger.info("Training Claim Severity Prediction Model...")
    
    # For standard dataset, use median threshold to create "claims" subset
    if 'TotalClaims' in df.columns:
        threshold = df['TotalClaims'].median()
        df_with_claims = df[df['TotalClaims'] > threshold].copy()
        logger.info(f"Using threshold of ${threshold:.2f} to define 'claims' subset")
    else:
        logger.warning("TotalClaims column not found")
        return {}
    
    if len(df_with_claims) == 0:
        logger.warning("No policies with claims found")
        return {}
    
    logger.info(f"Training on {len(df_with_claims)} policies with claims")
    
    X = df_with_claims[feature_cols].fillna(0)
    y = df_with_claims['TotalClaims']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1),
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED)
    else:
        logger.warning("XGBoost not available, skipping XGBoost model")
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        metrics = {
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            'train_r2': float(r2_score(y_train, y_pred_train)),
            'test_r2': float(r2_score(y_test, y_pred_test)),
            'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
            'test_mae': float(mean_absolute_error(y_test, y_pred_test))
        }
        
        results[model_name] = {
            'model': model,
            'metrics': metrics
        }
        
        logger.info(f"{model_name} - Test RMSE: {metrics['test_rmse']:.2f}, R²: {metrics['test_r2']:.4f}")
    
    # Select best model
    best_model_name = min(results.keys(), key=lambda k: results[k]['metrics']['test_rmse'])
    best_model = results[best_model_name]['model']
    
    # Save best model
    model_path = MODELS_DIR / "claim_severity_model.pkl"
    joblib.dump(best_model, model_path)
    logger.info(f"Best model ({best_model_name}) saved to {model_path}")
    
    # SHAP analysis for best model
    if SHAP_AVAILABLE and hasattr(best_model, 'feature_importances_'):
        logger.info("Calculating SHAP values...")
        try:
            # Use TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test.iloc[:min(100, len(X_test))])  # Sample for speed
            
            # Feature importance from SHAP
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'shap_importance': np.abs(shap_values).mean(axis=0),
                'impact_direction': ['positive' if shap_values[:, i].mean() > 0 else 'negative' 
                                     for i in range(len(X_test.columns))]
            }).sort_values('shap_importance', ascending=False)
            
            shap_path = FINAL_REPORTS_DIR / "claim_severity_shap_importance.csv"
            feature_importance.to_csv(shap_path, index=False)
            logger.info(f"SHAP feature importance saved to {shap_path}")
            
            results['shap_importance'] = feature_importance.head(10).to_dict('records')
            
            # Generate business interpretation
            top_feature = feature_importance.iloc[0]
            logger.info(f"Top feature: {top_feature['feature']} (importance: {top_feature['shap_importance']:.4f})")
            
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
            # Fallback to feature importances
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_test.columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                results['feature_importance'] = feature_importance.head(10).to_dict('records')
    else:
        # Use built-in feature importances
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            results['feature_importance'] = feature_importance.head(10).to_dict('records')
            logger.info("Using model feature importances (SHAP not available)")
    
    return {
        'models': results,
        'best_model': best_model_name,
        'feature_importance': results.get('shap_importance', [])
    }


def train_premium_model(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Train model to predict optimal premium.
    Target: TotalPremium (or calculated premium)
    """
    logger.info("Training Premium Optimization Model...")
    
    if 'TotalPremium' not in df.columns:
        logger.warning("TotalPremium column not found")
        return {}
    
    X = df[feature_cols].fillna(0)
    y = df['TotalPremium']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1),
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED)
    else:
        logger.warning("XGBoost not available, skipping XGBoost model")
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        y_pred_test = model.predict(X_test)
        
        metrics = {
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            'test_r2': float(r2_score(y_test, y_pred_test)),
            'test_mae': float(mean_absolute_error(y_test, y_pred_test))
        }
        
        results[model_name] = {
            'model': model,
            'metrics': metrics
        }
        
        logger.info(f"{model_name} - Test RMSE: {metrics['test_rmse']:.2f}, R²: {metrics['test_r2']:.4f}")
    
    best_model_name = min(results.keys(), key=lambda k: results[k]['metrics']['test_rmse'])
    best_model = results[best_model_name]['model']
    
    model_path = MODELS_DIR / "premium_model.pkl"
    joblib.dump(best_model, model_path)
    logger.info(f"Best premium model ({best_model_name}) saved to {model_path}")
    
    return {
        'models': results,
        'best_model': best_model_name
    }


def train_claim_probability_model(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Train binary classification model to predict probability of claim occurring.
    Target: Binary (1 if TotalClaims > 0, else 0)
    """
    logger.info("Training Claim Probability Model (Binary Classification)...")
    
    # Create binary target
    df['has_claim'] = (df['TotalClaims'] > 0).astype(int)
    
    X = df[feature_cols].fillna(0)
    y = df['has_claim']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    from sklearn.ensemble import RandomForestClassifier
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1),
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED)
    else:
        logger.warning("XGBoost not available, skipping XGBoost model")
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        # Get probability predictions
        proba = model.predict_proba(X_test)
        if proba.shape[1] == 2:
            y_pred_proba = proba[:, 1]  # Probability of positive class
        else:
            # Only one class predicted, use the single column
            y_pred_proba = proba[:, 0]
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba)) if len(np.unique(y_test)) > 1 else 0.0
        }
        
        results[model_name] = {
            'model': model,
            'metrics': metrics
        }
        
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['f1_score'])
    best_model = results[best_model_name]['model']
    
    model_path = MODELS_DIR / "claim_probability_model.pkl"
    joblib.dump(best_model, model_path)
    logger.info(f"Best probability model ({best_model_name}) saved to {model_path}")
    
    return {
        'models': results,
        'best_model': best_model_name
    }


def calculate_risk_based_premium(df: pd.DataFrame, prob_model, severity_model, feature_cols: list) -> pd.Series:
    """
    Calculate Risk-Based Premium using:
    Premium = (Predicted Probability of Claim * Predicted Claim Severity) + Expense Loading + Profit Margin
    """
    logger.info("Calculating Risk-Based Premium...")
    
    X = df[feature_cols].fillna(0)
    
    # Predict probability
    if hasattr(prob_model, 'predict_proba'):
        prob_claim = prob_model.predict_proba(X)[:, 1]  # Probability of claim
    else:
        prob_claim = prob_model.predict(X)
    prob_claim = np.clip(prob_claim, 0, 1)  # Ensure between 0 and 1
    
    # Predict severity (for all policies, but will be adjusted by probability)
    severity = severity_model.predict(X)
    severity = np.clip(severity, 0, None)  # Ensure non-negative
    
    # Expense loading (20%) and profit margin (15%)
    expense_loading = 0.20
    profit_margin = 0.15
    
    # Risk-based premium calculation
    risk_based_premium = (prob_claim * severity) * (1 + expense_loading + profit_margin)
    
    return pd.Series(risk_based_premium, index=df.index)


def main():
    """Main modeling pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Advanced Predictive Modeling Pipeline")
    logger.info("=" * 60)
    
    # Load processed data
    data_path = PROCESSED_DATA_DIR / "insurance_cleaned.csv"
    if not data_path.exists():
        logger.error(f"Processed data not found at {data_path}")
        raise FileNotFoundError(f"Processed data not found at {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Data loaded. Shape: {df.shape}")
    
    # Prepare data
    df, feature_cols, label_encoders = prepare_modeling_data(df)
    logger.info(f"Features prepared: {len(feature_cols)} features")
    
    # Train models
    severity_results = train_claim_severity_model(df, feature_cols)
    premium_results = train_premium_model(df, feature_cols)
    probability_results = train_claim_probability_model(df, feature_cols)
    
    # Combine results
    all_results = {
        'claim_severity': severity_results,
        'premium_optimization': premium_results,
        'claim_probability': probability_results
    }
    
    # Save results
    results_path = FINAL_REPORTS_DIR / "model_evaluation_results.json"
    with open(results_path, 'w') as f:
        # Convert models to None for JSON serialization
        json_results = {}
        for key, value in all_results.items():
            if isinstance(value, dict):
                json_results[key] = {k: v for k, v in value.items() if k != 'model'}
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2, default=str)
    
    logger.info(f"Model evaluation results saved to {results_path}")
    logger.info("=" * 60)
    
    return all_results


if __name__ == "__main__":
    main()

