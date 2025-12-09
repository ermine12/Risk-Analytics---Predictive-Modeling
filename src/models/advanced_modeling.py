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
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
import shap

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import PROCESSED_DATA_DIR, MODELS_DIR, FINAL_REPORTS_DIR, RANDOM_SEED
from utils.logger import logger


def prepare_modeling_data(df: pd.DataFrame) -> tuple:
    """Prepare data for modeling with feature engineering and encoding."""
    logger.info("Preparing data for modeling...")
    df = df.copy()
    
    # Feature engineering
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        df['TransactionYear'] = df['TransactionMonth'].dt.year
        df['TransactionMonthNum'] = df['TransactionMonth'].dt.month
    
    if 'TotalClaims' in df.columns and 'TotalPremium' in df.columns:
        df['loss_ratio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, np.nan)
        df['margin'] = df['TotalPremium'] - df['TotalClaims']
    
    if 'RegistrationYear' in df.columns and 'TransactionMonth' in df.columns:
        current_year = df['TransactionMonth'].dt.year if 'TransactionMonth' in df.columns else pd.Timestamp.now().year
        if isinstance(current_year, pd.Series):
            df['VehicleAge'] = current_year - df['RegistrationYear']
        else:
            df['VehicleAge'] = current_year - df['RegistrationYear']
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Select features (exclude target columns)
    exclude_cols = ['TotalClaims', 'TotalPremium', 'loss_ratio', 'margin', 'PolicyID']
    exclude_cols += list(categorical_cols)  # Use encoded versions
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    return df, feature_cols, label_encoders


def train_claim_severity_model(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Train model to predict TotalClaims for policies with claims > 0.
    Target: TotalClaims (subset where claims > 0)
    Metrics: RMSE, R-squared
    """
    logger.info("Training Claim Severity Prediction Model...")
    
    # Filter to policies with claims
    df_with_claims = df[df['TotalClaims'] > 0].copy()
    
    if len(df_with_claims) == 0:
        logger.warning("No policies with claims found")
        return {}
    
    X = df_with_claims[feature_cols].fillna(0)
    y = df_with_claims['TotalClaims']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED)
    }
    
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
    logger.info("Calculating SHAP values...")
    try:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test.iloc[:100])  # Sample for speed
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        shap_path = FINAL_REPORTS_DIR / "claim_severity_shap_importance.csv"
        feature_importance.to_csv(shap_path, index=False)
        logger.info(f"SHAP feature importance saved to {shap_path}")
        
        results['shap_importance'] = feature_importance.head(10).to_dict('records')
        
    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}")
    
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
        'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED)
    }
    
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
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED)
    }
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict(X_test)
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

