"""Train model to recommend optimal premiums."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import PROCESSED_DATA_DIR, MODELS_DIR, INTERIM_REPORTS_DIR, RANDOM_SEED
from utils.logger import logger


def load_data() -> pd.DataFrame:
    """
    Load processed insurance data for premium model training.
    
    Returns:
        DataFrame with cleaned insurance data. Empty DataFrame if file not found.
        
    Assumptions:
        - Processed data file exists at expected location
    """
    filepath = PROCESSED_DATA_DIR / "insurance_cleaned.csv"
    
    if not filepath.exists():
        logger.warning(f"{filepath} not found.")
        return pd.DataFrame()
    
    logger.info(f"Loading data from {filepath}")
    return pd.read_csv(filepath)


def calculate_optimal_premium(df: pd.DataFrame) -> pd.Series:
    """
    Calculate optimal premium based on risk factors and target loss ratio.
    
    Uses target loss ratio of 0.7 (70%) to determine premium that covers
    expected claims plus margin. Blends claims-based calculation with
    historical premium if available.
    
    Args:
        df: DataFrame with TotalClaims and optionally TotalPremium columns.
        
    Returns:
        Series of calculated optimal premiums. Empty Series if insufficient data.
        
    Assumptions:
        - TotalClaims or TotalPremium column exists
        - Target loss ratio of 0.7 is appropriate
        - Minimum premium is 10% above claims
    """
    # Premium should cover expected claims plus margin
    # Target loss ratio of 0.7 (70%) means we want premium to be claims / 0.7
    target_loss_ratio = 0.7
    
    if 'TotalClaims' in df.columns:
        # Base premium calculation: claims / target_loss_ratio
        optimal_premium = df['TotalClaims'] / target_loss_ratio
        
        # If we have historical premium, use it as a reference
        if 'TotalPremium' in df.columns:
            # Blend: 70% based on claims, 30% based on current premium
            optimal_premium = 0.7 * optimal_premium + 0.3 * df['TotalPremium']
        
        # Ensure minimum premium
        optimal_premium = optimal_premium.clip(lower=df['TotalClaims'] * 1.1)  # At least 10% above claims
        
        return optimal_premium
    elif 'TotalPremium' in df.columns:
        # Fallback: use current premium as base
        return df['TotalPremium'] * 1.1  # 10% increase
    else:
        return pd.Series()


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features and target for premium recommendation model.
    
    Calculates optimal premium as target and encodes categorical variables.
    
    Args:
        df: DataFrame with insurance data.
        
    Returns:
        Tuple of (X, y, feature_cols, df_encoded):
        - X: Feature matrix
        - y: Target vector (optimal premium)
        - feature_cols: List of feature column names
        - df_encoded: DataFrame with encoded categorical variables
        
    Assumptions:
        - Optimal premium can be calculated from available columns
        - Categorical variables can be one-hot encoded
    """
    if df.empty:
        return None, None, None, None
    
    # Calculate optimal premium as target
    optimal_premium = calculate_optimal_premium(df)
    
    if optimal_premium.empty:
        logger.warning("Could not calculate optimal premium")
        return None, None, None, None
    
    # Select features (exclude target-related columns)
    exclude_cols = ['TotalPremium', 'TotalClaims', 'loss_ratio', 'PolicyID', 'premium']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Encode categorical variables
    df_encoded = df.copy()
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col in feature_cols:
            df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=col, drop_first=True)
    
    # Update feature columns after encoding
    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
    
    X = df_encoded[feature_cols]
    y = optimal_premium
    
    return X, y, feature_cols, df_encoded


def train_model(X: pd.DataFrame, y: pd.Series) -> GradientBoostingRegressor:
    """
    Train Gradient Boosting regressor for premium recommendation.
    
    Args:
        X: Feature matrix (training data).
        y: Target vector (optimal premium).
        
    Returns:
        Trained GradientBoostingRegressor model.
        
    Assumptions:
        - X and y have matching indices
        - No missing values in X or y
    """
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        max_depth=5
    )
    
    model.fit(X, y)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate model performance using regression metrics.
    
    Calculates RMSE, MAE, R², and Mean Absolute Percentage Error.
    
    Args:
        model: Trained model with predict() method.
        X_test: Test feature matrix.
        y_test: Test target vector.
        
    Returns:
        Dictionary of metric_name -> metric_value.
        
    Assumptions:
        - Model has been trained
        - X_test and y_test have matching indices
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'mse': float(mean_squared_error(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'r2': float(r2_score(y_test, y_pred))
    }
    
    return metrics


def main():
    """Main training pipeline."""
    logger.info("Training premium recommendation model...")
    
    # Load data
    df = load_data()
    
    if df.empty:
        logger.warning("No data available for training")
        # Create empty metrics file
        metrics = {'error': 'No data available'}
        output_path = INTERIM_REPORTS_DIR / "premium_model_metrics.json"
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        return
    
    # Prepare features
    X, y, feature_cols, df_encoded = prepare_features(df)
    
    if X is None or y is None:
        logger.error("Failed to prepare features")
        return
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Samples: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Train model
    model = train_model(X_train, y_train)
    logger.info("Model trained")
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    logger.info("Model Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key.upper()}: {value:.4f}")
    
    # Save model
    model_path = MODELS_DIR / "premium_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save feature names
    feature_path = MODELS_DIR / "premium_model_features.pkl"
    joblib.dump(feature_cols, feature_path)
    
    # Save metrics
    metrics_path = INTERIM_REPORTS_DIR / "premium_model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

