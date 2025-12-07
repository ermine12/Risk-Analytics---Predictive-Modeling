"""Train model to predict total claims."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import PROCESSED_DATA_DIR, MODELS_DIR, INTERIM_REPORTS_DIR, RANDOM_SEED


def load_data() -> pd.DataFrame:
    """Load processed insurance data."""
    filepath = PROCESSED_DATA_DIR / "insurance_cleaned.csv"
    
    if not filepath.exists():
        print(f"Warning: {filepath} not found.")
        return pd.DataFrame()
    
    return pd.read_csv(filepath)


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features and target for claims prediction."""
    if df.empty:
        return None, None, None, None
    
    # Assuming 'charges' is the target (total claims)
    if 'charges' not in df.columns:
        print("Warning: 'charges' column not found")
        return None, None, None, None
    
    # Select features (exclude target)
    feature_cols = [col for col in df.columns if col != 'charges']
    
    # Encode categorical variables
    df_encoded = df.copy()
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col in feature_cols:
            df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=col, drop_first=True)
    
    # Update feature columns after encoding
    feature_cols = [col for col in df_encoded.columns if col != 'charges']
    
    X = df_encoded[feature_cols]
    y = df_encoded['charges']
    
    return X, y, feature_cols, df_encoded


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    """Train Random Forest model for claims prediction."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    model.fit(X, y)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate model performance."""
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
    print("Training claims prediction model...")
    
    # Load data
    df = load_data()
    
    if df.empty:
        print("No data available for training")
        # Create empty metrics file
        metrics = {'error': 'No data available'}
        output_path = INTERIM_REPORTS_DIR / "claims_model_metrics.json"
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        return
    
    # Prepare features
    X, y, feature_cols, df_encoded = prepare_features(df)
    
    if X is None or y is None:
        print("Failed to prepare features")
        return
    
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Train model
    model = train_model(X_train, y_train)
    print("Model trained")
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Model Metrics:")
    for key, value in metrics.items():
        print(f"  {key.upper()}: {value:.4f}")
    
    # Save model
    model_path = MODELS_DIR / "claims_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature names
    feature_path = MODELS_DIR / "claims_model_features.pkl"
    joblib.dump(feature_cols, feature_path)
    
    # Save metrics
    metrics_path = INTERIM_REPORTS_DIR / "claims_model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

