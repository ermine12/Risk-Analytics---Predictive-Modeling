"""Evaluate and compare models."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FINAL_REPORTS_DIR, INTERIM_REPORTS_DIR,
    PLOT_STYLE, FIGURE_SIZE, RANDOM_SEED
)
from utils.logger import logger


def load_models():
    """
    Load trained claims and premium models from disk.
    
    Returns:
        Tuple of (claims_model, premium_model). Either can be None if file not found.
        
    Assumptions:
        - Model files exist at expected locations
        - Models were saved using joblib
    """
    claims_model_path = MODELS_DIR / "claims_model.pkl"
    premium_model_path = MODELS_DIR / "premium_model.pkl"
    
    claims_model = None
    premium_model = None
    
    if claims_model_path.exists():
        claims_model = joblib.load(claims_model_path)
        logger.info(f"Loaded claims model from {claims_model_path}")
    
    if premium_model_path.exists():
        premium_model = joblib.load(premium_model_path)
        logger.info(f"Loaded premium model from {premium_model_path}")
    
    return claims_model, premium_model


def load_data() -> pd.DataFrame:
    """
    Load processed insurance data for model evaluation.
    
    Returns:
        DataFrame with cleaned insurance data. Empty DataFrame if file not found.
        
    Assumptions:
        - Processed data file exists at expected location
    """
    filepath = PROCESSED_DATA_DIR / "insurance_cleaned.csv"
    
    if not filepath.exists():
        logger.warning(f"{filepath} not found.")
        return pd.DataFrame()
    
    return pd.read_csv(filepath)


def create_evaluation_plots(claims_model, premium_model, df: pd.DataFrame, output_dir: Path):
    """
    Create evaluation visualizations for trained models.
    
    Generates prediction vs actual plots and residual plots for both models.
    
    Args:
        claims_model: Trained claims prediction model (can be None).
        premium_model: Trained premium recommendation model (can be None).
        df: DataFrame with test data.
        output_dir: Path object for saving plot files.
        
    Assumptions:
        - Models have predict() method
        - Output directory exists and is writable
        - Data has been encoded same way as training data
    """
    if df.empty or (claims_model is None and premium_model is None):
        logger.warning("No data or models available for plotting")
        return
    
    plt.style.use('seaborn-v0_8')
    
    # Prepare features (same as training)
    feature_cols = [col for col in df.columns if col != 'charges']
    df_encoded = df.copy()
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col in feature_cols:
            df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=col, drop_first=True)
    
    feature_cols_encoded = [col for col in df_encoded.columns if col != 'charges']
    X = df_encoded[feature_cols_encoded]
    
    # Plot 1: Actual vs Predicted for Claims Model
    if claims_model is not None and 'charges' in df.columns:
        y_actual = df['charges']
        y_pred = claims_model.predict(X)
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        ax.scatter(y_actual, y_pred, alpha=0.5)
        ax.plot([y_actual.min(), y_actual.max()], 
               [y_actual.min(), y_actual.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Claims', fontsize=12)
        ax.set_ylabel('Predicted Claims', fontsize=12)
        ax.set_title('Claims Model: Actual vs Predicted', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'claims_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Feature Importance (if available)
    if claims_model is not None and hasattr(claims_model, 'feature_importances_'):
        importances = claims_model.feature_importances_
        feature_names = feature_cols_encoded[:len(importances)]
        
        # Get top 10 features
        indices = np.argsort(importances)[::-1][:10]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Top 10 Feature Importances - Claims Model', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Residuals Plot
    if claims_model is not None and 'charges' in df.columns:
        y_actual = df['charges']
        y_pred = claims_model.predict(X)
        residuals = y_actual - y_pred
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Claims', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title('Residuals Plot - Claims Model', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'residuals_plot.png', dpi=300, bbox_inches='tight')
        plt.close()


def generate_comparison_report(claims_model, premium_model, output_dir: Path):
    """
    Generate comprehensive model comparison report.
    
    Compiles metrics from both models and creates HTML report with
    performance comparisons and visualizations.
    
    Args:
        claims_model: Trained claims prediction model (can be None).
        premium_model: Trained premium recommendation model (can be None).
        output_dir: Path object for saving report files.
        
    Returns:
        None. Saves JSON and HTML reports to output_dir.
        
    Assumptions:
        - Model metrics files exist if models are trained
        - Output directory exists and is writable
    """
    comparison = {
        'models_trained': {
            'claims_model': claims_model is not None,
            'premium_model': premium_model is not None
        },
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Load metrics if available
    claims_metrics_path = INTERIM_REPORTS_DIR / "claims_model_metrics.json"
    premium_metrics_path = INTERIM_REPORTS_DIR / "premium_model_metrics.json"
    
    if claims_metrics_path.exists():
        with open(claims_metrics_path, 'r') as f:
            comparison['claims_model_metrics'] = json.load(f)
    
    if premium_metrics_path.exists():
        with open(premium_metrics_path, 'r') as f:
            comparison['premium_model_metrics'] = json.load(f)
    
    # Save comparison
    comparison_path = output_dir / "model_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Model comparison saved to {comparison_path}")
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .metric {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
        </style>
    </head>
    <body>
        <h1>Model Evaluation Report</h1>
        
        <h2>Models Status</h2>
        <div class="metric">
            <p><strong>Claims Model:</strong> {'Trained' if claims_model is not None else 'Not Available'}</p>
            <p><strong>Premium Model:</strong> {'Trained' if premium_model is not None else 'Not Available'}</p>
        </div>
    """
    
    if 'claims_model_metrics' in comparison:
        html_content += """
        <h2>Claims Model Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
        """
        for metric, value in comparison['claims_model_metrics'].items():
            if isinstance(value, (int, float)):
                html_content += f"<tr><td>{metric.upper()}</td><td>{value:.4f}</td></tr>"
        html_content += "</table>"
    
    if 'premium_model_metrics' in comparison:
        html_content += """
        <h2>Premium Model Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
        """
        for metric, value in comparison['premium_model_metrics'].items():
            if isinstance(value, (int, float)):
                html_content += f"<tr><td>{metric.upper()}</td><td>{value:.4f}</td></tr>"
        html_content += "</table>"
    
    html_content += """
    </body>
    </html>
    """
    
    report_path = output_dir / "evaluation_report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Evaluation report saved to {report_path}")


def main():
    """Main evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("Evaluating models...")
    logger.info("=" * 60)
    
    # Load models
    claims_model, premium_model = load_models()
    
    # Load data
    df = load_data()
    
    # Create evaluation plots
    create_evaluation_plots(claims_model, premium_model, df, FINAL_REPORTS_DIR)
    logger.info("Evaluation plots created")
    
    # Generate comparison report
    generate_comparison_report(claims_model, premium_model, FINAL_REPORTS_DIR)
    logger.info("Model evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

