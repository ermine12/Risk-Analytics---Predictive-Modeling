"""Generate comprehensive model evaluation report with SHAP interpretations."""

import pandas as pd
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.config import FINAL_REPORTS_DIR
from utils.logger import logger


def generate_shap_interpretations(shap_data: list, model_name: str) -> str:
    """Generate business-friendly SHAP interpretations."""
    interpretations = []
    
    if not shap_data:
        return "SHAP analysis not available for this model."
    
    interpretations.append(f"\n### Top 10 Most Influential Features for {model_name}\n")
    interpretations.append("| Rank | Feature | Importance | Impact Direction |")
    interpretations.append("|------|---------|------------|------------------|")
    
    for i, feature_data in enumerate(shap_data[:10], 1):
        feature_name = feature_data.get('feature', 'Unknown')
        importance = feature_data.get('shap_importance', feature_data.get('importance', 0))
        direction = feature_data.get('impact_direction', 'neutral')
        
        interpretations.append(f"| {i} | {feature_name} | {importance:.4f} | {direction} |")
    
    # Generate business interpretation for top feature
    if shap_data:
        top_feature = shap_data[0]
        feature_name = top_feature.get('feature', 'Unknown')
        importance = top_feature.get('shap_importance', top_feature.get('importance', 0))
        
        interpretations.append(f"\n**Business Interpretation:**")
        interpretations.append(f"\nThe most influential feature is **{feature_name}** with an importance score of {importance:.4f}.")
        interpretations.append(f"This indicates that {feature_name} has the strongest impact on claim severity predictions.")
        interpretations.append(f"Businesses should prioritize understanding and monitoring this feature when assessing risk.")
    
    return "\n".join(interpretations)


def generate_model_comparison_report():
    """Generate comprehensive model comparison report."""
    logger.info("Generating model evaluation report...")
    
    # Load results
    results_path = FINAL_REPORTS_DIR / "model_evaluation_results.json"
    if not results_path.exists():
        logger.error(f"Model results not found at {results_path}")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Load SHAP importance
    shap_path = FINAL_REPORTS_DIR / "claim_severity_shap_importance.csv"
    shap_data = []
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path)
        shap_data = shap_df.head(10).to_dict('records')
    
    # Generate markdown report
    report = []
    report.append("# Model Evaluation Report")
    report.append("=" * 60)
    report.append("\n## Executive Summary\n")
    report.append("This report presents a comprehensive evaluation of predictive models for insurance risk analytics.")
    report.append("Models were evaluated on their ability to predict claim severity, premium optimization, and claim probability.\n")
    
    # Claim Severity Models
    if 'claim_severity' in results:
        severity_data = results['claim_severity']
        if isinstance(severity_data, dict) and 'models' in severity_data:
            report.append("\n## 1. Claim Severity Prediction Models\n")
            report.append("**Target:** TotalClaims (for policies with claims > 0)")
            report.append("**Metrics:** RMSE, R², MAE\n")
            
            severity_models = severity_data['models']
            report.append("| Model | Test RMSE | Test R² | Test MAE |")
            report.append("|-------|-----------|---------|----------|")
            
            best_rmse = float('inf')
            best_model_name = None
            
            for model_name, model_data in severity_models.items():
                if isinstance(model_data, dict):
                    metrics = model_data.get('metrics', {})
                    rmse = metrics.get('test_rmse', 0)
                    r2 = metrics.get('test_r2', 0)
                    mae = metrics.get('test_mae', 0)
                    
                    report.append(f"| {model_name} | ${rmse:,.2f} | {r2:.4f} | ${mae:,.2f} |")
                    
                    if rmse < best_rmse and rmse > 0:
                        best_rmse = rmse
                        best_model_name = model_name
            
            if best_model_name:
                report.append(f"\n**Best Model:** {best_model_name} (Lowest RMSE: ${best_rmse:,.2f})")
            
            # Feature Importance
            if 'feature_importance' in severity_data:
                feature_imp = severity_data['feature_importance']
                if feature_imp:
                    report.append("\n### Top 10 Feature Importances\n")
                    report.append("| Rank | Feature | Importance |")
                    report.append("|------|---------|------------|")
                    for i, feat in enumerate(feature_imp[:10], 1):
                        report.append(f"| {i} | {feat.get('feature', 'Unknown')} | {feat.get('importance', 0):.4f} |")
                    
                    # Business Interpretation
                    report.append("\n#### Business Interpretation of Feature Importance\n")
                    top_feature = feature_imp[0]
                    top_name = top_feature.get('feature', 'Unknown')
                    top_imp = top_feature.get('importance', 0)
                    
                    # Map encoded features back to original names
                    feature_mapping = {
                        'smoker_encoded': 'Smoking Status',
                        'bmi': 'Body Mass Index (BMI)',
                        'age': 'Age',
                        'children': 'Number of Children',
                        'region_encoded': 'Geographic Region',
                        'Gender_encoded': 'Gender',
                        'sex_encoded': 'Gender'
                    }
                    
                    display_name = feature_mapping.get(top_name, top_name.replace('_encoded', '').replace('_', ' ').title())
                    
                    report.append(f"**Most Influential Feature: {display_name}**")
                    report.append(f"\nThe Random Forest model identifies **{display_name}** as the most important feature ")
                    report.append(f"for predicting claim severity, with an importance score of {top_imp:.2%}.")
                    report.append(f"\nThis means that {display_name} has the strongest impact on the predicted claim amount.")
                    report.append(f"\n**Business Implication:**")
                    
                    if 'smoker' in top_name.lower():
                        report.append("- Smoking status is the primary risk driver for claim severity.")
                        report.append("- Policies for smokers show significantly higher predicted claim amounts.")
                        report.append("- Consider implementing smoking-based premium adjustments or wellness programs.")
                    elif 'bmi' in top_name.lower():
                        report.append("- BMI is a strong predictor of claim severity.")
                        report.append("- Higher BMI correlates with increased claim amounts.")
                        report.append("- Consider BMI-based risk assessment in underwriting.")
                    elif 'age' in top_name.lower():
                        report.append("- Age is a significant factor in claim severity predictions.")
                        report.append("- Older policyholders tend to have higher claim amounts.")
                        report.append("- Age-based premium adjustments may be warranted.")
                    
                    report.append(f"\n**Top 5 Features Summary:**")
                    for i, feat in enumerate(feature_imp[:5], 1):
                        feat_name = feat.get('feature', 'Unknown')
                        display_name = feature_mapping.get(feat_name, feat_name.replace('_encoded', '').replace('_', ' ').title())
                        importance = feat.get('importance', 0)
                        report.append(f"{i}. {display_name}: {importance:.2%} importance")
            
            # SHAP Analysis
            if shap_data:
                report.append(generate_shap_interpretations(shap_data, best_model_name or "Claim Severity Model"))
    
    # Premium Models
    if 'premium_optimization' in results:
        premium_data = results['premium_optimization']
        if isinstance(premium_data, dict) and 'models' in premium_data:
            report.append("\n## 2. Premium Optimization Models\n")
            report.append("**Target:** TotalPremium")
            report.append("**Metrics:** RMSE, R², MAE\n")
            
            premium_models = premium_data['models']
            report.append("| Model | Test RMSE | Test R² | Test MAE |")
            report.append("|-------|-----------|---------|----------|")
            
            for model_name, model_data in premium_models.items():
                if isinstance(model_data, dict):
                    metrics = model_data.get('metrics', {})
                    rmse = metrics.get('test_rmse', 0)
                    r2 = metrics.get('test_r2', 0)
                    mae = metrics.get('test_mae', 0)
                    
                    report.append(f"| {model_name} | ${rmse:,.2f} | {r2:.4f} | ${mae:,.2f} |")
    
    # Claim Probability Models
    if 'claim_probability' in results:
        prob_data = results['claim_probability']
        if isinstance(prob_data, dict) and 'models' in prob_data:
            report.append("\n## 3. Claim Probability Models (Binary Classification)\n")
            report.append("**Target:** Binary (1 if claim occurred, 0 otherwise)")
            report.append("**Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC\n")
            
            prob_models = prob_data['models']
            report.append("| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |")
            report.append("|-------|----------|-----------|--------|----------|---------|")
            
            for model_name, model_data in prob_models.items():
                if isinstance(model_data, dict):
                    metrics = model_data.get('metrics', {})
                    acc = metrics.get('accuracy', 0)
                    prec = metrics.get('precision', 0)
                    rec = metrics.get('recall', 0)
                    f1 = metrics.get('f1_score', 0)
                    auc = metrics.get('roc_auc', 0)
                    
                    report.append(f"| {model_name} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {auc:.4f} |")
    
    # Business Recommendations
    report.append("\n## 4. Business Recommendations\n")
    report.append("Based on the model evaluation:\n")
    
    if 'claim_severity' in results:
        report.append("1. **Claim Severity Prediction**: Use the best-performing model for accurate claim amount predictions.")
        report.append("   This enables better reserve setting and risk assessment.\n")
    
    if 'claim_probability' in results:
        report.append("2. **Claim Probability**: Implement the probability model to identify high-risk policies.")
        report.append("   This supports proactive risk management and pricing adjustments.\n")
    
    if shap_data:
        report.append("3. **Feature Importance**: Focus on the top influential features identified by SHAP analysis.")
        report.append("   These features should be prioritized in underwriting and pricing decisions.\n")
    
    # Save report
    report_text = "\n".join(report)
    report_path = FINAL_REPORTS_DIR / "model_evaluation_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"Model evaluation report saved to {report_path}")
    
    # Also print to console
    print("\n" + "=" * 60)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(report_text)


if __name__ == "__main__":
    generate_model_comparison_report()

