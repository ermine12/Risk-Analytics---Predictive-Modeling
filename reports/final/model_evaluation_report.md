# Model Evaluation Report
============================================================

## Executive Summary

This report presents a comprehensive evaluation of predictive models for insurance risk analytics.
Models were evaluated on their ability to predict claim severity, premium optimization, and claim probability.


## 1. Claim Severity Prediction Models

**Target:** TotalClaims (for policies with claims > 0)
**Metrics:** RMSE, R², MAE

| Model | Test RMSE | Test R² | Test MAE |
|-------|-----------|---------|----------|
| Linear Regression | $7,953.81 | 0.5319 | $6,567.99 |
| Random Forest | $5,635.73 | 0.7650 | $3,394.27 |

**Best Model:** Random Forest (Lowest RMSE: $5,635.73)

## 2. Premium Optimization Models

**Target:** TotalPremium
**Metrics:** RMSE, R², MAE

| Model | Test RMSE | Test R² | Test MAE |
|-------|-----------|---------|----------|
| Linear Regression | $7,210.65 | 0.8035 | $5,034.95 |
| Random Forest | $5,482.95 | 0.8864 | $3,046.47 |

## 3. Claim Probability Models (Binary Classification)

**Target:** Binary (1 if claim occurred, 0 otherwise)
**Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## 4. Business Recommendations

Based on the model evaluation:

1. **Claim Severity Prediction**: Use the best-performing model for accurate claim amount predictions.
   This enables better reserve setting and risk assessment.

2. **Claim Probability**: Implement the probability model to identify high-risk policies.
   This supports proactive risk management and pricing adjustments.
