# Tasks 3 & 4 Implementation Summary

## Task 3: A/B Hypothesis Testing ✅

### Implementation

**File**: `src/eda/hypothesis_testing.py`

### Hypotheses Tested

1. **H₀: No risk differences across provinces**
   - Claim Frequency: Chi-squared test
   - Claim Severity: ANOVA + pairwise t-tests
   - Results saved to: `reports/interim/hypothesis_test_results.json`

2. **H₀: No risk differences between zip codes**
   - Claim Frequency: Chi-squared test (top 20 zip codes by sample size)
   - Claim Severity: ANOVA
   - Results include zipcodes tested and significance

3. **H₀: No margin differences between zip codes**
   - Margin = TotalPremium - TotalClaims
   - ANOVA test
   - Identifies highest/lowest margin zip codes

4. **H₀: No risk differences between Women and Men**
   - Claim Frequency: Chi-squared test
   - Claim Severity: t-test
   - Includes difference percentage

### Metrics Used

- **Claim Frequency**: Proportion of policies with at least one claim
- **Claim Severity**: Average claim amount (given claim occurred)
- **Margin**: TotalPremium - TotalClaims

### Usage

```bash
python src/eda/hypothesis_testing.py
```

### Output

- JSON results file with p-values, test statistics, and interpretations
- Business recommendations generated automatically
- Logged to `logs/pipeline.log`

## Task 4: Advanced Predictive Modeling ✅

### Implementation

**File**: `src/models/advanced_modeling.py`

### Models Implemented

1. **Claim Severity Prediction**
   - Target: `TotalClaims` (for policies with claims > 0)
   - Models: Linear Regression, Random Forest, XGBoost
   - Metrics: RMSE, R², MAE
   - SHAP analysis for feature importance
   - Best model saved to: `models/claim_severity_model.pkl`

2. **Premium Optimization**
   - Target: `TotalPremium`
   - Models: Linear Regression, Random Forest, XGBoost
   - Metrics: RMSE, R², MAE
   - Best model saved to: `models/premium_model.pkl`

3. **Claim Probability Prediction** (Binary Classification)
   - Target: Binary (1 if TotalClaims > 0, else 0)
   - Models: Random Forest, XGBoost
   - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
   - Best model saved to: `models/claim_probability_model.pkl`

4. **Risk-Based Premium Calculation**
   - Formula: `Premium = (Predicted Probability × Predicted Severity) × (1 + Expense Loading + Profit Margin)`
   - Expense Loading: 20%
   - Profit Margin: 15%

### Feature Engineering

- Date parsing and extraction
- Vehicle age calculation
- Loss ratio and margin calculation
- Categorical encoding (Label Encoding)
- Missing value handling

### SHAP Analysis

- Top 10 most influential features identified
- Saved to: `reports/final/claim_severity_shap_importance.csv`
- Explains how features impact predictions

### Usage

```bash
python src/models/advanced_modeling.py
```

### Output

- Trained models saved to `models/` directory
- Evaluation metrics saved to: `reports/final/model_evaluation_results.json`
- SHAP feature importance: `reports/final/claim_severity_shap_importance.csv`

## Improvements Made

### 1. Structured Logging ✅
- **File**: `src/utils/logger.py`
- Replaces print statements with structured logging
- Logs to both console and file (`logs/pipeline.log`)
- Different log levels (DEBUG, INFO, WARNING, ERROR)

### 2. Enhanced Data Cleaning ✅
- **File**: `src/data/prepare_data.py`
- Explicit cleaning steps with logging:
  1. Handle missing values
  2. Remove duplicates
  3. Handle outliers (cap instead of remove)
  4. Validate data types
  5. Handle zero/invalid values
- Detailed logging of each step

### 3. Bar Charts for Categorical Variables ✅
- **File**: `src/eda/run_eda.py`
- Automatically generates bar charts for top 5 categorical variables
- Value labels on bars
- Saved to: `reports/interim/bar_chart_*.png`

### 4. Box Plots for Outlier Detection ✅
- **File**: `src/eda/run_eda.py`
- Box plots for `TotalPremium` and `TotalClaims`
- Statistics displayed (Q1, Q3, IQR, outlier count)
- Box plots by Province for claims
- Saved to: `reports/interim/boxplot_*.png`

### 5. DVC Configuration ✅
- **File**: `init_dvc.py`
- Script to initialize DVC and configure remote
- Updated `.dvcignore`
- Pipeline stages added for hypothesis testing and advanced modeling

### 6. CI/CD Fix ✅
- **File**: `.github/workflows/ci.yml`
- Replaced `dvc validate` (doesn't exist) with `dvc dag`
- Validates pipeline structure

## DVC Pipeline Stages

1. `prepare_data` - Data cleaning and preprocessing
2. `eda` - Exploratory data analysis
3. `hypothesis_testing` - A/B hypothesis tests (NEW)
4. `train_claims_model` - Basic claims model
5. `train_premium_model` - Basic premium model
6. `advanced_modeling` - Advanced models (severity, premium, probability) (NEW)
7. `evaluate` - Model evaluation
8. `recommendations` - Low-risk group recommendations

## Running the Complete Pipeline

```bash
# Initialize DVC (first time only)
python init_dvc.py

# Add data (if not already tracked)
dvc add data/raw/insurance.csv
git add data/raw/insurance.csv.dvc .gitignore
git commit -m "task-2: add raw data tracked by dvc"
dvc push

# Run complete pipeline
dvc repro

# Or run specific stages
dvc repro hypothesis_testing
dvc repro advanced_modeling
```

## Next Steps

1. **Merge task-3 to main via PR** (as per requirements)
2. **Create task-4 branch** (if separate from task-3)
3. **Run models on actual data** to generate results
4. **Generate final reports** with business interpretations

## Files Created/Modified

### New Files
- `src/utils/logger.py` - Structured logging
- `src/eda/hypothesis_testing.py` - Hypothesis testing module
- `src/models/advanced_modeling.py` - Advanced modeling module
- `init_dvc.py` - DVC initialization script
- `TASK_3_4_SUMMARY.md` - This file

### Modified Files
- `src/data/prepare_data.py` - Enhanced cleaning with logging
- `src/eda/run_eda.py` - Added bar charts and box plots
- `dvc.yaml` - Added new pipeline stages
- `.github/workflows/ci.yml` - Fixed DVC validation
- `requirements.txt` - Added xgboost

## Business Recommendations Format

The hypothesis testing module automatically generates business recommendations in this format:

> "We reject the null hypothesis for provinces (p < 0.05). Specifically, [Province A] vs [Province B] shows a X% difference in claim severity, suggesting regional risk adjustment to premiums may be warranted."

## Model Interpretability

SHAP analysis provides:
- Top 10 most influential features
- Feature importance scores
- Explanation of how features impact predictions
- Business-ready interpretations (e.g., "For every year older a vehicle is, predicted claim amount increases by X Rand")

