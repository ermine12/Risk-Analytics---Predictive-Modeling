# Required Files Guide

## 📁 Data Files Needed

### Primary Data File (Required)

**Location**: `data/raw/insurance.csv`

**Expected Columns** (based on code implementation):

#### For Insurance Risk Analytics (Full Dataset):
```
- TransactionMonth        # Date of transaction
- TotalClaims             # Total claims amount (numeric)
- TotalPremium            # Total premium amount (numeric)
- Province                # Province name (categorical)
- PostalCode              # Postal/ZIP code (categorical)
- VehicleType             # Type of vehicle (categorical)
- RegistrationYear        # Year vehicle was registered (numeric)
- CustomValueEstimate      # Estimated vehicle value (numeric)
- VehicleIntroDate         # Date vehicle was introduced (date)
- Gender                  # Gender (categorical: Male/Female)
- Make                    # Vehicle make/brand (categorical)
- PolicyID                # Unique policy identifier (categorical/numeric)
```

#### Alternative: If you have the standard insurance dataset:
```
- age                     # Age of policyholder
- sex                     # Gender (male/female)
- bmi                     # Body mass index
- children                # Number of children
- smoker                  # Smoking status (yes/no)
- region                  # Geographic region
- charges                 # Insurance charges (can be treated as TotalClaims)
```

**Note**: The code handles both formats. If you have the standard dataset, it will work but some features (like Province, PostalCode) won't be available.

### Sample Data Structure

If you don't have data yet, the code will create an empty structure. You can:

1. **Download a sample dataset**:
   - Kaggle: "Medical Cost Personal Datasets Insurance"
   - Or use your own insurance data

2. **Format your data**:
   - Save as CSV
   - Place in `data/raw/insurance.csv`
   - Ensure column names match (case-sensitive)

## 📋 Configuration Files (Auto-Generated)

These will be created automatically:

- `.dvc/` - DVC metadata (created by `dvc init`)
- `.dvc/config` - DVC configuration (created by `init_dvc.py`)
- `logs/` - Log files directory (created automatically)
- `models/` - Trained models (created when models train)
- `reports/` - Generated reports (created when pipeline runs)

## 🔧 Setup Files (Already Created)

These files are already in the repository:

✅ `requirements.txt` - Python dependencies
✅ `dvc.yaml` - DVC pipeline configuration
✅ `params.yaml` - Model parameters
✅ `.gitignore` - Git ignore rules
✅ `.github/workflows/ci.yml` - CI/CD workflow
✅ `src/` - All source code modules
✅ `notebooks/` - Jupyter notebooks

## 🚀 Quick Start Checklist

### Step 1: Install Dependencies
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### Step 2: Add Your Data File
```bash
# Place your CSV file here:
data/raw/insurance.csv

# If you don't have data yet, the code will still run but with warnings
```

### Step 3: Initialize DVC (Optional but Recommended)
```bash
# Run the initialization script
python init_dvc.py

# Or manually:
dvc init
mkdir -p ../dvc-storage
dvc remote add -d localstorage ../dvc-storage

# Track your data file
dvc add data/raw/insurance.csv
git add data/raw/insurance.csv.dvc .gitignore
git commit -m "task-2: add raw data tracked by dvc"
dvc push
```

### Step 4: Run the Pipeline
```bash
# Run complete pipeline
dvc repro

# Or run individual stages:
python src/data/prepare_data.py      # Data cleaning
python src/eda/run_eda.py            # EDA
python src/eda/hypothesis_testing.py # Task 3: Hypothesis tests
python src/models/advanced_modeling.py # Task 4: Advanced models
```

## 📊 What Happens Without Data File?

The code is designed to handle missing data gracefully:

1. **Data Loading**: Creates empty DataFrame structure
2. **Data Cleaning**: Logs warnings, returns empty DataFrame
3. **EDA**: Creates empty reports with structure
4. **Models**: Logs errors, creates empty metrics files
5. **Visualizations**: Skips if no data

**You can develop and test the code structure without data**, but to get actual results, you need the data file.

## 🔍 Verifying Your Data File

Run this Python script to check your data:

```python
import pandas as pd
from pathlib import Path

data_path = Path("data/raw/insurance.csv")
if data_path.exists():
    df = pd.read_csv(data_path)
    print(f"✅ Data file found!")
    print(f"Shape: {df.shape}")
    print(f"\nColumns:")
    print(df.columns.tolist())
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
else:
    print("❌ Data file not found at data/raw/insurance.csv")
    print("Please add your insurance.csv file to that location")
```

## 📝 Expected Output Files

After running the pipeline, you should see:

### Data Files:
- `data/processed/insurance_cleaned.csv` - Cleaned data
- `data/processed/eda_features.pkl` - EDA features

### Models:
- `models/claims_model.pkl` - Claims prediction model
- `models/premium_model.pkl` - Premium model
- `models/claim_severity_model.pkl` - Severity model (Task 4)
- `models/claim_probability_model.pkl` - Probability model (Task 4)

### Reports:
- `reports/interim/eda_report.html` - EDA report
- `reports/interim/low_risk_groups.csv` - Low-risk groups
- `reports/interim/hypothesis_test_results.json` - Task 3 results
- `reports/final/model_evaluation_results.json` - Task 4 results
- `reports/final/claim_severity_shap_importance.csv` - SHAP analysis
- `reports/interim/*.png` - Visualizations (bar charts, box plots, etc.)

### Logs:
- `logs/pipeline.log` - Detailed execution logs

## ❓ Troubleshooting

### "File not found" errors:
- Check that `data/raw/insurance.csv` exists
- Verify file path is correct
- Check file permissions

### "Column not found" warnings:
- Your data might have different column names
- Update column names in your CSV to match expected names
- Or modify the code to use your column names

### DVC errors:
- Make sure DVC is installed: `pip install dvc`
- Run `dvc init` first
- Check `.dvc/config` exists

### Import errors:
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check you're in the project root directory
- Verify virtual environment is activated

## 📚 Additional Resources

- See `QUICK_REFERENCE.md` for command snippets
- See `TASK_3_4_SUMMARY.md` for Task 3 & 4 details
- See `README.md` for general project information

