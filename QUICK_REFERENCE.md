# Quick Reference - Commands & Snippets

## Data Processing

### Parse date + loss ratio
```python
df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
df['loss_ratio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, np.nan)
```

### Group loss ratio by province
```python
prov = df.groupby('Province').agg({'TotalClaims':'sum','TotalPremium':'sum'})
prov['loss_ratio'] = prov['TotalClaims']/prov['TotalPremium']
prov.sort_values('loss_ratio')
```

## Model Training

### Train Random Forest
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = df[feature_cols]
y = df['TotalClaims']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
```

## Feature Engineering

### Age of vehicle
```python
df['VehicleAge'] = df['TransactionMonth'].dt.year - df['RegistrationYear']
```

### Vehicle value bins
```python
df['VehicleValueBin'] = pd.cut(
    df['CustomValueEstimate'],
    bins=[0, 10000, 25000, 50000, 100000, np.inf],
    labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
)
```

### Time since vehicle introduced
```python
df['TimeSinceIntro'] = (df['TransactionMonth'] - df['VehicleIntroDate']).dt.days / 365.25
```

### CRESTA zone encoding (one-hot)
```python
df = pd.get_dummies(df, columns=['CRESTA'], prefix='CRESTA', drop_first=True)
```

### Interaction features
```python
df['Gender_VehicleType'] = df['Gender'].astype(str) + '_' + df['VehicleType'].astype(str)
df['Make_YearBucket'] = df['Make'].astype(str) + '_' + df['RegistrationYearBucket'].astype(str)
```

### Aggregates: historical claims per PolicyID
```python
df['HistoricalClaimsPerPolicy'] = df.groupby('PolicyID')['TotalClaims'].transform('sum')
```

## Low-Risk Group Recommendations

### For each candidate low-risk group, show:
1. **Group definition**: Province X, PostalCode Y, VehicleType Z
2. **Historical loss ratio and CI**: Calculate confidence intervals
3. **Number of policies**: Sample size
4. **Proposed premium adjustment**: e.g., "reduce premium by 10% → expected conversion +X% and projected loss ratio remains ≤ target"
5. **Financial impact simulation**: Projected profit change from proposed premium change

## Important Notes

- **Always set `random_state=42`** everywhere
- **Log steps and decisions** in README (e.g., how you handle zeros in TotalPremium)
- **Use joblib.dump** for models: `joblib.dump(model, 'models/rf_v1.joblib')`
- **For feature importance, prioritize SHAP** for business explainability
- **Use stratified sample for CI runs** if dataset is big; full DVC pipeline for final runs

## DVC Commands

```bash
# Initialize DVC
dvc init

# Add data
dvc add data/raw/insurance.csv
git add data/raw/insurance.csv.dvc .gitignore
git commit -m "task-2: add raw data tracked by dvc"
dvc push

# Run pipeline
dvc repro

# Run specific stage
dvc repro prepare_data
```

## Git Commands

```bash
# Commit at least 3x/day
git add .
git commit -m "task-1: eda - missing values cleanup"

# Create task-2 from task-1
git checkout task-1
git checkout -b task-2
```

