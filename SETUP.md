# Setup Instructions

## Initial Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone <your-repo-url>
   cd insurance-risk-analytics
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .venv\Scripts\activate
   
   # Linux/Mac
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize DVC**:
   ```bash
   # Windows PowerShell
   .\setup_dvc.ps1
   
   # Linux/Mac
   chmod +x setup_dvc.sh
   ./setup_dvc.sh
   
   # Or manually:
   dvc init
   mkdir -p ../dvc-storage
   dvc remote add -d localstorage ../dvc-storage
   ```

5. **Add your data**:
   ```bash
   # Place your insurance.csv file in data/raw/
   # Then track it with DVC:
   dvc add data/raw/insurance.csv
   git add data/raw/insurance.csv.dvc .gitignore
   git commit -m "task-2: add raw data tracked by dvc"
   dvc push
   ```

## Running the Pipeline

Once data is added, you can run the complete pipeline:

```bash
dvc repro
```

Or run individual stages:

```bash
dvc repro prepare_data
dvc repro eda
dvc repro train_claims_model
dvc repro train_premium_model
dvc repro evaluate
```

## Development Workflow

1. **Make changes** to code
2. **Test locally**:
   ```bash
   pytest src/tests/
   black src/ --check
   flake8 src/
   ```
3. **Commit changes** (at least 3x per day):
   ```bash
   git add .
   git commit -m "task-1: description of changes"
   ```
4. **Push to branch**:
   ```bash
   git push origin task-1
   ```

## Branching Strategy

- `task-1`: Initial development branch
- `task-2`: Feature branch (created from task-1 when ready)
- `main`: Production-ready code

To create task-2 from task-1:
```bash
git checkout task-1
git checkout -b task-2
# Make changes and commit
# Open PR to main when ready
```

