# Insurance Risk Analytics & Predictive Modeling

A comprehensive machine learning project for insurance risk analysis, featuring data versioning with DVC, exploratory data analysis, hypothesis testing, predictive modeling, and automated CI/CD pipelines.

## 🎯 Project Overview

This repository implements a complete insurance risk analytics pipeline that:
- **Versions data** using DVC for reproducible experiments
- **Performs EDA** with hypothesis tests to identify low-risk customer groups
- **Trains models** to predict total claims and recommend optimal premiums
- **Includes CI/CD** for automated testing and quality checks
- **Generates reports** with creative visualizations and insights

## 📁 Repository Structure

```
insurance-risk-analytics/
├── data/                  # Data directory (raw files tracked by DVC)
│   ├── raw/              # Raw insurance data
│   └── processed/        # Processed/cleaned data
├── src/                  # Source code
│   ├── data/             # Data loaders, cleaning, featurizers
│   ├── eda/              # EDA notebooks & plotting scripts
│   ├── models/           # Model training & evaluation code
│   ├── tests/            # Unit tests
│   └── utils/            # Helper functions and config
├── notebooks/            # Exploratory notebooks (clean copies for reports)
├── reports/               # Generated reports
│   ├── interim/          # Interim report
│   └── final/            # Final report & slides
├── dvc.yaml              # DVC pipeline configuration
├── .dvc/                 # DVC metadata
├── requirements.txt      # Python dependencies
├── .github/workflows/    # CI/CD workflows
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git
- DVC

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd insurance-risk-analytics
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   
   # On Windows:
   .venv\Scripts\activate
   
   # On Linux/Mac:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize DVC:**
   ```bash
   dvc init
   ```

5. **Set up DVC remote (local storage example):**
   ```bash
   mkdir -p ../dvc-storage
   dvc remote add -d localstorage ../dvc-storage
   ```

6. **Pull data (if available):**
   ```bash
   dvc pull
   ```

## 📊 Usage

### Running EDA

```bash
# Run EDA notebooks
jupyter notebook notebooks/eda_exploration.ipynb
```

### Training Models

```bash
# Run model training pipeline
dvc repro
```

### Running Tests

```bash
# Run unit tests
pytest src/tests/

# Run with coverage
pytest src/tests/ --cov=src
```

### Code Quality

```bash
# Format code
black src/

# Lint code
flake8 src/
```

## 🔄 DVC Workflow

### Adding Data

```bash
# Add raw data file
dvc add data/raw/insurance.csv
git add data/raw/insurance.csv.dvc .gitignore
git commit -m "task-2: add raw data tracked by dvc"
dvc push
```

### Running Pipeline

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro prepare_data
```

### Pulling Data

```bash
# Pull data from remote
dvc pull
```

## 🧪 CI/CD

The repository includes GitHub Actions workflows for:
- Automated testing on push/PR
- Code quality checks (black, flake8)
- DVC pipeline validation

See `.github/workflows/ci.yml` for details.

## 📈 Key Features

1. **Data Versioning**: All datasets are versioned using DVC
2. **EDA & Hypothesis Testing**: Comprehensive analysis to identify low-risk groups
3. **Predictive Modeling**: Models for claims prediction and premium optimization
4. **Reproducibility**: Complete pipeline with DVC for reproducible experiments
5. **CI/CD**: Automated testing and quality checks
6. **Creative Visualizations**: 3+ unique plots for insights
7. **Documentation**: Interim and final reports with findings

## 📝 Reports

- **Interim Report**: `reports/interim/` - Initial findings and EDA results
- **Final Report**: `reports/final/` - Complete analysis, model results, and recommendations

## 🤝 Contributing

### Git Workflow & Pull Requests

This project uses a **Pull Request (PR) workflow** for all changes. All task branches must be merged into `main` via Pull Requests.

#### Branching Strategy

- `main`: Production-ready code (protected branch)
- `task-1`, `task-2`, `task-3`, etc.: Task branches for specific features
- `feature/description`: Feature branches for new functionality
- `fix/description`: Bug fix branches

#### Workflow Steps

1. **Create a task branch:**
   ```bash
   git checkout -b task-X
   ```

2. **Make changes and commit:**
   ```bash
   git add .
   git commit -m "task-X: descriptive commit message"
   ```

3. **Push to remote:**
   ```bash
   git push origin task-X
   ```

4. **Create Pull Request:**
   - Go to GitHub repository
   - Click "Pull Requests" → "New Pull Request"
   - Select base: `main`, compare: `task-X`
   - Fill out PR template with:
     - Description of changes
     - Testing performed
     - Business impact
     - DVC pipeline changes (if any)

5. **Review Process:**
   - Code is reviewed using [Code Review Guidelines](.github/CODE_REVIEW_GUIDELINES.md)
   - Address feedback and re-request review
   - CI/CD checks must pass

6. **Merge:**
   - After approval, merge via GitHub UI (usually "Squash and Merge")
   - Delete branch after merge

#### Commit Guidelines

- Commit at least 3x per day while working
- Use descriptive messages: `task-X: eda - missing values cleanup`
- Format: `task-X: brief description of changes`

#### PR Requirements

- ✅ All CI/CD checks pass
- ✅ Code follows style guidelines
- ✅ Tests are passing
- ✅ Documentation is updated
- ✅ PR template is filled out
- ✅ At least one approval (if required)

**See [Pull Request Guide](.github/PULL_REQUEST_GUIDE.md) for detailed instructions.**

## 📄 License

[Specify your license here]

## 👥 Authors

[Your name/team]

## 🙏 Acknowledgments

[Any acknowledgments]

