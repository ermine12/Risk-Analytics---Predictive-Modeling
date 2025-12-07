"""Configuration settings for the project."""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"

# Reports paths
REPORTS_DIR = PROJECT_ROOT / "reports"
INTERIM_REPORTS_DIR = REPORTS_DIR / "interim"
FINAL_REPORTS_DIR = REPORTS_DIR / "final"

# Notebooks path
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
                 INTERIM_REPORTS_DIR, FINAL_REPORTS_DIR, NOTEBOOKS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42

# Plotting style
PLOT_STYLE = "seaborn-v0_8"
FIGURE_SIZE = (12, 6)

