"""Data preparation and cleaning pipeline."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_data(filepath: str = None) -> pd.DataFrame:
    """Load raw insurance data."""
    if filepath is None:
        filepath = RAW_DATA_DIR / "insurance.csv"
    
    if not Path(filepath).exists():
        print(f"Warning: {filepath} not found. Creating sample data structure.")
        return create_sample_structure()
    
    return pd.read_csv(filepath)


def create_sample_structure() -> pd.DataFrame:
    """Create sample data structure for testing."""
    print("Creating sample data structure...")
    return pd.DataFrame({
        'age': [],
        'sex': [],
        'bmi': [],
        'children': [],
        'smoker': [],
        'region': [],
        'charges': []
    })


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the insurance data."""
    df = df.copy()
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        print(f"Found {df.isnull().sum().sum()} missing values")
        df = df.dropna()
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Handle outliers (using IQR method for numerical columns)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > 0:
            print(f"Found {outliers} outliers in {col}")
            # Optionally remove or cap outliers
            # df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for modeling."""
    df = df.copy()
    
    # Encode categorical variables if needed
    # This is a placeholder - actual encoding will depend on the data
    
    return df


def main():
    """Main data preparation pipeline."""
    print("Starting data preparation...")
    
    # Load data
    df = load_data()
    
    if df.empty:
        print("No data to process. Please add insurance.csv to data/raw/")
        # Create empty processed file
        df_cleaned = pd.DataFrame()
    else:
        # Clean data
        df_cleaned = clean_data(df)
        
        # Prepare features
        df_cleaned = prepare_features(df_cleaned)
    
    # Save processed data
    output_path = PROCESSED_DATA_DIR / "insurance_cleaned.csv"
    df_cleaned.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    print(f"Shape: {df_cleaned.shape}")


if __name__ == "__main__":
    main()

