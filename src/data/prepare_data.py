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
        'TransactionMonth': [],
        'TotalClaims': [],
        'TotalPremium': [],
        'Province': [],
        'PostalCode': [],
        'VehicleType': [],
        'RegistrationYear': [],
        'CustomValueEstimate': [],
        'VehicleIntroDate': [],
        'Gender': [],
        'Make': [],
        'PolicyID': []
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
    """Prepare features for modeling with feature engineering."""
    df = df.copy()
    
    # Parse date
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        df['TransactionYear'] = df['TransactionMonth'].dt.year
        df['TransactionMonthNum'] = df['TransactionMonth'].dt.month
    
    # Calculate loss ratio
    if 'TotalClaims' in df.columns and 'TotalPremium' in df.columns:
        df['loss_ratio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, np.nan)
    
    # Feature engineering: Age of vehicle
    if 'RegistrationYear' in df.columns and 'TransactionMonth' in df.columns:
        current_year = df['TransactionMonth'].dt.year if 'TransactionMonth' in df.columns else pd.Timestamp.now().year
        if isinstance(current_year, pd.Series):
            df['VehicleAge'] = current_year - df['RegistrationYear']
        else:
            df['VehicleAge'] = current_year - df['RegistrationYear']
    
    # Feature engineering: Vehicle value bins
    if 'CustomValueEstimate' in df.columns:
        df['VehicleValueBin'] = pd.cut(
            df['CustomValueEstimate'],
            bins=[0, 10000, 25000, 50000, 100000, np.inf],
            labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
        )
    
    # Feature engineering: Time since vehicle introduced
    if 'VehicleIntroDate' in df.columns and 'TransactionMonth' in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df['VehicleIntroDate']):
            df['TimeSinceIntro'] = (df['TransactionMonth'] - df['VehicleIntroDate']).dt.days / 365.25
        else:
            df['VehicleIntroDate'] = pd.to_datetime(df['VehicleIntroDate'], errors='coerce')
            df['TimeSinceIntro'] = (df['TransactionMonth'] - df['VehicleIntroDate']).dt.days / 365.25
    
    # Feature engineering: Interaction features
    if 'Gender' in df.columns and 'VehicleType' in df.columns:
        df['Gender_VehicleType'] = df['Gender'].astype(str) + '_' + df['VehicleType'].astype(str)
    
    if 'Make' in df.columns and 'RegistrationYear' in df.columns:
        # Create registration year buckets
        df['RegistrationYearBucket'] = pd.cut(
            df['RegistrationYear'],
            bins=[0, 2010, 2015, 2020, np.inf],
            labels=['Old', 'Medium-Old', 'Recent', 'New']
        )
        df['Make_YearBucket'] = df['Make'].astype(str) + '_' + df['RegistrationYearBucket'].astype(str)
    
    # Feature engineering: Historical aggregates (if PolicyID exists)
    if 'PolicyID' in df.columns and 'TotalClaims' in df.columns:
        # Historical claims per policy
        policy_claims = df.groupby('PolicyID')['TotalClaims'].transform('sum')
        df['HistoricalClaimsPerPolicy'] = policy_claims
    
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

