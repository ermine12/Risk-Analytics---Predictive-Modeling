"""Data preparation and cleaning pipeline."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.logger import logger


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load raw insurance data from CSV file.
    
    Args:
        filepath: Optional path to data file. If None, uses default location.
        
    Returns:
        DataFrame with insurance data. Empty DataFrame if file not found.
        
    Assumptions:
        - CSV file exists and is readable
        - File contains insurance-related columns
    """
    if filepath is None:
        filepath = RAW_DATA_DIR / "insurance.csv"
    
    if not Path(filepath).exists():
        logger.warning(f"{filepath} not found. Creating sample data structure.")
        return create_sample_structure()
    
    logger.info(f"Loading data from {filepath}")
    return pd.read_csv(filepath)


def create_sample_structure() -> pd.DataFrame:
    """
    Create empty DataFrame with expected insurance data structure.
    
    Returns:
        Empty DataFrame with expected column names for insurance data.
        
    Note:
        Used when actual data file is not available for testing purposes.
    """
    logger.info("Creating sample data structure...")
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
    """
    Clean and preprocess the insurance data with explicit steps.
    
    Steps:
    1. Handle missing values
    2. Remove duplicates
    3. Handle outliers in critical numeric features
    4. Validate data types
    5. Handle zero/invalid values in premium and claims
    """
    df = df.copy()
    initial_shape = df.shape
    logger.info(f"Starting data cleaning. Initial shape: {initial_shape}")
    
    # Step 1: Handle missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        logger.warning(f"Found missing values:\n{missing_counts[missing_counts > 0]}")
        
        # For critical columns, log details
        critical_cols = ['TotalClaims', 'TotalPremium', 'Province', 'PostalCode']
        for col in critical_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                logger.warning(f"{col}: {df[col].isnull().sum()} missing values ({df[col].isnull().sum()/len(df)*100:.2f}%)")
        
        # Drop rows with missing critical values
        df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
        logger.info(f"After dropping missing critical values: {df.shape}")
    
    # Step 2: Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows ({duplicates_removed/initial_rows*100:.2f}%)")
    
    # Step 3: Handle outliers in critical numeric features
    critical_numeric = ['TotalClaims', 'TotalPremium']
    for col in critical_numeric:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outliers > 0:
                logger.warning(f"{col}: Found {outliers} outliers ({outliers/len(df)*100:.2f}%)")
                logger.debug(f"{col}: Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
                # Cap outliers instead of removing (preserve data)
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
                logger.info(f"{col}: Capped outliers to bounds")
    
    # Step 4: Handle zero/invalid values
    if 'TotalPremium' in df.columns:
        zero_premium = (df['TotalPremium'] <= 0).sum()
        if zero_premium > 0:
            logger.warning(f"Found {zero_premium} rows with TotalPremium <= 0. Removing...")
            df = df[df['TotalPremium'] > 0]
    
    if 'TotalClaims' in df.columns:
        negative_claims = (df['TotalClaims'] < 0).sum()
        if negative_claims > 0:
            logger.warning(f"Found {negative_claims} rows with negative TotalClaims. Setting to 0...")
            df.loc[df['TotalClaims'] < 0, 'TotalClaims'] = 0
    
    # Step 5: Validate data types
    if 'TotalClaims' in df.columns:
        df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce')
    if 'TotalPremium' in df.columns:
        df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce')
    
    final_shape = df.shape
    logger.info(f"Data cleaning complete. Final shape: {final_shape}")
    logger.info(f"Rows removed: {initial_shape[0] - final_shape[0]} ({((initial_shape[0] - final_shape[0])/initial_shape[0]*100):.2f}%)")
    
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for modeling.
    
    Creates derived features including:
    - Date parsing and extraction (year, month)
    - Loss ratio calculation
    - Vehicle age (if registration year available)
    - Vehicle value bins
    - Time since vehicle introduction
    - Interaction features (Gender×VehicleType, Make×YearBucket)
    - Historical aggregates per policy
    
    Args:
        df: Input DataFrame with raw/cleaned insurance data.
        
    Returns:
        DataFrame with original columns plus engineered features.
        
    Assumptions:
        - Date columns can be parsed as datetime
        - Numeric columns exist for calculations
    """
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
    logger.info("=" * 60)
    logger.info("Starting data preparation pipeline")
    logger.info("=" * 60)
    
    try:
        # Load data
        df = load_data()
        
        if df.empty:
            logger.error("No data to process. Please add insurance.csv to data/raw/")
            raise ValueError("No data available for processing")
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
        # Clean data
        df_cleaned = clean_data(df)
        
        # Prepare features
        logger.info("Starting feature engineering...")
        df_cleaned = prepare_features(df_cleaned)
        logger.info("Feature engineering complete")
        
        # Save processed data
        output_path = PROCESSED_DATA_DIR / "insurance_cleaned.csv"
        df_cleaned.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        logger.info(f"Final shape: {df_cleaned.shape}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in data preparation pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

