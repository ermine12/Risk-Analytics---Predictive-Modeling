"""Unit tests for data processing."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.prepare_data import clean_data, prepare_features


def test_clean_data():
    """Test data cleaning function."""
    # Create sample data with missing values and duplicates
    df = pd.DataFrame({
        'age': [25, 30, None, 25, 35],
        'charges': [1000, 2000, 3000, 1000, 4000]
    })
    
    df_cleaned = clean_data(df)
    
    # Should remove missing values and duplicates
    assert df_cleaned.isnull().sum().sum() == 0
    assert len(df_cleaned) <= len(df)


def test_prepare_features():
    """Test feature preparation function."""
    df = pd.DataFrame({
        'age': [25, 30, 35],
        'sex': ['male', 'female', 'male'],
        'charges': [1000, 2000, 3000]
    })
    
    df_prepared = prepare_features(df)
    
    # Should return a DataFrame
    assert isinstance(df_prepared, pd.DataFrame)
    assert len(df_prepared) == len(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

