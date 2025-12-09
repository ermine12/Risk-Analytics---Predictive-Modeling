"""Quick script to check if required data file exists and verify its structure."""

import pandas as pd
from pathlib import Path
import sys

def check_data_file():
    """Check if data file exists and display its structure."""
    print("=" * 60)
    print("Data File Checker")
    print("=" * 60)
    
    data_path = Path("data/raw/insurance.csv")
    
    if not data_path.exists():
        print(f"❌ Data file NOT found at: {data_path}")
        print("\n📋 Expected location: data/raw/insurance.csv")
        print("\n💡 To fix:")
        print("   1. Place your insurance.csv file in data/raw/")
        print("   2. Ensure the file is named 'insurance.csv'")
        print("   3. Run this script again to verify")
        return False
    
    print(f"✅ Data file found at: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"\n📊 File Statistics:")
        print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"   Size: {data_path.stat().st_size / 1024:.2f} KB")
        
        print(f"\n📋 Columns ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            print(f"   {i:2d}. {col:25s} ({dtype:10s}) - {null_count:4d} nulls ({null_pct:5.2f}%)")
        
        print(f"\n📈 Data Types Summary:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} columns")
        
        print(f"\n🔍 Missing Values:")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("   ✅ No missing values!")
        else:
            print("   ⚠️  Missing values found:")
            for col, count in missing[missing > 0].items():
                print(f"      {col}: {count} ({count/len(df)*100:.2f}%)")
        
        print(f"\n📋 First 5 rows:")
        print(df.head().to_string())
        
        print(f"\n✅ Data file is ready to use!")
        print("\n💡 Next steps:")
        print("   1. Run: python src/data/prepare_data.py")
        print("   2. Or run full pipeline: dvc repro")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error reading data file: {e}")
        print("\n💡 Possible issues:")
        print("   - File might be corrupted")
        print("   - File might not be a valid CSV")
        print("   - Encoding issues (try UTF-8)")
        return False

if __name__ == "__main__":
    success = check_data_file()
    sys.exit(0 if success else 1)

