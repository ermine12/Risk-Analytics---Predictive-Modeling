"""Initialize DVC and configure remote storage."""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, check=True):
    """Run shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and check:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result

def main():
    """Initialize DVC repository."""
    print("=" * 60)
    print("Initializing DVC Repository")
    print("=" * 60)
    
    # Check if DVC is installed
    result = run_command("dvc --version", check=False)
    if result.returncode != 0:
        print("ERROR: DVC is not installed. Please run: pip install dvc")
        sys.exit(1)
    
    print(f"DVC version: {result.stdout.strip()}")
    
    # Initialize DVC
    print("\n1. Initializing DVC...")
    run_command("dvc init", check=False)  # May already be initialized
    
    # Create local storage directory
    storage_path = Path("../dvc-storage").resolve()
    storage_path.mkdir(exist_ok=True)
    print(f"\n2. Created storage directory: {storage_path}")
    
    # Configure remote
    print("\n3. Configuring DVC remote...")
    run_command(f"dvc remote add -d localstorage {storage_path}", check=False)
    
    # Verify configuration
    print("\n4. Verifying DVC configuration...")
    result = run_command("dvc remote list", check=False)
    print(result.stdout)
    
    print("\n" + "=" * 60)
    print("DVC Initialization Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Add your data file:")
    print("   dvc add data/raw/insurance.csv")
    print("2. Commit DVC tracking files:")
    print("   git add data/raw/insurance.csv.dvc .gitignore")
    print("   git commit -m 'task-2: add raw data tracked by dvc'")
    print("3. Push to remote:")
    print("   dvc push")

if __name__ == "__main__":
    main()

