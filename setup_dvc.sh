#!/bin/bash
# Setup script for DVC initialization

echo "Setting up DVC..."

# Initialize DVC
dvc init

# Create local storage directory (adjust path as needed)
STORAGE_PATH="../dvc-storage"
mkdir -p "$STORAGE_PATH"

# Add local remote
dvc remote add -d localstorage "$STORAGE_PATH"

echo "DVC setup complete!"
echo "To add data: dvc add data/raw/insurance.csv"
echo "Then commit: git add data/raw/insurance.csv.dvc .gitignore"
echo "And push: dvc push"

