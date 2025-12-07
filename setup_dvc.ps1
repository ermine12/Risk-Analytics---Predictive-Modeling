# PowerShell setup script for DVC initialization

Write-Host "Setting up DVC..." -ForegroundColor Green

# Initialize DVC
dvc init

# Create local storage directory (adjust path as needed)
$storagePath = "..\dvc-storage"
New-Item -ItemType Directory -Force -Path $storagePath | Out-Null

# Add local remote
dvc remote add -d localstorage $storagePath

Write-Host "DVC setup complete!" -ForegroundColor Green
Write-Host "To add data: dvc add data/raw/insurance.csv"
Write-Host "Then commit: git add data/raw/insurance.csv.dvc .gitignore"
Write-Host "And push: dvc push"

