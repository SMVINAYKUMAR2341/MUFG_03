# Hugging Face Spaces Deployment Script (PowerShell)
# Quick deploy for Heart Disease Prediction app

Write-Host "`n🤗 Hugging Face Spaces Deployment" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Check if huggingface-cli is installed
$hfInstalled = Get-Command huggingface-cli -ErrorAction SilentlyContinue

if (-not $hfInstalled) {
    Write-Host "📦 Installing Hugging Face CLI..." -ForegroundColor Yellow
    pip install huggingface_hub
}

# Login to Hugging Face
Write-Host "🔐 Login to Hugging Face..." -ForegroundColor Yellow
Write-Host "You'll need your Hugging Face access token" -ForegroundColor Gray
Write-Host "Get it from: https://huggingface.co/settings/tokens`n" -ForegroundColor Gray
huggingface-cli login

# Get username
Write-Host "`nEnter your Hugging Face username:" -ForegroundColor Cyan
$HF_USERNAME = Read-Host

# Create space
Write-Host "`n🚀 Creating Space: heart-disease-prediction" -ForegroundColor Green
huggingface-cli repo create heart-disease-prediction --type space --space_sdk gradio

# Clone the space
Write-Host "`n📥 Cloning Space repository..." -ForegroundColor Yellow
git clone "https://huggingface.co/spaces/$HF_USERNAME/heart-disease-prediction"
Set-Location heart-disease-prediction

# Copy files
Write-Host "`n📋 Copying project files..." -ForegroundColor Yellow
Copy-Item ..\app.py .
Copy-Item ..\README_HF.md README.md
Copy-Item ..\requirements_hf.txt requirements.txt

# Copy models
New-Item -ItemType Directory -Path "models" -Force | Out-Null
New-Item -ItemType Directory -Path "data\processed" -Force | Out-Null
Copy-Item ..\models\best_model.pkl models\
Copy-Item ..\data\processed\scaler.pkl data\processed\

# Setup git lfs
Write-Host "`n🔧 Setting up Git LFS for model files..." -ForegroundColor Yellow
git lfs install
git lfs track "*.pkl"

# Commit and push
Write-Host "`n📤 Pushing to Hugging Face..." -ForegroundColor Yellow
git add .
git commit -m "Initial deployment: Heart Disease Prediction app"
git push

Write-Host "`n✅ Deployment complete!" -ForegroundColor Green
Write-Host "`n🌐 Your app will be live at:" -ForegroundColor Cyan
Write-Host "   https://huggingface.co/spaces/$HF_USERNAME/heart-disease-prediction" -ForegroundColor Yellow
Write-Host "`n⏱️  Build takes ~2-3 minutes" -ForegroundColor Gray
Write-Host "📊 Check build logs in the Space page`n" -ForegroundColor Gray
