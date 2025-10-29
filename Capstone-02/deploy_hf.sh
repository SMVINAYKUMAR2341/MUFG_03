#!/bin/bash

# Hugging Face Spaces Deployment Script
# Quick deploy for Heart Disease Prediction app

echo "🤗 Hugging Face Spaces Deployment"
echo "=================================="
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null
then
    echo "📦 Installing Hugging Face CLI..."
    pip install huggingface_hub
fi

# Login to Hugging Face
echo "🔐 Login to Hugging Face..."
echo "You'll need your Hugging Face access token"
echo "Get it from: https://huggingface.co/settings/tokens"
echo ""
huggingface-cli login

# Get username
echo ""
echo "Enter your Hugging Face username:"
read HF_USERNAME

# Create space
echo ""
echo "🚀 Creating Space: heart-disease-prediction"
huggingface-cli repo create heart-disease-prediction --type space --space_sdk gradio

# Clone the space
echo ""
echo "📥 Cloning Space repository..."
git clone https://huggingface.co/spaces/$HF_USERNAME/heart-disease-prediction
cd heart-disease-prediction

# Copy files
echo ""
echo "📋 Copying project files..."
cp ../app.py .
cp ../README_HF.md README.md
cp ../requirements_hf.txt requirements.txt

# Copy models
mkdir -p models data/processed
cp ../models/best_model.pkl models/
cp ../data/processed/scaler.pkl data/processed/

# Setup git lfs
echo ""
echo "🔧 Setting up Git LFS for model files..."
git lfs install
git lfs track "*.pkl"

# Commit and push
echo ""
echo "📤 Pushing to Hugging Face..."
git add .
git commit -m "Initial deployment: Heart Disease Prediction app"
git push

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Your app will be live at:"
echo "   https://huggingface.co/spaces/$HF_USERNAME/heart-disease-prediction"
echo ""
echo "⏱️  Build takes ~2-3 minutes"
echo "📊 Check build logs in the Space page"
