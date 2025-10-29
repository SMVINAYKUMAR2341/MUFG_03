#!/bin/bash

# Quick Deploy Script for Vercel
# Run this script to deploy your Heart Disease Prediction app

echo "🚀 Starting Vercel Deployment..."
echo ""

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null
then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
    echo "✅ Vercel CLI installed!"
fi

# Login to Vercel
echo ""
echo "🔐 Logging into Vercel..."
vercel login

# Deploy to production
echo ""
echo "📦 Deploying to production..."
vercel --prod

echo ""
echo "✅ Deployment complete!"
echo "🌐 Your app is now live!"
