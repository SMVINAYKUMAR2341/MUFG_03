# Quick Deploy Script for Vercel (PowerShell)
# Run this script to deploy your Heart Disease Prediction app

Write-Host "`n🚀 Starting Vercel Deployment...`n" -ForegroundColor Green

# Check if Vercel CLI is installed
$vercelInstalled = Get-Command vercel -ErrorAction SilentlyContinue

if (-not $vercelInstalled) {
    Write-Host "❌ Vercel CLI not found. Installing..." -ForegroundColor Red
    npm install -g vercel
    Write-Host "✅ Vercel CLI installed!`n" -ForegroundColor Green
} else {
    Write-Host "✅ Vercel CLI found!`n" -ForegroundColor Green
}

# Login to Vercel
Write-Host "🔐 Logging into Vercel...`n" -ForegroundColor Cyan
vercel login

# Deploy to production
Write-Host "`n📦 Deploying to production...`n" -ForegroundColor Cyan
vercel --prod

Write-Host "`n✅ Deployment complete!" -ForegroundColor Green
Write-Host "🌐 Your app is now live!`n" -ForegroundColor Green
