# Vercel Deployment Guide for Heart Disease Prediction

## Prerequisites

1. **Vercel Account**: Sign up at [https://vercel.com](https://vercel.com)
2. **Vercel CLI**: Install globally
   ```bash
   npm install -g vercel
   ```

## Deployment Steps

### Option 1: Deploy via Vercel CLI (Recommended)

1. **Navigate to project directory**:
   ```bash
   cd d:\CapstoneProject2\Capstone-02
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy to production**:
   ```bash
   vercel --prod
   ```

4. **Follow the prompts**:
   - Set up and deploy: `Y`
   - Which scope: Select your account
   - Link to existing project: `N`
   - Project name: `heart-disease-prediction`
   - Directory: `./`
   - Override settings: `N`

### Option 2: Deploy via GitHub Integration

1. **Push your code to GitHub** (Already done! ✅)

2. **Import to Vercel**:
   - Go to [https://vercel.com/new](https://vercel.com/new)
   - Click "Import Git Repository"
   - Select `SMVINAYKUMAR2341/MUFG_03`
   - Set root directory to `Capstone-02`
   - Click "Deploy"

3. **Configure build settings** (auto-detected from vercel.json):
   - Framework Preset: `Other`
   - Build Command: (leave empty)
   - Output Directory: `static`
   - Install Command: `pip install -r requirements.txt`

## Important Notes

### Model Files
⚠️ **Large File Warning**: Your `models/best_model.pkl` and `data/processed/scaler.pkl` files might be too large for Vercel's free tier (max 100MB total).

**Solution Options**:

1. **Use External Storage** (Recommended):
   - Upload models to AWS S3, Google Cloud Storage, or Azure Blob Storage
   - Load models from URL at runtime
   
2. **Use Vercel Blob Storage**:
   ```bash
   npm install @vercel/blob
   ```

3. **Optimize Model Size**:
   - Use model compression
   - Reduce Random Forest estimators
   - Use lighter model (Logistic Regression)

### Environment Variables

If needed, set environment variables in Vercel dashboard:
- Go to Project Settings → Environment Variables
- Add any API keys or configuration

## Post-Deployment

### Test Your API

Once deployed, test endpoints:

```bash
# Health check
curl https://your-app.vercel.app/health

# Model info
curl https://your-app.vercel.app/model-info

# Frontend
https://your-app.vercel.app/
```

### Custom Domain

1. Go to Project Settings → Domains
2. Add your custom domain
3. Follow DNS configuration instructions

## Troubleshooting

### Issue: Build Failed
- Check logs in Vercel dashboard
- Ensure all dependencies in requirements.txt
- Verify Python version compatibility

### Issue: Model Not Found
- Ensure model files are in correct directory
- Check .vercelignore doesn't exclude models/
- Consider using external storage for large models

### Issue: Static Files Not Serving
- Verify `static/` directory structure
- Check vercel.json routes configuration
- Ensure StaticFiles mount in main.py

### Issue: API Timeout
- Vercel free tier: 10s timeout
- Optimize model loading
- Use caching for model/scaler

## Performance Optimization

1. **Enable Caching**:
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1)
   def load_model():
       return joblib.load(MODEL_PATH)
   ```

2. **Use Edge Functions**:
   - Move static content to Edge Network
   - Faster global response times

3. **Compress Model**:
   ```python
   joblib.dump(model, 'model.pkl', compress=3)
   ```

## Monitoring

- View deployment logs: Vercel Dashboard → Deployments
- Check analytics: Vercel Dashboard → Analytics
- Set up alerts: Vercel Dashboard → Settings → Notifications

## Rollback

If deployment fails:
```bash
vercel rollback
```

Or use Vercel Dashboard → Deployments → Previous Deployment → Promote to Production

## Cost Considerations

**Vercel Free Tier Limits**:
- 100GB Bandwidth/month
- 100 Deployments/day
- Serverless Function Execution: 100GB-Hrs
- Edge Middleware: 1M requests/month

**Upgrade to Pro** if you need:
- Custom domains with SSL
- More bandwidth
- Team collaboration
- Priority support

## Additional Resources

- [Vercel Documentation](https://vercel.com/docs)
- [FastAPI on Vercel](https://vercel.com/docs/frameworks/python)
- [Vercel CLI Reference](https://vercel.com/docs/cli)

---

## Quick Start Command

```bash
# One-command deployment
cd d:\CapstoneProject2\Capstone-02 && vercel --prod
```

🚀 **Your app will be live at**: `https://your-project-name.vercel.app`
