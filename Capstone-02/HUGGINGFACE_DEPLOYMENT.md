# 🤗 Hugging Face Spaces Deployment Guide

## Overview

Deploy your Heart Disease Prediction app to Hugging Face Spaces with Gradio interface.

## Prerequisites

1. **Hugging Face Account**: Sign up at [https://huggingface.co/join](https://huggingface.co/join)
2. **Git & Git LFS**: For pushing large model files

## Deployment Steps

### Option 1: Deploy via Web Interface (Easiest) ⭐

1. **Create New Space**
   - Go to [https://huggingface.co/new-space](https://huggingface.co/new-space)
   - Space name: `heart-disease-prediction`
   - License: `MIT`
   - SDK: Select `Gradio`
   - Space hardware: `CPU basic` (free tier)
   - Click **Create Space**

2. **Upload Files**
   
   Upload these files to your Space:
   
   **Required Files:**
   ```
   ├── app.py                          # Main Gradio application
   ├── README_HF.md                    # Space card (rename to README.md)
   ├── requirements_hf.txt             # Dependencies (rename to requirements.txt)
   ├── models/
   │   └── best_model.pkl             # Trained model
   └── data/
       └── processed/
           └── scaler.pkl             # Feature scaler
   ```

3. **Configure Space**
   - The app will auto-deploy from `app.py`
   - Build takes ~2-3 minutes
   - Once ready, your app will be live!

### Option 2: Deploy via Git (Advanced)

1. **Install Git LFS** (for large files)
   ```bash
   # Windows
   git lfs install
   
   # Mac
   brew install git-lfs
   git lfs install
   
   # Linux
   curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
   sudo apt-get install git-lfs
   git lfs install
   ```

2. **Clone Your Space**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/heart-disease-prediction
   cd heart-disease-prediction
   ```

3. **Copy Files**
   ```bash
   # Copy from your project
   cp ../Capstone-02/app.py .
   cp ../Capstone-02/README_HF.md README.md
   cp ../Capstone-02/requirements_hf.txt requirements.txt
   
   # Copy models
   mkdir -p models data/processed
   cp ../Capstone-02/models/best_model.pkl models/
   cp ../Capstone-02/data/processed/scaler.pkl data/processed/
   ```

4. **Track Large Files with Git LFS**
   ```bash
   git lfs track "*.pkl"
   git add .gitattributes
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "Initial deployment of Heart Disease Prediction app"
   git push
   ```

6. **Wait for Build**
   - Check build logs at your Space URL
   - App will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/heart-disease-prediction`

### Option 3: Connect GitHub Repository

1. **Go to Your Space Settings**
   - Navigate to your Space on Hugging Face
   - Click **Settings** tab

2. **Link Repository**
   - Find "Repository" section
   - Click **Link a GitHub repository**
   - Select: `SMVINAYKUMAR2341/MUFG_03`
   - Set path: `Capstone-02/`

3. **Configure Files**
   - Rename `README_HF.md` to `README.md`
   - Rename `requirements_hf.txt` to `requirements.txt`
   - Ensure `app.py` is in root of Capstone-02

4. **Auto-Deploy**
   - Push changes to GitHub = Auto-deploy to HF!

## File Structure for Hugging Face

```
heart-disease-prediction/
├── app.py                    # Gradio interface (REQUIRED)
├── README.md                 # Space card with metadata (REQUIRED)
├── requirements.txt          # Python dependencies (REQUIRED)
├── models/
│   └── best_model.pkl       # ML model (~50KB)
└── data/
    └── processed/
        └── scaler.pkl       # Feature scaler (~5KB)
```

## Important Notes

### Model Files

✅ **Good News**: Your model files are small (~56KB total)
- `best_model.pkl`: ~50KB
- `scaler.pkl`: ~5KB

These are well under Hugging Face's limits, so Git LFS is optional but recommended.

### Space Card (README.md)

The README_HF.md contains YAML metadata at the top:
```yaml
---
title: Heart Disease Prediction
emoji: ❤️
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---
```

This configures your Space's appearance and behavior.

### Hardware Requirements

- **Free Tier (CPU basic)**: Perfect for this app
- **Upgrade**: If needed, can upgrade to GPU for faster inference

## Gradio Interface Features

Your app includes:

✅ **13 Input Parameters**:
- Demographics (age, sex)
- Vital signs (blood pressure, cholesterol, heart rate)
- Medical tests (ECG, blood sugar)
- Cardiac symptoms (chest pain, angina)
- Advanced tests (ST depression, vessels, thalassemia)

✅ **Real-time Predictions**:
- Disease detection (Yes/No)
- Probability percentage
- Risk level (Low/Moderate/High/Very High)

✅ **Professional UI**:
- Color-coded risk levels
- Organized input sections
- Clear result display
- Mobile responsive

## Testing Your Deployment

Once deployed, test these scenarios:

1. **Low Risk Patient**:
   - Age: 35, Male, No chest pain
   - Normal vitals
   - Expected: Low risk

2. **High Risk Patient**:
   - Age: 65, Male, Typical angina
   - High BP, High cholesterol
   - Expected: High/Very High risk

3. **Sample Data**: Use the default values for quick test

## Troubleshooting

### Build Fails

**Issue**: Dependencies not installing
**Solution**: Check `requirements.txt` syntax, ensure package names are correct

**Issue**: Model file not found
**Solution**: Verify file paths in `app.py` match your structure

### Runtime Errors

**Issue**: "Model not loaded"
**Solution**: 
- Check model files uploaded correctly
- Verify paths: `models/best_model.pkl` and `data/processed/scaler.pkl`

**Issue**: "Input validation error"
**Solution**: Check feature names match training data

### Performance Issues

**Issue**: Slow predictions
**Solution**: 
- Use CPU basic tier (sufficient for this model)
- Model is lightweight (~50KB), should be fast

## Monitoring & Analytics

- **Usage Stats**: View in Space settings
- **Logs**: Check build and runtime logs
- **Updates**: Push to GitHub = Auto-redeploy

## Sharing Your Space

Once deployed, share:
- **Direct URL**: `https://huggingface.co/spaces/YOUR_USERNAME/heart-disease-prediction`
- **Embed**: Use embed code in websites/blogs
- **API Access**: Gradio provides automatic API

## Cost

✅ **FREE** on Hugging Face Spaces!
- CPU basic tier: Free forever
- No credit card required
- Unlimited inference requests

## Next Steps

After deployment:

1. ✅ Test the interface thoroughly
2. ✅ Share your Space URL
3. ✅ Add to your portfolio/resume
4. ✅ Monitor usage analytics
5. ✅ Collect user feedback
6. ✅ Iterate and improve

## Additional Features (Optional)

### Add Examples

```python
demo.launch(
    examples=[
        [55, 1, 0, 132, 342, 0, 1, 166, 0, 1.2, 2, 0, 2],  # Low risk
        [70, 1, 3, 145, 274, 0, 1, 125, 1, 2.6, 0, 0, 3],  # High risk
    ]
)
```

### Enable API

```python
demo.launch(
    share=False,
    enable_queue=True,  # Better for multiple users
    show_api=True       # Show API documentation
)
```

### Add Analytics

Integrate with your analytics platform for usage tracking.

## Support

- **Hugging Face Docs**: [https://huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- **Gradio Docs**: [https://gradio.app/docs](https://gradio.app/docs)
- **Community**: [https://huggingface.co/spaces](https://huggingface.co/spaces)

---

## Quick Commands

```bash
# Create new space
huggingface-cli login
huggingface-cli repo create heart-disease-prediction --type space --space_sdk gradio

# Push to space
git add . && git commit -m "Update app" && git push
```

🚀 **Ready to deploy? Visit https://huggingface.co/new-space now!**
