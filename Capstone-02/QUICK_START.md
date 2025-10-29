# 🚀 Vercel Deployment Quick Reference

## ⚡ Quick Deploy (3 Steps)

1. **Go to**: https://vercel.com/new
2. **Import**: `SMVINAYKUMAR2341/MUFG_03`
3. **Set Root**: `Capstone-02` → **Deploy!**

## 🔗 Important URLs

- **GitHub Repo**: https://github.com/SMVINAYKUMAR2341/MUFG_03/tree/main/Capstone-02
- **Deploy Dashboard**: https://vercel.com/new
- **Full Guide**: [VERCEL_DEPLOYMENT.md](./VERCEL_DEPLOYMENT.md)

## 📦 What Gets Deployed

✅ FastAPI Backend (api/main.py)  
✅ ML Models (best_model.pkl, scaler.pkl)  
✅ Web Frontend (static/)  
✅ All dependencies from requirements.txt  

## 🎯 After Deployment

Your app will be live at: `https://your-project.vercel.app`

Test these endpoints:
- `/` - Frontend interface
- `/health` - API health check
- `/docs` - API documentation
- `/predict` - Make predictions

## ⚙️ Configuration Files

- `vercel.json` - Deployment config
- `index.py` - Serverless entry point
- `runtime.txt` - Python version (3.11)
- `.vercelignore` - Excluded files

## 🐛 Troubleshooting

**Build fails?**
- Check Vercel logs in dashboard
- Verify all dependencies in requirements.txt

**Models not loading?**
- Models are ~56KB total (under 100MB limit ✅)
- Check paths in api/main.py

**API timeout?**
- Free tier: 10s limit
- Optimize model loading with caching

## 💡 Pro Tips

1. **Custom Domain**: Settings → Domains
2. **Environment Variables**: Settings → Environment Variables  
3. **Analytics**: Check deployment logs and analytics
4. **Auto Deploy**: Push to GitHub = Auto deploy!

## 🆘 Need Help?

- Read: [VERCEL_DEPLOYMENT.md](./VERCEL_DEPLOYMENT.md)
- Vercel Docs: https://vercel.com/docs
- Issues: Check deployment logs

---

**Ready to deploy?** → https://vercel.com/new 🚀
