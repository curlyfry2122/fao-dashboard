# Quick Deploy Guide - FAO Dashboard

## 🚀 Deploy in 3 Steps

### Option 1: Streamlit Cloud (Easiest)
1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Deploy FAO Dashboard"
   git push origin main
   ```

2. **Deploy**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app" → Select your repo → Choose `app.py`
   - Click "Deploy"

3. **Done!** Your dashboard is live at `https://your-app-name.streamlit.app`

---

### Option 2: Heroku
1. **Setup**
   ```bash
   # Install Heroku CLI first: https://devcenter.heroku.com/articles/heroku-cli
   heroku login
   heroku create your-app-name
   ```

2. **Deploy**
   ```bash
   git push heroku main
   ```

3. **Done!** Dashboard at `https://your-app-name.herokuapp.com`

---

### Option 3: Local Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py

# Open: http://localhost:8501
```

## 🔧 If Something Goes Wrong

**Common fixes:**
- Make sure `requirements.txt` has all dependencies
- Check that `app.py` exists in root folder
- Wait 2-3 minutes for first deployment
- Check platform logs for error messages

**Need help?** Check `DEPLOYMENT_README.md` for detailed instructions.