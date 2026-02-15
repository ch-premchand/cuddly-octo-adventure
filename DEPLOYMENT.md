# Machine Learning Classification Models - Deployment Guide

## Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub** (already done)
   - Repository: https://github.com/ch-premchand/cuddly-octo-adventure

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Sign in with your GitHub account
   - Click "New app"
   - Select repository: `ch-premchand/cuddly-octo-adventure`
   - Branch: `main` or `copilot/implement-classification-models`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Access Your App**
   - Your app will be available at: `https://[your-app-name].streamlit.app`

### Option 2: Local Deployment

```bash
# Clone the repository
git clone https://github.com/ch-premchand/cuddly-octo-adventure.git
cd cuddly-octo-adventure

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Access at: http://localhost:8501

### Option 3: Heroku

1. Create a `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.headless=true
```

2. Create a `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Option 4: Docker

1. Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

2. Build and run:
```bash
docker build -t ml-classifier .
docker run -p 8501:8501 ml-classifier
```

## Testing the Deployed App

1. Upload the `sample_dataset.csv` file
2. Select "target" as the target column
3. Click "Train Models"
4. View the results and metrics

## Requirements for Submission

✅ Dataset: 1000 rows, 17 features (exceeds minimum requirements)
✅ 6 Models: All implemented and tested
✅ Metrics: All 6 metrics computed
✅ Streamlit App: Fully functional with all features
✅ GitHub Repo: Code, requirements.txt, README
✅ Documentation: Comprehensive README and deployment guide

## Live App Example

After deployment, your app URL will be something like:
- Streamlit Cloud: `https://cuddly-octo-adventure.streamlit.app`
- Heroku: `https://your-app-name.herokuapp.com`
- Local: `http://localhost:8501`

## Screenshots for Documentation

Take screenshots of:
1. App home page
2. Dataset upload interface
3. Model selection options
4. Training results with metrics
5. Confusion matrix
6. Model comparison charts

## Notes

- The app uses a free-tier deployment (no credit card required for Streamlit Cloud)
- Sample dataset is included in the repository
- All dependencies are specified in requirements.txt
- App is mobile-responsive and works on all devices
