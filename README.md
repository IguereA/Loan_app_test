# Mobile Loan Default Prediction

A machine learning project to predict loan default risk for a Nigerian mobile lending platform.

## Problem Statement

Mobile lending apps in Nigeria process thousands of loan applications daily. Manually assessing each application is time-consuming and prone to bias. This ML model automates the risk assessment process by predicting which applicants are likely to default on their loans.

##  Project Objectives

- Build a classification model to predict loan defaults
- Deploy the model as a web application
- Make the model accessible to non-technical users
- Help lenders make faster, data-driven decisions

## Dataset Description

The dataset contains **5000 loan applications** with the following features:

- **Demographic**: Age, state, education level
- **Financial**: Monthly income, credit score, bank account
- **Loan Details**: Loan amount, duration, previous loans
- **Risk Indicators**: Previous defaults, dependents, employment type

**Target Variable**: `loan_default` (0 = No Default, 1 = Default)

See `columns_description.txt` for detailed feature explanations.

## Technologies Used

- **PyCaret**: AutoML library for model training
- **Streamlit**: Web app framework
- **Pandas & NumPy**: Data manipulation
- **Git & GitHub**: Version control
- **Streamlit Cloud**: Deployment platform

## Setup


### Step 1: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

##  How to Train the Model

Run the training script:
```bash
python train.py
```

This will:
1. Load the dataset
2. Set up PyCaret environment
3. Compare multiple ML models
4. Select and tune the best model
5. Save the model as `loan_default_model.pkl`

**Expected Training Time**: 2-5 minutes

## How to Run the Streamlit App

### Locally
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Test the App

1. Enter applicant details in the sidebar
2. Click "Predict Default Risk"
3. View the prediction and risk assessment

##  Git & GitHub Workflow

### Initialize Git Repository
```bash
git init
git add .
git commit -m "Initial commit: Loan default prediction project"
```

### Create GitHub Repository

1. Go to [github.com](https://github.com)
2. Click "New Repository"
3. Name it: `loan-default-prediction`
4. Don't initialize with README (we already have one)

### Push to GitHub
```bash
git remote add origin https://github.com/your-username/loan-default-prediction.git
git branch -M main
git push -u origin main
```

## Deployment to Streamlit Cloud

### Step 1: Prepare for Deployment

Make sure you have:
- `requirements.txt` file
- `app.py` file
- Trained model (`loan_default_model.pkl`)
- All files pushed to GitHub

### Step 2: Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your GitHub repository
4. Set main file: `app.py`
5. Click "Deploy"

**Deployment Time**: 5-10 minutes

### Step 3: Share Your App

Once deployed, you'll get a URL like:
```
https://your-username-loan-default-prediction.streamlit.app
```







##  Troubleshooting

**Model file not found**:
- Make sure you ran `train.py` first
- Check that `loan_default_model.pkl` exists in the project folder

**Streamlit app crashes**:
- Check your Python version (use Python 3.8-3.10)
- Reinstall requirements: `pip install -r requirements.txt --upgrade`

**Deployment fails**:
- Ensure all files are pushed to GitHub
- Check that `requirements.txt` has correct versions
- Make sure model file size is under 100MB

## Resources

- [PyCaret Documentation](https://pycaret.gitbook.io/docs/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Git Tutorial](https://www.atlassian.com/git/tutorials)

