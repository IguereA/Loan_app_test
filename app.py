import streamlit as st
import pandas as pd
import numpy as np
import joblib
# from pycaret.classification import load_model, predict_model

# --- PAGE CONFIG ---
st.set_page_config(page_title="Loan Risk AI", page_icon="🏦", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Ensure this filename matches exactly what you saved in your notebook
    return joblib.load('models/xgb_loan_model.joblib')

model_pipeline = load_model()

# --- MODEL COLUMN ORDER ---
# This must match X_train.columns.tolist() exactly
MODEL_COLUMNS = [
    'age', 'monthly_income', 'loan_amount', 'loan_duration_months',
    'previous_loans', 'previous_defaults', 'account_age_months',
    'num_dependents', 'education_level', 'has_bank_account',
    'credit_score', 'debt_to_income_ratio', 'estimated_monthly_payment',
    'payment_to_income_ratio', 'default_history_ratio', 'income_per_dependent',
    'employment_type_Freelancer', 'employment_type_Salary_Earner', 'employment_type_Self_Employed',
    'residential_status_Own_House', 'residential_status_Renting',
    'state_Enugu', 'state_Ibadan', 'state_Kano', 'state_Lagos', 'state_Port_Harcourt',
    'credit_score_band_Poor', 'credit_score_band_Good', 'credit_score_band_Excellent'
]

# --- UI SETUP ---
st.title("🇳🇬 Banking Credit Risk Assessment Tool")
st.markdown("This AI model predicts the likelihood of loan default based on applicant profiles.")
st.divider()

# --- INPUT SECTION ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Personal Details")
    age = st.number_input("Age", 18, 100, 30)
    edu = st.selectbox("Education Level", ['Secondary', 'OND', 'HND', 'BSc', 'MSc'])
    emp = st.selectbox("Employment Type", ['Business_Owner', 'Salary_Earner', 'Self_Employed', 'Freelancer'])
    res = st.selectbox("Residential Status", ['Living_with_Parents', 'Renting', 'Own_House'])
    state = st.selectbox("State", ['Abuja', 'Lagos', 'Port_Harcourt', 'Ibadan', 'Kano', 'Enugu'])

with col2:
    st.subheader("💰 Financials")
    income = st.number_input("Monthly Income (₦)", min_value=1000.0, value=150000.0)
    loan_amt = st.number_input("Loan Amount (₦)", min_value=1000.0, value=500000.0)
    duration = st.number_input("Loan Duration (Months)", 1, 120, 12)
    deps = st.number_input("Number of Dependents", 0, 20, 0)
    bank_acc = st.radio("Has Bank Account?", ["Yes", "No"])

with col3:
    st.subheader("📊 Credit History")
    score = st.slider("Credit Score", 300, 850, 650)
    prev_loans = st.number_input("Previous Loans", 0, 50, 0)
    prev_def = st.number_input("Previous Defaults", 0, 50, 0)
    acc_age = st.number_input("Account Age (Months)", 0, 600, 24)

# --- FEATURE ENGINEERING (Replicating your Notebook Logic) ---
# 1. Map Education (Ordinal)
education_map = {'Secondary': 1, 'OND': 2, 'HND': 3, 'BSc': 4, 'MSc': 5}
edu_val = education_map[edu]

# 2. Raw Ratios
dti = loan_amt / income
est_monthly_payment = loan_amt / duration
pay_to_inc_ratio = est_monthly_payment / income
def_history_ratio = prev_def / (prev_loans + 1)
inc_per_dep = income / (deps + 1)
bank_acc_val = 1 if bank_acc == "Yes" else 0

# 3. Credit Score Banding
if score <= 500: band = "Very_Poor"
elif score <= 650: band = "Poor"
elif score <= 750: band = "Good"
else: band = "Excellent"

# --- DATA ASSEMBLY ---
# Initialize all columns to 0
input_dict = {col: 0 for col in MODEL_COLUMNS}

# Update Numerical/Ordinal Values
input_dict.update({
    'age': age, 'monthly_income': income, 'loan_amount': loan_amt,
    'loan_duration_months': duration, 'previous_loans': prev_loans,
    'previous_defaults': prev_def, 'account_age_months': acc_age,
    'num_dependents': deps, 'education_level': edu_val,
    'has_bank_account': bank_acc_val, 'credit_score': score,
    'debt_to_income_ratio': dti, 'estimated_monthly_payment': est_monthly_payment,
    'payment_to_income_ratio': pay_to_inc_ratio, 'default_history_ratio': def_history_ratio,
    'income_per_dependent': inc_per_dep
})

# Handle One-Hot Encoding (N-1 logic)
# If user picks the 'dropped' column (e.g., Business_Owner), all encoded cols remain 0
if f"employment_type_{emp}" in input_dict: input_dict[f"employment_type_{emp}"] = 1
if f"residential_status_{res}" in input_dict: input_dict[f"residential_status_{res}"] = 1
if f"state_{state}" in input_dict: input_dict[f"state_{state}"] = 1
if f"credit_score_band_{band}" in input_dict: input_dict[f"credit_score_band_{band}"] = 1

# Convert to DataFrame in exact order
final_df = pd.DataFrame([input_dict])[MODEL_COLUMNS]

# --- PREDICTION ---
st.divider()
if st.button("🔍 Run Risk Analysis", use_container_width=True):
    prediction = model_pipeline.predict(final_df)[0]
    probability = model_pipeline.predict_proba(final_df)[0][1]    
    safe_probability = float(np.clip(probability, 0.0, 1.0))
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        if prediction == 1:
            st.error("### Result: HIGH RISK")
            st.write("Model recommends **Rejection** due to high default probability.")
            # Show red-ish progress for high risk
            st.progress(safe_probability)
        else:
            st.success("### Result: LOW RISK")
            st.write("Model recommends **Approval**.")
            # Show green-ish progress for low risk
            st.progress(safe_probability)
            st.balloons()
            
    with col_res2:
        st.metric("Probability of Default", f"{probability:.2%}")
        st.progress(safe_probability)

st.sidebar.info("Model Info: Tuned XGBoost | Recall: 94.27%")
