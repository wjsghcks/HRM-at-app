import streamlit as st
import pandas as pd
import numpy as np # Needed for logit-probability conversion

# List of variables included in the final model (Model 9)
final_variables = [
    'Age', 'BusinessTravel', 'DistanceFromHome', 'EnvironmentSatisfaction',
    'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'NumCompaniesWorked',
    'OverTime', 'RelationshipSatisfaction', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager'
]

# Unstandardized coefficients (B) for each variable
coefficients = {
    '(Constant)': 0.713, # Constant B value
    'Age': -0.004,
    'BusinessTravel': 0.082, # BusinessTravel B value (numeric coding)
    'DistanceFromHome': 0.004,
    'EnvironmentSatisfaction': -0.040,
    'JobInvolvement': -0.065,
    'JobLevel': -0.024,
    'JobSatisfaction': -0.037,
    'NumCompaniesWorked': 0.017,
    'OverTime': 0.204, # OverTime_Num B value
    'RelationshipSatisfaction': -0.022,
    'StockOptionLevel': -0.055,
    'TotalWorkingYears': -0.004,
    'TrainingTimesLastYear': -0.012,
    'WorkLifeBalance': -0.025,
    'YearsAtCompany': 0.006,
    'YearsInCurrentRole': -0.010,
    'YearsSinceLastPromotion': 0.012,
    'YearsWithCurrManager': -0.010
}

# --- 2. App Interface Configuration ---
st.title("🧑‍💼 Employee Turnover Prediction")
st.write("Enter employee details to predict the probability of turnover.")

# Dictionary to store user inputs
inputs = {}

st.sidebar.header("Enter Employee Details")

# Input widgets for each variable
inputs['Age'] = st.sidebar.slider("Age", 18, 60, 30)

# BusinessTravel
# Translated options: 1=Non-Travel, 2=Travel Rarely, 3=Travel Frequently
travel_options = {1: 'Non-Travel', 2: 'Travel Rarely', 3: 'Travel Frequently'}
selected_travel_text = st.sidebar.selectbox(
    "Business Travel Frequency",
    options=list(travel_options.values()),
    index=1 # Default: Travel Rarely
)
# Convert selected text back to number
inputs['BusinessTravel'] = [k for k, v in travel_options.items() if v == selected_travel_text][0]


inputs['DistanceFromHome'] = st.sidebar.slider("Distance From Home (km)", 1, 30, 5)

inputs['EnvironmentSatisfaction'] = st.sidebar.select_slider(
    "Environment Satisfaction (1: Low ~ 4: High)", options=[1, 2, 3, 4], value=3)

inputs['JobInvolvement'] = st.sidebar.select_slider(
    "Job Involvement (1: Low ~ 4: High)", options=[1, 2, 3, 4], value=3)

inputs['JobLevel'] = st.sidebar.select_slider(
    "Job Level (1 ~ 5)", options=[1, 2, 3, 4, 5], value=2)

inputs['JobSatisfaction'] = st.sidebar.select_slider(
    "Job Satisfaction (1: Low ~ 4: High)", options=[1, 2, 3, 4], value=3)

inputs['NumCompaniesWorked'] = st.sidebar.slider("Num. of Companies Worked", 0, 10, 2)

# OverTime Handling (0=No, 1=Yes)
overtime_option = st.sidebar.radio("Overtime", ('No', 'Yes'), index=0)
inputs['OverTime'] = 1 if overtime_option == 'Yes' else 0

inputs['RelationshipSatisfaction'] = st.sidebar.select_slider(
    "Relationship Satisfaction (1: Low ~ 4: High)", options=[1, 2, 3, 4], value=3)

inputs['StockOptionLevel'] = st.sidebar.select_slider(
    "Stock Option Level (0 ~ 3)", options=[0, 1, 2, 3], value=0)

inputs['TotalWorkingYears'] = st.sidebar.slider("Total Working Years", 0, 40, 5)

inputs['TrainingTimesLastYear'] = st.sidebar.slider("Training Times Last Year", 0, 6, 2)

inputs['WorkLifeBalance'] = st.sidebar.select_slider(
    "Work-Life Balance (1: Low ~ 4: High)", options=[1, 2, 3, 4], value=3)

inputs['YearsAtCompany'] = st.sidebar.slider("Years at Company", 0, 40, 3)

inputs['YearsInCurrentRole'] = st.sidebar.slider("Years in Current Role", 0, 20, 2)

inputs['YearsSinceLastPromotion'] = st.sidebar.slider("Years Since Last Promotion", 0, 20, 1)

inputs['YearsWithCurrManager'] = st.sidebar.slider("Years with Current Manager", 0, 20, 2)


# --- 3. Calculate Turnover Probability ---
logit = coefficients['(Constant)']
for var in final_variables:
    if var in inputs and var != '(Constant)': # Constant is already added
        logit += coefficients[var] * inputs[var]

# Convert logit to probability (Sigmoid function)
probability = 1 / (1 + np.exp(-logit))

# --- 4. Display Results ---
st.subheader("📊 Prediction Result")
probability_percent = probability * 100
st.metric(label="Turnover Probability", value=f"{probability_percent:.2f}%")

# Risk level indication based on probability
if probability_percent >= 50:
    st.error("🚨 High Turnover Risk")
elif probability_percent >= 30:
    st.warning("⚠️ Moderate Turnover Risk")
else:
    st.success("✅ Low Turnover Risk")
