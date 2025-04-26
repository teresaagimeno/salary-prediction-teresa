
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the best model
with open('salary_predictor_best.pkl', 'rb') as f:
    model = pickle.load(f)

# ---------------------------
# ✨ Streamlit App Layout ✨
# ---------------------------

st.title("Salary Prediction App 💼📈")
st.subheader("Predict your estimated yearly salary based on your background!")

st.write("""
Please fill out the form below and click **Predict Salary** to see your estimated compensation.
""")

# ---------------------------
# ✨ User Inputs
# ---------------------------

country = st.selectbox(
    "🌎 In which country do you currently reside?",
    ['United States of America', 'India', 'Germany', 'United Kingdom of Great Britain and Northern Ireland', 
     'Canada', 'France', 'Australia', 'Brazil', 'Spain', 'Other']
)

education = st.selectbox(
    "🎓 What is the highest level of education you have completed?",
    ['Bachelor’s degree', 'Master’s degree', 'Doctoral degree', 'Some college/university study', 'High school diploma', 'Other']
)

experience = st.selectbox(
    "⌛ For how many years have you been coding?",
    ['< 1 years', '1-3 years', '3-5 years', '5-10 years', '10-20 years', '20+ years']
)

age = st.selectbox(
    "🎂 What is your age group?",
    ['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-69', '70+']
)

role = st.selectbox(
    "👩‍💻 Select your current job title:",
    ['Data Scientist', 'Software Engineer', 'Data Analyst', 'Research Scientist', 'Consultant', 'Developer Advocate', 'Other']
)

company_size = st.selectbox(
    "🏢 What is the size of your company?",
    ['0-49 employees', '50-249 employees', '250-999 employees', '1000-9,999 employees', '10,000 or more employees']
)

# ---------------------------
# ✨ Prediction
# ---------------------------

input_dict = {
    'In which country do you currently reside?': country,
    'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?': education,
    'For how many years have you been writing code and/or programming?': experience,
    'What is your age (# years)?': age,
    'Select the title most similar to your current role (or most recent title if retired):': role,
    'What is the size of the company where you are employed?': company_size
}

input_df = pd.DataFrame([input_dict])
input_encoded = pd.get_dummies(input_df)

model_features = model.feature_names_in_
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

if st.button("🔮 Predict Salary"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"💰 Estimated Salary: ${prediction:,.2f} USD")

# ---------------------------
# ✨ Footer
# ---------------------------

st.markdown("""
---
App created by **Teresa Gimeno**.
""")
    