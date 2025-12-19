import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="HR Attrition Prediction System",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------------------------
# Load Model, Scaler, Feature Names
# ----------------------------------------------------
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

# ----------------------------------------------------
# HR Recommendation Engine
# ----------------------------------------------------
def hr_recommendation(data, probability):
    rec = []

    if data['satisfaction_level'].values[0] < 0.4:
        rec.append("Improve employee engagement and satisfaction initiatives.")

    if data['average_montly_hours'].values[0] > 250:
        rec.append("Reduce workload to prevent burnout.")

    if data['promotion_last_5years'].values[0] == 0:
        rec.append("Provide career growth or promotion opportunities.")

    if 'salary' in data.columns and data['salary'].values[0] == 0:
        rec.append("Review compensation or offer incentives.")

    if probability < 0.3:
        return ["Employee is stable. Maintain current engagement strategies."]

    return rec

# ----------------------------------------------------
# App Title
# ----------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>HR Attrition Prediction & Decision Support System</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

# ----------------------------------------------------
# Sidebar Inputs
# ----------------------------------------------------
st.sidebar.header("Employee Details")

satisfaction = st.sidebar.slider("Satisfaction Level", 0.0, 1.0, 0.5)
evaluation = st.sidebar.slider("Last Evaluation Score", 0.0, 1.0, 0.7)
projects = st.sidebar.slider("Number of Projects", 2, 7, 4)
monthly_hours = st.sidebar.slider("Average Monthly Hours", 90, 310, 160)
time_spend = st.sidebar.slider("Years at Company", 1, 10, 3)
work_accident = st.sidebar.selectbox("Work Accident", [0, 1])
promotion = st.sidebar.selectbox("Promotion in Last 5 Years", [0, 1])

salary_label = st.sidebar.selectbox("Salary Level", ["Low", "Medium", "High"])
department = st.sidebar.selectbox(
    "Department",
    ["sales", "technical", "support", "IT", "RandD",
     "accounting", "hr", "management", "marketing", "product_mng"]
)

# ----------------------------------------------------
# Encode Salary
# ----------------------------------------------------
salary_map = {"Low": 0, "Medium": 1, "High": 2}
salary_encoded = salary_map[salary_label]

# ----------------------------------------------------
# Build Input with EXACT Feature Match
# ----------------------------------------------------
input_data = pd.DataFrame(0, index=[0], columns=feature_names)

input_data['satisfaction_level'] = satisfaction
input_data['last_evaluation'] = evaluation
input_data['number_project'] = projects
input_data['average_montly_hours'] = monthly_hours
input_data['time_spend_company'] = time_spend
input_data['Work_accident'] = work_accident
input_data['promotion_last_5years'] = promotion

if 'salary' in input_data.columns:
    input_data['salary'] = salary_encoded

dept_col = f"Department_{department}"
if dept_col in input_data.columns:
    input_data[dept_col] = 1

# ----------------------------------------------------
# Scale & Predict
# ----------------------------------------------------
input_scaled = scaler.transform(input_data.values)

prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# ----------------------------------------------------
# KPI Section
# ----------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Attrition Prediction", "YES" if prediction == 1 else "NO")

with col2:
    st.metric("Attrition Probability", f"{probability:.2%}")

with col3:
    if probability > 0.6:
        risk = "HIGH RISK"
    elif probability > 0.3:
        risk = "MEDIUM RISK"
    else:
        risk = "LOW RISK"

    st.metric("Risk Level", risk)

st.markdown("---")

# ----------------------------------------------------
# HR Recommendations (Solution Layer)
# ----------------------------------------------------
st.subheader("HR Intervention Recommendations")

recommendations = hr_recommendation(input_data, probability)

for r in recommendations:
    st.write("•", r)

# ----------------------------------------------------
# Feature Importance
# ----------------------------------------------------
if hasattr(model, "feature_importances_"):
    st.markdown("---")
    st.subheader("Model Feature Importance")

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(fi_df.set_index("Feature"))
