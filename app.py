import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('models/logistic_regression_model.pkl')
scaler = joblib.load('src/scaler.pkl')

# Title of the app
st.title("Heart Disease Prediction")

# Input fields for user data
st.header("Enter Patient Details:")
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=50, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (0=No, 1=Yes)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [1 if sex == "Male" else 0],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make predictions
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Display results
    st.subheader("Prediction:")
    if prediction == 1:
        st.error("The patient is likely to have heart disease.")
    else:
        st.success("The patient is unlikely to have heart disease.")
    st.write(f"Probability of Heart Disease: {probability:.2%}")