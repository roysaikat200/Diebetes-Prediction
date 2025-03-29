import numpy as np
import pickle
import streamlit as st

# Load the trained model
try:
    with open('trained_model.sav', 'rb') as f:
        loaded_model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found! Ensure 'trained_model.sav' is in the directory.")
    st.stop()

# Load the scaler (if you saved it during training)
try:
    with open('scaler.sav', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Scaler file not found! Ensure 'scaler.sav' is in the directory.")
    st.stop()

def predict_diabetes(input_data):
    """Predicts diabetes based on user input."""
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)  # Ensure it's 2D

    # Apply the pre-trained scaler
    standardized_data = scaler.transform(input_data_as_numpy_array)

    # Predict
    prediction = loaded_model.predict(standardized_data)

    return 1 if prediction[0] == 1 else 0

def main():
    st.title("Diabetes Prediction App")
    st.write("Enter patient details to predict the likelihood of diabetes:")

    # User inputs
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=5)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=199, value=166)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=122, value=72)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=99, value=19)
    insulin = st.number_input("Insulin", min_value=0, max_value=846, value=175)
    bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=25.8, format="%.1f")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.42, value=0.587, format="%.3f")
    age = st.number_input("Age", min_value=0, max_value=120, value=51)

    # Prepare input data
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age)

    # Predict
    if st.button("Predict"):
        prediction = predict_diabetes(input_data)
        if prediction == 0:
            st.success(f"Prediction: THE PERSON IS NOT DIABETIC",  icon="✅")
        else:
            st.error(f"Prediction: THE PERSON IS DIABETIC", icon="🚨")

if __name__ == "__main__":
    main()
