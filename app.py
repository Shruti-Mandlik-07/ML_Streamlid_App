
import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Height & Weight Prediction",
    page_icon="⚖️"
)

st.title("⚖️ Height & Weight Prediction")
st.write("Enter the details below to predict weight.")

# Load files
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("sex_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

st.success("Model loaded successfully!")

sex = st.selectbox(
    "Select Sex",
    encoder.classes_.tolist()
)

height = st.number_input(
    "Height (cm)",
    min_value=50.0,
    max_value=250.0,
    value=170.0,
    step=0.1
)

if st.button("Predict Weight"):

    sex_encoded = encoder.transform([sex])[0]

    data = np.array([
        [float(height), sex_encoded]
    ])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    st.success(
        f"Predicted Weight: {prediction:.2f} kg"
    )