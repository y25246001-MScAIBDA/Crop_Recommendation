# -*- coding: utf-8 -*-
"""
Created on Fri May  1 22:53:39 2026

@author: asusl
"""
import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="AgroPredict - Crop Recommendation System", page_icon="🌱", layout="centered")

st.title("🌱 Crop Recommendation System")
st.write("Enter all soil and weather values to get an accurate crop recommendation.")

# Load saved models
@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    minmax = pickle.load(open("minmaxscaler.pkl", "rb"))
    standard = pickle.load(open("standscaler.pkl", "rb"))
    return model, minmax, standard
# Label mapping (EDIT THIS according to your dataset)
crop_dict = {
    0: "Rice",
    1: "Maize",
    2: "Chickpea",
    3: "Kidney Beans",
    4: "Pigeon Peas",
    5: "Moth Beans",
    6: "Mung Bean",
    7: "Black Gram",
    8: "Lentil",
    9: "Pomegranate",
    10: "Banana",
    11: "Mango",
    12: "Grapes",
    13: "Watermelon",
    14: "Muskmelon",
    15: "Apple",
    16: "Orange",
    17: "Papaya",
    18: "Coconut",
    19: "Cotton",
    20: "Jute",
    21: "Coffee"
}

try:
    model, minmax, standard = load_models()
except:
    st.error("Required files (model.pkl, minmaxscaler.pkl, standscaler.pkl) not found.")
    st.stop()

# User Inputs
st.subheader("Enter Input Values")

N = st.number_input("Nitrogen (N)", min_value=0.0, value=0.0, step=1.0)
P = st.number_input("Phosphorus (P)", min_value=0.0, value=0.0, step=1.0)
K = st.number_input("Potassium (K)", min_value=0.0, value=0.0, step=1.0)
temperature = st.number_input("Temperature (°C)", value=0.0, step=0.1)
humidity = st.number_input("Humidity (%)", value=0.0, step=0.1)
ph = st.number_input("pH Value", value=0.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", value=0.0, step=0.1)

# Check if all inputs are valid (not default 0)
def valid_inputs(values):
    return all(v != 0.0 for v in values)

# Prediction
if st.button("Predict Crop"):
    inputs = [N, P, K, temperature, humidity, ph, rainfall]

    if not valid_inputs(inputs):
        st.warning("⚠️ Please enter valid (non-zero) values for all fields.")
    else:
        input_data = np.array([inputs])

        # Apply scalers
        input_data = minmax.transform(input_data)
        input_data = standard.transform(input_data)

        prediction = model.predict(input_data)[0]

        # Convert number to crop name
        crop_name = crop_dict.get(prediction, "Unknown Crop")

        st.success(f"🌱 Recommended Crop: {crop_name}")

# Footer
st.write("---")
st.write("Developed using Streamlit & Machine Learning")
