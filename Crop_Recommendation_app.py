# -*- coding: utf-8 -*-
"""
Created on Fri May  1 22:53:39 2026

@author: asusl
"""

import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Crop Recommendation System", layout="centered")

st.title("🌾 Crop Recommendation System")
st.write("Enter all soil and weather values to get an accurate crop recommendation.")

# Load saved models
@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    minmax = pickle.load(open("minmaxscaler.pkl", "rb"))
    standard = pickle.load(open("standscaler.pkl", "rb"))
    return model, minmax, standard

try:
    model, minmax, standard = load_models()
except:
    st.error("Required files (model.pkl, minmaxscaler.pkl, standscaler.pkl) not found.")
    st.stop()

# User Inputs
st.subheader("Enter Input Values")

N = st.number_input("Nitrogen (N)", min_value=0.0, value=None, placeholder="Enter Nitrogen value")
P = st.number_input("Phosphorus (P)", min_value=0.0, value=None, placeholder="Enter Phosphorus value")
K = st.number_input("Potassium (K)", min_value=0.0, value=None, placeholder="Enter Potassium value")
temperature = st.number_input("Temperature (°C)", value=None, placeholder="Enter Temperature")
humidity = st.number_input("Humidity (%)", value=None, placeholder="Enter Humidity")
ph = st.number_input("pH Value", value=None, placeholder="Enter pH value")
rainfall = st.number_input("Rainfall (mm)", value=None, placeholder="Enter Rainfall")

# Check if all inputs are filled
def all_inputs_filled(values):
    return all(v is not None for v in values)

# Prediction
if st.button("Predict Crop"):
    inputs = [N, P, K, temperature, humidity, ph, rainfall]
    
    if not all_inputs_filled(inputs):
        st.warning("⚠️ Please fill all input fields before prediction.")
    else:
        input_data = np.array([inputs])
        
        # Apply scalers
        input_data = minmax.transform(input_data)
        input_data = standard.transform(input_data)

        prediction = model.predict(input_data)
        st.success(f"🌱 Recommended Crop: {prediction[0]}")

# Footer
st.write("---")
st.write("Developed using Streamlit & Machine Learning")
st.write("---")
st.write("Developed using Streamlit & Machine Learning")
