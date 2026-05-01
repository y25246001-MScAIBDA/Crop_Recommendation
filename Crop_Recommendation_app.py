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
st.write("Enter soil and weather conditions to get the best crop recommendation.")

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
    st.error("Model or scaler files not found. Please make sure model.pkl, minmaxscaler.pkl, and standscaler.pkl are in the same folder.")
    st.stop()

# User Inputs
st.subheader("Enter Input Values")

N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorus (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH Value")
rainfall = st.number_input("Rainfall (mm)")

# Prediction
if st.button("Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Apply scalers (same order as training)
    input_data = minmax.transform(input_data)
    input_data = standard.transform(input_data)

    prediction = model.predict(input_data)

    st.success(f"🌱 Recommended Crop: {prediction[0]}")

# Footer
st.write("---")
st.write("Developed using Streamlit & Machine Learning")
