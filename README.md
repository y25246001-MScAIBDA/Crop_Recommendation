# Crop_Recommendation
This project is a Machine Learning-based Crop Recommendation System that suggests the most suitable crop to grow based on soil nutrients and environmental conditions.It helps farmers and agricultural enthusiasts make data-driven decisions for better yield and productivity.

The model takes the following input parameters:

Nitrogen (N)
Phosphorus (P)
Potassium (K)
Temperature (°C)
Humidity (%)
pH value
Rainfall (mm)

Using these inputs, the system predicts the best crop using a trained ML model integrated into an interactive Streamlit web application.

🚀 Features

🌱 Predicts the most suitable crop based on input conditions
📊 Uses trained Machine Learning model (model.pkl)
⚙️ Includes preprocessing using:
MinMaxScaler (minmaxscaler.pkl)
StandardScaler (standscaler.pkl)
💻 User-friendly interface built with Streamlit
⚡ Fast and real-time predictions

🛠️ Tech Stack

Python
Streamlit
Scikit-learn
NumPy
Pickle
📂 Project Structure
├── crop_recommendation_app.py
├── model.pkl
├── minmaxscaler.pkl
├── standscaler.pkl
├── crop_data.csv
└── README.md

▶️ How to Run

pip install -r requirements.txt
streamlit run crop_recommendation_app.py

🎯 Objective

To assist in smart agriculture by recommending the right crop based on scientific data analysis, improving productivity and reducing risk.
