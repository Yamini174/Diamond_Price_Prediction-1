
import streamlit as st
import pickle
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="centered"
)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Application title
st.title("💎 Diamond Price Prediction")
st.write("Predict the price of a diamond using Machine Learning (KNN).")

st.divider()

# User inputs
st.subheader("Enter Diamond Details")

carat = st.number_input(
    "Carat Weight",
    min_value=0.1,
    max_value=5.0,
    value=0.5,
    step=0.1
)

cut = st.selectbox(
    "Cut Quality",
    ["Ideal", "Premium", "Very Good", "Good", "Fair"]
)

color = st.selectbox(
    "Diamond Color",
    ["D", "E", "F", "G", "H", "I", "J"]
)

clarity = st.selectbox(
    "Diamond Clarity",
    ["IF", "VVS1", "VVS2", "VS1", "VS2",
     "SI1", "SI2", "I1"]
)

depth = st.number_input(
    "Depth Percentage",
    min_value=40.0,
    max_value=80.0,
    value=61.5
)

table = st.number_input(
    "Table Percentage",
    min_value=40.0,
    max_value=90.0,
    value=57.0
)

x = st.number_input(
    "Length (mm)",
    min_value=0.0,
    max_value=15.0,
    value=5.0
)

y = st.number_input(
    "Width (mm)",
    min_value=0.0,
    max_value=15.0,
    value=5.0
)

z = st.number_input(
    "Depth (mm)",
    min_value=0.0,
    max_value=10.0,
    value=3.0
)

# Prediction button
if st.button("Predict Diamond Price", type="primary"):

    input_data = pd.DataFrame([{
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z
    }])

    prediction = model.predict(input_data)

    st.success(
        f"Estimated Diamond Price: ${prediction[0]:,.2f}"
    )

    st.info("Prediction generated using the trained KNN model.")
