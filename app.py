import json
import os

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# Diamond Price Prediction System - Streamlit web app
# Loads the trained scikit-learn pipeline (pre-processing + model)
# and predicts the price of a diamond from its specifications.
# ---------------------------------------------------------------

MODEL_FILE = "diamond_price_model_full_dataset.pkl"
INFO_FILE = "model_info.json"

st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_info():
    if os.path.exists(INFO_FILE):
        with open(INFO_FILE) as f:
            return json.load(f)
    return {}


model = load_model()
info = load_info()

st.title("💎 Diamond Price Predictor")
st.write(
    "Enter the specifications of a diamond and the model will predict its price in US dollars. "
    "The model was selected by comparing 5 regression algorithms with 10-fold cross validation "
    "on the full Diamonds dataset (53,940 diamonds)."
)

st.subheader("Diamond specifications")

col1, col2 = st.columns(2)
with col1:
    carat = st.number_input("Carat (weight)", min_value=0.2, max_value=5.5, value=0.70, step=0.01)
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=4)
    color = st.selectbox("Color (D = best, J = worst)", ["D", "E", "F", "G", "H", "I", "J"], index=3)
    clarity = st.selectbox("Clarity (IF = best, I1 = worst)",
                           ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], index=2)
with col2:
    depth = st.number_input("Depth % (total depth percentage)", min_value=40.0, max_value=80.0, value=61.8, step=0.1)
    table = st.number_input("Table % (width of top relative to widest point)", min_value=40.0, max_value=100.0, value=57.0, step=0.1)
    x = st.number_input("Length x (mm)", min_value=2.0, max_value=11.0, value=5.70, step=0.01)
    y = st.number_input("Width y (mm)", min_value=2.0, max_value=60.0, value=5.72, step=0.01)
    z = st.number_input("Depth z (mm)", min_value=1.0, max_value=32.0, value=3.53, step=0.01)

if st.button("Predict Price", type="primary"):
    features = pd.DataFrame([{
        "carat": carat, "depth": depth, "table": table, "x": x, "y": y, "z": z,
        "cut": cut, "color": color, "clarity": clarity,
    }])
    predicted_price = float(model.predict(features)[0])
    st.success(f"Predicted price: **${predicted_price:,.2f}**")

    if info:
        st.caption(
            f"Model: {info.get('best_algorithm', 'N/A')} - typical error (10-fold CV RMSE) "
            f"about ${info.get('mean_rmse', 0):,.0f}."
        )

with st.expander("About this model"):
    if info:
        st.write(f"**Algorithm:** {info.get('best_algorithm')}")
        st.write(f"**Mean RMSE (10-fold CV):** ${info.get('mean_rmse', 0):,.2f} "
                 f"(+/- ${info.get('std_rmse', 0):,.2f})")
        st.write(f"**Trained on:** {info.get('n_samples', 0):,} diamonds")
    st.write("Course: AIC354 Machine Learning Fundamentals - COMSATS University Islamabad, Lahore Campus.")
