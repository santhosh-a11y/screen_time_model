import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="Screen Time Guesser", layout="centered")

st.title("📱 Screen Time Guesser")
st.write("Enter your age and the model will estimate your daily screen time.")

# Load the trained model
@st.cache_resource
def load_model():
    with open("screen_time_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# User input
age = st.number_input(
    "Enter your age",
    min_value=1,
    max_value=100,
    value=18,
    step=1
)

# Prediction
if st.button("Guess Screen Time"):
    # Model expects 2D array
    input_data = np.array([[age]])

    predicted_screen_time = model.predict(input_data)[0]

    st.success(f"🕒 Estimated Screen Time: {predicted_screen_time:.2f} hours/day")

st.markdown("---")
st.caption("⚠️ Prediction is based on a machine learning model and may not reflect actual usage.")
