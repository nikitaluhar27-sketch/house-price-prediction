# app.py
# ============================================
# PHASE 6: Streamlit Web App
# ============================================

import streamlit as st
import numpy as np
import pickle

# ---- Load the saved model ----
with open('house_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ============================================
# PAGE SETUP
# ============================================
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ---- Title Section ----
st.title("🏠 House Price Predictor")
st.markdown("### Predict California House Prices using Machine Learning")
st.markdown("---")

# ============================================
# INPUT SLIDERS
# ============================================
st.header("📋 Enter House Details")
st.markdown("Move the sliders to match your house features:")

# Create 2 columns for neat layout
col1, col2 = st.columns(2)

with col1:
    MedInc = st.slider(
        "💰 Median Income (x$10,000)",
        min_value=0.5,
        max_value=15.0,
        value=5.0,
        step=0.1,
        help="Median income of the neighborhood"
    )

    HouseAge = st.slider(
        "🏚️ House Age (years)",
        min_value=1,
        max_value=52,
        value=20,
        step=1,
        help="How old is the house?"
    )

    AveRooms = st.slider(
        "🚪 Average Rooms",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.1,
        help="Average number of rooms per house"
    )

    AveBedrms = st.slider(
        "🛏️ Average Bedrooms",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="Average number of bedrooms"
    )

with col2:
    Population = st.slider(
        "👥 Population",
        min_value=3.0,
        max_value=5000.0,
        value=1000.0,
        step=10.0,
        help="Population of the neighborhood"
    )

    AveOccup = st.slider(
        "🏘️ Average Occupants",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.1,
        help="Average number of people per house"
    )

    Latitude = st.slider(
        "🗺️ Latitude",
        min_value=32.5,
        max_value=42.0,
        value=37.0,
        step=0.1,
        help="Geographic latitude (N-S location)"
    )

    Longitude = st.slider(
        "🗺️ Longitude",
        min_value=-124.0,
        max_value=-114.0,
        value=-120.0,
        step=0.1,
        help="Geographic longitude (E-W location)"
    )

# ============================================
# PREDICTION BUTTON
# ============================================
st.markdown("---")

if st.button("🔮 Predict House Price!", use_container_width=True):

    # Prepare input as array for model
    input_features = np.array([[
        MedInc, HouseAge, AveRooms, AveBedrms,
        Population, AveOccup, Latitude, Longitude
    ]])

    # Make prediction
    prediction = model.predict(input_features)
    predicted_price = prediction[0] * 100000

    # Show result
    st.markdown("---")
    st.success(f"### 🏡 Predicted House Price: ${predicted_price:,.0f}")

    # Show price range
    low  = predicted_price * 0.90
    high = predicted_price * 1.10

    st.info(f"📊 Likely price range: ${low:,.0f} — ${high:,.0f}")

    # Show rating
    st.markdown("### 💰 Price Category:")
    if predicted_price < 100000:
        st.warning("🟡 Budget Home")
    elif predicted_price < 250000:
        st.success("🟢 Affordable Home")
    elif predicted_price < 400000:
        st.info("🔵 Mid-Range Home")
    else:
        st.error("🔴 Luxury Home")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    "Built with ❤️ using Python, Scikit-learn & Streamlit | "
    "ML Model: Linear Regression | "
    "Dataset: California Housing"
)
