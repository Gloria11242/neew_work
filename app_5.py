import streamlit as st
import joblib
import numpy as np

# ==============================
# Load the trained model
# ==============================
model = joblib.load("sales_prediction_model.pkl")

# ==============================
# Page Configuration
# ==============================
st.set_page_config(page_title="Sales Prediction App", page_icon="💰", layout="centered")

# ==============================
# Sidebar: Info & Instructions
# ==============================
st.sidebar.title(" About This App")
st.sidebar.markdown(
    """
    ###  Sales Prediction App
    This app predicts **total sales** based on two key inputs:
    - **Quantity Ordered**
    - **Price Each**

    ---
    ###  How the Model Works
    The model was trained on historical sales data to learn patterns between:
    - Number of items sold  
    - Price per item  
    - Total sales revenue  

    When you enter new values, the model estimates the likely sales amount using relationships it learned — roughly:  
    \`Predicted Sales ≈ Quantity × Price × (Model Adjustment)\`

    ---
    ###  How to Use
    1. Enter the **quantity ordered** and **price per item**.  
    2. Click **Predict Sales**.  
    3. View your predicted total sales instantly.  

    ---
     
    """
)

# ==============================
# Main App Content
# ==============================
st.title(" Sales Prediction App")
st.subheader("Predict your sales using quantity and price")

st.markdown("###  Enter Your Data")

# Input fields with default values set to 77 and 89
quantity = st.number_input("Enter Quantity Ordered:", min_value=0, value=77)
price = st.number_input("Enter Price Each ($):", min_value=0.0, value=89.0, step=0.5)

# Prediction button
if st.button(" Predict Sales"):
    # Prepare input for model
    X_new = np.array([[quantity, price]])

    # Predict
    predicted_sales = model.predict(X_new)[0]

    # Display result
    st.success(f"✅ **Predicted Sales:** ${predicted_sales:,.2f}")
