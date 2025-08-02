import streamlit as st
import pickle
import pandas as pd
import numpy as np
from babel.numbers import format_currency

st.set_page_config(page_title="Car Price Predictor", layout="centered")

try:
    with open('LinearModel.pkl', 'rb') as f:
        model = pickle.load(f)
    car_data = pd.read_csv('cleaned_car_data.csv')
except FileNotFoundError:
    st.error("Model or data files not found. Please run the model training script first to generate them.")
    st.stop()

st.markdown("<h1 style='text-align: center; color: #1f77b4;'>Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter your car details below to estimate its current market value.</p>", unsafe_allow_html=True)
st.markdown("---")

companies = sorted(car_data['company'].unique())
years = sorted(car_data['year'].unique(), reverse=True)
fuel_types = car_data['fuel_type'].unique()

with st.form("car_price_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        company = st.selectbox("Car Company", companies)
        year = st.selectbox("Year of Purchase", years)
        kms_driven = st.number_input("Kilometres Driven", min_value=0, step=500, help="Total distance the car has travelled.")
        
    with col2:
        car_models = sorted(car_data[car_data['company'] == company]['name'].unique())
        car_model = st.selectbox("Car Model", car_models)
        fuel_type = st.selectbox("Fuel Type", fuel_types)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    with st.spinner("Calculating estimated price..."):
        if kms_driven >= 0:
            input_data = pd.DataFrame(
                [[car_model, company, year, kms_driven, fuel_type]],
                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
            )
            
            try:
                prediction = model.predict(input_data)
                predicted_price = np.round(prediction[0], 2)
                formatted_price = format_currency(predicted_price, 'INR', locale='en_IN')

                st.markdown(f"""
                    <div style='
                        background-color: #e6f2ff;
                        padding: 20px;
                        border-left: 5px solid #1f77b4;
                        border-radius: 8px;
                        margin-top: 20px;'>
                        <h3 style='color: #1f77b4;'>Estimated Car Price: {formatted_price}</h3>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Please enter a valid number of kilometers.")

st.markdown("---")