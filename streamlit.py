#importing the required libraries
import numpy as np
import streamlit as st
import pickle
import joblib
import pandas as pd
import os
import pathlib

model_path = os.fspath(pathlib.Path(__file__).parent / 'base_model.pkl')
# loading the saved model
loaded_model = pickle.load(open(model_path,'rb'))

st.info(
    f"""
    Welcome to Precision Farming for Best Product Results with Data!

    This app assists farmers in predicting the optimal planting and harvesting times for crops, taking into account local weather conditions and soil quality. The machine learning model behind this app is tailored to empower farmers and improve crop yields.

    To get started:
    1. Enter your location and crop details in the user-friendly interface.
    2. The system, powered by machine learning, will analyze data to predict the best planting and harvesting times for your specific conditions.

    Let's revolutionize agriculture together with data-driven precision farming!

    [Explore and utilize the provided data for building your predictive models.](https://drive.google.com/drive/folders/19ZLOahGV9tS-xqwztQC92HS111uCUGH8)
    """
)


def crop_prediction(temperature, humidity, ph, water, season, country):

    # Convert values to dataframe
    values = [temperature, humidity, ph, water, season, country]
    df = pd.DataFrame([values], columns=['temperature', 'humidity', 'ph', 'water', 'season', 'Country'])

    # apply OHE
    encoder_path = os.fspath(pathlib.Path(__file__).parent / 'encoder.joblib')
    loaded_encoder = joblib.load(encoder_path)
    row_to_predict_ = loaded_encoder.transform(df)

    prediction = loaded_model.predict(row_to_predict_)
    return ' '.join(prediction)


def main():
    
    # getting the input from the user
    st.title("Precision Farming for Best Product Results with Data")
    st.write("Welcome to our Hackathon project focused on revolutionizing agriculture through precision farming!")

    temperature = float(st.number_input('Temperature', min_value=15, max_value=37, step=1))
    humidity = float(st.number_input('Humidity', min_value=14, max_value=94, step=1))
    ph = float(st.number_input('Soil pH', min_value=3, max_value=10, step=1))
    water = float(st.number_input('Water Availability', min_value=20, max_value=300, step=1))
    season = st.selectbox('Season', ('rain', 'winter', 'summer', 'spring'))
    country = st.selectbox('Country', ('Nigeria', 'South Africa', 'Kenya', 'Sudan'))
    
    
    # code for predicted price
    predicted_crop = ''

    # code for button
    if st.button("Crop Prediction"):
        predicted_crop = crop_prediction(temperature, humidity, ph, water, season, country)
        st.success(f'The predicted crop is {predicted_crop}')
    
    
if __name__ == '__main__':
    main()
    
    
