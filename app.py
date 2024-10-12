import streamlit as st
import pandas as pd
import numpy as np
import joblib

def predict(humidity, rainfall, pressure, wind_speed):
    model = joblib.load('models/weather_model_zip.joblib')
    data = np.expand_dims(np.array([humidity, rainfall, pressure, wind_speed]), axis=0)
    predictions = model.predict(data)
    return predictions[0]

# Заголовок застосунку
st.title('Передбачення дощу на завтра')
st.markdown('Ця модель передбачає погоду на завтра в австралії, базуючись на історичних данних')

# Заголовок секції з характеристиками рослини
st.header("Погода сьогодні")
col1, col2 = st.columns(2)

# Введення характеристик чашолистків
with col1:
    humidity = st.slider('Вологість повітря', 0, 100)
    rainfall = st.slider('Кількість опадів', 0, 350)

# Введення характеристик пелюсток
with col2:
    pressure = st.slider('Атмосферний тиск', 975, 1040)
    wind_speed = st.slider('Швидкість вітру', 6, 135)

# Кнопка для прогнозування
if st.button("Прогнозувати погоду"):
    # Викликаємо функцію прогнозування
    result = predict(humidity, rainfall, pressure, wind_speed)
    st.write(f"Чи буде завтра дош: {result}")
