import streamlit as st
import pandas as pd
st.title('Prediction of Cardiovascular Disease in Middle-Aged and Elderly Patients with Diabetes Mellitus')

st.info('This app is based on machine learning model')
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")

df
