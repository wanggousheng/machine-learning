
import streamlit as st
import pandas as pd
import joblib 
import shap
import matplotlib.pyplot as plt
st.title('Prediction of Cardiovascular Disease in Middle-Aged and Elderly Patients with Diabetes Mellitus')

st.info('This app is based on machine learning model')
df = pd.read_csv("X_test.csv")
model = joblib.load('XGB.pkl')


## 'srh', 'adlab_c', 'hibpe', 'lunge', 'dyslipe', 'kidneye', 'digeste',
##       'asthmae', 'memrye', 'mdact_c', 'hospital', 'retire', 'wrist_pain',
##       'chest_pain'
with st.sidebar:
  st.header("patient-related information")
  srh = st.selectbox(
    "Your Self-Reported Health Status Score",
    ("Very Good", "Good", "Fair","Poor",'Very Poor'),
    index=None,
    placeholder="Select contact method...",
)
