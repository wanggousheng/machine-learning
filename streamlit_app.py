
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
    placeholder="Select you score.",
  )
  
  adlab_c = st.selectbox(
    '''How many of the following daily living activities do you have difficulty with?
(Note: Daily living activities include: using the toilet, feeding yourself,
 dressing yourself, controlling bowel and bladder movements, getting in and out of bed, bathing yourself)''',
    ("0", "1", "2","3",'4','5','6'),
    index=None,
    placeholder='''Options: 0 item (no difficulty) / 1 item 
    / 2 items / 3 items / 4 items / 5 items / 6 items.''',
  )
